"""VideoPrism backbone (`google/videoprism-base-f16r288`, PyTorch port).

16 frames at 288px, patch 18 -> 16 * 16 * 16 = 4096 tokens of width 768.
No CLS token.

THE ENCODER IS FACTORISED, WHICH CHANGES WHAT A "LAYER" IS
-----------------------------------------------------------
ViViT and V-JEPA 2 run one flat stack of blocks over a joint space-time token
sequence. VideoPrism does not. It runs

    patchify -> spatial_encoder (12 blocks)  over (B*T, 256, D)   -- per frame
             -> temporal_encoder (4 blocks)  over (B*N, 16, D)    -- per position
             -> reshape                      to   (B, 4096, D)

so `layers` here is the 12 spatial blocks followed by the 4 temporal ones, 16
in total, and **the token axis means something different either side of the
boundary**. A spatial block outputs 256 spatial patches of one frame; a temporal
block outputs 16 timesteps of one spatial position. Both are (N, tokens, 768)
and both are fine to fit an SAE on -- but only `_to_grid()` (overridden below)
knows how to put either back into (B, T, H, W, D), the frame-major layout
`encode(video, layer=L)` returns, and `_from_grid()` how to undo it for `decode`.

`num_spatial_layers` is the boundary. Layer 11 is the last spatial block.

WEIGHTS AND CODE
----------------
Google published VideoPrism as a JAX/Flax `.npz` only, and `transformers`
4.56.1 has no VideoPrism. This loads the community PyTorch port, which is pure
`torch.nn` and reproduces the published architecture. It arrives through
`trust_remote_code=True`, so the code is fetched from the Hub and executed; the
registry pins `revision=`, which pins that code as well as the weights.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

from models.base import VideoBackbone


class _VideoProcessor:
    """Frames -> (B, T, H, W, 3) float in [0, 1], the port's input convention.

    The port ships a processor, but it takes a single video rather than the
    list-of-videos every dataset in this project passes. Doing the resize here
    keeps that contract and costs one interpolate.

    It is a straight bilinear resize of the FULL frame to 288 x 288 -- no crop,
    no antialiasing -- so non-square Kinetics frames (456 x 256) are squashed.
    The shipped SAEs were trained on exactly this input; do not "fix" it to a
    crop, or the activations will drift from what the SAEs expect.
    """

    def __init__(self, image_size: int = 288):
        self.image_size = image_size

    def __call__(self, videos, return_tensors=None, **kw):
        if videos and not isinstance(videos[0], (list, tuple, np.ndarray, torch.Tensor)):
            videos = [videos]
        out = []
        for v in videos:
            arr = torch.as_tensor(np.asarray(v))            # (T, H, W, 3) uint8
            x = arr.permute(0, 3, 1, 2).float() / 255.0     # (T, 3, H, W) in [0,1]
            if x.shape[-2:] != (self.image_size, self.image_size):
                x = nn.functional.interpolate(
                    x, size=(self.image_size, self.image_size),
                    mode="bilinear", align_corners=False)
            out.append(x.permute(0, 2, 3, 1))               # (T, H, W, 3)
        stacked = torch.stack(out)
        return {"pixel_values": stacked if return_tensors == "pt" else stacked.numpy()}


class VideoPrism(VideoBackbone):
    family = "videoprism"

    def __init__(
        self,
        checkpoint: str,
        revision: Optional[str] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.dtype = dtype

        self.model = AutoModel.from_pretrained(
            checkpoint, revision=revision, trust_remote_code=True, dtype=dtype).to(device).eval()
        # The port wraps the encoder one level down; the blocks and norms that
        # encode/decode use live on the inner FactorizedEncoder.
        self.backbone = self.model.videoprism
        self.config = self.model.config
        self.processor = _VideoProcessor(self.config.image_size)

    # ------------------------------------------------------- subclass contract

    @property
    def num_spatial_layers(self) -> int:
        """Layers 0 .. num_spatial_layers-1 are spatial, the rest temporal (12 + 4 for base)."""
        return len(self.backbone.spatial_encoder.x_layers)

    @property
    def layers(self) -> nn.ModuleList:
        """The 12 spatial blocks then the 4 temporal ones, as one list.

        Rebuilt on each access rather than cached: caching an nn.ModuleList of
        modules that already belong to the model would register them a second
        time and double what `.parameters()` reports.
        """
        return nn.ModuleList(list(self.backbone.spatial_encoder.x_layers)
                             + list(self.backbone.temporal_encoder.x_layers))

    @property
    def hidden_dim(self) -> int:
        return self.config.model_dim

    @property
    def num_frames(self) -> int:
        return self.config.num_frames

    @property
    def has_cls(self) -> bool:
        return False

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        side = self.config.image_size // self.config.patch_size
        return (self.config.num_frames, side, side)

    # ------------------------------------------------------------ token layout

    def _to_grid(self, acts: torch.Tensor, layer: int) -> torch.Tensor:
        """Block `layer`'s output -> (B, T, H, W, D), whichever side it came from.

        Spatial blocks emit (B*T, H*W, D) and temporal blocks (B*H*W, T, D).
        Both must come out in the same frame-major layout, and getting the
        permutation wrong silently mislabels every token, so it lives here
        rather than being re-derived per caller.
        """
        t, h, w = self.grid_shape
        n, tokens, d = acts.shape
        if layer < self.num_spatial_layers:
            if tokens != h * w:
                raise ValueError(f"spatial block {layer} should output {h*w} tokens, got {tokens}")
            return acts.reshape(n // t, t, h, w, d)
        if tokens != t:
            raise ValueError(f"temporal block {layer} should output {t} tokens, got {tokens}")
        b = n // (h * w)
        # (B*H*W, T, D) -> (B, H, W, T, D) -> (B, T, H, W, D)
        return acts.reshape(b, h, w, t, d).permute(0, 3, 1, 2, 4).contiguous()

    def _from_grid(self, grid: torch.Tensor, layer: int) -> torch.Tensor:
        """Inverse of `_to_grid`: (B, T, H, W, D) -> the block's own layout."""
        b, t, h, w, d = grid.shape
        if layer < self.num_spatial_layers:
            return grid.reshape(b * t, h * w, d)                                # (B*T, H*W, D)
        return grid.permute(0, 2, 3, 1, 4).contiguous().reshape(b * h * w, t, d)   # (B*H*W, T, D)

    # ----------------------------------------------------------------- forward

    def _features(self, px):
        """Mean-pooled (B, 768) features -- there is no CLS token to read instead."""
        out = self.model(pixel_values=px)
        tokens = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out).float()
        return tokens.mean(dim=1)

    def _resume(self, layer, hidden):
        """Resume the FACTORISED stack after `layer` (mirrors FactorizedEncoder.forward).

        spatial layer   hidden is (B*T, N, D): finish the spatial blocks, spatial_ln,
                        permute to (B*N, T, D), ADD the temporal position embedding,
                        then all four temporal blocks
        temporal layer  hidden is (B*N, T, D) and the temporal positions were already
                        added before the stack; adding them again would double-count
        """
        bb = self.backbone
        t, h, w = self.grid_shape
        n = h * w
        if layer < self.num_spatial_layers:
            for block in bb.spatial_encoder.x_layers[layer + 1:]:
                hidden = block(hidden)
            hidden = bb.spatial_ln(hidden)
            bt, _, d = hidden.shape
            b = bt // t
            hidden = hidden.reshape(b, t, n, d).permute(0, 2, 1, 3).contiguous().reshape(b * n, t, d)
            if bb.temporal_pos_emb.shape[0] != t:
                raise NotImplementedError(f"temporal position embedding has {bb.temporal_pos_emb.shape[0]} "
                                          f"steps, the grid {t}; the port interpolates, this does not")
            hidden = hidden + bb.temporal_pos_emb.unsqueeze(0)
            temporal_start = 0
        else:
            temporal_start = layer - self.num_spatial_layers + 1
        for block in bb.temporal_encoder.x_layers[temporal_start:]:
            hidden = block(hidden)
        hidden = bb.temporal_ln(hidden)
        bn, t2, d = hidden.shape
        b = bn // n
        # '(b n) t d -> b (t n) d', then the mean over tokens, as in _features
        hidden = hidden.reshape(b, n, t2, d).permute(0, 2, 1, 3).contiguous().reshape(b, t2 * n, d)
        return hidden.float().mean(dim=1)
