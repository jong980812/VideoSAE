"""VideoMAE V2 backbone (e.g. `OpenGVLab/VideoMAEv2-Base`).

16 frames -> 8 * 14 * 14 = 1568 tokens of width 768. No CLS token. 12 blocks.

THE CHECKPOINT
--------------
Loaded through `trust_remote_code`: the modelling code ships in the Hub repo and
is fetched with the weights; the registry pins `revision=`, which pins that code
too. `AutoModel` returns a thin `VideoMAEv2` PreTrainedModel whose `.model` is a
timm-style VisionTransformer -- `blocks`, `fc_norm`, `head` -- and that is what
`encode`/`decode` run.

`videomaev2-vitb-k710distill` (the ID the shipped SAEs were trained on) IS SUPERVISED,
whatever the model card says. The card for `OpenGVLab/VideoMAEv2-Base` calls the model
self-supervised (800 epochs on UnlabeledHybrid-1M), but outside the blocks the
weights hold only the patch embedding and a TRAINED `fc_norm` (weight mean 0.52,
where a fresh LayerNorm is 1.0) -- no decoder, no head -- and a MAE encoder has
no `fc_norm` at all. The official model zoo's only ViT-B is
`vit_b_k710_dl_from_giant`: distilled from a giant FINE-TUNED on Kinetics-710
labels, head stripped here. So these features have seen Kinetics labels, which
V-JEPA 2's have not.

WHAT DIFFERS FROM V-JEPA 2
--------------------------
**Pixels go in as (B, C, T, H, W).** VideoMAEImageProcessor emits
(B, T, C, H, W); HF's own VideoMAE permutes inside its embeddings, the Hub
code's Conv3d does not. `_to_model()` permutes before every forward.

**The blocks are not at `encoder.layer`.** They are `model.model.blocks`, and
each block's attention is `attn`, not `attention`. `layers` says so; everything
else in `VideoBackbone` works unchanged.

**Pooling is mean-then-norm**: `fc_norm(tokens.mean(1))`, the checkpoint's own
`forward_features`, which is what `encode()` returns. V-JEPA 2 norms per token
and then means.

**The clip length is fixed at 16.** The positional embedding is a sinusoid table
built once, at construction, for 1,568 positions and added with no resizing, so
any other length is a shape error in the forward.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, VideoMAEImageProcessor

from models.base import VideoBackbone


class VideoMAEv2(VideoBackbone):
    family = "videomaev2"

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
        # The VisionTransformer inside the PreTrainedModel shell.
        self.backbone = self.model.model
        self.config = self.model.config
        self._cfg = self.config.model_config
        self.processor = VideoMAEImageProcessor.from_pretrained(checkpoint, revision=revision)

    # ------------------------------------------------------- subclass contract

    @property
    def layers(self) -> nn.ModuleList:
        return self.backbone.blocks

    @property
    def has_cls(self) -> bool:
        return False

    @property
    def hidden_dim(self) -> int:
        return self._cfg["embed_dim"]

    @property
    def num_frames(self) -> int:
        return self._cfg["num_frames"]

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        side = self._cfg["img_size"] // self._cfg["patch_size"]
        return (self._cfg["num_frames"] // self._cfg["tubelet_size"], side, side)

    def _to_model(self, px):
        """Processor layout (B, T, C, H, W) -> the Conv3d's (B, C, T, H, W)."""
        if px.dim() != 5 or px.shape[2] != self._cfg["in_chans"]:
            raise ValueError(f"expected (B, T, C, H, W) pixels, got {tuple(px.shape)}")
        return px.permute(0, 2, 1, 3, 4)

    # ----------------------------------------------------------------- forward

    def _features(self, px):
        """Pooled (B, 768) features: fc_norm of the mean over all tokens."""
        # forward_features = patch embed, blocks, fc_norm(mean over tokens).
        return self.backbone.forward_features(self._to_model(px)).float()

    def _finish(self, hidden):
        # The end of forward_features: fc_norm of the mean over tokens.
        return self.backbone.fc_norm(hidden.mean(1)).float()
