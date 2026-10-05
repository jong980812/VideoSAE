"""V-JEPA 2 backbone (e.g. `facebook/vjepa2-vitl-fpc64-256`).

Natively 64 frames -> 32 * 16 * 16 = 8192 tokens of width 1024. No CLS token.
This repo runs it at 16 frames (`backbone.frames` in its config) -> 8 * 16 * 16 =
2048 tokens.

THREE THINGS DIFFER FROM ViViT, AND EACH IS HANDLED HERE
--------------------------------------------------------
**No classification head.** V-JEPA 2 is trained by predicting masked latents;
the Hub repo ships the encoder (plus the self-supervised predictor) and nothing
else. `encode()` returns the mean over the final tokens, and skips the
predictor, which runs after the encoder and changes no block's output.

**The forward takes `pixel_values_videos`.** Every caller in this project passes
`pixel_values`, so `_features()` hands it on under the other name and
`_VideoProcessor` renames the processor's output to match. Doing it here keeps
the datasets model-agnostic.

**`config.num_frames` does not exist** -- V-JEPA 2 calls it `frames_per_clip`,
and its spatial size `crop_size` rather than `image_size`. Both are overridden
below so `grid_shape` and the frame samplers keep working.
"""

from typing import Optional

import torch
from transformers import AutoVideoProcessor, VJEPA2Model

from models.base import VideoBackbone


class _VideoProcessor:
    """`VJEPA2VideoProcessor` under the key the rest of the project expects.

    Callers do `processor([list(v) for v in videos], return_tensors=...)` and
    read `["pixel_values"]`; V-JEPA 2's processor emits `pixel_values_videos`.
    Everything else (resize to shortest side 292, centre-crop 256, ImageNet
    mean/std) passes straight through.
    """

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, videos, return_tensors=None, **kw):
        out = self.inner(videos, return_tensors=return_tensors, **kw)
        if "pixel_values_videos" in out:
            out["pixel_values"] = out.pop("pixel_values_videos")
        return out

    def __getattr__(self, item):
        return getattr(self.inner, item)


class VJepa2(VideoBackbone):
    family = "vjepa2"

    def __init__(
        self,
        checkpoint: str,
        revision: Optional[str] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
        frames: Optional[int] = None,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.dtype = dtype

        self.model = VJEPA2Model.from_pretrained(
            checkpoint, revision=revision, dtype=dtype).to(device).eval()
        # The base class reaches blocks through `backbone.encoder.layer`, which
        # VJEPA2Model exposes directly, so the backbone IS the model here.
        self.backbone = self.model
        self.config = self.model.config
        if frames is not None:
            # V-JEPA 2's attention is RoPE -- positions are computed from the input
            # grid, not looked up in a learned table -- so it can read clips of
            # another length. `frames_per_clip` is the one field to set: both
            # `num_frames` (what the samplers read) and `grid_shape` derive from it.
            # It is a different representation from the native 64 frames, so
            # anything fitted at one length is off-distribution at the other.
            if frames % self.config.tubelet_size:
                raise ValueError(f"frames must be a multiple of the tubelet size "
                                 f"{self.config.tubelet_size}, got {frames}")
            self.config.frames_per_clip = frames
        self.processor = _VideoProcessor(
            AutoVideoProcessor.from_pretrained(checkpoint, revision=revision))

    # ------------------------------------------------------- subclass contract

    @property
    def has_cls(self) -> bool:
        # V-JEPA 2 prepends nothing; every token is a tubelet.
        return False

    @property
    def num_frames(self) -> int:
        return self.config.frames_per_clip

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        side = self.config.crop_size // self.config.patch_size
        return (self.config.frames_per_clip // self.config.tubelet_size, side, side)

    # ----------------------------------------------------------------- forward

    def _features(self, px):
        """Mean-pooled (B, 1024) features -- there is no CLS token to read instead."""
        # skip_predictor: the predictor runs AFTER the encoder, so skipping it
        # changes no block's output and saves 12 blocks of compute.
        out = self.model(pixel_values_videos=px, skip_predictor=True)
        tokens = out.last_hidden_state.float()          # (B, 2048 at 16 frames, 1024)
        return tokens.mean(dim=1)

    def _finish(self, hidden):
        # VJEPA2Encoder: final LayerNorm over every token; then the mean, as above.
        # (RoPE positions come from the token count, so the blocks need nothing else.)
        return self.backbone.encoder.layernorm(hidden).float().mean(dim=1)
