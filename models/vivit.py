"""ViViT backbone (e.g. `google/vivit-b-16x2-kinetics400`).

32 frames -> 1 CLS + 16*14*14 = 3137 tokens of width 768.

The CLS token is not part of any SAE's training data: `encode(video, layer=L)`
returns the 3136 patch tokens only and keeps the CLS token aside, and
`decode(patches, layer=L)` puts it back untouched. SAEs ship for blocks 0-10
only; the checkpoint's own classifier reads just the CLS token, so the last
block's patch outputs never reach it and no SAE was trained there.
"""

from typing import Optional

import torch
from transformers import VivitForVideoClassification, VivitImageProcessor

from models.base import VideoBackbone


class Vivit(VideoBackbone):
    family = "vivit"

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

        # The classification model, so the checkpoint's own Kinetics-400 head is
        # available (`native_classifier`). Its encoder, `.vivit`, is a VivitModel
        # without a pooler -- the blocks and their outputs are the same as a bare
        # VivitModel's.
        self.model = VivitForVideoClassification.from_pretrained(
            checkpoint, revision=revision, dtype=dtype).to(device).eval()
        self.backbone = self.model.vivit
        # Explicit class: ViViT is absent from AutoImageProcessor's mapping on
        # some transformers versions (its preprocessor_config has no type key).
        self.processor = VivitImageProcessor.from_pretrained(checkpoint, revision=revision)
        self.config = self.backbone.config

    @property
    def has_cls(self) -> bool:
        return True

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        tub_t, tub_h, tub_w = self.config.tubelet_size  # ViViT stores a 3-list
        img = self.config.image_size
        return (self.config.num_frames // tub_t, img // tub_h, img // tub_w)

    # ----------------------------------------------------------------- forward

    def _features(self, px):
        """(B, 768): the final-layernormed CLS token -- what the classifier reads."""
        return self.backbone(pixel_values=px).last_hidden_state[:, 0].float()

    def _finish(self, hidden):
        # VivitModel: final LayerNorm over every token; the classifier reads token 0.
        return self.backbone.layernorm(hidden)[:, 0].float()

    def native_classifier(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(W (400, 768), b (400,)) of the checkpoint's Kinetics-400 classifier.

        Its config labels the outputs only LABEL_0..LABEL_399; they are the 400
        Kinetics classes in sorted name order, the order of every clip list
        here (read that way, it scores the README's reference accuracy).
        """
        return self.model.classifier.weight.detach(), self.model.classifier.bias.detach()
