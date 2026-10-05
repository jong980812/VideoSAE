"""SigLIP SO400M as a single-frame "video" backbone.

An image model dressed as a video model, so the rest of the project does not
have to know the difference: it takes the same list-of-frames input every other
backbone takes, and quietly keeps only the middle frame.

Two things about SigLIP do not match the video models, and each is handled
here rather than in the callers:

**No CLS token.** SigLIP pools with an attention head (`SiglipMultiheadAttention
PoolingHead`) rather than carrying a CLS token, so `has_cls` is False and every
one of the 27 x 27 = 729 tokens is a patch. `encode()` returns that pooled
1152-d vector.

**Middle frame, cheaply.** `num_frames = 3` rather than 1. The frame samplers in
this project take `linspace(0, n-1, num_frames)`, so 3 gives
[first, middle, last] and index 1 is the true middle frame of the clip -- while
`num_frames = 1` would give `linspace(0, n-1, 1) == [0]`, the *first* frame. Three
frames get decoded instead of thirty-two, and only the middle one is used.

SigLIP's own processor squashes each frame to 384 x 384 (no crop).
"""

from typing import Optional

import numpy as np
import torch
from transformers import SiglipImageProcessor, SiglipModel

from models.base import VideoBackbone


class _MiddleFrameProcessor:
    """Adapter giving SigLIP's image processor the video processors' signature.

    Callers do `processor([list(v) for v in videos], return_tensors=...)` and read
    `["pixel_values"]`. This takes that same call, keeps frame `len//2` of each
    clip, and returns whatever SigLIP's processor makes of those stills.
    """

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, videos, return_tensors=None, **kw):
        # A bare list of frames (one clip) is accepted too, for convenience.
        if videos and not isinstance(videos[0], (list, tuple, np.ndarray)):
            videos = [videos]
        middles = [np.asarray(v[len(v) // 2]) for v in videos]
        return self.inner(images=middles, return_tensors=return_tensors, **kw)

    def __getattr__(self, item):
        return getattr(self.inner, item)


class Siglip(VideoBackbone):
    family = "siglip"

    def __init__(
        self,
        checkpoint: str,
        revision: Optional[str] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        self.checkpoint = checkpoint
        self.revision = revision
        self.device = device
        self.dtype = dtype

        self.model = SiglipModel.from_pretrained(
            checkpoint, revision=revision, dtype=dtype).to(device).eval()
        self.backbone = self.model.vision_model
        # Image processor only. AutoProcessor would also build SiglipTokenizer,
        # which needs sentencepiece -- a dependency the vision path never uses.
        self.processor = _MiddleFrameProcessor(
            SiglipImageProcessor.from_pretrained(checkpoint, revision=revision))
        self.config = self.model.config.vision_config

    # ------------------------------------------------------- subclass contract

    @property
    def has_cls(self) -> bool:
        # Pooling is an attention head, not a prepended token.
        return False

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        side = self.config.image_size // self.config.patch_size
        return (1, side, side)          # one frame, so the temporal axis is 1

    @property
    def num_frames(self) -> int:
        # 3, not 1: the samplers use linspace, and linspace(0, n-1, 1) is the
        # FIRST frame. Three gives [first, middle, last] and the wrapper keeps
        # the middle -- the true centre of the clip, for three decodes.
        return 3

    # ----------------------------------------------------------------- forward

    def _features(self, px):
        """Pooled (B, 1152) image features of each clip's middle frame."""
        if px.dim() == 5:                      # (B, T, 3, H, W) from a video collate
            px = px[:, px.shape[1] // 2]
        out = self.model.vision_model(pixel_values=px)
        return out.pooler_output.float()

    def _run_block(self, block, hidden):
        # SiglipEncoderLayer takes the attention mask positionally; there is none.
        return block(hidden, None)

    def _finish(self, hidden):
        # SiglipVisionTransformer: post_layernorm, then the attention-pooling head.
        vm = self.model.vision_model
        return vm.head(vm.post_layernorm(hidden)).float()

    @torch.no_grad()
    def zero_shot_classifier(self, class_names: list, prompt: str):
        """Text-tower zero-shot head: (text_emb (C, 1152), scale, bias).

        Each class name is put into `prompt` and encoded once; an image then
        scores `normalize(encode(x)) @ text_emb.T * scale + bias` -- SigLIP's
        trained temperature and bias, so the logits keep the model's own
        calibration. The tokenizer needs `sentencepiece`.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, revision=self.revision)
        tok = tokenizer([prompt.format(c) for c in class_names],
                        padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        emb = torch.nn.functional.normalize(self.model.get_text_features(**tok).float(), dim=-1)
        return emb, self.model.logit_scale.exp().float(), self.model.logit_bias.float()
