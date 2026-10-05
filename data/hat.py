"""HAT4 ActionSwap composites, one per item (config/datasets/hat.yaml).

Each sample is (fg_key, label, bg_key, bg_label): the PERSON of Kinetics-400 val
clip `fg_key` is pasted, with its soft segmentation mask, onto the inpainted
BACKGROUND of clip `bg_key`, which is of a different class. A model reading the
person should answer `label`; one reading the scene answers `bg_label`.

    <original_root>/<fg_key>/000001.jpg     the person's frames (Kinetics-400 val)
    <hat_root>/seg/<fg_key>/000001.png      its soft mask, 0-255
    <hat_root>/<inpaint_dir>/<bg_key>/...   the background clip with its person removed

How a composite is built:
  * frames are chosen from the frame ids present in all three directories
    (matched by id, never by position), `num_frames` evenly spaced;
  * every layer is canonicalised the same way BEFORE blending -- shortest edge
    to `size` (bilinear), centre crop to size x size -- so person and scene go
    through identical geometry;
  * composite = fg * mask + bg * (1 - mask), with the soft mask;
  * the model's processor then runs on each composite on its own.
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.clips import _frames_in


def _canonical(clip_dir, names, size: int, mode: str = "RGB") -> np.ndarray:
    """Resize shortest edge to `size`, then centre crop -- the processor's geometry."""
    out = []
    for n in names:
        im = Image.open(os.path.join(clip_dir, n)).convert(mode)
        w, h = im.size
        scale = size / min(w, h)
        im = im.resize((max(round(w * scale), size), max(round(h * scale), size)), Image.BILINEAR)
        w, h = im.size
        left, top = (w - size) // 2, (h - size) // 2
        out.append(np.array(im.crop((left, top, left + size, top + size))))
    return np.stack(out)


class HATComposites(Dataset):
    """[(fg_key, label, bg_key, bg_label), ...] -> one composite per item."""

    def __init__(self, samples: list, hat_root: str, original_root: str, num_frames: int,
                 processor, inpaint_dir: str = "inpaint", size: int = 224):
        self.samples = [tuple(s) for s in samples]
        self.hat, self.original = Path(hat_root), Path(original_root)
        self.num_frames, self.processor = num_frames, processor
        self.inpaint_dir, self.size = inpaint_dir, size

    def __len__(self):
        return len(self.samples)

    def _picked(self, fg_key: str, bg_key: str) -> list:
        dirs = [self.original / fg_key, self.hat / "seg" / fg_key,
                self.hat / self.inpaint_dir / bg_key]
        maps = [{os.path.splitext(n)[0]: n for n in _frames_in(d)} for d in dirs]
        stems = sorted(set.intersection(*(set(m) for m in maps)))
        if not stems:
            raise RuntimeError(f"no frame ids shared by {fg_key} and {bg_key}")
        idx = np.linspace(0, len(stems) - 1, num=self.num_frames).round().astype(int)
        return [[m[stems[i]] for i in idx] for m in maps]

    def __getitem__(self, i: int) -> dict:
        fg_key, label, bg_key, bg_label = self.samples[i]
        picked = self._picked(fg_key, bg_key)
        fg = _canonical(self.original / fg_key, picked[0], self.size)
        mask = _canonical(self.hat / "seg" / fg_key, picked[1], self.size,
                          mode="L").astype(np.float32)[..., None] / 255.0
        bg = _canonical(self.hat / self.inpaint_dir / bg_key, picked[2], self.size)
        video = (fg * mask + bg * (1.0 - mask)).astype(np.uint8)
        px = self.processor([list(video)], return_tensors=None)["pixel_values"][0]
        return {"pixel_values": torch.from_numpy(np.asarray(px, dtype=np.float32)),
                "label": label, "bg_label": bg_label, "key": fg_key, "bg_key": bg_key}


def hat_collate(batch):
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": torch.tensor([b["label"] for b in batch]),
            "bg_labels": torch.tensor([b["bg_label"] for b in batch]),
            "paths": [b["key"] for b in batch],
            "bg_keys": [b["bg_key"] for b in batch]}
