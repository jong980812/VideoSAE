"""Classification heads, built from the `head:` map of config/<model_id>.yaml.

The map has one entry per label space (kinetics400, nturgbd, ssv2, ...); a
dataset config names the label space it is scored in. Every head maps
`model.encode(...)` features (B, D) to logits (B, n_classes), with output i
meaning class `classes[i]` of that label space:

    native    the checkpoint's own classifier (ViViT on Kinetics-400; CLS token)
    probe     a linear map on the frozen features, weights/heads/<label_space>/<model_id>.pt
    zeroshot  SigLIP's text tower, one prompt per class name
"""

import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent


class LinearHead(nn.Module):
    def __init__(self, W: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.register_buffer("W", W.float().clone())
        self.register_buffer("b", b.float().clone())

    def forward(self, features):
        return F.linear(features, self.W, self.b)


class ZeroShotHead(nn.Module):
    def __init__(self, text_emb: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.register_buffer("text_emb", text_emb.float())
        self.register_buffer("scale", scale.float())
        self.register_buffer("bias", bias.float())

    def forward(self, features):
        return F.normalize(features, dim=-1) @ self.text_emb.T * self.scale + self.bias


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_probe(path) -> dict:
    """A probe file: {W (C, D), b (C,), classes, val_top1, ...}, one output per
    class of its label space (400 for Kinetics-400, 120 for NTU)."""
    return torch.load(path, map_location="cpu", weights_only=True)


def build_head(model, label_space: str, classes: list, device: str) -> tuple[nn.Module, dict]:
    """The model's head for `label_space` -> (module on `device`, description for results)."""
    from models import get_spec

    cfg = dict(get_spec(model.model_id).head(label_space))
    kind = cfg["type"]
    info = {"label_space": label_space, "type": kind}

    if kind == "native":
        W, b = model.native_classifier()
        if W.shape[0] != len(classes):
            raise ValueError(f"native head has {W.shape[0]} outputs, the clip list {len(classes)} classes")
        head = LinearHead(W, b)
        info["description"] = "the checkpoint's own classifier, outputs in sorted class-name order"

    elif kind == "probe":
        path = ROOT / cfg["path"]
        if not path.exists() or not cfg.get("sha256"):
            raise SystemExit(f"no {label_space} head for {model.model_id} yet: {cfg['path']} "
                             f"{'is missing' if not path.exists() else 'has no sha256 in the config'}. "
                             f"Fit it, then set head.{label_space}.sha256 in "
                             f"config/{model.model_id}.yaml.")
        digest = _sha256(path)
        if cfg["sha256"] != digest:
            raise ValueError(f"{path} sha256 {digest} != config's {cfg['sha256']}")
        blob = load_probe(path)
        if list(blob["classes"]) != list(classes):
            raise ValueError(f"{path}: its class order differs from the clip list's, "
                             f"so every label would be shifted")
        if blob.get("frames") not in (None, model.num_frames):
            print(f"note: the {label_space} head was fitted on {blob['frames']}-frame "
                  f"features; the model runs at {model.num_frames}", flush=True)
        head = LinearHead(blob["W"], blob["b"])
        info.update(path=cfg["path"], sha256=digest,
                    description="linear probe on frozen pooled features",
                    probe_val_top1=blob.get("val_top1"), probe_n_train=blob.get("n_train"),
                    probe_frames=blob.get("frames"))

    elif kind == "zeroshot":
        emb, scale, bias = model.zero_shot_classifier(classes, cfg["prompt"])
        head = ZeroShotHead(emb, scale, bias)
        info.update(prompt=cfg["prompt"],
                    description="zero-shot: cosine to one text-tower prompt per class")

    else:
        raise ValueError(f"unknown head type {kind!r} (native | probe | zeroshot)")
    return head.to(device).eval(), info
