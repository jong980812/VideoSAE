"""Backbone registry: one model ID per backbone *variant*, one YAML file per ID.

    from models import get_model
    model = get_model("videomaev2-base", device="cuda:0")

`config/<model_id>.yaml` holds everything about a model: its name, the wrapper class, the
Hugging Face checkpoint and pinned revision (and, optionally, the clip length to
run at), the shape the loaded model must have (`expect:`, checked on every load),
where its SAEs are, one classification head per label space (kinetics400,
nturgbd, ssv2, ...), and the defaults of extract_activations.py / train_probe.py
/ evaluate.py.

A model ID names exactly one checkpoint. The same ID names the SAE weights
directory (`weights/sae/<model_id>/`) and is recorded in every activation dump, SAE
config and result file, so an SAE can never be paired with a different
checkpoint of the "same" model -- e.g. an SAE fitted on VideoMAEv2-Base
activations refuses to load against VideoMAEv2-giant.

To add a variant (another size, another fine-tune), add a YAML file with its own
ID. If the new checkpoint has the same architecture as an existing one, the
existing wrapper class usually works unchanged.
"""

import importlib
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from models.base import VideoBackbone

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _parse_layers(spec) -> tuple:
    """'0-10' -> (0, ..., 10); '3,7' or [3, 7] -> (3, 7)."""
    if isinstance(spec, (list, tuple)):
        return tuple(int(x) for x in spec)
    spec = str(spec)
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return tuple(range(int(a), int(b) + 1))
    return tuple(int(x) for x in spec.split(",") if x)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    raw: dict                  # the whole YAML, for sections a script reads itself

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelSpec":
        raw = yaml.safe_load(path.read_text())
        if raw.get("model_id") != path.stem:
            raise ValueError(f"{path}: model_id {raw.get('model_id')!r} != file name {path.stem!r}")
        return cls(model_id=raw["model_id"], raw=raw)

    @property
    def name(self) -> str:
        """What the model is, in words (the model ID is only a key)."""
        return self.raw["name"]

    # backbone
    @property
    def wrapper(self) -> str:
        return self.raw["backbone"]["wrapper"]

    @property
    def checkpoint(self) -> str:
        return self.raw["backbone"]["checkpoint"]

    @property
    def revision(self) -> str:
        return self.raw["backbone"]["revision"]

    @property
    def expect(self) -> dict:
        return self.raw["expect"]

    @property
    def frames(self):
        """Clip length to run at, if the config overrides the checkpoint's own."""
        return self.raw["backbone"].get("frames")

    def head(self, label_space: str) -> dict:
        """The head that scores `label_space` (kinetics400, nturgbd, ssv2, ...)."""
        heads = self.raw["head"]
        if label_space not in heads:
            raise ValueError(f"config/{self.model_id}.yaml has no head for {label_space!r} "
                             f"(has: {', '.join(heads)})")
        return heads[label_space]

    def sae_dir(self, layer: int) -> Path:
        """The folder holding the block-`layer` SAE (ae.pt + config.json), from
        `sae.dir` in the config; relative paths are relative to the repo."""
        path = Path(self.raw["sae"]["dir"].format(layer=layer, model_id=self.model_id))
        return path if path.is_absolute() else CONFIG_DIR.parent / path

    # extract_activations.py defaults
    @property
    def layers(self) -> tuple:
        return _parse_layers(self.raw["extract"]["layers"])

    @property
    def batch_size(self) -> int:
        return int(self.raw["extract"]["batch_size"])

    @property
    def rows_per_shard(self) -> int:
        return int(self.raw["extract"]["rows_per_shard"])

    @property
    def probe_batch_size(self) -> int:
        return int(self.raw["probe"]["batch_size"])

    @property
    def eval_batch_size(self) -> int:
        return int(self.raw["eval"]["batch_size"])


MODELS = {p.stem: ModelSpec.from_yaml(p) for p in sorted(CONFIG_DIR.glob("*.yaml"))}

__all__ = ["MODELS", "ModelSpec", "VideoBackbone", "get_model", "get_spec"]


def get_spec(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id]
    except KeyError:
        raise ValueError(
            f"Unknown model ID {model_id!r}. Registered IDs: {', '.join(MODELS)}. "
            f"To add a checkpoint, add config/<model_id>.yaml."
        ) from None


def get_model(
    model_id: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float32,
) -> VideoBackbone:
    """Load the backbone registered under `model_id`, at its pinned revision.

    The shipped SAEs were fitted on fp32 activations; keep `dtype=torch.float32`
    when using them.
    """
    spec = get_spec(model_id)
    module_name, class_name = spec.wrapper.split(":")
    cls = getattr(importlib.import_module(f"models.{module_name}"), class_name)
    extra = {"frames": spec.frames} if spec.frames is not None else {}
    model = cls(spec.checkpoint, revision=spec.revision, device=device, dtype=dtype, **extra)
    model.model_id = model_id
    model.revision = spec.revision

    e = spec.expect
    got = {"blocks": model.num_layers, "width": model.hidden_dim, "frames": model.num_frames,
           "token_grid": list(model.grid_shape), "cls_token": model.has_cls}
    want = {k: (list(e[k]) if k == "token_grid" else e[k]) for k in got}
    if got != want:
        raise ValueError(f"{model_id}: loaded checkpoint does not match config/{model_id}.yaml "
                         f"`expect:` -- got {got}, expected {want}")
    return model
