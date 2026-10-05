"""The SAE weights: one sparse autoencoder per (model ID, layer).

    from models import get_model
    from saes import load_sae, list_saes

    list_saes()                                   # {"vivit-b-16x2-kinetics400": [0, ..., 10], ...}
    model = get_model("vjepa2-vitl-fpc64-256", device="cuda:0")
    sae = load_sae("vjepa2-vitl-fpc64-256", 18, device="cuda:0", backbone=model)

Where an SAE is loaded from: `sae.dir` in config/<model_id>.yaml, a folder
template with {layer} filled in -- for the shipped weights,
`weights/sae/<model_id>/l<layer>/{ae.pt, config.json}`. Change it there to use other
weights. Passing `weights_dir=` instead reads `<weights_dir>/<model_id>/l<layer>/`,
the layout train_sae.py writes.

`config.json` says which backbone checkpoint + revision, layer and site
(`resid_post`, the block's output) the SAE was fitted on and how it was trained; `load_sae` checks it against the
request (and against `backbone`, when given) so an SAE cannot be silently paired
with the wrong model.

What an SAE expects as input: RAW `resid_post` activations of PATCH tokens at
its layer, in fp32 -- exactly what `model.encode(video, layer=layer)` returns.
The training-time normalisation is folded into the saved weights, so do not
rescale activations yourself. `sae.encode(x)` uses the learned global threshold
(the deployment behaviour); `sae.encode(x, use_threshold=False)` gives exact
BatchTopK over the batch (pass 2-D (N, d) input for that).
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

import torch

from dictionary_learning.trainers import BatchTopKSAE, MatroyshkaBatchTopKSAE
from models import MODELS, get_spec

SAE_CLASSES = {
    "MatroyshkaBatchTopKSAE": MatroyshkaBatchTopKSAE,
    "BatchTopKSAE": BatchTopKSAE,
}

INPUT_NOTE = ("raw fp32 resid_post activations of PATCH tokens at `layer` (no CLS); "
              "the training-time normalisation is folded into the weights")


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_config(*, model_id: str, checkpoint: str, revision: str, layer: int,
                frames: int, token_grid, ae_path, training: dict, data: dict) -> dict:
    """The config.json written next to every ae.pt (shipped or newly trained).

    Architecture fields are read from `ae_path` itself, so they cannot disagree
    with the weights.
    """
    sd = torch.load(ae_path, map_location="cpu", weights_only=True, mmap=True)
    matryoshka = "W_enc" in sd
    activation_dim, dict_size = (sd["W_enc"].shape if matryoshka
                                 else tuple(reversed(sd["encoder.weight"].shape)))
    return {
        "model_id": model_id,
        "checkpoint": checkpoint,
        "revision": revision,
        "layer": int(layer),
        "site": "resid_post",
        "tokens": "patch",
        "frames": int(frames),
        "token_grid": list(token_grid),
        "sae_class": "MatroyshkaBatchTopKSAE" if matryoshka else "BatchTopKSAE",
        "activation_dim": int(activation_dim),
        "dict_size": int(dict_size),
        "k": int(sd["k"]),
        "group_sizes": sd["group_sizes"].tolist() if "group_sizes" in sd else None,
        "threshold": float(sd["threshold"]),
        "input": INPUT_NOTE,
        "training": training,
        "data": data,
        "sha256": file_sha256(ae_path),
    }


def sae_dir(model_id: str, layer: int, weights_dir=None) -> Path:
    """The block-`layer` SAE's folder: `sae.dir` from the model's config, or
    `<weights_dir>/<model_id>/l<layer>` when `weights_dir` is given."""
    if weights_dir is not None:
        return Path(weights_dir) / model_id / f"l{layer}"
    return get_spec(model_id).sae_dir(layer)


def list_saes(weights_dir=None) -> dict[str, list[int]]:
    """{model_id: [layers with an SAE]}: where each config's `sae.dir` points,
    or everything under `weights_dir` when it is given."""
    out = {}
    if weights_dir is None:
        for model_id, spec in MODELS.items():
            layers = [L for L in range(spec.expect["blocks"])
                      if (spec.sae_dir(L) / "ae.pt").exists()]
            if layers:
                out[model_id] = layers
        return out
    for model_dir in sorted(p for p in Path(weights_dir).iterdir() if p.is_dir()):
        layers = sorted(int(p.name[1:]) for p in model_dir.glob("l*")
                        if p.name[1:].isdigit() and (p / "ae.pt").exists())
        if layers:
            out[model_dir.name] = layers
    return out


def load_config(model_id: str, layer: int, weights_dir=None) -> dict:
    path = sae_dir(model_id, layer, weights_dir) / "config.json"
    if not path.exists():
        available = list_saes(weights_dir).get(model_id)
        hint = (f"layers with an SAE for {model_id}: {available}" if available
                else f"model IDs with SAEs: {list(list_saes(weights_dir))}")
        raise FileNotFoundError(f"no SAE at {path.parent} ({hint})")
    return json.loads(path.read_text())


def load_sae(
    model_id: str,
    layer: int,
    device: str = "cpu",
    weights_dir=None,
    backbone: Optional[object] = None,
):
    """Load the SAE for `model_id` at `layer`; `sae.config` holds its config.json.

    Pass the loaded `backbone` (from `models.get_model`) to also check that it
    is the checkpoint + revision + width the SAE was fitted on.
    """
    cfg = load_config(model_id, layer, weights_dir)
    if cfg["model_id"] != model_id or cfg["layer"] != layer:
        raise ValueError(f"{sae_dir(model_id, layer, weights_dir)}/config.json describes "
                         f"{cfg['model_id']} L{cfg['layer']}, not {model_id} L{layer}")
    if backbone is not None:
        got = (getattr(backbone, "model_id", None), getattr(backbone, "revision", None),
               backbone.hidden_dim)
        want = (cfg["model_id"], cfg["revision"], cfg["activation_dim"])
        if got != want:
            raise ValueError(f"SAE was fitted on (model_id, revision, width) = {want}; "
                             f"the backbone passed is {got}")
        if cfg.get("frames") not in (None, backbone.num_frames):
            print(f"note: this SAE was fitted on {cfg['frames']}-frame activations; "
                  f"the backbone runs at {backbone.num_frames} frames", flush=True)

    sae = SAE_CLASSES[cfg["sae_class"]].from_pretrained(
        str(sae_dir(model_id, layer, weights_dir) / "ae.pt"), device=device)
    if (int(sae.activation_dim), int(sae.dict_size)) != (cfg["activation_dim"], cfg["dict_size"]):
        raise ValueError(f"ae.pt is {sae.activation_dim} -> {sae.dict_size}, config.json says "
                         f"{cfg['activation_dim']} -> {cfg['dict_size']}")
    sae.config = cfg
    return sae.eval()
