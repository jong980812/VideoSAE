"""Matryoshka BatchTopK vs plain BatchTopK on the held-out layer-7 activations.

    .venv/bin/python runs/mat_vs_btk/analyze.py

Per SAE (thresholded encode, the deployment behaviour):
  fve, l0, dead      as train_sae.py's metrics.json, recomputed here in one place
  dense              fraction of latents firing on more than 10% of tokens
  prefix fve         FVE when only the first 1/16, 3/16, 7/16 of the latents are
                     kept (the Matryoshka group boundaries). A plain BatchTopK
                     dictionary has no order, so its prefix is a random subset.
  max-cos            each decoder atom's largest cosine with another atom: mean,
                     and the fraction above 0.9 (near-duplicate atoms)
Writes results.json next to this file.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data.activations import ActivationsDataset   # noqa: E402
from saes import load_sae                         # noqa: E402

MODEL, LAYER, DEVICE = "videomaev2-vitb-k710distill", 7, "cuda:0"
PREFIXES = (1 / 16, 3 / 16, 7 / 16)


def decoder_atoms(sae) -> torch.Tensor:
    """(dict_size, activation_dim) unit rows."""
    w = sae.W_dec if hasattr(sae, "W_dec") else sae.decoder.weight.T
    return torch.nn.functional.normalize(w.detach(), dim=1)


@torch.no_grad()
def measure(sae, acts: torch.Tensor) -> dict:
    n_dict = int(sae.dict_size)
    cuts = [int(f * n_dict) for f in PREFIXES]
    mean = acts.mean(0)
    total = float(((acts - mean) ** 2).sum())
    resid = 0.0
    resid_prefix = [0.0] * len(cuts)
    fired = torch.zeros(n_dict, device=acts.device)
    l0 = 0.0
    for x in acts.split(8192):
        codes = sae.encode(x)
        resid += float(((x - sae.decode(codes)) ** 2).sum())
        fired += (codes != 0).float().sum(0)
        l0 += float((codes != 0).sum())
        for i, cut in enumerate(cuts):
            kept = codes.clone()
            kept[:, cut:] = 0
            resid_prefix[i] += float(((x - sae.decode(kept)) ** 2).sum())
    freq = fired / acts.shape[0]

    atoms = decoder_atoms(sae)
    max_cos = torch.empty(n_dict, device=atoms.device)
    for s in range(0, n_dict, 2048):
        cos = atoms[s:s + 2048] @ atoms.T
        cos[torch.arange(cos.shape[0]), torch.arange(s, s + cos.shape[0])] = -1
        max_cos[s:s + 2048] = cos.max(1).values

    return {
        "fve": 1 - resid / total,
        "l0": l0 / acts.shape[0],
        "dead": float((fired == 0).float().mean()),
        "dense": float((freq > 0.10).float().mean()),
        "prefix_fve": {f"{c}": 1 - r / total for c, r in zip(cuts, resid_prefix)},
        "prefix_l0": {f"{c}": float(fired[:c].sum()) / acts.shape[0] for c in cuts},
        "max_cos_mean": float(max_cos.mean()),
        "max_cos_gt_0.9": float((max_cos > 0.9).float().mean()),
    }


def main():
    acts = torch.cat(ActivationsDataset(RUN / "acts_heldout" / f"l{LAYER}", device=DEVICE).cached_tensors)
    runs = {p.name: p for p in sorted((RUN / "weights").iterdir()) if p.is_dir()}
    results = {}
    for name, weights_dir in runs.items():
        results[name] = measure(load_sae(MODEL, LAYER, device=DEVICE, weights_dir=weights_dir), acts)
    # The shipped SAE, for reference only: it was fitted on JPEG-frame inputs, not these mp4s.
    results["shipped_matryoshka"] = measure(load_sae(MODEL, LAYER, device=DEVICE), acts)

    (RUN / "results.json").write_text(json.dumps(results, indent=2))
    cuts = list(next(iter(results.values()))["prefix_fve"])
    print(f"{'run':<28}{'fve':>7}{'l0':>7}{'dead':>7}{'dense':>7}"
          + "".join(f"{'fve@' + c:>10}" for c in cuts) + f"{'maxcos':>8}{'>0.9':>7}")
    for name, r in results.items():
        print(f"{name:<28}{r['fve']:>7.4f}{r['l0']:>7.1f}{r['dead']:>7.4f}{r['dense']:>7.4f}"
              + "".join(f"{r['prefix_fve'][c]:>10.4f}" for c in cuts)
              + f"{r['max_cos_mean']:>8.3f}{r['max_cos_gt_0.9']:>7.4f}")


if __name__ == "__main__":
    main()
