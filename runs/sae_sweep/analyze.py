"""One table for the sweep: every run under weights/, scored on the held-out
activations of its layer (thresholded encode, the deployment behaviour).

    .venv/bin/python runs/sae_sweep/analyze.py            # -> results.json + a table

  fve / l0 / dead   reconstruction, sparsity, latents that never fire (held-out, 128k tokens)
  fve_train         the same FVE on 128k rows the SAE was trained on (fve_train - fve = overfit)
  dense             fraction of latents firing on more than 10% of tokens
  first16           of the active latents per token, how many sit in the first 1/16 of the
                    dictionary, and the FVE those alone give (the Matryoshka inner group)
  top1 / agree      Kinetics-400 accuracy with the SAE spliced in and agreement with the
                    clean model's prediction, from splice/<split>.json when it exists
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUN))
from data.activations import ActivationsDataset   # noqa: E402
from saes import load_sae                         # noqa: E402
from sweep import BASE, MODEL, RUNS               # noqa: E402

DEVICE = "cuda:0"
N_TRAIN_ROWS = 128_000


def rows(directory, n=None) -> torch.Tensor:
    """A layer's activation rows on DEVICE; `n` evenly spaced ones when given."""
    x = torch.cat(ActivationsDataset(directory, device="cpu").cached_tensors)
    if n is not None and n < len(x):
        x = x[torch.linspace(0, len(x) - 1, n).long()]
    return x.to(DEVICE)


@torch.no_grad()
def measure(sae, acts: torch.Tensor) -> dict:
    n_dict = int(sae.dict_size)
    cut = n_dict // 16
    total = float(((acts - acts.mean(0)) ** 2).sum())
    resid = resid_first = l0 = 0.0
    fired = torch.zeros(n_dict, device=acts.device)
    for x in acts.split(4096):
        codes = sae.encode(x)
        resid += float(((x - sae.decode(codes)) ** 2).sum())
        fired += (codes != 0).float().sum(0)
        l0 += float((codes != 0).sum())
        codes[:, cut:] = 0
        resid_first += float(((x - sae.decode(codes)) ** 2).sum())
    n = acts.shape[0]
    return {"fve": 1 - resid / total, "l0": l0 / n,
            "dead": float((fired == 0).float().mean()),
            "dense": float((fired / n > 0.10).float().mean()),
            "first16_l0": float(fired[:cut].sum()) / n,
            "first16_fve": 1 - resid_first / total}


def main():
    path = RUN / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    cache = {}
    for d in sorted((RUN / "weights").iterdir()):
        name = d.name
        cfg = {**BASE, **RUNS.get(name, {})}
        if name in results or not (d / MODEL / f"l{cfg['layer']}" / "ae.pt").exists():
            continue
        L = cfg["layer"]
        if ("heldout", L) not in cache:
            cache[("heldout", L)] = rows(RUN / "acts" / "heldout" / f"l{L}")
        if (cfg["data"], L) not in cache:
            cache[(cfg["data"], L)] = rows(RUN / "acts" / cfg["data"] / f"l{L}", N_TRAIN_ROWS)
        sae = load_sae(MODEL, L, device=DEVICE, weights_dir=d)
        r = measure(sae, cache[("heldout", L)])
        r["fve_train"] = measure(sae, cache[(cfg["data"], L)])["fve"]
        r["config"] = {k: v for k, v in cfg.items() if k != "fractions"}
        r["dict_size"] = int(sae.dict_size)
        results[name] = r
        path.write_text(json.dumps(results, indent=2))

    splice = {}
    for f in sorted((RUN / "splice").glob("*.json")) if (RUN / "splice").exists() else []:
        splice[f.stem] = json.loads(f.read_text())
    cols = [(s, k) for s in splice for k in ("top1", "agree_with_clean")]
    print(f"{'run':<16}{'L':>3}{'k':>4}{'x':>4}{'fve':>8}{'train':>8}{'l0':>7}{'dead':>7}{'dense':>7}"
          f"{'f16 l0':>8}{'f16 fve':>9}" + "".join(f"{s + ':' + k[:5]:>14}" for s, k in cols))
    for name, r in sorted(results.items(), key=lambda kv: (kv[1]["config"]["layer"], kv[0])):
        c = r["config"]
        line = (f"{name:<16}{c['layer']:>3}{c['k']:>4}{c['x']:>4}{r['fve']:>8.4f}{r['fve_train']:>8.4f}"
                f"{r['l0']:>7.1f}{r['dead']:>7.3f}{r['dense']:>7.4f}{r['first16_l0']:>8.1f}"
                f"{r['first16_fve']:>9.4f}")
        for s, k in cols:
            v = splice[s].get(name, {}).get(k)
            line += f"{v:>14.4f}" if v is not None else f"{'':>14}"
        print(line)
    for s in splice:
        c = splice[s]["clean"]
        print(f"clean on {s}: top1 {c['top1']:.4f}  top5 {c['top5']:.4f}  (n={c['n']})")


if __name__ == "__main__":
    main()
