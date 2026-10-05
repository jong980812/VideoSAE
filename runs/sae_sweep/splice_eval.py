"""Kinetics-400 accuracy of videomaev2-base with each sweep SAE spliced in, all
SAEs scored on the same clips in one pass (the splice is evaluate.py's:
decode(sae.decode(sae.encode(encode(x, layer))), layer), then the kinetics400 head).

    .venv/bin/python runs/sae_sweep/splice_eval.py --split val4k --num_tasks 8 --task_id 0 --device cuda:0
    ...
    .venv/bin/python runs/sae_sweep/splice_eval.py --split val4k --num_tasks 8 --merge

Scores every run under weights/ (or the ones named with --runs). Each task writes
splice/<split>/task<i>.pt; --merge writes splice/<split>.json:
    {"clean": {top1, top5, n}, "<run>": {top1, top5, agree_with_clean, layer}, ...}
A run already in <split>.json is not scored again unless --force.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data.clips import ClipList, load_split, make_frame_collate_fn   # noqa: E402
from models import get_model                                         # noqa: E402
from models.heads import build_head                                  # noqa: E402
from saes import load_sae                                            # noqa: E402

MODEL = "videomaev2-base"


def sae_runs(names) -> dict:
    """{run name: layer} for each run under weights/ that has a finished SAE."""
    out = {}
    for d in sorted((RUN / "weights").iterdir()):
        if names and d.name not in names:
            continue
        for ae in (d / MODEL).glob("l*/ae.pt"):
            out[d.name] = int(ae.parent.name[1:])
    return out


def result_file(split: str) -> Path:
    return RUN / "splice" / f"{split}.json"


@torch.no_grad()
def score(args, runs: dict):
    split = load_split(RUN / "clips.json", args.split)
    samples = split["samples"]
    lo = len(samples) * args.task_id // args.num_tasks
    hi = len(samples) * (args.task_id + 1) // args.num_tasks

    model = get_model(MODEL, device=args.device)
    head, _ = build_head(model, "kinetics400", split["classes"], args.device)
    saes = {name: load_sae(MODEL, layer, device=args.device, weights_dir=RUN / "weights" / name,
                           backbone=model) for name, layer in runs.items()}
    layers = sorted(set(runs.values()))

    ds = ClipList(str(RUN / "k400_val"), samples, model.num_frames, split["short_side"])
    loader = DataLoader(Subset(ds, range(lo, hi)), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False,
                        collate_fn=make_frame_collate_fn(model.processor))

    labels, top5 = [], {name: [] for name in ["clean", *runs]}
    for b, batch in enumerate(loader):
        px = {"pixel_values": batch["pixel_values"]}
        clean = model.encode(px).float()
        top5["clean"].append(head(clean).topk(5, dim=-1).indices.cpu())
        labels.append(batch["labels"])
        patches = model.encode(px, layer=layers)
        if b == 0:      # the wrapper's resume must reproduce the full forward, as in evaluate.py
            for L in layers:
                dev = float((model.decode(patches[L], layer=L).float() - clean).norm() / clean.norm())
                if dev > 1e-3:
                    raise SystemExit(f"decode(encode(x, layer={L})) differs from the full forward by {dev:.2e}")
        for name, sae in saes.items():
            L = runs[name]
            feats = model.decode(sae.decode(sae.encode(patches[L])), layer=L).float()
            top5[name].append(head(feats).topk(5, dim=-1).indices.cpu())
        if b % 20 == 0:
            print(f"task {args.task_id}: batch {b}/{len(loader)}", flush=True)

    out = RUN / "splice" / args.split
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"lo": lo, "hi": hi, "labels": torch.cat(labels), "runs": runs,
                "top5": {k: torch.cat(v) for k, v in top5.items()}}, out / f"task{args.task_id}.pt")


def merge(args, runs: dict):
    parts = [torch.load(RUN / "splice" / args.split / f"task{i}.pt") for i in range(args.num_tasks)]
    n_listed = len(load_split(RUN / "clips.json", args.split)["samples"])
    labels = torch.cat([p["labels"] for p in parts])
    if len(labels) != n_listed or any(set(p["runs"]) != set(runs) for p in parts):
        raise SystemExit("the task files do not cover this split with these runs; re-run the tasks")
    top5 = {k: torch.cat([p["top5"][k] for p in parts]) for k in parts[0]["top5"]}

    path = result_file(args.split)
    results = json.loads(path.read_text()) if path.exists() else {}
    clean_pred = top5["clean"][:, 0]
    for name, t5 in top5.items():
        r = {"top1": float((t5[:, 0] == labels).float().mean()),
             "top5": float((t5 == labels[:, None]).any(1).float().mean()),
             "n": len(labels)}
        if name != "clean":
            r["agree_with_clean"] = float((t5[:, 0] == clean_pred).float().mean())
            r["layer"] = runs[name]
        results[name] = r
    path.write_text(json.dumps(results, indent=2))
    for name, r in results.items():
        print(f"{name:<18} top1 {r['top1']:.4f}  top5 {r['top5']:.4f}  "
              f"agree {r.get('agree_with_clean', 1):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val4k")
    p.add_argument("--runs", nargs="*", help="default: every finished run not yet in <split>.json")
    p.add_argument("--force", action="store_true")
    p.add_argument("--num_tasks", type=int, default=1)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()

    runs = sae_runs(args.runs)
    path = result_file(args.split)
    if path.exists() and not args.force:
        scored = set(json.loads(path.read_text()))
        runs = {n: L for n, L in runs.items() if n not in scored}
    if not runs:
        print("nothing to score")
        return
    if args.merge:
        merge(args, runs)
    else:
        score(args, runs)


if __name__ == "__main__":
    main()
