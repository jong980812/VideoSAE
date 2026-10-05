"""Evaluate a backbone on a dataset: clean, or with an SAE spliced in.

    python scripts/evaluate.py --model videoprism-base-f16r288                     # Kinetics-400, clean
    python scripts/evaluate.py --model videoprism-base-f16r288 --layer 11          # ... SAE at block 11
    python scripts/evaluate.py --model videoprism-base-f16r288 --dataset nturgbd --layer 11

`--dataset` names a config in config/datasets/ (kinetics400 by default; also
nturgbd, hat, ssv2). It says where the data lives (an .env entry), which clip
list to score, and which label space -- and so which head of the model's config
-- scores it. One run measures one condition:

    clean   (no --layer) the untouched model: `model.encode(clips)`
    sae     (--layer N)  block N's output replaced by the layer-N SAE's
            reconstruction (`sae.dir` in the model's config) -- every latent
            kept, nothing edited:

                patches = model.encode(clips, layer=N)
                feats   = model.decode(sae.decode(sae.encode(patches)), layer=N)

            (For ViViT only the patch tokens go through the SAE; the CLS token,
            which the SAEs never saw, passes untouched.)

`decode` runs blocks N+1.. and the model's ending through code of its own, so an
sae run first checks, on the first batch of every session, that
decode(encode(x, layer=N)) reproduces the full forward to within 1e-3 (relative),
and stops if not: a drift there would move every SAE number and show nowhere else.

Both are scored through the model's FIXED head for that label space (`head:` in
config/<model_id>.yaml). Writes one JSON file you can open and read from the top:

    results/<model_id>/<dataset>/clean.json
    results/<model_id>/<dataset>/sae_l<layer>.json

    {
      "summary": {"top1": ..., "top5": ..., "n_correct": ..., "n_items": 19877, ...},
      "config":  {checkpoint, revision, head, SAE, clip list, batch size, ...},
      "items": [
        {"key": "abseiling/0wR5jVB-WPk_000417_000427", "label": 0, "label_name": "abseiling",
         "pred": 0, "pred_name": "abseiling", "correct": true, "prob": 0.91,
         "top5": [0, 312, ...]},
        ...                                              one line per clip
      ]
    }

HAT (person of one clip on the background of another) reports, instead of
top1: `hu_acc` (the person's action predicted), `bg_err` (the background's
action predicted), `hu_acc_binary` (the person's logit beats the background's)
and `bg_hu_ratio` = bg_err / hu_acc; its items also carry the background clip.
NTU is scored over the 60 classes present, so chance is 1/60.

An `sae` summary also reports `agreement_with_clean` -- the fraction of clips whose
prediction the splice leaves unchanged -- when clean.json already exists.

RESUMABLE. Each finished batch is appended to
`results/<model_id>/<dataset>/.progress/<name>.jsonl`. If a run stops (time
limit, crash, Ctrl-C), run the same command again: it checks that the settings
match and continues from the last finished batch, so batches -- and so
predictions -- are exactly those of an uninterrupted run. The progress file is
deleted once the result file is written.

ALWAYS THE FULL SET. Each dataset's clip list is its full evaluation split, and
results measured on it are comparable across models, layers and months.
`--limit N` (N evenly spaced clips) exists for smoke tests only; its output goes
to `<dataset>_smoke/` so it can never be mistaken for a real number.

An existing result file is not re-measured; `--force` discards it (and any
progress) and starts over. To split one evaluation over several GPUs, run
`--num_tasks N --task_id i` for each i (each task is resumable on its own), then
once more with `--merge`.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.clips import stride_subsample                    # noqa: E402
from data.datasets import (DATASETS, build_dataset, check_roots, dataset_roots,  # noqa: E402
                           dataset_split, get_dataset)
from models import MODELS, get_model, get_spec             # noqa: E402
from models.heads import build_head                        # noqa: E402
from saes import load_sae, sae_dir                         # noqa: E402


def get_args_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--layer", type=int, default=None,
                   help="splice the SAE of this block (condition 'sae'); "
                        "omit it to measure the clean model")
    p.add_argument("--dataset", default="kinetics400", choices=list(DATASETS),
                   help="a config in config/datasets/")
    p.add_argument("--data_root", type=str, default=None,
                   help="override the dataset's root (default: its .env entry, e.g. K400_VAL); "
                        "only for datasets with a single root")
    p.add_argument("--results_dir", default=str(ROOT / "results"), type=str)
    p.add_argument("--batch_size", default=None, type=int,
                   help="default: eval.batch_size in the model's config")
    p.add_argument("--workers", default=7, type=int)
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--limit", default=None, type=int,
                   help="SMOKE TEST: N evenly spaced clips; written under <dataset>_smoke/")
    p.add_argument("--force", action="store_true",
                   help="discard an existing result file and any saved progress, start over")
    p.add_argument("--num_tasks", default=1, type=int)
    p.add_argument("--task_id", default=0, type=int)
    p.add_argument("--merge", action="store_true",
                   help="combine the --num_tasks parts into the result files, then stop")
    return p


# ------------------------------------------------------------------- writing


def write_result(path: Path, summary: dict, config: dict, items: list):
    """JSON with the summary first and one line per item, so it reads top-down."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        f.write("{\n")
        f.write(f'  "summary": {json.dumps(summary, indent=2).replace(chr(10), chr(10) + "  ")},\n')
        f.write(f'  "config": {json.dumps(config, indent=2).replace(chr(10), chr(10) + "  ")},\n')
        f.write('  "items": [\n')
        for i, item in enumerate(items):
            f.write("    " + json.dumps(item) + (",\n" if i < len(items) - 1 else "\n"))
        f.write("  ]\n}\n")
    tmp.replace(path)


def summarize(items: list, n_classes: int, condition: str, metrics: str,
              ident: dict, extra: dict) -> dict:
    """Accuracy right after what was measured, bookkeeping after that."""
    r = lambda x: round(float(x), 6)
    correct = np.array([it["correct"] for it in items])
    top5 = np.array([it["label"] in it["top5"] for it in items])
    if metrics == "hat":
        hu, bg = correct.mean(), np.mean([it["bg_hit"] for it in items])
        scores = {"hu_acc": r(hu), "bg_err": r(bg),
                  "hu_acc_binary": r(np.mean([it["margin"] > 0 for it in items])),
                  "bg_hu_ratio": r(bg / hu) if hu > 0 else None,
                  "top5": r(top5.mean())}
    else:
        scores = {"top1": r(correct.mean()), "top5": r(top5.mean())}
    return {
        **ident,
        "condition": condition,
        **scores,
        "n_correct": int(correct.sum()),
        "n_items": len(items),
        **extra,
        "n_classes": n_classes,
        "chance": round(1.0 / n_classes, 6),
        "distinct_preds": len({it["pred"] for it in items}),
        "n_substituted": sum("read" in it for it in items),
    }


# ------------------------------------------------------------------ progress
#
# <name>[.task<i>of<n>].jsonl, one JSON object per line, appended as it goes:
#     {"header":  {...}}                   once: what is being measured, and how
#     {"session": {...}}                   each time a run starts or resumes
#     {"start": i, "seconds": s, "items": [...]}   each finished batch
# A run killed mid-write leaves a torn last line; it is cut off on resume.

# Header fields that must match for saved progress to be continued.
IDENTITY = ("model_id", "dataset", "condition", "layer", "split_sha256", "limit", "batch_size",
            "num_tasks", "task_id", "lo", "hi")


def progress_path(results_dir: Path, model_id: str, dataset: str, name: str,
                  num_tasks: int, task_id: int) -> Path:
    task = f".task{task_id}of{num_tasks}" if num_tasks > 1 else ""
    return results_dir / model_id / dataset / ".progress" / f"{name}{task}.jsonl"


def read_progress(path: Path, identity: dict):
    """-> (header or None, items, sessions, seconds) of the saved progress."""
    if not path.exists():
        return None, [], [], 0.0
    raw = path.read_bytes()
    complete, _, torn = raw.rpartition(b"\n")
    if torn:                                   # killed mid-write: drop the partial line
        with open(path, "r+b") as f:
            f.truncate(len(complete) + 1 if complete else 0)
    records = [json.loads(line) for line in complete.split(b"\n") if line.strip()]
    if not records:
        return None, [], [], 0.0
    header = records[0].get("header")
    if header is None:
        raise SystemExit(f"{path} does not start with a header; delete it or use --force")
    diff = {k: (header.get(k), identity[k]) for k in IDENTITY if header.get(k) != identity[k]}
    if diff:
        raise SystemExit(
            f"{path} holds progress for different settings -- {diff} (saved, now). "
            f"Resume with the original settings, or pass --force to start over.")
    items, sessions, seconds = [], [], 0.0
    for r in records[1:]:
        if "session" in r:
            sessions.append(r["session"])
            continue
        if r["start"] != identity["lo"] + len(items):
            raise SystemExit(f"{path}: batch at clip {r['start']} does not follow clip "
                             f"{identity['lo'] + len(items) - 1}; delete it or use --force")
        items += r["items"]
        seconds += r["seconds"]
    return header, items, sessions, seconds


def append_line(f, record: dict):
    f.write(json.dumps(record) + "\n")
    f.flush()
    os.fsync(f.fileno())


# ------------------------------------------------------------------- scoring

RESUME_TOL = 1e-3


@torch.no_grad()
def check_resume(model, px, layer: int) -> float:
    """Relative deviation of decode(encode(x, layer)) from the full forward;
    stops the run above RESUME_TOL."""
    clean = model.encode({"pixel_values": px}).float()
    resumed = model.decode(model.encode({"pixel_values": px}, layer=layer), layer=layer).float()
    dev = ((resumed - clean).abs().max() / clean.abs().max().clamp_min(1e-12)).item()
    print(f"resume check, block {layer}: decode(encode(x)) vs the full forward, "
          f"max relative deviation {dev:.2e}", flush=True)
    if dev > RESUME_TOL:
        raise SystemExit(f"decode(encode(x, layer={layer})) differs from the full forward by "
                         f"{dev:.2e} (> {RESUME_TOL:g}); the sae condition would measure that too")
    return dev


@torch.no_grad()
def score(model, head, sae, layer, loader, samples, start, lo, hi, classes, score_cols,
          device, progress):
    """Score clips start..hi-1, appending one line per batch to `progress`.

    `score_cols` restricts the answer to those classes (NTU: the 60 present of the
    head's 120): argmax, top-5 and prob are taken over them only.
    """
    cols = None if score_cols is None else torch.as_tensor(score_cols, device=device)
    bar = tqdm(total=hi - lo, initial=start - lo, desc="eval", unit="clip", file=sys.stdout,
               dynamic_ncols=True, mininterval=5.0)
    i, t = start, time.time()
    checked = sae is None
    for batch in loader:
        px, labels, read = batch["pixel_values"], batch["labels"].tolist(), batch["paths"]
        if not checked:
            check_resume(model, px, layer)
            checked, t = True, time.time()        # not billed to the first batch
        if sae is None:
            feats = model.encode({"pixel_values": px})
        else:
            patches = model.encode({"pixel_values": px}, layer=layer)
            feats = model.decode(sae.decode(sae.encode(patches)), layer=layer)
        logits = head(feats.float())
        sub = logits if cols is None else logits[:, cols]
        best = sub.argmax(1)
        pmax = sub.softmax(1).gather(1, best[:, None]).squeeze(1).cpu().tolist()
        top5 = sub.topk(min(5, sub.shape[1]), dim=1).indices
        if cols is not None:                        # back to label-space indices
            best, top5 = cols[best], cols[top5]
        pred, top5 = best.cpu().tolist(), top5.cpu().tolist()
        bg = batch.get("bg_labels")
        if bg is not None:                          # HAT: person logit minus background logit
            margin = (logits.gather(1, batch["labels"].to(device)[:, None])
                      - logits.gather(1, bg.to(device)[:, None])).squeeze(1).cpu().tolist()
            bg = bg.tolist()
        items = []
        for b in range(len(pred)):
            key = samples[i + b][0]
            item = {"key": key, "label": labels[b], "label_name": classes[labels[b]],
                    "pred": pred[b], "pred_name": classes[pred[b]],
                    "correct": pred[b] == labels[b], "prob": round(pmax[b], 5), "top5": top5[b]}
            if bg is not None:
                item.update(bg_key=batch["bg_keys"][b], bg_label=bg[b], bg_name=classes[bg[b]],
                            bg_hit=pred[b] == bg[b], margin=round(margin[b], 5))
            if read[b] != key:          # unreadable clip: its neighbour was scored
                item["read"] = read[b]
            items.append(item)
        now = time.time()
        append_line(progress, {"start": i, "seconds": round(now - t, 3), "items": items})
        i, t = i + len(items), now
        bar.update(len(items))
    bar.close()


# ----------------------------------------------------------------- results


def result_path(results_dir: Path, model_id: str, dataset: str, name: str) -> Path:
    return results_dir / model_id / dataset / f"{name}.json"


def finalize(args, dataset, name, header, items, sessions, seconds):
    """Write the result file from finished progress."""
    ident = {"model_id": args.model, "dataset": dataset}
    extra = {}
    if header["condition"] == "sae":
        ident["layer"] = args.layer
        clean_file = result_path(Path(args.results_dir), args.model, dataset, "clean")
        if clean_file.exists():
            clean = json.loads(clean_file.read_text())
            if (clean["config"]["split"]["sha256"] == header["split"]["sha256"]
                    and len(clean["items"]) == len(items)):
                extra["agreement_with_clean"] = round(float(np.mean(
                    [a["pred"] == b["pred"] for a, b in zip(items, clean["items"])])), 6)
    ds = header["dataset_config"]
    summary = summarize(items, ds["n_classes"], header["condition"], ds["metrics"], ident, extra)
    config = {k: header[k] for k in ("model_id", "checkpoint", "revision", "dataset_config", "head",
                                     "sae", "frames", "batch_size", "dtype", "split",
                                     "limit", "num_tasks")
              if header.get(k) is not None}
    config.update(sessions=sessions, seconds=round(seconds, 1),
                  written=time.strftime("%Y-%m-%d %H:%M:%S"))
    out = result_path(Path(args.results_dir), args.model, dataset, name)
    write_result(out, summary, config, items)
    if ds["metrics"] == "hat":
        shown = (f"hu_acc {summary['hu_acc']:.2%}  bg_err {summary['bg_err']:.2%}  "
                 f"hu_acc_binary {summary['hu_acc_binary']:.2%}")
    else:
        shown = f"top1 {summary['top1']:.2%}  top5 {summary['top5']:.2%}"
    shown += f"  ({summary['n_correct']:,}/{summary['n_items']:,})"
    if "agreement_with_clean" in summary:
        shown += f"  agreement with clean {summary['agreement_with_clean']:.2%}"
    print(f"{name}: {shown}\n  -> {out}", flush=True)


def main(args):
    spec = get_spec(args.model)
    condition = "clean" if args.layer is None else "sae"
    name = "clean" if args.layer is None else f"sae_l{args.layer}"
    args.batch_size = args.batch_size or spec.eval_batch_size

    ds_spec = get_dataset(args.dataset)
    split = dataset_split(ds_spec)
    classes, samples = split["classes"], split["samples"]
    score_cols = None
    if ds_spec.get("restrict_to_score_classes"):
        score_cols = split["score_classes"]
        if not score_cols:
            raise SystemExit(f"{ds_spec['clips']} has no score_classes to restrict to")
    dataset = ds_spec["name"]
    if args.limit is not None:
        samples = stride_subsample(samples, args.limit)
        dataset += "_smoke"
    results_dir = Path(args.results_dir)

    out = result_path(results_dir, args.model, dataset, name)
    if out.exists() and not args.force:
        print(f"already measured: {out} (--force re-measures)")
        return
    per = -(-len(samples) // args.num_tasks)

    def identity(task_id):
        return {"model_id": args.model, "dataset": dataset, "condition": condition,
                "layer": args.layer, "split_sha256": split["sha256"], "limit": args.limit,
                "batch_size": args.batch_size, "num_tasks": args.num_tasks, "task_id": task_id,
                "lo": task_id * per, "hi": min((task_id + 1) * per, len(samples))}

    if args.merge:
        return merge(args, dataset, name, identity)

    ident = identity(args.task_id)
    lo, hi = ident["lo"], ident["hi"]
    prog = progress_path(results_dir, args.model, dataset, name, args.num_tasks, args.task_id)
    if args.force and prog.exists():
        prog.unlink()
    header, items, sessions, seconds = read_progress(prog, ident)
    if len(items) < hi - lo:
        roots = dataset_roots(ds_spec, args.data_root)
        check_roots(ds_spec, roots, samples)
        model = get_model(args.model, device=args.device)
        head, head_info = build_head(model, ds_spec["label_space"], classes, args.device)
        sae, sae_info = None, None
        if condition == "sae":
            sae = load_sae(args.model, args.layer, device=args.device, backbone=model)
            ae_path = sae_dir(args.model, args.layer) / "ae.pt"      # sae.dir in the config
            sae_info = {"path": str(ae_path.relative_to(ROOT) if ae_path.is_relative_to(ROOT)
                                    else ae_path),
                        "sha256": sae.config["sha256"], "dict_size": sae.config["dict_size"],
                        "k": sae.config["k"], "fitted_on_frames": sae.config.get("frames"),
                        "encode": "learned threshold (sae.encode default)",
                        "splice": "decode(sae.decode(sae.encode(encode(x, layer))), layer); "
                                  "patch tokens only, any CLS token passed through",
                        "resume_check": f"decode(encode(x, layer)) vs the full forward on each "
                                        f"session's first batch, relative tolerance {RESUME_TOL:g}"}
        dataset_config = {"name": ds_spec["name"], "label_space": ds_spec["label_space"],
                          "loader": ds_spec["loader"], "metrics": ds_spec["metrics"],
                          "n_classes": len(score_cols) if score_cols else len(classes),
                          "restricted_to_score_classes": score_cols is not None}
        now = dict(ident, checkpoint=spec.checkpoint, revision=spec.revision,
                   dataset_config=dataset_config, head=head_info, sae=sae_info,
                   frames=int(model.num_frames), dtype="float32",
                   split={k: split[k] for k in ("file", "split", "sha256", "description", "short_side")})
        if header is not None:
            # not in IDENTITY, because they need the model loaded to know
            for k in ("revision", "frames", "dataset_config", "head", "sae"):
                if header.get(k) != now[k]:
                    raise SystemExit(f"{prog}: saved progress used {k}={header.get(k)}, now "
                                     f"{now[k]}. Pass --force to start over.")
        start = lo + len(items)
        ds, collate = build_dataset(ds_spec, split, model, roots, samples=samples)
        loader = DataLoader(Subset(ds, range(start, hi)), batch_size=args.batch_size,
                            num_workers=args.workers, shuffle=False, collate_fn=collate)
        print(f"{args.model} {name} on {dataset}: clips {lo:,}-{hi - 1:,} of {len(samples):,}, "
              f"{model.num_frames} frames, batch {args.batch_size}, "
              f"{ds_spec['label_space']} head ({head_info['type']})"
              + (f"; resuming at clip {start:,} ({len(items):,} done)" if items else ""), flush=True)
        prog.parent.mkdir(parents=True, exist_ok=True)
        with open(prog, "a") as f:
            if header is None:
                append_line(f, {"header": now})
            append_line(f, {"session": {
                "started": time.strftime("%Y-%m-%d %H:%M:%S"), "from_clip": start,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "data_roots": {k: str(Path(v).resolve()) for k, v in roots.items()}}})
            score(model, head, sae, args.layer, loader, samples, start, lo, hi, classes,
                  score_cols, args.device, f)
        header, items, sessions, seconds = read_progress(prog, ident)
        if len(items) != hi - lo:
            raise SystemExit(f"{prog}: {len(items):,} of {hi - lo:,} clips scored; rerun to resume")

    if args.num_tasks > 1:
        print(f"task {args.task_id}/{args.num_tasks} complete ({len(items):,} clips) in {prog}; "
              f"run with --merge once every task is complete")
        return
    finalize(args, dataset, name, header, items, sessions, seconds)
    prog.unlink()


def merge(args, dataset, name, identity):
    """Combine the tasks' finished progress files into the result file."""
    progs, merged, sessions, seconds, header = [], [], [], 0.0, None
    for t in tqdm(range(args.num_tasks), desc="merge", unit="task"):
        ident = identity(t)
        prog = progress_path(Path(args.results_dir), args.model, dataset, name, args.num_tasks, t)
        h, items, s, sec = read_progress(prog, ident)
        if len(items) != ident["hi"] - ident["lo"]:
            raise SystemExit(f"task {t}: {len(items):,} of {ident['hi'] - ident['lo']:,} clips "
                             f"scored ({prog}); finish that task first")
        header = header or h
        merged += items
        sessions += s
        seconds += sec
        progs.append(prog)
    finalize(args, dataset, name, header, merged, sessions, seconds)
    for p in progs:
        p.unlink()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
