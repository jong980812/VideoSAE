"""Stage 1: resid_post activations at many layers, in one forward pass.

    python scripts/extract_activations.py --model videomaev2-vitb-k710distill --out_dir acts/videomaev2-vitb-k710distill

(the clips come from K400_TRAIN in .env; --data_root overrides it), or split over GPUs:

    python scripts/extract_activations.py ... --num_tasks 8 --task_id i    # x8, any order
    python scripts/extract_activations.py ... --merge                      # once, after all 8

Writes, per layer, the shard layout train_sae.py reads:

    <out_dir>/clips.json                    the clip list used, in order
    <out_dir>/l{N}/patch_l{N}_part{i}.pt    fp16 rows: tokens_per_clip per clip, clip order
    <out_dir>/l{N}/meta.json                model ID, layer, grid, ... (written by the merge)

Every default -- which layers, batch size, shard size -- comes from the
`extract:` section of config/<model_id>.yaml, so `--model` is the only choice
that has to be made.

THE CLIP LIST
-------------
`--clip_list` / `--split` default to datafile/kinetics400.json, split `train_sae`:
the exact 5,000 Kinetics-400 train clips the shipped SAEs were fitted on (evenly
spaced through the sorted 240,258-clip listing, so every class is present). Keys
are `<class>/<clip_id>`; each resolves under the data root (K400_TRAIN in .env,
or `--data_root`) to a frame directory or an .mp4 (see data/clips.py). To fit on
other data, write a new list (same format, one split) with

    python scripts/extract_activations.py --plan --data_root <root> --n_clips N --clip_list new.json

which reads one directory per class rather than stat-ing every clip.

WHY ONE PASS
------------
`model.encode(clips, layer=layers)` returns every layer from one forward, so 24
layers cost one pass over the clips rather than 24. The forward and the decode
feeding it are the expensive part; what each layer adds after it is a gather
and a copy.

SPLITTING OVER GPUS
-------------------
Task i takes a contiguous slice of the clip list and writes shards numbered from
i*1000+1, so the tasks never share a filename and train_sae.py, which orders
shards by that number, reads them back in clip order. Two things make the split
output the same data as one pass:

  * THE TOKEN PICK IS SEEDED PER CLIP, from (seed, clip's index in the list),
    so it does not depend on which task read the clip or what came before it.
  * AN UNREADABLE CLIP IS REPLACED BY THE SAME NEIGHBOUR. ClipList substitutes
    the next clip in its list; each task indexes the FULL list, so "next" is the
    clip one pass would have substituted. Substitutions are recorded in meta.json.

`--merge` checks that every task finished for every layer, that the rows add up
to clips x tokens and that no stray shard sits in a layer directory, and only
then writes meta.json.

TOKEN SUBSAMPLING
-----------------
`--tokens_per_clip` keeps a random subset of each clip's PATCH tokens (any CLS
token is dropped) -- 5,000 clips x 256 tokens = 1.28M rows per layer, the recipe
all shipped SAEs were fitted with. That is 2.0-2.9 GB per layer in fp16
(768-1,152 wide), 22-80 GB for a whole model: use `--layers` to extract a group,
train it, delete it, and move on (scripts/train_all_layers.sh does exactly that).

The subset is drawn **per clip and shared across layers**: the same token
positions are kept at every depth, so a row index means the same patch of the
same clip in every layer's file.

Of the shipped SAEs, only videomaev2-vitb-k710distill's were fitted on per-clip picks; the
other four drew theirs from one sequential stream in clip order. Same
distribution (256 uniform positions per clip), different rows.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.clips import (ClipList, check_data_root, list_clips, load_split,  # noqa: E402
                        make_frame_collate_fn, stride_subsample)
from models import MODELS, get_model, get_spec             # noqa: E402
from utils.paths import data_root                          # noqa: E402

DEFAULT_CLIP_LIST = ROOT / "datafile" / "kinetics400.json"
CLIPS = "clips.json"


def get_args_parser():
    p = argparse.ArgumentParser("resid_post activations at many layers, one pass")
    p.add_argument("--model", required=True, choices=list(MODELS),
                   help="model ID; picks the backbone class, checkpoint and defaults")
    p.add_argument("--data_root", type=str,
                   help="root the clip keys resolve under: <root>/<class>/<clip_id>/ "
                        "(frames) or <root>/<class>/<clip_id>.mp4 (default: K400_TRAIN "
                        "from .env)")
    p.add_argument("--out_dir", type=str, help="where the per-layer shards go")
    p.add_argument("--clip_list", default=str(DEFAULT_CLIP_LIST), type=str,
                   help="a datafile/*.json clip-list file")
    p.add_argument("--split", default="train_sae", type=str,
                   help="which split of --clip_list (also the split --plan writes)")
    p.add_argument("--layers", default=None, type=str,
                   help="inclusive range 'a-b' or a comma list; default: extract.layers "
                        "in the model's config (the ones with shipped SAEs)")
    p.add_argument("--tokens_per_clip", default=256, type=int)
    p.add_argument("--batch_size", default=None, type=int,
                   help="default: extract.batch_size in the model's config")
    p.add_argument("--workers", default=7, type=int)
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--rows_per_shard", default=None, type=int,
                   help="flush threshold; every extracted layer buffers up to this "
                        "many rows, so peak RAM ~ layers x rows_per_shard x width x 2 B. "
                        "Default: extract.rows_per_shard in the model's config.")
    p.add_argument("--num_tasks", default=1, type=int)
    p.add_argument("--task_id", default=0, type=int)
    p.add_argument("--merge", action="store_true",
                   help="check every task's output and write each layer's "
                        "meta.json, then stop. A one-task run merges itself.")
    p.add_argument("--limit", default=None, type=int,
                   help="SMOKE TEST: only the first N clips of the list; '_smoke' "
                        "is appended to --out_dir so the output is never mistaken "
                        "for a real dump")
    p.add_argument("--plan", action="store_true",
                   help="write a new clip list (--n_clips of --data_root) to "
                        "--clip_list and stop")
    p.add_argument("--n_clips", default=5000, type=int, help="with --plan")
    return p


def parse_layers(spec):
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x]


def plan(args):
    """Write a clip list (datafile format, one split named --split): `--n_clips`
    evenly spaced through `--data_root`'s listing."""
    classes, samples = list_clips(args.data_root)
    listed = len(samples)
    samples = stride_subsample(samples, args.n_clips)
    out = Path(args.clip_list)
    if out.exists():
        raise SystemExit(f"{out} exists; pick another --clip_list path")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "name": out.stem,
        "description": f"clips under {Path(args.data_root).name}",
        "classes": classes,
        "splits": {args.split: {
            "description": f"{len(samples)} of {listed} clips, evenly spaced through the "
                           f"sorted listing",
            "short_side": 256, "listed": listed, "samples": samples}},
    }, indent=1))
    print(f"planned {len(samples):,} of {listed:,} clips -> {out} (split {args.split!r})", flush=True)


@torch.no_grad()
def extract(args, layers, samples, clip_sha, clip_desc):
    out_dir = Path(args.out_dir)
    check_data_root(args.data_root, samples, env_name="K400_TRAIN")
    per = -(-len(samples) // args.num_tasks)
    lo, hi = args.task_id * per, min((args.task_id + 1) * per, len(samples))
    if lo >= hi:
        raise SystemExit(f"task {args.task_id}/{args.num_tasks} has no clips")

    model = get_model(args.model, device=args.device)
    bad = [L for L in layers if not 0 <= L < model.num_layers]
    if bad:
        raise SystemExit(f"{args.model} has blocks 0-{model.num_layers - 1}; got {bad}")
    grid = model.grid_shape
    n_patch = int(np.prod(grid))
    keep_n = min(args.tokens_per_clip or n_patch, n_patch)

    # A Subset of the FULL list rather than a list of the slice: an unreadable
    # clip is replaced by the next one in the list, and this keeps "next" the
    # clip a single pass would have read.
    ds = ClipList(args.data_root, samples, model.num_frames, args.short_side)
    loader = DataLoader(Subset(ds, range(lo, hi)), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False,
                        collate_fn=make_frame_collate_fn(model.processor))

    print(f"{args.model}: {model.num_layers} blocks, {model.num_frames} frames, "
          f"grid {grid} -> {n_patch} patch tokens, keeping {keep_n}/clip",
          flush=True)
    print(f"layers {layers}  ({len(layers)} from ONE forward)", flush=True)
    print(f"task {args.task_id}/{args.num_tasks}: clips {lo:,}-{hi - 1:,} of "
          f"{len(samples):,} -> {(hi - lo) * keep_n:,} rows per layer", flush=True)

    dirs = {L: out_dir / f"l{L}" for L in layers}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    buf = {L: [] for L in layers}
    shards = {L: [] for L in layers}
    rows = {L: 0 for L in layers}

    def flush(L):
        if not buf[L]:
            return
        name = f"patch_l{L}_part{args.task_id * 1000 + len(shards[L]) + 1}.pt"
        torch.save(torch.cat(buf[L], 0), dirs[L] / name)
        shards[L].append(name)
        rows[L] += sum(t.shape[0] for t in buf[L])
        buf[L] = []

    def pick(i):
        # Seeded by the clip's place in the planned list, so the same clip gets
        # the same tokens whichever task reads it.
        if keep_n >= n_patch:
            return np.arange(n_patch)
        return np.random.default_rng([args.seed, i]).choice(
            n_patch, size=keep_n, replace=False)

    substituted = []
    gi = lo
    for batch in tqdm(loader, unit="b", file=sys.stdout, dynamic_ncols=True,
                      mininterval=5.0):
        # {L: (B, n_patch, D)}: patch tokens only, frame-major -- ViViT's CLS
        # token dropped, VideoPrism's temporal-block outputs back in (T, H, W) order.
        acts = model.encode({"pixel_values": batch["pixel_values"]}, layer=layers)
        b = batch["pixel_values"].shape[0]
        for i, got in enumerate(batch["paths"]):
            if got != samples[gi + i][0]:
                substituted.append({"index": gi + i, "planned": samples[gi + i][0], "read": got})
        # One subset per clip, reused for every layer, so a row index refers to
        # the same patch of the same clip at every depth.
        sel = [pick(gi + i) for i in range(b)]
        gi += b
        for L in layers:
            # fp16 halves both the transfer off the GPU and the file size.
            flat = acts.pop(L).to(torch.float16).cpu()     # (B, n_patch, D)
            if flat.shape[0] != b or flat.shape[1] != n_patch:
                raise SystemExit(f"L{L}: got {tuple(flat.shape[:2])}, "
                                 f"expected ({b}, {n_patch})")
            for i in range(b):
                buf[L].append(flat[i, sel[i]])
            if sum(t.shape[0] for t in buf[L]) >= args.rows_per_shard:
                flush(L)

    if substituted:
        print(f"  {len(substituted)} unreadable clip(s) replaced by their neighbour "
              f"(recorded in meta.json)", flush=True)
    for L in layers:
        flush(L)
        part = {"task_id": args.task_id, "num_tasks": args.num_tasks,
                "clips": [lo, hi], "rows": rows[L], "shards": shards[L],
                "model_id": args.model, "checkpoint": get_spec(args.model).checkpoint,
                "revision": model.revision,
                "layer": L, "grid": list(grid), "dim": model.hidden_dim,
                "frames": int(model.num_frames), "tokens_per_clip": keep_n,
                "seed": args.seed, "clip_list_sha256": clip_sha,
                "data_root": str(Path(args.data_root).resolve()),
                "substituted": substituted}
        (dirs[L] / f"meta_t{args.task_id}.json").write_text(json.dumps(part, indent=2))
        print(f"  L{L:>2}: {len(shards[L])} shards, {rows[L]:,} rows -> {dirs[L]}",
              flush=True)

    if args.num_tasks == 1:
        merge(args, layers, samples, clip_sha, clip_desc)


def merge(args, layers, samples, clip_sha, clip_desc):
    """Check every task's output, then write clips.json and each layer's meta.json."""
    out_dir = Path(args.out_dir)
    n_clips = len(samples)
    for L in layers:
        d = out_dir / f"l{L}"
        parts = [json.loads(p.read_text()) for p in sorted(d.glob("meta_t*.json"))]
        if not parts:
            raise SystemExit(f"L{L}: no task wrote anything to {d}")
        nt = parts[0]["num_tasks"]
        ids = sorted(p["task_id"] for p in parts)
        if ids != list(range(nt)) or any(p["num_tasks"] != nt for p in parts):
            raise SystemExit(f"L{L}: tasks {ids} finished, expected 0-{nt - 1}")
        for field, want in (("model_id", args.model), ("clip_list_sha256", clip_sha)):
            got = {p[field] for p in parts}
            if got != {want}:
                raise SystemExit(f"L{L}: task outputs have {field} {sorted(got)}, "
                                 f"this merge was asked for {want!r}")
        keep_n = parts[0]["tokens_per_clip"]
        rows = sum(p["rows"] for p in parts)
        if rows != n_clips * keep_n:
            raise SystemExit(f"L{L}: {rows:,} rows, expected {n_clips:,} clips x "
                             f"{keep_n} = {n_clips * keep_n:,}")
        # train_sae.py reads every *_part*.pt in the directory, so a leftover
        # from an earlier run would be trained on silently.
        written = {s for p in parts for s in p["shards"]}
        on_disk = {f.name for f in d.glob("*_part*.pt")}
        if written != on_disk:
            raise SystemExit(f"L{L}: shards on disk != shards the tasks wrote; "
                             f"stray {sorted(on_disk - written)[:3]}, "
                             f"missing {sorted(written - on_disk)[:3]}")
        p0 = parts[0]
        meta = {"model_id": p0["model_id"], "checkpoint": p0["checkpoint"],
                "revision": p0["revision"], "layer": L,
                "site": "resid_post", "tokens": "patch",
                "grid": p0["grid"], "frames": p0["frames"], "dim": p0["dim"],
                "dtype": "float16", "n_clips": n_clips,
                "tokens_per_clip": keep_n, "rows": rows, "shards": len(written),
                "seed": p0["seed"], "tasks": nt,
                "token_pick": "per clip, default_rng([seed, clip index])",
                "clip_list": {"description": clip_desc, "clips_sha256": clip_sha,
                              "file": CLIPS},
                "data_root": p0["data_root"],
                "substituted": sorted((s for p in parts for s in p["substituted"]),
                                      key=lambda s: s["index"])}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  L{L:>2}: {nt} tasks, {len(written)} shards, {rows:,} rows", flush=True)
    (out_dir / CLIPS).write_text(json.dumps(
        {"description": clip_desc, "n_clips": n_clips, "samples": samples}))


def main(args):
    args.data_root = args.data_root or data_root("K400_TRAIN")
    if args.plan:
        if not args.data_root:
            raise SystemExit("--plan needs a data root: --data_root, or K400_TRAIN in .env")
        return plan(args)
    if not args.out_dir:
        raise SystemExit("--out_dir is required")

    spec = get_spec(args.model)
    layers = parse_layers(args.layers) if args.layers else list(spec.layers)
    if args.batch_size is None:
        args.batch_size = spec.batch_size
    if args.rows_per_shard is None:
        args.rows_per_shard = spec.rows_per_shard

    split = load_split(args.clip_list, args.split)
    samples, clip_desc = split["samples"], split["description"]
    clip_desc = f"{split['file']} [{args.split}]: {clip_desc}"
    args.short_side = split["short_side"]      # how .mp4 clips of this split are resized
    if args.limit is not None:
        samples = samples[:args.limit]
        clip_desc = f"SMOKE: first {len(samples)} clips of: {clip_desc}"
        if not args.out_dir.rstrip("/").endswith("_smoke"):
            args.out_dir = args.out_dir.rstrip("/") + "_smoke"
    # Identifies the clips themselves (not the file, which holds several splits
    # and whose formatting can change), so tasks of one dump can be checked
    # against each other.
    clip_sha = hashlib.sha256(json.dumps([list(s) for s in samples]).encode()).hexdigest()

    if args.merge:
        merge(args, layers, samples, clip_sha, clip_desc)
    else:
        extract(args, layers, samples, clip_sha, clip_desc)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
