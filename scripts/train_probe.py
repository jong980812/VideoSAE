"""Fit a Kinetics-400 linear probe on a backbone's frozen features.

    python scripts/train_probe.py --model videomaev2-vitb-k710distill

(clips from K400_TRAIN and K400_VAL in .env; --train_root / --val_root override)
writes weights/heads/kinetics400/<model_id>.pt, the `probe` head evaluate.py scores
Kinetics-400 (and HAT) through: point `head.kinetics400.path` in
config/<model_id>.yaml at it, and set `head.kinetics400.sha256`. (Kinetics-400
only, for now; the shipped NTU heads were fitted separately, on xsub_train --
see "Heads and probes" in the README.)

Two stages, both by default (`--stage extract|fit|all`):

  extract   `model.encode()` pooled features of the probe clips, cached as
            <work_dir>/<model_id>/emb_{train,val}[_part<i>].npz. Existing files
            are reused. `--num_tasks N --task_id i` splits it over GPUs.
  fit       multinomial logistic regression on those features
            -> weights/heads/kinetics400/<model_id>.pt

THE PROTOCOL is the one the shipped VideoPrism and V-JEPA 2 probes were fitted
with, so probes made here are comparable with them:

  * fit on datafile/kinetics400.json split `probe_train` -- 50 clips per class,
    20,000 -- and pick the epoch on split `probe_val` -- 8 per class, 3,200 (a
    subset of the val set that evaluate.py later scores; the one choice made on
    it is the epoch)
  * features under bf16 autocast (`--amp`): the pooled feature moves ~1% of the
    distance between two clips, so the head applies unchanged to the fp32
    features evaluate.py feeds it
  * features standardised; AdamW, lr 1e-3, weight decay 1e-4, batch 4,096, 200
    epochs with cosine decay, seed 0; val top-1 checked every 20 epochs and the
    best kept. The standardisation is folded into the saved W and b, so the head
    is a plain linear map on RAW pooled features.

The fit is seeded and runs on the GPU; from the same features it reproduces the
same head.
"""

import argparse
import glob
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.clips import (ClipList, check_data_root, load_split,  # noqa: E402
                        make_frame_collate_fn, stride_subsample)
from models import MODELS, get_model, get_spec             # noqa: E402
from utils.paths import data_root                          # noqa: E402

CLIP_LIST = ROOT / "datafile" / "kinetics400.json"
SPLITS = {"train": "probe_train", "val": "probe_val"}      # stage name -> split in CLIP_LIST


def get_args_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--stage", default="all", choices=["extract", "fit", "all"])
    p.add_argument("--train_root", type=str,
                   help="Kinetics-400 train: <class>/<clip_id>/ or .mp4 (default: K400_TRAIN from .env)")
    p.add_argument("--val_root", type=str,
                   help="Kinetics-400 val: <class>/<clip_id>/ or .mp4 (default: K400_VAL from .env)")
    p.add_argument("--work_dir", default=str(ROOT / "probe_work"), type=str,
                   help="feature cache; <work_dir>/<model_id>/emb_*.npz")
    p.add_argument("--emb_dir", default=None, type=str,
                   help="fit from features in this directory instead of <work_dir>/<model_id>")
    p.add_argument("--out", default=None, type=str,
                   help="default: weights/heads/kinetics400/<model_id>.pt")
    # extract
    p.add_argument("--amp", default="bf16", choices=["bf16", "off"])
    p.add_argument("--batch_size", default=None, type=int,
                   help="default: probe.batch_size in the model's config")
    p.add_argument("--workers", default=14, type=int)
    p.add_argument("--num_tasks", default=1, type=int)
    p.add_argument("--task_id", default=0, type=int)
    p.add_argument("--limit", default=None, type=int,
                   help="SMOKE TEST: N evenly spaced clips per split; '_smoke' is added "
                        "to the cache directory and the output head")
    # fit
    p.add_argument("--epochs", default=200, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--fit_batch", default=4096, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--device", default="cuda:0", type=str)
    return p


# ----------------------------------------------------------------- extract


@torch.no_grad()
def extract(args, model, split_name, root, emb_dir, env_name):
    split = load_split(CLIP_LIST, SPLITS[split_name])
    samples = split["samples"]
    if args.limit is not None:
        samples = stride_subsample(samples, args.limit)
    part = f"_part{args.task_id}" if args.num_tasks > 1 else ""
    out = emb_dir / f"emb_{split_name}{part}.npz"
    if out.exists():
        print(f"  {out} exists, reusing it")
        return
    check_data_root(root, samples, env_name=env_name)
    per = -(-len(samples) // args.num_tasks)
    lo, hi = args.task_id * per, min((args.task_id + 1) * per, len(samples))

    ds = ClipList(root, samples, model.num_frames, split["short_side"])
    dl = DataLoader(Subset(ds, range(lo, hi)), batch_size=args.batch_size,
                    num_workers=args.workers, shuffle=False,
                    collate_fn=make_frame_collate_fn(model.processor))
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if args.amp == "bf16" and "cuda" in args.device else nullcontext())
    feats, labels, keys = [], [], []
    for batch in tqdm(dl, desc=f"{split_name} features", unit="batch", file=sys.stdout,
                      dynamic_ncols=True, mininterval=5.0):
        with amp:
            f = model.encode({"pixel_values": batch["pixel_values"]})
        feats.append(f.float().cpu())
        labels.append(batch["labels"])
        keys.extend(batch["paths"])
    X = torch.cat(feats).numpy().astype(np.float32)
    y = torch.cat(labels).numpy()
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, y=y, keys=np.array(keys, dtype=object), amp=args.amp,
             model_id=args.model, split=split_name, split_sha256=split["sha256"],
             task_id=args.task_id, num_tasks=args.num_tasks)
    print(f"  wrote {out}: X {X.shape}, {len(set(y.tolist()))} classes")


# --------------------------------------------------------------------- fit


def load_features(emb_dir: Path, split: str):
    """emb_<split>.npz, or its _part* files concatenated in task order."""
    single = emb_dir / f"emb_{split}.npz"
    parts = sorted(glob.glob(str(emb_dir / f"emb_{split}_part*.npz")),
                   key=lambda p: int(p.split("_part")[-1].split(".")[0]))
    if parts:
        ids = [int(p.split("_part")[-1].split(".")[0]) for p in parts]
        n = int(np.load(parts[0], allow_pickle=True)["num_tasks"])
        if ids != list(range(n)):
            raise SystemExit(f"{split}: found parts {ids} of {n}; each part is a band of "
                             f"classes, so all of them are needed")
        files = parts
    elif single.exists():
        files = [str(single)]
    else:
        raise SystemExit(f"no emb_{split}*.npz in {emb_dir}: run --stage extract first")
    Xs, ys, amps = [], [], set()
    for f in tqdm(files, desc=f"load {split}", unit="file"):
        z = np.load(f, allow_pickle=True)
        Xs.append(z["X"]); ys.append(z["y"]); amps.add(str(z["amp"]))
    if len(amps) > 1:
        raise SystemExit(f"{split}: parts disagree on precision {amps}; re-extract with one --amp")
    X, y = np.concatenate(Xs), np.concatenate(ys)
    print(f"  {split}: {X.shape[0]:,} x {X.shape[1]} from {len(files)} file(s), amp={amps.pop()}")
    return X, y


def fit(Xtr, ytr, Xva, yva, n_classes, args):
    """Multinomial logistic regression, minibatch AdamW; -> (best val top-1, (W, b, top5))."""
    dev = args.device if torch.cuda.is_available() else "cpu"
    # Seeded before the head is built: its init and every epoch's permutation
    # draw from this, so the fit is reproducible from the cached features.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr_t = torch.from_numpy((Xtr - mu) / sd).float().to(dev)
    ytr_t = torch.from_numpy(ytr).long().to(dev)
    Xva_t = torch.from_numpy((Xva - mu) / sd).float().to(dev)
    yva_t = torch.from_numpy(yva).long().to(dev)

    head = nn.Linear(Xtr.shape[1], n_classes).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n = Xtr_t.shape[0]

    best = (0.0, None)
    bar = tqdm(range(args.epochs), desc="fit", unit="epoch", file=sys.stdout, dynamic_ncols=True)
    for ep in bar:
        head.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, args.fit_batch):
            b = perm[i : i + args.fit_batch]
            loss = nn.functional.cross_entropy(head(Xtr_t[b]), ytr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 20 == 0 or ep == args.epochs - 1:
            head.eval()
            with torch.no_grad():
                logits = head(Xva_t)
                tr = (head(Xtr_t).argmax(1) == ytr_t).float().mean().item()
                va = (logits.argmax(1) == yva_t).float().mean().item()
                t5 = logits.topk(5, 1).indices.eq(yva_t[:, None]).any(1).float().mean().item()
            tqdm.write(f"  epoch {ep+1:>4}  loss {loss.item():.3f}  train {tr*100:.2f}%  "
                       f"val {va*100:.2f}%  val@5 {t5*100:.2f}%")
            bar.set_postfix(val=f"{va*100:.2f}%", best=f"{max(va, best[0])*100:.2f}%")
            if va > best[0]:
                # Fold the standardisation in: a plain linear map on raw features.
                W = head.weight.detach().cpu() / torch.from_numpy(sd).float()
                b = head.bias.detach().cpu() - (W * torch.from_numpy(mu).float()).sum(1)
                best = (va, (W, b, t5))
    return best


def main(args):
    spec = get_spec(args.model)
    args.batch_size = args.batch_size or spec.probe_batch_size
    tag = "_smoke" if args.limit is not None else ""
    emb_dir = Path(args.emb_dir) if args.emb_dir else Path(args.work_dir) / f"{args.model}{tag}"
    out = (Path(args.out) if args.out else
           ROOT / "weights" / "heads" / "kinetics400" / f"{args.model}{tag}.pt")

    if args.stage in ("extract", "all"):
        args.train_root = args.train_root or data_root("K400_TRAIN")
        args.val_root = args.val_root or data_root("K400_VAL")
        model = get_model(args.model, device=args.device)
        print(f"{args.model}: probe features, amp={args.amp}, batch {args.batch_size}"
              + (f", task {args.task_id}/{args.num_tasks}" if args.num_tasks > 1 else ""))
        extract(args, model, "train", args.train_root, emb_dir, "K400_TRAIN")
        extract(args, model, "val", args.val_root, emb_dir, "K400_VAL")
        del model
        if args.num_tasks > 1 and args.stage == "all":
            print("split extraction: run --stage fit once every task has finished")
            return

    if args.stage in ("fit", "all"):
        if out.exists():
            raise SystemExit(f"{out} exists; refusing to overwrite it (pass another --out)")
        classes = load_split(CLIP_LIST, SPLITS["train"])["classes"]
        print(f"{args.model}: fitting a {len(classes)}-way linear probe")
        Xtr, ytr = load_features(emb_dir, "train")
        Xva, yva = load_features(emb_dir, "val")
        acc, (W, b, top5) = fit(Xtr, ytr, Xva, yva, len(classes), args)
        amp = str(np.load(sorted(glob.glob(str(emb_dir / "emb_val*.npz")))[0], allow_pickle=True)["amp"])
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"W": W, "b": b, "classes": classes, "val_top1": acc, "val_top5": top5,
                    "model_id": args.model, "checkpoint": spec.checkpoint, "revision": spec.revision,
                    "seed": args.seed, "n_train": int(Xtr.shape[0]), "n_val": int(Xva.shape[0]),
                    "dim": int(Xtr.shape[1]), "amp": amp, "epochs": args.epochs, "lr": args.lr,
                    "weight_decay": args.weight_decay, "fit_batch": args.fit_batch,
                    "train_split": json.dumps({k: load_split(CLIP_LIST, SPLITS["train"])[k]
                                               for k in ("file", "split", "sha256")}),
                    "val_split": json.dumps({k: load_split(CLIP_LIST, SPLITS["val"])[k]
                                             for k in ("file", "split", "sha256")})},
                   out)
        print(f"\nwrote {out}\n  val top-1 {acc*100:.2f}%   top-5 {top5*100:.2f}%  "
              f"(on the {Xva.shape[0]:,} probe-val clips)")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
