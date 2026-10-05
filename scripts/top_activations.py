"""What each SAE latent responds to: its top-activating clips and its classes.

    python scripts/top_activations.py --model videomaev2-base --layer 7
    python scripts/top_activations.py --model videomaev2-base --layer 7 \\
        --weights_dir runs/sae_sweep/weights/btk --out runs/sae_sweep/top/btk.pt \\
        --clips runs/sae_sweep/clips.json --split val4k --data_root runs/sae_sweep/k400_val

One pass over a clip list. Each clip gets one score per latent, the latent's
MAX activation over the clip's patch tokens (so a latent that fires on a small
region still ranks its clip), and from those scores the script keeps, per latent:

    top_val, top_clip   (dict_size, K)   its K highest-scoring clips: score, index into `keys`
                                         (-1 where fewer than K clips fired)
    class_mean          (dict_size, C)   its mean score over the clips of each class
    clip_freq           (dict_size,)     the fraction of clips it fires on

Writes one file, results/<model_id>/top_activations/l<layer>.pt by default, also
holding `keys`, `labels`, `classes` and the settings, among them which SAE it was
(`sae`: its config.json, `sae_dir`: its folder). Per-patch maps are not stored:
show_latent.py and sae_usage.ipynb recompute them for the few clips they show.

`--weights_dir` reads the SAE from <weights_dir>/<model_id>/l<layer> instead of
`sae.dir` in the model's config; give such a run its own `--out`.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.clips import (ClipList, check_data_root, load_split,       # noqa: E402
                        make_frame_collate_fn, stride_subsample)
from models import MODELS, get_model, get_spec                       # noqa: E402
from saes import load_sae, sae_dir                                   # noqa: E402
from utils.paths import data_root                                    # noqa: E402


def get_args_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--layer", required=True, type=int)
    p.add_argument("--weights_dir", default=None, type=str,
                   help="read the SAE from <weights_dir>/<model_id>/l<layer> "
                        "(default: sae.dir in the model's config)")
    p.add_argument("--clips", default=str(ROOT / "datafile" / "kinetics400.json"), type=str)
    p.add_argument("--split", default="val", type=str)
    p.add_argument("--data_root", default=None, type=str, help="default: K400_VAL in .env")
    p.add_argument("--limit", default=None, type=int, help="N evenly spaced clips of the split")
    p.add_argument("--topk", default=8, type=int, help="clips kept per latent")
    p.add_argument("--out", default=None, type=str,
                   help="default: results/<model_id>/top_activations/l<layer>.pt")
    p.add_argument("--batch_size", default=None, type=int,
                   help="default: eval.batch_size in the model's config")
    p.add_argument("--workers", default=7, type=int)
    p.add_argument("--device", default="cuda:0", type=str)
    return p


@torch.no_grad()
def main(args):
    split = load_split(args.clips, args.split)
    samples = stride_subsample(split["samples"], args.limit)
    root = args.data_root or data_root("K400_VAL")
    check_data_root(root, samples, "--data_root / K400_VAL")

    model = get_model(args.model, device=args.device)
    sae = load_sae(args.model, args.layer, device=args.device, weights_dir=args.weights_dir,
                   backbone=model)
    ds = ClipList(root, samples, model.num_frames, split["short_side"])
    loader = DataLoader(ds, batch_size=args.batch_size or get_spec(args.model).eval_batch_size,
                        num_workers=args.workers, shuffle=False,
                        collate_fn=make_frame_collate_fn(model.processor))

    F, C, K = int(sae.dict_size), len(split["classes"]), args.topk
    top_val = torch.zeros(F, K, device=args.device)
    top_clip = torch.full((F, K), -1, dtype=torch.long, device=args.device)
    class_sum = torch.zeros(F, C, device=args.device)
    class_n = torch.zeros(C, device=args.device)
    fired = torch.zeros(F, device=args.device)

    start = 0
    for batch in tqdm(loader, unit="batch"):
        patches = model.encode({"pixel_values": batch["pixel_values"]}, layer=args.layer)
        score = sae.encode(patches.float()).amax(1)                 # (B, F): max over the patches
        idx = torch.arange(start, start + len(score), device=args.device)
        start += len(score)
        # ClipList replaces an unreadable clip by the next one; leave those out, they are scored as themselves
        read = torch.tensor([samples[i][0] == key for i, key in zip(idx.tolist(), batch["paths"])],
                            device=args.device)
        score, idx, labels = score[read], idx[read], batch["labels"].to(args.device)[read]

        val = torch.cat([top_val, score.T], 1)                      # (F, K + B)
        clip = torch.cat([top_clip, idx.expand(F, -1)], 1)
        top_val, order = val.topk(K, dim=1)
        top_clip = clip.gather(1, order)
        class_sum.index_add_(1, labels, score.T)
        class_n += torch.bincount(labels, minlength=C)
        fired += (score > 0).sum(0)

    top_clip[top_val == 0] = -1
    n = int(class_n.sum())
    where = sae_dir(args.model, args.layer, args.weights_dir).resolve()
    out = Path(args.out or ROOT / "results" / args.model / "top_activations" / f"l{args.layer}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_id": args.model, "layer": args.layer, "sae": sae.config,
        "sae_dir": str(where.relative_to(ROOT) if where.is_relative_to(ROOT) else where),
        "clips": {k: split[k] for k in ("file", "split", "sha256", "short_side")},
        "data_root": str(Path(root).resolve()), "n_clips": n,
        "keys": [s[0] for s in samples], "labels": torch.tensor([s[1] for s in samples]),
        "classes": split["classes"],
        "top_val": top_val.cpu(), "top_clip": top_clip.cpu(),
        "class_mean": (class_sum / class_n.clamp(min=1)).cpu(), "clip_freq": (fired / n).cpu(),
    }, out)
    print(f"{args.model} L{args.layer}: {n:,} clips, {int((fired > 0).sum()):,} of {F:,} latents "
          f"fired on at least one -> {out}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
