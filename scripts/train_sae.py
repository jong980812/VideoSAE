"""Stage 2: train one sparse autoencoder on one layer's activation shards.

    python scripts/train_sae.py --activations_dir acts/videomaev2-vitb-k710distill/l9 --out_dir my_weights

writes `my_weights/<model_id>/l<layer>/{ae.pt, config.json}` -- the same layout
as the shipped `weights/sae/`, so `saes.load_sae(..., weights_dir="my_weights")`
loads it. The model ID and layer come from the shards' meta.json (written by
extract_activations.py), so an SAE always records which backbone it belongs to.

The defaults ARE the recipe every shipped SAE was trained with: Matryoshka
BatchTopK, k=20, dictionary 16 x width, lr 16/(125 sqrt(dict_size)), batch 4096,
20k steps, warmup 1000, decay from 80%, auxk_alpha 0.03, group fractions
[1/16, 1/8, 1/4, 9/16], activations normalised to unit mean-squared norm (the
factor is folded back into the weights before saving), fp32, no seed.
Keep --log_steps at 1000 to reproduce it: each logging step also updates the
BatchTopK threshold's running estimate, so the cadence moves the threshold.

Existing weights are never overwritten: if <out>/<model_id>/l<layer>/ae.pt
exists the run stops (with --skip_if_trained, quietly).
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.activations import ActivationsDataset            # noqa: E402
from dictionary_learning.trainers import BatchTopKTrainer, MatroyshkaBatchTopKTrainer  # noqa: E402
from dictionary_learning.training import trainSAE          # noqa: E402
from saes import SAE_CLASSES, make_config                  # noqa: E402

TRAINERS = {
    "matroyshka_batch_top_k": MatroyshkaBatchTopKTrainer,
    "batch_top_k": BatchTopKTrainer,
}
SAE_CLASS_NAMES = {
    "matroyshka_batch_top_k": "MatroyshkaBatchTopKSAE",
    "batch_top_k": "BatchTopKSAE",
}


def get_args_parser():
    p = argparse.ArgumentParser("Train an SAE on one layer's activation shards")
    p.add_argument("--activations_dir", required=True, type=str,
                   help="one layer's shards + meta.json, from extract_activations.py")
    p.add_argument("--out_dir", required=True, type=str,
                   help="weights root; the SAE goes to <out_dir>/<model_id>/l<layer>/")
    p.add_argument("--val_activations_dir", default=None, type=str,
                   help="optional held-out shards -> metrics.json (FVE, L0, dead latents)")
    p.add_argument("--sae_model", default="matroyshka_batch_top_k", choices=list(TRAINERS))
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--data_device", default=None, type=str,
                   help="where to hold activations (default: --device)")
    p.add_argument("--expansion_factor", type=int, default=16)
    p.add_argument("--lr", default="paper",
                   help="a float, or: 'paper' = 16/(125*sqrt(dict_size)) (arXiv 2504.02821; "
                        "the shipped recipe), 'auto' = the trainer's own "
                        "2e-4/sqrt(dict_size/2^14), 5x lower")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--save_steps", type=int, default=0,
                   help="also save intermediate checkpoints every N steps (0 = none)")
    p.add_argument("--log_steps", type=int, default=1000)
    p.add_argument("--skip_if_trained", action="store_true",
                   help="exit quietly if this layer's ae.pt already exists (evaluating it "
                        "first when --val_activations_dir is given)")
    # logging
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_entity", default="", type=str)
    p.add_argument("--wandb_project", default="video-sae", type=str)
    # LR schedule; defaults scale with --steps so short runs work
    p.add_argument("--warmup_steps", type=int, default=None, help="default: steps//20, capped at 1000")
    p.add_argument("--decay_start", type=int, default=None, help="default: 80%% of steps")
    # (Matryoshka) BatchTopK
    p.add_argument("--k", type=int, default=20, help="active latents per token")
    p.add_argument("--auxk_alpha", type=float, default=0.03)
    p.add_argument("--threshold_beta", type=float, default=0.999)
    p.add_argument("--threshold_start_step", type=int, default=1_000)
    p.add_argument("--group_fractions", type=float, nargs="+", default=[0.0625, 0.125, 0.25, 0.5625],
                   help="Matryoshka nested-group sizes, as fractions of the dictionary")
    return p


class ETAReporter:
    """Wraps the train loader to project a finish time.

    `trainSAE` wraps the loader in `itertools.cycle`, which caches the first
    epoch and replays it, so this only ever observes one epoch's batches. That
    is enough to measure a step rate and extrapolate -- but it means the
    estimate comes from the first epoch, warmup included, so it runs pessimistic
    early on. The actual elapsed time is printed at the end.
    """

    def __init__(self, loader, total_steps: int, warmup_batches: int = 20, skip_passes: int = 0):
        self.loader = loader
        self.total_steps = total_steps
        self.warmup_batches = warmup_batches
        # `trainSAE` iterates the loader once for get_norm_factor before training
        # starts. That pass does no backward, so timing it would be optimistic by
        # an order of magnitude -- skip it.
        self.skip_passes = skip_passes
        self.passes = 0
        self.reported = False

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        # Only one epoch is ever observed (see the class docstring), so the
        # threshold has to fit inside it -- a large batch over a small dataset
        # can leave fewer batches per epoch than the nominal warmup.
        try:
            threshold = max(2, min(self.warmup_batches, len(self.loader) - 1))
        except TypeError:
            threshold = self.warmup_batches

        self.passes += 1
        measuring = self.passes > self.skip_passes

        seen, start = 0, None
        for batch in self.loader:
            # Skip the first batch: it carries CUDA context setup and one-off
            # allocations that would badly skew a rate measured over ~50 steps.
            if seen == 1:
                start = time.time()
            seen += 1
            yield batch

            if measuring and not self.reported and start is not None and seen >= threshold:
                rate = (seen - 1) / max(time.time() - start, 1e-9)
                remaining = max(self.total_steps - seen, 0) / max(rate, 1e-9)
                finish = datetime.now() + timedelta(seconds=remaining)
                print(
                    f"  ~{rate:.1f} steps/s over the first {seen} steps -> "
                    f"{self.total_steps:,} steps in ~{remaining / 60:.0f} min, "
                    f"finishing ~{finish:%H:%M} ({finish:%a %d %b})",
                    flush=True,
                )
                self.reported = True


def build_loaders(args, train: bool = True):
    """`train=False` skips the train split entirely -- loading 1.3M rows onto the
    GPU just to evaluate a finished run is pure waste."""
    data_device = args.data_device or args.device

    def val_loader_for(val_ds):
        # drop_last=False: a val split smaller than one batch must still yield one.
        val_batch = min(args.batch_size, len(val_ds))
        print(f"val activations: {len(val_ds)} rows in "
              f"{-(-len(val_ds) // val_batch)} batch(es) of <= {val_batch}")
        return DataLoader(val_ds, batch_size=val_batch, shuffle=False, drop_last=False)

    if not train:
        return None, val_loader_for(ActivationsDataset(args.val_activations_dir, device=data_device))

    dataset = ActivationsDataset(args.activations_dir, device=data_device)
    if len(dataset) < args.batch_size:
        # drop_last=True would otherwise yield zero batches and surface as a bare
        # StopIteration several frames deep.
        raise SystemExit(
            f"--batch_size {args.batch_size} exceeds the {len(dataset):,} rows in "
            f"{args.activations_dir}; with drop_last there would be no batches. "
            f"Lower --batch_size to at most {len(dataset):,}."
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_loader = None
    if args.val_activations_dir:
        val_loader = val_loader_for(ActivationsDataset(args.val_activations_dir, device=data_device))
    print(f"train activations: {len(dataset)} rows")
    return loader, val_loader


@torch.no_grad()
def evaluate_sae(ae, loader, device: str) -> dict:
    """Final val metrics, accumulated globally rather than averaged per batch.

    Fraction of variance explained is scale-free, which is what makes it
    comparable across layers whose residual streams have very different norms.

    For the (Matryoshka) BatchTopK SAEs this reports *both* encode paths:
      `*_topk`      exact top-k, the training objective
      unsuffixed    the learned global threshold, i.e. how the SAE actually
                    behaves when spliced into a model
    They agree once training has run past `threshold_start_step`. If they do not,
    the threshold was never estimated and the unsuffixed numbers are meaningless
    -- `threshold_set` in the output says which case you are in.
    """
    import inspect

    ae = ae.to(device).eval()
    takes_threshold = "use_threshold" in inspect.signature(ae.encode).parameters

    rows, n_rows, running_sum = [], 0, None
    for act in tqdm(loader, desc="val load", unit="batch"):
        act = act.to(device).float()
        rows.append(act)
        running_sum = act.sum(0) if running_sum is None else running_sum + act.sum(0)
        n_rows += act.shape[0]
    if not rows:
        return {}
    mean = running_sum / n_rows

    def run(**encode_kwargs) -> dict:
        resid_sq = total_sq = l0_sum = 0.0
        fired = torch.zeros(int(ae.dict_size), device=device)
        mode = "top-k" if encode_kwargs.get("use_threshold") is False else "threshold"
        for act in tqdm(rows, desc=f"val metrics ({mode})", unit="batch"):
            codes = ae.encode(act, **encode_kwargs)
            recon = ae.decode(codes)
            resid_sq += float(((act - recon) ** 2).sum())
            total_sq += float(((act - mean) ** 2).sum())
            l0_sum += float((codes != 0).float().sum())
            fired += (codes != 0).float().sum(0)
        return {
            "frac_variance_explained": 1.0 - resid_sq / max(total_sq, 1e-9),
            "l0": l0_sum / max(n_rows, 1),
            "mse": resid_sq / max(n_rows, 1),
            "dead_latent_frac": float((fired == 0).float().mean()),
        }

    if not takes_threshold:
        return {**run(), "n_val_rows": n_rows}

    exact = run(use_threshold=False)
    thresholded = run(use_threshold=True)
    threshold = getattr(ae, "threshold", None)
    return {
        **thresholded,
        **{f"{k}_topk": v for k, v in exact.items()},
        "threshold_set": bool(threshold is not None and float(threshold) >= 0),
        "n_val_rows": n_rows,
    }


def load_trained(sae_model: str, path, device: str):
    return SAE_CLASSES[SAE_CLASS_NAMES[sae_model]].from_pretrained(str(path), device=device)


def write_metrics(args, ae, val_loader, save_dir, ae_path):
    """Evaluate on val and persist metrics.json; returns the metrics dict."""
    metrics = evaluate_sae(ae, val_loader, args.device)
    metrics.update(
        sae_model=args.sae_model,
        dict_size=int(ae.dict_size),
        activation_dim=int(ae.activation_dim),
        val_activations_dir=args.val_activations_dir,
        ae_path=str(ae_path),
    )
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    shown = [k for k in ("frac_variance_explained_topk", "l0_topk",
                         "frac_variance_explained", "l0", "dead_latent_frac")
             if k in metrics]
    print("final val: " + "  ".join(f"{k}={metrics[k]:.4f}" for k in shown))
    if metrics.get("threshold_set") is False:
        print(
            f"  WARNING: BatchTopK threshold was never estimated "
            f"(steps={args.steps} <= threshold_start_step={args.threshold_start_step}). "
            "The non-_topk numbers are meaningless; compare layers on the _topk columns "
            "or train for more steps."
        )
    print(f"Wrote {save_dir}/metrics.json")
    return metrics


def read_meta(activations_dir: str) -> dict:
    """The dump's meta.json: which model ID, checkpoint, layer and data it holds."""
    path = Path(activations_dir) / "meta.json"
    if not path.exists():
        raise SystemExit(f"no {path}: point --activations_dir at one layer's output of "
                         f"extract_activations.py (e.g. <out_dir>/l9), after its merge")
    meta = json.loads(path.read_text())
    for field in ("model_id", "checkpoint", "revision", "layer", "grid", "frames"):
        if field not in meta:
            raise SystemExit(f"{path} has no {field!r}; re-extract with extract_activations.py")
    return meta


def train_sae(args):
    meta = read_meta(args.activations_dir)
    save_dir = Path(args.out_dir) / meta["model_id"] / f"l{meta['layer']}"
    ae_path = save_dir / "ae.pt"

    # Never overwrite: a finished SAE may be what later results were built on,
    # and ae.pt may be a hardlink shared with another copy of the weights.
    if ae_path.exists():
        if not args.skip_if_trained:
            raise SystemExit(f"{ae_path} already exists; refusing to overwrite it "
                             f"(pass --skip_if_trained to skip, or pick another --out_dir)")
        print(f"{ae_path} already exists -> skipping training")
        if args.val_activations_dir:
            _, val_loader = build_loaders(args, train=False)
            write_metrics(args, load_trained(args.sae_model, ae_path, args.device),
                          val_loader, save_dir, ae_path)
        return
    val_meta = read_meta(args.val_activations_dir) if args.val_activations_dir else None
    if val_meta and (val_meta["model_id"], val_meta["layer"]) != (meta["model_id"], meta["layer"]):
        raise SystemExit(f"val activations are {val_meta['model_id']} L{val_meta['layer']}, "
                         f"train activations {meta['model_id']} L{meta['layer']}")

    loader, val_loader = build_loaders(args)

    sample = next(iter(loader))
    activation_dim = sample.shape[-1]
    dict_size = args.expansion_factor * activation_dim

    # Both candidate rules scale as 1/sqrt(width); they differ only in the
    # constant, by exactly 5x:
    #   library  2e-4 / sqrt(d / 2**14)  = 0.0256 / sqrt(d)
    #   paper    16 / (125 * sqrt(d))    = 0.128  / sqrt(d)   <- arXiv 2504.02821; shipped SAEs
    lr_rule = args.lr
    if isinstance(args.lr, str):
        if args.lr == "auto":
            args.lr = None      # the trainer applies 2e-4/sqrt(dict_size/2^14)
            print(f"lr=auto -> trainer default 2e-4/sqrt({dict_size}/2^14) = "
                  f"{2e-4 / (dict_size / 2**14) ** 0.5:.3e}")
        elif args.lr == "paper":
            args.lr = 16.0 / (125.0 * math.sqrt(dict_size))
            print(f"lr=paper -> {args.lr:.3e}  (16/(125*sqrt({dict_size})), 5x the trainer default)")
        else:
            args.lr = float(args.lr)
    shown = "trainer default" if args.lr is None else f"{args.lr:.3e}"
    print(f"activation_dim={activation_dim}  dict_size={dict_size}  lr={shown}  batch={sample.shape}")

    # `get_lr_schedule` requires warmup < decay_start < steps, and the library
    # defaults blow up on short runs, so derive them from --steps.
    warmup = args.warmup_steps if args.warmup_steps is not None else min(1000, max(1, args.steps // 20))
    decay_start = args.decay_start if args.decay_start is not None else int(args.steps * 0.8)
    decay_start = max(decay_start, warmup + 1)
    if decay_start >= args.steps:
        decay_start = None  # too few steps to decay; keep a flat LR after warmup

    run_name = f"{meta['model_id']}_l{meta['layer']}"
    cfg = {
        "trainer": TRAINERS[args.sae_model],
        "activation_dim": activation_dim,
        "dict_size": dict_size,
        "lr": args.lr,
        "device": args.device,
        "steps": args.steps,
        "warmup_steps": warmup,
        # Stored by the trainer only; recorded here so its own config is right.
        "layer": int(meta["layer"]),
        "lm_name": meta["model_id"],
        "submodule_name": "resid_post",
        "wandb_name": run_name,
        "k": args.k,
        "auxk_alpha": args.auxk_alpha,
        "decay_start": decay_start,
        "threshold_beta": args.threshold_beta,
        "threshold_start_step": args.threshold_start_step,
    }
    if args.sae_model == "matroyshka_batch_top_k":
        cfg["group_fractions"] = args.group_fractions

    run_cfg = {                      # extra columns on the wandb run, for grouping
        "model_id": meta["model_id"],
        "layer": meta["layer"],
        "sae_model": args.sae_model,
        "expansion_factor": args.expansion_factor,
        "batch_size": args.batch_size,
        "n_train_rows": len(loader.dataset),
    }

    # trainSAE writes <stage>/trainer_0/{config.json, ae.pt, checkpoints/}; the
    # result is moved into place only once training has finished.
    stage = save_dir / "_training"
    if stage.exists():
        shutil.rmtree(stage)          # leftovers of an interrupted run of this layer
    stage.mkdir(parents=True)
    save_steps = list(range(args.save_steps, args.steps, args.save_steps)) if args.save_steps else []

    started = datetime.now()
    print(f"training {meta['model_id']} L{meta['layer']}: {args.steps:,} steps, "
          f"started {started:%H:%M:%S}", flush=True)

    trainSAE(
        data=ETAReporter(loader, args.steps, skip_passes=1),
        val_data=val_loader,
        trainer_configs=[cfg],
        steps=args.steps,
        use_wandb=args.wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        run_cfg=run_cfg,
        save_steps=save_steps,
        save_dir=str(stage),
        log_steps=args.log_steps,
        normalize_activations=True,
        verbose=not args.wandb,
        device=args.device,
        autocast_dtype=torch.float32,
    )
    elapsed = datetime.now() - started
    print(f"training took {elapsed.total_seconds() / 60:.1f} min "
          f"({started:%H:%M:%S} -> {datetime.now():%H:%M:%S})")

    # os.replace renames the directory entry: it never writes through into an
    # existing file (which may be a hardlink shared with another weights copy).
    trainer_cfg = json.loads((stage / "trainer_0" / "config.json").read_text())["trainer"]
    os.replace(stage / "trainer_0" / "ae.pt", ae_path)
    if save_steps:
        os.replace(stage / "trainer_0" / "checkpoints", save_dir / "checkpoints")
    shutil.rmtree(stage)

    identity = {"layer", "lm_name", "submodule_name", "wandb_name", "device",
                "activation_dim", "dict_size", "group_sizes", "k"}
    training = {k: v for k, v in trainer_cfg.items() if k not in identity}
    training.update(lr_rule=lr_rule, batch_size=args.batch_size, log_steps=args.log_steps,
                    normalize_activations=True, dtype="float32",
                    n_train_rows=len(loader.dataset))
    data = {k: meta[k] for k in ("n_clips", "tokens_per_clip", "rows", "seed",
                                 "token_pick", "clip_list", "substituted") if k in meta}
    config = make_config(model_id=meta["model_id"], checkpoint=meta["checkpoint"],
                         revision=meta["revision"], layer=meta["layer"],
                         frames=meta["frames"], token_grid=meta["grid"],
                         ae_path=ae_path, training=training, data=data)
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Final SAE written to {ae_path} (+ config.json)")

    if val_loader is not None:
        write_metrics(args, load_trained(args.sae_model, ae_path, args.device),
                      val_loader, save_dir, ae_path)


if __name__ == "__main__":
    train_sae(get_args_parser().parse_args())
