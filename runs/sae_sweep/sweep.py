"""Train the sweep's SAEs, one GPU per run, as many at a time as there are GPUs.

    .venv/bin/python runs/sae_sweep/sweep.py base_r1 k32 k64      # named runs
    .venv/bin/python runs/sae_sweep/sweep.py --group round1        # a whole group
    .venv/bin/python runs/sae_sweep/sweep.py --list

Every run is scripts/train_sae.py with the shipped recipe (Matryoshka BatchTopK,
k=20, 16x, 20k steps, lr 'paper', train5k x 256 tokens, layer 7) except for the
fields its entry in RUNS overrides. A run lands in weights/<name>/videomaev2-vitb-k710distill/l<layer>/
with metrics.json from the held-out activations; finished runs are skipped.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
MODEL = "videomaev2-vitb-k710distill"

BASE = dict(data="c5k_t256", layer=7, sae="matroyshka_batch_top_k", k=20, x=16,
            steps=20_000, lr="paper", fractions=None)
BIG = {"c5k_t1024", "c20k_t256"}      # too many rows for one 24 GB GPU -> held on the CPU

RUNS = {
    # the shipped recipe, three unseeded repeats
    "base_r1": {}, "base_r2": {}, "base_r3": {},
    # round 1: one factor at a time
    "k32": dict(k=32),
    "k64": dict(k=64),
    "x8": dict(x=8),
    "x32": dict(x=32),
    "steps40k": dict(steps=40_000),
    "lr_auto": dict(lr="auto"),
    "tok1024": dict(data="c5k_t1024"),
    "c20k": dict(data="c20k_t256"),
    "btk": dict(sae="batch_top_k"),
    # round 2: the two round-1 winners (20k clips, lower lr) combined, across k
    "c20k_lrauto": dict(data="c20k_t256", lr="auto"),
    "c20k_lr5e-4": dict(data="c20k_t256", lr=5e-4),
    "c20k_lrauto_k32": dict(data="c20k_t256", lr="auto", k=32),
    "c20k_lrauto_k64": dict(data="c20k_t256", lr="auto", k=64),
    "c20k_lrauto_btk": dict(data="c20k_t256", lr="auto", sae="batch_top_k"),
    "c20k_lrauto_steps40k": dict(data="c20k_t256", lr="auto", steps=40_000),
    "lr_auto_r2": dict(lr="auto"),
    "lr_auto_k32": dict(lr="auto", k=32),
    # round 3: the candidate (20k clips, lr 5e-4) across k, and at layers 3 and 10
    "c20k_lr5e-4_k32": dict(data="c20k_t256", lr=5e-4, k=32),
    "c20k_lr5e-4_k64": dict(data="c20k_t256", lr=5e-4, k=64),
    "c20k_lr5e-4_btk_k64": dict(data="c20k_t256", lr=5e-4, k=64, sae="batch_top_k"),
    "l3_base": dict(layer=3),
    "l10_base": dict(layer=10),
    "l3_c20k_lr5e-4": dict(layer=3, data="c20k_t256", lr=5e-4),
    "l10_c20k_lr5e-4": dict(layer=10, data="c20k_t256", lr=5e-4),
    "l3_c20k_lr5e-4_k64": dict(layer=3, data="c20k_t256", lr=5e-4, k=64),
    "l10_c20k_lr5e-4_k64": dict(layer=10, data="c20k_t256", lr=5e-4, k=64),
}
GROUPS = {
    "base": ["base_r1", "base_r2", "base_r3"],
    "round1": ["k32", "k64", "x8", "x32", "steps40k", "lr_auto", "tok1024", "c20k"],
    # at most six c20k runs at a time: each holds ~31 GB of activations in RAM
    "round2": ["c20k_lrauto", "c20k_lr5e-4", "c20k_lrauto_k32", "c20k_lrauto_k64",
               "c20k_lrauto_btk", "c20k_lrauto_steps40k", "lr_auto_r2", "lr_auto_k32"],
    "round3": ["l3_base", "l10_base", "c20k_lr5e-4_k32", "c20k_lr5e-4_k64",
               "l3_c20k_lr5e-4", "l10_c20k_lr5e-4", "l3_c20k_lr5e-4_k64", "l10_c20k_lr5e-4_k64"],
    "round3b": ["c20k_lr5e-4_btk_k64"],
}


def config(name: str) -> dict:
    return {**BASE, **RUNS[name]}


def command(name: str, gpu: int) -> list:
    c = config(name)
    cmd = [str(ROOT / ".venv/bin/python"), "scripts/train_sae.py",
           "--activations_dir", str(RUN / "acts" / c["data"] / f"l{c['layer']}"),
           "--val_activations_dir", str(RUN / "acts" / "heldout" / f"l{c['layer']}"),
           "--out_dir", str(RUN / "weights" / name),
           "--sae_model", c["sae"], "--k", str(c["k"]), "--expansion_factor", str(c["x"]),
           "--steps", str(c["steps"]), "--lr", str(c["lr"]), "--device", f"cuda:{gpu}"]
    if c["data"] in BIG:
        cmd += ["--data_device", "cpu"]
    if c["fractions"]:
        cmd += ["--group_fractions", *map(str, c["fractions"])]
    return cmd


def done(name: str) -> bool:
    return (RUN / "weights" / name / MODEL / f"l{config(name)['layer']}" / "metrics.json").exists()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*")
    p.add_argument("--group", choices=list(GROUPS))
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list:
        for name in RUNS:
            print(f"{name:<16}{'done' if done(name) else '':<6}{RUNS[name]}")
        return

    names = list(args.names) + (GROUPS[args.group] if args.group else [])
    todo = [n for n in names if not done(n)]
    print(f"{len(names) - len(todo)} already done, training: {todo}", flush=True)
    (RUN / "logs").mkdir(exist_ok=True)

    free = [int(g) for g in args.gpus.split(",")]
    running, failed = {}, []
    while todo or running:
        while todo and free:
            name, gpu = todo.pop(0), free.pop(0)
            log = open(RUN / "logs" / f"train_{name}.log", "w")
            running[name] = (subprocess.Popen(command(name, gpu), cwd=ROOT, stdout=log,
                                              stderr=subprocess.STDOUT), gpu, time.time())
            print(f"start {name} on cuda:{gpu}", flush=True)
        time.sleep(10)
        for name, (proc, gpu, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[name]
            free.append(gpu)
            ok = proc.returncode == 0 and done(name)
            print(f"{'done' if ok else 'FAILED'} {name} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            if not ok:
                failed.append(name)
    if failed:
        sys.exit(f"failed: {failed} (see {RUN}/logs/train_<name>.log)")


if __name__ == "__main__":
    main()
