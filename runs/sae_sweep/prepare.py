"""Clip lists and symlink trees for the SAE setting sweep, from the Kinetics-400
copy on this machine (/local_datasets/kinetics400_320p/{train,val}/<youtube_id>.mp4).

    python runs/sae_sweep/prepare.py

Writes runs/sae_sweep/clips.json with splits
  train5k    datafile train_sae minus the clips missing here (the shipped recipe's clips)
  train20k   train5k + probe_train clips (not in train5k / heldout), 20,000 in all
  heldout    the 500 clips of runs/mat_vs_btk (never trained on)
  val4k      4,000 val clips evenly spaced through the val list (10 per class)
  val        every val clip present here
and k400_train/, k400_val/: <class>/<clip_id>.mp4 -> the local file.
"""

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
SRC = Path("/local_datasets/kinetics400_320p")


def youtube_id(key: str) -> str:
    return re.sub(r"_\d{6}_\d{6}$", "", key.split("/")[-1])


def stride(samples: list, n: int) -> list:
    step = len(samples) / n
    return [samples[int(i * step)] for i in range(n)]


def main():
    k400 = json.loads((ROOT / "datafile" / "kinetics400.json").read_text())
    have = {s: set(os.listdir(SRC / s)) for s in ("train", "val")}
    here = lambda split, key: youtube_id(key) + ".mp4" in have[split]

    train5k = [s for s in k400["splits"]["train_sae"]["samples"] if here("train", s[0])]
    heldout = json.loads((ROOT / "runs" / "mat_vs_btk" / "clips.json").read_text())["splits"]["heldout"]["samples"]
    taken = {k for k, _ in train5k} | {k for k, _ in heldout}
    extra = []
    for key, label in k400["splits"]["probe_train"]["samples"]:
        if key not in taken and here("train", key):
            taken.add(key)
            extra.append([key, label])
    train20k = train5k + stride(extra, 20_000 - len(train5k))
    val = [s for s in k400["splits"]["val"]["samples"] if here("val", s[0])]
    val4k = stride(val, 4000)

    def split(desc, samples, short_side):
        return {"description": desc, "short_side": short_side, "samples": samples}

    clips = {
        "name": "kinetics400-local",
        "description": "Kinetics-400 clips present in /local_datasets/kinetics400_320p, for the SAE setting sweep",
        "classes": k400["classes"],
        "splits": {
            "train5k": split("train_sae minus the clips missing locally", train5k, 256),
            "train20k": split("train5k + probe_train clips not in train5k/heldout, evenly spaced", train20k, 256),
            "heldout": split("500 probe_train clips never trained on", heldout, 256),
            "val4k": split("4,000 val clips evenly spaced through the sorted val list", val4k, None),
            "val": split("every val clip present locally", val, None),
        },
    }
    (RUN / "clips.json").write_text(json.dumps(clips))

    for name, src, samples in (("k400_train", "train", train20k + heldout), ("k400_val", "val", val)):
        for key, _ in samples:
            dst = RUN / name / (key + ".mp4")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.is_symlink():
                dst.symlink_to(SRC / src / (youtube_id(key) + ".mp4"))

    for name, s in clips["splits"].items():
        print(f"{name:<9}{len(s['samples']):>7,} clips, {len({l for _, l in s['samples']})} classes")
    print(f"val clips missing locally: {len(k400['splits']['val']['samples']) - len(val)}")


if __name__ == "__main__":
    main()
