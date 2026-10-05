"""Clip loading: a list of `<class>/<clip_id>` keys -> uint8 frames.

Each key resolves under the data root to one of

    <root>/<class>/<clip_id>/000001.jpg ...   a directory of JPEG frames
    <root>/<class>/<clip_id>.mp4              a video file, decoded on the fly

The shipped SAEs were trained on Kinetics-400 frame directories made from the
CVDF mp4 release with

    ffmpeg -i <clip>.mp4 -y -q:v 4 -r 30 \
        -vf "scale='if(gt(a,1),-2,256)':'if(gt(a,1),256,-2)'" <clip>/%06d.jpg

i.e. 30 fps, shortest side 256. Frame directories made that way reproduce the
training input. mp4 clips are resized to the same shortest side (256) after
decoding, so they are close -- but frame rate, decoder and JPEG compression
differ slightly, so activations are not bit-identical to the frame path.

Frames are always chosen the same way: `num_frames` indices spread uniformly
over the whole clip (`linspace(0, n - 1, num_frames)`). Resizing, cropping and
normalisation are the backbone's job (`model.processor`), not this module's.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

FRAME_EXTS = (".jpg", ".jpeg", ".png")
VIDEO_EXT = ".mp4"
SHORT_SIDE = 256       # shortest side of the frames the shipped SAEs were trained on


def sample_frame_indices(total: int, num_frames: int) -> np.ndarray:
    """`num_frames` indices spread over the whole clip of `total` frames."""
    return np.linspace(0, max(total - 1, 0), num=num_frames).round().astype(np.int64)


# ------------------------------------------------------------ frame directories


def _frames_in(clip_dir: str) -> list[str]:
    return sorted(f for f in os.listdir(clip_dir) if f.lower().endswith(FRAME_EXTS))


def _load_frames(clip_dir: str, names: list[str]) -> np.ndarray:
    return np.stack([np.array(Image.open(os.path.join(clip_dir, n)).convert("RGB")) for n in names])


def _read_frame_dir(clip_dir: str, num_frames: int) -> np.ndarray:
    names = _frames_in(clip_dir)
    if not names:
        raise RuntimeError(f"No frames in {clip_dir}")
    idx = sample_frame_indices(len(names), num_frames)
    return _load_frames(clip_dir, [names[min(int(i), len(names) - 1)] for i in idx])


# ------------------------------------------------------------------ video files


def _decode_decord(path: str, num_frames: int) -> np.ndarray:
    import decord

    decord.bridge.set_bridge("native")
    reader = decord.VideoReader(path, num_threads=1)
    idx = sample_frame_indices(len(reader), num_frames)
    return reader.get_batch(list(idx)).asnumpy()          # (T, H, W, 3) uint8


def _decode_pyav(path: str, num_frames: int) -> np.ndarray:
    """Fallback decoder. Counts frames by decoding rather than trusting the
    container's frame-count metadata, which is often missing or wrong."""
    import av

    with av.open(path) as container:
        total = sum(1 for _ in container.decode(video=0))
    if total == 0:
        raise RuntimeError(f"No frames decoded from {path}")
    idx = sample_frame_indices(total, num_frames)
    wanted = set(int(i) for i in idx)
    frames = {}
    with av.open(path) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                frames[i] = frame.to_ndarray(format="rgb24")
                if len(frames) == len(wanted):
                    break
    return np.stack([frames[int(i)] for i in idx])


def _resize_short_side(frames: np.ndarray, short: Optional[int] = SHORT_SIDE) -> np.ndarray:
    """Match the Kinetics frame directories: shortest side `short`, aspect kept.
    `short=None` keeps the video's own resolution."""
    h, w = frames.shape[1:3]
    if short is None or min(h, w) == short:
        return frames
    if w >= h:
        size = (int(round(w * short / h / 2)) * 2, short)
    else:
        size = (short, int(round(h * short / w / 2)) * 2)
    return np.stack([np.array(Image.fromarray(f).resize(size, Image.BICUBIC)) for f in frames])


def _read_video(path: str, num_frames: int, short_side: Optional[int] = SHORT_SIDE) -> np.ndarray:
    try:
        frames = _decode_decord(path, num_frames)
    except ImportError:
        frames = _decode_pyav(path, num_frames)
    return _resize_short_side(frames, short_side)


def read_clip(path: str, num_frames: int, short_side: Optional[int] = SHORT_SIDE) -> np.ndarray:
    """One clip -- a frame directory or a video file -> (num_frames, H, W, 3) uint8.

    `short_side` applies to video files only (frame directories are read as they
    are): 256 matches the Kinetics train frames, None keeps the video's own
    resolution, which matches the Kinetics val frames.
    """
    if os.path.isdir(path):
        return _read_frame_dir(path, num_frames)
    if os.path.isfile(path):
        return _read_video(path, num_frames, short_side)
    raise FileNotFoundError(f"neither a frame directory nor a video file: {path}")


# --------------------------------------------------------------------- listing


def list_clips(root: str) -> tuple[list[str], list[tuple[str, int]]]:
    """(classes, [(key, label), ...]) for `<root>/<class>/<clip>` data.

    Classes are the sorted sub-directories of `root` and the label is a class's
    position in that list; clips are sorted within each class. A clip is a
    frame directory or an `.mp4` file (its key drops the extension). One
    directory read per class, so listing Kinetics costs ~400 reads.
    """
    with os.scandir(root) as it:
        classes = sorted(e.name for e in it if e.is_dir())
    samples = []
    for label, name in enumerate(classes):
        with os.scandir(os.path.join(root, name)) as it:
            clips = [e.name if e.is_dir() else e.name[: -len(VIDEO_EXT)]
                     for e in it if e.is_dir() or e.name.lower().endswith(VIDEO_EXT)]
        samples += [(f"{name}/{c}", label) for c in sorted(clips)]
    return classes, samples


def load_split(path, split: str) -> dict:
    """One split of a clip-list file (datafile/*.json, format in datafile/README.md)
    -> {samples, classes, score_classes, short_side, sha256, description, file, split}.

    `samples` holds one tuple per clip -- (key, label), or (fg_key, label, bg_key,
    bg_label) for HAT. `score_classes` may be None.
    """
    raw = Path(path).read_bytes()
    m = json.loads(raw)
    if split not in m.get("splits", {}):
        raise SystemExit(f"{path} has splits {list(m.get('splits', {}))}; asked for {split!r}")
    s = m["splits"][split]
    return {"samples": [tuple(x) for x in s["samples"]],
            "classes": m.get("classes"),
            "score_classes": m.get("score_classes"),
            "short_side": s.get("short_side", SHORT_SIDE),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "description": s.get("description", m.get("description", "")),
            "file": Path(path).name,
            "split": split}


def check_data_root(root: Optional[str], samples: list, env_name: str = "the data root"):
    """Fail fast, with a message that says why, if the keys do not resolve."""
    if not root:
        raise SystemExit(f"no data root: set {env_name} in .env (or pass it on the command line)")
    if not os.path.isdir(root):
        raise SystemExit(f"{env_name} = {root!r} is not a directory (see .env)")
    key = samples[0][0]
    path = os.path.join(root, key)
    if not (os.path.isdir(path) or os.path.isfile(path) or os.path.isfile(path + VIDEO_EXT)):
        raise SystemExit(
            f"the first clip in the list, {key!r}, is neither {path}/ (frames) nor a video "
            f"file ({path} or {path}{VIDEO_EXT}) -- is {env_name} the right root?")


def stride_subsample(samples: list, n: int) -> list:
    """`n` samples evenly spaced through the list, so every class is represented."""
    if n is None or n >= len(samples):
        return list(samples)
    step = len(samples) / n
    return [samples[int(i * step)] for i in range(n)]


# ---------------------------------------------------------------------- dataset


class ClipList(Dataset):
    """[(key, label), ...] -> (frames (T, H, W, 3) uint8, label, key).

    An unreadable clip (Kinetics-400 train has ~1,460 empty frame directories)
    is replaced by the next clip in the list, which is also how the clips behind
    the shipped SAEs were read. The returned `key` is the clip actually read, so
    callers can detect and record substitutions.
    """

    def __init__(self, root: str, samples: list, num_frames: int,
                 short_side: Optional[int] = SHORT_SIDE):
        self.root = root
        self.samples = [tuple(s) for s in samples]
        self.num_frames = num_frames
        self.short_side = short_side

    def __len__(self):
        return len(self.samples)

    def load(self, key: str) -> np.ndarray:
        path = os.path.join(self.root, key)
        if not os.path.isdir(path) and os.path.isfile(path + VIDEO_EXT):
            path += VIDEO_EXT
        return read_clip(path, self.num_frames, self.short_side)

    def __getitem__(self, i: int):
        for offset in range(8):
            key, label = self.samples[(i + offset) % len(self.samples)]
            try:
                return self.load(key), label, key
            except Exception as e:
                if offset == 0:
                    print(f"[ClipList] failed on {key}: {e}", flush=True)
        raise RuntimeError(f"Could not read any clip near index {i}")


def make_frame_collate_fn(processor):
    """ClipList items -> the backbone's `pixel_values` plus labels and keys."""

    def collate_fn(batch):
        inputs = processor([list(item[0]) for item in batch], return_tensors="pt")
        inputs["labels"] = torch.tensor([item[1] for item in batch])
        inputs["paths"] = [item[2] for item in batch]
        return inputs

    return collate_fn
