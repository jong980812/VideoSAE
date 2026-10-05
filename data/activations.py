"""Activation shards (`*_part{n}.pt`, written by extract_activations.py) as a dataset."""

import bisect
import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ActivationsDataset(Dataset):
    """All activation shards of one layer held in memory, as float32 rows.

    ~1.3M rows per layer (5,000 clips x 256 tokens) is 4-6 GB in fp32, so this
    fits on one GPU; pass `device="cpu"` otherwise.
    """

    def __init__(self, directory, device="cpu", dtype=torch.float32):
        self.directory = directory
        self.files = _sorted_shards(directory)
        self.device = device

        self.cached_tensors = []
        self.cumulative_lengths = []
        cumulative = 0
        for file in tqdm(self.files, desc=f"load {os.path.basename(os.path.normpath(directory))}",
                         unit="shard"):
            tensor = torch.load(os.path.join(directory, file), map_location="cpu")
            tensor = tensor.to(device=device, dtype=dtype)
            self.cached_tensors.append(tensor)
            cumulative += tensor.size(0)
            self.cumulative_lengths.append(cumulative)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        shard = bisect.bisect_right(self.cumulative_lengths, idx)
        start = self.cumulative_lengths[shard - 1] if shard > 0 else 0
        return self.cached_tensors[shard][idx - start]


def _sorted_shards(directory: str) -> list[str]:
    """Shards in the order they were written: by the number after `_part`."""
    files = [f for f in os.listdir(directory) if f.endswith(".pt") and "_part" in f]
    if not files:
        raise RuntimeError(f"No activation shards (*_part*.pt) found in {directory}")
    return sorted(files, key=lambda x: int(x.split("_part")[-1].split(".")[0]))
