"""Evaluation datasets: one YAML per dataset in config/datasets/.

    from data.datasets import DATASETS, dataset_split, dataset_roots, build_dataset
    spec = DATASETS["nturgbd"]
    split = dataset_split(spec)                  # clip list, classes, score_classes, ...
    roots = dataset_roots(spec)                  # data roots, from .env
    ds, collate = build_dataset(spec, split, model, roots)

A dataset config says where the data lives (`root:` / `roots:`, names of .env
entries), which clip list and split to score (`clips:`, `split:`), how an item is
loaded (`loader:` clips | hat), which of the model's heads scores it
(`label_space:`), and what is measured (`metrics:` classification | hat).
"""

from pathlib import Path

import yaml

from data.clips import ClipList, check_data_root, load_split, make_frame_collate_fn
from data.hat import HATComposites, hat_collate
from utils.paths import data_root

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "config" / "datasets"


def _load(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text())
    if spec.get("name") != path.stem:
        raise ValueError(f"{path}: name {spec.get('name')!r} != file name {path.stem!r}")
    return spec


DATASETS = {p.stem: _load(p) for p in sorted(DATASET_DIR.glob("*.yaml"))}


def get_dataset(name: str) -> dict:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset {name!r}. Configured: {', '.join(DATASETS)} "
                         f"(config/datasets/<name>.yaml)") from None


def dataset_split(spec: dict) -> dict:
    """The dataset's clip list: see data.clips.load_split."""
    split = load_split(ROOT / spec["clips"], spec.get("split"))
    if split["classes"] is None:
        raise SystemExit(f"{spec['clips']} has no 'classes' list; evaluation needs one")
    return split


def dataset_roots(spec: dict, override: str = None) -> dict:
    """{role: path} from the .env entries the config names. `override` replaces
    the single root of a one-root dataset (a --data_root flag)."""
    if "root" in spec:
        return {"root": override or data_root(spec["root"])}
    if override:
        raise SystemExit(f"{spec['name']} reads several roots ({', '.join(spec['roots'].values())}); "
                         f"set them in .env instead of --data_root")
    return {role: data_root(env) for role, env in spec["roots"].items()}


def check_roots(spec: dict, roots: dict, samples: list):
    """Fail fast, with a message naming the .env entry, if the data is not there."""
    if spec["loader"] == "clips":
        check_data_root(roots["root"], samples, env_name=spec["root"])
    elif spec["loader"] == "hat":
        fg, _, bg, _ = samples[0]
        env = spec["roots"]
        check_data_root(roots["original"], [(fg,)], env_name=env["original"])
        check_data_root(roots["hat"], [(f"seg/{fg}",)], env_name=env["hat"])
        check_data_root(roots["hat"], [(f"{spec['hat']['inpaint_dir']}/{bg}",)], env_name=env["hat"])


def build_dataset(spec: dict, split: dict, model, roots: dict, samples: list = None):
    """-> (Dataset, collate_fn) for `model`; `samples` defaults to the split's."""
    samples = split["samples"] if samples is None else samples
    if spec["loader"] == "clips":
        ds = ClipList(roots["root"], samples, model.num_frames, split["short_side"])
        return ds, make_frame_collate_fn(model.processor)
    if spec["loader"] == "hat":
        ds = HATComposites(samples, roots["hat"], roots["original"], model.num_frames,
                           model.processor, **spec.get("hat", {}))
        return ds, hat_collate
    raise ValueError(f"{spec['name']}: unknown loader {spec['loader']!r} (clips | hat)")
