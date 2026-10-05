"""Where the data lives: the roots in `.env` at the repo root.

    K400_TRAIN=/data/kinetics400/train
    K400_VAL=/data/kinetics400/val

The scripts use these as the defaults of their --data_root / --train_root /
--val_root flags. Precedence: the command-line flag, then an environment
variable of the same name, then `.env`.
"""

import os
from pathlib import Path
from typing import Optional

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"      # at the repo root


def _read_env(path: Path = ENV_FILE) -> dict:
    """KEY=VALUE lines; blank lines and # comments ignored; ~ and $VARS expanded."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        out[key.strip()] = os.path.expanduser(os.path.expandvars(value))
    return out


def data_root(name: str) -> Optional[str]:
    """The root called `name` (e.g. "K400_VAL"): environment first, then .env."""
    return os.environ.get(name) or _read_env().get(name) or None
