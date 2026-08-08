"""Path helpers for the training package (monorepo + package data)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# atpgllm/training/_paths.py → parents[2] = libatpgllm, parents[3] = monorepo root
_THIS = Path(__file__).resolve()
LIBATPGLLM_ROOT = _THIS.parents[2]
MONOREPO_ROOT = _THIS.parents[3]
DATA_PREPROCESSING = MONOREPO_ROOT / "data_preprocessing"
PACKAGE_DATA_DIR = _THIS.parent / "data"
DEFAULT_SIM_CONFIG = PACKAGE_DATA_DIR / "sim_config.json"


def ensure_data_preprocessing_on_path() -> Path:
    """Insert sibling ``data_preprocessing/`` on ``sys.path`` once; return its path."""
    path = str(DATA_PREPROCESSING)
    if path not in sys.path:
        sys.path.insert(0, path)
    return DATA_PREPROCESSING


def resolve_sim_config_path(config_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve ``sim_config.json`` via env override, explicit path, or package data."""
    override = os.environ.get("SIM_CONFIG")
    if override and config_path is None:
        config_path = override

    if config_path is None:
        return DEFAULT_SIM_CONFIG

    p = Path(config_path).expanduser()
    if p.is_absolute():
        return p if p.exists() else DEFAULT_SIM_CONFIG
    if p.exists():
        return p.resolve()
    # Bare default name (or relative miss) → package data next to this module
    if p.name == "sim_config.json" or str(p) in ("sim_config.json", "./sim_config.json"):
        return DEFAULT_SIM_CONFIG
    packaged = PACKAGE_DATA_DIR / p.name
    if packaged.exists():
        return packaged
    return p
