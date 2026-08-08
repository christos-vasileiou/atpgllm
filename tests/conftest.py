"""Shared pytest fixtures for libatpgllm."""

from pathlib import Path

import pytest

# libatpgllm repo root (tests/conftest.py → parents[1])
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CODE = REPO_ROOT / "scripts" / "train" / "training_code.py"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def training_code_path() -> Path:
    return TRAINING_CODE
