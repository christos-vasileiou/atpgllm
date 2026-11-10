"""Parallel failed-attempt search keeps the serial order and results."""

import importlib.util
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location(
        "multimodal_training_under_test", ROOT / "scripts" / "train" / "multimodal_training.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_searched_matches_serial_order_and_values(monkeypatch):
    mt = _module()

    def fake_failed_attempt(record, rng, max_flips):
        time.sleep(random.random() * 0.01)  # finish out of order
        return None if record["i"] % 7 == 0 else (record["i"], rng.random())

    monkeypatch.setattr(mt, "failed_attempt", fake_failed_attempt)
    records = [{"i": i, "circuit_id": i % 5, "fault": f"sa0 n{i}", "pattern_index": i,
                "input_vector": "{}"} for i in range(60)]
    args = SimpleNamespace(retry_fraction=0.5, max_flips=3, example_workers=1)
    serial = list(mt._searched(iter(records), args, "42"))
    args.example_workers = 6
    parallel = list(mt._searched(iter(records), args, "42"))
    assert [r["i"] for r, _ in parallel] == list(range(60))
    assert parallel == serial
    assert any(a is mt._NO_RETRY for _, a in serial) and any(a is None for _, a in serial)
