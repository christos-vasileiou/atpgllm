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


def test_in_background_keeps_order_and_raises_the_source_error():
    mt = _module()
    assert list(mt._in_background(iter(range(50)), ahead=4)) == list(range(50))

    def broken():
        yield 1
        raise KeyError("bad row")

    items = mt._in_background(broken(), ahead=4)
    assert next(items) == 1
    try:
        next(items)
    except KeyError as exc:
        assert exc.args == ("bad row",)
    else:
        raise AssertionError("the source error was swallowed")


def test_closing_in_background_ends_its_thread_and_closes_the_source():
    """A --stop-after run returns mid-stream: the example thread must be gone
    before the interpreter exits (juno 506526 and 506528 aborted on exit)."""
    import threading

    mt = _module()
    closed = []

    def source():
        try:
            i = 0
            while True:  # endless, like the repeating training rows
                yield i
                i += 1
        finally:
            closed.append(threading.current_thread().name)

    before = set(threading.enumerate())
    items = mt._in_background(source(), ahead=3)
    assert [next(items) for _ in range(5)] == [0, 1, 2, 3, 4]
    worker = (set(threading.enumerate()) - before).pop()
    items.close()
    assert not worker.is_alive()
    # Closed inside the worker, not from the consuming thread.
    assert closed == [worker.name]


def test_closing_searched_early_leaves_no_search_threads(monkeypatch):
    import threading

    mt = _module()
    started = []

    def slow_attempt(record, args, salt):
        started.append(record["i"])
        time.sleep(0.02)
        return record["i"]

    monkeypatch.setattr(mt, "_tool_attempt", slow_attempt)
    before = set(threading.enumerate())
    args = SimpleNamespace(example_workers=2)
    searched = mt._searched(({"i": i} for i in range(1000)), args, "42")
    assert next(searched)[1] == 0
    searched.close()
    assert not [t for t in set(threading.enumerate()) - before if t.is_alive()]
    assert len(started) < 20  # queued searches were dropped, not run out


def test_a_record_too_wide_to_fit_never_reaches_the_simulator(monkeypatch):
    """TetraMAX refuses the pattern file of a circuit with thousands of
    inputs. Such a record cannot fit the length cap either, so the search for
    a failed attempt skips it instead of failing the run."""
    import json

    mt = _module()

    def simulator(*_args, **_kwargs):
        raise AssertionError("the simulator was called")

    monkeypatch.setattr(mt, "failed_attempt", simulator)
    args = SimpleNamespace(retry_fraction=1.0, max_flips=3, max_seq_len=14336)
    wide = {"circuit_id": "wide", "fault": "sa1 a[0]",
            "input_vector": json.dumps({f"a[{i}]": 0 for i in range(2883)}),
            "expected_output": json.dumps({f"y[{i}]": 0 for i in range(338)})}
    assert mt._too_wide(wide, args)
    assert mt._tool_attempt(wide, args, "42") is mt._NO_RETRY

    narrow = {"circuit_id": "narrow", "fault": "sa1 a",
              "input_vector": json.dumps({"a": 0, "b": 1}), "expected_output": json.dumps({"y": 0})}
    assert not mt._too_wide(narrow, args)
    monkeypatch.setattr(mt, "failed_attempt", lambda *a, **k: "searched")
    assert mt._tool_attempt(narrow, args, "42") == "searched"
