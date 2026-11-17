"""Targets found by the scripted search (compacting the search into the graph model)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def sl():
    sys.path.insert(0, str(ROOT / "scripts" / "eval"))
    spec = importlib.util.spec_from_file_location("search_labels", ROOT / "scripts" / "eval" / "search_labels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RICH = {"inputs": {"a": [0.99, 0.98], "b": [0.9, 0.45], "c": [0.4, 0.9]}}
INPUTS = ["a", "b", "c", "d"]


def test_search_stops_at_the_first_detecting_attempt(sl):
    tried = []

    def simulate(vector):
        tried.append(dict(vector))
        return vector["b"] == 1          # only the flipped b detects

    found = sl.search({}, RICH, INPUTS, 16, simulate)
    assert found["found"] == 1 and found["tried"] == 2 and found["vector"] == {"a": 1, "b": 1}
    assert tried[0] == {"a": 1, "b": 0, "c": "x", "d": "x"} and len(tried) == 2
    none = sl.search({}, RICH, INPUTS, 16, lambda vector: False)
    assert none["found"] == -1 and none["vector"] is None and none["tried"] == none["candidates"]
    assert sl.search({}, RICH, INPUTS, 16, lambda vector: True)["found"] == 0


def test_relabel_and_merge(sl, tmp_path):
    row = {"netlist_key": "0a", "fault": "sa0 n1", "inputs": INPUTS, "outputs": ["y"],
           "care_freq": {"a": 1.0, "b": 0.6, "c": 0.4, "d": 0.0},
           "one_freq": {"a": 1.0, "b": 0.3, "c": 0.9, "d": 0.5},
           "search": {"found": 1, "tried": 2, "candidates": 3, "vector": {"a": 1, "b": 1}}}
    hard = sl.relabel(row, 1.0)
    assert hard["care_freq"] == {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0}
    assert hard["one_freq"] == {"a": 1.0, "b": 1.0, "c": 0.9, "d": 0.5}      # unknown inputs keep their value
    assert hard["search_found"] == 1 and "search" not in hard
    half = sl.relabel(row, 0.5)
    assert half["care_freq"]["b"] == pytest.approx(0.8) and half["care_freq"]["c"] == pytest.approx(0.2)
    assert half["one_freq"]["b"] == pytest.approx(0.65)
    assert sl.relabel({**row, "search": {"found": -1, "vector": None}}, 1.0) is None

    labels, out, into = tmp_path / "labels", tmp_path / "out", tmp_path / "into"
    labels.mkdir(), out.mkdir()
    (labels / "circuits_train.jsonl").write_text(json.dumps({"netlist_key": "0a", "netlist": "x"}) + "\n"
                                                 + json.dumps({"netlist_key": "0b", "netlist": "y"}) + "\n")
    for name in ("circuits_validation.jsonl", "problems_validation.jsonl"):
        (labels / name).write_text("{}\n")
    first = {**row, "fault": "sa1 n1", "search": {"found": 0, "tried": 1, "candidates": 3, "vector": {"a": 1}}}
    lost = {**row, "fault": "sa0 y", "search": {"found": -1, "tried": 3, "candidates": 3, "vector": None}}
    (out / "rows.s000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (row, first, lost)))
    args = type("A", (), {"into": into, "out": out, "labels": labels, "split": "train",
                          "eval_split": "validation", "blend": 1.0, "only_later": False})()
    sl.run_merge(args)
    rows = [json.loads(line) for line in (into / "problems_train.jsonl").open()]
    assert [r["fault"] for r in rows] == ["sa0 n1", "sa1 n1"]
    assert [json.loads(line)["netlist_key"] for line in (into / "circuits_train.jsonl").open()] == ["0a"]
    assert (into / "problems_validation.jsonl").is_symlink()
    summary = json.loads((into / "summary.json").read_text())
    assert summary["searched"] == 3 and summary["first"] == 1 and summary["later"] == 1 and summary["written"] == 2
    args.only_later, args.into = True, tmp_path / "later"
    sl.run_merge(args)
    assert len((tmp_path / "later" / "problems_train.jsonl").read_text().splitlines()) == 1


def test_wanted_problems_parses_only_the_wanted_rows(sl, tmp_path):
    rows = [{"netlist_key": "0a", "netlist_id": "n", "circuit_id": "c", "fault": "sa0 \\add/n1", "inputs": ["a"]},
            {"netlist_key": "0b", "netlist_id": "n", "circuit_id": "c", "fault": "sa1 y", "inputs": ["b"]}]
    (tmp_path / "problems_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = list(sl.wanted_problems(tmp_path, "train", {("0a", "sa0 \\add/n1"): {}}))
    assert [r["netlist_key"] for r in got] == ["0a"]
