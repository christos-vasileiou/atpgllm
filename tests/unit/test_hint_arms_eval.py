"""Rollout scoring and the graph-only retry policy of the hint arm check, without GPUs."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROW = json.loads((ROOT / "tests" / "unit" / "fixtures" / "v5_scan_row.json").read_text())


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location(
        "hint_arms_eval", ROOT / "scripts" / "eval" / "hint_arms_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["hint_arms_eval"] = module
    spec.loader.exec_module(module)
    return module


def test_answer_vector_fills_unnamed_inputs_with_x(ev):
    text = 'INPUT_VECTOR: "a[1]: 0, sel: 1, ghost: 1"\nEXPECTED_OUTPUT: "y: 1"'
    assert ev.answer_vector(text, ["a[0]", "a[1]", "sel"]) == {"a[0]": "x", "a[1]": 0, "sel": 1}
    assert ev.answer_vector("no report here", ["a"]) is None


def test_location_names_state_inputs(ev):
    assert ev.kind(ROW) == "sequential" and ev.kind({"fault": "sa0 a"}) == "combinational"
    assert ev.location(dict(ROW, fault="sa0 s")) == "state_in"        # a flop's QN net
    assert ev.location(dict(ROW, fault="sa0 tx1")) == "pi_stem"
    assert ev.location(ROW) == "internal"


def test_decoder_attempts_change_one_decision_each_least_sure_first(ev):
    rich = {"inputs": {"a": [0.99, 0.98], "b": [0.9, 0.55], "c": [0.4, 0.1], "d": [0.25, 0.9]}}
    vectors = ev.decoder_attempts(rich, ["a", "b", "c", "d", "e"], attempts=4)
    assert vectors[0] == {"a": 1, "b": 1, "c": "x", "d": "x", "e": "x"}          # the plain hint
    assert vectors[1] == {**vectors[0], "b": 0}      # the value nearest a coin flip is flipped first
    assert vectors[2] == {**vectors[0], "c": 0}      # then the likeliest input the hint left out
    assert vectors[3] == {**vectors[0], "d": 1}
    assert len(ev.decoder_attempts(rich, ["a", "b", "c", "d", "e"], attempts=9)) == 5


def test_report_scores_first_call_retries_and_final_answer(ev, tmp_path):
    def roll(row, flags, final, circuit):
        return {"row": row, "circuit_id": circuit, "kind": "sequential" if row % 2 else "combinational",
                "location": "internal", "difficulty": "easy", "final_vector": {"a": 1}, "truncated": False,
                "calls": [{"detected": f} for f in flags], "final_detected": final}

    plain = [roll(0, [False, True], True, 1), roll(1, [False, False, False], False, 1), roll(2, [True], True, 2)]
    rich = [roll(0, [True], True, 1), roll(1, [False, True], True, 1), roll(2, [True], True, 2),
            roll(3, [True], True, 2)]                        # row 3 only here: not in the shared set
    (tmp_path / "rollout_plain.jsonl").write_text("".join(json.dumps(r) + "\n" for r in plain))
    (tmp_path / "rollout_rich.s0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rich[:2]))
    (tmp_path / "rollout_rich.s1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rich[2:]))
    decoder = [{"row": i, "circuit_id": 1, "kind": "combinational", "location": "internal", "difficulty": "easy",
                "attempts": [i == 2, True, True], "answer_key": True} for i in range(3)]
    (tmp_path / "decoder.jsonl").write_text("".join(json.dumps(r) + "\n" for r in decoder)
                                            + json.dumps({"row": 3, "rejected": True}) + "\n")
    report = ev.build_report(tmp_path, calls=3)
    assert report["rows"] == 3
    table = report["tables"]["all"]
    assert table["plain"]["first"] == pytest.approx(1 / 3)
    assert table["plain"]["within_2"] == pytest.approx(2 / 3) and table["plain"]["within_3"] == pytest.approx(2 / 3)
    assert table["plain"]["final"] == pytest.approx(2 / 3) and table["plain"]["mean_calls"] == pytest.approx(2.0)
    assert table["rich"]["first"] == pytest.approx(2 / 3) and table["rich"]["within_2"] == 1.0
    assert table["graph_decoder"]["first"] == pytest.approx(1 / 3) and table["graph_decoder"]["within_2"] == 1.0
    assert table["graph_decoder"]["final"] is None and table["answer_key"]["first"] == 1.0
    pair = report["paired"]["plain minus rich, first"]
    assert pair["difference"] == pytest.approx(-1 / 3) and pair["rows"] == 3
    assert "plain minus rich, final" in report["paired"]
    assert "| rich | 3 |" in ev.markdown(report)


def test_second_pass_overlays_the_rows_that_ran_into_a_limit(ev, tmp_path):
    def roll(row, flags, final, **more):
        return {"row": row, "circuit_id": 1, "kind": "combinational", "location": "internal",
                "difficulty": "easy", "final_vector": {"a": 1} if final is not None else None,
                "truncated": False, "calls": [{"detected": f} for f in flags],
                "final_detected": bool(final), **more}

    base, out = tmp_path / "base", tmp_path / "long"
    base.mkdir(), out.mkdir()
    first = [roll(0, [True], True), roll(1, [False], None, truncated=True),
             roll(2, [False, False, False], None, call_limit_hit=True)]
    (base / "rollout_plain.jsonl").write_text("".join(json.dumps(r) + "\n" for r in first))
    assert ev.limit_rows(base, "plain") == {1, 2}
    again = [roll(1, [False, True], True), roll(2, [False, False, False, False, True], True)]
    (out / "rollout_plain.s0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in again))
    table = ev.build_report(out, calls=5, base=[base])["tables"]["all"]["plain"]
    assert table["n"] == 3 and table["first"] == pytest.approx(1 / 3)
    assert table["within_2"] == pytest.approx(2 / 3) and table["within_3"] == pytest.approx(2 / 3)
    assert table["within_5"] == 1.0 and table["final"] == 1.0 and table["truncated"] == 0.0
    assert ev.build_report(base, calls=3)["tables"]["all"]["plain"]["truncated"] == pytest.approx(1 / 3)
    (out / "rollout_other.jsonl").write_text(json.dumps(roll(1, [True], True)) + "\n")
    assert ev.build_report(out, calls=5, base=[base])["rows"] == 1
    assert list(ev.build_report(out, calls=5, base=[base], only=["plain"])["tables"]["all"]) == ["plain"]


def test_final_answer_ends_where_the_fault_list_begins(ev):
    import torch

    class Tok:
        words = {1: "INPUT_VECTOR: a: 1 ", 2: "DETECTED_FAULTS", 3: ": sa0 n1", 9: "<end>"}

        def decode(self, ids, **kwargs):
            return "".join(self.words.get(int(i), "?") for i in ids)

        def batch_decode(self, rows, **kwargs):
            return [self.decode(row) for row in rows]

    rule = ev.ends_answer(Tok(), 9)
    done = rule(torch.tensor([[1, 2], [1, 1], [1, 9]]), None)
    assert done.tolist() == [True, False, True]
    assert ev.close_answer(Tok(), [1, 2], 9) == [1, 2, 9]
    assert ev.close_answer(Tok(), [1, 9], 9) == [1, 9] and ev.close_answer(Tok(), [1, 1], 9) == [1, 1]


def test_rollout_row_reads_the_tool_and_simulates_the_answer(ev, monkeypatch):
    monkeypatch.setenv("FAULT_SIM_BACKEND", "fast")
    from atpgllm.training._paths import ensure_data_preprocessing_on_path

    ensure_data_preprocessing_on_path()
    pytest.importorskip("scan_view", reason="needs a data_preprocessing checkout with scan_view")
    import multimodal_training as mt
    from atpgllm.multimodal.data import netlist_key
    from atpgllm.training.revert_template import parse_tool_call

    record = dict(ROW)
    vector = json.loads(record["input_vector"])
    filled = {name: (0 if value == "x" else value) for name, value in vector.items()}

    def call(v):
        return ("<tool_call>\n<function=fault_simulation_tool>\n<parameter=input_vector>\n" + json.dumps(v)
                + "\n</parameter>\n<parameter=output_vector>\n" + record["expected_output"]
                + "\n</parameter>\n<parameter=fault>\n" + record["fault"] + "\n</parameter>\n<parameter=doc_id>\n"
                + netlist_key(record) + "\n</parameter>\n</function>\n</tool_call>\n")

    wrong = {**filled, "tx0": 1, "tx1": 1}
    turns = [call(wrong), call(filled), 'INPUT_VECTOR: "txreset: 0, tx1: 0, tx0: 0, s: 0"\n']
    results = mt._tool_runner(dict(record), "po_fault")([parse_tool_call(t) for t in turns[:2]])
    trajectory = SimpleNamespace(turns=turns, tool_results=results, truncated=False, call_limit_hit=False,
                                 reused_calls=0, completion_ids=[0] * 10)
    row = ev.rollout_row(trajectory, record)
    assert [c["vector"] for c in row["calls"]] == [wrong, filled]
    assert row["calls"][1]["detected"] and not row["calls"][1]["malformed"]
    assert row["final_vector"] == {"txreset": 0, "tx1": 0, "tx0": 0, "d": "x", "s": 0}
    assert ev.outcomes(row, 3)["within_2"] and ev.outcomes(row, 3)["first"] == row["calls"][0]["detected"]
