"""Protocol invariants: metrics must not hide malformed or unsound tests."""
import importlib.util
from pathlib import Path
import sys
import json
import hashlib
import subprocess

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/eval"
spec = importlib.util.spec_from_file_location("lot_metrics", SCRIPTS / "lot_metrics.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def parse(text):
    return m.parse_answer(text, ["a"], ["y"])


def test_detection_is_not_usable_when_good_circuit_would_fail():
    p = parse('INPUT_VECTOR: "a: 1"\nEXPECTED_OUTPUT: "y: 0"')
    result = m.outcomes(p, {"y": 1}, {"y": 0})
    assert result["D"] and not result["S"] and not result["U"]
    p = parse('INPUT_VECTOR: "a: 1"\nEXPECTED_OUTPUT: "y: 1"')
    assert m.outcomes(p, {"y": 1}, {"y": 0})["U"]  # no target-name requirement


@pytest.mark.parametrize("value", ['a: 1, a: 0', 'a: X', 'a: 1, extra: 0', '{}', '{"a": 1, "a": 0}', '{"a": True}'])
def test_ambiguous_or_incomplete_vectors_are_not_stimuli(value):
    with pytest.raises((ValueError, SyntaxError)):
        m.assignments(value, ["a"])


def test_fault_status_unknown_is_not_nondetection_and_soundness_is_independent():
    p = parse('INPUT_VECTOR: "a: 1"\nEXPECTED_OUTPUT: "y: 1"')
    result = m.outcomes(p, {"y": 1}, None)
    assert result["S"] and not result["D"] and not result["verification_known"]


def test_final_fields_do_not_use_tool_or_earlier_reasoning():
    text = '<think>INPUT_VECTOR: "a: 0"</think>\n<tool_response>INPUT_VECTOR: "a: 0"</tool_response>\nINPUT_VECTOR: "a: 1"'
    assert parse(text)["input"] == {"a": 1}
    assert parse('INPUT_VECTOR: "a: 0"\nINPUT_VECTOR: "a: 1"')["input"] is None


def test_pass_at_k_and_impossible_k():
    assert m.pass_at_k(10, 2, 2) == pytest.approx(1 - 28 / 45)
    assert m.pass_at_k(5, 0, 5) == 0
    assert m.pass_at_k(5, 1, 5) == 1
    with pytest.raises(ValueError):
        m.pass_at_k(1, 1, 2)


def test_macro_weights_circuits_and_keeps_failed_slots():
    rows = [dict(example_id=str(i), circuit_id="large", Vx=True, D=True, S=True, U=True, verification_known=True) for i in range(9)]
    rows += [dict(example_id="failed", circuit_id="small", Vx=False, D=False, S=False, U=False, verification_known=False)]
    summary = m.summarize(rows)
    assert summary["U_pass@1"]["macro"] == .5
    assert summary["U_pass@1"]["micro"] == .9
    assert summary["unknown_verification_slots"] == 1


def test_pattern_duplicates_count_in_cost_but_do_not_increase_coverage():
    stream = [{"a"}, {"a"}, {"b"}, {"a", "b"}]
    curve = m.coverage_curve(stream, ["a", "b", "c"])
    assert [r["detected"] for r in curve] == [1, 1, 2, 2]
    assert m.compact(stream) == [3]
    with pytest.raises(ValueError):
        m.coverage_curve([{"outside"}], ["a"])


@pytest.mark.parametrize("terminal,successes", [("stop", 1), ("length", 0)])
def test_replay_keeps_missing_and_truncated_slots(tmp_path, terminal, successes):
    example = {"_fixed_eval_id": "one", "fault": "sa0 a", "module_name": "tiny",
               "netlist": "module tiny(a,y);\ninput a;\noutput y;\nassign y=a;\nendmodule", "prompt": "test"}
    examples = [example]
    checksum = hashlib.sha256(json.dumps(examples, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    protocol = {"generations": 2, "split": "validation"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"examples": examples, "examples_sha256": checksum, "protocol": protocol}))
    records = tmp_path / "records.json"
    records.write_text(json.dumps({"step": 0, "examples_sha256": checksum, "protocol": protocol,
        "records": [{"example_id": "one", "terminal_status": terminal,
                     "completion": 'INPUT_VECTOR: "a: 1"\nEXPECTED_OUTPUT: "y: 1"'}]}))
    output = tmp_path / "out"
    run = subprocess.run([sys.executable, str(SCRIPTS / "execute_language_of_test.py"), "replay",
        "--manifest", str(manifest), "--records", str(records), "--output", str(output), "--k", "1", "2"],
        text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "incomplete_source"
    assert summary["internal"]["slots"] == 2
    assert summary["internal"]["U_pass@1"]["micro"] == successes / 2
    assert summary["internal"]["U_pass@2"]["micro"] == successes


def test_prepare_control_preserves_the_scoring_target_and_loads_policy(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import generate_lot_no_feedback as gen
    from types import SimpleNamespace
    source = {"protocol": {}, "examples": [{"_fixed_eval_id": "id", "fault": "sa0 a", "netlist": "module x; endmodule"}]}
    monkeypatch.setattr(gen, "load_manifest", lambda _: source)
    manifest = tmp_path / "source.json"
    manifest.write_text("frozen")
    class Tokenizer:
        chat_template = "same"
        def apply_chat_template(self, messages, **kwargs):
            return str(messages)
        def encode(self, text, **kwargs):
            return list(text)
        def get_vocab(self):
            return {"a": 1}
    args = SimpleNamespace(manifest=manifest, condition="opposite-fault", n=4, seed=7,
        temperature=.6, max_new_tokens=100, context_length=1000, tokenizer=tmp_path)
    result = gen.freeze(args, Tokenizer())
    assert result["examples"][0]["fault"] == "sa0 a"
    assert result["examples"][0]["prompt_fault"] == "sa1 a"
    (tmp_path / "policy").mkdir()
    (tmp_path / "policy/adapter_config.json").write_text("{}")
    (tmp_path / "adapter_config.json").write_text("{}")
    assert gen.resolve_adapter(tmp_path) == tmp_path / "policy"


def test_unknown_reference_cells_create_bounds_not_false_certainty(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import execute_language_of_test as execute
    from types import SimpleNamespace
    source = tmp_path / "source"
    source.mkdir()
    circuit = {"module_name": "tiny", "netlist": "tiny", "outputs": ["y"], "faults": ["sa0 y", "sa1 y"]}
    (source / "fault_manifest.json").write_text(json.dumps({"circuits": {"c": circuit}}))
    (source / "slots.jsonl").write_text(json.dumps({"circuit_id": "c", "parsed": {"expected": {"y": 1}}}) + "\n")
    (source / "coverage-c.json").write_text(json.dumps({
        "patterns": {"m": {"a": 1}, "r": {"a": 0}},
        "detected_fault_matrix": {"m": ["sa0 y"], "r": []},
        "streams": {"model_stimulus": {"ordered_vector_ids": ["m"]}, "uniform_stimulus": {"ordered_vector_ids": ["r"]}}}))
    def reference(text, vector, fault, output_names):
        if fault == "sa1 y":
            return {"status": "unknown", "error": "reference unavailable"}
        return {"status": "verified", "good": {"y": vector["a"]}, "bad": {"y": 0}}
    monkeypatch.setattr(execute, "reference", reference)
    output = tmp_path / "out"
    execute.audit_coverage(SimpleNamespace(replay=source, output=output, workers=1, reuse_reference=None))
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "incomplete_reference_lower_bounds"
    assert summary["macro_FC_all_bounds"]["model_stimulus"] == [.5, 1]
    assert summary["macro_FC_all_bounds"]["uniform_stimulus"] == [0, .5]
    assert summary["circuits"]["c"]["model_testable_coverage_bounds"] is None
