"""Evaluation tables: definitions, and the paper's stored numbers as a reference."""
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atpgllm.training import eval_report as report

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO.parent / "libatpgllm"


def test_pass_at_k_is_the_unbiased_estimator():
    assert report.pass_at_k(16, 0, 1) == 0.0
    assert report.pass_at_k(16, 4, 1) == 0.25
    assert report.pass_at_k(4, 1, 4) == 1.0


def slot(pid, detected, *, vector=None, calls=1, executions=1, activated=None, final=None):
    vector = {"a": 1} if vector is None else vector
    return {"example_id": pid, "vector": vector, "usage": {"simulator_executions": executions, "tool_calls": calls},
            "observations": [{"vector": vector}] * calls, "final_answer": final,
            "reward_components": {"fault_detected_by_pred_input_vector_acc_logonly": float(detected),
                                  "fault_site_activated_acc_logonly": float(detected if activated is None else activated)}}


REFERENCE = {"p1": {"level": "easy", "p": 0.5, "circuit_id": "c1", "fault": "sa0 n", "random_activation": 0.5,
                    "random_observability": 0.5},
             "p2": {"level": "hard", "p": 0.0, "circuit_id": "c2", "fault": "sa0 m", "random_activation": 0.25,
                    "random_observability": None}}


def row(result, table, name, **match):
    return next(r for r in result[table] if r["bin"] == name and all(r[k] == v for k, v in match.items()))


def test_an_answer_with_no_real_call_counts_only_in_the_lenient_curve():
    slots = [slot("p1", True, calls=0), slot("p1", False), slot("p2", True), slot("p2", False, vector={})]
    result = report.analyze(slots, REFERENCE, tag="m")
    assert row(result, "effort", "easy", k=1)["pass_at_k"] == 0.5
    assert row(result, "effort", "easy", k=1)["strict_pass_at_k"] == 0.0
    assert row(result, "effort", "hard", k=1)["strict_pass_at_k"] == 0.5
    assert row(result, "outcomes", "easy")["no_real_call"] == 0.5
    assert row(result, "outcomes", "hard")["no_vector"] == 0.5


def test_propagation_follows_the_paper_and_keeps_the_wider_count_beside_it():
    slots = [slot("p1", True), slot("p1", False, activated=True), slot("p2", True)]
    result = report.analyze(slots, REFERENCE, tag="m")
    assert row(result, "mechanism", "easy")["propagation"] == 0.5
    assert row(result, "mechanism", "hard")["propagation_all"] == 1.0
    assert row(result, "mechanism", "hard")["propagation"] != row(result, "mechanism", "hard")["propagation"]  # nan


def test_false_detection_claims_and_retries():
    slots = [slot("p1", False, final='DETECTED_FAULTS: "sa0 n"'), slot("p1", False, calls=2, final='DETECTED_FAULTS: ""')]
    result = report.analyze(slots, REFERENCE, tag="m")
    assert row(result, "outcomes", "easy")["false_detection_claims"] == 0.5
    assert row(result, "outcomes", "easy")["retried"] == 0.5


def test_extra_slots_on_a_few_problems_do_not_create_a_larger_k():
    slots = [slot(p, True) for p in ("p1", "p2") for _ in range(8)] + [slot("p1", True) for _ in range(8)]
    reference = {**REFERENCE, "p3": dict(REFERENCE["p2"])}
    slots += [slot("p3", True) for _ in range(8)]
    result = report.analyze(slots, reference, tag="m")
    assert result["slots_per_problem"] == 8
    assert max(r["k"] for r in result["effort"]) == 8


@pytest.mark.skipif(not (PAPER / "runs/eval_results_grpo_granite_4.2_8b_repaired_a100_bon32_policy").exists(),
                    reason="paper runs not present")
def test_stored_ckpt25_numbers_are_reproduced():
    runs = PAPER / "runs/eval_results_grpo_granite_4.2_8b_repaired_a100_bon32_policy"
    bins = {"easy": "easy", "medium": "medium", "hard": "hard", "very_hard": "very_hard", "unresolved": "random_never_detects"}

    def number(text):
        try:
            return float(text)
        except (TypeError, ValueError):
            return None

    reference = {}
    for r in csv.DictReader(open(PAPER / "analysis/shared_eval_manifest_20260930_65k/manifest_difficulty_65536.problems.csv")):
        reference[r["problem_id"]] = dict(
            level=next(v for k, v in bins.items() if r["difficulty_bin"].startswith(k)),
            p=number(r["random_detection_probability"]), random_activation=number(r.get("activation_probability")),
            random_observability=number(r.get("conditional_observability")), circuit_id=r.get("circuit_id"),
            fault=r.get("fault"))
    slots = []
    for line in open(runs / "checkpoint-25_passatk_single_completion_csv1_nc16_t0.7_topp0.95_b64.slots.jsonl"):
        record = json.loads(line)
        if "evaluation_problem_id" in record:
            record.pop("messages", None)
            slots.append(record)
    result = report.analyze(slots, reference, tag="ckpt25")
    assert row(result, "effort", "all", k=1)["pass_at_k"] == pytest.approx(0.281, abs=5e-4)
    assert row(result, "effort", "all", k=16)["pass_at_k"] == pytest.approx(0.725, abs=5e-4)
    assert row(result, "effort", "hard_three", k=1)["pass_at_k"] == pytest.approx(0.138, abs=5e-4)
    for name, act, prop in (("all", 0.603, 0.593), ("easy", 0.707, 0.888), ("hard", 0.528, 0.321)):
        assert row(result, "mechanism", name)["activation"] == pytest.approx(act, abs=5e-4)
        assert row(result, "mechanism", name)["propagation"] == pytest.approx(prop, abs=5e-4)


def test_figures_are_written(tmp_path):
    slots = [slot("p1", True), slot("p1", False), slot("p2", False)]
    result = report.analyze(slots, REFERENCE, tag="round_6")
    report.effort_figure([result], REFERENCE, tmp_path / "effort")
    report.mechanism_figure([result], tmp_path / "mechanism")
    assert (tmp_path / "effort.png").stat().st_size and (tmp_path / "mechanism.pdf").stat().st_size


def test_stop_after_step_requests_save_and_stop_without_touching_max_steps():
    pytest.importorskip("transformers")
    from atpgllm.training.callbacks import StopAfterStepCallback
    control = SimpleNamespace(should_training_stop=False, should_save=False)
    args = SimpleNamespace(max_steps=100)
    callback = StopAfterStepCallback(4)
    callback.on_step_end(args, SimpleNamespace(global_step=3), control)
    assert not control.should_training_stop
    callback.on_step_end(args, SimpleNamespace(global_step=4), control)
    assert control.should_training_stop and control.should_save and args.max_steps == 100


def test_evaluate_keeps_only_whole_problems_on_resume(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("rft_evaluate", REPO / "scripts/rft/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._line_id('{"example_id": "a"}\n') == "a"
    assert module._line_id('{"example') is None
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"examples": [{"_fixed_eval_id": "a", "prompt": "p"}, {"_fixed_eval_id": "b", "prompt": "q"}]}))
    assert [item[1] for item in module.manifest_items(manifest)] == ["fixed", "fixed"]
    (tmp_path / "m.json.reference.json").write_text(json.dumps({"a": {"level": "hard"}, "b": {"level": "easy"}}))
    assert [item[1] for item in module.manifest_items(manifest)] == ["hard", "easy"]
