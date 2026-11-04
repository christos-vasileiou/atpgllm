"""The RFT driver's round loop: when rounds stop and which problems stay available."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def run_rft():
    spec = importlib.util.spec_from_file_location("run_rft", REPO / "scripts" / "rft" / "run_rft.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scores(*values):
    return {str(n): {"val": value, "fixed": 0.0} for n, value in enumerate(values)}


def test_stalled_rounds_counts_back_to_the_last_rise(run_rft):
    results = scores(0.10, 0.20, 0.20, 0.15, 0.14)
    assert [run_rft.stalled_rounds(results, n) for n in range(1, 5)] == [0, 1, 2, 3]


def test_a_rise_resets_the_count_even_below_the_best(run_rft):
    results = scores(0.30, 0.20, 0.25, 0.24)
    assert [run_rft.stalled_rounds(results, n) for n in range(1, 4)] == [1, 0, 1]


def test_equal_scores_with_different_float_bits_count_as_stalled(run_rft):
    assert run_rft.stalled_rounds(scores(0.22, 0.21999999999999997), 1) == 1
    assert run_rft.stalled_rounds(scores(0.21999999999999997, 0.22), 1) == 1


def drive(run_rft, monkeypatch, tmp_path, values, *, patience=3, rounds=20):
    """Run the loop with stubbed stages; returns the Run and the models each round started from."""
    plan = {"run_dir": str(tmp_path / "run"), "start_adapter": "a0", "rounds": rounds, "patience": patience,
            "deadline_hours": 1000, "round_hours_estimate": 1, "eval": {}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    started_from = []
    values = iter(values)

    def evaluate_model(r, tag, adapter):
        return {"adapter": adapter, "fixed": 0.0, "val": next(values), "val_by_level": {}}

    monkeypatch.setattr(run_rft, "select_validation", lambda r: None)
    monkeypatch.setattr(run_rft, "select_round", lambda r, n: None)
    monkeypatch.setattr(run_rft, "merge_export", lambda r, adapter, name: None)
    monkeypatch.setattr(run_rft, "generate", lambda r, n, model: started_from.append(model))
    monkeypatch.setattr(run_rft, "build", lambda r, n: {"sft_examples": 100})
    monkeypatch.setattr(run_rft, "train", lambda r, n, model, examples: f"a{n}")
    monkeypatch.setattr(run_rft, "evaluate_model", evaluate_model)
    r = run_rft.Run(path)
    (r.dir / "plan.json").write_text(json.dumps(plan))  # as when editing a running plan
    return r, run_rft.rounds(r), started_from


def test_rounds_stop_after_three_stalled_rounds_and_return_the_best(run_rft, monkeypatch, tmp_path):
    r, best, started_from = drive(run_rft, monkeypatch, tmp_path, [0.1, 0.2, 0.3, 0.3, 0.25, 0.2, 0.9])
    assert sorted(map(int, r.state["results"])) == [0, 1, 2, 3, 4, 5]
    assert "ending at round 5" in r.state["rounds_stopped"]
    assert best == "a2"
    # Every round continues from the one before it, stalled or not.
    assert started_from == ["a0", "a1", "a2", "a3", "a4"]


def test_a_rise_keeps_the_rounds_going(run_rft, monkeypatch, tmp_path):
    r, best, _ = drive(run_rft, monkeypatch, tmp_path, [0.1, 0.5, 0.4, 0.3, 0.35, 0.3, 0.2, 0.1])
    assert sorted(map(int, r.state["results"])) == list(range(8))
    assert best == "a1"


def test_restart_replays_finished_rounds_without_stopping_early(run_rft, monkeypatch, tmp_path):
    _, _, first = drive(run_rft, monkeypatch, tmp_path, [0.1, 0.2, 0.3], rounds=2)
    r, best, again = drive(run_rft, monkeypatch, tmp_path, [0.3, 0.2, 0.1], rounds=5)
    assert first == ["a0", "a1"]
    assert again == ["a2", "a3", "a4"]
    assert "ending at round 5" in r.state["rounds_stopped"]
    assert best == "a2"


def test_attempted_problems_excludes_only_what_was_sampled(run_rft, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    (tmp_path / "plan.json").write_text(json.dumps({"run_dir": str(tmp_path / "run")}))
    r = run_rft.Run(tmp_path / "plan.json")
    gen = r.path("round_1", "gen")
    gen.mkdir(parents=True)
    (gen / "slots.shard0.jsonl").write_text(
        json.dumps({"problem": ["n1", "sa0 a"]}) + "\n" + json.dumps({"problem": ["n1", "sa0 a"]}) + "\n"
        + '{"problem": ["n9", "half written')
    (gen / "slots.shard1.jsonl").write_text(json.dumps({"problem": ["n2", "sa1 b"]}) + "\n")
    rows = pq.read_table(run_rft.attempted_problems(r, 1)).to_pylist()
    assert rows == [{"netlist_id": "n1", "fault": "sa0 a"}, {"netlist_id": "n2", "fault": "sa1 b"}]


def driven_with_bench(run_rft, monkeypatch, tmp_path, *, from_round, fail=False):
    calls = []

    def run_benchmark(r, adapter, out):
        calls.append(adapter)
        if fail:
            raise RuntimeError("benchmark broke")
        return {"robust": 0.5, "level_mean": 0.5}

    monkeypatch.setattr(run_rft, "run_benchmark", run_benchmark)
    monkeypatch.setattr(run_rft, "write_report", lambda r, tag, directory: True)
    plan = {"run_dir": str(tmp_path / "run"), "start_adapter": "a0", "rounds": 3, "patience": 9,
            "deadline_hours": 1000, "round_hours_estimate": 1, "max_failures": 2,
            "eval": {"bench": {"manifest": "m", "from_round": from_round}}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    values = iter([0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr(run_rft, "evaluate_model",
                        lambda r, tag, adapter: {"adapter": adapter, "fixed": None, "val": next(values), "val_by_level": {}})
    for name, stub in (("select_validation", lambda r: None), ("select_round", lambda r, n: None),
                       ("merge_export", lambda r, adapter, name: None), ("generate", lambda r, n, model: None),
                       ("build", lambda r, n: {"sft_examples": 100}), ("train", lambda r, n, model, examples: f"a{n}")):
        monkeypatch.setattr(run_rft, name, stub)
    r = run_rft.Run(path)
    return r, run_rft.rounds(r), calls


def test_benchmark_runs_only_from_its_first_round(run_rft, monkeypatch, tmp_path):
    r, best, calls = driven_with_bench(run_rft, monkeypatch, tmp_path, from_round=2)
    assert calls == ["a2", "a3"]
    assert "robust" not in r.state["results"]["1"] and r.state["results"]["2"]["robust"] == 0.5
    assert "round_1/bench" not in r.state["done"] and "round_2/report" in r.state["done"]


def test_a_failing_benchmark_does_not_halt_the_rounds(run_rft, monkeypatch, tmp_path):
    r, best, calls = driven_with_bench(run_rft, monkeypatch, tmp_path, from_round=1, fail=True)
    assert best == "a3" and len(calls) == 3
    assert not r.state.get("halted") and r.state["failures"]["round_1/bench"] == 1


def test_a_finished_round_is_not_rerun_on_replay(run_rft, monkeypatch, tmp_path):
    r, _, calls = driven_with_bench(run_rft, monkeypatch, tmp_path, from_round=2)
    again = []
    monkeypatch.setattr(run_rft, "run_benchmark", lambda r, adapter, out: again.append(adapter))
    assert run_rft.rounds(r) == "a3"
    assert again == [] and r.state["results"]["3"]["robust"] == 0.5


def test_gdpo_waits_while_held(run_rft, monkeypatch, tmp_path):
    plan = {"run_dir": str(tmp_path / "run"), "gdpo": {"hold": True}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    r = run_rft.Run(path)
    sleeps = []

    def release(seconds):
        sleeps.append(seconds)
        (r.dir / "plan.json").write_text(json.dumps({**plan, "gdpo": {"hold": False}}))

    monkeypatch.setattr(run_rft.time, "sleep", release)
    run_rft.wait_while_held(r)
    assert sleeps == [60]
    run_rft.wait_while_held(r)
    assert sleeps == [60]


def test_gdpo_chunks_stop_at_even_steps_and_benchmark_each(run_rft, monkeypatch, tmp_path):
    plan = {"run_dir": str(tmp_path / "run"), "gpus": [0],
            "eval": {"bench": {"manifest": str(tmp_path / "m.json")}},
            "gdpo": {"conf": "c", "output_dir": str(tmp_path / "gdpo"), "eval_every": 2, "max_steps": 4}}
    (tmp_path / "m.json").write_text("{}")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    r = run_rft.Run(path)
    step = [None]
    launches, benches = [], []

    def run(cmd, logfile, env=None, cwd=None):
        launches.append((env["RESUME_FROM"], env["RESUME_TRAINING_STATE"], env["STOP_AFTER_STEP"]))
        step[0] = int(env["STOP_AFTER_STEP"])

    monkeypatch.setattr(run_rft, "run", run)
    monkeypatch.setattr(run_rft, "wait_for_free_gpus", lambda plan: None)
    monkeypatch.setattr(run_rft, "latest_gdpo_step", lambda out: step[0])
    monkeypatch.setattr(run_rft, "merge_export", lambda r, adapter, name: None)
    monkeypatch.setattr(run_rft, "benchmark_model", lambda r, tag, adapter, directory, report: benches.append(tag))
    run_rft.gdpo(r, "best")
    assert launches == [("best", "False", "2"), ("latest", "True", "4")]
    assert benches == ["gdpo_2", "gdpo_4"]
    assert json.loads((tmp_path / "gdpo" / "fixed_eval_manifest.json").read_text()) == {}
