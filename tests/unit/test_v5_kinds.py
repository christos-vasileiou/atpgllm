"""Dataset v5 circuit kinds in training order, selection, evaluation and the schedule."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atpgllm.training.kinds import (
    V5_PROMPT_WINDOW, even_spacing_order, interleave_by_kind, parse_kind_shares, prompt_window_notice,
    shard_kinds, split_counts,
)
from atpgllm.training.stream_buffer import ShardPermutation

REPO = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dataset(tmp_path, kinds=(("data", "combinational", [3, 1, 40, 2]), ("seq", "sequential", [60, 5, 9])),
            version="v5"):
    """One shard per circuit, combinational shards first, as the v5 assembly writes them."""
    root = tmp_path / "ds"
    (root / "train").mkdir(parents=True)
    shards, counts = {}, {}
    for prefix, kind, sizes in kinds:
        for i, n in enumerate(sizes):
            name = f"train/{prefix}-{i:05d}.parquet"
            rows = [{"circuit_kind": kind, "netlist": f"{prefix}{i}", "fault": f"sa0 n{j}", "difficulty": "easy",
                     "netlist_id": f"{prefix}{i}"} for j in range(n)]
            pq.write_table(pa.Table.from_pylist(rows), root / name)
            shards[name] = {"rows": n}
            counts[kind] = counts.get(kind, 0) + n
    (root / "split_manifest.json").write_text(json.dumps(
        {"dataset_version": version, "shards": shards, "kind_row_counts": {"train": counts}}))
    return root


def window_kinds(order, row_kinds, width):
    return [set(row_kinds[order[i:i + width]].tolist()) for i in range(0, len(order) - width + 1, width)]


def test_even_spacing_keeps_each_kind_in_order_and_mixes_every_window():
    kinds = np.array([0] * 30 + [1] * 70)
    order = even_spacing_order(kinds)
    assert sorted(order.tolist()) == list(range(100))
    assert [i for i in order if kinds[i] == 0] == list(range(30))
    # Any 7 consecutive positions hold both kinds (2 * 100 / 30 < 7).
    assert all(len(set(kinds[order[i:i + 7]])) == 2 for i in range(94))


def test_interleave_by_kind_is_a_no_op_for_one_kind():
    rows = [{"circuit_kind": "combinational", "i": i} for i in range(5)]
    assert interleave_by_kind(rows) is rows
    assert interleave_by_kind([{"i": 1}, {"i": 2}])[0]["i"] == 1  # v4 rows: no kind column


def test_shard_permutation_spaces_kinds_on_v5(tmp_path):
    root = dataset(tmp_path)
    perm = ShardPermutation(root, "train", seed=3)
    assert perm.kind_names == ("combinational", "sequential")
    order = perm.order(0)
    assert sorted(order.tolist()) == list(range(perm.total))
    # 46 combinational of 120 rows: every window of 6 holds both kinds.
    assert all(len(s) == 2 for s in window_kinds(order, perm.row_kinds, 6))
    # Each kind keeps the plain permutation's relative order.
    plain = ShardPermutation(root, "train", seed=3, stratify_kinds=False).order(0)
    for code in (0, 1):
        assert [i for i in order if perm.row_kinds[i] == code] == [i for i in plain if perm.row_kinds[i] == code]


def test_shard_permutation_reads_kinds_from_the_manifest(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    manifest = json.loads((root / "split_manifest.json").read_text())
    for name, entry in manifest["shards"].items():
        entry["kind"] = "sequential" if "/seq-" in name else "combinational"
    (root / "split_manifest.json").write_text(json.dumps(manifest))
    import atpgllm.training.kinds as kinds
    monkeypatch.setattr(kinds, "shard_kinds", lambda paths: pytest.fail("footers must not be read"))
    with_manifest = ShardPermutation(root, "train", seed=3)
    assert with_manifest.kind_names == ("combinational", "sequential")
    monkeypatch.undo()
    assert np.array_equal(with_manifest.order(0), ShardPermutation(dataset(tmp_path / "b"), "train", seed=3).order(0))


def test_shard_permutation_unchanged_for_one_kind(tmp_path):
    root = dataset(tmp_path, kinds=(("data", "combinational", [4, 9, 2]),), version="v4")
    a = ShardPermutation(root, "train", seed=1).order(0)
    b = ShardPermutation(root, "train", seed=1, stratify_kinds=False).order(0)
    assert np.array_equal(a, b)


def test_shard_kinds_reads_footer_statistics(tmp_path):
    root = dataset(tmp_path)
    paths = sorted((root / "train").glob("*.parquet"))
    assert shard_kinds(paths) == ["combinational"] * 4 + ["sequential"] * 3


def test_prompt_window_notice(tmp_path):
    v5 = dataset(tmp_path)
    window, notice = prompt_window_notice(v5, 8192)
    assert window == V5_PROMPT_WINDOW and "Changing the prompt window from 8192 to 14336" in notice
    assert "6.8% of sequential" in notice and "2.9% of combinational" in notice
    assert prompt_window_notice(v5, 16384)[0] == 16384
    v4 = dataset(tmp_path / "v4", kinds=(("data", "combinational", [2]),), version="v4")
    assert prompt_window_notice(v4, 8192) == (8192, None)


def test_kind_shares_split_each_level_exactly():
    shares = parse_kind_shares("combinational=1,sequential=1")
    assert shares == {"combinational": 0.5, "sequential": 0.5}
    counts = split_counts({"easy": 1001, "hard": 60}, shares)
    assert counts[("combinational", "easy")] + counts[("sequential", "easy")] == 1001
    assert counts[("combinational", "hard")] == counts[("sequential", "hard")] == 30
    with pytest.raises(ValueError):
        parse_kind_shares("analog=1")


def test_selection_takes_problems_from_each_kind(tmp_path):
    rationalize = load("rationalize", "scripts/star/rationalize.py")
    root = dataset(tmp_path)
    rows = rationalize.read_selected(root, {"easy": 6}, seed=1, kind_shares={"combinational": 0.5, "sequential": 0.5})
    kinds = [r["circuit_kind"] for r in rows]
    assert kinds.count("combinational") == 9 and kinds.count("sequential") == 9  # 3 x surplus
    assert rationalize.stratum(rows[0], {"combinational": 0.5}) == "combinational/easy"


def test_evaluation_summary_by_kind(tmp_path):
    evaluate = load("evaluate_rft", "scripts/rft/evaluate.py")
    slots = []
    for kind, level, hit in [("combinational", "easy", 1), ("combinational", "hard", 0),
                             ("sequential", "easy", 1), ("sequential", "easy", 1), ("sequential", "hard", 1)]:
        slots.append({"example_id": f"{kind}{level}{len(slots)}", "level": level, "kind": kind,
                      "reward_components": {"detection": float(hit)}})
    (tmp_path / "slots.jsonl").write_text("".join(json.dumps(s) + "\n" for s in slots))
    metrics = evaluate.summarize(tmp_path)
    assert metrics["by_kind"]["combinational"]["level_mean_detection"] == 0.5
    assert metrics["by_kind"]["sequential"]["level_mean_detection"] == 1.0
    assert metrics["kind_level_mean_detection"] == 0.75


def test_best_gdpo_takes_the_best_existing_checkpoint(tmp_path, monkeypatch):
    run_rft = load("run_rft", "scripts/rft/run_rft.py")
    out = tmp_path / "gdpo"
    for step in (2, 4):  # checkpoint-6 was pruned
        (out / f"checkpoint-{step}" / "policy").mkdir(parents=True)
    plan = {"run_dir": str(tmp_path / "run"), "gdpo": {"output_dir": str(out)}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    r = run_rft.Run(tmp_path / "plan.json")
    r.state["done"].update({
        "gdpo_2/bench": {"result": {"level_mean": 0.4, "kind_level_mean": 0.30}},
        "gdpo_4/bench": {"result": {"level_mean": 0.3, "kind_level_mean": 0.35}},
        "gdpo_6/bench": {"result": {"level_mean": 0.9, "kind_level_mean": 0.90}},
    })
    best = run_rft.best_gdpo(r)
    assert best["step"] == 4 and best["policy"].endswith("checkpoint-4/policy")
    assert json.loads((tmp_path / "run" / "gdpo_best.json").read_text())["step"] == 4


def test_schedule_starts_each_phase_from_the_one_before(tmp_path):
    load("run_rft", "scripts/rft/run_rft.py")
    run_schedule = load("run_schedule", "scripts/rft/run_schedule.py")
    phase_1 = tmp_path / "phase_1"
    phase_1.mkdir()
    (phase_1 / "gdpo_best.json").write_text(json.dumps({"policy": "/x/checkpoint-8/policy"}))
    state = {"warmup": "/w/checkpoint-100", "phases": {"0": {"run_dir": str(phase_1)}}}
    assert run_schedule.phase_start(state, 0) == "/w/checkpoint-100"
    assert run_schedule.phase_start(state, 1) == "/x/checkpoint-8/policy"


SCHEDULES = ["scripts/rft/configs/schedule_v5_juno_h100.json", "scripts/rft/configs/schedule_v5_juno_h200.json"]


@pytest.mark.parametrize("path", SCHEDULES)
def test_v5_configs_agree(path):
    schedule = json.loads((REPO / path).read_text())
    for phase in schedule["phases"]:
        plan = json.loads((REPO / phase["plan"]).read_text())
        assert plan["sample"]["max_prompt_length"] == V5_PROMPT_WINDOW
        assert plan["kind_shares"] and plan["eval"]["val_kind_shares"]
        assert plan["gdpo"]["hold"] is False
        gdpo = (REPO / plan["gdpo"]["conf"]).read_text()
        assert "MAX_PROMPT_LENGTH=14336" in gdpo and "FIXED_EVAL_SELECTION=stratified_family_kind_v1" in gdpo
        assert f"MAX_STEPS=${{MAX_STEPS:-{plan['gdpo']['max_steps']}}}" in gdpo
        sft = (REPO / plan["sft"]["conf"]).read_text()
        assert "MAX_PROMPT_LENGTH=14336" in sft and f"MAX_MODEL_LEN={plan['sft']['max_model_len']}" in sft
        assert "SFT_CIRCUIT_VALIDATION=True" in sft
    warm = (REPO / schedule["warmup"]["conf"]).read_text()
    assert "SFT_CIRCUIT_VALIDATION=True" in warm and "MAX_PROMPT_LENGTH=14336" in warm


def _conf_int(text, key):
    import re

    return int(re.search(rf"^{key}=(?:\$\{{{key}:-)?(\d+)", text, re.M).group(1))


def test_the_h200_schedule_keeps_the_step_sizes_of_the_h100_schedule():
    """Two GPUs in place of four: a step must hold the same conversations
    (SFT) and the same completions (GDPO: accumulation times trainer ranks,
    where the vLLM server takes TP x DP of the node's GPUs)."""
    sizes = []
    for path in SCHEDULES:
        schedule = json.loads((REPO / path).read_text())
        warm = (REPO / schedule["warmup"]["conf"]).read_text()
        assert _conf_int(warm, "GPUS_PER_NODE") == len(schedule["gpus"])
        row = [_conf_int(warm, "GPUS_PER_NODE") * _conf_int(warm, "GRADIENT_ACCUMULATION_STEPS")]
        for phase in schedule["phases"]:
            plan = json.loads((REPO / phase["plan"]).read_text())
            assert plan["gpus"] == schedule["gpus"] and plan["sft"]["gpus"] == len(plan["gpus"])
            sft = (REPO / plan["sft"]["conf"]).read_text()
            assert _conf_int(sft, "GPUS_PER_NODE") == plan["sft"]["gpus"]
            assert _conf_int(sft, "GRADIENT_ACCUMULATION_STEPS") == plan["sft"]["gradient_accumulation_steps"]
            row.append(plan["sft"]["gpus"] * plan["sft"]["gradient_accumulation_steps"])
            gdpo = (REPO / plan["gdpo"]["conf"]).read_text()
            trainers = _conf_int(gdpo, "GPUS_PER_NODE") - _conf_int(gdpo, "TENSOR_PARALLEL_SIZE") * _conf_int(gdpo, "DATA_PARALLEL_SIZE")
            assert trainers >= 1 and _conf_int(gdpo, "GPUS_PER_NODE") == len(plan["gpus"])
            assert _conf_int(gdpo, "STEPS_PER_GENERATION") == _conf_int(gdpo, "GRADIENT_ACCUMULATION_STEPS")
            row.append(trainers * _conf_int(gdpo, "GRADIENT_ACCUMULATION_STEPS"))
        sizes.append(row)
    assert sizes[0] == sizes[1] == [1024, 1024, 1152, 1024, 1152]


@pytest.mark.parametrize("path", SCHEDULES)
def test_gdpo_phases_read_different_prompts(path):
    schedule = json.loads((REPO / path).read_text())
    seeds = [json.loads((REPO / p["plan"]).read_text())["gdpo"].get("env", {}).get("GRPO_BUFFER_SEED", 42)
             for p in schedule["phases"]]
    assert len(set(seeds)) == len(seeds)


def test_gdpo_stage_passes_the_phase_env(tmp_path, monkeypatch):
    run_rft = load("run_rft", "scripts/rft/run_rft.py")
    out = tmp_path / "gdpo"
    plan = {"run_dir": str(tmp_path / "run"), "gpus": [], "eval": {"fixed_manifest": str(tmp_path / "m.json")},
            "gdpo": {"conf": "c.conf", "output_dir": str(out), "env": {"GRPO_BUFFER_SEED": 4242}}}
    (tmp_path / "m.json").write_text("{}")
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    r = run_rft.Run(tmp_path / "plan.json")
    seen = []
    monkeypatch.setattr(run_rft, "wait_for_free_gpus", lambda plan: None)
    monkeypatch.setattr(run_rft, "latest_gdpo_step", lambda out: None)
    monkeypatch.setattr(run_rft, "run", lambda cmd, log, env=None, cwd=None: seen.append(env))
    run_rft.gdpo(r, "/adapter")
    assert seen[0]["GRPO_BUFFER_SEED"] == "4242" and seen[0]["RESUME_FROM"] == "/adapter"


def test_phase_handover_excludes_problems_and_continues_the_stream(tmp_path):
    run_rft = load("run_rft", "scripts/rft/run_rft.py")
    run_schedule = load("run_schedule", "scripts/rft/run_schedule.py")
    prev = tmp_path / "phase_1"
    for n in (1, 2):
        (prev / f"round_{n}").mkdir(parents=True)
        pq.write_table(pa.table({"netlist_id": [f"n{n}"], "fault": ["sa0 a"]}), prev / f"round_{n}" / "attempted.parquet")
    (prev / "plan.json").write_text(json.dumps({"run_dir": str(prev)}))
    (prev / "state.json").write_text(json.dumps({"results": {
        "0": {"adapter": "/warm/checkpoint-100"},
        "1": {"adapter": str(prev / "round_1/sft/checkpoint-10")},
        "2": {"adapter": str(prev / "round_2/sft/checkpoint-10")}}}))
    out = run_schedule.handover({"run_dir": str(prev)})
    assert out["skip_from"].endswith("round_2/sft/checkpoint-10")
    assert sorted(Path(p).parent.name for p in out["exclude_problems"]) == ["round_1", "round_2"]

    # The first SFT round of the next phase takes the stream offset from skip_from.
    ck = prev / "round_2/sft/checkpoint-10"
    ck.mkdir(parents=True)
    (ck / "training_state_summary.json").write_text(json.dumps({"cumulative_skip_buffer_size": 123456}))
    plan = {"run_dir": str(tmp_path / "phase_2"), "gpus": [], "sft": {"conf": "c", "gpus": 4, "epochs": 1,
            "extra_fraction": 0.5, "learning_rate": 1e-5, "skip_from": str(ck)}}
    (tmp_path / "p2.json").write_text(json.dumps(plan))
    r = run_rft.Run(tmp_path / "p2.json")
    seen = []
    run_rft.wait_for_free_gpus = lambda plan: None
    run_rft.run = lambda cmd, log, env=None, cwd=None: seen.append(env)
    run_rft.complete_checkpoint = lambda d, s: bool(seen)
    run_rft.train(r, 1, str(tmp_path / "gdpo/checkpoint-8/policy"), 100)
    assert seen[0]["SKIP_BUFFER_SIZE"] == "123456"


def test_read_rows_picks_rows_across_row_groups(tmp_path):
    from atpgllm.training.stream_buffer import read_rows
    path = tmp_path / "s.parquet"
    pq.write_table(pa.table({"i": list(range(25)), "s": [f"x{i}" * 3 for i in range(25)]}), path, row_group_size=4)
    rows = read_rows(path, [24, 0, 5, 5, 13])
    assert [r["i"] for r in rows] == [24, 0, 5, 5, 13] and rows[2]["s"] == "x5x5x5"


def test_gdpo_from_last_starts_from_the_last_trained_round(tmp_path, monkeypatch):
    run_rft = load("run_rft", "scripts/rft/run_rft.py")
    plan = {"run_dir": str(tmp_path / "run"), "start_adapter": "/warm/checkpoint-100", "gdpo": {"output_dir": "x"}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    r = run_rft.Run(tmp_path / "plan.json")
    assert run_rft.last_trained(r) == (None, "/warm/checkpoint-100")
    # Round 2 trained but its job ended before the evaluation; round 3 only sampled.
    r.state["done"].update({"round_1/sft": {"result": "/r1/checkpoint-10"}, "round_1/evaluate": {"result": {}},
                            "round_2/sft": {"result": "/r2/checkpoint-10"}, "round_3/generate": {"result": "g"}})
    assert run_rft.last_trained(r) == (2, "/r2/checkpoint-10")
    seen = []
    monkeypatch.setattr(run_rft, "gdpo", lambda r, adapter: seen.append(adapter) or "out")
    monkeypatch.setattr(run_rft, "best_gdpo", lambda r: None)
    run_rft.gdpo_from_last(r)
    assert seen == ["/r2/checkpoint-10"] and r.state["gdpo_start"]["round"] == 2 and r.state["rounds_stopped"]
    # A later launch resumes the same GDPO even when more rounds appear in the state.
    r.state["done"].pop("gdpo")
    r.state["done"]["round_3/sft"] = {"result": "/r3/checkpoint-10"}
    run_rft.gdpo_from_last(r)
    assert seen == ["/r2/checkpoint-10", "/r2/checkpoint-10"]


@pytest.mark.parametrize("stage, flag", [("rft", "--rounds_only"), ("gdpo", "--gdpo_from_last")])
def test_schedule_only_runs_one_stage_of_the_first_phase(tmp_path, monkeypatch, stage, flag):
    load("run_rft", "scripts/rft/run_rft.py")
    run_schedule = load("run_schedule", "scripts/rft/run_schedule.py")
    plan = tmp_path / "phase.json"
    plan.write_text(json.dumps({"run_dir": "unused", "sft": {}}))
    schedule = {"run_dir": str(tmp_path / "run"), "warmup": {"conf": "w.conf", "output_dir": "w", "max_steps": 100},
                "phases": [{"name": "phase_1", "plan": str(plan)}, {"name": "phase_2", "plan": str(plan)}]}
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    seen = []
    monkeypatch.setattr(run_schedule, "run", lambda cmd, log, env=None: seen.append([str(c) for c in cmd]))
    monkeypatch.setattr(run_schedule, "warmup", lambda schedule, run_dir: "/warm/checkpoint-100")
    state = {} if stage == "rft" else {"warmup": "/warm/checkpoint-100"}
    run_schedule.run_only(schedule, run_dir, state, stage)
    if stage == "rft":   # the warm-up is finished first when an earlier job ran out of time
        assert state["warmup"] == "/warm/checkpoint-100"
    assert len(seen) == 1 and seen[0][-1] == flag and seen[0][-2].endswith("phase_1")
    written = json.loads((run_dir / "phase_1" / "plan.source.json").read_text())
    assert written["start_adapter"] == "/warm/checkpoint-100" and "phase_2" not in state.get("phases", {})


def test_schedule_only_warmup_stops_after_the_warmup(tmp_path, monkeypatch):
    load("run_rft", "scripts/rft/run_rft.py")
    run_schedule = load("run_schedule", "scripts/rft/run_schedule.py")
    seen = []
    monkeypatch.setattr(run_schedule, "run", lambda *a, **k: seen.append(a))
    monkeypatch.setattr(run_schedule, "warmup", lambda schedule, run_dir: "/warm/checkpoint-100")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = {}
    run_schedule.run_only({"phases": []}, run_dir, state, "warmup")
    assert state == {"warmup": "/warm/checkpoint-100"} and not seen
