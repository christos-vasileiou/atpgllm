#!/usr/bin/env python
"""A training schedule: SFT warm-up, then RFT phases, each ending in GDPO.

    python scripts/rft/run_schedule.py --schedule scripts/rft/configs/schedule_v5_a100.json

The v5 schedule is SFT warm-up -> RFT -> GDPO -> RFT -> GDPO:

1. ``warmup``: SFT from the base model on the dataset (``run_training_code.sh``
   with the warm-up config) for ``max_steps`` steps.
2. Every entry of ``phases`` is one ``run_rft.py`` plan: RFT rounds until the
   validation score plateaus, then GDPO from the best round. The first phase
   starts from the warm-up's last checkpoint. A later phase starts from the
   best benchmarked GDPO checkpoint of the phase before it (``gdpo_best.json``,
   written by ``run_rft.py``: the highest benchmark score, earliest on ties,
   among checkpoints that still exist).

The schedule writes each phase's plan to ``<run_dir>/<phase>/plan.source.json``
with ``start_adapter`` and ``run_dir`` filled in, and runs ``run_rft.py`` on it,
so every phase has its own state, markers and logs and resumes on its own.
A rerun skips finished steps; ``state.json`` records what each one produced.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rft  # noqa: E402
from run_rft import REPO, PYTHON, complete_checkpoint, log, read_json, run, wait_for_free_gpus, write_json  # noqa: E402


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def warmup(schedule, run_dir):
    w = schedule["warmup"]
    out = resolve(w["output_dir"])
    steps = int(w["max_steps"])
    if complete_checkpoint(out, steps):
        return str(out / f"checkpoint-{steps}")
    import subprocess
    has_checkpoint = out.exists() and subprocess.run(
        [PYTHON, "atpgllm/training/checkpoints.py", str(out)], cwd=REPO, capture_output=True).returncode == 0
    env = {"OUTPUT_DIR": str(out), "MAX_STEPS": str(steps), **{k: str(v) for k, v in w.get("env", {}).items()}}
    if has_checkpoint:
        env.update({"RESUME_FROM": "latest", "RESUME_TRAINING_STATE": "True"})
    wait_for_free_gpus(schedule)
    log(f"warm-up SFT into {out}, {steps} steps" + (" (resuming)" if has_checkpoint else ""))
    run(["bash", "scripts/train/run_training_code.sh", w["conf"]], run_dir / "logs" / "warmup_sft.log", env=env)
    if not complete_checkpoint(out, steps):
        raise RuntimeError(f"warm-up SFT ended without checkpoint-{steps}")
    return str(out / f"checkpoint-{steps}")


def phase_start(state, index):
    """The adapter phase ``index`` starts from."""
    if index == 0:
        return state["warmup"]
    previous = state["phases"][str(index - 1)]
    best = read_json(Path(previous["run_dir"]) / "gdpo_best.json")
    if not best:
        raise RuntimeError(f"phase {index - 1} has no gdpo_best.json; it must finish GDPO first")
    return best["policy"]


def handover(previous):
    """What a phase takes from the one before it: the problems it sampled (never chosen
    again) and its last SFT checkpoint, whose stream offset the next SFT continues."""
    prev_dir = Path(previous["run_dir"])
    prev = run_rft.Run(prev_dir / "plan.json", prev_dir)
    rounds = sorted(int(p.name.split("_")[1]) for p in prev_dir.glob("round_*") if p.name.split("_")[1].isdigit())
    exclude = [p for p in (run_rft.attempted_problems(prev, n) for n in rounds if n > 0) if p]
    results = prev.state.get("results", {})
    trained = [int(k) for k, v in results.items() if int(k) > 0 and "/sft/checkpoint-" in str(v.get("adapter"))]
    return {"exclude_problems": exclude,
            "skip_from": results[str(max(trained))]["adapter"] if trained else None}


def run_phase(schedule, run_dir, state, index):
    phase = schedule["phases"][index]
    name = phase.get("name", f"phase_{index + 1}")
    phase_dir = run_dir / name
    phase_dir.mkdir(parents=True, exist_ok=True)
    recorded = state.setdefault("phases", {}).get(str(index))
    if recorded and recorded.get("finished"):
        return recorded
    start = recorded["start_adapter"] if recorded else phase_start(state, index)
    source = phase_dir / "plan.source.json"
    if not source.exists():
        plan = read_json(resolve(phase["plan"]))
        plan.update({"run_dir": str(phase_dir), "start_adapter": start})
        if index > 0:
            plan.update(handover(state["phases"][str(index - 1)]))
            plan["sft"]["skip_from"] = plan.pop("skip_from", None)
        write_json(source, plan)
    state["phases"][str(index)] = recorded = {"name": name, "run_dir": str(phase_dir), "start_adapter": start}
    write_json(run_dir / "state.json", state)
    log(f"{name}: RFT rounds then GDPO from {start}")
    run([PYTHON, "scripts/rft/run_rft.py", "--plan", source, "--run_dir", phase_dir],
        run_dir / "logs" / f"{name}_driver.log")
    phase_state = read_json(phase_dir / "state.json", {})
    if not phase_state.get("finished"):
        raise RuntimeError(f"{name} stopped before finishing (see {phase_dir}/state.json)")
    recorded.update({"finished": phase_state["finished"], "best_round": phase_state.get("best"),
                     "gdpo_best": phase_state.get("gdpo_best")})
    write_json(run_dir / "state.json", state)
    return recorded


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schedule", required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, run_rft.on_signal)
    signal.signal(signal.SIGHUP, run_rft.on_signal)
    signal.signal(signal.SIGINT, run_rft.on_signal)
    os.chdir(REPO)
    schedule = read_json(args.schedule)
    run_dir = resolve(schedule["run_dir"])
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    if not (run_dir / "schedule.json").exists():
        write_json(run_dir / "schedule.json", schedule)
    schedule = read_json(run_dir / "schedule.json")  # the run's copy, editable while it goes on
    state = read_json(run_dir / "state.json", {})
    state.setdefault("started", time.time())
    (run_dir / "schedule.pid").write_text(f"{os.getpid()}\n")
    if "warmup" not in state:
        state["warmup"] = warmup(schedule, run_dir)
        write_json(run_dir / "state.json", state)
    for index in range(len(schedule["phases"])):
        run_phase(schedule, run_dir, state, index)
    state["finished"] = run_rft.now()
    write_json(run_dir / "state.json", state)
    log(f"schedule finished: {json.dumps(state['phases'], default=str)[:400]}")


if __name__ == "__main__":
    main()
