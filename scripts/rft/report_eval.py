#!/usr/bin/env python
"""Tables and figures of one benchmark, plus history from stored evaluations.

    python scripts/rft/report_eval.py --run_dir runs/rft_v4_a100 --tag round_6 \
        --eval_dir runs/rft_v4_a100/round_6/eval_bench --reference <manifest>.reference.json
    python scripts/rft/report_eval.py --run_dir runs/rft_v4_a100 --history_only

CPU only. A benchmark goes to ``<run_dir>/reports/<tag>/``, and the figures in
``reports/figures/`` are redrawn over every benchmark so far. History is
measured on other problems and stays apart in ``reports_history/``: greedy
validation of each round (300 problems) and the unhinted generation of each
round (T=1, training circuits, the model of the round before).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from atpgllm.training import eval_report as report  # noqa: E402


def load_audit():
    try:
        from atpgllm.training.reasoning_audit import audit_slot
    except ImportError:
        return None
    return audit_slot


def read_index(directory):
    path = directory / "index.json"
    return json.loads(path.read_text()) if path.exists() else []


def add_to_index(directory, tag):
    index = read_index(directory)
    if tag not in index:
        index.append(tag)
        (directory / "index.json").write_text(json.dumps(index, indent=1))
    return index


def load_results(directory, tags):
    return [json.loads((directory / tag / "metrics.json").read_text()) for tag in tags
            if (directory / tag / "metrics.json").exists()]


def benchmark(args):
    reference = json.loads(Path(args.reference).read_text())
    reports = Path(args.run_dir) / "reports"
    result = report.analyze(report.load_slots(args.eval_dir), reference, report.Simulator(), tag=args.tag,
                            audit=load_audit())
    report.write_tables(result, reports / args.tag)
    history = load_results(reports, add_to_index(reports, args.tag))
    report.effort_figure(history, reference, reports / "figures" / "detection_effort")
    report.mechanism_figure(history, reports / "figures" / "mechanism")
    report.calls_figure(history, reports / "figures" / "tool_calls")
    print(json.dumps({"tag": args.tag, **report.headline(result),
                      "simulator_checks": result["simulator_checks"],
                      "simulator_disagreements": result["simulator_disagreements"]}))


def validation_reference(run_dir, simulator):
    """Reference of the 300 validation problems, computed once."""
    path = run_dir / "eval" / "val_reference.json"
    if path.exists():
        return json.loads(path.read_text())
    import pyarrow.parquet as pq
    from fault_difficulty import Analyzer
    from atpgllm.training.fixed_eval import write_json_atomic
    analyzer = Analyzer(config=Path(simulator.fs.__file__).parent / "sim_config.json", samples=65536,
                        exact_max_inputs=16, seed=1729)
    reference = {}
    for row in pq.read_table(run_dir / "eval" / "val_problems.parquet").to_pylist():
        key = "|".join([str(row.get("netlist_id") or row.get("netlist")), str(row["fault"])])
        reference[key] = report.reference_entry(row, simulator, analyzer=analyzer)
    write_json_atomic(path, reference)
    return reference


def generation_slots(directory):
    """Unhinted generation slots with what the tables read and nothing else; each holds a netlist prompt."""
    for path in sorted(Path(directory).glob("slots*.jsonl")):
        for line in open(path):
            if not line.endswith("\n"):
                continue
            try:
                slot = json.loads(line)
            except json.JSONDecodeError:
                continue
            if slot.get("hint_mode") != "none":
                continue
            yield {"problem": slot["problem"], "difficulty": slot["difficulty"], "fault": slot["fault"],
                   "circuit_id": slot.get("circuit_id"), "p": slot.get("random_detection_probability"),
                   "usage": slot.get("usage"), "status": slot.get("status"),
                   "reward_components": slot.get("reward_components"), "vector": slot.get("vector"),
                   "final_answer": slot.get("final_answer"),
                   "observations": [{"vector": o.get("vector")} for o in slot.get("observations") or []]}


def history(run_dir):
    run_dir = Path(run_dir)
    out = run_dir / "reports_history"
    state = json.loads((run_dir / "state.json").read_text())
    done = state.get("done", {})
    simulator = report.Simulator()
    val_reference = validation_reference(run_dir, simulator)

    tags = [(f"round_{n}", run_dir / f"round_{n}" / "eval_val") for n in sorted(map(int, state.get("results", {})))
            if n == 0 or f"round_{n}/evaluate" in done]
    tags.insert(1, ("gdpo_step2", run_dir / "gdpo_step2" / "eval_val"))
    for tag, directory in tags:
        name = f"val_{tag}"
        if not (out / name / "metrics.json").exists() and (directory / "metrics.json").exists():
            result = report.analyze(report.load_slots(directory), val_reference, simulator, tag=tag,
                                    source="greedy validation, 300 problems", audit=load_audit())
            report.write_tables(result, out / name)
            print(f"[history] {name}: {report.headline(result)} disagreements "
                  f"{result['simulator_disagreements']} of {result['simulator_checks']}", flush=True)
    names = [f"val_{tag}" for tag, _ in tags]
    report.mechanism_figure(load_results(out, names), out / "figures" / "mechanism_validation")
    report.calls_figure(load_results(out, names), out / "figures" / "tool_calls_validation")

    merged = {}
    names = []
    for n in range(1, 1000):
        directory = run_dir / f"round_{n}" / "gen"
        if f"round_{n}/generate" not in done:
            if not directory.exists():
                break
            continue
        name = f"gen_round_{n}"
        names.append(name)
        slots = list(generation_slots(directory))
        reference = {}
        for slot in slots:
            reference.setdefault("|".join(map(str, slot["problem"])), {
                "level": slot["difficulty"], "fault": slot["fault"], "circuit_id": slot["circuit_id"], "p": slot["p"]})
        merged.update(reference)
        if not (out / name / "metrics.json").exists():
            result = report.analyze(slots, reference, tag=f"model {n - 1}",
                                    source="unhinted generation, T=1, training circuits")
            report.write_tables(result, out / name)
            print(f"[history] {name}: {report.headline(result)}", flush=True)
    if names:
        report.effort_figure(load_results(out, names), merged, out / "figures" / "effort_generation")
        report.calls_figure(load_results(out, names), out / "figures" / "tool_calls_generation")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--tag")
    parser.add_argument("--eval_dir")
    parser.add_argument("--reference")
    parser.add_argument("--history_only", action="store_true")
    parser.add_argument("--no_history", action="store_true", help="For a run without stored RFT rounds")
    args = parser.parse_args()
    if not args.history_only:
        benchmark(args)
    if not args.no_history:
        history(args.run_dir)


if __name__ == "__main__":
    main()
