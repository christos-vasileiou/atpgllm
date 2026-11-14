"""Hint arms on v5: what does Granite add to the graph model's hint?

The step 40 check of 11-12 scored only the first tool call, where the hint
arm sat level with the graph decoder alone. This script lets each arm use
the tool: a rollout is the model's turns with the fault simulator answering
its calls, up to ``--max-tool-calls``, then its final answer. All passes
write into ``--out`` and skip finished rows when run again.

``rows`` (CPU)
    Picks ``--per-circuit`` problems of every validation circuit of
    ``--dataset`` (both kinds of circuit) and writes ``rows.jsonl``.
``rollout`` (GPU, simulator through ``FAULT_SIM_BACKEND``)
    Greedy rollouts of one arm (``--name``). Per row: every call's vector
    and whether the simulator saw the fault detected, and the final answer's
    vector, simulated the same way (x stays unknown).
``decoder`` (simulator)
    The graph model alone on the same rows. ``attempt 1`` is its plain hint
    as a vector. Attempts 2 and up change one decision of that hint each,
    least sure first (flip a value close to a coin flip, or set an input it
    left out with a care probability under 0.5): a retry policy that needs
    no language model, so a rollout's gain from retries can be compared with
    it. ``answer_key`` is the vote of the dataset's own patterns.
``report``
    Detection on the first call, within k calls and by the final answer, by
    kind of circuit, fault location and difficulty, with paired differences
    between arms (bootstrap over circuits).

    PYTHONPATH=$PWD python scripts/eval/hint_arms_eval.py rows --dataset <v5> --out runs/hint_arms_eval
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import retest_r6 as r6  # noqa: E402  (also puts scripts/train on the path)

_QN = re.compile(r"QN=(\S+)")
_ANSWER = re.compile(r'INPUT_VECTOR:\s*"([^"]*)"')
LOCATIONS = ("internal", "pi_stem", "state_in", "po_stem")
KINDS = ("combinational", "sequential")


# ---------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------

def kind(record: Dict[str, Any]) -> str:
    return record.get("circuit_kind") or "combinational"


def location(record: Dict[str, Any]) -> str:
    """Where the fault sits. A fault on a flop's QN net is on a state input."""
    net = str(record["fault"]).strip().split(None, 1)[1]
    if net in _QN.findall(record.get("scan_chain") or ""):
        return "state_in"
    return r6.location(record)


def load_rows(out: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in (out / "rows.jsonl").open()]


def shard_file(out: Path, stem: str, args) -> Path:
    """One file per shard, so shards can run as separate jobs."""
    return out / (f"{stem}.jsonl" if args.shards == 1 else f"{stem}.s{args.shard}.jsonl")


def read_method(out: Path, stem: str) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    for path in sorted(list(out.glob(f"{stem}.jsonl")) + list(out.glob(f"{stem}.s*.jsonl"))):
        for row in map(json.loads, path.open()):
            if not row.get("rejected"):
                rows[row["row"]] = row
    return rows


def run_rows(args) -> None:
    import pyarrow.parquet as pq

    from atpgllm.graph.circuit_sampling import split_files

    rng = random.Random(args.seed)
    by_circuit: Dict[str, Dict[str, List[dict]]] = collections.OrderedDict()
    for path in split_files(str(args.dataset), "validation", None):
        file = pq.ParquetFile(path)
        for group in range(file.num_row_groups):
            for row in file.read_row_group(group).to_pylist():
                faults = by_circuit.setdefault(str(row["circuit_id"]), collections.OrderedDict())
                faults.setdefault(str(row["fault"]).strip(), []).append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    count = collections.Counter()
    with (args.out / "rows.jsonl").open("w") as fh:
        for circuit, faults in by_circuit.items():
            names = sorted(faults)
            for fault in rng.sample(names, min(args.per_circuit, len(names))):
                row = rng.choice(faults[fault])
                fh.write(json.dumps(row, default=str) + "\n")
                count[kind(row)] += 1
    print(f"rows: {sum(count.values())} from {len(by_circuit)} circuits {dict(count)}")


# ---------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------

def answer_vector(text: str, inputs: List[str]) -> Optional[Dict[str, Any]]:
    """The final answer's input vector over all inputs. The answer states
    care bits only (``INPUT_VECTOR: "a: 0, b: 1"``); the rest are x."""
    match = _ANSWER.search(text)
    if match is None:
        return None
    vector: Dict[str, Any] = {name: "x" for name in inputs}
    for item in match.group(1).split(","):
        if ":" not in item:
            continue
        name, value = (part.strip() for part in item.rsplit(":", 1))
        if name in vector and value in ("0", "1"):
            vector[name] = int(value)
    return vector


def call_vector(turn: str) -> Optional[Dict[str, Any]]:
    from atpgllm.training.revert_template import parse_tool_call

    call = parse_tool_call(turn)
    if not call or not isinstance(call.get("arguments"), dict):
        return None
    return r6.cs._vector(call["arguments"].get("input_vector"))


def rollout_row(trajectory, record) -> Dict[str, Any]:
    """What one trajectory did: its calls, each with the detection the tool
    reported, and its final answer."""
    from atpgllm.training.tool_feedback import read_snapshot

    outputs = r6.outputs_of(record)
    calls = []
    for turn, result in zip(trajectory.turns, trajectory.tool_results):
        try:
            detected = bool(read_snapshot(result, record["fault"], outputs).detected)
        except (ValueError, KeyError, TypeError):
            detected = False
        calls.append({"vector": call_vector(turn), "detected": detected,
                      "malformed": result.startswith("Tool execution failed")})
    last = trajectory.turns[-1] if trajectory.turns else ""
    answered = len(trajectory.turns) > len(trajectory.tool_results)
    final = answer_vector(last, r6.inputs_of(record)) if answered else None
    return {"calls": calls, "final_vector": final,
            "final_detected": bool(final) and r6._simulate(record, final)[0],
            "truncated": trajectory.truncated, "call_limit_hit": trajectory.call_limit_hit,
            "reused_calls": trajectory.reused_calls, "tokens": len(trajectory.completion_ids),
            "turns": [turn[-1200:] for turn in trajectory.turns]}


def run_rollout(args) -> None:
    import torch
    from torch_geometric.data import Batch

    import multimodal_training as mt
    from atpgllm.multimodal.tool_rollout import rollout_group

    records = load_rows(args.out)
    model, tokenizer, gate_funcs, vocab, session = r6.load_model(args.checkpoint, args.adapter)
    hints, _ = r6.hint_tables(args, session)
    out_path = shard_file(args.out, f"rollout_{args.name}", args)
    done = r6._done(out_path, lambda r: r["row"])
    stop = [t for t in (tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id)
            if isinstance(t, int) and t >= 0]
    device = model.llm.get_input_embeddings().weight.device
    todo = [i for i in range(len(records)) if i % args.shards == args.shard and i not in done]
    if args.limit:
        todo = todo[:args.limit]
    started = time.time()
    with out_path.open("a") as fh:
        for count, index in enumerate(todo):
            record = records[index]
            base = {"row": index, "name": args.name, "circuit_id": record["circuit_id"],
                    "fault": record["fault"], "kind": kind(record), "location": location(record),
                    "difficulty": record.get("difficulty")}
            shown = r6.example(record, tokenizer, gate_funcs, vocab, session, hints)
            if shown is None:
                fh.write(json.dumps({**base, "rejected": True}) + "\n")
                continue
            prompt_ids = list(shown["input_ids"][:shown["turns"][0][0]])
            with torch.no_grad():
                prefix = model.graph_soft_prompt(Batch.from_data_list([shown["graph"]]).to(device))

            def generate_turn(contexts, max_new_tokens):
                return model.generate_turns(
                    prefix, contexts, pad_token_id=tokenizer.pad_token_id,
                    graph_at=shown["graph_at"], max_new_tokens=max_new_tokens,
                    do_sample=False, eos_token_id=stop)

            trajectory = rollout_group(
                generate_turn, tokenizer, prompt_ids,
                mt._tool_runner(copy.deepcopy(record), session.get("tool_view", "po_fault")),
                num_generations=1, max_tool_calls=args.max_tool_calls,
                max_completion_tokens=args.max_completion_tokens)[0]
            fh.write(json.dumps({**base, **rollout_row(trajectory, record)}) + "\n")
            fh.flush()
            print(f"rollout {args.name} {count + 1}/{len(todo)} {time.time() - started:.0f}s", flush=True)


# ---------------------------------------------------------------------
# Graph model alone
# ---------------------------------------------------------------------

def decoder_attempts(rich: Dict[str, Any], inputs: List[str], attempts: int) -> List[Dict[str, Any]]:
    """The plain hint as a vector, then one changed decision per attempt.

    Candidates, least sure first: flip the value of a suggested input (score
    1 at a coin flip, 0 at a certain value), or set an input the hint left
    out (score twice its care probability, under 1)."""
    probabilities = rich.get("inputs") or {}
    base: Dict[str, Any] = {name: "x" for name in inputs}
    changes = []
    for name, (care, one) in probabilities.items():
        if name not in base:
            continue
        value = int(one >= 0.5)
        if care >= 0.5:
            base[name] = value
            changes.append((1.0 - 2.0 * abs(one - 0.5), name, 1 - value))
        else:
            changes.append((2.0 * care, name, value))
    changes.sort(key=lambda item: -item[0])
    vectors = [base]
    for _, name, value in changes[:max(0, attempts - 1)]:
        vectors.append({**base, name: value})
    return vectors


def run_decoder(args) -> None:
    from atpgllm.multimodal.data import netlist_key

    records = load_rows(args.out)
    hints: Dict[tuple, dict] = {}
    for path in sorted(Path(args.hints).glob("validation_*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            hints[(row["netlist_key"], row["fault"])] = row
    problems: Dict[tuple, dict] = {}
    wanted = {(netlist_key(r), str(r["fault"]).strip()) for r in records}
    for directory in args.labels:
        with (Path(directory) / "problems_validation.jsonl").open() as fh:
            for line in fh:
                head = json.loads(line)
                key = (head["netlist_key"], head["fault"])
                if key in wanted:
                    problems[key] = head
    out_path = shard_file(args.out, "decoder", args)
    done = r6._done(out_path, lambda r: r["row"])
    started = time.time()
    with out_path.open("a") as fh:
        for index, record in enumerate(records):
            if index in done or index % args.shards != args.shard:
                continue
            key = (netlist_key(record), str(record["fault"]).strip())
            base = {"row": index, "circuit_id": record["circuit_id"], "fault": record["fault"],
                    "kind": kind(record), "location": location(record),
                    "difficulty": record.get("difficulty")}
            hint, problem = hints.get(key), problems.get(key)
            if hint is None or problem is None:
                fh.write(json.dumps({**base, "rejected": True}) + "\n")
                continue
            inputs = r6.inputs_of(record)
            vectors = decoder_attempts(hint["rich"], inputs, args.attempts)
            detected = [r6._simulate(record, vector)[0] for vector in vectors]
            vote = {n: (int(problem["one_freq"].get(n, 0) >= 0.5)
                        if problem["care_freq"].get(n, 0) >= 0.5 else "x") for n in inputs}
            fh.write(json.dumps({**base, "attempts": detected, "candidates": len(vectors),
                                 "answer_key": r6._simulate(record, vote)[0]}) + "\n")
            fh.flush()
            if index % 50 == 0:
                print(f"decoder row {index}/{len(records)} {time.time() - started:.0f}s", flush=True)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def outcomes(row: Dict[str, Any], calls: int) -> Dict[str, bool]:
    """One method's row as ``{measure: detected}``."""
    if "attempts" in row:      # the graph model alone
        flags = row["attempts"]
        out = {f"within_{k}": any(flags[:k]) for k in range(1, calls + 1)}
        out["first"] = bool(flags and flags[0])
        return out
    flags = [c["detected"] for c in row["calls"]]
    out = {f"within_{k}": any(flags[:k]) for k in range(1, calls + 1)}
    out["first"] = bool(flags and flags[0])
    out["final"] = bool(row["final_detected"])
    return out


def _rate(values: List[bool]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def bootstrap(pairs: List[tuple], resamples: int = 2000, seed: int = 0) -> List[float]:
    """95% interval of the mean paired difference, resampling circuits."""
    by_circuit: Dict[str, List[float]] = collections.defaultdict(list)
    for circuit, difference in pairs:
        by_circuit[circuit].append(difference)
    groups = list(by_circuit.values())
    if not groups:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [x for _ in groups for x in rng.choice(groups)]
        means.append(sum(sample) / len(sample))
    means.sort()
    return [means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]]


def build_report(out: Path, calls: int) -> Dict[str, Any]:
    methods: Dict[str, Dict[int, dict]] = {}
    names = {re.sub(r"\.s\d+$", "", path.stem)[len("rollout_"):] for path in out.glob("rollout_*.jsonl")}
    for name in sorted(names):
        methods[name] = read_method(out, f"rollout_{name}")
    rows = read_method(out, "decoder")
    if rows:
        methods["graph_decoder"] = rows
        methods["answer_key"] = {i: {**r, "attempts": [r["answer_key"]]} for i, r in rows.items()}
    shared = set.intersection(*(set(rows) for rows in methods.values())) if methods else set()
    measures = ["first"] + [f"within_{k}" for k in range(2, calls + 1)] + ["final"]

    def table(rows_of: Dict[int, dict], keep) -> Dict[str, Any]:
        rows = [r for i, r in sorted(rows_of.items()) if i in shared and keep(r)]
        scored = [outcomes(r, calls) for r in rows]
        line: Dict[str, Any] = {"n": len(rows)}
        for measure in measures:
            line[measure] = _rate([s[measure] for s in scored if measure in s])
        if rows and "calls" in rows[0]:
            line["mean_calls"] = sum(len(r["calls"]) for r in rows) / len(rows)
            line["answered"] = _rate([r["final_vector"] is not None for r in rows])
            line["truncated"] = _rate([bool(r["truncated"]) for r in rows])
        return line

    groups = {"all": lambda r: True}
    groups.update({f"kind:{k}": (lambda r, k=k: r["kind"] == k) for k in KINDS})
    groups.update({f"loc:{l}": (lambda r, l=l: r["location"] == l) for l in LOCATIONS})
    groups.update({f"difficulty:{d}": (lambda r, d=d: r.get("difficulty") == d) for d in r6.DIFFICULTIES})
    report: Dict[str, Any] = {"rows": len(shared), "calls": calls, "measures": measures,
                              "tables": {g: {m: table(rows, keep) for m, rows in methods.items()}
                                         for g, keep in groups.items()}}
    pairs = {}
    names = sorted(methods)
    for a in names:
        for b in names:
            if a >= b or "answer_key" in (a, b):
                continue
            for measure in measures:
                diffs = []
                for i in shared:
                    oa, ob = outcomes(methods[a][i], calls), outcomes(methods[b][i], calls)
                    if measure in oa and measure in ob:
                        diffs.append((str(methods[a][i]["circuit_id"]), float(oa[measure]) - float(ob[measure])))
                if diffs:
                    pairs[f"{a} minus {b}, {measure}"] = {
                        "difference": sum(d for _, d in diffs) / len(diffs),
                        "circuit_bootstrap_95": bootstrap(diffs), "rows": len(diffs)}
    report["paired"] = pairs
    return report


def _pct(x) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.1f}%"


def markdown(report: Dict[str, Any]) -> str:
    measures = report["measures"]
    lines = ["# Hint arms on v5: first call, retries and final answer", "",
             f"Rows scored by every method: {report['rows']}. A rollout may make up to "
             f"{report['calls']} tool calls. x stays unknown in every simulation.", ""]
    for group, methods in report["tables"].items():
        if not any(line["n"] for line in methods.values()):
            continue
        lines += [f"## {group}", "", "| method | n | " + " | ".join(measures) + " | calls | answered |",
                  "|---|---|" + "---|" * (len(measures) + 2)]
        for name, line in methods.items():
            lines.append(f"| {name} | {line['n']} | " + " | ".join(_pct(line.get(m)) for m in measures)
                         + f" | {line.get('mean_calls', float('nan')):.2f} | {_pct(line.get('answered'))} |")
        lines.append("")
    lines += ["## Paired differences (all rows)", "",
              "| comparison | difference | 95% interval over circuits | rows |", "|---|---|---|---|"]
    for name, pair in report["paired"].items():
        low, high = pair["circuit_bootstrap_95"]
        lines.append(f"| {name} | {100 * pair['difference']:+.1f} | {100 * low:+.1f} to {100 * high:+.1f} | {pair['rows']} |")
    return "\n".join(lines) + "\n"


def run_report(args) -> None:
    report = build_report(args.out, args.max_tool_calls)
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    (args.out / "report.md").write_text(markdown(report))
    print(markdown(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pass_", metavar="pass", choices=("rows", "rollout", "decoder", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--per-circuit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", help="The arm's label in the report (rollout).")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--hints", type=Path,
                        help="rollout: replaces the path the arm trained with. decoder: the hint files.")
    parser.add_argument("--labels", type=Path, nargs="+", default=[])
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    {"rows": run_rows, "rollout": run_rollout, "decoder": run_decoder, "report": run_report}[args.pass_](args)


if __name__ == "__main__":
    main()
