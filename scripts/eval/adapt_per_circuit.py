"""Does the graph model learn a new circuit from its own simulator search?

Test generation runs over all the faults of one design. So on a circuit the
model has never seen, the search it does for the first faults can teach it
the circuit for the remaining ones. For every held-out circuit:

1. ``test`` faults are set aside, and the model's attempts on them are
   scored (``before``).
2. On ``adapt`` other faults of the same circuit the scripted search runs
   (first pattern, then one changed decision per attempt) until a pattern
   detects. Nothing but the simulator's yes or no is used: no dataset
   pattern of this circuit.
3. A copy of the model takes a few training steps on the patterns the search
   found (care and value heads only).
4. The copy's attempts on the ``test`` faults are scored (``after``).

If ``after`` beats ``before`` on the first attempt, the search has been
compacted into the model for that circuit, and later faults need fewer
simulator calls.

    PYTHONPATH=$PWD python scripts/eval/adapt_per_circuit.py run --checkpoint runs/bridges_retest_seq/r3s/best.pt \\
        --labels ../data/freeset/retest_labels_v5 --out runs/adapt_per_circuit/a1 --shard 0 --shards 32
    PYTHONPATH=$PWD python scripts/eval/adapt_per_circuit.py report --out runs/adapt_per_circuit/a1
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import search_labels as sl  # noqa: E402  (also loads hint_arms_eval and retest_r6)

ev, r6 = sl.ev, sl.r6
LOSS_ARGS = SimpleNamespace(query_weight=1.0, text_weight=0.0)


def record_of(problem: dict, circuit: dict) -> Dict[str, Any]:
    return {"netlist": circuit["netlist"], "fault": problem["fault"],
            "input_vector": problem["example_input_vector"],
            "expected_output": problem["example_expected_output"],
            "scan_chain": circuit.get("scan_chain") or ""}


def attempts_of(model, builder, problem: dict, circuit: dict, device, attempts: int, stop: bool):
    """The simulator's verdict on each attempt for one fault (all of them, or
    up to the first that detects), with the rich probabilities behind them.
    None when the graph cannot be built."""
    from atpgllm.graph.scripts import retest_hints as rh

    graph = builder.build(problem)
    if graph is None:
        return None
    _, rich = rh.decode(model, graph, device)
    record = record_of(problem, circuit)
    if stop:
        return sl.search(record, rich, list(problem["inputs"]), attempts)
    vectors = ev.decoder_attempts(rich, list(problem["inputs"]), attempts)
    return [bool(r6._simulate(record, vector)[0]) for vector in vectors]


def adapt(model, builder, rows: List[dict], device, steps: int, batch_size: int, lr: float, rng):
    """A copy of ``model`` after ``steps`` updates on the found patterns.
    Only the care and value losses are used: the path and observation labels
    of a new circuit would come from an ATPG run we do not have."""
    import torch
    from torch_geometric.data import Batch

    from atpgllm.graph.scripts.train_retest import compute_losses

    graphs = [g for g in (builder.build(row) for row in rows) if g is not None]
    adapted = copy.deepcopy(model)
    if not graphs or steps <= 0:
        return adapted
    for parameter in adapted.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(adapted.parameters(), lr=lr, weight_decay=0.0)
    adapted.eval()                  # normalisation statistics stay those of training
    for _ in range(steps):
        batch = Batch.from_data_list(rng.sample(graphs, min(batch_size, len(graphs)))).to(device)
        enc, queries = adapted(batch)
        losses = compute_losses(adapted, batch, enc, queries, LOSS_ARGS)
        loss = losses["care"] + losses["value"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted.parameters(), 1.0)
        optimizer.step()
    return adapted


def run(args) -> None:
    import torch

    from atpgllm.graph.scripts import retest_hints as rh
    from atpgllm.graph.scripts.train_retest import ProblemGraphs
    from atpgllm.training._paths import DEFAULT_SIM_CONFIG

    device = torch.device(args.device)
    model, vocab = rh.load_graph_model(args.checkpoint, device)
    gate_funcs = json.loads(Path(DEFAULT_SIM_CONFIG).read_text())["gate_funcs"]
    circuits, by_circuit = {}, collections.defaultdict(list)
    with (args.labels / f"circuits_{args.split}.jsonl").open() as fh:
        for line in fh:
            circuit = json.loads(line)
            circuits[circuit["netlist_key"]] = circuit
    with (args.labels / f"problems_{args.split}.jsonl").open() as fh:
        for line in fh:
            problem = json.loads(line)
            by_circuit[problem["netlist_key"]].append(problem)
    keys = sorted(k for k, rows in by_circuit.items() if len(rows) >= args.min_problems)
    builder = ProblemGraphs(circuits, gate_funcs, vocab)
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"circuits.s{args.shard:03d}.jsonl"
    done = r6._done(out_path, lambda r: r["netlist_key"])
    todo = [k for i, k in enumerate(keys) if i % args.shards == args.shard and k not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(keys)} circuits with at least {args.min_problems} problems, {len(todo)} in this shard", flush=True)
    started = time.time()
    with out_path.open("a") as fh:
        for count, key in enumerate(todo):
            rng = random.Random(f"{args.seed}:{key}")
            circuit, rows = circuits[key], list(by_circuit[key])
            rng.shuffle(rows)
            test, pool = rows[:args.test], rows[args.test:args.test + args.adapt]
            with torch.no_grad():
                before = [attempts_of(model, builder, p, circuit, device, args.attempts, stop=False) for p in test]
                found = [(p, attempts_of(model, builder, p, circuit, device, args.attempts, stop=True)) for p in pool]
            found = [(p, f) for p, f in found if f is not None]
            relabeled = [r for r in (sl.relabel({**p, "search": f}, 1.0) for p, f in found) if r is not None]
            adapted = adapt(model, builder, relabeled, device, args.steps, args.batch_size, args.lr, rng)
            with torch.no_grad():
                after = [attempts_of(adapted, builder, p, circuit, device, args.attempts, stop=False) for p in test]
            del adapted
            row = {"netlist_key": key, "kind": "sequential" if circuit.get("scan_chain") else "combinational",
                   "problems": len(rows), "adapt": len(found),
                   "adapt_first": sum(f["found"] == 0 for _, f in found),
                   "adapt_later": sum(f["found"] > 0 for _, f in found),
                   "adapt_simulations": sum(f["tried"] for _, f in found),
                   "test": [{"fault": p["fault"], "difficulty": p.get("difficulty"), "before": b, "after": a}
                            for p, b, a in zip(test, before, after) if b is not None and a is not None]}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"shard {args.shard}: {count + 1}/{len(todo)} circuits, {time.time() - started:.0f}s", flush=True)


def summarize(rows: List[dict], ks=(1, 2, 3, 8, 16)) -> Dict[str, Any]:
    """Detection within k attempts before and after, over all test faults,
    and the paired first-attempt difference with a 95% interval over circuits."""
    tests = [(r["netlist_key"], t) for r in rows for t in r["test"]]
    out: Dict[str, Any] = {"circuits": len(rows), "test_faults": len(tests),
                           "adapt_faults": sum(r["adapt"] for r in rows),
                           "adapt_first": sum(r["adapt_first"] for r in rows),
                           "adapt_later": sum(r["adapt_later"] for r in rows),
                           "adapt_simulations": sum(r["adapt_simulations"] for r in rows)}
    if not tests:
        return out
    for name in ("before", "after"):
        out[name] = {f"within_{k}": sum(any(t[name][:k]) for _, t in tests) / len(tests) for k in ks}
    by = collections.defaultdict(list)
    for key, t in tests:
        by[key].append(float(t["after"][:1] == [True]) - float(t["before"][:1] == [True]))
    groups, rng, means = list(by.values()), random.Random(0), []
    for _ in range(2000):
        sample = [x for _ in groups for x in rng.choice(groups)]
        means.append(sum(sample) / len(sample))
    means.sort()
    out["first_attempt_gain"] = sum(x for g in groups for x in g) / len(tests)
    out["first_attempt_gain_95"] = [means[50], means[1949]]
    return out


def report(args) -> None:
    rows = [json.loads(line) for path in sorted(args.out.glob("circuits.s*.jsonl")) for line in path.open()]
    result = {"all": summarize(rows)}
    for kind in ("combinational", "sequential"):
        result[kind] = summarize([r for r in rows if r["kind"] == kind])
    (args.out / "report.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pass_", metavar="pass", choices=("run", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--min-problems", type=int, default=40)
    parser.add_argument("--test", type=int, default=16, help="Faults per circuit that are only scored.")
    parser.add_argument("--adapt", type=int, default=64, help="Other faults per circuit the search runs on.")
    parser.add_argument("--attempts", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    {"run": run, "report": report}[args.pass_](args)


if __name__ == "__main__":
    main()
