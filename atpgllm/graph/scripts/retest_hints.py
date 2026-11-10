"""Text hints for the hint arm of the BRIDGES re-test.

For every problem of a split, the trained graph model (R3 checkpoint) marks
the inputs to set (care probability at least ``--threshold``) and picks
their values. This is the vector R4 scored. It is written here as JSONL,
one row per problem, and the Stage C trainer puts it into the prompt as
plain text (``--hints``), with no graph tokens at all.

The hint arm asks whether a latent channel into the LLM is needed: if
Granite with this text does as well as Granite with the graph tokens, the
graph model is better used as a proposer the LLM reads.

Each run also writes ``summary_<split>.json`` with how often a hint agrees
with the dataset's patterns (the answer key vote of R4). The graph model was
trained on the train split's labels, so its train hints are better than its
validation hints, and a model trained on them may trust a hint more than it
should on new circuits. Compare the two summaries before reading a result.

    PYTHONPATH=$PWD python -m atpgllm.graph.scripts.retest_hints \\
        --checkpoint runs/bridges_retest/r3/best.pt \\
        --labels ../data/freeset/retest_labels_v4 --split validation \\
        --out runs/bridges_retest/hints
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch_geometric.data import Batch

from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.scripts.train_retest import ProblemGraphs, RetestGraphModel, load_split
from atpgllm.training._paths import DEFAULT_SIM_CONFIG


def load_graph_model(checkpoint: str, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    a = state["args"]
    vocab = GateAttributeVocab.from_dict(state["vocab"])
    model = RetestGraphModel(vocab, node_dim=a["node_dim"], gin_layers=a["gin_layers"],
                             q_hidden=a["q_hidden"], slots=a["slots"], q_layers=a["q_layers"])
    missing, _ = model.load_state_dict(state["model"], strict=False)
    if any(not k.startswith("text_proj") for k in missing):
        raise RuntimeError(f"checkpoint is missing weights: {missing}")
    return model.to(device).eval(), vocab


@torch.no_grad()
def decode_hint(model, graph, device: torch.device, threshold: float = 0.5) -> dict:
    """``{input: 0 or 1}`` for the inputs the care head marks."""
    _, queries = model(Batch.from_data_list([graph]).to(device))
    former = model.q_former
    nodes = former.last["nodes"]
    care = torch.sigmoid(former.pointer_logits("care", queries, nodes))[0]
    value = torch.sigmoid(former.value_logits(queries, nodes))[0]
    return {
        name: int(value[i] >= 0.5)
        for i, name in enumerate(graph.node_names)
        if bool(graph.pi_node_mask[i]) and care[i] >= threshold
    }


def agreement(hint: dict, problem: dict, threshold: float = 0.5) -> dict:
    """The hint against the vote of the dataset's patterns for this problem."""
    key = {n: int(problem["one_freq"][n] >= 0.5)
           for n, c in problem["care_freq"].items() if c >= threshold}
    shared = set(hint) & set(key)
    return {
        "same_inputs": set(hint) == set(key),
        "same_vector": hint == key,
        "shared": len(shared),
        "shared_same_value": sum(hint[n] == key[n] for n in shared),
        "hinted": len(hint),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1, help="Number of shards; this process writes shard --shard.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)   # one process per shard, small graphs
    device = torch.device(args.device)
    model, vocab = load_graph_model(args.checkpoint, device)
    gate_funcs = json.loads(open(DEFAULT_SIM_CONFIG, encoding="utf-8").read())["gate_funcs"]
    problems, circuits = load_split(args.labels, args.split, args.limit, 0)
    problems = problems[args.shard::args.shards]
    builder = ProblemGraphs(circuits, gate_funcs, vocab)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.split}_{args.shard:03d}.jsonl"
    done = set()
    if path.exists():
        done = {(r["netlist_key"], r["fault"]) for r in map(json.loads, path.open())}
    totals = {"problems": 0, "same_inputs": 0, "same_vector": 0, "shared": 0,
              "shared_same_value": 0, "hinted": 0, "empty": 0}
    with path.open("a") as fh:
        for index, p in enumerate(problems):
            if (p["netlist_key"], p["fault"]) in done:
                continue
            graph = builder.build(p)
            if graph is None:
                continue
            hint = decode_hint(model, graph, device, args.threshold)
            fh.write(json.dumps({"netlist_key": p["netlist_key"], "fault": p["fault"], "hint": hint}) + "\n")
            for key, value in agreement(hint, p, args.threshold).items():
                totals[key] += int(value)
            totals["problems"] += 1
            totals["empty"] += not hint
            if index % 5000 == 0:
                print(f"{args.split} shard {args.shard}: {index}/{len(problems)}", flush=True)
    totals["rejected"] = builder.rejected
    totals["resumed_past"] = len(done)
    summary = args.out / f"summary_{args.split}_{args.shard:03d}.json"
    summary.write_text(json.dumps(totals, indent=1) + "\n")
    n = max(1, totals["problems"])
    print(json.dumps({"split": args.split, "shard": args.shard, "problems": totals["problems"],
                      "same_inputs": totals["same_inputs"] / n, "same_vector": totals["same_vector"] / n,
                      "value_agreement": totals["shared_same_value"] / max(1, totals["shared"]),
                      "hinted_per_problem": totals["hinted"] / n, "empty": totals["empty"] / n}))


if __name__ == "__main__":
    os.environ.setdefault("FAULT_SIM_BACKEND", "fast")
    main()
