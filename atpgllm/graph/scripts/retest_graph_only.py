"""Graph-only check for the BRIDGES re-test (plan phase R4, gates G1 and G2).

No LLM. For each problem the trained care and value heads give a test
vector: inputs whose care probability is at least ``--threshold`` take the
predicted value, the rest are filled at random. The fast simulator scores it.

Policies, all sharing the same random bits for free inputs:

- ``model``: care and value heads, random fill
- ``random``: every input random (the floor)
- ``oracle``: care and value frequencies from the labels, random fill
  (a check on the decoder and the simulator, should be near 1.0)

G1 passes when internal-fault detection of ``model`` beats ``random`` by at
least ``--g1-margin`` (10 points in the plan). G2 reads the query collapse
metrics on the same graphs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import torch
from torch_geometric.data import Batch

from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.retest_qformer import collapse_metrics
from atpgllm.graph.scripts.train_retest import ProblemGraphs, RetestGraphModel, load_split
from atpgllm.training._paths import DEFAULT_SIM_CONFIG


def location(problem: dict, state_inputs=()) -> str:
    """``state_inputs`` are the QN nets of a scan circuit: a fault there is on
    a state bit, which the test loads through the chain."""
    net = problem["fault"].split(None, 1)[1]
    if net in state_inputs:
        return "state_in"
    if net in problem["inputs"]:
        return "pi_stem"
    if net in problem["outputs"]:
        return "po_stem"
    return "internal"


def detects(sim_model, vector: dict, problem: dict) -> bool:
    from atpgllm.training.tool_feedback import read_snapshot
    from atpgllm.training.tools import run_fault_simulation

    expected = json.loads(problem["example_expected_output"])
    snap = run_fault_simulation(vector, expected, problem["fault"], model=sim_model)
    if "error" in snap.columns:
        return False
    return read_snapshot(snap[["Good Machine", "Bad Machine"]].to_json(), problem["fault"],
                         list(expected)).detected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--labels", required=True, nargs="+",
                    help="One or more label directories (v4, v5).")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-problems", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--g1-margin", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("FAULT_SIM_BACKEND", "fast")
    from atpgllm.training.tools import simulation_model

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    a = state["args"]
    gate_funcs = json.loads(open(DEFAULT_SIM_CONFIG, encoding="utf-8").read())["gate_funcs"]
    vocab = GateAttributeVocab.from_dict(state["vocab"])
    model = RetestGraphModel(vocab, node_dim=a["node_dim"], gin_layers=a["gin_layers"], q_hidden=a["q_hidden"],
                             slots=a["slots"], q_layers=a["q_layers"])
    missing, _ = model.load_state_dict(state["model"], strict=False)
    if any(not k.startswith("text_proj") for k in missing):
        raise RuntimeError(f"checkpoint is missing weights: {missing}")
    device = torch.device(args.device)
    model.to(device).eval()

    problems, circuits = load_split(args.labels, args.split, args.max_problems, args.seed)
    builder = ProblemGraphs(circuits, gate_funcs, vocab)
    sims = {}
    rng = random.Random(args.seed)
    tally = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    queries = []
    for p in problems:
        g = builder.build(p)
        if g is None:
            continue
        batch = Batch.from_data_list([g]).to(device)
        with torch.no_grad():
            _, q = model(batch)
            f = model.q_former
            nodes = f.last["nodes"]
            care = torch.sigmoid(f.pointer_logits("care", q, nodes))[0]
            value = torch.sigmoid(f.value_logits(q, nodes))[0]
        queries.append(q.cpu())
        names = list(g.node_names)
        pi_index = {names[i]: i for i in range(len(names)) if bool(g.pi_node_mask[i])}
        free_bits = {net: rng.randint(0, 1) for net in p["inputs"]}
        vectors = {
            "random": dict(free_bits),
            "model": {net: (int(value[pi_index[net]] >= 0.5) if net in pi_index and care[pi_index[net]] >= args.threshold
                            else free_bits[net]) for net in p["inputs"]},
            "oracle": {net: (int(p["one_freq"][net] >= 0.5) if p["care_freq"][net] >= args.threshold
                             else free_bits[net]) for net in p["inputs"]},
        }
        key = p["netlist_key"]
        if key not in sims:
            sims[key] = simulation_model(circuits[key]["netlist"])
        state_inputs = {names[i] for i in g.state_in_mask.nonzero().flatten().tolist()}
        loc = location(p, state_inputs)
        kind = "sequential" if state_inputs else "combinational"
        for policy, vec in vectors.items():
            hit = detects(sims[key], vec, p)
            for group in ("all", f"kind:{kind}", f"loc:{loc}", f"difficulty:{p.get('difficulty')}"):
                tally[policy][group][0] += int(hit)
                tally[policy][group][1] += 1
    rates = {policy: {group: {"detect": h / n, "n": n} for group, (h, n) in groups.items()}
             for policy, groups in tally.items()}
    internal = {k: rates[k].get("loc:internal", {"detect": float("nan")})["detect"] for k in rates}
    g2 = collapse_metrics(torch.cat(queries)) if queries else {}
    result = {
        "checkpoint": args.checkpoint, "split": args.split, "problems": tally["model"]["all"][1],
        "rejected": builder.rejected, "threshold": args.threshold, "rates": rates,
        "G1": {"internal_model": internal.get("model"), "internal_random": internal.get("random"),
               "margin": args.g1_margin,
               "pass": bool(internal.get("model", 0) - internal.get("random", 0) >= args.g1_margin)},
        "G2": {**g2, "pass": bool(g2 and g2["query_cosine"] < 0.5 and g2["circuit_variance_share"] > 0.30)},
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "graph_only.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps({"G1": result["G1"], "G2": result["G2"], "all": {k: rates[k]["all"] for k in rates}}, indent=1))


if __name__ == "__main__":
    main()
