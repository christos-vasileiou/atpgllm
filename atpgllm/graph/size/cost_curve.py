"""S0 cost curve: time and memory of the per-circuit and per-fault stages against V, E and L.

Runs on CPU so it can run beside GPU jobs. For each netlist: parse and
levelize, the trunk sweep (``--dim``) without gradients, the spectrum, and
``--faults`` views plus fault-head passes. Memory is the trunk state that
must be kept, V·d floats.

Usage::

    python -m atpgllm.graph.size.cost_curve --coverage dataset_analysis/size_windows_20261105/parser_coverage.json \
        --stats dataset_analysis/size_windows_20261105/graph_stats.json --out cost_curve.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
from torch_geometric.data import Batch

from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.size.check1 import FREESET
from atpgllm.graph.size.fault_head import FaultHead, view_tokens
from atpgllm.graph.size.fovea import build_view
from atpgllm.graph.size.spectrum import spectrum_dim
from atpgllm.graph.size.trunk import TopoSweepTrunk
from atpgllm.training._paths import DEFAULT_SIM_CONFIG


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--per-bucket", type=int, default=4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--faults", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("FAULT_SIM_BACKEND", "fast")
    torch.set_grad_enabled(False)
    from atpgllm.multimodal.size_stack import build_circuit_cache

    gate_funcs = json.loads(open(DEFAULT_SIM_CONFIG).read())["gate_funcs"]
    vocab = GateAttributeVocab(gate_funcs)
    accepted = set(json.load(open(args.coverage))["accepted"])
    stats = [s for s in json.load(open(args.stats)) if s["f"] in accepted]
    rng = random.Random(args.seed)
    chosen = []
    for lo, hi in ((1_000, 3_000), (3_000, 10_000), (10_000, 50_000), (50_000, 10**9)):
        group = [s for s in stats if lo <= s["V"] < hi]
        rng.shuffle(group)
        chosen += group[: args.per_bucket]
    trunk = TopoSweepTrunk(vocab, dim=args.dim).eval()
    head = FaultHead(args.dim, spectrum_dim()).eval()
    rows = []
    for s in sorted(chosen, key=lambda r: r["V"]):
        t0 = time.time()
        cache = build_circuit_cache(open(os.path.join(FREESET, s["f"])).read(), gate_funcs, vocab)
        t_prep = time.time() - t0
        t0 = time.time()
        vecs = trunk(Batch.from_data_list([cache.pyg]))["node_embs"]
        t_trunk = time.time() - t0
        nets = [n for n in cache.graph.order if not n.isdigit()]
        t_view = t_head = 0.0
        tokens = []
        for net in rng.sample(nets, min(args.faults, len(nets))):
            t0 = time.time()
            view = build_view(cache.graph, net)
            vt = view_tokens(view, cache.graph)
            t_view += time.time() - t0
            idx = torch.tensor([cache.node_of_net.get(n, 0) for n in vt["nets"]])
            t0 = time.time()
            head(vecs[idx], torch.tensor(vt["roles"]), torch.tensor(vt["hops"]), 0, cache.spectrum)
            t_head += time.time() - t0
            tokens.append(len(vt["nets"]))
        n = max(1, min(args.faults, len(nets)))
        rows.append({"file": s["f"], "V": cache.pyg.num_nodes, "E": int(cache.pyg.edge_index.size(1)),
                     "L": int(cache.pyg.level.max()), "prepare_s": round(t_prep, 2), "trunk_s": round(t_trunk, 3),
                     "trunk_state_mb": round(cache.pyg.num_nodes * args.dim * 4 / 2**20, 1),
                     "view_ms_per_fault": round(1000 * t_view / n, 2), "head_ms_per_fault": round(1000 * t_head / n, 2),
                     "graph_tokens_mean": round(sum(tokens) / max(1, len(tokens)), 1)})
        print(json.dumps(rows[-1]), flush=True)
    json.dump({"args": vars(args), "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
