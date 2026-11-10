"""Check 1 of the size stack: does the view hold the answer? No training.

For sampled faults on large netlists, random patterns are simulated with
the fast simulator until one detects. The nets that differ between the good
and faulty machine under that pattern are the fault effect. A fault passes
for a given (cap, k) when some chain of differing nets from the fault to a
detecting output lies in the S1 view and every other input of the gates on
that chain is in the view or in the S2 side tokens.

Bias: only faults that random patterns detect are scored, so hard faults
are missing. The same harness gives the labels for large-circuit training.

Usage::

    python -m atpgllm.graph.size.check1 --stats dataset_analysis/size_windows_20261105/graph_stats.json \
        --out dataset_analysis/size_windows_20261105/check1.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict

from .fovea import build_view, effect_chain_covered
from .netgraph import build_netgraph, clean_netlist

FREESET = "/proj/trela/christos/transformers_atpg/data/structural.v.freeset.asap7sc7p5t_28.rvt.tt"
BUCKETS = ((1_000, 3_000, "1k-3k"), (3_000, 10_000, "3k-10k"), (10_000, 10**9, "10k+"))


def bucket_of(v: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= v < hi:
            return name
    return "small"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stats", required=True, help="graph_stats.json (file, V, L per netlist)")
    ap.add_argument("--freeset", default=FREESET)
    ap.add_argument("--per-bucket", type=int, default=10, help="netlists per bucket (10k+ takes all)")
    ap.add_argument("--faults", type=int, default=16)
    ap.add_argument("--patterns", type=int, default=16)
    ap.add_argument("--caps", default="128,256,512")
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--side-cap", type=int, default=64)
    ap.add_argument("--max-seconds", type=float, default=6 * 3600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("FAULT_SIM_BACKEND", "fast")
    from atpgllm.training.tool_feedback import read_snapshot
    from atpgllm.training.tools import run_fault_simulation, simulation_model

    rng = random.Random(args.seed)
    stats = json.load(open(args.stats))
    chosen = []
    for lo, hi, name in BUCKETS:
        group = [s for s in stats if lo <= s["V"] < hi]
        rng.shuffle(group)
        chosen += group if name == "10k+" else group[: args.per_bucket]
    caps = [int(c) for c in args.caps.split(",")]
    ks = [int(k) for k in args.ks.split(",")]
    records = []
    start = time.time()
    for s in chosen:
        if time.time() - start > args.max_seconds:
            break
        text = clean_netlist(open(os.path.join(args.freeset, s["f"])).read())
        try:
            model = simulation_model(text)
            g = build_netgraph(model)
        except Exception as exc:  # a netlist the simulator cannot compile
            records.append({"file": s["f"], "error": repr(exc)[:200]})
            continue
        candidates = [n for n in g.order if not n.isdigit()]
        for net in rng.sample(candidates, min(args.faults, len(candidates))):
            fault = f"sa{rng.randint(0, 1)} {net}"
            expected = {o: 0 for o in model.output_nets}
            hit = None
            for _ in range(args.patterns):
                vector = {i: rng.randint(0, 1) for i in model.input_nets}
                snap = run_fault_simulation(vector, expected, fault, model=model)
                if "error" in snap.columns:
                    break
                reading = read_snapshot(snap[["Good Machine", "Bad Machine"]].to_json(), fault,
                                        list(model.output_nets))
                if reading.detected:
                    hit = reading
                    break
            rec = {"file": s["f"], "V": s["V"], "L": s["L"], "bucket": bucket_of(s["V"]), "fault": fault,
                   "location": "pi" if net in g.inputs else ("po" if net in set(g.outputs) else "internal"),
                   "detected": hit is not None}
            if hit is not None:
                differing = list(hit.differing_internal) + list(hit.differing_outputs)
                rec["effect_nets"] = len(differing)
                for cap in caps:
                    for k in ks:
                        view = build_view(g, net, cap=cap, k=k, side_cap=args.side_cap)
                        rec[f"pass_c{cap}_k{k}"] = bool(effect_chain_covered(
                            g, view, differing, list(hit.differing_outputs)))
                        rec[f"tokens_c{cap}_k{k}"] = view.tokens
                        rec[f"fanin_cut_c{cap}_k{k}"] = view.fanin_truncated
            records.append(rec)
        print(s["f"], s["V"], round(time.time() - start), flush=True)
    summary = defaultdict(dict)
    for name in [b[2] for b in BUCKETS] + ["all"]:
        rs = [r for r in records if "fault" in r and (name == "all" or r["bucket"] == name)]
        det = [r for r in rs if r["detected"]]
        summary[name]["faults"] = len(rs)
        summary[name]["random_detected"] = len(det)
        for cap in caps:
            for k in ks:
                key = f"c{cap}_k{k}"
                if det:
                    summary[name][f"pass_{key}"] = round(sum(r[f"pass_{key}"] for r in det) / len(det), 3)
                internal = [r for r in det if r["location"] == "internal"]
                if internal:
                    summary[name][f"pass_internal_{key}"] = round(
                        sum(r[f"pass_{key}"] for r in internal) / len(internal), 3)
    json.dump({"args": vars(args), "summary": summary, "records": records}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
