"""
Precompute one natural-language *testability description* per unique
netlist in the ASAP7 ATPG dataset.

Why
---
Stage-1 of the BRIDGES-style training (`atpgllm.graph.Stage1Trainer`) needs
``(graph, text)`` pairs in a *one graph → one description* discipline.
The raw HF dataset has many records per design (one per ``(fault,
test_vector)``) so the existing per-record caption breaks the
contrastive loss. This script aggregates *across all records of the same
netlist* into a single description that mixes:

- **Structural facts** (cheap, deterministic, derived from the netlist)
- **Empirical testability facts** (cheap, derived from aggregating the
  dataset's per-record fault-propagation / backtrack / detection
  statistics)

Output
------
- A JSON file keyed by netlist sha1 ::

    {
      "<sha1>": {
        "module_name": str,
        "netlist": str,                  # original Verilog
        "n_records": int,
        "structural": {...stats...},
        "empirical":  {...stats...},
        "description": str               # the rendered NL paragraph
      },
      ...
    }

- A human-readable text file with N sample descriptions for review.

CLI
---
::

    python -m atpgllm.graph.scripts.precompute_design_descriptions \\
        --dataset chrivasileiou/asap7-language-of-test-v2 \\
        --split train \\
        --min-records 10 \\
        --output /tmp/design_descriptions.json \\
        --samples-output /tmp/design_descriptions_samples.txt
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset

from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.netlist_parser import (
    ParsedNetlistGraph,
    compute_structural_features,
    parse_verilog_to_graph,
)

# atpgllm/graph/scripts/*.py → parents[3] == libatpgllm package root
_PKG_ROOT = Path(__file__).resolve().parents[3]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# Cell-name canonicaliser for human-readable descriptions.
# ASAP7 cells follow ``<FUNC><inputs?><variant?>x<drive>_ASAP7_<...>``.
# We anchor on the *lowercase* ``x`` that delimits the drive-strength
# suffix: everything before it is the functional prefix.
_BASE_CELL_RE = re.compile(r"^([A-Z][A-Z0-9]*)x")


def _base_cell_name(cell: str) -> str:
    """Strip ASAP7 drive-strength / variant suffix.

    Examples: ``"NAND2x1_ASAP7_75t_R"`` -> ``"NAND2"``,
    ``"INVxp33_ASAP7_75t_R"`` -> ``"INV"``,
    ``"AOI22x1_ASAP7_75t_R"`` -> ``"AOI22"``,
    ``"A2O1A1O1Ix1_ASAP7_75t_R"`` -> ``"A2O1A1O1I"``,
    ``"FAx1_ASAP7_75t_R"`` -> ``"FA"``.
    """
    m = _BASE_CELL_RE.match(cell)
    if m:
        return m.group(1)
    # Cells without the ``x<drive>`` suffix (rare): fall back to the
    # part before ``_ASAP7``.
    return cell.split("_ASAP7", 1)[0]


def _split_csv(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def _classify_fault_site(net: str, inputs: set[str], outputs: set[str]) -> str:
    """Return ``"pi"`` / ``"po"`` / ``"internal"`` for a fault net name."""
    if net in outputs:
        return "po"
    if net in inputs:
        return "pi"
    # Common quirk: synthesiser uses ``y_82`` as a wire driving ``y[82]``.
    # Treat the bracket-form match as PO too.
    m = re.match(r"^([A-Za-z][\w]*?)_(\d+)$", net)
    if m:
        bracket = f"{m.group(1)}[{m.group(2)}]"
        if bracket in outputs:
            return "po"
        if bracket in inputs:
            return "pi"
    return "internal"


def _percentile(xs: List[int], p: float) -> int:
    """Crude percentile (linear-interp) for a list of ints; 0 on empty."""
    if not xs:
        return 0
    xs_sorted = sorted(xs)
    if p <= 0:
        return xs_sorted[0]
    if p >= 100:
        return xs_sorted[-1]
    k = (len(xs_sorted) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs_sorted) - 1)
    frac = k - lo
    return int(round(xs_sorted[lo] * (1 - frac) + xs_sorted[hi] * frac))


# ----------------------------------------------------------------------
# Per-design aggregator
# ----------------------------------------------------------------------


@dataclass
class DesignAccumulator:
    netlist_hash: str
    netlist: str
    module_name: str

    # Counters that accumulate across records of the same design.
    n_records: int = 0
    n_sa0: int = 0
    n_sa1: int = 0
    site_counts: collections.Counter = None  # pi/po/internal
    prop_gate_counts: collections.Counter = None
    backtrack_gate_counts: collections.Counter = None
    obs_net_counts: collections.Counter = None
    backtrack_net_counts: collections.Counter = None
    prop_cone_sizes: List[int] = None
    backtrack_depths: List[int] = None
    detected_fault_counts: List[int] = None  # detected per record

    def __post_init__(self) -> None:
        self.site_counts = collections.Counter()
        self.prop_gate_counts = collections.Counter()
        self.backtrack_gate_counts = collections.Counter()
        self.obs_net_counts = collections.Counter()
        self.backtrack_net_counts = collections.Counter()
        self.prop_cone_sizes = []
        self.backtrack_depths = []
        self.detected_fault_counts = []

    def update(self, record: Dict[str, Any], inputs: set[str], outputs: set[str]) -> None:
        self.n_records += 1

        fault = (record.get("fault") or "").strip()
        m = re.match(r"^(sa[01])\s+(.+)$", fault)
        if m:
            kind, net = m.group(1), m.group(2).strip()
            if kind == "sa0":
                self.n_sa0 += 1
            else:
                self.n_sa1 += 1
            self.site_counts[_classify_fault_site(net, inputs, outputs)] += 1

        prop_gates = _split_csv(record.get("fault_propagation_gates"))
        bt_gates = _split_csv(record.get("backtrack_gates"))
        prop_nets = _split_csv(record.get("fault_propagation_nets"))
        bt_nets = _split_csv(record.get("backtrack_nets"))
        detected = _split_csv(record.get("detected_faults"))

        for g in prop_gates:
            self.prop_gate_counts[g] += 1
        for g in bt_gates:
            self.backtrack_gate_counts[g] += 1
        # Observation cluster = primary-output nets that show up in
        # ``fault_propagation_nets`` (these are the POs the fault
        # eventually reaches).
        for n in prop_nets:
            if n in outputs:
                self.obs_net_counts[n] += 1
        for n in bt_nets:
            self.backtrack_net_counts[n] += 1

        self.prop_cone_sizes.append(len(prop_gates))
        self.backtrack_depths.append(len(bt_gates))
        self.detected_fault_counts.append(len(detected))


# ----------------------------------------------------------------------
# Structural analysis (one-shot per unique netlist)
# ----------------------------------------------------------------------


def _expand_pi_po_groups(parsed: ParsedNetlistGraph) -> Tuple[List[str], List[str]]:
    """Best-effort: collapse expanded scalar PI/PO names back to ``base[n]``
    bus groupings. Returns two lists of strings like ``"ctrl[8]"`` (8-bit
    bus). Falls back to bare names when no bracket form is detected.
    """
    def _group(names: List[str]) -> List[str]:
        buckets: Dict[str, int] = {}
        scalars: List[str] = []
        for n in names:
            m = re.match(r"^([A-Za-z][\w]*)\[(\d+)\]$", n)
            if m:
                buckets[m.group(1)] = max(int(m.group(2)) + 1, buckets.get(m.group(1), 0))
            else:
                scalars.append(n)
        out = [f"{k}[{w}]" for k, w in sorted(buckets.items())]
        out.extend(sorted(scalars))
        return out

    return _group(parsed.inputs), _group(parsed.outputs)


def _structural_stats(
    netlist: str,
    gate_funcs: Dict[str, Any],
    vocab: GateAttributeVocab,
) -> Optional[Dict[str, Any]]:
    """Parse + compute design-level structural facts. Returns ``None`` if
    parsing yields a degenerate graph.
    """
    try:
        parsed = parse_verilog_to_graph(netlist, gate_funcs)
    except Exception:
        return None
    if not parsed.gates:
        return None

    # Build edges (driver -> sink) for depth + fanout computation.
    edge_src: List[int] = []
    edge_dst: List[int] = []
    for net, drivers in parsed.net_drivers.items():
        sinks = parsed.net_sinks.get(net, [])
        if not drivers or not sinks:
            continue
        for d in drivers:
            for s in sinks:
                edge_src.append(d)
                edge_dst.append(s)

    import torch
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.int64)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.int64)

    sf = compute_structural_features(edge_index, len(parsed.gates))
    # Columns: forward_depth, backward_depth, log_in_deg, log_out_deg
    # (normalised). Recover unnormalised forward depth by * max_fwd.
    fwd_norm = sf[:, 0].tolist() if sf.numel() else []

    # Recompute discrete fwd depth (already done internally; expose max).
    # The normalised column is in [0,1]; scale up using the count of
    # edges to estimate depth in a topo sense — but it's cleaner to
    # recompute from edges directly.
    n = len(parsed.gates)
    in_deg = [0] * n
    out_deg = [0] * n
    succ: List[List[int]] = [[] for _ in range(n)]
    for s, d in zip(edge_src, edge_dst):
        succ[s].append(d)
        in_deg[d] += 1
        out_deg[s] += 1

    # Topological order + longest-path forward depth.
    fwd_depth = [0] * n
    from collections import deque
    q: deque[int] = deque(i for i, x in enumerate(in_deg) if x == 0)
    remaining = list(in_deg)
    while q:
        v = q.popleft()
        for w in succ[v]:
            fwd_depth[w] = max(fwd_depth[w], fwd_depth[v] + 1)
            remaining[w] -= 1
            if remaining[w] == 0:
                q.append(w)

    max_depth = max(fwd_depth) if fwd_depth else 0

    # Cell-base histogram (top by count).
    cell_counter: collections.Counter = collections.Counter()
    for g in parsed.gates:
        cell_counter[_base_cell_name(g.cell)] += 1
    total = sum(cell_counter.values()) or 1
    cell_mix = [
        {"cell": c, "count": k, "pct": round(100.0 * k / total, 1)}
        for c, k in cell_counter.most_common(8)
    ]

    # Top fanout *nets* (not gates): a net's fanout = number of sinks.
    net_fanout: collections.Counter = collections.Counter()
    for net, sinks in parsed.net_sinks.items():
        net_fanout[net] = len(sinks)
    top_fanout_nets = [
        {"net": n, "fanout": f}
        for n, f in net_fanout.most_common(10)
        if f >= 4
    ][:5]

    pi_groups, po_groups = _expand_pi_po_groups(parsed)

    return {
        "n_gates": len(parsed.gates),
        "n_pi_bits": len(parsed.inputs),
        "n_po_bits": len(parsed.outputs),
        "pi_groups": pi_groups,
        "po_groups": po_groups,
        "logic_depth": max_depth,
        "cell_mix": cell_mix,
        "top_fanout_nets": top_fanout_nets,
        "n_high_fanout_nets": sum(1 for v in net_fanout.values() if v >= 4),
        "_inputs_set": set(parsed.inputs),
        "_outputs_set": set(parsed.outputs),
    }


# ----------------------------------------------------------------------
# Empirical reduction
# ----------------------------------------------------------------------


def _empirical_stats(acc: DesignAccumulator, top_k: int) -> Dict[str, Any]:
    """Reduce accumulators into a JSON-serialisable summary."""
    n = acc.n_records or 1

    def _rank(counter: collections.Counter, k: int) -> List[Dict[str, Any]]:
        return [
            {"name": name, "count": cnt, "pct": round(100.0 * cnt / n, 1)}
            for name, cnt in counter.most_common(k)
        ]

    sites_total = sum(acc.site_counts.values()) or 1
    site_dist = {
        site: round(100.0 * acc.site_counts.get(site, 0) / sites_total, 1)
        for site in ("pi", "po", "internal")
    }

    return {
        "n_records": acc.n_records,
        "fault_kind_pct": {
            "sa0": round(100.0 * acc.n_sa0 / n, 1),
            "sa1": round(100.0 * acc.n_sa1 / n, 1),
        },
        "fault_site_pct": site_dist,
        "top_propagation_gates": _rank(acc.prop_gate_counts, top_k),
        "top_backtrack_gates": _rank(acc.backtrack_gate_counts, top_k),
        "top_observation_pos": _rank(acc.obs_net_counts, top_k),
        "top_backtrack_nets": _rank(acc.backtrack_net_counts, top_k),
        "prop_cone_size": {
            "median": _percentile(acc.prop_cone_sizes, 50),
            "p90": _percentile(acc.prop_cone_sizes, 90),
            "max": max(acc.prop_cone_sizes) if acc.prop_cone_sizes else 0,
            "n_zero": sum(1 for x in acc.prop_cone_sizes if x == 0),
        },
        "backtrack_depth": {
            "median": _percentile(acc.backtrack_depths, 50),
            "p90": _percentile(acc.backtrack_depths, 90),
            "max": max(acc.backtrack_depths) if acc.backtrack_depths else 0,
            "n_zero": sum(1 for x in acc.backtrack_depths if x == 0),
            "n_deep": sum(1 for x in acc.backtrack_depths if x >= 10),
        },
    }


# ----------------------------------------------------------------------
# Description renderer (NL paragraph, deterministic templates)
# ----------------------------------------------------------------------


def _humanize_count(n: int, unit: str) -> str:
    if n == 1:
        return f"1 {unit}"
    return f"{n} {unit}s"


def _join(items: List[str], conj: str = "and") -> str:
    items = [x for x in items if x]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conj} {items[1]}"
    return ", ".join(items[:-1]) + f", {conj} {items[-1]}"


def render_description(
    structural: Dict[str, Any],
    empirical: Dict[str, Any],
) -> str:
    """Render a name-anonymised, ATPG-methodology-oriented paragraph.

    The text deliberately uses D-algorithm vocabulary (excite, sensitise,
    forward-propagate, backtrack for consistency, observe a difference
    between good and faulty machine) so that the Stage-1 text encoder
    learns to align graph queries with the canonical language of test
    pattern generation while still being driven by *design-specific*
    aggregate statistics.
    """

    def _maybe_truncate(groups: List[str], k: int = 8) -> str:
        if len(groups) <= k:
            return _join(groups, conj="and")
        return _join(groups[:k], conj="and") + f", ... (+{len(groups) - k} more)"

    pi_str = _maybe_truncate(structural["pi_groups"]) or "<none>"
    po_str = _maybe_truncate(structural["po_groups"]) or "<none>"

    # ----- 1. Structural overview (no module name; widths + counts only) -----
    interface = (
        f"This combinational ASAP7 standard-cell netlist exposes "
        f"{structural['n_pi_bits']} primary input bits ({pi_str}) and "
        f"{structural['n_po_bits']} primary output bits ({po_str})."
    )

    # ----- 2. Implementation -----
    mix = structural["cell_mix"][:5]
    mix_str = ", ".join(f"{c['cell']} ({c['pct']}%)" for c in mix)
    impl = (
        f"The implementation uses {structural['n_gates']} cells, dominated by "
        f"{mix_str}; the longest combinational path traverses "
        f"{structural['logic_depth']} levels of logic."
    )
    if structural["top_fanout_nets"]:
        fanout_str = ", ".join(
            f"{n['net']} (fanout {n['fanout']})"
            for n in structural["top_fanout_nets"]
        )
        impl += (
            f" High-fanout stems concentrate on {fanout_str}, with "
            f"{structural['n_high_fanout_nets']} signals exceeding fanout 3."
        )

    # ----- 3. ATPG-corpus summary -----
    n_rec = empirical["n_records"]
    site = empirical["fault_site_pct"]
    sa = empirical["fault_kind_pct"]
    corpus_sent = (
        f"Across {n_rec} ATPG records for this design, the targets split "
        f"{sa['sa0']}% stuck-at-0 / {sa['sa1']}% stuck-at-1, sited on "
        f"internal nets ({site['internal']}%), primary-output drivers "
        f"({site['po']}%), and primary-input drivers ({site['pi']}%)."
    )

    # ----- 4. Excitation framing -----
    # We don't have per-design "controlling PI" statistics, so this
    # sentence is methodology-only but tied to the corpus statistics.
    excite_sent = (
        "Each fault is excited by selecting primary-input assignments that "
        "drive the target net to its non-faulty logic value, placing a D "
        "(or D-bar) on the fault site."
    )

    # ----- 5. Forward propagation through D-algorithm -----
    prop = empirical["top_propagation_gates"]
    if prop:
        prop_str = ", ".join(f"{g['name']} ({g['pct']}%)" for g in prop[:4])
        prop_sent = (
            f"Forward propagation of the fault effect via the D-algorithm "
            f"typically traverses {prop_str}, the recurrent observability "
            f"bottlenecks where side-inputs must be set to non-controlling "
            f"values to keep the D path alive."
        )
    else:
        prop_sent = (
            "Most faults are sited directly at primary-output drivers, so "
            "forward propagation is trivial (the fault site is already an "
            "observation point) for the bulk of records."
        )

    # ----- 6. Backward consistency check -----
    bt = empirical["top_backtrack_gates"]
    if bt:
        bt_str = ", ".join(f"{g['name']} ({g['pct']}%)" for g in bt[:4])
        bt_sent = (
            f"The backward consistency check between propagation requirements "
            f"and excitation assignments most frequently backtracks through "
            f"{bt_str}, marking these cells as the controllability hubs of "
            f"the circuit."
        )
    else:
        bt_sent = (
            "The backward consistency check is shallow: backtrack rarely "
            "leaves the immediate fan-in cone of the fault site."
        )

    # ----- 7. Observation point -----
    obs = empirical["top_observation_pos"]
    if obs:
        obs_str = ", ".join(f"{o['name']} ({o['pct']}%)" for o in obs[:5])
        obs_sent = (
            f"The fault is finally detected as a logic difference between "
            f"the good and faulty machine at primary outputs {obs_str}, the "
            f"dominant observation cluster."
        )
    else:
        obs_sent = ""

    # ----- 8. Test-difficulty distribution -----
    cone = empirical["prop_cone_size"]
    bt_d = empirical["backtrack_depth"]
    dist_sent = (
        f"Across the corpus, propagation cones have median size "
        f"{cone['median']} gates (p90 {cone['p90']}, max {cone['max']}) "
        f"and backtrack depth has median {bt_d['median']} (p90 "
        f"{bt_d['p90']}, max {bt_d['max']})."
    )

    # ----- 9. Hard-fault callout -----
    n_deep = bt_d["n_deep"]
    if n_deep > 0:
        hard_sent = (
            f"{n_deep} of the {n_rec} faults required deep backtrack "
            f"(\u226510 gates), evidence of hard-to-control sub-circuits "
            f"that dominate the test-generation cost for this design."
        )
    else:
        hard_sent = ""

    parts = [
        interface, impl, corpus_sent, excite_sent, prop_sent,
        bt_sent, obs_sent, dist_sent, hard_sent,
    ]
    return " ".join(p for p in parts if p)


# ----------------------------------------------------------------------
# Two-pass driver
# ----------------------------------------------------------------------


def _pass1_count_hashes(stream) -> Dict[str, int]:
    """Count how many records each unique netlist (by sha1) appears in."""
    counts: Dict[str, int] = collections.Counter()
    for i, rec in enumerate(stream):
        nl = rec.get("netlist") or ""
        if not nl:
            continue
        counts[_sha1(nl.strip())] += 1
        if (i + 1) % 5000 == 0:
            print(f"  [pass-1] scanned {i+1:>7d} records, "
                  f"{len(counts):>5d} unique designs so far...", flush=True)
    return counts


def _pass2_aggregate(
    stream,
    qualifying: set[str],
    gate_funcs: Dict[str, Any],
    vocab: GateAttributeVocab,
) -> Dict[str, Tuple[DesignAccumulator, Dict[str, Any]]]:
    """Aggregate per-record stats and structural facts for qualifying designs."""
    accs: Dict[str, DesignAccumulator] = {}
    structurals: Dict[str, Dict[str, Any]] = {}
    parse_failures: set[str] = set()

    for i, rec in enumerate(stream):
        nl = rec.get("netlist") or ""
        if not nl:
            continue
        h = _sha1(nl.strip())
        if h not in qualifying or h in parse_failures:
            continue

        if h not in structurals:
            s = _structural_stats(nl, gate_funcs, vocab)
            if s is None:
                parse_failures.add(h)
                continue
            structurals[h] = s
            accs[h] = DesignAccumulator(
                netlist_hash=h,
                netlist=nl,
                module_name=str(rec.get("module_name") or ""),
            )

        s = structurals[h]
        accs[h].update(rec, s["_inputs_set"], s["_outputs_set"])

        if (i + 1) % 5000 == 0:
            print(f"  [pass-2] processed {i+1:>7d} records, "
                  f"{len(accs):>5d} designs accumulated, "
                  f"{len(parse_failures):>4d} parse failures...", flush=True)

    return accs, structurals, parse_failures


def _strip_internal(structural: Dict[str, Any]) -> Dict[str, Any]:
    """Drop fields prefixed with ``_`` before JSON serialisation."""
    return {k: v for k, v in structural.items() if not k.startswith("_")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="chrivasileiou/asap7-language-of-test-v2")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--sim-config",
        default=str(_PKG_ROOT / "tests" / "sim_config.json"),
        help="Path to sim_config.json (used for gate_funcs).",
    )
    p.add_argument("--min-records", type=int, default=10)
    p.add_argument("--top-k", type=int, default=5,
                   help="K for top-K propagation/backtrack hubs.")
    p.add_argument("--output", default="design_descriptions.json")
    p.add_argument("--samples-output", default="design_descriptions_samples.txt")
    p.add_argument("--num-samples", type=int, default=20)
    args = p.parse_args()

    print(f"Loading gate_funcs from {args.sim_config}", flush=True)
    with open(args.sim_config, "r", encoding="utf-8") as f:
        gate_funcs = json.load(f)["gate_funcs"]
    vocab = GateAttributeVocab(gate_funcs)

    print(f"\nPass 1: counting unique netlists in "
          f"{args.dataset!r} split={args.split!r}", flush=True)
    ds1 = load_dataset(args.dataset, split=args.split, streaming=True)
    counts = _pass1_count_hashes(iter(ds1))

    qualifying = {h for h, c in counts.items() if c >= args.min_records}
    print(f"\n  Total unique netlists:    {len(counts):>6d}")
    print(f"  Designs with ≥{args.min_records:<2d} records: {len(qualifying):>6d}")
    print(f"  Total records:            {sum(counts.values()):>6d}")
    print(f"  Records covered by ≥{args.min_records}:  "
          f"{sum(c for h, c in counts.items() if h in qualifying):>6d}")

    if not qualifying:
        print("\nNo designs meet the threshold. Exiting.", flush=True)
        return

    print(f"\nPass 2: aggregating empirical stats per qualifying design", flush=True)
    ds2 = load_dataset(args.dataset, split=args.split, streaming=True)
    accs, structurals, parse_failures = _pass2_aggregate(
        iter(ds2), qualifying, gate_funcs, vocab,
    )
    print(f"\n  Designs aggregated:    {len(accs):>5d}")
    print(f"  Parse failures:        {len(parse_failures):>5d}")

    print(f"\nRendering descriptions", flush=True)
    output: Dict[str, Dict[str, Any]] = {}
    for h, acc in accs.items():
        s = structurals[h]
        e = _empirical_stats(acc, args.top_k)
        desc = render_description(s, e)
        output[h] = {
            "module_name": acc.module_name,
            "netlist": acc.netlist,
            "n_records": acc.n_records,
            "structural": _strip_internal(s),
            "empirical": e,
            "description": desc,
        }

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(output)} design descriptions to {out_path}")

    # Write sample file (sorted by record count desc, then alphabetical).
    samples = sorted(
        output.items(),
        key=lambda kv: (-kv[1]["n_records"], kv[1]["module_name"]),
    )[: args.num_samples]
    sample_path = Path(args.samples_output).expanduser().resolve()
    with sample_path.open("w", encoding="utf-8") as f:
        for i, (h, item) in enumerate(samples):
            f.write(f"=== Sample {i+1}/{len(samples)}  "
                    f"module={item['module_name']}  "
                    f"hash={h[:10]}  n_records={item['n_records']} ===\n")
            f.write(item["description"])
            f.write("\n\n")
    print(f"Wrote {len(samples)} sample descriptions to {sample_path}")


if __name__ == "__main__":
    main()
