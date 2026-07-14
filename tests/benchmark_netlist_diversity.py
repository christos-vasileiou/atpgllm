#!/usr/bin/env python3
"""Benchmark netlist-diversity per effective batch for the three buffer-ordering
strategies used by ``buffer_streaming_dataset`` / ``interleave_for_field_diversity``:

    random        — seeded global shuffle (today's baseline)
    round_robin   — cycle netlist groups one item at a time (largest first)
    even_spacing  — spread each netlist's faults uniformly across the epoch

The strategy implementations below are byte-for-byte equivalent to those in
``dataset_utils.py`` (kept inline so this script stays dependency-free).

Effective batch (sequences) = num_processes * per_device_train_batch_size *
gradient_accumulation_steps. Each *unique prompt* is replicated
``num_generations`` times by TRL's RepeatSampler, so the number of distinct
prompts (hence netlists) available in one optimizer step is

    W_prompts = effective_batch_sequences / num_generations

Netlist diversity can only be created across those W_prompts, so that is the
window we score.

Usage:
    python benchmark_netlist_diversity.py                 # synthetic regimes
    python benchmark_netlist_diversity.py --real dist.json  # real fault-count list
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import deque


# --------------------------------------------------------------------------
# Ordering strategies (mirror dataset_utils.interleave_for_field_diversity)
# --------------------------------------------------------------------------
def _grouped_shuffled(labels, seed):
    rng = random.Random(seed)
    groups = {}
    for x in labels:
        groups.setdefault(x, []).append(x)
    ordered = sorted(groups.keys(), key=lambda k: (-len(groups[k]), k))
    out = []
    for k in ordered:
        items = groups[k]
        rng.shuffle(items)
        out.append(items)
    return out


def order_random(labels, seed=42):
    out = list(labels)
    random.Random(seed).shuffle(out)
    return out


def order_round_robin(labels, seed=42):
    queues = deque(deque(g) for g in _grouped_shuffled(labels, seed))
    out = []
    while queues:
        q = queues.popleft()
        out.append(q.popleft())
        if q:
            queues.append(q)
    return out


def order_even_spacing(labels, seed=42):
    decorated = []
    for gi, items in enumerate(_grouped_shuffled(labels, seed)):
        n = len(items)
        for rank, x in enumerate(items):
            decorated.append(((rank + 0.5) / n, gi, x))
    decorated.sort(key=lambda t: (t[0], t[1]))
    return [x for _, _, x in decorated]


STRATEGIES = {
    "random": order_random,
    "round_robin": order_round_robin,
    "even_spacing": order_even_spacing,
}


# --------------------------------------------------------------------------
# Synthetic netlist fault-count distributions (~N prompts each)
# --------------------------------------------------------------------------
def make_labels_from_counts(counts):
    labels = []
    for nid, c in enumerate(counts):
        labels.extend([nid] * c)
    return labels


def dist_uniform(N=10000, n_netlists=200):
    base = N // n_netlists
    counts = [base] * n_netlists
    for i in range(N - base * n_netlists):
        counts[i] += 1
    return counts


def dist_zipf(N=10000, n_netlists=150, s=1.0):
    raw = [1.0 / (i + 1) ** s for i in range(n_netlists)]
    tot = sum(raw)
    counts = [max(1, round(N * r / tot)) for r in raw]
    return counts


def dist_few_netlists(N=10000, n_netlists=8):
    base = N // n_netlists
    counts = [base] * n_netlists
    counts[0] += N - base * n_netlists
    return counts


REGIMES = {
    "uniform(200 netlists)":      lambda: dist_uniform(10000, 200),
    "moderate_zipf(s=1.0,~150)":  lambda: dist_zipf(10000, 150, 1.0),
    "heavy_zipf(s=1.4,~120)":     lambda: dist_zipf(10000, 120, 1.4),
    "few_netlists(8)":            lambda: dist_few_netlists(10000, 8),
}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def window_unique_stats(ordered, W):
    """Distinct labels per non-overlapping window of length W."""
    uniques = [
        len(set(ordered[i:i + W]))
        for i in range(0, len(ordered) - W + 1, W)
    ]
    if not uniques:
        uniques = [len(set(ordered))]
    return uniques


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=str, default=None,
                    help="JSON file: list of fault-counts per netlist (real distribution).")
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--effective_batches", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    G = args.num_generations

    if args.real:
        counts = json.load(open(args.real))
        regimes = {f"REAL({len(counts)} netlists, N={sum(counts)})": (lambda c=counts: c)}
    else:
        regimes = REGIMES

    for name, builder in regimes.items():
        counts = builder()
        labels = make_labels_from_counts(counts)
        N = len(labels)
        n_netlists = len(counts)
        top_share = max(counts) / N * 100
        print("\n" + "=" * 96)
        print(f"REGIME: {name}   | prompts={N}  netlists={n_netlists}  "
              f"largest={max(counts)} ({top_share:.0f}%)  num_generations={G}")
        print("=" * 96)
        header = (f"{'eff_batch(seq)':>14} {'W=prompts/step':>15} {'cap':>5} | "
                  f"{'random mean(min)':>18} {'round_robin mean(min)':>22} "
                  f"{'even_spacing mean(min)':>23}")
        print(header)
        print("-" * len(header))
        for eff in args.effective_batches:
            W = max(1, eff // G)
            cap = min(W, n_netlists)  # max distinct netlists achievable in a window
            cells = []
            for strat in ("random", "round_robin", "even_spacing"):
                ordered = STRATEGIES[strat](labels, seed=args.seed)
                u = window_unique_stats(ordered, W)
                cells.append(f"{statistics.mean(u):.1f}({min(u)})")
            print(f"{eff:>14} {W:>15} {cap:>5} | "
                  f"{cells[0]:>18} {cells[1]:>22} {cells[2]:>23}")
    print("\nLegend: 'mean(min)' = mean and minimum distinct netlists across "
          "non-overlapping windows. 'cap' = ceiling = min(W, #netlists).")


if __name__ == "__main__":
    main()
