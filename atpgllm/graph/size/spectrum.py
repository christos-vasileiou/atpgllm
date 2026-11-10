"""S4 circuit spectrum: a fixed-size gist vector for any netlist size.

Like a k-mer spectrum of a genome, it counts small local functions as
shares, so the vector has the same meaning at 30 gates and at 100k:

- the NPN class of each net's 4-input cuts (222 classes), as a histogram;
- relative-level, log-fanout, signal-probability and observability profiles.

Nets are sampled when the netlist is large (``max_nets``), which is valid
because every entry is a density. Cost: O(min(V, max_nets) · C · cone).
"""

from __future__ import annotations

import itertools
import random
from functools import lru_cache
from typing import Dict, FrozenSet, List, Sequence, Tuple

import numpy as np

from .netgraph import NetGraph

K = 4
_LEAF_TT = (0xAAAA, 0xCCCC, 0xF0F0, 0xFF00)
PROFILE_BINS = {"level": 16, "fanout": 8, "sp": 10, "obs": 10}


@lru_cache(maxsize=1)
def npn_classes() -> Tuple[np.ndarray, int]:
    """Class index of every 4-input truth table under input permutation,
    input negation and output negation, and the number of classes (222)."""
    tables = np.arange(1 << 16, dtype=np.uint32)
    best = np.full(1 << 16, 0xFFFF, dtype=np.uint32)
    for perm in itertools.permutations(range(K)):
        for neg in range(1 << K):
            index = []
            for i in range(16):
                bits = [((i >> perm[j]) & 1) ^ ((neg >> j) & 1) for j in range(K)]
                index.append(sum(b << j for j, b in enumerate(bits)))
            new = np.zeros_like(tables)
            for i, src in enumerate(index):
                new |= ((tables >> src) & 1) << i
            best = np.minimum(best, np.minimum(new, new ^ 0xFFFF))
    canon, cls = np.unique(best, return_inverse=True)
    return cls.astype(np.int32), int(len(canon))


def _cuts(g: NetGraph, nets: Sequence[str], per_net: int) -> Dict[str, List[FrozenSet[str]]]:
    cuts: Dict[str, List[FrozenSet[str]]] = {}
    for net in g.order:
        own = frozenset([net])
        if net not in g.gates:
            cuts[net] = [own]
            continue
        ins = [i for i in g.gates[net][1] if not i.isdigit()]
        merged = {own}
        for combo in itertools.product(*(cuts.get(i, [frozenset([i])]) for i in ins)):
            leaves = frozenset().union(*combo) if combo else frozenset()
            if len(leaves) <= K:
                merged.add(leaves)
            if len(merged) > 4 * per_net:
                break
        cuts[net] = sorted(merged, key=len)[:per_net]
    return cuts


def _truth_table(g: NetGraph, net: str, leaves: Sequence[str]) -> int:
    memo = {leaf: _LEAF_TT[j] for j, leaf in enumerate(leaves)}

    def tt(n: str) -> int:
        if n in memo:
            return memo[n]
        if n.isdigit():
            return 0xFFFF if int(n) else 0
        table, ins = g.gates[n]
        args = [tt(i) for i in ins]
        out = 0
        for p in range(16):
            idx = 0
            for j, a in enumerate(args):
                idx |= ((a >> p) & 1) << j
            out |= (table[idx] & 1) << p
        memo[n] = out
        return out

    return tt(net)


def circuit_spectrum(g: NetGraph, *, max_nets: int = 20_000, cuts_per_net: int = 8, seed: int = 0) -> np.ndarray:
    """Fixed-length vector: 222 NPN cut shares, then the four profiles."""
    cls, n_cls = npn_classes()
    gate_nets = [n for n in g.order if n in g.gates]
    rng = random.Random(seed)
    sample = gate_nets if len(gate_nets) <= max_nets else rng.sample(gate_nets, max_nets)
    cuts = _cuts(g, sample, cuts_per_net)
    hist = np.zeros(n_cls)
    for net in sample:
        for leaves in cuts.get(net, []):
            if leaves == frozenset([net]):
                continue
            hist[cls[_truth_table(g, net, sorted(leaves))]] += 1
    hist = hist / max(1.0, hist.sum())
    depth = max(1, g.depth)
    nets = [n for n in g.order if not n.isdigit()]
    level = np.histogram([g.level[n] / depth for n in nets], bins=PROFILE_BINS["level"], range=(0, 1))[0]
    fanout = np.histogram([np.log2(1 + len(g.fanout.get(n, ()))) for n in nets], bins=PROFILE_BINS["fanout"],
                          range=(0, 8))[0]
    sp = np.histogram([g.sp.get(n, 0.5) for n in nets], bins=PROFILE_BINS["sp"], range=(0, 1))[0]
    obs = np.histogram([g.obs.get(n, 0.0) for n in nets], bins=PROFILE_BINS["obs"], range=(0, 1))[0]
    profiles = [p / max(1, p.sum()) for p in (level, fanout, sp, obs)]
    return np.concatenate([hist, *profiles]).astype(np.float32)


def spectrum_dim() -> int:
    return npn_classes()[1] + sum(PROFILE_BINS.values())
