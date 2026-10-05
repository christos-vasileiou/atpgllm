"""Stage-B validation set: held-out (graph, caption) pairs grouped by circuit.

v3 repeats each (circuit, fault) once per test pattern; those rows share one
graph but have different captions, so keeping two of them would make a
retrieval tie that no encoder can break. The set keeps one pair per
(circuit, fault) and a few faults per circuit, so retrieval can be read both
across circuits and among the faults of one circuit
(:func:`atpgllm.graph.hpo.metrics.grouped_retrieval_metrics`).
"""

from __future__ import annotations

import collections
from typing import Any, Dict, Iterable, List


def build_alignment_validation_set(
    examples: Iterable[Dict[str, Any]],
    *,
    per_circuit: int,
    limit: int,
) -> List[Dict[str, Any]]:
    """Up to ``limit`` examples, one per (circuit, fault), at most
    ``per_circuit`` per circuit, in the order they arrive."""
    seen = set()
    per: collections.Counter = collections.Counter()
    items: List[Dict[str, Any]] = []
    for example in examples:
        if len(items) >= limit:
            break
        key = (example["circuit"], example["fault"])
        if key in seen or per[example["circuit"]] >= per_circuit:
            continue
        seen.add(key)
        per[example["circuit"]] += 1
        items.append(example)
    return items
