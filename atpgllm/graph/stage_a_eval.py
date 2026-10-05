"""Stage-A validation: stratified by circuit size, read against a structural rule.

A single average over validation graphs is dominated by whichever circuits are
most numerous; with a per-design cap that is the many small circuits, where the
tasks are easiest. This module

- builds a validation set with up to N graphs per circuit-size bucket
  (optionally per size bucket and difficulty label),
- reports metrics per size bucket and per difficulty label,
- scores the same graphs with a label-free structural rule (downstream cone of
  the fault for propagation/discrepancy, its fan-in for backtrack), so each
  head's average precision can be read as a lift over structure alone, and
- picks checkpoints by a size-balanced score (mean over filled buckets).
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from .dataset import collate_graph_pretrain_batch, has_pretraining_labels, record_to_pyg
from .gate_features import GateAttributeVocab
from .hpo.manifests import design_hash
from .hpo.metrics import binary_average_precision, graph_pretrain_score, mean_defined
from .hpo.validation import evaluate_graph_pretraining
from .pretrain import GraphPretrainingModel


SIZE_BUCKETS: Tuple[Tuple[str, float], ...] = (
    ("<=50 gates", 50),
    ("51-200 gates", 200),
    (">200 gates", math.inf),
)
# A bucket with fewer graphs than this is reported but left out of the
# balanced score, which would otherwise swing on a handful of circuits.
MIN_BUCKET_GRAPHS = 20


def size_bucket(graph: Data) -> str:
    gates = int(graph.gate_node_mask.sum())
    return next(name for name, limit in SIZE_BUCKETS if gates <= limit)


@dataclass
class ValidationItem:
    graph: Data
    size: str
    difficulty: str


def build_validation_set(
    records: Iterable[Mapping[str, Any]],
    gate_funcs: Dict[str, Dict[str, str]],
    vocab: GateAttributeVocab,
    *,
    per_bucket: int,
    per_design: int,
    scan_limit: int,
    per_difficulty: Optional[int] = None,
) -> Tuple[List[ValidationItem], collections.Counter, int]:
    """Collect up to ``per_bucket`` labelled graphs per size bucket, at most
    ``per_design`` per circuit, reading at most ``scan_limit`` records.

    With ``per_difficulty``, the caps are per difficulty label instead: up to
    ``per_difficulty`` graphs per (size bucket, difficulty) and ``per_design``
    per (circuit, difficulty), so rare hard faults are not crowded out by the
    easy majority.
    """
    rejections: collections.Counter = collections.Counter()
    per_design_seen: collections.Counter = collections.Counter()
    counts: collections.Counter = collections.Counter()
    items: List[ValidationItem] = []
    scanned = 0
    for record in records:
        if scanned >= scan_limit or (
            per_difficulty is None
            and all(counts[name] >= per_bucket for name, _ in SIZE_BUCKETS)
        ):
            break
        scanned += 1
        difficulty = str(record.get("difficulty") or "unknown")
        key = design_hash(record)
        if per_difficulty is not None:
            key = (key, difficulty)
        if per_design_seen[key] >= per_design:
            continue
        graph = record_to_pyg(dict(record), gate_funcs, vocab, rejections=rejections)
        if graph is None or not has_pretraining_labels(graph):
            continue
        bucket = size_bucket(graph)
        cell = bucket if per_difficulty is None else (bucket, difficulty)
        if counts[cell] >= (per_bucket if per_difficulty is None else per_difficulty):
            continue
        per_design_seen[key] += 1
        counts[cell] += 1
        items.append(ValidationItem(graph=graph, size=bucket, difficulty=difficulty))
    return items, rejections, scanned


def structural_rule_scores(graph: Data) -> Dict[str, Tensor]:
    """Label-free node scores from the fault site and topology alone.

    Propagation and discrepancy: 1 / (1 + distance) inside the fault net's
    downstream cone, else 0. Backtrack: 1 / (1 + distance) upstream of that
    cone (the cone itself scores 1), else 0.
    """
    n = graph.num_nodes
    src, dst = graph.edge_index.tolist()
    succ: List[List[int]] = [[] for _ in range(n)]
    pred: List[List[int]] = [[] for _ in range(n)]
    for s, d in zip(src, dst):
        succ[s].append(d)
        pred[d].append(s)
    start = {s for s, flag in zip(src, graph.fault_edge.tolist()) if flag}

    def bfs(seeds, neighbours) -> Dict[int, int]:
        dist = {v: 0 for v in seeds}
        queue = collections.deque(seeds)
        while queue:
            v = queue.popleft()
            for w in neighbours[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
        return dist

    cone = bfs(start, succ)
    upstream = bfs(set(cone), pred)

    def scores(dist: Dict[int, int]) -> Tensor:
        out = torch.zeros(n)
        for v, d in dist.items():
            out[v] = 1.0 / (1 + d)
        return out

    downstream = scores(cone)
    return {
        "propagation": downstream,
        "discrepancy": downstream,
        "backtrack": scores(upstream),
    }


def _rule_metrics(items: List[ValidationItem]) -> Dict[str, Optional[float]]:
    average_precisions: Dict[str, List[Optional[float]]] = collections.defaultdict(list)
    for item in items:
        batch = Batch.from_data_list([item.graph])
        rule = structural_rule_scores(item.graph)
        for task, (label_field, present_field) in GraphPretrainingModel.LABEL_FIELDS.items():
            if not bool(getattr(item.graph, present_field).item()):
                continue
            mask = GraphPretrainingModel.scored_nodes(task, batch)
            ap = binary_average_precision(
                rule[task][mask], getattr(item.graph, label_field)[mask]
            )
            if ap is not None:
                average_precisions[task].append(ap)
    metrics: Dict[str, Optional[float]] = {
        f"{task}_ap": mean_defined(average_precisions[task])
        for task in GraphPretrainingModel.LABEL_FIELDS
    }
    metrics["semantic_score"] = (
        graph_pretrain_score(metrics)
        if any(value is not None for value in metrics.values())
        else None
    )
    return metrics


def _model_metrics(model, items, device, batch_size) -> Dict[str, Any]:
    loader = DataLoader(
        [item.graph for item in items],
        batch_size=batch_size,
        collate_fn=collate_graph_pretrain_batch,
    )
    return evaluate_graph_pretraining(
        model, loader, device=device, max_batches=len(loader)
    )


def _groups(items: List[ValidationItem]) -> Dict[str, Dict[str, List[ValidationItem]]]:
    by_size: Dict[str, List[ValidationItem]] = collections.defaultdict(list)
    by_difficulty: Dict[str, List[ValidationItem]] = collections.defaultdict(list)
    for item in items:
        by_size[item.size].append(item)
        by_difficulty[item.difficulty].append(item)
    groups = {"by_size": dict(by_size)}
    if set(by_difficulty) != {"unknown"}:
        groups["by_difficulty"] = dict(by_difficulty)
    return groups


def _balanced(by_size: Mapping[str, Mapping[str, Any]]) -> Optional[float]:
    scores = [
        group["semantic_score"]
        for group in by_size.values()
        if group["graphs"] >= MIN_BUCKET_GRAPHS and group["semantic_score"] is not None
    ]
    return sum(scores) / len(scores) if scores else None


def rule_baseline(items: List[ValidationItem]) -> Dict[str, Any]:
    """Structural-rule metrics on the validation set, grouped like the model's."""
    report: Dict[str, Any] = {"overall": _rule_metrics(items)}
    for name, groups in _groups(items).items():
        report[name] = {
            key: {**_rule_metrics(group), "graphs": len(group)}
            for key, group in groups.items()
        }
    report["score"] = _balanced(report["by_size"])
    return report


def evaluate_stage_a(
    model,
    items: List[ValidationItem],
    *,
    device: torch.device,
    batch_size: int,
    baseline: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Model metrics overall, per size bucket and per difficulty; with a
    ``baseline`` from :func:`rule_baseline`, also each head's AP lift over it."""
    report: Dict[str, Any] = {
        "overall": _model_metrics(model, items, device, batch_size)
    }
    for name, groups in _groups(items).items():
        report[name] = {
            key: {**_model_metrics(model, group, device, batch_size), "graphs": len(group)}
            for key, group in groups.items()
        }
    report["score"] = _balanced(report["by_size"])
    if baseline is not None:
        report["lift_vs_rule"] = {
            key: {
                task: (
                    None
                    if group.get(f"{task}_ap") is None
                    or baseline["by_size"][key].get(f"{task}_ap") is None
                    else group[f"{task}_ap"] - baseline["by_size"][key][f"{task}_ap"]
                )
                for task in GraphPretrainingModel.LABEL_FIELDS
            }
            for key, group in report["by_size"].items()
        }
    return report
