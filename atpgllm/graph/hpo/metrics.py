"""Semantic validation metrics for graph and graph-text stages."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor


def _as_float_tensor(value: Tensor | Sequence[float]) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float64).reshape(-1)


def binary_average_precision(
    scores: Tensor | Sequence[float],
    labels: Tensor | Sequence[int],
) -> float | None:
    """Average precision; return ``None`` when no positive label exists."""
    score = _as_float_tensor(scores)
    target = torch.as_tensor(labels, dtype=torch.bool).reshape(-1)
    if score.numel() != target.numel() or score.numel() == 0:
        raise ValueError("scores and labels must be non-empty and equally sized")
    positives = int(target.sum())
    if positives == 0:
        return None
    order = torch.argsort(score, descending=True, stable=True)
    ranked = target[order].to(torch.float64)
    precision = ranked.cumsum(0) / torch.arange(
        1,
        ranked.numel() + 1,
        dtype=torch.float64,
    )
    return float(precision[ranked.bool()].mean())


def binary_roc_auc(
    scores: Tensor | Sequence[float],
    labels: Tensor | Sequence[int],
) -> float | None:
    """Pairwise AUROC with tie handling; undefined one-class cases return None."""
    score = _as_float_tensor(scores)
    target = torch.as_tensor(labels, dtype=torch.bool).reshape(-1)
    if score.numel() != target.numel() or score.numel() == 0:
        raise ValueError("scores and labels must be non-empty and equally sized")
    positive = score[target]
    negative = score[~target]
    if positive.numel() == 0 or negative.numel() == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (comparisons.gt(0).to(torch.float64) + 0.5 * comparisons.eq(0)).mean()
    )


def binary_iou(
    scores: Tensor | Sequence[float],
    labels: Tensor | Sequence[int],
    threshold: float = 0.5,
) -> float:
    prediction = _as_float_tensor(scores) >= threshold
    target = torch.as_tensor(labels, dtype=torch.bool).reshape(-1)
    if prediction.numel() != target.numel() or prediction.numel() == 0:
        raise ValueError("scores and labels must be non-empty and equally sized")
    union = (prediction | target).sum()
    if union == 0:
        return 1.0
    return float((prediction & target).sum() / union)


def mean_defined(values: Iterable[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def graph_pretrain_score(metrics: Mapping[str, float | None]) -> float:
    weights = {
        "propagation_ap": 0.45,
        "discrepancy_ap": 0.35,
        "backtrack_ap": 0.20,
    }
    available = {
        name: weight
        for name, weight in weights.items()
        if metrics.get(name) is not None
    }
    if not available:
        raise ValueError("No defined Stage-A semantic metrics.")
    normalizer = sum(available.values())
    return sum(
        float(metrics[name]) * weight
        for name, weight in available.items()
    ) / normalizer


def retrieval_metrics(similarity: Tensor) -> dict[str, float]:
    """Bidirectional Recall@1/5 and MRR for a square matched-pair matrix."""
    if similarity.ndim != 2 or similarity.size(0) != similarity.size(1):
        raise ValueError("similarity must be a square [N, N] matrix")
    count = similarity.size(0)
    if count == 0:
        raise ValueError("similarity matrix cannot be empty")

    def direction(matrix: Tensor) -> tuple[float, float, float]:
        order = torch.argsort(matrix, dim=1, descending=True)
        target = torch.arange(count, device=matrix.device).unsqueeze(1)
        rank = order.eq(target).to(torch.int64).argmax(dim=1) + 1
        recall1 = float(rank.le(1).to(torch.float32).mean())
        recall5 = float(rank.le(min(5, count)).to(torch.float32).mean())
        mrr = float(rank.to(torch.float32).reciprocal().mean())
        return recall1, recall5, mrr

    g1, g5, gm = direction(similarity)
    t1, t5, tm = direction(similarity.t())
    return {
        "graph_to_text_r1": g1,
        "graph_to_text_r5": g5,
        "graph_to_text_mrr": gm,
        "text_to_graph_r1": t1,
        "text_to_graph_r5": t5,
        "text_to_graph_mrr": tm,
        "mean_r1": 0.5 * (g1 + t1),
        "mean_r5": 0.5 * (g5 + t5),
        "mean_mrr": 0.5 * (gm + tm),
    }


def graph_text_alignment_score(metrics: Mapping[str, float | None]) -> float:
    required = ("mean_r1", "mean_mrr", "matching_ap", "gtg_loss")
    if any(metrics.get(name) is None for name in required):
        missing = [name for name in required if metrics.get(name) is None]
        raise ValueError(f"Undefined Stage-B semantic metrics: {missing}")
    gtg_quality = math.exp(-min(float(metrics["gtg_loss"]), 20.0))
    return (
        0.45 * float(metrics["mean_r1"])
        + 0.25 * float(metrics["mean_mrr"])
        + 0.20 * float(metrics["matching_ap"])
        + 0.10 * gtg_quality
    )
