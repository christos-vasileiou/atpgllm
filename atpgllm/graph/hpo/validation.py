"""Deterministic semantic validation for Stage-A and Stage-B pruning."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import torch
from torch import nn

from .metrics import (
    binary_average_precision,
    binary_iou,
    graph_pretrain_score,
    graph_text_alignment_score,
    mean_defined,
    retrieval_metrics,
)


@torch.no_grad()
def evaluate_graph_pretraining(
    model,
    batches: Iterable[Any],
    *,
    device: torch.device,
    max_batches: int,
) -> dict[str, float | int | None]:
    was_training = model.training
    model.eval()
    task_fields = {
        "propagation": ("propagation_mask", "has_propagation_labels"),
        "backtrack": ("backtrack_mask", "has_backtrack_labels"),
        "discrepancy": ("discrepancy_mask", "has_discrepancy_labels"),
    }
    average_precisions: dict[str, list[float | None]] = defaultdict(list)
    ious: dict[str, list[float]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)

    for batch_number, graph in enumerate(batches):
        if batch_number >= max_batches:
            break
        graph = graph.to(device)
        outputs = model(graph)
        graph_count = int(graph.batch.max().item()) + 1
        for task, (label_field, present_field) in task_fields.items():
            probabilities = outputs.logits[task].sigmoid()
            labels = getattr(graph, label_field)
            present = getattr(graph, present_field).reshape(-1).bool()
            for graph_index in range(graph_count):
                if not bool(present[graph_index]):
                    skipped[task] += 1
                    continue
                node_mask = graph.batch == graph_index
                ap = binary_average_precision(
                    probabilities[node_mask].cpu(),
                    labels[node_mask].cpu(),
                )
                if ap is None:
                    skipped[task] += 1
                    continue
                average_precisions[task].append(ap)
                ious[task].append(
                    binary_iou(
                        probabilities[node_mask].cpu(),
                        labels[node_mask].cpu(),
                    )
                )

    metrics: dict[str, float | int | None] = {}
    for task in task_fields:
        metrics[f"{task}_ap"] = mean_defined(average_precisions[task])
        metrics[f"{task}_iou"] = mean_defined(ious[task])
        metrics[f"{task}_skipped_graphs"] = skipped[task]
    metrics["semantic_score"] = graph_pretrain_score(metrics)
    if was_training:
        model.train()
    return metrics


@torch.no_grad()
def evaluate_graph_text_alignment(
    trainer,
    batches: Iterable[dict[str, Any]],
    *,
    max_batches: int,
) -> dict[str, float | None]:
    model = trainer.model
    was_training = model.training
    model.eval()
    trainer.gtm_loss.eval()
    trainer.gtg_loss.eval()
    graph_representations = []
    text_representations = []
    matching_scores = []
    matching_labels = []
    gtg_losses = []

    for batch_number, batch in enumerate(batches):
        if batch_number >= max_batches:
            break
        prepared = trainer._prepare_batch(batch)
        outputs = model(
            prepared["g"],
            prepared["input_ids"],
            prepared["attention_mask"],
        )
        graph_q = model.projected_graph_repr_per_query(outputs)
        text = model.projected_text_repr(outputs)
        graph_representations.append(
            nn.functional.normalize(graph_q, dim=-1).float().cpu()
        )
        text_representations.append(
            nn.functional.normalize(text, dim=-1).float().cpu()
        )

        if graph_q.size(0) > 1:
            positive = trainer.gtm_loss.head(graph_q, text)
            negative = trainer.gtm_loss.head(graph_q, torch.roll(text, 1, 0))
            matching_scores.extend(
                torch.cat([positive, negative]).float().cpu().tolist()
            )
            matching_labels.extend(
                [1] * positive.numel() + [0] * negative.numel()
            )

        gtg_loss = trainer.gtg_loss(
            model,
            outputs,
            prepared["decoder_input_ids"],
            prepared["decoder_attention_mask"],
            prepared["decoder_labels"],
        )
        gtg_losses.append(float(gtg_loss.float().cpu()))

    if not graph_representations:
        raise ValueError("Stage-B validation produced no batches.")
    graph_q = torch.cat(graph_representations, dim=0)
    text = torch.cat(text_representations, dim=0)
    similarity = torch.einsum("iqd,jd->iqj", graph_q, text).max(dim=1).values
    metrics: dict[str, float | None] = retrieval_metrics(similarity)
    metrics["matching_ap"] = binary_average_precision(
        matching_scores,
        matching_labels,
    ) if matching_scores else None
    metrics["gtg_loss"] = sum(gtg_losses) / len(gtg_losses)
    metrics["semantic_score"] = graph_text_alignment_score(metrics)
    if was_training:
        model.train()
    return metrics
