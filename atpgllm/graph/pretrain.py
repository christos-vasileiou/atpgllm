"""Target-fault-conditioned DAG encoder pretraining objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn

from .gate_features import GateAttributeVocab
from .models_stage1 import NetlistGraphEncoder


@dataclass
class GraphPretrainOutputs:
    node_embs: Tensor
    graph_embs: Tensor
    logits: Dict[str, Tensor]


@dataclass
class GraphPretrainLosses:
    propagation: Tensor
    backtrack: Tensor
    discrepancy: Tensor

    @property
    def total(self) -> Tensor:
        return self.propagation + self.backtrack + self.discrepancy


class GraphPretrainingModel(nn.Module):
    """Pretrain the DAG encoder on ATPG paths and good/bad discrepancies."""

    LABEL_FIELDS = {
        "propagation": ("propagation_mask", "has_propagation_labels"),
        "backtrack": ("backtrack_mask", "has_backtrack_labels"),
        "discrepancy": ("discrepancy_mask", "has_discrepancy_labels"),
    }

    def __init__(
        self,
        vocab: GateAttributeVocab,
        *,
        node_dim: int = 256,
        gin_hidden_dim: int = 256,
        gin_num_layers: int = 6,
        per_attr_dim: int = 16,
        attr_mlp_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.graph_encoder = NetlistGraphEncoder(
            vocab=vocab,
            node_dim=node_dim,
            gin_hidden_dim=gin_hidden_dim,
            gin_num_layers=gin_num_layers,
            per_attr_dim=per_attr_dim,
            attr_mlp_hidden=attr_mlp_hidden,
            dropout=dropout,
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(self.graph_encoder.out_dim, 1)
            for name in self.LABEL_FIELDS
        })
        self.architecture_config = {
            "graph_encoder": dict(self.graph_encoder.config),
            "pretraining_heads": list(self.LABEL_FIELDS),
        }

    def forward(self, data) -> GraphPretrainOutputs:
        encoded = self.graph_encoder(data)
        node_embs = encoded["node_embs"]
        return GraphPretrainOutputs(
            node_embs=node_embs,
            graph_embs=encoded["graph_embs"],
            logits={
                name: head(node_embs).squeeze(-1)
                for name, head in self.heads.items()
            },
        )

    @staticmethod
    def _node_loss(
        logits: Tensor,
        labels: Tensor,
        graph_has_labels: Tensor,
        batch_index: Tensor,
    ) -> Tensor:
        valid = graph_has_labels.reshape(-1).bool()[batch_index]
        if not valid.any():
            return logits.sum() * 0.0
        selected_logits = logits[valid]
        selected_labels = labels[valid].to(selected_logits.dtype)
        positives = selected_labels.sum()
        negatives = selected_labels.numel() - positives
        pos_weight = (
            (negatives / positives.clamp(min=1.0)).clamp(min=1.0, max=100.0)
            if positives.item() > 0
            else selected_logits.new_tensor(1.0)
        )
        return nn.functional.binary_cross_entropy_with_logits(
            selected_logits,
            selected_labels,
            pos_weight=pos_weight,
        )

    def compute_losses(
        self,
        data,
        outputs: GraphPretrainOutputs | None = None,
    ) -> GraphPretrainLosses:
        outputs = outputs or self(data)
        losses = {}
        for name, (label_field, present_field) in self.LABEL_FIELDS.items():
            losses[name] = self._node_loss(
                outputs.logits[name],
                getattr(data, label_field),
                getattr(data, present_field),
                data.batch,
            )
        return GraphPretrainLosses(**losses)
