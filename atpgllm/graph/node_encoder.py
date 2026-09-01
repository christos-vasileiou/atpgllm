"""
Node feature encoder using Attribute Decomposition.

Instead of learning one monolithic embedding per cell type, we decompose each
gate into a versioned ordered set of categorical/binary attributes and learn a
small embedding per attribute value. The concatenation is fused through an MLP
to produce the initial node embedding for the GNN.

Additionally, 4 continuous **structural features** (forward depth, backward
depth, in-degree, out-degree) are concatenated before the MLP.  These
replace RWPE / Laplacian PE — which are degenerate on DAGs — with
task-native features that directly encode SCOAP-like controllability and
observability information. Five inference-safe target-fault features identify
the fault site's driver/sink neighborhood and stuck-at polarity.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import nn, Tensor

from .fault_context import NUM_FAULT_FEATURES
from .gate_features import GateAttributeVocab
from .netlist_parser import NUM_STRUCTURAL_FEATURES


class AttributeDecompositionEncoder(nn.Module):
    """Encode gate attributes + structural features → initial node embedding.

    Parameters
    ----------
    vocab_sizes : dict
        ``{attribute_name: num_classes}`` from
        :attr:`GateAttributeVocab.vocab_sizes`.
    per_attr_dim : int
        Embedding dimensionality for each categorical attribute.
    structural_dim : int
        Number of continuous structural features per node (default 4).
    hidden_dim : int
        Width of the fusion MLP hidden layer.
    out_dim : int
        Final node embedding dimensionality (must match GIN hidden_dim).
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        attribute_names: Optional[Sequence[str]] = None,
        per_attr_dim: int = 16,
        structural_dim: int = NUM_STRUCTURAL_FEATURES,
        context_dim: int = NUM_FAULT_FEATURES,
        hidden_dim: int = 128,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.attribute_names = tuple(attribute_names or vocab_sizes.keys())
        self.per_attr_dim = per_attr_dim
        self.structural_dim = structural_dim
        self.context_dim = context_dim

        ordered_sizes = [vocab_sizes[name] for name in self.attribute_names]
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_classes, per_attr_dim)
            for n_classes in ordered_sizes
        ])

        cat_dim = len(self.attribute_names) * per_attr_dim + structural_dim + context_dim
        self.mlp = nn.Sequential(
            nn.Linear(cat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        gate_attrs: Tensor,
        structural_feats: Tensor,
        fault_feats: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        gate_attrs : Tensor[N, NUM_ATTRIBUTES]  (long)
            Per-node attribute indices (from ``GateAttributeVocab.encode``).
            The columns correspond to ``ATTRIBUTE_NAMES`` in order.
        structural_feats : Tensor[N, NUM_STRUCTURAL_FEATURES]  (float)
            Per-node continuous features, one column per name in
            ``STRUCTURAL_FEATURE_NAMES`` (already normalised — see
            ``compute_structural_features``).

        Returns
        -------
        Tensor[N, out_dim]
            Initial node embeddings ready for the GNN.
        """
        if gate_attrs.size(1) != len(self.embeddings):
            raise ValueError(
                f"Expected {len(self.embeddings)} gate-attribute columns "
                f"{self.attribute_names}, got {gate_attrs.size(1)}."
            )
        if structural_feats.size(1) != self.structural_dim:
            raise ValueError(
                f"Expected {self.structural_dim} structural features, "
                f"got {structural_feats.size(1)}."
            )
        if fault_feats is None:
            fault_feats = structural_feats.new_zeros(
                (gate_attrs.size(0), self.context_dim)
            )
        if fault_feats.size(1) != self.context_dim:
            raise ValueError(
                f"Expected {self.context_dim} fault-context features, "
                f"got {fault_feats.size(1)}."
            )

        parts = [emb(gate_attrs[:, i]) for i, emb in enumerate(self.embeddings)]
        parts.append(structural_feats)
        parts.append(fault_feats)
        return self.mlp(torch.cat(parts, dim=-1))

    @classmethod
    def from_vocab(
        cls,
        vocab: GateAttributeVocab,
        **kwargs,
    ) -> "AttributeDecompositionEncoder":
        """Convenience constructor from a ``GateAttributeVocab``."""
        return cls(
            vocab_sizes=vocab.vocab_sizes,
            attribute_names=vocab.attribute_names,
            **kwargs,
        )
