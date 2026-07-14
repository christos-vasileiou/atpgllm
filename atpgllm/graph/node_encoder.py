"""
Node feature encoder using Attribute Decomposition.

Instead of learning one monolithic embedding per cell type (~160 in ASAP7),
we decompose each gate into 8 categorical/binary attributes and learn a
small embedding per attribute value.  The concatenation is fused through
an MLP to produce the initial node embedding for the GNN.

Additionally, 4 continuous **structural features** (forward depth, backward
depth, in-degree, out-degree) are concatenated before the MLP.  These
replace RWPE / Laplacian PE — which are degenerate on DAGs — with
task-native features that directly encode SCOAP-like controllability and
observability information.

Total learnable embedding parameters: ~52 vectors × 16 dims = 832 params.
Compare with monolithic: 160 × 256 = 40 960 params — and the decomposed
version generalises better to unseen gate combinations.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn, Tensor

from .gate_features import (
    ATTRIBUTE_NAMES,
    NUM_ATTRIBUTES,
    GateAttributeVocab,
)
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
        per_attr_dim: int = 16,
        structural_dim: int = NUM_STRUCTURAL_FEATURES,
        hidden_dim: int = 128,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.per_attr_dim = per_attr_dim
        self.structural_dim = structural_dim

        ordered_sizes = [vocab_sizes[name] for name in ATTRIBUTE_NAMES]
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_classes, per_attr_dim)
            for n_classes in ordered_sizes
        ])

        # Fusion MLP: concat(8 × per_attr_dim, structural_dim) → out_dim
        cat_dim = NUM_ATTRIBUTES * per_attr_dim + structural_dim
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
        parts = [emb(gate_attrs[:, i]) for i, emb in enumerate(self.embeddings)]
        parts.append(structural_feats)
        return self.mlp(torch.cat(parts, dim=-1))

    @classmethod
    def from_vocab(
        cls,
        vocab: GateAttributeVocab,
        **kwargs,
    ) -> "AttributeDecompositionEncoder":
        """Convenience constructor from a ``GateAttributeVocab``."""
        return cls(vocab_sizes=vocab.vocab_sizes, **kwargs)
