"""Construct graph/Q-Former stacks from validated alignment checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atpgllm.graph.checkpoints import (
    STAGE_GRAPH_TEXT_ALIGNMENT,
    load_stage_checkpoint,
)
from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.models_stage1 import GraphQFormer, NetlistGraphEncoder


def _substate(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    dotted = prefix + "."
    return {
        key[len(dotted):]: value
        for key, value in state.items()
        if key.startswith(dotted)
    }


def load_aligned_graph_stack(
    checkpoint_path: str | Path,
    vocab: GateAttributeVocab,
) -> tuple[NetlistGraphEncoder, GraphQFormer, dict[str, Any]]:
    """Load only inference-time graph modules, without the Stage-1 text encoder."""
    checkpoint = load_stage_checkpoint(
        checkpoint_path,
        expected_stages=[STAGE_GRAPH_TEXT_ALIGNMENT],
        vocab=vocab,
    )
    architecture = checkpoint["architecture"]
    graph_config = dict(architecture["graph_encoder"])
    graph_encoder = NetlistGraphEncoder(vocab=vocab, **graph_config)
    q_former = GraphQFormer(
        d_node=graph_encoder.out_dim,
        hidden_dim=architecture["qformer_hidden_dim"],
        num_queries=architecture["num_queries"],
        num_layers=architecture["qformer_layers"],
        num_heads=architecture["qformer_heads"],
        ffn_dim=(
            architecture.get("qformer_ffn_multiplier", 4)
            * architecture["qformer_hidden_dim"]
        ),
        cross_attn_every_n=architecture["qformer_cross_every_n"],
        dropout=architecture.get("qformer_dropout", 0.0),
    )
    model_state = checkpoint["states"]["model"]
    graph_encoder.load_state_dict(
        _substate(model_state, "graph_encoder"),
        strict=True,
    )
    q_former.load_state_dict(
        _substate(model_state, "q_former"),
        strict=True,
    )
    return graph_encoder, q_former, checkpoint
