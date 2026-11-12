"""Construct graph/Q-Former stacks from validated alignment checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def local_pretrained(name: str) -> str:
    """``name`` itself, or its local Hub snapshot when ``HF_HUB_OFFLINE`` is on.

    With a Hub id, transformers 4.57's tokenizer loader (``_patch_mistral_regex``)
    asks the Hub API for the model's info even with ``local_files_only``, so an
    offline load by id fails (juno pilot 473238). A local snapshot path skips
    that call and loads the same files."""
    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")
    if not offline or Path(name).exists():
        return name
    from huggingface_hub import snapshot_download

    return snapshot_download(name, local_files_only=True)

from atpgllm.graph.checkpoints import (
    STAGE_GRAPH_TEXT_ALIGNMENT,
    load_stage_checkpoint,
)
from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.models_stage1 import GraphQFormer, NetlistGraphEncoder

RETEST_FORMAT = "bridges_retest_v1"


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
    """Load only inference-time graph modules, without the Stage-1 text encoder.

    Also accepts a BRIDGES re-test checkpoint (``train_retest``), whose
    Q-Former is the fault-seeded :class:`RoleQueryFormer`."""
    import torch

    peek = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(peek, dict) and peek.get("format") == RETEST_FORMAT:
        return _load_retest_stack(peek, vocab)
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


def _load_retest_stack(payload: dict[str, Any], vocab: GateAttributeVocab):
    """Encoder and role Q-Former from a ``train_retest`` checkpoint."""
    from atpgllm.graph.retest_qformer import RoleQueryFormer

    saved = GateAttributeVocab.from_dict(payload["vocab"])
    if saved.fingerprint == vocab.with_state_cells().fingerprint:
        # Trained with state cells: graphs must be encoded with them too. The
        # caller takes this vocabulary from the returned ``vocab`` entry.
        vocab = saved
    elif saved.fingerprint != vocab.fingerprint:
        raise ValueError("re-test checkpoint was trained with another gate vocabulary")
    a = payload["args"]
    graph_encoder = NetlistGraphEncoder(vocab=vocab, node_dim=a["node_dim"], gin_hidden_dim=a["node_dim"],
                                        gin_num_layers=a["gin_layers"])
    q_former = RoleQueryFormer(d_node=graph_encoder.out_dim, hidden_dim=a["q_hidden"],
                               slots_per_role=a["slots"], num_layers=a["q_layers"])
    state = payload["model"]
    graph_encoder.load_state_dict(_substate(state, "encoder"), strict=True)
    q_former.load_state_dict(_substate(state, "q_former"), strict=True)
    architecture = {
        "format": RETEST_FORMAT,
        "graph_encoder": dict(graph_encoder.config),
        "qformer": "role_query_former",
        "qformer_hidden_dim": a["q_hidden"],
        "num_queries": q_former.num_queries,
        "slots_per_role": a["slots"],
        "qformer_layers": a["q_layers"],
        "train_step": payload.get("step"),
    }
    return graph_encoder, q_former, {"architecture": architecture, "stage": "bridges_retest", "vocab": vocab}
