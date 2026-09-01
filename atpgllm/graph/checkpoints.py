"""Versioned checkpoint contracts for graph-to-LM training sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch

from .gate_features import GateAttributeVocab


CHECKPOINT_FORMAT = "atpgllm.graph-stack"
CHECKPOINT_VERSION = 1

STAGE_GRAPH_PRETRAIN = "graph_pretrain"
STAGE_GRAPH_TEXT_ALIGNMENT = "graph_text_alignment"
STAGE_MULTIMODAL_SFT = "multimodal_sft"
STAGE_MULTIMODAL_GRPO = "multimodal_grpo"

_ALLOWED_PARENTS = {
    STAGE_GRAPH_PRETRAIN: {None, STAGE_GRAPH_PRETRAIN},
    STAGE_GRAPH_TEXT_ALIGNMENT: {
        STAGE_GRAPH_PRETRAIN,
        STAGE_GRAPH_TEXT_ALIGNMENT,
    },
    STAGE_MULTIMODAL_SFT: {
        STAGE_GRAPH_TEXT_ALIGNMENT,
        STAGE_MULTIMODAL_SFT,
    },
    STAGE_MULTIMODAL_GRPO: {
        STAGE_MULTIMODAL_SFT,
        STAGE_MULTIMODAL_GRPO,
    },
}


def validate_stage_transition(stage: str, parent_stage: Optional[str]) -> None:
    if stage not in _ALLOWED_PARENTS:
        raise ValueError(f"Unknown checkpoint stage {stage!r}.")
    if parent_stage not in _ALLOWED_PARENTS[stage]:
        raise ValueError(
            f"Invalid checkpoint transition {parent_stage!r} -> {stage!r}; "
            f"allowed parents are {sorted(x for x in _ALLOWED_PARENTS[stage] if x)}."
        )


def _assert_subset(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    prefix: str = "architecture",
) -> None:
    for key, value in expected.items():
        location = f"{prefix}.{key}"
        if key not in actual:
            raise ValueError(f"Checkpoint is missing {location}.")
        if isinstance(value, Mapping):
            if not isinstance(actual[key], Mapping):
                raise ValueError(f"Checkpoint field {location} is not a mapping.")
            _assert_subset(value, actual[key], location)
        elif actual[key] != value:
            raise ValueError(
                f"Checkpoint mismatch at {location}: "
                f"checkpoint={actual[key]!r}, runtime={value!r}."
            )


def save_stage_checkpoint(
    path: str | Path,
    *,
    stage: str,
    vocab: GateAttributeVocab,
    architecture: Mapping[str, Any],
    states: Mapping[str, Any],
    step: int,
    parent_stage: Optional[str],
    optimizer_state: Optional[Mapping[str, Any]] = None,
    session: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically save one session boundary with its compatibility manifest."""
    validate_stage_transition(stage, parent_stage)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "stage": stage,
        "parent_stage": parent_stage,
        "step": int(step),
        "vocab": vocab.to_dict(),
        "vocab_fingerprint": vocab.fingerprint,
        "architecture": dict(architecture),
        "states": dict(states),
        "optimizer": optimizer_state,
        "session": dict(session or {}),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_stage_checkpoint(
    path: str | Path,
    *,
    expected_stages: Iterable[str],
    vocab: Optional[GateAttributeVocab] = None,
    expected_architecture: Optional[Mapping[str, Any]] = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate stage, vocabulary, and requested architecture fields."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is a legacy/unknown checkpoint. Expected format "
            f"{CHECKPOINT_FORMAT!r}; legacy checkpoints lack the vocabulary "
            "and architecture contract required for safe stage transitions."
        )
    if payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {payload.get('format_version')!r}; "
            f"expected {CHECKPOINT_VERSION}."
        )
    validate_stage_transition(
        payload.get("stage"),
        payload.get("parent_stage"),
    )
    allowed = set(expected_stages)
    if payload.get("stage") not in allowed:
        raise ValueError(
            f"Checkpoint stage {payload.get('stage')!r} is not one of "
            f"{sorted(allowed)}."
        )

    checkpoint_vocab = GateAttributeVocab.from_dict(payload["vocab"])
    if checkpoint_vocab.fingerprint != payload.get("vocab_fingerprint"):
        raise ValueError("Checkpoint gate-vocabulary fingerprint is corrupt.")
    if vocab is not None:
        vocab.assert_compatible(checkpoint_vocab)
    if expected_architecture is not None:
        _assert_subset(expected_architecture, payload.get("architecture", {}))
    return payload
