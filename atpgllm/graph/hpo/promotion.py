"""Deterministic top-K checkpoint promotion between pipeline stages."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

from .manifests import checkpoint_identity, write_json_atomic


PROMOTION_MANIFEST_VERSION = 1


def _metric(row: Mapping[str, Any], name: str) -> float | None:
    value: Any = row
    for component in name.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_top_k(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric: str,
    top_k: int,
    direction: str = "maximize",
    require_checkpoint: bool = True,
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be maximize or minimize")
    candidates = []
    for row in rows:
        score = (
            float(row["promotion_score"])
            if row.get("promotion_score") is not None
            else _metric(row, metric)
        )
        checkpoint = row.get("checkpoint_path")
        if score is None or (require_checkpoint and not checkpoint):
            continue
        candidates.append({**dict(row), "promotion_score": score})
    candidates.sort(
        key=lambda row: (
            -row["promotion_score"]
            if direction == "maximize"
            else row["promotion_score"],
            int(row.get("trial_number", 2**31 - 1)),
            int(row.get("seed", 2**31 - 1)),
        )
    )
    selected = []
    seen_hashes: set[str] = set()
    for row in candidates:
        config_hash = str(row.get("config_hash", ""))
        if config_hash and config_hash in seen_hashes:
            continue
        if config_hash:
            seen_hashes.add(config_hash)
        selected.append(row)
        if len(selected) >= top_k:
            break
    return selected


def aggregate_seed_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric: str,
    min_seeds: int,
) -> list[dict[str, Any]]:
    """Collapse repeated configurations to median score across distinct seeds."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        config_hash = str(row.get("config_hash") or "")
        if config_hash:
            grouped.setdefault(config_hash, []).append(row)
    aggregated = []
    for config_hash, members in grouped.items():
        by_seed = {
            int(member["seed"]): member
            for member in members
            if member.get("seed") is not None and _metric(member, metric) is not None
        }
        if len(by_seed) < min_seeds:
            continue
        scores = [float(_metric(member, metric)) for member in by_seed.values()]
        median = float(statistics.median(scores))
        representative = min(
            by_seed.values(),
            key=lambda member: abs(float(_metric(member, metric)) - median),
        )
        aggregated.append({
            **dict(representative),
            "promotion_score": median,
            "seed_count": len(by_seed),
            "seed_scores": {
                str(seed): float(_metric(member, metric))
                for seed, member in sorted(by_seed.items())
            },
            "config_hash": config_hash,
        })
    return aggregated


def build_promotion_manifest(
    *,
    source_stage: str,
    target_stage: str,
    study_name: str,
    metric: str,
    direction: str,
    selected: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    entries = []
    for rank, row in enumerate(selected, start=1):
        checkpoint = row.get("checkpoint_path")
        entry = {
            "rank": rank,
            "trial_number": row.get("trial_number"),
            "seed": row.get("seed"),
            "score": row.get("promotion_score", _metric(row, metric)),
            "config_hash": row.get("config_hash"),
            "params": row.get("params", {}),
            "checkpoint_path": checkpoint,
            "trial_manifest_path": row.get("trial_manifest_path"),
            "seed_count": row.get("seed_count", 1),
            "seed_scores": row.get("seed_scores", {}),
        }
        if checkpoint and Path(checkpoint).is_file():
            entry["checkpoint"] = checkpoint_identity(checkpoint)
        entries.append(entry)
    return {
        "version": PROMOTION_MANIFEST_VERSION,
        "study": study_name,
        "source_stage": source_stage,
        "target_stage": target_stage,
        "metric": metric,
        "direction": direction,
        "entries": entries,
    }


def save_promotion_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return write_json_atomic(path, payload)


def load_promotion_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != PROMOTION_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported promotion manifest version {payload.get('version')!r}."
        )
    if not isinstance(payload.get("entries"), list):
        raise ValueError("Promotion manifest requires an entries list.")
    return payload
