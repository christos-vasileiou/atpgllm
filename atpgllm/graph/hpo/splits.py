"""Design-disjoint dataset split manifests keyed by netlist SHA-256."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .manifests import design_hash, write_json_atomic


SPLIT_MANIFEST_VERSION = 1
SPLIT_NAMES = ("train", "validation", "test")


def assign_design_hash(
    value: str,
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> str:
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError("split ratios must contain three non-negative values")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("split ratios must have positive total")
    normalized = tuple(ratio / total for ratio in ratios)
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    if unit < normalized[0]:
        return "train"
    if unit < normalized[0] + normalized[1]:
        return "validation"
    return "test"


@dataclass(frozen=True)
class DesignSplitManifest:
    dataset: str
    revision: str | None
    source_split: str
    seed: int
    ratios: tuple[float, float, float]
    designs: Mapping[str, str]
    record_counts: Mapping[str, int]
    max_records_scanned: int

    def split_for_hash(self, value: str) -> str:
        existing = self.designs.get(value)
        if existing is not None:
            return existing
        return assign_design_hash(value, seed=self.seed, ratios=self.ratios)

    def split_for_record(self, record: Mapping[str, Any]) -> str:
        return self.split_for_hash(design_hash(record))

    def filter_records(
        self,
        records: Iterable[Mapping[str, Any]],
        split: str,
    ) -> Iterator[Mapping[str, Any]]:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown design split {split!r}.")
        for record in records:
            if self.split_for_record(record) == split:
                yield record

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SPLIT_MANIFEST_VERSION,
            "dataset": self.dataset,
            "revision": self.revision,
            "source_split": self.source_split,
            "seed": self.seed,
            "ratios": list(self.ratios),
            "designs": dict(sorted(self.designs.items())),
            "record_counts": dict(sorted(self.record_counts.items())),
            "max_records_scanned": self.max_records_scanned,
        }

    def save(self, path: str | Path) -> Path:
        return write_json_atomic(path, self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DesignSplitManifest":
        if payload.get("version") != SPLIT_MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported split manifest version {payload.get('version')!r}."
            )
        designs = dict(payload.get("designs", {}))
        invalid = set(designs.values()) - set(SPLIT_NAMES)
        if invalid:
            raise ValueError(f"Invalid split labels: {sorted(invalid)}")
        return cls(
            dataset=str(payload["dataset"]),
            revision=payload.get("revision"),
            source_split=str(payload.get("source_split", "train")),
            seed=int(payload["seed"]),
            ratios=tuple(float(x) for x in payload["ratios"]),
            designs=designs,
            record_counts={
                str(key): int(value)
                for key, value in payload.get("record_counts", {}).items()
            },
            max_records_scanned=int(payload.get("max_records_scanned", 0)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DesignSplitManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_design_split_manifest(
    records: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    revision: str | None,
    source_split: str,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    max_records: int = 100_000,
) -> DesignSplitManifest:
    designs: dict[str, str] = {}
    record_counts = {name: 0 for name in SPLIT_NAMES}
    scanned = 0
    for record in records:
        if scanned >= max_records:
            break
        scanned += 1
        value = design_hash(record)
        split = assign_design_hash(value, seed=seed, ratios=ratios)
        previous = designs.setdefault(value, split)
        if previous != split:
            raise AssertionError("Design split assignment changed within one manifest.")
        record_counts[split] += 1
    if not designs:
        raise ValueError("Cannot build a design split from zero records.")
    return DesignSplitManifest(
        dataset=dataset,
        revision=revision,
        source_split=source_split,
        seed=seed,
        ratios=ratios,
        designs=designs,
        record_counts=record_counts,
        max_records_scanned=scanned,
    )
