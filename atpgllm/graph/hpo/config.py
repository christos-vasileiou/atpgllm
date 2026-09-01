"""Versioned YAML configuration for staged graph HPO."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


HPO_CONFIG_VERSION = 1
HPO_STAGES = (
    "graph_pretrain",
    "graph_text_alignment",
    "multimodal_sft",
    "multimodal_grpo",
)


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


@dataclass(frozen=True)
class StageHPOConfig:
    name: str
    trials: int
    fidelities: tuple[int, ...]
    batch_size: int
    grad_accum: int
    validation_batches: int
    search: Mapping[str, Any]
    fixed: Mapping[str, Any]

    @property
    def max_steps(self) -> int:
        return self.fidelities[-1]


@dataclass(frozen=True)
class PipelineHPOConfig:
    path: Path
    profile: str
    raw: Mapping[str, Any]

    @property
    def study_name(self) -> str:
        return str(self.raw["study"]["name"])

    @property
    def artifacts_root(self) -> Path:
        return Path(self.raw["study"]["artifacts_root"])

    @property
    def sampler(self) -> Mapping[str, Any]:
        return self.raw["study"]["sampler"]

    @property
    def pruner(self) -> Mapping[str, Any]:
        return self.raw["study"]["pruner"]

    @property
    def storage(self) -> Mapping[str, Any]:
        return self.raw["study"]["storage"]

    @property
    def dataset(self) -> Mapping[str, Any]:
        return self.raw["dataset"]

    def stage(self, name: str) -> StageHPOConfig:
        if name not in HPO_STAGES:
            raise ValueError(f"Unknown HPO stage {name!r}.")
        value = self.raw["stages"][name]
        fidelities = tuple(int(x) for x in value.get("fidelities", ()))
        if not fidelities:
            raise ValueError(f"Stage {name!r} must define at least one fidelity.")
        return StageHPOConfig(
            name=name,
            trials=int(value.get("trials", 0)),
            fidelities=fidelities,
            batch_size=int(value.get("batch_size", 1)),
            grad_accum=int(value.get("grad_accum", 1)),
            validation_batches=int(value.get("validation_batches", 1)),
            search=dict(value.get("search", {})),
            fixed=dict(value.get("fixed", {})),
        )


def _validate(raw: Mapping[str, Any]) -> None:
    if raw.get("version") != HPO_CONFIG_VERSION:
        raise ValueError(
            f"Unsupported HPO config version {raw.get('version')!r}; "
            f"expected {HPO_CONFIG_VERSION}."
        )
    for section in ("study", "dataset", "stages"):
        if not isinstance(raw.get(section), Mapping):
            raise ValueError(f"HPO config requires mapping section {section!r}.")
    stages = raw["stages"]
    for stage in HPO_STAGES:
        if stage not in stages:
            raise ValueError(f"HPO config is missing stage {stage!r}.")
        value = stages[stage]
        fidelities = [int(x) for x in value.get("fidelities", ())]
        if not fidelities or fidelities != sorted(set(fidelities)):
            raise ValueError(
                f"Stage {stage!r} fidelities must be unique and increasing."
            )
        if int(value.get("batch_size", 0)) < 1:
            raise ValueError(f"Stage {stage!r} batch_size must be >= 1.")
        if int(value.get("grad_accum", 0)) < 1:
            raise ValueError(f"Stage {stage!r} grad_accum must be >= 1.")


def load_hpo_config(
    path: str | Path,
    *,
    profile: str = "production",
) -> PipelineHPOConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("HPO YAML root must be a mapping.")
    merged = dict(raw)
    if profile != "production":
        profiles = raw.get("profiles", {})
        if profile not in profiles:
            raise ValueError(f"HPO config has no profile {profile!r}.")
        merged = _deep_merge(dict(raw), profiles[profile])
    _validate(merged)
    return PipelineHPOConfig(
        path=config_path.resolve(),
        profile=profile,
        raw=merged,
    )
