"""Optuna study construction shared by coordinator and Slurm workers."""

from __future__ import annotations

from typing import Any, Mapping

from .config import PipelineHPOConfig


def stage_study_name(config: PipelineHPOConfig, stage: str) -> str:
    return f"{config.study_name}-{stage}"


def create_storage(config: PipelineHPOConfig, url: str):
    import optuna

    settings = config.storage
    return optuna.storages.RDBStorage(
        url=url,
        heartbeat_interval=int(settings.get("heartbeat_interval_seconds", 60)),
        grace_period=int(settings.get("grace_period_seconds", 180)),
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(
            max_retry=int(settings.get("max_retry", 1)),
        ),
    )


def create_sampler(config: PipelineHPOConfig):
    import optuna

    value = config.sampler
    return optuna.samplers.TPESampler(
        seed=int(value.get("seed", 42)),
        n_startup_trials=int(value.get("n_startup_trials", 16)),
        n_ei_candidates=int(value.get("n_ei_candidates", 48)),
        multivariate=bool(value.get("multivariate", True)),
        group=bool(value.get("group", True)),
        constant_liar=bool(value.get("constant_liar", True)),
    )


def create_pruner(config: PipelineHPOConfig, stage: str):
    import optuna

    value = config.pruner
    stage_config = config.stage(stage)
    return optuna.pruners.HyperbandPruner(
        min_resource=int(value.get("min_resource", stage_config.fidelities[0])),
        max_resource=int(value.get("max_resource", stage_config.fidelities[-1])),
        reduction_factor=int(value.get("reduction_factor", 3)),
        bootstrap_count=int(value.get("bootstrap_count", 0)),
    )


def load_or_create_study(
    config: PipelineHPOConfig,
    *,
    stage: str,
    storage_url: str,
):
    import optuna

    return optuna.create_study(
        study_name=stage_study_name(config, stage),
        storage=create_storage(config, storage_url),
        sampler=create_sampler(config),
        pruner=create_pruner(config, stage),
        direction="maximize",
        load_if_exists=True,
    )


def completed_trial_rows(study: Any) -> list[dict[str, Any]]:
    import optuna

    rows = []
    for trial in study.get_trials(deepcopy=False):
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            continue
        rows.append({
            "trial_number": trial.number,
            "seed": trial.user_attrs.get("seed"),
            "value": float(trial.value),
            "semantic_score": float(trial.value),
            "checkpoint_path": trial.user_attrs.get("checkpoint_path"),
            "trial_manifest_path": trial.user_attrs.get("trial_manifest_path"),
            "config_hash": trial.user_attrs.get("config_hash"),
            "params": dict(trial.params),
            "metrics": trial.user_attrs.get("final_metrics", {}),
            "parent_checkpoint": trial.user_attrs.get("parent_checkpoint"),
        })
    return rows
