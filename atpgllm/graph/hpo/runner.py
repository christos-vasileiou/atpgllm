"""Executable single-GPU Optuna objectives for Stage A and Stage B."""

from __future__ import annotations

import json
from itertools import chain, islice
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from atpgllm.graph.checkpoints import (
    STAGE_GRAPH_PRETRAIN,
    STAGE_GRAPH_TEXT_ALIGNMENT,
    load_stage_checkpoint,
    save_stage_checkpoint,
)
from atpgllm.graph.dataset import (
    ASAP7GraphPretrainDataset,
    ASAP7GraphTextDataset,
    collate_graph_pretrain_batch,
    collate_graph_text_batch,
    record_to_pyg,
)
from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.models_stage1 import Stage1GraphTextModel
from atpgllm.graph.pretrain import GraphPretrainingModel
from atpgllm.graph.train_stage1 import LossWeights, Stage1Trainer
from atpgllm.training._paths import resolve_sim_config_path

from .config import PipelineHPOConfig
from .manifests import (
    append_jsonl,
    build_trial_manifest,
    canonical_hash,
    checkpoint_filename,
    trial_artifact_dir,
    write_json_atomic,
)
from .promotion import load_promotion_manifest
from .spaces import (
    assert_graph_architecture_locked,
    suggest_graph_pretrain,
    suggest_graph_text_alignment,
)
from .splits import DesignSplitManifest
from .validation import (
    evaluate_graph_pretraining,
    evaluate_graph_text_alignment,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _raw_stream(
    config: PipelineHPOConfig,
    *,
    seed: int,
    shuffle: bool,
):
    dataset = config.dataset
    kwargs: dict[str, Any] = {
        "path": dataset["name"],
        "split": dataset.get("source_split", "train"),
        "streaming": True,
    }
    if dataset.get("revision"):
        kwargs["revision"] = dataset["revision"]
    stream = load_dataset(**kwargs)
    if shuffle:
        stream = stream.shuffle(
            seed=seed,
            buffer_size=int(dataset.get("shuffle_buffer", 2048)),
        )
    return stream


def _records_for_split(
    config: PipelineHPOConfig,
    split_manifest: DesignSplitManifest,
    split: str,
    *,
    seed: int,
    shuffle: bool,
) -> Iterable[Mapping[str, Any]]:
    return split_manifest.filter_records(
        _raw_stream(config, seed=seed, shuffle=shuffle),
        split,
    )


def _gate_library(
    fixed: Mapping[str, Any],
) -> tuple[dict[str, dict[str, str]], GateAttributeVocab]:
    sim_path = resolve_sim_config_path(fixed.get("sim_config"))
    gate_funcs = json.loads(sim_path.read_text(encoding="utf-8"))["gate_funcs"]
    return gate_funcs, GateAttributeVocab(gate_funcs)


def _trial_paths(
    config: PipelineHPOConfig,
    stage: str,
    trial_number: int,
    seed: int,
) -> tuple[Path, Path, Path]:
    directory = trial_artifact_dir(
        config.artifacts_root,
        config.study_name,
        stage,
        trial_number,
        seed,
    )
    directory.mkdir(parents=True, exist_ok=True)
    return (
        directory,
        directory / "trial_manifest.json",
        directory / "metrics.jsonl",
    )


def _record_trial_start(
    *,
    trial: Any,
    config: PipelineHPOConfig,
    stage: str,
    seed: int,
    params: Mapping[str, Any],
    fixed: Mapping[str, Any],
    split_manifest_path: Path,
    parent_checkpoint: Path | None,
) -> tuple[Path, dict[str, Any], Path]:
    directory, manifest_path, metrics_path = _trial_paths(
        config, stage, trial.number, seed
    )
    manifest = build_trial_manifest(
        study_name=config.study_name,
        stage=stage,
        trial_number=trial.number,
        seed=seed,
        fidelities=list(config.stage(stage).fidelities),
        params=params,
        fixed=fixed,
        split_manifest=split_manifest_path,
        repository=_repository_root(),
        parent_checkpoint=parent_checkpoint,
    )
    write_json_atomic(manifest_path, manifest)
    trial.set_user_attr("seed", seed)
    trial.set_user_attr("config_hash", manifest["config_hash"])
    trial.set_user_attr("trial_manifest_path", str(manifest_path.resolve()))
    if parent_checkpoint is not None:
        trial.set_user_attr("parent_checkpoint", str(parent_checkpoint.resolve()))
    return directory, manifest, metrics_path


def _save_result(
    directory: Path,
    *,
    status: str,
    score: float | None,
    metrics: Mapping[str, Any] | None,
    checkpoint_path: Path | None,
) -> None:
    write_json_atomic(
        directory / "result.json",
        {
            "status": status,
            "semantic_score": score,
            "metrics": dict(metrics or {}),
            "checkpoint_path": (
                str(checkpoint_path.resolve())
                if checkpoint_path is not None
                else None
            ),
        },
    )


def _validation_graphs(
    records: Iterable[Mapping[str, Any]],
    gate_funcs: dict[str, dict[str, str]],
    vocab: GateAttributeVocab,
    *,
    limit: int,
) -> list[Any]:
    graphs = []
    for record in records:
        graph = record_to_pyg(dict(record), gate_funcs, vocab)
        if graph is not None:
            graphs.append(graph)
        if len(graphs) >= limit:
            break
    if not graphs:
        raise ValueError("No valid validation graphs matched the split manifest.")
    return graphs


def run_graph_pretrain_trial(
    trial: Any,
    *,
    config: PipelineHPOConfig,
    split_manifest_path: str | Path,
    seed: int,
    device: str,
) -> float:
    import optuna

    seed = int(trial.user_attrs.get("requested_seed", seed))
    stage = STAGE_GRAPH_PRETRAIN
    stage_config = config.stage(stage)
    params = suggest_graph_pretrain(trial, stage_config.search)
    fixed = dict(stage_config.fixed)
    split_path = Path(split_manifest_path)
    split_manifest = DesignSplitManifest.load(split_path)
    gate_funcs, vocab = _gate_library(fixed)
    directory, manifest, metrics_path = _record_trial_start(
        trial=trial,
        config=config,
        stage=stage,
        seed=seed,
        params=params,
        fixed=fixed,
        split_manifest_path=split_path,
        parent_checkpoint=None,
    )

    torch.manual_seed(seed)
    target_device = torch.device(device)
    model = GraphPretrainingModel(
        vocab,
        node_dim=int(params["node_dim"]),
        gin_hidden_dim=int(params["gin_hidden"]),
        gin_num_layers=int(params["gin_layers"]),
        dropout=float(params["dropout"]),
        per_attr_dim=int(params["per_attr_dim"]),
        attr_mlp_hidden=int(params["attr_mlp_hidden"]),
    ).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )

    train_dataset = ASAP7GraphPretrainDataset(
        _records_for_split(
            config,
            split_manifest,
            fixed.get("train_split", "train"),
            seed=seed,
            shuffle=True,
        ),
        gate_funcs,
        vocab,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=stage_config.batch_size,
        num_workers=int(fixed.get("num_workers", 0)),
        collate_fn=collate_graph_pretrain_batch,
    )
    validation_graphs = _validation_graphs(
        _records_for_split(
            config,
            split_manifest,
            fixed.get("validation_split", "validation"),
            seed=seed,
            shuffle=False,
        ),
        gate_funcs,
        vocab,
        limit=(
            stage_config.batch_size
            * stage_config.validation_batches
        ),
    )

    def validation_loader():
        return DataLoader(
            validation_graphs,
            batch_size=stage_config.batch_size,
            shuffle=False,
            collate_fn=collate_graph_pretrain_batch,
        )

    weights = params["loss_weights"]
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    optimizer_step = 0
    latest_checkpoint: Path | None = None
    latest_metrics: Mapping[str, Any] | None = None
    rung_set = set(stage_config.fidelities)
    use_amp = (
        target_device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    try:
        for graph in train_loader:
            graph = graph.to(target_device)
            with torch.autocast(
                device_type=target_device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                losses = model.compute_losses(graph)
                total = (
                    weights["propagation"] * losses.propagation
                    + weights["backtrack"] * losses.backtrack
                    + weights["discrepancy"] * losses.discrepancy
                ) / stage_config.grad_accum
            total.backward()
            micro_step += 1
            if micro_step % stage_config.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(params["grad_clip"]),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            if optimizer_step in rung_set:
                metrics = evaluate_graph_pretraining(
                    model,
                    validation_loader(),
                    device=target_device,
                    max_batches=stage_config.validation_batches,
                )
                score = float(metrics["semantic_score"])
                architecture = model.architecture_config
                filename = checkpoint_filename(
                    config.study_name,
                    stage,
                    trial.number,
                    seed,
                    optimizer_step,
                    manifest["config_hash"],
                )
                latest_checkpoint = save_stage_checkpoint(
                    directory / filename,
                    stage=stage,
                    parent_stage=None,
                    vocab=vocab,
                    architecture=architecture,
                    states={
                        "graph_encoder": model.graph_encoder.state_dict(),
                        "pretraining_heads": model.heads.state_dict(),
                    },
                    optimizer_state=optimizer.state_dict(),
                    step=optimizer_step,
                    session={
                        "hpo_trial_manifest": str(
                            (directory / "trial_manifest.json").resolve()
                        ),
                        "semantic_metrics": metrics,
                    },
                )
                latest_metrics = metrics
                append_jsonl(
                    metrics_path,
                    {
                        "step": optimizer_step,
                        "semantic_score": score,
                        "metrics": metrics,
                        "checkpoint_path": str(latest_checkpoint.resolve()),
                    },
                )
                trial.set_user_attr(
                    "checkpoint_path", str(latest_checkpoint.resolve())
                )
                trial.report(score, optimizer_step)
                if trial.should_prune():
                    _save_result(
                        directory,
                        status="pruned",
                        score=score,
                        metrics=metrics,
                        checkpoint_path=latest_checkpoint,
                    )
                    raise optuna.TrialPruned(
                        f"Stage-A trial pruned at step {optimizer_step}."
                    )
            if optimizer_step >= stage_config.max_steps:
                break
        if latest_metrics is None or latest_checkpoint is None:
            raise RuntimeError("Stage-A stream ended before the first validation rung.")
        score = float(latest_metrics["semantic_score"])
        trial.set_user_attr("final_metrics", dict(latest_metrics))
        _save_result(
            directory,
            status="complete",
            score=score,
            metrics=latest_metrics,
            checkpoint_path=latest_checkpoint,
        )
        return score
    except optuna.TrialPruned:
        raise
    except Exception as error:
        _save_result(
            directory,
            status="failed",
            score=None,
            metrics={"error": repr(error)},
            checkpoint_path=latest_checkpoint,
        )
        raise


def _select_parent(trial: Any, promotion_manifest_path: str | Path) -> Path:
    promotion = load_promotion_manifest(promotion_manifest_path)
    entries = promotion["entries"]
    if not entries:
        raise ValueError("Stage-B promotion manifest contains no parent checkpoints.")
    by_id = {
        f"rank-{int(entry['rank']):03d}-{str(entry.get('config_hash', ''))[:8]}": entry
        for entry in entries
    }
    parent_id = trial.suggest_categorical("parent_id", sorted(by_id))
    parent = by_id[parent_id].get("checkpoint_path")
    if not parent:
        raise ValueError(f"Promoted parent {parent_id!r} has no checkpoint path.")
    return Path(parent)


def _set_alignment_graph_policy(model: Stage1GraphTextModel, policy: str) -> None:
    for parameter in model.graph_encoder.parameters():
        parameter.requires_grad = policy == "full"
    if policy == "last_layer":
        for parameter in model.graph_encoder.dag_gin.layers[-1].parameters():
            parameter.requires_grad = True
        if model.graph_encoder.dag_gin.jk_proj is not None:
            for parameter in model.graph_encoder.dag_gin.jk_proj.parameters():
                parameter.requires_grad = True


def _alignment_transfer_state(model: Stage1GraphTextModel) -> dict[str, Any]:
    """Exclude the reloadable frozen text encoder from HPO transfer checkpoints."""
    prefixes = ("graph_encoder.", "q_former.", "graph_proj.", "text_proj.")
    return {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }


def run_graph_text_alignment_trial(
    trial: Any,
    *,
    config: PipelineHPOConfig,
    split_manifest_path: str | Path,
    promotion_manifest_path: str | Path,
    seed: int,
    device: str,
) -> float:
    import optuna

    seed = int(trial.user_attrs.get("requested_seed", seed))
    stage = STAGE_GRAPH_TEXT_ALIGNMENT
    stage_config = config.stage(stage)
    parent_checkpoint = _select_parent(trial, promotion_manifest_path)
    params = suggest_graph_text_alignment(trial, stage_config.search)
    fixed = dict(stage_config.fixed)
    split_path = Path(split_manifest_path)
    split_manifest = DesignSplitManifest.load(split_path)
    gate_funcs, vocab = _gate_library(fixed)
    parent_payload = load_stage_checkpoint(
        parent_checkpoint,
        expected_stages=[STAGE_GRAPH_PRETRAIN],
        vocab=vocab,
    )
    parent_graph_architecture = parent_payload["architecture"]["graph_encoder"]

    directory, manifest, metrics_path = _record_trial_start(
        trial=trial,
        config=config,
        stage=stage,
        seed=seed,
        params=params,
        fixed=fixed,
        split_manifest_path=split_path,
        parent_checkpoint=parent_checkpoint,
    )
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(fixed["text_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Stage1GraphTextModel(
        vocab=vocab,
        text_model_name=fixed["text_model"],
        graph_encoder_config=parent_graph_architecture,
        qformer_hidden_dim=int(params["qformer_hidden"]),
        qformer_layers=int(params["qformer_layers"]),
        qformer_heads=int(params["qformer_heads"]),
        qformer_cross_every_n=int(params["cross_attn_every_n"]),
        qformer_ffn_multiplier=int(params["ffn_multiplier"]),
        qformer_dropout=float(params["qformer_dropout"]),
        num_queries=int(params["num_queries"]),
        proj_dim=int(params["proj_dim"]),
        freeze_text=bool(fixed.get("freeze_text", True)),
    )
    assert_graph_architecture_locked(
        parent_graph_architecture,
        model.graph_encoder.config,
    )
    model.graph_encoder.load_state_dict(
        parent_payload["states"]["graph_encoder"],
        strict=True,
    )
    _set_alignment_graph_policy(model, str(params["graph_policy"]))

    target_device = torch.device(device)
    weights = params["loss_weights"]
    trainer = Stage1Trainer(
        model,
        vocab_size=len(tokenizer),
        weights=LossWeights(
            gtc=float(weights["gtc"]),
            gtm=float(weights["gtm"]),
            gtg=float(weights["gtg"]),
        ),
        grad_clip=float(params["grad_clip"]),
        device=target_device,
        gtg_max_seq_len=int(fixed.get("max_answer_len", 512)),
        use_bf16=target_device.type == "cuda",
    )
    trainable = [
        parameter
        for parameter in chain(
            trainer.model.parameters(),
            trainer.gtm_loss.parameters(),
            trainer.gtg_loss.parameters(),
        )
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )

    def make_dataset(split: str, shuffle: bool):
        return ASAP7GraphTextDataset(
            _records_for_split(
                config,
                split_manifest,
                split,
                seed=seed,
                shuffle=shuffle,
            ),
            gate_funcs,
            vocab,
            tokenizer,
            max_text_len=int(fixed.get("max_text_len", 512)),
            max_answer_len=int(fixed.get("max_answer_len", 512)),
        )

    def collate(items):
        return collate_graph_text_batch(
            items,
            tokenizer,
            max_text_len=int(fixed.get("max_text_len", 512)),
            max_answer_len=int(fixed.get("max_answer_len", 512)),
        )

    train_loader = DataLoader(
        make_dataset(fixed.get("train_split", "train"), True),
        batch_size=stage_config.batch_size,
        num_workers=int(fixed.get("num_workers", 0)),
        collate_fn=collate,
        drop_last=True,
    )
    validation_items = list(islice(
        make_dataset(fixed.get("validation_split", "validation"), False),
        stage_config.batch_size * stage_config.validation_batches,
    ))
    if len(validation_items) < 2:
        raise ValueError("Stage-B validation needs at least two valid examples.")

    def validation_loader():
        return DataLoader(
            validation_items,
            batch_size=stage_config.batch_size,
            shuffle=False,
            collate_fn=collate,
        )

    optimizer_step = 0
    accum_index = 0
    latest_checkpoint: Path | None = None
    latest_metrics: Mapping[str, Any] | None = None
    rung_set = set(stage_config.fidelities)
    try:
        for batch in train_loader:
            trainer.train_step(
                batch,
                optimizer,
                accum_index=accum_index,
                grad_accum_steps=stage_config.grad_accum,
            )
            accum_index = (accum_index + 1) % stage_config.grad_accum
            if accum_index:
                continue
            optimizer_step += 1
            if optimizer_step in rung_set:
                metrics = evaluate_graph_text_alignment(
                    trainer,
                    validation_loader(),
                    max_batches=stage_config.validation_batches,
                )
                score = float(metrics["semantic_score"])
                filename = checkpoint_filename(
                    config.study_name,
                    stage,
                    trial.number,
                    seed,
                    optimizer_step,
                    manifest["config_hash"],
                )
                latest_checkpoint = save_stage_checkpoint(
                    directory / filename,
                    stage=stage,
                    parent_stage=STAGE_GRAPH_PRETRAIN,
                    vocab=vocab,
                    architecture=trainer.model.architecture_config,
                    states={
                        "model": _alignment_transfer_state(trainer.model),
                        "gtm_loss": trainer.gtm_loss.state_dict(),
                    },
                    optimizer_state=None,
                    step=optimizer_step,
                    session={
                        "hpo_trial_manifest": str(
                            (directory / "trial_manifest.json").resolve()
                        ),
                        "parent_checkpoint": str(parent_checkpoint.resolve()),
                        "semantic_metrics": metrics,
                        "hpo_transfer_checkpoint": True,
                    },
                )
                latest_metrics = metrics
                append_jsonl(
                    metrics_path,
                    {
                        "step": optimizer_step,
                        "semantic_score": score,
                        "metrics": metrics,
                        "checkpoint_path": str(latest_checkpoint.resolve()),
                    },
                )
                trial.set_user_attr(
                    "checkpoint_path", str(latest_checkpoint.resolve())
                )
                trial.report(score, optimizer_step)
                if trial.should_prune():
                    _save_result(
                        directory,
                        status="pruned",
                        score=score,
                        metrics=metrics,
                        checkpoint_path=latest_checkpoint,
                    )
                    raise optuna.TrialPruned(
                        f"Stage-B trial pruned at step {optimizer_step}."
                    )
            if optimizer_step >= stage_config.max_steps:
                break
        if latest_metrics is None or latest_checkpoint is None:
            raise RuntimeError("Stage-B stream ended before the first validation rung.")
        score = float(latest_metrics["semantic_score"])
        trial.set_user_attr("final_metrics", dict(latest_metrics))
        _save_result(
            directory,
            status="complete",
            score=score,
            metrics=latest_metrics,
            checkpoint_path=latest_checkpoint,
        )
        return score
    except optuna.TrialPruned:
        raise
    except Exception as error:
        _save_result(
            directory,
            status="failed",
            score=None,
            metrics={"error": repr(error)},
            checkpoint_path=latest_checkpoint,
        )
        raise
