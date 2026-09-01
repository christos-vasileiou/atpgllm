"""Coordinate staged, checkpoint-compatible graph-pipeline HPO."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

from atpgllm.graph.checkpoints import (
    STAGE_GRAPH_PRETRAIN,
    STAGE_GRAPH_TEXT_ALIGNMENT,
    STAGE_MULTIMODAL_GRPO,
    STAGE_MULTIMODAL_SFT,
)
from atpgllm.graph.hpo.config import HPO_STAGES, load_hpo_config
from atpgllm.graph.hpo.manifests import (
    canonical_hash,
    checkpoint_identity,
    write_json_atomic,
)
from atpgllm.graph.hpo.promotion import (
    aggregate_seed_rows,
    build_promotion_manifest,
    load_promotion_manifest,
    save_promotion_manifest,
    select_top_k,
)
from atpgllm.graph.hpo.runner import (
    run_graph_pretrain_trial,
    run_graph_text_alignment_trial,
)
from atpgllm.graph.hpo.spaces import (
    downstream_screening_configs,
    suggest_graph_pretrain,
    suggest_graph_text_alignment,
)
from atpgllm.graph.hpo.splits import (
    DesignSplitManifest,
    build_design_split_manifest,
)
from atpgllm.graph.hpo.study import (
    completed_trial_rows,
    load_or_create_study,
    stage_study_name,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "train"
    / "configs"
    / "hpo_graph_pipeline.yaml"
)

_NEXT_STAGE = {
    STAGE_GRAPH_PRETRAIN: STAGE_GRAPH_TEXT_ALIGNMENT,
    STAGE_GRAPH_TEXT_ALIGNMENT: STAGE_MULTIMODAL_SFT,
    STAGE_MULTIMODAL_SFT: STAGE_MULTIMODAL_GRPO,
    STAGE_MULTIMODAL_GRPO: "final",
}


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", default="production")


def _storage(args: argparse.Namespace) -> str:
    value = args.storage or os.environ.get("OPTUNA_STORAGE")
    if not value:
        raise ValueError(
            "Optuna storage is required. Pass --storage or set OPTUNA_STORAGE "
            "(PostgreSQL is required for distributed workers)."
        )
    if value.startswith("sqlite:///"):
        database_path = value.removeprefix("sqlite:///")
        if database_path and database_path != ":memory:":
            Path(database_path).expanduser().resolve().parent.mkdir(
                parents=True,
                exist_ok=True,
            )
    return value


def _load_dataset_stream(config):
    dataset = config.dataset
    kwargs: dict[str, Any] = {
        "path": dataset["name"],
        "split": dataset.get("source_split", "train"),
        "streaming": True,
    }
    if dataset.get("revision"):
        kwargs["revision"] = dataset["revision"]
    return load_dataset(**kwargs)


def command_prepare_split(args: argparse.Namespace) -> int:
    config = load_hpo_config(args.config, profile=args.profile)
    dataset = config.dataset
    manifest = build_design_split_manifest(
        _load_dataset_stream(config),
        dataset=dataset["name"],
        revision=dataset.get("revision"),
        source_split=dataset.get("source_split", "train"),
        seed=int(dataset.get("split_seed", 42)),
        ratios=tuple(float(x) for x in dataset["split_ratios"]),
        max_records=(
            args.max_records
            if args.max_records is not None
            else int(dataset.get("split_scan_records", 100_000))
        ),
    )
    manifest.save(args.output)
    design_counts = {
        split: sum(1 for value in manifest.designs.values() if value == split)
        for split in ("train", "validation", "test")
    }
    print(json.dumps({
        "design_counts": design_counts,
        "record_counts": manifest.record_counts,
        "records_scanned": manifest.max_records_scanned,
    }, indent=2))
    print(f"Wrote design-disjoint split manifest: {args.output}")
    return 0


def command_validate_split(args: argparse.Namespace) -> int:
    manifest = DesignSplitManifest.load(args.manifest)
    counts = {name: 0 for name in ("train", "validation", "test")}
    for split in manifest.designs.values():
        counts[split] += 1
    if len(manifest.designs) != sum(counts.values()):
        raise AssertionError("A design appears in more than one split.")
    print(json.dumps({
        "design_counts": counts,
        "record_counts": manifest.record_counts,
        "manifest": str(args.manifest.resolve()),
    }, indent=2))
    return 0


def command_create_study(args: argparse.Namespace) -> int:
    config = load_hpo_config(args.config, profile=args.profile)
    study = load_or_create_study(
        config,
        stage=args.stage,
        storage_url=_storage(args),
    )
    print(f"Study ready: {study.study_name}")
    print(f"Direction: {study.direction.name.lower()}")
    return 0


def command_worker(args: argparse.Namespace) -> int:
    config = load_hpo_config(args.config, profile=args.profile)
    study = load_or_create_study(
        config,
        stage=args.stage,
        storage_url=_storage(args),
    )
    if args.stage == STAGE_GRAPH_PRETRAIN:
        objective = lambda trial: run_graph_pretrain_trial(
            trial,
            config=config,
            split_manifest_path=args.split_manifest,
            seed=args.seed,
            device=args.device,
        )
    elif args.stage == STAGE_GRAPH_TEXT_ALIGNMENT:
        if args.promotion_manifest is None:
            raise ValueError("Stage-B worker requires --promotion-manifest.")
        objective = lambda trial: run_graph_text_alignment_trial(
            trial,
            config=config,
            split_manifest_path=args.split_manifest,
            promotion_manifest_path=args.promotion_manifest,
            seed=args.seed,
            device=args.device,
        )
    else:
        raise ValueError(
            "Optuna workers currently support graph_pretrain and "
            "graph_text_alignment only. Use plan-downstream for SFT/GRPO."
        )
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        gc_after_trial=True,
        catch=(RuntimeError, ValueError, torch.cuda.OutOfMemoryError),
    )
    return 0


def command_promote(args: argparse.Namespace) -> int:
    config = load_hpo_config(args.config, profile=args.profile)
    study = load_or_create_study(
        config,
        stage=args.stage,
        storage_url=_storage(args),
    )
    rows = completed_trial_rows(study)
    if args.min_seeds > 1:
        rows = aggregate_seed_rows(
            rows,
            metric="semantic_score",
            min_seeds=args.min_seeds,
        )
    selected = select_top_k(
        rows,
        metric="semantic_score",
        top_k=args.top_k,
        direction="maximize",
    )
    if not selected:
        raise ValueError("No completed trials with checkpoints can be promoted.")
    target = args.target_stage or _NEXT_STAGE[args.stage]
    payload = build_promotion_manifest(
        source_stage=args.stage,
        target_stage=target,
        study_name=stage_study_name(config, args.stage),
        metric="semantic_score",
        direction="maximize",
        selected=selected,
    )
    save_promotion_manifest(args.output, payload)
    print(f"Promoted {len(payload['entries'])} checkpoint(s) to {args.output}")
    return 0


def command_enqueue_confirmations(args: argparse.Namespace) -> int:
    config = load_hpo_config(args.config, profile=args.profile)
    study = load_or_create_study(
        config,
        stage=args.stage,
        storage_url=_storage(args),
    )
    promotion = load_promotion_manifest(args.promotion)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    existing = {
        (str(row.get("config_hash")), int(row["seed"]))
        for row in completed_trial_rows(study)
        if row.get("config_hash") and row.get("seed") is not None
    }
    enqueued = 0
    for entry in promotion["entries"]:
        params = dict(entry.get("params", {}))
        if not params:
            raise ValueError(
                f"Promotion rank {entry.get('rank')} has no Optuna params."
            )
        for seed in seeds:
            if (str(entry.get("config_hash")), seed) in existing:
                continue
            study.enqueue_trial(
                params,
                user_attrs={
                    "requested_seed": seed,
                    "confirmation_of": entry.get("trial_number"),
                },
            )
            enqueued += 1
    print(f"Enqueued {enqueued} fixed-configuration confirmation trial(s).")
    return 0


def _sft_parent_settings(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    session = payload.get("session", {})
    architecture = payload.get("architecture", {})
    return {
        "alignment_checkpoint": (
            str(session.get("alignment_ckpt"))
            if session.get("alignment_ckpt")
            else None
        ),
        "base_model": architecture.get("base_model"),
        "lora": architecture.get("lora", {}),
    }


def _plan_command(
    stage: str,
    parent_checkpoint: Path,
    params: dict[str, Any],
    *,
    seed: int,
    output_dir: Path,
    stage_config,
    alignment_checkpoint: str | None,
    parent_settings: dict[str, Any] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/train/multimodal_training.py",
        "--method",
        "sft" if stage == STAGE_MULTIMODAL_SFT else "grpo",
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--graph-policy",
        str(params["graph_policy"]),
        "--lr",
        str(params["lr"]),
        "--graph-lr",
        str(params["graph_lr"]),
        "--max-steps",
        str(stage_config.fixed.get(
            "pilot_optimizer_steps",
            stage_config.fidelities[-1],
        )),
        "--grad-accum",
        str(stage_config.grad_accum),
    ]
    if stage == STAGE_MULTIMODAL_SFT:
        command += [
            "--alignment-ckpt",
            str(parent_checkpoint),
            "--lora-r",
            str(params["lora_r"]),
            "--lora-alpha",
            str(int(params["lora_r"] * params["lora_alpha_ratio"])),
            "--max-seq-len",
            str(stage_config.fixed.get("max_seq_len", 4096)),
        ]
    else:
        if not alignment_checkpoint:
            raise ValueError(
                f"Cannot derive alignment checkpoint from {parent_checkpoint}; "
                "pass --alignment-checkpoint."
            )
        command += [
            "--alignment-ckpt",
            alignment_checkpoint,
            "--sft-ckpt",
            str(parent_checkpoint),
            "--beta",
            str(params["beta"]),
            "--temperature",
            str(params["temperature"]),
            "--clip-epsilon",
            str(params["clip_epsilon"]),
            "--num-generations",
            str(params["num_generations"]),
            "--max-prompt-length",
            str(stage_config.fixed.get("max_prompt_length", 2048)),
            "--max-completion-length",
            str(stage_config.fixed.get("max_completion_length", 2048)),
        ]
        settings = parent_settings or {}
        if settings.get("base_model"):
            command += ["--llm", str(settings["base_model"])]
        lora = settings.get("lora", {})
        if lora:
            command += [
                "--lora-r",
                str(lora["r"]),
                "--lora-alpha",
                str(lora["alpha"]),
                "--lora-dropout",
                str(lora["dropout"]),
            ]
    return command


def command_plan_downstream(args: argparse.Namespace) -> int:
    config = load_hpo_config(args.config, profile=args.profile)
    stage_config = config.stage(args.stage)
    promotion = load_promotion_manifest(args.parents)
    limit = (
        args.configs_per_parent
        if args.configs_per_parent is not None
        else int(stage_config.fixed.get("configs_per_parent", 3))
    )
    screening = downstream_screening_configs(args.stage, limit=limit)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    entries = []
    index = 0
    for parent in promotion["entries"]:
        parent_checkpoint = Path(parent["checkpoint_path"]).resolve()
        parent_identity = (
            parent.get("checkpoint")
            or checkpoint_identity(parent_checkpoint)
        )
        alignment_checkpoint = args.alignment_checkpoint
        parent_settings = None
        if args.stage == STAGE_MULTIMODAL_GRPO:
            parent_settings = _sft_parent_settings(parent_checkpoint)
            alignment_checkpoint = (
                alignment_checkpoint
                or parent_settings["alignment_checkpoint"]
            )
        for params in screening:
            for seed in seeds:
                config_hash = canonical_hash({
                    "stage": args.stage,
                    "parent": parent_identity,
                    "params": params,
                })
                run_hash = canonical_hash({
                    "config_hash": config_hash,
                    "seed": seed,
                })
                output_dir = (
                    config.artifacts_root
                    / config.study_name
                    / args.stage
                    / f"screen-{index:05d}-{run_hash[:8]}"
                )
                command = _plan_command(
                    args.stage,
                    parent_checkpoint,
                    params,
                    seed=seed,
                    output_dir=output_dir,
                    stage_config=stage_config,
                    alignment_checkpoint=alignment_checkpoint,
                    parent_settings=parent_settings,
                )
                entries.append({
                    "index": index,
                    "stage": args.stage,
                    "seed": seed,
                    "params": params,
                    "config_hash": config_hash,
                    "run_hash": run_hash,
                    "parent_checkpoint": parent_identity,
                    "output_dir": str(output_dir.resolve()),
                    "command": command,
                    "selection_metric": stage_config.fixed.get(
                        "heldout_metric", "heldout_pass_at_1"
                    ),
                    "selection_status": "requires_graph_aware_heldout_evaluation",
                })
                index += 1
    payload = {
        "version": 1,
        "stage": args.stage,
        "source_promotion": str(args.parents.resolve()),
        "entries": entries,
        "limitation": (
            "Stage C/D are explicit screening plans, not Optuna objectives: "
            "the repository does not yet have a graph-aware held-out simulator "
            "evaluator suitable for safe pruning."
        ),
    }
    write_json_atomic(args.output, payload)
    print(f"Wrote {len(entries)} planned {args.stage} run(s): {args.output}")
    return 0


def command_execute_plan(args: argparse.Namespace) -> int:
    payload = json.loads(args.plan.read_text(encoding="utf-8"))
    index = args.index
    if index is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise ValueError("Pass --index or run inside a Slurm array.")
        index = int(raw)
    entries = {int(entry["index"]): entry for entry in payload["entries"]}
    if index not in entries:
        raise ValueError(f"Plan has no entry {index}.")
    entry = entries[index]
    if args.dry_run:
        print(" ".join(str(part) for part in entry["command"]))
        return 0
    result = subprocess.run(entry["command"], cwd=Path(__file__).resolve().parents[3])
    write_json_atomic(
        Path(entry["output_dir"]) / "launch_result.json",
        {
            "plan": str(args.plan.resolve()),
            "index": index,
            "returncode": result.returncode,
        },
    )
    return result.returncode


def command_promote_evaluations(args: argparse.Namespace) -> int:
    payload = json.loads(args.evaluations.read_text(encoding="utf-8"))
    rows = payload["entries"] if isinstance(payload, dict) else payload
    if args.min_seeds > 1:
        rows = aggregate_seed_rows(
            rows,
            metric=args.metric,
            min_seeds=args.min_seeds,
        )
    selected = select_top_k(
        rows,
        metric=args.metric,
        top_k=args.top_k,
        direction=args.direction,
    )
    promotion = build_promotion_manifest(
        source_stage=args.stage,
        target_stage=args.target_stage or _NEXT_STAGE[args.stage],
        study_name=args.study_name,
        metric=args.metric,
        direction=args.direction,
        selected=selected,
    )
    save_promotion_manifest(args.output, promotion)
    print(f"Promoted {len(promotion['entries'])} held-out result(s).")
    return 0


def command_export_best(args: argparse.Namespace) -> int:
    promotion = load_promotion_manifest(args.promotion)
    entries = promotion["entries"]
    if not entries:
        raise ValueError("Promotion manifest has no entries.")
    rank = max(1, args.rank)
    matching = [entry for entry in entries if int(entry["rank"]) == rank]
    if not matching:
        raise ValueError(f"Promotion manifest has no rank {rank}.")
    payload = {
        "version": 1,
        "source_promotion": str(args.promotion.resolve()),
        "selected": matching[0],
    }
    write_json_atomic(args.output, payload)
    print(f"Exported rank {rank} lineage to {args.output}")
    return 0


def command_mock_study(args: argparse.Namespace) -> int:
    config = load_hpo_config(args.config, profile=args.profile)
    study = load_or_create_study(
        config,
        stage=args.stage,
        storage_url=_storage(args),
    )
    stage_config = config.stage(args.stage)

    def objective(trial):
        if args.stage == STAGE_GRAPH_PRETRAIN:
            params = suggest_graph_pretrain(trial, stage_config.search)
        elif args.stage == STAGE_GRAPH_TEXT_ALIGNMENT:
            params = suggest_graph_text_alignment(trial, stage_config.search)
        else:
            raise ValueError("Mock study supports Stage A/B.")
        target = int(canonical_hash(params, 8), 16) / float(16**8)
        score = 0.0
        for step in stage_config.fidelities:
            score = 1.0 - abs(target - 0.37) - 1.0 / (step + 1)
            trial.report(score, step)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()
        trial.set_user_attr("seed", args.seed)
        trial.set_user_attr("config_hash", canonical_hash(params))
        trial.set_user_attr("final_metrics", {"semantic_score": score})
        return score

    study.optimize(objective, n_trials=args.trials)
    print(
        json.dumps({
            "study": study.study_name,
            "trials": len(study.trials),
            "best_value": study.best_value,
        }, indent=2)
    )
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    config = load_hpo_config(args.config, profile=args.profile)
    study = load_or_create_study(
        config,
        stage=args.stage,
        storage_url=_storage(args),
    )
    print(json.dumps(completed_trial_rows(study), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-split")
    _common(prepare)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--max-records", type=int)
    prepare.set_defaults(func=command_prepare_split)

    validate = subparsers.add_parser("validate-split")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.set_defaults(func=command_validate_split)

    for name, function in (
        ("create-study", command_create_study),
        ("inspect", command_inspect),
    ):
        child = subparsers.add_parser(name)
        _common(child)
        child.add_argument(
            "--stage",
            choices=(STAGE_GRAPH_PRETRAIN, STAGE_GRAPH_TEXT_ALIGNMENT),
            required=True,
        )
        child.add_argument("--storage")
        child.set_defaults(func=function)

    worker = subparsers.add_parser("worker")
    _common(worker)
    worker.add_argument(
        "--stage",
        choices=(STAGE_GRAPH_PRETRAIN, STAGE_GRAPH_TEXT_ALIGNMENT),
        required=True,
    )
    worker.add_argument("--storage")
    worker.add_argument("--split-manifest", type=Path, required=True)
    worker.add_argument("--promotion-manifest", type=Path)
    worker.add_argument("--trials", type=int, default=1)
    worker.add_argument("--timeout", type=int)
    worker.add_argument("--seed", type=int, default=42)
    worker.add_argument("--device", default="cuda")
    worker.set_defaults(func=command_worker)

    promote = subparsers.add_parser("promote")
    _common(promote)
    promote.add_argument(
        "--stage",
        choices=(STAGE_GRAPH_PRETRAIN, STAGE_GRAPH_TEXT_ALIGNMENT),
        required=True,
    )
    promote.add_argument("--storage")
    promote.add_argument("--top-k", type=int, required=True)
    promote.add_argument(
        "--min-seeds",
        type=int,
        default=1,
        help="Require this many distinct seeds per config and rank by median score.",
    )
    promote.add_argument("--target-stage", choices=HPO_STAGES)
    promote.add_argument("--output", type=Path, required=True)
    promote.set_defaults(func=command_promote)

    confirm = subparsers.add_parser("enqueue-confirmations")
    _common(confirm)
    confirm.add_argument(
        "--stage",
        choices=(STAGE_GRAPH_PRETRAIN, STAGE_GRAPH_TEXT_ALIGNMENT),
        required=True,
    )
    confirm.add_argument("--storage")
    confirm.add_argument("--promotion", type=Path, required=True)
    confirm.add_argument("--seeds", default="17,42,73")
    confirm.set_defaults(func=command_enqueue_confirmations)

    plan = subparsers.add_parser("plan-downstream")
    _common(plan)
    plan.add_argument(
        "--stage",
        choices=(STAGE_MULTIMODAL_SFT, STAGE_MULTIMODAL_GRPO),
        required=True,
    )
    plan.add_argument("--parents", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--seeds", default="42")
    plan.add_argument("--configs-per-parent", type=int)
    plan.add_argument("--alignment-checkpoint")
    plan.set_defaults(func=command_plan_downstream)

    execute = subparsers.add_parser("execute-plan")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--index", type=int)
    execute.add_argument("--dry-run", action="store_true")
    execute.set_defaults(func=command_execute_plan)

    evaluations = subparsers.add_parser("promote-evaluations")
    evaluations.add_argument("--evaluations", type=Path, required=True)
    evaluations.add_argument("--stage", choices=HPO_STAGES, required=True)
    evaluations.add_argument("--target-stage", choices=HPO_STAGES)
    evaluations.add_argument("--study-name", required=True)
    evaluations.add_argument("--metric", default="heldout_pass_at_1")
    evaluations.add_argument(
        "--direction",
        choices=("maximize", "minimize"),
        default="maximize",
    )
    evaluations.add_argument("--top-k", type=int, required=True)
    evaluations.add_argument("--min-seeds", type=int, default=1)
    evaluations.add_argument("--output", type=Path, required=True)
    evaluations.set_defaults(func=command_promote_evaluations)

    export = subparsers.add_parser("export-best")
    export.add_argument("--promotion", type=Path, required=True)
    export.add_argument("--rank", type=int, default=1)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(func=command_export_best)

    mock = subparsers.add_parser("mock-study")
    _common(mock)
    mock.add_argument(
        "--stage",
        choices=(STAGE_GRAPH_PRETRAIN, STAGE_GRAPH_TEXT_ALIGNMENT),
        default=STAGE_GRAPH_PRETRAIN,
    )
    mock.add_argument("--storage")
    mock.add_argument("--trials", type=int, default=4)
    mock.add_argument("--seed", type=int, default=42)
    mock.set_defaults(func=command_mock_study)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
