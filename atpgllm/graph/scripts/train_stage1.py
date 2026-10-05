"""
Stage 1 training entry point: BRIDGES-style graph-text pre-training on
``chrivasileiou/asap7-language-of-test``.

Loads a versioned Stage-A DAG checkpoint, then trains
``Stage1GraphTextModel`` with the combined GTC + GTM + GTG loss. The DAG
encoder is frozen by default; Q-Former and graph/text projections train.

Example
-------

Activate the project env (alias: ``activate``), then::

    python -m atpgllm.graph.scripts.train_stage1 \\
        --sim-config atpgllm/training/data/sim_config.json \\
        --graph-ckpt runs/graph_pretrain/graph_pretrain_final.pt \\
        --output-dir checkpoints/graph_stage1 \\
        --per-device-train-batch-size 4 --grad-accum 4 --max-steps 50000 --lr 1e-4

The script streams the dataset, so no full-dataset download is
required. For datasets stored grouped by circuit (asap7-language-of-test-v3)
use ``--load parquet``: rows are sampled at random from the cached parquet
files, at most ``--max-rows-per-circuit`` per circuit.

With ``--eval-split`` the model is scored on held-out (graph, caption) pairs
at step 0 and every ``--eval-every`` steps (``metrics.jsonl``; the best by
``--select-metric`` is saved as ``stage1_best.pt``): retrieval across
circuits and among the faults of one circuit, matching AP and generation
loss. ``--graph-init random`` builds the same encoder with random weights, the
control arm for asking whether Stage-A pretraining helps.

By default training uses **full bfloat16** on CUDA when supported; pass
``--fp32`` for float32 everywhere.

Optional Weights & Biases: set ``--wandb-project NAME`` (or ``WANDB_PROJECT``)
to log per-step losses (``train/loss_*``, ``train/loss_weighted``) and final
summaries; optional ``WANDB_ENTITY``, ``WANDB_GROUP``, ``WANDB_TAGS`` match the
CLI flags when those are omitted. Install with ``pip install wandb`` or
``pip install 'atpgllm[graph-track]'``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from atpgllm.graph import (
    ASAP7DesignDataset,
    ASAP7GraphTextDataset,
    GateAttributeVocab,
    Stage1GraphTextModel,
    Stage1Trainer,
    collate_graph_text_batch,
)
from atpgllm.graph.circuit_sampling import sampled_records
from atpgllm.graph.hpo.validation import evaluate_graph_text_alignment
from atpgllm.graph.stage_b_eval import build_alignment_validation_set
from atpgllm.graph.train_stage1 import LossWeights
from atpgllm.graph.checkpoints import (
    STAGE_GRAPH_PRETRAIN,
    STAGE_GRAPH_TEXT_ALIGNMENT,
    load_stage_checkpoint,
    save_stage_checkpoint,
)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _wandb_env_str(name: str) -> str | None:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return None
    return str(v).strip()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph-text Stage 1 trainer")
    p.add_argument("--sim-config", type=Path, required=True,
                   help="Path to sim_config.json with gate_funcs dict.")
    p.add_argument("--dataset", type=str, default="chrivasileiou/asap7-language-of-test")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--revision", help="Dataset revision (commit) to pin.")
    p.add_argument(
        "--load",
        choices=("streaming", "parquet"),
        default="streaming",
        help="parquet: sample rows at random from the split's cached parquet "
             "files (needed when rows are grouped by circuit, as in v3).",
    )
    p.add_argument(
        "--max-rows-per-circuit",
        type=int,
        default=0,
        help="parquet only: at most this many random training rows per "
             "circuit (0 = all).",
    )
    p.add_argument(
        "--eval-split",
        help="Held-out split, circuit-disjoint from --split (v3: 'validation'). "
             "Enables evaluation at step 0 and every --eval-every steps.",
    )
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument(
        "--eval-rows-per-circuit",
        type=int,
        default=20,
        help="parquet only: random held-out rows read per circuit.",
    )
    p.add_argument(
        "--eval-faults-per-circuit",
        type=int,
        default=6,
        help="Held-out pairs per circuit, one per distinct fault.",
    )
    p.add_argument("--eval-pairs", type=int, default=1024,
                   help="At most this many held-out pairs.")
    p.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Seed of the held-out sample, separate from --seed so runs with "
             "different training seeds score the same pairs.",
    )
    p.add_argument(
        "--select-metric",
        default="semantic_score",
        help="Held-out metric (higher is better) that picks stage1_best.pt.",
    )
    p.add_argument(
        "--descriptions-json",
        type=Path,
        default=None,
        help="If set, use the precomputed per-design descriptions JSON "
             "(produced by precompute_design_descriptions.py) as the "
             "Stage-1 (graph, text) source instead of streaming the raw HF "
             "dataset. This is the BRIDGES-faithful 'one graph -> one "
             "description' pairing.",
    )
    p.add_argument("--text-model", type=str, default="answerdotai/ModernBERT-base")
    p.add_argument("--output-dir", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--graph-ckpt",
        type=Path,
        help="Stage-A graph_pretrain checkpoint: the DAG encoder's architecture "
             "and (with --graph-init pretrained) its weights.",
    )
    source.add_argument(
        "--resume",
        type=Path,
        help="Resume a graph_text_alignment checkpoint including optimizer state.",
    )

    p.add_argument(
        "--graph-init",
        choices=("pretrained", "random"),
        default="pretrained",
        help="random: same encoder architecture as --graph-ckpt, random "
             "weights (control arm).",
    )
    p.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=16,
        help="Micro-batch size per optimizer accumulation step (per device).",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        metavar="N",
        help="Gradient accumulation steps; optimizer effective batch is "
             "per_device_train_batch_size * N (same pattern as train_stage2). "
             "Loss is scaled by 1/N each micro-step. GTC/GTM still use each "
             "forward's micro-batch for in-batch negatives only.",
    )
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--shuffle-buffer", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--max-text-len", type=int, default=4096)
    p.add_argument("--max-answer-len", type=int, default=4096)
    p.add_argument("--num-queries", type=int, default=32)
    p.add_argument("--qformer-layers", type=int, default=6)
    p.add_argument("--qformer-hidden", type=int, default=512)
    p.add_argument("--proj-dim", type=int, default=512)

    p.add_argument("--w-gtc", type=float, default=1.0)
    p.add_argument("--w-gtm", type=float, default=1.0)
    p.add_argument("--w-gtg", type=float, default=1.0)

    p.add_argument("--freeze-text", action="store_true",
                   help="Freeze ModernBERT weights (train only projectors + Q-Former + graph).")
    p.add_argument(
        "--graph-policy",
        choices=("frozen", "last_layer", "full"),
        default="frozen",
        help="DAG encoder policy for alignment. Q-Former/projectors remain trainable.",
    )
    p.add_argument(
        "--fp32",
        action="store_true",
        help="Disable bfloat16: train in float32 (default is full bf16 on CUDA "
             "when torch.cuda.is_bf16_supported(), else float32).",
    )
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--metrics-json", type=Path, default=None,
                   help="If set, write training summary metrics to this path (JSON).")
    p.add_argument("--metrics-window", type=int, default=20,
                   help="With --metrics-json, average weighted loss over the last N "
                        "log intervals (default 20).")
    p.add_argument("--no-save", action="store_true",
                   help="Skip all checkpoint writes (for short search / smoke runs).")

    p.add_argument(
        "--wandb-project",
        type=str,
        default=_wandb_env_str("WANDB_PROJECT"),
        help="If set, log metrics to Weights & Biases (requires: pip install wandb). "
        "Default: WANDB_PROJECT env if set.",
    )
    p.add_argument(
        "--wandb-entity",
        type=str,
        default=_wandb_env_str("WANDB_ENTITY"),
        help="W&B entity (team). Default: WANDB_ENTITY env if set.",
    )
    p.add_argument(
        "--wandb-group",
        type=str,
        default=_wandb_env_str("WANDB_GROUP"),
        help="W&B group (e.g. one HPO sweep); child runs share the same group. "
        "Default: WANDB_GROUP env if set.",
    )
    p.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name.")
    p.add_argument(
        "--wandb-tags",
        type=str,
        default=_wandb_env_str("WANDB_TAGS") or "",
        help="Comma-separated W&B tags. Default: WANDB_TAGS env if set.",
    )
    p.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
        help="W&B mode.",
    )
    return p.parse_args()


def _set_graph_policy(model: Stage1GraphTextModel, policy: str) -> None:
    for parameter in model.graph_encoder.parameters():
        parameter.requires_grad = policy == "full"
    if policy == "last_layer":
        for parameter in model.graph_encoder.dag_gin.layers[-1].parameters():
            parameter.requires_grad = True
        if model.graph_encoder.dag_gin.jk_proj is not None:
            for parameter in model.graph_encoder.dag_gin.jk_proj.parameters():
                parameter.requires_grad = True


def _wandb_config_dict(args: argparse.Namespace) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in vars(args).items():
        if k.startswith("wandb_"):
            continue
        if isinstance(v, Path):
            d[k] = str(v)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            d[k] = v
        else:
            d[k] = repr(v)
    return d


def _maybe_wandb_init(args: argparse.Namespace) -> Any:
    if not args.wandb_project:
        return None
    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "wandb is not installed. Run: pip install wandb   "
            "or: pip install 'atpgllm[graph-track]'"
        ) from e

    tags = [t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_run_name,
        tags=tags or None,
        config=_wandb_config_dict(args),
        mode=args.wandb_mode,
    )
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    return run


def _wandb_finish(run: Any) -> None:
    if run is None:
        return
    try:
        import wandb  # type: ignore[import-not-found]

        wandb.finish()
    except Exception:
        pass


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def _model_state(model: Stage1GraphTextModel, freeze_text: bool) -> dict[str, Any]:
    """The model's state; without the frozen text encoder, which reloads
    from ``text_model_name`` (Stage C reads only the graph encoder and
    Q-Former)."""
    state = model.state_dict()
    if freeze_text:
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith("text_encoder.")
        }
    return state


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def main() -> None:
    args = _parse_args()
    # The tokenizer runs in the main process (held-out pairs) before
    # DataLoader workers fork; its thread pool is not fork-safe.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.resume is not None and args.graph_init != "pretrained":
        raise ValueError("--graph-init applies to --graph-ckpt, not --resume.")

    # --- Load gate_funcs + vocab ---
    with args.sim_config.open() as fh:
        sim_config = json.load(fh)
    gate_funcs = sim_config["gate_funcs"]
    vocab = GateAttributeVocab(gate_funcs)
    vocab.report_collapse()

    # --- Tokenizer (shared by text encoder and GTG decoder) ---
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Dataset ---
    if args.descriptions_json is not None:
        dataset = ASAP7DesignDataset(
            descriptions_path=args.descriptions_json,
            gate_funcs=gate_funcs,
            vocab=vocab,
            tokenizer=tokenizer,
            shuffle=True,
            seed=args.seed,
            repeat=True,
        )
        print(
            f"Stage-1 source: per-design descriptions "
            f"({len(dataset)} unique netlists from {args.descriptions_json})",
            flush=True,
        )
    else:
        if args.load == "parquet":
            # The rows this run will consume, plus 20% for rejected records.
            needed = int(
                1.2 * args.max_steps * args.per_device_train_batch_size * args.grad_accum
            )
            records = sampled_records(
                args.dataset,
                args.split,
                revision=args.revision,
                per_circuit=args.max_rows_per_circuit,
                total=needed,
                seed=args.seed,
                repeat=True,
                columns=None,
            )
        else:
            records = load_dataset(
                args.dataset, split=args.split, streaming=True, revision=args.revision
            ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
        dataset = ASAP7GraphTextDataset(
            hf_stream=records,
            gate_funcs=gate_funcs,
            vocab=vocab,
            tokenizer=tokenizer,
            max_text_len=args.max_text_len,
            max_answer_len=args.max_answer_len,
            shard_by_worker=args.load == "parquet",
        )
        print(
            f"Stage-1 source: {args.load} {args.dataset!r} split={args.split!r} "
            f"(per-record captions)",
            flush=True,
        )

    def collate(items):
        return collate_graph_text_batch(
            items,
            tokenizer=tokenizer,
            max_text_len=args.max_text_len,
            max_answer_len=args.max_answer_len,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
    )

    # --- Held-out pairs ---
    validation_items: list[dict[str, Any]] = []
    if args.eval_split:
        if args.load == "parquet":
            eval_records = sampled_records(
                args.dataset,
                args.eval_split,
                revision=args.revision,
                per_circuit=args.eval_rows_per_circuit,
                total=None,
                seed=args.eval_seed,
                columns=None,
            )
        else:
            eval_records = load_dataset(
                args.dataset, split=args.eval_split, streaming=True,
                revision=args.revision,
            )
        eval_dataset = ASAP7GraphTextDataset(
            hf_stream=eval_records,
            gate_funcs=gate_funcs,
            vocab=vocab,
            tokenizer=tokenizer,
            max_text_len=args.max_text_len,
            max_answer_len=args.max_answer_len,
        )
        validation_items = build_alignment_validation_set(
            eval_dataset,
            per_circuit=args.eval_faults_per_circuit,
            limit=args.eval_pairs,
        )
        circuits = {item["circuit"] for item in validation_items}
        print(
            f"validation pairs={len(validation_items)} circuits={len(circuits)} "
            f"rejected={dict(eval_dataset.rejections)}",
            flush=True,
        )
        if len(validation_items) < 2:
            raise ValueError("Stage-B evaluation needs at least two held-out pairs.")
    validation_groups = [item["circuit"] for item in validation_items]

    # --- Model ---
    # Seeded again so both --graph-init arms build every module from the
    # same random state; only the graph encoder's weights then differ.
    torch.manual_seed(args.seed)
    resume_payload = None
    if args.graph_ckpt is not None:
        graph_checkpoint = load_stage_checkpoint(
            args.graph_ckpt,
            expected_stages=[STAGE_GRAPH_PRETRAIN],
            vocab=vocab,
        )
        graph_config = graph_checkpoint["architecture"]["graph_encoder"]
    else:
        resume_payload = load_stage_checkpoint(
            args.resume,
            expected_stages=[STAGE_GRAPH_TEXT_ALIGNMENT],
            vocab=vocab,
        )
        graph_config = resume_payload["architecture"]["graph_encoder"]
    model = Stage1GraphTextModel(
        vocab=vocab,
        text_model_name=args.text_model,
        graph_encoder_config=graph_config,
        qformer_hidden_dim=args.qformer_hidden,
        qformer_layers=args.qformer_layers,
        num_queries=args.num_queries,
        proj_dim=args.proj_dim,
        freeze_text=args.freeze_text,
    )
    if resume_payload is not None:
        missing, unexpected = model.load_state_dict(
            resume_payload["states"]["model"], strict=False
        )
        if unexpected or any(not key.startswith("text_encoder.") for key in missing):
            raise ValueError(
                f"Resume state does not match the model: missing={missing[:5]} "
                f"unexpected={unexpected[:5]}"
            )
    elif args.graph_init == "pretrained":
        model.graph_encoder.load_state_dict(
            graph_checkpoint["states"]["graph_encoder"],
            strict=True,
        )
    print(f"Graph encoder: {args.graph_init if resume_payload is None else 'resumed'}",
          flush=True)
    _set_graph_policy(model, args.graph_policy)

    trainer = Stage1Trainer(
        model=model,
        vocab_size=len(tokenizer),
        weights=LossWeights(gtc=args.w_gtc, gtm=args.w_gtm, gtg=args.w_gtg),
        grad_clip=args.grad_clip,
        device=args.device,
        gtg_max_seq_len=args.max_answer_len,
        use_bf16=not args.fp32,
    )
    dev = torch.device(args.device)
    if not args.fp32 and trainer.param_dtype is torch.float32 and dev.type == "cuda":
        print(
            "Note: bfloat16 is not supported on this CUDA device; training in float32.",
            file=sys.stderr,
        )
    elif trainer.param_dtype == torch.bfloat16:
        print("Training dtype: bfloat16 (model + GTM/GTG heads).")

    trainable = [
        p
        for p in chain(
            trainer.model.parameters(),
            trainer.gtm_loss.parameters(),
            trainer.gtg_loss.parameters(),
        )
        if p.requires_grad
    ]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(
        p.numel()
        for p in chain(
            trainer.model.parameters(),
            trainer.gtm_loss.parameters(),
            trainer.gtg_loss.parameters(),
        )
    )
    print(f"Trainable params: {n_train:,} / {n_total:,} "
          f"({100.0 * n_train / n_total:.2f}%)")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if resume_payload is not None:
        trainer.gtm_loss.load_state_dict(
            resume_payload["states"]["gtm_loss"], strict=True
        )
        trainer.gtg_loss.load_state_dict(
            resume_payload["states"]["gtg_loss"], strict=True
        )
        if resume_payload.get("optimizer"):
            optimizer.load_state_dict(resume_payload["optimizer"])

    wb_run = None
    try:
        wb_run = _maybe_wandb_init(args)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1) from e

    parent_stage = (
        STAGE_GRAPH_TEXT_ALIGNMENT if resume_payload is not None else STAGE_GRAPH_PRETRAIN
    )

    def save(path: Path, step: int) -> None:
        save_stage_checkpoint(
            path,
            stage=STAGE_GRAPH_TEXT_ALIGNMENT,
            parent_stage=parent_stage,
            vocab=vocab,
            architecture=model.architecture_config,
            states={
                "model": _model_state(trainer.model, args.freeze_text),
                "gtm_loss": trainer.gtm_loss.state_dict(),
                "gtg_loss": trainer.gtg_loss.state_dict(),
            },
            optimizer_state=optimizer.state_dict(),
            step=step,
            session=vars(args),
        )

    metrics_path = args.output_dir / "metrics.jsonl"
    best_score: float | None = None

    def evaluate(step: int) -> None:
        nonlocal best_score
        validation_loader = DataLoader(
            validation_items,
            batch_size=args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        metrics = evaluate_graph_text_alignment(
            trainer,
            validation_loader,
            max_batches=len(validation_loader),
            groups=validation_groups,
        )
        row: dict[str, Any] = {"step": step, **metrics}
        if args.num_workers == 0:
            # DataLoader workers count rejections in their own processes.
            row["train_rejected"] = dict(getattr(dataset, "rejections", {}))
        _append_jsonl(metrics_path, row)
        print("eval " + json.dumps(row), flush=True)
        if wb_run is not None:
            import wandb  # type: ignore[import-not-found]

            wandb.log({
                "train/step": step,
                **{
                    f"eval/{key}": value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                },
            })
        score = metrics.get(args.select_metric)
        if score is None:
            raise ValueError(f"--select-metric {args.select_metric!r} is undefined.")
        if step > 0 and not args.no_save and (best_score is None or score > best_score):
            best_score = float(score)
            save(args.output_dir / "stage1_best.pt", step)
            print(f"New best {args.select_metric}={score:.4f} at step {step}", flush=True)

    # --- Train ---
    step = int(resume_payload["step"]) if resume_payload is not None else 0
    track_metrics = args.metrics_json is not None
    history_window = track_metrics or (wb_run is not None)
    weighted_history: deque[float] = deque(
        maxlen=max(1, args.metrics_window)
    ) if history_window else deque()
    last_components: dict[str, float] | None = None
    accum_index = 0
    evaluated_at: int | None = None
    try:
        if validation_items and resume_payload is None:
            evaluate(0)
            evaluated_at = 0
        for batch in loader:
            losses = trainer.train_step(
                batch,
                optimizer,
                accum_index=accum_index,
                grad_accum_steps=args.grad_accum,
            )
            accum_index = (accum_index + 1) % args.grad_accum
            if accum_index != 0:
                continue
            step += 1

            if step % args.log_every == 0:
                d = losses.as_dict()
                last_components = d
                total = d["gtc"] + d["gtm"] + d["gtg"]
                wsum = (
                    args.w_gtc * d["gtc"]
                    + args.w_gtm * d["gtm"]
                    + args.w_gtg * d["gtg"]
                )
                if history_window:
                    weighted_history.append(wsum)
                print(
                    f"step={step:>6} "
                    f"total={total:.4f} "
                    f"gtc={d['gtc']:.4f} "
                    f"gtm={d['gtm']:.4f} "
                    f"gtg={d['gtg']:.4f}",
                    flush=True,
                )
                if wb_run is not None:
                    import wandb  # type: ignore[import-not-found]

                    wandb.log(
                        {
                            "train/step": step,
                            "train/loss_gtc": d["gtc"],
                            "train/loss_gtm": d["gtm"],
                            "train/loss_gtg": d["gtg"],
                            "train/loss_unweighted_sum": total,
                            "train/loss_weighted": wsum,
                        }
                    )

            if validation_items and step % args.eval_every == 0:
                evaluate(step)
                evaluated_at = step

            if not args.no_save and step % args.save_every == 0:
                ckpt_path = args.output_dir / f"stage1_step{step}.pt"
                save(ckpt_path, step)
                print(f"Saved {ckpt_path}")

            if step >= args.max_steps:
                break

        if validation_items and evaluated_at != step:
            evaluate(step)
        if args.num_workers == 0 and getattr(dataset, "rejections", None):
            print(f"Training records rejected: {dict(dataset.rejections)}", flush=True)

        wh = list(weighted_history)
        mean_w = sum(wh) / len(wh) if wh else None

        if args.metrics_json is not None:
            args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "step": step,
                "weights": {
                    "w_gtc": args.w_gtc,
                    "w_gtm": args.w_gtm,
                    "w_gtg": args.w_gtg,
                },
                "mean_weighted_loss_last_window": mean_w,
                "metrics_window": args.metrics_window,
                "n_weighted_samples": len(wh),
                "last_components": last_components,
            }
            args.metrics_json.write_text(json.dumps(payload, indent=2))
            print(f"Wrote metrics to {args.metrics_json}")

        if wb_run is not None:
            import wandb  # type: ignore[import-not-found]

            summary: dict[str, Any] = {
                "final/step": step,
                "final/trainable_params": n_train,
                "final/total_params": n_total,
            }
            if last_components is not None:
                summary["final/loss_gtc"] = last_components["gtc"]
                summary["final/loss_gtm"] = last_components["gtm"]
                summary["final/loss_gtg"] = last_components["gtg"]
            if wh:
                summary["final/mean_weighted_loss_window"] = sum(wh) / len(wh)
                summary["final/min_weighted_loss_window"] = min(wh)
                summary["final/max_weighted_loss_window"] = max(wh)
            wandb.log(summary)
            wandb.summary["w_gtc"] = args.w_gtc
            wandb.summary["w_gtm"] = args.w_gtm
            wandb.summary["w_gtg"] = args.w_gtg
            wandb.summary["final_step"] = step
            if mean_w is not None:
                wandb.summary["mean_weighted_loss_last_window"] = mean_w

        # Final save
        if not args.no_save:
            final = args.output_dir / "stage1_final.pt"
            save(final, step)
            print(f"Final checkpoint: {final}")
    finally:
        _wandb_finish(wb_run)


if __name__ == "__main__":
    main()
