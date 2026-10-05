"""Stage A: pretrain the target-fault-conditioned circuit/DAG encoder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from atpgllm.graph.checkpoints import (
    STAGE_GRAPH_PRETRAIN,
    load_stage_checkpoint,
    save_stage_checkpoint,
)
from atpgllm.graph.circuit_sampling import sampled_records
from atpgllm.graph.dataset import (
    ASAP7GraphPretrainDataset,
    collate_graph_pretrain_batch,
)
from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.hpo.splits import DesignSplitManifest
from atpgllm.graph.pretrain import GraphPretrainingModel
from atpgllm.graph.stage_a_eval import (
    build_validation_set,
    evaluate_stage_a,
    rule_baseline,
)
from atpgllm.training._paths import resolve_sim_config_path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-config", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        default="chrivasileiou/asap7-language-of-test-v2",
    )
    parser.add_argument("--split", default="train")
    held_out = parser.add_mutually_exclusive_group(required=True)
    held_out.add_argument(
        "--split-manifest",
        type=Path,
        help="Design-disjoint split (search_graph_pipeline prepare-split) of "
        "--split; trains on its 'train' designs, evaluates on 'validation'.",
    )
    held_out.add_argument(
        "--eval-split",
        help="A dataset split that is already circuit-disjoint from --split "
        "(e.g. 'validation' in asap7-language-of-test-v3).",
    )
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-graphs-per-bucket", type=int, default=200)
    parser.add_argument("--eval-faults-per-design", type=int, default=2)
    parser.add_argument("--eval-scan-records", type=int, default=60_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--revision", help="Dataset revision (commit) to pin.")
    parser.add_argument(
        "--load",
        choices=("streaming", "parquet"),
        default="streaming",
        help="parquet: sample rows at random from the split's cached parquet "
        "files and shuffle them globally; needed when rows are grouped by "
        "circuit (v3). Requires a circuit_id column and --eval-split.",
    )
    parser.add_argument(
        "--max-rows-per-circuit",
        type=int,
        default=0,
        help="parquet only: at most this many random training rows per "
        "circuit (0 = all), so a few huge circuits do not dominate.",
    )
    parser.add_argument(
        "--eval-rows-per-circuit",
        type=int,
        default=20,
        help="parquet only: random validation rows read per circuit before "
        "the size-bucket and per-design caps apply.",
    )
    parser.add_argument("--shuffle-buffer", type=int, default=2_048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gin-hidden", type=int, default=256)
    parser.add_argument("--gin-layers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    sim_path = resolve_sim_config_path(args.sim_config)
    gate_funcs = json.loads(sim_path.read_text())["gate_funcs"]
    vocab = GateAttributeVocab(gate_funcs)
    model = GraphPretrainingModel(
        vocab,
        gin_hidden_dim=args.gin_hidden,
        gin_num_layers=args.gin_layers,
    )
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    step = 0
    parent_stage = None
    if args.resume:
        checkpoint = load_stage_checkpoint(
            args.resume,
            expected_stages=[STAGE_GRAPH_PRETRAIN],
            vocab=vocab,
            expected_architecture=model.architecture_config,
        )
        model.graph_encoder.load_state_dict(checkpoint["states"]["graph_encoder"])
        model.heads.load_state_dict(checkpoint["states"]["pretraining_heads"])
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
        parent_stage = STAGE_GRAPH_PRETRAIN

    if args.load == "parquet" and args.split_manifest:
        raise ValueError("--load parquet uses the dataset's own splits; pass --eval-split")
    if args.max_rows_per_circuit and args.load != "parquet":
        raise ValueError("--max-rows-per-circuit needs --load parquet")

    def stream(split: str, buffer_size: int):
        return load_dataset(
            args.dataset, split=split, streaming=True, revision=args.revision
        ).shuffle(seed=args.seed, buffer_size=buffer_size)

    if args.split_manifest:
        manifest = DesignSplitManifest.load(args.split_manifest)
        if (manifest.dataset, manifest.source_split) != (args.dataset, args.split):
            raise ValueError(
                f"Split manifest is for {manifest.dataset}:{manifest.source_split}, "
                f"not {args.dataset}:{args.split}."
            )
        validation_records = manifest.filter_records(
            stream(args.split, 10_000), "validation"
        )
        train_records = manifest.filter_records(
            stream(args.split, args.shuffle_buffer), "train"
        )
    elif args.load == "parquet":
        validation_records = sampled_records(
            args.dataset,
            args.eval_split,
            revision=args.revision,
            per_circuit=args.eval_rows_per_circuit,
            total=None,
            seed=args.seed,
        )
        # The rows this run will consume, plus 20% for unlabelled records.
        needed = int(1.2 * args.max_steps * args.batch_size * args.grad_accum)
        train_records = sampled_records(
            args.dataset,
            args.split,
            revision=args.revision,
            per_circuit=args.max_rows_per_circuit,
            total=needed,
            seed=args.seed,
            repeat=True,
        )
    else:
        validation_records = stream(args.eval_split, 10_000)
        train_records = stream(args.split, args.shuffle_buffer)

    started = time.time()
    validation_items, validation_rejections, scanned = build_validation_set(
        validation_records,
        gate_funcs,
        vocab,
        per_bucket=args.eval_graphs_per_bucket,
        per_design=args.eval_faults_per_design,
        scan_limit=args.eval_scan_records,
    )
    if not validation_items:
        raise ValueError("No labelled validation graphs were found.")
    baseline = rule_baseline(validation_items)
    (args.output_dir / "rule_baseline.json").write_text(json.dumps(baseline, indent=2))
    print(
        f"validation graphs={len(validation_items)} "
        f"by_size={ {k: v['graphs'] for k, v in baseline['by_size'].items()} } "
        f"scanned={scanned} rejected={dict(validation_rejections)} "
        f"seconds={time.time() - started:.0f}",
        flush=True,
    )
    print("rule_baseline " + json.dumps(baseline), flush=True)

    dataset = ASAP7GraphPretrainDataset(train_records, gate_funcs, vocab)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_graph_pretrain_batch,
    )

    use_amp = (
        not args.fp32
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    optimizer.zero_grad(set_to_none=True)

    def save(name: str) -> None:
        save_stage_checkpoint(
            args.output_dir / name,
            stage=STAGE_GRAPH_PRETRAIN,
            parent_stage=parent_stage,
            vocab=vocab,
            architecture=model.architecture_config,
            states={
                "graph_encoder": model.graph_encoder.state_dict(),
                "pretraining_heads": model.heads.state_dict(),
            },
            optimizer_state=optimizer.state_dict(),
            step=step,
            session=vars(args),
        )

    best_score = None

    def evaluate() -> None:
        nonlocal best_score
        metrics = evaluate_stage_a(
            model,
            validation_items,
            device=device,
            batch_size=args.batch_size,
            baseline=baseline,
        )
        record = {"step": step, **metrics, "rejected": dict(dataset.rejections)}
        print("eval " + json.dumps(record), flush=True)
        with (args.output_dir / "metrics.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        score = metrics["score"]
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            save("graph_pretrain_best.pt")

    micro_step = 0
    for graph in loader:
        graph = graph.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            losses = model.compute_losses(graph)
            loss = losses.total / args.grad_accum
        loss.backward()
        micro_step += 1
        if micro_step % args.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            print(
                f"step={step} total={losses.total.item():.4f} "
                f"propagation={losses.propagation.item():.4f} "
                f"backtrack={losses.backtrack.item():.4f} "
                f"discrepancy={losses.discrepancy.item():.4f}",
                flush=True,
            )
        if step % args.save_every == 0:
            save(f"graph_pretrain_step{step}.pt")
        if step % args.eval_every == 0:
            evaluate()
        if step >= args.max_steps:
            break

    if step % args.eval_every:
        evaluate()
    save("graph_pretrain_final.pt")


if __name__ == "__main__":
    main()
