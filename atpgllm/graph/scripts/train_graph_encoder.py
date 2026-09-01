"""Stage A: pretrain the target-fault-conditioned circuit/DAG encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from atpgllm.graph.checkpoints import (
    STAGE_GRAPH_PRETRAIN,
    load_stage_checkpoint,
    save_stage_checkpoint,
)
from atpgllm.graph.dataset import (
    ASAP7GraphPretrainDataset,
    collate_graph_pretrain_batch,
)
from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.pretrain import GraphPretrainingModel
from atpgllm.training._paths import resolve_sim_config_path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-config", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        default="chrivasileiou/asap7-language-of-test-v2",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=20)
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

    stream = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    dataset = ASAP7GraphPretrainDataset(stream, gate_funcs, vocab)
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
        if step >= args.max_steps:
            break

    save("graph_pretrain_final.pt")


if __name__ == "__main__":
    main()
