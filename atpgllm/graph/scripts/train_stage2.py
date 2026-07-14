"""
Stage 2 training entry point: fine-tune a graph-text-to-text model on
``chrivasileiou/asap7-language-of-test``.

Architecture::

    netlist -> graph encoder (AttributeDecompositionEncoder + DAGGINEncoder)
            -> Q-Former -> Linear projection
            -> soft-prompt tokens prepended to the LLM input

    user_content (text) -> tokenizer -> LLM -> answer_content (SFT target)

Graph encoder + Q-Former are initialised from a Stage-1 checkpoint and
frozen by default. The projector is always trainable. The LLM may be
trained fully (if VRAM allows) or with LoRA (recommended).

Example
-------

::

    python -m atpgllm.graph.scripts.train_stage2 \\
        --sim-config tests/sim_config.json \\
        --stage1-ckpt checkpoints/graph_stage1/stage1_final.pt \\
        --llm Qwen/Qwen2.5-7B-Instruct \\
        --output-dir checkpoints/graph_stage2 \\
        --batch-size 1 --grad-accum 16 --max-steps 20000 --lr 5e-5 \\
        --use-lora
"""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from atpgllm.graph import (
    GateAttributeVocab,
    Stage1GraphTextModel,
    Stage2GraphTextLM,
    build_stage2_inputs,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph-text-to-text Stage 2 trainer")
    p.add_argument("--sim-config", type=Path, required=True)
    p.add_argument("--dataset", type=str, default="chrivasileiou/asap7-language-of-test")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--stage1-ckpt", type=Path, required=True)
    p.add_argument("--llm", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--text-model", type=str, default="answerdotai/ModernBERT-base",
                   help="Must match the text model used in Stage 1.")
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--shuffle-buffer", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--num-queries", type=int, default=32)
    p.add_argument("--qformer-layers", type=int, default=6)
    p.add_argument("--qformer-hidden", type=int, default=512)
    p.add_argument("--gin-layers", type=int, default=6)
    p.add_argument("--gin-hidden", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=512)

    p.add_argument("--freeze-graph", action="store_true", default=True)
    p.add_argument("--unfreeze-graph", dest="freeze_graph", action="store_false")
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--load-4bit", action="store_true",
                   help="Load LLM in 4-bit via bitsandbytes (requires bnb).")

    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _load_llm(args: argparse.Namespace, tokenizer) -> torch.nn.Module:
    kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    llm = AutoModelForCausalLM.from_pretrained(args.llm, **kwargs)
    llm.config.pad_token_id = tokenizer.pad_token_id

    if args.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if args.load_4bit:
            llm = prepare_model_for_kbit_training(llm)
        llm = get_peft_model(
            llm,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            ),
        )
    return llm


def _batched(iterable, n: int):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    with args.sim_config.open() as fh:
        gate_funcs = json.load(fh)["gate_funcs"]
    vocab = GateAttributeVocab(gate_funcs)

    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Stage 1 model ---
    stage1 = Stage1GraphTextModel(
        vocab=vocab,
        text_model_name=args.text_model,
        gin_hidden_dim=args.gin_hidden,
        gin_num_layers=args.gin_layers,
        qformer_hidden_dim=args.qformer_hidden,
        qformer_layers=args.qformer_layers,
        num_queries=args.num_queries,
        proj_dim=args.proj_dim,
        freeze_text=True,  # text encoder unused in Stage 2
    )
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    stage1.load_state_dict(ckpt["model"], strict=False)
    # The text_encoder submodule is unused at Stage 2 — drop it to save memory.
    stage1.text_encoder = torch.nn.Identity()  # type: ignore[assignment]

    llm = _load_llm(args, tokenizer)

    model = Stage2GraphTextLM.from_stage1(
        stage1=stage1,
        llm=llm,
        freeze_graph=args.freeze_graph,
    ).to(args.device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_train:,} / {n_total:,} "
          f"({100.0 * n_train / n_total:.2f}%)")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --- Data ---
    hf_stream = (
        load_dataset(args.dataset, split=args.split, streaming=True)
        .shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    )

    # --- Train ---
    step = 0
    accum_step = 0
    optimizer.zero_grad(set_to_none=True)

    for records in _batched(hf_stream, args.batch_size):
        try:
            batch = build_stage2_inputs(
                records,
                gate_funcs=gate_funcs,
                vocab=vocab,
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len,
            )
        except ValueError:
            # All records failed to parse; skip.
            continue

        batch = {k: (v.to(args.device) if hasattr(v, "to") else v)
                 for k, v in batch.items()}

        out = model(**batch)
        loss = out.loss / args.grad_accum
        loss.backward()
        accum_step += 1

        if accum_step < args.grad_accum:
            continue

        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        step += 1

        if step % args.log_every == 0:
            print(f"step={step:>6} loss={out.loss.item():.4f}")

        if step % args.save_every == 0:
            ckpt_path = args.output_dir / f"stage2_step{step}.pt"
            torch.save(
                {
                    "projector": model.graph_to_llm.state_dict(),
                    "stage1": stage1.state_dict(),
                    "step": step,
                    "args": vars(args),
                },
                ckpt_path,
            )
            if args.use_lora:
                model.llm.save_pretrained(args.output_dir / f"lora_step{step}")
            print(f"Saved {ckpt_path}")

        if step >= args.max_steps:
            break

    torch.save(
        {"projector": model.graph_to_llm.state_dict(), "stage1": stage1.state_dict()},
        args.output_dir / "stage2_final.pt",
    )
    if args.use_lora:
        model.llm.save_pretrained(args.output_dir / "lora_final")


if __name__ == "__main__":
    main()
