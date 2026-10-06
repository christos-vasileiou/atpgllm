"""QLoRA SFT/GRPO for the DAG encoder + Q-Former + decoder LM stack.

SFT (Stage C) trains the graph encoder, the Q-Former, the graph-to-LLM
projector and QLoRA adapters together (``--graph-policy full``). With
``--tool-format`` (default) each example is a chat with the fault-simulation
tool: the model reasons, calls the tool, and reads the simulator's answer
before reporting. ``--retry-fraction`` of the examples first show a real
failed call (a nearby pattern that does not detect the fault, confirmed by
simulation) as untrained context, then train the model to read why it failed
from the snapshot and call the tool again with a fixed pattern
(:mod:`atpgllm.training.tool_trajectories`).

``--eval-split`` scores held-out loss (single-call and retry examples
separately); ``--probe-examples`` runs the model on held-out circuits and
simulates its tool calls (:mod:`atpgllm.multimodal.tool_probe`).
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import random
import time
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from atpgllm.graph.checkpoints import (
    STAGE_GRAPH_TEXT_ALIGNMENT,
    STAGE_MULTIMODAL_GRPO,
    STAGE_MULTIMODAL_SFT,
    load_stage_checkpoint,
    save_stage_checkpoint,
)
from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.multimodal import (
    GraphConditionedCausalLM,
    build_graph_prompt_example,
    build_multimodal_sft_batch,
    graph_grpo_loss,
    group_relative_advantages,
    load_aligned_graph_stack,
)
from atpgllm.graph.circuit_sampling import sampled_records
from atpgllm.multimodal.data import build_tool_sft_example, collate_tool_sft
from atpgllm.multimodal.tool_probe import ProbeItem, run_tool_probe
from atpgllm.training._paths import resolve_sim_config_path
from atpgllm.training.tool_trajectories import failed_attempt, record_seed
from atpgllm.training.reward_function_factory import RewardFunctionFactory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("sft", "grpo"))
    parser.add_argument("--sim-config", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        default="chrivasileiou/asap7-language-of-test-v2",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", help="Dataset revision (commit) to pin.")
    parser.add_argument(
        "--load",
        choices=("streaming", "parquet"),
        default="streaming",
        help="parquet: sample rows per circuit from the cached parquet files "
             "(needed for v3, whose rows are grouped by circuit).",
    )
    parser.add_argument("--max-rows-per-circuit", type=int, default=0)
    parser.add_argument(
        "--tool-format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SFT on chats with the fault-simulation tool (default). "
             "--no-tool-format keeps the older format, where the snapshot is "
             "pasted inside the reasoning.",
    )
    parser.add_argument(
        "--retry-fraction",
        type=float,
        default=0.3,
        help="Share of tool-format examples that start with a real failed "
             "call and train the diagnosis and retry.",
    )
    parser.add_argument(
        "--tool-view",
        choices=("po_fault", "full"),
        default="po_fault",
        help="What the fault-simulation tool shows the model: primary outputs "
             "and the fault site (default), or the full Good/Bad table. GRPO "
             "rollouts must use the view the SFT model was trained with.",
    )
    parser.add_argument("--max-flips", type=int, default=3,
                        help="Inputs flipped to make a failed first call.")
    parser.add_argument("--eval-split", help="Held-out split (v3: validation).")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--eval-rows-per-circuit", type=int, default=4)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument(
        "--probe-examples",
        type=int,
        default=16,
        help="Held-out circuits for the tool-use probe (0 = off). Each "
             "probe decodes two assistant turns greedily.",
    )
    parser.add_argument("--probe-every", type=int, default=0,
                        help="Probe every N steps (0 = only after training).")
    parser.add_argument("--probe-max-new-tokens", type=int, default=1024)
    parser.add_argument("--alignment-ckpt", type=Path, required=True)
    parser.add_argument("--sft-ckpt", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--graph-policy",
        choices=("frozen", "qformer", "last_layer", "full"),
        default="full",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--shuffle-buffer", type=int, default=2_048)
    parser.add_argument("--max-seq-len", type=int, default=4_096)
    parser.add_argument("--max-prompt-length", type=int, default=2_048)
    parser.add_argument("--max-completion-length", type=int, default=2_048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--graph-lr",
        type=float,
        default=1e-5,
        help="Learning rate of the pretrained graph encoder and Q-Former. The "
             "new graph-to-LLM projector learns at --lr with the adapters.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--load-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _batched(iterable: Iterable[dict[str, Any]], size: int):
    iterator = iter(iterable)
    while True:
        values = list(islice(iterator, size))
        if not values:
            return
        yield values


def _lora_config(args: argparse.Namespace):
    from peft import LoraConfig

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def _base_llm(args: argparse.Namespace):
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if args.load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return AutoModelForCausalLM.from_pretrained(args.llm, **kwargs)


def _load_sft_llm(
    args: argparse.Namespace,
    adapter_path: Path | None,
):
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

    base = _base_llm(args)
    if args.load_4bit:
        base = prepare_model_for_kbit_training(base)
    if adapter_path is not None:
        return PeftModel.from_pretrained(
            base,
            adapter_path,
            is_trainable=True,
        )
    return get_peft_model(base, _lora_config(args))


def _load_grpo_llm(
    args: argparse.Namespace,
    adapter_path: Path,
    *,
    resume: bool,
):
    from peft import PeftModel, prepare_model_for_kbit_training
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

    base = _base_llm(args)
    if args.load_4bit:
        base = prepare_model_for_kbit_training(base)
    if resume:
        llm = PeftModel.from_pretrained(
            base,
            adapter_path / "reference",
            adapter_name="reference",
            is_trainable=False,
        )
        llm.load_adapter(
            adapter_path / "policy",
            adapter_name="policy",
            is_trainable=True,
        )
    else:
        llm = PeftModel.from_pretrained(
            base,
            adapter_path,
            adapter_name="reference",
            is_trainable=False,
        )
        reference_config = copy.deepcopy(llm.peft_config["reference"])
        reference_config.inference_mode = False
        llm.add_adapter("policy", reference_config)
        reference_state = get_peft_model_state_dict(
            llm,
            adapter_name="reference",
        )
        set_peft_model_state_dict(
            llm,
            reference_state,
            adapter_name="policy",
        )
    llm.set_adapter("policy")
    for name, parameter in llm.named_parameters():
        if ".reference." in name:
            parameter.requires_grad = False
    return llm


def _module_state(model: GraphConditionedCausalLM) -> dict[str, Any]:
    return {
        "graph_encoder": model.graph_encoder.state_dict(),
        "q_former": model.q_former.state_dict(),
        "graph_to_llm": model.graph_to_llm.state_dict(),
    }


def _load_module_state(
    model: GraphConditionedCausalLM,
    states: dict[str, Any],
) -> None:
    model.graph_encoder.load_state_dict(states["graph_encoder"], strict=True)
    model.q_former.load_state_dict(states["q_former"], strict=True)
    model.graph_to_llm.load_state_dict(states["graph_to_llm"], strict=True)


def _architecture(
    model: GraphConditionedCausalLM,
    alignment_checkpoint: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "alignment": alignment_checkpoint["architecture"],
        "multimodal": model.architecture_config,
        "base_model": args.llm,
        "quantization": "nf4-double" if args.load_4bit else "bf16",
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
    }


def _graph_device(model: GraphConditionedCausalLM) -> torch.device:
    return model.llm.get_input_embeddings().weight.device


def _optimizer(
    model: GraphConditionedCausalLM,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    # The encoder and Q-Former come pretrained (Stages A and B) and move
    # slowly; the projector starts from random weights, like the adapters.
    graph_parameters = [
        parameter
        for module in (model.graph_encoder, model.q_former)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    new_parameters = [
        parameter
        for module in (model.graph_to_llm, model.llm)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if not new_parameters or not any(
        p.requires_grad for p in model.llm.parameters()
    ):
        raise RuntimeError(
            "Expected trainable projector and LoRA parameters; "
            "check --graph-policy and adapter loading."
        )
    groups = [{"params": new_parameters, "lr": args.lr}]
    if graph_parameters:
        groups.insert(0, {"params": graph_parameters, "lr": args.graph_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def _save(
    model: GraphConditionedCausalLM,
    optimizer: torch.optim.Optimizer,
    *,
    path: Path,
    adapter_path: Path,
    stage: str,
    parent_stage: str,
    vocab: GateAttributeVocab,
    architecture: dict[str, Any],
    step: int,
    args: argparse.Namespace,
    reference_modules: tuple[torch.nn.Module, ...] | None = None,
) -> None:
    if stage == STAGE_MULTIMODAL_GRPO:
        model.llm.save_pretrained(
            adapter_path,
            selected_adapters=["reference", "policy"],
        )
    else:
        model.llm.save_pretrained(adapter_path)
    states = _module_state(model)
    if reference_modules is not None:
        states.update({
            "reference_graph_encoder": reference_modules[0].state_dict(),
            "reference_q_former": reference_modules[1].state_dict(),
            "reference_graph_to_llm": reference_modules[2].state_dict(),
        })
    save_stage_checkpoint(
        path,
        stage=stage,
        parent_stage=parent_stage,
        vocab=vocab,
        architecture=architecture,
        states=states,
        optimizer_state=optimizer.state_dict(),
        step=step,
        session={
            **vars(args),
            "adapter_path": str(adapter_path.resolve()),
        },
    )


def _load_common(args: argparse.Namespace):
    sim_path = resolve_sim_config_path(args.sim_config)
    gate_funcs = json.loads(sim_path.read_text())["gate_funcs"]
    vocab = GateAttributeVocab(gate_funcs)
    graph_encoder, q_former, alignment = load_aligned_graph_stack(
        args.alignment_ckpt,
        vocab,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return gate_funcs, vocab, graph_encoder, q_former, alignment, tokenizer


def _records(args: argparse.Namespace, split: str, *, per_circuit: int,
             total: int | None, seed: int, repeat: bool) -> Iterable[dict[str, Any]]:
    if args.load == "parquet":
        return sampled_records(
            args.dataset, split, revision=args.revision, per_circuit=per_circuit,
            total=total, seed=seed, repeat=repeat, columns=None,
        )
    stream = load_dataset(args.dataset, split=split, streaming=True, revision=args.revision)
    return stream.shuffle(seed=seed, buffer_size=args.shuffle_buffer) if repeat else stream


def _tool_example(record, args, gate_funcs, vocab, tokenizer, rejections, salt):
    """One SFT example; a retry with probability --retry-fraction. A retry
    that does not fit falls back to the single-call example of the record."""
    rng = random.Random(record_seed(record, salt))
    attempt = None
    if rng.random() < args.retry_fraction:
        attempt = failed_attempt(record, rng, max_flips=args.max_flips)
        if attempt is None:
            rejections["no_failed_attempt"] += 1
    example = None
    if attempt is not None:
        example = build_tool_sft_example(
            record, gate_funcs, vocab, tokenizer, max_seq_len=args.max_seq_len,
            failed_attempt=attempt, rejections=rejections, tool_view=args.tool_view,
        )
    if example is None:
        example = build_tool_sft_example(
            record, gate_funcs, vocab, tokenizer, max_seq_len=args.max_seq_len,
            rejections=rejections, tool_view=args.tool_view,
        )
    return example


def _held_out(args, gate_funcs, vocab, tokenizer):
    """Fixed held-out SFT examples (one per circuit and fault) and probe items."""
    rejections: collections.Counter = collections.Counter()
    examples, probes, seen = [], [], set()
    for record in _records(args, args.eval_split, per_circuit=args.eval_rows_per_circuit,
                           total=None, seed=args.eval_seed, repeat=False):
        key = (record.get("circuit_id"), record.get("fault"))
        if key in seen:
            continue
        seen.add(key)
        if len(examples) < args.eval_examples:
            example = _tool_example(record, args, gate_funcs, vocab, tokenizer,
                                    rejections, salt="eval")
            if example is not None:
                examples.append(example)
                continue
        if len(probes) < args.probe_examples:
            attempt = failed_attempt(record, random.Random(record_seed(record, "probe")),
                                     max_flips=args.max_flips)
            if attempt is not None and build_tool_sft_example(
                record, gate_funcs, vocab, tokenizer, max_seq_len=args.max_seq_len,
                failed_attempt=attempt, tool_view=args.tool_view,
            ) is not None:
                probes.append(ProbeItem(record=record, failed_attempt=attempt))
        if len(examples) >= args.eval_examples and len(probes) >= args.probe_examples:
            break
    return examples, probes, rejections


@torch.no_grad()
def _held_out_loss(model, examples, pad_token_id, device) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    losses: dict[str, list[float]] = {"single": [], "retry": []}
    for example in examples:
        batch = collate_tool_sft([example], pad_token_id)
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        loss = model(**batch, use_cache=False).loss
        losses["retry" if example.retry else "single"].append(float(loss))
    if was_training:
        model.train()
    out: dict[str, Any] = {
        f"heldout_loss_{kind}": (sum(v) / len(v) if v else None)
        for kind, v in losses.items()
    }
    every = losses["single"] + losses["retry"]
    out["heldout_loss"] = sum(every) / len(every) if every else None
    out["heldout_examples"] = {kind: len(v) for kind, v in losses.items()}
    return out


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def train_sft(args: argparse.Namespace) -> None:
    (
        gate_funcs,
        vocab,
        graph_encoder,
        q_former,
        alignment,
        tokenizer,
    ) = _load_common(args)
    resume_payload = None
    adapter_path = args.adapter
    if args.resume:
        resume_payload = load_stage_checkpoint(
            args.resume,
            expected_stages=[STAGE_MULTIMODAL_SFT],
            vocab=vocab,
        )
        adapter_path = adapter_path or Path(resume_payload["session"]["adapter_path"])
    llm = _load_sft_llm(args, adapter_path)
    model = GraphConditionedCausalLM(graph_encoder, q_former, llm)
    model.set_graph_policy(args.graph_policy)
    device = _graph_device(model)
    for module in (model.graph_encoder, model.q_former, model.graph_to_llm):
        module.to(device)
    if resume_payload:
        _load_module_state(model, resume_payload["states"])
    architecture = _architecture(model, alignment, args)
    if resume_payload and resume_payload["architecture"] != architecture:
        raise ValueError("SFT resume architecture differs from the checkpoint.")
    optimizer = _optimizer(model, args)
    if resume_payload and resume_payload.get("optimizer"):
        optimizer.load_state_dict(resume_payload["optimizer"])
    step = int(resume_payload["step"]) if resume_payload else 0
    parent_stage = STAGE_MULTIMODAL_SFT if resume_payload else STAGE_GRAPH_TEXT_ALIGNMENT
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"graph policy={args.graph_policy} trainable params={trainable:,}", flush=True)

    held_out: list = []
    probes: list = []
    metrics_path = args.output_dir / "metrics.jsonl"
    if args.eval_split and args.tool_format:
        held_out, probes, eval_rejections = _held_out(args, gate_funcs, vocab, tokenizer)
        print(
            f"held-out examples={len(held_out)} "
            f"(retry={sum(e.retry for e in held_out)}) probes={len(probes)} "
            f"rejected={dict(eval_rejections)}",
            flush=True,
        )

    def evaluate(probe: bool) -> None:
        row: dict[str, Any] = {"step": step}
        if held_out:
            row.update(_held_out_loss(model, held_out, tokenizer.pad_token_id, device))
        if probe and probes:
            result = run_tool_probe(
                model, tokenizer, probes, gate_funcs, vocab, device=device,
                max_new_tokens=args.probe_max_new_tokens, tool_view=args.tool_view,
            )
            samples = result.pop("samples")
            row.update({f"probe_{k}": v for k, v in result.items()})
            (args.output_dir / f"probe_samples_step{step}.json").write_text(
                json.dumps(samples, indent=2)
            )
        if len(row) > 1:
            _append_jsonl(metrics_path, row)
            print("eval " + json.dumps(row), flush=True)

    def save(name: str) -> None:
        _save(
            model,
            optimizer,
            path=args.output_dir / f"multimodal_sft_{name}.pt",
            adapter_path=args.output_dir / f"adapter_{name}",
            stage=STAGE_MULTIMODAL_SFT,
            parent_stage=parent_stage,
            vocab=vocab,
            architecture=architecture,
            step=step,
            args=args,
        )

    needed = int(1.3 * args.max_steps * args.grad_accum * args.batch_size)
    records = _records(args, args.split, per_circuit=args.max_rows_per_circuit,
                       total=needed, seed=args.seed, repeat=True)
    rejections: collections.Counter = collections.Counter()
    kinds: collections.Counter = collections.Counter()

    def tool_batches():
        pending = []
        for record in records:
            example = _tool_example(record, args, gate_funcs, vocab, tokenizer,
                                    rejections, salt=str(args.seed))
            if example is None:
                continue
            kinds["retry" if example.retry else "single"] += 1
            pending.append(example)
            if len(pending) == args.batch_size:
                yield collate_tool_sft(pending, tokenizer.pad_token_id)
                pending = []

    def legacy_batches():
        for chunk in _batched(records, args.batch_size):
            try:
                yield build_multimodal_sft_batch(
                    chunk, gate_funcs, vocab, tokenizer, max_seq_len=args.max_seq_len,
                )
            except ValueError:
                continue

    # from_pretrained returns eval mode: gradient checkpointing, LoRA dropout
    # and the graph encoder's BatchNorm statistics need train mode.
    model.train()
    if step == 0:
        evaluate(probe=args.probe_every > 0)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    window: list[float] = []
    started = time.time()
    for batch in (tool_batches() if args.tool_format else legacy_batches()):
        batch = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        output = model(**batch, use_cache=False)
        (output.loss / args.grad_accum).backward()
        window.append(float(output.loss))
        micro_step += 1
        if micro_step % args.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.grad_clip,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0:
            print(
                f"step={step} sft_loss={sum(window) / len(window):.4f} "
                f"sec_per_step={(time.time() - started) / args.log_every:.1f} "
                f"examples={dict(kinds)} rejected={dict(rejections)}",
                flush=True,
            )
            window = []
            started = time.time()
        if args.eval_every and step % args.eval_every == 0 and step < args.max_steps:
            evaluate(probe=bool(args.probe_every) and step % args.probe_every == 0)
        if step % args.save_every == 0:
            save(f"step{step}")
        if step >= args.max_steps:
            break
    evaluate(probe=True)
    save("final")


def _reference_prefix(
    reference_modules: tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module],
    graph,
) -> torch.Tensor:
    graph_encoder, q_former, projector = reference_modules
    encoded = graph_encoder(graph)
    queries = q_former(encoded["node_embs"], graph.batch)
    return projector(queries)


def train_grpo(args: argparse.Namespace) -> None:
    if args.sft_ckpt is None and args.resume is None:
        raise ValueError("GRPO requires --sft-ckpt or --resume.")
    (
        gate_funcs,
        vocab,
        graph_encoder,
        q_former,
        alignment,
        tokenizer,
    ) = _load_common(args)
    resume_payload = None
    if args.resume:
        resume_payload = load_stage_checkpoint(
            args.resume,
            expected_stages=[STAGE_MULTIMODAL_GRPO],
            vocab=vocab,
        )
        source_payload = resume_payload
        adapter_path = args.adapter or Path(source_payload["session"]["adapter_path"])
    else:
        source_payload = load_stage_checkpoint(
            args.sft_ckpt,
            expected_stages=[STAGE_MULTIMODAL_SFT],
            vocab=vocab,
        )
        adapter_path = args.adapter or Path(source_payload["session"]["adapter_path"])

    llm = _load_grpo_llm(
        args,
        adapter_path,
        resume=resume_payload is not None,
    )
    model = GraphConditionedCausalLM(graph_encoder, q_former, llm)
    _load_module_state(model, source_payload["states"])
    model.set_graph_policy(args.graph_policy)
    device = _graph_device(model)
    for module in (model.graph_encoder, model.q_former, model.graph_to_llm):
        module.to(device)

    if resume_payload:
        reference_modules = (
            copy.deepcopy(model.graph_encoder),
            copy.deepcopy(model.q_former),
            copy.deepcopy(model.graph_to_llm),
        )
        reference_modules[0].load_state_dict(
            resume_payload["states"]["reference_graph_encoder"]
        )
        reference_modules[1].load_state_dict(
            resume_payload["states"]["reference_q_former"]
        )
        reference_modules[2].load_state_dict(
            resume_payload["states"]["reference_graph_to_llm"]
        )
    else:
        reference_modules = (
            copy.deepcopy(model.graph_encoder),
            copy.deepcopy(model.q_former),
            copy.deepcopy(model.graph_to_llm),
        )
    for module in reference_modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    architecture = _architecture(model, alignment, args)
    if source_payload["architecture"] != architecture:
        raise ValueError(
            "SFT/GRPO architecture differs from the source checkpoint."
        )
    optimizer = _optimizer(model, args)
    if resume_payload and resume_payload.get("optimizer"):
        optimizer.load_state_dict(resume_payload["optimizer"])
    step = int(resume_payload["step"]) if resume_payload else 0
    reward_fn = RewardFunctionFactory(
        str(resolve_sim_config_path(args.sim_config))
    ).create_reward_function(return_component_dicts=False)

    stream = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum_step = 0
    for record in stream:
        example = build_graph_prompt_example(
            record,
            gate_funcs,
            vocab,
            tokenizer,
            max_prompt_length=args.max_prompt_length,
        )
        if example is None:
            continue
        graph = example.graph.to(device)
        prompt_ids = example.prompt_ids.to(device)
        prompt_mask = example.prompt_mask.to(device)
        model.llm.set_adapter("policy")
        completion_ids, completion_mask = model.sample_group(
            graph,
            prompt_ids,
            prompt_mask,
            num_generations=args.num_generations,
            max_new_tokens=args.max_completion_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            temperature=args.temperature,
        )
        completions = tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
        )
        reward_kwargs = {
            key: [value] * args.num_generations
            for key, value in record.items()
        }
        reward_values = reward_fn(
            prompts=[example.prompt] * args.num_generations,
            completions=completions,
            **reward_kwargs,
        )
        rewards = torch.tensor(reward_values, dtype=torch.float32, device=device)
        advantages = group_relative_advantages(rewards)
        repeated_prompt_ids = prompt_ids.repeat(args.num_generations, 1)
        repeated_prompt_mask = prompt_mask.repeat(args.num_generations, 1)

        with torch.no_grad():
            policy_prefix = model.graph_soft_prompt(graph).repeat(
                args.num_generations, 1, 1
            )
            old_log_probs = model.completion_log_probs(
                graph,
                repeated_prompt_ids,
                repeated_prompt_mask,
                completion_ids,
                completion_mask,
                prefix=policy_prefix,
            )
            model.llm.set_adapter("reference")
            reference_prefix = _reference_prefix(
                reference_modules,
                graph,
            ).repeat(args.num_generations, 1, 1)
            reference_log_probs = model.completion_log_probs(
                graph,
                repeated_prompt_ids,
                repeated_prompt_mask,
                completion_ids,
                completion_mask,
                prefix=reference_prefix,
            )
            model.llm.set_adapter("policy")

        policy_prefix = model.graph_soft_prompt(graph).repeat(
            args.num_generations, 1, 1
        )
        policy_log_probs = model.completion_log_probs(
            graph,
            repeated_prompt_ids,
            repeated_prompt_mask,
            completion_ids,
            completion_mask,
            prefix=policy_prefix,
        )
        loss = graph_grpo_loss(
            policy_log_probs,
            old_log_probs,
            reference_log_probs,
            completion_mask,
            advantages,
            clip_epsilon=args.clip_epsilon,
            beta=args.beta,
        )
        (loss / args.grad_accum).backward()
        accum_step += 1
        if accum_step < args.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.grad_clip,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        step += 1
        if step % args.log_every == 0:
            print(
                f"step={step} grpo_loss={loss.item():.4f} "
                f"reward_mean={rewards.mean().item():.4f} "
                f"reward_std={rewards.std(unbiased=False).item():.4f}",
                flush=True,
            )
        if step % args.save_every == 0:
            _save(
                model,
                optimizer,
                path=args.output_dir / f"multimodal_grpo_step{step}.pt",
                adapter_path=args.output_dir / f"adapters_step{step}",
                stage=STAGE_MULTIMODAL_GRPO,
                parent_stage=STAGE_MULTIMODAL_GRPO if resume_payload else STAGE_MULTIMODAL_SFT,
                vocab=vocab,
                architecture=architecture,
                step=step,
                args=args,
                reference_modules=reference_modules,
            )
        if step >= args.max_steps:
            break
    _save(
        model,
        optimizer,
        path=args.output_dir / "multimodal_grpo_final.pt",
        adapter_path=args.output_dir / "adapters_final",
        stage=STAGE_MULTIMODAL_GRPO,
        parent_stage=STAGE_MULTIMODAL_GRPO if resume_payload else STAGE_MULTIMODAL_SFT,
        vocab=vocab,
        architecture=architecture,
        step=step,
        args=args,
        reference_modules=reference_modules,
    )


def main() -> None:
    args = _parse_args()
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")
    if args.method == "grpo" and args.num_generations < 2:
        raise ValueError("--num-generations must be >= 2 for GRPO")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.method == "sft":
        train_sft(args)
    else:
        train_grpo(args)


if __name__ == "__main__":
    main()
