"""QLoRA SFT/GRPO for the DAG encoder + Q-Former + decoder LM stack."""

from __future__ import annotations

import argparse
import copy
import json
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
from atpgllm.training._paths import resolve_sim_config_path
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
    parser.add_argument("--graph-lr", type=float, default=1e-5)
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
    graph_parameters = [
        parameter
        for module in (
            model.graph_encoder,
            model.q_former,
            model.graph_to_llm,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    llm_parameters = [
        parameter
        for parameter in model.llm.parameters()
        if parameter.requires_grad
    ]
    if not graph_parameters or not llm_parameters:
        raise RuntimeError(
            "Expected trainable graph/projector and LoRA parameters; "
            "check --graph-policy and adapter loading."
        )
    return torch.optim.AdamW(
        [
            {"params": graph_parameters, "lr": args.graph_lr},
            {"params": llm_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )


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

    stream = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    for records in _batched(stream, args.batch_size):
        try:
            batch = build_multimodal_sft_batch(
                records,
                gate_funcs,
                vocab,
                tokenizer,
                max_seq_len=args.max_seq_len,
            )
        except ValueError:
            continue
        batch = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        output = model(**batch, use_cache=False)
        (output.loss / args.grad_accum).backward()
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
            print(f"step={step} sft_loss={output.loss.item():.4f}", flush=True)
        if step % args.save_every == 0:
            _save(
                model,
                optimizer,
                path=args.output_dir / f"multimodal_sft_step{step}.pt",
                adapter_path=args.output_dir / f"adapter_step{step}",
                stage=STAGE_MULTIMODAL_SFT,
                parent_stage=STAGE_MULTIMODAL_SFT if resume_payload else STAGE_GRAPH_TEXT_ALIGNMENT,
                vocab=vocab,
                architecture=architecture,
                step=step,
                args=args,
            )
        if step >= args.max_steps:
            break
    _save(
        model,
        optimizer,
        path=args.output_dir / "multimodal_sft_final.pt",
        adapter_path=args.output_dir / "adapter_final",
        stage=STAGE_MULTIMODAL_SFT,
        parent_stage=STAGE_MULTIMODAL_SFT if resume_payload else STAGE_GRAPH_TEXT_ALIGNMENT,
        vocab=vocab,
        architecture=architecture,
        step=step,
        args=args,
    )


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
