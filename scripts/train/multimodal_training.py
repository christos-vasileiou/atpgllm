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
separately) and keeps the checkpoint with the lowest single-call loss as
``best``; ``--probe-examples`` runs the model on held-out circuits and
simulates its tool calls (:mod:`atpgllm.multimodal.tool_probe`). Under
torchrun SFT runs data-parallel: each rank holds the whole 4-bit model on its
GPU and trains on its own share of the sampled rows, so one optimizer step
sees ``ranks x --batch-size x --grad-accum`` examples.

GRPO (Stage D) rolls out multi-turn tool use: the model writes a turn, a tool
call in it is simulated (the same handler and view as SFT, with the netlist
from the record), the answer is appended as context, and the model continues,
up to ``--max-tool-calls`` (:mod:`atpgllm.multimodal.tool_rollout`). The
reward scores the trajectory's tool calls, simulating only the last turn that
called the tool (:func:`atpgllm.llm.reward_funcs.test_generation_grpo_reward`);
advantages are GDPO (default, per-objective) or GRPO (scalar). Only the
model's own tokens are trained.
"""

from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import copy
import hashlib
import json
import math
import os
import random
import shutil
import time
from datetime import datetime, timedelta
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch_geometric.data import Batch
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
    build_multimodal_sft_batch,
    graph_grpo_loss,
    group_relative_advantages,
    load_aligned_graph_stack,
)
from atpgllm.graph.circuit_sampling import sampled_records
from atpgllm.graph.dataset import record_to_pyg
from atpgllm.llm.reward_funcs import train_scalar_from_reward_components
from atpgllm.multimodal.data import (
    _compact_netlist_payload,
    _template_ids,
    build_tool_sft_example,
    collate_tool_sft,
)
from atpgllm.multimodal.tool_rollout import reward_text, reward_trace, rollout_group, turn_stop
from atpgllm.training.conversation import ConversationExample
from atpgllm.training.gdpo import (
    compute_gdpo_advantages,
    reward_output_to_tensors,
    select_objective_columns,
)
from atpgllm.training.tool_feedback import read_snapshot
from atpgllm.training.tools import (
    SimulationError,
    resolve_fault_sim_runner,
    ToolContext,
    execute_tool_batch,
    fault_simulation_tool_handler,
)
from atpgllm.multimodal.tool_probe import ProbeItem, run_tool_probe
from atpgllm.training._paths import (
    LIBATPGLLM_ROOT,
    ensure_data_preprocessing_on_path,
    resolve_sim_config_path,
)
from atpgllm.training.git_utils import get_git_info
from atpgllm.training.tool_trajectories import failed_attempt, record_seed
from atpgllm.training.reward_function_factory import RewardFunctionFactory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("sft", "grpo"))
    parser.add_argument("--sim-config", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        default="chrivasileiou/asap7-language-of-test-v2",
        help="Hub dataset id, or a local dataset directory (<dir>/<split>/*.parquet, "
             "for example the v4 x-fill build); --load parquet reads either.",
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
    parser.add_argument("--llm", default="ibm-granite/granite-4.2-8b")
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
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="constant",
        help="SFT: cosine decays every group from its LR to 0 over --max-steps, "
             "without warmup (as the text SFT's TRL default).",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention kernel of the LLM (sdpa, flash_attention_2).",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=0,
        help="SFT: keep this many most recent step checkpoints (0 = all); the "
             "best and final checkpoints are always kept.",
    )
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
    parser.add_argument("--max-tool-calls", type=int, default=3,
                        help="GRPO: tool calls per trajectory (SFT shows up to 2).")
    # Smoke tests only: an untrained policy scores 0 everywhere, so every
    # group is skipped and the update path never runs.
    parser.add_argument("--debug-random-advantages", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "--advantage",
        choices=("gdpo", "grpo"),
        default="gdpo",
        help="GRPO: per-objective GDPO advantages (default) or scalar GRPO.",
    )
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
    # One process spreads the model over every visible GPU; a data-parallel
    # rank holds a full copy on its own GPU.
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": args.local_rank} if args.world_size > 1 else "auto",
        "attn_implementation": args.attn_implementation,
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
        # Non-reentrant checkpointing, which DDP needs to see each parameter's
        # gradient once.
        base = prepare_model_for_kbit_training(
            base, gradient_checkpointing_kwargs={"use_reentrant": False},
        )
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


def _record_launch(args: argparse.Namespace) -> None:
    """Pin what this run depends on outside its arguments: the simulator the
    tool, retries and rewards use, both code checkouts, and the dataset build.
    Saved in each checkpoint's session and appended to code_provenance.jsonl."""
    data_preprocessing = ensure_data_preprocessing_on_path()
    args.simulator = resolve_fault_sim_runner().__name__
    manifest = Path(args.dataset) / "split_manifest.json"
    args.dataset_manifest_sha256 = (
        hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None
    )
    record = {
        "launched_at": datetime.now().isoformat(timespec="seconds"),
        "method": args.method,
        "simulator": args.simulator,
        "dataset": args.dataset,
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "llm": args.llm,
        **get_git_info(LIBATPGLLM_ROOT),
        **get_git_info(data_preprocessing, prefix="dp_git_"),
        "dp_root": str(data_preprocessing),
    }
    if args.rank:
        return
    with (args.output_dir / "code_provenance.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"[code_provenance] {record}", flush=True)


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
             total: int | None, seed: int, repeat: bool,
             shard: bool = False) -> Iterable[dict[str, Any]]:
    """Rows of ``split``; with ``shard``, this rank's disjoint share of them."""
    rank, world_size = (args.rank, args.world_size) if shard else (0, 1)
    if args.load == "parquet":
        return sampled_records(
            args.dataset, split, revision=args.revision, per_circuit=per_circuit,
            total=total, seed=seed, repeat=repeat, columns=None,
            rank=rank, world_size=world_size,
        )
    if world_size > 1:
        raise ValueError("Data-parallel training needs --load parquet.")
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
def _held_out_loss(model, examples, pad_token_id, device, *,
                   rank: int = 0, world_size: int = 1) -> dict[str, Any]:
    """Mean held-out loss per example kind. Data-parallel ranks score
    disjoint slices of ``examples`` and sum them."""
    was_training = model.training
    model.eval()
    kinds = ("single", "retry")
    # Row 0: summed loss, row 1: example count, one column per kind.
    totals = torch.zeros((2, len(kinds)), dtype=torch.float64, device=device)
    for example in examples[rank::world_size]:
        batch = collate_tool_sft([example], pad_token_id)
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        loss = model(**batch, use_cache=False).loss
        column = kinds.index("retry" if example.retry else "single")
        totals[0, column] += float(loss)
        totals[1, column] += 1
    if world_size > 1:
        dist.all_reduce(totals)
    if was_training:
        model.train()
    sums, counts = totals.tolist()
    out: dict[str, Any] = {
        f"heldout_loss_{kind}": (sums[i] / counts[i] if counts[i] else None)
        for i, kind in enumerate(kinds)
    }
    out["heldout_loss"] = sum(sums) / sum(counts) if sum(counts) else None
    out["heldout_examples"] = {kind: int(counts[i]) for i, kind in enumerate(kinds)}
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
    rank, world_size = args.rank, args.world_size
    main_process = rank == 0
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
    # Before loading the optimizer state, which holds already-decayed rates.
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    if resume_payload and resume_payload.get("optimizer"):
        optimizer.load_state_dict(resume_payload["optimizer"])
    step = int(resume_payload["step"]) if resume_payload else 0
    parent_stage = STAGE_MULTIMODAL_SFT if resume_payload else STAGE_GRAPH_TEXT_ALIGNMENT
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if main_process:
        print(f"graph policy={args.graph_policy} trainable params={trainable:,} "
              f"ranks={world_size}", flush=True)
    trained = model
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel

        trained = DistributedDataParallel(model, device_ids=[args.local_rank])

    held_out: list = []
    probes: list = []
    metrics_path = args.output_dir / "metrics.jsonl"
    if args.eval_split and args.tool_format:
        # Every rank builds the same fixed set and scores its slice of it.
        held_out, probes, eval_rejections = _held_out(args, gate_funcs, vocab, tokenizer)
        if main_process:
            print(
                f"held-out examples={len(held_out)} "
                f"(retry={sum(e.retry for e in held_out)}) probes={len(probes)} "
                f"rejected={dict(eval_rejections)}",
                flush=True,
            )
    best = math.inf

    def evaluate(probe: bool) -> dict[str, Any]:
        row: dict[str, Any] = {"step": step}
        if held_out:
            row.update(_held_out_loss(model, held_out, tokenizer.pad_token_id, device,
                                      rank=rank, world_size=world_size))
        if probe and probes and main_process:
            result = run_tool_probe(
                model, tokenizer, probes, gate_funcs, vocab, device=device,
                max_new_tokens=args.probe_max_new_tokens, tool_view=args.tool_view,
            )
            samples = result.pop("samples")
            row.update({f"probe_{k}": v for k, v in result.items()})
            (args.output_dir / f"probe_samples_step{step}.json").write_text(
                json.dumps(samples, indent=2)
            )
        if len(row) > 1 and main_process:
            _append_jsonl(metrics_path, row)
            print("eval " + json.dumps(row), flush=True)
        return row

    def save(name: str) -> None:
        if main_process:
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
        if world_size > 1:
            dist.barrier()

    def keep_best(row: dict[str, Any]) -> None:
        # Single-call loss only: like the text SFT's validation, which has no
        # retry conversations.
        nonlocal best
        loss = row.get("heldout_loss_single")
        if loss is not None and loss < best:
            best = loss
            save("best")
            if main_process:
                (args.output_dir / "best.json").write_text(
                    json.dumps({"step": step, "heldout_loss_single": loss}) + "\n"
                )

    def prune() -> None:
        if not main_process or args.save_total_limit <= 0:
            return
        steps = sorted(
            int(path.stem.removeprefix("multimodal_sft_step"))
            for path in args.output_dir.glob("multimodal_sft_step*.pt")
        )
        for old in steps[:-args.save_total_limit]:
            (args.output_dir / f"multimodal_sft_step{old}.pt").unlink()
            shutil.rmtree(args.output_dir / f"adapter_step{old}", ignore_errors=True)

    # Per rank; 30% spare for examples rejected while building.
    needed = int(1.3 * args.max_steps * args.grad_accum * args.batch_size)
    records = _records(args, args.split, per_circuit=args.max_rows_per_circuit,
                       total=needed * world_size, seed=args.seed, repeat=True,
                       shard=True)
    if step:
        # Resume past the rows already trained on (one row per example; rows
        # rejected while building make this slightly short).
        records = islice(records, step * args.grad_accum * args.batch_size, None)
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

    def set_lr() -> None:
        # The rate of the update that takes the run from ``step`` to step + 1.
        if args.lr_schedule == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * step / args.max_steps))
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * factor

    # from_pretrained returns eval mode: gradient checkpointing, LoRA dropout
    # and the graph encoder's BatchNorm statistics need train mode.
    model.train()
    if step == 0:
        keep_best(evaluate(probe=args.probe_every > 0))
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    window: list[float] = []
    started = time.time()
    for batch in (tool_batches() if args.tool_format else legacy_batches()):
        batch = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        micro_step += 1
        update = micro_step % args.grad_accum == 0
        # Gradients are averaged across ranks only on the micro-step that updates.
        with (trained.no_sync() if world_size > 1 and not update else contextlib.nullcontext()):
            output = trained(**batch, use_cache=False)
            (output.loss / args.grad_accum).backward()
        window.append(float(output.loss))
        if not update:
            continue
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.grad_clip,
        )
        set_lr()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0:
            loss = torch.tensor([sum(window), len(window)], dtype=torch.float64, device=device)
            if world_size > 1:
                dist.all_reduce(loss)
            row = {
                "step": step,
                "sft_loss": float(loss[0] / loss[1]),
                "lr": optimizer.param_groups[-1]["lr"],
                "sec_per_step": (time.time() - started) / args.log_every,
            }
            if main_process:
                _append_jsonl(metrics_path, row)
                print(
                    f"step={step} sft_loss={row['sft_loss']:.4f} lr={row['lr']:.3g} "
                    f"sec_per_step={row['sec_per_step']:.1f} "
                    f"examples={dict(kinds)} rejected={dict(rejections)}",
                    flush=True,
                )
            window = []
            started = time.time()
        if args.eval_every and step % args.eval_every == 0 and step < args.max_steps:
            keep_best(evaluate(probe=bool(args.probe_every) and step % args.probe_every == 0))
        if step % args.save_every == 0:
            save(f"step{step}")
            prune()
        if step >= args.max_steps:
            break
    keep_best(evaluate(probe=True))
    save("final")


def _reference_prefix(
    reference_modules: tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module],
    graph,
) -> torch.Tensor:
    graph_encoder, q_former, projector = reference_modules
    encoded = graph_encoder(graph)
    queries = q_former(encoded["node_embs"], graph.batch)
    return projector(queries)


def _grpo_prompt(record: dict[str, Any], tokenizer) -> list[int]:
    """System + user turn with the tool schema; the netlist is the graph."""
    messages = ConversationExample.prompt_messages_from_record(copy.deepcopy(record))
    full, compact = _compact_netlist_payload(record)
    for message in messages:
        if message["role"] == "user":
            message["content"] = message["content"].replace(full, compact)
    return _template_ids(tokenizer, messages, generation_prompt=True)


def _tool_runner(record: dict[str, Any], view: str):
    """Runs a batch of the model's calls for this record's circuit and fault.
    Errors in model-written arguments come back to the model as text;
    simulator infrastructure failures stop the run before any update."""
    full, _ = _compact_netlist_payload(record)
    netlist = record["netlist"]["netlist"] if isinstance(record["netlist"], dict) else record["netlist"]
    context = ToolContext(
        netlist=netlist,
        expected_doc_id=ast.literal_eval(full)["doc_id"],
        expected_fault=record["fault"],
        view=view,
    )
    functions = {"fault_simulation_tool": fault_simulation_tool_handler}

    def run(calls: list[dict[str, Any]]) -> list[str]:
        outcomes = execute_tool_batch(calls, [context] * len(calls), functions)
        if any(failed for _, failed in outcomes):
            raise SimulationError(
                "Simulator infrastructure failure during a rollout; no update performed."
            )
        return [content for content, _ in outcomes]

    return run


def _rollout_stats(trajectories, record) -> dict[str, float]:
    outputs = list(json.loads(record["expected_output"]))
    stats = collections.Counter()
    for trajectory in trajectories:
        results = trajectory.tool_results
        stats["trajectories"] += 1
        stats["tool_calls"] += len(results)
        stats["malformed_calls"] += sum(r.startswith("Tool execution failed") for r in results)
        stats["reused_calls"] += trajectory.reused_calls
        stats["truncated"] += trajectory.truncated
        stats["call_limit_hit"] += trajectory.call_limit_hit
        detected = []
        for result in results:
            try:
                detected.append(read_snapshot(result, record["fault"], outputs).detected)
            except (ValueError, KeyError, TypeError):
                detected.append(False)
        if detected:
            stats["called"] += 1
            stats["first_call_detects"] += detected[0]
            if not detected[0]:
                stats["first_call_missed"] += 1
                stats["recovered"] += any(detected[1:])
    return stats


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
    else:
        source_payload = load_stage_checkpoint(
            args.sft_ckpt,
            expected_stages=[STAGE_MULTIMODAL_SFT],
            vocab=vocab,
        )
    adapter_path = args.adapter or Path(source_payload["session"]["adapter_path"])
    sft_view = source_payload["session"].get("tool_view")
    if sft_view is not None and sft_view != args.tool_view:
        raise ValueError(
            f"--tool-view {args.tool_view!r} differs from the SFT model's {sft_view!r}."
        )
    # The model learned to read one simulator's answers (TetraMAX and the fast
    # simulator differ on x values, alias wires and TIE nets).
    sft_simulator = source_payload["session"].get("simulator")
    if sft_simulator is not None and sft_simulator != args.simulator:
        raise ValueError(
            f"FAULT_SIM_BACKEND gives {args.simulator}, but the SFT model was "
            f"trained on {sft_simulator}."
        )

    llm = _load_grpo_llm(args, adapter_path, resume=resume_payload is not None)
    # As TRL does for GRPO: sampling and the trained policy should be the
    # same function, so the LM's dropout is off.
    for module in llm.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model = GraphConditionedCausalLM(graph_encoder, q_former, llm)
    _load_module_state(model, source_payload["states"])
    model.set_graph_policy(args.graph_policy)
    device = _graph_device(model)
    for module in (model.graph_encoder, model.q_former, model.graph_to_llm):
        module.to(device)

    reference_modules = (
        copy.deepcopy(model.graph_encoder),
        copy.deepcopy(model.q_former),
        copy.deepcopy(model.graph_to_llm),
    )
    if resume_payload:
        for module, name in zip(reference_modules, ("graph_encoder", "q_former", "graph_to_llm")):
            module.load_state_dict(resume_payload["states"][f"reference_{name}"])
    for module in reference_modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    architecture = _architecture(model, alignment, args)
    if source_payload["architecture"] != architecture:
        raise ValueError("SFT/GRPO architecture differs from the source checkpoint.")
    optimizer = _optimizer(model, args)
    if resume_payload and resume_payload.get("optimizer"):
        optimizer.load_state_dict(resume_payload["optimizer"])
    step = int(resume_payload["step"]) if resume_payload else 0
    parent_stage = STAGE_MULTIMODAL_GRPO if resume_payload else STAGE_MULTIMODAL_SFT
    reward_fn = RewardFunctionFactory(
        str(resolve_sim_config_path(args.sim_config))
    ).create_reward_function(return_component_dicts=True)
    _, stop = turn_stop(tokenizer)
    metrics_path = args.output_dir / "metrics.jsonl"
    probes: list = []
    if args.eval_split and args.probe_examples:
        saved = args.eval_examples
        args.eval_examples = 0
        _, probes, _ = _held_out(args, gate_funcs, vocab, tokenizer)
        args.eval_examples = saved

    def train_mode() -> None:
        # The LM trains with gradient checkpointing (train mode); the graph
        # encoder keeps its BatchNorm statistics and no dropout, so the
        # prefix it trains is the one generation used. Its weights still learn.
        model.train()
        model.graph_encoder.eval()

    def save(name: str) -> None:
        _save(
            model,
            optimizer,
            path=args.output_dir / f"multimodal_grpo_{name}.pt",
            adapter_path=args.output_dir / f"adapters_{name}",
            stage=STAGE_MULTIMODAL_GRPO,
            parent_stage=parent_stage,
            vocab=vocab,
            architecture=architecture,
            step=step,
            args=args,
            reference_modules=reference_modules,
        )

    def probe() -> None:
        if not probes:
            return
        result = run_tool_probe(
            model, tokenizer, probes, gate_funcs, vocab, device=device,
            max_new_tokens=args.probe_max_new_tokens, tool_view=args.tool_view,
        )
        samples = result.pop("samples")
        (args.output_dir / f"probe_samples_step{step}.json").write_text(json.dumps(samples, indent=2))
        row = {"step": step, **{f"probe_{k}": v for k, v in result.items()}}
        _append_jsonl(metrics_path, row)
        print("eval " + json.dumps(row), flush=True)
        train_mode()

    needed = int(3 * args.max_steps * args.grad_accum)
    records = _records(args, args.split, per_circuit=args.max_rows_per_circuit,
                       total=needed, seed=args.seed, repeat=True)
    rejections: collections.Counter = collections.Counter()
    window: collections.Counter = collections.Counter()
    rewards_seen: list[float] = []
    detections: list[float] = []
    train_mode()
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    groups_seen = 0
    started = time.time()
    for record in records:
        graph = record_to_pyg(record, gate_funcs, vocab, rejections=rejections)
        if graph is None:
            continue
        prompt_ids = _grpo_prompt(record, tokenizer)
        if len(prompt_ids) > args.max_prompt_length:
            rejections["prompt_too_long"] += 1
            continue
        graph = Batch.from_data_list([graph]).to(device)

        model.eval()
        with torch.no_grad():
            prefix = model.graph_soft_prompt(graph)

        def generate_turn(contexts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
            return model.generate_turns(
                prefix, contexts, pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=1.0, top_k=0, eos_token_id=stop,
            )

        trajectories = rollout_group(
            generate_turn, tokenizer, prompt_ids, _tool_runner(record, args.tool_view),
            num_generations=args.num_generations,
            max_tool_calls=args.max_tool_calls,
            max_completion_tokens=args.max_completion_length,
        )
        train_mode()
        window.update(_rollout_stats(trajectories, record))

        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        rows = reward_fn(
            prompts=[prompt_text] * len(trajectories),
            completions=[reward_text(t) for t in trajectories],
            tool_trace=[reward_trace(t) for t in trajectories],
            **{key: [value] * len(trajectories) for key, value in record.items()},
        )
        scalars = torch.tensor(
            [train_scalar_from_reward_components(row) for row in rows], device=device
        )
        rewards_seen.extend(scalars.tolist())
        # The final answer's detection; the objective divides it by attempts.
        detections.extend(
            float(row.get("fault_detected_by_pred_input_vector_acc_logonly", 0.0)) for row in rows
        )
        if args.advantage == "gdpo":
            _, components, keys = reward_output_to_tensors(rows, device)
            objectives = select_objective_columns(
                components, keys, reward_fn.gdpo_objective_keys
            )
            advantages = compute_gdpo_advantages(
                objectives, len(trajectories), reward_fn.gdpo_objective_weights
            ).advantages
        else:
            advantages = group_relative_advantages(scalars)
        if args.debug_random_advantages:
            advantages = torch.randn(len(trajectories), device=device)
        groups_seen += 1
        signal = bool(advantages.ne(0).any())
        print(
            f"group {groups_seen}: rewards={[round(x, 3) for x in scalars.tolist()]} "
            f"calls={[len(t.tool_results) for t in trajectories]} "
            f"tokens={[len(t.completion_ids) for t in trajectories]} "
            f"{'update' if signal else 'skipped: no signal'}",
            flush=True,
        )
        if not signal:
            window["zero_advantage_groups"] += 1
            continue
        window["groups"] += 1

        prompt = torch.tensor([prompt_ids], device=device)
        prompt_mask = torch.ones_like(prompt)
        for index, trajectory in enumerate(trajectories):
            if not any(trajectory.loss_mask):
                continue
            completion = torch.tensor([trajectory.completion_ids], device=device)
            attention = torch.ones_like(completion)
            with torch.no_grad():
                model.llm.set_adapter("reference")
                reference = model.completion_log_probs(
                    graph, prompt, prompt_mask, completion, attention,
                    prefix=_reference_prefix(reference_modules, graph),
                )
                model.llm.set_adapter("policy")
            policy = model.completion_log_probs(
                graph, prompt, prompt_mask, completion, attention,
                prefix=model.graph_soft_prompt(graph),
            )
            loss = graph_grpo_loss(
                policy,
                policy.detach(),  # one update per rollout: the sampler is this policy
                reference,
                torch.tensor([trajectory.loss_mask], device=device),
                advantages[index:index + 1],
                clip_epsilon=args.clip_epsilon,
                beta=args.beta,
            )
            (loss / (len(trajectories) * args.grad_accum)).backward()
        accumulated += 1
        if accumulated % args.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.grad_clip,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0:
            n = max(1, window["trajectories"])
            row = {
                "step": step,
                "sec_per_step": round((time.time() - started) / args.log_every, 1),
                "reward_mean": sum(rewards_seen) / max(1, len(rewards_seen)),
                "detection_rate": sum(detections) / max(1, len(detections)),
                "tool_calls_per_trajectory": window["tool_calls"] / n,
                "first_call_detects": window["first_call_detects"] / max(1, window["called"]),
                "recovery_rate": window["recovered"] / max(1, window["first_call_missed"]),
                "malformed_call_share": window["malformed_calls"] / max(1, window["tool_calls"]),
                "reused_call_share": window["reused_calls"] / max(1, window["tool_calls"]),
                "truncated_share": window["truncated"] / n,
                "zero_advantage_group_share": window["zero_advantage_groups"]
                / max(1, window["groups"] + window["zero_advantage_groups"]),
                "rejected": dict(rejections),
            }
            _append_jsonl(metrics_path, row)
            print("train " + json.dumps(row), flush=True)
            window.clear()
            rewards_seen.clear()
            detections.clear()
            started = time.time()
        if args.probe_every and step % args.probe_every == 0 and step < args.max_steps:
            probe()
        if step % args.save_every == 0:
            save(f"step{step}")
        if step >= args.max_steps:
            break
    probe()
    save("final")


def _init_distributed() -> tuple[int, int, int]:
    """(rank, local rank, world size); joins the process group under torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Long timeout: the other ranks wait while rank 0 alone runs the tool
    # probe and writes checkpoints.
    dist.init_process_group("nccl", timeout=timedelta(hours=4))
    return dist.get_rank(), local_rank, world_size


def main() -> None:
    args = _parse_args()
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")
    if args.method == "grpo" and args.num_generations < 2:
        raise ValueError("--num-generations must be >= 2 for GRPO")
    args.rank, args.local_rank, args.world_size = _init_distributed()
    if args.world_size > 1 and args.method != "sft":
        raise ValueError("Only SFT runs data-parallel; launch GRPO as one process.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _record_launch(args)
    torch.manual_seed(args.seed)
    if args.method == "sft":
        train_sft(args)
    else:
        train_grpo(args)
    if args.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
