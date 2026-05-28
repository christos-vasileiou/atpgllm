"""
callbacks.py
============

Training callbacks for SFT and GRPO training pipelines.

Contains
--------
- ThroughputMetricsCallback   : Tracks tokens-per-second throughput.
- ContextLengthHistogramCallback : Saves context-length histograms per checkpoint.
- SFTStoppingCallback          : Monitors format compliance and output diversity
                                 to stop SFT at the right time before switching
                                 to GRPO.

SFT Stopping Criteria
---------------------
The SFT phase should teach the model to **follow the output format** without
destroying its **creative diversity**.  ``SFTStoppingCallback`` evaluates two
hard criteria at every checkpoint save (after ``min_steps``):

1. **Format Compliance** (default ≥ 95 %)
   Generate one completion per validation prompt (with multi-turn tool calling).
   Check whether the reward-function parsers can extract every required field:

       <think>(.*)</think> .*
       <tool_call>(.*)</tool_call> .*
       <tool_response>(.*)</tool_response> .*
       INPUT_VECTOR:(.*)
       EXPECTED_OUTPUT:(.*)
       DETECTED_FAULTS:(.*)

2. **Output Diversity** (default ≥ 30 % unique)
   Pick **one** prompt and generate ``diversity_num_generations`` completions
   (default 10).  Count distinct ``INPUT_VECTOR`` values.  If the model has
   collapsed to memorising a single answer the ratio drops to ≈ 0.

Additionally, the callback tracks an *informational* metric:

3. **Loss Plateau** (not a hard gate)
   If the training loss has not improved by ``loss_delta`` over the last
   ``loss_window`` logging steps, it is reported as a soft signal of
   convergence.

Training stops when criteria (1) AND (2) are **both** met for
``patience`` consecutive evaluations, and ``min_steps`` has been reached.
"""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from typing_extensions import deprecated

import numpy as np
import regex as re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from tqdm import tqdm
from tools import ToolHelper, TOOLS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# =====================================================================
# Files produced by the HuggingFace Trainer checkpoint mechanism
# =====================================================================
_TRAINER_STATE_FILE = "trainer_state.json"
_OPTIMIZER_FILES = ("optimizer.pt", "optimizer.safetensors")
_SCHEDULER_FILE = "scheduler.pt"
_RNG_STATE_PREFIX = "rng_state"

# =====================================================================
# Regex patterns for format checking (mirrors RewardFunctionFactory)
# =====================================================================
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
INPUT_VECTOR_RE = re.compile(r'INPUT_VECTOR:\s*"(.*?)"', re.DOTALL)
EXPECTED_OUTPUT_RE = re.compile(r'EXPECTED_OUTPUT:\s*"(.*?)"', re.DOTALL)
DETECTED_FAULTS_RE = re.compile(r'DETECTED_FAULTS:\s*"(.*?)"', re.DOTALL)

# Combined full-format regex (informational – individual checks are more
# granular, but this one is useful for a quick boolean "parseable?" test).
FULL_FORMAT_RE = re.compile(
    r"<think>.*?</think>"
    r".*?<tool_call>.*?</tool_call>"
    r".*?<tool_response>.*?</tool_response>"
    r'.*?INPUT_VECTOR:\s*".*?"'
    r'.*?EXPECTED_OUTPUT:\s*".*?"'
    r'.*?DETECTED_FAULTS:\s*".*?"',
    re.DOTALL,
)


def _ddp_barrier() -> None:
    """Synchronise all ranks when torch.distributed is initialised (no-op otherwise)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _move_underlying_params_buffers_to_cpu(underlying: torch.nn.Module) -> None:
    """Move tensors to CPU in-place so nn.Parameter identity is preserved (DDP-safe)."""
    for p in underlying.parameters():
        p.data = p.data.cpu()
        if p.grad is not None:
            p.grad = p.grad.cpu()
    for b in underlying.buffers():
        b.data = b.data.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _move_underlying_params_buffers_to_device(
    underlying: torch.nn.Module, device: torch.device
) -> None:
    nb = device.type == "cuda"
    for p in underlying.parameters():
        p.data = p.data.to(device, non_blocking=nb)
    for b in underlying.buffers():
        b.data = b.data.to(device, non_blocking=nb)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


# =====================================================================
# ThroughputMetricsCallback
# =====================================================================
class ThroughputMetricsCallback(TrainerCallback):
    """
    Callback that tracks tokens-per-second throughput metrics during training.

    Metrics logged (appear in wandb / console logs):
    - throughput/tokens_per_sec: Tokens processed per second over the last
      logging interval.
    - throughput/overall_tokens_per_sec: Average tokens/sec since training
      started.

    Works with both SFT (requires ``include_num_input_tokens_seen=True`` in
    config) and GRPO (which tracks ``num_input_tokens_seen`` natively).
    """

    def __init__(self):
        self._train_start_time: float | None = None
        self._last_log_time: float | None = None
        self._last_tokens_seen: int = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start_time = time.perf_counter()
        self._last_log_time = self._train_start_time
        self._last_tokens_seen = getattr(state, "num_input_tokens_seen", 0)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        now = time.perf_counter()
        tokens_seen = getattr(state, "num_input_tokens_seen", 0)

        # Interval throughput (tokens/sec since last log event)
        if self._last_log_time is not None:
            interval_time = now - self._last_log_time
            interval_tokens = tokens_seen - self._last_tokens_seen
            if interval_time > 0 and interval_tokens > 0:
                logs["throughput/tokens_per_sec"] = round(
                    interval_tokens / interval_time, 2
                )

        # Overall throughput since training started
        if self._train_start_time is not None and tokens_seen > 0:
            total_time = now - self._train_start_time
            if total_time > 0:
                logs["throughput/overall_tokens_per_sec"] = round(
                    tokens_seen / total_time, 2
                )

        self._last_log_time = now
        self._last_tokens_seen = tokens_seen


# =====================================================================
# ContextLengthHistogramCallback
# =====================================================================
class ContextLengthHistogramCallback(TrainerCallback):
    """
    Records the token-level context length of every sequence the model
    processes during training and saves a histogram figure into each
    checkpoint directory (i.e. every ``save_steps``).

    How it works
    ------------
    A forward hook is registered on the model's **input embedding layer**.
    Each time the embedding layer is called with a batch of ``input_ids``
    the hook counts the number of non-padding tokens per sequence and
    appends them to an internal list.

    To avoid double-counting caused by gradient-checkpointing (which
    replays forward passes during backward), recording is gated by a
    per-step flag that is toggled via ``on_step_begin`` / ``on_step_end``.

    Metrics saved per checkpoint
    ----------------------------
    * ``context_length_histogram.png`` – histogram figure.
    * ``context_length_stats.json``  – summary statistics (mean, median,
      percentiles, min/max, total sequences seen so far).

    Parameters
    ----------
    pad_token_id : int
        Token id used for padding.  Non-pad tokens are counted as the
        effective context length.
    """

    def __init__(self, pad_token_id: int = 0, tokenizer: AutoTokenizer = None):
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.context_lengths: list[int] = []
        self._hook_handle = None
        # Gating flags to avoid double-counting from gradient checkpointing
        self._recording = False
        self._recorded_this_step = False

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        embed_layer = model.get_input_embeddings()

        pad_id = self.pad_token_id
        cb = self  # capture reference for the closure

        def _embed_hook(_module, _input, _output):
            if not cb._recording or cb._recorded_this_step:
                return
            inp = _input[0]  # nn.Embedding receives (input_ids,)
            if inp.dim() == 2 and inp.shape[1] > 1:
                for seq in inp:
                    n_tokens = int((seq != pad_id).sum())
                    cb.context_lengths.append(n_tokens)
                cb._recorded_this_step = True

        self._hook_handle = embed_layer.register_forward_hook(_embed_hook)

    def on_train_end(self, args, state, control, **kwargs):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ------------------------------------------------------------------
    # Per-step gating
    # ------------------------------------------------------------------
    def on_step_begin(self, args, state, control, **kwargs):
        self._recording = True
        self._recorded_this_step = False

    def on_step_end(self, args, state, control, **kwargs):
        self._recording = False

    # ------------------------------------------------------------------
    # Histogram saving (fires every save_steps)
    # ------------------------------------------------------------------
    def on_save(self, args, state, control, **kwargs):
        if not self.context_lengths:
            return

        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._save_histogram(checkpoint_dir, state.global_step)

    def _save_histogram(self, directory: str, step: int) -> None:
        lengths = np.array(self.context_lengths)

        # ---- figure -------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))

        n_bins = min(50, max(10, len(set(lengths)) // 2 or 10))
        ax.hist(
            lengths,
            bins=n_bins,
            edgecolor="black",
            alpha=0.75,
            color="steelblue",
        )

        mean_val = float(np.mean(lengths))
        median_val = float(np.median(lengths))
        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_val:,.0f}",
        )
        ax.axvline(
            median_val,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"Median: {median_val:,.0f}",
        )

        ax.set_xlabel("Context Length (tokens)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"Context Window Length Distribution — Step {step}\n"
            f"({len(lengths):,} sequences,  min={int(lengths.min())},  "
            f"max={int(lengths.max())})",
            fontsize=13,
        )
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        fig_path = os.path.join(directory, "context_length_histogram.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- summary JSON -------------------------------------------
        stats = {
            "step": step,
            "total_sequences": len(lengths),
            "mean": round(mean_val, 2),
            "median": round(median_val, 2),
            "std": round(float(np.std(lengths)), 2),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "percentile_25": round(float(np.percentile(lengths, 25)), 2),
            "percentile_75": round(float(np.percentile(lengths, 75)), 2),
            "percentile_95": round(float(np.percentile(lengths, 95)), 2),
        }
        is_distributed = torch.distributed.is_initialized()
        is_main = not is_distributed or torch.distributed.get_rank() == 0
        stats_path = os.path.join(directory, "context_length_stats.json")
        if is_main:
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)

            print(
                f"[ContextLengthHistogram] Saved to {fig_path}\n"
                f"  Sequences: {len(lengths):,} | Mean: {mean_val:,.0f} | "
                f"Median: {median_val:,.0f} | "
                f"Range: [{int(lengths.min())}, {int(lengths.max())}]"
            )


# =====================================================================
# TrainingStateCheckpointCallback
# =====================================================================
class TrainingStateCheckpointCallback(TrainerCallback):
    """
    Ensures every checkpoint is fully resumable and logs a clear summary.

    The HuggingFace Trainer already persists optimizer state, LR
    scheduler state, RNG seeds, and training progress (step / epoch) in
    every checkpoint directory.  This callback adds two things on top:

    1. **Verification** – after each save it confirms that all required
       files (``trainer_state.json``, optimizer, scheduler, RNG) are
       present and warns loudly if anything is missing.
    2. **Visibility** – prints step, epoch, current LR, and a
       *RESUMABLE / INCOMPLETE* tag so the user can tell at a glance
       which checkpoints are safe to resume from.

    Additionally, a small ``training_state_summary.json`` file is
    written into each checkpoint with the same information in a
    machine-readable format.

    To resume from a checkpoint, combine the two CLI flags::

        python training_code.py \\
            --resume_from <output_dir>/checkpoint-<N> \\
            --resume_training_state \\
            --output_dir <output_dir> ...

    ``--resume_from`` loads the adapter weights; adding
    ``--resume_training_state`` also restores the optimizer, LR
    schedule, global step, epoch counter, and RNG seeds so training
    continues exactly where it left off.

    Stream-position bookkeeping
    ---------------------------
    The summary JSON additionally records the cumulative
    ``skip_buffer_size`` that should be passed to the **next** training
    job that resumes from this checkpoint as a fresh run on new data
    (i.e. without ``--resume_training_state``). This removes the need
    to compute the next stream offset by hand:

    * **SFT** — ``cumulative_skip_buffer_size = launch_skip_buffer_size
      + global_step * effective_batch_size`` where
      ``effective_batch_size = per_device_train_batch_size * world_size
      * gradient_accumulation_steps``. The Trainer consumes one
      post-filter row per logical micro-batch slot, so this matches the
      number of valid rows the streaming pipeline yielded.
    * **GRPO** — ``cumulative_skip_buffer_size = launch_skip_buffer_size
      + buffer_size``. The buffer is populated once at startup with the
      next ``buffer_size`` valid rows from the stream; we treat the
      whole buffer as "consumed" regardless of how many ``max_steps``
      were actually reached, so the next run picks up from beyond the
      buffered slice.

    Parameters
    ----------
    method : str, optional
        ``"sft"`` or ``"grpo"``. Selects the cumulative skip formula.
        When ``None`` the SFT formula is used (back-compatible default).
    launch_skip_buffer_size : int
        The ``skip_buffer_size`` value that was passed to this run
        (after any ``--auto_skip_from_resume`` resolution). Used as the
        offset added to ``consumed_in_run``.
    buffer_size : int, optional
        GRPO only. The ``buffer_size`` that ``buffer_streaming_dataset``
        was asked to fill. Required when ``method="grpo"``.
    """

    def __init__(
        self,
        method: Optional[str] = None,
        launch_skip_buffer_size: int = 0,
        buffer_size: Optional[int] = None,
    ):
        self.method = method.lower() if isinstance(method, str) else None
        try:
            self.launch_skip_buffer_size = max(0, int(launch_skip_buffer_size or 0))
        except (TypeError, ValueError):
            self.launch_skip_buffer_size = 0
        if buffer_size is None:
            self.buffer_size: Optional[int] = None
        else:
            try:
                self.buffer_size = max(0, int(buffer_size))
            except (TypeError, ValueError):
                self.buffer_size = None

    def _compute_skip_state(self, args, state) -> dict:
        """Return cumulative_skip_buffer_size + supporting metadata for the summary."""
        try:
            world_size = int(getattr(args, "world_size", 1) or 1)
        except (TypeError, ValueError):
            world_size = 1
        try:
            per_device = int(getattr(args, "per_device_train_batch_size", 1) or 1)
        except (TypeError, ValueError):
            per_device = 1
        try:
            grad_acc = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
        except (TypeError, ValueError):
            grad_acc = 1
        effective_batch_size = max(1, per_device) * max(1, world_size) * max(1, grad_acc)

        if self.method == "grpo" and self.buffer_size is not None:
            consumed = int(self.buffer_size)
        else:
            consumed = int(state.global_step) * effective_batch_size
        cumulative = self.launch_skip_buffer_size + consumed

        skip_state: dict = {
            "method": self.method,
            "launch_skip_buffer_size": self.launch_skip_buffer_size,
            "effective_batch_size": effective_batch_size,
            "per_device_train_batch_size": per_device,
            "gradient_accumulation_steps": grad_acc,
            "world_size": world_size,
            "consumed_in_run": consumed,
            "cumulative_skip_buffer_size": cumulative,
        }
        if self.method == "grpo":
            skip_state["buffer_size"] = self.buffer_size
        return skip_state

    def on_save(self, args, state, control, **kwargs):
        is_distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if is_distributed and torch.distributed.get_rank() != 0:
            return

        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        if not os.path.isdir(checkpoint_dir):
            return

        # -- Collect file inventory ------------------------------------
        has_trainer_state = os.path.exists(
            os.path.join(checkpoint_dir, _TRAINER_STATE_FILE)
        )
        has_optimizer = any(
            os.path.exists(os.path.join(checkpoint_dir, f))
            for f in _OPTIMIZER_FILES
        )
        has_scheduler = os.path.exists(
            os.path.join(checkpoint_dir, _SCHEDULER_FILE)
        )
        rng_files = [
            f for f in os.listdir(checkpoint_dir)
            if f.startswith(_RNG_STATE_PREFIX)
        ]
        has_rng = len(rng_files) > 0

        # -- Extract current learning rate -----------------------------
        current_lr = None
        if state.log_history:
            for entry in reversed(state.log_history):
                if "learning_rate" in entry:
                    current_lr = entry["learning_rate"]
                    break

        resumable = (
            has_trainer_state and has_optimizer
            and has_scheduler and has_rng
        )
        status = "RESUMABLE" if resumable else "INCOMPLETE"
        lr_str = f"{current_lr:.2e}" if current_lr is not None else "N/A"

        # -- Compute cumulative stream offset for next fresh-run --------
        try:
            skip_state = self._compute_skip_state(args, state)
        except Exception as exc:
            logger.warning(
                "Could not compute cumulative_skip_buffer_size: %s", exc
            )
            skip_state = {}

        print(
            f"[TrainingStateCheckpoint] Step {state.global_step} | "
            f"Epoch {state.epoch:.4f} | LR {lr_str} | {status}"
        )
        if not resumable:
            missing = []
            if not has_trainer_state:
                missing.append(_TRAINER_STATE_FILE)
            if not has_optimizer:
                missing.append("optimizer.pt/.safetensors")
            if not has_scheduler:
                missing.append(_SCHEDULER_FILE)
            if not has_rng:
                missing.append("rng_state_*.pth")
            print(
                f"  WARNING: Checkpoint incomplete — missing: "
                f"{', '.join(missing)}"
            )
        else:
            print(
                f"  Resume (crash recovery): --resume_from {checkpoint_dir} "
                f"--resume_training_state"
            )
            cumulative = skip_state.get("cumulative_skip_buffer_size")
            if cumulative is not None:
                print(
                    f"  Resume (fresh run on new data): --resume_from "
                    f"{checkpoint_dir} --skip_buffer_size {cumulative} "
                    f"(or rely on --auto_skip_from_resume, default ON)"
                )

        # -- Write machine-readable summary ----------------------------
        summary = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "max_steps": state.max_steps,
            "learning_rate": current_lr,
            "num_input_tokens_seen": getattr(
                state, "num_input_tokens_seen", 0
            ),
            "total_flos": state.total_flos,
            "best_metric": state.best_metric,
            "resumable": resumable,
            "files": {
                "trainer_state": has_trainer_state,
                "optimizer": has_optimizer,
                "scheduler": has_scheduler,
                "rng_states": rng_files,
            },
            **skip_state,
        }
        summary_path = os.path.join(
            checkpoint_dir, "training_state_summary.json"
        )
        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        except OSError as exc:
            logger.warning(
                "Could not write training state summary: %s", exc
            )


def _read_skip_summary_int(checkpoint_dir: str, key: str) -> Optional[int]:
    """Internal: read a non-negative integer field from ``training_state_summary.json``.

    Returns ``None`` (and does not raise) when the summary file is
    missing, malformed, or predates this feature.
    """
    summary_path = os.path.join(checkpoint_dir, "training_state_summary.json")
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s for %s: %s", summary_path, key, exc)
        return None
    val = data.get(key)
    if val is None:
        return None
    try:
        return max(0, int(val))
    except (TypeError, ValueError):
        return None


def read_cumulative_skip_from_checkpoint(checkpoint_dir: str) -> Optional[int]:
    """Return ``cumulative_skip_buffer_size`` from a checkpoint summary, or ``None``.

    Used by ``--auto_skip_from_resume`` for **fresh-on-new-data** resumes
    (i.e. without ``--resume_training_state``) — the next run skips past
    every valid row the previous run consumed, so training never revisits
    samples already covered.

    Returns ``None`` for legacy checkpoints without the new field; callers
    should fall back to the user-supplied value or 0.
    """
    return _read_skip_summary_int(checkpoint_dir, "cumulative_skip_buffer_size")


def read_launch_skip_from_checkpoint(checkpoint_dir: str) -> Optional[int]:
    """Return ``launch_skip_buffer_size`` from a checkpoint summary, or ``None``.

    Used by ``--auto_skip_from_resume`` for **crash-recovery** resumes
    (i.e. with ``--resume_training_state``) — the next run uses the same
    stream offset that the checkpoint's optimizer state was trained
    against, so HF Trainer's built-in step-resume skip lands on the same
    rows the original run consumed past ``global_step``.

    Returns ``None`` for legacy checkpoints without the new field.
    """
    return _read_skip_summary_int(checkpoint_dir, "launch_skip_buffer_size")


def validate_training_state_checkpoint(checkpoint_dir: str) -> None:
    """Raise if *checkpoint_dir* lacks files needed for state resumption.

    Called from the training entry-points when ``--resume_training_state``
    is set, **before** the Trainer is constructed, so the user gets an
    immediate, actionable error instead of a cryptic failure mid-training.
    """
    trainer_state = os.path.join(checkpoint_dir, _TRAINER_STATE_FILE)
    if not os.path.exists(trainer_state):
        raise FileNotFoundError(
            f"Cannot resume training state: '{trainer_state}' not found.\n"
            f"The --resume_from path must point to a Trainer checkpoint "
            f"directory (e.g. output_dir/checkpoint-50/) that was saved "
            f"during a previous training run — not a bare adapter directory "
            f"produced by trainer.save_model().\n"
            f"Contents of '{checkpoint_dir}': "
            f"{os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir) else 'NOT A DIRECTORY'}"
        )
    has_optimizer = any(
        os.path.exists(os.path.join(checkpoint_dir, f))
        for f in _OPTIMIZER_FILES
    )
    if not has_optimizer:
        raise FileNotFoundError(
            f"Cannot resume training state: no optimizer file found in "
            f"'{checkpoint_dir}'.\nExpected one of: {_OPTIMIZER_FILES}"
        )
    if not os.path.exists(os.path.join(checkpoint_dir, _SCHEDULER_FILE)):
        logger.warning(
            "Scheduler state file '%s' not found in '%s'. "
            "The LR schedule will restart from the beginning.",
            _SCHEDULER_FILE, checkpoint_dir,
        )


# =====================================================================
# SFTStoppingCallback
# =====================================================================
@deprecated("Use `python evaluate_model.py --sft-stop` instead")
class SFTStoppingCallback(TrainerCallback):
    """
    Monitors SFT training and stops when the model has learned the output
    format sufficiently without collapsing into repetitive outputs.

    The callback evaluates two hard criteria (and one soft metric) at each
    checkpoint save after ``min_steps`` training steps.

    Hard Criteria (both must pass for ``patience`` consecutive evaluations)
    -----------------------------------------------------------------------
    1. **Format Compliance** (``format_threshold``, default 0.95)
       For each prompt in the buffered validation set, generate a completion
       with full multi-turn tool calling (generate → tool_call → tool
       response → continue).  Check whether the reward-function parsers can
       extract every structured field.

    2. **Output Diversity** (``diversity_threshold``, default 0.3)
       Pick a **single** prompt and generate ``diversity_num_generations``
       completions (default 10).  Count how many produce distinct
       ``INPUT_VECTOR`` values.  Ratio of unique / total must exceed the
       threshold.

    Soft Metric (informational, logged but not blocking)
    ----------------------------------------------------
    3. **Loss Plateau**
       If the training loss has not improved by ``loss_delta`` over the last
       ``loss_window`` logging steps, it is reported.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        The tokenizer used for encoding / decoding.
    dataset_path : str
        HuggingFace dataset identifier (e.g.
        ``"chrivasileiou/asap7-language-of-test"``).
    eval_buffer_size : int
        Number of validation examples to buffer from the ``"test"`` split
        (default 30).
    tool_functions : dict[str, Callable], optional
        Mapping from tool function name to callable.  Supports both sync
        and async callables.
    tools_schema : list, optional
        The JSON schema list passed to ``apply_chat_template(tools=...)``.
    format_threshold : float
        Minimum fraction of parseable validation outputs (default 0.95).
    diversity_threshold : float
        Minimum fraction of unique outputs in the diversity check
        (default 0.3).
    diversity_num_generations : int
        How many completions to generate for the single-prompt diversity
        check (default 10).
    use_vllm : bool
        If ``True``, use a **persistent vLLM server** for eval generation.
        The server must be started separately (e.g. on a spare GPU) before
        training with dynamic LoRA loading enabled::

            VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \\
                CUDA_VISIBLE_DEVICES=<gpu> vllm serve <model> \\
                --enable-lora --max-lora-rank 64 --port 8000

        Falls back to ``model.generate()`` if the server is unreachable.
    vllm_server_url : str, optional
        Base URL of the running vLLM server (default
        ``"http://localhost:8000"`` when ``use_vllm=True``).
    max_new_tokens : int
        Maximum new tokens per generation turn (default 4096).
    min_steps : int
        Do not evaluate before this many training steps (default 50).
    patience : int
        Number of consecutive evaluations that must pass before stopping
        (default 1).  Increase to 2–3 for extra stability.
    temperature : float
        Sampling temperature for generation (default 0.7).
    loss_delta : float
        Minimum loss improvement to *not* be flagged as a plateau
        (default 0.01).
    loss_window : int
        Number of recent log entries to compare for plateau detection
        (default 10).
    eval_every_n_saves : int
        Only run the full evaluation every *N* checkpoint saves (default 1).
        Increase if evaluation is too slow relative to ``save_steps``.
    generation_batch_size : int
        Maximum number of prompts to feed into ``model.generate()`` in one
        call (default 8).  The two-turn generation pipeline batches both
        Turn 1 (think + tool_call) and Turn 2 (summary) using this size,
        reducing wall-clock time by ~7× compared to sequential generation.
        Increase on high-VRAM GPUs (e.g. H100 80 GB) or decrease if you
        see OOM during evaluation.
    vllm_max_context : int
        vLLM server's max_model_len (default 32768). Used to cap max_tokens
        per request so input_tokens + max_tokens <= vllm_max_context.
    """
    THINK_RE = THINK_RE
    TOOL_CALL_RE = TOOL_CALL_RE
    TOOL_RESPONSE_RE = TOOL_RESPONSE_RE
    INPUT_VECTOR_RE = INPUT_VECTOR_RE
    EXPECTED_OUTPUT_RE = EXPECTED_OUTPUT_RE
    DETECTED_FAULTS_RE = DETECTED_FAULTS_RE

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_path: str,
        eval_buffer_size: int = 30,
        tool_functions: Optional[Dict[str, Callable]] = None,
        tools_schema: Optional[list] = None,
        format_threshold: float = 0.95,
        diversity_threshold: float = 0.3,
        diversity_num_generations: int = 10,
        use_vllm: bool = False,
        vllm_server_url: Optional[str] = None,
        max_new_tokens: int = 8192,
        max_prompt_length: int = 4096,
        min_steps: int = 50,
        patience: int = 1,
        temperature: float = 0.7,
        loss_delta: float = 0.01,
        loss_window: int = 10,
        eval_every_n_saves: int = 1,
        generation_batch_size: int = 8,
        vllm_max_context: int = 32768,
        device_map: str = "auto",
    ):
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.eval_buffer_size = eval_buffer_size
        self.tool_functions = tool_functions or {}
        self.tools_schema = tools_schema
        self.format_threshold = format_threshold
        self.diversity_threshold = diversity_threshold
        self.diversity_num_generations = diversity_num_generations
        self.use_vllm = use_vllm and importlib.util.find_spec("vllm") is not None
        if self.use_vllm:
            self._max_tool_rounds = 1
        self.vllm_server_url = vllm_server_url
        self.max_new_tokens = max_new_tokens
        self.max_prompt_length = max_prompt_length
        self.min_steps = min_steps
        self.patience = patience
        self.temperature = temperature
        self.loss_delta = loss_delta
        self.loss_window = loss_window
        self.eval_every_n_saves = eval_every_n_saves
        self._generation_batch_size = generation_batch_size
        self.vllm_max_context = vllm_max_context
        self.device_map = device_map

        # Lazy-loaded eval dataset
        self._eval_dataset = None
        self._eval_prompts: List[str] = []

        # State
        self._recent_losses: List[float] = []
        self._consecutive_passes: int = 0
        self._stopped: bool = False
        self._save_count: int = 0

        # vLLM server state
        self._current_checkpoint_dir: Optional[str] = None
        self._vllm_adapter_loaded: Optional[str] = None

    # ------------------------------------------------------------------
    # Lazy eval dataset loading
    # ------------------------------------------------------------------
    def _load_eval_dataset(self) -> None:
        """Buffer the ``test`` split for validation (lazy, runs once)."""
        if self._eval_dataset is not None:
            return

        # Lazy import to avoid circular dependency with training_code.py
        from dataset_utils import (
            TrainingMode,
            buffer_streaming_dataset,
            format_dataset_for_training,
        )

        print(
            f"[SFTStoppingCallback] Loading eval dataset "
            f"from {self.dataset_path} (buffer={self.eval_buffer_size})..."
        )
        eval_data = load_dataset(self.dataset_path, split="test", streaming=True)
        eval_formatted_dataset = format_dataset_for_training(
            eval_data, self.tokenizer, TrainingMode.GRPO
        )
        self._eval_dataset = buffer_streaming_dataset(
            eval_formatted_dataset,
            buffer_size=self.eval_buffer_size,
            shuffle=False,
            tokenizer=self.tokenizer,
            max_prompt_length=self.max_prompt_length,
        )
        self._eval_prompts = [ex["prompt"] for ex in self._eval_dataset]
        print(f"[SFTStoppingCallback] Loaded {len(self._eval_prompts)} eval prompts")

    # ------------------------------------------------------------------
    # Loss tracking
    # ------------------------------------------------------------------
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        if loss is not None:
            self._recent_losses.append(float(loss))
            # Keep a bounded window
            max_keep = self.loss_window * 3
            if len(self._recent_losses) > max_keep:
                self._recent_losses = self._recent_losses[-max_keep:]

    # ------------------------------------------------------------------
    # Main evaluation hook (fires at every checkpoint save)
    # ------------------------------------------------------------------
    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None or self._stopped:
            return

        self._save_count += 1

        # Respect eval_every_n_saves
        if self._save_count % self.eval_every_n_saves != 0:
            return

        # Don't evaluate before minimum steps
        if state.global_step < self.min_steps:
            return

        # ----- DDP vs single-process -----
        # In-process vLLM: every rank moves the training weights to CPU so each
        # local GPU is free; rank 0 then loads vLLM.  All ranks must enter the
        # same barriers so no rank runs Trainer/NCCL collectives while others
        # are blocked in eval (rank skew causes watchdog timeouts).
        is_distributed = torch.distributed.is_initialized()
        is_main = not is_distributed or torch.distributed.get_rank() == 0
        should_stop = False

        underlying = model.module if hasattr(model, "module") else model
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = (
            torch.device(f"cuda:{local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        use_inproc_vllm = self.use_vllm and not self.vllm_server_url

        if is_main:
            self._current_checkpoint_dir = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )
            backend_label = (
                "vLLM (in-process)"
                if use_inproc_vllm
                else "vLLM server"
                if (self.use_vllm and self.vllm_server_url)
                else "HF batched"
            )
            print(f"\n{'=' * 60}")
            print(f"[SFTStoppingCallback] Evaluating at step {state.global_step}")
            print(
                f"  Backend: {backend_label}"
                f"  |  gen_batch_size: {self._generation_batch_size}"
                f"  |  eval prompts: "
                f"{len(self._eval_prompts) if self._eval_prompts else self.eval_buffer_size}"
            )
            print(f"{'=' * 60}")
            self._load_eval_dataset()

        was_training = model.training
        model.eval()

        from model_utils import load_optimizer_state_from_cpu, save_optimizer_state_to_cpu

        base_opt = (
            kwargs["optimizer"].optimizer
            if hasattr(kwargs["optimizer"], "optimizer")
            else kwargs["optimizer"]
        )
        opt_state_cpu = None
        old_padding_side = self.tokenizer.padding_side
        vllm_model = None

        if use_inproc_vllm:
            opt_state_cpu = save_optimizer_state_to_cpu(kwargs["optimizer"])
            _move_underlying_params_buffers_to_cpu(underlying)
            _ddp_barrier()

        if is_main:
            model_to_generate = None
            try:
                if use_inproc_vllm:
                    vllm_model, self._vllm_lora_request, self._vllm_generation_config = self._load_vllm_model(
                        Path(self._current_checkpoint_dir),
                        tp_size=1,
                        gpu_memory_utilization=0.85,
                        qlora=False,
                        temperature=0.7,
                        top_p=0.95,
                        max_new_tokens=8192,
                    )
                    model_to_generate = vllm_model
                else:
                    # HF batched generate or vLLM HTTP server (server path may
                    # fall back to HF on the training model).
                    model_to_generate = underlying

                eval_start = time.perf_counter()

                # --- Criterion 1: Format Compliance ---
                format_score, format_details = self._check_format_compliance(model_to_generate)

                # --- Criterion 2: Output Diversity ---
                diversity_score, diversity_details = self._check_diversity(model_to_generate)

                # --- Criterion 3: Loss Plateau (soft) ---
                loss_plateaued = self._check_loss_plateau()

                eval_elapsed = time.perf_counter() - eval_start
                print(f"\n  [Timing] Total eval wall-clock: {eval_elapsed:.1f}s")
                
                # Log results to console & checkpoint dir
                self._log_results(
                    state,
                    format_score,
                    diversity_score,
                    loss_plateaued,
                    format_details,
                    diversity_details,
                    args,
                )

                # Check if criteria are met
                format_passed = format_score >= self.format_threshold
                diversity_passed = diversity_score >= self.diversity_threshold

                if format_passed and diversity_passed:
                    self._consecutive_passes += 1
                    if self._consecutive_passes >= self.patience:
                        print(
                            f"\n{'*' * 60}\n"
                            f"  SFT STOPPING CRITERIA MET at step {state.global_step}\n"
                            f"  Format:    {format_score:.1%} >= {self.format_threshold:.1%}\n"
                            f"  Diversity: {diversity_score:.1%} >= {self.diversity_threshold:.1%}\n"
                            f"  Patience:  {self._consecutive_passes}/{self.patience} consecutive passes\n"
                            f"  Loss plateau: {'Yes' if loss_plateaued else 'No'}\n"
                            f"  --> Stopping SFT.  Ready to switch to GRPO.\n"
                            f"{'*' * 60}"
                        )
                        should_stop = True
                    else:
                        print(
                            f"\n  Criteria passed ({self._consecutive_passes}/{self.patience}).  "
                            f"Waiting for {self.patience - self._consecutive_passes} more."
                        )
                else:
                    self._consecutive_passes = 0
                    reasons = []
                    if not format_passed:
                        reasons.append(
                            f"Format {format_score:.1%} < {self.format_threshold:.1%}"
                        )
                    if not diversity_passed:
                        reasons.append(
                            f"Diversity {diversity_score:.1%} < {self.diversity_threshold:.1%}"
                        )
                    print(f"\n  Continuing SFT: {'; '.join(reasons)}")

            finally:
                if use_inproc_vllm and vllm_model is not None:
                    try:
                        vllm_model.llm_engine.engine_core.shutdown()
                    except Exception as exc:
                        logger.warning("vLLM engine shutdown: %s", exc)
                    del vllm_model
                    model_to_generate = None
                    if hasattr(self, "_vllm_lora_request"):
                        del self._vllm_lora_request
                    if hasattr(self, "_vllm_generation_config"):
                        del self._vllm_generation_config
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    gc.collect()
                    try:
                        import ray

                        if ray.is_initialized():
                            ray.shutdown()
                    except Exception:
                        pass

        if use_inproc_vllm:
            assert opt_state_cpu is not None
            _ddp_barrier()
            _move_underlying_params_buffers_to_device(underlying, device)
            load_optimizer_state_from_cpu(
                base_opt, opt_state_cpu, model=underlying
            )
            self.tokenizer.padding_side = old_padding_side

        if was_training:
            model.train()
        
        # Synchronise the stop decision across all DDP ranks so that
        # every process breaks out of the training loop together.
        if is_distributed:
            device_for_broadcast = next(model.parameters()).device
            stop_tensor = torch.tensor(
                int(should_stop), dtype=torch.int32, device=device_for_broadcast,
            )
            torch.distributed.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

        if should_stop:
            control.should_training_stop = True
            self._stopped = True

    def _load_vllm_model(self, adapter_path: Path, tp_size: int = 1, gpu_memory_utilization: float = 0.85, qlora: bool = False, temperature: float = 0.7, top_p: float = 0.95, max_new_tokens: int = 8192):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        with open(adapter_path / "adapter_config.json", "r") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config["base_model_name_or_path"]

        self.old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"  # For generation

        quant_kwargs = {}
        if qlora:
            # Activates handling for bitsandbytes quantized base models natively.
            quant_kwargs["quantization"] = "bitsandbytes"
            quant_kwargs["load_format"] = "bitsandbytes"

        # Initialize vLLM with unmerged dynamic LoRA
        model = LLM(
            model=base_model_name,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=adapter_config.get("r", 8),
            trust_remote_code=True,
            **quant_kwargs
        )
        lora_request = LoRARequest("active_adapter", 1, str(adapter_path))
        generation_config = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.pad_token_id],
        )
        print(f"vLLM Engine initialized on {tp_size} GPUs.")
        print(f"Base Model: {base_model_name}")
        print(f"Adapter dynamically mounted from: {adapter_path.as_posix()}")

        return model, lora_request, generation_config

    # ------------------------------------------------------------------
    # Criterion 1 – Format Compliance
    # ------------------------------------------------------------------
    def _check_format_compliance(self, model) -> tuple[float, dict]:
        """Generate one completion per eval prompt and check parseability."""
        print("[Format Check] Generating completions for eval prompts...")

        total = len(self._eval_prompts)
        details: Dict[str, int] = {
            "total": total,
            "think_ok": 0,
            "tool_call_ok": 0,
            "tool_call_json_ok": 0,
            "input_vector_ok": 0,
            "expected_output_ok": 0,
            "detected_faults_ok": 0,
            "fully_parseable": 0,
        }

        # Process in proper non-overlapping batches
        batch_size = self._generation_batch_size
        all_completions = []
        num_batches = (total + batch_size - 1) // batch_size

        for batch_idx in tqdm(
            range(num_batches), desc="Format check batches", file=sys.stdout
        ):
            start = batch_idx * batch_size
            batch_prompts = self._eval_prompts[start : start + batch_size]
            try:
                completions = self._generate_with_tool_calling(
                    model, batch_prompts, num_return_sequences=1
                )
                all_completions.extend(completions)
            except Exception as e:
                logger.warning(f"Generation failed for batch {batch_idx} (start={start}): {e}")
                all_completions.extend([""] * len(batch_prompts))

        for completion in all_completions[:total]:
            # Check individual components
            has_think = bool(self.THINK_RE.search(completion))
            tc_match = self.TOOL_CALL_RE.search(completion)
            has_tool_call = bool(tc_match)
            has_tool_json = False
            if tc_match:
                try:
                    json.loads(tc_match.group(1))
                    has_tool_json = True
                except (json.JSONDecodeError, ValueError):
                    pass
            has_iv = bool(self.INPUT_VECTOR_RE.search(completion))
            has_eo = bool(self.EXPECTED_OUTPUT_RE.search(completion))
            has_df = bool(self.DETECTED_FAULTS_RE.search(completion))
            
            details["think_ok"] += int(has_think)
            details["tool_call_ok"] += int(has_tool_call)
            details["tool_call_json_ok"] += int(has_tool_json)
            details["input_vector_ok"] += int(has_iv)
            details["expected_output_ok"] += int(has_eo)
            details["detected_faults_ok"] += int(has_df)
            
            # Fully parseable = valid tool call + all summary fields
            if has_tool_json and has_iv and has_eo and has_df:
                details["fully_parseable"] += 1

        score = details["fully_parseable"] / total if total > 0 else 0.0
        return score, details

    # ------------------------------------------------------------------
    # Criterion 2 – Output Diversity
    # ------------------------------------------------------------------
    def _check_diversity(self, model) -> tuple[float, dict]:
        """Generate N completions for a single prompt; count unique vectors."""
        n = self.diversity_num_generations
        print(f"[Diversity Check] Generating {n} completions for a single prompt...")

        prompt = self._eval_prompts[0]
        
        try:
            completions = self._generate_with_tool_calling(
                model, [prompt], num_return_sequences=n
            )
        except Exception as e:
            logger.warning(f"Diversity generation failed: {e}")
            return 0.0, {"error": str(e)}

        # Extract INPUT_VECTORs
        input_vectors: List[Optional[str]] = []
        # not accounted for the score
        expected_outputs: List[Optional[str]] = []
        # not accounted for the score
        detected_faults: List[Optional[str]] = []
        for comp in completions:
            m = self.INPUT_VECTOR_RE.search(comp)
            input_vectors.append(m.group(1).strip() if m else None)
            m = self.EXPECTED_OUTPUT_RE.search(comp)
            expected_outputs.append(m.group(1).strip() if m else None)
            m = self.DETECTED_FAULTS_RE.search(comp)
            detected_faults.append(m.group(1).strip() if m else None)

        parseable = [v for v in input_vectors if v is not None]
        unique_vectors = len(set(parseable)) if parseable else 0
        none_count = sum(1 for v in input_vectors if v is None)

        # Score: unique parseable vectors / total generations
        # Un-parseable outputs count as distinct (different failure modes)
        effective_unique = unique_vectors + min(none_count, 1)
        score = effective_unique / len(completions) if completions else 0.0

        # Also compute raw-text diversity
        unique_texts = len(set(c.strip() for c in completions))

        details = {
            "total_generations": len(completions),
            "parseable_count": len(parseable),
            "unique_input_vectors": unique_vectors,
            "unique_texts": unique_texts,
            "unparseable_count": none_count,
            "effective_unique": effective_unique,
            "sample_vectors": parseable[:3],
        }
        return score, details

    # ------------------------------------------------------------------
    # Criterion 3 – Loss Plateau (soft)
    # ------------------------------------------------------------------
    def _check_loss_plateau(self) -> bool:
        if len(self._recent_losses) < self.loss_window:
            return False
        recent = self._recent_losses[-self.loss_window :]
        start_idx = max(0, len(self._recent_losses) - 2 * self.loss_window)
        end_idx = len(self._recent_losses) - self.loss_window
        older = self._recent_losses[start_idx:end_idx]
        if not older:
            return False
        improvement = float(np.mean(older)) - float(np.mean(recent))
        return improvement < self.loss_delta

    # ==================================================================
    # Generation with multi-turn tool calling
    # ==================================================================
    def _generate_with_tool_calling(
        self,
        model,
        prompts: List[str],
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate completions with multi-turn tool calling.

        Flow
        ----
        1. Generate first assistant turn (stops at EOS / ``<|im_end|>``).
        2. Parse ``<tool_call>`` from the completion.
        3. Execute the tool and obtain a result string.
        4. Build a continuation prompt with the tool response injected.
        5. Generate second assistant turn (the summary).
        6. Return the concatenated full completion for each sequence.

        Parameters
        ----------
        model
            The (unwrapped) model to generate with.
        prompt : str
            The formatted prompt string (from ``apply_chat_template``).
        num_return_sequences : int
            Number of independent completions to produce.

        Returns
        -------
        list[str]
            Full completions (first_turn + tool_response + second_turn).
        """
        if self.use_vllm:
            if self.vllm_server_url:
                return self._generate_vllm_server(model, prompts, num_return_sequences)
            else:
                return self._generate_n_completions_vllm(model, prompts, num_return_sequences)
        return self._generate_hf(model, prompts, num_return_sequences)

    # ------------------------------------------------------------------
    # HuggingFace model.generate() backend  (batched two-turn)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_hf(
        self,
        model,
        prompts: List[str],
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Optimised **batched** two-turn generation using ``model.generate()``.

        Performance strategy
        --------------------
        Instead of issuing 2 × N *sequential* ``model.generate()`` calls
        (one per prompt per turn), this method processes prompts in
        sub-batches:

        1. **Turn 1** — all prompts batched → first-turn completions
           (think + tool_call).
        2. **Tool execution** — parse ``<tool_call>`` JSON, run the tool
           function on CPU (negligible wall-clock time).
        3. **Turn 2** — all continuation prompts batched → second-turn
           completions (summary with INPUT_VECTOR / EXPECTED_OUTPUT /
           DETECTED_FAULTS).

        For ``eval_buffer_size=30`` and ``generation_batch_size=8`` this
        gives ~8 batched calls instead of ~60 sequential ones (≈7× faster).

        Returns
        -------
        list[str]
            Full completions.
            Length = ``len(prompts) * num_return_sequences``.
        """
        device = next(model.parameters()).device
        orig_pad_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        gen_bs = self._generation_batch_size

        try:
            # ==============================================================
            # TURN 1: Batch-generate first assistant turn (think + tool_call)
            # ==============================================================
            first_turns: List[str] = []
            # Maps each element of first_turns → index of source prompt
            prompt_index_map: List[int] = []

            t0 = time.perf_counter()

            for start in range(0, len(prompts), gen_bs):
                batch = prompts[start : start + gen_bs]

                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_new_tokens,
                ).to(device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                decoded = self.tokenizer.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                first_turns.extend(decoded)

                # num_return_sequences completions per prompt in the batch
                for j in range(len(batch)):
                    for _ in range(num_return_sequences):
                        prompt_index_map.append(start + j)

            t1 = time.perf_counter()
            logger.info(
                "[SFTEval] Turn 1: %d completions in %.1fs (%.1f comp/s)",
                len(first_turns),
                t1 - t0,
                len(first_turns) / max(t1 - t0, 1e-6),
            )

            # ==============================================================
            # TOOL EXECUTION  (CPU-bound — fast)
            # ==============================================================
            full_completions: List[str] = list(first_turns)

            # Collect items that need a second-turn generation
            # Each item: (index-into-first_turns, first_turn_text,
            #             tool_result_str, continuation_prompt_str)
            second_turn_items: List[tuple] = []

            for idx, ft in enumerate(first_turns):
                tool_call = self._parse_tool_call(ft)
                if not tool_call:
                    continue

                tool_call["arguments"].update({"netlist": ToolHelper.get_netlist(prompts[prompt_index_map[idx]])})
                tool_result = self._execute_tool_call(tool_call)

                try:
                    messages = self._build_continued_messages(
                        prompts[prompt_index_map[idx]],
                        ft,
                        tool_call,
                        tool_result,
                    )
                    cont_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        tools=self.tools_schema,
                        add_generation_prompt=True,
                    )
                    second_turn_items.append(
                        (idx, ft, tool_result, cont_prompt)
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to build continuation for turn %d: %s",
                        idx,
                        exc,
                    )

            t2 = time.perf_counter()
            logger.info(
                "[SFTEval] Tool execution: %d tool calls in %.1fs",
                len(second_turn_items),
                t2 - t1,
            )

            # ==============================================================
            # TURN 2: Batch-generate second assistant turn (summary)
            # ==============================================================
            if second_turn_items:
                cont_prompts = [item[3] for item in second_turn_items]
                second_turns: List[str] = []

                for start in range(0, len(cont_prompts), gen_bs):
                    batch = cont_prompts[start : start + gen_bs]

                    cont_inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_new_tokens // 2,
                    ).to(device)

                    cont_outputs = model.generate(
                        **cont_inputs,
                        max_new_tokens=self.max_new_tokens // 2,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=0.9,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                    decoded = self.tokenizer.batch_decode(
                        cont_outputs[:, cont_inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                    second_turns.extend(decoded)

                # Stitch: first_turn + <tool_response>…</tool_response> + 2nd turn
                for j, (orig_idx, ft, tool_result, _) in enumerate(
                    second_turn_items
                ):
                    full_completions[orig_idx] = (
                        ft
                        + f"\n<tool_response>\n{tool_result}\n</tool_response>\n"
                        + second_turns[j]
                    )

                t3 = time.perf_counter()
                logger.info(
                    "[SFTEval] Turn 2: %d completions in %.1fs (%.1f comp/s)",
                    len(second_turns),
                    t3 - t2,
                    len(second_turns) / max(t3 - t2, 1e-6),
                )

            return full_completions

        finally:
            self.tokenizer.padding_side = orig_pad_side

    # ------------------------------------------------------------------
    # vLLM server backend (persistent server on a separate GPU)
    # ------------------------------------------------------------------
    def _generate_vllm_server(
        self,
        model,
        prompts: List[str],
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate completions via a **persistent** vLLM server.

        Expects ``vllm serve <base_model> --enable-lora`` to be running
        (e.g. on a spare GPU) at :pyattr:`vllm_server_url`.  At each
        evaluation the current LoRA adapter is loaded on the server from
        the latest checkpoint directory — no model migration, no engine
        creation / destruction.

        Falls back to :meth:`_generate_hf` when the server is unreachable
        or the adapter cannot be loaded.
        """
        import requests as http_requests

        base_url = self.vllm_server_url.rstrip("/")

        # Health check
        try:
            resp = http_requests.get(f"{base_url}/health", timeout=10)
            if resp.status_code != 200:
                raise ConnectionError(f"status {resp.status_code}")
        except Exception as exc:
            logger.warning(
                "[SFTEval] vLLM server unreachable at %s (%s). "
                "Falling back to HF generation. Start a server with:\n"
                "  CUDA_VISIBLE_DEVICES=<spare_gpu> vllm serve <model> "
                "--enable-lora --port 8000",
                base_url,
                exc,
            )
            return self._generate_hf(model, prompts, num_return_sequences)

        # Load / refresh LoRA adapter on the server
        if not self._load_vllm_adapter(self._current_checkpoint_dir):
            logger.warning(
                "[SFTEval] Could not load LoRA adapter on vLLM server. "
                "Falling back to HF generation."
            )
            return self._generate_hf(model, prompts, num_return_sequences)

        adapter_name = "sft_eval"
        # ==============================================================
        # TURN 1: think + tool_call
        # ==============================================================
        t0 = time.perf_counter()
        first_turns, prompt_index_map = self._vllm_server_complete(
            base_url,
            adapter_name,
            prompts,
            n=num_return_sequences,
            max_tokens=self.max_new_tokens,
        )
        t1 = time.perf_counter()
        logger.info(
            "[SFTEval/vLLM] Turn 1: %d completions in %.1fs (%.1f comp/s)",
            len(first_turns),
            t1 - t0,
            len(first_turns) / max(t1 - t0, 1e-6),
        )

        # ==============================================================
        # TOOL EXECUTION  (CPU-bound — fast)
        # ==============================================================
        full_completions: List[str] = list(first_turns)
        second_turn_items: List[tuple] = []

        for idx, ft in enumerate(first_turns):
            tool_call = self._parse_tool_call(ft)
            tool_call["arguments"].update({"netlist": ToolHelper.get_netlist(prompts[prompt_index_map[idx]])})
            tool_result = self._execute_tool_call(tool_call)
            
            try:
                messages = self._build_continued_messages(
                    prompts[prompt_index_map[idx]],
                    ft,
                    tool_call,
                    tool_result,
                )
                cont_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    tools=self.tools_schema,
                    add_generation_prompt=True,
                )
                second_turn_items.append(
                    (idx, ft, tool_result, cont_prompt)
                )
            except Exception as exc:
                logger.warning(
                    "Failed to build continuation for turn %d: %s",
                    idx,
                    exc,
                )
            
        t2 = time.perf_counter()
        logger.info(
            "[SFTEval/vLLM] Tool execution: %d tool calls in %.1fs",
            len(second_turn_items),
            t2 - t1,
        )

        # ==============================================================
        # TURN 2: summary
        # ==============================================================
        if second_turn_items:
            cont_prompts = [item[3] for item in second_turn_items]
            second_turns, _ = self._vllm_server_complete(
                base_url,
                adapter_name,
                cont_prompts,
                n=1,
                max_tokens=self.max_new_tokens // 2,
            )
            
            for j, (orig_idx, ft, tool_result, _) in enumerate(
                second_turn_items
            ):
                full_completions[orig_idx] = (
                    ft
                    + f"\n<tool_response>\n{tool_result}\n</tool_response>\n"
                    + second_turns[j]
                )
            
            t3 = time.perf_counter()
            logger.info(
                "[SFTEval/vLLM] Turn 2: %d completions in %.1fs (%.1f comp/s)",
                len(second_turns),
                t3 - t2,
                len(second_turns) / max(t3 - t2, 1e-6),
            )

        return full_completions

    # ------------------------------------------------------------------
    # vLLM server helpers
    # ------------------------------------------------------------------
    def _vllm_server_complete(
        self,
        base_url: str,
        adapter_name: str,
        prompts: List[str],
        n: int,
        max_tokens: int,
    ) -> tuple[List[str], List[int]]:
        """Send prompts to the vLLM ``/v1/completions`` endpoint.

        Uses ``max_tokens`` (OpenAI-compatible) to control output length.
        Caps max_tokens so input_tokens + max_tokens <= vllm_max_context
        (avoids CUDA out-of-bounds when exceeding model's max_position_embeddings).
        """
        import requests as http_requests

        # Cap max_tokens so we never exceed vllm_max_context (e.g. 32768 for Qwen2.5)
        max_input_len = max(
            len(self.tokenizer.encode(p, add_special_tokens=True)) for p in prompts
        )
        effective_max_tokens = min(max_tokens, self.vllm_max_context - max_input_len)
        effective_max_tokens = max(1, effective_max_tokens)  # ensure at least 1

        json_data = {
            "model": adapter_name,
            "prompt": prompts,
            "max_tokens": effective_max_tokens,
            "temperature": self.temperature,
            "top_p": 0.9,
            "n": n,
        }
        resp = http_requests.post(
            f"{base_url}/v1/completions",
            json=json_data,
            timeout=1200,
        )
        resp.raise_for_status()
        data = resp.json()

        choices = sorted(data["choices"], key=lambda c: c["index"])
        texts = [c["text"] for c in choices]
        index_map = [c["index"] for c in choices]
        return texts, index_map

    def _load_vllm_adapter(self, checkpoint_dir: Optional[str]) -> bool:
        """Load a LoRA adapter on the vLLM server (idempotent per path).

        Uses the ``/v1/load_lora_adapter`` endpoint which requires
        ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`` when starting the
        server (vLLM >= 0.11 V1 engine).
        """
        if checkpoint_dir is None:
            return False
        if self._vllm_adapter_loaded == checkpoint_dir:
            return True

        import requests as http_requests

        base_url = self.vllm_server_url.rstrip("/")
        adapter_name = "sft_eval"
        abs_path = os.path.abspath(checkpoint_dir)

        # Unload previous adapter (ignore errors on first call)
        try:
            http_requests.post(
                f"{base_url}/v1/unload_lora_adapter",
                json={"lora_name": adapter_name},
                timeout=30,
            )
        except Exception:
            pass

        try:
            resp = http_requests.post(
                f"{base_url}/v1/load_lora_adapter",
                json={"lora_name": adapter_name, "lora_path": abs_path},
                timeout=60,
            )
            if resp.status_code == 200:
                self._vllm_adapter_loaded = checkpoint_dir
                print(f"[SFTEval] Loaded LoRA adapter from {abs_path}")
                return True

            if resp.status_code == 404:
                logger.warning(
                    "[SFTEval] /v1/load_lora_adapter returned 404.  "
                    "The vLLM V1 engine requires the env var "
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True to expose "
                    "dynamic LoRA endpoints.  Restart the server:\n"
                    "  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True "
                    "CUDA_VISIBLE_DEVICES=<gpu> vllm serve <model> "
                    "--enable-lora --max-lora-rank 64 --port <port>"
                )
                return False

            logger.warning("[SFTEval] Adapter load failed: %s", resp.text)
            return False
        except Exception as exc:
            logger.warning("[SFTEval] Adapter load error: %s", exc)
            return False

    def _generate_n_completions_vllm(
        self,
        llm,
        prompts,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate n completions simultaneously using vLLM.
        Tracks state of n independent generation paths to manage divergent tool calls natively.
        """
        # Initialize n completely independent conversation tracks
        states = [
            {
                "prompt_idx": prompt_idx,
                "current_input": prompt,
                "full_completion": "",
                "done": False,
            }
            for prompt_idx, prompt in enumerate(prompts)
            for _ in range(num_return_sequences)
        ]

        for _round in range(self._max_tool_rounds + 1):
            # Identify paths that still need generation
            active_indices = [i for i, state in enumerate(states) if not state["done"]]
            if not active_indices:
                break
            
            active_inputs = [states[i]["current_input"] for i in active_indices]
            
            # Batch generate for all active paths using vLLM
            outputs = llm.generate(
                active_inputs,
                sampling_params=self._vllm_generation_config,
                lora_request=self._vllm_lora_request,
                use_tqdm=False, # Disable nested tiny progress bars
            )
            
            for i, output in zip(active_indices, outputs):
                completion_text = output.outputs[0].text
                states[i]["full_completion"] += completion_text
                
                if _round < self._max_tool_rounds:
                    tool_call = self._parse_tool_call(completion_text)
                    if tool_call is not None:
                        # Get the netlist from the current input
                        tool_call["arguments"].update({"netlist": ToolHelper.get_netlist(states[i]["current_input"])})
                        
                        # Execute tool synchronously for this specific path
                        tool_result = self._execute_tool_call(tool_call)
                        
                        # Rebuild the conversation history just for this specific track
                        messages = self._build_continued_messages(
                            states[i]["current_input"],
                            completion_text,
                            tool_call,
                            tool_result,
                        )
                        
                        states[i]["current_input"] = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            tools=TOOLS,
                            add_generation_prompt=True,
                        )
                        states[i]["full_completion"] += f"\n<tool_response>\n{tool_result}\n</tool_response>\n"
                        # This track needs another round - leave done=False
                    else:
                        states[i]["done"] = True
                else:
                    # No tool call or max rounds reached
                    states[i]["done"] = True

        return [state["full_completion"] for state in states]

    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse a tool call from model completion text.
        
        Expected format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        
        Parameters
        ----------
        text : str
            The completion text to parse.
        
        Returns
        -------
        Optional[Dict]
            Parsed tool call dict with 'name' and 'arguments', or None.
        """
        import regex as re
        match = self.TOOL_CALL_RE.search(text)
        if match:
            try:
                tool_call_data = json.loads(match.group(1))
                if "name" in tool_call_data:
                    return tool_call_data
            except json.JSONDecodeError:
                pass
        return None


    # ------------------------------------------------------------------
    # Tool execution helper
    # ------------------------------------------------------------------
    def _execute_tool_call(self, tool_call: dict) -> str:
        """Execute a tool function (handles both sync and async)."""
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})
        func = self.tool_functions.get(tool_name)
        if func is None:
            return f"Error: Unknown tool '{tool_name}'"
        try:
            if asyncio.iscoroutinefunction(func):
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(func(**tool_args))
                finally:
                    loop.close()
            else:
                result = func(**tool_args)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    # ------------------------------------------------------------------
    # Message construction helper
    # ------------------------------------------------------------------
    def _build_continued_messages(
        self,
        original_prompt: str,
        first_turn: str,
        tool_call_data: dict,
        tool_result: str,
    ) -> list:
        """
        Parse the original prompt back to messages, append the first-turn
        assistant message (with tool call) and the tool response, ready for
        ``apply_chat_template(add_generation_prompt=True)``.
        """
        from revert_template import revert_qwen2_5_template, revert_chat_template 

        # Strip the trailing generation prompt suffix before parsing
        clean = original_prompt
        gen_suffix = "<|im_start|>assistant\n"
        if clean.endswith(gen_suffix):
            clean = clean[: -len(gen_suffix)]

        messages = revert_qwen2_5_template(clean)

        # Content before <tool_call>
        tc_start = first_turn.find("<tool_call>")
        content_before = first_turn[:tc_start].strip() if tc_start >= 0 else first_turn.strip()

        assistant_msg: Dict[str, Any] = {"role": "assistant"}
        if content_before:
            assistant_msg["content"] = content_before
        assistant_msg["tool_calls"] = [
            {"type": "function", "function": tool_call_data}
        ]
        messages.append(assistant_msg)

        # Tool response
        messages.append(
            {"role": "tool", "name": tool_call_data.get("name", ""), "content": tool_result}
        )
        return messages

    # ------------------------------------------------------------------
    # Logging & persistence
    # ------------------------------------------------------------------
    def _log_results(
        self,
        state,
        format_score: float,
        diversity_score: float,
        loss_plateaued: bool,
        format_details: dict,
        diversity_details: dict,
        args,
    ) -> None:
        step = state.global_step

        print(f"\n[SFTStoppingCallback] Results at step {step}:")
        print(
            f"  Format Compliance:  {format_score:.1%}  "
            f"(threshold: {self.format_threshold:.1%})"
        )
        if format_details:
            t = format_details["total"]
            for key in [
                "think_ok",
                "tool_call_ok",
                "tool_call_json_ok",
                "input_vector_ok",
                "expected_output_ok",
                "detected_faults_ok",
                "fully_parseable",
            ]:
                label = key.replace("_ok", "").replace("_", " ").title()
                print(f"    {label:.<28s} {format_details[key]:>3d}/{t}")

        print(
            f"  Output Diversity:   {diversity_score:.1%}  "
            f"(threshold: {self.diversity_threshold:.1%})"
        )
        if isinstance(diversity_details, dict) and "error" not in diversity_details:
            print(f"    Generations ............ {diversity_details.get('total_generations', 'N/A')}")
            print(f"    Parseable .............. {diversity_details.get('parseable_count', 'N/A')}")
            print(f"    Unique INPUT_VECTORs ... {diversity_details.get('unique_input_vectors', 'N/A')}")
            print(f"    Unique texts ........... {diversity_details.get('unique_texts', 'N/A')}")

        print(f"  Loss Plateau:       {'Yes' if loss_plateaued else 'No'}")

        # Persist to checkpoint directory
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        results = {
            "step": step,
            "format_score": format_score,
            "format_threshold": self.format_threshold,
            "format_details": format_details,
            "diversity_score": diversity_score,
            "diversity_threshold": self.diversity_threshold,
            "diversity_details": {
                k: v
                for k, v in diversity_details.items()
                if k != "sample_vectors"
            }
            if isinstance(diversity_details, dict)
            else {},
            "loss_plateaued": loss_plateaued,
            "criteria_met": (
                format_score >= self.format_threshold
                and diversity_score >= self.diversity_threshold
            ),
            "consecutive_passes": self._consecutive_passes,
        }
        results_path = os.path.join(
            checkpoint_dir, "sft_stopping_results.json"
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Results saved to {results_path}")
