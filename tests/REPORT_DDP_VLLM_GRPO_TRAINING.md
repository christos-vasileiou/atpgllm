# DDP + vLLM Server — GRPO Training Pipeline Report

> **Scope:** DualAdapterGRPOTrainer running with `accelerate launch --multi_gpu`
> and a separate `trl vllm-serve` process for generation.
>
> **Key files:**
> - `run_training_code.sh` — GPU allocation, vLLM server launch, accelerate launch
> - `training_code.py` — dataset prep, model loading, GRPOConfig, trainer instantiation
> - `dual_adapter_grpo_trainer.py` — dual-adapter logic, generation, tool-call loop, scoring
> - TRL `grpo_trainer.py` (upstream) — base training loop, buffered generation, loss computation

---

## Table of Contents

1. [Hardware Layout](#1-hardware-layout)
2. [Dual-Adapter Architecture](#2-dual-adapter-architecture)
3. [High-Level Training Loop](#3-high-level-training-loop)
4. [Step-by-Step: What Happens in a Single Optimizer Step](#4-step-by-step-what-happens-in-a-single-optimizer-step)
5. [Communication Patterns Between GPUs](#5-communication-patterns-between-gpus)
6. [Multi-Turn Tool Calling Loop (DDP-Safe)](#6-multi-turn-tool-calling-loop-ddp-safe)
7. [GPU Utilisation Patterns](#7-gpu-utilisation-patterns)
8. [Generalisation to Arbitrary num_generations and steps_per_generation](#8-generalisation-to-arbitrary-num_generations-and-steps_per_generation)
9. [Bugs Found and Fixes Applied](#9-bugs-found-and-fixes-applied)

---

## 1. Hardware Layout

`run_training_code.sh` allocates 4 GPUs and splits them as follows:

| GPU | Process | Role |
|-----|---------|------|
| #0 | `accelerate` rank 0 (main process) | Training + DDP coordination |
| #1 | `accelerate` rank 1 | Training |
| #2 | `accelerate` rank 2 | Training |
| #3 | `trl vllm-serve` | Inference-only generation server |

The vLLM server is started **before** training on the last GPU
(`VLLM_GPU=${GPU_ARRAY[-1]}`).  `CUDA_VISIBLE_DEVICES` is then trimmed to the
first N-1 GPUs for the `accelerate launch` command, so training processes never
compete with vLLM for GPU memory.

Communication between the training processes and vLLM happens over two channels:

- **HTTP** — rank 0 sends generation requests to `http://localhost:<port>/generate/`
- **NCCL** — rank 0 pushes updated model weights to GPU #3 via
  `vllm_client.update_named_param()`, which uses an NCCL communicator
  initialised at startup (`/init_communicator` endpoint)

---

## 2. Dual-Adapter Architecture

Standard GRPOTrainer calls `merge_and_unload()` when given a PeftModel, fusing
the SFT LoRA into the 4-bit base weights.  This causes precision loss because
the merged result gets re-quantised to 4-bit.

DualAdapterGRPOTrainer avoids this by keeping **two LoRA adapters** side by side:

```
Base model (4-bit quantised, frozen)
  └── "reference" adapter  (frozen SFT LoRA)
  └── "policy" adapter     (trainable LoRA, initialised from SFT weights)
```

During forward passes:

- **Policy logps:** both adapters are active →
  `output = base + reference_ΔW + policy_ΔW`
- **Reference logps:** only the SFT adapter is active →
  `output = base + reference_ΔW`

This is achieved by monkey-patching `model.disable_adapter()` so that when the
parent class enters its `with model.disable_adapter():` block for reference
computation, it actually switches to "SFT-only" mode rather than disabling all
adapters.

A DDP-specific subtlety: during the reference forward pass, the policy adapter's
parameters are not used.  DDP normally expects every parameter that
`requires_grad=True` to appear in the computational graph (to synchronise
gradient buckets).  Since the policy adapter isn't used, DDP would deadlock
waiting for gradient buckets that never arrive.  The fix is to bypass the DDP
wrapper and use the unwrapped model for the reference forward pass.

---

## 3. High-Level Training Loop

TRL's GRPOTrainer uses a **buffered generation** strategy controlled by two
parameters:

| Parameter | Meaning |
|-----------|---------|
| `steps_per_generation` (S) | How many training micro-steps each generation batch covers |
| `gradient_accumulation_steps` (A) | How many micro-steps before an optimizer step |
| `num_generations` (G) | Completions generated per prompt |
| `per_device_train_batch_size` (B) | Micro-batch size per GPU |

The dataloader loads `B × S` prompts per GPU per generation call.  Each prompt
is duplicated `G` times (for GRPO's group-relative scoring), yielding
`B × S × G` samples per GPU.  These are split into `S` slices of `B × G` for
training.

**One generation cycle:**

```
Step 0:  Generate & score  →  buffer S slices  →  train on slice 0
Step 1:                                         →  train on slice 1
  ...
Step S-1:                                       →  train on slice S-1
Step S:  Generate & score  →  buffer S slices  →  train on slice 0
  ...
```

Each training step performs a full forward + backward pass.  Every `A` steps the
gradients are all-reduced across ranks and the optimizer updates weights.

---

## 4. Step-by-Step: What Happens in a Single Optimizer Step

Using the concrete configuration from the user's command:

```
B=1, A=128, G=2, S=2, max_model_len=8192, max_completion_length=6144
```

### 4.1. Generation and Scoring (every 2 micro-steps)

This is the expensive phase.  All ranks enter `_prepare_inputs` →
`_generate_and_score_completions` together.

#### 4.1.1. Weight Push to vLLM

Only once per generation batch (when `global_step` has changed):

1. **Rank 0** iterates over every LoRA-wrapped layer in the model.
2. For each layer:
   - Dequantises the 4-bit base weight → bfloat16
   - Adds `ΔW = B·A·scaling` for every active adapter (reference + policy)
   - Moves the merged tensor to `cuda:0`
   - Calls `vllm_client.update_named_param(name, tensor)` which sends the
     tensor over NCCL to GPU #3
3. **Ranks 1, 2** are idle during this phase (no collective ops involved).
4. **GPU #3** receives the updated weights via NCCL.

#### 4.1.2. Initial Generation via vLLM

`_generate_single_turn()` in vLLM server mode:

1. **All ranks** call `gather_object(prompts)` — an NCCL collective that
   serialises each rank's prompt list and sends them to rank 0.
2. **Rank 0** deduplicates prompts (takes every `G`-th prompt since they arrive
   duplicated `G` times), applies the chat template, and sends an HTTP POST to
   `http://localhost:<port>/generate/` with `n=G` (request `G` completions per
   unique prompt).
3. **GPU #3** runs autoregressive generation (attention, sampling, KV cache)
   and returns `prompt_ids`, `completion_ids`, `logprobs`.
4. **Rank 0** receives the HTTP response and calls
   `broadcast_object_list(payload, from_process=0)`.
5. **All ranks** receive the broadcast.  Each rank slices its portion:
   ```python
   process_slice = slice(rank * len(local_prompts), (rank + 1) * len(local_prompts))
   ```

#### 4.1.3. Decoding and Tool-Call Loop

After initial generation, completions are decoded from token IDs to text.  Then
the multi-turn tool calling loop runs (detailed in
[Section 6](#6-multi-turn-tool-calling-loop-ddp-safe)).

#### 4.1.4. Metric Aggregation

After the tool loop completes, all ranks gather completion lengths, truncation
flags, and tool-call counts via `self.accelerator.gather()`.  These are logged
to wandb.

#### 4.1.5. Scoring — Policy Log-Probabilities (`old_per_token_logps`)

Back in `_generate_and_score_completions` (the parent class), all ranks
perform a **full forward pass** of the model on the concatenated
`[prompt_ids | completion_ids]` tensors to compute per-token log-probabilities
under the current policy.

This is needed for importance sampling correction: vLLM's weights may be
slightly stale (they were last synced before tool continuations modified
the conversation), so the ratio `π_current / π_vllm` corrects for the mismatch.

**All 3 training GPUs** are active here — each processes its local batch slice.

#### 4.1.6. Scoring — Reference Log-Probabilities (`ref_per_token_logps`)

Another **full forward pass** on all 3 GPUs, this time with only the SFT adapter
active.  The monkey-patched `disable_adapter()` context manager switches to
SFT-only mode.  The DDP wrapper is bypassed for this pass to avoid gradient
bucket deadlocks.

#### 4.1.7. Reward Computation

The reward function (e.g., `fault_simulation_tool_handler`-based reward) is
called on all ranks.  Results are gathered across ranks because GRPO normalises
rewards per group, and completions may be distributed across processes.

#### 4.1.8. Advantage Computation

Rewards are grouped by prompt (groups of `G` completions), mean-subtracted, and
optionally normalised.  The resulting advantages determine which completions
the policy should learn to produce more often.

#### 4.1.9. Buffering

The scored batch is split into `S` slices and buffered:
```python
generation_batches = split_tensor_dict(generation_batch, steps_per_generation)
self._buffered_inputs = [...]
```

### 4.2. Training Micro-Step (every step, including non-generation steps)

For every micro-step, all 3 GPUs perform:

1. **Policy forward pass** — compute `per_token_logps` and entropy under the
   current policy (both adapters active)
2. **Loss computation** — DAPO loss using advantages, importance sampling ratios,
   and optionally the KL divergence penalty `β · KL(π_ref || π_policy)`
3. **Backward pass** — compute gradients through the model (gradient
   checkpointing is enabled to reduce memory)
4. Every `A=128` micro-steps: **gradient all-reduce** across all 3 ranks via
   NCCL, followed by optimizer step and learning rate scheduler step

---

## 5. Communication Patterns Between GPUs

### 5.1. Communication Diagram

```
GPU #0 (rank 0)          GPU #1 (rank 1)          GPU #2 (rank 2)           GPU #3 (vLLM)
      │                        │                        │                         │
      │── NCCL update_named_param (weight push) ─────────────────────────────────►│
      │                        │                        │                         │
      │◄── gather_object ─────►│◄── gather_object ─────►│                         │
      │   (prompts to rank 0)  │   (prompts to rank 0)  │                         │
      │                        │                        │                         │
      │── HTTP POST /generate/ ──────────────────────────────────────────────────►│
      │                        │                        │  (autoregressive gen)   │
      │◄── HTTP 200 OK ─────────────────────────────────────────────────────────◄─│
      │                        │                        │                         │
      │── broadcast_object_list ───────────────────────►│                         │
      │   (completions)        │◄──────────────────────►│                         │
      │                        │                        │                         │
      │    ┌───── TOOL CALL LOOP (all ranks in lockstep) ──────┐                  │
      │    │  gather(sync_flag)  — any rank has tool calls?    │                  │
      │    │  gather(max_tokens) — agree on generation budget  │                  │
      │    │  gather_object(prompts) — collect to rank 0       │                  │
      │    │  HTTP POST to vLLM (rank 0 only) ───────────────────────────────────►│
      │    │  broadcast_object_list(results) to all ranks      │                  │
      │    │  (repeat until no rank has tool calls)            │                  │
      │    └───────────────────────────────────────────────────┘                  │
      │                        │                        │                         │
      │◄── gather (metrics) ──►│◄── gather (metrics) ──►│                         │
      │                        │                        │                         │
      │    ┌───── SCORING (all ranks compute locally) ─────────┐                  │
      │    │  Forward pass: old_per_token_logps (policy)       │                  │
      │    │  Forward pass: ref_per_token_logps (SFT only)     │                  │
      │    │  Reward function evaluation                       │                  │
      │    │  gather(rewards) — for group normalisation        │                  │
      │    └───────────────────────────────────────────────────┘                  │
      │                        │                        │                         │
      │    ┌───── TRAINING (all ranks compute locally) ─────────┐                 │
      │    │  Forward pass: per_token_logps + entropy           │                 │
      │    │  Loss computation (DAPO)                           │                 │
      │    │  Backward pass                                     │                 │
      │    │  Every A steps: NCCL all-reduce gradients          │                 │
      │    │  Optimizer step                                    │                 │
      │    └────────────────────────────────────────────────────┘                 │
```

### 5.2. Summary of Collective Operations

| Operation | When | Participants | Type |
|-----------|------|-------------|------|
| `update_named_param` | Before first generation of each step | Rank 0 → GPU #3 | NCCL point-to-point |
| `gather_object` | Initial generation + each tool continuation | Ranks 0,1,2 | NCCL all-gather |
| `broadcast_object_list` | After vLLM returns results | Ranks 0,1,2 | NCCL broadcast |
| `accelerator.gather(sync_tensor)` | Tool loop Phase 2 | Ranks 0,1,2 | NCCL all-gather |
| `accelerator.gather(metrics)` | After generation completes | Ranks 0,1,2 | NCCL all-gather |
| `gather(rewards_per_func)` | During reward computation | Ranks 0,1,2 | NCCL all-gather |
| DDP gradient sync | Every `A` micro-steps | Ranks 0,1,2 | NCCL all-reduce |

---

## 6. Multi-Turn Tool Calling Loop (DDP-Safe)

The tool-call loop is the most complex part of the pipeline because:

- Different ranks may have different numbers of samples containing tool calls
- vLLM generation requires all ranks to participate in collective operations
- The loop must terminate synchronously across all ranks

### 6.1. Overview

After the initial generation, each completion is scanned for
`<tool_call>{"name": ..., "arguments": ...}</tool_call>` tags.  If any sample
on any rank contains a tool call, the loop executes:

```
WHILE any rank has pending tool calls:
    Phase 1 (local):    Execute tools, check overlong, prepare prompts
    Phase 2 (collective): Vote on termination, sync max_tokens
    Phase 3 (collective): Generate continuations via vLLM
    Phase 4 (local):    Stitch results, parse next tool calls
```

### 6.2. Phase Details

#### Phase 1 — Local Tool Execution

Each rank independently:
1. Builds the full conversation: `[prompt] + [assistant completion] + [tool result]`
2. Calls the matched tool function (e.g., `fault_simulation_tool_handler`)
3. Appends the tool result as a `{"role": "tool", ...}` message
4. Tokenises the full conversation and checks if `len(tokens) >= max_model_len`
5. **Overlong handling:** if the conversation exceeds the context window,
   the completion is truncated to `max_completion_length` tokens and the
   sample is removed from further tool-call processing
6. For surviving samples, computes `local_max_tokens = min(max_completion_length,
   max_model_len - max_conversation_length)` — the budget for the next generation

#### Phase 2 — DDP Synchronisation

All ranks participate in two collective operations:

1. **Termination vote:** each rank creates a tensor `[1]` if it has prompts
   needing generation, `[0]` otherwise.  `accelerator.gather()` collects these.
   If the sum is 0, all ranks break out of the loop together.

2. ~~**Max tokens agreement** (removed — see Section 9.4)~~.
   Previously, each rank gathered its `local_max_tokens` and the minimum across
   all ranks was used.  This was removed because it degraded training quality;
   vLLM's per-prompt capping makes it unnecessary.

#### Phase 3 — DDP-Safe Generation

All ranks call `_generate_tool_continuation()` together, even those with no
local prompts (they pass empty lists).

Inside `_generate_tool_continuation`:
1. Each rank sends its prompt count via `accelerator.gather(counts_tensor)`
2. `gather_object(prompts)` collects all prompts to rank 0
3. Rank 0 applies chat templates, sends HTTP POST to vLLM with `n=1`
   (one completion per unique prompt — no deduplication needed because tool
   continuations are unique per sample) and `max_tokens=max_completion_length`.
   vLLM internally caps each completion to
   `min(max_tokens, max_model_len - prompt_length)`, so short prompts get the
   full budget while long prompts are naturally limited.
4. `broadcast_object_list(payload)` sends results back to all ranks
5. Each rank computes its offset from the gathered counts and slices its
   portion of the results

#### Phase 4 — Result Stitching

Each rank locally:
1. Verifies that re-tokenised conversations preserve the original prompt prefix
2. Truncates if `old_completion + tool_tokens + new_output > max_completion_length`
3. Updates `completion_ids = old_completion + tool_result_tokens + new_model_tokens`
4. Updates `tool_mask`: `[...existing 1s...] + [0s for tool tokens] + [1s for model tokens]`
5. Updates `logprobs`: `[...existing...] + [0.0 for tool tokens] + [new logprobs from vLLM]`
6. Decodes the new model output and parses for further `<tool_call>` tags
7. If found, the loop repeats from Phase 1

### 6.3. Tool Mask

The `tool_mask` is a per-token binary list (same length as `completion_ids`)
where:
- `1` = token was produced by the model (participates in loss computation)
- `0` = token was injected as a tool result (excluded from loss)

This ensures the policy is only trained on tokens it actually generated, not
on the deterministic tool output that was inserted into the conversation.

---

## 7. GPU Utilisation Patterns

### 7.1. Why GPUs #1 and #2 show high utilisation at all times

Two factors:

**A. Training compute dominates the timeline.**

With `steps_per_generation=2`, every generation call is followed by 2 training
micro-steps.  Each micro-step involves a full forward + backward pass through
a 7B-parameter model.  Even with 4-bit quantisation and LoRA, the backward pass
(with gradient checkpointing) is expensive.  The generation pause (vLLM HTTP
round-trip + autoregressive decoding on GPU #3) is a small fraction of the total
wall-clock time.  GPUs #1 and #2 are computing for the vast majority of the time.

**B. NCCL polling during collective operations.**

During the brief moments when ranks 1 and 2 are waiting for rank 0 to finish
the HTTP call to vLLM (inside `gather_object` and `broadcast_object_list`),
NCCL runs GPU-side polling kernels.  `nvidia-smi` reports GPU utilisation as
"percentage of time at least one kernel was running."  NCCL's polling kernels
count as running kernels even though they're doing no useful math — just
spinning while waiting for data.

### 7.2. Expected Utilisation by Phase

| Phase | GPU #0 | GPU #1 | GPU #2 | GPU #3 |
|-------|--------|--------|--------|--------|
| Weight push to vLLM | NCCL send | idle | idle | NCCL receive |
| Initial generation (vLLM) | HTTP wait + NCCL | NCCL wait | NCCL wait | **Active** |
| Tool execution | CPU-bound | CPU-bound | CPU-bound | idle |
| Tool-continuation generation | HTTP wait + NCCL | NCCL wait | NCCL wait | **Active** |
| Scoring: old_per_token_logps | **Forward pass** | **Forward pass** | **Forward pass** | idle |
| Scoring: ref_per_token_logps | **Forward pass** | **Forward pass** | **Forward pass** | idle |
| Reward computation | **Compute** | **Compute** | **Compute** | idle |
| Training forward + backward | **Forward+Backward** | **Forward+Backward** | **Forward+Backward** | idle |
| Gradient all-reduce | NCCL all-reduce | NCCL all-reduce | NCCL all-reduce | idle |

The "NCCL wait" entries appear as GPU utilisation in `nvidia-smi` but represent
no useful computation.  This is normal and expected.

---

## 8. Generalisation to Arbitrary num_generations and steps_per_generation

### 8.1. num_generations (G)

`G` controls how many completions are sampled per prompt.  These completions
form a "group" for GRPO's advantage computation (mean-subtraction within group).

- **Initial generation:** the dataloader duplicates each prompt `G` times.
  `_generate_single_turn` deduplicates by taking every `G`-th prompt, sends
  unique prompts to vLLM with `n=G`, then re-expands results.

- **Tool continuations:** each of the `G` completions may diverge (different
  tool calls, different results).  `_generate_tool_continuation` handles this
  with `n=1` (each prompt gets exactly one completion, no deduplication).

- **Batch size constraint:** `per_device_train_batch_size × G` must fit in
  memory during scoring (two forward passes over the full prompt+completion
  sequence).

### 8.2. steps_per_generation (S)

`S` controls amortisation: generate once, train `S` times on different slices.

- **Dataloader:** loads `B × S` prompts per GPU (the prompts for `S`
  micro-steps worth of training are fetched at once).
- **Buffering:** after generation and scoring, the batch is split into `S`
  slices stored in `_buffered_inputs`. Each training step consumes one slice.
- **Generation frequency:** completions are regenerated every
  `S × num_iterations` micro-steps.  Between regenerations, the model trains
  on stale completions, which is corrected by importance sampling
  (`old_per_token_logps / current_per_token_logps`).

**Trade-offs:**

| S | Pros | Cons |
|---|------|------|
| Small (1–2) | Fresh completions, small memory peak | More vLLM calls, lower throughput |
| Large (=A) | Fewer vLLM calls, higher throughput | Stale completions, larger memory peak |

### 8.3. Interaction Between G and S

The total number of samples generated per GPU per generation call is `B × S × G`.
All of these are scored (2 forward passes + rewards) before any training step
runs.  The memory high-water mark during scoring is proportional to
`B × S × G × (prompt_length + completion_length)`.

Example configurations:

| B | G | S | A | Samples per GPU per gen | Training steps between gens |
|---|---|---|---|------------------------|---------------------------|
| 1 | 2 | 2 | 128 | 4 | 2 |
| 2 | 4 | 4 | 64 | 32 | 4 |
| 1 | 8 | 1 | 128 | 8 | 1 |

---

## 9. Bugs Found and Fixes Applied

### 9.1. Bug #1 — Context Window Overflow

**Symptom:**
```
ValueError: The decoder prompt (length 8867) is longer than the
maximum model length of 8192.
```
Training deadlocked because the vLLM server rejected the prompt, the HTTP
connection hung, and all ranks blocked waiting for results.

**Root cause:**
The tool-call loop used the model's `max_position_embeddings` (131072 for
Qwen2.5) as the ceiling for conversation length, instead of the vLLM server's
`--max-model-len` (8192).  After a few tool-call rounds, conversations grew
beyond 8192 tokens and were sent to vLLM, which rejected them.

Additionally, `max_model_len` was not being passed from `run_training_code.sh`
through `training_code.py` to the trainer.

**Fix:**
1. `run_training_code.sh` now passes `--max_model_len`, `--max_completion_length`,
   and `--max_prompt_length` as CLI arguments.
2. `training_code.py` forwards `vllm_max_model_len` to the trainer constructor.
3. `_custom_tool_call_loop_impl` uses `vllm_max_model_len` as the hard ceiling
   for the overlong check and dynamically caps `max_tokens` for each
   continuation so that `conversation_length + max_tokens <= max_model_len`.

---

### 9.2. Bug #2a — Prompt Deduplication Mismatch in Tool Continuations

**Symptom:**
Training freezes silently.  vLLM logs show `200 OK` (generation succeeded) but
the training process hangs after receiving the response.  GPU #3 goes idle.
No error message anywhere.

**Root cause:**
`_generate_single_turn` was designed for the initial generation where prompts
arrive duplicated `G` times.  It deduplicates by taking every `G`-th prompt:

```python
ordered_set_of_prompts = all_prompts[::num_generations]  # e.g. [::2]
output = vllm_client.generate(prompts=..., n=num_generations)
```

For initial generation with prompts `[A, A, B, B]`, `[::2]` → `[A, B]`, and
`n=2` gives 2 completions each.  Correct.

But during tool continuations, every prompt is **unique** — different tool
results make each conversation diverge.  With prompts `[A', B']`:
- `[::2]` → `[A']` — prompt `B'` is silently dropped
- vLLM generates for `A'` only
- Rank tries to use results for both samples, gets misaligned data

This led to corrupted conversation state, infinite loops of nonsensical tool
calls, or silent hangs.

**Fix:**
Created `_generate_tool_continuation()` — a dedicated generation method for
tool continuations that uses `n=1` (one completion per unique prompt, no
deduplication).

---

### 9.3. Bug #2b — DDP Collective Operation Deadlock in Tool Loop

**Symptom:**
Same as 2a — silent hang, regardless of DDP or not (in non-DDP mode, Bug 2a
alone caused the hang; in DDP mode, both bugs contributed).

**Root cause:**
In DDP mode, `_generate_single_turn` internally uses `gather_object()` and
`broadcast_object_list()` — these are NCCL collective operations requiring all
ranks to participate simultaneously.

In the tool loop, different ranks could finish tool processing at different
times.  Example deadlock:

```
Rank 0: sample has a tool call → enters _generate_single_turn →
         calls gather_object(), waits for rank 1

Rank 1: sample has NO tool call → exits tool loop → moves to
         metrics section → calls accelerator.gather(metrics),
         waits for rank 0
```

Rank 0 is stuck in `gather_object()` waiting for rank 1.
Rank 1 is stuck in `accelerator.gather()` waiting for rank 0.
**Classic deadlock.** Neither will ever proceed.

**Fix:**
Rewrote `_custom_tool_call_loop_impl` with four DDP-safe phases:

1. **Phase 1 (local):** execute tools, check overlong, prepare prompts
2. **Phase 2 (collective vote):** all ranks vote on whether *any* rank still
   has work.  If no rank has work, all break together.  If any does, all ranks
   agree on the tightest `max_tokens` budget.
3. **Phase 3 (collective generation):** all ranks enter
   `_generate_tool_continuation()` together.  Ranks with no local work pass
   empty prompt lists but still participate in the internal `gather_object` /
   `broadcast_object_list` calls.
4. **Phase 4 (local):** stitch results, update masks/logprobs, parse for
   further tool calls.

This guarantees that no rank ever calls a collective operation while another
rank is executing a different collective elsewhere.

---

### 9.4. Issue #3 — Cross-Rank max_tokens Agreement Degraded Training Quality

**Symptom:**
No crash or deadlock, but training quality was silently degraded.

**Root cause:**
During tool-call continuations, Phase 2 gathered each rank's `local_max_tokens`
(the headroom before the longest conversation on that rank would hit
`max_model_len`) and used the **minimum** across all ranks as the generation
budget for every prompt.

This meant: if Rank 1 had a 7500-token conversation (leaving 692 tokens of
headroom) and Rank 0 had a 3000-token conversation (leaving 5192 tokens of
headroom), **all** completions on **all** ranks were limited to 692 tokens.
Rank 0's completions — which had room for a full, detailed response — were
truncated to a fraction of their potential length.

In multi-turn tool-calling, the model's second-turn answer (after receiving tool
results) is often the most important part of the completion: it contains the
final reasoning and answer.  Truncating it forces the model to learn from
incomplete responses, which directly harms reward signal quality.

**Why the old approach was unnecessary:**
vLLM already handles per-prompt length limits natively.  When you send
`max_tokens=6144` but a particular prompt is already 7000 tokens long with
`max_model_len=8192`, vLLM generates at most `8192 - 7000 = 1192` tokens for
that prompt — it never errors, it just generates fewer tokens.  The error only
occurs when the prompt *itself* exceeds `max_model_len`, and Phase 1's overlong
check already removes those.

**Fix:**
Removed the `gather(max_tokens)` collective and the min-across-ranks logic.
`_generate_tool_continuation` now always uses `max_completion_length` as the
`max_tokens` parameter.  vLLM's internal per-prompt capping ensures that long
conversations don't exceed the context window, while short conversations get
the full generation budget.

---

*Report generated from codebase analysis.  All fixes are in
`dual_adapter_grpo_trainer.py`.*
