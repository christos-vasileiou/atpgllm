# H100 GRPO configuration review — 2026-09-24

The two H100 configs use three training GPUs and one dedicated vLLM completion
GPU. Both preserve the A100 training objective, repaired SFT checkpoint-90,
best-of-32 selection retaining 16 trajectories, 32K total context, learning rate,
and fixed evaluation settings. Output directories and evaluation manifests stay
separate between simulators. No jobs were submitted.

## Changes

- Custom fast simulator: `FAULT_SIM_BACKEND=fast`, `TMAX_MANAGE_SERVICE=False`.
  The draft enabled service management, which the launcher rejects for fast mode.
- Native simulator: `FAULT_SIM_BACKEND=tetramax`, managed service, 16-worker
  shared license budget, `po` reward profile, and pipelining disabled pending a
  GPU pilot. A working Synopsys environment must be loaded or supplied through
  `TMAX_SERVICE_MODULE`/`TMAX_BIN` as documented in `docs/TETRAMAX.md`.
- Both: TP=1, DP=1, batch size 1, accumulation 384, and generation interval 64.
  Corrected copied A100 topology, context, and unverified runtime comments.
- Shared `ds_zero2.json`: both communication buckets are now explicit
  16,777,216-element integers. The reduction bucket equals Granite's 4096 squared
  hidden size, matching Transformers' normal auto resolution. The all-gather
  bucket decreases from 500,000,000 elements. This lowers communication-buffer
  memory at a possible communication-throughput cost. Batch, accumulation,
  clipping, and BF16 remain `auto`, so A100 and other users retain their own
  training settings. `auto` reduction was not intrinsically invalid when
  Transformers resolved it; explicit sizing removes that dependency.

## Arithmetic and memory

| Quantity | Value |
| --- | --- |
| Training ranks | 4 - (1 TP × 1 DP) = 3 |
| Retained sequences per optimizer step | 3 × 1 × 384 = 1,152 |
| Prompt groups per optimizer step | 1,152 / 16 = 72 |
| Retained sequences per generation batch | 3 × 1 × 64 = 192 |
| Candidate trajectories per generation batch | (192 / 16) × 32 = 384 |
| Generation batches per optimizer step | 384 / 64 = 6 |

The generation batch is divisible by 16, and accumulation is divisible by 64.
The A100 reference uses 2 × 1 × 576 = 1,152, so optimizer batch size is preserved.
Three training ranks need not themselves be divisible by 16.

The [published Granite config](https://huggingface.co/ibm-granite/granite-4.2-8b/raw/main/config.json)
has 40 layers, 8 KV heads, hidden size 4096, and 32 attention heads. With BF16 KV
cache, a full 32,768-token trajectory needs
`2 × 40 × 8 × (4096 / 32) × 2 × 32768 = 5 GiB` before sharing or overhead.
The approximately 8B BF16 serving weights need roughly 15–16 GiB. One 80GB H100
therefore has plausible capacity for weights and several full-length sequences,
but cannot hold all 32 independent full-length trajectories simultaneously.
vLLM schedules within its cache budget; 384 candidates is rollout volume, not a
promise of simultaneous residency. Actual cache capacity, activation peaks,
preemption, and throughput require a GPU pilot. Training uses the existing 4-bit
model loading and gradient checkpointing; ZeRO-2 does not shard model weights.

## Operational findings and validation

Live Slurm queries confirmed `g-04-02` has four full H100 80GB HBM3 GPUs and the
`h100` partition permits a two-day allocation. The partition also contains MIG
resources, so the existing node pin is retained. Availability can change.
The dataset directory and SFT checkpoint exist locally; the adapter metadata
matches Granite 4.2 8B, rank 32, and alpha 64.

Both configs pass shell syntax and arithmetic checks. Both real launcher dry
runs select GPUs 0–2 for three training processes and GPU 3 for vLLM; only the
native config requests the TetraMAX service. Mock-sbatch checks confirm config
freezing and all requested Slurm resource flags. DeepSpeed schema validation and
Transformers config reconciliation pass for both 3 × 384 and 2 × 576 layouts,
resolving BF16 and the 1,152-sequence training batch correctly.
Dry runs do not validate GPU memory,
native simulator execution, or actual training.

Before a long run, use a separate pilot output/manifest and a short MAX_STEPS
schedule to measure peak memory, generation speed, simulator queue delays, fixed
evaluation, and checkpoint save/resume. If buffering or preemption is excessive,
STEPS_PER_GENERATION=16 is arithmetically valid: 48 retained / 96 candidate
trajectories per generation batch, with the same 1,152-sequence optimizer batch.
This changes generation grouping, so compare observed training behavior. Keep the
16-worker license budget independent of rollout count. If generation dominates,
a 2-training/2-serving alternative uses TP=2 and accumulation 576; it requires a
new run or weights-only restart when changing the optimizer partition count.

The original A100 fast config also contains `TMAX_MANAGE_SERVICE=True`, so it has
the same launcher rejection if used as-is. It was retained as the requested
reference; change that setting to False before launching that A100 fast variant.
