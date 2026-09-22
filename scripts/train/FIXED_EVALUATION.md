**Granite GRPO/GDPO training across Slurm allocations**

The main Granite 4.2 config and its resume config enable the same fixed
evaluation protocol. They retain `MAX_STEPS=200`, LR `5e-6`, ten warmup steps,
and the original batch sizes. The two-day allocation is a wall-time limit;
200 is the cumulative optimizer-step target across allocations.

From the `atpgllm` repository root, start a new SFT-to-GRPO run with:

```bash
./scripts/train/submit_training_code.sh configs/grpo_granite_4.2_8b.conf
```

After that job exits, continue its checkpoint with:

```bash
./scripts/train/submit_training_code.sh configs/grpo_granite_4.2_8b_resume.conf
```

The resume config restores the policy/reference adapters, optimizer, scheduler,
global step and RNG state. `AUTO_SKIP_FROM_RESUME=True` recovers the original
stream-buffer offset; it does not skip ahead another 60,000 examples.
Keep the same total steps, LR/warmup settings, training GPU count, batch settings,
dataset, and stream offset across recovery jobs. Training finishes when the
cumulative target is reached; there is no automatic early-stopping callback.
The optional `grpo_granite_4.2_8b_pilot.conf` remains a separate 20-step experiment.

`RESUME_FROM=latest` now selects the **most recently completed save**, using
`training_state_summary.json` publication time. It checks trainer step,
optimizer, scheduler, and all expected rank RNG files and skips partial saves.
This prevents an older run's higher-numbered checkpoint from winning inside a
reused output directory. A new save clears an old completion marker before
writing files, then publishes its summary atomically. Use a distinct output
directory for each new experiment; avoid concurrent jobs writing the same one.
Checkpoints are still saved every optimizer step, so work since the last
completed save can be repeated after Slurm terminates a job.

**Loss and prompt corrections**

With three training GPUs, accumulation 384 and generation steps 16, each update
contains 24 global generation batches of 48 completions. TRL 0.26.1's DAPO
denominator covers a generation batch, and its base Trainer bypasses the usual
accumulation division. `GenerationBatchLossMixin` multiplies training DAPO/CISPO
contributions by `steps_per_generation / current_gradient_accumulation_steps`
(1/24 here), so the update averages generation-batch token means. Variable
lengths and tool masks retain their token weighting *within* a generation batch;
this is not a single token average over all 1,152 completions. It requires no
larger rollout buffer. Short final accumulations use their actual length.
Evaluation and loss types that already divide by accumulation are not rescaled.
The fused Liger DAPO/CISPO path is rejected because it has a different contract.
The CPU regression checks exercise the installed TRL loss directly, so rerun
them before changing TRL/Transformers versions.

Both custom trainers remove only the open generation placeholder before
converting rendered prompts back to messages. Completed assistant history is
preserved. Granite's template-owned `<think>` opening is restored in assistant
message history for tool continuation, while completion token IDs and their
log-probabilities are unchanged. The dual-adapter weight sync invalidates the
vLLM prefix cache and synchronizes ranks after each transfer, including the
evaluation-to-training transition.

Applying these corrections to an existing checkpoint preserves optimizer and
scheduler state but changes the subsequent loss scaling and prompts. The
continuation is therefore not a bitwise replay of the old implementation.

**Frozen faults and continuation data**

Both long-run configs use:

```text
FIXED_EVAL_SIZE=72
FIXED_EVAL_SPLIT=test
FIXED_EVAL_MANIFEST=runs/grpo_granite_4.2_8b_bon32/fixed_eval_manifest.json
FIXED_EVAL_SEED=1729
FIXED_EVAL_STEPS=5
FIXED_EVAL_GENERATIONS=1
FIXED_EVAL_BATCH_SIZE=1
```

The first launch freezes 72 unique circuit/fault pairs from a bounded pool of
576 eligible test examples, using seeded circuit-round-robin selection. It
stores exact rendered prompts, authoritative faults, netlists, tokenizer and
protocol hashes. This is a bounded evaluation sample, not a uniform sample of
the entire test split. A missing split or too few eligible faults fails clearly.

For a new SFT-to-GRPO run, evaluation circuits are excluded from the training
buffer by document identity **or** exact netlist content before the first update.
For an older GRPO checkpoint that has no evaluation metadata, introducing a
holdout must not delete training rows and invalidate its saved sampler position.
In that case the training buffer is preserved and evaluation candidates are
filtered to circuits absent from the whole buffered training set. Finding the
first eligible pool may require scanning further into the test split. If the
dataset does not contain enough disjoint test faults, startup fails instead of
silently evaluating training circuits or changing the resume cursor.

Every new checkpoint carries `fixed_eval_manifest.json` and
`grpo_data_state.json`. Later resumes recover a missing external manifest from
the checkpoint, reject a conflicting external manifest, reapply the original
holdout policy, and verify the exact ordered training-buffer fingerprint.
Keep the dataset source available and unchanged so the buffer can be rebuilt.
Tokenizer/template, generation, simulator, topology, and protocol changes are
rejected on continuation. Start a new experiment/manifest for such changes.

Circuit exclusion guards against GRPO leakage; it does not establish that the
earlier SFT dataset lacked those circuits, or detect renamed/structurally
equivalent circuits. Keep this limitation in mind when interpreting quality.

**Evaluation curves and artifacts**

Evaluation runs before training begins and every five cumulative optimizer
updates. A fresh run's initial policy is the SFT baseline. A resumed run's initial
evaluation scores the restored policy at its restored step. For a legacy run,
that provides the first fixed baseline; no historical SFT score is invented.

Trainer history uses `eval_fixed/*`; Transformers rewrites these names to
**`eval/fixed/*` in W&B**. The W&B x-axis is `eval/fixed/global_step`, which
continues the optimizer-step numbering across allocations. Use:

- `eval/fixed/detection`: raw simulator detections / all completions (greedy pass@1).
- `eval/fixed/solved_at_k`: faults detected by the single greedy completion (`k=1`).
- `eval/fixed/activation_without_detection` and `all_fail_fraction`.
- `eval/fixed/simulation_valid_fraction` and `simulator_error_fraction`.
- `eval/fixed/pi_complete_fraction` and `expected_output_exact_fraction`.

Simulator errors and missing/unusable predictions remain failures in the
detection denominator. The separate error and validity metrics distinguish
infrastructure errors from unsuccessful predictions. A reward-factory failure
returning scalar zeros aborts evaluation instead of producing a false score.
`pi_complete_fraction` requires complete valid PI assignments as scored by the
reward function. Exact expected-output accuracy requires every scored primary
output to match the simulator.

Training `train/rewards/reward_fn/raw_mean` records the weighted objective sum
before GDPO normalization; raw component means remain available separately.
Training/group/diversity diagnostics gather across ranks, so local prompt groups
are not mistaken for global batches. `train/reward` and `eval/reward` remain
normalized GDPO diagnostics with means near zero, and are not quality curves.

Each evaluation writes `OUTPUT_DIR/fixed_eval/step-000014.json`, for example,
with the protocol/checksum, raw metrics and every completion's components.
Repeated evaluation at a resumed step gets a `-repeat-<id>` suffix so earlier
records are preserved. These files remain together even when resumed jobs have
separate W&B run IDs. Counts are checked across ranks; missing or duplicated
samples fail rather than silently biasing a metric.

Evaluation uses temperature 0, top-p 1, one greedy completion per fault, and
fixed request seeds, including tool continuations. Training temperature remains 1.
Use `eval/fixed/detection` in W&B (`eval_fixed/detection` in trainer history)
as the greedy quality curve. TRL group-standard-deviation
diagnostics are undefined for one completion and are not evaluation quality metrics.
The evaluation kwargs are scoped to `evaluate()` and restored even on failure.
Old stochastic evaluation manifests are incompatible: begin a new experiment
with a new output/manifest path; full-state resumes must retain the saved protocol.
The main/repaired/TetraMAX configs use separate `_bon32` output directories.
Do not compare the old sampled pass@3 directly with the new greedy detection curve.
Python/NumPy/Torch RNG state, generation kwargs, completion logs and model
mode are restored afterward. Fixed seeds/settings do not guarantee bitwise
identity across different hardware or policies. Compare fixed evaluation over
several checkpoints and report uncertainty; 72 faults are useful for catching
large regressions but not establishing very small improvements. Evaluation
consumes time inside each Slurm allocation.

**Training best-of-N (independent of fixed evaluation)**

The main, repaired, TetraMAX, and main resume configs set:

```text
NUM_GENERATIONS=16
TRAIN_BEST_OF_N=32
```

For each training prompt, sample 32 trajectories at temperature 1, finish all
tool calls, score using the configured simulator, and retain the highest-reward
16. `NUM_GENERATIONS` continues to control the optimizer group and batch layout.
`TRAIN_BEST_OF_N=0` (the CLI default), or setting N equal to G, retains ordinary
i.i.d. sampling. N must be a multiple of G and at least G; G must be at least 2.
The candidate pool is expanded before generation, so N/G increases rollout memory,
generation work, and simulator requests; policy/reference forwards only see G.
The shared TetraMAX worker/license limit is unchanged.

Ranking uses raw reward objectives and their configured weights, before GDPO
normalization, with stable ties. Log-only diagnostics are excluded. The native
reward profile controls available objectives; simulator infrastructure errors
still abort, and a group with fewer than G finite scores fails explicitly.
Selected trajectories are redistributed in global prompt-group order, including
groups spanning ranks. Tokens, tool masks, log probabilities, and extra fields
move together. Selected rows are scored again through the normal reward path
(which can reuse simulator caches), and GDPO normalizes only the retained group.
Generation/token counters include candidates; the DAPO denominator counts only
retained model tokens. Both adapter trainers use this path.

This is reward-selected GRPO/GDPO, not an unbiased estimator of ordinary on-policy
GRPO. Existing vLLM importance ratios correct model/engine log-probability
differences; they do not correct the reward-selection distribution. Selecting
only high scores can reduce within-group variation and therefore the learning
signal. Keep G >= 2, monitor group degeneracy and greedy fixed evaluation, and
compare against `TRAIN_BEST_OF_N=0` before treating this as an improvement.

`sampling/best_of_n/` logs candidate count, retained count, candidate raw reward
mean, and selected raw reward mean. The launcher snapshot and W&B config record
`TRAIN_BEST_OF_N`; keep it unchanged across full-state resumes. This is implemented
in the trainer, rather than vLLM's `best_of` option: selection requires the final
simulator reward after tool execution.
