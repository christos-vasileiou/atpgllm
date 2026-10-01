# Executing the Language-of-Test evaluation

The implemented CPU replay follows the D/S/U definitions in
[the protocol](proposals/LANGUAGE_OF_TEST_EVALUATION_PROTOCOL.md). The first
campaign and its limitations are recorded in
[the execution report](LANGUAGE_OF_TEST_EXECUTION_20260925.md).

All commands below run from `libatpgllm`. Use an environment with the project's
simulator dependencies. Output paths must be new: the scripts refuse to replace
an existing campaign. These scripts do not modify training or its manifests.

## Replay, split audit, and coverage

```bash
python scripts/eval/execute_language_of_test.py audit-split \
  --dataset ../data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.stil_repaired_v1 \
  --read-parquet --output analysis/new_campaign/split_audit.json

python scripts/eval/execute_language_of_test.py replay \
  --manifest runs/grpo_granite_4.2_8b_repaired_a100_bon32/fixed_eval_manifest.json \
  --records runs/grpo_granite_4.2_8b_repaired_a100_bon32/fixed_eval/step-000005.json \
  --output analysis/new_campaign/grpo5 --reference tetramax --coverage

python scripts/eval/execute_language_of_test.py audit-coverage \
  --replay analysis/new_campaign/grpo5 \
  --output analysis/new_campaign/grpo5_coverage --workers 2
```

TetraMAX commands require the site's existing Synopsys environment and valid
64-bit library paths. They use the existing shared license-seat pool. No GPU is
used. There is no fallback to Python on native failure. In the September 25
campaign, commands ran on `eng-tamale3` with
`/proj/trela/christos/myenv/bin/python`, after sourcing
`/proj/cad/startup/profile.synopsys_2018_vcs_2021` and placing
`/home/eng/c/cxv200006/usr/lib64` first in `LD_LIBRARY_PATH`.

`slots.jsonl` retains raw answers, strict assignments, D/S/U, missing/error
statuses, good/faulty outputs, and native provenance. `fault_manifest.json`
freezes both polarities of every declared-net stem, including raw alias labels;
it does not include pin branches. Coverage records save the ordered streams,
deduplicated vectors, detection matrices, uniform controls, and deterministic
greedy compaction at each stream's own witnessed coverage. Counts at different
coverage are not a compaction improvement. Historical generation costs are
unavailable; offline audit runtime is reported separately.

Unknown native matrix entries create coverage bounds. They do not become
verified negatives or proven untestable faults. Testability bounds based only
on complete detection measurements are omitted for an incomplete matrix.
The optional `--reuse-reference PATH` explicitly reuses old measured cells,
including unknowns, for identical circuit/vector/fault identities. The source
matrix hash is saved; this is not a new verification run.

## Fresh no-feedback and fault-conditioning runs

Two CPU-prepared manifests are already available in
`analysis/language_of_test_20260925/prepared_original/manifest.json` and
`analysis/language_of_test_20260925/prepared_opposite_fault/manifest.json`.
Each reserves 16 independently seeded slots for each of 72 validation problems,
temperature 0.6, top-p 0.95, 4,096 output tokens, and a 32,768-token context.
These are pilot settings, not settings selected and locked on a separate test
set. All full formatted prompts fit: maximum 800 tokens.

The control changes only the prompted stuck polarity, and keeps the original
target as the scoring fault. Score both outputs against that same original
target. It tests fault conditioning; it does not replace the required
renaming, family holdout, or representation ablations.

On an idle GPU, evaluate a checkpoint explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/generate_lot_no_feedback.py \
  --prepared analysis/language_of_test_20260925/prepared_original/manifest.json \
  --tokenizer runs/sft_granite_4.2_8b_repaired/checkpoint-90 \
  --checkpoint runs/grpo_granite_4.2_8b_repaired_a100_tetramax_bon32/checkpoint-1 \
  --output analysis/new_campaign/grpo_tetramax1_original

python scripts/eval/execute_language_of_test.py replay \
  --manifest analysis/new_campaign/grpo_tetramax1_original/manifest.json \
  --records analysis/new_campaign/grpo_tetramax1_original/records.json \
  --reference tetramax --coverage --k 1 5 10 16 \
  --output analysis/new_campaign/grpo_tetramax1_original_replay
```

Repeat with the opposite-fault manifest and a new output directory. Use the
same manifests/tokenizer for the base model (omit `--checkpoint`), SFT checkpoint
90, custom-simulator GRPO checkpoints 5 and 7, and TetraMAX GRPO checkpoints 1–3.
The adapter paths and September 25 weight hashes are in
`analysis/language_of_test_20260925/checkpoint_matrix.json`. A GRPO root resolves
to its `policy/` adapter. The loader activates that adapter explicitly, with
the NF4 base and bfloat16 compute used by the existing training loader.

Generation records per-slot seeds, token counts, terminal reasons, cumulative
wall time and model requests. It provides no simulator feedback or repairs.
Missing, context-exhausted, truncated, and generation-error slots remain
failures. Models are loaded from the local cache. CPU preparation and protocol
tests passed; fresh GPU model loading/generation has **not** been validated in
this campaign because no free working GPU was available.

## Evidence still required

Before a generalization claim, reserve a family-disjoint test suite that was
never used to choose checkpoints or budgets, audit canonical structural overlap,
and complete semantic renaming/reordering and conditioned-fault controls.
Add base-model and representation/rationale ablations, ATPG and budget-matched
model-free controls, actual generation/search cost trajectories, and training
seed replication. Neither the existing validation pilot nor extra proxy
metrics establish the full language-of-test hypothesis.
