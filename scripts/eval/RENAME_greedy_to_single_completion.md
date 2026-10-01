# Rename: sampling method `greedy` → `single_completion` (2026-10-01)

The evaluator's model-based sampling method formerly called `greedy` is now `single_completion`.
It was never greedy decoding: it is one independent model trajectory per slot (temperature/top-p apply,
tool calls allowed). `SAMPLING_METHOD=greedy` / `--sampling_method greedy` are now rejected (no alias).

Deliberately **unchanged** (they really are greedy, or are unrelated):
- `--greedy_tool_calls` / `GREEDY_TOOL_CALLS` / `GreedyToolCallGenerator` / `_gtc` tags (temperature-0 tool-call bodies).
- Training fixed evaluation (`"sampling_method": "greedy"` with temperature 0.0 in `scripts/train/training_code.py`,
  `runs/*/fixed_eval*`, `analysis/language_of_test_20260925/`, fault-difficulty pilot rows) — true greedy decoding,
  and its SHA-256 provenance chain stays valid.
- Greedy set-cover compaction, non-greedy regexes, tokenizer vocab, and model completions that say "greedy approach".

Originals of every edited/renamed data file are backed up outside the repo in
`/proj/trela/christos/rename_backups/greedy_to_single_completion_20261001/`
(`data_runs_wandb.backup/`, `wandb_yaml.backup/`, `analysis_backup/`, old→new rename map `file_renames.tsv`,
pre-rename working-tree diff `pre_rename.patch`, and the rename/regeneration scripts used).

## Changes in `scripts/eval/`

- `README.md` — `SAMPLING_METHOD` table
- `analyze_decisive_bits.py` — runs `random` + `single_completion`; globs `*_passatk_single_completion_*`
- `eval_grpo_policy_checkpoints.sh` — default, validation `case`, comments (replaced atomically: a job was running from it)
- `eval_sft_policy_checkpoints.sh` — default, validation `case`, comments
- `evaluate_model.py` — default/choices/help/docstrings of `--sampling_method`

## Rerun status

New evaluations write `*_passatk_single_completion_*` / `*_single_completion_*_stdout.log` and W&B run names `..._policy_single_completion`. Anything outside this repo that calls `SAMPLING_METHOD=greedy` must be updated.
