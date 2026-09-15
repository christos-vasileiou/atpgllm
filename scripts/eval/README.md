# Checkpoint evaluation

From the `atpgllm` repository, supply only the checkpoint:

```bash
./scripts/eval/eval_sft_policy_checkpoints.sh runs/sft_granite_4.2_8b/checkpoint-200
./scripts/eval/eval_grpo_policy_checkpoints.sh runs/grpo_granite_4.2_8b/checkpoint-40
```

The same scripts work with any base model supported by `evaluate_model.py` and
the installed inference backend. The evaluator reads the model identifier and
LoRA configuration from `adapter_config.json`; no model name or size is required.
These replace the four `eval_{sft,grpo}_{7b,32b}_policy_checkpoints.sh` scripts.

Paths can be absolute, relative to the current directory, relative to the
repository, or relative to `runs/` (with `tests/` as a legacy fallback), in that
order. SFT checkpoints contain `adapter_config.json` directly. GRPO checkpoints
contain `policy/adapter_config.json` or legacy `combined/policy/adapter_config.json`.
You can also pass either GRPO policy directory directly to select that adapter.

Passing an experiment directory evaluates its `checkpoint-*` children in numeric
order, excluding `*_merged_bf16` exports. Missing adapters are skipped during a
multi-checkpoint run; an invalid single checkpoint or no usable adapters fails.
With no positional argument, the scripts accept `CHECKPOINT`, `EXP_ROOT`, or
`POLICY_FOLDER` (in that precedence order). There is no default experiment.

```bash
# Preview without loading a model or creating output directories.
DRY_RUN=1 ./scripts/eval/eval_sft_policy_checkpoints.sh runs/sft_granite_4.2_8b/checkpoint-200

# Evaluate all saved GRPO checkpoints with a different sampling strategy.
SAMPLING_METHOD=best_of_n BEST_OF_N_WIDTH=4 \
  ./scripts/eval/eval_grpo_policy_checkpoints.sh runs/grpo_granite_4.2_8b
```

Evaluation uses vLLM and the active Python environment, falling back to the
existing cluster virtual environments when none is active. GPU count is inferred
from `CUDA_VISIBLE_DEVICES`, otherwise `nvidia-smi -L`. `TP_SIZE` overrides it when
the model or allocation needs a different tensor parallel size. A dry run on a
machine without `nvidia-smi` uses `TP_SIZE=1`. Actual evaluation requires enough
GPU memory for the model and evaluation settings.

The previous runtime defaults are shared by both scripts:

| Environment variable | Default |
| --- | --- |
| `EVAL_DATASET` | `chrivasileiou/asap7-language-of-test-v2` |
| `SAMPLING_METHOD` | `greedy` (`random`, `best_of_n`, `mcts`, `evolutionary`, `vector_evolutionary` also supported) |
| `NUM_COMPLETIONS` | `50` (`NUM_SAMPLES` is an alias) |
| `PASS_AT_K` | `1 2 4 8 16` (each must be ≤ `NUM_COMPLETIONS`) |
| `SEARCH_BUDGET` | `50`, only for `mcts` / `evolutionary` / `vector_evolutionary` |
| `SEARCH_CONFIG` | Unset; optional validated search JSON file |
| `BEST_OF_N_WIDTH` | `4`, only for `best_of_n` |
| `TEMPERATURE` / `TOP_P` | `0.7` / `0.95` |
| `MAX_NEW_TOKENS` / `MAX_PROMPT_LENGTH` | `16384` / `4096` |
| `MAX_EVAL_SAMPLES` | `512` |
| `EVAL_PROMPT_BATCH_SIZE` / `GENERATION_MICRO_BATCH_SIZE` | `16` / `16` |
| `GPU_MEMORY_UTILIZATION` | `0.85` |
| `THRESHOLD_MODE` | `fault_detected` |
| `MERGE_DEQUANT` | `1`: merged bf16 export from the NF4 base used in QLoRA training; `0`: clean base + dynamic LoRA |
| `REPORT_TO` | `wandb` (`none` disables reporting) |
| `WANDB_RUN_NAME` | Derived from experiment, checkpoint, and sampling method |
| `EVAL_RESULTS_DIR` | `runs/eval_results_<experiment>_policy` |
| `DRY_RUN` | `0` |

Each checkpoint produces a metrics JSON and a stdout log in `EVAL_RESULTS_DIR`.
Choose `EVAL_DATASET` to match the training dataset when using a different source.

For the MCTS and evolutionary sampling design, historical implementation
review, and scientific references, see
[MCTS and evolutionary sampling with tools](../../docs/search/README.md).
The documents distinguish delivered behavior from remaining experiments.

The pass@k evaluator now uses `conversation-search-v1` for tool-aware model
sampling, with independent per-completion budgets and final-answer scoring.
See [run instructions and configuration](../../docs/search/IMPLEMENTATION.md).
`SEARCH_CONFIG` (or `--search_config`) accepts a JSON configuration; a starting
file is [conversation_search.json](configs/conversation_search.json).
The `vector_evolutionary` model-free baseline accepts `SEARCH_BUDGET` as well.
New launcher filenames include `csv1`; each metrics JSON also has a neighboring
`*.slots.jsonl` with trajectories and usage. Training reward parsing is unchanged.
