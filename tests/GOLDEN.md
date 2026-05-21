# Golden training recipe: Qwen2.5 + asap7-language-of-test-v2

Frozen reference for the two-session pipeline: **SFT** then **GRPO** on
`chrivasileiou/asap7-language-of-test-v2` (synthesized structural netlists, language-of-test).

## Code pin

| Item | Value |
|------|--------|
| Branch | `golden/qwen2.5-sft-grpo-asap7` |
| Base commit | `09b5154` (`main` at branch creation) |
| Entrypoint | `tests/training_code.py` |
| Slurm wrapper | `tests/run_training_code.sh` |

## Workspace layout (required at runtime)

Clone or check out **both** trees so `tests/training_code.py` can import fault simulation:

```text
<workspace>/
├── atpgllm/                 ← this repo (golden branch)
└── data_preprocessing/      ← sibling repo; provides fault_sim.py for GRPO rewards
```

`training_code.py` and `reward_function_factory.py` add `../../data_preprocessing` to
`sys.path`. Use a `data_preprocessing` revision compatible with `fault_sim.py` on your
machine.

## Python environment

- Python 3.11+ with CUDA
- Install: `pip install -r requirements.txt` from repo root
- Versions expected by `training_code.py`: `transformers==4.57.3`, `trl==0.26.1`,
  `peft==0.13.2`, `bitsandbytes==0.49.0`

## Model and dataset

| Phase | Model | Dataset |
|-------|--------|---------|
| SFT | `Qwen/Qwen2.5-32B-Instruct` | `chrivasileiou/asap7-language-of-test-v2` |
| GRPO | same base (via SFT adapter) | same |

## Session 1 — SFT

Supervised fine-tuning with **assistant-only loss** (Qwen chat template patched for
`{% generation %}` masks). Streaming dataset pipeline in `dataset_utils.py`.

Typical settings (32B on 4× H100):

| Parameter | Value |
|-----------|--------|
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 64 |
| `max_steps` | 100 |
| `max_prompt_length` | 4096 |
| `assistant_only_loss` | True |
| `use_dual_adapter` | False |

```bash
cd tests
METHOD=sft \
MODEL=Qwen/Qwen2.5-32B-Instruct \
TRAIN_DATASET=chrivasileiou/asap7-language-of-test-v2 \
OUTPUT_DIR=<your_sft_output_dir> \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=64 \
MAX_STEPS=100 \
MAX_PROMPT_LENGTH=4096 \
ASSISTANT_ONLY_LOSS=True \
sbatch run_training_code.sh
```

Or direct Python:

```bash
cd tests
python training_code.py \
  --method sft \
  --model_name Qwen/Qwen2.5-32B-Instruct \
  --dataset chrivasileiou/asap7-language-of-test-v2 \
  --output_dir <your_sft_output_dir> \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --max_steps 100 \
  --max_prompt_length 4096 \
  --assistant_only_loss
```

## Session 2 — GRPO

Group Relative Policy Optimization with **dual-adapter** mode (`DualAdapterGRPOTrainer`):
frozen SFT reference adapter + trainable policy adapter. Reward via
`reward_function_factory.py` → `atpgllm.llm.reward_funcs` → `data_preprocessing.fault_sim`
(`FAULT_SIM_BACKEND=fast` recommended for long runs).

Start GRPO only after SFT has finished. Pass the SFT adapter directory to `--resume_from`
(or `RESUME_FROM` for the shell script); do not merge the SFT LoRA into base weights.

Typical settings (32B, vLLM server on one GPU, train on the rest):

| Parameter | Value |
|-----------|--------|
| `use_dual_adapter` | True |
| `use_vllm` | True |
| `vllm_mode` | server (`trl vllm-serve`, not plain `vllm serve`) |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 128 |
| `max_steps` | 80–100 |
| `buffer_size` | 40000 |
| `num_generations` | 16 |
| `steps_per_generation` | 16 |
| `max_model_len` | 16384 |
| `max_prompt_length` | 4096 |
| `max_completion_length` | 6144–12288 |

```bash
cd tests
METHOD=grpo \
MODEL=Qwen/Qwen2.5-32B-Instruct \
TRAIN_DATASET=chrivasileiou/asap7-language-of-test-v2 \
OUTPUT_DIR=<your_grpo_output_dir> \
RESUME_FROM=<your_sft_output_dir> \
USE_DUAL_ADAPTER=True \
USE_VLLM=True \
VLLM_MODE=server \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=128 \
MAX_STEPS=80 \
BUFFER_SIZE=40000 \
NUM_GENERATIONS=16 \
STEPS_PER_GENERATION=16 \
MAX_MODEL_LEN=16384 \
MAX_PROMPT_LENGTH=4096 \
MAX_COMPLETION_LENGTH=6144 \
sbatch run_training_code.sh
```

## Core modules (this branch)

| Role | File |
|------|------|
| CLI / orchestration | `training_code.py` |
| Conversation format | `conversation.py`, `template_rendering.py` |
| Dataset streaming | `dataset_utils.py` |
| Model / LoRA / Qwen template | `model_utils.py` |
| GRPO trainer | `dual_adapter_grpo_trainer.py` |
| Rewards | `reward_function_factory.py`, `atpgllm/llm/reward_funcs.py` |
| Tool schema | `tools.py` |
| Reports | `docs/reports/CALL_TREE.md`, `docs/reports/REPORT_DDP_VLLM_GRPO_TRAINING.md` |

## What stays out of git

Training outputs (`wandb/`, `jobs/`, `logs/`, experiment output directories) are
gitignored. Reproduce weights locally or publish models separately if needed.
