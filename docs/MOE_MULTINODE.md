# Mixture-of-experts training across juno H200 nodes

Branch `moe-multinode` (from `develop`). The multimodal (graph + language) side lives on `graph-moe`, which is `graph-modality` with this branch merged in. Both pair with data_preprocessing `develop`.

## 1. Which MoE models, and why

Checked on 2026-10-14 against each model card and `config.json`.

| Model | Experts (active) | Params total / active | Context | Tool calls | Stack needed | Role here |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | 128 (top 8) | 30.5B / 3.3B | 262k | Hermes JSON, same as Qwen2.5 | transformers 4.57, vLLM 0.11 (current env) | **First real MoE run** |
| `allenai/OLMoE-1B-7B-0125-Instruct` | 64 (top 8) | 7B / 1.3B | **4,096** | none in template | current env | Plumbing smoke test only |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 128 + 1 shared (top 6) | 30B / 3.5B | 262k (1M) | qwen3_coder XML | transformers ≥ 5.3, vLLM ≥ 0.12, mamba-ssm | Second, after a separate env |
| `nvidia/Cosmos3-Nano` | not an MoE | 16B | 256k (reasoner) | not documented | diffusers main, vLLM 0.21 + plugin | Not a training target (see §6) |

OLMoE cannot hold a v4 conversation. Its context is 4,096 tokens while v4 prompts reach 8,192, and its chat template drops tool calls. It still exercises every MoE code path cheaply, so it is the first thing to run on a new node or after a launcher change.

Qwen3-30B-A3B is the first model that can learn the v4 protocol. It runs on the libraries we already have, its template is ChatML with Hermes tool calls (the Qwen2.5 format our parsers and the assistant mask already handle), and its active size of 3.3B is below Granite 4.2 8B.

## 2. What the branch adds

| Piece | Where | What it does |
|---|---|---|
| `MOE=True` (`--moe`) | `training_code.py` | Resolves LoRA targets from the loaded model's own `nn.Linear` names (attention, expert MLP under either `gate/up/down_proj` or Mixtral's `w1/w2/w3`, and Mamba `in_proj/out_proj`). Sets `ddp_find_unused_parameters=True` (dense runs keep `False`), because an expert that receives no token in a micro batch has no gradient and plain DDP aborts on that. Adds expert-load logging. |
| `TUNE_MOE_ROUTER=1` | same | Adds the router (`gate`) to the LoRA targets and turns on `output_router_logits`, so the model's load balancing loss joins the SFT loss. GRPO computes its loss from logits and ignores it. |
| `BASE_PRECISION=bf16` | `model_utils.load_quantised_model` | Loads the base in bfloat16 instead of NF4. A 30B MoE is about 61GB in bf16 and fits one H200 (141GB) with room for activations, so each DDP rank holds a replica and nothing is sharded. The LoRA preparation skips `prepare_model_for_kbit_training`, which would otherwise upcast the whole base to fp32. vLLM weight sync already handles a base without `quant_state`. |
| Fused expert warning | `moe_utils.resolve_moe_lora_targets` | GraniteMoE (and many models under transformers v5) store a layer's experts as one 3D parameter. LoRA `target_modules` cannot reach those. The run prints how many such parameters exist instead of silently training attention only. |
| `ExpertLoadCallback` | `moe_utils.py` | Counts the top-k choices of every `nn.Linear` router during training forwards and logs `moe/load_entropy_mean`, `moe/load_entropy_min` (1.0 is a uniform load), `moe/max_expert_share` and `moe/dead_expert_fraction`. A falling minimum entropy is the early sign of expert collapse. |
| Qwen3 2507 assistant mask | `patch_qwen_chat_template_for_assistant_mask` | The 2507 template has one assistant branch, which the old patcher did not recognise and refused. The new branch masks the same tokens as Qwen2.5 (checked on the real tokenizer with a tool call turn). |
| `PYTHON_VENV` | `run_training_code.sh`, `_mn_launch.sh` | Activates another environment on the batch node and on every srun node. |
| `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_DEBUG`, `GLOO_SOCKET_IFNAME`, `HF_HOME`, `HF_HUB_OFFLINE` | `run_training_code.sh` | Values set in a config are exported, so they reach the other nodes. On juno: `ib0` and `mlx5_0`, and a model cache on `/work` filled before the job. |
| `RESUME_ON_REQUEUE` (default True) | `run_training_code.sh` | The h200 partition preempts with `REQUEUE`. A requeued job (`SLURM_RESTART_COUNT > 0`) now resumes from its own latest checkpoint with full trainer state, instead of restarting from the configured start point and overwriting `OUTPUT_DIR`. |
| Configs | `scripts/train/configs/` | `sft_olmoe_1b_7b_h200_juno_smoke.conf`, `sft_qwen3_30b_a3b_v4_h200_2node_juno.conf`, `grpo_qwen3_30b_a3b_v4_h200_3node_juno_tetramax.conf`, `sft_nemotron3_nano_v4_h200_2node_juno.conf` (blocked, see §4). |

Tests: `tests/unit/test_moe_utils.py` (tiny random OLMoE, Mixtral, GraniteMoE and Llama models on CPU) and a Qwen3 2507 case in `tests/unit/test_chat_templates.py`. The full unit suite passes on CPU.

## 3. Juno layout

Measured on 2026-10-14 from `juno-vscode` (`sinfo`, `scontrol show partition h200`).

* `h200` partition: 26 nodes (`g-07-[01-13]`, `g-08-[01-13]`), each 2× H200 NVL (141GB), 64 cores, 384GB RAM. MaxTime 2 days, `PreemptMode=REQUEUE`. InfiniBand (`mlx5_0`, `ib0`).
* `juno-vscode` is a submit host. Jobs run on the g-07 and g-08 nodes, which see `/work` (MooseFS), where staging, `.runtime/tetramax` and the HF cache should live.
* Modules: `tetramax/vO-2018.06-SP1`, `cuda/12.4 … 13.3`, `nccl/2.29.2`, `nccl-tests/cuda12.4`, `synopsys/W-2024.09`.
* Existing envs: `/work/cxv200006/myenv` (same versions as the local one) and `myenv2` (torch 2.10, vLLM 0.17, PEFT 0.18.1, transformers 4.57.6).

The existing `*_h200_2node.conf` and `*_h200_3node.conf` files ask for `TIME_LIMIT=7-00:00:00`, which juno's h200 partition rejects. The new configs ask for 2 days.

SFT uses every allocated GPU for DDP. GRPO in server mode gives the last node to `trl vllm-serve` and trains on the rest. With 2 GPUs per node, 3 nodes give 4 training ranks and 2 vLLM engines.

## 4. Nemotron 3 Nano environment

Nemotron needs a separate env (`/work/cxv200006/myenv-moe` in its config). Do not upgrade the shared `myenv`: Granite runs, the dual adapter trainer and every analysis notebook use it. Requirements from the model card and its remote code:

* transformers ≥ 5.3 (native `NemotronHForCausalLM`), or the remote code with `trust_remote_code=True`.
* `mamba-ssm` and `causal-conv1d` built against the env's torch. Without them the Mamba layers fall back to a slow torch path.
* vLLM ≥ 0.12 for generation, with `--reasoning-parser nano_v3` and `--tool-call-parser qwen3_coder` if served outside TRL.
* PEFT ≥ 0.17 if experts load fused under transformers v5 (`target_parameters`).
* TRL that supports transformers v5. `myenv2`'s TRL 0.24 is older than our 0.26.1.

Check before any run, inside the new env, with `PYTHONPATH=<libatpgllm_moe-multinode>`:

```bash
python -c "import torch, transformers, trl, peft, vllm, mamba_ssm, causal_conv1d; print(transformers.__version__, trl.__version__, peft.__version__, vllm.__version__)"
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= python -m pytest tests/unit -q   # the same suite must pass under v5
```

Facts from the remote `modeling_nemotron_h.py` that shape the work:

* Experts are `nn.Linear` `up_proj` and `down_proj` (squared ReLU, no gate), so LoRA reaches them under the remote code.
* The MoE forward loops in Python over all 128 experts and runs a dummy pass through idle ones. DDP therefore never sees unused expert parameters, but the loop is slow. Expect much lower throughput than Qwen3 until the native v5 kernels or Megatron Bridge are used.
* The router (`NemotronHTopkRouter`) is a module with an fp32 weight and an `e_score_correction_bias` buffer (balancing without an auxiliary loss). It is not `nn.Linear`, so expert-load logging turns itself off and `TUNE_MOE_ROUTER` does not apply.
* The chat template opens a reasoning block by default and uses the qwen3_coder XML tool format. Our parsers already read that XML format for Granite 4.2. The assistant mask patcher does not know this template yet.

NVIDIA's own fine-tuning path is Megatron Bridge or NeMo RL in the `nemo:25.11.nemotron_3_nano` container (LoRA with expert parallelism 2 on 2× 80GB). That is a different training stack. Use it only if the HF path is too slow, and keep our reward and TetraMAX code as the environment.

## 5. Validation ladder

Run each rung on juno before the next. All of them are short jobs.

| Rung | Command / config | Pass when |
|---|---|---|
| Setup | On the login node: `HF_HOME=/work/cxv200006/hf_cache hf download <model>` for every model a config names | Weights are on `/work`. The configs then set `HF_HUB_OFFLINE=1`, so no rank or vLLM node downloads during a job |
| 0. Fabric | 2 node `all_reduce_perf` from `nccl-tests` with `NCCL_SOCKET_IFNAME=ib0 NCCL_IB_HCA=mlx5_0` | Bus bandwidth is IB class (tens of GB/s), not TCP over Ethernet |
| 1. OLMoE, 1 node | `sft_olmoe_1b_7b_h200_juno_smoke.conf` | Loss falls over 20 steps, `moe/*` metrics logged, no unused parameter error, checkpoint saved |
| 2. OLMoE, 2 nodes | same with `NUM_NODES=2` | Same, with 4 ranks. `scancel --signal=TERM` then requeue resumes from the checkpoint |
| 3. Qwen3 MoE SFT, 2 nodes | `sft_qwen3_30b_a3b_v4_h200_2node_juno.conf` (`MAX_STEPS=2`) | Peak memory below 120GB per GPU, tokens/s recorded, LoRA targets printed include expert projections |
| 4. Qwen3 MoE GRPO, 3 nodes | `grpo_qwen3_30b_a3b_v4_h200_3node_juno_tetramax.conf` (`MAX_STEPS=2`) | TetraMAX preflight passes on the compute node, vLLM health from both training nodes, weight sync time per step recorded |
| 5. Nemotron | after §4 | Rungs 1 and 3 again under the new env |

Only after rung 3 gives a throughput number should `MAX_STEPS` be set for a real run (one step is 1,024 conversations, and each job must finish or checkpoint inside 2 days).

## 6. Risks and open questions

* **HF MoE speed.** transformers 4.57 runs OLMoE and Qwen3 MoE experts in a Python loop over the experts that received tokens. With 128 experts this can dominate the step. Measure at rung 3 before anything else.
* **Weight sync volume.** GRPO pushes every LoRA-touched matrix to vLLM with one HTTP call each. With experts targeted, Qwen3 has 18,624 such matrices. If sync dominates, train attention-only LoRA for the GRPO stage and compare.
* **DDP vs ZeRO-2.** The MoE configs use plain DDP, since a bf16 replica fits and `--moe` sets `find_unused_parameters` on that path. `ds_zero2.json` is untested with MoE here.
* **TetraMAX on juno.** The module exists and the binary is on `PATH` on the login node. The preflight in rung 4 is the first check on a compute node. Seats are shared through `.runtime/tetramax` on `/work`. With `TMAX_MAX_CONCURRENT=16`, simulation throughput does not grow with more training nodes.
* **Cosmos3-Nano.** Its card describes a Mixture-of-Transformers, which is a vision language reasoner tower plus a diffusion generator tower for video, audio and actions. It is not a sparse MoE. Serving needs vLLM 0.21 with the `vllm-cosmos3` plugin and diffusers main, and no fine-tuning recipe is published. It is not a fit for netlist test generation. The analog plan (`docs/proposals/ANALOG_TEST_PLAN.md` on `research-roadmap`) considers its reasoner tower for reading waveform images.
