# Training on dataset v5

Dataset v5 (`chrivasileiou/asap7-language-of-test-v5`, local copy `data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.v5`) holds the v4 combinational rows and the full-scan sequential rows. Simulating sequential rows needs data_preprocessing at branch `seq-v5` or later.

## Schedule

SFT warm-up, then RFT, GDPO, RFT and GDPO, run by `scripts/rft/run_schedule.py` from `scripts/rft/configs/schedule_v5_juno_h100.json`.

1. **Warm-up SFT.** 100 steps of 1,024 conversations from the base model (`sft_granite_4.2_8b_v5_warmup_juno_h100.conf`).
2. **Phase 1.** `run_rft.py` with `rft_v5_phase1_juno_h100.json`. RFT rounds run until the validation score has not risen above the best round for 3 rounds. GDPO then starts from the best round and runs 40 steps (`grpo_granite_4.2_8b_v5_juno_h100_tetramax_after_rft.conf`), benchmarked every 2 steps.
3. **Phase 2.** The same with `rft_v5_phase2_juno_h100.json`, starting from the best benchmarked GDPO checkpoint of phase 1 (`runs/v5_schedule/phase_1/gdpo_best.json`).

Every stage writes a marker when it finishes, so a rerun continues where the last one stopped. `plan.json` in each phase directory and `schedule.json` in the run directory are read before each stage and can be edited while the run goes on.

## Both circuit kinds in every step

`atpgllm/training/kinds.py` merges the kinds by even spacing. Every window of consecutive rows holds each kind in its split proportion, give or take one row, and each kind keeps its own order.

- **SFT.** `ShardPermutation` applies it to every pass when a split holds more than one `circuit_kind`. The v5 SFT configs set `SFT_CIRCUIT_VALIDATION=True`, which streams through it. The plain streaming path would read every combinational shard first.
- **GRPO.** The rolling buffer applies it to every chunk after the netlist interleave. GRPO on a mixed dataset refuses to run without `BUFFER_ROLLING=True`.
- **RFT.** Problems are selected per kind and level (`kind_shares`, half each). Validation and the benchmark are split the same way, and rounds and GDPO checkpoints are ranked by the mean over kind and level.
- **Measurement.** `scripts/dataset/check_kind_mix.py` measures the kinds per optimizer step on the real SFT stream and the real GRPO buffer.

## Prompt window

v5 sequential prompts carry the scan netlist and the chain table (median 5,740 tokens; 27.2% exceed 8,192). Every v5 config sets `MAX_PROMPT_LENGTH=14336`, and `training_code.py` prints the window at start. On a v5 dataset it raises a smaller window to 14,336 and says so. 6.8% of sequential and 2.9% of combinational problems still exceed 14,336 and are skipped.

The context grows to match: 24,576 tokens for SFT, 32,768 for RFT sampling, and 32,768 for GDPO (prompt 14,336 plus completion 18,432).

## Launch on juno

The tree is `/work/cxv200006/atpg_v5`: `libatpgllm` at branch `seq-v5`, with `data` and `data_preprocessing` linked to the v5 data and to data_preprocessing `seq-v5`. The benchmark manifest is built once, before the first job:

```
python scripts/rft/build_eval_manifest.py --dataset ../data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.v5 \
  --kind_shares combinational=0.5,sequential=0.5 --per_level 50 --max_prompt_length 14336 \
  --max_completion_length 18432 --max_model_len 32768 --per_device_batch_size 1 --world_size 2 \
  --backend tetramax --out runs/v5_schedule/eval/v5_bench_manifest.json
```

Then submit a chain of 2-day jobs on the h100 node (4x H100 80GB). Each job resumes the one before it:

```
cd /work/cxv200006/atpg_v5/libatpgllm
scripts/rft/submit_juno_schedule.sh 8
```

To monitor the run, read these files:

- `runs/v5_schedule/state.json`
- `runs/v5_schedule/phase_<n>/state.json`
- `runs/v5_schedule/logs/`
- `jobs/v5_schedule_<id>.out`
