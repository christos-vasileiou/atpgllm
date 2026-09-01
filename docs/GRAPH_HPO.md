# Staged Graph-Pipeline HPO

This workflow uses Optuna grouped multivariate TPE, constant-liar distributed
sampling, Hyperband pruning, semantic validation, and explicit checkpoint
promotion. Ordinary trials run exactly one stage. A Stage-B trial consumes a
promoted Stage-A checkpoint; it does not retrain Stage A. SFT and GRPO are
screened only after promotion.

Run all commands from `libatpgllm/` after activating the project environment.
The production configuration is
`scripts/train/configs/hpo_graph_pipeline.yaml`.

## 1. Install search and PostgreSQL support

```bash
activate && pip install -e '.[graph-search]'
export OPTUNA_STORAGE='postgresql+psycopg://USER:PASSWORD@HOST:5432/atpgllm_hpo'
```

W&B may track trials, but Optuna is the optimizer and PostgreSQL is the study
authority. SQLite is supported only for a single local worker.

## 2. Freeze a design-disjoint split

The split is keyed by SHA-256 of the Verilog netlist, so faults from one design
cannot cross train/validation/test boundaries.

```bash
activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  prepare-split \
  --config scripts/train/configs/hpo_graph_pipeline.yaml \
  --output runs/hpo/asap7_v2_design_split.json

activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  validate-split --manifest runs/hpo/asap7_v2_design_split.json
```

The default manifest scans 100,000 streamed records. Record the Hub dataset
revision in the YAML before production use.

## 3. Stage A: graph-pretraining search

Create the durable study:

```bash
activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  create-study --stage graph_pretrain \
  --config scripts/train/configs/hpo_graph_pipeline.yaml \
  --storage "$OPTUNA_STORAGE"
```

Launch one GPU trial per Slurm array worker:

```bash
export HPO_ARRAY=0-23
export HPO_TRIALS_PER_WORKER=3   # 24 workers × 3 = 72 requested trials
bash scripts/train/submit_graph_hpo.sh \
  scripts/train/configs/hpo_graph_pipeline.yaml \
  graph_pretrain \
  runs/hpo/asap7_v2_design_split.json
```

Workers report propagation, discrepancy, and backtrack macro AUPRC/IoU at
steps 500, 1,500, 4,500, and 13,500. Hyperband may prune at any real
validation rung.

Inspect and promote the initial top eight:

```bash
activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  inspect --stage graph_pretrain \
  --config scripts/train/configs/hpo_graph_pipeline.yaml \
  --storage "$OPTUNA_STORAGE"

activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  promote --stage graph_pretrain --top-k 8 \
  --config scripts/train/configs/hpo_graph_pipeline.yaml \
  --storage "$OPTUNA_STORAGE" \
  --output runs/hpo/stage_a_top8.json
```

Confirm those fixed configurations with three seeds:

```bash
activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  enqueue-confirmations --stage graph_pretrain \
  --promotion runs/hpo/stage_a_top8.json --seeds 17,42,73 \
  --config scripts/train/configs/hpo_graph_pipeline.yaml \
  --storage "$OPTUNA_STORAGE"

# Resubmit workers; enqueued fixed trials are claimed before new suggestions.
export HPO_ARRAY=0-7 HPO_TRIALS_PER_WORKER=2
bash scripts/train/submit_graph_hpo.sh \
  scripts/train/configs/hpo_graph_pipeline.yaml \
  graph_pretrain runs/hpo/asap7_v2_design_split.json

activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  promote --stage graph_pretrain --top-k 3 --min-seeds 3 \
  --config scripts/train/configs/hpo_graph_pipeline.yaml \
  --storage "$OPTUNA_STORAGE" \
  --output runs/hpo/stage_a_finalists.json
```

Seed-confirmed configurations are ranked by median semantic score. The
representative checkpoint is the seed nearest that median.

## 4. Stage B: Q-Former/alignment search

Stage B samples only Q-Former, projection, optimization, graph-freeze policy,
and normalized GTC/GTM/GTG ratios. GIN architecture is loaded from and locked
to the selected Stage-A parent.

```bash
activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  create-study --stage graph_text_alignment \
  --config scripts/train/configs/hpo_graph_pipeline.yaml \
  --storage "$OPTUNA_STORAGE"

export HPO_ARRAY=0-15 HPO_TRIALS_PER_WORKER=3
bash scripts/train/submit_graph_hpo.sh \
  scripts/train/configs/hpo_graph_pipeline.yaml \
  graph_text_alignment \
  runs/hpo/asap7_v2_design_split.json \
  runs/hpo/stage_a_finalists.json
```

Stage-B pruning uses bidirectional graph/text Recall@1 and MRR, matching average
precision, and held-out GTG loss—not sampled weighted training loss. Promote
the top six, enqueue seeds `17,42,73`, run confirmation workers with the same
Stage-A promotion manifest, then promote `--top-k 4 --min-seeds 3`.

```bash
activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  promote --stage graph_text_alignment --top-k 6 \
  --config scripts/train/configs/hpo_graph_pipeline.yaml \
  --storage "$OPTUNA_STORAGE" \
  --output runs/hpo/stage_b_top6.json

activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  enqueue-confirmations --stage graph_text_alignment \
  --promotion runs/hpo/stage_b_top6.json --seeds 17,42,73 \
  --config scripts/train/configs/hpo_graph_pipeline.yaml \
  --storage "$OPTUNA_STORAGE"

export HPO_ARRAY=0-5 HPO_TRIALS_PER_WORKER=2
bash scripts/train/submit_graph_hpo.sh \
  scripts/train/configs/hpo_graph_pipeline.yaml \
  graph_text_alignment \
  runs/hpo/asap7_v2_design_split.json \
  runs/hpo/stage_a_finalists.json

activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  promote --stage graph_text_alignment --top-k 4 --min-seeds 3 \
  --config scripts/train/configs/hpo_graph_pipeline.yaml \
  --storage "$OPTUNA_STORAGE" \
  --output runs/hpo/stage_b_finalists.json
```

## 5. Stage C: multimodal SFT screening

The current repository lacks a graph-aware held-out simulator evaluator, so
Stage C is deliberately an explicit launch plan rather than a fake Optuna
objective.

```bash
activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  plan-downstream --stage multimodal_sft \
  --parents runs/hpo/stage_b_finalists.json \
  --seeds 17,42,73 \
  --config scripts/train/configs/hpo_graph_pipeline.yaml \
  --output runs/hpo/stage_c_plan.json

bash scripts/train/submit_graph_hpo_plan.sh runs/hpo/stage_c_plan.json
```

Each entry preserves the parent checkpoint SHA-256, config hash, seed, exact
command, and output directory. Evaluate every completed SFT checkpoint on the
same held-out design/fault set and write:

```json
{
  "entries": [
    {
      "checkpoint_path": "/absolute/path/multimodal_sft_final.pt",
      "config_hash": "...",
      "seed": 42,
      "heldout_pass_at_1": 0.42,
      "format_valid_rate": 0.98
    }
  ]
}
```

Promote only held-out results:

```bash
activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  promote-evaluations \
  --evaluations runs/hpo/stage_c_evaluations.json \
  --stage multimodal_sft --top-k 2 --min-seeds 3 \
  --study-name asap7-graph-pipeline-v1 \
  --metric heldout_pass_at_1 \
  --output runs/hpo/stage_c_top2.json
```

## 6. Stage D: GRPO pilots

```bash
activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  plan-downstream --stage multimodal_grpo \
  --parents runs/hpo/stage_c_top2.json \
  --seeds 17,42,73 \
  --config scripts/train/configs/hpo_graph_pipeline.yaml \
  --output runs/hpo/stage_d_plan.json

bash scripts/train/submit_graph_hpo_plan.sh runs/hpo/stage_d_plan.json
```

Do not select from GRPO training reward. Use scheduled held-out evaluations,
the median of the final three evaluations or validation AUC, bootstrap
confidence intervals, format validity, simulator errors, diversity, and KL to
the SFT reference. Express Stage-D fidelity in verifier rollouts (the YAML
records recommended rungs of 2k/6k/18k); the current trainer stops by optimizer
steps, so the generated pilot plan uses a conservative 20-step cap.

After held-out evaluation, use `promote-evaluations` again with
`--stage multimodal_grpo --top-k 1`.

## 7. Export the best lineage

```bash
activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  export-best \
  --promotion runs/hpo/stage_d_best.json \
  --output runs/hpo/final_best_lineage.json
```

## 8. Resume interrupted workers

RDB heartbeat marks abandoned trials failed and the configured retry callback
requeues one retry. Resubmit the same Slurm array command; workers claim
remaining or enqueued trials from PostgreSQL. Never map trial numbers modulo
local GPU IDs.

## 9. Cheap local smoke study

This exercises YAML parsing, conditional sampling, Optuna storage, Hyperband
reporting/pruning, and coordinator persistence without downloading models or
datasets:

```bash
activate && python -m atpgllm.graph.scripts.search_graph_pipeline \
  mock-study --profile smoke --stage graph_pretrain --trials 4 \
  --config scripts/train/configs/hpo_graph_pipeline.yaml \
  --storage sqlite:////tmp/atpgllm_graph_hpo_smoke.db
```

An actual Stage-A/B worker requires the streamed dataset; Stage B additionally
downloads the configured text encoder. PostgreSQL, Slurm, GPU execution, and
the graph-aware SFT/GRPO held-out evaluator are not exercised by the mock.

To run one real local Stage-A smoke trial after preparing a small split:

```bash
activate && python -m atpgllm.graph.scripts.search_graph_pipeline worker \
  --profile smoke --stage graph_pretrain --trials 1 --device cpu \
  --config scripts/train/configs/hpo_graph_pipeline.yaml \
  --split-manifest runs/hpo/asap7_v2_design_split.json \
  --storage sqlite:////tmp/atpgllm_graph_hpo_real_smoke.db
```
