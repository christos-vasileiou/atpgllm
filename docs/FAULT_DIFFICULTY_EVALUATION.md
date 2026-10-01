# Fault difficulty and Language-of-Test evaluation

`evaluate_model.py` now characterizes every selected circuit/fault pair and reports
outcomes by circuit, stuck-at polarity, exact fault location, input/internal/output
location, site depth, circuit depth, gate count, PI count, PO count, output distance,
controllability, observability, combined SCOAP cost, and empirical difficulty.
The GRPO launcher forwards the controls below. Existing training code is unaffected.

## Which metrics answer which question?

| Metric | Meaning | Difficulty direction |
|---|---|---|
| `cc0`, `cc1` | Combinational SCOAP costs to force the site to 0 or 1 | Larger is harder |
| `activation_cost` | CC1 for sa0; CC0 for sa1 | Larger is harder |
| `co` | SCOAP cost to propagate a change to any observed output | Larger is harder |
| `scoap_cost` | Activation cost + CO | Larger is harder |
| `site_depth` | Longest gate path from a PI/constant to the site | Structural depth only |
| `distance_to_output` | Shortest structural gate path to a PO | Structural distance only |
| `circuit_depth` | Maximum depth at primary outputs | Structural depth only |
| `fanin_support`, `fanout` | PI support size and downstream cell-output dependency count | Context, not a proof of difficulty |
| `activation_probability` | Uniform PI probability of driving the site opposite the stuck value | Smaller is harder |
| `conditional_observability` | Probability of detection **given activation** | Smaller is harder |
| `random_detection_probability` | Probability one uniform PI vector detects the requested fault | Smaller is harder |
| `difficulty_bits` | `-log2(random_detection_probability)` | Larger is harder |
| `expected_random_vectors` | `1 / random_detection_probability` | Larger is harder |

The probabilities obey `P(detect) = P(activate) * P(detect | activate)`.
No independence between activation and propagation is assumed. A PI fault is easy
to activate but may still be hard to observe. A one-gate AND output can be hard to
activate for sa0, and easy for sa1. Reconvergence may cancel an activated fault.

SCOAP uses PI cost 1, PO observability 0, and one cost unit per cell. Direct wire
aliases add no depth/cost. Constants have cost 0 for their fixed value and infinite
cost for its opposite. Arbitrary supported cell functions use minimum forcing
cubes (including don't-cares); sensitization uses the Boolean derivative. Treat
these as **cell-level generalized SCOAP** scores, dependent on library mapping.
They are heuristics across reconvergent paths, not minimum independent PI counts.
The theoretical basis and limitations are described in the
[University of New Mexico testability notes](https://ece-research.unm.edu/jimp/vlsi_test/slides/html/testability_measures.html),
including Goldstein's fault-specific CC+CO construction.

## Interface size

`num_inputs` and `num_outputs` count scalar primary-port bits; a 32-bit bus counts
as 32. Output ports connected to the same signal still count as separate ports;
`num_output_signals` separately records the number of distinct aliased signals.
Reports group detection and random baselines by each port count,
by count crossed with difficulty, and by joint input/output count bins. Bin edges
are 0, 1, 2, 4, 8, 16, 32, 64 and 128, followed by >128; unknown counts stay separate.
The historical report includes input/output detection curves and stacked easy/hard
composition plots (`interface_difficulty.png` and `.pdf`), with sample counts.
Difficulty labels continue to use random detectability, not interface size itself.

## Probability measurement and limits

Up to 12 primary inputs, all vectors are enumerated. Larger designs use 4,096
seeded, independent uniform vectors by default. Each probability includes a 95%
Wilson interval for Monte Carlo estimates; exhaustive values have zero sampling
uncertainty. Circuit-derived seeds make results independent of evaluation order
and share vectors across a circuit's faults. The difficulty seed is separate from
the model generation seed. Increase trials to resolve rare faults.

An exhaustive zero is `proven_undetectable_exhaustive` **under the implemented
Boolean circuit and stated constraints**. A sampled zero is `unresolved_zero_hits`,
never proof of redundancy. Infinite structural costs are null plus a separate
`structural_unreachable` flag; unsupported syntax has an explicit error status.
Unsupported and failed examples stay in model denominators. Neither zero category
is silently ranked as an ordinary finite hard fault.

The analyzer supports combinational, unconstrained binary PIs, all POs observed,
and a single declared-net stem stuck-at fault. Aliased wires represent the same
physical stem; faults on an assigned PO can therefore differ from the historical
Python simulator's directional-assignment behavior. Branch faults, sequential
state, black boxes, multiple drivers, cycles, and unsupported syntax are rejected.
The packed execution loop is independent of `fast_fault_sim`, but shares the
project's cell-function definitions and canonical port parser. It is not an
independent validation of that library or a native TetraMAX execution. Keep the
model scoring backend and characterization backend in the report provenance.

## Sampling and frozen membership

Previously the evaluator took a prefix with `shuffle=False, unique_by="netlist"`:
the first eligible fault per netlist, up to 512. It did not randomly select faults.
It also passed `-1` to a helper that rejects negative buffer sizes.

The new default `uniform_faults` scans the **complete test stream**, deduplicates
exact `(netlist text, fault)` pairs, filters the actual templated prompt against
`MAX_PROMPT_LENGTH`, and uses seeded reservoir sampling. Memory for candidate
records is bounded; the identity set scales with the number of unique source
problems. Multiple faults per circuit are permitted. `MAX_EVAL_SAMPLES=-1` means
all eligible pairs and consequently materializes all records.

`stratified` first takes a uniform reservoir of `EVAL_CANDIDATE_POOL` pairs, then
balances `(fault polarity, location class, empirical difficulty)` strata. The pool
counts and selected distribution are saved. Rare strata absent from the reservoir
cannot be recovered; enlarge the pool. These are **challenge-set scores**, not
estimates of the original dataset's average. Macro circuit averages reduce the
influence of circuits with many selected faults, but do not establish independent
design families. The fixed difficulty bands are p>=1/4, [1/16,1/4),
[1/256,1/16), (0,1/256), exhaustive-zero, unresolved-zero, and unknown.
Monte Carlo band membership is estimated; inspect the per-problem intervals.

`legacy_prefix` retains the prefix/one-netlist selection policy for comparison,
but uses the corrected full-prompt length filter. To reproduce an exact historical
workload, use a saved manifest or saved results, rather than expecting this option
to undo tokenizer/filter changes.

`EVAL_MANIFEST` creates a checksummed, exclusive-write problem manifest when absent
and reuses its exact membership when present. Existing membership overrides sample
count/selection flags. A tokenizer/length incompatibility fails rather than silently
changing the set. The launcher defaults to `$EVAL_RESULTS_DIR/eval_manifest.json`.
Direct Python evaluation also creates/reuses `<output stem>.manifest.json` when
no explicit manifest path is supplied. Use a new path to change the workload.
For different simulator backends, retain
separate manifests/result directories and verify equal example checksums for
matched membership. Save the dataset revision externally for whole-population
provenance; the manifest freezes exact examples, not the upstream dataset.

```bash
# From libatpgllm; checkpoint may be an individual GRPO checkpoint or experiment.
EVAL_SELECTION=stratified EVAL_CANDIDATE_POOL=8192 MAX_EVAL_SAMPLES=512 \
EVAL_MANIFEST=analysis/difficulty_eval/fast_stratified.json \
DIFFICULTY_RANDOM_SAMPLES=16384 \
EVAL_RESULTS_DIR=analysis/difficulty_eval/grpo_fast \
bash scripts/eval/eval_grpo_policy_checkpoints.sh runs/YOUR_RUN/checkpoint-N
```

For SFT, invoke `evaluate_model.py --adapter PATH --eval_manifest SAME_PROBLEMS ...`
or set `EVAL_MANIFEST` when using the SFT launcher; Python accepts the same
environment settings. Freeze a representative uniform set and a balanced challenge
set separately. Reuse each within a matched checkpoint/baseline comparison.

## Reports and interpretation

The main evaluation JSON includes `difficulty_analysis`, `difficulty_provenance`,
`selection_audit`, raw component rewards, and per-problem identities/characteristics.
Companion `*.difficulty.json`, `*.difficulty.problems.csv`, and
`*.difficulty.problems.jsonl` contain grouped results and an easy-to-hard ranking.
CSV numeric columns permit any alternate sort. Exact locations are keyed by
circuit hash plus net name, preventing unrelated `n1` nets from being merged.

The only outcome in the difficulty analysis is **D (detection)**: applying the
input vector makes the faulty circuit differ from the good circuit at **at least
one primary output**. Activation at an internal fault site alone is insufficient.
Correct model-predicted expected outputs are not required. Input/output counts
refer to scalar primary-port bits, including individual bits of buses.

Failed/missing slots count in the requested number of completions. Detection
reporting is independent of `threshold_mode`; use `fault_detected` (the default)
for the main evaluator's pass@k too. Online D uses the recorded detection reward;
the CPU analysis replays the input vector and compares simulated primary outputs.
Historical source records remain available with their original fields.

Reports include pass@k, problem-weighted and circuit-macro means, circuit bootstrap
intervals (1,000 replicates), and paired lifts over random vectors. These intervals
are exploratory: design-family and training-seed independence is unestablished,
and lift intervals condition on the measured random probabilities. A single-circuit
stratum has no bootstrap interval. Counts always accompany strata; sparse groups
should not support strong conclusions.

`uniform_random_k_vectors = 1-(1-p)^k` describes **k uniform input vectors**.
A search-policy completion may consume several model generations and simulator
calls. Its pass@k is not a cost-matched comparison to k vectors. Aggregate attempts,
simulator requests/executions and generated tokens are retained by stratum. Compare
searches against a random baseline with the same predeclared vector budget; do not
turn outcome-dependent early-stopping costs into an unbiased counterfactual.

## CPU-only historical analysis

```bash
python scripts/eval/analyze_fault_difficulty.py \
  --input analysis/language_of_test_20260925/sft90_reference/source_manifest.json \
  --slots analysis/language_of_test_20260925/sft90_reference/slots.jsonl \
  --output analysis/difficulty_eval/sft_reference.json

python scripts/eval/analyze_fault_difficulty.py --evaluation-results \
  --input runs/EVAL_FOLDER/checkpoint-N_passatk_single_completion.json \
  --output analysis/difficulty_eval/checkpoint-N.json --k 1 4 16
```

For model-free saved results without prompts, `--problem-source` accepts an
identity-matched evaluator result containing prompts, or a new checksummed example
manifest. Joins never use row positions or module names. Strict saved replay slots
must cover all declared completion indices; run `execute_language_of_test.py replay`
first to materialize missing failures. JSON/JSONL example inputs without slots can
also characterize difficulty without loading an LLM.

The reproducible historical audit and figures are under
`analysis/fault_difficulty_20260928/`. These are retrospective CPU measurements,
not new model inference. To judge whether a model learned the language of test,
require fault-conditioned detection advantage in difficult strata on locked,
circuit/family-disjoint data, with equal inference/simulation budgets. A high
aggregate detection rate dominated by easy faults does not establish that claim.

The regenerated figures include `interface_detectability.png` / `.pdf`: separate
model-D and random-D heatmaps for input-count-by-difficulty and
output-count-by-difficulty, with denominators in every populated cell. Grey cells
mean no examples, not zero detection. `interface_difficulty` shows the distribution
of easy/hard faults, and `difficulty_curves` includes D versus input/output counts.
