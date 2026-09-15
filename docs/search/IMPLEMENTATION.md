# Running the implemented search

Implementation status: 15 September 2026. The pass@k evaluator now uses
`conversation-search-v1`. The original [design](DESIGN.md),
[review](IMPLEMENTATION_REVIEW.md), and [literature-backed plan](IMPLEMENTATION_PLAN.md)
remain the rationale and experimental roadmap. Performance comparisons on trained
checkpoints still need to be run; implementation does not establish a quality gain.

## What is implemented

| Component | Implementation |
| --- | --- |
| Full assistant/tool histories, pending-call execution, bounded protocol repair, final-answer boundaries | [`completion_runner.py`](../../atpgllm/training/completion_runner.py) |
| Per-slot seeds, token/simulator caps, configuration validation, shared acceptance predicate | [`search_types.py`](../../atpgllm/training/search_types.py) |
| Explicit final-answer scoring, authoritative fault/netlist binding, PI/PO validation, matching observations and simulator cache | [`search_verifier.py`](../../atpgllm/training/search_verifier.py) |
| MCTS with complete-action nodes, progressive widening, PUCT, uniform/LM priors, bounded duplicate proposals | [`search_policies.py`](../../atpgllm/training/search_policies.py) |
| Evolution with exact seed allocation, feedback continuation, PI mutation/crossover, diversity, fresh seeds, answer repair | Same policy module |
| Independent random and vector-only genetic baselines | [`sampling_strategies.py`](../../atpgllm/training/sampling_strategies.py) |
| Backend finish reasons, request seeds, token counts, stopping at tool calls | [`search_backends.py`](../../atpgllm/training/search_backends.py) |

`greedy`, `best_of_n`, `mcts`, and `evolutionary` all use the common runner in
pass@k evaluation. Existing SFT stopping-criteria helpers and training tool loops
keep their separate paths. Training reward parsing is unchanged; the new
evaluation verifier supplies the final answer and its matching observation to
the existing reward calculation.

Each returned completion has an independent search context and cache. vLLM
batches work across these contexts with per-request sampling settings. HF runs
requests individually inside a saved/restored RNG scope because its generation
API uses a shared RNG; `generation_micro_batch_size` does not batch the new HF
search path. Simulator calls are sequential and use existing backend timeout
and license-seat controls.

## Run a small evaluation first

From the `atpgllm` repository, in the configured evaluation environment:

```bash
SAMPLING_METHOD=mcts SEARCH_BUDGET=8 NUM_COMPLETIONS=4 PASS_AT_K="1 2 4" \
MAX_TOOL_ROUNDS=2 MAX_EVAL_SAMPLES=32 REPORT_TO=none \
SEARCH_CONFIG=scripts/eval/configs/conversation_search.json \
  ./scripts/eval/eval_sft_policy_checkpoints.sh runs/sft_granite_4.2_8b/checkpoint-200
```

Set `SAMPLING_METHOD=evolutionary` for evolution. The GRPO checkpoint launcher
accepts the same settings. Set `DRY_RUN=1` to preview either launcher.

For direct Python invocation, pass
`--search_config scripts/eval/configs/conversation_search.json`,
`--sampling_method mcts` or `evolutionary`, `--budget 8`, and the usual checkpoint,
backend, dataset, and completion arguments. `SEARCH_CONFIG` is also read directly
from the environment. Unknown configuration keys and invalid values fail before
model loading.

For best-of-N use `SAMPLING_METHOD=best_of_n BEST_OF_N_WIDTH=8` and unset
`SEARCH_BUDGET`. For the vector-only baseline use
`SAMPLING_METHOD=vector_evolutionary SEARCH_BUDGET=8`; this loads the tokenizer
for the existing dataset formatting flow but no LLM. Its expected outputs come
from simulation, so compare detection versus simulator cost separately from
model answer fidelity. `random` still samples both PI and PO values.

## Budget and policy details

N (`NUM_COMPLETIONS`) is the number of independent search results. B
(`SEARCH_BUDGET`, or `BEST_OF_N_WIDTH`) is a maximum number of attempts within
each result. Invalid/incomplete attempts count; already completed slots do not
cancel other slots. Every policy stops its slot on acceptance. Consequently,
best-of-N may return before drawing all B candidates when it already has an
accepted one.

The JSON limits apply across all branches of a slot. `MAX_NEW_TOKENS` still caps
an individual backend request; `action_tokens` can impose a smaller cap. A
truncated buffer is resumed until it reaches a tool/final boundary or exhausts
its token, context, or action allowance. `max_actions` is a safety limit on
generation segments and executed tool transitions along a path. It is not B.

A tool call counts toward the path's inherited `MAX_TOOL_ROUNDS`. A path hitting
that cap can fail while other branches continue. Simulator-request and total
token exhaustion apply to the entire slot. Protocol repair and infrastructure
retry limits are distinct. Backend errors with unknown token usage conservatively
charge the entire reserved request allowance and increment
`generation_usage_unknown`; those counts are bounds rather than measured tokens.

Mutation/crossover start from the original problem plus an explicit controller
instruction naming the proposed vector. This discards dependent old text and
observations. Feedback continuation instead preserves a failed tool exchange and
adds a request to reconsider it. Set `controller_edits=false` to disable direct
vector mutation/crossover; protocol repair and feedback instructions remain
visible in the conversation. Operator weights are ordered as feedback, mutation,
crossover, fresh seed. Duplicate edited vectors fall back to fresh generation.

Evolution keeps a bounded population with vector diversity and increases fresh
seeds after stagnation. MCTS retains complete tool states, excludes closed
terminal nodes from selection, and limits duplicate proposals. If all current
children are closed, it can admit another action up to `max_children` without
waiting for the ordinary widening threshold. Terminal duplicates reuse their
score without adding another backup. All generated proposals still cost tokens.

`full_accuracy` requires correct complete PI/PO reporting as well as detection.
Detection alone is the default acceptance rule. A final vector different from
the tool's vector triggers its own verification; no earlier answer field can
silently replace the final vector.

## Outputs and compatibility

Checkpoint launchers include `csv1` in output filenames to distinguish the new
protocol from older evaluations. Use a separate results directory when comparing
different JSON configurations; the filenames do not encode every configuration
field.

The metrics JSON includes resolved configuration, protocol version, per-problem
completions and per-slot usage. A neighboring `*.slots.jsonl` contains selected
trajectories, observations, final answers, scores, seeds, and costs. Enable
`save_trace` for generation/candidate/tool events and cumulative budget snapshots.
Failures count in pass@k and aggregate component denominators.

`simulator_requests` includes tool and verifier cache hits;
`simulator_executions` counts calls through the configured simulator wrapper.
That wrapper can have its own backend cache, so this counter is not a count of
TetraMAX subprocess launches. `generator_calls` remains a compatibility field
and now counts per-sequence backend requests; use the structured usage fields
for new comparisons.

## Verification and remaining experiments

The focused tests cover conversation histories, JSON/XML calls, small budgets,
prefixes, terminal/duplicate MCTS handling, vector edits, acceptance, slot seeds,
real reward/cache behavior, real fast simulation, local Granite/Qwen tokenizers,
a tiny CPU HF model, the vLLM adapter contract, and evaluator metrics/trajectory
files including failure slots. They require no model download.

```bash
python -m pytest tests/unit/test_conversation_search.py \
  tests/unit/test_search_integration.py tests/unit/test_random_sampling.py \
  tests/unit/test_reward_objectives.py tests/unit/test_chat_templates.py \
  tests/unit/test_eval_scripts.py -q
```

A full trained-checkpoint GPU evaluation and production TetraMAX comparison are
still experimental validation work. Optional propagation-distance values,
netlist-cone mutation targeting, multiple concurrent rollouts in one tree, and
cross-slot simulator caches are not enabled. The default implementation uses
activation-based values and independent per-slot caches. The cited papers and
the original plan explain why these choices should be tested rather than assumed
to improve accuracy.
