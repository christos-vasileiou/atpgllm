# Implementation and evaluation plan

This plan records the staged implementation of the [proposed design](DESIGN.md).
An initial `conversation-search-v1` implementation is now available. See
[implementation status](IMPLEMENTATION.md) for delivered modules, configuration,
tests, exact behavior, and remaining experimental validation. The sections below
preserve the original roadmap and its scientific basis.

## Phase 1: correct the completion contract

The first deliverable is reliable generation and scoring, before changing search
selection rules.

| Work | Existing location | Done when |
| --- | --- | --- |
| Preserve complete history and execute pending calls before continuation. | `Generator.generate_with_tools` in [`sampling_strategies.py`](../../atpgllm/training/sampling_strategies.py) | A model continuation after two calls sees both complete exchanges. |
| Condition all prefixed generation on the actual prefix. | `SamplingStrategy._generate`, MCTS `_rollout`, evolution `_spawn_children` | Both tool and non-tool backends receive the selected prefix exactly once. |
| Replace unsafe crossover/mutation cuts with valid saved boundaries. | Evolution `_plan_children` and `_first_segment` | Editing a proposal cannot leave its old observation or dependent answer attached to the replacement. |
| Extract the explicit final answer for scoring; bind fidelity to the matching observation. | [`reward_function_factory.py`](../../atpgllm/training/reward_function_factory.py), [`reward_funcs.py`](../../atpgllm/llm/reward_funcs.py) | An earlier failed vector and later corrected vector are scored using the later final answer. |
| Share acceptance logic with search. | `is_completion_correct`, `Verifier`, strategy factory | `full_accuracy` continues after detection if the final output still needs repair. |
| Fix exact seed allocation and per-operator generation settings. | Evolution `_seed_population`, `_spawn_children` | Budgets 1–8 generate exactly the intended seeds; mixed batches preserve each operator's settings. |

Introduce the final-answer parser as an explicit evaluation path first. The
reward module is shared with training: retain legacy entry points until their
single-answer behavior is checked, and version evaluation output when parsing
semantics change. Add no reference answer fields to search prompts.

## Phase 2: one runner, structured results, and cost accounting

Extract the common functionality so `evaluate_model.py` is a caller rather than
an imported utility dependency of the training package.

Suggested module boundaries under `atpgllm/training/`:

| Proposed module | Responsibility |
| --- | --- |
| `search_types.py` | Problem context, immutable states, generation/tool results, candidates, typed statuses. |
| `completion_runner.py` | Render messages, continue token buffers, resolve calls, stop at safe boundaries, finalize a path. |
| `search_budget.py` | Budget reservation, usage counters, stable seed derivation, attempt limits. |
| `search_verifier.py` | Final-answer validation, simulator result cache, acceptance and search value. |
| `sampling_strategies.py` | Public strategy factory and algorithms; preserve its existing caller interface during migration. |

Keep model-specific rendering in the existing
[`revert_template.py`](../../atpgllm/training/revert_template.py) utilities and
tool schema/handler code in [`tools.py`](../../atpgllm/training/tools.py).
The runner should receive canonical messages from prompt construction; reversing
a rendered prompt is only a compatibility fallback. A failed fallback is an
explicit error, not an empty system/user context.

Route search, best-of-N, and the greedy evaluator through this runner, with
backend adapters returning finish metadata and measured usage. Migrate one
caller at a time and compare its rendered prompts against the old single-round
path. Preserve the final flattened output ordering expected by
`run_strategy_batch`.

Keep the existing `SamplingResult` fields temporarily and add structured usage,
statuses, and trace IDs. Mark `generator_calls` as a legacy estimate; do not
quietly reinterpret historical data. Keep `raw_results` in the evaluator and
serialize per-slot costs as well as totals.

Gate: a complete replayable trajectory and a consistent budget ledger for each
returned slot, including failure slots. No new algorithm is benchmark-ready
before this gate passes.

## Phase 3: establish corrected baselines and implement evolution

Run corrected `greedy` and `best_of_n` through the common runner. These establish
whether the tool protocol and final-answer parser work on actual checkpoints.

Then implement evolution in this order:

1. Candidates with canonical vectors and saved conversation states.
2. Feedback continuation and fresh-seed operators.
3. Valid vector mutation, with controller instructions recorded explicitly.
4. Bit-subset crossover, regeneration of dependent text, and operator-specific
   batching.
5. Bounded population selection with vector diversity and stagnation handling.

Keep the operator mix fixed during an evaluation run. Run a pure-LM variant and
a controller-assisted variant separately to measure how much improvement comes
from direct vector edits. This distinction is part of the sampling policy and
must appear in results.

Gate: no stale observations, exact budget enforcement, and an informative
comparison with best-of-N at equal resources. Lack of quality improvement is a
valid experimental result; it should not be hidden by spending more tokens.

## Phase 4: implement MCTS over complete actions

Replace `_MCTSNode.prefix` with a saved conversation state while retaining a
readable prefix for debugging. Implement boundary-aware expansion, bounded
rollouts, terminal-node closure, progressive widening, and backup of each new
outcome once. Store the best final candidate separately from tree statistics.

Begin with uniform priors and activation-based values. Add LM-scored priors as
a switch after correctness checks pass. Keep one active rollout per tree and
batch across independent searches. Only then consider within-tree concurrency
or extra propagation diagnostics.

Gate: controlled examples show that the tree can branch after a real tool
observation, keep siblings isolated, revisit promising nonterminal branches,
and terminate when no frontier or resources remain.

## Configuration and compatibility

Keep the existing CLI meanings that users rely on:

| Existing setting | Keep |
| --- | --- |
| `--sampling_method` | `greedy`, `best_of_n`, `mcts`, `evolutionary`, `random`. |
| `--num_completions` | N independent applications of the selected policy. |
| `--n` | B for best-of-N only. |
| `--budget` | Per-slot search limit for MCTS/evolution; record the new attempt-based semantics as a versioned change. |
| `--max_tool_rounds` | Per-path cap, including tool exchanges inherited from a saved prefix. |
| `--threshold_mode` | The same acceptance predicate for search and reported metrics. |

The implemented validated search configuration object is exposed through the
`--search_config` file option. Reject unknown
keys and nonsensical values instead of silently ignoring them. Persist the full
resolved configuration, including temperatures currently hard-coded in Python.

Initial experiment settings, subject to validation-set tuning:

| Proposed setting | Starting choice | Reason |
| --- | --- | --- |
| Search budget B | 8, 16, 32 | Observe quality/cost scaling before large runs. |
| Tool rounds per path | 2; compare 0, 1, 3 | Two rounds allow a failed proposal followed by a repair. |
| Per-slot generated-token cap | 32,768 | Explicit total across all branches; may bind before B. |
| Per-slot logical simulator-request cap | 32 | Count both tool requests and final verification. |
| Per-slot actual simulator-execution cap | 32 | Include retries; cache hits consume no execution. |
| Finalization token reserve | 1,024 | Leave room to turn a tool result into an answer. |
| MCTS prior | Uniform first | Separates action/state correctness from prior quality. |
| MCTS widening / exploration | Rule in the design; `c=1.25` | Small initial branching, bounded later expansion. |
| Evolution population | `min(6, B)` | Retains a manageable batched population. |
| Duplicate proposal limit | 3 per requested new action/child | Prevent no-progress loops. |
| Tool/parse repair limit | 1 per failed action | Recovery stays bounded and accounted for. |

The 32,768-token starting cap is a work allowance, not a claim that every
checkpoint fits it well. Tune on validation trajectories and publish the number
of paths truncated by each limit. Set a per-generation cap no larger than the
remaining allowance and available model context. Select simulator timeouts using
observed backend latency; do not guess a universal cluster timeout.

Legacy runs and corrected runs need different `search_protocol_version` values
and distinct output filenames. Record the old/new final-answer parser version,
budget semantics, tool protocol, and success criterion. Fixing the parser can
change reported scores even if model weights are identical.

## Tests that establish useful behavior

These are the roadmap's required checks. The current automated coverage is
listed in [implementation status](IMPLEMENTATION.md). Use recording fake
generators and deterministic tiny circuits before GPUs.

| Test | Required observation |
| --- | --- |
| Two or three tool rounds | Every later prompt contains all preceding assistant/tool pairs in order. |
| Pending request in a saved prefix | The tool executes before the next model call, exactly once for that action. |
| Partial JSON/XML, multiple calls, unknown tool | Each follows the documented incomplete/error policy; no silent first-call success. |
| Tool cap reached with pending request | Incomplete failure status; no successful final answer inferred. |
| Prefix with tools on/off | Backend sees the prefix once; output rendering contains it once. |
| Nonempty EOS vs token-limit stop | Terminal answer is closed; truncated buffer can resume only within allowance. |
| Backend returns too few/many outputs | Explicit cardinality error with no cross-slot reassignment. |
| Earlier vector A, final vector B | B drives final simulation; matching observation drives fidelity. |
| Replacement mutation after a tool call | Old dependent suffix is removed; changed vector gets a new matching result. |
| Appended repair | Earlier failed exchanges remain intact as history; later final vector is scored. |
| Same vector, different expected outputs | Simulation may hit cache; final-answer fidelity is scored separately. |
| Wrong doc ID, fault, PI/PO names or duplicate bits | Validation fails before an unbound/wrong-target simulation. |
| Cache key varies by netlist/backend/fault | No result leaks across different simulation contexts. |
| MCTS widening | Additional actions become available at a previously visited internal node. |
| Cached/closed terminal MCTS leaf | It does not create repeated evidence or consume all remaining iterations. |
| Evolution budgets 1–8 and short final batch | Exact admitted counts and no hidden generated samples. |
| Mixed evolution operators | Each request retains its own generation settings. |
| Detection with incorrect expected output | Stops under the configured detection criterion; continues repair under full accuracy. |
| Infrastructure timeout | Error recorded separately, bounded retry charged, other slots retain their state. |
| Exhaustion during expansion, tool call, or repair | No overspend; return exactly N slots with explicit statuses. |
| Reorder problems or batch scheduling | Stable per-slot request seeds and isolated search state. |

Integration checks should render Qwen-style JSON and Granite 4.2 XML tool
exchanges with actual tokenizers. Then run a small HF and vLLM smoke evaluation
with the same checkpoint, criterion, and resource limits. Compare protocol and
usage behavior, not exact generated strings across backends. Validate simulator
cache equivalence on the configured production backend as well as tiny circuits.

Run existing chat-template, reward, random-sampling, and evaluation-script tests
when the affected code changes. A documentation-only change does not require a
training job or the full model test suite.

## Evaluation that answers whether search is useful

Use a fixed validation subset first, stratified by circuit size and target-fault
type where practical. Freeze the dataset revision, split, checkpoint, simulator
backend/configuration, tokenizer, and success criterion. Tune only on validation;
freeze choices before the held-out evaluation.

Start with 32 problems, N=4, B=8 as a smoke experiment; use k values no larger
than N. Expand to the fixed comparison set with N=16 and B in {8,16,32}, subject
to the shared caps. Replicate the comparison with at least three run seeds if
resources permit. Report these as proposed run sizes, not a power calculation.

| Comparison | Question |
| --- | --- |
| Corrected greedy vs best-of-N | How much does extra independent sampling already buy? |
| Best-of-N vs evolution vs MCTS | Does adaptive search improve outcomes at the same resources? |
| Tools 0/1/2/3 rounds | Is there enough useful feedback depth to justify search? |
| Evolution with/without vector edits and diversity | Do concrete candidate changes help beyond suffix resampling? |
| MCTS fixed branching vs widening; uniform vs LM priors | Which added selection mechanisms justify their cost? |
| Activation-only vs optional propagation diagnostics | Does more simulator guidance help on unseen problems? |

The primary comparison gives each returned slot the same generated-token and
logical simulator-request caps, plus a common actual-execution cap. Publish
success-versus-consumed-cost curves, since caps alone do not force equal usage.
Report latency separately: batching, prompt lengths, prefix caching, and simulator
cache hits can make equal generated tokens take different time. Best-of-N must
charge all its samples and their tool loops to the same slot ledger.

Report pass@k for the search-augmented policy. If a problem has C accepted
completions among N independent search slots, use the existing estimator:

```text
pass@k = 1 - choose(N - C, k) / choose(N, k)
```

Use a numerically stable implementation and zero combinations where appropriate.
Never count internal branches as extra N slots or stop generating the remaining
slots because an earlier slot succeeded. Internal attempts may be adaptive and
correlated; the complete search procedure must be fixed and independently
repeated across the output slots.

Also report accepted completions, verified detection, full accuracy, valid final
answer rate, unique vector count, cache-hit rate, malformed-call rate, truncation
rate, infrastructure errors, generated/prompt tokens, simulator requests and
executions, wall time, and time/cost to first acceptance. Include failures in the
primary denominator and publish error counts separately.

Use paired comparisons on the same problems. Estimate uncertainty by resampling
problems, keeping all slots of a problem together. If several problems are faults
from the same circuit, cluster the resampling by circuit to avoid treating those
correlated cases as independent evidence.

Adopt a more complex policy when it improves acceptance at comparable measured
cost, or reduces cost at comparable acceptance, with no unexplained protocol
failures. If the apparent gain disappears against best-of-N or after accounting
for tool calls, keep the simpler policy and document the negative result.

## Artifacts to save from an implemented run

Keep aggregate metrics and add one per-slot JSONL record with:

```text
problem_id, completion_slot, search_protocol_version, seed
checkpoint/dataset/simulator/tokenizer identities and resolved config
selected trajectory, explicit final answer, status, reward components
generated/prompt tokens, backend requests, simulator requests/executions
cache hits, stop reason, elapsed time, trace_id
```

The optional search trace records node/parent IDs or candidate/operator lineage,
message state hashes, proposal parameters, action/observation references, and
budget deltas. It should make stale observations and duplicate work observable
without requiring inspection of every full completion. Store large histories
once and reference them from branches.

## Scientific basis and what this plan adds

The plan combines established search methods with adaptations for this
repository's tool-using ATPG completions. The papers below support particular
ingredients; they do not validate this exact combination, its numerical settings,
or its performance on ASAP7. Treat it as a literature-grounded experimental
design until the implementation and comparisons above provide that evidence.

### Which paper supports which part

| Plan component | Published basis | Scope of the support and difference from this plan |
| --- | --- | --- |
| Continue generation using actual tool observations. | [R1: ReAct](https://arxiv.org/abs/2210.03629), §2. | Interleaves reasoning, actions, and environment observations. Our immutable messages, pending-call handling, and JSON/XML execution rules are implementation choices. |
| Search alternative conversation paths and back up their outcomes. | [R2: LATS](https://proceedings.mlr.press/v235/zhou24r.html), §§3.2 and 4. | Applies MCTS to LM agents with environment feedback. LATS uses UCT, LM-based evaluation, and reflection; this plan uses PUCT and simulator-based values, without requiring reflection. |
| Use PUCT for language generation guided by an outcome metric. | [R3: Leblond et al.](https://aclanthology.org/2021.emnlp-main.662/), §4.2. | Gives a PUCT decoding formulation for translation. Moving from token actions to assistant/tool actions, using rollout simulation, and adding a `+1` visit offset are adaptations here. |
| Admit more actions as a state receives more visits. | [R4: Continuous RAVE](https://proceedings.mlr.press/v20/couetoux11.html), §2.2. | Describes progressive widening, including square-root growth. The minimum of two children, ceiling, visit offset, and maximum of eight are our finite-budget choices. We do not implement its RAVE machinery. |
| Evolve input vectors and evaluate them with a fault simulator. | [R5: Rudnick et al.](https://www.researchgate.net/publication/4225889_Sequential_Circuit_Test_Generation_in_a_Genetic_Algorithm_Framework), §§II–III, author-uploaded paper. | Direct ATPG precedent for selection, crossover, mutation, and simulator fitness. It searches vectors/sequences for circuit testing; it does not evolve LLM conversations or require consistent assistant/tool histories. |
| Use an LLM to generate improved candidates and retain useful diversity. | [R6: FunSearch](https://www.nature.com/articles/s41586-023-06924-6), “FunSearch” and “Evolutionary method and program selection.” | Combines LM proposals, executable evaluation, and diverse program populations. Our candidates are vector-bearing trajectories; Hamming-distance selection and the four-operator mix are not the FunSearch algorithm. |
| Estimate pass@k from N samples and C successes. | [R7: Chen et al.](https://arxiv.org/abs/2107.03374), §2.1 and Appendix A. | Provides the estimator and its justification for independent samples. Treating one complete adaptive search as one policy sample is our application of that sampling requirement. |

These distinctions affect implementation. Uniform priors in our PUCT formula do
not turn it into LATS's UCT formula. Likewise, length-normalized action scores
are heuristic priors, not the token-policy probabilities used in translation.
The fixed eight-child ceiling can permanently exclude actions, so this plan
does not inherit an asymptotic completeness or optimality guarantee from
progressive-widening research. Its intended justification is finite-budget
experimental performance. See [R2](https://arxiv.org/pdf/2310.04406),
[R3](https://aclanthology.org/2021.emnlp-main.662.pdf), and
[R4](https://proceedings.mlr.press/v20/couetoux11/couetoux11.pdf) for the respective
selection and widening formulations.

### Project-specific additions worth reporting

“Addition” here means a choice specified by this plan rather than a method
directly taken from the cited papers. It does **not** establish that the idea is
new across the research literature. A novelty claim would require a broader
related-work comparison and experimental results.

| Addition in this plan | Why it matters | Status and evidence needed |
| --- | --- | --- |
| Rebuild all dependent messages when an earlier vector proposal changes; distinguish replacement from appending a repair. | Prevents a new vector from inheriting an old simulator result or explanation. | Engineering adaptation to trajectory editing. Test both edit modes and measure stale-result failures. |
| Make the final answer explicit and match simulator observations by vector and provenance. | Earlier failed attempts remain in history without becoming the scored answer. | Correctness contract motivated by the local parser behavior, not a new search algorithm. Test corrected answers and mismatched observations. |
| Separate verified detection, final-answer acceptance, and search value. | A detecting vector can still need expected-output repair under `full_accuracy`. | ATPG completion objective design. Compare acceptance-aligned stopping against detection-only stopping. |
| Preserve a detecting vector while repairing only its final answer. | Can reuse valid simulation work instead of searching for another detecting vector. | Efficiency hypothesis. Measure simulator executions and full accuracy at equal budgets. |
| Combine feedback continuation, PI mutation, PI-subset crossover, and fresh seeds in one trajectory population. | Connects language-based repair with concrete circuit inputs. | Proposed hybrid policy. Compare pure-LM, controller-assisted, and vector-only variants. Individual GA operators are established by [R5](https://www.researchgate.net/publication/4225889_Sequential_Circuit_Test_Generation_in_a_Genetic_Algorithm_Framework). |
| Charge all branches, discarded tokens, repairs, and tool/verifier requests to one slot; distinguish cache hits from executions. | Makes expensive adaptive searches comparable with best-of-N. | Evaluation and systems design. Audit ledgers and compare success against measured resource use. |
| Repeat the entire search independently for each output slot, with isolated feedback and stable seeds. | Gives pass@k the meaning of success for the fixed search-augmented policy. | Application of [R7](https://arxiv.org/abs/2107.03374), not a new estimator. Test isolation and report policy/configuration versions. |
| Exact value levels, population policy, operator percentages, retry limits, and resource caps. | Make an implementable first experiment but can strongly affect results. | Untuned hypotheses: values `0/0.2/0.8/1`, mix `50/25/15/10%`, population six, and token/tool budgets need validation-set ablations. |

Netlist-derived mutation targeting and extra propagation diagnostics are optional
extensions, not established improvements of this plan. Earlier genetic ATPG work
already used fault-propagation and circuit-activity information in fitness, so
the general idea of richer simulator guidance is not new. What remains to define
and test here is a useful diagnostic for this single-target completion task and
its supported backends.
[R5: Rudnick et al., fitness functions in §III](https://www.researchgate.net/publication/4225889_Sequential_Circuit_Test_Generation_in_a_Genetic_Algorithm_Framework).

The strongest potential contribution is the evaluated combination: search over
tool-using ATPG conversations while keeping edited vectors, their observations,
and the final answer consistent under an explicit resource budget. State
preservation, caching, input validation, and the repairs listed in the
[implementation review](IMPLEMENTATION_REVIEW.md) are necessary engineering,
not evidence of scientific novelty by themselves.

### An additional baseline suggested by the literature

Add a **vector-only genetic search** using the same simulator, target faults,
PI representation, and simulator-request allowance. It proposes binary vectors
using selection, mutation, and crossover, without an LLM. This isolates whether
the language model contributes beyond established genetic ATPG.
[R5: Rudnick et al.](https://www.researchgate.net/publication/4225889_Sequential_Circuit_Test_Generation_in_a_Genetic_Algorithm_Framework).

This would be a new experimental baseline, not the existing `random` strategy,
which samples PI/PO assignments. Compare detection versus simulator work
directly. If the vector-only baseline constructs expected outputs from the
simulator, label that as simulator-derived reporting and keep it separate from
the model's final-answer fidelity results. This baseline is an addition to the
original comparison matrix prompted by the literature review.

### References

**R1.** Shunyu Yao et al. **ReAct: Synergizing Reasoning and Acting in Language
Models.** ICLR, 2023. [Paper](https://arxiv.org/abs/2210.03629).

**R2.** Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and
Yu-Xiong Wang. **Language Agent Tree Search Unifies Reasoning, Acting, and
Planning in Language Models.** ICML, PMLR 235:62138–62160, 2024.
[Proceedings and paper](https://proceedings.mlr.press/v235/zhou24r.html).

**R3.** Rémi Leblond et al. **Machine Translation Decoding beyond Beam Search.**
EMNLP, pp. 8410–8434, 2021.
[Proceedings and paper](https://aclanthology.org/2021.emnlp-main.662/).

**R4.** Adrien Couëtoux, Mario Milone, Mátyás Brendel, Hassan Doghmen, Michèle
Sebag, and Olivier Teytaud. **Continuous Rapid Action Value Estimates.** ACML,
PMLR 20:19–31, 2011.
[Proceedings and paper](https://proceedings.mlr.press/v20/couetoux11.html).
Used here for its description of progressive widening, not as a claim that this
paper first introduced widening.

**R5.** Elizabeth M. Rudnick, Janak H. Patel, Gary S. Greenstein, and Thomas M.
Niermann. **Sequential Circuit Test Generation in a Genetic Algorithm
Framework.** 31st ACM/IEEE Design Automation Conference, pp. 698–704, 1994.
DOI: `10.1109/DAC.1994.204191`.
[Author-uploaded paper](https://www.researchgate.net/publication/4225889_Sequential_Circuit_Test_Generation_in_a_Genetic_Algorithm_Framework).

**R6.** Bernardino Romera-Paredes et al. **Mathematical discoveries from program
search with large language models.** Nature 625:468–475, 2024; published online
14 December 2023. DOI: `10.1038/s41586-023-06924-6`.
[FunSearch paper](https://www.nature.com/articles/s41586-023-06924-6).

**R7.** Mark Chen et al. **Evaluating Large Language Models Trained on Code.**
arXiv:2107.03374, 2021. [Paper](https://arxiv.org/abs/2107.03374).
