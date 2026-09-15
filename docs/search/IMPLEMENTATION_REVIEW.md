# Review of the existing implementation

Reviewed on 14 September 2026 at commit `01aaff3`. Findings below come from source
inspection and small isolated checks of the current Python methods. No GPU
generation, checkpoint evaluation, or production simulator benchmark was run.

## How a completion reaches the search

The evaluation path is:

```text
dataset record
  -> format_eval_prompt: system and user messages, rendered with the tokenizer
  -> make_strategy: choose MCTS or evolution; enable tools if max_tool_rounds >= 1
  -> run_strategy_batch: run N independent searches for each problem
  -> Verifier: score each candidate using the fault simulator
  -> choose one completion from each search
  -> is_completion_correct: calculate the pass@k success count
```

[`sampling_strategies.py`](../../atpgllm/training/sampling_strategies.py) contains
the strategies, generator adapters, and verifier. The evaluator sets most search
parameters directly in `evaluate_pass_at_k`; they are not currently individual
CLI options.

| Method | Current behavior per returned completion |
| --- | --- |
| `greedy` | One model trajectory through the evaluator's generation/tool loop. The name does not imply temperature zero. |
| `best_of_n` | Generate B model trajectories; select by detection, then scalar reward. |
| `mcts` | Run up to B selection iterations over text prefixes, score rollouts, return the best observed completion. |
| `evolutionary` | Seed candidates, choose elites, generate modified children, score up to B candidates, return the best. |
| `random` | Construct random PI/PO assignments without the model. |

PI means primary input; PO means primary output. The current separation between N
returned completions and B internal attempts is worth preserving.

## What MCTS currently does

`_MCTSNode` stores a raw text prefix, children, visit count, accumulated value,
prior, and a terminal/value cache. It has no structured conversation or tool
observation state.

`MCTSStrategy._search_once` starts by completing the empty root. On failure it
expands that node into three sampled chunks, each up to 256 tokens. Later
iterations choose a leaf using PUCT, finish its prefix, score the finished text,
back up the score to its ancestors, and expand the leaf for future iterations.
Detection ends that search early.

PUCT combines the average outcome of a branch with an exploration bonus based
on its prior and visit count. vLLM provides mean token log-probabilities; the
code softmaxes them over sampled siblings. Hugging Face falls back to uniform
priors. These length-normalized chunk scores are a heuristic preference among
sampled chunks, not a calibrated probability distribution over all possible
conversation actions. Duplicate chunk strings are merged.

Expansion bypasses tools. A tool request in a chunk is resolved later during a
rollout. Consequently, the tree branches within the first assistant text, while
the useful decisions after a real tool reply exist only inside rollouts.

The defaults are branching 3, chunk size 256, exploration constant 1.25, chunk
temperature 0.9, and rollout temperature 0.7. Changing the evaluator's ordinary
temperature does not override these hard-coded search temperatures.

## What evolution currently does

`EvolutionaryStrategy` seeds candidates at temperatures 0.5, 0.8, and 1.0, using
a default population size of 6. It selects the top 3 candidates from the entire
accumulated archive by scalar reward. This is archive-based elitism; the active
parent pool is not a separately bounded, diverse population.

Children use one of two operations:

- **Crossover:** Replace the host's entire quoted `INPUT_VECTOR` field with a
  donor's field. Cut after that field and generate the rest. This transfers an
  existing vector; it does not combine the parents' individual input bits.
- **Mutation:** Cut at a random character position, approximately within the
  middle half of the first assistant segment, and regenerate the suffix.

Seeds and children can use the tool loop. Any detected candidate stops further
generations. Final selection ranks detection first and scalar reward second.

## Correctness issues to fix first

The function names below refer to
[`sampling_strategies.py`](../../atpgllm/training/sampling_strategies.py) unless
another source is linked.

| ID | Finding and source | Why it matters |
| --- | --- | --- |
| C1 | `Generator.generate_with_tools` rebuilds each round from `base_messages_cache` plus only the current assistant/tool pair. | At the second tool execution, the next model prompt loses the first exchange. The readable completion still contains it, so the saved text and actual model context disagree. |
| C2 | `EvolutionaryStrategy._plan_children` applies crossover to the full parent, although mutation uses `_first_segment`. | If the answer follows a tool reply, a crossover prefix contains the old reply. This violates `generate_with_tools`' documented prefix constraint and can preserve a simulation for a different vector. |
| C3 | `SamplingStrategy._generate` without tools sends only `prompts` to the generator and adds `prefixes` to the returned text afterward. | Evolutionary children are not conditioned on their retained parent prefix. MCTS has an explicit workaround in `_rollout`; evolution does not. |
| C4 | `generate_with_tools` generates more text before checking whether its initial prefix already contains a complete tool request. `_expand` also continues raw prefixes. | The model can continue past a tool request without seeing its result. A branch may contain text written before the observation it appears to use. |
| C5 | `Generator.generate` returns text without finish reasons or stop-token metadata. `_expand` treats only empty continuations as terminal. | A nonempty chunk ending at EOS is indistinguishable from a truncated chunk and can be extended again. Ending an assistant tool-request turn must also be distinguished from ending the whole task. |
| C6 | [`reward_funcs.py`](../../atpgllm/llm/reward_funcs.py), `test_generation_grpo_reward`, selects `[0]` from all extracted answer fields. Its tool-table helpers also select the first matching response. | If a later assistant turn corrects an earlier vector, the verifier can score the earlier field. Tool fidelity can compare the final vector with an earlier, unrelated observation. |
| C7 | MCTS and evolution stop on `score.detected`; [`is_completion_correct`](../../scripts/eval/evaluate_model.py) also supports `full_accuracy`. | A detecting vector with an incorrect predicted output can stop search before satisfying the requested evaluation criterion. The search never receives that criterion. |

C1 is specific to the shared search tool loop. The evaluator's existing greedy
loops reconstruct from their current input instead; do not assume the same
history-loss mechanism applies to them. They still need parity tests when a
common runner replaces the separate implementations.

## Search quality and resource issues

| ID | Finding | Consequence |
| --- | --- | --- |
| Q1 | Mutation cuts by character; crossover keeps the host's preceding reasoning. | It can cut inside JSON/XML or preserve reasoning that assumes a different vector. |
| Q2 | There is no candidate-vector diversity policy or simulator-result cache in these strategies. | Different wording can repeatedly test the same vector; the elite archive can collapse to near-duplicates. The reward factory does cache parsed netlists, which is a different cache. |
| Q3 | `_spawn_children` uses crossover temperature for the entire batch if any child is a crossover; it also applies one shared token cap. | Mutation parameters depend on unrelated children in the batch. |
| Q4 | `_seed_population` uses floor division across temperatures, then truncates. | B=1 generates three samples and scores one. B=4 or 5 generates only three seeds. Default B>=6/population=6 does not show this particular mismatch. |
| Q5 | `generator_calls` counts different units across strategies and does not count every internal tool-round generation. The evaluator discards `raw_results`. | Existing output cannot establish equal generation or simulator cost. |
| Q6 | MCTS's B counts loop iterations, including cached terminal visits; expansion and tool work sit outside that counter. Token caps apply to individual generation calls. | B is not an end-to-end work limit. Repeated cached leaves can consume iterations without finding another candidate. |
| Q7 | MCTS and evolution process independent problems/searches sequentially. | GPU batching across independent work is largely unavailable, despite batched child generation inside evolution. |
| Q8 | `Verifier.score_many` converts an unexpected batch exception to empty reward dictionaries. Ordinary simulator failures appear in reward components. | Search does not explicitly distinguish infrastructure errors from valid failed candidates. |
| Q9 | Global Python/NumPy/Torch seeding exists, but there is no explicit seed per search slot and generation request. | Reordering batches, early stopping, or future parallel execution can change later searches' random streams. |

Both tool loops parse only the first supported call from an assistant segment.
Malformed or multiple calls do not have an explicit transition policy. The
search tool loop also lacks the greedy vLLM loop's output-count check, so a
backend cardinality mismatch can silently leave paths incomplete.

### The reward has less guidance than the comments suggest

The current scalar in
[`train_scalar_from_reward_components`](../../atpgllm/llm/reward_funcs.py) is:

```text
detection + 0.25 * activation + 0.20 * fidelity + 0.05 * format
```

The reward implementation gates fidelity and format by detection. Before
detection, a normally scored candidate therefore has scalar 0 or 0.25, according
to whether the fault site was activated. MCTS transforms these into approximately
0.5000 and 0.50625. Online min/max normalization can enlarge that difference,
but it cannot create guidance among candidates tied on activation. This is not
evidence for adding a learned process-reward model; first use the existing
simulator's activation signal explicitly and measure whether more diagnostics help.

## Isolated checks performed for this review

The relevant class definitions were extracted with Python's AST and executed
with recording generators, a recording chat-template stub, and a zero-score
verifier. This avoids loading a model or importing the production simulator.
The checks exercise control flow, not real tokenizer or simulator behavior.

| Check | Observed result |
| --- | --- |
| `_generate(["PROMPT"], prefixes=["PREFIX"])` with tools disabled | The backend saw `PROMPT`, not `PROMPTPREFIX`. |
| Two tool executions through `generate_with_tools` | Each rendered history contained only one tool message; the second history lacked the first exchange. |
| Forced crossover of parents with answer fields after a tool reply | The child prefix retained `</tool_response>`. |
| Seed budgets 1, 4, 5 | Generated/scored counts were 3/1, 3/3, and 3/3. |
| MCTS values for non-detected scores 0 and 0.25 | 0.5 and 0.5062496745. |

The existing tests include random-sampling helpers, chat-template parsing,
rewards, and evaluation scripts. A search of `tests/` found no direct tests for
`MCTSStrategy`, `EvolutionaryStrategy`, or the search `generate_with_tools` loop.
The [implementation plan](IMPLEMENTATION_PLAN.md) specifies the missing behavior
tests and real-backend checks.
