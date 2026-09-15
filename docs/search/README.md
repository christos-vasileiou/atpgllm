# MCTS and evolutionary sampling with tools

Design review dated 14 September 2026, based on commit `01aaff3`. The initial
implementation is now available as `conversation-search-v1` (15 September 2026).
Start with [running the implemented search](IMPLEMENTATION.md) for commands,
configuration, behavior, and validation limits. Checkpoint performance gains
have not yet been established.

The recommendation is to make a completion a saved conversation: assistant text,
tool requests, actual simulator replies, and a final answer. MCTS should explore
alternative next steps in that conversation. Evolution should improve candidate
input vectors and regenerate everything that depends on a changed vector.

Start here, then read the documents in order:

| Document | What it explains |
| --- | --- |
| [Running the implementation](IMPLEMENTATION.md) | Current behavior, configuration, commands, output files, and verification. |
| [Implementation review](IMPLEMENTATION_REVIEW.md) | Historical behavior at `01aaff3`, confirmed problems, and their consequences. |
| [Proposed design](DESIGN.md) | How both methods should work inside completions that use tools, with examples. |
| [Implementation and evaluation plan](IMPLEMENTATION_PLAN.md) | Which changes to make first, configuration, tests, and fair comparisons. |

## The two methods in simple words

**MCTS:** Save a point in the conversation. Try several next steps. Check their
outcomes with the simulator. Spend more attempts on steps that lead to better
outcomes. A useful saved point is immediately after a simulator reply, because
the next step can respond to what the simulator actually found.

**Evolution:** Keep a small group of candidate solutions. Improve a candidate,
combine compatible input assignments from two candidates, or start a fresh one.
Simulate the new candidate and keep useful, different solutions. Changing an
input means the old simulation and answer must be reconsidered.

Both return one selected completion per search. `--num_completions N` means
running that whole search independently N times; `--budget B` controls the work
inside each search. Taking N winners from one shared tree or population would
change the meaning of the current pass@k evaluation.

## Recommended order

1. Correct conversation history, prefix continuation, tool execution, and final
   answer extraction. Add accurate cost accounting.
2. Establish a corrected tool-using `greedy` and `best_of_n` baseline.
3. Implement evolution over valid conversation steps and structured vectors.
4. Implement MCTS over the same steps, reusing the same execution and scoring code.
5. Compare quality at equal token and simulator budgets before increasing scale.

Evolution is a useful first experiment because the task has a concrete object to
improve: the primary-input vector. MCTS is especially worth testing when several
tool exchanges help the model diagnose and repair a candidate. These are design
hypotheses, not measured advantages on this dataset.

The proposal covers inference-time evaluation in
[`evaluate_model.py`](../../scripts/eval/evaluate_model.py). Reusing selected
trajectories in GRPO training would require a separate treatment of sampling
probabilities and tool-token masks.
