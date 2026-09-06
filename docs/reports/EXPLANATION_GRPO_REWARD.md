# GDPO reward for ATPG, in simple terms

The model must produce an input pattern that exposes a stuck-at fault at a
primary output. The external fault simulator decides whether that happened.
The model does not receive functional credit merely because its explanation
looks convincing.

## Why the old reward was changed

The old implementation added detection, fault activation, output accuracy,
formatting, tool output, and fault-name mention into one number. GRPO then
compared those totals inside each group of generated answers.

This can hide important differences. For example, two answers may receive the
same total even though one detects the fault and the other only follows the
format well. The [GDPO paper](../../../docs/papers/GDPO.pdf) calls this
multi-reward signal collapse. Turning off GRPO's standard-deviation division
does not fully solve it.

The current implementation is therefore GDPO-style:

1. Compare answers separately for each meaningful goal.
2. Normalize each goal inside the group.
3. Give the goals explicit priorities.
4. Combine them and normalize once across the full distributed generation
   batch.

## The four training goals

1. **Detection, weight 1.00.** Did any real primary output differ between the
   good and faulty circuits? This is the main objective.
2. **Activation, weight 0.25.** At the fault site, did the good circuit produce
   the opposite of the stuck value? This gives useful progress when every
   answer in a group fails to propagate the fault to an output.
3. **Output fidelity, weight 0.20.** After detection, did the claimed
   `EXPECTED_OUTPUT` match the simulator's good-machine outputs?
4. **Interface and format, weight 0.05.** After detection, was the answer
   parseable and was the PI assignment complete?

The weights are applied after each goal is normalized. This matters: multiplying
a raw GDPO reward by 12 instead of 1 does not reliably make it 12 times more
important, because normalization removes that raw scale.

## Why easy rewards wait for detection

Formatting and output reporting are easier than ATPG. If they were rewarded on
failed patterns, the model could improve its score without learning to detect
faults. They are now conditioned on simulator-confirmed detection:

- no detection: fidelity = 0 and format = 0;
- detection: fidelity and format can distinguish better detecting answers.

Activation remains available before detection. It is physically meaningful and
prevents every all-fail group from being automatically useless when some
patterns at least excite the target fault.

## What is logging only

Fault-name mention, tool-response agreement, simulation-table presence, and
detailed accuracy flags end in `_logonly`. They appear in diagnostics but do not
change the policy. A model can copy a fault name or table layout without
creating a valid test, so these are unsafe training goals.

The generated PI vector is also not compared with the dataset's one reference
vector. Many different vectors may detect the same fault. Instead, the reward
checks that the required PI names have valid binary values and lets simulation
judge the vector.

## Simulator safety

The model's claimed output values are never inserted into the good circuit.
Canonical output names come from the netlist, and their values are computed by
the circuit. Detection can still be scored when `EXPECTED_OUTPUT` is missing or
wrong; those mistakes only reduce fidelity or format.

## Important limitations

- GDPO cannot create information when every answer has the same result. A
  constant goal contributes zero advantage.
- If all patterns fail both detection and activation, that group has no
  task-reward learning signal. The KL term can still pull the policy toward the
  fixed SFT reference.
- Activation says the fault was excited, not that it was propagated close to an
  output.
- The four weights are engineering priorities, not experimentally proven
  optima.
- Unit tests verify the equations and simulator independence, but improved
  convergence must still be confirmed by a controlled GRPO-versus-GDPO
  training run.

Implementation: `atpgllm/llm/reward_funcs.py`,
`atpgllm/training/gdpo.py`, and the two custom GRPO trainers. Exact equations
are in [`TECHNICAL_REPORT_GRPO_REWARD.md`](TECHNICAL_REPORT_GRPO_REWARD.md).
