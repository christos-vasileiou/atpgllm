# Reward scenarios in training and inference

This note shows, with small flow graphs, how the GRPO reward scores every kind of trajectory during training, and how the same kinds of trajectory are judged at inference. The reward is implemented in `atpgllm/llm/reward_funcs.py` (`test_generation_grpo_reward`), and the reuse of repeated calls in `atpgllm/training/tools.py` (`ToolScheduler`). The trajectory reward was merged into `develop` as `c18dcba`. The uniqueness component and the reuse of repeated calls are on the branch `reward-uniqueness`.

## The two pipelines

During training the model sees only the observation returned by the simulator. The reward is computed once, after the whole trajectory has finished.

```
TRAINING
Prompt ──▶ turn 1 (think + call) ──▶ simulator ──▶ observation appended to text
                                                        │
       ┌────────────────────────────────────────────────┘
       ▼
     turn 2 (think + call again, or final answer) ──▶ ... ──▶ final answer
                                                                  │
                                                                  ▼
                                    REWARD over the whole trajectory (computed once)
                                    detection, activation, fidelity, format
                                                                  │
                                                                  ▼
                                    GDPO compares the trajectories of the same prompt

INFERENCE
Prompt ──▶ turn 1 (think + call) ──▶ simulator ──▶ observation appended to text
                                                        │
       ┌────────────────────────────────────────────────┘
       ▼
     turn 2 ──▶ ... ──▶ final answer ──▶ simulator runs the FINAL vector ──▶ success or failure
```

## Notation

1. A, B and C are input vectors. ✔ means the vector detects the fault and ✘ means it does not.
2. "act" says whether the vector activates the fault, which means it drives the faulty net to the opposite of its stuck value.
3. Format is shown as passed checks out of all checks. With E executed calls there are 4E + 4 checks, and with no call there are 6.
4. "uniq" is the uniqueness component.

## How each component is computed (short version)

1. **Attempts.** Every call the model sent to the simulator, in order, stopping at the first one that detects the fault. A repeated vector counts as another attempt, and a rejected call counts as a failed attempt.
2. **Detection.** 1/K if attempt K is the first that detects, and 0 if no attempt detects or there is no call. A repeated vector before the success therefore lowers detection.
3. **Activation.** The average over the attempts of whether each vector activates the fault.
4. **Fidelity.** How well the final answer matches the last call that detected the fault. It averages the share of inputs that agree with that call and the share of outputs that agree with the good circuit values it produced. It is 0 when no call detected the fault.
5. **Format.** The share of protocol checks that pass: reason then call, after a negative result reason and call with a new vector, after a positive result give the final answer with all three fields.
6. **Uniqueness.** The number of distinct vectors among the attempts divided by the number of attempts, and 0 when there is no call. Two calls are the same vector when they set the same inputs to the same values, in any order.
7. **Which vectors are simulated.** Earlier calls are scored from the result the simulator already returned. The calls of the last turn that called the simulator are simulated again, once for each distinct vector. The final answer is never simulated in training.
8. **Repeated calls are not run twice.** During training, a call that repeats an earlier call of the same trajectory (same input vector, same output nets, same fault, same circuit) gets the earlier observation back without running the simulator. It still counts as an attempt.
9. **Weights.** Detection 1.00, activation 0.25, fidelity 0.20, format 0.05, uniqueness 0.10.

# Training scenarios

### T1. First try succeeds (the ideal)

```
Prompt
 │
 ▼
Turn 1   think, call A
 │
 ▼
Sim      A ✔                  (observation appended)
 │
 ▼
Turn 2   final answer  INPUT_VECTOR = A, outputs copied from the observation
 │
 ▼
Reward   attempts   A✔
         detection  1/1 = 1        activation 1
         fidelity   1              format 8/8 = 1
         uniq       1/1 = 1
```

This is the best possible score.

### T2. One failure, then a correct retry

```
Prompt
 │
 ▼
Turn 1   think, call A
 │
 ▼
Sim      A ✘ (not activated)
 │
 ▼
Turn 2   think again, call B (new vector)
 │
 ▼
Sim      B ✔
 │
 ▼
Turn 3   final answer  INPUT_VECTOR = B
 │
 ▼
Reward   attempts   A✘ B✔
         detection  1/2 = 0.5      activation (0+1)/2 = 0.5
         fidelity   1              format 12/12 = 1
         uniq       2/2 = 1
```

The protocol is followed perfectly. The model loses detection only because it needed two tries.

### T3. Two failures, then success

```
Turn 1 call A ──▶ A ✘ (not activated)
Turn 2 call B ──▶ B ✘ (activated, but blocked before the outputs)
Turn 3 call C ──▶ C ✔
Turn 4 final answer = C

Reward   attempts   A✘ B✘ C✔
         detection  1/3 = 0.33     activation (0+1+1)/3 = 0.67
         fidelity   1              format 16/16 = 1
         uniq       3/3 = 1
```

Activation gives credit for B, which reached the fault but not an output.

### T4. Gives up after failures

```
Turn 1 call A ──▶ A ✘ (not activated)
Turn 2 call B ──▶ B ✘ (activated)
Turn 3 final answer = B   (answers after a negative observation)

Reward   attempts   A✘ B✘
         detection  0              activation 0.5
         fidelity   0 (no confirmed test to report)
         format     11/12 = 0.92   (fails "answer follows a detection")
         uniq       2/2 = 1
```

### T5. No tool call, the model just guesses

```
Turn 1   think, final answer = X     (X happens to detect)

Reward   attempts   none
         detection  0              activation 0
         fidelity   0              format 5/6 = 0.83 (fails "answer follows a detection")
         uniq       0
```

The guess earns nothing, because detection must come from a tool call.

### T6. Repeats a failed vector

```
Turn 1 call A ──▶ A ✘ (not activated)
Turn 2 call A ──▶ earlier observation of A returned, simulator not run again
Turn 3 call B ──▶ B ✔
Turn 4 final answer = B

Reward   attempts   A✘ A✘ B✔      (the repeat is an attempt too)
         detection  1/3 = 0.33     activation (0+0+1)/3 = 0.33
         fidelity   1              format 15/16 = 0.94 (fails "new vector")
         uniq       2/3 = 0.67
```

Compared with T3, which also succeeds on the third attempt with three different vectors, detection is the same, but uniqueness and format are lower. The simulator ran only twice during generation.

### T7. Keeps calling after a success

```
Turn 1 call A ──▶ A ✔
Turn 2 call B ──▶ B ✘   (unnecessary call)
Turn 3 final answer = A

Reward   attempts   A✔            (the list stops at the first success)
         detection  1              activation 1
         fidelity   1 (compared with A, the last call that detected)
         format     10/12 = 0.83  (fails "no call after a detection"
                                   and "answer follows a detection")
         uniq       1/1 = 1       (B comes after the success, so it is not an attempt)
```

Extra calls never raise detection, and format penalizes them.

### T8. Success, but the answer misreports the test

```
Turn 1 call A ──▶ A ✔   (A = a1=1 a2=0 a3=1 a4=1, good outputs y1=1 y2=0)
Turn 2 final answer   INPUT_VECTOR    a1=1 a2=0 a3=0 a4=1   (one input changed)
                      EXPECTED_OUTPUT y1=1 y2=1             (one output wrong)

Reward   attempts   A✔
         detection  1              activation 1
         fidelity   (3/4 inputs + 1/2 outputs)/2 = 0.63
         format     8/8 = 1
         uniq       1
```

Training does not simulate the answer itself, so fidelity is what catches the difference. At inference the changed vector would be simulated and might fail.

### T9. A rejected call (malformed vector or wrong fault)

```
Turn 1 call R  ──▶ simulator rejects it ("Tool execution failed ...")
Turn 2 call A  ──▶ A ✔
Turn 3 final answer = A

Reward   attempts   R✘ A✔          (a rejected call is a failed attempt)
         detection  1/2 = 0.5      activation (0+1)/2 = 0.5
         fidelity   1              format 11/12 = 0.92 (fails "call accepted")
         uniq       2/2 = 1
```

### T10. Runs out of tokens before the answer

```
Turn 1 call A ──▶ A ✔
(token budget ends, no final answer)

Reward   attempts   A✔             (A is simulated again, since it is the last calling turn)
         detection  1              activation 1
         fidelity   0 (no INPUT_VECTOR or EXPECTED_OUTPUT to compare)
         format     2/8 = 0.25     (all six final turn checks fail)
         uniq       1
```

### T11. The model writes a fake observation

```
Turn 1   think, then the model itself types
         <tool_response> {fake table: detected} </tool_response>
         and gives a final answer
         (no real call, so nothing is in the trainer's record)

Reward   attempts   none
         detection  0              activation 0
         fidelity   0              format 5/6 = 0.83
         uniq       0
```

Fake text earns nothing, because the reward reads only the record of executed calls.

### T12. Two calls written in one turn

```
Turn 1   think, call A, call B          (only A is executed during generation)
 │
 ▼
Sim      A ✘                            (the model never sees a result for B)
 │
 ▼
Turn 2   final answer = B
 │
 ▼
Reward   Turn 1 is the last calling turn, so A and B are both simulated now
         attempts   A✘ B✔
         detection  1/2 = 0.5      activation 0.5
         fidelity   inputs 1, outputs only as good as the model's own guess for B
         format     7/8 = 0.88     (fails "answer follows a detection",
                                    because the model was only shown A✘)
         uniq       2/2 = 1
```

If the listed call B names a different circuit or lacks one of the tool's arguments, the simulator would have rejected it, so the reward does not simulate it and B counts as a failed attempt.

### T13. Calls without reasoning

```
Turn 1   call A, with no reasoning block ──▶ A ✔
Turn 2   final answer = A

Reward   detection 1   activation 1   fidelity 1   uniq 1
         format    7/8 = 0.88   (fails "reasoning before the call")
```

### T14. Keeps repeating the same failed vector

```
Turn 1 call A ──▶ A ✘ (not activated)
Turn 2 call A ──▶ earlier observation returned, simulator not run
Turn 3 call A ──▶ earlier observation returned, simulator not run
Turn 4 final answer = A

Reward   attempts   A✘ A✘ A✘
         detection  0              activation 0
         fidelity   0              format 13/16 = 0.81 (fails "new vector" twice
                                                       and "answer follows a detection")
         uniq       1/3 = 0.33
```

Even when nothing in the group detects the fault, uniqueness and format push this trajectory below one that tries new vectors, as in T4.

### How GDPO then uses these numbers

```
Prompt P ──▶ 16 trajectories (for example T1, T2, T4, T5, T7, ...)
                │
                ▼
   Each component is compared inside the group of 16 separately
   (above the group average is good, below it is bad)
                │
                ▼
   weighted sum   1.00 detection + 0.25 activation + 0.20 fidelity
                + 0.05 format + 0.10 uniqueness
                │
                ▼
   T1 is pushed up the most, T5, T11 and T14 are pushed down
```

If every trajectory in a group has the same value for a component, that component contributes nothing for that prompt.

# Inference scenarios

There is no reward at inference. The observation is appended to the text exactly as in training, and success is decided only by simulating the final INPUT_VECTOR. The reuse of repeated calls applies only to the training loop, so at inference every executed call runs the simulator.

### I1. One call succeeds and the answer copies it

```
Turn 1 call A ──▶ A ✔ ──▶ final answer = A ──▶ simulate A ✔ ──▶ SUCCESS
```

### I2. Retry within the allowed tool rounds

```
Turn 1 call A ──▶ A ✘ ──▶ Turn 2 call B ──▶ B ✔ ──▶ final answer = B ──▶ simulate B ✔ ──▶ SUCCESS
```

### I3. The answer differs from the tested vector

```
Turn 1 call A ──▶ A ✔ ──▶ final answer = A' (one bit changed) ──▶ simulate A'
                                                                     │
                                         ┌───────────────────────────┴──────────────┐
                                         ▼                                          ▼
                                   A' ✔ SUCCESS                               A' ✘ FAILURE
```

### I4. No tool call

```
Turn 1 final answer = X ──▶ simulate X ──▶ SUCCESS if X detects, otherwise FAILURE
```

This differs from training, where the same guess earns 0 detection.

### I5. Tool round limit reached

```
(max tool rounds = 1)
Turn 1 call A ──▶ A ✘ ──▶ Turn 2 writes call B (not executed), no final answer
                                   │
                                   ▼
                        no INPUT_VECTOR ──▶ INVALID ──▶ FAILURE
```

### I6. Invalid final answer

```
final answer leaves an input unassigned
   or writes INPUT_VECTOR twice
   or the INPUT_VECTOR sits inside the reasoning instead of after it
        │
        ▼
     INVALID ──▶ FAILURE (nothing is simulated)
```

### I7. Several answer slots per problem (pass@k)

```
Problem ──▶ slot 1 (any of I1 to I6) ──▶ success or failure
        ──▶ slot 2                    ──▶ success or failure
        ──▶ ...
        ──▶ slot 16                   ──▶ success or failure
                     │
                     ▼
        pass@k = chance that at least one of k slots succeeds
```

# The key difference between training and inference

In training, the model is rewarded for what it verified with the simulator, and the final answer is held to that verification through fidelity and format. At inference, only the final vector counts. T8 and I3 show the one place where the two can disagree: an answer that changes the tested vector loses fidelity in training and is checked directly at inference.
