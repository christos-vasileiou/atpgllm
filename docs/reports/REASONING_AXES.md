# Reasoning audit and candidate GDPO reward axes

Measured 2026-11-02 on stored RFT rollouts, CPU only. Nothing was trained and no reward code was changed. Scripts: `scripts/star/score_reward_axes.py`, `scripts/star/check_reasoning_trace.py`. Numbers: `analysis/reasoning_axes_20261102/`.

## Recommendation

No axis that reads the reasoning text is ready to be a reward today. The model's internal claims are true about as often as random claims (0.49 against 0.50), and it almost never states a gate step that can be checked, so such a reward would mostly pay for noise.

The robust axis available now is a vector ladder, which is the mean of `activate_distance` and `detect_distance`. It gives failed attempts a signal in 84% of the groups where no attempt detects, it needs no format, and a detecting attempt is always at its maximum. It has one problem still open: a constant vector beats real attempts on it, and the first guard failed.

The way to reach reasoning is to teach it first and reward it second. Gate by gate demonstrations and retry rows go into SFT, in many phrasings and with no forced format. Reasoning axes become candidates only once the model's claims rise clearly above the 0.50 that random claims score.

## How each axis did

Attempts: 6,008 unhinted attempts in 750 prompt groups from rounds 1 to 5, 30 problems per level and round, training circuits. "All-fail signal" is the share of the 321 to 339 groups with no detection where the axis still varies. GDPO normalizes inside a group, so an axis that is constant there teaches nothing. "Predicts solved" correlates a problem's mean score on failed attempts with whether another attempt solved it, within a level. It is independent of how each axis is defined.

| Axis | All-fail signal | Predicts solved | Corr. with detection in group | Detecting attempt at group max | Honest mean | Best exploit | Verdict |
|---|---|---|---|---|---|---|---|
| **ladder** | **0.84** | **0.24** | 0.72 | 1.00 | 0.485 | all zeros 0.565 | **Best, needs a constant guard** |
| `detect_distance` | 0.58 | 0.29 | 0.81 | 1.00 | 0.332 | all zeros 0.393 | Part of the ladder |
| `activate_distance` | 0.80 | 0.18 | 0.46 | 1.00 | 0.637 | all zeros 0.736 | Part of the ladder |
| `propagation` (staged) | 0.67 | 0.21 | 0.95 | 1.00 | 0.262 | all zeros 0.314 | Usable, close to detection |
| `controlling` (staged) | 0.61 | 0.14 | 0.82 | 1.00 | 0.315 | all zeros 0.399 | Usable, weaker |
| `activated` (current) | 0.66 | 0.17 | 0.49 | 1.00 | 0.486 | all zeros 0.555 | Coarser than the ladder |
| `output_prediction` | 0.97 | -0.07 | 0.08 | 0.48 | 0.668 | all-zero outputs 0.848 | **Fails, a constant beats it** |
| `claim_truth` | 0.84 | -0.03 | 0.20 | 0.54 | 0.491 | random claims 0.503 | **Fails today, keep as a gauge** |
| `structural_validity` | 0.84 | 0.09 | 0.06 | 0.33 | 0.598 | copied netlist 0.621 | **Fails, copying beats it** |
| `verdict_truth` | 0.02 | 0.06 | 0.97 | 1.00 | 0.254 | not applicable | **No signal, every answer claims success** |
| `implication_truth` | none | none | none | none | 0.867 on rare cases | not measured | **Too rare to score** |
| `repair_gain` | not measured | | | | | | **Untested, retries are about 0.16% of attempts** |

The ladder is 1 for a detecting vector, 0.75 when activated and one input change detects, 0.5 when activated, 0.25 when one input change activates, and 0 otherwise. Its statistics come from the saved attempts. Its exploit figure is computed from the means of its two parts.

**Cost.** Audit 5 ms and simulation 9 ms per attempt. One input change search takes 176 ms per attempt on one core, about 7 minutes of CPU for a 2,304 attempt GDPO step, against a step of about 5.8 h. The Python simulator would supply the ladder's partial credit, and TetraMAX would still decide detection.

## The finding that changes the design: constant vectors

An all-zeros vector beats real attempts on every vector axis, and detects more often (0.242 against 0.209 on the subset used for exploits). The same holds on held-out problems:

| Validation, 300 problems | Level mean | Hard three levels |
|---|---|---|
| Uniform random, 1 vector | 0.147 | 0.008 |
| All zeros | 0.180 | 0.078 |
| All zeros or all ones (2 simulations) | 0.263 | 0.106 |
| Round 3, greedy | 0.243 | 0.200 |
| Round 4, greedy | 0.250 | 0.211 |
| Round 5, greedy | 0.227 | 0.194 |

So the model still leads about two to one on the hard levels, which is its real skill, but on the level mean two constant vectors match or beat it. Any reward on vectors, including detection itself, can pull a policy toward constants on problems where they work. Earlier work reached the same conclusion about ckpt-25, and the paper leaves this out by decision. This report is internal and does not change the paper.

**The first guard failed.** Scoring each attempt against the better of the two constant vectors for its problem drops the all-fail signal from 0.84 to 0.22 and "predicts solved" from 0.25 to 0.03. Constants score 0.58 to 0.94 on the ladder by level and detect 37% of the problems, so too few attempts remain above them (`constant_guard.json`). The ladder does not go into GDPO with this guard. Other options, such as lower weight on problems constants detect, are not scored yet.

**Best attempt makes constants stronger.** The staged reward now scores activation, propagation and controlling on the best attempt, by the user's decision of 2026-11-02, so a retry that does worse costs nothing. The cost is that one constant followed by the other keeps the better of the two. On the 250 fault benchmark (`best_attempt_check.py`, `best_of_k_real.py`):

| Stage | Real, 1 attempt | Real, 2 | Real, 3 | Real, 5 | All zeros then all ones, best | Same pair, mean as before |
|---|---|---|---|---|---|---|
| Activation | 0.507 | 0.654 | 0.721 | 0.789 | 0.836 | 0.482 |
| Propagation | 0.301 | 0.430 | 0.507 | 0.598 | 0.471 | 0.273 |
| Controlling | 0.360 | 0.505 | 0.586 | 0.675 | 0.558 | 0.316 |

Real attempts here are separate slots of the `pilot_start` model with no feedback between them, so they stand in for retries and are not a measurement of them. Under the mean the constant pair scored below one real attempt on all three stages. Under the best attempt it beats two real attempts on all three, needs three real attempts to be passed on propagation and controlling, and stays ahead on activation even at five. Very hard and random never detects are where the pair leads most. A guard against constants therefore matters more than it did, and it should be settled before GDPO leaves hold.

## Tool calls

The model makes exactly one call, and GDPO moved toward one call over its 14 steps. Script and full table: `analysis/tool_calls_20261102/`.

| Source | Attempts | Mean calls | Detected, 0 calls | Detected, 1 call | Detected, 2 or more | No detection |
|---|---|---|---|---|---|---|
| RFT round 0, greedy validation | 300 | 0.88 | 0.000 | 0.147 | 0.000 | 0.853 |
| RFT round 5, greedy validation | 300 | 0.92 | 0.007 | 0.220 | 0.000 | 0.773 |
| RFT model 4, sampled at $T=1$ | 20,480 | 0.86 | 0.019 | 0.186 | 0.000 | 0.795 |
| GDPO step 0, fixed set | 72 | 1.65 | 0.000 | 0.292 | 0.014 | 0.694 |
| GDPO step 2, fixed set | 72 | 1.07 | 0.000 | 0.375 | 0.042 | 0.583 |
| GDPO step 14, fixed set | 72 | 1.00 | 0.000 | 0.347 | 0.000 | 0.653 |

The GDPO rows use its own tool loop on the two-netlist set, so they compare only with each other. Only 14 of 100 steps ran. The cause of the drift is not established: a retry that detects still scores well above stopping on a failure, so the reward alone does not explain it, and retries rarely succeed today.

## Decisive bits and don't-cares

From greedy validation, round 0 to round 5. A care bit is activating when its flip takes the site off the excited value, and propagating when the site stays excited but the effect no longer reaches an output. A label is one test among many, so a match below 1 does not by itself mean a wrong bit.

| Level | Activating share of care bits | Propagating share | Match, activating | Match, propagating | Don't-cares left x |
|---|---|---|---|---|---|
| Easy | 0.44 | 0.11 | 0.66 to 0.54 | 0.49 to 0.41 | 0.81 to 0.55 |
| Medium | 0.46 | 0.41 | 0.80 to 0.63 | 0.43 to 0.58 | 0.64 to 0.57 |
| Hard | 0.30 | 0.59 | 0.67 to 0.75 | 0.60 to 0.70 | 0.83 to 0.58 |
| Very hard | 0.26 | 0.64 | 0.67 to 0.65 | 0.69 to 0.73 | 0.64 to 0.46 |
| Random never detects | 0.12 | 0.85 | 0.65 to 0.66 | 0.72 to 0.74 | 0.54 to 0.30 |
| All | 0.32 | 0.52 | 0.69 to 0.65 | 0.61 to 0.68 | 0.70 to 0.50 |

## Vectors before reasoning

At round 5, 41% of attempts state input values in the reasoning, about 60% of the way through on average, and in 68% of those the input values come before any internal net is discussed. Where the reasoning states a bit that the call also sets, the call repeats it 98% of the time. So the call follows what was written, and a vector written before the derivation decides the answer. The pilot's demonstrations derive first and name inputs only as a conclusion, activating and propagating inputs apart, with the remaining inputs left x.

## What the audit says about the reasoning now

- **Claims are coin flips.** Honest internal net claims are true 0.49 of the time, and random claims score 0.50. This did not move over rounds 1 to 5 (0.50, 0.47, 0.48, 0.51, 0.51). It matches the earlier result that activation behind gates is near or below random.
- **Gate steps are almost never stated.** `implication_truth` could not be scored in any group.
- **Failed answers claim success.** Every answer claims detection, so `verdict_truth` simply tracks detection (0.97).
- **Two figures differ.** The audit's `site_target_correct` is 0.86 and the earlier estimate was 99%. The audit also counts a site's value when the reasoning lists a vector or outputs, which the earlier narrower pattern skipped. So 0.86 is a lower bound, and the concept of activation is still mostly right.

## Keeping the model free to reason

- No axis above requires a format. The audit ignores any sentence it cannot check, neither rewarding nor penalizing it.
- No surviving axis pays for length (within group correlation at most 0.17 in size).
- Favouring any surviving axis does not narrow the search. The top half of a group by ladder parts keeps 0.87 to 0.89 distinct vectors, against 0.83 for all attempts.
- Demonstrations use several phrasings, and none is enforced at inference.

## Ways to guide reasoning, compared

| Option | What it would teach | Evidence | Cost | Risk | Order |
|---|---|---|---|---|---|
| Gate by gate demonstrations in SFT | The computation itself | Renderer: 1,000 of 1,000 rows pass the audit, corrupted traces score 0.50, 228 fewer tokens than today's gold at the median, same 94% fit | A short SFT pilot, about 30 steps at 35 min | The model copies the form and not the logic, which `claim_truth` would show | 1 |
| Retry rows with real negative results | Reading the simulator and trying again | Graph branch has a builder. Today 0.16% of attempts retry and all failed answers claim success | Port the builder to text SFT | Rows that fail on purpose may lower first attempts | 1, in the same pilot |
| Ladder in GDPO | Getting closer on failed attempts | Best offline score above | Small change to the staged reward | Constants, so the guard comes first | 2 |
| STaR hints that show a correct trace | Reasoning toward a known answer | Hinted rates in rounds 1 to 5 stay low, and 16 to 24% of rows are hinted | Hint mode change | The model learns to depend on a hint it will not get | 3 |

**An LLM judge** was considered and rejected as a reward. It can be gamed by wording and cannot be proved correct.

## Pilot, running on eng-tamale2 since 2026-11-02 13:46

30 SFT steps from round 4, the best round so far. Half of each step is 13,760 built rows (12,285 demonstrations and 1,475 retries, the model's own failed attempts followed by a reading of the real result), and the other half is ordinary rows. It is judged on the 250 fault benchmark with 16 slots at the start and at steps 10, 20 and 30, scored with the staged reward and the Python simulator. Launcher: `scripts/star/run_reasoning_pilot.sh`, log `runs/reasoning_pilot/pilot.log`, W&B `reasoning_pilot/benchmark` in `rft-training`.

| Check | Now | Go if |
|---|---|---|
| `claim_truth` on benchmark | about 0.49 | at least 0.70 |
| Activation with 1 or 2 gates to the site, all attempts | 0.431 and 0.340 | above random, 0.509 and 0.416 |
| Attempts with a checkable gate step | about 0 | at least 25% |
| Retry after a negative result | about 0.16% | at least 10% |
| Benchmark pass@1, hard three levels | pilot_start | no significant drop |
| Distinct vectors per problem | pilot_start | no drop above 10% |
| Don't-cares left x | about 0.50 | above pilot_start |

If these pass, the next proposal is GDPO on the staged reward, once a working guard against constants exists, with `claim_truth` at a small weight only if it survives the exploit suite. GDPO stays on hold until the user clears it.

## Limits of this evidence

- Offline correlations describe rollouts and do not show that training on an axis will work.
- Rollouts come from training circuits, sampled at $T=1$, the temperature parameter of generation.
- Exploit means use about a quarter of the groups, and the ladder's exploit figure is derived from its parts.
- A working constant guard, `repair_gain` and the pilot outcome are still open.
- Family splits and per level tables for the axes were not computed. `attempts.jsonl` allows them.
