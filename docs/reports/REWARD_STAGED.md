# Staged reward, profile `staged`

The existing profiles (`full`, `po`, `detection`) are unchanged. Set `TMAX_REWARD_PROFILE=staged` to score with contract `atpg-rewards-staged`. It works with the TetraMAX and the fast simulator. A checkpoint trained on the existing profiles refuses to resume under it, because `simulator_provenance.py` records the contract, the profile, the objectives and their weights.

## Why a new contract

Detection needs the fault to be activated and then propagated to a primary output. Until now only activation gave partial credit, so on hard faults every failed attempt looked alike and the only signal left was activation. The final answer was also never simulated, so a trajectory could detect with a call and still write a wrong summary at full detection. The staged reward adds two continuous stages and judges the answer the way deployment does.

## Objectives and initial weights

| Objective | Weight | Meaning |
|---|---|---|
| detection | 1.00 | 1/K for the first detecting attempt K, but only if the final answer vector also detects the fault |
| activation | 0.25 | 1 if any attempt excites the fault site, else 0 |
| propagation | 0.25 | how much of the fault's output cone carries the fault effect, on the best attempt |
| controlling | 0.15 | how many side inputs of the gates the effect reached are set to let it pass, on the best attempt |
| fidelity | 0.20 | 0 unless the answer detects, else the mean of the output agreement and the fault-claim validity |
| format | 0.05 | structure only |
| uniqueness | 0.10 | distinct vectors over attempts (unchanged) |

GDPO normalizes each objective inside the prompt group before weighting. A stage that is constant inside a group therefore adds nothing there, and the earlier stages carry the learning on hard faults.

## Propagation and controlling

Both are read from the Good and Bad machine table of one attempt, which holds every net. A D-net is a net whose two values are definite and differ. TetraMAX does not flag path or controlling nets. The dataset builder derives them from the same values and the netlist, and the reward does the same. With the TetraMAX backend the netlist object only holds ports, so the Verilog text is compiled once with the gate library into an `OptimizedNetlist` that supplies the structure (`atpgllm/llm/propagation_signals.py`).

1. **Propagation** is the number of D-nets divided by the number of nets in the fault's fanout cone that can still reach a primary output. Only cone nets present in the table count.
2. **Controlling** looks at every gate that reads a D-net. A side input of such a gate is correctly assigned when its value is 0 or 1 and some assignment of the gate's other side inputs lets the effect through. The gate's own truth table decides this, so complex cells such as AOI and OAI need no controlling-value table. A blocked 3-input AND with side inputs 1 and 0 scores 1/2. An x side input is not correct. The ratio is correct side inputs over all side inputs.
3. A fault that is not activated scores 0 on both, and an attempt that detects the fault scores 1 on both. A wide dead-end region therefore never outranks a detecting vector.
4. Each ratio r is sensitized as ((1 + r)^3 - 1) / 7. This is the cubic (1 + r)^3 with zero mapped to zero and the maximum to one. An attempt the simulator rejected counts as 0.

`propagation_depth_logonly` records how close the deepest D-net got to the nearest output. Within a prompt group the propagation denominator is constant, so propagation behaves like a D-net count. The depth log shows whether spread and progress disagree.

## Best attempt

Activation, propagation and controlling each take the highest score over the attempts, up to the first that detects. A mean would lower the score whenever a retry did worse, which punishes the search the tool loop is there to allow. Under the best attempt a retry can only help the stages. Detection stays 1/K, so a late success still costs more than an early one, and uniqueness still charges for a repeated vector. The other profiles keep the mean. On the evaluation path the reward sees one call, so the best and the mean are the same number there.

Two consequences need watching in GDPO. More calls can only raise the stage scores, so watch `attempts_logonly` for a policy that sprays vectors. A constant vector now keeps its full score when it is followed by a retry, so all zeros then all ones scores the better of the two on the stages. On the benchmark that pair beats two real attempts on all three stages (see `docs/reports/REASONING_AXES.md`).

## The answer

The final INPUT_VECTOR lists care bits and a missing input is x. The reward fills the full vector, reuses the result of a call with the same care bits, and otherwise simulates it. Writing `x` and omitting an input are the same. An answer that names a net that is not a primary input does not detect. EXPECTED_OUTPUT is compared with the good values of that simulation, and DETECTED_FAULTS is checked against its table. The target must be listed. Any other claim `saV net` needs a definite difference on that net with bad value V. This is evidence from the table and not a proof that the extra fault is detected, so listing only the target is the safe optimum. `claims_count_logonly` and `claims_valid_logonly` show whether the model enumerates path faults.

## Format

Per call: a closed thinking block with non-empty reasoning, and a tool call whose arguments are exactly the tool schema. Final turn: no pending call, thinking closed, the three fields present, and the target fault named in DETECTED_FAULTS. What the vectors contain (0, 1 or x) and what the calls found belong to the other objectives. Calls after a detection are no longer penalized by format, so watch `calls_after_detection_logonly`.

## Checks run

1. `tests/unit/test_reward_staged.py` covers the signals on hand-built and real ASAP7 cells, the answer simulation, the claims, the profile, the scalar fallback and the provenance. The whole unit suite passes.
2. On 200 validation rows of the v4 dataset, the D-nets computed from the dataset's TetraMAX snapshot equal the dataset's `fault_propagation_nets` (restricted to nets that reach an output) in 185 rows. In 13 of the other 15 the fault sits on a primary output, which is detected by definition. The remaining two are an escaped output name and a tie-cell net.
3. Real TetraMAX on eng-tamale2 with a private two-seat pool and 10 small circuits. The dataset vector scores 1 everywhere. With one care bit flipped, detection falls to 0.2, propagation averages 0.30 and activation 0.5, so failed attempts spread over the range. A detecting call followed by a wrong answer keeps propagation and controlling at 1 and drops detection and fidelity to the answer's own result. No simulator errors.
