# Reward shaping for ATPG-oriented language-model training

This note summarizes the scalar reward used to train a model (e.g., with GRPO) to generate **test patterns** for **stuck-at faults** on gate-level netlists. The model is encouraged to output structured fields (primary-input assignment, claimed good-machine outputs, optional tool calls) such that an external **fault simulator** confirms that the pattern **detects** the requested fault at a primary output.

Implementation: `test_generation_grpo_reward` in `atpgllm/llm/reward_funcs.py`. The trainer typically uses the **sum** of all reported sub-scores as the per-sample return. Default weights below refer to the built-in hyperparameters unless overridden.

---

## Ground truth and objective

The **simulator** is the source of truth: given the model’s declared input pattern, expected outputs, target fault, and netlist, it computes good-machine and faulty-machine logic values. The learning objective is **functional**—a high reward means the pattern actually distinguishes faulty from fault-free behavior at an observable output—not that the prose merely sounds plausible.

---

## Key reward components (conceptual)

Four ideas dominate the design:

1. **Observable detection** — At least one primary output must differ between good and faulty simulation. This is the main signal that the pattern is a valid test for the target fault.

2. **Fault-site consistency** — The faulty machine should exhibit the stuck-at value on the fault net, and that net should differ from the good machine (fault is excited, not masked).

3. **Consistency with simulation** — The model’s **declared** primary inputs and primary outputs are compared to the **simulated** good machine. This penalizes answers that detect the fault but misreport the nominal response (a common failure mode when the summary is wrong but the tool call was right).

4. **Process and reporting cues** — Smaller terms reward use of the intended template (e.g., tool response alignment, mention of the target fault in a dedicated field, light format shaping). These shape the **interface** the model uses without replacing simulator-based credit.

Auxiliary terms compare any **embedded simulation table** in the completion (e.g., JSON from a tool block) to the same gold simulation, and expose binary “perfect match” flags for logging; they also enter the summed scalar unless the training code filters them.

---

## Illustrative corner cases (default weights, order of magnitude)

The following are **approximate** totals when sub-scores are summed as in the reference implementation. Optional extractors (thinking / tool call / tool response) are assumed present and successful in favorable cases, and absent or failing only where noted. Values round to the nearest whole or half unit for readability.

| Scenario | Approx. total | Rationale |
|----------|----------------|-----------|
| **1. Empty or invalid completion** | **≈ −3** | No extractable input/output vectors: the simulator is not run. Only **format** penalties apply (missing required fields; optional −1 to −2 if chain-of-thought or tool markers are also missing). Substantially below zero discourages degenerate outputs. |
| **2. Well-formed but non-detecting pattern** | **≈ 0–2** | Vectors parse and simulation runs, but **no** primary output differs between good and faulty machines, and the fault site is not activated as required. **Detection** and **site** terms are zero; small credit may remain from partial PI/PO agreement with the good machine and mild positive **format**. |
| **3. Strong test pattern, inconsistent narrative** | **≈ 19–22** | **Detection** and **site** terms fire (**≈ 12 + 4 = 16** with defaults). **Input** side can be near full credit if the pattern matches simulated PIs (**up to ~3**), while **expected output** can be **~0** if the model’s stated good-machine outputs disagree with simulation. Tool and table bonuses may add **~0–2**. This case shows that the reward **does not** require a correct verbal summary to grant large credit for a valid test. |
| **4. Near-ideal completion** | **≈ 33–36** | Full **detection** and **site** (**16**), full **PO** and **PI** consistency (**5 + 3**), strong **tool** alignment (**~1–2**), **fault mention** (**~1.5**), high **pred** vs gold table match (**~2.5** including accuracy sub-scores), **sim_table** bonus (**~0.35**), and positive **format** (**~1**). Exact total depends on how many auxiliary counters are included in the sum. |

These four cases bracket **failure to engage the task**, **engagement without detection**, **detection with sloppy reporting**, and **aligned detection plus reporting**.

---

## How learning pressure increases with this reward

Under policy-gradient-style training, trajectories with **higher** summed return are reinforced. The gradient flow implied by this design pushes the policy toward:

- Proposing **primary input assignments** that actually **propagate** the fault to an output, because that unlocks the largest stepwise gain (the detection-weighted term).

- Satisfying **fault-site** conditions so that credit is not limited to coincidental PO differences unrelated to the target fault.

- Aligning **declared** vectors with **simulated** good-machine values, which raises the return even after detection is achieved—rewarding **honest** reporting of the nominal response.

- Adopting the **tool-and-template** habit that yields stable parsing and optional consistency bonuses, which matters early in training when simulation might not yet succeed.

Thus improvement is not a single leap: the model can first learn **format and parseable vectors**, then **detection**, then **tighter consistency** and **tool trace** alignment, with the simulator acting as a **hard filter** on the main credit.

---

## Limitations and disadvantages

- **Simulator trust** — Errors, modeling mismatches, or overly coarse fault models in the simulator are treated as ground truth; the policy may be rewarded or penalized incorrectly.

- **Credit assignment** — The scalar mixes **functional success**, **reporting accuracy**, and **auxiliary metrics** (including some 0/1 logging-style terms) into one sum. Without per-term baselines or masking, the trainer may overweight incidental components or add variance to the gradient.

- **Reward hacking** — A pattern that maximizes **detection** might still be **non-minimal** or **hard to apply** in practice; there is no explicit cost for pattern length or don't-care density. **Mention** of the fault in text can be satisfied superficially.

- **Partial credit asymmetry** — High reward can occur with **incorrect EXPECTED_OUTPUT** if detection is strong (Case 3), which may or may not match the desired curriculum if the publication goal is **fully** correct ATPG **reports**, not only **valid tests**.

- **Parsing fragility** — Extractors and templates tie reward to **surface form**; valid semantically equivalent outputs that miss a tag can receive harsh **format** penalties despite a correct underlying pattern.

---

## Reference

Full formulas and keyword arguments: implementation in `atpgllm/llm/reward_funcs.py` (`test_generation_grpo_reward`). A compact technical appendix for the codebase lives in `test_generation_grpo_reward.md` in this directory.
