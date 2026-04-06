# `test_generation_grpo_reward` — design and formulas

This document describes the reward logic in [`atpgllm/llm/reward_funcs.py`](atpgllm/llm/reward_funcs.py) for GRPO / RL on **ATPG-style** completions: the model should produce a **test pattern** (primary inputs + claimed good-machine outputs) such that the **target stuck-at fault** is **detectable** at a primary output when the circuit is simulated.

The implementation is **`test_generation_grpo_reward`**; **`test_generation_reward`** is an alias that calls the same code.

---

## What is treated as ground truth?

**The fault simulator** is authoritative—not the model’s natural-language summary.

- With precompiled netlists (typical training path), the runner is **`fast_fault_sim`** (passed as `fault_sim` in kwargs when `lib_gate_funcs` is set).
- Otherwise the legacy **`fault_sim`** from `fault_coverage_calc` is used with a Verilog string netlist.

For each `(prompt, completion, netlist)` triple, the function:

1. Parses the **target fault** from the prompt via **`fault_fn`** (e.g. `sa0`, `sel1`).
2. Parses from the completion: **`INPUT_VECTOR`**, **`EXPECTED_OUTPUT`**, **`DETECTED_FAULTS`** via the supplied extractors.
3. Runs the simulator with the model’s vectors and fault string `"<fault> <net>"` (e.g. `"sa0 sel1"`).
4. Scores the result using the terms below.

If fault/net/netlist/vectors are missing, or the simulator errors, most terms stay at zero (with only format / `sim_table_bonus` possibly nonzero).

---

## Reward components (non-overlapping)

Each completion returns a **dictionary** of floats. Keys are chosen so that **`sum(dict.values())`** is a single scalar total for the trainer—**no double-counting** across the main terms.

| Key | What it rewards |
|-----|------------------|
| **`fault_detect_inpvector`** | **Detection + activation** (see formulas below). This is the main RL signal for “did this pattern actually test the fault?” |
| **`expected_output`** | How well **`EXPECTED_OUTPUT`** matches the **simulated good machine** on **primary outputs** (POs). |
| **`input_vector`** | How well **`INPUT_VECTOR`** matches the **simulated good machine** on **primary inputs** (PIs). |
| **`fault_simulation`** | Consistency of **`<tool_response>`** JSON (Good/Bad per net) with the **same** gold simulation on POs. |
| **`detected_faults`** | Whether **`DETECTED_FAULTS`** text **mentions** the target fault (e.g. `sa0 sel1`). |
| **`pred_simulation`** | If a simulation table exists in the completion, PO-level match of Good/Bad columns vs gold (scaled). |
| **`pred_vs_fault_sim_acc`** | Raw **0–1** accuracy for that PO Good/Bad comparison (useful for logging). |
| **`format`** | Light shaping: optional thinking / tool_call / tool_response extractors, plus penalties if INPUT/OUTPUT/DETECTED tags are missing. |
| **`sim_table_bonus`** | Small fixed bonus when **any** parseable simulation snapshot was found in the completion. |

Binary **`*_acc`** keys (`fault_detected_by_pred_input_vector_acc`, `expected_output_acc`, `input_vector_acc`, `detected_faults_acc`) are **0 or 1** flags for perfect or detected outcomes; they are included in the sum and are useful for metrics.

---

## Core formulas (default weights)

Let:

- **`detected`** = true iff **some PO** has Good Machine ≠ Bad Machine (fault **observable** at an output).
- **`site_ok`** = true iff at the **fault net**: Bad Machine equals the **stuck-at value** (`sa0` → 0, `sa1` → 1) **and** Good ≠ Bad (**activated** fault site).
- **`po_score`** ∈ [0, 1] = fraction of POs where the parsed **`EXPECTED_OUTPUT`** agrees with the **gold** good-machine simulation (nets listed in the completion must cover POs; missing keys hurt the score).
- **`pi_score`** ∈ [0, 1] = same for PIs vs **`INPUT_VECTOR`**.
- **`tool_bonus`** ∈ [0, 1.5] = average match of **tool JSON** Good/Bad PO entries vs gold, scaled by **1.5** inside `_tool_response_po_consistency_bonus`.
- **`mention`** ∈ {0, 1} = whether **`DETECTED_FAULTS`** contains the target fault phrase (normalized whitespace).

Default weights (overridable via kwargs):

| Kwarg | Default | Role |
|-------|---------|------|
| `reward_weight_fault_detected_po` | 12 | Weight for **`detected`** |
| `reward_weight_fault_site` | 4 | Weight for **`site_ok`** |
| `reward_weight_po_match` | 5 | Weight for **`po_score`** |
| `reward_weight_pi_match` | 3 | Weight for **`pi_score`** |
| `reward_weight_tool_json_bonus` | 1 | Multiplier on **`tool_bonus`** |
| `reward_weight_fault_mention` | 1.5 | Multiplier on **`mention`** |
| `reward_format_weight` | 0.12 | Per successful optional format piece |

**Main detection term:**

```text
fault_detect_inpvector = w_detect * I(detected) + w_site * I(site_ok)
```

**Vector and tool terms:**

```text
expected_output  = w_po  * po_score
input_vector     = w_pi  * pi_score
fault_simulation = w_tool_json * tool_bonus
detected_faults  = w_fault_mention * mention
```

**Pred vs gold table** (if a completion-side table aligns with gold PO rows):

```text
pred_vs_fault_sim_acc = mean over matching (PO, column) cells of I(pred == gold)
pred_simulation       = 3.0 * (pred_vs_fault_sim_acc - 0.5)
```

So perfect agreement yields `pred_simulation = +1.5`, random-level (~0.5) yields ~0.

---

## How the completion simulation table is parsed

**`_simulation_table_from_completion`** tries, in order:

1. **JSON inside `<tool_response>...</tool_response>`** with top-level `Good Machine` / `Bad Machine` dicts → built into a DataFrame (index = net names).
2. Otherwise **`simulation_fn(completion)`** → first string passed to **`convert_to_df`** (legacy markdown / fixed-width table format).

That order favors the **tool-use** workflow (JSON) used in your chat template.

---

## Format shaping (details)

- For each of **`thinking_fn`**, **`tool_call_fn`**, **`tool_response_fn`** (if not `None`): **+`reward_format_weight`** if the extractor returns a match, else **−0.35** (or **−0.35** on exception).
- **INPUT_VECTOR** present: **+format_weight**; missing: **−0.8**.
- **EXPECTED_OUTPUT** present: **+format_weight**; missing: **−0.8**.
- **DETECTED_FAULTS** present: **+format_weight**; missing: **−0.5**.

This keeps structure useful for training without dominating the simulator-based terms.

---

## Intended training incentive (summary)

1. **Maximize observable fault detection** at POs (`detected`) and correct **fault-site behavior** (`site_ok`).
2. **Align** the final **`INPUT_VECTOR`** / **`EXPECTED_OUTPUT`** strings with the **actual good-machine** result of the simulator (consistent ATPG answer).
3. **Align** the **tool** JSON with the same gold simulation when the model uses **`fault_simulation_tool`**.
4. **Nudge** the model to **name** the target fault in **`DETECTED_FAULTS`** and to follow the **template** (format keys).

Tune the **`reward_weight_*`** kwargs if PO detection should dominate even more, or if format penalties are too harsh for early training.

---

## Code reference

- Entry point: **`test_generation_grpo_reward`** in [`reward_funcs.py`](atpgllm/llm/reward_funcs.py).
- Helpers: **`_fault_detected_at_pos`**, **`_fault_site_activated`**, **`_po_prediction_score`**, **`_pi_assignment_score`**, **`_tool_response_po_consistency_bonus`**, **`_simulation_table_from_completion`**, **`_mentions_target_fault`**.
