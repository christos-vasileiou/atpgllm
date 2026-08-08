# `test_generation_grpo_reward` — design and formulas

Reward logic in [`atpgllm/llm/reward_funcs.py`](../../atpgllm/llm/reward_funcs.py) for GRPO on ATPG-style completions: produce a test pattern (PIs + claimed good-machine POs) such that the target stuck-at fault is observable at a PO under fault simulation.

Entry: **`test_generation_grpo_reward`**. Alias: **`test_generation_reward`** (same code).

Wiring: [`atpgllm/training/reward_function_factory.py`](../../atpgllm/training/reward_function_factory.py) → `RewardFunctionFactory.create_reward_function(return_component_dicts=True)` (training) / scalars via `train_scalar_from_reward_components`.

---

## Ground truth

The **fault simulator** is authoritative—not model prose.

- Production path: factory always passes `lib_gate_funcs` from package `sim_config.json` (`atpgllm/training/data/sim_config.json`, overridable via `SIM_CONFIG`) and `fault_sim=resolve_fault_sim_runner()` → typically **`fast_fault_sim`** (`data_preprocessing/fault_sim.py`). Backend: `FAULT_SIM_BACKEND=fast|tetramax|hybrid`.
- Legacy: if `lib_gate_funcs is None`, uses `fault_coverage_calc.fault_sim` on a Verilog string (not used by the factory).

Per `(prompt, completion, netlist)`:

1. Parse target fault via `fault_fn` (e.g. `sa0`, `sel1`).
2. Parse `INPUT_VECTOR`, `EXPECTED_OUTPUT`, `DETECTED_FAULTS` from the completion.
3. Run simulator with vectors and fault string `"<fault> <net>"`.
4. Score with the terms below.

Missing fault/net/netlist/vectors or sim errors → most terms stay 0 (format / `sim_table_bonus` may still fire).

---

## Training scalar vs log-only

Returns a **dict of floats** per completion.

- **Train scalar** = `train_scalar_from_reward_components(dict)` = sum of all values **except** keys ending in **`_logonly`**.
- Trainer (`DualAdapterGRPOTrainer`) and factory both use that helper. Do **not** assume `sum(dict.values())` for the policy loss—`*_acc_logonly` are dashboard-only.

| Train key | Role |
|-----------|------|
| **`fault_detect_inpvector`** | Binary PO detection + site activation (shaped when undetected) |
| **`expected_output`** | PO match of `EXPECTED_OUTPUT` vs gold good machine (+ perfect bonus) |
| **`input_vector`** | PI match of `INPUT_VECTOR` vs gold (+ perfect bonus) |
| **`fault_simulation`** | `<tool_response>` JSON Good/Bad vs gold POs |
| **`detected_faults`** | Target fault mentioned in `DETECTED_FAULTS` (+ perfect bonus) |
| **`pred_simulation`** | Completion sim-table vs gold POs (biased accuracy) |
| **`format`** | Template shaping |
| **`sim_table_bonus`** | Fixed +0.35 if any parseable sim table in completion |

Log-only (excluded from loss): `fault_detected_by_pred_input_vector_acc_logonly`, `expected_output_acc_logonly`, `input_vector_acc_logonly`, `detected_faults_acc_logonly`, `pred_vs_fault_sim_acc_logonly`.

---

## Core formulas (defaults)

Let:

- **`detected`** = some PO has Good ≠ Bad (or TetraMAX detection flag when that backend reports it).
- **`site_ok`** = at fault net: Bad == stuck-at and Good ≠ Bad.
- **`shaping`** = `1.0` if `detected` else **`reward_undetected_shaping_scale`** (default **0.2**). Dampens site / PO / PI / mention credit when the pattern does not detect.
- **`po_score`**, **`pi_score`** ∈ [0, 1] — fraction of POs/PIs where parsed vectors match gold good machine.
- **`tool_bonus`** ∈ [0, 1.5] — PO Good/Bad match of tool JSON vs gold, scaled ×1.5 in `_tool_response_po_consistency_bonus`.
- **`mention`** ∈ {0, 1} — `DETECTED_FAULTS` contains target fault phrase.

### Active kwargs

| Kwarg | Default | Role |
|-------|---------|------|
| `reward_weight_fault_detected_po` | 12 | `I(detected)` |
| `reward_weight_fault_site` | 4 | `I(site_ok)` (× shaping) |
| `reward_undetected_shaping_scale` | 0.2 | Multiplier when not detected |
| `reward_weight_po_match` | 5 | × `po_score` |
| `reward_weight_pi_match` | 3 | × `pi_score` |
| `reward_perfect_po_bonus` | 1 | Extra if `po_score ≥ 0.999` |
| `reward_perfect_pi_bonus` | 1 | Extra if `pi_score ≥ 0.999` |
| `reward_perfect_mention_bonus` | 1 | Extra if mention |
| `reward_weight_tool_json_bonus` | 1 | × `tool_bonus` |
| `reward_weight_fault_mention` | 1.5 | × `mention` |
| `reward_weight_pred_table` | 2.75 | Pred-table train term |
| `reward_pred_table_acc_bias` | 0.2 | Floor for pred-table credit |
| `reward_format_weight` | 0.12 | Per successful format piece |

**Accepted but unused in the current body** (parsed, no effect): `reward_weight_po_obs`, `reward_site_partial_credit`. Do not rely on them until wired in.

```text
fault_detect_inpvector = w_detect * I(detected) + w_site * I(site_ok) * shaping

expected_output  = (w_po * po_score + perfect_po_if_full) * shaping
input_vector     = (w_pi * pi_score + perfect_pi_if_full) * shaping
fault_simulation = w_tool_json * tool_bonus          # not shaped
detected_faults  = (w_fault_mention * mention + perfect_mention_if) * shaping

pred_simulation  = w_pred_table * max(0, acc - pred_table_acc_bias)   # if table present
# pred_vs_fault_sim_acc_logonly = acc
```

---

## Simulation table parsing

`_simulation_table_from_completion`:

1. JSON inside `<tool_response>...</tool_response>` with Good/Bad Machine → DataFrame.
2. Else `simulation_fn(completion)` → `convert_to_df` (legacy table string).

---

## Format shaping

- Each of `thinking_fn` / `tool_call_fn` / `tool_response_fn` (if set): **+format_weight** on hit, else **−0.35**.
- `INPUT_VECTOR` / `EXPECTED_OUTPUT` / `DETECTED_FAULTS` present: **+format_weight** each; missing: **−0.5** each.

---

## Intended incentive

1. Maximize observable detection (`detected`); site credit and reporting are damped without it.
2. Align declared PI/PO strings with gold good machine (perfect bonuses when exact).
3. Align tool JSON with gold when using `fault_simulation_tool`.
4. Name the target fault; keep template parseable.

---

## Code reference

- Reward: `atpgllm/llm/reward_funcs.py` — `test_generation_grpo_reward`, `train_scalar_from_reward_components`
- Helpers: `_fault_detected_at_pos`, `_fault_site_activated`, `_po_prediction_score`, `_pi_assignment_score`, `_tool_response_po_consistency_bonus`, `_simulation_table_from_completion`, `_mentions_target_fault`
- Factory: `atpgllm/training/reward_function_factory.py`
- Train CLI: `scripts/train/training_code.py` (uses `return_component_dicts=True`)
