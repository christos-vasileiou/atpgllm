# Reward shaping for ATPG-oriented language-model training

Scalar reward for GRPO-style training to generate **test patterns** for **stuck-at faults**. Structured fields (PI assignment, claimed good-machine POs, optional tool calls) are scored so an external **fault simulator** confirms detection at a PO.

**Impl:** `test_generation_grpo_reward` in `atpgllm/llm/reward_funcs.py`.  
**Train scalar:** `train_scalar_from_reward_components` — sum of components **excluding** `*_logonly`.  
**Formulas / weights:** [`TECHNICAL_REPORT_GRPO_REWARD.md`](TECHNICAL_REPORT_GRPO_REWARD.md).

---

## Ground truth

Simulator is ground truth. High reward ⇒ pattern distinguishes faulty vs fault-free at an observable PO—not that the prose sounds plausible.

---

## Key ideas

1. **Observable detection** — Main signal: some PO Good ≠ Bad (`reward_weight_fault_detected_po`, default 12).
2. **Fault-site consistency** — Stuck-at value at the fault net with Good ≠ Bad (`reward_weight_fault_site`, default 4).
3. **Undetected dampening** — If not detected, site / PO / PI / mention terms are scaled by `reward_undetected_shaping_scale` (default **0.2**) so “looks-right-but-doesn’t-detect” cannot outscore real detections.
4. **Consistency with simulation** — Declared PIs/POs vs gold good machine; perfect-match bonuses when scores ≈ 1.
5. **Process cues** — Tool JSON alignment, fault mention, light format shaping; pred-table term uses `w * max(0, acc − bias)`.

`*_logonly` accuracy flags are for logging/W&B only (not in the policy loss).

---

## Corner cases (default weights, order of magnitude)

Approximate **train** totals (log-only keys excluded). Optional format extractors assumed OK when “favorable.”

| Scenario | Approx. total | Rationale |
|----------|----------------|-----------|
| **1. Empty / invalid** | **≈ −1.5 to −2.5** | No vectors → no sim. Format: missing INPUT/OUTPUT/DETECTED (−0.5 each) plus optional extractor misses (−0.35 each). |
| **2. Well-formed, non-detecting** | **≈ 0–3** | Detection/site main terms ~0 (site at most `4×0.2` if site_ok). PO/PI/mention ×0.2; mild format / tool credit. |
| **3. Strong test, sloppy EXPECTED_OUTPUT** | **≈ 16–22** | Detection + site ≈ 16; PI near full (~3–4 with perfect bonus) ×1; PO near 0; tool/mention/table optional. Shows detection can dominate without a correct PO report. |
| **4. Near-ideal** | **≈ 30–36** | Detection+site 16; PO+PI with perfect bonuses (~5+1 + 3+1); tool ~1–1.5; mention ~1.5+1; pred_table up to ~2.2; sim_table 0.35; format ~0.5–1. Exact total depends on tool/table presence. |

---

## Learning pressure

Policy is pushed toward: PI assignments that **propagate** the fault; site activation; honest PI/PO reporting after detection; tool/template habits early on. Curriculum roughly: format → detection → consistency/tool alignment. Simulator is a hard filter on main credit.

---

## Limitations

- Simulator errors become ground truth.
- Scalar mixes functional success, reporting, and auxiliaries; `*_logonly` exclusion helps but remaining terms still share one sum.
- No explicit cost for pattern length / don't-cares; fault **mention** can be superficial.
- High reward still possible with wrong `EXPECTED_OUTPUT` if detection is strong (case 3).
- Format penalties are surface-form fragile.

---

## Reference

`atpgllm/llm/reward_funcs.py` · factory `atpgllm/training/reward_function_factory.py` · formulas [`TECHNICAL_REPORT_GRPO_REWARD.md`](TECHNICAL_REPORT_GRPO_REWARD.md).
