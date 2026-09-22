# TypeSafe-inspired ATPG: experiment analysis and preregistration

Prepared 16 September 2026. Companion:
[research and implementation design](TYPESAFE_ATPG_DESIGN_REPORT.md).
Primary scope: combinational single stuck-at faults, complete binary PI vectors,
and canonical observable POs. This document specifies experiments to run.
**No new model-training or ATPG performance experiments have been performed.**

Use the existing
[Language-of-test evaluation protocol](LANGUAGE_OF_TEST_EVALUATION_PROTOCOL.md)
for semantic definitions, independent verification, denominators and coverage
accounting. This plan adds architecture attribution and probability evaluation.

## 1. Questions and falsifiable hypotheses

| ID | Hypothesis | Evidence required | Outcome that rejects or limits it |
| --- | --- | --- | --- |
| H1 | A typed decision model can prioritize detecting candidates efficiently | Better independently verified success at fixed total cost than random ordering and simple heuristics | Scorer overhead exceeds saved simulation/search work, or ranking improvement disappears on held-out families |
| H2 | Joint masked denoising helps vector generation beyond independent bit heads | Better detection/coverage at matched wall time and training budget, especially on dependence-heavy faults | Equal-cost AR vectors or independent heads dominate |
| H3 | Graph conditioning improves generalization | Benefit survives representation, capacity, data and fault-conditioning controls | Gain disappears after removing label-bearing inputs, or under design-family holdout |
| H4 | A success head offers actionable calibrated probabilities | Good NLL/Brier and useful risk–coverage curves on both all candidates and selected outputs | Apparent calibration comes from constant probabilities, selection bias, or easy-fault imbalance |
| H5 | Verified-data iteration or diffusion-native RL adds value | Gain over SFT/denoising survives equal candidate-generation and verifier budgets | More simulator work or changed test distribution explains the gain |
| H6 | Fewer sampling rounds or distillation gives a better deployment tradeoff | Quality retention and diversity at lower measured end-to-end latency | Hard-fault success or coverage collapses despite similar bit accuracy |
| H7 | A narrow ATPG interface can extend to unseen typed tasks | Held-out task/schema success and calibration after separate broad training | Results restricted to familiar ATPG labels and schemas |

H1–H6 are about our proposed models. None tests whether Jev secretly uses the same
architecture. H7 is a separate programme, not a required early ATPG milestone.

## 2. What is already known and what remains unmeasured

| Item | Evidence status | Consequence |
| --- | --- | --- |
| Causal SFT, LoRA loading, simulator rewards, causal search adapters | Source inspected | Retain as baselines; new model families need explicit trainers/backends |
| Fault-conditioned graph encoder and Q-Former | Source inspected | Reuse is feasible at interface level; checkpoint quality not measured here |
| Graph auxiliary labels distinct from encoder features | Source inspected | Add regression checks so future changes preserve the boundary |
| Family-level train/calibration/test cleanliness | Not measured dataset-wide | Audit manifests and ancestors before training |
| Hardware capacity and throughput | Unavailable in this session; NVIDIA driver query failed | Profile on the selected training host; no asserted fit or runtime |
| Jev implementation and private RLCD recipe | Not disclosed in reviewed material | Treat architecture rankings as hypotheses |
| Model accuracy, speedups, coverage, calibration | Not measured | Leave result cells empty; do not turn design targets into findings |

Analytical dependence example: a distribution uniform on `01` and `10` has
perfectly plausible 0.5 bit marginals, but an independent sampler detects with
probability 0.5. For eight independent opposite-bit pairs, this becomes
`1/256 = 0.00390625`. This calculation explains a test requirement; it is not a
measured neural-model result.

## 3. Data construction and split protocol

1. Inventory actual supported circuit families, gate libraries, sizes/depths,
   primary-port counts and fault types. Record parse/verification exclusions.
2. Assign every transformation, resynthesis, renamed variant and fault record to
   its source family before splitting. Preserve known ancestral relationships.
3. Freeze train, model-selection validation, calibration and final-test manifests.
   Use grouped development folds if there are too few families for four stable
   partitions. Never move a test design into training because it is difficult.
4. Construct the fault universe independently of existing successful ATPG rows.
   Define stems/branches, collapsing and fault weights. Begin with the exact
   supported fault semantics; do not imply branch-fault coverage if not implemented.
5. Independently verify teacher vectors and good outputs. Keep candidate negatives,
   multiple distinct successes, failed attempts, unresolved faults and proof-backed
   untestability statuses. A timeout does not supply an untestability label.
6. Save the candidate source and sampling probabilities. Build a development
   calibration mixture matching the intended deployment proposer and selector.
7. Use a fixed, bounded random-detectability probe to stratify faults; record its
   simulator cost. Do not silently remove faults on which the probe finds nothing.

Training target sets must not expose valid vectors, good/faulty snapshots,
propagation witnesses or answer-derived captions to a no-feedback inference path.
Solver-derived labels are allowed as supervision on training families. During
testing, any computed solver information supplied to the model is an online tool
and must be charged and shared according to the declared comparison policy.

Sampling each circuit equally and then each fault within it is the initial
training policy. Run a fault-weighted alternative only as an explicit ablation.
Deduplicate training copies with provenance retained. Do not deduplicate the
independent evaluation sample pool before calculating pass@k.

## 4. Baseline and experiment matrix

Each row produces saved candidates, verification outcomes and resource traces.
Start with small smoke runs; use three independent training seeds for finalists,
with inference seeds shared by problem identity where meaningful. Equal seeds
do not imply equivalent samples across different model architectures.

| ID | Method | Main purpose / controls |
| --- | --- | --- |
| B0 | Uniform random PI vectors and validation-tuned weighted random vectors | Lower bound and detectability control; no expected-output guessing handicap |
| B1 | Existing base causal LM, SFT and SFT+GRPO | Current capability and training contribution; same problem manifest |
| B2 | Compact typed AR vector model | Match proposed denoiser graph input, width/parameter band and binary output contract; isolate AR versus diffusion |
| B3 | Existing causal LM with shortest valid vector format and constrained serialization where supported | Control for removing rationale, punctuation and generated net names |
| B4 | Classical ATPG/SAT, plus random-seeded cleanup if applicable | Practical reference; include startup, license waits, seed cost and equal final coverage |
| A0 | Independent PI Bernoulli heads on the proposed graph encoder | Dependence failure control; same training positives |
| A1 | Candidate success scorer on a fixed saved candidate pool | Isolate ranking without changing the proposer or candidate count |
| A2 | Candidate scorer plus live candidate production | Test real cost including generation, scoring and simulation |
| D0 | Typed masked denoiser with fixed random unmasking | Main joint-proposal experiment |
| D1 | D0 with confidence-based commitment / different step counts | Sampler tradeoff, not an architecture change |
| D2 | D0 plus verified successful-data iteration | Isolate data improvement before RL |
| D3 | D0 plus a specified diffusion-native RL algorithm | Optional; same rollout/verifier budget as D2 |
| D4 | Distilled few-step or latent decoder | Test reduced-cost joint sampling and diversity retention |
| L0 | Native pretrained diffusion model adapted to ATPG | Optional language-capable reference, separated by model size/training budget |

Do not compare a small graph denoiser only to a large LM generating long reasoning
and attribute the entire difference to diffusion. B2 and B3 are essential. If
parameter/data equality is impossible, report both resource-matched comparisons
and useful deployment frontiers, clearly labelling the distinction.

For A1, evaluate saved candidates exhaustively offline to measure ranking quality,
but do not expose those labels to the selector. In the online simulation, charge
only the declared scored/verified selection sequence; report the offline audit
cost separately. Compare to directly simulating the same pool. For cheap local
fault simulation, a neural scorer may have no economic role even if accurate.

## 5. Minimal staged campaign

| Stage | Runs | Gate before spending more |
| --- | --- | --- |
| S0: correctness | Tiny exhaustive circuits; random and simple hand-designed models; corruption/sampler checks | No unresolved semantic discrepancies or label leakage |
| S1: data pilot | Up to 100k verified candidate records, constrained by available families; A0/A1/B0 | Stable learnability, useful ranking and measured simulation/scoring cost |
| S2: generation pilot | B2/D0 at comparable size and data, plus B1/B3 snapshots | Some equal-cost improvement or a defensible Pareto tradeoff |
| S3: ablation | Best two model families; graph/text, conditioning, round-count and diversity controls | Improvement survives family holdout and equal-budget controls |
| S4: refinement | D2 first; D3 only if warranted; then D4 | Benefit exceeds uncertainty and added training/serving cost |
| S5: locked evaluation | Finalists on untouched in-distribution and OOD test families | Report all outcomes, including failures of the proposed model |

Initial development grids: sampling rounds `{1,4,8,16,32}`, candidate pool sizes
`{1,8,32}`, and pass@k at `{1,4,8}` when at least eight independent slots are
generated. Clip impractical schedules using a documented rule based on free-slot
count. For development, use identical simulator-execution caps `{1,8,32}`;
set absolute wall-time budgets after the hardware pilot and freeze them before
opening the final test.

Every candidate selected after a shared ranking/search procedure belongs to that
procedure. For pass@k across searches, reset the entire search k times; correlated
members of one pool are not independent attempts. Report oracle-best-in-pool as a
separate upper bound, never as the returned policy's quality.

The plan proposes a deployment promotion rule: lower 95% confidence bound on the
paired macro success difference is at least -0.02 versus the selected strongest
baseline, while the speed improvement at the declared quality target is at least
2× with uncertainty reported. These are engineering defaults to finalize on
development data, not properties of TypeSafe or promises of our model. Alternatively
promote a model that demonstrably improves quality at equal cost. Predeclare which
branch is primary; do not switch criteria after seeing the test.

## 6. Metrics and interpretation

### 6.1 Semantic performance

Retain the existing protocol's predicates:

* Vx: complete, unambiguous, legal binary assignment.
* D: valid stimulus detects the target fault.
* S: proposed expected outputs exactly match the good circuit on required POs.
* U: S and D; a usable detecting test.

For vector-only systems, D is the primary raw-model endpoint. For a deployable
pipeline, provide the same deterministic good-output generator to every eligible
method, charge it, and report verified U. Retain raw S/U for models that predict
outputs. Separate model-produced outputs from tool-supplied/replicated outputs.

Report D-pass@1, U-pass@1, pass@k, assignment validity, full-fault-set coverage,
coverage versus wall time/simulator executions/applied patterns, and compacted
pattern count at equal coverage. Use the same fault universe and compactor. Macro
average over circuits and report the pooled micro result separately.

Keep witnessed testable, proven untestable and unresolved fault counts visible.
Success on training-data ATPG targets alone is not full-circuit fault coverage.
All invalid, exhausted and failed completion slots remain unsuccessful in the
end-to-end denominator; infrastructure errors also have a separate status.

### 6.2 Probabilities

For specified candidate outcomes y and predicted detection probability p, record:

```text
Brier = mean((p-y)^2)
NLL   = mean(-y*log(p) - (1-y)*log(1-p))
```

Use documented numerical clipping only for metric computation. Retain unclipped
probabilities and clipping counts. Compare against the development-estimated
base-rate predictor; add AUROC and especially precision–recall measures when
success is rare. Calibration does not imply useful ranking.

Plot reliability diagrams using predeclared bins, with sample counts and clustered
uncertainty. Report binning sensitivity; ECE alone is not a proper optimization
target and can conceal poor discrimination. Evaluate by circuit family, size,
difficulty, candidate source, sampler rounds and selected-versus-all candidates.

For an acceptance threshold tau, report acceptance fraction, failure rate among
accepted candidates, and the cost/quality of fallback on rejected cases. A high
threshold that accepts almost nothing is not an automation success. Keep the
simulator verification policy explicit; calibration is not a substitute for
verification when claiming individually sound tests.

Fit calibrators only on development predictions from frozen model versions.
If a model ranks candidates and returns only the top candidate, calibrate/evaluate
that selected distribution too. Bit probabilities and confidence summaries are
diagnostics, not interchangeable with whole-vector detection probabilities.

### 6.3 Efficiency and diversity

Save per-request timestamps and measure p50/p95 latency; do not substitute batch
time divided by batch size for individual latency. Separately report batched
throughput, GPU/CPU time, peak memory, parsing, graph construction, cache hits,
simulator startup and fallback. Include all failed attempts in cost.

Count model calls, neural evaluations, refinement rounds, processed slots/tokens,
candidate vectors, simulator requests and executions. An API that emits numeric
arrays still consumes compute. Separate paid-service price from measured hardware
efficiency. Report the actual precision and batch size.

For diversity, count unique complete vectors, distinct verified detecting vectors,
incremental fault coverage and repeated detections per fault. Marginal entropy or
varied textual wording is not evidence of useful joint diversity. Report repeated
identical outputs; do not hide them with pre-metric deduplication.

## 7. Ablations and correctness tests

The following are meaningful implementation checks to add when implementing the
models. They are not tests executed by this documentation change.

| Check | Required behaviour / failure exposed |
| --- | --- |
| Exhaustive tiny circuits with independently implemented good/faulty logic | Every generated vector's D and good output agree with the reference; audit negative cases as well as successes |
| Corruption estimator on tiny vectors | Compare Monte Carlo loss with exact enumeration for a small fixed-t objective; verify zero-mask and padding handling |
| Noncausal attention | A masked slot can depend on visible slots on both sides; padding/other examples cannot leak; optimized kernel agrees with eager mode |
| Independent question semantics | Unrelated question reordering/addition does not change isolated question output, apart from documented numerical tolerance |
| Graph-label boundary | Deleting/permuting snapshot and rationale labels leaves inference features and outputs unchanged |
| Fault-conditioning control | Correct f versus informative wrong f' changes target-specific performance; shared easy detecting vectors are accounted for |
| PI/PO mapping | Renaming and declaration permutations preserve the mapped circuit, constraints and fault; unused ports and direct wires remain represented |
| Dependence construction | Opposite-bit/parity cases expose independent-head limitations; compare learned joint distribution or validity, not only marginal accuracy |
| Seed and resume | Restored optimizer, data position, corruption RNG and scheduler reproduce the next update within backend tolerance |
| Serialization | Every successful output has exactly the declared ports and binary values; NaN, invalid masks and unfinished slots produce explicit errors |
| Calibration isolation | Calibrator never reads final-test labels; deploying a changed proposer/selector invalidates stale calibration metadata |
| Cache validity | Different faults/candidates/constraints/checkpoints cannot share incompatible cached activations |
| RL estimator, if used | Tiny-action-space gradient or estimator check against a tractable reference; finite rewards and correct old-policy tracking |

Additional scientific ablations: graph versus text versus fused state; candidate
scorer versus simple controllability/observability features; one-shot versus
iterative decoding; exact binary vector output versus full rationale output;
multiple diverse teachers versus one canonical solution; calibrated versus raw
logits; successful-data iteration versus RL at equal verifier work; fixed versus
confidence-driven schedules; matched-context cached versus uncached serving.

Do not conflate the current structural depth/degree features with exact SCOAP
controllability/observability. Implement and verify the actual heuristic if using
it as a named baseline.

## 8. Statistical analysis

Use the circuit/design family as the primary resampling unit. Faults and vectors
within one family are correlated. Report paired differences on common manifests;
use a family-level paired bootstrap (proposed default: 2,000 draws) and disclose
small-family instability. Across training seeds, report individual results and
their spread, rather than pretending every vector is a new trained-model replicate.

Freeze one primary endpoint and cost budget before final testing. Treat other
round counts, metrics and ablations as secondary; avoid searching the test set for
the most favourable operating point. Use pilot variance to determine whether the
available number of independent families can resolve the desired quality margin.
There is no justified universal minimum sample count from the information here.

If no verifier disagreements occur, report the audit size and an uncertainty bound;
zero observed failures does not prove zero error. An independent-trial binomial
bound applies only when that independence assumption is justified. For clustered
faults, use an analysis respecting the family structure.

For a wall-time budget B, report success@B and capped first-success time
`mean(min(T_i,B))`, with non-success T_i treated as infinity, plus actual total
runtime. Show complete cost curves from logged events; successful-case latency
alone can conceal failures.

## 9. Result templates

All cells below are intentionally unmeasured. Fill only from saved run artifacts.

| Model / seed | Families / faults | D-pass@1 | Verified U-pass@1 | Macro coverage at B | NLL / Brier | p50 / p95 latency | GPU h / simulator executions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Best model-free baseline | Not run | — | — | — | N/A unless probabilistic | — | — |
| AR typed-vector baseline | Not run | — | — | — | — if success head provided | — | — |
| Candidate scorer pipeline | Not run | — | — | — | — | — | — |
| Typed masked denoiser | Not run | — | — | — | — if success head provided | — | — |
| Selected refined model | Not run | — | — | — | — | — | — |

| Required failure report | Value |
| --- | --- |
| Unsupported/excluded circuits with reasons | Not measured |
| Invalid, incomplete, exhausted and infrastructure-failed attempts | Not measured |
| Internal versus independent verifier confusion counts | Not measured |
| Testable / proven untestable / unresolved faults | Not measured |
| Selected-candidate calibration and acceptance–risk curve | Not measured |
| OOD family/size/depth degradation | Not measured |
| Incremental coverage over strongest baseline, including extra cost | Not measured |

## 10. Run contract and reproducibility

Each run must save:

```text
identity: run_id, source_commit, local_diff_hash, environment_lock,
          architecture_config, checkpoint_revision, tokenizer_or_slot_schema
data: split_manifest_hash, family_ids, circuit_and_library_hashes,
      fault_manifest, constraints, PI/PO maps, teacher provenance
training: seed, optimizer, schedule, precision, corruption distribution,
          free-slot normalization, batch policy, processed examples/slots,
          checkpoint/resume state, verifier-work ledger
inference: proposal_and_selector_versions, candidate_count, sampler_rounds,
           unmasking_policy, RNG_seeds, batch_size, cache_policy, calibration_hash
events: raw_candidate, candidate_origin, predicted_probabilities, selection,
        good/faulty outputs, D/S/U, independent_verification_status,
        timestamps, resource counters, final status and stop reason
```

Budget online model work, verification and fallback jointly. A training teacher's
simulation budget also belongs in the training-cost ledger. Keep external audit
cost separate but publish it. Derived tables should be reproducible from saved
records without rerunning either neural inference or the simulator.

## 11. Interpretation and next actions

If A1 succeeds but D0 fails, a useful deliverable is a calibrated candidate-ranking
component; do not rename it an autonomous generator. If D0 only beats verbose LMs,
the result may be structured-output efficiency. If D0 beats B2 on coupled hard
faults at equal cost, that supports joint non-autoregressive generation for this
domain. If calibration fails under OOD conditions, keep verification/fallback and
limit confidence claims to the evaluated distribution.

The next implementation action is P0/S0: build the typed problem/result contract,
freeze the family/fault manifest, and validate the independent verifier on tiny
circuits. Then implement the candidate scorer and small denoiser as separate paths
against the common harness. Do not begin frontier-scale pretraining before these
comparisons show a reproducible benefit.

## Appendix: limitations and artifact status

Public interface observations cannot uniquely identify Jev. No Jev API measurements
or model experiments were made. Proposed margins, sizes, seeds and budgets are
development defaults to freeze before final evaluation. The local evaluation
protocol is itself a proposal; references to it are methodological alignment, not
claims that all its checks already exist in code.

This is a Markdown experiment plan with empty result tables, not a completed
experiment report or rendered template. Template production stopped because the
selected [Experiment Analysis skill](/home/eng/c/cxv200006/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-experiment-analysis/SKILL.md)
requires: “If no such capability can be identified and read, say it is unavailable
and stop; do not recreate or install it.” No compatible advertised document
capability or connected session was available. The retained template was unchanged.
