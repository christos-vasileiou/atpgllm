**Language-of-test evaluation: proposed protocol for a TCAD submission**

Prepared 15 September 2026. This is an evaluation proposal grounded in the current
worktree and the papers cited below, not a report of completed experiments or a
claim of publication readiness. Code inspected: `libatpgllm` HEAD `f34692a`,
including existing uncommitted evaluator changes. No evaluator code was changed
for this note. The definitions proposed here are adaptations for this task;
the cited papers motivate the methodology, not these exact new metric names.

The central claim should be falsifiable: a fault-conditioned language model can
generate semantically valid tests for previously unseen combinational circuits,
and its contribution remains measurable when verification and search costs are
accounted for. A separate claim concerns its utility inside an ATPG workflow.
Neither claim requires claiming superiority to industrial ATPG on every circuit.
No metric can eliminate every reviewer objection; explicit denominators,
independent checks, strong baselines, and appropriately bounded claims are the
defensible approach.

The inspected implementation supports combinational single stuck-at testing.
This proposal treats that as the primary scope. Scan, transition faults,
physical defect coverage, and silicon measurements are separate extensions,
with additional protocols specified below.

**1. Define the object being evaluated**

A problem is a circuit, a target fault, and a fixed test protocol:
`(C, f, constraints, observable_outputs)`. A completion proposes a test
`t = (x, y_hat)` and optionally a set of claimed detected faults. Let `g_C(x)`
and `g_C,f(x)` be independently evaluated good and faulty outputs. Use all
canonical primary outputs for the present combinational task. Any future mask,
scan access, reset, clock, or timing restrictions must be specified in advance.

Keep three predicates separate:

* `Vx(t)`: the input assignment is unambiguous, complete, binary, and legal.
* `D(t,f) = Vx(t) AND [g_C(x) != g_C,f(x)]` on observable outputs: the
  proposed stimulus detects the target under the declared fault model.
* `S(t)`: the input and expected-output fields are legal and complete, and
  `y_hat = g_C(x)` on every required output. This is a sound executable test.
* `U(t,f) = S(t) AND D(t,f)`: a usable detecting test, which both accepts the
  good circuit and rejects the target faulty circuit.

The first item validates syntax and assignments; detection establishes stimulus
quality; soundness validates the response oracle; their conjunction validates
the delivered test. A missing or false fault-name report is a separate semantic
reporting failure unless that report is required by the deployment interface.
Do not reward a test for rejecting a faulty circuit if it also rejects the
fault-free circuit because its expected output is wrong.

This distinction follows the good/faulty circuit formulation used in ATPG;
SAT can also establish whether a detecting assignment exists. See Larrabee
[R1](https://www.eecg.toronto.edu/~ece1767/project/larrabee.pdf).

All denominators below include malformed, truncated, exhausted, and failed
completion slots as unsuccessful attempts. Separately enumerate their causes.
An infrastructure failure is an unknown measurement, not evidence of a logic
error: retain it as unsuccessful in the end-to-end score, report the missing
verification rate, and use a predeclared identical retry policy for all methods.
Do not present an interrupted evaluation as a complete benchmark.

**2. Primary endpoints**

| Endpoint | Definition and interpretation | Required comparison |
| --- | --- | --- |
| Detection pass@k | Probability that at least one of k independently generated completion slots satisfies D | Base model, SFT, SFT+GRPO, random input vectors |
| Usable-test pass@k | Same calculation using U; primary endpoint for a model that emits both inputs and expected outputs | All language-model variants under the same tool policy |
| Sound-test rate | Mean S over all completion slots | Reveals failures hidden by detection-only scoring |
| Full-fault-set coverage | Fraction of a prespecified circuit fault universe detected by the union of produced patterns | Commercial or classical ATPG and budgeted random/vector search |
| Coverage versus cost | Coverage at fixed generation time, simulator budget, or test-application budget | Matched resource settings and full Pareto curves |
| Generalization delta | Paired change in usable-test success and coverage on held-out designs and semantics-preserving transformations | Renaming, serialization, size, topology, and library controls |

For n independent completion slots on problem i and c_i successful slots:

`pass_i@k = 1 - choose(n-c_i, k) / choose(n, k)`, for `1 <= k <= n`.

Use the established estimator of Chen et al.
[R2](https://arxiv.org/abs/2107.03374). Apply it separately to D and U from
the same saved outputs. Do not rerank outputs separately with the test oracle
and then call both scores the quality of the same deployed selection policy.
Report success of the actual returned answer as well as any oracle-selected
best-of-k upper bound.

For MCTS, evolutionary search, or best-of-N, an independently reset complete
search can be one slot. Then pass@k is success across k entire searches, and
must be labeled with each search's budgets. Dependent candidates inside a single
tree or population are not independent completion slots. Use independent search
repeats to estimate success, and trajectory-based budget curves for adaptive
search. A deterministic policy rerun with identical settings adds no sampling
diversity. Do not deduplicate the stochastic sample pool before computing
pass@k; that changes the distribution being measured.

With several faults per circuit, average within each circuit first, then report
the mean across circuits (macro average). Also report the pooled fault-weighted
micro average and circuit-level distributions. Equal weighting of dataset rows
can overrepresent circuits with many ATPG patterns or duplicated fault records.

**3. Circuit coverage: a different experiment from targeted pass@k**

Build a fixed fault manifest `F_C` independently of what the model or the
training-data ATPG happened to detect. Specify stems versus branches, sa0 and
sa1, fault collapsing, constraints, and supported cells. For small circuits,
enumerating the declared full fault universe is preferable to sampling.

For the unique generated stimulus set X:

`Fdet_C(X) = {f in F_C : exists x in X with g_C(x) != g_C,f(x)}`.

`FC_all(C) = |Fdet_C(X)| / |F_C|`.

This counts incidental detections as well as the requested target. Report both
stimulus coverage and deployable coverage; the latter uses only tests passing S.
If a deterministic simulator supplies or repairs expected outputs, label that
pipeline explicitly, account for its cost, and retain the raw language-model
soundness result. Fault detection sets must come from independent fault
simulation, not the completion's list of claimed faults.

Partition the manifest into witnessed testable faults T, proven untestable
faults Z, and unresolved faults R. An ATPG timeout is unresolved, not untestable.
Use proof-capable ATPG/SAT with the exact legal-input constraints for Z. Keep
the classification common across methods; adjudicate any new witnesses under
a predeclared protocol before producing the final common manifest.

Always report `|T|`, `|Z|`, `|R|`, and FC_all. When R is nonempty,
`|Fdet intersect T|/|T|` is coverage of confirmed-testable faults, not proven
coverage of all testable faults. If all detected faults have been added to T,
the unknown true testable-fault coverage lies between
`|Fdet|/(|T|+|R|)` and `|Fdet|/|T|`, when these denominators are nonzero.
Label the bounds; do not remove unresolved faults to manufacture high coverage.
If using collapsed faults, publish the equivalence mapping and raw-fault
weights or report representative-fault coverage separately.

Record the ordered test stream as well as its deduplicated set. Report:

* Coverage after m applied patterns, and after common simulator/time budgets.
* Pattern count needed to reach a prespecified coverage q. If q is not reached,
  report that outcome rather than averaging it away.
* Unique and total emitted pattern counts; compacted count under the same
  compaction algorithm and settings for every method.
* Incremental coverage beyond an ATPG baseline:
  `|Fdet_LLM minus Fdet_ATPG|/|F_C|`, with extra compute and pattern cost.
* For an LLM-seeded ATPG claim, total seed-generation plus cleanup cost versus
  ATPG from scratch at the same final coverage, including random-seeded control.

Compare compacted sizes at equal coverage, or show the coverage-size frontier.
A tiny low-coverage set is not a compaction improvement. The practical relevance
of pattern count and multiple-fault targeting is supported by Eggersgluss et al.
[R3](https://agra.informatik.uni-bremen.de/doc/konf/12DDECS-MultipleTarget.pdf).

**4. Search efficiency and attribution to the language model**

Publish two experiments: model generation without interactive tool feedback,
and the model-plus-tools/search system. An offline evaluator can check the
former without showing its results to the model. Online simulator queries,
candidate ranking, retries, repair, and controller-generated vector edits all
belong to the latter system and consume its budget.

For each candidate event save cumulative wall time, generated and input tokens,
model requests, simulator requests, actual simulator executions, cache hits,
and candidate origin (model, random, mutation, crossover, deterministic repair).
Record first model proposal before feedback, first verified success, and final
returned test. Report both initial-proposal quality and search improvement.
This distinguishes a useful learned proposal distribution from successful
black-box search around a weak model.

Compare at fixed wall time on disclosed hardware, and separately at fixed
simulator-execution budgets. Within a given LLM/backend also compare token
budgets; equal token counts across different tokenizers/models are not equal
compute. Include CPU/GPU count, batching, memory, precision, cache policy, tool
startup, license waits, and cold versus amortized model load time. Test-generation
time and physical test-application time are different costs.

For a predeclared scalar budget b in [0,B], plot success/coverage Q(b). An optional
summary is `AUC_B = integral_0^B Q(b) db / B`. Use the same B and axis measure
for every method. If using a log axis, define the integration measure explicitly.
Do not manufacture intermediate results from the final winner: reconstruct
them from timestamped trajectories and a fixed selection/verification policy.

For first-success cost T_i, define T_i = infinity for no success before B.
Report `mean(min(T_i,B))` along with success@B; this capped first-success cost
keeps failures in the denominator. It is not the same as actual mean runtime.
Also report actual total resource cost across all successes and failures.
Median time among successful cases alone can favor a method that solves only
the easy cases. For wall-time curves, verification must finish before the
deadline to count as an available success.

Use external audit verification without feedback to assess all methods.
Account for its cost separately from online generation/search, and give the
end-to-end total if claiming deployable turnaround time.

**5. Metrics that specifically test a language-of-test hypothesis**

Good fault coverage alone does not isolate the contribution of a language
representation. The following are proposed controlled experiments, motivated
by semantic/metamorphic testing; they do not prove human-like reasoning.
Metamorphic testing checks relations between transformed inputs and outcomes;
see Chen et al. [R4](https://doi.org/10.1145/3143561).

| Experiment | Metric | Interpretation and control |
| --- | --- | --- |
| Rename nets, ports, and instances consistently | Paired change in U-pass@k; success on both original and renamed problems | Map ports and the target fault bijectively; independently check semantic equivalence |
| Reorder declarations/gates; change valid whitespace | Paired change in U-pass@k and malformed-output rate | Same circuit and fault; measure tokenizer-length changes |
| Paraphrase the fault request | Same outcomes across held-out templates | Preserve the exact machine-readable task and constraints |
| Hold out RTL families, sizes, depths, and library cells | Absolute U-pass@k, FC, cost, and drop from the in-distribution benchmark | Keep base-design variants in one split; disclose unsupported/out-of-context cases |
| Change fault conditioning within the same circuit | Detection probability for requested f with prompt f versus prompt f' | Score both outputs against f; stratify by fault difficulty and use pairs for which conditioning can matter |
| Controlled logic edits | Success on independently relabeled edited circuits | Edits must change the relevant semantics; retain separate checks for the unchanged parts |
| Structured trace or witness, if claimed | Fraction of stated values/edges consistent with independent good/faulty simulation; full-certificate validity | Fix required fields to prevent empty or trivial certificates from scoring perfectly |

For renamed problems, require semantic success after mapping back, not the same
input-vector bytes: many distinct test vectors are valid. For logic edits or
resynthesis, a functionally equivalent good circuit does not automatically
preserve internal fault identities; use a proven fault mapping or treat it as
a new fault problem. Equal random seeds do not guarantee identical samples
across differently tokenized prompts.

Define a fault-conditioning contrast, for example:
`E[D_f(x drawn from model(C,f)) - D_f(x drawn from model(C,f'))]`.
Use a controlled wrong-fault prompt f' and identical budgets. Random vectors and
fault-blind vector search are negative controls. Faults can share detecting
patterns, so a zero contrast for universally easy faults is inconclusive;
characterize detectability and analyze a prespecified informative subset.

A rationale merely agreeing with the final answer is not a reasoning-fidelity
metric. Tool-produced tables establish tool consistency, not the model's
ability to simulate. If claiming circuit reasoning, evaluate model-produced
values before feedback and/or complete machine-checkable witnesses.

Reference-vector exact match, BLEU/ROUGE, perplexity, and text diversity are not
primary evidence of test effectiveness. Functional equivalence admits many
valid answers; execution-based evaluation is the relevant precedent
[R2](https://arxiv.org/abs/2107.03374). Benchmark adequacy also matters:
EvalPlus shows that stronger functional checks can change apparent correctness
and model rankings [R5](https://arxiv.org/abs/2305.01210).

**6. Secondary diagnostics and test-set quality**

Keep activation rate, output bit accuracy, exact expected-output accuracy,
assignment validity, schema conformance, and error categories. Report these
unconditionally, with conditional versions explicitly labeled. For example,
`P(detection | activated)` diagnoses propagation difficulty, but cannot replace
overall detection. Missing outputs are incorrect, not removed from bit-accuracy
denominators. Report per-pattern exact output correctness as well as bitwise
accuracy, since a high bit score can coexist with no fully correct tests.

For a claimed detected-fault set H(t), compare against the independently
simulated set Fdet(t) within a declared fault universe. Report precision,
recall, and exact-set match. Empty predictions have undefined precision unless
a convention is declared; report their count. If the field is intended to name
only the requested fault, score that narrower claim and do not demand recall
of every incidental detection. Mere target-name mention is not fault detection.

For useful diversity, count distinct canonical stimulus vectors, then compute:

`Ndetect_r(X) = |{f in F_C: number of distinct x detecting f >= r}| / |F_C|`.

Use r = 1,2,4,8 as an example prespecified grid, with actual counts reported.
Distinct wording or repeated identical inputs do not increase r. Compare at
equal pattern/compute budgets. Distinct vectors may exercise the same local
fault behavior; record observation outputs or fault-site neighborhoods as
optional diagnostics. N-detect does not establish physical defect coverage;
Pomeranz and Reddy explicitly analyze limitations of finite n
[R6](https://arxiv.org/abs/0710.4735).

If test cubes with X values are introduced, distinguish an existential cube
(some fill detects) from a robust cube (every allowed fill detects). State the
fill policy; a single successful fill does not validate every fill. The present
final-answer verifier requires binary assignments, so cube metrics are an
extension, not an existing capability.

For scan/test-power claims, add measured or timing-aware estimated peak and
average shift/capture switching, constraint violations, data volume, and
application cycles under one fixed scan protocol. Input-vector Hamming distance
alone is not a measure of internal switching power. These are optional until
the paper claims low power, scan applicability, or production-test cost.

**7. Independent verification and physical hardware evidence**

The training reward and evaluator currently share reward and simulation code.
Keep that fast feedback path, but audit results with an independently
implemented reference: commercial fault simulation, a SAT circuit model, or an
independent HDL good/faulty replay flow. A separate function calling the same
simulator is not implementation independence. Library functions, net naming,
pin order, fault locations, and observation rules must agree.

On tiny circuits, exhaustively check all legal inputs and declared faults.
On the main benchmark, replay all final tests if feasible, plus a stratified
sample of rejected candidates and malformed inputs to audit false negatives
and parsing. Do not validate only apparent successes.

Report internal-versus-reference detection confusion counts, disagreement rate,
reference-verified fraction, and false acceptance among internally accepted
tests: `count(internal=1, reference=0) / count(internal=1)`. Unknown reference
outcomes get their own status; do not silently turn them into negative results
or omit them. Also check good-output agreement and compilation/pin-mapping
success. Expand any disagreement into a reproducible counterexample.

An independent simulator validates the declared digital fault model. If actual
hardware is part of the claim, use a separate experiment:

* FPGA replay: compile a golden design and controlled fault-injected variants,
  preserve the intended fault site through synthesis, replay exported tests,
  and compare captured outputs. Measure export/execution success, good-design
  rejection rate, injected-fault detection and escape rate, simulation-to-FPGA
  disagreement, and test cycles/latency. State clock/reset/voltage conditions and
  repeated-trial design. FPGA injection demonstrates hardware execution under
  the injected model, not ASIC manufacturing-defect yield.
* Silicon/ATE: validate pin timing, masks, reset/scan sequence, golden-device
  behavior, and failures on independently characterized defective devices.
  Report incremental confirmed failures, false rejects, application time, and
  adjudicated escapes with device-level denominators. Uncharacterized failures
  are not automatically true defect detections. A small experiment cannot
  establish a production DPPM rate.
* Broader defect quality: independently evaluate bridging, transition, or
  cell-internal defects only with the necessary physical models and stimuli.
  Transition testing needs a defined launch/capture sequence and timing
  assumptions; a single static vector does not test that claim.

The gap between stuck-at success and production defect quality is substantive:
Hapke et al. evaluate cell-internal defects and production data in *Cell-Aware
Test*, IEEE TCAD [R7](https://doi.org/10.1109/TCAD.2014.2323216).

**8. Benchmark design and uncertainty**

Split by source RTL design/family before expanding faults and patterns. Keep
renamed, resynthesized, parameterized, or trivially edited descendants in the
same group. Audit train-validation-test overlap using provenance and canonical
structural fingerprints, followed by equivalence checks where appropriate.
Text hashes alone do not detect renamed duplicates. Public benchmarks can be
familiar to pretrained models; include newly generated held-out structures and
avoid claiming complete pretraining-contamination exclusion without evidence.

Create separate suites for (a) held-out designs within the supported small
combinational regime, (b) harder/larger and structurally different designs,
(c) semantic transformations, and (d) supported external benchmarks. Freeze a
model-independent manifest and disclose all context-length and cell-support
exclusions. For cross-tokenizer comparisons report a common eligible subset
and eligibility rate on the full suite. Do not equate a token-filtered stream
prefix with a representative sample of the corpus.

Stratify by gate/PI/PO count, logic depth, reconvergence, fault-site/sa0/sa1,
and a separately measured difficulty proxy. Random detectability estimated
from a fixed independent sample is useful; zero observed random hits is not a
proof of undetectability. Report the uncertainty of that difficulty estimate.
The easy, moderate, and hard partitions must be defined without model outcomes.

Choose the primary endpoint and operating budget on validation data. Select
checkpoints and temperatures there; then evaluate the locked test suite.
Reporting the best checkpoint on the test set is selection bias. If many
secondary comparisons are tested, control multiplicity or label exploratory
findings clearly.

Use paired confidence intervals for method differences, resampling whole
independent design families and preserving all their faults/completions.
Account separately for independent training seeds and inference/search seeds;
many completions from one trained checkpoint do not establish training
stability. For crossed designs and training seeds, use a bootstrap/hierarchical
model that respects both factors, not a binomial interval over all rows.
Report 95% intervals, absolute percentage-point gains, circuit-level wins/ties/
losses, and performance distributions. State whether inference targets the
fixed benchmark or a wider design population. Few families imply weak evidence
for population-wide claims, regardless of row count. This follows the emphasis
on interval estimates and performance profiles in Agarwal et al.
[R8](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).

Plan replication using pilot variability and a practically meaningful effect;
there is no universal number of seeds that guarantees sufficient power.
For a simple independent Bernoulli audit with zero observed errors in n trials,
the exact one-sided 95% upper error bound is `1 - 0.05^(1/n)` (approximately
3/n). Repeated correlated faults on one circuit are not n independent audit
trials. Report this limitation instead of calling zero observed errors proof
of zero error.

**9. Baselines and ablations that answer likely objections**

| Objection | Required experiment/evidence |
| --- | --- |
| Random inputs would work just as well | Uniform and validation-tuned weighted random inputs, equal resources; results by independent random-detectability strata |
| The simulator/search does the work | No-feedback model generation, tool-enabled generation, model-free vector evolution/random search, and candidate-origin accounting |
| Fine-tuning adds nothing | Same base model, SFT, and SFT+GRPO; identical test manifest, inference budgets, and seed protocol |
| More search explains the improvement | Success/coverage versus wall time and simulator executions; same-budget best-of-N and search comparisons |
| It memorizes names or circuits | Family-disjoint split, overlap audit, renaming/reordering tests, controlled fault changes |
| The reward is being exploited | Independent verification, strict output checking, verifier confusion matrix, adversarial parser cases |
| It is not useful compared with ATPG | Classical/commercial ATPG reference; same coverage/pattern/cost frontier; optional LLM-seeded cleanup including seed cost |
| The language representation is unnecessary | Direct structured bit-vector output versus full language-of-test format with comparable training/inference resources; optionally a suitable graph/ML proposal baseline |
| Reasoning text is decorative | No-rationale versus rationale ablation and independently verified structured witnesses; no claim that plausible prose proves reasoning |
| It only works on easy tiny circuits | Explicit supported-size statement, scale curves, hard-fault strata, external/OOD suite, inclusion/exclusion counts |
| Results are cherry-picked | Locked validation-selected settings, held-out test, paired intervals, all failures, training and search replication |
| Stuck-at coverage is not silicon quality | Restricted modeled-fault claim, or additional defect models and independently characterized hardware experiments |

For random/vector-only methods, evaluate D and circuit stimulus coverage fairly.
For a deployable pipeline comparison, give every method the same deterministic
expected-output generator, explicitly charge it, and report raw language-model
U separately. Requiring random baselines to guess expected outputs would
artificially handicap the stimulus baseline. Likewise, do not mislabel a system
that delegates generation to a conventional ATPG tool as autonomous LLM ATPG.

**10. Concrete gaps in the current worktree**

| Evidence | Consequence and proposed action |
| --- | --- |
| `scripts/eval/evaluate_model.py` loads the test split and calls buffering with `unique_by="netlist"`, `shuffle=False`, default cap 512 | Current target pass@k is a bounded sample of unique-netlist records, not all-fault circuit coverage. Use an explicit design/fault manifest for the latter. |
| `atpgllm/training/dataset_utils.py`, `unique_batch_generator` and `process_batch` | Deduplication uses the raw selected field; length filtering uses that field rather than necessarily the complete formatted prompt. Audit canonical identities and actual prompt lengths. |
| `data_preprocessing/final_dataset_creation.py:518` assigns rows using a random mask; the manuscript also describes row-level splitting | Unseen-design separation is not guaranteed. Measure actual overlap of the released/training artifacts and build a design-family split; the source alone does not quantify deployed leakage. |
| `atpgllm/training/search_types.py:21` | Default acceptance is D; `full_accuracy` adds output/input checks and target mention. Compute D and U side by side; do not use positive reward as correctness. |
| `atpgllm/llm/reward_funcs.py` | Input accuracy is assignment completeness; detected-fault accuracy is a target-name mention. Rename/document them and add exact semantic set checking where applicable. |
| `atpgllm/training/search_verifier.py`, `score_state` | Scoring reconstructs final fields and may append an actual matching tool observation. Simulation-table agreement can reflect supplied tool evidence, not an independently predicted model trace. |
| `data_preprocessing/fault_sim.py:725` and `resolve_fault_sim_runner` | Default is Python. Even requested TetraMAX depends on executable discovery. Record actual backend and enforce the declared independent-verification policy. |
| `data_preprocessing/fault_sim.py:989` | `tetramax_available` is only set when the detected set is nonempty. A valid empty reference result can therefore be treated as unavailable, leaving reward code to use Python detection. Audit this negative-result path before asserting independent verification. |
| `evaluate_model.py`, `rewards_summary` | `mean_total_reward` sums diagnostic fields and `mean_fault_sim_reward` looks up an absent legacy component. Neither should be a headline metric. |
| `evaluate_model.py`, `time_seconds` | Per-problem time is batch duration divided by batch size; it is not measured per-problem first-success latency. Save timestamps/cumulative counters for cost curves. |
| `evaluate_model.py`, W&B calls | Running pass@k and a running rate are logged at batch boundaries divisible by ten; full accuracies/usage at the end. Component means remain in JSON. Running rate excludes problem errors while final pass@k includes failed slots. Use explicit consistent denominators. |

These are source-level observations. No new performance runs, dataset-wide
overlap measurements, independent simulator campaigns, or hardware experiments
were performed in preparing this proposal.

**11. Reporting and implementation order**

First freeze semantics, circuit/fault manifests, family splits, independent
verification rules, and resource budgets. Then instrument predictions, selected
outputs, fault-detection matrices, and timestamps. Derived summaries should be
recomputable from those artifacts without rerunning the language model.

Recommended paper artifacts:

1. A benchmark table: design families, sizes, fault-model population, testability
   statuses, support exclusions, overlap audit, and splits.
2. A main results table: D-pass@1, U-pass@1 and U-pass@k, soundness, independent
   verification status, macro/micro FC, total cost, with paired intervals.
3. Coverage-versus-time, coverage-versus-simulator-work, and coverage-versus-
   pattern-count plots, plus a matched-coverage pattern-count table.
4. Generalization and ablation tables: design families, size/depth, difficult
   faults, renaming, fault conditioning, tool access, and model-free search.
5. A hardware/defect-quality table only when those experiments have been run.

During evaluation, log cumulative D/U pass@k, soundness, failures, and cumulative
resources after every completed batch or when a monotonic logging threshold is
crossed. Label these as prefix estimates. With multiple faults per circuit,
report provisional circuit coverage separately until its campaign is complete.
Log budget curves against real cumulative resources, not W&B's implicit step.
Compute final CIs, macro/micro coverage, and independent-verification results
after the full manifest is processed. Streaming progress must not be mistaken
for independent statistical replicates.

Save circuit and source-family IDs, manifest/dataset revision and hashes,
fault IDs, checkpoint and library hashes, seeds, prompt/template version,
raw answer, parsed input/output, candidate origin, raw and repaired outcomes,
actual simulator backend/version, reference status, good/faulty output vectors,
detected-fault sets, selection policy, cumulative costs, and terminal reasons.
Keep audit artifacts sufficient to reproduce both a success and a failure.

The minimum defensible primary bundle for this work is usable-test and detection
pass@k, independent verification, full-fault-set coverage, cost/compaction,
and design-disjoint semantic generalization, each with uncertainty. N-detect,
trace quality, power, and physical-defect measurements support additional
specific claims; accumulating more proxy scores cannot replace this bundle.

**References and their role**

R1. T. Larrabee, “Test Pattern Generation Using Boolean Satisfiability,”
*IEEE Transactions on Computer-Aided Design of Integrated Circuits and
Systems*, vol. 11, no. 1, pp. 4–15, 1992. DOI: 10.1109/43.108614.
[Paper](https://www.eecg.toronto.edu/~ece1767/project/larrabee.pdf).
Supports the good/faulty Boolean formulation and SAT ATPG reference.

R2. M. Chen et al., “Evaluating Large Language Models Trained on Code,”
arXiv:2107.03374, 2021.
[Paper](https://arxiv.org/abs/2107.03374).
Supports functional correctness and the pass@k estimator; it is not an ATPG
paper and does not establish equal compute for different search strategies.

R3. S. Eggersgluss, R. Krenz-Baath, A. Glowatz, F. Hapke, and R. Drechsler,
“A New SAT-based ATPG for Generating Highly Compacted Test Sets,”
*IEEE Symposium on Design and Diagnostics of Electronic Circuits and Systems
(DDECS)*, pp. 230–235, 2012.
[Author-hosted paper](https://agra.informatik.uni-bremen.de/doc/konf/12DDECS-MultipleTarget.pdf).
Supports pattern compaction and integration with an industrial ATPG flow.

R4. T. Y. Chen, F.-C. Kuo, H. Liu, P.-L. Poon, D. Towey, T. H. Tse, and
Z. Q. Zhou, “Metamorphic Testing: A Review of Challenges and Opportunities,”
*ACM Computing Surveys*, vol. 51, no. 1, article 4, 2018.
[DOI](https://doi.org/10.1145/3143561);
[author manuscript](https://eprints.nottingham.ac.uk/51607/1/__MTChallOpporCSUR.accepted.20170922.pdf).
Motivates semantic transformation tests; the circuit-specific experiments
above are proposed adaptations.

R5. J. Liu, C. S. Xia, Y. Wang, and L. Zhang, “Is Your Code Generated by ChatGPT
Really Correct? Rigorous Evaluation of Large Language Models for Code
Generation,” *NeurIPS*, 2023.
[Paper](https://arxiv.org/abs/2305.01210).
Supports strengthening semantic checks rather than relying on weak acceptance
criteria; it does not itself validate this hardware evaluator.

R6. I. Pomeranz and S. M. Reddy, “Worst-Case and Average-Case Analysis of
n-Detection Test Sets,” *Design, Automation and Test in Europe (DATE)*, 2005.
[Paper record](https://arxiv.org/abs/0710.4735).
The arXiv deposit is dated 2007; the listed conference is DATE 2005.
Supports treating n-detect as evidence with limits, not a physical-defect
coverage guarantee.

R7. F. Hapke et al., “Cell-Aware Test,” *IEEE Transactions on Computer-Aided
Design of Integrated Circuits and Systems*, vol. 33, no. 9, pp. 1396–1409, 2014.
[DOI](https://doi.org/10.1109/TCAD.2014.2323216);
[author-uploaded full text](https://www.researchgate.net/publication/264900648_Cell-Aware_Test).
Supports the distinction between modeled stuck-at coverage, cell-internal
defects, and production-test evidence.

R8. R. Agarwal, M. Schwarzer, P. S. Castro, A. C. Courville, and M. G. Bellemare,
“Deep Reinforcement Learning at the Edge of the Statistical Precipice,”
*NeurIPS*, 2021.
[Paper](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).
Supports interval estimates and performance distributions; family/seed-aware
inference above is an adaptation to the dependency structure of circuit data.
