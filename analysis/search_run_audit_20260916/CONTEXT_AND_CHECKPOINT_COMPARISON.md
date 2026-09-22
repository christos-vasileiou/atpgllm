**The SFT/GRPO context mismatch is confirmed. Its causal contribution to output errors is not yet established.** Bus indexing is corrected and regression-tested. The added strict rejection of unresolved canonical outputs has been removed at the user's request; model-generated expected outputs remain untouched.

**September 17 update: all 12 amended checkpoint jobs completed. A subsequent label audit also established incorrect cached training targets and contradictory snapshots.** The consolidated interpretation and prioritized remaining work are in [Resolved findings and next steps](RESOLVED_FINDINGS_AND_NEXT_STEPS.md). This file retains the detailed experiment history and reproduction instructions.

This follow-up uses the three requested W&B training runs, local checkpoint/tokenizer artifacts, cached train/test data, and fresh paired evaluation. It does not use the future design reports.

**Verified training configuration**

| Run | Sequence limit | Prompt filter in launch configuration | Completion setting | Starting checkpoint |
|---|---:|---:|---:|---|
| [SFT n5wplmt8](https://wandb.ai/chrivasileiou/sft-training/runs/n5wplmt8) | 8,192 | 2,048 | 6,144 in launch config | Granite 4.2 8B |
| [GRPO 5gmb8rfn](https://wandb.ai/chrivasileiou/grpo-training/runs/5gmb8rfn) | 16,384 | 4,096 | 12,288 | SFT checkpoint 200 |
| [GRPO f9uuxsrt](https://wandb.ai/chrivasileiou/grpo-training/runs/f9uuxsrt) | 16,384 | 4,096 | 12,288 | GRPO checkpoint 34, full-state resume |

SFT's operative sequence limit is `SFTConfig(max_length=8192)`; its completion setting is not a separate SFT generation budget. GRPO's logged generic `max_length=20` is not its training sequence limit. Its trainer-level `max_prompt_length=null` also should not be substituted for the launch/data-filter setting. The W&B configurations are preserved in [context_window_runs.json](context_window_runs.json).

The SFT formatter, reasoning-template renderer, and dataset utilities have no committed changes between the recorded SFT commit `eec7da1` and evaluation commit `f34692a`. This supports using the local formatter to investigate truncation risk, while not proving the exact cached dataset revision or consumed training stream.

**What the saved 512-circuit greedy trajectories say about length**

Lengths below reconstruct the saved histories with the SFT checkpoint's tokenizer/template and production tool schema. They are observational strata, not a training-window intervention.

| Initial rendered prompt | Slots | Exact expected-output accuracy |
|---|---:|---:|
| ≤2,048 tokens | 3,536 | 25.11% |
| 2,049–4,096 | 3,872 | 8.55% |
| >4,096 | 784 | 4.72% |

Among slots with a tool observation, exact output accuracy is 17.59% when the reconstructed post-tool prefix is at most 8,192 tokens, versus 1.34% beyond 8,192. For the intersection of initial prompt ≤2,048 and post-tool prefix ≤8,192, it is still only 25.43% (888/3,492). These prefixes do not reconstruct every unsaved intermediate generation token or establish the position of every final answer token.

Longer examples also have more outputs, making exact whole-vector correctness harder even without a per-bit accuracy decline. Restricting to valid finals whose vector matches the last observed tool vector gives:

| Initial prompt | Mean output bits | Mean per-pattern output-bit accuracy |
|---|---:|---:|
| ≤2,048 | 11.67 | 65.80% |
| 2,049–4,096 | 35.07 | 66.01% |
| >4,096 | 48.35 | 61.05% |

Thus the sharp decline in *exact* accuracy cannot be attributed to context length alone. This conditional bit-level analysis excludes invalid finals, changed vectors, and unresolved tool outputs; it is not an unconditional replacement metric. Reproduction and measurements: [context_length_diagnostics.py](context_length_diagnostics.py), [context_length_diagnostics.json](context_length_diagnostics.json), [context_output_bit_diagnostics.json](context_output_bit_diagnostics.json).

**Direct SFT truncation-risk probe**

A prespecified Bernoulli sample (probability 0.001, seed 20260916) selected 736 cached training rows before examining their lengths. Applying the SFT rendered-prompt filter leaves 91 rows. One of these 91 has a sequence longer than 8,192 and places the final `EXPECTED_OUTPUT` field after that boundary; none lacks the field in the untruncated formatted example. The tokenizer's assistant masks retain supervised expected-output tokens in all 90 other examples; none has retained output tokens that are completely masked from the loss.

This sample demonstrates that truncation can remove final-output supervision, but it does not show widespread removal in the eligible cached rows. It is a small probe of the formatter and cache, not an audit of the exact examples consumed by checkpoint 200 or its dependency revisions. Prefix-token positions are approximate at tokenization boundaries. See [sft_truncation_probe.py](sft_truncation_probe.py) and [sft_truncation_probe.json](sft_truncation_probe.json).

A separate mechanism is visible in the training formatter: SFT tool requests receive the stored expected-output vector, and the final answer repeats that vector. These examples teach repetition of the supplied label; they do not demonstrate correcting a wrong request after feedback. The subsequent label audit shows why calling these values "gold" or "already correct" would be misleading: only 43/91 sampled eligible labels agree with the corrected circuit simulation, and three simple mismatches are independently confirmed from wiring. Forty-seven incorrect targets retain supervised tokens within the SFT cap. The observed 98.39% copying in the historical greedy trajectories is compatible with this supervision, although its causal contribution still requires repaired-data training experiments. See [the label audit](sft_training_label_probe.json) and [independent label checks](sft_label_independent_checks.json).

**Bus correction and deferred output enforcement**

`data_preprocessing/netlist_utils.py` preserves the actual declared range bounds, including ascending and nonzero ranges, and expands unpacked dimensions before packed bits. The graph parser shares this expansion. This follow-up additionally fixes whitespace handling for multiple declared names and allows signed range bounds in the reward parser.

All 17 bus-identity regression cases pass. Both all-zero and all-one stimuli now produce resolved good/faulty canonical outputs for historical problematic indices 108, 321, 395, and 474. That is a simulator check; it does not substitute simulated outputs into model answers. The three-line strict unresolved-output rejection added after the historical runs is deliberately deferred. See [comparison_simulation_checks.json](comparison_simulation_checks.json).

**Paired checkpoint experiment**

The full cached training split has 2,848 distinct normalized netlists; the cached test split has 2,870. Only 22 test netlists are absent by text hash. Excluding compiled-cone structural duplicates and unsupported/unresolved transformation cases leaves **nine eligible circuits**. No training circuit with a matching PI/PO shape failed the structural exclusion scan. The frozen manifest contains four versions of each: original, renamed nets/modules/instances, reordered logic statements, and both transformations. Good-circuit and target-cone structural identities agree across each group. Cell types, pin labels, bus bounds, and logical connectivity are preserved.

This disjointness check covers exact content plus compiled Boolean cones invariant to the supported renaming/reordering. It is not complete arbitrary netlist-equivalence checking, source-family separation, or evidence about pretraining exposure. The nine circuits are a limited, mostly easy pilot: uniform detection probabilities range from 25% to 100%, with five between approximately 25% and 75% and four with all probe vectors detecting. Seven probabilities are exhaustive; two use 8,192 fixed random vectors.

Output predictability further limits interpretation. `multiple_module_opt2`, `dummy_generator`, and `ar_rxd_2` have constant outputs under exhaustive input enumeration; `slave2` has constant observed outputs over 8,192 random probes. The most frequent output pattern on `pc_sel` occurs in approximately 97% of probes. The corresponding frequencies on `half_wave`, `minus_8`, `dimc_macro_add_3u_82_4`, and `adder` are 50%, 1.56%, 12.5%, and 6.25%. These are oracle descriptions of output distributions, not an implementable baseline without circuit information, and model-selected inputs need not be uniform. See [comparison_output_baselines.py](comparison_output_baselines.py) and [comparison_output_baselines.json](comparison_output_baselines.json).

The initial manifest is [circuit_comparison_manifest.json](circuit_comparison_manifest.json). It fixes tokenizer vocabulary/template hashes, tool schema, source identities, transformed target faults, and complete messages. Its original prompts are within the SFT prompt window (543–1,025 tokens). The amended comparison uses [circuit_comparison_explicit_manifest.json](circuit_comparison_explicit_manifest.json), preserving every circuit, fault, transformation, and user message while making the system instructions explicit. All amended prompts, including transformed variants, are 890–1,537 tokens. Neither set establishes behavior on genuinely long held-out circuit prompts.

The amendment followed a concrete protocol failure in the initial base pilot. With one tool round and an 8K context, all 144 slots failed the final-answer contract (114 exhausted, 30 invalid), despite some inspected answers containing correct circuit reasoning. The prompt requested "quoted assignments" without specifying the required colon syntax, and did not disclose the one-call tool limit. The zero-tool base condition likewise had no valid finals (113 exhausted, 31 invalid). These are failures of the tested protocol, not proof of absent circuit understanding. Remaining initial-plan launches were cancelled while active jobs were allowed to finish. All initial outputs remain in `runs/circuit_comparison_20260916`.

The new shared instructions specify the exact three-field grammar, scalar/bus-bit enumeration, stuck-at notation, document identity, the one-call limit when a tool is available, and the meaning of the simulator's Good Machine column. They supply no circuit-specific answers. This is a documented prompt amendment prompted by a base-model failure, not a blind preregistered comparison. Reproduce it with [refine_comparison_manifest.py](refine_comparison_manifest.py). Scores from the two prompt versions must not be pooled.

For example, the original `multiple_module_opt2` base completion correctly identifies the tie-low output as `y=0` and the target `sa1 y` as detectable, then ends with `"INPUT_VECTOR = a=0, b=0, c=0, d=0"`, `"EXPECTED_OUTPUT = y=0"`, and `"DETECTED_FAULTS = sa1 y"`. This fails the verifier's colon-based syntax despite a semantically correct answer on this trivial circuit. It is direct evidence against interpreting that pilot's zero score as zero understanding.

All four active initial jobs completed without infrastructure errors. Their counts are preserved in `runs/circuit_comparison_20260916/pilot_status.json`; the other eight initial-plan jobs were cancelled, not scored as failures.

| Initial 8K condition | Valid finals / 144 | Detecting slots / 144 | Correct expected outputs / 144 |
|---|---:|---:|---:|
| Base, no tool | 0 | 0 | 0 |
| SFT, no tool | 19 | 13 | 7 |
| GRPO, no tool | 27 | 14 | 5 |
| Base, one tool | 0 | 0 | 0 |

The initial SFT answers also contain errors beyond punctuation. One short `multiple_module_opt2` completion treats its `TIELO` output as 1 and reverses the stuck-at-1 behavior. This completion used 2,030 generated tokens with a short initial prompt, so direct inference-context overflow does not explain that particular mistake. These examples are diagnostic observations, not a causal training-window experiment or an estimate of general reasoning ability.

The experiment compares the unadapted base, SFT checkpoint 200, and GRPO checkpoint 50 policy under both 8K/16K inference limits and zero/one tool rounds. Each cell uses seed 42, temperature 0.7, top-p 0.95, four independent slots per transformed problem, the same search limits, and detection-only selection. Raw expected-output and full-answer accuracy are reported separately. No output repair is applied. This is 12 jobs, each with 36 paired records and 144 completion slots.

The actual pilot uses vLLM with a common bf16 base and optional SFT/GRPO LoRA adapters. This holds the serving backbone fixed across the three checkpoints but differs from the older QLoRA-faithful merged runs. The runner also supports a separate HF/NF4 condition. No claim should attribute differences from the historical six-run scores solely to training or context.

The exact local adapter files and configurations are fingerprinted in [comparison_checkpoint_identities.json](comparison_checkpoint_identities.json).

Base and SFT tokenizer vocabularies and EOS IDs are identical. Their stored chat templates differ; every comparison condition explicitly uses the SFT template, with the tool schema omitted by the conversation runner when the tool budget is zero. This tests checkpoints under a shared format, not each checkpoint under its separately optimized native prompt. See [comparison_tokenizer_check.json](comparison_tokenizer_check.json). Per-call generation is capped at 4,096 tokens and the controller uses 2,048-token action chunks; continued chunks can consume the available context up to the configured overall search budget. The 8K/16K contrast can therefore affect long generated histories even though initial prompts are short.

GPU access required execution outside the sandbox. Initial adapter jobs failed because Triton's JIT could not locate Python development headers; the locally installed matching headers were supplied through the child processes' include path. Such setup failures are not model correctness outcomes.

**Fresh evaluation results**

All 12 amended jobs completed, covering 1,728 slots with zero infrastructure errors. Manifest/pairing checks passed and the adapter fingerprints remained unchanged. The plan is in `runs/circuit_comparison_explicit_20260916/plan.json`; the raw result hashes are in [the receipt](checkpoint_comparison_receipt.json). Full aggregates and paired circuit-bootstrap intervals are preserved in [CSV](checkpoint_comparison_summary.csv) and [JSON](checkpoint_comparison_summary.json). The original plan is retained as an incomplete, superseded pilot, with only completed results interpreted.

| Checkpoint | Context | Tool rounds | Valid final | Detection | Exact expected output |
|---|---:|---:|---:|---:|---:|
| Base | 8K | 0 | 47.92% | 47.22% | 42.36% |
| SFT | 8K | 0 | 54.86% | 47.22% | 18.06% |
| GRPO | 8K | 0 | 51.39% | 41.67% | 21.53% |
| Base | 8K | 1 | 40.97% | 40.97% | 40.28% |
| SFT | 8K | 1 | 78.47% | 59.03% | 24.31% |
| GRPO | 8K | 1 | 81.94% | 65.28% | 28.47% |
| Base | 16K | 0 | 95.14% | 90.97% | 81.94% |
| SFT | 16K | 0 | 67.36% | 53.47% | 21.53% |
| GRPO | 16K | 0 | 74.31% | 60.42% | 29.17% |
| Base | 16K | 1 | 89.58% | 89.58% | 88.19% |
| SFT | 16K | 1 | 78.47% | 62.50% | 29.17% |
| GRPO | 16K | 1 | 92.36% | 68.06% | 34.03% |

The 16K one-tool condition exposes a persistent feedback failure. Among valid finals using the simulated input, base corrects 43/43 wrong requested outputs, SFT 5/72, and GRPO 4/84. Even restricting the entire reconstructed history to at most 7,680 tokens, SFT copies 62/67 wrong requests and GRPO 71/75. On the five circuits with varying outputs, neither adapted checkpoint corrects a wrong request in the 16K tool condition (0/47 SFT and 0/57 GRPO), while base corrects 23/23. These conditional counts are not identical-prefix intervention trials. See [diagnostics](checkpoint_comparison_diagnostics.json) and [history-length checks](checkpoint_feedback_context.json).

A separate execution loop agrees with all scored output labels. Hand-derived equations on three pilot circuits agree with 463 final detection labels, 463 final-output correctness labels, and 494 simulator-output tables. See [independent checks](checkpoint_independent_checks.json). This validates representative scoring cases without claiming independent validation of every circuit.

From the `libatpgllm` directory, the amended matrix can be resumed with:

```bash
python scripts/eval/run_circuit_comparison.py \
  --manifest docs/reports/search_run_audit_20260916/circuit_comparison_explicit_manifest.json \
  --output runs/circuit_comparison_explicit_20260916 \
  --backend vllm --tp-size 1 --devices 0 1 2 3 \
  --python-include /proj/trela/christos/python_headers/usr/include/python3.11 \
  --wait-for-idle-devices --resume --execute
python scripts/eval/summarize_circuit_comparison.py runs/circuit_comparison_explicit_20260916
python docs/reports/search_run_audit_20260916/comparison_diagnostics.py runs/circuit_comparison_explicit_20260916
```

Resume after any existing coordinator has stopped; it skips completed matching files, not unfinished jobs in another process. Use the project environment's Python. The include path is specific to this host. Aggregation checks the manifest, checkpoint adapter paths, shared serving/sampling configuration, ordered problem identities, slot counts, and paired per-slot seeds. Confidence intervals resample source circuits with all their variants kept together.

**Interpretation boundary**

Changing an existing checkpoint's inference limit tests sensitivity to inference context constraints. It cannot identify the effect of having trained SFT at 8K versus 16K. Establishing that effect requires a matched-data SFT training ablation that separates the sequence cap from changes to prompt eligibility, followed by identical GRPO and held-out evaluation. These nine short circuits, one evaluation seed, and one training trajectory are a diagnostic pilot rather than evidence of general ATPG intelligence.
