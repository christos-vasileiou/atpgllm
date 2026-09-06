# GDPO objective for ATPG test generation

## Scope and classification

The previous training path fell within the multi-reward GRPO setting analyzed
by Liu et al., *GDPO: Group reward-Decoupled Normalization Policy Optimization
for Multi-reward RL Optimization* ([arXiv:2601.05242](https://arxiv.org/abs/2601.05242)).
`test_generation_grpo_reward` returned heterogeneous component dictionaries,
but `DualAdapterGRPOTrainer` summed every non-`_logonly` entry before TRL formed
group-relative advantages. Setting `scale_rewards=False` removed division by
the summed reward's group standard deviation, but it did not preserve separate
objective information. The paper's ablation also shows that removing this
division is not a reliable substitute for reward-decoupled normalization.

The implementation now uses a repository-local GDPO estimator because the
active environment has TRL 0.26.1, while stock
`multi_objective_aggregation="normalize_then_sum"` begins in TRL 0.27.0. This
avoids coupling the reward correction to a separate trainer upgrade.

## Authoritative simulation

For target fault $f=(n,s)$, where $n$ is the fault net and $s\in\{0,1\}$
is the stuck value, let $\mathbf{x}$ be the generated PI assignment and let
$\mathrm{Sim}(\mathbf{x},f)$ produce good- and faulty-machine values
$(g_v,b_v)$ for net $v$.

Production uses `resolve_fault_sim_runner()` and therefore the configured
`fast`, `tetramax`, or `hybrid` backend. If TetraMAX reports an authoritative
detection flag, that flag defines detection; otherwise the Python simulator
checks the primary outputs.

Canonical PI and PO names come from `OptimizedNetlist.input_nets` and
`OptimizedNetlist.output_nets`. The generated `EXPECTED_OUTPUT` values are
never inserted into the simulated good machine. They are predictions scored
after simulation. This prevents claimed values or omitted PO names from
fabricating or hiding a detection.

## Four semantic reward objectives

The only policy-loss keys are, in fixed order,

```text
("detection", "activation", "fidelity", "format")
```

All four raw values lie in $[0,1]$.

### Detection

$$
r_{\mathrm{det}}
=
\mathbb{I}\!\left[\exists o\in\mathrm{PO}: g_o\neq b_o\right].
$$

For a TetraMAX-backed call, the corresponding TetraMAX detection result
replaces the Python PO predicate. This is the highest-priority objective.

### Fault-site activation

$$
r_{\mathrm{act}}
=
\mathbb{I}[b_n=s]\,
\mathbb{I}[g_n\neq b_n].
$$

Activation is retained when detection is zero. It is the only pre-detection
shaping objective and supplies physically grounded variation in some all-fail
groups.

### Output fidelity

For the parsed claimed good-machine outputs $\hat{\mathbf{y}}$,

$$
\rho_{\mathrm{PO}}
=
\frac{1}{|\mathrm{PO}|}
\sum_{o\in\mathrm{PO}}
\mathbb{I}[o\text{ is supplied}]\,
\mathbb{I}[\hat y_o=g_o],
\qquad
r_{\mathrm{fid}}=r_{\mathrm{det}}\rho_{\mathrm{PO}}.
$$

The denominator is every canonical PO, not only supplied names. Missing and
invalid values therefore receive no fidelity credit.

### Interface and format compliance

Let $\rho_{\mathrm{PI}}$ be the fraction of canonical PI names supplied with a
binary value. This is completeness, not agreement with the dataset's one
reference vector; multiple distinct detecting patterns are valid.

Let $\rho_{\mathrm{struct}}$ be the mean of the available structural checks:
`<think>`, `<tool_call>`, `<tool_response>`, `INPUT_VECTOR`,
`EXPECTED_OUTPUT`, and `DETECTED_FAULTS`. Checks whose extractor is not
configured are omitted. The final objective is

$$
r_{\mathrm{fmt}}
=
r_{\mathrm{det}}\,
\frac{\rho_{\mathrm{struct}}+\rho_{\mathrm{PI}}}{2}.
$$

Both fidelity and format are conditioned on detection, following the paper's
recommendation to prevent easier objectives from dominating a harder,
prioritized correctness objective.

## GDPO advantage

For prompt $i$, rollout $j\in\{1,\ldots,G\}$, and objective
$k\in\{1,\ldots,K\}$, the implementation first computes

$$
Z_{ijk}
=
\frac{r_{ijk}-\mu_{ik}}{\sigma_{ik}+\epsilon},
\qquad
\epsilon=10^{-4},
$$

where $\mu_{ik}$ and $\sigma_{ik}$ are the mean and sample standard
deviation over valid rollouts for objective $k$ within prompt group $i$.
Weights are then applied after normalization:

$$
u_{ij}
=
\sum_k w_k Z_{ijk},
\qquad
(w_{\mathrm{det}},w_{\mathrm{act}},w_{\mathrm{fid}},w_{\mathrm{fmt}})
=
(1.00,0.25,0.20,0.05).
$$

Finally, over every valid sequence in the globally gathered generation batch
$\mathcal{B}$,

$$
A_{ij}
=
\frac{u_{ij}-\operatorname{mean}_{\mathcal{B}}(u)}
{\operatorname{std}_{\mathcal{B}}(u)+\epsilon}.
$$

The final normalization is sequence-level: each completion contributes once,
independent of completion length.

Raw scaling cannot express GDPO priority because a positive multiplicative
factor is removed by per-objective normalization. The explicit $w_k$ values
are therefore the only cross-objective priorities. A common positive scaling
of all $w_k$ is mostly canceled by final batch normalization.

## Degenerate and missing rewards

- A constant objective within one prompt group contributes zero.
- A group/objective slice with fewer than two valid observations contributes
  zero.
- Non-finite values are excluded from that objective's statistics rather than
  converted to zero rewards.
- A completion missing every objective is excluded from final statistics and
  receives zero advantage.
- If all objectives are constant, all final advantages are finite zeros.

These rules avoid the missing-reward behavior of early GDPO implementations,
where replacing `None` with zero could create a false preference.

## Trainer integration and distributed behavior

`RewardFunctionFactory.create_reward_function(return_component_dicts=True)`
attaches the objective key order and weights to the reward callable. Both
`DualAdapterGRPOTrainer` and `ToolCallingGRPOTrainer`:

1. preserve the component matrix;
2. gather it across all DDP ranks;
3. select only the four explicit objectives;
4. compute GDPO on the global matrix;
5. place the final advantages in the single reward column consumed by TRL.

TRL 0.26.1 still subtracts a per-group mean afterward. Every GDPO objective has
zero group mean before global scaling, so this inherited centering is a
numerical no-op. `scale_rewards=False` is required and checked at runtime to
prevent a second standard-deviation scaling pass.

The configured `loss_type="dapo"` and sequence-level importance ratios govern
the policy surrogate after advantage estimation; they do not change the GDPO
reward-decoupling equations.

## Diagnostics excluded from the loss

Keys ending in `_logonly` are retained for W&B and evaluation, including:

- simulator-confirmed detection and site activation flags;
- PI completeness and exact PO-report accuracy;
- tool-response and predicted simulation-table agreement;
- format-check average, table presence, and target-fault mention.

Fault mention and parseable table presence were removed from the policy reward
because they are inexpensive surface proxies. Tool-response agreement is also
diagnostic: tool-result tokens are external and masked from policy updates.

`train_scalar_from_reward_components` remains for legacy/evaluation callers
and applies the same four priorities to the raw objectives, but the training
path does not use that scalar to estimate advantages.

## ToolRL comparison

[ToolRL](https://github.com/qiancheng0/ToolRL) supplies useful dense partial
credit for tool names and arguments, but its released GRPO path sums format and
correctness before normalization. Copying its raw score ranges would reproduce
the failure mode addressed by GDPO. ATPG also differs fundamentally because
there may be many valid test vectors, so matching one reference vector is not a
sound correctness reward.

## Remaining limitations and validation requirement

- Fault-simulator errors remain authoritative errors.
- Site activation does not measure how close propagation came to a PO.
- Groups with neither detection nor activation variation have zero task
  advantage.
- With `beta=0.03`, a zero-task-advantage group can still receive a KL-only
  update toward the fixed SFT reference. This is intentional canonical GRPO
  behavior, not a hidden task reward.
- The four weights encode engineering priorities; they require ablation before
  being presented as optimal.
- Unit tests establish equation behavior, missing-value handling, canonical
  PI/PO scoring, and independence from claimed PO values. They do not establish
  better convergence. That claim requires a controlled GRPO-versus-GDPO A/B
  training run with identical prompts, seeds, rollout count, and optimizer
  settings.

## Code map

- Reward semantics: `atpgllm/llm/reward_funcs.py`
- GDPO estimator: `atpgllm/training/gdpo.py`
- Reward construction: `atpgllm/training/reward_function_factory.py`
- Trainer wiring: `atpgllm/training/dual_adapter_grpo_trainer.py` and
  `atpgllm/training/tool_calling_grpo_trainer.py`
- Authoritative simulation: `data_preprocessing/fault_sim.py`
- Training configuration: `scripts/train/training_code.py`
