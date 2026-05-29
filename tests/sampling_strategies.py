"""
sampling_strategies.py
======================

Pluggable inference-time search strategies for ``evaluate_model.py``.

Each strategy explores the model's output space under a fixed call budget
(``--n``) and stops as soon as the **external fault simulator** confirms the
target fault is detected (or the model emits EOS / the budget is exhausted).

Implemented (Phase 1):

  * ``best_of_n``    — Independent N samples with diversified temperature.
  * ``mcts``         — UCT search over partial completions; rollouts run to
                       EOS and the simulator-derived reward is backpropagated.
  * ``evolutionary`` — Genetic search; crossover splices ``INPUT_VECTOR``
                       between parents, mutation regenerates the suffix.

All three bypass model-issued tool calls during search and use the
simulator as the verifier / oracle. The original ``random`` path in
``evaluate_model.py`` (which still issues tool calls) is unaffected.

Integration: ``evaluate_model.py`` selects a strategy via
``--sampling_method``. For non-random strategies, the per-batch loop calls
:func:`run_strategy_batch` instead of ``generate_batch_n_completions_*``.

Design choices
--------------
* Stop conditions are uniformly two:
    1. Tokenizer EOS (handled by ``stop_token_ids`` in the underlying
       generator — same as the existing eval pipeline).
    2. Verifier reports ``detected=True`` from the external fault simulator
       (``resolve_fault_sim_runner()``).
* The verifier reads ``fault_detected_by_pred_input_vector_acc_logonly``
  *and* the non-suffixed key, so it works whether or not future reward
  refactors drop the ``_logonly`` suffix.
* PRMs are intentionally not used here. There is no separately-trained
  process-reward model in this repo; faking one would amount to
  format-heuristics, which the verifier already covers via the
  ``RewardFunctionFactory`` reward components.
"""

from __future__ import annotations

import math
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import regex as re

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_preprocessing"))

from fault_sim import resolve_fault_sim_runner
from reward_function_factory import RewardFunctionFactory


# Module-local regexes (decoupled from the factory so partial completions
# can be inspected without re-importing).
_INPUT_VECTOR_LINE_RE = re.compile(r'(INPUT_VECTOR:\s*"[^"]*")', re.DOTALL)
_EXPECTED_OUTPUT_LINE_RE = re.compile(r'(EXPECTED_OUTPUT:\s*"[^"]*")', re.DOTALL)
_DETECTED_FAULTS_LINE_RE = re.compile(r'(DETECTED_FAULTS:\s*"[^"]*")', re.DOTALL)


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class CompletionScore:
    """Verifier output for a single completion."""

    detected: bool
    scalar: float
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class SamplingResult:
    """Per-problem search result."""

    completions: List[str]
    scores: List[CompletionScore]
    detected_any: bool
    generator_calls: int  # total distinct generator invocations spent


# Keys produced by ``test_generation_grpo_reward`` that mark fault detection.
# The ``_logonly`` key is the one currently emitted; we keep both for
# forward-compatibility with any future reward refactor.
DETECTED_KEYS: Tuple[str, ...] = (
    "fault_detected_by_pred_input_vector_acc_logonly",
    "fault_detected_by_pred_input_vector_acc",
)


def reward_is_detected(reward: Dict[str, float]) -> bool:
    """True when the external simulator confirmed the fault was excited & propagated."""
    return any(float(reward.get(k, 0.0)) >= 1.0 for k in DETECTED_KEYS)


# =============================================================================
# Generator abstraction
# =============================================================================

class Generator(ABC):
    """Abstract generator. ``generate`` returns ``[[c_1..c_n] per prompt]``."""

    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        n: int = 1,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> List[List[str]]:
        ...


class VLLMGenerator(Generator):
    """vLLM-backed generator. Stop-on-EOS is enforced via ``stop_token_ids``."""

    def __init__(self, llm, tokenizer, lora_request, base_sampling_params):
        from vllm import SamplingParams  # local import keeps HF-only users happy

        self._SamplingParams = SamplingParams
        self.llm = llm
        self.tokenizer = tokenizer
        self.lora_request = lora_request
        self._base = base_sampling_params

    def _params(self, n, max_tokens, temperature, top_p):
        return self._SamplingParams(
            n=n,
            temperature=self._base.temperature if temperature is None else temperature,
            top_p=self._base.top_p if top_p is None else top_p,
            max_tokens=self._base.max_tokens if max_tokens is None else max_tokens,
            stop_token_ids=list(self._base.stop_token_ids or []),
        )

    def generate(self, prompts, n=1, max_tokens=None, temperature=None, top_p=None):
        sp = self._params(n, max_tokens, temperature, top_p)
        outputs = self.llm.generate(
            prompts,
            sampling_params=sp,
            lora_request=self.lora_request,
            use_tqdm=False,
        )
        return [[sub.text for sub in o.outputs] for o in outputs]


class HFGenerator(Generator):
    """
    HF Transformers fallback generator. Functional but slow for tree/MCTS
    methods at large budgets; prefer vLLM where possible.
    """

    def __init__(self, model, tokenizer, base_generation_config, micro_batch_size: int = 8):
        import torch

        self._torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self._base = base_generation_config
        self.micro_batch_size = micro_batch_size

    def generate(self, prompts, n=1, max_tokens=None, temperature=None, top_p=None):
        from transformers import GenerationConfig

        cfg = GenerationConfig(
            max_new_tokens=(
                self._base.max_new_tokens if max_tokens is None else max_tokens
            ),
            temperature=(
                self._base.temperature if temperature is None else temperature
            ),
            top_p=self._base.top_p if top_p is None else top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        device = next(self.model.parameters()).device
        orig_pad_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        try:
            expanded = [p for p in prompts for _ in range(n)]
            decoded_all: List[str] = []
            with self._torch.no_grad():
                for start in range(0, len(expanded), self.micro_batch_size):
                    batch = expanded[start : start + self.micro_batch_size]
                    enc = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.tokenizer.model_max_length,
                    ).to(device)
                    out = self.model.generate(**enc, generation_config=cfg)
                    decoded_all.extend(
                        self.tokenizer.batch_decode(
                            out[:, enc["input_ids"].shape[1]:],
                            skip_special_tokens=True,
                        )
                    )
        finally:
            self.tokenizer.padding_side = orig_pad_side

        return [decoded_all[i * n : (i + 1) * n] for i in range(len(prompts))]


# =============================================================================
# Verifier
# =============================================================================

class Verifier:
    """
    Batched scorer wrapping ``RewardFunctionFactory`` + the configured fault
    simulator (TetraMAX or fast python sim — see ``FAULT_SIM_BACKEND``).
    """

    def __init__(self, reward_factory: RewardFunctionFactory):
        from atpgllm.llm.reward_funcs import (
            test_generation_reward,
            train_scalar_from_reward_components,
        )

        self._test_generation_reward = test_generation_reward
        self._scalar = train_scalar_from_reward_components
        self.reward_factory = reward_factory
        self.fault_sim = resolve_fault_sim_runner()

    def _build_kwargs(self, prompts: List[str], records: List[Dict[str, Any]]) -> Dict[str, Any]:
        netlists: List[Any] = []
        for p, r in zip(prompts, records):
            netlist_field = r.get("netlist", "")
            try:
                resolved = self.reward_factory.validate_and_get_netlist_from_prompt(
                    p, netlist_field
                )
            except Exception:
                resolved = None
            netlists.append(resolved if resolved is not None else netlist_field)
        return {
            "fault_fn": lambda x, **kw: RewardFunctionFactory.fault_fn(x, **kw),
            "simulation_fn": RewardFunctionFactory.simulation_fn,
            "input_vector_fn": RewardFunctionFactory.input_vector_fn,
            "expected_output_fn": RewardFunctionFactory.expected_output_fn,
            "detected_faults_fn": RewardFunctionFactory.detected_faults_fn,
            "thinking_fn": RewardFunctionFactory.thinking_fn,
            "tool_call_fn": RewardFunctionFactory.tool_call_fn,
            "tool_response_fn": RewardFunctionFactory.tool_response_fn,
            "eval_mode": True,
            "lib_gate_funcs": self.reward_factory.gate_funcs,
            "fault_sim": self.fault_sim,
            "netlists": netlists,
            "fault": [r.get("fault", "") for r in records],
            "module_name": [r.get("module_name", "") for r in records],
        }

    def score_many(
        self,
        prompts: List[str],
        completions: List[str],
        records: List[Dict[str, Any]],
    ) -> List[CompletionScore]:
        if not completions:
            return []
        kwargs = self._build_kwargs(prompts, records)
        try:
            comps = self._test_generation_reward(prompts, completions, **kwargs)
        except Exception as e:
            print(f"[Verifier] reward calc failed: {e}")
            comps = [{} for _ in completions]
        scored: List[CompletionScore] = []
        for c in comps:
            scored.append(
                CompletionScore(
                    detected=reward_is_detected(c),
                    scalar=self._scalar(c) if c else 0.0,
                    components=c or {},
                )
            )
        return scored

    def score(
        self, prompt: str, completion: str, record: Dict[str, Any],
    ) -> CompletionScore:
        return self.score_many([prompt], [completion], [record])[0]


# =============================================================================
# Base strategy
# =============================================================================

class SamplingStrategy(ABC):
    """All strategies operate per-problem; ``sample_batch`` may parallelise."""

    name: str = "abstract"

    def __init__(self, generator: Generator, verifier: Verifier, budget: int, **_):
        self.gen = generator
        self.verifier = verifier
        self.budget = budget

    @abstractmethod
    def sample_batch(
        self,
        prompts: List[str],
        records: List[Dict[str, Any]],
    ) -> List[SamplingResult]:
        ...


# =============================================================================
# BEST-OF-N
# =============================================================================

class BestOfNStrategy(SamplingStrategy):
    """
    Independent N samples per prompt across a small temperature grid (more
    diverse than a single fixed T). All N are scored; results are returned
    in generation order so the caller can still compute ``pass@k`` over the
    full set. Early stop is not possible inside a single vLLM call, but
    the strategy reports ``detected_any`` so downstream code can short-
    circuit further work on detected problems if desired.
    """

    name = "best_of_n"

    def __init__(
        self,
        generator: Generator,
        verifier: Verifier,
        budget: int,
        temperatures: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(generator, verifier, budget)
        self.temperatures = list(temperatures) if temperatures else [0.4, 0.7, 1.0]

    def sample_batch(self, prompts, records):
        per_temp = max(1, self.budget // len(self.temperatures))
        # gathered[i] is the list of completions accumulated for prompt i.
        gathered: List[List[str]] = [[] for _ in prompts]
        for t in self.temperatures:
            batch = self.gen.generate(prompts, n=per_temp, temperature=t)
            for i, cs in enumerate(batch):
                gathered[i].extend(cs)

        # Normalize to exactly ``budget`` completions per prompt: pad with
        # extra samples at the lowest temperature if a temperature grid
        # didn't divide evenly.
        deficits = [self.budget - len(cs) for cs in gathered]
        if any(d > 0 for d in deficits):
            top_up_idx = [i for i, d in enumerate(deficits) if d > 0]
            top_up_prompts = [prompts[i] for i in top_up_idx]
            top_up_n = max(deficits[i] for i in top_up_idx)
            extra = self.gen.generate(
                top_up_prompts, n=top_up_n, temperature=self.temperatures[0],
            )
            for slot, i in enumerate(top_up_idx):
                gathered[i].extend(extra[slot][: deficits[i]])
        for i in range(len(gathered)):
            gathered[i] = gathered[i][: self.budget]

        # Verify all in one batched call per problem.
        results: List[SamplingResult] = []
        for prompt, record, cs in zip(prompts, records, gathered):
            scores = self.verifier.score_many(
                [prompt] * len(cs), cs, [record] * len(cs),
            )
            results.append(
                SamplingResult(
                    completions=cs,
                    scores=scores,
                    detected_any=any(s.detected for s in scores),
                    generator_calls=len(cs),
                )
            )
        return results


# =============================================================================
# MCTS WITH FAULT-COVERAGE BACKPROP
# =============================================================================

@dataclass
class _MCTSNode:
    prefix: str  # text appended to the prompt to reach this node
    parent: Optional["_MCTSNode"] = None
    children: List["_MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    terminal: bool = False

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0


class MCTSStrategy(SamplingStrategy):
    """
    UCT search over partial completions.

    Each iteration:
      1. Select a leaf via UCT.
      2. Expand: generate ``branching`` short chunks (``chunk_tokens``).
      3. Rollout: pick one child, continue it to EOS.
      4. Backprop: score the full completion via the simulator; propagate
         a value in ``[0, 1]`` (1.0 if detected; otherwise a sigmoid of the
         scalar verifier reward) up to the root.

    Budget: ``iterations = budget // 2`` (each iter costs ≈ 2 generator
    calls: one expansion + one rollout).
    """

    name = "mcts"

    def __init__(
        self,
        generator: Generator,
        verifier: Verifier,
        budget: int,
        branching: int = 3,
        chunk_tokens: int = 256,
        rollout_max_tokens: Optional[int] = None,
        ucb_c: float = 1.4,
        chunk_temperature: float = 0.9,
        rollout_temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(generator, verifier, budget)
        self.branching = branching
        self.chunk_tokens = chunk_tokens
        self.rollout_max_tokens = rollout_max_tokens
        self.ucb_c = ucb_c
        self.chunk_temperature = chunk_temperature
        self.rollout_temperature = rollout_temperature

    def _select(self, root: _MCTSNode) -> _MCTSNode:
        node = root
        while node.children and not node.terminal:
            log_n = math.log(max(1, node.visits))

            def _ucb(child: _MCTSNode) -> float:
                if child.visits == 0:
                    return float("inf")
                return child.mean_value + self.ucb_c * math.sqrt(log_n / child.visits)

            node = max(node.children, key=_ucb)
        return node

    def _expand(self, prompt: str, node: _MCTSNode) -> None:
        if node.terminal or node.children:
            return
        continuations = self.gen.generate(
            [prompt + node.prefix],
            n=self.branching,
            max_tokens=self.chunk_tokens,
            temperature=self.chunk_temperature,
        )[0]
        for c in continuations:
            node.children.append(_MCTSNode(prefix=node.prefix + c, parent=node))

    def _rollout(self, prompt: str, leaf: _MCTSNode) -> str:
        completion_tail = self.gen.generate(
            [prompt + leaf.prefix],
            n=1,
            max_tokens=self.rollout_max_tokens,
            temperature=self.rollout_temperature,
        )[0][0]
        return leaf.prefix + completion_tail

    @staticmethod
    def _backprop(leaf: _MCTSNode, value: float) -> None:
        node: Optional[_MCTSNode] = leaf
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    @staticmethod
    def _value_from_score(score: CompletionScore) -> float:
        if score.detected:
            return 1.0
        # Squash scalar reward to (0, 1); positive scalars → > 0.5.
        return 1.0 / (1.0 + math.exp(-0.1 * score.scalar))

    def _run_single(self, prompt: str, record: Dict[str, Any]) -> SamplingResult:
        root = _MCTSNode(prefix="")
        completions: List[str] = []
        scores: List[CompletionScore] = []
        calls = 0
        iterations = max(1, self.budget // 2)

        for _ in range(iterations):
            leaf = self._select(root)
            self._expand(prompt, leaf)
            calls += 1
            if leaf.children:
                # Prefer an unvisited child; fall back to random among siblings.
                unvisited = [c for c in leaf.children if c.visits == 0]
                target = unvisited[0] if unvisited else random.choice(leaf.children)
            else:
                target = leaf

            full_completion = self._rollout(prompt, target)
            calls += 1
            score = self.verifier.score(prompt, full_completion, record)
            target.terminal = True
            self._backprop(target, self._value_from_score(score))
            completions.append(full_completion)
            scores.append(score)
            if score.detected:
                break

        return SamplingResult(
            completions=completions,
            scores=scores,
            detected_any=any(s.detected for s in scores),
            generator_calls=calls,
        )

    def sample_batch(self, prompts, records):
        # Different prompts spawn different trees; run sequentially.
        return [self._run_single(p, r) for p, r in zip(prompts, records)]


# =============================================================================
# EVOLUTIONARY / GENETIC SEARCH
# =============================================================================

class EvolutionaryStrategy(SamplingStrategy):
    """
    Population-based search over completions.

    Each generation:
      1. Score the current population via the verifier (fault sim).
      2. Select the top ``elite_fraction`` as parents.
      3. Produce children, batched in one generator call:
         * **crossover**: splice donor's ``INPUT_VECTOR`` into host's
           completion, then truncate after the spliced line and let the
           model re-derive ``EXPECTED_OUTPUT`` / ``DETECTED_FAULTS``.
         * **mutation**: cut the parent at a random position and resample
           the suffix with a higher temperature.

    Stops early when any child detects the fault.
    """

    name = "evolutionary"

    def __init__(
        self,
        generator: Generator,
        verifier: Verifier,
        budget: int,
        population_size: int = 6,
        elite_fraction: float = 0.5,
        mutation_temperature: float = 1.0,
        crossover_temperature: float = 0.7,
        crossover_max_tokens: int = 2048,
        seed_temperatures: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(generator, verifier, budget)
        self.population_size = population_size
        self.elite_count = max(1, int(round(population_size * elite_fraction)))
        self.mutation_temperature = mutation_temperature
        self.crossover_temperature = crossover_temperature
        self.crossover_max_tokens = crossover_max_tokens
        self.seed_temperatures = list(seed_temperatures) if seed_temperatures else [0.5, 0.8, 1.0]

    @staticmethod
    def _splice_input_vector(donor: str, host: str) -> Optional[str]:
        """Replace host's INPUT_VECTOR line with donor's; None if either lacks it."""
        d = _INPUT_VECTOR_LINE_RE.search(donor)
        h = _INPUT_VECTOR_LINE_RE.search(host)
        if not d or not h:
            return None
        return host[: h.start()] + d.group(1) + host[h.end():]

    def _seed_population(
        self,
        prompt: str,
        record: Dict[str, Any],
    ) -> Tuple[List[str], List[CompletionScore], int]:
        per_temp = max(1, self.population_size // len(self.seed_temperatures))
        completions: List[str] = []
        for t in self.seed_temperatures:
            completions.extend(self.gen.generate([prompt], n=per_temp, temperature=t)[0])
        completions = completions[: self.population_size]
        scores = self.verifier.score_many(
            [prompt] * len(completions), completions, [record] * len(completions),
        )
        return completions, scores, len(completions)

    def _plan_children(
        self, elites: List[str], k: int,
    ) -> Tuple[List[str], List[str], List[Optional[int]]]:
        """
        For each of *k* children, decide crossover vs mutation and prepare
        the prefix to continue from. Returns (prefixes, kinds, max_tokens).
        ``kind`` is ``"c"`` for crossover or ``"m"`` for mutation.
        ``max_tokens`` is per-child, or ``None`` to use the default.
        """
        prefixes: List[str] = []
        kinds: List[str] = []
        max_tokens: List[Optional[int]] = []

        for _ in range(k):
            if len(elites) >= 2 and random.random() < 0.5:
                donor, host = random.sample(elites, 2)
                spliced = self._splice_input_vector(donor, host)
                if spliced is not None:
                    iv = _INPUT_VECTOR_LINE_RE.search(spliced)
                    if iv is not None:
                        prefixes.append(spliced[: iv.end()])
                        kinds.append("c")
                        max_tokens.append(self.crossover_max_tokens)
                        continue
            parent = random.choice(elites)
            # Avoid degenerate empty / huge cuts.
            min_cut = max(1, len(parent) // 4)
            max_cut = max(min_cut + 1, 3 * len(parent) // 4)
            cut = random.randint(min_cut, max_cut)
            prefixes.append(parent[:cut])
            kinds.append("m")
            max_tokens.append(None)

        return prefixes, kinds, max_tokens

    def _spawn_children(
        self,
        prompt: str,
        elites: List[str],
        k: int,
    ) -> List[str]:
        """Generate *k* children in a single batched generator call."""
        prefixes, kinds, max_tokens_list = self._plan_children(elites, k)
        # vLLM accepts heterogeneous prompts but a single max_tokens per call;
        # use the maximum so longer crossovers don't get truncated.
        unified_max = max(
            (m for m in max_tokens_list if m is not None), default=None,
        )
        # Temperature: use crossover_temperature if any crossover, else mutation_temperature.
        temp = self.crossover_temperature if "c" in kinds else self.mutation_temperature
        batch_prompts = [prompt + pre for pre in prefixes]
        batch_outputs = self.gen.generate(
            batch_prompts, n=1, max_tokens=unified_max, temperature=temp,
        )
        children: List[str] = []
        for pre, out_list in zip(prefixes, batch_outputs):
            tail = out_list[0] if out_list else ""
            children.append(pre + tail)
        return children

    def _run_single(self, prompt: str, record: Dict[str, Any]) -> SamplingResult:
        population, scores, calls = self._seed_population(prompt, record)
        all_completions: List[str] = list(population)
        all_scores: List[CompletionScore] = list(scores)

        if any(s.detected for s in all_scores):
            return SamplingResult(
                completions=all_completions,
                scores=all_scores,
                detected_any=True,
                generator_calls=calls,
            )

        while calls < self.budget:
            ranked = sorted(
                zip(all_completions, all_scores),
                key=lambda cs: cs[1].scalar,
                reverse=True,
            )
            elites = [c for c, _ in ranked[: self.elite_count]]

            remaining = self.budget - calls
            k = min(self.population_size, remaining)
            if k <= 0:
                break

            new_completions = self._spawn_children(prompt, elites, k)
            calls += k
            new_scores = self.verifier.score_many(
                [prompt] * len(new_completions),
                new_completions,
                [record] * len(new_completions),
            )
            all_completions.extend(new_completions)
            all_scores.extend(new_scores)
            if any(s.detected for s in new_scores):
                break

        return SamplingResult(
            completions=all_completions,
            scores=all_scores,
            detected_any=any(s.detected for s in all_scores),
            generator_calls=calls,
        )

    def sample_batch(self, prompts, records):
        return [self._run_single(p, r) for p, r in zip(prompts, records)]


# =============================================================================
# Factory + integration helper
# =============================================================================

STRATEGY_REGISTRY: Dict[str, type] = {
    "best_of_n": BestOfNStrategy,
    "mcts": MCTSStrategy,
    "evolutionary": EvolutionaryStrategy,
}


def list_available_strategies() -> List[str]:
    """Strategy names exposed to ``evaluate_model.py --sampling_method``."""
    return sorted(STRATEGY_REGISTRY)


def make_strategy(
    name: str,
    generator: Generator,
    verifier: Verifier,
    budget: int,
    **kwargs,
) -> SamplingStrategy:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown sampling strategy '{name}'. "
            f"Available: {list_available_strategies()}."
        )
    return STRATEGY_REGISTRY[name](generator, verifier, budget, **kwargs)


def normalize_to_n(
    result: SamplingResult, n: int,
) -> Tuple[List[str], List[Dict[str, float]]]:
    """
    Pad / truncate a ``SamplingResult`` so the caller can plug it into the
    existing ``pass@k`` pipeline which assumes exactly ``n`` completions
    per problem.

    Padding is empty strings + zero-reward dicts so the pad slots can
    never appear "correct".
    """
    completions = list(result.completions[:n])
    components: List[Dict[str, float]] = [s.components for s in result.scores[:n]]
    if len(completions) < n:
        pad = n - len(completions)
        completions.extend([""] * pad)
        components.extend([{} for _ in range(pad)])
    return completions, components


def run_strategy_batch(
    strategy: SamplingStrategy,
    prompts: List[str],
    records: List[Dict[str, Any]],
    n: int,
) -> Tuple[List[str], List[Dict[str, float]], List[SamplingResult]]:
    """
    Execute *strategy* on a batch and return three aligned views:

    * ``flat_completions``  — length ``len(prompts) * n`` (padded / truncated)
    * ``flat_rewards``      — same length; the verifier's component dicts
    * ``raw_results``       — the per-problem ``SamplingResult`` objects for
                              callers that want search-cost diagnostics.
    """
    raw_results = strategy.sample_batch(prompts, records)
    flat_completions: List[str] = []
    flat_rewards: List[Dict[str, float]] = []
    for res in raw_results:
        comps, rewards = normalize_to_n(res, n)
        flat_completions.extend(comps)
        flat_rewards.extend(rewards)
    return flat_completions, flat_rewards, raw_results


# =============================================================================
# Public factory helpers used by evaluate_model.py
# =============================================================================

def make_vllm_generator(llm, tokenizer, lora_request, sampling_params) -> VLLMGenerator:
    return VLLMGenerator(llm, tokenizer, lora_request, sampling_params)


def make_hf_generator(model, tokenizer, generation_config, micro_batch_size: int = 8) -> HFGenerator:
    return HFGenerator(model, tokenizer, generation_config, micro_batch_size)
