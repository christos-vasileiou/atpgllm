"""
sampling_strategies.py
======================

Pluggable inference-time search strategies for ``evaluate_model.py``.

``evaluate_model.py`` passes two orthogonal knobs:

  * ``--num_completions`` (N) — completions per problem; this *is* the pass@k
    pool.  Every strategy returns exactly N completions per problem, each one
    produced by a single independent application of the strategy, so the N
    outputs are i.i.d. draws from the (search-augmented) policy and the pass@k
    estimator stays valid.
  * the per-completion search width (B) — how much search backs *each* of the
    N completions: ``--n`` for ``best_of_n``, ``--budget`` for ``mcts`` /
    ``evolutionary``.  Total work per problem is ``N * (cost of one width-B
    application)``.

Implemented:

  * ``best_of_n``    — each completion is the best of ``B`` i.i.d. samples
                       (drawn at the eval temperature; ``B == 1`` reduces to
                       the plain i.i.d. baseline).
  * ``mcts``         — each completion is the best of one UCT search of ``B``
                       rollouts; rollouts run to EOS and the simulator-derived
                       reward is backpropagated.
  * ``evolutionary`` — each completion is the best of one genetic search of
                       ``B`` evaluations; crossover splices ``INPUT_VECTOR``
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
from tools import TOOLS, ToolHelper
from revert_template import get_generation_prompt_suffix, revert_chat_template


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

def _load_tool_helpers():
    """
    Lazily import the tool-call parser / executor from ``evaluate_model``.

    Lazy (call-time) to avoid an import cycle: ``evaluate_model`` imports this
    module at top level, so importing it back here at top level would deadlock
    the import machinery. By call time both modules are fully loaded.
    """
    from evaluate_model import (
        ensure_tool_call_arguments_dict,
        execute_tool_call,
        parse_tool_call,
    )

    return parse_tool_call, ensure_tool_call_arguments_dict, execute_tool_call


# Marks the boundary of a *resolved* tool exchange in the readable completion.
# Text after the final occurrence is the still-open assistant turn (where a new
# tool call may appear and must be executed).
_TOOL_RESPONSE_CLOSE = "</tool_response>"


class Generator(ABC):
    """
    Abstract generator. ``generate`` returns ``[[c_1..c_n] per prompt]``.

    Concrete subclasses must set ``self.tokenizer`` (used by
    :meth:`generate_with_tools` to revert / re-apply the chat template across
    tool rounds).
    """

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

    # -------------------------------------------------------------------------
    # Tool-aware generation (shared by all strategies)
    # -------------------------------------------------------------------------

    def _base_messages(self, prompt: str) -> List[Dict[str, Any]]:
        """
        System + user messages for *prompt* (a fully templated generation
        prompt). The trailing ``add_generation_prompt`` suffix is stripped so
        the revert parser doesn't emit a spurious empty assistant turn.
        """
        suffix = get_generation_prompt_suffix(tokenizer=self.tokenizer)
        clean = prompt[: -len(suffix)] if suffix and prompt.endswith(suffix) else prompt
        try:
            msgs = revert_chat_template(clean, tokenizer=self.tokenizer)
        except ValueError:
            return []
        # Keep only the context turns; any assistant content here would be the
        # (empty) open turn we just stripped.
        return [m for m in msgs if m.get("role") in ("system", "user", "tool")]

    @staticmethod
    def _open_turn_text(prefix: str) -> str:
        """Portion of *prefix* belonging to the still-open assistant turn."""
        if _TOOL_RESPONSE_CLOSE in prefix:
            return prefix.rsplit(_TOOL_RESPONSE_CLOSE, 1)[1]
        return prefix

    def generate_with_tools(
        self,
        prompts: List[str],
        n: int = 1,
        prefixes: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tool_rounds: int = 1,
    ) -> List[List[str]]:
        """
        Multi-round tool-aware generation, batched in lockstep across all
        active paths (mirrors ``generate_batch_n_completions_vllm`` but uses
        :meth:`generate` as the per-round primitive so it works for any
        backend).

        Each ``(prompt, prefix)`` pair expands to *n* independent paths. The
        returned **readable** completion for a path is
        ``prefix + model_text + <tool_response>…</tool_response> + …`` — the
        same format the reward parser consumes. Generation continues from a
        properly re-templated conversation after every tool call, so the model
        always sees the in-distribution tool format (not the readable markers).

        Constraint: a non-empty *prefix* must lie within the first assistant
        turn (it may contain an unexecuted ``<tool_call>`` but no already
        *resolved* ``<tool_response>``). All strategy call sites honour this.
        """
        parse_tool_call, ensure_args, execute_tool_call = _load_tool_helpers()

        if prefixes is None:
            prefixes = [""] * len(prompts)
        if len(prefixes) != len(prompts):
            raise ValueError("prefixes must align with prompts")

        base_messages_cache = [self._base_messages(p) for p in prompts]

        states: List[Dict[str, Any]] = []
        for p_idx, (prompt, prefix) in enumerate(zip(prompts, prefixes)):
            for _ in range(n):
                states.append({
                    "prompt": prompt,
                    "p_idx": p_idx,
                    "current_input": prompt + prefix,
                    "readable": prefix,
                    "assistant_acc": self._open_turn_text(prefix),
                    "done": False,
                })

        for _round in range(max_tool_rounds + 1):
            active = [i for i, s in enumerate(states) if not s["done"]]
            if not active:
                break
            
            inputs = [states[i]["current_input"] for i in active]
            outputs = self.generate(
                inputs, n=1, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p,
            )
            
            for i, out_list in zip(active, outputs):
                s = states[i]
                text = out_list[0] if out_list else ""
                s["readable"] += text
                s["assistant_acc"] += text
                
                if _round >= max_tool_rounds:
                    s["done"] = True
                    continue
                
                tool_call = parse_tool_call(s["assistant_acc"])
                if tool_call is None:
                    s["done"] = True
                    continue
                
                try:
                    ensure_args(tool_call)
                    tool_call["arguments"].update(
                        {"netlist": ToolHelper.get_netlist(s["prompt"])}
                    )
                    tool_result = execute_tool_call(tool_call)
                    messages = list(base_messages_cache[s["p_idx"]])
                    messages.append(
                        {"role": "assistant", "content": s["assistant_acc"]}
                    )
                    messages.append({
                        "role": "tool",
                        "name": tool_call.get("name", "fault_simulation_tool"),
                        "content": tool_result,
                    })
                    s["current_input"] = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        tools=TOOLS,
                        add_generation_prompt=True,
                    )
                    s["readable"] += (
                        f"\n<tool_response>\n{tool_result}\n</tool_response>\n"
                    )
                    s["assistant_acc"] = ""
                except Exception as e:
                    s["readable"] += (
                        f"\n<tool_response>\nTool round failed: {e}\n</tool_response>\n"
                    )
                    s["done"] = True

        grouped: List[List[str]] = [[] for _ in prompts]
        for s in states:
            grouped[s["p_idx"]].append(s["readable"])
        return grouped


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
        # Coerce to built-in scalars: vLLM's msgpack encoder rejects numpy
        # types (e.g. a numpy.float64 temperature from np.linspace).
        temp = self._base.temperature if temperature is None else temperature
        tp = self._base.top_p if top_p is None else top_p
        mt = self._base.max_tokens if max_tokens is None else max_tokens
        return self._SamplingParams(
            n=int(n),
            temperature=float(temp),
            top_p=float(tp),
            max_tokens=int(mt),
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
            max_new_tokens=int(
                self._base.max_new_tokens if max_tokens is None else max_tokens
            ),
            temperature=float(
                self._base.temperature if temperature is None else temperature
            ),
            top_p=float(self._base.top_p if top_p is None else top_p),
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

    def __init__(
        self,
        generator: Generator,
        verifier: Verifier,
        num_completions: int = 1,
        use_tools: bool = False,
        max_tool_rounds: int = 1,
        **_,
    ):
        self.gen = generator
        self.verifier = verifier
        # ``num_completions`` (N) is the pass@k pool size. Every strategy
        # returns exactly N completions per problem, each from one independent
        # application of the strategy (see module docstring).
        self.num_completions = num_completions
        self.use_tools = use_tools
        self.max_tool_rounds = max_tool_rounds

    def _generate(
        self,
        prompts: List[str],
        n: int = 1,
        prefixes: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> List[List[str]]:
        """
        Route generation through the tool-aware loop (when ``use_tools``) or the
        plain primitive. With plain generation, any *prefixes* are prepended to
        the returned text so callers get the same readable shape either way.
        """
        if self.use_tools:
            return self.gen.generate_with_tools(
                prompts, n=n, prefixes=prefixes,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                max_tool_rounds=self.max_tool_rounds,
            )
        out = self.gen.generate(
            prompts, n=n, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p,
        )
        if prefixes is not None:
            out = [[prefixes[i] + t for t in sub] for i, sub in enumerate(out)]
        return out

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
    Best-of-``width`` selection, repeated ``num_completions`` times per problem.

    Each of the N pass@k slots is one independent draw of ``width`` (B) i.i.d.
    samples at the eval temperature, keeping the single best by verifier
    ``detected`` then ``scalar``. ``B == 1`` degenerates to the plain i.i.d.
    baseline. All ``N * B`` samples are drawn in one batched generation per
    problem and grouped into N blocks of B.
    """

    name = "best_of_n"

    def __init__(
        self,
        generator: Generator,
        verifier: Verifier,
        num_completions: int,
        width: int,
        **kwargs,
    ):
        super().__init__(
            generator, verifier, num_completions=num_completions, **kwargs
        )
        self.width = width

    def sample_batch(self, prompts, records):
        total = self.num_completions * self.width
        # One batched draw of N*B i.i.d. samples per prompt at the eval
        # temperature (no temperature grid: the pool must be i.i.d.).
        batch = self._generate(prompts, n=total)

        results: List[SamplingResult] = []
        for prompt, record, samples in zip(prompts, records, batch):
            # Pad defensively so grouping always yields N blocks of exactly B.
            if len(samples) < total:
                samples = list(samples) + [""] * (total - len(samples))
            scores = self.verifier.score_many(
                [prompt] * len(samples), samples, [record] * len(samples),
            )
            best_completions: List[str] = []
            best_scores: List[CompletionScore] = []
            for g in range(self.num_completions):
                block = list(
                    zip(
                        samples[g * self.width : (g + 1) * self.width],
                        scores[g * self.width : (g + 1) * self.width],
                    )
                )
                best_c, best_s = max(
                    block, key=lambda cs: (cs[1].detected, cs[1].scalar),
                )
                best_completions.append(best_c)
                best_scores.append(best_s)
            results.append(
                SamplingResult(
                    completions=best_completions,
                    scores=best_scores,
                    detected_any=any(s.detected for s in best_scores),
                    generator_calls=len(samples),
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

    Width: each search runs up to ``budget`` (B) scored rollouts (one per
    iteration; each iteration costs ≈ 2 generator calls: one expansion + one
    rollout) and returns its single best completion. The strategy runs
    ``num_completions`` (N) such independent searches per problem.
    """

    name = "mcts"

    def __init__(
        self,
        generator: Generator,
        verifier: Verifier,
        num_completions: int,
        budget: int,
        branching: int = 3,
        chunk_tokens: int = 256,
        rollout_max_tokens: Optional[int] = None,
        ucb_c: float = 1.4,
        chunk_temperature: float = 0.9,
        rollout_temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(
            generator, verifier, num_completions=num_completions, **kwargs
        )
        self.budget = budget
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
        # Expansion is intentionally plain: chunks are partial assistant text,
        # so tool calls are not executed here. A ``<tool_call>`` landing inside
        # a chunk is resolved later by the tool-aware rollout from that node.
        continuations = self.gen.generate(
            [prompt + node.prefix],
            n=self.branching,
            max_tokens=self.chunk_tokens,
            temperature=self.chunk_temperature,
        )[0]
        for c in continuations:
            node.children.append(_MCTSNode(prefix=node.prefix + c, parent=node))

    def _rollout(self, prompt: str, leaf: _MCTSNode) -> str:
        # Tool-aware when use_tools: any tool call in (prefix + rollout tail)
        # is executed and the simulator output fed back before continuing.
        return self._generate(
            [prompt],
            n=1,
            prefixes=[leaf.prefix],
            max_tokens=self.rollout_max_tokens,
            temperature=self.rollout_temperature,
        )[0][0]

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

    def _search_once(
        self, prompt: str, record: Dict[str, Any],
    ) -> Tuple[str, CompletionScore, int]:
        """One UCT search of ``budget`` rollouts; returns its single best."""
        root = _MCTSNode(prefix="")
        best: Optional[Tuple[str, CompletionScore]] = None
        calls = 0

        for _ in range(self.budget):
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
            if best is None or (score.detected, score.scalar) > (
                best[1].detected, best[1].scalar
            ):
                best = (full_completion, score)
            if score.detected:
                break

        if best is None:
            best = ("", CompletionScore(detected=False, scalar=0.0))
        return best[0], best[1], calls

    def _run_single(self, prompt: str, record: Dict[str, Any]) -> SamplingResult:
        # ``num_completions`` independent searches; each contributes its single
        # best completion, giving an i.i.d. pass@k pool of size N per problem.
        completions: List[str] = []
        scores: List[CompletionScore] = []
        calls = 0
        for _ in range(self.num_completions):
            c, s, cc = self._search_once(prompt, record)
            completions.append(c)
            scores.append(s)
            calls += cc
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

    Width: each search evaluates up to ``budget`` (B) completions and returns
    its single best; stops early when any child detects the fault. The strategy
    runs ``num_completions`` (N) such independent searches per problem.

    Tool calling: when ``use_tools`` is set, seeding and child continuations
    run through the tool-aware loop. Seeds (no prefix) get full multi-round
    tool support. Child prefixes are cut from the parent's *first* assistant
    segment (before any resolved ``<tool_response>``) so a tool call in the
    regenerated tail is re-templated against the correct system+user context.
    """

    name = "evolutionary"

    def __init__(
        self,
        generator: Generator,
        verifier: Verifier,
        num_completions: int,
        budget: int,
        population_size: int = 6,
        elite_fraction: float = 0.5,
        mutation_temperature: float = 1.0,
        crossover_temperature: float = 0.7,
        crossover_max_tokens: int = 2048,
        seed_temperatures: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(
            generator, verifier, num_completions=num_completions, **kwargs
        )
        self.budget = budget
        self.population_size = population_size
        self.elite_count = max(1, int(round(population_size * elite_fraction)))
        self.mutation_temperature = mutation_temperature
        self.crossover_temperature = crossover_temperature
        self.crossover_max_tokens = crossover_max_tokens
        self.seed_temperatures = list(seed_temperatures) if seed_temperatures else [0.5, 0.8, 1.0]

    @staticmethod
    def _first_segment(text: str) -> str:
        """Parent text up to (not including) the first resolved tool response."""
        marker = "\n<tool_response>"
        return text.split(marker, 1)[0] if marker in text else text

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
        seed_count = min(self.population_size, self.budget)
        per_temp = max(1, seed_count // len(self.seed_temperatures))
        completions: List[str] = []
        for t in self.seed_temperatures:
            completions.extend(self._generate([prompt], n=per_temp, temperature=t)[0])
        completions = completions[:seed_count]
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
            # Mutate within the first assistant segment so the regenerated tail
            # re-templates cleanly if it issues a tool call (see class docstring).
            seg = self._first_segment(parent)
            # Avoid degenerate empty / huge cuts.
            min_cut = max(1, len(seg) // 4)
            max_cut = max(min_cut + 1, 3 * len(seg) // 4)
            cut = random.randint(min_cut, max_cut)
            prefixes.append(seg[:cut])
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
        # ``_generate`` already prepends each prefix (and appends any tool
        # responses), so each output is the full readable child completion.
        batch_outputs = self._generate(
            [prompt] * len(prefixes),
            n=1,
            prefixes=prefixes,
            max_tokens=unified_max,
            temperature=temp,
        )
        return [out_list[0] if out_list else "" for out_list in batch_outputs]

    def _search_once(
        self, prompt: str, record: Dict[str, Any],
    ) -> Tuple[str, CompletionScore, int]:
        """One genetic search (≤ ``budget`` evals); returns its single best."""
        population, scores, calls = self._seed_population(prompt, record)
        all_completions: List[str] = list(population)
        all_scores: List[CompletionScore] = list(scores)

        while not any(s.detected for s in all_scores) and (
            len(all_completions) < self.budget
        ):
            ranked = sorted(
                zip(all_completions, all_scores),
                key=lambda cs: cs[1].scalar,
                reverse=True,
            )
            elites = [c for c, _ in ranked[: self.elite_count]]

            remaining = self.budget - len(all_completions)
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

        if not all_completions:
            return "", CompletionScore(detected=False, scalar=0.0), calls
        best_c, best_s = max(
            zip(all_completions, all_scores),
            key=lambda cs: (cs[1].detected, cs[1].scalar),
        )
        return best_c, best_s, calls

    def _run_single(self, prompt: str, record: Dict[str, Any]) -> SamplingResult:
        # ``num_completions`` independent searches; each contributes its single
        # best completion, giving an i.i.d. pass@k pool of size N per problem.
        completions: List[str] = []
        scores: List[CompletionScore] = []
        calls = 0
        for _ in range(self.num_completions):
            c, s, cc = self._search_once(prompt, record)
            completions.append(c)
            scores.append(s)
            calls += cc
        return SamplingResult(
            completions=completions,
            scores=scores,
            detected_any=any(s.detected for s in scores),
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
    *,
    num_completions: int,
    width: Optional[int] = None,
    use_tools: bool = False,
    max_tool_rounds: int = 1,
    **kwargs,
) -> SamplingStrategy:
    """
    Build a strategy by name.

    ``num_completions`` (N) is the pass@k pool size for every strategy: the
    strategy returns exactly N completions per problem, each from one
    independent application of the strategy.

    ``width`` (B) is the per-completion search width and is required for all
    registered strategies:
      * ``best_of_n``    — i.i.d. samples drawn per completion (best is kept);
      * ``mcts``         — scored rollouts per search;
      * ``evolutionary`` — completions evaluated per search.

    ``use_tools`` enables the in-search fault-simulation tool loop (the model
    can call ``fault_simulation_tool`` mid-generation, exactly like the random
    eval path); ``max_tool_rounds`` caps how many tool calls each path may make.
    Extra ``kwargs`` are strategy-specific knobs (branching, population_size, …).
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown sampling strategy '{name}'. "
            f"Available: {list_available_strategies()}."
        )
    if width is None:
        raise ValueError(f"make_strategy({name}) requires width=...")
    common = dict(
        num_completions=num_completions,
        use_tools=use_tools,
        max_tool_rounds=max_tool_rounds,
        **kwargs,
    )
    if name == "best_of_n":
        return BestOfNStrategy(generator, verifier, width=width, **common)
    if name in ("mcts", "evolutionary"):
        cls = STRATEGY_REGISTRY[name]
        return cls(generator, verifier, budget=width, **common)
    raise ValueError(f"Unhandled strategy: {name}")


def select_for_pass_at_k(
    result: SamplingResult, num_completions: int,
) -> Tuple[List[str], List[Dict[str, float]]]:
    """
    Normalize a ``SamplingResult`` to exactly ``num_completions`` completions.

    Strategies already return exactly N independent completions, so this is a
    defensive normalizer only: it truncates / pads *in order* and never
    reorders by score (reordering would bias the i.i.d. pass@k pool). Pad slots
    are empty strings with zero-reward dicts so they can never count as
    correct.
    """
    completions = list(result.completions)
    components: List[Dict[str, float]] = [s.components for s in result.scores]
    if len(completions) > num_completions:
        completions = completions[:num_completions]
        components = components[:num_completions]
    elif len(completions) < num_completions:
        pad = num_completions - len(completions)
        completions.extend([""] * pad)
        components.extend([{} for _ in range(pad)])
    return completions, components


def run_strategy_batch(
    strategy: SamplingStrategy,
    prompts: List[str],
    records: List[Dict[str, Any]],
    num_completions: int,
) -> Tuple[List[str], List[Dict[str, float]], List[SamplingResult]]:
    """
    Execute *strategy* on a batch and return three aligned views:

    * ``flat_completions``  — length ``len(prompts) * num_completions``
    * ``flat_rewards``      — same length; the verifier's component dicts
    * ``raw_results``       — the per-problem ``SamplingResult`` objects for
                              callers that want search-cost diagnostics.
    """
    raw_results = strategy.sample_batch(prompts, records)
    flat_completions: List[str] = []
    flat_rewards: List[Dict[str, float]] = []
    for res in raw_results:
        comps, rewards = select_for_pass_at_k(res, num_completions)
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
