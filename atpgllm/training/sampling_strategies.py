"""Inference policies for ATPG evaluation.

Version conversation-search-v1 uses structured assistant/tool histories,
per-slot budgets, independent request seeds, and explicit final-answer scoring.
Training reward parsing is unchanged. Random remains a model-free PI/PO baseline.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from ._paths import ensure_data_preprocessing_on_path
ensure_data_preprocessing_on_path()
from fault_sim import OptimizedNetlist
from .reward_function_factory import RewardFunctionFactory
from .search_types import CompletionScore, SamplingResult, SearchConfig, PROTOCOL_VERSION, stable_seed
from .search_verifier import Verifier
from .search_backends import HFGenerator, VLLMGenerator, make_hf_generator, make_vllm_generator
from .search_policies import SamplingStrategy, BestOfNStrategy, MCTSStrategy, EvolutionaryStrategy

def _netlist_text_from_record(record: Dict[str, Any]) -> str:
    """Extract Verilog text from a dataset ``netlist`` field (str or payload)."""
    field = record.get("netlist", "")
    if isinstance(field, dict):
        return str(field.get("netlist", "") or "")
    return str(field or "")


def sample_bit_string(n_bits: int, rng: random.Random) -> str:
    """
    Draw ``u ~ Unif{0, …, 2^{n}-1}`` and return its ``n``-bit binary string
    (MSB on the left). Uses ``getrandbits`` so large ``n`` never enumerates.
    """
    if n_bits <= 0:
        return ""
    return format(rng.getrandbits(n_bits), f"0{n_bits}b")


def format_nets_bit_assignment(nets: List[str], bit_str: str) -> str:
    """
    Assign bits to nets in declaration / ``OptimizedNetlist`` order.

    Matches the gold ``answer_content`` style: ``net: 0, net: 1, …``.
    """
    if len(bit_str) < len(nets):
        bit_str = bit_str.zfill(len(nets))
    elif len(bit_str) > len(nets):
        bit_str = bit_str[-len(nets) :]
    return ", ".join(f"{net}: {bit}" for net, bit in zip(nets, bit_str))


def format_random_answer(
    input_nets: List[str],
    output_nets: List[str],
    fault: str,
    rng: random.Random,
) -> str:
    """
    Minimal scorable completion: INPUT_VECTOR / EXPECTED_OUTPUT /
    DETECTED_FAULTS (same keys the reward factory parses).
    """
    iv = format_nets_bit_assignment(
        input_nets, sample_bit_string(len(input_nets), rng),
    )
    eo = format_nets_bit_assignment(
        output_nets, sample_bit_string(len(output_nets), rng),
    )
    df = (fault or "").strip()
    return (
        f'INPUT_VECTOR: "{iv}"\n'
        f'EXPECTED_OUTPUT: "{eo}"\n'
        f'DETECTED_FAULTS: "{df}"'
    )


def format_vector_answer(vector, expected, fault):
    inputs = ", ".join(f"{k}: {v}" for k, v in vector.items())
    outputs = ", ".join(f"{k}: {v}" for k, v in expected.items())
    return f'INPUT_VECTOR: "{inputs}"\nEXPECTED_OUTPUT: "{outputs}"\nDETECTED_FAULTS: "{fault}"'


class RandomStrategy:
    """Model-free independent PI/PO draws, with the common verifier and usage."""
    name = "random"

    def __init__(self, verifier, num_completions=1, seed=42, *, budget=1,
                 threshold_mode="fault_detected", search_config=None):
        from .search_types import accepted
        if num_completions < 1 or budget < 1:
            raise ValueError("num_completions and budget must be positive")
        accepted({}, threshold_mode)
        self.verifier, self.num_completions, self.seed = verifier, num_completions, seed
        self.budget, self.mode = budget, threshold_mode
        self.config = SearchConfig.load(search_config)

    def sample_batch(self, prompts, records):
        from dataclasses import asdict
        from .search_types import SearchContext, ConversationState, BudgetExceeded, Usage, accepted
        if len(prompts) != len(records):
            raise ValueError("Prompts and records must align")
        results = []
        for prompt, record in zip(prompts, records):
            completions, scores, slots = [], [], []
            try:
                problem = self.verifier.problem(prompt, record)
            except Exception as exc:
                results.append(SamplingResult([""] * self.num_completions,
                    [self.verifier.failure("INFRA_ERROR") for _ in range(self.num_completions)], False, 0,
                    [{"status": "INFRA_ERROR", "error": str(exc), "completion_slot": i,
                      "usage": asdict(Usage())} for i in range(self.num_completions)]))
                continue
            for slot in range(self.num_completions):
                context = SearchContext(stable_seed(self.seed, problem.problem_id, slot), self.config, self.budget)
                population, seen, best = [], set(), None
                last_failure = self.verifier.failure("EXHAUSTED")
                while context.begin_attempt():
                    rng = context.rng
                    vector = {k: rng.getrandbits(1) for k in problem.input_nets}
                    if self.name == "vector_evolutionary" and len(population) >= min(self.config.population_size, self.budget):
                        ranked = sorted(population, key=lambda pair: pair[1].rank(self.mode), reverse=True)
                        vector = dict(rng.choice(ranked[:max(1, len(ranked) // 2)])[0])
                        if len(ranked) > 1 and rng.random() < 0.5:
                            donor = rng.choice(ranked)[0]
                            vector = {k: donor[k] if rng.random() < 0.5 else v for k, v in vector.items()}
                        if vector:
                            bit = rng.choice(list(vector))
                            vector[bit] ^= 1
                    if tuple(vector.items()) in seen and self.name == "vector_evolutionary":
                        continue
                    seen.add(tuple(vector.items()))
                    expected = {k: rng.getrandbits(1) for k in problem.output_nets}
                    text = format_vector_answer(vector, expected, problem.fault)
                    if self.name == "vector_evolutionary":
                        try:
                            frame, _ = self.verifier.simulate(problem, vector, context)
                            expected = {k: int(frame.loc[k, "Good Machine"]) for k in problem.output_nets}
                            text = format_vector_answer(vector, expected, problem.fault)
                        except BudgetExceeded as exc:
                            last_failure = self.verifier.failure("EXHAUSTED")
                            context.stop_reason = str(exc)
                            break
                        except Exception as exc:
                            last_failure = self.verifier.failure("INFRA_ERROR")
                            context.usage.infrastructure_errors += 1
                            context.event(kind="simulator_error", error=str(exc))
                            if context.usage.infrastructure_errors > self.config.infrastructure_retry_limit:
                                context.stop_reason = "infrastructure_retry_limit"
                                break
                            continue
                    state = ConversationState(status="FINAL", final_answer=text, readable=text)
                    score, actual_vector = self.verifier.score_state(problem, state, context)
                    population.append((actual_vector or vector, score))
                    population = sorted(population, key=lambda pair: pair[1].rank(self.mode), reverse=True)[:self.config.population_size]
                    if best is None or score.rank(self.mode) > best[1].rank(self.mode):
                        best = (text, score)
                    if accepted(score.components, self.mode):
                        context.stop_reason = "accepted"
                        break
                    if score.status == "EXHAUSTED":
                        break
                    if score.status == "INFRA_ERROR" and context.usage.infrastructure_errors > self.config.infrastructure_retry_limit:
                        context.stop_reason = "infrastructure_retry_limit"
                        break
                text, score = best or ("", last_failure)
                completions.append(text)
                scores.append(score)
                slots.append({"completion_slot": slot, "seed": context.seed, "problem_id": problem.problem_id,
                    "status": score.status, "stop_reason": context.stop_reason, "usage": asdict(context.usage),
                    "reward_components": score.components, "unique_vectors": len(seen),
                    "answer_source": "simulator" if self.name == "vector_evolutionary" else "random"})
            results.append(SamplingResult(completions, scores, any(s.detected for s in scores), 0, slots))
        return results


class VectorEvolutionaryStrategy(RandomStrategy):
    """Vector-only genetic baseline; reported PO values come from simulation."""
    name = "vector_evolutionary"


STRATEGY_REGISTRY = {
    "best_of_n": BestOfNStrategy,
    "mcts": MCTSStrategy,
    "evolutionary": EvolutionaryStrategy,
    "random": RandomStrategy,
    "vector_evolutionary": VectorEvolutionaryStrategy,
}
MODEL_FREE_STRATEGY_NAMES = frozenset({"random", "vector_evolutionary"})


def list_available_strategies():
    return sorted(STRATEGY_REGISTRY)


def make_strategy(name, generator, verifier, *, num_completions, width=None,
                  use_tools=False, max_tool_rounds=1, seed=42,
                  threshold_mode="fault_detected", search_config=None):
    if name in MODEL_FREE_STRATEGY_NAMES:
        if name == "vector_evolutionary" and (width is None or width < 1):
            raise ValueError("vector_evolutionary requires a positive budget")
        return STRATEGY_REGISTRY[name](verifier, num_completions=num_completions, seed=seed,
            budget=width if name == "vector_evolutionary" else 1,
            threshold_mode=threshold_mode, search_config=search_config)
    if name not in STRATEGY_REGISTRY and name != "greedy":
        raise ValueError(f"Unknown sampling strategy: {name}")
    if generator is None:
        raise ValueError(f"{name} requires a model generator")
    if name != "greedy" and (width is None or width < 1):
        raise ValueError(f"{name} requires a positive search width")
    cls = SamplingStrategy if name == "greedy" else STRATEGY_REGISTRY[name]
    return cls(generator, verifier, num_completions=num_completions,
               budget=1 if name == "greedy" else width, use_tools=use_tools,
               max_tool_rounds=max_tool_rounds, seed=seed,
               threshold_mode=threshold_mode, search_config=search_config)


def select_for_pass_at_k(result, num_completions):
    if len(result.completions) != num_completions or len(result.scores) != num_completions:
        raise RuntimeError("Search did not return exactly N aligned slots")
    return list(result.completions), [s.components for s in result.scores]


def run_strategy_batch(strategy, prompts, records, num_completions):
    results = strategy.sample_batch(prompts, records)
    if len(results) != len(prompts):
        raise RuntimeError("Search did not return one result per problem")
    flat, rewards = [], []
    for result in results:
        texts, components = select_for_pass_at_k(result, num_completions)
        flat.extend(texts)
        rewards.extend(components)
    return flat, rewards, results
