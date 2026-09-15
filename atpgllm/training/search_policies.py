"""Search policies over complete assistant/tool actions.

Each scheduler sweep admits at most one attempt per independent slot. vLLM
batches those slots while every slot retains its own seeds, cache and budget.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

from .completion_runner import CompletionRunner
from .search_types import (Candidate, ConversationState, CompletionScore, SearchConfig,
                           SearchContext, SamplingResult, Usage, accepted, stable_seed)


@dataclass
class Node:
    state: ConversationState
    parent: Node | None = None
    children: list = field(default_factory=list)
    visits: int = 0
    total: float = 0.0
    prior: float = 1.0
    closed: bool = False
    duplicates: int = 0
    candidate: Candidate | None = None


@dataclass
class Slot:
    problem: object
    initial: ConversationState
    context: SearchContext
    index: int
    root: Node
    best: Candidate | None = None
    population: list = field(default_factory=list)
    unique_vectors: set = field(default_factory=set)
    stagnant: int = 0
    infrastructure_failures: int = 0
    done: bool = False
    started: float = field(default_factory=time.monotonic)


class SamplingStrategy:
    name = "greedy"

    def __init__(self, generator, verifier, num_completions=1, *, budget=1, width=None,
                 use_tools=True, max_tool_rounds=1, seed=42, threshold_mode="fault_detected",
                 search_config=None):
        if num_completions < 1 or (width if width is not None else budget) < 1:
            raise ValueError("num_completions and search budget must be positive")
        accepted({}, threshold_mode)
        self.gen, self.verifier = generator, verifier
        self.num_completions = num_completions
        self.budget = width if width is not None else budget
        self.seed, self.mode = seed, threshold_mode
        self.config = SearchConfig.load(search_config)
        self.runner = CompletionRunner(generator, verifier,
            max_tool_rounds=max_tool_rounds if use_tools else 0,
            temperature=generator.temperature, top_p=generator.top_p)

    def plan(self, slot):
        return slot.initial, self.gen.temperature, "seed", None

    def update(self, slot, candidate, parent):
        pass

    def _record(self, slot, candidate, parent):
        if slot.best is None or candidate.score.rank(self.mode) > slot.best.score.rank(self.mode):
            slot.best = candidate
        if candidate.vector is not None:
            slot.unique_vectors.add(tuple(candidate.vector.items()))
        self.update(slot, candidate, parent)
        if self.config.save_trace:
            slot.context.event(kind="candidate", operator=candidate.operator, vector=candidate.vector,
                               status=candidate.score.status, components=candidate.score.components,
                               state_id=stable_seed(candidate.state.messages, candidate.state.open_text),
                               selected_node_id=stable_seed(parent.state.messages, parent.state.open_text)
                                   if parent is not None else None,
                               parent_state_id=stable_seed(parent.parent.state.messages, parent.parent.state.open_text)
                                   if parent is not None and parent.parent is not None else None,
                               messages=list(candidate.state.messages), open_text=candidate.state.open_text,
                               final_answer=candidate.state.final_answer, observations=list(candidate.state.observations))
        if accepted(candidate.score.components, self.mode):
            slot.context.stop_reason = "accepted"
            slot.done = True
        elif candidate.score.status == "EXHAUSTED":
            reason = candidate.state.reason or slot.context.stop_reason
            # Per-path caps close only that path. Other branches can still fit
            # the slot's remaining resource budget.
            if reason in ("simulator_request_limit", "simulator_execution_limit") or slot.context.remaining_tokens() <= 0:
                slot.context.stop_reason = reason
                slot.done = True
        elif candidate.score.status == "INFRA_ERROR":
            slot.infrastructure_failures += 1
            if slot.infrastructure_failures > self.config.infrastructure_retry_limit:
                slot.context.stop_reason = "infrastructure_retry_limit"
                slot.done = True

    def sample_batch(self, prompts, records):
        if len(prompts) != len(records):
            raise ValueError("Prompts and records must align")
        slots, failed = [], {}
        for problem_index, (prompt, record) in enumerate(zip(prompts, records)):
            try:
                problem = self.verifier.problem(prompt, record)
                initial = self.runner.initial(prompt, record.get("_search_messages"))
            except Exception as exc:
                failed[problem_index] = str(exc)
                continue
            for i in range(self.num_completions):
                context = SearchContext(stable_seed(self.seed, problem.problem_id, i), self.config, self.budget)
                slots.append(Slot(problem, initial, context,
                                  problem_index * self.num_completions + i, Node(initial)))
        while True:
            active, jobs, plans = [], [], []
            for slot in slots:
                if slot.done:
                    continue
                if not slot.context.begin_attempt():
                    slot.done = True
                    if slot.context.remaining_tokens() <= 0:
                        slot.context.stop_reason = "token_limit"
                    continue
                plan = self.plan(slot)
                if plan is None:
                    slot.context.usage.attempts -= 1
                    slot.context.stop_reason = "frontier_closed"
                    slot.done = True
                    continue
                state, temperature, operator, parent = plan
                active.append(slot)
                jobs.append((state, slot.problem, slot.context, temperature))
                plans.append((operator, parent))
            if not jobs:
                break
            if self.name == "mcts":
                states = self.runner.advance_many(jobs)
                for index, (slot, state, (_, parent)) in enumerate(zip(active, states, plans)):
                    child = Node(state, parent, closed=state.status in ("FINAL", "INVALID", "EXHAUSTED", "INFRA_ERROR"))
                    identity = repr(state.messages) + state.open_text
                    duplicate = any(repr(c.state.messages) + c.state.open_text == identity for c in parent.children)
                    if duplicate:
                        parent.duplicates += 1
                    else:
                        parent.children.append(child)
                        self._priors(parent)
                    # Backup is to the actual selected edge. Duplicate actions
                    # reuse the existing node, not an unreachable new child.
                    if duplicate:
                        child = next(c for c in parent.children if repr(c.state.messages) + c.state.open_text == identity)
                    plans[index] = ("mcts", child)
                jobs = [(s, *job[1:]) for s, job in zip(states, jobs)]
            results = self.runner.complete_many(jobs)
            for slot, (state, checkpoints), (operator, parent), job in zip(active, results, plans, jobs):
                if self.name == "mcts" and job[0].status == "READY" and job[0].observations:
                    checkpoints = (job[0],) + checkpoints
                if self.name == "mcts" and parent.closed and parent.candidate is not None:
                    # A duplicate proposal costs generation, but is not another
                    # independent observation of the same terminal outcome.
                    self._record(slot, parent.candidate, None)
                    continue
                score, vector = self.verifier.score_state(slot.problem, state, slot.context)
                candidate = Candidate(state, score, vector, checkpoints, operator)
                if self.name == "mcts" and parent.closed:
                    parent.candidate = candidate
                self._record(slot, candidate, parent)
        output = []
        by_index = {slot.index: slot for slot in slots}
        for p_idx in range(len(prompts)):
            completions, scores, diagnostics = [], [], []
            for i in range(self.num_completions):
                slot = by_index.get(p_idx * self.num_completions + i)
                if slot is None:
                    completions.append("")
                    scores.append(self.verifier.failure("INFRA_ERROR"))
                    diagnostics.append({"status": "INFRA_ERROR", "error": failed[p_idx],
                                        "completion_slot": i, "usage": asdict(Usage())})
                    continue
                candidate = slot.best or Candidate(slot.initial, self.verifier.failure("EXHAUSTED"))
                completions.append(candidate.state.readable)
                scores.append(candidate.score)
                diagnostics.append({
                    "completion_slot": i, "problem_id": slot.problem.problem_id,
                    "seed": slot.context.seed, "status": candidate.score.status,
                    "stop_reason": slot.context.stop_reason, "usage": asdict(slot.context.usage),
                    "unique_vectors": len(slot.unique_vectors), "final_answer": candidate.state.final_answer,
                    "reward_components": candidate.score.components, "vector": candidate.vector,
                    "simulation_identity": getattr(slot.problem, "simulation_identity", None),
                    "messages": list(candidate.state.messages), "observations": list(candidate.state.observations),
                    "elapsed_seconds": time.monotonic() - slot.started,
                    "trace": slot.context.trace if self.config.save_trace else None,
                })
            output.append(SamplingResult(completions, scores, any(s.detected for s in scores),
                sum(d.get("usage", {}).get("generation_requests", 0) for d in diagnostics), diagnostics))
        return output


class BestOfNStrategy(SamplingStrategy):
    name = "best_of_n"


class MCTSStrategy(SamplingStrategy):
    name = "mcts"

    def _priors(self, node):
        values = [c.state.mean_logprob for c in node.children]
        if self.config.prior == "lm" and all(v is not None and math.isfinite(v) for v in values):
            hi = max(values)
            weights = [math.exp(v - hi) for v in values]
        else:
            weights = [1.0] * len(values)
        for child, weight in zip(node.children, weights):
            child.prior = weight / sum(weights)

    def plan(self, slot):
        node = slot.root
        while not node.closed:
            allowed = min(self.config.max_children, max(2, math.ceil(math.sqrt(1 + node.visits))))
            if len(node.children) < allowed and node.duplicates < self.config.duplicate_limit:
                return node.state, self.gen.temperature, "mcts", node
            open_children = [c for c in node.children if not c.closed]
            if not open_children:
                # Nodes at the current widening limit can be exhausted before
                # further visits: admit another alternative up to the hard cap.
                if len(node.children) < self.config.max_children and node.duplicates < self.config.duplicate_limit:
                    return node.state, self.gen.temperature, "mcts", node
                node.closed = True
                if node.parent is None:
                    return None
                node = node.parent
                continue
            def value(child):
                if child.visits == 0:
                    return math.inf
                return child.total / child.visits + self.config.c_puct * child.prior * math.sqrt(1 + node.visits) / (1 + child.visits)
            node = max(open_children, key=value)
        return None

    def update(self, slot, candidate, node):
        # Infrastructure failures do not supply negative evidence about a vector.
        if candidate.score.status == "INFRA_ERROR":
            return
        value = candidate.score.value(self.mode)
        while node is not None:
            node.visits += 1
            node.total += value
            node = node.parent


class EvolutionaryStrategy(SamplingStrategy):
    name = "evolutionary"

    def plan(self, slot):
        cfg, rng = self.config, slot.context.rng
        attempt = slot.context.usage.attempts - 1
        if attempt < min(cfg.population_size, self.budget) or not slot.population:
            return slot.initial, cfg.seed_temperatures[attempt % len(cfg.seed_temperatures)], "seed", None
        parents = sorted(slot.population, key=lambda c: c.score.rank(self.mode), reverse=True)
        parent = rng.choice(parents[:max(1, (len(parents) + 1) // 2)])
        # For an already detecting vector, repair the answer without revising
        # the vector or disclosing a private verifier snapshot to the model.
        if parent.score.detected and not accepted(parent.score.components, self.mode):
            text = "Keep INPUT_VECTOR unchanged. Correct the final EXPECTED_OUTPUT and DETECTED_FAULTS using the conversation. Return only the final answer."
            return self.runner.instruct(parent.state, text), self.gen.temperature, "answer_repair", None
        weights = list(cfg.operator_weights)
        if not cfg.controller_edits:
            weights[1] = weights[2] = 0
        if slot.stagnant >= 2 * cfg.population_size:
            weights[3] += sum(weights)
        if sum(weights) == 0:
            weights[3] = 1
        operator = rng.choices(["feedback", "mutation", "crossover", "seed"], weights=weights)[0]
        if operator == "feedback":
            points = [s for s in parent.checkpoints if s.tool_rounds < self.runner.max_tool_rounds]
            if points:
                state = rng.choice(points)
                return self.runner.instruct(state, "Review the simulator result. Revise the input if needed, or return the final answer."), self.gen.temperature, operator, None
            operator = "seed"
        if operator in ("mutation", "crossover") and parent.vector:
            vector = dict(parent.vector)
            if operator == "crossover":
                donors = [c for c in parents if c.vector and c.vector != vector]
                donor = rng.choice(donors).vector if donors else vector
                differing = [name for name in vector if donor[name] != vector[name]]
                if len(differing) >= 2:
                    for name in rng.sample(differing, rng.randint(1, len(differing) - 1)):
                        vector[name] = donor[name]
                else:
                    operator = "mutation"
            if operator == "mutation":
                for name in rng.sample(list(vector), rng.randint(1, min(3, len(vector)))):
                    vector[name] ^= 1
            if tuple(vector.items()) not in slot.unique_vectors:
                proposed = ", ".join(f"{k}: {v}" for k, v in vector.items())
                instruction = (f"Consider this candidate input assignment: {proposed}. "
                               "Check it and produce a consistent tool request and final answer. You may revise the candidate.")
                state = self.runner.instruct(slot.initial, instruction)
                temp = cfg.mutation_temperature if operator == "mutation" else cfg.crossover_temperature
                return state, temp, operator, None
        return slot.initial, rng.choice(cfg.seed_temperatures), "seed", None

    def update(self, slot, candidate, parent):
        population = slot.population + [candidate]
        ranked = sorted(population, key=lambda c: c.score.rank(self.mode), reverse=True)
        selected, seen = [], set()
        for c in ranked:
            if c.vector is None:
                continue
            key = tuple(c.vector.items())
            if key not in seen:
                selected.append(c)
                seen.add(key)
        # Retain the best, then balance quality ties with PI Hamming distance.
        diverse = selected[:1]
        pool = selected[1:]
        while pool and len(diverse) < self.config.population_size:
            def priority(c):
                distance = min(sum(c.vector[k] != d.vector[k] for k in c.vector) / max(1, len(c.vector)) for d in diverse)
                return c.score.rank(self.mode)[:4], distance, c.score.scalar
            best = max(pool, key=priority)
            diverse.append(best)
            pool.remove(best)
        old_vectors = {tuple(c.vector.items()) for c in slot.population if c.vector}
        slot.stagnant = slot.stagnant + 1 if candidate.vector is None or tuple(candidate.vector.items()) in old_vectors else 0
        # Give a novel valid arrival a parent position even if the established
        # elites have slightly better auxiliary scores. The best is preserved.
        if (self.config.population_size > 1 and candidate.vector is not None
                and candidate.score.status == "FINAL" and tuple(candidate.vector.items()) not in old_vectors
                and all(c is not candidate for c in diverse)):
            diverse = diverse[:self.config.population_size - 1] + [candidate]
        slot.population = diverse
