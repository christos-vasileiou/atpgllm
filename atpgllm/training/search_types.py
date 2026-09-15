"""Versioned contracts for inference search. No model or simulator imports."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "conversation-search-v1"
FINAL_STATUSES = {"FINAL", "INVALID", "EXHAUSTED", "INFRA_ERROR"}


def stable_seed(*parts: Any) -> int:
    raw = json.dumps(parts, sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def accepted(reward: dict, mode: str = "fault_detected") -> bool:
    if reward.get("search_failure_logonly", 0):
        return False
    def acc(key):
        return float(reward.get(key, reward.get(key + "_logonly", 0))) >= 1
    detected = acc("fault_detected_by_pred_input_vector_acc")
    if mode == "fault_detected":
        return detected
    if mode == "full_accuracy":
        return detected and all(acc(key) for key in (
            "input_vector_acc", "expected_output_acc", "detected_faults_acc"))
    if mode == "positive_reward":
        return sum(reward.values()) > 0
    raise ValueError(f"Unknown threshold mode: {mode}")


@dataclass(frozen=True)
class SearchConfig:
    max_generated_tokens: int = 32768
    max_simulator_requests: int = 32
    max_simulator_executions: int = 32
    finalization_tokens: int = 1024
    action_tokens: int = 2048
    max_actions: int = 32
    repair_limit: int = 1
    infrastructure_retry_limit: int = 1
    duplicate_limit: int = 3
    max_children: int = 8
    c_puct: float = 1.25
    prior: str = "uniform"
    population_size: int = 6
    seed_temperatures: tuple = (0.5, 0.8, 1.0)
    mutation_temperature: float = 1.0
    crossover_temperature: float = 0.7
    operator_weights: tuple = (0.5, 0.25, 0.15, 0.10)
    controller_edits: bool = True
    save_trace: bool = False

    def __post_init__(self):
        for name in ("max_generated_tokens", "max_simulator_requests",
                     "max_simulator_executions", "finalization_tokens", "action_tokens",
                     "max_actions", "duplicate_limit", "max_children", "population_size"):
            val = getattr(self, name)
            if type(val) is not int or val < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("repair_limit", "infrastructure_retry_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.finalization_tokens > self.max_generated_tokens:
            raise ValueError("finalization_tokens exceeds max_generated_tokens")
        if self.prior not in ("uniform", "lm"):
            raise ValueError("prior must be uniform or lm")
        for name in ("c_puct", "mutation_temperature", "crossover_temperature"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ValueError(f"Invalid {name}")
        if not self.seed_temperatures or any(
            isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0
            for t in self.seed_temperatures
        ):
            raise ValueError("seed_temperatures must contain finite nonnegative values")
        if len(self.operator_weights) != 4 or any(
            isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
            for w in self.operator_weights
        ) or sum(self.operator_weights) <= 0:
            raise ValueError("operator_weights requires four nonnegative weights with positive sum")
        if type(self.controller_edits) is not bool or type(self.save_trace) is not bool:
            raise ValueError("controller_edits and save_trace must be booleans")

    @classmethod
    def load(cls, value=None):
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            value = json.loads(Path(value).read_text())
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("search_config must be a JSON object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown search configuration keys: {sorted(unknown)}")
        return cls(**value)


@dataclass
class Usage:
    attempts: int = 0
    generated_tokens: int = 0
    prompt_tokens: int = 0
    generation_requests: int = 0
    simulator_requests: int = 0
    simulator_executions: int = 0
    cache_hits: int = 0
    tool_calls: int = 0
    infrastructure_errors: int = 0
    invalid_actions: int = 0
    discarded_tokens: int = 0
    generation_usage_unknown: int = 0


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class SearchContext:
    seed: int
    config: SearchConfig
    budget: int
    usage: Usage = field(default_factory=Usage)
    cache: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    stop_reason: str = "attempt_limit"

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def remaining_tokens(self):
        return self.config.max_generated_tokens - self.usage.generated_tokens

    def begin_attempt(self):
        if self.usage.attempts >= self.budget or self.remaining_tokens() <= 0:
            return False
        self.usage.attempts += 1
        return True

    def request_seed(self):
        seed = stable_seed(self.seed, "generation", self.usage.generation_requests)
        self.usage.generation_requests += 1
        return seed

    def event(self, **event):
        if self.config.save_trace:
            self.trace.append({**event, "usage": asdict(self.usage)})


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    seed: int
    logprobs: bool = False


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: tuple = ()
    finish_reason: str = "stop"
    mean_logprob: float | None = None
    prompt_tokens: int = 0
    error: str | None = None


@dataclass(frozen=True)
class ConversationState:
    # Messages are never mutated; runners copy before template serialization.
    messages: tuple = ()
    open_text: str = ""
    readable: str = ""
    status: str = "READY"
    tool_rounds: int = 0
    actions: int = 0
    repairs: int = 0
    final_answer: str = ""
    reason: str = ""
    mean_logprob: float | None = None
    observations: tuple = ()
    open_logprob_sum: float = 0.0
    open_logprob_tokens: int = 0
    open_logprob_missing: bool = False


@dataclass
class CompletionScore:
    detected: bool = False
    scalar: float = 0.0
    components: dict = field(default_factory=dict)
    status: str = "FINAL"

    def rank(self, mode):
        return (accepted(self.components, mode), self.detected,
                self.components.get("simulation_valid_logonly", 0),
                self.components.get("activation", 0), self.scalar)

    def value(self, mode):
        if accepted(self.components, mode):
            return 1.0
        if self.detected:
            return 0.8
        return 0.2 if self.components.get("activation", 0) else 0.0


@dataclass
class Candidate:
    state: ConversationState
    score: CompletionScore
    vector: dict | None = None
    checkpoints: tuple = ()
    operator: str = "seed"


@dataclass
class SamplingResult:
    completions: list
    scores: list
    detected_any: bool
    generator_calls: int
    slots: list = field(default_factory=list)
