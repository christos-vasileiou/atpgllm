"""Reward-ranked training rollouts, selected before policy/reference forwards.

This is a biased rollout-selection variant of GRPO, not an importance-corrected
estimator of the ordinary on-policy objective. vLLM's likelihood ranking is not
used: candidates finish their tool trajectories and are scored by the configured
reward functions (including the selected fault simulator).
"""

from __future__ import annotations

import copy
import math


def validate_best_of_n(candidates: int, retained: int) -> None:
    if retained < 2:
        raise ValueError("Training requires NUM_GENERATIONS >= 2")
    if candidates and (candidates < retained or candidates % retained):
        raise ValueError("TRAIN_BEST_OF_N must be 0 or a multiple of NUM_GENERATIONS >= NUM_GENERATIONS")


def select_best_indices(scores, candidates: int, retained: int) -> list[int]:
    """Stable top-G per contiguous prompt group, including groups spanning ranks."""
    validate_best_of_n(candidates, retained)
    if not candidates or not scores or len(scores) % candidates:
        raise ValueError("Best-of-N requires complete candidate groups")
    selected = []
    for start in range(0, len(scores), candidates):
        valid = [i for i in range(start, start + candidates) if math.isfinite(scores[i])]
        if len(valid) < retained:
            raise ValueError("Best-of-N group has fewer than G finite simulator rewards")
        selected.extend(sorted(valid, key=lambda i: -scores[i])[:retained])
    return selected


def select_rollouts(bundles, indices):
    """Move complete trajectories together; masks/logprobs must never be reranked separately."""
    columns = []
    for column in (0, 1, 2, 3, 5):
        values = [bundle[column] for bundle in bundles]
        if all(value is None for value in values):
            columns.append(None)
        elif any(value is None for value in values):
            raise ValueError("Inconsistent rollout fields across ranks")
        else:
            flat = [row for value in values for row in value]
            columns.append([flat[i] for i in indices])
    extra = {}
    for key in bundles[0][6]:
        values = [bundle[6][key] for bundle in bundles]
        if isinstance(values[0], list):
            if any(len(value) != len(bundle[0]) for value, bundle in zip(values, bundles)):
                raise ValueError(f"Rollout extra field {key} is not per-completion")
            flat = [row for value in values for row in value]
            extra[key] = [flat[i] for i in indices]
        else:
            extra[key] = values[0]
    prompt_ids, completion_ids, masks, completions, logprobs = columns
    total = sum(sum(mask) for mask in masks) if masks is not None else sum(map(len, completion_ids))
    return prompt_ids, completion_ids, masks, completions, total, logprobs, extra


class BestOfNTrainingMixin:
    """Shared by both adapter lifecycles; evaluation never enters selection."""

    def _generate_and_score_completions(self, inputs):
        candidates = getattr(self, "train_best_of_n", 0)
        validate_best_of_n(candidates, self.num_generations)
        enabled = self.model.training and candidates > self.num_generations
        self._best_of_inputs = inputs if enabled else None
        try:
            return super()._generate_and_score_completions(inputs)
        finally:
            self._best_of_inputs = None

    def _generate_best_of_n(self, prompts):
        import torch
        from accelerate.utils import gather_object

        retained = self.num_generations
        candidates = self.train_best_of_n
        factor = candidates // retained
        expanded_inputs = [copy.deepcopy(row) for row in self._best_of_inputs for _ in range(factor)]
        expanded_prompts = [copy.deepcopy(prompt) for prompt in prompts for _ in range(factor)]
        self._ranking_candidates = True
        self.num_generations = candidates
        try:
            bundle = self._generate(expanded_prompts)
            for key, values in bundle[6].items():
                for i, row in enumerate(expanded_inputs):
                    row[key] = values[i] if isinstance(values, list) else values
            # The trainers return raw weighted objectives here, before GDPO
            # normalization. Selected rows are scored normally by the parent.
            rewards = self._calculate_rewards(expanded_inputs, expanded_prompts, bundle[3], bundle[1])
            weighted = rewards * self.reward_weights.to(rewards.device).unsqueeze(0)
            scores = weighted.sum(dim=1).tolist()
        finally:
            self.num_generations = retained
            self._ranking_candidates = False

        indices = select_best_indices(scores, candidates, retained)
        # Accelerator's object gather flattens lists, hence the one-item wrapper.
        # The candidate token total may be a CUDA scalar; selection recomputes
        # its denominator, so do not serialize that device tensor across ranks.
        bundle = (*bundle[:4], 0, *bundle[5:])
        bundles = gather_object([bundle])
        selected = select_rollouts(bundles, indices)
        local_count = len(prompts)
        start = self.accelerator.process_index * local_count
        local_indices = list(range(start, start + local_count))
        local = list(select_rollouts([selected], local_indices))
        # DAPO's loss denominator is global and counts selected model tokens only.
        local[4] = torch.tensor(selected[4], device=self.accelerator.device)
        metrics = self._metrics["train"]
        finite_scores = [score for score in scores if math.isfinite(score)]
        for key, value in {
            "candidates_per_prompt": candidates,
            "retained_per_prompt": retained,
            "candidate_reward_mean": sum(finite_scores) / len(finite_scores),
            "selected_reward_mean": sum(scores[i] for i in indices) / len(indices),
        }.items():
            metrics.setdefault(f"sampling/best_of_n/{key}", []).append(value)
        return tuple(local)
