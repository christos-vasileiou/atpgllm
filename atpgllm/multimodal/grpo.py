"""Small, testable GRPO objective used by graph-conditioned training."""

from __future__ import annotations

import torch
from torch import Tensor


def group_relative_advantages(rewards: Tensor, eps: float = 1e-6) -> Tensor:
    """Normalize rewards within one prompt's generation group."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("GRPO rewards must be a 1-D group with at least two values.")
    centered = rewards - rewards.mean()
    scale = rewards.std(unbiased=False)
    if scale <= eps:
        return torch.zeros_like(rewards)
    return centered / (scale + eps)


def graph_grpo_loss(
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    reference_log_probs: Tensor,
    completion_mask: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float = 0.2,
    beta: float = 0.03,
) -> Tensor:
    """Clipped sequence-level GRPO surrogate plus fixed-reference KL."""
    if policy_log_probs.shape != old_log_probs.shape:
        raise ValueError("policy_log_probs and old_log_probs must have equal shape.")
    if policy_log_probs.shape != reference_log_probs.shape:
        raise ValueError(
            "policy_log_probs and reference_log_probs must have equal shape."
        )
    if completion_mask.shape != policy_log_probs.shape:
        raise ValueError("completion_mask must match log-probability tensors.")
    if advantages.shape != policy_log_probs.shape[:1]:
        raise ValueError("advantages must contain one value per completion.")

    mask = completion_mask.to(policy_log_probs.dtype)
    lengths = mask.sum(dim=1).clamp(min=1.0)
    log_ratio = ((policy_log_probs - old_log_probs) * mask).sum(dim=1) / lengths
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    surrogate = torch.minimum(unclipped, clipped)

    ref_minus_policy = reference_log_probs - policy_log_probs
    per_token_kl = ref_minus_policy.exp() - ref_minus_policy - 1.0
    kl = (per_token_kl * mask).sum(dim=1) / lengths
    return -(surrogate - beta * kl).mean()
