"""GDPO advantage computation for ATPG multi-objective rewards."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor


GDPO_EPS = 1e-4


@dataclass(frozen=True)
class GDPOResult:
    """Intermediate values retained for diagnostics and semantic tests."""

    advantages: Tensor
    normalized_objectives: Tensor
    pre_batch_advantages: Tensor
    active_group_objectives: Tensor
    valid_rows: Tensor


def reward_output_to_tensors(
    output_reward_func: list,
    device: torch.device,
    *,
    logonly_suffix: str = "_logonly",
) -> tuple[Tensor, Tensor | None, list[str] | None]:
    """Convert scalar or component-dict reward outputs to tensors."""
    first_dict = next((row for row in output_reward_func if isinstance(row, dict)), None)
    if first_dict is None:
        scalars = [
            float("nan")
            if row is None or (isinstance(row, float) and math.isnan(row))
            else float(row)
            for row in output_reward_func
        ]
        return torch.tensor(scalars, dtype=torch.float32, device=device), None, None

    component_keys = sorted(first_dict)
    scalar_rows: list[float] = []
    component_rows: list[list[float]] = []
    for row in output_reward_func:
        if isinstance(row, dict):
            scalar_rows.append(
                float(
                    sum(
                        value
                        for key, value in row.items()
                        if not str(key).endswith(logonly_suffix)
                    )
                )
            )
            component_rows.append(
                [float(row.get(key, 0.0)) for key in component_keys]
            )
        elif row is None or (isinstance(row, float) and math.isnan(row)):
            scalar_rows.append(float("nan"))
            component_rows.append([float("nan")] * len(component_keys))
        else:
            scalar_rows.append(float(row))
            component_rows.append([float("nan")] * len(component_keys))

    return (
        torch.tensor(scalar_rows, dtype=torch.float32, device=device),
        torch.tensor(component_rows, dtype=torch.float32, device=device),
        component_keys,
    )


def select_objective_columns(
    component_matrix: Tensor,
    component_keys: Sequence[str],
    objective_keys: Sequence[str],
) -> Tensor:
    """Select objective columns in an explicit, stable order."""
    if component_matrix.ndim != 2:
        raise ValueError("component_matrix must have shape (completions, components)")
    if component_matrix.shape[1] != len(component_keys):
        raise ValueError("component key count does not match component_matrix width")

    key_to_index = {key: idx for idx, key in enumerate(component_keys)}
    missing = [key for key in objective_keys if key not in key_to_index]
    if missing:
        raise ValueError(f"Missing GDPO objective component(s): {missing}")
    return component_matrix[:, [key_to_index[key] for key in objective_keys]]


def compute_gdpo_advantages(
    rewards_per_objective: Tensor,
    num_generations: int,
    objective_weights: Sequence[float] | Tensor,
    *,
    eps: float = GDPO_EPS,
) -> GDPOResult:
    """Compute group reward-decoupled, globally normalized advantages.

    Non-finite entries are missing observations. They are excluded from
    per-objective statistics. A group/objective slice needs at least two valid,
    non-constant values to contribute. Rows with no valid objectives are
    excluded from final batch statistics and receive zero advantage.
    """
    if rewards_per_objective.ndim != 2:
        raise ValueError(
            "rewards_per_objective must have shape (completions, objectives)"
        )
    if num_generations < 1:
        raise ValueError("num_generations must be at least 1")

    num_rows, num_objectives = rewards_per_objective.shape
    if num_rows == 0 or num_rows % num_generations != 0:
        raise ValueError(
            "completion count must be non-zero and divisible by num_generations"
        )

    rewards = rewards_per_objective.to(dtype=torch.float32)
    weights = torch.as_tensor(
        objective_weights, dtype=torch.float32, device=rewards.device
    )
    if weights.ndim != 1 or weights.numel() != num_objectives:
        raise ValueError("objective_weights must contain one value per objective")
    if not torch.isfinite(weights).all():
        raise ValueError("objective_weights must be finite")

    grouped = rewards.reshape(-1, num_generations, num_objectives)
    valid = torch.isfinite(grouped)
    counts = valid.sum(dim=1, keepdim=True)

    safe_values = torch.where(valid, grouped, torch.zeros_like(grouped))
    means = safe_values.sum(dim=1, keepdim=True) / counts.clamp(min=1)
    centered = torch.where(valid, grouped - means, torch.zeros_like(grouped))

    # GDPO and TRL use sample standard deviation (correction=1).
    variances = centered.square().sum(dim=1, keepdim=True) / (counts - 1).clamp(min=1)
    stds = variances.sqrt()
    active = (counts > 1) & (stds > eps)

    normalized = torch.where(
        valid & active,
        centered / (stds + eps),
        torch.zeros_like(centered),
    )
    normalized_flat = normalized.reshape(num_rows, num_objectives)
    pre_batch = (normalized_flat * weights.unsqueeze(0)).sum(dim=1)

    valid_rows = valid.any(dim=2).reshape(num_rows)
    final = torch.zeros_like(pre_batch)
    valid_pre_batch = pre_batch[valid_rows]
    if valid_pre_batch.numel() > 1:
        batch_std = valid_pre_batch.std(unbiased=True)
        if torch.isfinite(batch_std) and batch_std > eps:
            final[valid_rows] = (
                valid_pre_batch - valid_pre_batch.mean()
            ) / (batch_std + eps)

    return GDPOResult(
        advantages=final,
        normalized_objectives=normalized_flat,
        pre_batch_advantages=pre_batch,
        active_group_objectives=active.squeeze(1),
        valid_rows=valid_rows,
    )
