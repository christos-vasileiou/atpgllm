import pytest
import torch

from atpgllm.training.gdpo import (
    compute_gdpo_advantages,
    reward_output_to_tensors,
    select_objective_columns,
)


def test_gdpo_preserves_reward_combinations_hidden_by_scalar_grpo():
    # Group 1 totals are (0, 1); group 2 totals are (0, 2). Ordinary
    # group-standardized GRPO maps both pairs to the same advantages.
    rewards = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    result = compute_gdpo_advantages(rewards, 2, [1.0, 1.0])

    assert result.advantages.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert result.advantages.std(unbiased=True).item() == pytest.approx(
        1.0, abs=2e-4
    )
    assert abs(result.advantages[2]) > abs(result.advantages[0])
    assert abs(result.advantages[3]) > abs(result.advantages[1])


def test_raw_positive_scales_do_not_change_gdpo_advantages():
    rewards = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.2, 0.1],
            [0.8, 0.9],
        ]
    )

    base = compute_gdpo_advantages(rewards, 2, [1.0, 0.25]).advantages
    rescaled = compute_gdpo_advantages(
        rewards * torch.tensor([12.0, 0.05]), 2, [1.0, 0.25]
    ).advantages

    # The fixed epsilon makes this approximate for very small raw scales.
    torch.testing.assert_close(base, rescaled, atol=2e-3, rtol=2e-3)


def test_weights_are_applied_after_per_objective_normalization():
    rewards = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    detection_only = compute_gdpo_advantages(rewards, 2, [1.0, 0.0])
    both = compute_gdpo_advantages(rewards, 2, [1.0, 1.0])

    assert detection_only.pre_batch_advantages[0] < 0
    assert both.pre_batch_advantages[0] == pytest.approx(0.0, abs=1e-6)


def test_constant_and_insufficient_objectives_contribute_zero():
    rewards = torch.tensor(
        [
            [1.0, 0.0, float("nan")],
            [1.0, 1.0, 2.0],
            [1.0, 0.0, float("nan")],
            [1.0, 1.0, float("nan")],
        ]
    )

    result = compute_gdpo_advantages(rewards, 2, [1.0, 1.0, 1.0])

    assert not result.active_group_objectives[:, 0].any()
    assert not result.active_group_objectives[:, 2].any()
    assert torch.isfinite(result.advantages).all()
    assert result.normalized_objectives[:, 0].eq(0).all()
    assert result.normalized_objectives[:, 2].eq(0).all()


def test_all_missing_row_is_excluded_and_receives_zero():
    rewards = torch.tensor(
        [
            [float("nan"), float("nan")],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    result = compute_gdpo_advantages(rewards, 2, [1.0, 1.0])

    assert not result.valid_rows[0]
    assert result.advantages[0] == 0
    assert torch.isfinite(result.advantages).all()


def test_all_constant_rewards_return_finite_zeros():
    rewards = torch.ones(4, 3)
    result = compute_gdpo_advantages(rewards, 2, [1.0, 0.25, 0.1])
    assert result.advantages.eq(0).all()


def test_objective_selection_uses_explicit_order():
    components = torch.tensor([[1.0, 2.0, 3.0]])
    selected = select_objective_columns(
        components,
        ["format", "detection", "metric_logonly"],
        ["detection", "format"],
    )
    torch.testing.assert_close(selected, torch.tensor([[2.0, 1.0]]))

    with pytest.raises(ValueError, match="Missing GDPO objective"):
        select_objective_columns(components, ["a", "b", "c"], ["detection"])


def test_component_dict_conversion_excludes_logonly_values_from_scalar():
    rows = [
        {"detection": 1.0, "format": 0.5, "mention_logonly": 100.0},
        {"detection": 0.0, "format": 0.0, "mention_logonly": 0.0},
    ]
    scalars, matrix, keys = reward_output_to_tensors(rows, torch.device("cpu"))

    torch.testing.assert_close(scalars, torch.tensor([1.5, 0.0]))
    assert matrix is not None
    assert keys == ["detection", "format", "mention_logonly"]


def test_parent_group_centering_is_noop_after_gdpo():
    rewards = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    advantages = compute_gdpo_advantages(rewards, 2, [1.0, 0.25]).advantages
    parent_centered = advantages - advantages.reshape(-1, 2).mean(
        dim=1
    ).repeat_interleave(2)
    torch.testing.assert_close(advantages, parent_centered, atol=1e-6, rtol=0)
