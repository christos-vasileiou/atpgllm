"""Training selection invariants without loading vLLM or model weights."""
import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atpgllm.training.best_of_n import (
    BestOfNTrainingMixin, select_best_indices, select_rollouts, validate_best_of_n,
)


def bundle(ids):
    return ([[99] for i in ids], [[i, i] for i in ids], [[1, 0] for i in ids],
            [f"trajectory-{i}" for i in ids], len(ids),
            [[-i, 0] for i in ids], {"candidate": list(ids)})


def test_selection_crosses_rank_boundaries_and_preserves_trajectory_alignment():
    # Two prompt groups of six, partitioned into three ranks of four.
    indices = select_best_indices([0, 8, 1, 9, 2, 3, 7, 0, 6, 1, 5, 2], 6, 2)
    assert indices == [3, 1, 6, 8]
    result = select_rollouts([bundle(range(0, 4)), bundle(range(4, 8)), bundle(range(8, 12))], indices)
    assert result[1] == [[i, i] for i in indices]
    assert result[2] == [[1, 0]] * 4
    assert result[3] == [f"trajectory-{i}" for i in indices]
    assert result[4] == 4
    assert result[5] == [[-i, 0] for i in indices]
    assert result[6] == {"candidate": indices}


def test_ties_missing_rewards_and_invalid_groups():
    assert select_best_indices([1, 1, float("nan"), 0], 4, 2) == [0, 1]
    with pytest.raises(ValueError, match="finite"):
        select_best_indices([float("nan"), float("inf"), 1, float("nan")], 4, 2)
    with pytest.raises(ValueError, match="complete"):
        select_best_indices([1, 2, 3], 4, 2)
    for n, g in [(-1, 2), (3, 2), (2, 4), (4, 1)]:
        with pytest.raises(ValueError):
            validate_best_of_n(n, g)
    validate_best_of_n(0, 16)
    validate_best_of_n(32, 16)


class Parent:
    def _generate_and_score_completions(self, inputs):
        return self._generate([row["prompt"] for row in inputs])


class Trainer(BestOfNTrainingMixin, Parent):
    def __init__(self):
        self.train_best_of_n = 4
        self.num_generations = 2
        self.model = SimpleNamespace(training=True)
        self.accelerator = SimpleNamespace(device="cpu", process_index=0)
        self.reward_weights = torch.tensor([1.])
        self._metrics = {"train": defaultdict(list)}
        self.calls = []

    def _generate(self, prompts):
        if self._best_of_inputs is not None and not getattr(self, "_ranking_candidates", False):
            return self._generate_best_of_n(prompts)
        self.calls.append((len(prompts), self.num_generations))
        return bundle(range(len(prompts)))

    def _calculate_rewards(self, inputs, prompts, completions, ids):
        assert self._ranking_candidates
        assert self.num_generations == 4
        assert [row["candidate"] for row in inputs] == list(range(4))
        if getattr(self, "fail", False):
            raise RuntimeError("simulator failed")
        return torch.tensor([[0.], [3.], [2.], [1.]])


def test_selects_before_parent_forwards_and_bypasses_eval(monkeypatch):
    monkeypatch.setattr("accelerate.utils.gather_object", lambda rows: rows)
    trainer = Trainer()
    inputs = [{"prompt": "p", "fault": "sa0"}] * 2
    result = trainer._generate_and_score_completions(inputs)
    assert result[1] == [[1, 1], [2, 2]]
    assert result[4].item() == 2
    assert trainer.calls == [(4, 4)]
    assert trainer.num_generations == 2
    assert trainer._best_of_inputs is None
    assert not trainer._ranking_candidates
    assert "candidate" not in inputs[0]
    trainer.model.training = False
    trainer._generate_and_score_completions(inputs)
    assert trainer.calls[-1] == (2, 2)


def test_simulator_failure_restores_training_state(monkeypatch):
    monkeypatch.setattr("accelerate.utils.gather_object", lambda rows: rows)
    trainer = Trainer()
    trainer.fail = True
    with pytest.raises(RuntimeError, match="simulator failed"):
        trainer._generate_and_score_completions([{"prompt": "p"}] * 2)
    assert trainer.num_generations == 2
    assert trainer._best_of_inputs is None
    assert not trainer._ranking_candidates


@pytest.mark.parametrize("name", ["dual_adapter_grpo_trainer", "tool_calling_grpo_trainer"])
def test_both_trainers_rank_raw_rewards_and_normalize_selected_groups(name):
    # Execute the real reward method with a CPU fake trainer to verify the hook
    # precedes GDPO and honors objective weights for either adapter lifecycle.
    from atpgllm.training.gdpo import reward_output_to_tensors, select_objective_columns, compute_gdpo_advantages
    source = Path(__file__).resolve().parents[2] / f"atpgllm/training/{name}.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and "GRPOTrainer" in n.name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_calculate_rewards")
    fn.decorator_list = []
    import asyncio
    from contextlib import nullcontext
    scope = dict(torch=torch, nn=torch.nn, asyncio=asyncio, gather=lambda x: x,
                 profiling_context=lambda *a: nullcontext(), select_objective_columns=select_objective_columns,
                 compute_gdpo_advantages=compute_gdpo_advantages)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), scope)
    reward = lambda **kw: [{"detection": float(i), "format": 1.} for i in range(len(kw["prompts"]))]
    reward.gdpo_objective_keys = ["detection", "format"]
    reward.gdpo_objective_weights = [1., 0.05]
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(device="cpu"), model=SimpleNamespace(training=True),
        reward_funcs=[reward], reward_processing_classes=[None], reward_func_names=["sim"],
        processing_class=SimpleNamespace(batch_decode=lambda ids, **kw: ["text"] * len(ids)),
        state=None, print=False, num_generations=4, scale_rewards="none", _ranking_candidates=True,
        _metrics={"train": {}}, _log_netlist_diversity_diagnostics=lambda *a: None,
        _log_reward_component_means=lambda *a: None, _log_group_sampling_diagnostics=lambda *a: None,
        _log_gdpo_diagnostics=lambda *a: None,
        _reward_output_to_scalars_and_components=lambda rows, device: reward_output_to_tensors(rows, device))
    rows = [{"prompt": "p", "fault": "sa0"}] * 4
    raw = scope[fn.name](trainer, rows, ["p"] * 4, ["c"] * 4, [[1]] * 4)
    assert raw[:, 0].tolist() == pytest.approx([0.05, 1.05, 2.05, 3.05])
    trainer._ranking_candidates = False
    trainer.num_generations = 2
    normalized = scope[fn.name](trainer, rows[:2], ["p"] * 2, ["c"] * 2, [[1]] * 2)
    assert normalized[0, 0] < 0 < normalized[1, 0]
    assert normalized.mean().item() == pytest.approx(0)
