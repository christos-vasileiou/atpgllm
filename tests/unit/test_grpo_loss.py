"""Exercise the installed TRL loss on CPU against independent objectives."""
import ast
from collections import defaultdict
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atpgllm.training.grpo_loss import GenerationBatchLossMixin


@pytest.fixture(scope="module")
def trainer_class():
    # Importing GRPOTrainer imports vLLM/CUDA. Extract its actual loss method
    # so these regression checks also run on a CPU login node.
    path = Path(next(iter(find_spec("trl").submodule_search_locations))) / "trainer/grpo_trainer.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GRPOTrainer")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_compute_loss")
    fn.decorator_list = []
    scope = {"torch": torch, "nanmin": lambda t: t.min(), "nanmax": lambda t: t.max()}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
    base = type("InstalledTRL", (), {"_compute_loss": scope["_compute_loss"]})
    return type("CorrectedTrainer", (GenerationBatchLossMixin, base), {})


def make_trainer(cls, parameter, generation_steps, accumulation, world=1):
    trainer = cls()
    trainer.model = SimpleNamespace(training=True)
    trainer.args = SimpleNamespace(steps_per_generation=generation_steps, delta=None, use_bias_correction_kl=False)
    trainer.current_gradient_accumulation_steps = accumulation
    trainer.loss_type = "dapo"
    trainer.beta = 0.03
    trainer.tools = ["simulator"]
    trainer.top_entropy_quantile = 1
    trainer.importance_sampling_level = "token"
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.use_vllm = False
    trainer.accelerator = SimpleNamespace(num_processes=world, gather=lambda t: t)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._get_per_token_logps_and_entropies = lambda model, ids, mask, keep, **kw: (
        parameter * ids[:, -keep:].float(), torch.zeros_like(ids[:, -keep:], dtype=torch.float))
    return trainer


def batch(offset, count):
    features = (torch.arange(count * 4).reshape(count, 4) + offset) % 7 + 1
    mask = torch.ones(count, 4)
    mask[::2, -1] = 0  # padding / variable completion lengths
    if offset:
        mask[0, 0] = 0  # generation batches also have different token counts
    tool_mask = torch.ones_like(mask)
    tool_mask[:, 1] = 0  # injected tool results never receive a gradient
    return {
        "prompt_ids": torch.ones(count, 1, dtype=torch.long), "prompt_mask": torch.ones(count, 1),
        "completion_ids": features, "completion_mask": mask, "tool_mask": tool_mask,
        "advantages": torch.linspace(-1, 1, count), "old_per_token_logps": torch.zeros(count, 4),
        "ref_per_token_logps": torch.full((count, 4), 0.15),
    }


@pytest.mark.parametrize("world", [1, 3])
@pytest.mark.parametrize("microbatch", [1, 2, 4])
def test_accumulation_matches_mean_of_generation_batch_token_objectives(trainer_class, world, microbatch):
    parameter = torch.tensor(0.01, requires_grad=True)
    reference_parameter = parameter.detach().clone().requires_grad_()
    generations = [batch(offset, 4) for offset in (0, 2, 5)]
    trainer = make_trainer(trainer_class, parameter, 4 // microbatch, 12 // microbatch, world)
    total = 0
    references = []
    for inputs in generations:
        mask = inputs["completion_mask"] * inputs["tool_mask"]
        # Independent full generation-batch objective, with nonzero KL.
        logps = reference_parameter * inputs["completion_ids"]
        diff = inputs["ref_per_token_logps"] - logps
        per_token = -logps.exp() * inputs["advantages"][:, None] + 0.03 * (diff.exp() - diff - 1)
        references.append((per_token * mask).sum() / mask.sum())
        for start in range(0, 4, microbatch):
            chunk = {key: value[start:start + microbatch] for key, value in inputs.items()}
            chunk["num_items_in_batch"] = mask.sum() * world
            total = total + trainer._compute_loss(trainer.model, chunk)
    expected = torch.stack(references).mean()
    total.backward()
    expected.backward()
    torch.testing.assert_close(total, expected)
    torch.testing.assert_close(parameter.grad, reference_parameter.grad)


def test_run_384_16_scale_and_evaluation(trainer_class):
    parameter = torch.tensor(0.0, requires_grad=True)
    trainer = make_trainer(trainer_class, parameter, 16, 384, 3)
    inputs = batch(0, 2)
    inputs["num_items_in_batch"] = (inputs["completion_mask"] * inputs["tool_mask"]).sum() * 3
    train_loss = trainer._compute_loss(trainer.model, inputs)
    assert trainer._metrics["train"]["loss/accumulation_scale"] == [1 / 24]
    trainer.model.training = False
    eval_loss = trainer._compute_loss(trainer.model, inputs)
    torch.testing.assert_close(eval_loss, train_loss * 24)
    # A short final accumulation uses its actual size, including after resume.
    trainer.model.training = True
    trainer.current_gradient_accumulation_steps = 16
    torch.testing.assert_close(trainer._compute_loss(trainer.model, inputs), eval_loss)


def test_other_loss_types_are_not_scaled_twice(trainer_class):
    parameter = torch.tensor(0.0, requires_grad=True)
    trainer = make_trainer(trainer_class, parameter, 16, 384)
    trainer.loss_type = "grpo"
    inputs = batch(0, 2)
    train_loss = trainer._compute_loss(trainer.model, inputs)
    trainer.current_gradient_accumulation_steps = 1
    torch.testing.assert_close(trainer._compute_loss(trainer.model, inputs), train_loss * 384)
    assert "loss/accumulation_scale" not in trainer._metrics["train"]
