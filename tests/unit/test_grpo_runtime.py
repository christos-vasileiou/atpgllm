"""CPU checks for distributed bookkeeping and cache invalidation."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2] / "atpgllm/training/dual_adapter_grpo_trainer.py"


def method(name, **scope):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DualAdapterGRPOTrainer")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    namespace = {"torch": torch, **scope}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("main", [True, False])
def test_weight_sync_invalidates_cache_once_and_synchronizes(main):
    events = []
    trainer = SimpleNamespace(
        model=SimpleNamespace(named_modules=lambda: []), vllm_mode="server",
        vllm_client=SimpleNamespace(reset_prefix_cache=lambda: events.append("reset")),
        accelerator=SimpleNamespace(is_main_process=main, wait_for_everyone=lambda: events.append("barrier")))
    method("_move_model_to_vllm")(trainer)
    assert events == (["reset", "barrier"] if main else ["barrier"])


def test_netlist_window_is_one_global_optimizer_step():
    trainer = SimpleNamespace(model=SimpleNamespace(training=True), num_generations=16,
                              _metrics={"train": {}}, _netlist_seen=set(),
                              _eff_batch_netlist_window=[], _prompts_per_effective_batch=72,
                              _netlist_identity=lambda value: value)
    for i in range(24):
        global_ids = [name for name in (f"a{i}", f"b{i}", f"c{i}") for _ in range(16)]
        fn = method("_log_netlist_diversity_diagnostics", gather_object=lambda rows: global_ids)
        fn(trainer, [{"netlist": f"a{i}"}] * 16)
    metrics = trainer._metrics["train"]
    assert metrics["diagnostics/netlist_diversity/unique_in_generation_batch"] == [3.0] * 24
    assert metrics["diagnostics/netlist_diversity/effective_batch_unique_count"] == [72.0]
    assert metrics["diagnostics/netlist_diversity/cumulative_unique_seen"] == [72.0]


def test_eval_groups_split_across_ranks_are_aggregated():
    trainer = SimpleNamespace(model=SimpleNamespace(training=False), num_generations=16,
                              num_generations_eval=3, _metrics={"eval": {}})
    scalars = torch.tensor([0., 1., 0.])
    matrix = scalars[:, None]
    fn = method("_log_group_sampling_diagnostics",
                gather=lambda value: scalars if value.ndim == 1 else matrix)
    fn(trainer, scalars[:1], matrix[:1], ["fault_detected_by_pred_input_vector_acc_logonly"], "reward_fn")
    assert trainer._metrics["eval"]["diagnostics/group_sampling/reward_fn/pass_at_1"] == pytest.approx([1/3])


def test_checkpoint_marker_removed_before_rewriting_weights(tmp_path):
    from atpgllm.training.grpo_loss import GenerationBatchLossMixin
    marker = tmp_path / "checkpoint-5/training_state_summary.json"
    marker.parent.mkdir()
    marker.write_text('{"resumable": true}')
    class Base:
        def _save_checkpoint(self, model, trial):
            assert not marker.exists()
    class Trainer(GenerationBatchLossMixin, Base):
        pass
    trainer = Trainer()
    trainer.args = SimpleNamespace(output_dir=tmp_path)
    trainer.state = SimpleNamespace(global_step=5)
    trainer.accelerator = SimpleNamespace(is_main_process=True, wait_for_everyone=lambda: None)
    trainer._save_checkpoint(None, None)
