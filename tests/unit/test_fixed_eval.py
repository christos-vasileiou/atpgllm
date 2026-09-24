"""CPU-only invariants; runnable with Python's unittest (no ML imports)."""
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[2] / "atpgllm/training/fixed_eval.py"
spec = importlib.util.spec_from_file_location("fixed_eval_under_test", MODULE)
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)


def example(circuit, fault="sa0 y[1]", doc=None):
    return {"prompt": f"Test {circuit}: {fault}", "fault": fault,
            "netlist": {"doc_id": doc or circuit, "netlist": f"module {circuit}; endmodule"}}


def components(detected=0, activated=1):
    return {"detection": detected, "activation": activated,
            "simulation_valid_logonly": 1, "simulator_error_logonly": 0,
            "pi_completeness_logonly": 1, "expected_output_acc_logonly": 0}


class FixedEvalTests(unittest.TestCase):
    def test_selection_is_order_independent_unique_and_diverse(self):
        rows = [example(c, f"sa0 y[{i}]") for c in "abc" for i in range(3)]
        selected = fixed.select_examples(rows + rows, 3, 7)
        self.assertEqual(selected, fixed.select_examples(list(reversed(rows)), 3, 7))
        self.assertEqual(len({r["netlist"]["doc_id"] for r in selected}), 3)
        self.assertTrue(all("[" in r["fault"] for r in selected))
        with self.assertRaises(ValueError):
            fixed.select_examples(rows, 10, 7)

    def test_holdout_excludes_other_faults_and_both_identity_aliases(self):
        held = example("a", doc="shared")
        train = [example("a", "sa1 z", doc="different"),
                 example("renamed", doc="shared"), example("safe")]
        self.assertEqual(fixed.training_indices_without_holdout(train, [held]), [2])
        with self.assertRaises(ValueError):
            fixed.training_indices_without_holdout(train[:2], [held])

    def test_manifest_reuses_rows_and_rejects_protocol_or_content_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            protocol = {"size": 2, "seed": 9}
            manifest = fixed.load_or_create_manifest(path, protocol, lambda: [example("a"), example("b")])
            self.assertEqual(manifest, fixed.load_or_create_manifest(path, protocol, lambda: self.fail("resampled")))
            with self.assertRaises(ValueError):
                fixed.load_or_create_manifest(path, {**protocol, "seed": 8}, lambda: [])
            manifest["examples"][0]["fault"] = "sa1 changed"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                fixed.load_or_create_manifest(path, protocol, lambda: [])

    def test_layout_prevents_partial_groups_and_padding(self):
        fixed.validate_eval_layout(72, 1, 1, 3)
        fixed.validate_eval_layout(72, 3, 1, 3)
        fixed.validate_eval_layout(72, 4, 4, 1)
        for layout in [(72, 4, 1, 3), (73, 2, 2, 3), (72, 0, 1, 3)]:
            with self.assertRaises(ValueError):
                fixed.validate_eval_layout(*layout)

    def test_resume_recovers_frozen_manifest_and_rejects_a_different_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "checkpoint-14"
            checkpoint.mkdir()
            path = root / "manifest.json"
            protocol = {"size": 2, "seed": 9}
            manifest = fixed.load_or_create_manifest(path, protocol, lambda: [example("a"), example("b")])
            fixed.write_json_atomic(checkpoint / "fixed_eval_manifest.json", manifest)
            path.unlink()
            restored = fixed.load_or_create_manifest(
                path, protocol, lambda: self.fail("Resuming must not resample"), resume_checkpoint=checkpoint)
            self.assertEqual(restored, manifest)
            self.assertEqual(json.loads(path.read_text()), manifest)
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "differs from the resumed checkpoint"):
                fixed.load_or_create_manifest(path, protocol, lambda: [], resume_checkpoint=checkpoint)

    def test_resume_buffer_detects_reordering_filtering_and_fault_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [example("a"), example("b")]
            state = fixed.training_buffer_state(rows)
            fixed.write_json_atomic(Path(folder) / "grpo_data_state.json", state)
            fixed.validate_resume_buffer(folder, fixed.training_buffer_state(rows))
            for changed in [list(reversed(rows)), rows[:1], [example("a", "sa1 z"), rows[1]]]:
                with self.assertRaisesRegex(ValueError, "training buffer changed"):
                    fixed.validate_resume_buffer(folder, fixed.training_buffer_state(changed))

    def test_checkpoint_carries_evaluation_and_buffer_metadata(self):
        class Base:
            def _save_checkpoint(self, model, trial):
                (Path(self.args.output_dir) / "checkpoint-14").mkdir()
        class Trainer(fixed.FixedEvaluationMixin, Base):
            pass
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer()
            trainer.state = SimpleNamespace(global_step=14)
            trainer.args = SimpleNamespace(output_dir=folder)
            trainer.accelerator = SimpleNamespace(is_main_process=True, wait_for_everyone=lambda: None)
            manifest = {"examples": [example("a")]}
            data_state = fixed.training_buffer_state([example("b")])
            trainer.configure_fixed_evaluation(manifest, [], folder, "checkpoint-13", data_state)
            trainer._save_checkpoint(None, None)
            checkpoint = Path(folder) / "checkpoint-14"
            self.assertEqual(json.loads((checkpoint / "fixed_eval_manifest.json").read_text()), manifest)
            self.assertEqual(json.loads((checkpoint / "grpo_data_state.json").read_text()), data_state)

    def test_counts_metrics_and_missing_samples(self):
        records = [{"example_id": key, "components": components(d)}
                   for key, d in [("a", 1), ("a", 0), ("b", 0), ("b", 0)]]
        metrics = fixed.summarize_records(records, ["a", "b"], 2)
        self.assertEqual(metrics["detection"], 0.25)
        self.assertEqual(metrics["solved_at_k"], 0.5)
        self.assertEqual(metrics["activation_without_detection"], 0.75)
        for bad in [records[:-1], records + records[:1]]:
            with self.assertRaises(ValueError):
                fixed.summarize_records(bad, ["a", "b"], 2)

    def test_reward_capture_preserves_gdpo_metadata_and_ignores_train(self):
        records = []
        def reward_fn(**kwargs):
            return [components(1)]
        reward_fn.gdpo_objective_keys = ("detection",)
        wrapped = fixed.capture_rewards(reward_fn, records)
        self.assertEqual(wrapped.gdpo_objective_keys, ("detection",))
        wrapped(prompts=["p"], completions=["c"])
        self.assertEqual(records, [])
        wrapped(prompts=["p"], completions=["c"], _fixed_eval_id=["a"])
        self.assertEqual(records[0]["components"]["detection"], 1)
        with self.assertRaises(ValueError):
            fixed.capture_rewards(lambda **kw: [0.0], [])(
                prompts=["p"], completions=["c"], _fixed_eval_id=["a"])

    def test_summary_uses_activation_diagnostic_without_training_objective(self):
        records = []
        for key, detected, activated in [("a", 1, 1), ("a", 0, 1), ("b", 0, 0), ("b", 0, 1)]:
            row = components(detected, activated)
            row["fault_site_activated_acc_logonly"] = row.pop("activation")
            records.append({"example_id": key, "components": row})
        metrics = fixed.summarize_records(records, ["a", "b"], 2)
        self.assertEqual(metrics["detection"], 0.25)
        self.assertEqual(metrics["activation_without_detection"], 0.5)
        self.assertEqual(metrics["activation_available_fraction"], 1.0)

    def test_unknown_activation_is_not_reported_as_zero(self):
        for unknown in (None, float("nan")):
            row = components(0)
            row.pop("activation")
            if unknown is not None:
                row["fault_site_activated_acc_logonly"] = unknown
            metrics = fixed.summarize_records([
                {"example_id": "a", "components": components(0)},
                {"example_id": "b", "components": row},
            ], ["a", "b"], 1)
            self.assertNotIn("activation_without_detection", metrics)
            self.assertEqual(metrics["activation_available_fraction"], 0.5)
            self.assertEqual(metrics["detection"], 0.0)

    def test_evaluation_restores_state_on_success_and_failure(self):
        class Base:
            def evaluate(self):
                self.model.train(False)
                random.random()
                self._logs["completion"].append("eval")
                assert self.args.generation_kwargs == {"existing": True, "seed": 9,
                    "temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "n": 1}
                assert self.temperature == 1.0
                if self.fail_eval:
                    raise RuntimeError("generation failed")
                row = components(1)
                if self.profile != "full":
                    row["fault_site_activated_acc_logonly"] = row.pop("activation")
                self.fixed_eval_records.extend([
                    {"example_id": "a", "components": row}])
                return {}
            def log(self, metrics):
                self.logged = metrics

        class Trainer(fixed.FixedEvaluationMixin, Base):
            pass

        numpy_rng = SimpleNamespace(get_state=lambda: "original", set_state=lambda state: None, seed=lambda seed: None)
        fake_modules = {
            "numpy": SimpleNamespace(random=numpy_rng),
            "torch": SimpleNamespace(random=SimpleNamespace(
                fork_rng=lambda **kw: nullcontext(),
                default_generator=SimpleNamespace(manual_seed=lambda seed: None))),
            "accelerate.utils": SimpleNamespace(gather_object=lambda rows: list(rows)),
        }
        for fail, profile in [(False, "full"), (False, "po"), (False, "detection"), (True, "po")]:
            with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, fake_modules):
                trainer = Trainer()
                trainer.model = SimpleNamespace(training=True)
                trainer.model.train = lambda mode: setattr(trainer.model, "training", mode)
                trainer.accelerator = SimpleNamespace(device=SimpleNamespace(type="cpu"),
                                                     is_main_process=True, wait_for_everyone=lambda: None)
                trainer.state = SimpleNamespace(global_step=0)
                trainer.args = SimpleNamespace(generation_kwargs={"existing": True})
                trainer.num_generations_eval = 1
                trainer.temperature = 1.0
                trainer._logs = {"completion": deque(["train"])}
                original_logs = trainer._logs
                trainer.fail_eval = fail
                trainer.profile = profile
                manifest = {"protocol": {"seed": 9}, "examples": [{"_fixed_eval_id": "a"}], "examples_sha256": "digest"}
                trainer.configure_fixed_evaluation(manifest, [], folder, "sft/checkpoint-200")
                rng_state = random.getstate()
                if fail:
                    with self.assertRaises(RuntimeError):
                        trainer.evaluate()
                else:
                    result = trainer.evaluate()
                    self.assertEqual(result["eval_fixed/detection"], 1.0)
                    self.assertEqual(result["eval_fixed/activation_without_detection"], 0.0)
                    artifact = json.loads((Path(folder) / "fixed_eval/step-000000.json").read_text())
                    self.assertEqual(artifact["metrics"]["activation_available_fraction"], 1.0)
                    if profile != "full":
                        self.assertNotIn("activation", artifact["records"][0]["components"])
                self.assertEqual(random.getstate(), rng_state)
                self.assertIs(trainer._logs, original_logs)
                self.assertEqual(list(trainer._logs["completion"]), ["train"])
                self.assertTrue(trainer.model.training)
                self.assertEqual(trainer.args.generation_kwargs, {"existing": True})
                self.assertEqual(trainer.fixed_eval_records, [])
                self.assertEqual(trainer._last_loaded_step, -1)


if __name__ == "__main__":
    unittest.main()
