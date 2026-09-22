"""Frozen ATPG evaluation examples and an opt-in GRPO evaluation mixin.

Manifest/aggregation helpers deliberately have no ML dependencies, so their
leakage and accounting invariants can be checked without GPUs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import uuid
from collections import Counter, defaultdict
from functools import wraps
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def write_json_atomic(path, value, *, overwrite=True):
    """Publish complete metadata even if Slurm terminates the writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            temporary.replace(path)
        else:
            os.link(temporary, path)  # exclusive, fully written publication
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_manifest(checkpoint):
    if not checkpoint:
        return None
    path = Path(checkpoint) / "fixed_eval_manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def training_buffer_state(rows):
    """Fingerprint the ordered buffer, including faults, prompts and circuits."""
    hasher = hashlib.sha256()
    count = 0
    for row in rows:
        hasher.update(digest([row.get("prompt"), row.get("fault"),
                              row.get("netlist"), row.get("module_name")]).encode())
        count += 1
    return {"version": 1, "num_rows": count, "ordered_rows_sha256": hasher.hexdigest()}


def validate_resume_buffer(checkpoint, data_state):
    if checkpoint:
        path = Path(checkpoint) / "grpo_data_state.json"
        if path.exists() and json.loads(path.read_text()) != data_state:
            raise ValueError("GRPO training buffer changed since the checkpoint; "
                             "restore the original data, stream offset and evaluation manifest")


def circuit_ids(row: dict) -> set[str]:
    """Match either document identity or exact circuit content, not prompt prose."""
    netlist = row["netlist"]
    doc_id = row.get("doc_id")
    if isinstance(netlist, dict):
        doc_id = netlist.get("doc_id") or netlist.get("id") or doc_id
        netlist = netlist["netlist"]
    if not isinstance(netlist, str) or not netlist.strip():
        raise ValueError("Fixed evaluation requires a nonempty circuit netlist")
    ids = {"net:" + digest(netlist.strip().replace("\r\n", "\n"))}
    if doc_id:
        ids.add("doc:" + str(doc_id))
    return ids


def select_examples(candidates, size: int, seed: int) -> list[dict]:
    """Deterministic, circuit-round-robin selection from a bounded candidate pool."""
    if size < 1:
        raise ValueError("fixed_eval_size must be positive")
    groups = defaultdict(list)
    seen = set()
    for row in candidates:
        if not row.get("fault") or not row.get("prompt"):
            raise ValueError("Fixed evaluation requires authoritative fault and prompt fields")
        ids = circuit_ids(row)
        net_id = next(key for key in ids if key.startswith("net:"))
        example_id = digest([net_id, row["fault"]])
        if example_id in seen:
            continue
        seen.add(example_id)
        example = {key: copy.deepcopy(row[key]) for key in
                   ("prompt", "fault", "netlist", "module_name", "doc_id") if key in row}
        example["_fixed_eval_id"] = example_id
        groups[net_id].append(example)
    ordered = sorted(groups, key=lambda key: digest([seed, key]))
    for rows in groups.values():
        rows.sort(key=lambda row: digest([seed, row["_fixed_eval_id"]]))
    selected = []
    depth = 0
    while len(selected) < size:
        added = False
        for key in ordered:
            if depth < len(groups[key]):
                selected.append(groups[key][depth])
                added = True
                if len(selected) == size:
                    return selected
        if not added:
            raise ValueError(f"Only {len(selected)} unique evaluation faults; requested {size}")
        depth += 1
    return selected


def load_or_create_manifest(path, protocol: dict, candidates_factory, *, resume_checkpoint=None) -> dict:
    """Reuse frozen prompts verbatim; reject drift instead of silently resampling.

    Call inside PartialState.main_process_first() on shared storage.
    """
    path = Path(path)
    saved = checkpoint_manifest(resume_checkpoint)
    if saved is not None or path.exists():
        manifest = saved if saved is not None else json.loads(path.read_text())
        if manifest.get("version") != 1 or manifest.get("protocol") != protocol:
            raise ValueError(f"Fixed-evaluation protocol changed: use a new manifest path ({path})")
        if digest(manifest["examples"]) != manifest["examples_sha256"]:
            raise ValueError(f"Fixed-evaluation manifest checksum mismatch: {path}")
        if saved is not None and path.exists() and json.loads(path.read_text()) != saved:
            raise ValueError(f"Evaluation manifest differs from the resumed checkpoint: {path}")
        if not path.exists():
            write_json_atomic(path, manifest, overwrite=False)
        return manifest
    examples = select_examples(candidates_factory(), protocol["size"], protocol["seed"])
    manifest = {"version": 1, "protocol": protocol, "examples": examples,
                "examples_sha256": digest(examples)}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, manifest, overwrite=False)
    return manifest


def training_indices_without_holdout(training_rows, examples) -> list[int]:
    held_out = set().union(*(circuit_ids(row) for row in examples))
    indices = [i for i, row in enumerate(training_rows) if circuit_ids(row).isdisjoint(held_out)]
    if not indices:
        raise ValueError("Circuit holdout removed the entire GRPO training buffer")
    return indices


def validate_eval_layout(size: int, generations: int, per_device_batch: int, world: int):
    if min(size, generations, per_device_batch, world) < 1:
        raise ValueError("Fixed GRPO evaluation requires positive sizes")
    batch = per_device_batch * world
    if batch % generations or (size * generations) % batch:
        raise ValueError(
            "Fixed evaluation needs complete prompt groups and no padded final batch: "
            f"size={size}, generations={generations}, global_eval_batch={batch}"
        )


def capture_rewards(reward_fn, records: list):
    """Capture simulator components before GDPO replaces them with advantages."""
    @wraps(reward_fn)
    def wrapped(prompts, completions, **kwargs):
        output = reward_fn(prompts=prompts, completions=completions, **kwargs)
        ids = kwargs.get("_fixed_eval_id")
        if ids is not None:
            if len(output) != len(ids) or any(not isinstance(row, dict) for row in output):
                raise ValueError("Fixed evaluation reward calculation failed; refusing a false zero score")
            for index, (example_id, components) in enumerate(zip(ids, output)):
                records.append({"example_id": example_id,
                                "completion": copy.deepcopy(completions[index]),
                                "components": dict(components)})
        return output
    return wrapped


def summarize_records(records: list[dict], example_ids: list[str], generations: int) -> dict:
    counts = Counter(row["example_id"] for row in records)
    expected = Counter({key: generations for key in example_ids})
    if len(expected) != len(example_ids) or counts != expected:
        raise ValueError("Fixed evaluation sample counts mismatch (missing/duplicated faults or DDP padding)")
    grouped = defaultdict(list)
    for row in records:
        grouped[row["example_id"]].append(row["components"])
    components = [row["components"] for row in records]
    n = len(components)
    return {
        "detection": sum(c["detection"] for c in components) / n,
        "activation_without_detection": sum(c["activation"] > 0 and c["detection"] == 0 for c in components) / n,
        "solved_at_k": sum(any(c["detection"] > 0 for c in group) for group in grouped.values()) / len(grouped),
        "all_fail_fraction": sum(all(c["detection"] == 0 for c in group) for group in grouped.values()) / len(grouped),
        "simulation_valid_fraction": sum(c["simulation_valid_logonly"] for c in components) / n,
        "simulator_error_fraction": sum(c["simulator_error_logonly"] for c in components) / n,
        "pi_complete_fraction": sum(c["pi_completeness_logonly"] >= 0.999 for c in components) / n,
        "expected_output_exact_fraction": sum(c["expected_output_acc_logonly"] for c in components) / n,
        "num_faults": len(grouped), "num_completions": n, "k": generations,
    }


class FixedEvaluationMixin:
    """Evaluate through the existing DDP/tool loop, with isolated RNG and artifacts."""

    def configure_fixed_evaluation(self, manifest, records, output_dir, initial_checkpoint,
                                   data_state=None):
        self.fixed_eval_manifest = manifest
        self.fixed_eval_records = records
        self.fixed_eval_output_dir = Path(output_dir) / "fixed_eval"
        self.fixed_eval_initial_checkpoint = initial_checkpoint
        self.fixed_eval_data_state = data_state

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.accelerator.is_main_process:
            directory = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            write_json_atomic(directory / "fixed_eval_manifest.json", self.fixed_eval_manifest)
            if self.fixed_eval_data_state is not None:
                write_json_atomic(directory / "grpo_data_state.json", self.fixed_eval_data_state)
        self.accelerator.wait_for_everyone()

    def evaluate(self, *args, **kwargs):
        import numpy as np
        import torch
        from accelerate.utils import gather_object

        if self.num_generations_eval != 1:
            raise ValueError("Greedy fixed evaluation requires FIXED_EVAL_GENERATIONS=1")
        seed = self.fixed_eval_manifest["protocol"]["seed"]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        was_training = self.model.training
        generation_kwargs = self.args.generation_kwargs
        old_logs = self._logs
        # TRL completion tables should not mix training samples into evaluation.
        self._logs = copy.deepcopy(old_logs)
        for value in self._logs.values():
            if isinstance(value, dict):
                for entries in value.values():
                    entries.clear()
            elif hasattr(value, "clear"):
                value.clear()
        self.fixed_eval_records.clear()
        self._last_loaded_step = -1
        devices = [self.accelerator.device.index] if self.accelerator.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                random.seed(seed)
                np.random.seed(seed)
                torch.random.default_generator.manual_seed(seed)
                for device in devices:
                    torch.cuda.default_generators[device].manual_seed(seed)
                # Both initial generation and tool continuations forward these kwargs.
                # Override decoding only. Keep self.temperature at its training
                # value: TRL also uses it as a divisor when computing logprobs.
                self.args.generation_kwargs = {
                    **(generation_kwargs or {}), "seed": seed,
                    "temperature": 0.0, "top_p": 1.0, "top_k": -1,
                    "min_p": 0.0, "n": 1,
                }
                result = super().evaluate(*args, **kwargs)
                records = gather_object(self.fixed_eval_records)
                metrics = summarize_records(
                    records, [r["_fixed_eval_id"] for r in self.fixed_eval_manifest["examples"]],
                    self.num_generations_eval,
                )
                logged = {f"eval_fixed/{key}": value for key, value in metrics.items()}
                logged["eval_fixed/global_step"] = self.state.global_step
                self.log(logged)
                result.update(logged)
                if self.accelerator.is_main_process:
                    self.fixed_eval_output_dir.mkdir(parents=True, exist_ok=True)
                    step = self.state.global_step
                    payload = {"step": step, "initial_checkpoint": self.fixed_eval_initial_checkpoint,
                               "examples_sha256": self.fixed_eval_manifest["examples_sha256"],
                               "protocol": self.fixed_eval_manifest["protocol"], "metrics": metrics,
                               "records": records}
                    target = self.fixed_eval_output_dir / f"step-{step:06d}.json"
                    if target.exists():
                        target = target.with_name(f"step-{step:06d}-repeat-{uuid.uuid4().hex[:8]}.json")
                    write_json_atomic(target, payload)
                self.accelerator.wait_for_everyone()
                return result
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            self.args.generation_kwargs = generation_kwargs
            self.model.train(was_training)
            self._logs = old_logs
            self.fixed_eval_records.clear()
            # Force a fresh training sync/cache reset after evaluation.
            self._last_loaded_step = -1
