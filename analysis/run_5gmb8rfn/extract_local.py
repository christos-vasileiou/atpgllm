"""Snapshot local W&B history, checkpoint metrics, and paired fixed evaluation."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
RUN = REPO / "wandb/run-20260909_135227-5gmb8rfn"
reader = DataStore()
reader.open_for_scan(str(RUN / "run-5gmb8rfn.wandb"))
history, config, counts = [], {}, Counter()
tail_error = None
while True:
    try:
        data = reader.scan_data()
    except Exception as exc:
        tail_error = f"{type(exc).__name__}: {exc}"
        break
    if data is None:
        break
    record = wandb_internal_pb2.Record()
    record.ParseFromString(data)
    kind = record.WhichOneof("record_type")
    counts[kind] += 1
    if kind == "history":
        row = {}
        for item in record.history.item:
            key = item.key or "/".join(item.nested_key)
            row[key] = json.loads(item.value_json)
        history.append(row)
    elif kind in ("config", "run"):
        entries = record.config.update if kind == "config" else record.run.config.update
        for item in entries:
            config[item.key or "/".join(item.nested_key)] = json.loads(item.value_json)

metrics = [r for r in history if "train/reward" in r or "eval_fixed/detection" in r]
(OUT / "grpo_wandb_snapshot.json").write_text(json.dumps({
    "counts": counts, "tail_error": tail_error, "config": config, "metrics": metrics}, indent=2))
train = [r for r in metrics if "train/reward" in r]
keys = sorted(set().union(*(r.keys() for r in train)))
with (OUT / "grpo_metrics.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(train)

manifest = json.loads((REPO / "runs/grpo_granite_4.2_8b/fixed_eval_manifest.json").read_text())
examples = {r["_fixed_eval_id"]: r for r in manifest["examples"]}
evals = {}
for path in sorted((REPO / "runs/grpo_granite_4.2_8b/fixed_eval").glob("step-*.json")):
    value = json.loads(path.read_text())
    if value["examples_sha256"] != manifest["examples_sha256"]:
        continue
    evals[value["step"]] = value
grouped = {}
for step, value in evals.items():
    groups = defaultdict(list)
    for record in value["records"]:
        groups[record["example_id"]].append(record)
    assert len(groups) == 72 and all(len(v) == 3 for v in groups.values())
    grouped[step] = groups

ids = sorted(examples)
rng = np.random.default_rng(1729)
paired = {}
base = np.array([np.mean([r["components"]["detection"] for r in grouped[0][i]]) for i in ids])
for step, groups in grouped.items():
    current = np.array([np.mean([r["components"]["detection"] for r in groups[i]]) for i in ids])
    difference = current - base
    boot = difference[rng.integers(0, len(ids), size=(20000, len(ids)))].mean(axis=1)
    paired[step] = {"detection_delta": float(difference.mean()),
                    "paired_fault_bootstrap_95ci": np.quantile(boot, [.025, .975]).tolist(),
                    "faults_improved": int((difference > 0).sum()),
                    "faults_worsened": int((difference < 0).sum()),
                    "faults_unchanged": int((difference == 0).sum()),
                    "metrics": evals[step]["metrics"]}
    rows = []
    for i in ids:
        example = examples[i]
        records = groups[i]
        rows.append({"id": i, "doc_id": example["netlist"].get("doc_id"),
                     "module": example.get("module_name"), "fault": example["fault"],
                     "detections": sum(r["components"]["detection"] for r in records),
                     "unique_completions": len(set(r["completion"] for r in records))})
    (OUT / f"eval_faults_step_{step}.json").write_text(json.dumps(rows, indent=2))

summary = {"history_counts": counts, "tail_error": tail_error,
           "optimizer_steps": [r.get("train/global_step") for r in train],
           "eval_circuits": len(set(r["netlist"]["doc_id"] for r in examples.values())),
           "paired_evaluation": paired}
(OUT / "local_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
