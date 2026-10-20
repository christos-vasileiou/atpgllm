"""Stream a multimodal (graph) training run's metrics.jsonl to Weights & Biases.

multimodal_training.py writes its train, held-out and probe rows only to
``<OUTPUT_DIR>/metrics.jsonl``. This follower runs beside it (or after it),
logs every row, including those already written, and keeps following until
the run writes its final checkpoint. Training is not touched.

Metric names follow the text SFT's Trainer logs so the curves overlay in one
project: train/loss, train/learning_rate, train/global_step, and the
single-call held-out loss as eval/loss (the text validation has no retries).
Runs are tagged and grouped ``graph-modality``.

    python scripts/train/wandb_follow_metrics.py runs/multimodal_sft_v4 \
        --log logs/multimodal_sft_v4_<ts>.log --project sft-training

A restarted follower resumes the same W&B run and skips rows it has logged
(state in ``<OUTPUT_DIR>/wandb_follow.json``).
"""

from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path
from typing import Any

# metrics.jsonl key -> W&B key.
RENAMES = {
    "sft_loss": "train/loss",
    "lr": "train/learning_rate",
    "sec_per_step": "perf/sec_per_step",
    "micro_batches_per_step": "perf/micro_batches_per_step",
    "heldout_loss_single": "eval/loss",
    "heldout_loss_retry": "eval/loss_retry",
    "heldout_loss": "eval/loss_all",
}


def _launch_config(run_dir: Path, log: Path | None) -> dict[str, Any]:
    """The launch arguments (from the log's ``Command:`` line) and the code
    provenance of the run's latest launch."""
    config: dict[str, Any] = {"modality": "graph", "output_dir": str(run_dir.resolve())}
    if log is not None and log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("Command:"):
                words = shlex.split(line.removeprefix("Command:"))
                for i, word in enumerate(words):
                    if word.startswith("--"):
                        value = words[i + 1] if i + 1 < len(words) else ""
                        config[word[2:].replace("-", "_")] = (
                            True if value.startswith("--") else value
                        )
                break
    provenance = run_dir / "code_provenance.jsonl"
    if provenance.is_file():
        lines = provenance.read_text().strip().splitlines()
        if lines:
            config.update({f"provenance/{k}": v for k, v in json.loads(lines[-1]).items()})
    return config


def _row(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"train/global_step": record["step"]}
    for key, value in record.items():
        if key == "step":
            continue
        if key.startswith("probe_"):
            out[f"probe/{key.removeprefix('probe_')}"] = value
        elif key == "heldout_examples" and isinstance(value, dict):
            out.update({f"eval/examples_{kind}": n for kind, n in value.items()})
        elif isinstance(value, (int, float)) and value is not None:
            out[RENAMES.get(key, key)] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--log", type=Path, help="The run's launcher log (for its arguments).")
    parser.add_argument("--project", default="sft-training")
    parser.add_argument("--entity")
    parser.add_argument("--name", help="Default: the run directory's name.")
    parser.add_argument("--tags", default="graph-modality,stage-c,v4")
    parser.add_argument("--group", default="graph-modality")
    parser.add_argument("--poll", type=float, default=30.0)
    args = parser.parse_args()

    import wandb

    metrics = args.run_dir / "metrics.jsonl"
    state_path = args.run_dir / "wandb_follow.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name or args.run_dir.name,
        group=args.group,
        tags=[tag for tag in args.tags.split(",") if tag],
        config=_launch_config(args.run_dir, args.log),
        id=state.get("run_id"),
        resume="allow",
    )
    run.define_metric("train/global_step")
    run.define_metric("*", step_metric="train/global_step")
    state["run_id"] = run.id
    logged = int(state.get("lines", 0))
    print(f"following {metrics} -> {run.url} (skipping {logged} logged rows)", flush=True)

    final = args.run_dir / "multimodal_sft_final.pt"
    while True:
        finished = final.is_file()  # checked first: rows written before it are read below
        data = metrics.read_bytes() if metrics.is_file() else b""
        lines = data.decode().splitlines()
        # A last line without its newline may still be being written.
        complete = len(lines) if data.endswith(b"\n") else max(0, len(lines) - 1)
        for line in lines[logged:complete]:
            run.log(_row(json.loads(line)))
        if complete > logged:
            logged = complete
            state["lines"] = logged
            state_path.write_text(json.dumps(state) + "\n")
        if finished:
            break
        time.sleep(args.poll)
    best = args.run_dir / "best.json"
    if best.is_file():
        run.summary.update({f"best/{k}": v for k, v in json.loads(best.read_text()).items()})
    run.finish()


if __name__ == "__main__":
    main()
