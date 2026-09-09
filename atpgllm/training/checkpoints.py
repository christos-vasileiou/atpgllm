"""Choose the most recently completed checkpoint without importing ML packages."""

import json
from pathlib import Path
import sys


def latest_complete_checkpoint(output_dir):
    candidates = []
    for directory in Path(output_dir).glob("checkpoint-*"):
        summary_path = directory / "training_state_summary.json"
        try:
            step = int(directory.name.removeprefix("checkpoint-"))
            summary = json.loads(summary_path.read_text())
            state = json.loads((directory / "trainer_state.json").read_text())
            if not summary.get("resumable") or state["global_step"] != step:
                continue
            if summary.get("global_step") != step:
                continue
            world = int(summary.get("world_size", 1))
            required = [directory / "scheduler.pt"]
            required += ([directory / f"rng_state_{rank}.pth" for rank in range(world)]
                         if world > 1 else [directory / "rng_state.pth"])
            if not all(path.is_file() for path in required):
                continue
            if not any((directory / name).is_file() for name in ("optimizer.pt", "optimizer.safetensors")):
                continue
            # on_save publishes this summary after optimizer/scheduler/RNG and
            # fixed-evaluation metadata. Numeric steps can belong to older runs
            # when a directory has been reused; publication time is authoritative.
            candidates.append((summary_path.stat().st_mtime_ns, step, directory))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not candidates:
        raise ValueError(f"No completed resumable checkpoint under {output_dir}")
    return max(candidates)[2]


if __name__ == "__main__":
    try:
        print(latest_complete_checkpoint(sys.argv[1]))
    except (ValueError, IndexError) as exc:
        sys.exit(str(exc))
