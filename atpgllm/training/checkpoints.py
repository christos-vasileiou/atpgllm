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
                tag = directory / "latest"
                ds_shards = (list((directory / tag.read_text().strip()).glob("*optim_states.pt"))
                             if tag.is_file() else [])
                if len(ds_shards) != world:
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


DETECTION_KEY = "eval_fixed/detection"
TOTAL_REWARD_KEY = "eval_rewards/reward_fn/raw_mean"


def fixed_eval_scores(log_history):
    """Map step -> (detection, total reward) from the fixed-evaluation log entries.

    Detection is the fraction of held-out faults whose final answer TetraMAX
    finds detected (activated and propagated to an output). The total reward is
    the weighted objective sum on the same greedy completions. A step evaluated
    twice (resume re-evaluates its start) keeps the later entry.
    """
    detection, reward = {}, {}
    for entry in log_history:
        step = entry.get("step")
        if DETECTION_KEY in entry:
            detection[step] = float(entry[DETECTION_KEY])
        if TOTAL_REWARD_KEY in entry:
            reward[step] = float(entry[TOTAL_REWARD_KEY])
    scores = {}
    for step, value in detection.items():
        total = reward.get(step, float("-inf"))
        scores[step] = (value, float("-inf") if total != total else total)
    return scores


def rank_checkpoints(output_dir, log_history, keep):
    """Split checkpoints into (kept ranking, directories to delete).

    The ranking holds the ``keep`` best scored checkpoints by detection, then
    total reward, then the later step. The newest checkpoint and the latest
    complete one are never deleted, so a resume always has a target. A
    checkpoint without a fixed-evaluation score cannot rank and is deleted once
    newer ones exist. Nothing is deleted while no checkpoint is complete.
    """
    directories = {}
    for directory in Path(output_dir).glob("checkpoint-*"):
        try:
            directories[int(directory.name.removeprefix("checkpoint-"))] = directory
        except ValueError:
            continue
    try:
        latest = latest_complete_checkpoint(output_dir)
    except ValueError:
        return [], []
    scores = fixed_eval_scores(log_history)
    ranked = sorted((step for step in directories if step in scores),
                    key=lambda step: (*scores[step], step), reverse=True)[:keep]
    protected = set(ranked) | {max(directories), int(latest.name.removeprefix("checkpoint-"))}
    ranking = [{"step": step, "checkpoint": directories[step].name,
                "detection": scores[step][0], "total_reward": scores[step][1]} for step in ranked]
    return ranking, [directories[step] for step in sorted(directories) if step not in protected]


if __name__ == "__main__":
    try:
        print(latest_complete_checkpoint(sys.argv[1]))
    except (ValueError, IndexError) as exc:
        sys.exit(str(exc))
