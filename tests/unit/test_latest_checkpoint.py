import json
import os

import pytest

from atpgllm.training.checkpoints import latest_complete_checkpoint, rank_checkpoints


def checkpoint(root, step, timestamp):
    path = root / f"checkpoint-{step}"
    path.mkdir()
    for name in ("optimizer.pt", "scheduler.pt", "rng_state_0.pth", "rng_state_1.pth", "rng_state_2.pth"):
        (path / name).write_bytes(b"fixture")
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    marker = path / "training_state_summary.json"
    marker.write_text(json.dumps({"global_step": step, "resumable": True, "world_size": 3}))
    os.utime(marker, (timestamp, timestamp))
    return path


def test_latest_uses_completed_save_time_and_skips_incomplete(tmp_path):
    checkpoint(tmp_path, 24, 100)
    current = checkpoint(tmp_path, 14, 200)
    incomplete = checkpoint(tmp_path, 15, 300)
    (incomplete / "rng_state_2.pth").unlink()
    assert latest_complete_checkpoint(tmp_path) == current
    (incomplete / "rng_state_2.pth").write_bytes(b"complete")
    assert latest_complete_checkpoint(tmp_path) == incomplete
    (incomplete / "training_state_summary.json").write_text("partial json")
    assert latest_complete_checkpoint(tmp_path) == current


def test_deepspeed_checkpoint_needs_one_optimizer_shard_per_rank(tmp_path):
    path = checkpoint(tmp_path, 7, 100)
    (path / "optimizer.pt").unlink()
    (path / "latest").write_text("global_step7")
    shards = path / "global_step7"
    shards.mkdir()
    for rank in range(2):
        (shards / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="No completed resumable"):
        latest_complete_checkpoint(tmp_path)
    (shards / "bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt").write_bytes(b"fixture")
    assert latest_complete_checkpoint(tmp_path) == path


def test_no_complete_checkpoint_fails_instead_of_selecting_partial(tmp_path):
    with pytest.raises(ValueError, match="No completed resumable"):
        latest_complete_checkpoint(tmp_path)


def evaluation(step, detection, reward):
    return [{"eval_rewards/reward_fn/raw_mean": reward, "step": step},
            {"eval_fixed/detection": detection, "step": step}]


def test_ranking_keeps_best_detection_then_reward_and_the_latest(tmp_path):
    history = []
    for step, detection, reward in [(1, 0.30, 0.9), (2, 0.40, 0.5), (3, 0.40, 0.7), (4, 0.20, 1.0)]:
        checkpoint(tmp_path, step, 100 + step)
        history += evaluation(step, detection, reward)
    checkpoint(tmp_path, 5, 50)  # unscored and older on disk than the rest
    ranking, prune = rank_checkpoints(tmp_path, history, keep=2)
    assert [entry["step"] for entry in ranking] == [3, 2]
    # 4 is the latest complete checkpoint and 5 the newest directory.
    assert [path.name for path in prune] == ["checkpoint-1"]


def test_ranking_uses_the_later_evaluation_of_a_step_and_never_prunes_without_a_complete_checkpoint(tmp_path):
    history = evaluation(1, 0.9, 0.0) + evaluation(2, 0.5, 0.0) + evaluation(1, 0.1, 0.0)
    assert rank_checkpoints(tmp_path, history, keep=1) == ([], [])
    for step in (1, 2, 3):
        checkpoint(tmp_path, step, 100 + step)
    ranking, prune = rank_checkpoints(tmp_path, history, keep=1)
    assert [entry["step"] for entry in ranking] == [2]
    assert [path.name for path in prune] == ["checkpoint-1"]
