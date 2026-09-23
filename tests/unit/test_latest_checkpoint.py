import json
import os

import pytest

from atpgllm.training.checkpoints import latest_complete_checkpoint


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
