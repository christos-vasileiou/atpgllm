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


def test_no_complete_checkpoint_fails_instead_of_selecting_partial(tmp_path):
    with pytest.raises(ValueError, match="No completed resumable"):
        latest_complete_checkpoint(tmp_path)
