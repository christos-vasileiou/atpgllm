"""Check that NCCL can run one collective across the local GPUs.

Run with ``torchrun --standalone --nproc-per-node N``. Exits non-zero, within
the 90 s timeout, when the first all_gather hangs or fails. Training sets a
4 hour timeout, so without this check a broken transport looks like a job
that runs for 4 hours and then aborts in the DDP constructor."""

import datetime
import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    value = torch.full((1,), float(rank + 1), device="cuda")
    gathered = [torch.zeros(1, device="cuda") for _ in range(world)]
    dist.all_gather(gathered, value)
    torch.cuda.synchronize()
    expected = [float(r + 1) for r in range(world)]
    got = [t.item() for t in gathered]
    if got != expected:
        raise SystemExit(f"rank {rank}: all_gather returned {got}, expected {expected}")
    if rank == 0:
        print(f"[nccl_check] all_gather ok on {world} GPUs "
              f"(NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', 'unset')})", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
