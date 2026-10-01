"""Two-rank collective timeout regression without loading a model.

Run via torchrun --standalone --nproc_per_node=2. Hide CUDA for a CPU/Gloo
reproduction with --timeout 3 --delay 5 --expect-timeout, then use --timeout 15
to verify the repair. With two visible GPUs, the default tests NCCL and the
DeepSpeed GRPOConfig handoff using the production one-hour timeout.
"""
import argparse
from datetime import timedelta
import tempfile
import time

import torch
import torch.distributed as dist
from accelerate.utils import broadcast_object_list
from trl import GRPOConfig

from atpgllm.training.distributed_runtime import initialize_grpo_distributed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--delay', type=float, default=3)
    parser.add_argument('--expect-timeout', action='store_true')
    args = parser.parse_args()
    state = initialize_grpo_distributed(args.timeout)
    assert state.num_processes == 2
    group = dist.distributed_c10d._get_default_group()
    # Inspect the actual C++ backend, not just the Python configuration value.
    backend = group._get_backend(state.device)
    assert backend.options._timeout == timedelta(seconds=args.timeout)
    try:
        with tempfile.TemporaryDirectory(prefix='grpo-timeout-') as directory:
            config = GRPOConfig(
                output_dir=directory, ddp_timeout=args.timeout, report_to='none',
                use_cpu=state.device.type == 'cpu', bf16=state.device.type == 'cuda',
                num_generations=2, per_device_train_batch_size=1,
                deepspeed={'zero_optimization': {'stage': 2}} if state.device.type == 'cuda' else None,
            )
            # TrainingArguments resets Accelerate's Python state. It must retain
            # the existing group and the timeout chosen before fixed evaluation.
            assert config.device == state.device
            assert dist.distributed_c10d._get_default_group() is group
            assert backend.options._timeout == timedelta(seconds=args.timeout)
            dist.barrier()
            if state.is_main_process:
                time.sleep(args.delay)  # Stand-in for the blocking vLLM HTTP call.
            payload = [{'completion_ids': [[1, 2, 3]]} if state.is_main_process else None]
            try:
                broadcast_object_list(payload, from_process=0)
            except RuntimeError as exc:
                if not args.expect_timeout:
                    raise
                assert any(text in str(exc).lower() for text in ('timed out', 'timeout', 'closed by peer'))
                print(f'Rank {state.process_index}: reproduced short-timeout failure', flush=True)
            else:
                assert not args.expect_timeout, 'Expected the short timeout to fail'
                assert payload == [{'completion_ids': [[1, 2, 3]]}]
                print(f'Rank {state.process_index}: {dist.get_backend()} broadcast passed; '
                      f'actual timeout={backend.options._timeout}', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
