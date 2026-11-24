"""Initialize distributed communication before model loading, validation or fixed evaluation."""
import os
import socket
from datetime import timedelta

RANK_VARIABLES = ("LOCAL_RANK", "RANK", "WORLD_SIZE")


def give_lone_process_rank(environ=os.environ):
    """Give a DDP run with a single process the environment torchrun would give it.

    ``accelerate launch --num_processes 1`` (the GRPO layout of one training
    GPU beside one vLLM GPU, and any 1-GPU DDP run) starts the script without
    rank variables, so accelerate runs it undistributed and never creates a
    process group. A DeepSpeed config then switches the trainer's accelerator
    to DEEPSPEED all the same, and the first barrier (GRPOTrainer.__init__'s
    wait_for_everyone) fails with "Default process group has not been
    initialized" (juno 525530, 2026-11-24). With these variables PartialState
    creates a one-rank NCCL group, exactly as under ``torchrun
    --nproc_per_node 1``, so the run takes the same path as a multi-GPU one.

    Returns True when it set the variables, False when a launcher already did.
    """
    if any(environ.get(name) not in (None, "") for name in RANK_VARIABLES):
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environ.update({"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1",
                    "MASTER_ADDR": environ.get("MASTER_ADDR") or "127.0.0.1",
                    "MASTER_PORT": environ.get("MASTER_PORT") or str(port)})
    return True


def initialize_grpo_distributed(timeout_seconds: int, use_ddp: bool = False):
    """Set the collective timeout on the first process-group initialization.

    Non-main ranks wait in collectives while rank zero waits for vLLM
    generation (GRPO) or builds the SFT stream alone inside TRL's
    main_process_first. Either can exceed NCCL's default ten-minute timeout.
    Setting only ddp_timeout on the trainer config later cannot update an
    already-created process group.

    With ``use_ddp`` a lone process gets rank variables first
    (:func:`give_lone_process_rank`), so it has a process group like every
    rank of a multi-GPU run. Without it (one process spreading the model over
    the visible GPUs) nothing changes.
    """
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ValueError('ddp_timeout must be a positive integer number of seconds')
    if use_ddp and give_lone_process_rank():
        print('[distributed] One process: running it as rank 0 of 1 (torchrun layout), so DeepSpeed '
              'and the DDP code path have a process group.', flush=True)
    from accelerate import PartialState

    state = PartialState(timeout=timedelta(seconds=timeout_seconds))
    if state.num_processes > 1:
        state.print(f'[distributed] Collective timeout: {timeout_seconds}s.')
    return state


def pin_vllm_group_port(port: int):
    """Make GRPOTrainer's vLLM server client use ``port`` for weight sync.

    TRL 0.26 builds ``VLLMClient(base_url=...)`` inside GRPOTrainer.__init__
    with the default group port 51216, which lies in Linux's ephemeral range
    (32768-60999). An outbound connection that happens to hold 51216 as its
    source port makes the vLLM worker's bind fail with EADDRINUSE. The server
    takes the port from the client's init_communicator request, so pinning the
    client is enough.
    """
    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        raise ValueError('vLLM group port must be an integer in 1-65535')
    import functools
    import trl.trainer.grpo_trainer as grpo_trainer
    from trl.extras.vllm_client import VLLMClient

    grpo_trainer.VLLMClient = functools.partial(VLLMClient, group_port=port)
