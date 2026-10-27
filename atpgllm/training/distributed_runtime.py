"""Initialize distributed communication before model loading, validation or fixed evaluation."""
from datetime import timedelta


def initialize_grpo_distributed(timeout_seconds: int):
    """Set the collective timeout on the first process-group initialization.

    Non-main ranks wait in collectives while rank zero waits for vLLM
    generation (GRPO) or builds the SFT stream alone inside TRL's
    main_process_first. Either can exceed NCCL's default ten-minute timeout.
    Setting only ddp_timeout on the trainer config later cannot update an
    already-created process group.
    """
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ValueError('ddp_timeout must be a positive integer number of seconds')
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
