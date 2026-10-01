"""Initialize GRPO communication before model loading or fixed evaluation."""
from datetime import timedelta


def initialize_grpo_distributed(timeout_seconds: int):
    """Set the collective timeout on the first process-group initialization.

    Non-main ranks enter broadcasts while rank zero waits for vLLM generation.
    Long completions can exceed NCCL's default ten-minute timeout. Setting only
    GRPOConfig.ddp_timeout later cannot update an already-created process group.
    """
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ValueError('ddp_timeout must be a positive integer number of seconds')
    from accelerate import PartialState

    state = PartialState(timeout=timedelta(seconds=timeout_seconds))
    if state.num_processes > 1:
        state.print(f'[GRPO distributed] Collective timeout: {timeout_seconds}s '
                    '(includes waiting for vLLM generation).')
    return state
