"""Graph-conditioned SFT and GRPO integration."""

from .data import (  # noqa: F401
    GraphPromptExample,
    build_graph_prompt_example,
    build_multimodal_sft_batch,
)
from .grpo import graph_grpo_loss, group_relative_advantages  # noqa: F401
from .loading import load_aligned_graph_stack  # noqa: F401
from .model import GraphConditionedCausalLM  # noqa: F401

__all__ = [
    "GraphConditionedCausalLM",
    "GraphPromptExample",
    "build_graph_prompt_example",
    "build_multimodal_sft_batch",
    "graph_grpo_loss",
    "group_relative_advantages",
    "load_aligned_graph_stack",
]
"""
Reserved package for cross-modal integration of ``atpgllm.llm`` and
``atpgllm.graph``.

Planned responsibilities (not yet implemented):
  - Soft-prompt / projector wiring between graph query tokens and the LLM
  - Shared collate and conversation formatting for graph-conditioned ATPG
  - Stage-2/3 training entrypoints that reuse SFT / GRPO reward paths

Import graph encoders from ``atpgllm.graph`` and LLM utilities from
``atpgllm.llm`` until this package lands.
"""

__all__: list[str] = []
