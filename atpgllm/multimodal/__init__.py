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
