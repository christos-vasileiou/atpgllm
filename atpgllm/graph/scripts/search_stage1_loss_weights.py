"""Deprecated entry point for the former unsafe Stage-1 search driver.

The old implementation compared weighted training losses while sampling those
same weights, varied GIN architecture after loading a fixed Stage-A checkpoint,
and used an asynchronous modulo-GPU pool that could oversubscribe devices.
Use :mod:`atpgllm.graph.scripts.search_graph_pipeline` instead.
"""

from __future__ import annotations

import sys


MIGRATION = """
search_stage1_loss_weights is deprecated.

Use the staged Optuna workflow:
  python -m atpgllm.graph.scripts.search_graph_pipeline --help

Typical migration:
  1. prepare-split
  2. create-study --stage graph_pretrain
  3. worker --stage graph_pretrain
  4. promote --stage graph_pretrain
  5. create-study/worker --stage graph_text_alignment --promotion-manifest ...

The former flags (--gpus, --sampler, --w-*-range, and passthrough arguments)
are intentionally not translated because their objective and GPU scheduling
were not safe.
""".strip()


def main() -> int:
    print(MIGRATION, file=sys.stderr)
    return 0 if any(arg in {"-h", "--help"} for arg in sys.argv[1:]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
