"""
Plotly helpers for W&B: CDN-sized HTML and parsing ``fig.write_html`` exports.

W&B’s HTML panel often fails to run full **inline** plotly.js bundles; use
``plotly.io.to_html(..., full_html=False, include_plotlyjs="cdn")`` so the
iframe loads Plotly from a CDN and the figure renders in the UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ``Plotly.newPlot`` is always in the last part of a ``write_html`` file.
_READ_TAIL_BYTES = 3_000_000


def figure_from_plotly_write_html_file(path: Path) -> Any | None:
    """
    Rebuild a :class:`plotly.graph_objects.Figure` from a file written by
    ``fig.write_html`` (or Optuna’s Plotly outputs), by parsing the trailing
    ``Plotly.newPlot( id, data, layout, config )`` call.
    """
    try:
        from plotly.graph_objects import Figure
    except ImportError:
        return None
    try:
        sz = path.stat().st_size
    except OSError:
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            if sz > _READ_TAIL_BYTES:
                f.seek(max(0, sz - _READ_TAIL_BYTES))
            tail = f.read()
    except OSError:
        return None
    i = tail.rfind("Plotly.newPlot")
    if i < 0:
        return None
    s = tail[i:]
    paren = s.find("(")
    if paren < 0:
        return None
    j = paren + 1
    while j < len(s) and s[j] in " \n\t":
        j += 1
    dec = json.JSONDecoder()
    try:
        _div_id, j = dec.raw_decode(s, j)
        while j < len(s) and s[j] in " \n\t,":
            j += 1
        data, j = dec.raw_decode(s, j)
        while j < len(s) and s[j] in " \n\t,":
            j += 1
        layout, j = dec.raw_decode(s, j)
        while j < len(s) and s[j] in " \n\t,":
            j += 1
        _config, j = dec.raw_decode(s, j)  # noqa: F841 — must consume for valid parse
    except (json.JSONDecodeError, ValueError, IndexError):
        return None
    try:
        return Figure(data=data, layout=layout)
    except (ValueError, TypeError):
        return None


def figure_to_wandb_html_cdn(fig: Any) -> str:
    import plotly.io as pio

    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
