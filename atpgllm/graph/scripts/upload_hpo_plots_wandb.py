"""
Re-upload or backfill Optuna/Plotly ``hpo_plots/*.html`` to Weights & Biases.

This does **not** create a new W&B *project*. You pass the **existing** project
(``--project`` or ``WANDB_PROJECT``, e.g. ``atpgllm-graph-stage1``). W&B will add
a **new run** in that project to hold the HTML media (runs are what hold panels
and artifacts). To keep runs next to a sweep, pass the same values you used
for the search, e.g. ``--entity`` / ``WANDB_ENTITY`` and ``--group optuna`` to
match ``--wandb-group`` from ``search_stage1_loss_weights``.

**Preferred path:** run ``search_stage1_loss_weights`` with ``--wandb-project``;
when the HPO search finishes, the **hpo_summary** run already logs these plots
(see script docstring there) — use this uploader only for old runs or retries.

Each ``*.html`` is parsed, converted with Plotly to **CDN** HTML
(``to_html(..., full_html=False, include_plotlyjs="cdn")``) so the W&B panel can
render interactive charts. The folder is also stored as a W&B Artifact for backup.

Example::

    activate
    export WANDB_ENTITY=your-username
    export WANDB_PROJECT=atpgllm-graph-stage1
    python -m atpgllm.graph.scripts.upload_hpo_plots_wandb \\
        --group optuna \\
        --plots-dir checkpoints/graph_stage1_search/hpo_plots
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def _env_str(name: str) -> str | None:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return None
    return str(v).strip()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload hpo_plots/*.html to a W&B run (HTML panels + artifact)."
    )
    p.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("checkpoints/graph_stage1_search/hpo_plots"),
        help="Directory containing Plotly / Optuna HTML exports (default: path under checkpoints/).",
    )
    p.add_argument(
        "--project",
        type=str,
        default=_env_str("WANDB_PROJECT"),
        help="Existing W&B project (e.g. atpgllm-graph-stage1); or set WANDB_PROJECT.",
    )
    p.add_argument(
        "--entity",
        type=str,
        default=_env_str("WANDB_ENTITY"),
        help="W&B entity (user or team); recommended for a stable URL. Or WANDB_ENTITY.",
    )
    p.add_argument(
        "--group",
        type=str,
        default=None,
        help="Run group, e.g. same as HPO --wandb-group (optuna) to sit with the sweep.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name (default: auto).",
    )
    p.add_argument(
        "--tags",
        type=str,
        default=_env_str("WANDB_TAGS") or "hpo_plots,html",
        help="Comma-separated tags (default: hpo_plots,html).",
    )
    p.add_argument(
        "--mode",
        type=str,
        default=_env_str("WANDB_MODE") or "online",
        choices=["online", "offline", "disabled"],
        help="W&B mode (or WANDB_MODE; default: online).",
    )
    p.add_argument(
        "--max-log-bytes",
        type=int,
        default=12_000_000,
        help="Skip per-file wandb.Html log when file size exceeds this (artifact still used).",
    )
    p.add_argument(
        "--no-artifact",
        action="store_true",
        help="Do not create a W&B Artifact; only log HTML panels.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.project:
        print(
            "Set --project or WANDB_PROJECT for the destination W&B project.",
            file=sys.stderr,
        )
        return 2
    plots_dir: Path = args.plots_dir.expanduser().resolve()
    if not plots_dir.is_dir():
        print(f"Not a directory: {plots_dir}", file=sys.stderr)
        return 1

    html_files = sorted(plots_dir.glob("*.html"))
    if not html_files:
        print(f"No .html files under {plots_dir}", file=sys.stderr)
        return 1

    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError:
        print("Install wandb: pip install 'atpgllm[graph-track]'", file=sys.stderr)
        return 1
    try:
        import plotly.io  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        print("Install plotly: pip install 'atpgllm[graph-track]'", file=sys.stderr)
        return 1
    from atpgllm.graph.scripts.hpo_wandb_plotly import (
        figure_from_plotly_write_html_file,
        figure_to_wandb_html_cdn,
    )

    tag_list = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    tag_list.extend(["hpo_plots_upload", "optuna_plots"])
    if args.mode == "disabled":
        print("W&B mode is disabled; nothing to upload.", file=sys.stderr)
        return 0

    run: Any = wandb.init(
        project=args.project,
        entity=args.entity or None,
        group=args.group,
        name=args.run_name,
        tags=tag_list,
        job_type="hpo_plots",
        config={"plots_dir": str(plots_dir), "n_html": len(html_files)},
        mode=args.mode,
    )

    payload: dict[str, Any] = {}
    skipped_parse: list[str] = []
    skipped_large: list[str] = []
    for hf in html_files:
        n = hf.name
        key = f"hpo_plots/{hf.stem}"
        fig = figure_from_plotly_write_html_file(hf)
        if fig is None:
            skipped_parse.append(n)
            print(
                f"Could not parse Plotly figure from {n!r} "
                f"(re-save with a recent plotly, or use atpgllm.graph HPO to regenerate).",
                file=sys.stderr,
            )
            continue
        try:
            html = figure_to_wandb_html_cdn(fig)
        except Exception as e:  # noqa: BLE001
            print(f"plotly to_html failed for {n!r}: {e}", file=sys.stderr)
            skipped_parse.append(n)
            continue
        b = len(html.encode("utf-8"))
        if b > int(args.max_log_bytes):
            skipped_large.append(n)
            print(
                f"Skip wandb.Html for {n} (encoded {b} bytes > {args.max_log_bytes}); "
                f"use the artifact to view the file, or increase --max-log-bytes",
                file=sys.stderr,
            )
            continue
        payload[key] = wandb.Html(html)

    if payload:
        wandb.log(payload, step=0)
    if skipped_parse:
        print(
            f"Parsed {len(html_files) - len(skipped_parse)}/{len(html_files)} files for HTML panels.",
            file=sys.stderr,
        )
    if (skipped_large or skipped_parse) and not args.no_artifact:
        print("Open the logged artifact to view full Plotly .html or retry parsing.", file=sys.stderr)

    if not args.no_artifact:
        art = wandb.Artifact(
            "hpo_plots_html",
            type="hpo",
            description="Optuna/Plotly HTML from hpo_plots (open .html in Artifacts for full-size plots).",
        )
        art.add_dir(str(plots_dir), name="hpo_plots")
        run.log_artifact(art)

    url = getattr(run, "url", None)
    if url:
        print(f"W&B run: {url}")
    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
