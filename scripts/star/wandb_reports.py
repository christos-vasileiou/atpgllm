#!/usr/bin/env python
"""Upload the benchmark reports of a run outside the RFT driver, such as the reasoning pilot.

    python scripts/star/wandb_reports.py --run_dir runs/reasoning_pilot

Same tables and figures as ``scripts/rft/wandb_log.py``, in one W&B run per run
directory; what was already sent is recorded in ``<run_dir>/wandb_logged.json``.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "rft")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--project", default="rft-training")
    args = parser.parse_args()
    import wandb
    import wandb_log
    run_dir = Path(args.run_dir).resolve()
    id_path = run_dir / "wandb_run_id"
    if not id_path.exists():
        id_path.write_text(wandb.util.generate_id() + "\n")
    run = wandb.init(project=args.project, id=id_path.read_text().strip(), resume="allow",
                     name=f"{run_dir.name}/benchmark", group=run_dir.name, dir=str(run_dir))
    wandb.define_metric("eval_index")
    wandb.define_metric("eval/*", step_metric="eval_index")
    wandb.define_metric("round")
    wandb.define_metric("round/*", step_metric="round")
    logged = wandb_log.read_json(run_dir / "wandb_logged.json", None) or {}
    wandb_log.sync_reports(run_dir, logged)
    wandb_log.write_json(run_dir / "wandb_logged.json", logged)
    run.finish()


if __name__ == "__main__":
    main()
