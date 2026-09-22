"""Read-only W&B export for the requested SFT run; never prints credentials."""
import json
from pathlib import Path
import wandb

OUT = Path(__file__).resolve().parent
api = wandb.Api(timeout=45)
run = api.run("chrivasileiou/sft-training/n5wplmt8")
history = list(run.scan_history(page_size=1000))
files = [{"name": f.name, "size": f.size} for f in run.files()]
payload = {"path": "/".join(run.path), "name": run.name, "state": run.state,
           "config": run.config, "summary": dict(run.summary),
           "history": history, "files": files}
(OUT / "sft_online.json").write_text(json.dumps(payload, indent=2, default=str))
print(json.dumps({"path": payload["path"], "state": run.state,
                  "history_rows": len(history), "files": files,
                  "history_keys": sorted(set().union(*(r.keys() for r in history)))}, indent=2))
