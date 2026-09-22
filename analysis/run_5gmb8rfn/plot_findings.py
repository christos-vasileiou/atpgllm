"""Reproduce experiment diagnostics from saved exports; no network or training."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
data = json.loads((OUT / "grpo_wandb_snapshot.json").read_text())
rows = [r for r in data["metrics"] if "train/reward" in r]
local = json.loads((OUT / "local_summary.json").read_text())
sft = [r for r in json.loads((OUT / "sft_online.json").read_text())["history"] if r.get("train/loss") is not None]
evals = local["paired_evaluation"]
audit = json.loads((OUT / "completion_audit.json").read_text())
steps = [r["train/global_step"] for r in rows]
def values(key): return np.array([r["train/"+key] for r in rows])
stats = {}
for key in ("reward", "rewards/reward_fn/raw_mean", "rewards/reward_fn/component_mean/detection", "entropy", "grad_norm", "kl", "completions/clipped_ratio", "gdpo/objective_active_fraction/detection", "frac_reward_zero_std", "step_time"):
    v = values(key)
    stats[key] = {"first_five_mean":float(v[:5].mean()),"last_five_mean":float(v[-5:].mean()),"mean":float(v.mean()),"min":float(v.min()),"max":float(v.max())}
(OUT / "metric_statistics.json").write_text(json.dumps(stats,indent=2))
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(2,3,figsize=(15,8.5),layout="constrained")
ax=axes[0,0]
ax.plot(steps,100*values("rewards/reward_fn/component_mean/detection"),"o-",color="#86a5c6",label="Training; changing prompts")
evsteps=sorted(map(int,evals))
ax.plot(evsteps,[100*evals[str(s)]["metrics"]["detection"] for s in evsteps],"o-",color="#b24432",lw=2,label="Fixed 72 faults × 3 samples")
ax.set(title="Raw detection: no demonstrated gain",xlabel="GRPO optimizer step",ylabel="Detection (%)",ylim=(45,80));ax.legend(fontsize=8)
ax=axes[0,1]
d=[evals[str(s)]["detection_delta"]*100 for s in evsteps[1:]]
ci=np.array([evals[str(s)]["paired_fault_bootstrap_95ci"] for s in evsteps[1:]])*100
ax.errorbar(evsteps[1:],d,yerr=[np.array(d)-ci[:,0],ci[:,1]-d],fmt="o",capsize=6,color="#b24432")
ax.axhline(0,color="gray",ls="--");ax.set(title="Paired change from SFT baseline",xlabel="GRPO optimizer step",ylabel="Detection change (percentage points)",xticks=evsteps[1:]);ax.text(.04,.04,"95% paired bootstrap over 72 faults/circuits\nOne fixed seed; circuit overlap with SFT",transform=ax.transAxes,fontsize=8)
ax=axes[0,2]
ax.semilogy([r["train/global_step"] for r in sft],[r["train/loss"] for r in sft],color="#355d83")
ax.set(title="Online SFT: training loss only",xlabel="SFT optimizer step",ylabel="Training NLL (log scale)");ax.text(.25,.8,"Final loss 0.0152\nToken accuracy 99.28%\nNo validation history",transform=ax.transAxes)
ax=axes[1,0]
ax.plot(steps,values("entropy"),"o-",color="#355d83");ax.set(title="No observed GRPO entropy collapse",xlabel="GRPO optimizer step",ylabel="Mean token entropy",ylim=(0,.06))
ax=axes[1,1]
ax.plot(steps,values("gdpo/objective_active_fraction/detection")*100,"o-",label="Groups with detection variance",color="#356e57")
ax.plot(steps,values("frac_reward_zero_std")*100,"o-",label="Zero scalar-variance groups",color="#b24432")
ax.set(title="Useful reward differences are present",xlabel="GRPO optimizer step",ylabel="Prompt groups (%)",ylim=(-3,103));ax.legend(fontsize=8)
ax=axes[1,2]
positions=np.arange(len(evsteps));width=.36
ax.bar(positions-width/2,[100*audit[str(s)]["counts"]["final_output_equals_pretool_guess"]/audit[str(s)]["counts"]["comparable_tool_final"] for s in evsteps],width,label="Final output repeats pre-tool guess",color="#d29854")
ax.bar(positions+width/2,[100*evals[str(s)]["metrics"]["expected_output_exact_fraction"] for s in evsteps],width,label="All expected outputs correct",color="#355d83")
ax.set(title="Tool feedback rarely changes the answer",xlabel="GRPO optimizer step",ylabel="Responses (%)",xticks=positions,xticklabels=evsteps,ylim=(0,112));ax.legend(fontsize=8,loc="center right")
fig.suptitle("Run 5gmb8rfn: metric semantics, slow updates, and learned tool-use behavior",fontsize=16)
fig.savefig(OUT / "diagnostics.png",dpi=170)
fig.savefig(OUT / "diagnostics.pdf")
print("Saved diagnostics.png, diagnostics.pdf, and metric_statistics.json")
