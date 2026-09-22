"""Standalone experiment figures and size-stratified evaluation export."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[1]
summary=json.loads((OUT/'summary.json').read_text())
train=pd.read_csv(OUT/'training_metrics.csv')
ev=pd.DataFrame(summary['evaluations']).T
ev.index=ev.index.astype(int)
ev=ev.sort_index()
ev.drop(columns=['protocol','initial_checkpoint']).to_csv(OUT/'fixed_evaluation_metrics.csv',index_label='optimizer_step')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.grid':True,'grid.alpha':.18})
fig,axes=plt.subplots(2,2,figsize=(13,8.2),layout='constrained')
fig.suptitle('GRPO: modest training changes, no sustained functional gain',fontsize=18,fontweight='bold')
x=train['train/global_step']
ax=axes[0,0]
det=train['train/rewards/reward_fn/component_mean/detection']*100
ax.plot(x,det,color='#246F8B',alpha=.3,lw=1,label='Training update')
ax.plot(x,det.rolling(7,min_periods=7).mean(),color='#246F8B',lw=2.3,label='7-update trailing mean')
ax.set(title='Training detection varies with the prompt batch',ylabel='Detected samples (%)',ylim=(50,75))
ax.legend(frameon=False,fontsize=9)

ax=axes[0,1]
ax.plot(ev.index,ev.detection.astype(float)*100,'o-',color='#AC4B36',lw=2)
ax.axhline(float(ev.loc[0,'detection'])*100,color='#666666',ls=':',label='SFT checkpoint 200 baseline')
ax.set(title='Fixed detection: 64.81% → 63.43%',ylabel='Detected samples (%)',ylim=(50,75))
ax.legend(frameon=False,fontsize=9)

ax=axes[1,0]
ax.plot(ev.index,ev.expected_output_exact_fraction.astype(float)*100,'o-',color='#246F8B',label='All expected outputs correct')
ax.plot(ev.index,ev.usable_logged.astype(float)*100,'s-',color='#AC4B36',label='Detection + correct outputs + complete inputs')
ax.set(title='Usable answers remain rare: 7.41% → 6.02%',ylabel='All 216 completions (%)',ylim=(0,17))
ax.legend(frameon=False,fontsize=8.5,loc='upper left')

ax=axes[1,1]
steps=list(ev.index)
delta=[summary['paired_detection'][f'0_to_{s}']['delta']*100 for s in steps]
lo=[summary['paired_detection'][f'0_to_{s}']['ci95'][0]*100 for s in steps]
hi=[summary['paired_detection'][f'0_to_{s}']['ci95'][1]*100 for s in steps]
ax.errorbar(steps,delta,yerr=[np.array(delta)-lo,np.array(hi)-delta],fmt='o',capsize=3,color='#5D557A',ms=4)
ax.axhline(0,color='#666666',ls=':')
ax.set(title='Paired detection change remains uncertain',ylabel='Change vs SFT baseline (percentage points)',ylim=(-14,16))
for ax in axes.flat:
    ax.axvline(34.5,color='#888888',ls='--',lw=1,alpha=.65)
    ax.set(xlabel='Cumulative optimizer step',xlim=(-2,70))
fig.supxlabel('Dashed line: session boundary. Fixed set: 72 circuits × 3 samples; latest evaluation is step 65, latest checkpoint is 67.\nIntervals: paired circuit bootstrap (20,000 draws), one seed; historical set overlaps SFT training.',fontsize=9)
fig.savefig(OUT/'diagnostics.png',dpi=180)
fig.savefig(OUT/'diagnostics.pdf')
plt.close(fig)

tokens=json.loads((REPO/'analysis/run_5gmb8rfn/token_audit.json').read_text())
short={r['id'] for r in tokens['prompts'] if r['below_sft_prompt_limit']}
strata={}
for step in steps:
    payload=json.loads((REPO/f'runs/grpo_granite_4.2_8b/fixed_eval/step-{step:06}.json').read_text())
    strata[step]={}
    for label,is_short in [('prompt_lt_2048',True),('prompt_ge_2048',False)]:
        rs=[r for r in payload['records'] if (r['example_id'] in short)==is_short]
        strata[step][label]={
            'n':len(rs),
            'detection':np.mean([r['components']['detection'] for r in rs]),
            'exact_output':np.mean([r['components']['expected_output_acc_logonly'] for r in rs]),
            'usable':np.mean([r['components']['detection']==1 and r['components']['expected_output_acc_logonly']==1 and r['components']['pi_completeness_logonly']==1 for r in rs]),
        }
(OUT/'length_strata.json').write_text(json.dumps(strata,indent=2)+'\n')
print(json.dumps({k:strata[k] for k in [0,30,65]},indent=2))
