"""Tokenize frozen prompts with the actual local SFT tokenizer, CPU only."""
import json
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
tokenizer = AutoTokenizer.from_pretrained(str(REPO / "runs/sft_granite_4.2_8b/checkpoint-200"), local_files_only=True)
manifest = json.loads((REPO / "runs/grpo_granite_4.2_8b/fixed_eval_manifest.json").read_text())
rows = []
for e in manifest["examples"]:
    n = len(tokenizer.encode(e["prompt"], add_special_tokens=False))
    rows.append({"id":e["_fixed_eval_id"], "length":n, "below_sft_prompt_limit":n < 2048,
                 "assistant_prefix_count":e["prompt"].count("<|im_start|>assistant")})
lookup = {r["id"]:r for r in rows}
results = {}
for step in (0,5,10):
    payload = json.loads((REPO / f"runs/grpo_granite_4.2_8b/fixed_eval/step-{step:06d}.json").read_text())
    groups = defaultdict(list)
    for r in payload["records"]:
        label = "below_2048" if lookup[r["example_id"]]["below_sft_prompt_limit"] else "2048_to_4095"
        groups[label].append(r["components"])
    results[step] = {k:{"completions":len(v), "detection":sum(x["detection"] for x in v)/len(v),
                            "exact_output":sum(x["expected_output_acc_logonly"] for x in v)/len(v)} for k,v in groups.items()}
out = {"prompts":rows,"by_length":results,"below_2048":sum(r["below_sft_prompt_limit"] for r in rows),
       "prefix_counts":sorted(set(r["assistant_prefix_count"] for r in rows))}
(OUT / "token_audit.json").write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k != "prompts"},indent=2))
