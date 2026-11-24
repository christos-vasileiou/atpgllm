#!/usr/bin/env python
"""Can the model read a fault simulator answer? A plain question, no SFT template.

Each gate slot is cut after its first tool answer and gets one user question:
the primary outputs are listed, and the model must say which of them differ
between the Good and the Bad Machine (0 in one, 1 in the other) and end with
``VERDICT: DETECTED`` or ``VERDICT: NOT DETECTED``. Greedy decoding, thinking
on (Granite 4.2's generation prompt opens it). The truth is gate_report's
reading of the same snapshot.

The same contexts go to every model, so base Granite, the v4 SFT checkpoints,
RFT round 11 and the retry SFT are compared on one test: does training on
"every tool answer means detected" take away a reading skill the base model has?

    python scripts/retry/reading_test.py --name base --gate runs/retry_sft_v4/gate/retry_ckpt20 \\
        --manifest runs/retry_sft_v4/gate/fixed_eval_manifest.json --out runs/retry_sft_v4/reading_test
    python scripts/retry/reading_test.py --name sft250 --adapter <checkpoint> ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
sys.path.insert(0, str(REPO / "scripts" / "star"))
sys.path.insert(0, str(REPO / "scripts" / "retry"))

QUESTION = (
    "Before anything else, read the fault simulation result above. The primary outputs of this "
    "circuit are: {outputs}. A primary output differs only when it is 0 in one machine and 1 in the "
    "other (x is unknown and never counts). List the primary outputs whose Good Machine and Bad Machine "
    "values differ, or write none. Then finish with exactly one line: VERDICT: DETECTED if at least one "
    "primary output differs, otherwise VERDICT: NOT DETECTED."
)
VERDICT_RE = re.compile(r"VERDICT:\s*(NOT\s+DETECTED|DETECTED)", re.IGNORECASE)


def contexts(run, manifest, tokenizer):
    from gate_report import _outputs, faults_of, slot_facts
    from atpgllm.training.revert_template import stringify_tool_arguments_for_template
    from atpgllm.training.tools import TOOLS
    faults = faults_of(manifest)
    out = []
    for path in sorted(glob.glob(str(Path(run) / "slots*.jsonl"))):
        for line in open(path):
            slot = json.loads(line)
            fault = faults.get(slot["example_id"])
            messages = slot.get("messages") or []
            tool = next((i for i, m in enumerate(messages) if m.get("role") == "tool"), None)
            if fault is None or tool is None:
                continue
            outputs = _outputs(slot["observations"][0])
            if not outputs:
                continue
            truth = slot_facts(slot, fault)["detected"][0]
            history = deepcopy(messages[:tool + 1])
            history.append({"role": "user", "content": QUESTION.format(outputs=", ".join(outputs))})
            stringify_tool_arguments_for_template(history, tokenizer.chat_template)
            prompt = tokenizer.apply_chat_template(history, tokenize=False, tools=TOOLS,
                                                   add_generation_prompt=True, truncate_history_thinking=False)
            out.append({"id": slot["example_id"], "level": slot.get("level"), "truth": bool(truth),
                        "prompt": prompt})
    return out


def score(records):
    def rate(group, ok):
        return sum(ok(r) for r in group) / len(group) if group else None
    detects = [r for r in records if r["truth"]]
    misses = [r for r in records if not r["truth"]]
    recall_detect = rate(detects, lambda r: r["verdict"] == "detected")
    recall_miss = rate(misses, lambda r: r["verdict"] == "not_detected")
    return {"n": len(records), "detects": len(detects), "misses": len(misses),
            "answered": rate(records, lambda r: r["verdict"] is not None),
            "accuracy": rate(records, lambda r: r["verdict"] == ("detected" if r["truth"] else "not_detected")),
            "recall_detected": recall_detect, "recall_not_detected": recall_miss,
            "balanced_accuracy": (recall_detect + recall_miss) / 2 if detects and misses else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--adapter", help="Checkpoint (LoRA); omit for the base model")
    parser.add_argument("--base", default="ibm-granite/granite-4.2-8b")
    parser.add_argument("--gate", required=True, help="Gate arm dir whose first calls give the contexts")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer
    from rationalize import local_model
    if args.adapter:
        adapter = Path(args.adapter).resolve()
        base = local_model(json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"])
        from evaluate_model import resolve_merged_bf16_dir
        model = resolve_merged_bf16_dir(adapter)
    else:
        base = local_model(args.base)
        model = base
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    items = contexts(args.gate, args.manifest, tokenizer)
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(model), tokenizer=str(base), dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.9, enable_prefix_caching=True, trust_remote_code=True, seed=0)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    prompts = [{"prompt_token_ids": tokenizer.encode(item["prompt"], add_special_tokens=False)} for item in items]
    outputs = llm.generate(prompts, sampling_params=params)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    with open(out / f"{args.name}.jsonl", "w") as sink:
        for item, output in zip(items, outputs):
            text = output.outputs[0].text
            answer = text.rsplit("</think>", 1)[-1]
            found = VERDICT_RE.findall(answer) or VERDICT_RE.findall(text)
            verdict = None
            if found:
                verdict = "not_detected" if found[-1].upper().startswith("NOT") else "detected"
            record = {"id": item["id"], "level": item["level"], "truth": item["truth"], "verdict": verdict,
                      "tokens": len(output.outputs[0].token_ids), "answer_tail": answer[-400:]}
            records.append(record)
            sink.write(json.dumps(record) + "\n")
    result = {"name": args.name, "model": str(model), "all": score(records),
              "by_level": {level: score([r for r in records if r["level"] == level])
                           for level in sorted({r["level"] for r in records})}}
    (out / f"{args.name}_summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
