#!/usr/bin/env python
"""Does the model read the simulator's answer? The detect-or-miss fork, teacher forced.

After a tool answer the retry SFT teaches one of two readings: "The Good and
Bad Machine differ at ..." (detected, then the final answer) or "The Good and
Bad Machine agree on every primary output ..." / "No primary output has ..."
(not detected, then a new call). The choice sits on two tokens. This script
puts each context in front of the model and reads those next-token
probabilities:

* ``p_no``: "No" against "The" right after the ``<think>\\n`` that opens the turn;
* ``p_agree``: " agree" against " differ" (and the old " are") after
  ``<think>\\nThe Good and Bad Machine``.

``q`` = P(a not-detected reading) / P(either reading), with
P(not detected) = p_no + p_the * p_agree and P(detected) = p_the * p_differ.

Context sets:

* ``train_miss`` / ``train_detect``: the reading turns of the retry rows
  (``extra.jsonl``), rendered exactly as SFT tokenized them, labelled by the
  reading the row trains;
* ``gate_miss`` / ``gate_detect``: each gate slot up to its first tool answer,
  rendered as the evaluator renders it, labelled by whether that call detects
  (gate_report's reading of the snapshot).

A model that reads the answer has high ``q`` on the miss sets, low on the
detect sets (AUC near 1). One that learned the fork on its rows but does not
carry it over has a high AUC on the train sets and not on the gate sets. One
that never learned it has AUC near 0.5 on both.

One checkpoint per process (vLLM keeps its GPU memory until the process ends):

    python scripts/retry/fork_probe.py --name retry_ckpt20 --adapter runs/retry_sft_v4/sft/checkpoint-20 \\
        --extra runs/retry_sft_v4/data/extra.jsonl --gate runs/retry_sft_v4/gate/retry_ckpt20 \\
        --manifest runs/retry_sft_v4/gate/fixed_eval_manifest.json --out runs/retry_sft_v4/fork_probe
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean, median

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
sys.path.insert(0, str(REPO / "scripts" / "star"))
sys.path.insert(0, str(REPO / "scripts" / "retry"))

THINK = "<think>\n"
STEM = "The Good and Bad Machine"
MISS_HEADS = ("The Good and Bad Machine agree", "No primary output")
DETECT_HEAD = "The Good and Bad Machine differ"
# Granite 4.2's generation prompt opens the think block itself.
ROLE = "<|im_start|>assistant\n"
ASSISTANT = ROLE + THINK


def reading_label(content):
    """'miss', 'detect' or None for an assistant turn's text."""
    text = (content or "").lstrip()
    if text.startswith(THINK):
        text = text[len(THINK):]
    if text.startswith(MISS_HEADS):
        return "miss"
    if text.startswith(DETECT_HEAD):
        return "detect"
    return None


def train_contexts(path, tokenizer, per_label, seed, max_tokens):
    """Reading turns of the retry rows, as SFT rendered them."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    picked = {"miss": [], "detect": []}
    for row_index, row in enumerate(rows):
        if all(len(v) >= per_label for v in picked.values()):
            break
        messages = row["messages"]
        full = tokenizer.apply_chat_template(messages, tools=row.get("tools"), tokenize=False)
        starts = []
        position = 0
        while True:
            found = full.find(ROLE, position)
            if found < 0:
                break
            starts.append(found + len(ROLE))
            position = found + len(ROLE)
        assistant = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        if len(starts) != len(assistant):
            continue
        flags = row.get("assistant_train") or [True] * len(assistant)
        for k, i in enumerate(assistant):
            if not flags[k] or i == 0 or messages[i - 1].get("role") != "tool":
                continue
            if not full.startswith(THINK, starts[k]):
                continue
            label = reading_label(full[starts[k]:starts[k] + 200])
            if label is None or len(picked[label]) >= per_label:
                continue
            context = full[:starts[k] + len(THINK)]
            if len(tokenizer.encode(context, add_special_tokens=False)) > max_tokens:
                continue
            picked[label].append({"set": f"train_{label}", "id": f"row{row_index}_turn{k}",
                                  "retry": bool(row.get("retry")), "context": context})
    return picked["miss"] + picked["detect"]


def gate_contexts(run, manifest, tokenizer, max_tokens):
    """Each gate slot up to its first tool answer, as the evaluator renders it."""
    import glob
    from gate_report import faults_of, slot_facts
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
            facts = slot_facts(slot, fault)
            label = "detect" if facts["detected"][:1] == [True] else "miss"
            history = deepcopy(messages[:tool + 1])
            stringify_tool_arguments_for_template(history, tokenizer.chat_template)
            context = tokenizer.apply_chat_template(history, tokenize=False, tools=TOOLS,
                                                    add_generation_prompt=True,
                                                    truncate_history_thinking=False)
            if len(tokenizer.encode(context, add_special_tokens=False)) > max_tokens:
                continue
            following = messages[tool + 1]["content"] if tool + 1 < len(messages) else ""
            out.append({"set": f"gate_{label}", "id": slot["example_id"], "level": slot.get("level"),
                        "greedy": reading_label(following) or "other", "context": context})
    return out


def next_token_ids(tokenizer, prefix, words):
    """First token id of each word continuing ``prefix`` (prefix tokens must not change)."""
    base = tokenizer.encode(prefix, add_special_tokens=False)
    ids = {}
    for word in words:
        full = tokenizer.encode(prefix + word, add_special_tokens=False)
        if full[:len(base)] != base or len(full) <= len(base):
            raise ValueError(f"{word!r} changes the tokens of {prefix[-40:]!r}")
        ids[word] = full[len(base)]
    return ids


def auc(positives, negatives):
    """P(a random positive scores above a random negative), ties count half."""
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def summarize(records):
    sets = {}
    for record in records:
        sets.setdefault(record["set"], []).append(record["q"])
    out = {name: {"n": len(values), "mean_q": mean(values), "median_q": median(values),
                  "share_q_above_half": sum(v > 0.5 for v in values) / len(values)}
           for name, values in sorted(sets.items())}
    out["auc_train"] = auc(sets.get("train_miss", []), sets.get("train_detect", []))
    out["auc_gate"] = auc(sets.get("gate_miss", []), sets.get("gate_detect", []))
    gate = [r for r in records if r["set"].startswith("gate_")]
    if gate:
        out["gate_greedy_agreement"] = sum(
            (r["q"] > 0.5) == (r["greedy"] == "miss") for r in gate if r["greedy"] in ("miss", "detect")
        ) / max(1, sum(r["greedy"] in ("miss", "detect") for r in gate))
    return out


def probe(model_dir, tokenizer_dir, contexts, max_model_len, logprobs):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    llm = LLM(model=str(model_dir), tokenizer=str(tokenizer_dir), dtype="bfloat16", max_model_len=max_model_len,
              gpu_memory_utilization=0.9, enable_prefix_caching=True, max_logprobs=logprobs,
              trust_remote_code=True, seed=0)
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=logprobs)
    first = next_token_ids(tokenizer, ASSISTANT, ["The", "No"])
    second = next_token_ids(tokenizer, ASSISTANT + STEM, [" agree", " differ", " are"])
    prompts = []
    for context in contexts:
        for suffix in ("", STEM):
            prompts.append({"prompt_token_ids": tokenizer.encode(context["context"] + suffix,
                                                                 add_special_tokens=False)})
    outputs = llm.generate(prompts, sampling_params=params)
    floor = None
    records = []
    for index, context in enumerate(contexts):
        steps = []
        for output, ids in ((outputs[2 * index], first), (outputs[2 * index + 1], second)):
            table = output.outputs[0].logprobs[0]
            lowest = min(entry.logprob for entry in table.values())
            floor = lowest if floor is None else max(floor, lowest)
            # A word outside the top ``logprobs`` is below the lowest listed one.
            steps.append({word: math.exp(table[i].logprob if i in table else lowest)
                          for word, i in ids.items()})
        p_the, p_no = steps[0]["The"], steps[0]["No"]
        p_agree, p_differ = steps[1][" agree"], steps[1][" differ"]
        miss = p_no + p_the * p_agree
        detect = p_the * p_differ
        record = {k: v for k, v in context.items() if k != "context"}
        record.update({"p_the": p_the, "p_no": p_no, "p_agree": p_agree, "p_differ": p_differ,
                       "p_are": steps[1][" are"], "q": miss / (miss + detect) if miss + detect else 0.5})
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Arm name (output file stem)")
    parser.add_argument("--adapter", required=True, help="Checkpoint to probe")
    parser.add_argument("--extra", required=True, help="Retry SFT rows (extra.jsonl)")
    parser.add_argument("--gate", help="This checkpoint's gate arm dir (slots*.jsonl)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tokenizer", help="Base tokenizer dir (default: the adapter's base model)")
    parser.add_argument("--per_label", type=int, default=400)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--logprobs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", help="Build and check contexts only (no GPU)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter = Path(args.adapter).resolve()
    if args.tokenizer:
        tokenizer_dir = args.tokenizer
    else:
        from rationalize import local_model
        tokenizer_dir = local_model(json.loads((adapter / "adapter_config.json").read_text())
                                    ["base_model_name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    # The fork tokens must not merge with the prompt's last tokens.
    next_token_ids(tokenizer, ASSISTANT, ["The", "No"])
    next_token_ids(tokenizer, ASSISTANT + STEM, [" agree", " differ", " are"])
    limit = args.max_model_len - 64
    contexts = train_contexts(args.extra, tokenizer, args.per_label, args.seed, limit)
    if args.gate:
        contexts += gate_contexts(args.gate, args.manifest, tokenizer, limit)
    counts = {}
    for context in contexts:
        counts[context["set"]] = counts.get(context["set"], 0) + 1
    bad = sum(not c["context"].endswith(ASSISTANT) for c in contexts)
    print(f"[fork] {args.name}: {counts}, contexts not ending at a generation prompt: {bad}", flush=True)
    if args.dry_run:
        (out / f"{args.name}_dry_run.json").write_text(json.dumps({"contexts": counts, "bad_contexts": bad}))
        return
    from evaluate_model import resolve_merged_bf16_dir
    merged = resolve_merged_bf16_dir(adapter)
    records = probe(merged, tokenizer_dir, contexts, args.max_model_len, args.logprobs)
    with open(out / f"{args.name}.jsonl", "w") as sink:
        for record in records:
            sink.write(json.dumps(record) + "\n")
    result = {"adapter": str(adapter), "merged": str(merged), "contexts": counts, **summarize(records)}
    (out / f"{args.name}_summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
