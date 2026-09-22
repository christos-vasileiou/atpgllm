"""Reconstruct saved greedy prompt lengths; no model loading or generation.

These are observational length strata, not a causal test of SFT truncation.
Lengths use the local SFT tokenizer/template and the production tool schema.
"""
import collections
import importlib.util
import json
import re
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from prepare_circuit_comparison import parser_and_tools


def bit_diagnostics(data, lengths):
    groups = collections.defaultdict(collections.Counter)
    def assignment(value):
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value)
        except ValueError:
            return {k.strip(): int(v) for k, v in [item.rsplit(":", 1) for item in value.split(",")]}
    for row, length in zip(data["per_problem_results"], lengths):
        n = length["initial_prompt_tokens"]
        bucket = "initial_le_2048" if n <= 2048 else "initial_2049_4096" if n <= 4096 else "initial_gt_4096"
        for slot in row["search_slots"]:
            if slot["status"] != "FINAL" or not slot.get("observations"):
                continue
            obs = slot["observations"][-1]
            fields = dict(re.findall(r'\b(INPUT_VECTOR|EXPECTED_OUTPUT):\s*"([^"]*)"', slot["final_answer"]))
            try:
                vector = assignment(fields.get("INPUT_VECTOR", ""))
                expected = assignment(fields.get("EXPECTED_OUTPUT", ""))
                if vector != obs["vector"]:
                    continue
                outputs = assignment(obs["arguments"]["output_vector"])
                good = json.loads(obs["result"])["Good Machine"]
                if not all(good.get(k) in (0, 1) for k in outputs):
                    continue
            except (ValueError, KeyError, TypeError):
                continue
            bits = sum(expected.get(k) == good[k] for k in outputs)
            groups[bucket].update(slots=1, output_bits=len(outputs), correct_bits=bits,
                                 exact=int(bits == len(outputs)), pattern_bit_accuracy_sum=bits/len(outputs))
    for values in groups.values():
        values["micro_bit_accuracy"] = values["correct_bits"]/values["output_bits"]
        values["macro_pattern_bit_accuracy"] = values["pattern_bit_accuracy_sum"]/values["slots"]
        values["mean_output_bits"] = values["output_bits"]/values["slots"]
    return dict(scope="FINAL slots matching the last tool vector; missing output bits are incorrect. Conditional sample; length is confounded with output width and circuit difficulty.", groups=groups)


def main():
    from transformers import AutoTokenizer
    _, tools = parser_and_tools()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "runs/sft_granite_4.2_8b/checkpoint-200", local_files_only=True)
    path = next((ROOT / "runs/eval_results_grpo_granite_4.2_8b_policy").glob("*_greedy_*.json"))
    data = json.loads(path.read_text())
    groups = collections.defaultdict(collections.Counter)
    per_problem = []
    def length(messages):
        text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                add_generation_prompt=True, truncate_history_thinking=False)
        return len(tokenizer.encode(text, add_special_tokens=False))
    for i, row in enumerate(data["per_problem_results"]):
        initial = [m for m in row["search_slots"][0]["messages"] if m["role"] in ["system", "user"]][:2]
        prompt_len = length(initial)
        bucket = "initial_le_2048" if prompt_len <= 2048 else "initial_2049_4096" if prompt_len <= 4096 else "initial_gt_4096"
        counters = collections.Counter()
        for slot in row["search_slots"]:
            keys = ["all", bucket]
            # Reconstruct the saved conversation after the last tool response.
            # It does not recover unsaved intermediate token-level prefixes.
            messages = slot["messages"]
            indices = [j for j, m in enumerate(messages) if m["role"] == "tool"]
            if indices:
                post_tool_len = length(messages[:indices[-1]+1])
                keys.append("post_tool_le_8192" if post_tool_len <= 8192 else "post_tool_gt_8192")
                if prompt_len <= 2048 and post_tool_len <= 8192:
                    keys.append("initial_le_2048_and_post_tool_le_8192")
                counters["post_tool_tokens_sum"] += post_tool_len
                counters["tool_slots"] += 1
            comps = slot["reward_components"]
            values = {"slots": 1, "valid_finals": int(slot["status"] == "FINAL"),
                      "correct_expected_outputs": int(comps.get("expected_output_acc_logonly", 0) >= 1),
                      "detection": int(comps.get("detection", 0) >= 1)}
            for key in keys:
                groups[key].update(values)
            counters.update(values)
        per_problem.append(dict(idx=i, module=row["module_name"], fault=row["fault"],
                                initial_prompt_tokens=prompt_len, **counters))
        if i % 64 == 0:
            print(f"Length diagnostics {i+1}/512", flush=True)
    result = dict(caveat="Reconstructed saved histories; no causal training-length intervention and no measurement of actual SFT label truncation.",
                  groups={key: {**value, "expected_output_accuracy": value["correct_expected_outputs"]/value["slots"]}
                          for key, value in groups.items()}, per_problem=per_problem)
    (OUT / "context_length_diagnostics.json").write_text(json.dumps(result, indent=2)+"\n")
    (OUT / "context_output_bit_diagnostics.json").write_text(json.dumps(bit_diagnostics(data, per_problem), indent=2)+"\n")
    print(json.dumps(result["groups"], indent=2))


if __name__ == "__main__":
    main()
