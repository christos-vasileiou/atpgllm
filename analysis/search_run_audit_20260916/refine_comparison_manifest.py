"""Make output grammar and tool limits explicit, retaining the frozen faults.

The initial base pilot exposed a format ambiguity. Preserve its manifest and
results; apply this shared prompt amendment to every checkpoint in a new run.
"""
import copy
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from prepare_circuit_comparison import comparison, parser_and_tools

PROTOCOL = '''You are an ATPG assistant for structural Verilog netlists. A fault "sa0 NET" means NET stuck at 0; "sa1 NET" means NET stuck at 1. Find a binary input vector that makes at least one primary output differ between the fault-free and faulty circuits.

The final answer must contain exactly these three fields, with a colon after each field name and its value in double quotes. This example illustrates syntax only; replace all example names and values with the actual circuit's signals and target fault:
INPUT_VECTOR: "a: 0, b[0]: 1, b[1]: 0"
EXPECTED_OUTPUT: "y: 1"
DETECTED_FAULTS: "sa0 n1"

INPUT_VECTOR must list every primary-input scalar and bus bit exactly once. EXPECTED_OUTPUT must list every primary-output scalar and bus bit exactly once, using their values in the FAULT-FREE circuit for that input. Use the actual declared bit indices, comma-separated name: 0 or name: 1 entries, and no aggregate buses, ranges, ellipses, or omitted bits. DETECTED_FAULTS lists the exact target fault if detected, otherwise an empty quoted string. Do not place equals signs between names and values.

If fault_simulation_tool is available, you may call it at most once. Supply exactly its four arguments: input_vector and output_vector dictionaries with binary values as strings, the exact fault string, and the exact doc_id from the user. The simulator response's Good Machine column contains fault-free values; read the primary-output entries for EXPECTED_OUTPUT. After a tool response, finish with the three final fields; do not request another simulation. If no tool is provided, solve directly. Keep reasoning concise and finish the final answer within the available context.'''


def main():
    from transformers import AutoTokenizer
    source = OUT / "circuit_comparison_manifest.json"
    manifest = comparison.load_manifest(source)
    original = copy.deepcopy(manifest["examples"])
    tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer"], local_files_only=True)
    _, tools = parser_and_tools()
    for row in manifest["examples"]:
        row["comparison_messages"][0]["content"] = PROTOCOL
        rendered = tokenizer.apply_chat_template(row["comparison_messages"], tools=tools,
            tokenize=False, add_generation_prompt=True)
        row["prompt_tokens"] = len(tokenizer.encode(rendered, add_special_tokens=False))
        if row["prompt_tokens"] > 2048:
            raise ValueError("Explicit protocol exceeds the SFT prompt window")
    for before, after in zip(original, manifest["examples"]):
        for key in set(before) - {"comparison_messages", "prompt_tokens"}:
            if before[key] != after[key]:
                raise ValueError(f"Prompt amendment changed {key}")
        if before["comparison_messages"][1:] != after["comparison_messages"][1:]:
            raise ValueError("Prompt amendment changed circuit/fault presentation")
    manifest["examples_sha256"] = comparison.digest(manifest["examples"])
    manifest["protocol_amendment"] = dict(source_manifest=str(source),
        source_examples_sha256=comparison.digest(original),
        reason="Initial base 8K one-tool pilot had 0/144 valid finals; inspected completions exposed ambiguous output grammar and an undisclosed one-tool limit. All checkpoints receive the same explicit instructions in a separate run; no circuit or fault selection changes.")
    dest = OUT / "circuit_comparison_explicit_manifest.json"
    dest.write_text(json.dumps(manifest, indent=2)+"\n")
    comparison.load_manifest(dest)
    print(f"Preserved {len(original)} paired records; prompt tokens {min(r['prompt_tokens'] for r in manifest['examples'])}–{max(r['prompt_tokens'] for r in manifest['examples'])}; wrote {dest}")


if __name__ == "__main__":
    main()
