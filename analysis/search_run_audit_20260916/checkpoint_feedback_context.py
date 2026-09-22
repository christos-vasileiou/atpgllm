"""Locate wrong-output copying relative to the old SFT context boundary."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from prepare_circuit_comparison import parser_and_tools
from comparison_diagnostics import assignment


def main():
    from transformers import AutoTokenizer
    _, tools = parser_and_tools()
    manifest = json.loads((OUT / "circuit_comparison_explicit_manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer"], local_files_only=True)
    variable_ids = {r["source_circuit_id"] for r in json.loads((OUT / "comparison_output_baselines.json").read_text())["circuits"] if r["observed_variable_output_bits"]}
    output = ROOT / "runs/circuit_comparison_explicit_20260916"
    results, examples = {}, {}
    for path in sorted(output.glob('*_tools1_seed42.json')):
        data = json.loads(path.read_text())
        groups = defaultdict(Counter)
        for row in data['per_problem_results']:
            for slot in row['search_slots']:
                if slot['status'] != 'FINAL' or not slot['observations']:
                    continue
                obs = slot['observations'][-1]
                if slot['vector'] != obs['vector']:
                    continue
                fields = dict(re.findall(r'\b(INPUT_VECTOR|EXPECTED_OUTPUT):\s*"([^"]*)"', slot['final_answer'].rsplit('</think>', 1)[-1]))
                expected = assignment(fields.get('EXPECTED_OUTPUT', ''))
                requested = assignment(obs['arguments']['output_vector'])
                good = {k: json.loads(obs['result'])['Good Machine'][k] for k in requested}
                full = tokenizer.apply_chat_template(slot['messages'], tools=tools, tokenize=False,
                    add_generation_prompt=False, truncate_history_thinking=False)
                length = len(tokenizer.encode(full, add_special_tokens=False))
                counts = dict(matched_final_slots=1, wrong_request=int(requested != good),
                    corrected_wrong_request=int(requested != good and expected == good),
                    copied_wrong_request=int(requested != good and expected == requested))
                for group in ['all'] + (['full_history_le_8192'] if length <= 8192 else []) + (['full_history_le_7680'] if length <= 7680 else []):
                    groups[group].update(counts)
                if row['source_circuit_id'] in variable_ids and counts['copied_wrong_request'] and row['variant'] == 'original':
                    item = dict(job=path.stem, comparison_id=row['comparison_id'], module=row['module_name'],
                        completion_slot=slot['completion_slot'], rendered_full_history_tokens=length,
                        vector=slot['vector'], requested_output=requested, good_output=good,
                        final_expected_output=expected, detection=slot['reward_components'].get('detection', 0))
                    previous = examples.get(path.stem)
                    if previous is None or length < previous['rendered_full_history_tokens']:
                        examples[path.stem] = item
        results[path.stem] = dict(groups)
    result = dict(groups=results, short_wrong_copy_examples=examples,
        caveat="Valid finals whose input equals the last simulated input. Full histories are reconstructed with the shared tokenizer/template and thinking preserved, not a trace of every intermediate generation prefix. The 7,680-token stratum leaves a margin below 8,192. This diagnoses copying within the earlier context size; it does not isolate the causal effect of SFT training length.")
    (OUT / 'checkpoint_feedback_context.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
