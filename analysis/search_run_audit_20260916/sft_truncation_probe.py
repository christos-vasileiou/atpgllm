"""Sample cached training rows and locate final-answer labels under the SFT cap.

This probes the stored formatter/template, not the checkpoint's exact consumed
training stream. Bernoulli row sampling is fixed before examining lengths.
"""
import collections
import copy
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pyarrow as pa

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from prepare_circuit_comparison import parser_and_tools


def main():
    from transformers import AutoTokenizer
    _, tools = parser_and_tools()
    for name, path in [("atpgllm", ROOT / "atpgllm"), ("atpgllm.training", ROOT / "atpgllm/training")]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    tool_module = types.ModuleType("atpgllm.training.tools")
    tool_module.FAULT_SIMULATION_TOOL = tools[0]
    sys.modules[tool_module.__name__] = tool_module
    from atpgllm.training.conversation import ConversationExample
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "runs/sft_granite_4.2_8b/checkpoint-200", local_files_only=True)
    cache = Path('/home/eng/c/cxv200006/.cache/huggingface/datasets/chrivasileiou___asap7-language-of-test-v2/default/0.0.0/d35cfea64eadf30fb3b39735b0e8d20bffcc3345')
    rng = np.random.default_rng(20260916)
    counts, examples = collections.Counter(), []
    for file_index, path in enumerate(sorted(cache.glob('*-train-*.arrow'))):
        with pa.memory_map(str(path), 'r') as source:
            for batch in pa.ipc.open_stream(source):
                chosen = np.flatnonzero(rng.random(batch.num_rows) < .001)
                if not len(chosen):
                    continue
                for record in batch.take(pa.array(chosen)).to_pylist():
                    counts['sampled_rows'] += 1
                    try:
                        messages = ConversationExample.from_record(copy.deepcopy(record), use_tools=True).messages
                        prompt = [m for m in messages if m['role'] in ['system', 'user']]
                        prompt_ids = tokenizer.apply_chat_template(prompt, tools=tools, add_generation_prompt=True)
                        if len(prompt_ids) >= 2048:
                            counts['excluded_prompt_length'] += 1
                            continue
                        full = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
                        ids = tokenizer.encode(full, add_special_tokens=False)
                        masked = tokenizer.apply_chat_template(messages, tools=tools,
                            return_dict=True, return_assistant_tokens_mask=True)
                        if masked['input_ids'] != ids:
                            raise ValueError('Assistant-mask tokenization differs from full sequence')
                        last_answer = full.rfind('EXPECTED_OUTPUT:')
                        # Prefix tokenization may differ at the cut boundary by
                        # a token; interpret positions far from 8192 robustly.
                        before_output = len(tokenizer.encode(full[:last_answer], add_special_tokens=False)) if last_answer >= 0 else None
                        output_end = full.find('DETECTED_FAULTS:', last_answer) if last_answer >= 0 else -1
                        output_end = len(full) if output_end < 0 else output_end
                        offsets = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
                        output_tokens = [i for i, (start, end) in enumerate(offsets)
                                         if last_answer >= 0 and end > last_answer and start < output_end]
                        retained_output_tokens = [i for i in output_tokens if i < 8192]
                        supervised_output_tokens = [i for i in retained_output_tokens if masked['assistant_masks'][i]]
                        counts['eligible_rows'] += 1
                        counts['sequence_exceeds_8192'] += int(len(ids) > 8192)
                        counts['expected_output_starts_after_8192'] += int(before_output is not None and before_output >= 8192)
                        counts['missing_expected_output_field'] += int(last_answer < 0)
                        counts['rows_with_retained_output_tokens'] += bool(retained_output_tokens)
                        counts['rows_with_supervised_output_tokens'] += bool(supervised_output_tokens)
                        counts['rows_with_retained_but_fully_masked_output'] += bool(retained_output_tokens) and not bool(supervised_output_tokens)
                        examples.append(dict(netlist_sha256=hashlib.sha256(record['netlist'].encode()).hexdigest(),
                            fault=record['fault'], prompt_tokens=len(prompt_ids), total_tokens=len(ids),
                            expected_output_start=before_output,
                            retained_output_tokens=len(retained_output_tokens),
                            supervised_output_tokens=len(supervised_output_tokens)))
                    except Exception as exc:
                        counts['format_errors'] += 1
                        examples.append(dict(error=str(exc)))
        print(f"SFT truncation probe {file_index+1}/31", flush=True)
    result = dict(sample_probability=.001, seed=20260916, counts=counts, examples=examples,
        caveat="Cached training rows under the unchanged SFT formatter and checkpoint-200 template; actual training order/exposure and exact dependency revision not reconstructed. This is a truncation-risk probe, not a causal experiment.")
    (OUT / 'sft_truncation_probe.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
