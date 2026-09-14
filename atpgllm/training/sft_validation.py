"""Audited local circuit splits and a fixed, balanced SFT validation sample."""
import copy
import json
from pathlib import Path

from ._paths import ensure_data_preprocessing_on_path

ensure_data_preprocessing_on_path()
from circuit_split import audit_dataset, digest, file_digest


def check_sft_dataset(directory, resume_from=None):
    directory = Path(directory)
    manifest = audit_dataset(directory)
    fingerprint = file_digest(directory / 'split_manifest.json')
    if resume_from:
        checkpoint = Path(resume_from)
        candidates = (checkpoint / 'sft_data_manifest.json', checkpoint.parent / 'sft_data_manifest.json')
        previous = next((p for p in candidates if p.is_file()), None)
        if previous is None or json.loads(previous.read_text()).get('dataset_sha256') != fingerprint:
            raise ValueError('Resume checkpoint has no matching repaired dataset provenance. '
                             'Start this SFT experiment from the base model: old SFT weights '
                             'may already have seen these validation circuits.')
    return manifest, fingerprint


def load_circuit_train(directory):
    from datasets import load_dataset
    files = sorted(str(p) for p in (Path(directory) / 'train').glob('*.parquet'))
    return load_dataset('parquet', data_files={'train': files}, split='train', streaming=True)


def prepare_circuit_validation(directory, manifest, tokenizer, *, sft_format,
                               max_prompt_length, max_model_len, per_circuit=8):
    from collections import Counter
    import pyarrow.parquet as pq
    from datasets import Dataset
    from .dataset_utils import _classify_sft_messages_batch, _format_sft_messages_record, _gate_filter_passes
    from .tools import TOOLS

    if per_circuit < 1:
        raise ValueError('SFT validation examples per circuit must be positive')
    use_tools = bool(getattr(tokenizer, 'chat_template', None) and 'tool' in tokenizer.chat_template)
    groups = {r['circuit_id']: [] for r in manifest['circuits'] if r.get('split') == 'validation'}
    for path in sorted((Path(directory) / 'validation').glob('*.parquet')):
        for row in pq.read_table(path).to_pylist():
            groups[row['circuit_id']].append(row)
    selected, identities, counts = [], [], {}
    for circuit, rows in sorted(groups.items()):
        # Ordering is independent of shard order, process count and Python hash seed.
        def key(row):
            return digest(json.dumps([manifest['seed'], circuit, row['fault'], row['input_vector']], sort_keys=True))
        rows.sort(key=key)
        statuses = (_classify_sft_messages_batch(rows, tokenizer, use_tools, max_prompt_length)
                    if sft_format == 'messages' else ['ok' if _gate_filter_passes(r) else 'gate' for r in rows])
        reasons = Counter(statuses)
        kept_faults = set()
        for row, status in zip(rows, statuses, strict=True):
            if status != 'ok' or row['fault'] in kept_faults or len(kept_faults) >= per_circuit:
                continue
            formatted = _format_sft_messages_record(copy.deepcopy(row), use_tools)
            rendered = tokenizer.apply_chat_template(formatted['messages'], tokenize=False,
                                                      tools=TOOLS if use_tools else None)
            length = len(tokenizer(rendered, add_special_tokens=False)['input_ids'])
            if length > max_model_len:
                reasons['full_sequence_too_long'] += 1
                continue
            selected.append(formatted if sft_format == 'messages' else {'text': rendered})
            kept_faults.add(row['fault'])
            identities.append({k: row[k] for k in ('circuit_id', 'netlist_id', 'source_module_name',
                                                    'module_name', 'fault', 'pattern_index')})
        if not kept_faults:
            raise ValueError(f'Validation circuit {circuit} has no examples after SFT filtering: {dict(reasons)}')
        counts[circuit] = dict(selected=len(kept_faults), filtering=dict(reasons))
    if not selected:
        raise ValueError('No circuit validation examples')
    return Dataset.from_list(selected), {'examples': identities, 'circuits': counts,
                                         'max_prompt_length': max_prompt_length,
                                         'max_model_len': max_model_len, 'per_circuit': per_circuit}


def sft_evaluation_config(steps):
    if steps < 1:
        raise ValueError('SFT evaluation interval must be positive')
    return dict(eval_strategy='steps', eval_steps=steps, save_steps=steps,
                eval_on_start=True, per_device_eval_batch_size=1,
                prediction_loss_only=True, load_best_model_at_end=True,
                metric_for_best_model='eval_loss', greater_is_better=False)
