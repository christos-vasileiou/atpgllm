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


def load_circuit_train(directory, seed=None):
    """The train split, streamed in a seeded order across all circuits.

    Shards hold one circuit after another; reading them in file order would
    train on a few circuits at a time. The order depends only on the shards
    and ``seed`` (default: the split manifest's seed), so counting consumed
    rows still resumes at the same place.
    """
    from .stream_buffer import streaming_split
    if seed is None:
        manifest = Path(directory) / 'split_manifest.json'
        seed = json.loads(manifest.read_text()).get('seed', 0) if manifest.is_file() else 0
    return streaming_split(directory, 'train', seed=seed)


_REASON_TEXT = {
    'length': 'prompt reaches max_prompt_length',
    'too_long': 'full conversation exceeds max_model_len',
    'gate': 'netlist removed by the gate-count filter (1, 5 or 84 gates)',
    'full_sequence_too_long': 'full conversation exceeds max_model_len',
}


def _ordering_index(directory, circuits, seed):
    """Per validation circuit, its rows in a fixed order, from the ordering columns only."""
    import pyarrow.parquet as pq
    index = {circuit: [] for circuit in circuits}
    for path in sorted((Path(directory) / 'validation').glob('*.parquet')):
        table = pq.read_table(path, columns=['circuit_id', 'fault', 'input_vector'])
        for position, (circuit, fault, vector) in enumerate(zip(
                *(table.column(c).to_pylist() for c in ('circuit_id', 'fault', 'input_vector')))):
            if circuit not in index:
                raise KeyError(f'Validation row of circuit {circuit} is missing from split_manifest.json')
            # Ordering is independent of shard order, process count and Python hash seed.
            key = digest(json.dumps([seed, circuit, fault, vector], sort_keys=True))
            index[circuit].append((key, str(path), position))
    for rows in index.values():
        rows.sort()
    return index


def _skip_message(circuit, info, examined, reasons, shortest_prompt, max_prompt_length, max_model_len):
    causes = '; '.join(f'{_REASON_TEXT.get(r, r)} ({n} rows)' for r, n in reasons.most_common())
    return (f"[SFT validation] Skipping validation circuit {circuit[:12]} "
            f"({info.get('source_module_name', '?')}, netlist {str(info.get('netlist_id', '?'))[:12]}): "
            f"0 of {examined} rows usable after filtering. Causes: {causes or 'none recorded'}. "
            f"Shortest prompt {shortest_prompt} tokens vs max_prompt_length={max_prompt_length}; "
            f"max_model_len={max_model_len}. The circuit is excluded from validation only; "
            f"training data is unaffected.")


def _shortest_prompt(rows, tokenizer, use_tools, sample=32):
    from .conversation import ConversationExample
    from .tools import TOOLS
    shortest = None
    for row in rows[:sample]:
        try:
            messages = ConversationExample.prompt_messages_from_record(copy.deepcopy(row), use_tools=use_tools)
            text = tokenizer.apply_chat_template(messages, tools=TOOLS if use_tools else None,
                                                 add_generation_prompt=True, tokenize=False)
        except Exception:
            continue
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        shortest = len(ids) if shortest is None else min(shortest, len(ids))
    return shortest


def prepare_circuit_validation(directory, manifest, tokenizer, *, sft_format,
                               max_prompt_length, max_model_len, per_circuit=8,
                               batch_size=16):
    """A fixed, circuit-balanced validation sample: up to ``per_circuit``
    distinct faults per held-out circuit, in a seed-defined order.

    Rows are read and tokenized lazily, a few at a time, until a circuit has
    its examples, instead of tokenizing the whole split. A circuit with no row
    that passes the training filters is skipped with a logged reason and
    recorded under ``skipped_circuits``; only a sample with no circuit left
    is an error.
    """
    from collections import Counter
    import pyarrow.parquet as pq
    from datasets import Dataset
    from .dataset_utils import _classify_sft_messages_batch, _format_sft_messages_record, _gate_filter_passes
    from .tools import TOOLS

    if per_circuit < 1:
        raise ValueError('SFT validation examples per circuit must be positive')
    use_tools = bool(getattr(tokenizer, 'chat_template', None) and 'tool' in tokenizer.chat_template)
    info = {r['circuit_id']: r for r in manifest['circuits'] if r.get('split') == 'validation'}
    index = _ordering_index(directory, info, manifest['seed'])
    tables = {}

    def load(entries):
        rows = []
        for _, path, position in entries:
            if path not in tables:
                tables[path] = pq.read_table(path)
            rows.append(tables[path].slice(position, 1).to_pylist()[0])
        return rows

    selected, identities, counts, skipped = [], [], {}, {}
    for circuit, entries in sorted(index.items()):
        tables.clear()  # A circuit's rows sit in one or a few shards.
        reasons, kept_faults, examined = Counter(), set(), 0
        for start in range(0, len(entries), batch_size):
            if len(kept_faults) >= per_circuit:
                break
            rows = load(entries[start:start + batch_size])
            examined += len(rows)
            statuses = (_classify_sft_messages_batch(rows, tokenizer, use_tools, max_prompt_length, max_model_len)
                        if sft_format == 'messages' else ['ok' if _gate_filter_passes(r) else 'gate' for r in rows])
            for row, status in zip(rows, statuses, strict=True):
                reasons[status] += 1
                if status != 'ok' or row['fault'] in kept_faults or len(kept_faults) >= per_circuit:
                    continue
                formatted = _format_sft_messages_record(copy.deepcopy(row), use_tools)
                rendered = tokenizer.apply_chat_template(formatted['messages'], tokenize=False,
                                                          tools=TOOLS if use_tools else None)
                length = len(tokenizer(rendered, add_special_tokens=False)['input_ids'])
                if length > max_model_len:
                    reasons['ok'] -= 1
                    reasons['full_sequence_too_long'] += 1
                    continue
                selected.append(formatted if sft_format == 'messages' else {'text': rendered})
                kept_faults.add(row['fault'])
                identities.append({k: row[k] for k in ('circuit_id', 'netlist_id', 'source_module_name',
                                                        'module_name', 'fault', 'pattern_index')})
        reasons = Counter({k: v for k, v in reasons.items() if v > 0})
        if not kept_faults:
            shortest = _shortest_prompt(load(entries), tokenizer, use_tools) if entries else None
            print(_skip_message(circuit, info[circuit], examined, reasons, shortest,
                                max_prompt_length, max_model_len), flush=True)
            skipped[circuit] = dict(rows=len(entries), filtering=dict(reasons), shortest_prompt=shortest)
            continue
        counts[circuit] = dict(selected=len(kept_faults), filtering=dict(reasons))
    if not selected:
        raise ValueError(f'No circuit validation examples: all {len(index)} validation circuits were '
                         f'skipped (max_prompt_length={max_prompt_length}, max_model_len={max_model_len})')
    print(f"[SFT validation] {len(selected)} examples on {len(counts)} of {len(index)} validation "
          f"circuits; {len(skipped)} skipped.", flush=True)
    return Dataset.from_list(selected), {'examples': identities, 'circuits': counts,
                                         'skipped_circuits': skipped,
                                         'max_prompt_length': max_prompt_length,
                                         'max_model_len': max_model_len, 'per_circuit': per_circuit}


def cached_circuit_validation(cache_path, key, build):
    """Build the validation sample on the main process once; other ranks load it.

    ``key`` describes everything the sample depends on; a cache made under a
    different key is rebuilt rather than reused.
    """
    from accelerate import PartialState
    from datasets import Dataset
    cache_path = Path(cache_path)
    state = PartialState()
    with state.main_process_first():
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text())
            if cached.get('key') == key:
                print(f"[SFT validation] Loaded {len(cached['rows'])} cached examples from {cache_path}")
                return Dataset.from_list(cached['rows']), cached['report']
        if not state.is_main_process:
            raise RuntimeError(f'Main process did not write a matching validation cache at {cache_path}')
        dataset, report = build()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'key': key, 'rows': dataset.to_list(), 'report': report}))
        temporary.replace(cache_path)
        return dataset, report


def sft_evaluation_config(steps):
    if steps < 1:
        raise ValueError('SFT evaluation interval must be positive')
    return dict(eval_strategy='steps', eval_steps=steps, save_steps=steps,
                eval_on_start=True, per_device_eval_batch_size=1,
                prediction_loss_only=True, load_best_model_at_end=True,
                metric_for_best_model='eval_loss', greater_is_better=False)
