import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atpgllm.training.sft_validation import (
    check_sft_dataset, prepare_circuit_validation, sft_evaluation_config,
)


class FakeTokenizer:
    chat_template = None

    def apply_chat_template(self, messages, **kwargs):
        return ' '.join(m['content'] for m in messages)

    def __call__(self, text, **kwargs):
        return {'input_ids': [t.split() for t in text]} if isinstance(text, list) else {'input_ids': text.split()}


def write_validation(tmp_path):
    folder = tmp_path / 'validation'
    folder.mkdir()
    rows = []
    for circuit in ('one', 'two'):
        for i in range(3):
            rows.append(dict(circuit_id=circuit, netlist_id=circuit + '_netlist',
                             source_module_name=circuit, module_name=circuit,
                             netlist='BUF_X u0 (.A(a));\nBUF_X u1 (.A(a));\n',
                             system_content='system', user_content='user', reasoning_content='',
                             answer_content='answer', fault=f'sa0 n{i}', pattern_index=i,
                             input_vector='{}', expected_output='{}'))
    pq.write_table(pa.Table.from_pylist(rows), folder / 'data.parquet')
    return dict(seed=42, circuits=[dict(circuit_id=cid, split='validation') for cid in ('one', 'two')])


def test_validation_balances_circuits_and_is_deterministic(tmp_path):
    manifest = write_validation(tmp_path)
    kwargs = dict(sft_format='messages', max_prompt_length=100, max_model_len=100, per_circuit=2)
    a, report_a = prepare_circuit_validation(tmp_path, manifest, FakeTokenizer(), **kwargs)
    b, report_b = prepare_circuit_validation(tmp_path, manifest, FakeTokenizer(), **kwargs)
    assert a.to_list() == b.to_list()
    assert report_a == report_b
    assert len(a) == 4
    assert all(info['selected'] == 2 for info in report_a['circuits'].values())
    assert a.column_names == ['messages']


@pytest.mark.parametrize('max_prompt,max_length', [(1, 100), (100, 1)])
def test_validation_cannot_silently_lose_a_circuit(tmp_path, max_prompt, max_length):
    manifest = write_validation(tmp_path)
    with pytest.raises(ValueError, match='no examples after SFT filtering'):
        prepare_circuit_validation(tmp_path, manifest, FakeTokenizer(), sft_format='messages',
                                   max_prompt_length=max_prompt, max_model_len=max_length)


def test_resume_requires_matching_repaired_dataset_provenance(tmp_path, monkeypatch):
    import atpgllm.training.sft_validation as module
    monkeypatch.setattr(module, 'audit_dataset', lambda _: {'verified': True})
    (tmp_path / 'split_manifest.json').write_text('{}')
    checkpoint = tmp_path / 'old_run' / 'checkpoint-10'
    checkpoint.mkdir(parents=True)
    with pytest.raises(ValueError, match='old SFT weights'):
        check_sft_dataset(tmp_path, checkpoint)
    _, fingerprint = check_sft_dataset(tmp_path)
    sidecar = checkpoint.parent / 'sft_data_manifest.json'
    sidecar.write_text(json.dumps({'dataset_sha256': 'different'}))
    with pytest.raises(ValueError, match='provenance'):
        check_sft_dataset(tmp_path, checkpoint)
    sidecar.write_text(json.dumps({'dataset_sha256': fingerprint}))
    assert check_sft_dataset(tmp_path, checkpoint)[1] == fingerprint


def test_sft_evaluation_selects_lowest_loss_at_matching_save_steps():
    config = sft_evaluation_config(10)
    assert config['eval_on_start']
    assert config['eval_steps'] == config['save_steps'] == 10
    assert config['load_best_model_at_end'] and not config['greater_is_better']
    with pytest.raises(ValueError):
        sft_evaluation_config(0)
