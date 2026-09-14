#!/usr/bin/env python3
"""Run the repaired dataset/tokenizer preflight without loading model weights."""

import argparse
from datetime import datetime, timezone
import faulthandler
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=repo.parent / 'data/freeset/'
                        'dataset.freeset.asap7sc7p5t_28.rvt.tt.stil_repaired_v1')
    parser.add_argument('--tokenizer', type=Path, default=repo / 'runs/sft_granite_4.2_8b')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--max-prompt-length', type=int, default=2048)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--per-circuit', type=int, default=8)
    args = parser.parse_args()
    report_path = args.report or args.dataset / 'build_validation_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'dataset': str(args.dataset.resolve()),
              'tokenizer': str(args.tokenizer.resolve()),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'started_at': datetime.now(timezone.utc).isoformat()}

    def save():
        with tempfile.NamedTemporaryFile(mode='w', dir=report_path.parent,
                                         prefix=report_path.name + '.', delete=False) as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
            temporary = Path(stream.name)
        temporary.replace(report_path)

    def stage(name):
        report['stage'] = name
        report['updated_at'] = datetime.now(timezone.utc).isoformat()
        save()
        print(f"[{report['updated_at']}] {name}", flush=True)

    # Keep normal backend detection: importing atpgllm also imports PyTorch
    # model classes through __init__ -> utils. Disabling USE_TORCH hides
    # GenerationMixin and breaks that import, even without loading weights.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    sys.path.insert(0, str(repo))
    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    try:
        stage('Import dataset audit libraries')
        from atpgllm.training.sft_validation import (
            check_sft_dataset, load_circuit_train, prepare_circuit_validation,
        )

        stage('Audit every shard checksum, row identity, count, and split membership')
        manifest, fingerprint = check_sft_dataset(args.dataset)
        report.update(dataset_sha256=fingerprint, row_counts=manifest['row_counts'],
                      train_validation_overlap={'circuit_id': 0, 'netlist_id': 0, 'module_name': 0},
                      rejected_circuits=manifest['rejected_circuits'],
                      mapping_version=manifest['mapping_version'],
                      fault_claims_version=manifest['fault_claims_version'])

        stage('Import tokenizer and SFT formatting libraries')
        from transformers import AutoTokenizer
        from atpgllm.training.dataset_utils import (
            TrainingMode, count_prompt_tokens, format_dataset_for_training,
        )
        from atpgllm.training.tools import TOOLS

        stage('Load the cached tokenizer locally')
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
        template = tokenizer.get_chat_template()
        if 'generation' not in template:
            raise ValueError('Cached tokenizer template lacks assistant generation markers')
        tool_schema = TOOLS if 'tool' in template else None

        stage('Prepare the fixed circuit-balanced SFT validation sample')
        validation, sample = prepare_circuit_validation(
            args.dataset, manifest, tokenizer, sft_format='messages',
            max_prompt_length=args.max_prompt_length, max_model_len=args.max_model_len,
            per_circuit=args.per_circuit,
        )
        expected_groups = {r['circuit_id'] for r in manifest['circuits']
                           if r.get('split') == 'validation'}
        if set(sample['circuits']) != expected_groups:
            raise ValueError('Validation sample lost a held-out circuit group')

        stage('Verify validation token lengths and assistant loss masks')
        lengths, prompts, assistant_tokens = [], [], []
        for row in validation:
            encoded = tokenizer.apply_chat_template(
                row['messages'], tools=tool_schema, tokenize=True,
                return_dict=True, return_assistant_tokens_mask=True,
            )
            mask = encoded['assistant_masks']
            length = len(encoded['input_ids'])
            prompt = count_prompt_tokens(tokenizer, row['messages'], tools=tool_schema)
            if len(mask) != length or sum(mask) <= 0:
                raise ValueError('Validation example has an empty or misaligned assistant mask')
            if length > args.max_model_len or prompt >= args.max_prompt_length:
                raise ValueError('Validation example exceeds configured token limits')
            lengths.append(length)
            prompts.append(prompt)
            assistant_tokens.append(sum(mask))

        stage('Check the first retained example from the actual training stream')
        formatted = format_dataset_for_training(
            load_circuit_train(args.dataset), tokenizer, TrainingMode.SFT,
            sft_format='messages', max_prompt_length=args.max_prompt_length,
            skip_batch_size=32, skip_num_workers=1,
        )
        first = next(iter(formatted))
        if not first.get('messages'):
            raise ValueError('Training stream did not produce formatted messages')
        report.update(
            status='passed', stage='Complete',
            validation_circuit_groups=len(expected_groups),
            validation_source_variants=sum(r.get('split') == 'validation' for r in manifest['circuits']),
            fixed_sft_validation_examples=len(validation),
            validation_prompt_token_range=[min(prompts), max(prompts)],
            validation_full_sequence_token_range=[min(lengths), max(lengths)],
            validation_assistant_token_range=[min(assistant_tokens), max(assistant_tokens)],
            training_stream_formats_successfully=True, validation_sample=sample,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        save()
        print(f'PASS: {report_path.resolve()}', flush=True)
        return 0
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                      finished_at=datetime.now(timezone.utc).isoformat())
        save()
        traceback.print_exc()
        print(f'FAIL: {report_path.resolve()}', flush=True)
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    sys.exit(main())
