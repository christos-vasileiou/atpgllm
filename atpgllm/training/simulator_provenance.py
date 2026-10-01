"""Record simulator/reward identity and prevent incompatible full-state resumes."""
import json
import os
from pathlib import Path
from urllib import request as http

from transformers import TrainerCallback
from ._paths import ensure_data_preprocessing_on_path


def runtime_provenance():
    ensure_data_preprocessing_on_path()
    from fault_sim import resolve_fault_sim_runner
    from atpgllm.llm.reward_funcs import active_reward_objectives
    native = resolve_fault_sim_runner().__name__ == 'tetramax_fault_sim'
    keys, weights = active_reward_objectives()
    result = {'backend': 'tetramax' if native else 'fast', 'reward_contract': 'atpg-rewards-v2',
              'objectives': list(keys), 'weights': list(weights),
              'profile': os.environ.get('TMAX_REWARD_PROFILE', 'po') if native else 'full'}
    if native:
        from tetramax_backend import configuration, _identity, VERSION
        if os.environ.get('TMAX_SERVER_FILE') or os.environ.get('TMAX_SERVER_URL'):
            from tetramax_service import credentials
            from tetramax_seats import SimulationError
            url, token = credentials()
            req = http.Request(url+'/health', headers={'Authorization':'Bearer '+token})
            try:
                with http.build_opener(http.ProxyHandler({})).open(req, timeout=5) as response:
                    result['simulator'] = json.load(response)['simulator']
            except (OSError, ValueError, KeyError) as exc:
                raise SimulationError(f'TetraMAX service preflight failed: {exc}') from exc
        else:
            binary, libs = configuration()
            fingerprint, version = _identity({}, binary, libs)
            result['simulator'] = {'schema': VERSION, 'fingerprint': fingerprint, 'tool_version': version}
    return result


def _differences(previous, current, prefix=''):
    for key in sorted(previous.keys() | current.keys()):
        name = f'{prefix}{key}'
        old, new = previous.get(key), current.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            yield from _differences(old, new, name + '.')
        elif old != new:
            yield f'{name}: {old!r} -> {new!r}'


def validate_simulator_resume(resume_checkpoint, provenance):
    """Return an audited fingerprint transition, or reject incompatible state."""
    path = Path(resume_checkpoint) / 'simulator_provenance.json'
    previous = json.loads(path.read_text()) if path.exists() else {'backend': 'fast'}
    if previous != provenance and 'tetramax' in (previous.get('backend'), provenance.get('backend')):
        # A byte fingerprint is stricter than training-state compatibility: a
        # reviewed adapter repair need not invalidate Adam/scheduler/RNG state.
        # Scope the exception to BOTH exact fingerprints, and require every
        # other identity field (including reward semantics) to remain equal.
        old_simulator, new_simulator = previous.get('simulator', {}), provenance.get('simulator', {})
        old, new = old_simulator.get('fingerprint'), new_simulator.get('fingerprint')
        transition = os.environ.get('TMAX_RESUME_FINGERPRINT_TRANSITION', '')
        expected = dict(previous, simulator=dict(old_simulator, fingerprint=new))
        if (previous.get('backend') == provenance.get('backend') == 'tetramax'
                and old and new and old != new and expected == provenance
                and transition == f'{old}:{new}'):
            return {'checkpoint': str(Path(resume_checkpoint).resolve()),
                    'previous': previous, 'current': provenance,
                    'authorization': 'TMAX_RESUME_FINGERPRINT_TRANSITION'}
        changes = '\n  '.join(_differences(previous, provenance))
        raise ValueError(
            f'Simulator/reward identity differs from checkpoint {resume_checkpoint}:\n  {changes}\n'
            'The simulator fingerprint includes the Python adapter, Tcl script, STIL writer, '
            'tool and cell libraries; adapter fixes also change it. '
            'For a reviewed adapter-only repair with all other identity fields unchanged, '
            'set TMAX_RESUME_FINGERPRINT_TRANSITION=<checkpoint fingerprint>:<current fingerprint> '
            'to continue with the saved optimizer/scheduler/RNG state and record the transition. '
            'Backend, reward, profile, schema and tool-version changes remain incompatible. '
            'Alternatively, a separate run with RESUME_TRAINING_STATE=False, a new OUTPUT_DIR '
            'and FIXED_EVAL_MANIFEST can load weights with a fresh optimizer.'
        )


class SimulatorProvenanceCallback(TrainerCallback):
    def __init__(self, resume_checkpoint=None):
        self.provenance = runtime_provenance()
        self.resume_history = []
        if resume_checkpoint:
            transition = validate_simulator_resume(resume_checkpoint, self.provenance)
            history = Path(resume_checkpoint) / 'simulator_resume_history.json'
            if history.exists():
                self.resume_history = json.loads(history.read_text())
                if not isinstance(self.resume_history, list):
                    raise ValueError(f'Invalid simulator resume history: {history}')
            if transition:
                self.resume_history.append(transition)
                print('[Simulator resume] Accepted the configured adapter fingerprint transition; '
                      'restoring full training state. The old checkpoint provenance is preserved.')

    def _write(self, path, state):
        if state.is_world_process_zero:
            path.mkdir(parents=True, exist_ok=True)
            (path/'simulator_provenance.json').write_text(json.dumps(self.provenance, indent=2)+'\n')
            if self.resume_history:
                (path/'simulator_resume_history.json').write_text(json.dumps(self.resume_history, indent=2)+'\n')

    def on_train_begin(self, args, state, control, **kwargs):
        self._write(Path(args.output_dir), state)

    def on_save(self, args, state, control, **kwargs):
        self._write(Path(args.output_dir)/f'checkpoint-{state.global_step}', state)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check simulator compatibility before GPU startup.')
    parser.add_argument('checkpoint')
    args = parser.parse_args()
    try:
        SimulatorProvenanceCallback(args.checkpoint)
    except ValueError as exc:
        parser.exit(1, f'ERROR: {exc}\n')
    print('Simulator/reward checkpoint compatibility verified.')
