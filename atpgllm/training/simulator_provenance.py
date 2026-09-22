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


class SimulatorProvenanceCallback(TrainerCallback):
    def __init__(self, resume_checkpoint=None):
        self.provenance = runtime_provenance()
        if resume_checkpoint:
            path = Path(resume_checkpoint)/'simulator_provenance.json'
            previous = json.loads(path.read_text()) if path.exists() else {'backend':'fast'}
            if previous != self.provenance and ('tetramax' in (previous['backend'], self.provenance['backend'])):
                raise ValueError('Simulator/reward identity differs from the checkpoint. Start a new run with a fresh optimizer instead of resuming training state.')

    def _write(self, path, state):
        if state.is_world_process_zero:
            path.mkdir(parents=True, exist_ok=True)
            (path/'simulator_provenance.json').write_text(json.dumps(self.provenance, indent=2)+'\n')

    def on_train_begin(self, args, state, control, **kwargs):
        self._write(Path(args.output_dir), state)

    def on_save(self, args, state, control, **kwargs):
        self._write(Path(args.output_dir)/f'checkpoint-{state.global_step}', state)
