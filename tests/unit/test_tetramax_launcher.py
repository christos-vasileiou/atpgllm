"""Exercise launcher lifecycle with the real coordinator and a license-free backend."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launcher(tmp_path):
    repo = tmp_path / 'libatpgllm'
    scripts = repo / 'scripts/train'
    scripts.mkdir(parents=True)
    for name in ('run_training_code.sh', '_tetramax_lifecycle.sh'):
        shutil.copy(ROOT / 'scripts/train' / name, scripts / name)
    backend = tmp_path / 'data_preprocessing'
    backend.mkdir()
    for name in ('tetramax_service.py', 'tetramax_seats.py'):
        shutil.copy(ROOT.parent / 'data_preprocessing' / name, backend / name)
    (backend / 'tetramax_backend.py').write_text('''
VERSION = 'test'
def configuration(): return 'test-binary', []
def _identity(*args): return 'test-fingerprint', 'test-version'
def simulate(*args, **kwargs): return {}
''')
    # The training stand-in verifies that credentials are exported and that
    # authenticated service health works before any training work begins.
    (scripts / 'training_code.py').write_text('''
import json, os, time
from pathlib import Path
from urllib import request
credentials_path = Path(os.environ['TMAX_SERVER_FILE'])
assert credentials_path.is_absolute()
credentials = json.loads(credentials_path.read_text())
req = request.Request(credentials['url'] + '/health',
    headers={'Authorization': 'Bearer ' + credentials['token']})
with request.build_opener(request.ProxyHandler({})).open(req) as response:
    assert json.load(response)['status'] == 'ok'
Path('training-started').write_text(str(os.getpid()))
time.sleep(float(os.environ.get('TEST_TRAIN_SLEEP', '0')))
raise SystemExit(int(os.environ.get('TEST_TRAIN_RC', '0')))
''')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name, body in {
        'python': f'exec "{sys.executable}" "$@"',
        'nvidia-smi': 'echo "GPU 0: test"',
        'accelerate': 'while [[ "$1" != *.py ]]; do shift; done\nexec python "$@"',
        'scontrol': 'printf "test-head\\ntest-worker\\n"',
        'srun': 'while [[ "$1" == --* ]]; do shift; done\nexec "$@"',
        'sbatch': 'printf "%s\\n" "$@" > "$TEST_SBATCH_ARGS"',
    }.items():
        path = bin_dir / name
        path.write_text('#!/bin/bash\n' + body + '\n')
        path.chmod(0o755)
    (scripts / '_mn_launch.sh').write_text(
        '#!/bin/bash\nexec python "$(dirname "$0")/training_code.py"\n')
    (scripts / '_mn_launch.sh').chmod(0o755)
    config = repo / 'test.conf'
    config.write_text(f'''
PATH="{bin_dir}:$PATH"
METHOD=sft
MODEL=test
TRAIN_DATASET=test
OUTPUT_DIR=runs/test
USE_VLLM=False
USE_DDP=False
FAULT_SIM_BACKEND=tetramax
TMAX_MANAGE_SERVICE=True
TMAX_MAX_CONCURRENT=1
TMAX_SERVICE_STARTUP_TIMEOUT_S=4
''')
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('TMAX_', 'SLURM_', 'SIM_PYTHON'))}
    env['CUDA_VISIBLE_DEVICES'] = '0'

    class Launch:
        command = ['bash', str(scripts / 'run_training_code.sh'), str(config)]
        credentials = tmp_path / '.runtime/tetramax/server.json'

        def append(self, text):
            with config.open('a') as f:
                f.write(text + '\n')

        def run(self):
            return subprocess.run(self.command, cwd=repo, env=env,
                                  capture_output=True, text=True, timeout=20)

    launch = Launch()
    launch.repo, launch.backend, launch.env = repo, backend, env
    launch.scripts, launch.config = scripts, config
    return launch


@pytest.mark.parametrize('rc', [0, 7])
@pytest.mark.parametrize('ddp', ['True', 'False'])
def test_training_success_and_failure_stop_service(launcher, rc, ddp):
    launcher.append('USE_DDP=' + ddp)
    launcher.env['TEST_TRAIN_RC'] = str(rc)
    result = launcher.run()
    assert result.returncode == rc, result.stdout + result.stderr
    assert (launcher.repo / 'training-started').exists()
    assert 'TetraMAX service is ready' in result.stdout
    assert 'Stopping TetraMAX service' in result.stdout
    assert not launcher.credentials.exists()


def test_service_startup_failure_prevents_training(launcher):
    (launcher.backend / 'tetramax_backend.py').write_text(
        'raise RuntimeError("test simulator unavailable")\n')
    result = launcher.run()
    assert result.returncode == 1
    assert 'test simulator unavailable' in result.stdout
    assert not (launcher.repo / 'training-started').exists()
    assert not launcher.credentials.exists()


def test_service_readiness_timeout_stops_child(launcher):
    (launcher.backend / 'tetramax_service.py').write_text('import time; time.sleep(60)\n')
    launcher.append('TMAX_SERVICE_STARTUP_TIMEOUT_S=1')
    result = launcher.run()
    assert result.returncode == 1
    assert 'did not become ready within 1s' in result.stdout
    assert 'Stopping TetraMAX service' in result.stdout
    assert not (launcher.repo / 'training-started').exists()


def test_health_requires_valid_authentication(launcher):
    service = launcher.backend / 'tetramax_service.py'
    service.write_text(service.read_text().replace("'token':token,'workers'", "'token':token+'invalid','workers'"))
    launcher.append('TMAX_SERVICE_STARTUP_TIMEOUT_S=2')
    result = launcher.run()
    assert result.returncode == 1
    assert 'did not become ready within 2s' in result.stdout
    assert not (launcher.repo / 'training-started').exists()
    assert not launcher.credentials.exists()


def test_dry_run_does_not_start_service(launcher):
    launcher.append('DRY_RUN=True\nTMAX_SERVICE_MODULE=nonexistent-module')
    result = launcher.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'DRY RUN: would wait' in result.stdout
    assert not launcher.credentials.parent.exists()
    assert not (launcher.repo / 'training-started').exists()


@pytest.mark.parametrize('sig,rc', [(signal.SIGTERM, 143), (signal.SIGINT, 130)])
def test_signal_during_training_stops_service(launcher, sig, rc):
    launcher.env['TEST_TRAIN_SLEEP'] = '60'
    proc = subprocess.Popen(launcher.command, cwd=launcher.repo, env=launcher.env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 10
        marker = launcher.repo / 'training-started'
        while not marker.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists()
        proc.send_signal(sig)
        output, _ = proc.communicate(timeout=10)
        assert proc.returncode == rc, output
        assert 'Stopping training' in output
        assert 'Stopping TetraMAX service' in output
        assert not launcher.credentials.exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)


def test_multinode_uses_one_service_and_exports_credentials(launcher):
    launcher.append('NUM_NODES=2\nGPUS_PER_NODE=1\nTMAX_SERVICE_ADVERTISE_HOST=127.0.0.1')
    launcher.env.update(SLURM_JOB_ID='test', SLURM_JOB_NODELIST='test-head,test-worker')
    result = launcher.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert '--host 0.0.0.0' in result.stdout
    assert result.stdout.count('Starting TetraMAX service;') == 1
    assert (launcher.repo / 'training-started').exists()
    assert not launcher.credentials.exists()


def test_submit_only_queues_config_with_managed_setting(launcher):
    shutil.copy(ROOT / 'scripts/train/submit_training_code.sh', launcher.scripts)
    capture = launcher.repo / 'sbatch-args'
    launcher.env['TEST_SBATCH_ARGS'] = str(capture)
    result = subprocess.run(['bash', str(launcher.scripts / 'submit_training_code.sh'),
                             str(launcher.config)], cwd=launcher.repo, env=launcher.env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    args = capture.read_text().splitlines()
    assert args[-2] == str(launcher.scripts / 'run_training_code.sh')
    assert 'TMAX_MANAGE_SERVICE=True' in Path(args[-1]).read_text()
    assert not launcher.credentials.parent.exists()


@pytest.mark.parametrize('managed', ['True', 'False'])
def test_existing_service_is_never_stopped(launcher, managed):
    env = dict(launcher.env, TMAX_LOCK_DIR=str(launcher.credentials.parent), TMAX_MAX_CONCURRENT='1')
    external = subprocess.Popen(
        [sys.executable, str(launcher.backend / 'tetramax_service.py'), '--workers', '1', '--port', '0'],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 10
        while not launcher.credentials.exists() and external.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert launcher.credentials.exists()
        original = launcher.credentials.read_bytes()
        launcher.append(f'TMAX_MANAGE_SERVICE={managed}\nTMAX_SERVER_FILE={launcher.credentials}')
        result = launcher.run()
        assert external.poll() is None
        assert launcher.credentials.read_bytes() == original
        if managed == 'True':
            assert result.returncode == 1, result.stdout + result.stderr
            assert 'already owns this project pool' in result.stdout
            assert not (launcher.repo / 'training-started').exists()
        else:
            assert result.returncode == 0, result.stdout + result.stderr
            assert (launcher.repo / 'training-started').exists()
            assert 'Stopping TetraMAX service' not in result.stdout
    finally:
        external.terminate()
        external.communicate(timeout=10)
