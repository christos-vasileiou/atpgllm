#!/usr/bin/env python
"""Run one command per GPU and wait for all of them.

    python scripts/rft/fanout.py --gpus 0,1,2,3 --log_dir <dir> -- python x.py --shard {shard} --num_shards {num_shards}

``{shard}`` and ``{num_shards}`` are substituted in every argument; shard i
runs with ``CUDA_VISIBLE_DEVICES`` set to the i-th GPU and logs to
``<log_dir>/shard<i>.log``. If any shard fails, or this process receives
SIGTERM or SIGINT, the other shards are stopped and the exit status is nonzero.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", required=True, help="Comma list of GPU indices, one shard each")
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given")
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    children = []
    for shard, gpu in enumerate(gpus):
        argv = [part.format(shard=shard, num_shards=len(gpus)) for part in command]
        log = open(log_dir / f"shard{shard}.log", "a")
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} GPU {gpu}: {' '.join(argv)}\n")
        log.flush()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        children.append(subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True))

    def stop(*_):
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        deadline = time.time() + 60
        for child in children:
            try:
                child.wait(timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)

    def on_signal(signum, _):
        stop()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    while True:
        codes = [child.poll() for child in children]
        failed = [i for i, code in enumerate(codes) if code not in (None, 0)]
        if failed:
            print(f"[fanout] shard {failed[0]} exited with {codes[failed[0]]}; stopping the others "
                  f"(see {log_dir}/shard{failed[0]}.log)", flush=True)
            stop()
            sys.exit(1)
        if all(code == 0 for code in codes):
            print(f"[fanout] all {len(children)} shards finished", flush=True)
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
