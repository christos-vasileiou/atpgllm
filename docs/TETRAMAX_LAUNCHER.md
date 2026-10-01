# TetraMAX service lifecycle during training

Set these in a training config to let `run_training_code.sh` start the service,
wait for authenticated `/health`, run training, and stop its service on exit:

```bash
FAULT_SIM_BACKEND=tetramax
TMAX_MANAGE_SERVICE=True
```

This works for direct execution, direct `sbatch`, and
`submit_training_code.sh`. Submission only queues the job; the service starts
on the batch host when the allocation begins, before vLLM or model loading.
Normal completion, training failure, SIGINT, and SIGTERM trigger cleanup.
SIGKILL cannot run shell cleanup.

The batch host must have a working TetraMAX executable and cell libraries.
Use the loaded Synopsys environment / `TMAX_BIN`, or configure a module:

```bash
TMAX_SERVICE_MODULE=tetramax/vO-2018.06-SP1
```

The service uses the active Python environment (`SIM_PYTHON` overrides it).
Logs are retained in `jobs/tetramax_<job-or-local>_<unique-id>.log`. Startup
failure aborts the launch and prints the end of the service log.

Optional settings:

| Setting | Default | Purpose |
| --- | --- | --- |
| `TMAX_SERVICE_STARTUP_TIMEOUT_S` | `60` | Maximum readiness wait in seconds |
| `TMAX_SERVICE_WORKERS` | `TMAX_MAX_CONCURRENT`, or `16` | Service workers |
| `TMAX_QUEUE_SIZE` | `128` | Maximum queued requests |
| `TMAX_SERVICE_PORT` | `0` | OS-selected free port; set a fixed port if needed |
| `TMAX_LOCK_DIR` | Project `.runtime/tetramax` | Existing shared license pool |
| `TMAX_SERVICE_ADVERTISE_HOST` | Batch host's fully qualified hostname | Reachable host for multi-node clients |

Managed mode sets `TMAX_SERVER_FILE` to the absolute
`TMAX_LOCK_DIR/server.json` path, replacing any configured client endpoint.
The service creates the file and removes it during orderly shutdown. Single-node
runs bind to loopback. Multi-node runs bind to all interfaces and export the
shared credentials path to training ranks; the project filesystem and service
port must be reachable from those nodes.

Only one coordinator can own the project pool. Managed startup fails if another
coordinator owns it, and does not stop that coordinator. The pool also remembers
its host and seat allocation; launching on a different host requires the existing
drain/reconfiguration procedure. The launcher does not reset the pool.

To use a separately managed service, set `TMAX_MANAGE_SERVICE=False` and supply
its `TMAX_SERVER_FILE` (or URL/token). This is also the default for configs that
omit the setting. `DRY_RUN=True` prints the planned managed startup and does not
start the service, load its module, or launch training.

For GRPO, `DDP_TIMEOUT=3600` sets the distributed collective timeout in seconds
(also the default; direct Python launches accept `--ddp_timeout`). Non-main
training ranks wait in a broadcast while rank zero requests vLLM generation, so
this limit must exceed the longest generation request. The GRPO path initializes
the group with this timeout before model loading and fixed-evaluation setup,
then passes the same value to `GRPOConfig`. Setting it only on `GRPOConfig` is
too late if fixed evaluation already created the group with NCCL's ten-minute
default. This is separate from the TetraMAX simulation timeout and vLLM server
startup timeout. It does not change sampling, rewards, or checkpoint state.

Full training-state recovery checks the checkpoint's simulator/reward identity
against the current service. The fingerprint includes the Python adapter, Tcl
script, STIL writer, executable path/version and cell libraries, so an adapter
fix changes the fingerprint even with the same TetraMAX version.
The launcher checks this after service readiness and before vLLM starts, and
reports the changed fields. Direct Python training checks before model loading.

For a reviewed adapter repair, full-state continuation is supported with
`TMAX_RESUME_FINGERPRINT_TRANSITION=<old fingerprint>:<new fingerprint>`.
Both hashes must match exactly and every other identity field must be unchanged.
This does not allow backend, objective, weight, profile, schema or tool-version
changes. The new hash also pins the reviewed executable path and cell libraries.
The launcher exports this setting to training ranks and records it in its launch
snapshot. The trainer writes the accepted transition to
`simulator_resume_history.json` alongside the current provenance in the run and
subsequent checkpoints, preserving the original checkpoint unchanged. Later
resumes with an already matching fingerprint need no exception.

Use the original `OUTPUT_DIR` and `FIXED_EVAL_MANIFEST`, set `RESUME_FROM` to
the saved checkpoint, and enable `RESUME_TRAINING_STATE=True` and
`AUTO_SKIP_FROM_RESUME=True`. Trainer recovery restores weights, optimizer,
scheduler, saved RNG and training progress. Work after the last saved checkpoint
must be repeated; changing the simulator does not promise bit-identical future
samples or results.

If a separate fresh-optimizer run is desired instead, keep learned policy/reference weights by setting
`RESUME_FROM` to the old checkpoint, `RESUME_TRAINING_STATE=False`,
`AUTO_SKIP_FROM_RESUME=False` and `SKIP_BUFFER_SIZE=0`. Use a new `OUTPUT_DIR`
and a `FIXED_EVAL_MANIFEST` inside that directory. This starts a separate run
with a fresh optimizer, scheduler, step count and RNG state, replaying the
training buffer. It preserves the old run and does not rewrite its provenance.
