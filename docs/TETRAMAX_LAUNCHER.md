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
