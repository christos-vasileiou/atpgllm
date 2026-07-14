"""
Parallel hyperparameter search for Stage 1: loss weights (w_gtc, w_gtm, w_gtg)
and architecture/optimizer hparams (lr, Q-Former / GIN layers and widths, proj-dim).

Runs multiple ``train_stage1`` jobs as subprocesses, pinning one GPU per worker
(default 4 workers on GPUs 0–3). Each worker reloads the streaming dataset
(HuggingFace); use the same ``--seed`` and ``--shuffle-buffer`` in the passthrough
arguments for comparable trials.

Samplers (``--sampler``):

- ``sobol`` / ``random``: built-in (no extra packages); pre-generates all trials.
- ``optuna``: TPE or random sampler; parallel ``n_jobs`` = number of GPUs. Install:
  ``pip install 'atpgllm[graph-search]'`` or ``pip install optuna``.
- ``ax``: Service API with Sobol → GPEI; batched parallel trials via
  ``get_next_trials``. Install: ``pip install 'atpgllm[graph-search]'`` or
  ``pip install ax-platform``.

Usage::

    activate
    python -m atpgllm.graph.scripts.search_stage1_loss_weights \\
        --num-trials 16 \\
        --gpus 0,1,2,3 \\
        --sampler sobol \\
        -- \\
        --sim-config tests/sim_config.json \\
        --output-dir checkpoints/graph_stage1_search \\
        --per-device-train-batch-size 64 --max-steps 2000 --lr 1e-4 \\
        --shuffle-buffer 2097152 --log-every 50 \\
        --save-every 999999999 --no-save

Everything after ``--`` is forwarded to ``train_stage1`` except loss weights, learning
rate, Q-Former / GIN / ``--proj-dim`` values, and ``--output-dir`` / ``--metrics-json``
/ ``--device`` / W&B name flags, which this script sets per trial. After a run,
``hpo_trial_results.json`` and (if ``optuna`` is installed) ``hpo_plots/*.html`` are
written under the same ``--output-dir`` for analysis.

Weights & Biases: pass ``--wandb-project myproj`` (and optional ``--wandb-group``,
``--wandb-entity``, ``--wandb-tags``) so each trial logs a separate run in the same
group, then a short ``hpo_summary`` run logs an ``hpo/trials_table`` plus best metrics.
The same values can be supplied via ``WANDB_PROJECT``, ``WANDB_ENTITY``, ``WANDB_GROUP``,
and ``WANDB_TAGS`` when the matching CLI flag is omitted. The HPO **summary** run
(``job_type=hpo_summary``) is in the **same** ``--wandb-project`` and
``--wandb-group`` as the trial runs; it logs the same HPO figures as
``plotly.io.to_html(..., full_html=False, include_plotlyjs="cdn")`` in ``wandb.Html``
so interactive charts show under the run’s Media / panels, and it uploads a W&B
**Artifact** (``hpo_trial_results.json`` + ``hpo_plots/``) for download or Artifact
preview. Set ``WANDB_ENTITY`` to your user or team so the project appears under the
right organization in the UI.

Example with explicit flags (replace ``YOUR_ENTITY`` with your W&B username or team slug)::

    python -m atpgllm.graph.scripts.search_stage1_loss_weights \\
        --num-trials 16 --gpus 0,1,2,3 --sampler sobol \\
        --wandb-project atpgllm-graph-stage1 \\
        --wandb-entity YOUR_ENTITY \\
        --wandb-group stage1_loss_hpo_sobol \\
        --wandb-tags stage1,hpo,sobol \\
        -- \\
        --sim-config tests/sim_config.json \\
        --output-dir checkpoints/graph_stage1_search \\
        --per-device-train-batch-size 64 --max-steps 2000 --lr 1e-4 \\
        --shuffle-buffer 2097152 --log-every 50 \\
        --save-every 999999999 --no-save

Example using only environment (omit ``WANDB_GROUP`` here to keep the script's
auto-generated ``stage1_hpo_<sampler>_seed<seed>_<UTC>`` group names)::

    export WANDB_ENTITY=YOUR_ENTITY
    export WANDB_PROJECT=atpgllm-graph-stage1
    export WANDB_TAGS=stage1,hpo
    python -m atpgllm.graph.scripts.search_stage1_loss_weights \\
        --num-trials 16 --gpus 0,1,2,3 --sampler sobol \\
        -- \\
        --sim-config tests/sim_config.json \\
        --output-dir checkpoints/graph_stage1_search \\
        --per-device-train-batch-size 64 --max-steps 2000 --lr 1e-4 \\
        --shuffle-buffer 2097152 --log-every 50 \\
        --save-every 999999999 --no-save
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch


# ---------------------------------------------------------------------------
# Passthrough argv helpers
# ---------------------------------------------------------------------------

_FLAGS_WITH_VALUE = frozenset(
    {
        "--w-gtc",
        "--w-gtm",
        "--w-gtg",
        "--output-dir",
        "--metrics-json",
        "--device",
        "--wandb-project",
        "--wandb-entity",
        "--wandb-group",
        "--wandb-run-name",
        "--wandb-tags",
        "--wandb-mode",
        "--lr",
        "--qformer-layers",
        "--qformer-hidden",
        "--gin-layers",
        "--gin-hidden",
        "--proj-dim",
    }
)


def _strip_conflicting_train_flags(argv: Sequence[str]) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        if tok in _FLAGS_WITH_VALUE:
            i += 2
            continue
        if "=" in tok:
            name = tok.split("=", 1)[0]
            if name in _FLAGS_WITH_VALUE:
                i += 1
                continue
        out.append(tok)
        i += 1
    return out


def _get_flag_value(argv: Sequence[str], flag: str) -> str | None:
    eq = f"{flag}="
    for i, t in enumerate(argv):
        if t == flag and i + 1 < len(argv):
            return argv[i + 1]
        if t.startswith(eq):
            return t[len(eq) :]
    return None


# ---------------------------------------------------------------------------
# Optional backends
# ---------------------------------------------------------------------------

_OPTUNA_INSTALL = "pip install optuna   # or: pip install 'atpgllm[graph-search]'"
_AX_INSTALL = "pip install ax-platform   # or: pip install 'atpgllm[graph-search]'"


def _import_optuna() -> Any:
    try:
        import optuna  # type: ignore[import-not-found]

        return optuna
    except ImportError as e:
        print(f"Error: --sampler optuna requires Optuna. {_OPTUNA_INSTALL}", file=sys.stderr)
        raise SystemExit(1) from e


def _import_ax() -> tuple[Any, Any]:
    try:
        from ax.service.ax_client import AxClient  # type: ignore[import-not-found]
        from ax.service.utils.instantiation import (  # type: ignore[import-not-found]
            ObjectiveProperties,
        )

        return AxClient, ObjectiveProperties
    except ImportError as e:
        print(f"Error: --sampler ax requires Ax. {_AX_INSTALL}", file=sys.stderr)
        raise SystemExit(1) from e


# ---------------------------------------------------------------------------
# Search space + trial sampling (sobol / random)
# ---------------------------------------------------------------------------

# Discretize loss weights w_gtc / w_gtm / w_gtg on a linear grid (Optuna, Ax, sobol, random).
W_LOSS_WEIGHT_STEP = 1e-2


@dataclass(frozen=True, slots=True)
class HparamSpace:
    """Bounds for loss-weight ranges (step :data:`W_LOSS_WEIGHT_STEP`) and discretized extras."""

    w_gtc: tuple[float, float]
    w_gtm: tuple[float, float]
    w_gtg: tuple[float, float]
    lr: tuple[float, float, float]  # (low, high, step) linear in lr
    qformer_layers: tuple[int, int]  # inclusive
    qformer_hidden: tuple[int, int, int]  # (low, high, step) inclusive grid
    gin_layers: tuple[int, int]
    gin_hidden: tuple[int, int, int]
    proj_dim: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class SearchTrial:
    """One joint sample: loss weights + model / lr hyperparameters."""

    w_gtc: float
    w_gtm: float
    w_gtg: float
    lr: float
    qformer_layers: int
    qformer_hidden: int
    gin_layers: int
    gin_hidden: int
    proj_dim: int


def _n_linear_lr_levels(lo: float, hi: float, step: float) -> int:
    n = int(round((hi - lo) / step)) + 1
    return max(1, n)


def _n_int_grid_levels(low: int, high: int, step: int) -> int:
    n = (high - low) // step + 1
    return max(1, n)


def _n_loss_weight_levels(lo: float, hi: float) -> int:
    """Count of values ``lo, lo+step, …`` up to ``hi`` (inclusive), step ``W_LOSS_WEIGHT_STEP``."""
    n = int(round((hi - lo) / W_LOSS_WEIGHT_STEP)) + 1
    return max(1, n)


def _weight_from_index(lo: float, hi: float, i: int) -> float:
    n = _n_loss_weight_levels(lo, hi)
    ii = max(0, min(n - 1, int(i)))
    v = lo + ii * W_LOSS_WEIGHT_STEP
    return min(hi, max(lo, v))


def _snap_weight_to_grid(lo: float, hi: float, v: float) -> float:
    """Map a weight onto the 1e-2 grid in ``[lo, hi]`` (for Optuna dist validation)."""
    n = _n_loss_weight_levels(lo, hi)
    if n <= 1:
        return min(hi, max(lo, float(v)))
    i = int(round((float(v) - lo) / W_LOSS_WEIGHT_STEP))
    return _weight_from_index(lo, hi, i)


def _unit_to_index(u: float, n: int) -> int:
    uu = min(1.0 - 1e-9, max(1e-9, u))
    return int(uu * n) % n


def _index_to_float_grid(lo: float, step: float, i: int) -> float:
    return lo + i * step


def _index_to_int_grid(low: int, step: int, i: int) -> int:
    return low + i * step


def _sample_trial_from_units(u: list[float], space: HparamSpace) -> SearchTrial:
    w_lo = (space.w_gtc[0], space.w_gtm[0], space.w_gtg[0])
    w_hi = (space.w_gtc[1], space.w_gtm[1], space.w_gtg[1])
    lr_lo, lr_hi, lr_s = space.lr
    n_lr = _n_linear_lr_levels(lr_lo, lr_hi, lr_s)
    li = _unit_to_index(u[3], n_lr)
    lr = _index_to_float_grid(lr_lo, lr_s, li)
    if lr > lr_hi:  # numeric edge
        lr = lr_hi

    qfl_lo, qfl_hi = space.qformer_layers
    n_ql = qfl_hi - qfl_lo + 1
    qf_layers = qfl_lo + _unit_to_index(u[4], n_ql)

    qfh_lo, qfh_hi, qfh_s = space.qformer_hidden
    n_qfh = _n_int_grid_levels(qfh_lo, qfh_hi, qfh_s)
    qf_hidden = _index_to_int_grid(qfh_lo, qfh_s, _unit_to_index(u[5], n_qfh))

    gil_lo, gil_hi = space.gin_layers
    n_gil = gil_hi - gil_lo + 1
    g_layers = gil_lo + _unit_to_index(u[6], n_gil)

    gih_lo, gih_hi, gih_s = space.gin_hidden
    n_gih = _n_int_grid_levels(gih_lo, gih_hi, gih_s)
    g_hidden = _index_to_int_grid(gih_lo, gih_s, _unit_to_index(u[7], n_gih))

    pd_lo, pd_hi, pd_s = space.proj_dim
    n_pd = _n_int_grid_levels(pd_lo, pd_hi, pd_s)
    pr_dim = _index_to_int_grid(pd_lo, pd_s, _unit_to_index(u[8], n_pd))

    n_w0 = _n_loss_weight_levels(w_lo[0], w_hi[0])
    n_w1 = _n_loss_weight_levels(w_lo[1], w_hi[1])
    n_w2 = _n_loss_weight_levels(w_lo[2], w_hi[2])
    w_gtc = _weight_from_index(w_lo[0], w_hi[0], _unit_to_index(u[0], n_w0))
    w_gtm = _weight_from_index(w_lo[1], w_hi[1], _unit_to_index(u[1], n_w1))
    w_gtg = _weight_from_index(w_lo[2], w_hi[2], _unit_to_index(u[2], n_w2))

    return SearchTrial(
        w_gtc=w_gtc,
        w_gtm=w_gtm,
        w_gtg=w_gtg,
        lr=lr,
        qformer_layers=qf_layers,
        qformer_hidden=qf_hidden,
        gin_layers=g_layers,
        gin_hidden=g_hidden,
        proj_dim=pr_dim,
    )


def _sobol_dim() -> int:
    return 9  # 3 loss-weight steps + 6 discretized controls


def sample_trials_sobol(n: int, space: HparamSpace, seed: int) -> list[SearchTrial]:
    eng = torch.quasirandom.SobolEngine(dimension=_sobol_dim(), scramble=True, seed=seed)
    raw = eng.draw(n)
    raw = torch.clamp(raw, 1e-6, 1.0 - 1e-6)
    return [_sample_trial_from_units(raw[i].tolist(), space) for i in range(n)]


def sample_trials_random(n: int, space: HparamSpace, seed: int) -> list[SearchTrial]:
    g = torch.Generator().manual_seed(seed)
    trials: list[SearchTrial] = []
    for _ in range(n):
        u = [torch.rand(1, generator=g).item() for _ in range(_sobol_dim())]
        trials.append(_sample_trial_from_units(u, space))
    return trials


@dataclass
class TrialResult:
    trial_id: int
    hparams: SearchTrial
    gpu: int
    mean_weighted: float | None
    metrics_path: Path
    returncode: int
    stderr_tail: str

    @property
    def w_gtc(self) -> float:
        return self.hparams.w_gtc

    @property
    def w_gtm(self) -> float:
        return self.hparams.w_gtm

    @property
    def w_gtg(self) -> float:
        return self.hparams.w_gtg


def _wandb_child_argv(
    *,
    project: str,
    entity: str | None,
    group: str,
    run_name: str,
    tags: str,
    mode: str,
) -> list[str]:
    out: list[str] = [
        "--wandb-project",
        project,
        "--wandb-group",
        group,
        "--wandb-run-name",
        run_name,
        "--wandb-mode",
        mode,
    ]
    if entity:
        out += ["--wandb-entity", entity]
    if tags.strip():
        out += ["--wandb-tags", tags]
    return out


@dataclass
class WandbHpoContext:
    """Forwarded to each train_stage1 subprocess when running an HPO sweep."""

    project: str
    entity: str | None
    group: str
    tags: str
    mode: str
    run_name_prefix: str

    def argv_for_trial(self, trial_id: int) -> list[str]:
        return _wandb_child_argv(
            project=self.project,
            entity=self.entity,
            group=self.group,
            run_name=f"{self.run_name_prefix}_{trial_id:04d}",
            tags=self.tags,
            mode=self.mode,
        )


def _run_subprocess(
    trial_id: int,
    hparams: SearchTrial,
    gpu_id: int,
    filtered_train_argv: list[str],
    trial_dir: Path,
    metrics_path: Path,
    extra_train_args: Sequence[str] = (),
) -> TrialResult:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable,
        "-m",
        "atpgllm.graph.scripts.train_stage1",
        *filtered_train_argv,
        *extra_train_args,
        "--output-dir",
        str(trial_dir),
        "--w-gtc",
        str(hparams.w_gtc),
        "--w-gtm",
        str(hparams.w_gtm),
        "--w-gtg",
        str(hparams.w_gtg),
        "--lr",
        str(hparams.lr),
        "--qformer-layers",
        str(hparams.qformer_layers),
        "--qformer-hidden",
        str(hparams.qformer_hidden),
        "--gin-layers",
        str(hparams.gin_layers),
        "--gin-hidden",
        str(hparams.gin_hidden),
        "--proj-dim",
        str(hparams.proj_dim),
        "--metrics-json",
        str(metrics_path),
        "--device",
        "cuda",
    ]
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )
    err_tail = (proc.stderr or "")[-4000:]
    mean_w: float | None = None
    if metrics_path.is_file():
        try:
            data = json.loads(metrics_path.read_text())
            mean_w = data.get("mean_weighted_loss_last_window")
        except (json.JSONDecodeError, OSError):
            mean_w = None
    return TrialResult(
        trial_id=trial_id,
        hparams=hparams,
        gpu=gpu_id,
        mean_weighted=mean_w,
        metrics_path=metrics_path,
        returncode=proc.returncode,
        stderr_tail=err_tail,
    )


def _run_parallel_subprocesses(
    work: list[tuple[int, SearchTrial, int, Path, Path, tuple[str, ...]]],
    filtered: list[str],
    max_workers: int,
) -> list[TrialResult]:
    results: list[TrialResult] = []

    def _job(
        item: tuple[int, SearchTrial, int, Path, Path, tuple[str, ...]],
    ) -> TrialResult:
        tid, hp, gpu_slot, tdir, mpath, extra = item
        return _run_subprocess(
            tid, hp, gpu_slot, filtered, tdir, mpath, extra_train_args=extra
        )

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        futures = [ex.submit(_job, item) for item in work]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


# ---------------------------------------------------------------------------
# Optuna driver
# ---------------------------------------------------------------------------

_AX_OBJECTIVE = "mean_weighted_loss"


def _optuna_suggest_hparams(trial: Any, space: HparamSpace) -> SearchTrial:
    wlo, whi = space.w_gtc[0], space.w_gtc[1]
    w_gtc = trial.suggest_float("w_gtc", wlo, whi, step=W_LOSS_WEIGHT_STEP)
    wlo, whi = space.w_gtm[0], space.w_gtm[1]
    w_gtm = trial.suggest_float("w_gtm", wlo, whi, step=W_LOSS_WEIGHT_STEP)
    wlo, whi = space.w_gtg[0], space.w_gtg[1]
    w_gtg = trial.suggest_float("w_gtg", wlo, whi, step=W_LOSS_WEIGHT_STEP)
    lr_lo, lr_hi, lr_s = space.lr
    lr = trial.suggest_float("lr", lr_lo, lr_hi, step=lr_s)
    qfl0, qfl1 = space.qformer_layers
    qformer_layers = trial.suggest_int("qformer_layers", qfl0, qfl1)
    qh0, qh1, qh_s = space.qformer_hidden
    qformer_hidden = trial.suggest_int("qformer_hidden", qh0, qh1, step=qh_s)
    gil0, gil1 = space.gin_layers
    gin_layers = trial.suggest_int("gin_layers", gil0, gil1)
    gh0, gh1, gh_s = space.gin_hidden
    gin_hidden = trial.suggest_int("gin_hidden", gh0, gh1, step=gh_s)
    pd0, pd1, pd_s = space.proj_dim
    proj_dim = trial.suggest_int("proj_dim", pd0, pd1, step=pd_s)
    return SearchTrial(
        w_gtc=w_gtc,
        w_gtm=w_gtm,
        w_gtg=w_gtg,
        lr=lr,
        qformer_layers=qformer_layers,
        qformer_hidden=qformer_hidden,
        gin_layers=gin_layers,
        gin_hidden=gin_hidden,
        proj_dim=proj_dim,
    )


def _run_optuna(
    *,
    num_trials: int,
    seed: int,
    space: HparamSpace,
    gpus: list[int],
    root_out: Path,
    filtered: list[str],
    optuna_sampler: str,
    optuna_storage: str | None,
    wandb_ctx: WandbHpoContext | None,
) -> tuple[list[TrialResult], Any]:
    opt = _import_optuna()
    if optuna_sampler == "tpe":
        sampler = opt.samplers.TPESampler(seed=seed)
    else:
        sampler = opt.samplers.RandomSampler(seed=seed)

    storage = optuna_storage
    if storage is None:
        storage = f"sqlite:///{root_out.resolve()}/optuna_study.db"
    
    study = opt.create_study(
        study_name="stage1_loss_weights",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    collected: list[TrialResult] = []
    coll_lock = threading.Lock()

    def objective(trial: Any) -> float:
        hparams = _optuna_suggest_hparams(trial, space)
        tid = trial.number
        gpu_slot = gpus[tid % len(gpus)]
        trial_dir = root_out / f"trial_optuna_{tid:04d}"
        metrics_path = trial_dir / "metrics.json"
        trial_dir.mkdir(parents=True, exist_ok=True)
        extra = wandb_ctx.argv_for_trial(tid) if wandb_ctx else ()
        r = _run_subprocess(
            tid,
            hparams,
            gpu_slot,
            filtered,
            trial_dir,
            metrics_path,
            extra_train_args=extra,
        )
        with coll_lock:
            collected.append(r)
        if r.returncode != 0 or r.mean_weighted is None:
            return float("inf")
        return float(r.mean_weighted)

    study.optimize(
        objective,
        n_trials=num_trials,
        n_jobs=len(gpus),
        show_progress_bar=True,
    )

    collected.sort(key=lambda r: r.trial_id)
    return collected, study


# ---------------------------------------------------------------------------
# Ax driver
# ---------------------------------------------------------------------------


def _ax_params_to_search_trial(p: dict[str, Any], space: HparamSpace) -> SearchTrial:
    """Map Ax-sampled dict (including index params for discretized grids) to ``SearchTrial``."""
    n_wc = _n_loss_weight_levels(space.w_gtc[0], space.w_gtc[1])
    n_wm = _n_loss_weight_levels(space.w_gtm[0], space.w_gtm[1])
    n_wg = _n_loss_weight_levels(space.w_gtg[0], space.w_gtg[1])
    wci = max(0, min(n_wc - 1, int(p["w_gtc_index"])))
    wmi = max(0, min(n_wm - 1, int(p["w_gtm_index"])))
    wgi = max(0, min(n_wg - 1, int(p["w_gtg_index"])))
    w_gtc = _weight_from_index(space.w_gtc[0], space.w_gtc[1], wci)
    w_gtm = _weight_from_index(space.w_gtm[0], space.w_gtm[1], wmi)
    w_gtg = _weight_from_index(space.w_gtg[0], space.w_gtg[1], wgi)
    lr_lo, lr_hi, lr_s = space.lr
    n_lr = _n_linear_lr_levels(lr_lo, lr_hi, lr_s)
    lri = int(p["lr_index"])
    lri = max(0, min(n_lr - 1, lri))
    lr = _index_to_float_grid(lr_lo, lr_s, lri)
    if lr > lr_hi:
        lr = lr_hi

    qfl0, qfl1 = space.qformer_layers
    qf_layers = int(p["qformer_layers"])
    qf_layers = max(qfl0, min(qfl1, qf_layers))

    qh0, qh1, qh_s = space.qformer_hidden
    n_qfh = _n_int_grid_levels(qh0, qh1, qh_s)
    qfhi = int(p["qformer_hidden_index"])
    qfhi = max(0, min(n_qfh - 1, qfhi))
    qf_hidden = _index_to_int_grid(qh0, qh_s, qfhi)

    gil0, gil1 = space.gin_layers
    g_layers = int(p["gin_layers"])
    g_layers = max(gil0, min(gil1, g_layers))

    gh0, gh1, gh_s = space.gin_hidden
    n_gh = _n_int_grid_levels(gh0, gh1, gh_s)
    ghi = int(p["gin_hidden_index"])
    ghi = max(0, min(n_gh - 1, ghi))
    g_hidden = _index_to_int_grid(gh0, gh_s, ghi)

    pd0, pd1, pd_s = space.proj_dim
    n_pd = _n_int_grid_levels(pd0, pd1, pd_s)
    pdi = int(p["proj_dim_index"])
    pdi = max(0, min(n_pd - 1, pdi))
    pr_dim = _index_to_int_grid(pd0, pd_s, pdi)
    return SearchTrial(
        w_gtc=w_gtc,
        w_gtm=w_gtm,
        w_gtg=w_gtg,
        lr=lr,
        qformer_layers=qf_layers,
        qformer_hidden=qf_hidden,
        gin_layers=g_layers,
        gin_hidden=g_hidden,
        proj_dim=pr_dim,
    )


def _ax_parameter_list(space: HparamSpace) -> list[dict[str, Any]]:
    lr_lo, lr_hi, lr_s = space.lr
    n_lr = _n_linear_lr_levels(lr_lo, lr_hi, lr_s)
    if n_lr < 2:
        n_lr = 2
    qh0, qh1, qh_s = space.qformer_hidden
    n_qfh = _n_int_grid_levels(qh0, qh1, qh_s)
    gh0, gh1, gh_s = space.gin_hidden
    n_gh = _n_int_grid_levels(gh0, gh1, gh_s)
    pd0, pd1, pd_s = space.proj_dim
    n_pd = _n_int_grid_levels(pd0, pd1, pd_s)
    qf_lo, qf_hi = space.qformer_layers
    gi_lo, gi_hi = space.gin_layers
    n_wc = max(1, _n_loss_weight_levels(space.w_gtc[0], space.w_gtc[1]))
    n_wm = max(1, _n_loss_weight_levels(space.w_gtm[0], space.w_gtm[1]))
    n_wg = max(1, _n_loss_weight_levels(space.w_gtg[0], space.w_gtg[1]))
    return [
        {
            "name": "w_gtc_index",
            "type": "range",
            "bounds": [0, n_wc - 1],
            "value_type": "int",
        },
        {
            "name": "w_gtm_index",
            "type": "range",
            "bounds": [0, n_wm - 1],
            "value_type": "int",
        },
        {
            "name": "w_gtg_index",
            "type": "range",
            "bounds": [0, n_wg - 1],
            "value_type": "int",
        },
        {
            "name": "lr_index",
            "type": "range",
            "bounds": [0, n_lr - 1],
            "value_type": "int",
        },
        {
            "name": "qformer_layers",
            "type": "range",
            "bounds": [qf_lo, qf_hi],
            "value_type": "int",
        },
        {
            "name": "qformer_hidden_index",
            "type": "range",
            "bounds": [0, n_qfh - 1],
            "value_type": "int",
        },
        {
            "name": "gin_layers",
            "type": "range",
            "bounds": [gi_lo, gi_hi],
            "value_type": "int",
        },
        {
            "name": "gin_hidden_index",
            "type": "range",
            "bounds": [0, n_gh - 1],
            "value_type": "int",
        },
        {
            "name": "proj_dim_index",
            "type": "range",
            "bounds": [0, n_pd - 1],
            "value_type": "int",
        },
    ]


def _run_ax(
    *,
    num_trials: int,
    seed: int,
    space: HparamSpace,
    gpus: list[int],
    root_out: Path,
    filtered: list[str],
    wandb_ctx: WandbHpoContext | None,
) -> list[TrialResult]:
    AxClient, ObjectiveProperties = _import_ax()

    ax_client = AxClient(
        random_seed=seed,
        enforce_sequential_optimization=False,
        verbose_logging=False,
    )
    ax_client.create_experiment(
        name="atpgllm_graph_stage1_loss_weights",
        parameters=_ax_parameter_list(space),
        objectives={_AX_OBJECTIVE: ObjectiveProperties(minimize=True)},
        overwrite_existing_experiment=True,
    )

    results: list[TrialResult] = []
    n_completed = 0

    while n_completed < num_trials:
        remaining = num_trials - n_completed
        batch_cap = min(len(gpus), remaining)
        trials_dict, optimization_complete = ax_client.get_next_trials(
            max_trials=batch_cap
        )
        if not trials_dict:
            if optimization_complete:
                print("Ax: optimization complete (no further trials).")
            else:
                print("Ax: no trials returned; stopping.")
            break

        work: list[
            tuple[int, SearchTrial, int, Path, Path, tuple[str, ...]]
        ] = []
        sorted_items = sorted(trials_dict.items(), key=lambda kv: kv[0])
        for idx_within, (trial_index, params) in enumerate(sorted_items):
            hp = _ax_params_to_search_trial(params, space)
            gpu_slot = gpus[idx_within % len(gpus)]
            trial_dir = root_out / f"trial_ax_{trial_index:04d}"
            metrics_path = trial_dir / "metrics.json"
            trial_dir.mkdir(parents=True, exist_ok=True)
            extra = (
                wandb_ctx.argv_for_trial(trial_index) if wandb_ctx else ()
            )
            work.append(
                (trial_index, hp, gpu_slot, trial_dir, metrics_path, extra)
            )

        def _one(
            item: tuple[int, SearchTrial, int, Path, Path, tuple[str, ...]],
        ) -> TrialResult:
            tid, hparams, gpu_slot, tdir, mpath, extra = item
            return _run_subprocess(
                tid,
                hparams,
                gpu_slot,
                filtered,
                tdir,
                mpath,
                extra_train_args=extra,
            )

        batch_results: list[TrialResult] = []
        with ThreadPoolExecutor(max_workers=len(work)) as ex:
            futs = [ex.submit(_one, item) for item in work]
            for fut in futs:
                batch_results.append(fut.result())

        for r in batch_results:
            results.append(r)
            if r.returncode == 0 and r.mean_weighted is not None:
                ax_client.complete_trial(
                    trial_index=r.trial_id,
                    raw_data={_AX_OBJECTIVE: (float(r.mean_weighted), None)},
                )
            else:
                ax_client.log_trial_failure(
                    trial_index=r.trial_id,
                    metadata={"stderr": r.stderr_tail[:800]},
                )

        n_completed += len(trials_dict)
        if optimization_complete and n_completed >= num_trials:
            break

    results.sort(key=lambda x: x.trial_id)
    return results


# ---------------------------------------------------------------------------
# Optuna visualization (HTML under output dir; best-effort)
# ---------------------------------------------------------------------------


def _hparams_to_param_dict(h: SearchTrial) -> dict[str, Any]:
    return {
        "w_gtc": h.w_gtc,
        "w_gtm": h.w_gtm,
        "w_gtg": h.w_gtg,
        "lr": h.lr,
        "qformer_layers": h.qformer_layers,
        "qformer_hidden": h.qformer_hidden,
        "gin_layers": h.gin_layers,
        "gin_hidden": h.gin_hidden,
        "proj_dim": h.proj_dim,
    }


def _optuna_distributions_for_space(space: HparamSpace) -> dict[str, Any]:
    from optuna.distributions import FloatDistribution, IntDistribution

    lr_lo, lr_hi, lr_s = space.lr
    qh0, qh1, qh_s = space.qformer_hidden
    gh0, gh1, gh_s = space.gin_hidden
    pd0, pd1, pd_s = space.proj_dim
    return {
        "w_gtc": FloatDistribution(
            space.w_gtc[0], space.w_gtc[1], step=W_LOSS_WEIGHT_STEP
        ),
        "w_gtm": FloatDistribution(
            space.w_gtm[0], space.w_gtm[1], step=W_LOSS_WEIGHT_STEP
        ),
        "w_gtg": FloatDistribution(
            space.w_gtg[0], space.w_gtg[1], step=W_LOSS_WEIGHT_STEP
        ),
        "lr": FloatDistribution(lr_lo, lr_hi, step=lr_s),
        "qformer_layers": IntDistribution(
            space.qformer_layers[0], space.qformer_layers[1]
        ),
        "qformer_hidden": IntDistribution(qh0, qh1, step=qh_s),
        "gin_layers": IntDistribution(space.gin_layers[0], space.gin_layers[1]),
        "gin_hidden": IntDistribution(gh0, gh1, step=gh_s),
        "proj_dim": IntDistribution(pd0, pd1, step=pd_s),
    }


def _build_minimal_optuna_study(
    space: HparamSpace,
    results: list[TrialResult],
) -> Any | None:
    """In-memory study with only successful trials (for Optuna visualization)."""
    try:
        import optuna  # type: ignore[import-not-found]
        from optuna.trial import create_trial, TrialState
    except ImportError:
        return None

    dists = _optuna_distributions_for_space(space)
    to_add: list[Any] = []
    for r in results:
        if r.returncode != 0 or r.mean_weighted is None:
            continue
        p = _hparams_to_param_dict(r.hparams)
        p["w_gtc"] = _snap_weight_to_grid(space.w_gtc[0], space.w_gtc[1], float(p["w_gtc"]))
        p["w_gtm"] = _snap_weight_to_grid(space.w_gtm[0], space.w_gtm[1], float(p["w_gtm"]))
        p["w_gtg"] = _snap_weight_to_grid(space.w_gtg[0], space.w_gtg[1], float(p["w_gtg"]))
        to_add.append(
            create_trial(
                state=TrialState.COMPLETE,
                value=float(r.mean_weighted),
                params=p,
                distributions=dists,
            )
        )
    if not to_add:
        return None
    study = optuna.create_study(direction="minimize")
    study.add_trials(to_add)
    return study


def _set_hpo_slice_figure_yaxis_title(fig: Any, study: Any) -> None:
    """Label the objective on the y-axis of slice subplots, including min/max sense."""
    try:
        import optuna  # type: ignore[import-not-found]
    except ImportError:
        y_title = "Objective value (lower is better)"
    else:
        d = study.direction
        y_title = (
            "Objective value (higher is better)"
            if d == optuna.study.StudyDirection.MAXIMIZE
            else "Objective value (lower is better)"
        )

    def _yax_each(a: Any) -> None:
        if a is not None:
            a.update(title_text=y_title)

    if hasattr(fig, "for_each_yaxis"):
        try:
            fig.for_each_yaxis(_yax_each)  # type: ignore[no-untyped-call]
            return
        except (TypeError, AttributeError, ValueError) as e:
            pass
    try:
        fig.update_yaxes(title_text=y_title)
    except (TypeError, AttributeError, ValueError) as e:
        try:
            fig.update_layout(yaxis_title=y_title)  # type: ignore[union-attr]
        except (TypeError, AttributeError, ValueError) as e2:  # noqa: BLE001
            pass


def _optuna_study_figure_pairs(study: Any, space: HparamSpace) -> list[tuple[str, Any]]:
    """
    Build the same Optuna/Plotly figures that we save under ``hpo_plots/``.

    Returns ``(name, fig)`` only for figures that were produced successfully.
    """
    import optuna.visualization as vis  # type: ignore[import-not-found]

    all_params = list(_optuna_distributions_for_space(space).keys())
    contour_a, contour_b = "lr", "qformer_hidden"
    if contour_a not in all_params or contour_b not in all_params:
        contour_a, contour_b = all_params[0], all_params[1]
    w_first: tuple[str, ...] = ("w_gtc", "w_gtm", "w_gtg")
    rest = [p for p in all_params if p not in w_first]
    _known_rest: tuple[str, ...] = (
        "lr",
        "qformer_layers",
        "qformer_hidden",
        "gin_layers",
        "gin_hidden",
        "proj_dim",
    )
    rest_ordered: list[str] = [p for p in _known_rest if p in rest]
    for p in rest:
        if p not in rest_ordered:
            rest_ordered.append(p)
    slice_params: list[str] = [
        p for p in (*w_first, *rest_ordered) if p in all_params
    ]
    out: list[tuple[str, Any]] = []
    for name, factory, args in (
        ("optimization_history", vis.plot_optimization_history, (study,)),
        ("param_importances", vis.plot_param_importances, (study,)),
        ("parallel_coordinate", vis.plot_parallel_coordinate, (study,)),
    ):
        try:
            out.append((name, factory(*args)))
        except Exception as e:  # noqa: BLE001
            print(f"HPO plot {name}: {e}", file=sys.stderr)
    try:
        out.append(
            (
                f"contour__{contour_a}__{contour_b}",
                vis.plot_contour(study, params=[contour_a, contour_b]),
            )
        )
    except Exception as e:  # noqa: BLE001
        print(f"HPO plot contour: {e}", file=sys.stderr)
    try:
        if slice_params:
            sfig = vis.plot_slice(study, params=slice_params)
            _set_hpo_slice_figure_yaxis_title(sfig, study)
            out.append(("slice", sfig))
    except Exception as e:  # noqa: BLE001
        print(f"HPO plot slice: {e}", file=sys.stderr)
    return out


def _save_hpo_plots(
    space: HparamSpace,
    results: list[TrialResult],
    root_out: Path,
) -> None:
    """
    Write Optuna Plotly HTML reports (mirrors the useful plots in ``optuna_exper.py``).

    We skip ``plot_intermediate_values`` (no per-step HPO progress is recorded here).
    Contour / slice / importance are best-effort: they can fail with too few points
    or highly discrete parameters. Files use ``include_plotlyjs="cdn"`` to keep
    on-disk size reasonable (same as W&B ``wandb.Html`` panels).
    """
    study = _build_minimal_optuna_study(space, results)
    if study is None:
        print(
            "HPO plots: no successful trials with metrics; skip.",
            file=sys.stderr,
        )
        return
    try:
        import optuna.visualization  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        print(
            "HPO plots: optuna (with visualization) not available; skip.",
            file=sys.stderr,
        )
        return
    n = len(study.get_trials())
    if n < 2:
        print(
            f"HPO plots: need at least 2 successful trials, got {n}; skip.",
            file=sys.stderr,
        )
        return

    out = root_out / "hpo_plots"
    out.mkdir(parents=True, exist_ok=True)

    def _write(fig: Any, name: str) -> None:
        p = out / f"{name}.html"
        try:
            fig.write_html(  # type: ignore[union-attr]
                str(p), include_plotlyjs="cdn", full_html=True
            )
            print(f"Wrote {p}")
        except Exception as e:  # noqa: BLE001
            print(f"HPO plot {name}: {e}", file=sys.stderr)

    for name, fig in _optuna_study_figure_pairs(study, space):
        _write(fig, name)


# ---------------------------------------------------------------------------
# HPO output files (local + W&B Artifacts)
# ---------------------------------------------------------------------------


def _write_hpo_trial_results_json(
    root_out: Path, results: list[TrialResult]
) -> bool:
    """Write ``hpo_trial_results.json`` under ``root_out``. Return True on success."""
    try:
        rows: list[dict[str, Any]] = []
        for r in results:
            hp = r.hparams
            row = {
                "trial_id": r.trial_id,
                "returncode": r.returncode,
                "mean_weighted": r.mean_weighted,
                "w_gtc": hp.w_gtc,
                "w_gtm": hp.w_gtm,
                "w_gtg": hp.w_gtg,
                "lr": hp.lr,
                "qformer_layers": hp.qformer_layers,
                "qformer_hidden": hp.qformer_hidden,
                "gin_layers": hp.gin_layers,
                "gin_hidden": hp.gin_hidden,
                "proj_dim": hp.proj_dim,
                "metrics_path": str(r.metrics_path),
            }
            rows.append(row)
        out_path = root_out / "hpo_trial_results.json"
        out_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")
    except OSError as e:
        print(f"Could not write hpo_trial_results.json: {e}", file=sys.stderr)
        return False
    return True


def _safe_wandb_artifact_name(group: str) -> str:
    s = "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in (group or "hpo")
    ).strip("_")
    s = f"hpo_{s}" if s else "hpo_artifact"
    return s[:64]


def _log_wandb_hpo_artifact(
    run: Any,
    root_out: Path,
    group: str,
    mode: str,
) -> None:
    """Attach ``hpo_trial_results.json`` and ``hpo_plots/`` to the summary run (UI)."""
    if mode == "disabled":
        return
    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError:
        return
    hpo_json = root_out / "hpo_trial_results.json"
    hpo_plots = root_out / "hpo_plots"
    has_plots = (
        hpo_plots.is_dir()
        and any(
            p.is_file() and p.suffix.lower() in (".html", ".json")
            for p in hpo_plots.rglob("*")
        )
    )
    if not hpo_json.is_file() and not has_plots:
        print(
            "W&B: no hpo_trial_results.json or hpo_plots/*.html to upload; "
            "skip artifact.",
            file=sys.stderr,
        )
        return
    art = wandb.Artifact(
        _safe_wandb_artifact_name(group),
        type="hpo",
        description="HPO per-trial JSON + Optuna/Plotly HTML (open HTML in Artifacts for interactive plots).",
    )
    if hpo_json.is_file():
        art.add_file(str(hpo_json), name="hpo_trial_results.json")
    if has_plots:
        art.add_dir(str(hpo_plots), name="hpo_plots")
    try:
        run.log_artifact(art)
        print(
            f"W&B: logged artifact {art.name!r} (JSON + hpo_plots) — "
            "open the Artifacts tab on this run to preview files."
        )
    except Exception as e:  # noqa: BLE001
        print(f"W&B: could not log HPO artifact: {e}", file=sys.stderr)


def _hpo_plots_wandb_html_payload(
    wandb_mod: Any,
    max_log_bytes: int,
    space: HparamSpace,
    results: list[TrialResult],
) -> dict[str, Any]:
    """
    ``wandb.Html`` entries for HPO plot panels (rebuilt from Optuna; CDN-thin HTML).

    Uses ``plotly.io.to_html(..., full_html=False, include_plotlyjs="cdn")`` so
    interactive figures render in the W&B UI (unlike multi‑MB inline ``write_html`` files).
    """
    try:
        import plotly.io as pio  # type: ignore[import-not-found]
    except ImportError:
        print(
            "W&B: plotly not installed; skip HPO plot HTML panels "
            "(install atpgllm[graph-search] or plotly).",
            file=sys.stderr,
        )
        return {}

    study = _build_minimal_optuna_study(space, results)
    if study is None or len(study.get_trials()) < 2:
        return {}
    out: dict[str, Any] = {}
    for name, fig in _optuna_study_figure_pairs(study, space):
        try:
            html = pio.to_html(  # type: ignore[union-attr]
                fig, full_html=False, include_plotlyjs="cdn"
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"W&B: plotly to_html failed for {name!r}: {e}",
                file=sys.stderr,
            )
            continue
        b = len(html.encode("utf-8"))
        if b > max_log_bytes:
            print(
                f"W&B: skip wandb.Html for {name!r} "
                f"({b} bytes > {max_log_bytes}).",
                file=sys.stderr,
            )
            continue
        out[f"hpo_plots/{name}"] = wandb_mod.Html(html)
    return out


# ---------------------------------------------------------------------------
# W&B HPO summary (orchestrator run in the same group as child training runs)
# ---------------------------------------------------------------------------


def _log_wandb_hpo_summary(
    *,
    project: str,
    entity: str | None,
    group: str,
    tags: str,
    mode: str,
    sampler: str,
    num_trials: int,
    space: HparamSpace,
    results: list[TrialResult],
    root_out: Path,
) -> None:
    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError:
        print(
            "wandb not installed; skipping HPO summary run "
            "(child runs may still have logged if wandb was available to them).",
            file=sys.stderr,
        )
        return

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    tag_list.extend(["hpo_summary", sampler, "stage1_hpo"])
    meta: dict[str, Any] = {
        "sampler": sampler,
        "num_trials": num_trials,
        "w_gtc_bounds": [space.w_gtc[0], space.w_gtc[1]],
        "w_gtm_bounds": [space.w_gtm[0], space.w_gtm[1]],
        "w_gtg_bounds": [space.w_gtg[0], space.w_gtg[1]],
        "lr": list(space.lr),
        "qformer_layers": list(space.qformer_layers),
        "qformer_hidden": list(space.qformer_hidden),
        "gin_layers": list(space.gin_layers),
        "gin_hidden": list(space.gin_hidden),
        "proj_dim": list(space.proj_dim),
        "output_dir": str(root_out.resolve()),
    }
    safe_name = f"_hpo_summary_{group}".replace("/", "_")[:120]
    run = wandb.init(
        project=project,
        entity=entity or None,
        group=group,
        name=safe_name,
        tags=tag_list,
        job_type="hpo_summary",
        config=meta,
        mode=mode,
    )
    url = getattr(run, "url", None)
    if url:
        print(f"W&B HPO summary run: {url}")

    ok = [r for r in results if r.returncode == 0 and r.mean_weighted is not None]
    best = min(ok, key=lambda r: r.mean_weighted) if ok else None

    tbl = wandb.Table(
        columns=[
            "trial_id",
            "gpu",
            "status",
            "w_gtc",
            "w_gtm",
            "w_gtg",
            "lr",
            "qformer_layers",
            "qformer_hidden",
            "gin_layers",
            "gin_hidden",
            "proj_dim",
            "mean_weighted_loss",
            "returncode",
        ]
    )
    for r in sorted(results, key=lambda x: x.trial_id):
        st = "ok" if r.returncode == 0 else "fail"
        mw = r.mean_weighted if r.mean_weighted is not None else float("nan")
        hp = r.hparams
        tbl.add_data(
            r.trial_id,
            r.gpu,
            st,
            hp.w_gtc,
            hp.w_gtm,
            hp.w_gtg,
            hp.lr,
            hp.qformer_layers,
            hp.qformer_hidden,
            hp.gin_layers,
            hp.gin_hidden,
            hp.proj_dim,
            mw,
            r.returncode,
        )

    payload: dict[str, Any] = {
        "hpo/trials_table": tbl,
        "hpo/n_trials": len(results),
        "hpo/n_success": len(ok),
        "hpo/n_failed_return": sum(1 for r in results if r.returncode != 0),
        "hpo/n_missing_metric": sum(
            1 for r in results if r.returncode == 0 and r.mean_weighted is None
        ),
    }
    if best is not None:
        bhp = best.hparams
        payload["hpo/best_mean_weighted"] = float(best.mean_weighted)
        payload["hpo/best_trial_id"] = int(best.trial_id)
        payload["hpo/best_w_gtc"] = float(bhp.w_gtc)
        payload["hpo/best_w_gtm"] = float(bhp.w_gtm)
        payload["hpo/best_w_gtg"] = float(bhp.w_gtg)
        payload["hpo/best_lr"] = float(bhp.lr)
        payload["hpo/best_qformer_layers"] = int(bhp.qformer_layers)
        payload["hpo/best_qformer_hidden"] = int(bhp.qformer_hidden)
        payload["hpo/best_gin_layers"] = int(bhp.gin_layers)
        payload["hpo/best_gin_hidden"] = int(bhp.gin_hidden)
        payload["hpo/best_proj_dim"] = int(bhp.proj_dim)

    if mode != "disabled":
        payload.update(
            _hpo_plots_wandb_html_payload(wandb, 12_000_000, space, results)
        )

    wandb.log(payload)
    _log_wandb_hpo_artifact(run, root_out, group, mode)
    if mode != "disabled" and (root_out / "hpo_plots").is_dir():
        n_html = len(list((root_out / "hpo_plots").glob("*.html")))
        if n_html:
            print(
                "W&B: Optuna/Plotly HTML is on this hpo_summary run under hpo_plots/* "
                "(Media / workspace panels) and in the Artifacts tab."
            )
    wandb.finish()


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def _wandb_env_str(name: str) -> str | None:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return None
    return str(v).strip()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel search for Stage 1 loss weights and model/lr hparams "
        "(see HparamSpace defaults: lr, Q-Former, GIN, proj-dim).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-trials", type=int, default=16)
    p.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated physical GPU indices for workers.",
    )
    p.add_argument(
        "--sampler",
        choices=("sobol", "random", "optuna", "ax"),
        default="sobol",
        help="sobol/random: built-in; optuna: TPE or random (install optuna); "
        "ax: Service API / GPEI (install ax-platform).",
    )
    p.add_argument(
        "--optuna-sampler",
        choices=("tpe", "random"),
        default="tpe",
        help="Used when --sampler optuna.",
    )
    p.add_argument(
        "--optuna-storage",
        type=str,
        default=None,
        help="Optuna RDB URL (default: sqlite under --output-dir).",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default=_wandb_env_str("WANDB_PROJECT"),
        help="If set, each training subprocess logs to this W&B project and a "
        "summary table run is created in the same group. Default: WANDB_PROJECT env.",
    )
    p.add_argument(
        "--wandb-entity",
        type=str,
        default=_wandb_env_str("WANDB_ENTITY"),
        help="W&B entity (team). Default: WANDB_ENTITY env if set.",
    )
    p.add_argument(
        "--wandb-group",
        type=str,
        default=_wandb_env_str("WANDB_GROUP"),
        help="W&B group for all trials (default: auto from sampler + seed + UTC time "
        "when unset and WANDB_GROUP is unset).",
    )
    p.add_argument(
        "--wandb-tags",
        type=str,
        default=_wandb_env_str("WANDB_TAGS") or "",
        help="Comma-separated tags added to every run (including summary). "
        "Default: WANDB_TAGS env if set.",
    )
    p.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
        help="W&B mode for subprocesses and the HPO summary run.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--w-gtc-range",
        type=float,
        nargs=2,
        default=(0.1, 2.0),
        metavar=("LOW", "HIGH"),
        help="Bounds for w_gtc. Values are drawn on a linear grid with step 0.01 in [LOW, HIGH].",
    )
    p.add_argument(
        "--w-gtm-range",
        type=float,
        nargs=2,
        default=(0.1, 2.0),
        metavar=("LOW", "HIGH"),
        help="Bounds for w_gtm (grid step 0.01).",
    )
    p.add_argument(
        "--w-gtg-range",
        type=float,
        nargs=2,
        default=(0.1, 2.0),
        metavar=("LOW", "HIGH"),
        help="Bounds for w_gtg (grid step 0.01).",
    )
    p.add_argument(
        "--lr-search",
        type=float,
        nargs=3,
        default=(1e-5, 5e-3, 1e-5),
        metavar=("LOW", "HIGH", "STEP"),
        help="Linear grid for --lr: inclusive [LOW, HIGH] in steps of STEP (default: "
        "1e-5 … 5e-3, step 1e-5).",
    )
    p.add_argument(
        "--qformer-layers-search",
        type=int,
        nargs=2,
        default=(3, 6),
        metavar=("LOW", "HIGH"),
        help="Integer range (inclusive) for --qformer-layers.",
    )
    p.add_argument(
        "--qformer-hidden-search",
        type=int,
        nargs=3,
        default=(128, 512, 32),
        metavar=("LOW", "HIGH", "STEP"),
        help="Inclusive grid for --qformer-hidden.",
    )
    p.add_argument(
        "--gin-layers-search",
        type=int,
        nargs=2,
        default=(3, 6),
        metavar=("LOW", "HIGH"),
        help="Integer range (inclusive) for --gin-layers.",
    )
    p.add_argument(
        "--gin-hidden-search",
        type=int,
        nargs=3,
        default=(128, 512, 32),
        metavar=("LOW", "HIGH", "STEP"),
        help="Inclusive grid for --gin-hidden.",
    )
    p.add_argument(
        "--proj-dim-search",
        type=int,
        nargs=3,
        default=(128, 512, 32),
        metavar=("LOW", "HIGH", "STEP"),
        help="Inclusive grid for --proj-dim.",
    )
    p.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Pass-through to train_stage1; start with -- after search flags.",
    )
    args = p.parse_args()
    if args.train_args and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]
    return args


def _hparam_space_from_args(a: argparse.Namespace) -> HparamSpace:
    lr0, lr1, lr_s = a.lr_search
    qh = a.qformer_hidden_search
    gi = a.gin_hidden_search
    pd_ = a.proj_dim_search
    return HparamSpace(
        w_gtc=(float(a.w_gtc_range[0]), float(a.w_gtc_range[1])),
        w_gtm=(float(a.w_gtm_range[0]), float(a.w_gtm_range[1])),
        w_gtg=(float(a.w_gtg_range[0]), float(a.w_gtg_range[1])),
        lr=(float(lr0), float(lr1), float(lr_s)),
        qformer_layers=(int(a.qformer_layers_search[0]), int(a.qformer_layers_search[1])),
        qformer_hidden=(int(qh[0]), int(qh[1]), int(qh[2])),
        gin_layers=(int(a.gin_layers_search[0]), int(a.gin_layers_search[1])),
        gin_hidden=(int(gi[0]), int(gi[1]), int(gi[2])),
        proj_dim=(int(pd_[0]), int(pd_[1]), int(pd_[2])),
    )


def _print_summary(results: list[TrialResult]) -> None:
    ok = [r for r in results if r.returncode == 0 and r.mean_weighted is not None]
    failed = [r for r in results if r.returncode != 0]

    print("-" * 72)
    print("Per-trial summary:")
    for r in sorted(results, key=lambda x: x.trial_id):
        st = "ok" if r.returncode == 0 else f"exit {r.returncode}"
        mw = f"{r.mean_weighted:.6f}" if r.mean_weighted is not None else "n/a"
        h = r.hparams
        print(
            f"  trial {r.trial_id:4d}  gpu {r.gpu}  {st:12s}  "
            f"mean_w={mw}  "
            f"w=({h.w_gtc:.4g},{h.w_gtm:.4g},{h.w_gtg:.4g})  "
            f"lr={h.lr:g}  QF(L,H)=({h.qformer_layers},{h.qformer_hidden})  "
            f"Gin(L,H)=({h.gin_layers},{h.gin_hidden})  "
            f"proj={h.proj_dim}"
        )
    if failed:
        print()
        print(
            f"{len(failed)} trial(s) failed. Last stderr chunk from trial "
            f"{failed[-1].trial_id}:"
        )
        print(failed[-1].stderr_tail)

    print()
    print("=" * 72)
    if ok:
        best = min(ok, key=lambda r: r.mean_weighted)  # type: ignore[arg-type,type-var]
        b = best.hparams
        print("Best configuration (lowest mean weighted loss in window):")
        print(
            f"  trial_id={best.trial_id}  "
            f"w=({b.w_gtc:g},{b.w_gtm:g},{b.w_gtg:g})  "
            f"lr={b.lr:g}  "
            f"--qformer-layers {b.qformer_layers} --qformer-hidden {b.qformer_hidden}  "
            f"--gin-layers {b.gin_layers} --gin-hidden {b.gin_hidden} --proj-dim {b.proj_dim}"
        )
        print(f"  mean weighted loss: {best.mean_weighted:.6f}")
        print(f"  metrics: {best.metrics_path}")
        print()
        print("Suggested train_stage1 flags:")
        print(
            f"  --w-gtc {b.w_gtc:g} --w-gtm {b.w_gtm:g} --w-gtg {b.w_gtg:g}  "
            f"--lr {b.lr:g} --qformer-layers {b.qformer_layers} "
            f"--qformer-hidden {b.qformer_hidden} --gin-layers {b.gin_layers} "
            f"--gin-hidden {b.gin_hidden} --proj-dim {b.proj_dim}"
        )
    else:
        print("No successful trials with metrics; inspect stderr and metrics.json paths.")
    print("=" * 72)


def main() -> None:
    args = _parse_args()
    train_argv = list(args.train_args)
    if not train_argv:
        print(
            "Error: no training arguments. After search options, use -- then "
            "the same flags you pass to train_stage1 (must include --output-dir).",
            file=sys.stderr,
        )
        sys.exit(2)

    root_out_s = _get_flag_value(train_argv, "--output-dir")
    if not root_out_s:
        print(
            "Error: passthrough must include --output-dir (search uses subfolders).",
            file=sys.stderr,
        )
        sys.exit(2)
    root_out = Path(root_out_s)

    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        print("Error: --gpus must list at least one GPU index.", file=sys.stderr)
        sys.exit(2)

    space = _hparam_space_from_args(args)

    print("=" * 72)
    print("Stage 1 HPO: loss weights + lr + Q-Former + GIN + proj-dim")
    print("=" * 72)
    print()
    print("Exploration space:")
    print(
        f"  w_gtc: [{space.w_gtc[0]:g}, {space.w_gtc[1]:g}]  "
        f"w_gtm: [{space.w_gtm[0]:g}, {space.w_gtm[1]:g}]  "
        f"w_gtg: [{space.w_gtg[0]:g}, {space.w_gtg[1]:g}] (step {W_LOSS_WEIGHT_STEP:g} per weight)"
    )
    print(
        f"  lr: grid [{space.lr[0]:g}, {space.lr[1]:g}] step {space.lr[2]:g}"
    )
    print(
        f"  qformer_layers: {space.qformer_layers[0]}..{space.qformer_layers[1]}  "
        f"qformer_hidden: {list(space.qformer_hidden)}  "
        f"gin_layers: {space.gin_layers[0]}..{space.gin_layers[1]}"
    )
    print(
        f"  gin_hidden: {list(space.gin_hidden)}  "
        f"proj_dim: {list(space.proj_dim)}"
    )
    print(
        f"Sampler: {args.sampler}  |  trials: {args.num_trials}  |  workers: {len(gpus)}"
    )
    if args.sampler == "optuna":
        print(
            f"  Optuna: inner sampler={args.optuna_sampler}, "
            f"storage={args.optuna_storage or '(default sqlite under output-dir)'}"
        )
    print()
    print(
        "Objective: minimize mean weighted loss "
        "(w_gtc*L_gtc + w_gtm*L_gtm + w_gtg*L_gtg) over the last "
        "`--metrics-window` log steps (see train_stage1 --metrics-window)."
    )
    print()
    print(
        "Sampler guide: use sobol/random for dependency-free screening; optuna (TPE) "
        "for adaptive single-objective search with parallel n_jobs=GPUs; ax for "
        "Sobol→Bayesian (GPEI) with batched parallel trials (install ax-platform)."
    )
    print()
    print(f"GPU pool (CUDA_VISIBLE_DEVICES per worker): {gpus}")
    print(f"Run root: {root_out.resolve()}")
    print()

    wandb_ctx: WandbHpoContext | None = None
    wb_group_for_log: str | None = None
    if args.wandb_project:
        wb_group_for_log = args.wandb_group or (
            f"stage1_hpo_{args.sampler}_seed{args.seed}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        run_prefix = {
            "sobol": "trial_sobol",
            "random": "trial_random",
            "optuna": "trial_optuna",
            "ax": "trial_ax",
        }.get(args.sampler, "trial")
        wandb_ctx = WandbHpoContext(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=wb_group_for_log,
            tags=args.wandb_tags,
            mode=args.wandb_mode,
            run_name_prefix=run_prefix,
        )
        print(
            f"W&B: project={args.wandb_project}  group={wb_group_for_log}  "
            f"mode={args.wandb_mode}"
        )
        print()

    filtered = _strip_conflicting_train_flags(train_argv)
    root_out.mkdir(parents=True, exist_ok=True)

    results: list[TrialResult]

    if args.sampler in ("sobol", "random"):
        if args.sampler == "sobol":
            trials = sample_trials_sobol(args.num_trials, space, args.seed)
        else:
            trials = sample_trials_random(args.num_trials, space, args.seed)

        print("Trial list (abbrev.; full hparams in each trial dir / logs):")
        for i, t in enumerate(trials):
            print(
                f"  {i:4d}  w=({t.w_gtc:.4g},{t.w_gtm:.4g},{t.w_gtg:.4g})  "
                f"lr={t.lr:g}  QF=({t.qformer_layers},{t.qformer_hidden})  "
                f"G=({t.gin_layers},{t.gin_hidden})  proj={t.proj_dim}"
            )
        print()

        work: list[
            tuple[int, SearchTrial, int, Path, Path, tuple[str, ...]]
        ] = []
        for i, hp in enumerate(trials):
            gpu_slot = gpus[i % len(gpus)]
            trial_dir = root_out / f"trial_{i:04d}"
            metrics_path = trial_dir / "metrics.json"
            trial_dir.mkdir(parents=True, exist_ok=True)
            extra = tuple(wandb_ctx.argv_for_trial(i)) if wandb_ctx else ()
            work.append((i, hp, gpu_slot, trial_dir, metrics_path, extra))

        raw = _run_parallel_subprocesses(work, filtered, max_workers=len(gpus))
        results = sorted(raw, key=lambda r: r.trial_id)

    elif args.sampler == "optuna":
        results, _ = _run_optuna(
            num_trials=args.num_trials,
            seed=args.seed,
            space=space,
            gpus=gpus,
            root_out=root_out,
            filtered=filtered,
            optuna_sampler=args.optuna_sampler,
            optuna_storage=args.optuna_storage,
            wandb_ctx=wandb_ctx,
        )
        print("Optuna finished; per-trial metrics under trial_optuna_*/metrics.json")
        print()

    elif args.sampler == "ax":
        results = _run_ax(
            num_trials=args.num_trials,
            seed=args.seed,
            space=space,
            gpus=gpus,
            root_out=root_out,
            filtered=filtered,
            wandb_ctx=wandb_ctx,
        )
        print("Ax finished; per-trial metrics under trial_ax_*/metrics.json")
        print()

    else:
        raise AssertionError(args.sampler)

    _write_hpo_trial_results_json(root_out, results)
    print("HPO visualizations (Optuna / Plotly HTML):")
    _save_hpo_plots(space, results, root_out)

    if args.wandb_project and wb_group_for_log:
        _log_wandb_hpo_summary(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=wb_group_for_log,
            tags=args.wandb_tags,
            mode=args.wandb_mode,
            sampler=args.sampler,
            num_trials=args.num_trials,
            space=space,
            results=results,
            root_out=root_out,
        )
    _print_summary(results)


if __name__ == "__main__":
    main()
