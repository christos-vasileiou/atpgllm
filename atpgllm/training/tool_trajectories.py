"""Failed first attempts for retry SFT trajectories.

A retry trajectory shows the model a pattern that does not detect the fault,
the simulator's real answer for it, a diagnosis read from that answer, and the
corrected pattern. The failed pattern is the dataset's detecting pattern with
one to three primary inputs flipped, preferring the inputs the backtrace
names (they decide activation and propagation). Every candidate is simulated
with the same simulator the tool uses, and only a confirmed miss is kept.
"""

from __future__ import annotations

import collections
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from atpgllm.training._paths import (
    ensure_data_preprocessing_on_path,
    resolve_sim_config_path,
)
from atpgllm.training.tool_feedback import SimulationReading, read_snapshot

ensure_data_preprocessing_on_path()

from fault_sim import fast_fault_sim, prepare_netlist  # noqa: E402


@dataclass(frozen=True)
class FailedAttempt:
    input_vector: Dict[str, Any]
    output_vector: Dict[str, Any]   # Good Machine values at the primary outputs
    snapshot: str                   # what the tool returns for input_vector
    reading: SimulationReading
    changes: List[Tuple[str, Any]]  # (input, detecting value) that fix it


_MODELS: "collections.OrderedDict[str, Any]" = collections.OrderedDict()
_GATES: Optional[Dict[str, Any]] = None


def _netlist_text(record: Dict[str, Any]) -> str:
    netlist = record.get("netlist")
    if isinstance(netlist, dict):
        netlist = netlist.get("netlist")
    return netlist or ""


def _simulator_model(netlist: str):
    global _GATES
    from atpgllm.training.reward_function_factory import RewardFunctionFactory as RF

    if _GATES is None:
        with resolve_sim_config_path().open(encoding="utf-8") as fh:
            _GATES = json.load(fh)
    key = hashlib.sha256(netlist.encode("utf-8")).hexdigest()
    model = _MODELS.get(key)
    if model is None:
        model = prepare_netlist(netlist, _GATES, RF.DECL_RE, RF.NAME_RE)
        _MODELS[key] = model
        if len(_MODELS) > 256:
            _MODELS.popitem(last=False)
    else:
        _MODELS.move_to_end(key)
    return model


def simulate(record: Dict[str, Any], input_vector: Dict[str, Any]) -> str:
    """The tool's answer for ``input_vector`` on this record's fault."""
    model = _simulator_model(_netlist_text(record))
    expected = json.loads(record["expected_output"])
    snapshot = fast_fault_sim(input_vector, expected, record["fault"], model, _GATES)
    if "error" in snapshot.columns:
        raise ValueError(str(snapshot.loc[0, "error"]))
    return snapshot[["Good Machine", "Bad Machine"]].to_json()


def record_seed(record: Dict[str, Any], salt: str = "") -> int:
    """A stable seed per record, so trajectories are reproducible."""
    key = "|".join(
        str(record.get(name, ""))
        for name in ("circuit_id", "fault", "pattern_index", "input_vector")
    )
    return int(hashlib.sha256((salt + key).encode("utf-8")).hexdigest()[:12], 16)


def failed_attempt(
    record: Dict[str, Any],
    rng: random.Random,
    *,
    max_flips: int = 3,
    tries: int = 24,
) -> Optional[FailedAttempt]:
    """A confirmed non-detecting pattern near the record's detecting one, or
    None if ``tries`` candidates all still detect the fault."""
    golden = json.loads(record["input_vector"])
    outputs = list(json.loads(record["expected_output"]))
    inputs = [net for net in golden if str(golden[net]) in ("0", "1")]
    if not inputs or not outputs:
        return None
    backtrace = {
        net.strip()
        for net in str(record.get("backtrack_nets") or "").split(",")
        if net.strip()
    }
    controlling = [net for net in inputs if net in backtrace]
    # Random flips mostly break activation. Aim for each kind of miss half
    # the time, so the model learns to read both from the snapshot.
    want_activated = rng.random() < 0.5
    fallback: Optional[FailedAttempt] = None
    for _ in range(tries):
        # Activation misses are common, so only a search for a propagation
        # miss is worth the remaining tries.
        if fallback is not None and not want_activated:
            break
        pool = controlling if controlling and rng.random() < 0.8 else inputs
        count = rng.choice([1, 1, 2, 3][: max_flips + 1])
        flips = rng.sample(pool, min(len(pool), count))
        vector = dict(golden)
        for net in flips:
            vector[net] = 1 - int(vector[net])
        try:
            snapshot = simulate(record, vector)
        except (ValueError, KeyError, TypeError):
            return fallback
        reading = read_snapshot(snapshot, record["fault"], outputs)
        if reading.detected:
            continue
        good = json.loads(snapshot)["Good Machine"]
        attempt = FailedAttempt(
            input_vector=vector,
            output_vector={net: good[net] for net in outputs if net in good},
            snapshot=snapshot,
            reading=reading,
            changes=[(net, golden[net]) for net in inputs if net in flips],
        )
        if reading.activated == want_activated:
            return attempt
        fallback = fallback or attempt
    return fallback
