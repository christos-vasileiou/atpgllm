import sys

# Add data_preprocessing to sys.path for shared utilities
from atpgllm.training._paths import ensure_data_preprocessing_on_path
ensure_data_preprocessing_on_path()

import pandas as pd
import regex as re
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from .fault_coverage_calc import fault_sim, logic_and, logic_buf, logic_not, logic_nand, logic_nor, logic_or, logic_xor, logic_xnor
from ..utils import is_main_process
import torch.distributed as dist
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import warnings
import os
import math

from fault_sim import convert_string_to_dict, _use_tetramax_backend as using_native_simulator

warnings.filterwarnings("ignore")

# Keys ending with this suffix are logged to dashboards but excluded from the GRPO scalar
# (see ``train_scalar_from_reward_components`` and DualAdapterGRPOTrainer reward parsing).
REWARD_LOGONLY_SUFFIX = "_logonly"

# GDPO priorities are applied after each objective is normalized. Raw reward
# magnitudes therefore do not encode cross-objective importance.
ATPG_GDPO_OBJECTIVE_KEYS = ("detection", "activation", "fidelity", "format", "uniqueness")
ATPG_GDPO_OBJECTIVE_WEIGHTS = (1.0, 0.25, 0.20, 0.05, 0.10)


def active_reward_objectives():
    """A fixed per-run profile; unavailable internals are not negative rewards."""
    if not using_native_simulator():
        return ATPG_GDPO_OBJECTIVE_KEYS, ATPG_GDPO_OBJECTIVE_WEIGHTS
    profile = os.environ.get('TMAX_REWARD_PROFILE', 'po')
    profiles = {'detection': (('detection', 'format', 'uniqueness'), (1.0, 0.05, 0.10)),
                'po': (('detection', 'fidelity', 'format', 'uniqueness'), (1.0, 0.20, 0.05, 0.10)),
                'full': (ATPG_GDPO_OBJECTIVE_KEYS, ATPG_GDPO_OBJECTIVE_WEIGHTS)}
    if profile not in profiles:
        raise ValueError(f'Unknown TMAX_REWARD_PROFILE: {profile}')
    return profiles[profile]


def train_scalar_from_reward_components(components: Dict[str, float]) -> float:
    """Weighted scalar fallback for callers that cannot consume GDPO objectives."""
    objective_weights = dict(
        zip(ATPG_GDPO_OBJECTIVE_KEYS, ATPG_GDPO_OBJECTIVE_WEIGHTS)
    )
    total = 0.0
    for key, value in components.items():
        if str(key).endswith(REWARD_LOGONLY_SUFFIX):
            continue
        if math.isfinite(float(value)):
            total += float(value) * objective_weights.get(key, 1.0)
    return float(total)


def extract_json_tool_response_and_convert_to_df(text: str) -> Optional[pd.DataFrame]:
    """Parse first ``<tool_response>{...}</tool_response>`` blob into a DataFrame, or None."""
    match = re.search(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", text, re.DOTALL)
    if not match:
        return None
    blob = match.group(1)
    parsed: Any = None
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        try:
            import ast
            parsed = ast.literal_eval(blob)
        except (SyntaxError, ValueError, TypeError):
            return None
    try:
        return pd.DataFrame.from_dict(parsed)
    except (ValueError, TypeError):
        # e.g. nested dict layout pandas rejects ("Mixing dicts with non-Series...")
        return None


def extract_markdown_table(text: str) -> str:
    """
    Extracts the first Markdown table found in a text block.
    Returns the table as a string.
    """
    pattern = re.compile(
        r"(\|.*\|\n\|[-| ]+\|\n(?:\|.*\|\n?)*)",
        re.MULTILINE
    )

    match = pattern.search(text)
    if not match:
        raise ValueError("No markdown table found in text")

    return match.group(1)


def markdown_table_to_dataframe(table_str: str) -> pd.DataFrame:
    lines = [
        line.strip()
        for line in table_str.strip().splitlines()
        if line.strip().startswith("|") and not set(line.strip()) <= {"|", "-", " "}
    ]

    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in lines
    ]

    header = rows[0]
    data = rows[1:]

    df = pd.DataFrame(data, columns=header)
    df.set_index(df.columns[0], inplace=True)
    df = df.apply(pd.to_numeric, errors="ignore")

    return df


def convert_to_df(pred_simulation):
    df = pd.read_csv(StringIO(pred_simulation), sep="\s{2,}", header=None, skiprows=1)
    df.columns = ['ID', 'Good Machine', 'Bad Machine']
    df = df.set_index('ID')
    df.index.name = None
    return df


def _cell_to_int(val: Any) -> Optional[int]:
    if val == "x" or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _simulation_table_from_completion(
    completion: str, simulation_fn: Callable[[str], List[str]]
) -> Optional[pd.DataFrame]:
    """Prefer JSON <tool_response> (Good/Bad machine dicts); else legacy markdown / string table."""
    df = extract_json_tool_response_and_convert_to_df(completion)
    if df is not None and not df.empty:
        if "Good Machine" in df.columns and "Bad Machine" in df.columns:
            return df
    rows = simulation_fn(completion)
    if not rows:
        return None
    try:
        return convert_to_df(rows[0])
    except Exception:
        return None


def _fault_detected_at_pos(simulation_df: pd.DataFrame) -> bool:
    """True if some primary output differs between good and bad machine (observable detection)."""
    if simulation_df is None or simulation_df.empty or "POs" not in simulation_df.columns:
        return False
    po_rows = simulation_df.loc[simulation_df["POs"]]
    if po_rows.empty:
        return False
    for _, row in po_rows.iterrows():
        g = _cell_to_int(row["Good Machine"])
        b = _cell_to_int(row["Bad Machine"])
        if g is None or b is None:
            continue
        if g != b:
            return True
    return False


_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"
_TOOL_RESPONSE_BLOCK = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
# Checks on the turn that ends a trajectory; a missing final turn fails all.
_FINAL_TURN_CHECKS = 7


class Trajectory:
    """A completion as model turns and the tool calls that ended them.

    ``turns[k]`` is the text the model wrote in turn ``k``; ``calls[k]`` is the
    call executed after it, ``{"arguments": ..., "result": <tool text>}``. A
    trajectory that ends on the model's answer has one more turn than calls.
    Trainers pass the executed calls they recorded (``tool_trace``); for other
    callers the completion text is split at its ``<tool_response>`` blocks.
    """

    def __init__(self, turns: List[str], calls: List[Dict[str, Any]]):
        self.turns = [turn or "" for turn in turns]
        self.calls = list(calls)

    @property
    def final_turn(self) -> Optional[str]:
        return self.turns[len(self.calls)] if len(self.turns) > len(self.calls) else None

    @classmethod
    def from_trace(cls, trace: Dict[str, Any]) -> "Trajectory":
        return cls(list(trace.get("turns") or []), list(trace.get("calls") or []))

    @classmethod
    def from_text(cls, completion: str) -> "Trajectory":
        from atpgllm.training.revert_template import parse_tool_call
        turns, calls, start = [], [], 0
        for match in _TOOL_RESPONSE_BLOCK.finditer(completion):
            turn = completion[start:match.start()]
            call = parse_tool_call(turn)
            turns.append(turn)
            calls.append({"arguments": call.get("arguments") if call else None,
                          "result": match.group(1).strip()})
            start = match.end()
        turns.append(completion[start:])
        return cls(turns, calls)


def prompt_opens_thinking(prompt: Any) -> bool:
    """Whether the chat template opens ``<think>`` for each assistant turn.

    Granite 4.2's generation prompt ends with ``<think>\\n``, so the opening tag
    belongs to the prompt (and, after a tool answer, to the inserted
    continuation prefix) and the model writes only the reasoning and
    ``</think>``.
    """
    return isinstance(prompt, str) and prompt.rstrip().endswith(_THINK_OPEN)


def _reasoned(turn: str, opened_by_template: bool) -> bool:
    """The model closed a thinking block holding non-empty reasoning."""
    close = turn.find(_THINK_CLOSE)
    if close < 0:
        return False
    head = turn[:close]
    opening = head.rfind(_THINK_OPEN)
    if opening >= 0:
        reasoning = head[opening + len(_THINK_OPEN):]
    elif opened_by_template:
        reasoning = head
    else:
        return False
    return bool(reasoning.strip())


def _thinking_closed(turn: str, opened_by_template: bool) -> bool:
    """A thinking block the turn has open (its own or the template's) is closed."""
    opening = turn.rfind(_THINK_OPEN)
    if opening >= 0:
        return _THINK_CLOSE in turn[opening:]
    return _THINK_CLOSE in turn if opened_by_template else True


def vector_key(vector: Any) -> Optional[Tuple[Tuple[str, str], ...]]:
    """Order-insensitive identity of a net→value vector (``a=1, b:0`` or a dict)."""
    if isinstance(vector, dict):
        items = vector.items()
    elif isinstance(vector, str):
        sep = ":" if ":" in vector else "="
        items = []
        for part in vector.split(","):
            if sep not in part:
                return None
            net, value = part.split(sep, 1)
            items.append((net, value))
    else:
        return None
    key = tuple(sorted((str(net).strip(), str(value).strip().lower()) for net, value in items))
    return key or None


def call_identity(call: Dict[str, Any]) -> Any:
    """The input vector a tool call submitted, so repeated vectors compare equal.

    Two calls are the same vector when they assign the same values to the same
    nets, in any order. A vector that cannot be read as such an assignment is
    compared by its text, and a call without a readable vector by its arguments.
    """
    arguments = call.get("arguments") if isinstance(call, dict) else None
    vector = arguments.get("input_vector") if isinstance(arguments, dict) else None
    if vector is None:
        return ("unparsed", json.dumps(arguments, sort_keys=True, default=str))
    return vector_key(vector) or ("unparsed", str(vector))


_TOOL_ARGUMENTS = {"input_vector", "output_vector", "fault", "doc_id"}


def _executable(call: Dict[str, Any], doc_id: Optional[str]) -> bool:
    """Whether the tool would run a call the environment never executed: the
    tool's argument set and, when the prompt carries the circuit, its doc_id."""
    arguments = call.get("arguments") if isinstance(call, dict) else None
    if not isinstance(arguments, dict) or set(arguments) != _TOOL_ARGUMENTS:
        return False
    return doc_id is None or str(arguments["doc_id"]) == str(doc_id)


def _tool_returned_table(result: Any) -> bool:
    """The tool simulated the call (error strings and truncated JSON did not)."""
    if not isinstance(result, str):
        return False
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(data, dict) and "Good Machine" in data and "Bad Machine" in data


def _pi_agreement(
    pred_input: Optional[str], reference: Any, inputs: List[str]
) -> float:
    """Fraction of primary inputs the answer states as the reference call did.

    The answer gives only care bits: an input the reference call set to 0/1
    must carry the same value, and an input it left as don't-care (``x``) must
    be omitted, since a missing input is read as ``x``. Writing ``x`` out
    disagrees. 0 without an INPUT_VECTOR.
    """
    if pred_input is None or not inputs:
        return 0.0
    pred, ref = dict(vector_key(pred_input) or ()), dict(vector_key(reference) or ())
    if not pred and any(ref.get(pi, "x") != "x" for pi in inputs):
        return 0.0

    def agrees(pi: str) -> bool:
        if ref.get(pi, "x") == "x":
            return pi not in pred
        return pred.get(pi) == ref[pi]

    return sum(agrees(pi) for pi in inputs) / len(inputs)


def care_bits(key: Optional[Tuple[Tuple[str, str], ...]]) -> Optional[Tuple[Tuple[str, str], ...]]:
    """A vector key without its don't-care (``x``) nets: what a final answer states."""
    if key is None:
        return None
    return tuple(item for item in key if item[1] != "x")


def _answer_has_dont_care(*vectors: Optional[str]) -> bool:
    """Whether a final-answer vector writes ``x`` instead of omitting the net."""
    return any(value == "x" for vector in vectors for _, value in (vector_key(vector) or ()))


def _po_agreement(
    pred_output: Optional[str], good: Dict[str, Any], outputs: List[str]
) -> float:
    """Fraction of resolved primary outputs EXPECTED_OUTPUT gives as ``good``.

    Outputs the simulation left unknown (TetraMAX ``x``) are not scored; NaN
    means none was known.
    """
    known = {po: _cell_to_int(good.get(po)) for po in outputs}
    known = {po: value for po, value in known.items() if value is not None}
    if not known:
        return float("nan")
    pred = dict(vector_key(pred_output) or ())
    return sum(pred.get(po) == str(value) for po, value in known.items()) / len(known)


def _tool_calls_in(text: Optional[str]) -> List[Dict[str, Any]]:
    """Every tool call written in one model turn, in order."""
    from atpgllm.training.revert_template import _tool_call_from_inner
    calls = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text or "", re.DOTALL):
        call = _tool_call_from_inner(match.group(1))
        calls.append(call if call is not None else {"arguments": None})
    return calls


def _outcome(good: Dict[str, Any], bad: Dict[str, Any], outputs: List[str],
             net: str, stuck_at: int, detected: Optional[bool] = None) -> Dict[str, Any]:
    """Detection and fault-site activation read from Good/Bad machine values."""
    if detected is None:
        detected = False
        for po in outputs:
            g, b = _cell_to_int(good.get(po)), _cell_to_int(bad.get(po))
            if g is not None and b is not None and g != b:
                detected = True
                break
    activated = float("nan")
    if net in good:
        g, b = _cell_to_int(good.get(net)), _cell_to_int(bad.get(net))
        activated = float(g is not None and b is not None and b == stuck_at and g != b)
    return {"good": good, "bad": bad, "detected": bool(detected), "activated": activated}


def _outcome_from_result(result: Any, outputs: List[str], net: str,
                         stuck_at: int) -> Optional[Dict[str, Any]]:
    """What an executed call showed the model; None when the tool did not simulate."""
    if not _tool_returned_table(result):
        return None
    data = json.loads(result)
    return _outcome(dict(data["Good Machine"]), dict(data["Bad Machine"]), outputs, net, stuck_at)


class _SimulatorFailure(Exception):
    """Infrastructure failure: the row is flagged and every rank aborts."""


def _mentions_target_fault(text: str, fault: str, fault_net: str) -> float:
    if not text:
        return 0.0
    t = re.sub(r"\s+", " ", text.lower())
    needle = f"{fault.lower()} {fault_net.lower()}"
    if needle in t:
        return 1.0
    if f"{fault.lower()}{fault_net.lower()}" in t.replace(" ", ""):
        return 1.0
    return 0.0


def test_generation_grpo_reward(prompts: list, completions: list, **kwargs) -> List[Dict[str, float]]:
    """
    ATPG reward with five explicit GDPO objectives over a tool-call trajectory.

    Only tool calls are scored for detection, and the final answer is not
    simulated. Calls the environment executed before the model's last
    generation that called the tool are scored from the result the tool
    returned. The calls of that last generation (several when the model lists
    several) are simulated again, once per distinct vector. An *attempt* is
    a call, repeats included, in order of issue up to the first that detects
    the fault.

    ``detection``: 1 / (number of attempts) when an attempt detects the fault,
    else 0, and 0 without a tool call. Calling again after a success cannot
    raise it, and every repeated vector before the success lowers it.
    ``activation``: mean fault-site activation over the attempts.
    ``fidelity``: agreement of the final INPUT_VECTOR and EXPECTED_OUTPUT with
    the last call that detected the fault (its input vector and good machine
    outputs), 0 when no call detected it.
    ``format``: the protocol. Think, then call. After a negative result think
    and call again with a new vector. After a positive result answer with all
    three fields and no further call.
    ``uniqueness``: distinct vectors among the attempts divided by the number
    of attempts, 0 without a tool call.
    ``*_logonly`` keys are diagnostics only.
    ``fault_detected_by_pred_input_vector_acc_logonly`` is 1 when the answer's
    vector is one a call showed to detect the fault.

    ``tool_trace`` (optional, one per completion) holds the turns and executed
    calls a trainer recorded (see :class:`Trajectory`); without it the
    completion text is split at its ``<tool_response>`` blocks.
    """
    netlists = kwargs.get("netlists", None)
    if netlists is None:
        raise ValueError("netlists must be provided")
    if (using_native_simulator() and len(prompts) > 1
        and not kwargs.get('_serial_native_reward')):
        from concurrent.futures import ThreadPoolExecutor
        def score_one(index):
            options = dict(kwargs, netlists=[netlists[index]], _serial_native_reward=True)
            for field in ('fault', 'module_name', 'tool_trace'):
                value = kwargs.get(field)
                if value is not None and not isinstance(value, str):
                    options[field] = [value[index]]
            return test_generation_grpo_reward([prompts[index]], [completions[index]], **options)[0]
        with ThreadPoolExecutor(max_workers=min(16, len(prompts))) as executor:
            return list(executor.map(score_one, range(len(prompts))))

    fault_fn = kwargs.get("fault_fn", None)
    simulation_fn = kwargs.get("simulation_fn", None)
    input_vector_fn = kwargs.get("input_vector_fn", None)
    expected_output_fn = kwargs.get("expected_output_fn", None)
    detected_faults_fn = kwargs.get("detected_faults_fn", None)
    if fault_fn is None or simulation_fn is None or input_vector_fn is None:
        raise ValueError("fault_fn, simulation_fn, and input_vector_fn must be provided")
    if expected_output_fn is None or detected_faults_fn is None:
        raise ValueError("expected_output_fn and detected_faults_fn must be provided")

    lib_gate_funcs = kwargs.get("lib_gate_funcs", None)

    if using_native_simulator():
        from fault_sim import resolve_fault_sim_runner
        gate_func = lib_gate_funcs
        fault_sim_runner = kwargs.get('fault_sim') or resolve_fault_sim_runner()
    elif lib_gate_funcs is None:
        gate_func = {
        "IB": logic_buf,
        "AN": logic_and,
        "OR": logic_or,
        "XO": logic_xor,
        "IV": logic_not,
        "ND": logic_nand,
        "NR": logic_nor,
        "XN": logic_xnor,
        }
        fault_sim_runner = fault_sim
    else:
        gate_func = lib_gate_funcs
        fault_sim_runner = kwargs.get("fault_sim", None)
        if fault_sim_runner is None:
            raise ValueError("fault_sim function must be provided when using lib_gate_funcs")

    rewards: List[Dict[str, float]] = []
    module_names = kwargs.get("module_name") or []
    dataset_faults = kwargs.get("fault")
    tool_traces = kwargs.get("tool_trace")

    for idx, (prompt, completion, netlist) in enumerate(zip(prompts, completions, netlists)):
        out: Dict[str, float] = {
            "detection": 0.0,
            "activation": 0.0,
            "fidelity": 0.0,
            "format": 0.0,
            "uniqueness": 0.0,
            "format_compliance_logonly": 0.0,
            "pi_completeness_logonly": 0.0,
            "tool_response_fidelity_logonly": 0.0,
            "fault_mention_logonly": 0.0,
            "sim_table_present_logonly": 0.0,
            "pred_vs_fault_sim_acc_logonly": 0.0,
            "fault_detected_by_pred_input_vector_acc_logonly": 0.0,
            "fault_site_activated_acc_logonly": 0.0,
            "expected_output_acc_logonly": 0.0,
            "input_vector_acc_logonly": 0.0,
            "detected_faults_acc_logonly": 0.0,
            "simulation_valid_logonly": 0.0,
            "simulator_error_logonly": 0.0,
            "tool_calls_logonly": 0.0,
            "simulated_calls_logonly": 0.0,
            "attempts_logonly": 0.0,
            "duplicate_calls_logonly": 0.0,
            "calls_after_detection_logonly": 0.0,
            "reasoned_call_turns_logonly": 0.0,
            "unverified_answer_logonly": 0.0,
            "answer_x_free_logonly": 0.0,
        }

        dataset_fault = None
        if isinstance(dataset_faults, str):
            dataset_fault = dataset_faults
        elif dataset_faults is not None and idx < len(dataset_faults):
            dataset_fault = dataset_faults[idx]
        fault_info = (
            fault_fn(prompt, fault=dataset_fault)
            if dataset_fault
            else fault_fn(prompt)
        )
        fault, net = (None, None)
        if fault_info:
            fault, net = fault_info[0]
        stuck_at = int(fault[-1]) if fault else -1

        trace = None
        if isinstance(tool_traces, (list, tuple)) and idx < len(tool_traces):
            trace = tool_traces[idx]
        trajectory = (
            Trajectory.from_trace(trace) if isinstance(trace, dict)
            else Trajectory.from_text(completion)
        )
        calls = trajectory.calls
        final_turn = trajectory.final_turn
        # Only the final turn answers; reports drafted in earlier turns do not.
        answer = final_turn or ""
        pred_input = (input_vector_fn(answer) or [None])[-1]
        pred_output = (expected_output_fn(answer) or [None])[-1]
        pred_faults = (detected_faults_fn(answer) or [None])[-1]

        sim_from_completion = _simulation_table_from_completion(completion, simulation_fn)
        if sim_from_completion is not None:
            out["sim_table_present_logonly"] = 1.0
        mention = _mentions_target_fault(pred_faults, fault, net) if (pred_faults and fault) else 0.0
        out["fault_mention_logonly"] = mention
        out["detected_faults_acc_logonly"] = 1.0 if mention >= 0.99 else 0.0
        out["tool_calls_logonly"] = float(len(calls))

        if not (fault and net and netlist):
            rewards.append(out)
            continue

        mod_name = module_names[idx] if idx < len(module_names) else None
        canonical_outputs = list(getattr(netlist, "output_nets", []) or [])
        canonical_inputs = list(getattr(netlist, "input_nets", []) or [])
        outputs = canonical_outputs or [po for po, _ in (vector_key(pred_output) or ())]
        # Values are placeholders only. fast_fault_sim uses these canonical names
        # to mark POs and computes their values from the circuit.
        simulation_outputs: Any = (
            {po: 0 for po in canonical_outputs} if canonical_outputs else pred_output
        )

        simulated: Dict[Any, Optional[Dict[str, Any]]] = {}

        def simulate(vector: Any) -> Optional[Dict[str, Any]]:
            """Authoritative simulation of one input vector, run once per vector.

            None when the simulator rejects the vector."""
            identity = vector_key(vector) or ("unparsed", str(vector))
            if identity not in simulated:
                simulated[identity] = run_simulation(vector)
            return simulated[identity]

        def run_simulation(vector: Any) -> Optional[Dict[str, Any]]:
            try:
                frame, sim_rewards = fault_sim_runner(
                    vector,
                    simulation_outputs,
                    f"{fault} {net}",
                    netlist,
                    gate_func,
                    module_name=mod_name,
                    return_rewards=True,
                )
            except Exception as exc:
                # Both DDP ranks abort after scoring. Preserve the original cause
                # here, before it is reduced to a numeric infrastructure flag.
                print(f"[rewards] Simulator failure for module={mod_name!r}, "
                      f"fault={fault} {net}: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc()
                raise _SimulatorFailure() from exc
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                print(f"[rewards] Simulator returned an empty or invalid result for "
                      f"module={mod_name!r}, fault={fault} {net}", file=sys.stderr, flush=True)
                raise _SimulatorFailure()
            if "error" in frame.columns:
                if sim_rewards.get("invalid_request", False):
                    return None
                print(f"[rewards] Simulator failure for module={mod_name!r}, "
                      f"fault={fault} {net}: {frame['error'].tolist()}",
                      file=sys.stderr, flush=True)
                raise _SimulatorFailure()
            tetramax = bool(sim_rewards.get("tetramax_available"))
            outcome = _outcome(
                frame["Good Machine"].to_dict(), frame["Bad Machine"].to_dict(), outputs,
                net, stuck_at,
                detected=bool(sim_rewards.get("tetramax_detected")) if tetramax else None,
            )
            if not tetramax and net not in frame.index:
                outcome["activated"] = 0.0
            outcome.update(frame=frame, rewards=sim_rewards)
            return outcome

        def vector_of(call: Dict[str, Any]) -> Any:
            arguments = call.get("arguments")
            if not isinstance(arguments, dict):
                return None
            claimed = arguments.get("fault")
            if claimed is not None and " ".join(str(claimed).split()) != f"{fault} {net}":
                return None
            return arguments.get("input_vector")

        # What each executed call showed the model (the protocol reacts to it).
        shown = [_outcome_from_result(c.get("result"), outputs, net, stuck_at)
                 if vector_of(c) is not None else None for c in calls]
        keys = [vector_key(vector_of(c)) if shown[k] is not None else None
                for k, c in enumerate(calls)]
        identities = [call_identity(c) for c in calls]

        # The model's last generation that called the tool is re-simulated; its
        # first call is the one the environment executed, if any.
        pending = _tool_calls_in(final_turn)
        if pending:
            last_calls, first_executed = pending, None
        elif calls:
            # The environment runs a turn's first call; further calls the model
            # listed in that turn are read from its text.
            listed = [calls[-1]] + _tool_calls_in(trajectory.turns[len(calls) - 1])[1:]
            last_calls, first_executed = listed, len(calls) - 1
        else:
            last_calls, first_executed = [], None
        # Calls the environment never executed must pass the tool's own checks.
        expected_doc_id = None
        if isinstance(prompt, str):
            from atpgllm.training.tools import ToolHelper
            document = ToolHelper.get_document(prompt)
            if isinstance(document, dict) and document.get("doc_id") is not None:
                expected_doc_id = document["doc_id"]
        # (identity, vector key, outcome) for every call, in order of issue.
        scored: List[Tuple[Any, Any, Optional[Dict[str, Any]]]] = [
            (identities[k], keys[k], shown[k])
            for k in range(len(calls) - (first_executed is not None))
        ]
        resimulated: List[Optional[Dict[str, Any]]] = []
        try:
            for position, call in enumerate(last_calls):
                vector = vector_of(call)
                if position == 0 and first_executed is not None:
                    vector = vector if shown[first_executed] is not None else None
                elif not _executable(call, expected_doc_id):
                    vector = None
                outcome = simulate(vector) if vector is not None else None
                resimulated.append(outcome)
                scored.append((call_identity(call),
                               vector_key(vector) if outcome is not None else None, outcome))
        except _SimulatorFailure:
            out["simulator_error_logonly"] = 1.0
            rewards.append(out)
            continue
        out["simulated_calls_logonly"] = float(len(simulated))

        # Attempts: every call in order of issue, repeats included, up to the
        # first detection. A call that did not simulate detected nothing.
        attempts: List[Optional[Dict[str, Any]]] = []
        attempted: List[Any] = []
        for identity, _, outcome in scored:
            attempts.append(outcome)
            attempted.append(identity)
            if outcome is not None and outcome["detected"]:
                break
        detected_any = bool(attempts and attempts[-1] and attempts[-1]["detected"])
        detection = 1.0 / len(attempts) if detected_any else 0.0
        uniqueness = len(set(attempted)) / len(attempted) if attempted else 0.0
        activations = [o["activated"] if o else 0.0 for o in attempts]
        activation = (
            float(np.nanmean(activations))
            if any(math.isfinite(a) for a in activations)
            else (float("nan") if activations else 0.0)
        )

        positives = [(key, o) for _, key, o in scored if o is not None and o["detected"]]
        fidelity = 0.0
        po_agreement = float("nan")
        if positives:
            ref_key, reference = positives[-1]
            inputs = canonical_inputs or [
                net_name for net_name, _ in (ref_key or ())
            ]
            parts = [_pi_agreement(pred_input, dict(ref_key or ()), inputs),
                     _po_agreement(pred_output, reference["good"], outputs)
                     if pred_output else 0.0]
            po_agreement = parts[1]
            finite = [p for p in parts if math.isfinite(p)]
            fidelity = sum(finite) / len(finite)

        opened = prompt_opens_thinking(prompt)
        checks: List[bool] = []
        reasoned_turns: List[bool] = []
        shown_detected = [k for k, o in enumerate(shown) if o and o["detected"]]
        first_detected = shown_detected[0] if shown_detected else None
        called: set = set()
        duplicates = 0
        for k in range(len(calls)):
            turn = trajectory.turns[k] if k < len(trajectory.turns) else ""
            reasoned_turns.append(_reasoned(turn, opened))
            checks.append(reasoned_turns[-1])
            checks.append(shown[k] is not None)
            if k > 0:
                checks.append(first_detected is None or k <= first_detected)
                checks.append(identities[k] not in called)
            duplicates += identities[k] in called
            called.add(identities[k])
        if final_turn is None:
            checks += [False] * _FINAL_TURN_CHECKS
        else:
            checks += [
                not pending,
                _thinking_closed(final_turn, opened),
                bool(pred_input),
                bool(pred_output),
                bool(pred_faults),
                bool(shown) and bool(shown[-1] and shown[-1]["detected"]),
                # Care bits only: don't-cares are omitted, never written as x.
                pred_input is not None and not _answer_has_dont_care(pred_input, pred_output),
            ]
        format_score = sum(checks) / len(checks)

        answer_key = vector_key(pred_input)
        # The answer states care bits only, so compare it with each detecting
        # call's care bits (a missing input is read as x).
        verified = {care_bits(key) for key, _ in positives}
        answer_care = care_bits(answer_key)
        out["detection"] = detection
        out["activation"] = activation
        out["fidelity"] = fidelity
        out["format"] = format_score
        out["uniqueness"] = uniqueness
        out["format_compliance_logonly"] = format_score
        out["attempts_logonly"] = float(len(attempts))
        out["duplicate_calls_logonly"] = float(duplicates)
        out["calls_after_detection_logonly"] = float(
            len(calls) - 1 - first_detected if first_detected is not None else 0
        )
        out["reasoned_call_turns_logonly"] = (
            sum(reasoned_turns) / len(reasoned_turns) if reasoned_turns else 0.0
        )
        out["unverified_answer_logonly"] = float(bool(pred_input) and answer_care not in verified)
        out["fault_detected_by_pred_input_vector_acc_logonly"] = float(
            answer_care is not None and answer_care in verified
        )
        out["expected_output_acc_logonly"] = 1.0 if po_agreement >= 0.999 else 0.0
        out["answer_x_free_logonly"] = float(
            pred_input is not None and not _answer_has_dont_care(pred_input, pred_output)
        )
        if canonical_inputs:
            # Care-bit coverage: inputs the detecting call set to 0/1 (every
            # input when no call detected) that the answer states as 0/1.
            assigned = dict(answer_key or ())
            reference_bits = dict(positives[-1][0] or ()) if positives else {}
            care = [pi for pi in canonical_inputs if reference_bits.get(pi, "0") != "x"]
            pi_score = (sum(assigned.get(pi) in ("0", "1") for pi in care) / len(care)) if care else 1.0
            out["pi_completeness_logonly"] = pi_score
            out["input_vector_acc_logonly"] = 1.0 if pi_score >= 0.999 else 0.0

        last = next((o for o in reversed(resimulated) if o is not None), None)
        if last is None:
            rewards.append(out)
            continue

        frame, sim_rewards = last["frame"], last["rewards"]
        out["simulation_valid_logonly"] = 1.0
        out["fault_site_activated_acc_logonly"] = last["activated"]
        if sim_rewards.get("tetramax_available"):
            po_rows = frame.loc[frame["POs"]]
            known = po_rows["Good Machine"].map(_cell_to_int).notna()
            out["known_po_fraction_logonly"] = float(known.mean()) if len(known) else 0.0
            out["indeterminate_detection_logonly"] = float(sim_rewards.get("tetramax_indeterminate", False))
        # Agreement between what the executed call showed and the re-simulation.
        if first_executed is not None and resimulated and resimulated[0] is not None:
            shown_last = shown[first_executed]
            matches = [
                float(_cell_to_int(shown_last[side].get(po)) == _cell_to_int(resimulated[0][side].get(po)))
                for po in outputs if po in shown_last["good"]
                for side in ("good", "bad")
            ]
            out["tool_response_fidelity_logonly"] = sum(matches) / len(matches) if matches else 0.0

        if sim_from_completion is not None and "POs" in frame.columns:
            try:
                po_gold = frame.loc[frame["POs"]]
                po_pred = sim_from_completion.reindex(po_gold.index)
                matches = []
                for po in po_gold.index:
                    if po not in po_pred.index:
                        continue
                    for col in ("Good Machine", "Bad Machine"):
                        a, b = _cell_to_int(po_gold.loc[po, col]), _cell_to_int(po_pred.loc[po, col])
                        if a is not None and b is not None:
                            matches.append(float(a == b))
                if matches:
                    out["pred_vs_fault_sim_acc_logonly"] = sum(matches) / len(matches)
            except Exception:
                pass

        rewards.append(out)

    active, _ = active_reward_objectives()
    if using_native_simulator():
        for row in rewards:
            for objective in ATPG_GDPO_OBJECTIVE_KEYS:
                if objective not in active:
                    row.pop(objective, None)
            row.setdefault('known_po_fraction_logonly', 0.0)
            row.setdefault('indeterminate_detection_logonly', 0.0)
    return rewards


def test_generation_reward(prompts: list, completions: list, **kwargs):
    """Alias for :func:`test_generation_grpo_reward` (same signature and kwargs)."""
    return test_generation_grpo_reward(prompts, completions, **kwargs)


# Example usage
if __name__ == "__main__":
    df_cot = pd.read_csv('/proj/trela/christos/transformers_atpg/data/cot_atpg_data_v1.csv')

    model = None
    torch.cuda.empty_cache()
    # Load the pre-trained Sentence-BERT model
    if model is None:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to('cuda:1')

    cot_block_re = re.compile(r'CHAIN_OF_THOUGHT:\n(.*?)SNAPSHOT', re.DOTALL)
    thought_pattern_re = re.compile(r'(\d+)\.(.*?)(?=\d+\.|$)', re.DOTALL)
    fault_re = re.compile(r"(sa\d)\s+(_\d+_)", re.DOTALL)
    simulation_re = re.compile(r"SNAPSHOT:\n```\n(.*?)```\s+INPUT_VECTOR", re.DOTALL)
    input_vector_re = re.compile(r"INPUT_VECTOR:\s\"(.*?)\"", re.DOTALL)
    expected_output_re = re.compile(r"EXPECTED_OUTPUT:\s\"(.*?)\"", re.DOTALL)
    detected_faults_re = re.compile(r"DETECTED_FAULTS:\s\"(.*?)\"", re.DOTALL)

    prompts = [df_cot.loc[0, 'text'].split("<</SYS>>")[1].strip().split("[/INST]")[0]]
    completions = [df_cot.loc[0, 'text'].split("[/INST]")[1].strip()]
    netlists = [df_cot.loc[0, 'netlist']]

    def fault_fn(x):
        return fault_re.findall(x)

    def simulation_fn(x):
        x_str = extract_markdown_table(x)
        df = markdown_table_to_dataframe(x_str)
        return [df.to_string()]

    def input_vector_fn(x):
        return input_vector_re.findall(x)

    def expected_output_fn(x):
        return expected_output_re.findall(x)

    def detected_faults_fn(x):
        return detected_faults_re.findall(x)

    rewards = test_generation_reward(
        prompts,
        completions,
        netlists,
        fault_re,
        simulation_re,
        input_vector_re,
        expected_output_re,
        detected_faults_re
    )
    print(f"Test Generation: {rewards}")
