import sys
from pathlib import Path

# Add data_preprocessing to sys.path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'data_preprocessing'))

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

from fault_sim import convert_string_to_dict

warnings.filterwarnings("ignore")

# Keys ending with this suffix are logged to dashboards but excluded from the GRPO scalar
# (see ``train_scalar_from_reward_components`` and DualAdapterGRPOTrainer reward parsing).
REWARD_LOGONLY_SUFFIX = "_logonly"

# GDPO priorities are applied after each objective is normalized. Raw reward
# magnitudes therefore do not encode cross-objective importance.
ATPG_GDPO_OBJECTIVE_KEYS = ("detection", "activation", "fidelity", "format")
ATPG_GDPO_OBJECTIVE_WEIGHTS = (1.0, 0.25, 0.20, 0.05)


def train_scalar_from_reward_components(components: Dict[str, float]) -> float:
  """Weighted scalar fallback for callers that cannot consume GDPO objectives."""
  objective_weights = dict(
    zip(ATPG_GDPO_OBJECTIVE_KEYS, ATPG_GDPO_OBJECTIVE_WEIGHTS)
  )
  total = 0.0
  for key, value in components.items():
    if str(key).endswith(REWARD_LOGONLY_SUFFIX):
      continue
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


def _fault_site_activated(simulation_df: pd.DataFrame, fault_net: str, stuck_at: int) -> bool:
  if simulation_df is None or fault_net not in simulation_df.index:
    return False
  row = simulation_df.loc[fault_net]
  g = _cell_to_int(row["Good Machine"])
  b = _cell_to_int(row["Bad Machine"])
  if g is None or b is None:
    return False
  return b == stuck_at and g != b


def _po_prediction_score(
    simulation_df: pd.DataFrame, pred_expected_output: str
) -> Tuple[float, int]:
  """Fraction of POs where EXPECTED_OUTPUT matches simulated good machine (0..1, count)."""
  if simulation_df is None or "POs" not in simulation_df.columns:
    return 0.0, 0
  po_rows = simulation_df.loc[simulation_df["POs"]]
  if po_rows.empty:
    return 0.0, 0
  sep = ":" if ":" in pred_expected_output else "="
  try:
    pred = convert_string_to_dict(pred_expected_output, sep=sep)
  except Exception:
    return 0.0, len(po_rows)
  ok = 0
  supplied = 0
  for po, row in po_rows.iterrows():
    if po not in pred:
      continue
    supplied += 1
    sim_g = _cell_to_int(row["Good Machine"])
    if sim_g is None:
      continue
    try:
      pred_value = int(pred[po])
    except (TypeError, ValueError):
      continue
    if pred_value == sim_g:
      ok += 1
  # Missing outputs are incorrect; scoring only the supplied subset lets a
  # one-output answer claim perfect fidelity on a multi-output design.
  denom = len(po_rows)
  return (ok / denom) if denom else 0.0, supplied


def _pi_assignment_score(
    simulation_df: pd.DataFrame,
    pred_input_vector: str,
    required_inputs: Optional[List[str]] = None,
) -> Tuple[float, int]:
  """Fraction of canonical PIs supplied with valid binary values."""
  if simulation_df is None:
    return 0.0, 0
  sep = ":" if ":" in pred_input_vector else "="
  try:
    pred = convert_string_to_dict(pred_input_vector, sep=sep)
  except Exception:
    return 0.0, len(required_inputs or [])

  if required_inputs is None:
    if "PIs" not in simulation_df.columns:
      return 0.0, 0
    required_inputs = list(simulation_df.index[simulation_df["PIs"]])
  if not required_inputs:
    return 0.0, 0

  ok = 0
  supplied = 0
  for pi in required_inputs:
    if pi not in pred:
      continue
    supplied += 1
    try:
      value = int(pred[pi])
    except (TypeError, ValueError):
      continue
    if value in (0, 1):
      ok += 1
  return ok / len(required_inputs), supplied


def _tool_response_po_consistency_bonus(completion: str, simulation_df: pd.DataFrame) -> float:
  """Small bonus if <tool_response> Good/Bad PO values match authoritative simulation."""
  raw = extract_json_tool_response_and_convert_to_df(completion)
  if (
    raw is None
    or raw.empty
    or simulation_df is None
    or "POs" not in simulation_df.columns
    or "Good Machine" not in raw.columns
    or "Bad Machine" not in raw.columns
  ):
    return 0.0
  po_rows = simulation_df.loc[simulation_df["POs"]]
  if po_rows.empty:
    return 0.0
  good_hits, bad_hits, n = 0, 0, 0
  for po in po_rows.index:
    if po not in raw.index:
      continue
    n += 1
    sg = _cell_to_int(po_rows.loc[po, "Good Machine"])
    sb = _cell_to_int(po_rows.loc[po, "Bad Machine"])
    tg = _cell_to_int(raw.loc[po, "Good Machine"])
    tb = _cell_to_int(raw.loc[po, "Bad Machine"])
    if sg is not None and tg is not None and sg == tg:
      good_hits += 1
    if sb is not None and tb is not None and sb == tb:
      bad_hits += 1
  if n == 0:
    return 0.0
  return 1.5 * ((good_hits + bad_hits) / (2 * n))


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
  ATPG reward with four explicit GDPO objectives.

  ``detection`` is the authoritative functional objective. ``activation`` is
  the only pre-detection progress signal. ``fidelity`` and ``format`` are
  conditioned on detection so easy reporting proxies cannot train the policy
  independently. Keys ending in ``_logonly`` are diagnostics only.
  """
  netlists = kwargs.get("netlists", None)
  if netlists is None:
    raise ValueError("netlists must be provided")

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
  thinking_fn = kwargs.get("thinking_fn", None)
  tool_call_fn = kwargs.get("tool_call_fn", None)
  tool_response_fn = kwargs.get("tool_response_fn", None)

  if lib_gate_funcs is None:
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

  for idx, (prompt, completion, netlist) in enumerate(zip(prompts, completions, netlists)):
    out: Dict[str, float] = {
      "detection": 0.0,
      "activation": 0.0,
      "fidelity": 0.0,
      "format": 0.0,
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
    }

    fault_info = fault_fn(prompt)
    fault, net = (None, None)
    if fault_info:
      fault, net = fault_info[0]
    stuck_at = int(fault[-1]) if fault else -1

    format_checks: List[float] = []
    for fn in (thinking_fn, tool_call_fn, tool_response_fn):
      if fn is not None:
        try:
          format_checks.append(float(bool(fn(completion))))
        except Exception:
          format_checks.append(0.0)

    pred_input = (input_vector_fn(completion) or [None])[0]
    pred_output = (expected_output_fn(completion) or [None])[0]
    pred_faults = (detected_faults_fn(completion) or [None])[0]

    format_checks.extend(
      [float(bool(pred_input)), float(bool(pred_output)), float(bool(pred_faults))]
    )
    format_score = (
      sum(format_checks) / len(format_checks) if format_checks else 0.0
    )
    out["format_compliance_logonly"] = format_score

    sim_from_completion = _simulation_table_from_completion(completion, simulation_fn)
    if sim_from_completion is not None:
      out["sim_table_present_logonly"] = 1.0

    if not (fault and net and pred_input and netlist):
      rewards.append(out)
      continue

    mod_name = module_names[idx] if idx < len(module_names) else None
    canonical_outputs = list(getattr(netlist, "output_nets", []) or [])
    canonical_inputs = list(getattr(netlist, "input_nets", []) or [])
    # Values are placeholders only. fast_fault_sim uses these canonical names
    # to mark POs and computes their values from the circuit.
    simulation_outputs: Any
    if canonical_outputs:
      simulation_outputs = {po: 0 for po in canonical_outputs}
    elif pred_output:
      simulation_outputs = pred_output
    else:
      rewards.append(out)
      continue

    try:
      result = fault_sim_runner(
        pred_input,
        simulation_outputs,
        f"{fault} {net}",
        netlist,
        gate_func,
        module_name=mod_name,
        return_rewards=True,
      )
      fault_simulation, _fault_sim_rewards = result
    except Exception:
      rewards.append(out)
      continue

    if not isinstance(fault_simulation, pd.DataFrame) or fault_simulation.empty:
      rewards.append(out)
      continue
    if "error" in fault_simulation.columns:
      rewards.append(out)
      continue

    if _fault_sim_rewards.get("tetramax_available"):
      detected = bool(_fault_sim_rewards.get("tetramax_detected"))
    else:
      detected = _fault_detected_at_pos(fault_simulation)
    site_ok = _fault_site_activated(fault_simulation, net, stuck_at)
    po_score, _ = (
      _po_prediction_score(fault_simulation, pred_output)
      if pred_output
      else (0.0, 0)
    )
    pi_score, _ = _pi_assignment_score(
      fault_simulation,
      pred_input,
      required_inputs=canonical_inputs or None,
    )

    tool_bonus = _tool_response_po_consistency_bonus(completion, fault_simulation)
    mention = _mentions_target_fault(pred_faults, fault, net) if pred_faults else 0.0

    detected_f = float(detected)
    site_ok_f = float(site_ok)

    out["detection"] = detected_f
    out["activation"] = site_ok_f
    out["fidelity"] = detected_f * po_score
    out["format"] = detected_f * (0.5 * format_score + 0.5 * pi_score)
    out["fault_detected_by_pred_input_vector_acc_logonly"] = detected_f
    out["fault_site_activated_acc_logonly"] = site_ok_f
    out["pi_completeness_logonly"] = pi_score
    out["tool_response_fidelity_logonly"] = min(1.0, tool_bonus / 1.5)
    out["fault_mention_logonly"] = mention
    out["expected_output_acc_logonly"] = 1.0 if po_score >= 0.999 else 0.0
    out["input_vector_acc_logonly"] = 1.0 if pi_score >= 0.999 else 0.0
    out["detected_faults_acc_logonly"] = 1.0 if mention >= 0.99 else 0.0

    if sim_from_completion is not None and "POs" in fault_simulation.columns:
      try:
        po_gold = fault_simulation.loc[fault_simulation["POs"]]
        po_pred = sim_from_completion.reindex(po_gold.index)
        matches = []
        for idx in po_gold.index:
          if idx not in po_pred.index:
            continue
          for col in ("Good Machine", "Bad Machine"):
            a, b = _cell_to_int(po_gold.loc[idx, col]), _cell_to_int(po_pred.loc[idx, col])
            if a is not None and b is not None:
              matches.append(float(a == b))
        if matches:
          acc = sum(matches) / len(matches)
          out["pred_vs_fault_sim_acc_logonly"] = acc
      except Exception:
        pass

    rewards.append(out)

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

