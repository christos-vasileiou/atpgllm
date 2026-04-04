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

def extract_json_tool_response_and_convert_to_df(text: str) -> Optional[pd.DataFrame]:
  match = re.search(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", text, re.DOTALL)
  if match:
      try:
          groups = match.groups()
          return pd.DataFrame.from_dict(json.loads(groups[0]))
      except json.JSONDecodeError:
          import ast
          return pd.DataFrame.from_dict(ast.literal_eval(groups[0]))
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
  total = 0
  for po, row in po_rows.iterrows():
    if po not in pred:
      continue
    total += 1
    sim_g = _cell_to_int(row["Good Machine"])
    if sim_g is None:
      continue
    if int(pred[po]) == sim_g:
      ok += 1
  denom = total if total else len(po_rows)
  return (ok / denom) if denom else 0.0, total


def _pi_assignment_score(
    simulation_df: pd.DataFrame, pred_input_vector: str
) -> Tuple[float, int]:
  """Fraction of PIs where INPUT_VECTOR matches simulated good machine."""
  if simulation_df is None or "PIs" not in simulation_df.columns:
    return 0.0, 0
  pi_rows = simulation_df.loc[simulation_df["PIs"]]
  if pi_rows.empty:
    return 0.0, 0
  sep = ":" if ":" in pred_input_vector else "="
  try:
    pred = convert_string_to_dict(pred_input_vector, sep=sep)
  except Exception:
    return 0.0, len(pi_rows)
  ok = 0
  total = 0
  for pi, row in pi_rows.iterrows():
    if pi not in pred:
      continue
    total += 1
    sim_g = _cell_to_int(row["Good Machine"])
    if sim_g is None:
      continue
    if int(pred[pi]) == sim_g:
      ok += 1
  denom = total if total else len(pi_rows)
  return (ok / denom) if denom else 0.0, total


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
  GRPO-oriented reward for ATPG-style completions: learn a test vector that *detects* the target fault.

  Authoritative signal: the fault simulator (``fault_sim`` / ``fast_fault_sim``) on the model's
  ``INPUT_VECTOR`` and ``EXPECTED_OUTPUT``. Components (non-overlapping so ``sum(values)`` is meaningful):

  - **fault_detect_inpvector** — bonus for PO observation (good≠bad on some PO) plus fault-site
    activation (bad value at fault net equals stuck-at and differs from good).
  - **expected_output** / **input_vector** — match of declared vectors to simulated good machine
    on POs / PIs.
  - **fault_simulation** — tool JSON ``<tool_response>`` PO values vs gold simulation.
  - **detected_faults** — target fault string appears in ``DETECTED_FAULTS``.
  - **pred_simulation** / **pred_vs_fault_sim_acc** — completion simulation table vs gold (if present).
  - **format** — light shaping for thinking / tool_call / tool_response / tags.

  Optional kwargs: ``reward_weight_fault_detected_po`` (default 12), ``reward_weight_fault_site`` (4),
  ``reward_weight_po_match`` (5), ``reward_weight_pi_match`` (3), ``reward_weight_tool_json_bonus`` (1),
  ``reward_weight_fault_mention`` (1.5), ``reward_format_weight`` (0.12).

  Returns per-completion component dicts (summed by the trainer).
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

  w_detect = float(kwargs.get("reward_weight_fault_detected_po", 12.0))
  w_site = float(kwargs.get("reward_weight_fault_site", 4.0))
  w_po = float(kwargs.get("reward_weight_po_match", 5.0))
  w_pi = float(kwargs.get("reward_weight_pi_match", 3.0))
  w_tool_json = float(kwargs.get("reward_weight_tool_json_bonus", 1.0))
  w_fault_mention = float(kwargs.get("reward_weight_fault_mention", 1.5))
  format_weight = float(kwargs.get("reward_format_weight", 0.12))

  rewards: List[Dict[str, float]] = []

  for prompt, completion, netlist in zip(prompts, completions, netlists):
    # Keys are non-overlapping so sum(...) is a well-defined total (GRPO / logging).
    out: Dict[str, float] = {
      "format": 0.0,
      "fault_detect_inpvector": 0.0,
      "fault_simulation": 0.0,
      "expected_output": 0.0,
      "input_vector": 0.0,
      "detected_faults": 0.0,
      "pred_simulation": 0.0,
      "pred_vs_fault_sim_acc": 0.0,
      "fault_detected_by_pred_input_vector_acc": 0.0,
      "expected_output_acc": 0.0,
      "input_vector_acc": 0.0,
      "detected_faults_acc": 0.0,
      # Extra scalars for dashboards (also included in sum — keep small):
      "sim_table_bonus": 0.0,
    }

    fault_info = fault_fn(prompt)
    fault, net = (None, None)
    if fault_info:
      fault, net = fault_info[0]
    stuck_at = int(fault[-1]) if fault else -1

    # Light format shaping (optional extractors)
    for fn in (thinking_fn, tool_call_fn, tool_response_fn):
      if fn is not None:
        try:
          out["format"] += format_weight if fn(completion) else -0.35
        except Exception:
          out["format"] -= 0.35

    pred_input = (input_vector_fn(completion) or [None])[0]
    pred_output = (expected_output_fn(completion) or [None])[0]
    pred_faults = (detected_faults_fn(completion) or [None])[0]

    if pred_input:
      out["format"] += format_weight
    else:
      out["format"] -= 0.8
    if pred_output:
      out["format"] += format_weight
    else:
      out["format"] -= 0.8
    if pred_faults:
      out["format"] += format_weight
    else:
      out["format"] -= 0.5

    sim_from_completion = _simulation_table_from_completion(completion, simulation_fn)
    if sim_from_completion is not None:
      out["sim_table_bonus"] = 0.35

    if not (fault and net and pred_input and pred_output and netlist):
      rewards.append(out)
      continue

    try:
      result = fault_sim_runner(
        pred_input,
        pred_output,
        f"{fault} {net}",
        netlist,
        gate_func,
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

    detected = _fault_detected_at_pos(fault_simulation)
    site_ok = _fault_site_activated(fault_simulation, net, stuck_at)
    po_score, _ = _po_prediction_score(fault_simulation, pred_output)
    pi_score, _ = _pi_assignment_score(fault_simulation, pred_input)

    tool_bonus = _tool_response_po_consistency_bonus(completion, fault_simulation)
    mention = _mentions_target_fault(pred_faults, fault, net) if pred_faults else 0.0

    out["fault_detect_inpvector"] = w_detect * float(detected) + w_site * float(site_ok)
    out["expected_output"] = w_po * po_score
    out["input_vector"] = w_pi * pi_score
    out["fault_simulation"] = w_tool_json * tool_bonus
    out["detected_faults"] = w_fault_mention * mention
    out["fault_detected_by_pred_input_vector_acc"] = float(detected)
    out["expected_output_acc"] = 1.0 if po_score >= 0.999 else 0.0
    out["input_vector_acc"] = 1.0 if pi_score >= 0.999 else 0.0
    out["detected_faults_acc"] = 1.0 if mention >= 0.99 else 0.0

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
          out["pred_vs_fault_sim_acc"] = acc
          out["pred_simulation"] = 3.0 * (acc - 0.5)
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

