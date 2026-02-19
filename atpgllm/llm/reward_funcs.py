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
from typing import Optional
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

# Extract the fault and net
def test_generation_reward(prompts: list, completions: list, **kwargs):
  """
  Calculate the reward for the test generation task.

  This function evaluates the quality of generated test vectors for fault detection in digital circuits.
  It analyzes the provided prompts, completions, and netlists to compute rewards based on various criteria
  such as simulation accuracy, input vector validity, and fault detection effectiveness.

  Parameters:
  prompts (list): A list of input prompts describing the fault detection scenarios.
  completions (list): A list of generated completions corresponding to each prompt.
  netlists (list): A list of netlists representing the circuit structures.
  fault_fn (function): Function to extract fault information.
  simulation_fn (function): Function to extract simulation results.
  input_vector_fn (function): Function to extract input vectors.
  expected_output_fn (function): Function to extract expected outputs.
  detected_faults_fn (function): Function to extract detected faults.
  lib_gate_funcs (dict): Dictionary containing the logic functions for each gate type.
  eval_mode (bool): A boolean indicating whether to evaluate the rewards.

  Returns:
  list: A list of dictionaries, each containing reward scores for different aspects of the test generation task.
        The keys in each dictionary are:
        - 'format': Reward for correct formatting of the completion.
        - 'pred_simulation': Reward for accuracy of the predicted simulation.
        - 'fault_simulation': Reward for accuracy of the fault simulation.
        - 'input_vector': Reward for correctness of the generated input vector.
        - 'expected_output': Reward for correctness of the expected output.
        - 'detected_faults': Reward for correctly identifying detected faults.
        - 'fault_detect_inpvector': Reward for effectiveness of the input vector in detecting the fault.
  """
  netlists = kwargs.get('netlists', None)
  if netlists is None:
    raise ValueError("netlists must be provided")
  fault_fn = kwargs.get('fault_fn', None)
  if fault_fn is None:
    raise ValueError("fault_fn must be provided")
  simulation_fn = kwargs.get('simulation_fn', None)
  if simulation_fn is None:
    raise ValueError("simulation_fn must be provided")
  input_vector_fn = kwargs.get('input_vector_fn', None)
  if input_vector_fn is None:
    raise ValueError("input_vector_fn must be provided")
  expected_output_fn = kwargs.get('expected_output_fn', None)
  if expected_output_fn is None:
    raise ValueError("expected_output_fn must be provided")
  detected_faults_fn = kwargs.get('detected_faults_fn', None)
  if detected_faults_fn is None:
    raise ValueError("detected_faults_fn must be provided")
  eval_mode = kwargs.get('eval_mode', False)
  lib_gate_funcs = kwargs.get('lib_gate_funcs', None)
  thinking_fn = kwargs.get('thinking_fn', None)
  tool_call_fn = kwargs.get('tool_call_fn', None)
  tool_response_fn = kwargs.get('tool_response_fn', None)
  
  if lib_gate_funcs is None:
    gate_func = {'IB': logic_buf, 'AN': logic_and, 'OR': logic_or, 'XO': logic_xor, 'IV': logic_not, 'ND': logic_nand, 'NR': logic_nor, 'XN': logic_xnor}
  else:
    gate_func = lib_gate_funcs
    fault_sim = kwargs.get('fault_sim', None)
    # When using library gate functions, fault_sim MUST be provided as it requires OptimizedNetlist
    if fault_sim is None:
      raise ValueError("fault_sim function must be provided when using lib_gate_funcs")

  rewards = []
  
  for prompt, completion, netlist in zip(prompts, completions, netlists):
    fault = fault_fn(prompt)
    if fault:
      fault, net = fault[0]
    
    # Calculate Reward for Fault Simulation
    reward = {'format': 0, 
              'pred_simulation': 0, 
              'fault_simulation': 0, 
              'input_vector': 0, 
              'expected_output': 0, 
              'detected_faults': 0, 
              'fault_detect_inpvector': 0,
              'pred_vs_fault_sim_acc': 0,
              'fault_detected_by_pred_input_vector_acc': 0,
              'expected_output_acc': 0,
              'input_vector_acc': 0,
              'detected_faults_acc': 0
              }
    
    # Extract the thinking
    if thinking_fn is not None:
      pred_thinking = thinking_fn(completion)
      if pred_thinking:
        pred_thinking = pred_thinking[0]
        reward['format'] += 0.125
      else:
        reward['format'] -= 1

    # Extract the tool call
    if tool_call_fn is not None:
      pred_tool_call = tool_call_fn(completion)
      if pred_tool_call:
        pred_tool_call = pred_tool_call[0]
        reward['format'] += 0.125
      else:
        reward['format'] -= 1

    # Extract the tool response
    if tool_response_fn is not None:
      pred_tool_response = tool_response_fn(completion)
      if pred_tool_response:
        pred_tool_response = pred_tool_response[0]
        reward['format'] += 0.125
      else:
        reward['format'] -= 1

    # Extract the simulation
    pred_simulation = simulation_fn(completion)
    if pred_simulation:
      pred_simulation = pred_simulation[0]
      reward['format'] += 0.125
    else:
      reward['format'] -= 1

    # Extract the input vector
    pred_input_vector = input_vector_fn(completion)
    if pred_input_vector:
      pred_input_vector = pred_input_vector[0]
      reward['format'] += 0.125
    else:
      reward['format'] -= 1

    # Extract the expected output
    pred_expected_output = expected_output_fn(completion)
    if pred_expected_output:
      pred_expected_output = pred_expected_output[0]
      reward['format'] += 0.125
    else:
      reward['format'] -= 1

    # Extract the detected faults
    pred_detected_faults = detected_faults_fn(completion)
    if pred_detected_faults:
      pred_detected_faults = pred_detected_faults[0]
      reward['format'] += 0.125
    else:
      reward['format'] -= 1

    if fault and pred_simulation:
      try:
        # +1 Parse the simulation and convert it to a DataFrame
        pred_simulation = convert_to_df(pred_simulation)
        if pred_simulation.loc[net, "Good Machine"] == 'x':
          reward['pred_simulation'] -= 2.5
        else:
          reward['pred_simulation'] += .5
          # Check if the Good Machine value is different than Bad Machine for the requested net + if the fault simulation trigger the requested fault
          reward['pred_simulation'] += int(pred_simulation.loc[net, "Good Machine"] != pred_simulation.loc[net, "Bad Machine"])
          reward['pred_simulation'] += int(pred_simulation.loc[net, "Bad Machine"] == int(fault[-1]))
      except:
        pred_simulation = None
        reward['pred_simulation'] -= 2.5
    else:
      reward['pred_simulation'] -= 3.5
    
    if pred_input_vector and pred_expected_output and fault and net and netlist:
      try:
        # Run Fault Simulation
        fault_simulation, fault_sim_rewards = fault_sim(pred_input_vector, pred_expected_output, f"{fault} {net}", netlist, gate_func, return_rewards=True)
        # Reward the simulation
        # +2 if LLM simulation and actual simulation are the same!
        if isinstance(pred_simulation, pd.DataFrame) and isinstance(fault_simulation, pd.DataFrame):
          # Reward the generated simulation
          # Calculate row-wise accuracy
          row_matches = pred_simulation.eq(fault_simulation[["Good Machine", "Bad Machine"]])
          
          # Weight each row based on its importance
          # Rows in fault path are most important (weight 2.0)
          # Primary outputs are next most important (weight 1.5) 
          # All other rows have base weight 1.0
          weights = pd.DataFrame(1.0, index=row_matches.index, columns=row_matches.columns)
          reward['pred_vs_fault_sim_acc'] += (row_matches.sum() / row_matches.count()).mean()
          weights[fault_simulation["Fault Propagation Path"] | fault_simulation["Backtrack Sensitizing Inputs"]] = 2
          weights.loc[net, :] = 5
          
          # Calculate weighted accuracy
          weighted_accuracy = ((row_matches * weights).sum() / weights.sum()).mean()
          
          # Scale reward exponentially to incentivize high accuracy
          # This gives:
          #  ^2 -----------------------------    ^3 -------------------------------   ^4 -------------------------------
          # 50% accuracy -> 2.25x base reward  | 50% accuracy -> 3.38x base reward   | 50% accuracy -> 5.06x base reward   |
          # 75% accuracy -> 3.06x base reward  | 75% accuracy -> 5.36x base reward   | 75% accuracy -> 9.38x base reward   |
          # 95% accuracy -> 3.80x base reward  | 95% accuracy -> 7.41x base reward   | 95% accuracy -> 14.46x base reward  |
          # 100% accuracy -> 4.00x base reward | 100% accuracy -> 9.00x base reward  | 100% accuracy -> 16.00x base reward |
          exponent = 4
          base_reward = 1.0 if weighted_accuracy < 0.9 else 2.0 # force >90% accuracy.
          reward['fault_simulation'] += base_reward * (1 + weighted_accuracy) ** exponent - base_reward
          
          # Calculate a smoother reward using weights for each subcondition
          fault_detection_condition = fault_simulation.loc[net, "Bad Machine"] == int(fault[-1]) and \
                                      fault_simulation.loc[net, 'Good Machine'] != fault_simulation.loc[net, 'Bad Machine']
          
          # Reward if the fault is detected by the predicted input vector
          reward['fault_detected_by_pred_input_vector_acc'] += int(fault_detection_condition)
          if not fault_detection_condition:
            # Penalize heavily if the fault is not detected
            reward['fault_detect_inpvector'] -= 5
          
          # Reward based on validity of generated Good-Machine input values and generated input vector
          if fault_sim_rewards.get('input_nets_match', False):
            # Convert the predicted input vector string to a dictionary
            input_vector_dict = convert_string_to_dict(pred_input_vector, sep=':' if ':' in pred_input_vector else '=')
            # Get the input values from the simulation
            pred_input_vector_from_sim_dict = pred_simulation[fault_simulation['PIs']]['Good Machine'].to_dict()
            # Get weights for primary inputs
            input_vector_weights = weights[fault_simulation['PIs']]['Good Machine'].to_dict()
            # Calculate weighted matches between predicted and actual input values
            weighted_input_matches = [(iv==pi)*wi for iv, pi, wi in zip(input_vector_dict.values(), pred_input_vector_from_sim_dict.values(), input_vector_weights.values())]
            # Calculate weighted accuracy for input vector
            weighted_input_accuracy = sum(weighted_input_matches) / sum(input_vector_weights.values())
            # Apply exponential scaling to reward
            exponent = 2
            base_reward = 2.0
            reward['fault_detect_inpvector'] += base_reward * (1 + weighted_input_accuracy) ** exponent - base_reward
            # Additional reward for perfect accuracy
            reward['input_vector_acc'] += int(weighted_input_accuracy==1)
          
          # Reward based on validity of generated Good-Machine output values and generated expected output vector
          if fault_sim_rewards.get('output_nets_match', False):
            # Convert the predicted output vector string to a dictionary
            output_vector_dict = convert_string_to_dict(pred_expected_output, sep=':' if ':' in pred_expected_output else '=')
            # Get the output values from the simulation
            pred_expected_output_from_sim_dict = pred_simulation[fault_simulation['POs']]['Good Machine'].to_dict()
            # Get weights for primary outputs
            output_vector_weights = weights[fault_simulation['POs']]['Good Machine'].to_dict()
            # Calculate weighted matches between predicted and actual output values
            weighted_output_matches = [(ov==pv)*wo for ov, pv, wo in zip(output_vector_dict.values(), pred_expected_output_from_sim_dict.values(), output_vector_weights.values())]
            # Calculate weighted accuracy for output vector
            weighted_output_accuracy = sum(weighted_output_matches) / sum(output_vector_weights.values())
            # Apply exponential scaling to reward
            exponent = 2
            base_reward = 2.0
            reward['expected_output'] += base_reward * (1 + weighted_output_accuracy) ** exponent - base_reward
            # Additional reward for perfect accuracy
            reward['expected_output_acc'] += int(weighted_output_accuracy==1)
          
          # Reward the detected Fault Path. From the point where the fault occurs and onwards
          # Extract fault path information from simulation
          detected_fault_path_df = fault_simulation[fault_simulation["Fault Propagation Path"]].reset_index()[["Bad Machine", "index"]]
          # Format the bad machine values as fault types (sa0, sa1)
          detected_fault_path_df['Bad Machine'] = detected_fault_path_df['Bad Machine'].apply(lambda x: f"sa{x}").values
          # Parse the predicted detected faults string into a numpy array
          pred_detected_faults_np = np.array([(fault, loc) for fault_loc in pred_detected_faults.split(',') for fault, loc in [fault_loc.strip().split()]])
          # Check if the predicted fault path matches the actual fault path
          if detected_fault_path_df.shape[0] == pred_detected_faults_np.shape[0] and np.array_equal(detected_fault_path_df['index'].values, pred_detected_faults_np[:, 1]):
            # Count how many values are equal between the two arrays
            equal_values = sum(a == b for a, b in zip(detected_fault_path_df['index'].values, pred_detected_faults_np[:, 1]))
            accuracy = equal_values / len(detected_fault_path_df['index'].values)
            # Apply exponential scaling to reward
            exponent = 2
            base_reward = 2.0 
            reward['detected_faults'] += base_reward * (1 + accuracy) ** exponent - base_reward
            # Additional reward for perfect accuracy
            reward['detected_faults_acc'] += int(accuracy==1)
        else:
          # Penalize if either simulation is missing
          reward['fault_detect_inpvector'] -= 2
          reward['fault_simulation'] -= 2
      except:
        pass
    else:
      # Penalize heavily if required inputs are missing
      reward['fault_detect_inpvector'] -= 5
      reward['fault_simulation'] -= 5
    rewards.append(reward)
  
  return rewards

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

