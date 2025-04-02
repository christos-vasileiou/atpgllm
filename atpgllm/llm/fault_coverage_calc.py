from io import StringIO
from typing import Union
import pandas as pd
import os
import regex as re
import numpy as np


def logic_buf(input: list):
  if len(input) != 1:
    raise ValueError("Input vector for logic_buf should have exactly one input")
  return input[0]

def logic_and(*inputs: list):
  return int(all(*inputs))

def logic_or(*inputs):
  return int(any(*inputs))

def logic_xor(*inputs):
  return sum(*inputs) % 2

def logic_not(input: list):
  if len(input) != 1:
    raise ValueError("Input vector for logic_buf should have exactly one input")
  return int(not input[0])

def logic_nand(*inputs: list):
  return int(not all(*inputs))

def logic_nor(*inputs):
  return int(not any(*inputs))

def logic_xnor(*inputs):
  return int(not (sum(*inputs) % 2))


def detect_and_get_fault(quotes):
  for s in quotes:
    if re.match(r"^sa\d _\d+_$", s):
      return s
  return ""


def detect_and_get_input_output_vectors(quotes):
  quotes = np.array(quotes)
  vectors_idx = np.array([all([char.isdigit() or char.isspace() for char in s]) for s in quotes])
  vectors = quotes[vectors_idx]
  return vectors[:2]

def fault_sim(input_vector: Union[str, dict], output_vector: Union[str, dict], fault: str, verilog_code: str, gate_func: dict = {'IB': logic_buf, 'AN': logic_and, 'OR': logic_or, 'XO': logic_xor, 'IV': logic_not, 'ND': logic_nand, 'NR': logic_nor, 'XN': logic_xnor}, return_rewards: bool =False):
  """
  Simulates a fault in a digital circuit described by a Verilog code.

  Parameters:
  input_vector (str, dict): Represents the input vector for the circuit. 
                            - When it's a string the values are separated by spaces.
                            - When it's a dictionary the values are either 0 or 1 and the keys are the input ports.
  output_vector (str, dict): Represents the expected output vector for the circuit. 
                             - When it's a string the values are separated by spaces.
                             - When it's a dictionary the values are either 0 or 1 and the keys are the output ports.
  fault (str): A string representing the fault to be simulated. The fault is in the format "saX _Y_", where X is the fault value (0 or 1) and Y is the faulty net name.
  verilog_code (str): A string containing the Verilog code of the digital circuit.
  gate_func (dict): A dictionary containing the logic functions for each gate type.
  return_rewards (bool): A boolean indicating whether to return the rewards.

  Returns:
  DataFrame: A tuple containing two dictionaries:
      - good_machine (dict): A dictionary representing the output values of the circuit for the given input vector without any faults.
      - bad_machine (dict): A dictionary representing the output values of the circuit for the given input vector with the simulated fault.
  """
  GATE_INGREDIENT = r"(\w+)\s+(_\p{N}+_)\s+\(\s*(_\p{N}+_)\s*,\s*(.+?)\s*\);"
  INPUTS_NETS = r"input\s*(.+?);"
  OUTPUTS_NETS = r"output\s*(.+?);"
  FAULT_VALUE = r"sa(\d) (_\d+_)"

  gate_ingredients = re.compile(GATE_INGREDIENT)
  inputs_nets = re.compile(INPUTS_NETS)
  outputs_nets = re.compile(OUTPUTS_NETS)
  fault_value = re.compile(FAULT_VALUE)

  # Parse verilog lines with keyword 'input'/'output'
  # Collect Primary Inputs and Primary Outputs of the model
  inputs = [net.strip() for match in inputs_nets.findall(verilog_code) for net in match.split(',')]
  outputs = [net.strip() for match in outputs_nets.findall(verilog_code) for net in match.split(',')]
  
  # Calculate rewards if input and output vectors have the correct length
  if return_rewards:
    rewards = {}
    # Check if the input vector has the correct length and if the nets names are correct
    # rewards['input_nets_match'] = len(inputs) == len(input_vector.split(',')) and all(net.strip() in inputs for net_value in input_vector.split(',') for net, value in [net_value.split(':')])
    # Check if the output vector has the correct length and if the nets names are correct
    # rewards['output_nets_match'] = len(outputs) == len(output_vector.split(',')) and all(net.strip() in outputs for net_value in output_vector.split(',') for net, value in [net_value.split(':')])
    
  # Get faulty value and faulty net
  faulty_value, faulty_net = next(iter(fault_value.findall(fault)))
  # keep track of fault path
  fault_path = [faulty_net]

  circuit = []
  # Map input and output nets to their values
  test_ivector = input_vector.copy() if isinstance(input_vector, dict) else {net.strip():int(value.strip()) for net_value in input_vector.split(',') for net, value in [net_value.split(':')]}
  test_ovector = output_vector.copy() if isinstance(output_vector, dict) else {net.strip():int(value.strip()) for net_value in output_vector.split(',') for net, value in [net_value.split(':')]}
  
  # Keep track of the circuit's inputs and outputs
  _inputs = inputs.copy()
  _outputs = outputs.copy()

  # Define good & bad machine simulation
  good_machine = {}
  bad_machine = {}

  # initialize input & output values in the good machine behavior
  good_machine.update(test_ivector)
  good_machine.update(test_ovector)

  # initialize values in the bad machine behavior
  bad_machine.update(test_ivector)

  # inject the fault value
  bad_machine.update({faulty_net: int(faulty_value)})

  # Simulation: Propagate input vector. 
  # If fault is given, force the value to the specified net (Bad Machine Simulation)
  # If no fault is given, propagate the value normally (Good Machine Simulation)
  for line in verilog_code.splitlines(): # Parse line-by-line the netlist
    if not line.strip():
      continue
    if len(gate_ingredients.findall(line))==0:
      continue
    # get ingredients using the regex pattern
    ingredients = gate_ingredients.findall(line)[0]
    # gather ingredients
    circuit.append(ingredients)
    # get ingredients from circuit
    gate_type, instance, output, *inputs = ingredients
    inputs = inputs[0].split(', ')
    if fault_path[-1] in inputs:
      fault_path.append(output)

    good_machine[output] = gate_func[gate_type[:2]]([good_machine[i] for i in inputs])
    if output not in bad_machine.keys():
      bad_machine[output] = gate_func[gate_type[:2]]([bad_machine[i] for i in inputs])
  simulation = pd.concat([pd.Series(good_machine), pd.Series(bad_machine)], axis=1)
  simulation[2] = simulation.index.isin(_inputs)
  simulation[3] = simulation.index.isin(_outputs)
  simulation[4] = simulation.index.isin(fault_path)

  simulation.columns = ["Good Machine", "Bad Machine", "PIs", "POs", "Fault Path"]

  if return_rewards:
    # Check if the faulty net has been correctly identified using the predicted input vector
    # rewards["trigger_fault_reward"] = 2*int(simulation.loc[faulty_net, "Good Machine"] != simulation.loc[faulty_net, "Bad Machine"])-1

    return simulation, rewards

  return simulation


def validate_generated_text(completion, eval_netlist, quotes_re = re.compile(r'"([^"]*[\w\s_][^"]*)"'), postproc_re = re.compile(r'(_\d+_)'), fault_sim_re = re.compile(r'```(.*?)```', re.DOTALL), gate_func = {'IB': logic_buf, 'AN': logic_and, 'OR': logic_or, 'XO': logic_xor, 'IV': logic_not, 'ND': logic_nand, 'NR': logic_nor, 'XN': logic_xnor}):
  try:
    start_token = completion.find("[/INST]") + len("[/INST]")
    generated_text = completion[start_token:].strip()
    quotes = quotes_re.findall(completion)
    # print(quotes)
    user_test_net, model_test_net = quotes[1:3]
    # print(f"Is the tested net the one that the user asked for? {user_test_net == model_test_net}")
    triple_backticks_contents = fault_sim_re.findall(generated_text)
    # print(triple_backticks_contents)
    if len(triple_backticks_contents) == 0:
      return 0
    fault_sim_str = triple_backticks_contents[0]
    postproc_fault_sim = postproc_re.sub(r'\n\1', fault_sim_str)
  except:
    return 0

  try:
    generated_fault_sim = pd.read_csv(StringIO(postproc_fault_sim), sep='\s+', skiprows=2, header=None).set_index(0)
    generated_fault_sim.columns = ["Good Machine", "Bad Machine"]
  except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
    # print("the file is empty")
    # file = "invalid_gen_texts.md"
    # mode = 'a' if os.path.exists(file) else 'w'
    # with open(file, mode) as f:
    #   f.write(f"{postproc_fault_sim}\n\n---------------------------------------------------------------\n\n")
    return 0

  try:
    fault = detect_and_get_fault(quotes)
    input_vector, output_vector = detect_and_get_input_output_vectors(quotes)
    verilog_code = eval_netlist
    # print(f"input: {input_vector}, output: {output_vector}, fault: {fault}")
    # print(fault)
    sim = fault_sim(input_vector, output_vector, fault, verilog_code, gate_func)[["Good Machine", "Bad Machine"]]
    faulty_net = fault.split()[1].strip()
    # print(f"Does the generated vector test the fault that user asked for? {sim.loc[fault.split()[1], 'Good Machine'] != fault.split()[0][2:]}")
    if faulty_net not in sim.index:
      return 0
    # if the values between good machine and bad machine differ means that the test_vector triggers the behavior
    cnt = int(sim.loc[faulty_net, "Good Machine"] != sim.loc[faulty_net, "Bad Machine"])
  except:
    return 0

  return cnt
