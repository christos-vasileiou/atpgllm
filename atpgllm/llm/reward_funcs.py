import pandas as pd
import regex as re
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from .fault_coverage_calc import fault_sim, logic_and, logic_buf, logic_not, logic_nand, logic_nor, logic_or, logic_xor, logic_xnor
from ..utils import is_main_process
import torch.distributed as dist
from io import StringIO
import warnings

warnings.filterwarnings("ignore")

cots = ["""CHAIN_OF_THOUGHT:
1. **Identify Relevant Components in Netlist**: Examine the netlist to locate components and connections related to the {fault}.
2. **Analyze Fault Mechanism**: Understand how the {fault} impacts the circuit's functionality and which parts are affected.
3. **Simulate the Fault**: Perform a fault simulation to observe how the {fault} impacts the circuit's behavior.
4. **Formulate Input Vector**: Create an input vector that effectively induces the {fault} by interacting with the identified components.
5. **Predict Expected Output**: Define the expected output when the input vector triggers the {fault}.
6. **Execute Simulation**: Perform a simulation using the input vector to capture the circuit's behavior, resulting in a snapshot.
7. **Confirm Fault Detection**: Compare the simulation snapshot with the expected output to verify the detection of the {fault} and enumerate any other faults detected.
""",
"""CHAIN_OF_THOUGHT:
1. **Review Netlist for Fault Analysis**: Analyze the netlist to pinpoint components and connections that are susceptible to the {fault}.
2. **Understand Fault Dynamics**: Examine how the {fault} affects the circuit's operation and which signals it disrupts.
3. **Simulate the Fault**: Perform a fault simulation to observe how the {fault} impacts the circuit's behavior.
4. **Create Input Vector**: Develop an input vector that specifically targets the {fault}, ensuring it activates the faulty behavior.
5. **Determine Expected Output**: Predict the output that should result when the input vector successfully induces the {fault}.
6. **Simulate Circuit Behavior**: Run a simulation with the input vector to observe the circuit's response and capture the simulation snapshot.
7. **Detect and List Faults**: Compare the simulation snapshot with the expected output to confirm the detection of the {fault} and identify any other faults present.
""",
"""CHAIN_OF_THOUGHT:
1. **Netlist Analysis**: Inspect the netlist to identify components and connections that could be affected by the {fault}.
2. **Fault Impact Evaluation**: Understand how the {fault} alters the circuit's behavior and which signals are disrupted.
3. **Simulate the Fault**: Perform a fault simulation to observe how the {fault} impacts the circuit's behavior.
4. **Input Vector Formulation**: Design an input vector that specifically induces the {fault}, targeting the affected areas.
5. **Expected Output Specification**: Define the expected output when the {fault} is present and the input vector is applied.
6. **Conduct Simulation**: Perform a simulation using the input vector to observe the actual output and capture the snapshot.
7. **Fault Detection Identification**: Compare the simulation results with the expected output to verify the detection of the {fault} and list any additional faults uncovered.
""",
"""CHAIN_OF_THOUGHT:
1. **Examine Netlist Components**: Analyze the netlist to identify parts of the circuit that are related to the {fault}.
2. **Determine Fault Influence**: Assess how the {fault} impacts signal flow and overall circuit performance.
3. **Simulate the Fault**: Perform a fault simulation to observe how the {fault} impacts the circuit's behavior.
4. **Develop Input Vector**: Create an input vector that will trigger the {fault}, ensuring it interacts with the affected components.
5. **Predict Expected Output**: Establish what the output should be when the input vector activates the {fault}.
6. **Simulate Circuit Response**: Run a simulation with the input vector to generate a snapshot of the circuit's actual output.
7. **List Detected Faults**: Analyze the simulation snapshot to confirm the detection of the {fault} and document any other faults identified.
""",
"""CHAIN_OF_THOUGHT:
1. **Netlist Review**: Examine the provided netlist to identify components and connections pertinent to the {fault}.
2. **Fault Mechanism Analysis**: Understand how the {fault} affects the circuit's behavior and which pathways are involved.
3. **Simulate the Fault**: Perform a fault simulation to observe how the {fault} impacts the circuit's behavior.
4. **Input Vector Creation**: Formulate an input vector that specifically targets and induces the {fault} within the circuit.
5. **Expected Output Determination**: Define the expected output that should result when the {fault} is activated by the input vector.
6. **Circuit Simulation**: Execute a simulation using the input vector to observe the circuit's behavior and capture the snapshot.
7. **Fault Detection Verification**: Compare the simulation snapshot with the expected output to verify the detection and coverage of the {fault}.
""",
"""CHAIN_OF_THOUGHT:
1. **Analyze Circuit Structure**: Review the netlist to understand the layout and identify sections related to the {fault}.
2. **Understand Fault Impact**: Examine how the {fault} disrupts normal circuit operations and which signals are affected.
3. **Simulate the Fault**: Perform a fault simulation to observe how the {fault} impacts the circuit's behavior.
4. **Design Test Input**: Develop an input vector that activates the {fault}, ensuring it interacts with the relevant components.
5. **Determine Expected Outcome**: Predict the output that should result from the input vector if the {fault} is present.
6. **Simulate Circuit Behavior**: Run a simulation using the input vector to generate a snapshot of the circuit's actual response.
7. **Identify Detected Faults**: Analyze the simulation snapshot to confirm the detection of the {fault} and list any other faults uncovered.
""",
"""CHAIN_OF_THOUGHT:
1. **Netlist Examination**: Inspect the netlist to identify components and connections relevant to the {fault}.
2. **Fault Analysis**: Determine how the {fault} affects signal flow and circuit functionality.
3. **Simulate the Fault**: Perform a fault simulation to observe how the {fault} impacts the circuit's behavior.
4. **Input Vector Formulation**: Create an input vector that induces the {fault} by interacting with the affected components.
5. **Expected Output Specification**: Establish the expected output behavior when the {fault} is active.
6. **Simulation Execution**: Perform a simulation with the input vector to capture the circuit's behavior, resulting in a snapshot.
7. **Fault Detection Confirmation**: Compare the simulation snapshot with the expected output to verify the detection of the {fault}.
""",
"""CHAIN_OF_THOUGHT:
1. **Review Netlist Details**: Analyze the netlist to comprehend the circuit's configuration and pinpoint areas susceptible to the {fault}.
2. **Characterize the Fault**: Understand the nature of the {fault} and its impact on circuit operations.
3. **Simulate the Fault**: Perform a fault simulation to observe how the {fault} impacts the circuit's behavior.
4. **Craft the Input Vector**: Design an input vector that specifically activates the {fault}, ensuring it influences the targeted components.
5. **Define Expected Output**: Predict the output that should result from the input vector if the {fault} is present.
6. **Conduct Simulation**: Run a simulation using the input vector to obtain a snapshot of the circuit's actual behavior.
7. **Extract Detected Faults**: Analyze the simulation results to confirm the presence and detection of the {fault}.
""",
"""CHAIN_OF_THOUGHT:
1. **Examine the Netlist**: Review the netlist to identify key components and pathways related to the {fault}.
2. **Understand Fault Mechanism**: Analyze how the {fault} affects the circuit's performance and which signals are impacted.
3. **Simulate the Fault**: Perform a fault simulation to observe how the {fault} impacts the circuit's behavior.
4. **Develop the Test Input**: Formulate an input vector that will induce the {fault}, ensuring it triggers the specific faulty behavior.
5. **Establish Expected Behavior**: Define what the circuit's output should be when the {fault} is correctly triggered by the input vector.
6. **Perform Circuit Simulation**: Execute a simulation with the input vector to observe the actual output and generate a snapshot of the circuit's state.
7. **List Detected Faults**: Determine which faults are identified by comparing the simulation snapshot with the expected behavior.
""",
"""CHAIN_OF_THOUGHT:
1. **Analyze the Netlist**: Examine the provided netlist to understand the circuit's structure and identify components related to the {fault}.
2. **Determine the Fault Characteristics**: Understand how the {fault} manifests within the circuit and which components are affected.
3. **Simulate the Fault**: Perform a fault simulation to observe how the {fault} impacts the circuit's behavior.
4. **Generate the Input Vector**: Create an input vector that specifically targets the {fault}, ensuring it activates the faulty behavior.
5. **Predict the Expected Output**: Based on the input vector and the nature of the {fault}, determine the expected output when the fault is present.
6. **Run the Circuit Simulation**: Use the input vector to simulate the circuit and capture the actual behavior, generating the simulation snapshot.
7. **Identify Detected Faults**: Compare the simulation results with the expected output to confirm the detection of the {fault}.
"""]


import torch
def parse_chain_of_thought(contents, cot_block_re, thought_pattern_re):
  thoughts = {}
  for content in contents:
    # Extract text between CHAIN_OF_THOUGHT and SNAPSHOT
    cot_block = cot_block_re.findall(content)
    if cot_block:
      cot = cot_block[0]
      # Find all numbered thoughts
      thought_matches = thought_pattern_re.findall(cot)
      # Group thoughts by number
      for num, content in thought_matches:
        num = int(num)
        thoughts.setdefault(num, [])
        thoughts[num].append(content.strip())
  return thoughts

def calculate_cos_sim(thoughts, model):
  similarities = []
  # Define the sentences
  for thought_idx, (sentence1, sentence2) in thoughts.items():
    # Encode the sentences to get their embeddings
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)

    # Compute the cosine similarity between the embeddings
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    similarities.append(similarity.item())

  return np.sum(similarities)

def thoughts_check(thoughts, fault_net):
  try:
    for i, thought_list in thoughts.items():
      if len(thought_list) == 1:
        thoughts[i] = ["", thought_list[0]]
      if len(thought_list) > 2:
        thoughts[i] = [thought_list[0], thought_list[-1]]
      thoughts[i][1] = thoughts[i][1].format(fault=fault_net)
    return True
  except KeyError:
    return False


# Usage
def cot_reward(prompts: list, completions: list, model: torch.nn.Sequential, cot_block_re: re.Pattern, thought_pattern_re: re.Pattern, fault_re: re.Pattern, reward_per_thought: bool = False):
  similarity_rewards = []
  for prompt, completion in zip(prompts, completions):
    # Randomly select a target chain of thought
    target_thoughts = np.random.choice(cots)

    # Extract the fault and net
    fault = fault_re.findall(prompt)
    if fault:
      fault, net = fault[0]

    try:
      if reward_per_thought:
        # Parse the chain of thought to get thought pairs
        thoughts = parse_chain_of_thought([completion, target_thoughts+"SNAPSHOT"], cot_block_re, thought_pattern_re)
        # Check if there is only one thought for each number
        if thoughts_check(thoughts, f"{fault} {net}"):    
          # Compute the cosine similarity between the thought pairs
          # similarity_rewards.append(calculate_cos_sim(thoughts, model))
          similarity_rewards.append(calculate_cos_sim(thoughts, model))
        else:
          similarity_rewards.append(0)  # No matching thought pairs found
      else:
        cot_block = cot_block_re.findall(completion)
        if cot_block:
          cot_block = cot_block[0]
        else:
          cot_block = ""
        target_thoughts = target_thoughts.replace("CHAIN_OF_THOUGHT:", "").format(fault=f"{fault} {net}")

        embedding1 = model.encode(cot_block, convert_to_tensor=True)
        embedding2 = model.encode(target_thoughts, convert_to_tensor=True)

        # Compute the cosine similarity between the embeddings
        similarity = util.pytorch_cos_sim(embedding1, embedding2)
        similarity_rewards.append(similarity.item()*5)
    except (ValueError, UnboundLocalError):
      similarity_rewards.append(-1)
  return similarity_rewards


# Extract the fault and net
def test_generation_reward(prompts: list, completions: list, netlists: list, fault_re: re.Pattern, simulation_re: re.Pattern, input_vector_re: re.Pattern, expected_output_re: re.Pattern, detected_faults_re: re.Pattern):
  gate_func = {'IB': logic_buf, 'AN': logic_and, 'OR': logic_or, 'XO': logic_xor, 'IV': logic_not, 'ND': logic_nand, 'NR': logic_nor, 'XN': logic_xnor}
  
  rewards = []
  for prompt, completion, netlist in zip(prompts, completions, netlists):
    fault = fault_re.findall(prompt)
    if fault:
      fault, net = fault[0]
      # print("Fault: ", fault, "Net: ", net)

    # Extract the simulation
    pred_simulation = simulation_re.findall(completion)
    if pred_simulation:
      pred_simulation = pred_simulation[0]
      # print("simulation:\n", pred_simulation)

    # Extract the input vector
    pred_input_vector = input_vector_re.findall(completion)
    if pred_input_vector:
      pred_input_vector = pred_input_vector[0]
      # print("Input Vector: ", pred_input_vector)

    # Extract the expected output
    pred_expected_output = expected_output_re.findall(completion)
    if pred_expected_output:
      pred_expected_output = pred_expected_output[0]
      # print("Expected Output: ", pred_expected_output)

    # Extract the detected faults
    pred_detected_faults = detected_faults_re.findall(completion)
    if pred_detected_faults:
      pred_detected_faults = pred_detected_faults[0]
      # print("Detected Faults: ", pred_detected_faults)

    # Calculate the reward
    reward = 0
    if fault and pred_simulation:
      try:
        # Format: Parse the simulation and convert it to a DataFrame
        pred_simulation = pd.read_csv(StringIO(pred_simulation), sep="\s{2,}")
        reward += 2
        # print("1. Correct simulation format: 1")

        # Fault Detection: Check if the fault simulation trigger the requested fault
        reward += 4*int(pred_simulation.loc[net, "Good Machine"] != pred_simulation.loc[net, "Bad Machine"]) - 2
        # print(f"2. LLM Fault Simulation reward: {int(pred_simulation.loc[net, 'Good Machine'] != pred_simulation.loc[net, 'Bad Machine'])} ")
      except:
        # print(f"1-2. Wrong simulation format, reward:0\n{pred_simulation}")
        pred_simulation = None
        reward = 0

    gate_func = {'IB': logic_buf, 'AN': logic_and, 'OR': logic_or, 'XO': logic_xor, 'IV': logic_not, 'ND': logic_nand, 'NR': logic_nor, 'XN': logic_xnor}
    if pred_input_vector and pred_expected_output and fault and net and netlist:
      try:
        fault_simulation, fault_sim_rewards = fault_sim(pred_input_vector, pred_expected_output, f"{fault} {net}", netlist, gate_func, return_rewards=True)
        # print(f"3. Fault Simulation Reward based on test input: {pred_input_vector}, expected output: {pred_expected_output}:\n{fault_sim_rewards}")

        # if LLM simulation and actual simulation have same:
        # r1=+1 input length
        # r2=+1 output length 
        # r3=+1/-1 triggered fault.
        # r4=4*sum(r1,r2,r3)-2 if LLM simulation and actual simulation match 
        r4 = sum(fault_sim_rewards.values())
        reward += 4*r4-2*r4
        
        # r5=+2/-2 if LLM simulation and actual simulation match 
        reward += 4*int(fault_simulation.equals(pred_simulation))-2
      except:
        # print(f"3. Wrong Fault Simulation, Reward:0")
        fault_simulation = None
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
  fault_re = re.compile("sa\d\s+_\d+_", re.DOTALL)
  simulation_re = re.compile("SNAPSHOT:\n```\n(.*?)```\s+INPUT_VECTOR", re.DOTALL)
  input_vector_re = re.compile("INPUT_VECTOR:\s\"(.*?)\"", re.DOTALL)
  expected_output_re = re.compile("EXPECTED_OUTPUT:\s\"(.*?)\"", re.DOTALL)
  detected_faults_re = re.compile("DETECTED_FAULTS:\s\"(.*?)\"", re.DOTALL)

  prompts = [df_cot.loc[0, 'text'].split("<</SYS>>")[1].strip().split("[/INST]")[0]]
  completions = [df_cot.loc[0, 'text'].split("[/INST]")[1].strip()]
  netlists = [df_cot.loc[0, 'netlist']]

  rewards = cot_reward(prompts, completions, model, cot_block_re, thought_pattern_re, fault_re)
  print(f"COT: {rewards}")

  rewards = test_generation_reward(prompts, completions, netlists, fault_re, simulation_re, input_vector_re, expected_output_re, detected_faults_re)
  print(f"Test Generation: {rewards}")

