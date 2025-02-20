
from .utils import *
from .llm import *
from .graph import *
import json

models_causal = {
  'tiny-mistral': "openaccess-ai-collective/tiny-mistral",
  'llama-2': "meta-llama/Llama-2-7b-chat-hf",
  'llama-2-simaw': "chrivasileiou/SimulationAwareGeneration",
  'llama-2-simaw-2': "chrivasileiou/SimulationAwareGeneration-2",
  'llama-2-atpg': "chrivasileiou/LlamaModelForCausalLM-ATPG",
  'llama-2-combin-atpg': "chrivasileiou/LlamaModelForCausalLM-Combin-ATPG",
  'llama-2-atpg-lora': "chrivasileiou/LlamaModelForCausalLM-ATPG-LoRA",
  'llama-2-combin-atpg-lora': "chrivasileiou/LlamaModelForCausalLM-Combin-ATPG-LoRA",
  "llama-2-cot": "chrivasileiou/COTSimAwareATPG",
  'test-model': "chrivasileiou/TestModel",
  'test-model-v2': "chrivasileiou/TestModel-2",
  'codegemma-2b': "google/codegemma-2b",
  'codegemma-7b': "google/codegemma-7b-it",
}

models_seq2seq = {
  't5-tiny': "google/t5-efficient-tiny",
  't5-l': "google-t5/t5-large",
  't5-xl': "google-t5/t5-3b",
}

current_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(current_dir, "atpgllm", "deepspeed_config.json")

with open(config_path, 'r') as f:
  deepspeed_config = json.load(f)

_system_prompts = [
  "You are an ATPG tool applied to integrated circuits for fault testing.",
  # "You play the role of a Automated Test Pattern Generation (ATPG) and you need to generate patterns.",
  # "Assume you are an ATPG tool and you generate test vectors (patters) to detect faults of ICs.",
  # "Assume you operate as a Test Pattern Generator for Integrated Circuits (ICs)",
  # "Assume you are a Test Vector Generator for Integrated Circuits (ICs)",
  # "Your goal is to generate test programs for given netlists",
  # "Pretend you are an expert on test program and test vector (patterns) generation",
]

_user_prompt_dict = {'coverage': '',
                    'module_name': '',
                    'coverage_type': '',
                    'faults_list': '',
                    'fault_type': '',
                    'netlist': '',
                    'input_vector': '',
                    'expected_output': '',
                    'test_vectors': '',
                    'snapshot': '',
                    'detected_faults': '',
                    }

# Faults List
_training_prompts_faults_list = [
  "Create a test vector for the circuit \"{module_name}\" to detect the \"{fault}\" in the following netlist:\n```\n{netlist}\n```\n",
  "Please generate a test pattern that targets the \"{fault}\" for the circuit \"{module_name}\" using this netlist:\n```\n{netlist}\n```\n",
  "Develop a test vector aimed at covering the \"{fault}\" in the circuit \"{module_name}\" based on the netlist below:\n```\n{netlist}\n```\n",
  "For the module \"{module_name}\", design a test vector that identifies the \"{fault}\" within this netlist:\n```\n{netlist}\n```\n",
  "I need a test pattern for the circuit \"{module_name}\" that can uncover the \"{fault}\" as specified in the netlist:\n```\n{netlist}\n```\n",
  "Generate a test vector targeting the \"{fault}\" for the circuit named \"{module_name}\" using the provided netlist:\n```\n{netlist}\n```\n",
  "Can you write a test pattern for \"{module_name}\" that covers the \"{fault}\" in the following netlist?\n```\n{netlist}\n```\n",
  "Please formulate a test vector for the circuit \"{module_name}\" to address the \"{fault}\" based on this netlist:\n```\n{netlist}\n```\n",
  "Construct a test pattern for \"{module_name}\" that targets the \"{fault}\" using the given netlist:\n```\n{netlist}\n```\n",
  "Design a test vector for the module \"{module_name}\" aimed at detecting the \"{fault}\" in the subsequent netlist:\n```\n{netlist}\n```\n",
]


_assistant_response_faults_list = [
    """CHAIN_OF_THOUGHT:
1. **Analyze the Netlist**: Examine the provided netlist to understand the circuit's structure and identify components related to the {fault}.
2. **Determine the Fault Characteristics**: Understand how the {fault} manifests within the circuit and which components are affected.
3. **Generate the Input Vector**: Create an input vector that specifically targets the {fault}, ensuring it activates the faulty behavior.
4. **Predict the Expected Output**: Based on the input vector and the nature of the {fault}, determine the expected output when the fault is present.
5. **Simulate the Circuit**: Run a simulation using the input vector to capture the actual behavior and generate the simulation snapshot.
6. **Identify Detected Faults**: Compare the simulation results with the expected output to confirm the detection of the {fault}.

INPUT_VECTOR: "{input_vector}"
EXPECTED_OUTPUT: "{expected_output}"
SNAPSHOT:
```
{snapshot}
```
DETECTED_FAULTS: "{detected_faults}"
""",
    """CHAIN_OF_THOUGHT:
1. **Examine the Netlist**: Review the netlist to identify key components and pathways related to the {fault}.
2. **Understand Fault Mechanism**: Analyze how the {fault} affects the circuit's performance and which signals are impacted.
3. **Develop the Test Input**: Formulate an input vector that will induce the {fault}, ensuring it triggers the specific faulty behavior.
4. **Establish Expected Behavior**: Define what the circuit's output should be when the {fault} is correctly triggered by the input vector.
5. **Perform Circuit Simulation**: Execute a simulation with the input vector to observe the actual output and generate a snapshot of the circuit's state.
6. **List Detected Faults**: Determine which faults are identified by comparing the simulation snapshot with the expected behavior.

INPUT_VECTOR: "{input_vector}"
EXPECTED_OUTPUT: "{expected_output}"
SNAPSHOT:
{snapshot}
DETECTED_FAULTS: "{detected_faults}"
""",
    """CHAIN_OF_THOUGHT:
1. **Review Netlist Details**: Analyze the netlist to comprehend the circuit's configuration and pinpoint areas susceptible to the {fault}.
2. **Characterize the Fault**: Understand the nature of the {fault} and its impact on circuit operations.
3. **Craft the Input Vector**: Design an input vector that specifically activates the {fault}, ensuring it influences the targeted components.
4. **Define Expected Output**: Predict the output that should result from the input vector if the {fault} is present.
5. **Conduct Simulation**: Run a simulation using the input vector to obtain a snapshot of the circuit's actual behavior.
6. **Extract Detected Faults**: Analyze the simulation results to confirm the presence and detection of the {fault}.

INPUT_VECTOR: "{input_vector}"
EXPECTED_OUTPUT: "{expected_output}"
SNAPSHOT:
{snapshot}
DETECTED_FAULTS: "{detected_faults}"
""",
    """CHAIN_OF_THOUGHT:
1. **Netlist Examination**: Inspect the netlist to identify components and connections relevant to the {fault}.
2. **Fault Analysis**: Determine how the {fault} affects signal flow and circuit functionality.
3. **Input Vector Formulation**: Create an input vector that induces the {fault} by interacting with the affected components.
4. **Expected Output Specification**: Establish the expected output behavior when the {fault} is active.
5. **Simulation Execution**: Perform a simulation with the input vector to capture the circuit's behavior, resulting in a snapshot.
6. **Fault Detection Confirmation**: Compare the simulation snapshot with the expected output to verify the detection of the {fault}.

INPUT_VECTOR: "{input_vector}"
EXPECTED_OUTPUT: "{expected_output}"
SNAPSHOT:
{snapshot}
DETECTED_FAULTS: "{detected_faults}"
""",
    """CHAIN_OF_THOUGHT:
1. **Analyze Circuit Structure**: Review the netlist to understand the layout and identify sections related to the {fault}.
2. **Understand Fault Impact**: Examine how the {fault} disrupts normal circuit operations and which signals are affected.
3. **Design Test Input**: Develop an input vector that activates the {fault}, ensuring it interacts with the relevant components.
4. **Determine Expected Outcome**: Predict the output that should result from the input vector if the {fault} is present.
5. **Simulate Circuit Behavior**: Run a simulation using the input vector to generate a snapshot of the circuit's actual response.
6. **Identify Detected Faults**: Analyze the simulation snapshot to confirm the detection of the {fault} and list any other faults uncovered.

INPUT_VECTOR: "{input_vector}"
EXPECTED_OUTPUT: "{expected_output}"
SNAPSHOT:
{snapshot}
DETECTED_FAULTS: "{detected_faults}"
""",
    """CHAIN_OF_THOUGHT:
1. **Netlist Review**: Examine the provided netlist to identify components and connections pertinent to the {fault}.
2. **Fault Mechanism Analysis**: Understand how the {fault} affects the circuit's behavior and which pathways are involved.
3. **Input Vector Creation**: Formulate an input vector that specifically targets and induces the {fault} within the circuit.
4. **Expected Output Determination**: Define the expected output that should result when the {fault} is activated by the input vector.
5. **Circuit Simulation**: Execute a simulation using the input vector to observe the circuit's behavior and capture the snapshot.
6. **Fault Detection Verification**: Compare the simulation snapshot with the expected output to verify the detection and coverage of the {fault}.

INPUT_VECTOR: "{input_vector}"
EXPECTED_OUTPUT: "{expected_output}"
SNAPSHOT:
{snapshot}
DETECTED_FAULTS: "{detected_faults}"
""",
    """CHAIN_OF_THOUGHT:
1. **Examine Netlist Components**: Analyze the netlist to identify parts of the circuit that are related to the {fault}.
2. **Determine Fault Influence**: Assess how the {fault} impacts signal flow and overall circuit performance.
3. **Develop Input Vector**: Create an input vector that will trigger the {fault}, ensuring it interacts with the affected components.
4. **Predict Expected Output**: Establish what the output should be when the input vector activates the {fault}.
5. **Simulate Circuit Response**: Run a simulation with the input vector to generate a snapshot of the circuit's actual output.
6. **List Detected Faults**: Analyze the simulation snapshot to confirm the detection of the {fault} and document any other faults identified.

INPUT_VECTOR: "{input_vector}"
EXPECTED_OUTPUT: "{expected_output}"
SNAPSHOT:
{snapshot}
DETECTED_FAULTS: "{detected_faults}"
""",
    """CHAIN_OF_THOUGHT:
1. **Netlist Analysis**: Inspect the netlist to identify components and connections that could be affected by the {fault}.
2. **Fault Impact Evaluation**: Understand how the {fault} alters the circuit's behavior and which signals are disrupted.
3. **Input Vector Formulation**: Design an input vector that specifically induces the {fault}, targeting the affected areas.
4. **Expected Output Specification**: Define the expected output when the {fault} is present and the input vector is applied.
5. **Conduct Simulation**: Perform a simulation using the input vector to observe the actual output and capture the snapshot.
6. **Fault Detection Identification**: Compare the simulation results with the expected output to verify the detection of the {fault} and list any additional faults uncovered.

INPUT_VECTOR: "{input_vector}"
EXPECTED_OUTPUT: "{expected_output}"
SNAPSHOT:
{snapshot}
DETECTED_FAULTS: "{detected_faults}"
""",
    """CHAIN_OF_THOUGHT:
1. **Review Netlist for Fault Analysis**: Analyze the netlist to pinpoint components and connections that are susceptible to the {fault}.
2. **Understand Fault Dynamics**: Examine how the {fault} affects the circuit's operation and which signals it disrupts.
3. **Create Input Vector**: Develop an input vector that specifically targets the {fault}, ensuring it activates the faulty behavior.
4. **Determine Expected Output**: Predict the output that should result when the input vector successfully induces the {fault}.
5. **Simulate Circuit Behavior**: Run a simulation with the input vector to observe the circuit's response and capture the simulation snapshot.
6. **Detect and List Faults**: Compare the simulation snapshot with the expected output to confirm the detection of the {fault} and identify any other faults present.

INPUT_VECTOR: "{input_vector}"
EXPECTED_OUTPUT: "{expected_output}"
SNAPSHOT:
{snapshot}
DETECTED_FAULTS: "{detected_faults}"
""",
    """CHAIN_OF_THOUGHT:
1. **Identify Relevant Components in Netlist**: Examine the netlist to locate components and connections related to the {fault}.
2. **Analyze Fault Mechanism**: Understand how the {fault} impacts the circuit's functionality and which parts are affected.
3. **Formulate Input Vector**: Create an input vector that effectively induces the {fault} by interacting with the identified components.
4. **Predict Expected Output**: Define the expected output when the input vector triggers the {fault}.
5. **Execute Simulation**: Perform a simulation using the input vector to capture the circuit's behavior, resulting in a snapshot.
6. **Confirm Fault Detection**: Compare the simulation snapshot with the expected output to verify the detection of the {fault} and enumerate any other faults detected.

INPUT_VECTOR: "{input_vector}"
EXPECTED_OUTPUT: "{expected_output}"
SNAPSHOT:
{snapshot}
DETECTED_FAULTS: "{detected_faults}"
""",
]


# Fault/Test Coverage
_training_prompts_coverage = [
  "Achieve over {coverage} {coverage_type} coverage for the circuit \"{module_name}\" with the netlist:\n\n```\n{netlist}\n```\n",
  "Target for over {coverage} {coverage_type} coverage for the below circuit:\n\n```{netlist}\n```\n",
  # "Write test vectors for the Integrated Circuit \"{module_name}\" that cover the stuck@ faults achieving over {coverage}:\n{netlist}",
  # "Write patterns for IC {module_name}, targeting for over {coverage} {coverage_type} coverage at the netlist:\n```{netlist}```",
  # "Given your expertise in IC testing, your task is to write test vectors that can identify {fault_type} faults in ICs and cover over {coverage}",
  # "You are generating test vectors for a new Integrated Circuit, focused on detecting functional faults (stuck@) ",
]

_assistant_response_coverage = [
  "By applying the test patterns: {test_vectors}\nWe achieve {coverage} {coverage_type} coverage. The simulation snapshot is:\n```\n{snapshot}\n```\nTherefore, the detected faults are: {detected_faults}",
  "The test patterns: {test_vectors}\nYield {coverage} {coverage_type} coverage. The simulation snapshot is as follows:\n```\n{snapshot}\n```\nThus, the faults detected are: {detected_faults}"
]

_details = [
  "\n\n",
  "\n\nPlease wrap the test vectors in ```.",
  "\n\nPlease wrap the result in ```.",
  "\n\nPlease wrap the patterns in ```.",
  "\n\nPlease wrap the test program in ```.",
]

_extras = [
  "",
  "Please outline the test vectors step by step, ensuring coverage for a range of possible issues.",
  "Describe the test vectors you would write for this IC, detailing the types of faults and defects each vector is intended to uncover.",
  "Develop test vectors specifically aimed at uncovering these stuck-at faults.",
  "Focus on the key principles and methodologies that should be considered in the test vector development process.",
  "Please think carefully.",
]


chat_template = """{% for message_group in messages %}
  {% if message_group[0]['role'] == 'system' %}
    {% set loop_messages = message_group[1:] %}
    {% set system_message = message_group[0]['content'] %}
  {% else %}
    {% set loop_messages = message_group %}
    {% set system_message = false %}
  {% endif %}
  {% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
      {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if loop.index0 == 0 and system_message != false %}
      {% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}
    {% else %}
      {% set content = message['content'] %}
    {% endif %}
    {% if message['role'] == 'user' %}
      {{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
      {{ ' '  + content.strip() + ' ' + eos_token }}
    {% endif %}
  {% endfor %}
{% endfor %}
"""

stil_template = """STIL 1.0 {{ Design 2005; }}
Signals {{
   {signals}
}}
SignalGroups {{
   {signalgroups}
}}
Timing {{
   WaveformTable "_default_WFT_" {{
      Period '100ns';
      Waveforms {{
         "_default_In_Timing_" {{ 0 {{ '0ns' D; }} }}
         "_default_In_Timing_" {{ 1 {{ '0ns' U; }} }}
         "_default_In_Timing_" {{ Z {{ '0ns' Z; }} }}
         "_default_In_Timing_" {{ N {{ '0ns' N; }} }}
         "_default_Out_Timing_" {{ X {{ '0ns' X; }} }}
         "_default_Out_Timing_" {{ H {{ '0ns' X; '40ns' H; }} }}
         "_default_Out_Timing_" {{ T {{ '0ns' X; '40ns' T; }} }}
         "_default_Out_Timing_" {{ L {{ '0ns' X; '40ns' L; }} }}
      }}
   }}
}}
ScanStructures {{
   // Uncomment and modify the following to suit your design
   // ScanChain "chain_name" {{ ScanIn "chain_input_name"; ScanOut "chain_output_name"; }}
}}
PatternBurst "_burst_" {{
   PatList {{ "_pattern_" {{
   }}
}}}}
PatternExec {{
   PatternBurst "_burst_";
}}
Procedures {{
   "capture" {{
      W "_default_WFT_";
      C {{ "_po"={_po}; }}
      "forcePI": V {{ "_pi"={forcePI}; }}
      "measurePO": V {{ "_po"={measurePO}; }}
   }}
   // Uncomment and modify the following to suit your design
   // load_unload {{
      // V {{ }} // force clocks off and scan enable pins active
      // Shift {{ V {{ _si=#; _so=#; }}}} // pulse shift clocks
   // }}
}}
MacroDefs {{
}}
Pattern "_pattern_" {{
   {patterns}
}}"""

def template_signals_groups(typo, x):
  return f"{typo} = \'\"" + "\" + \"".join(x) + f"\"\'; // #signals={len(x)}\n"

pattern_template = """\"pattern {i}\": Call \"capture\" {{ 
      \"_pi\"={pi}; \"_po\"={po}; }}"""
