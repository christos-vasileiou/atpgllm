
from .utils import *
from .llm import *
from .graph import *
import json

models_causal = {
  'tiny-mistral': "openaccess-ai-collective/tiny-mistral",
  'llama-2': "meta-llama/Llama-2-7b-chat-hf",
  'nousre-llama-2': "NousResearch/Llama-2-7b-chat-hf",
  'llama-2-atpg': "chrivasileiou/LlamaModelForCausalLM-ATPG",
  'llama-2-combin-atpg': "chrivasileiou/LlamaModelForCausalLM-Combin-ATPG",
  'llama-2-atpg-lora': "chrivasileiou/LlamaModelForCausalLM-ATPG-LoRA",
  'llama-2-combin-atpg-lora': "chrivasileiou/LlamaModelForCausalLM-Combin-ATPG-LoRA",
  'test-model': "chrivasileiou/TestModel",
  'codegemma-2b': "google/codegemma-2b",
  'codegemma-7b': "google/codegemma-7b-it",
}

models_seq2seq = {
  't5-tiny': "google/t5-efficient-tiny",
  't5-l': "google-t5/t5-large",
  't5-xl': "google-t5/t5-3b",
}


with open("../tests/deepspeed_config.json", 'r') as f:
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
  "Write a test vector for the circuit \"{module_name}\" that covers the {fault} in the netlist:\n```\n{netlist}\n```\n",
  "Your task is to write a test vector that covers the {fault}:\n\n```\n{netlist}\n```\n ",
  "For the circuit \"{module_name}\", generate a test pattern that covers the {fault} in the netlist:\n```\n{netlist}\n```\n",
]

_assistant_response_faults_list = [
  "The test vector that cover the {fault} is the input vector \"{input_vector}\" and the expected output \"{expected_output}\" and the corresponding simulation snapshot is:\n```\n{snapshot}\n```\n By using this test pattern is possible to detect the faults: \"{detected_faults}\"",
  "The {fault} can be detected by the test input pattern \"{input_vector}\" and the expected output \"{expected_output}\". Here's the simulation result:\n```\n{snapshot}\n```\n This test vector yields the fault list: \"{detected_faults}\"",
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
