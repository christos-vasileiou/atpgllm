from transformers.utils import get_json_schema
import json
import os
from typing import Dict
from pathlib import Path
import sys
import regex as re
import ast

# Add the parent directory of atpgllm to sys.path to allow importing from data_preprocessing
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data_preprocessing'))

from fault_sim import OptimizedNetlist, fast_fault_sim
from reward_function_factory import RewardFunctionFactory


def _resolve_sim_config_path() -> Path:
    """Config next to this module (stable regardless of cwd). Override with SIM_CONFIG."""
    override = os.environ.get("SIM_CONFIG")
    if not override:
        return Path(__file__).resolve().parent / "sim_config.json"
    p = Path(override).expanduser()
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent / p


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================
def fault_simulation_tool(input_vector: str | Dict[str, int], output_vector: str | Dict[str, int], fault: str, doc_id: str) -> str:
    """
    Perform fault simulation on a netlist injecting the specified fault.
    
    Args:
        input_vector: dictionary of input nets and their values
        output_vector: dictionary of output nets and their values
        fault: string representation of the fault to inject
        doc_id: Netlist ID to simulate
    """
    pass


# Get JSON schema for the model to learn the tool calling format (e.g. <tool_call>{"name": "fault_simulation_tool", "arguments": {}})</tool_call>)
# NOTE: Trick here is that the model will learn the tool calling format, but the tool handler will actually execute the tool. 
# Trick model. No need to generate netlist as input argument.
FAULT_SIMULATION_TOOL = get_json_schema(fault_simulation_tool)
TOOLS = [FAULT_SIMULATION_TOOL]


async def fault_simulation_tool_handler(input_vector: str | Dict[str, int], output_vector: str | Dict[str, int], fault: str, doc_id: str, netlist: str) -> str:
    try:
        with _resolve_sim_config_path().open("r", encoding="utf-8") as f:
            gate_func = json.load(f)
    except (OSError, json.JSONDecodeError):
        if isinstance(input_vector, str):
            raise
        return {"error": "I cannot find the dictionary of the gate functions."}
    optimized_netlist = OptimizedNetlist(netlist, gate_func=gate_func, decl_re=RewardFunctionFactory.DECL_RE, name_re=RewardFunctionFactory.NAME_RE)
    snapshot = fast_fault_sim(input_vector, output_vector, fault, optimized_netlist, gate_func, return_rewards=False)
    if "error" in snapshot.columns:
        return snapshot.loc[0, "error"]
    return snapshot[['Good Machine', 'Bad Machine']].to_json()


# =============================================================================
# TOOL HELPER
# =============================================================================
class ToolHelper:
    netlist_re = re.compile(r"\{\s*['\"]doc_id['\"]\s*:\s*['\"][a-fA-F0-9]+['\"]\s*,\s*['\"]netlist['\"]\s*:\s*(?:'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")\s*\}")
    
    @classmethod
    def get_document(cls, prompt: str) -> str:
        match = cls.netlist_re.search(prompt)
        if match:
            try:
                return ast.literal_eval(match.group(0))
            except:
                return "Could not parse the document."
        return "Could not find the document."
    
    @classmethod
    def get_netlist(cls, prompt: str) -> str:
        document = cls.get_document(prompt)
        if isinstance(document, dict):
            return document.get("netlist", "Could not find the netlist in the document.")
        return "Could not find the netlist in the document."
    
    @classmethod
    def get_doc_id(cls, prompt: str) -> str:
        document = cls.get_document(prompt)
        if isinstance(document, dict):
            return document.get("doc_id", "Could not find the doc_id in the document.")
        return "Could not find the doc_id in the document."
