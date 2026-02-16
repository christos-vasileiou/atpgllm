from transformers.utils import get_json_schema
import json
from typing import Dict
from pathlib import Path
import sys
# Add the parent directory of atpgllm to sys.path to allow importing from data_preprocessing
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data_preprocessing'))

from fault_sim import OptimizedNetlist, fast_fault_sim
from reward_function_factory import RewardFunctionFactory

# =============================================================================
# TOOL DEFINITIONS
# =============================================================================
async def fault_simulation_tool(input_vector: str | Dict[str, int], output_vector: str | Dict[str, int], fault: str, netlist: str) -> str:
    """
    Perform fault simulation on a netlist injecting the specified fault.
    
    Args:
        input_vector: dictionary of input nets and their values
        output_vector: dictionary of output nets and their values
        fault: string representation of the fault to inject
        netlist: string representation of the netlist to simulate
    """
    if isinstance(input_vector, str):
        with open('sim_config.json', 'r') as f:
            gate_func = json.load(f)
    else:
        try:
            with open('sim_config.json', 'r') as f:
                gate_func = json.load(f)
        except:
            return {"error": "I cannot find the dictionary of the gate functions."}
    optimized_netlist = OptimizedNetlist(netlist, gate_func=gate_func, decl_re=RewardFunctionFactory.DECL_RE, name_re=RewardFunctionFactory.NAME_RE)
    snapshot = fast_fault_sim(input_vector, output_vector, fault, optimized_netlist, gate_func, return_rewards=False)
    return snapshot[['Good Machine', 'Bad Machine']].to_json()


FAULT_SIMULATION_TOOL = get_json_schema(fault_simulation_tool)
TOOLS = [FAULT_SIMULATION_TOOL]