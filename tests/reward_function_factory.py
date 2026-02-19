import regex as re
import json
from typing import Dict, Any, List
from sympy import symbols, parse_expr
from sympy.core.symbol import Symbol
from pathlib import Path
import sys

# Add the parent directory of atpgllm to sys.path to allow importing from data_preprocessing
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data_preprocessing'))

from fault_sim import OptimizedNetlist, fast_fault_sim
from atpgllm.llm.reward_funcs import (
    extract_json_tool_response_and_convert_to_df, 
    extract_markdown_table, 
    markdown_table_to_dataframe, 
    test_generation_reward
)

# =============================================================================
# REWARD FUNCTION FACTORY
# =============================================================================
class RewardFunctionFactory:
    """
    Factory class that creates and manages reward functions for GRPO training.
    
    This class encapsulates:
    1. Gate function parsing from sim_config.json
    2. Netlist caching (OptimizedNetlist objects) for efficient simulation
    3. Parsing functions for extracting data from model completions
    4. The actual reward function that runs fault simulation
    
    Usage:
        factory = RewardFunctionFactory("sim_config.json")
        reward_fn = factory.create_reward_function()
        # Pass reward_fn to GRPOTrainer
    """
    
    # Regex patterns for parsing model completions
    FAULT_RE = re.compile(r"(sa\d)\s+(\w+)", re.DOTALL)
    INPUT_VECTOR_RE = re.compile(r"INPUT_VECTOR:\s\"(.*?)\"", re.DOTALL)
    EXPECTED_OUTPUT_RE = re.compile(r"EXPECTED_OUTPUT:\s\"(.*?)\"", re.DOTALL)
    DETECTED_FAULTS_RE = re.compile(r"DETECTED_FAULTS:\s\"(.*?)\"", re.DOTALL)
    SYMBOLS_RE = re.compile(r'\w+')
    THINKING_RE = re.compile(r"(?<=<think>)([\s\S]*?)(?=<\/think>)", re.DOTALL)
    TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
    
    # Regex to capture full Verilog declarations
    DECL_RE = re.compile(r"""
        ^\s*
        (?P<kind>input|output|inout|wire|reg)\b      # keyword
        \s*
        (?P<packed>\[\s*\d+\s*:\s*\d+\s*\])?         # optional packed bus, e.g. [15:0]
        \s*
        (?P<rest>[^;]+)                              # list of signals
        ;
    """, re.VERBOSE | re.MULTILINE)
    
    # Regex to capture each signal in the list
    NAME_RE = re.compile(r"""
        ^\s*
        (?P<name>(?:\\[^\s]+)|(?:[A-Za-z_]\w*))      # signal name
        \s*
        (?P<unpacked>\[\s*\d+\s*:\s*\d+\s*\])?       # optional unpacked dimension
        \s*$
    """, re.VERBOSE)
    
    def __init__(self, config_path: str = "sim_config.json", max_cache_size: int = 100):
        """
        Initialize the factory with gate functions from config.
        
        Parameters
        ----------
        config_path: str
            Path to JSON config file containing gate function definitions.
        max_cache_size: int
            Maximum number of netlists to cache. Prevents unbounded memory growth.
            Default is 100 netlists.
        """
        self.config_path = config_path
        self.gate_funcs = self._load_and_parse_gate_functions()
        self._netlist_cache: Dict[str, OptimizedNetlist] = {}
        self._max_cache_size = max_cache_size
    
    def _load_and_parse_gate_functions(self) -> Dict[str, Any]:
        """Load and parse gate functions from config into SymPy expressions."""
        with open(self.config_path, 'r') as f:
            raw_gate_funcs = json.load(f).get("gate_funcs", {})
        
        gate_funcs = {}
        for cell_name, ports in raw_gate_funcs.items():
            if not ports:  # Skip cells with no port definitions
                continue
            gate_funcs[cell_name] = {}
            for port, expr_str in ports.items():
                gate_funcs[cell_name][port] = self._parse_gate_function(expr_str)
        
        return gate_funcs
    
    def _parse_gate_function(self, expr_str: str) -> Dict[str, Any]:
        """
        Parse a boolean expression string into a SymPy symbolic expression.
        
        Args:
            expr_str: Boolean expression string (e.g., "A & B", "~(A | B)")
            
        Returns:
            Dict with parsed function, symbol names, and original expression
        """
        var_names = sorted(set(self.SYMBOLS_RE.findall(expr_str)))
        sym_objs = symbols(' '.join(var_names)) if var_names else []
        if isinstance(sym_objs, Symbol):
            sym_objs = [sym_objs]
        symbol_map = dict(zip(var_names, sym_objs))
        
        return {
            'function': parse_expr(expr_str, local_dict=symbol_map, evaluate=False),
            'symbols': dict.fromkeys(var_names),
            'expr_str': expr_str
        }
    
    def get_or_create_netlist(self, netlist_str: str) -> OptimizedNetlist:
        """
        Get a cached OptimizedNetlist or create and cache a new one.
        
        This provides significant speedup when the same netlist appears
        multiple times in the training data. Cache is limited to max_cache_size
        entries to prevent unbounded memory growth.
        """
        # Use hash of netlist string as cache key
        cache_key = hash(netlist_str)
        
        if cache_key not in self._netlist_cache:
            # Evict oldest entries if cache is full (simple FIFO eviction)
            if len(self._netlist_cache) >= self._max_cache_size:
                # Remove the first (oldest) entry
                oldest_key = next(iter(self._netlist_cache))
                del self._netlist_cache[oldest_key]
            
            self._netlist_cache[cache_key] = OptimizedNetlist(
                netlist_str, 
                self.gate_funcs, 
                self.DECL_RE, 
                self.NAME_RE
            )
        
        return self._netlist_cache[cache_key]
    
    def clear_cache(self):
        """Clear the netlist cache to free memory."""
        self._netlist_cache.clear()
    
    @staticmethod
    def fault_fn(x: str, **kwargs) -> List[tuple]:
        """Extract fault information from prompt."""
        fault = kwargs.get('fault', None)
        if fault is not None:
            x = fault
        return RewardFunctionFactory.FAULT_RE.findall(x)
    
    @staticmethod
    def simulation_fn(x: str) -> List[str]:
        """Extract simulation table from completion."""
        try:
            df = extract_json_tool_response_and_convert_to_df(x)
            return [df.to_string()]
        except Exception as e:
            try:
                x_str = extract_markdown_table(x)
                df = markdown_table_to_dataframe(x_str)
                return [df.to_string()]
            except Exception as e:
                return []

    @staticmethod
    def input_vector_fn(x: str) -> List[str]:
        """Extract input vector from completion."""
        return RewardFunctionFactory.INPUT_VECTOR_RE.findall(x)
    
    @staticmethod
    def expected_output_fn(x: str) -> List[str]:
        """Extract expected output from completion."""
        return RewardFunctionFactory.EXPECTED_OUTPUT_RE.findall(x)
    
    @staticmethod
    def detected_faults_fn(x: str) -> List[str]:
        """Extract detected faults from completion."""
        return RewardFunctionFactory.DETECTED_FAULTS_RE.findall(x)
    
    @staticmethod
    def thinking_fn(x: str) -> List[str]:
        """Extract thinking from completion."""
        return RewardFunctionFactory.THINKING_RE.findall(x)
    
    @staticmethod
    def tool_call_fn(x: str) -> List[str]:
        """Extract tool call from completion."""
        return RewardFunctionFactory.TOOL_CALL_RE.findall(x)
    
    @staticmethod
    def tool_response_fn(x: str) -> List[str]:
        """Extract tool response from completion."""
        return RewardFunctionFactory.TOOL_RESPONSE_RE.findall(x)
    
    def create_reward_function(self) -> callable:
        """
        Create a reward function suitable for GRPOTrainer.
        
        Returns a function with signature:
            reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]
        
        The reward function:
        1. Parses model completions to extract INPUT_VECTOR, EXPECTED_OUTPUT, etc.
        2. Runs actual fault simulation using the model's predicted inputs
        3. Compares predictions with actual simulation results
        4. Returns rewards based on accuracy and fault detection
        """
        # Capture factory state in closure
        gate_funcs = self.gate_funcs
        decl_re = self.DECL_RE
        name_re = self.NAME_RE
        get_or_create_netlist = self.get_or_create_netlist
        
        def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
            """
            Evaluate model-generated test vectors by running actual fault simulation.
            """
            # Build kwargs for test_generation_reward
            reward_kwargs = {
                "fault_fn": self.fault_fn,
                "simulation_fn": self.simulation_fn,
                "input_vector_fn": self.input_vector_fn,
                "expected_output_fn": self.expected_output_fn,
                "detected_faults_fn": self.detected_faults_fn,
                "thinking_fn": self.thinking_fn,
                "tool_call_fn": self.tool_call_fn,
                "tool_response_fn": self.tool_response_fn,
                "eval_mode": False,
                "lib_gate_funcs": gate_funcs,
                "fault_sim": fast_fault_sim,  # Pass the fast_fault_sim function
            }
            
            _ = kwargs.pop('system_content', None)
            _ = kwargs.pop('user_content', None)
            _ = kwargs.pop('reasoning_content', None)
            _ = kwargs.pop('answer_content', None)
            
            # Merge with dataset kwargs
            reward_kwargs.update(kwargs)
            
            # Handle netlists - convert raw strings to OptimizedNetlist objects
            netlists = reward_kwargs.pop('netlist', [])
            
            try:
                reward_kwargs['netlists'] = [
                    get_or_create_netlist(netlist) if isinstance(netlist, str) else netlist
                    for netlist in netlists
                ]
            except Exception as e:
                print(f"Warning: Failed to parse some netlists: {e}")
                return [0.0] * len(prompts)
            
            # Run the reward calculation
            try:
                ret_rewards = test_generation_reward(prompts, completions, **reward_kwargs)
                # Sum all reward components into a single scalar per completion
                ret_rewards = [sum(ret_r.values()) for ret_r in ret_rewards]
            except Exception as e:
                print(f"Warning: Reward calculation failed: {e}")
                import traceback
                traceback.print_exc()
                ret_rewards = [0.0] * len(prompts)
            
            return ret_rewards
        
        return reward_fn
