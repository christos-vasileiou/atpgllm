from transformers.utils import get_json_schema
import json
from typing import Dict
import regex as re
import ast
import asyncio
import threading
import os
import time
from concurrent.futures import ThreadPoolExecutor

from atpgllm.training._paths import ensure_data_preprocessing_on_path, resolve_sim_config_path
ensure_data_preprocessing_on_path()

from fault_sim import convert_string_to_dict, prepare_netlist, resolve_fault_sim_runner
from tetramax_seats import SimulationError, acquire_timeout_s, per_run_timeout_s


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


def _require_bit_mapping(name, vector):
    """Model-authored vectors must be caught as tool errors: the simulator raises
    AttributeError/IndexError on them, which would abort training as infrastructure."""
    if isinstance(vector, str):
        vector = convert_string_to_dict(vector, sep=':' if ':' in vector else '=')
    if not isinstance(vector, dict):
        raise TypeError(f"{name} must map net names to 0/1")
    bad = [net for net, bit in vector.items()
           if not isinstance(bit, bool) and str(bit).strip().lstrip('-').isdigit()
           and int(bit) not in (0, 1)]
    if bad:
        raise ValueError(f"{name} has non-binary values for nets {bad[:5]}")


async def fault_simulation_tool_handler(input_vector, output_vector, fault, doc_id, netlist,
                                        expected_doc_id=None, expected_fault=None, cancel=None) -> str:
    """Execute off the event loop; cancellation waits for supervised cleanup."""
    if expected_doc_id is not None and str(doc_id) != str(expected_doc_id):
        raise ValueError("Tool document does not match the authoritative problem")
    if expected_fault is not None and fault.strip() != expected_fault.strip():
        raise ValueError("Tool fault does not match the authoritative problem")
    _require_bit_mapping("input_vector", input_vector)
    _require_bit_mapping("output_vector", output_vector)
    cancelled = cancel if cancel is not None else threading.Event()
    def execute():
        runner = resolve_fault_sim_runner()
        if runner.__name__ == 'tetramax_fault_sim':
            model = prepare_netlist(netlist)
            snapshot = runner(input_vector, output_vector, fault, model, cancel=cancelled)
        else:
            from atpgllm.training.reward_function_factory import RewardFunctionFactory as RF
            with resolve_sim_config_path().open(encoding='utf-8') as f:
                gates = json.load(f)
            model = prepare_netlist(netlist, gates, RF.DECL_RE, RF.NAME_RE)
            snapshot = runner(input_vector, output_vector, fault, model, gates, module_name=doc_id)
        if 'error' in snapshot.columns:
            return str(snapshot.loc[0, 'error'])
        return snapshot[['Good Machine', 'Bad Machine']].to_json()
    task = asyncio.create_task(asyncio.to_thread(execute))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled.set()
        try:
            await asyncio.shield(task)
        except Exception:
            pass
        raise


def tool_deadline_s():
    return acquire_timeout_s() + per_run_timeout_s() + 20


def _execute_tool(item, functions, cancelled):
    call, prompt = item
    try:
        name = call.get('name')
        if name not in functions:
            raise ValueError(f'Unknown tool: {name}')
        args = call.get('arguments', {})
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(args, dict) or set(args) != {'input_vector','output_vector','fault','doc_id'}:
            raise ValueError('Tool arguments must match the fault_simulation_tool schema')
        args = dict(args)
        args.update(ToolHelper.context(prompt))
        if name == 'fault_simulation_tool':
            args['cancel'] = cancelled
        fn = functions[name]
        value = fn(**args)
        if asyncio.iscoroutine(value):
            value = asyncio.run(value)
        return str(value), False
    except (ValueError, TypeError, SyntaxError) as exc:
        return f'Tool execution failed: {exc}', False
    except (SimulationError, OSError, MemoryError) as exc:
        # The synchronized abort only carries a boolean across ranks. Preserve
        # the original cause and diagnostics path in this rank's training log.
        print(f'[tools] {type(exc).__name__}: {exc}', flush=True)
        return f'Tool infrastructure failed: {exc}', True
    except Exception as exc:
        # Other errors come from simulating model-authored arguments; aborting
        # every rank on them would let one malformed sample end the run.
        print(f'[tools] {type(exc).__name__} treated as a tool error: {exc}', flush=True)
        return f'Tool execution failed: {exc}', False

def execute_tool_batch(calls, prompts, functions):
    """Concurrent independent tools, preserving order and infrastructure failures."""
    cancelled = threading.Event()
    with ThreadPoolExecutor(max_workers=min(16, max(1,len(calls)))) as executor:
        try:
            return list(executor.map(lambda item: _execute_tool(item, functions, cancelled), zip(calls,prompts)))
        except BaseException:
            cancelled.set()
            raise


class ToolScheduler:
    """Keep slow simulations running while ready trajectories resume generation.

    All collectives occur on the trainer thread. Trajectory indices, token masks,
    log probabilities and full-group optimization stay with the existing loop.
    """
    def __init__(self, functions):
        self.functions = functions
        self.cancelled = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=16)
        self.pending = {}
        self.pipelined = os.environ.get('TMAX_PIPELINED_TOOLS', '0') == '1'

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cancelled.set()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def take(self, trainer, calls, indices, prompts):
        import torch
        for call, index in zip(calls, indices):
            if index in self.pending:
                raise RuntimeError('Duplicate in-flight tool for a trajectory')
            self.pending[index] = (call, self.executor.submit(
                _execute_tool, (call, prompts[index]), self.functions, self.cancelled))
        while True:
            ready = [idx for idx, (_, future) in self.pending.items() if future.done()]
            if not self.pipelined:
                ready = list(self.pending)
                for _, future in self.pending.values():
                    future.result()
                break
            flags = torch.tensor([bool(ready), bool(self.pending)], device=trainer.accelerator.device)
            if trainer.accelerator.num_processes > 1:
                flags = trainer.accelerator.gather(flags).reshape(-1, 2).any(dim=0)
            if flags[0].item() or not flags[1].item():
                break
            time.sleep(0.05)
        selected = [self.pending.pop(index) for index in ready]
        return [call for call, _ in selected], ready, [future.result() for _, future in selected]


def abort_on_simulator_failure(trainer, failed):
    """All DDP ranks stop before continuation or optimization on an infra failure."""
    import torch
    flag = torch.tensor([int(failed)], device=trainer.accelerator.device)
    if trainer.accelerator.num_processes > 1:
        flag = trainer.accelerator.gather(flag)
    if flag.any().item():
        raise SimulationError('TetraMAX infrastructure failure on a rank; no update performed. '
                              'Inspect simulator diagnostics and resume after recovery.')


# =============================================================================
# TOOL HELPER
# =============================================================================
class ToolHelper:
    netlist_re = re.compile(r"\{\s*['\"]doc_id['\"]\s*:\s*['\"][^\"\']+['\"]\s*,\s*['\"]netlist['\"]\s*:\s*(?:'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")\s*\}")
    
    @classmethod
    def context(cls, prompt):
        if isinstance(prompt, list):
            prompt = '\n'.join(str(m.get('content','')) for m in prompt if m.get('role') == 'user')
        document = cls.get_document(prompt)
        if not isinstance(document, dict):
            raise ValueError(document)
        # The authoritative instruction precedes the embedded Verilog document.
        match = re.search(r'\b(sa[01])\s+([^\s\"\']+)', prompt)
        if match is None:
            raise ValueError('Missing authoritative target fault')
        return {'netlist':document['netlist'], 'expected_doc_id':document['doc_id'],
                'expected_fault':match[1]+' '+match[2]}

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
