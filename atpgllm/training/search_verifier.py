"""Final-answer scoring and a per-search deterministic simulator cache.

Training reward parsing is deliberately unchanged. Evaluation passes only the
explicit answer and its matching observation to the existing reward function.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass

from .search_types import BudgetExceeded, CompletionScore

FIELD_RE = re.compile(r'\b(INPUT_VECTOR|EXPECTED_OUTPUT|DETECTED_FAULTS):\s*"([^"]*)"')


def final_fields(text):
    # Reasoning and earlier resolved exchanges cannot define the final answer.
    text = text.rsplit("</tool_response>", 1)[-1].rsplit("</think>", 1)[-1]
    fields = {}
    for name, value in FIELD_RE.findall(text):
        if name in fields:
            raise ValueError(f"Ambiguous final answer: repeated {name}")
        fields[name] = value
    if "INPUT_VECTOR" not in fields:
        raise ValueError("Final answer has no INPUT_VECTOR")
    return fields


def assignment(value, names):
    """Parse full binary assignments without silently accepting duplicate nets."""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("{"):
            tree = ast.parse(value, mode="eval").body
            if not isinstance(tree, ast.Dict):
                raise ValueError("Assignment must be an object")
            pairs = [(ast.literal_eval(k), ast.literal_eval(v)) for k, v in zip(tree.keys, tree.values)]
        else:
            pairs = []
            for item in value.split(","):
                key, bit = item.rsplit(":", 1)
                pairs.append((key.strip().strip("\"'"), bit.strip()))
    elif isinstance(value, dict):
        pairs = list(value.items())
    else:
        raise ValueError("Assignment must be an object or net: bit string")
    out = {}
    for key, bit in pairs:
        if key in out or not isinstance(key, str):
            raise ValueError("Duplicate or invalid net name")
        if type(bit) not in (str, int) or bit not in (0, 1, "0", "1"):
            raise ValueError(f"Nonbinary assignment for {key}")
        out[key] = int(bit)
    if set(out) != set(names):
        raise ValueError(f"Expected exactly these nets: {list(names)}")
    return {name: out[name] for name in names}


@dataclass(frozen=True)
class ProblemContext:
    prompt: str
    record: dict
    problem_id: str
    netlist: object
    fault: str
    doc_id: str
    module: str
    input_nets: tuple
    output_nets: tuple
    simulation_identity: str


class Verifier:
    def __init__(self, reward_factory):
        from ._paths import ensure_data_preprocessing_on_path
        ensure_data_preprocessing_on_path()
        from fault_sim import resolve_fault_sim_runner
        from atpgllm.llm.reward_funcs import test_generation_reward, train_scalar_from_reward_components
        self.reward_factory = reward_factory
        self.fault_sim = resolve_fault_sim_runner()
        self._reward = test_generation_reward
        self._scalar = train_scalar_from_reward_components

    def problem(self, prompt, record):
        from fault_sim import OptimizedNetlist, prepare_netlist
        from .reward_function_factory import RewardFunctionFactory
        field = record.get("netlist", "")
        raw = field.get("netlist", "") if isinstance(field, dict) else field
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Missing authoritative netlist")
        net_hash = hashlib.sha256(raw.encode()).hexdigest()
        doc_id = str(field.get("doc_id", "")) if isinstance(field, dict) else net_hash[:16]
        netlist = prepare_netlist(raw, self.reward_factory.gate_funcs,
                                   RewardFunctionFactory.DECL_RE, RewardFunctionFactory.NAME_RE)
        fault = str(record.get("fault", "")).strip()
        if not re.fullmatch(r"sa[01]\s+\S+", fault):
            raise ValueError("Expected authoritative sa0/sa1 target fault")
        module = str(record.get("module_name", "") or doc_id)
        identity = hashlib.sha256(json.dumps({
            "netlist": net_hash, "module": module, "fault": fault,
            "backend": f"{self.fault_sim.__module__}.{self.fault_sim.__name__}",
            "backend_mode": os.environ.get("FAULT_SIM_BACKEND", "fast"),
            "gates": self.reward_factory.gate_funcs,
        }, sort_keys=True, default=str).encode()).hexdigest()
        return ProblemContext(prompt, record, identity, netlist, fault, doc_id, module,
                              tuple(netlist.input_nets), tuple(netlist.output_nets), identity)

    def simulate(self, problem, vector, context):
        """Tools and final verification use the exact same canonical PO set."""
        key = (problem.simulation_identity, tuple((name, vector[name]) for name in problem.input_nets), problem.output_nets)
        use = context.usage
        if use.simulator_requests >= context.config.max_simulator_requests:
            raise BudgetExceeded("simulator_request_limit")
        use.simulator_requests += 1
        if key in context.cache:
            use.cache_hits += 1
            frame, rewards = context.cache[key]
            return frame.copy(deep=True), dict(rewards)
        if use.simulator_executions >= context.config.max_simulator_executions:
            raise BudgetExceeded("simulator_execution_limit")
        use.simulator_executions += 1
        frame, rewards = self.fault_sim(
            vector, {po: 0 for po in problem.output_nets}, problem.fault,
            problem.netlist, self.reward_factory.gate_funcs,
            module_name=problem.module, return_rewards=True,
        )
        if (frame is None or frame.empty or "error" in frame.columns
                or not {"Good Machine", "Bad Machine"}.issubset(frame.columns)):
            raise RuntimeError("Simulator returned an invalid result")
        context.cache[key] = (frame.copy(deep=True), dict(rewards))
        return frame, rewards

    def tool(self, call, problem, context):
        if call.get("name") != "fault_simulation_tool":
            raise ValueError("Unknown tool")
        args = call.get("arguments")
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be an object")
        if set(args) != {"input_vector", "output_vector", "fault", "doc_id"}:
            raise ValueError("Tool arguments must match the fault_simulation_tool schema")
        if str(args["fault"]).strip() != problem.fault or str(args["doc_id"]) != problem.doc_id:
            raise ValueError("Tool target does not match the problem")
        vector = assignment(args["input_vector"], problem.input_nets)
        assignment(args["output_vector"], problem.output_nets)
        frame, _ = self.simulate(problem, vector, context)
        return {"vector": vector, "result": frame[["Good Machine", "Bad Machine"]].to_json(),
                "identity": problem.simulation_identity}

    @staticmethod
    def failure(status, reason=""):
        return CompletionScore(components={"search_failure_logonly": 1.0,
            "simulator_error_logonly": float(status == "INFRA_ERROR")}, status=status)

    def score_state(self, problem, state, context):
        if state.status != "FINAL":
            return self.failure(state.status), None
        try:
            fields = final_fields(state.final_answer)
            vector = assignment(fields["INPUT_VECTOR"], problem.input_nets)
        except (ValueError, TypeError, SyntaxError) as exc:
            context.event(kind="invalid_final", error=str(exc))
            return self.failure("INVALID"), None
        try:
            frame, sim_rewards = self.simulate(problem, vector, context)
        except BudgetExceeded as exc:
            context.stop_reason = str(exc)
            return self.failure("EXHAUSTED"), vector
        except Exception as exc:
            context.usage.infrastructure_errors += 1
            context.event(kind="simulator_error", error=str(exc))
            return self.failure("INFRA_ERROR"), vector
        from .reward_function_factory import RewardFunctionFactory as RF
        text = "\n".join(f'{name}: "{val}"' for name, val in fields.items())
        matched = next((o for o in reversed(state.observations)
                        if o.get("vector") == vector and o.get("identity") == problem.simulation_identity), None)
        if matched:
            text += "\n<tool_response>" + matched["result"] + "</tool_response>"
        # The already accounted simulation is supplied to the reward code;
        # no extra backend call occurs when it calculates its components.
        try:
            comps = self._reward([problem.prompt], [text],
                netlists=[problem.netlist], fault=[problem.fault], module_name=[problem.module],
                lib_gate_funcs=self.reward_factory.gate_funcs,
                fault_sim=lambda *a, **k: (frame.copy(deep=True), dict(sim_rewards)),
                fault_fn=RF.fault_fn, simulation_fn=RF.simulation_fn,
                input_vector_fn=RF.input_vector_fn, expected_output_fn=RF.expected_output_fn,
                detected_faults_fn=RF.detected_faults_fn, eval_mode=True)[0]
            try:
                assignment(fields.get("EXPECTED_OUTPUT"), problem.output_nets)
            except (ValueError, TypeError, SyntaxError):
                comps["expected_output_acc_logonly"] = 0.0
            return CompletionScore(bool(comps.get("detection", 0)), self._scalar(comps), comps), vector
        except Exception as exc:
            context.usage.infrastructure_errors += 1
            context.event(kind="reward_error", error=str(exc))
            return self.failure("INFRA_ERROR"), vector

    def score_many(self, prompts, completions, records):
        # Compatibility for the model-free random baseline. Training callers do
        # not use this evaluation-specific verifier.
        from .search_types import ConversationState, SearchConfig, SearchContext
        if not len(prompts) == len(completions) == len(records):
            raise ValueError("Verifier inputs must align")
        scores = []
        for p, text, record in zip(prompts, completions, records):
            try:
                problem = self.problem(p, record)
                state = ConversationState(status="FINAL", final_answer=text, readable=text)
                score, _ = self.score_state(problem, state, SearchContext(0, SearchConfig(), 1))
            except Exception:
                score = self.failure("INFRA_ERROR")
            scores.append(score)
        return scores

    def score(self, prompt, completion, record):
        return self.score_many([prompt], [completion], [record])[0]
