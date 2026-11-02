"""Contract atpg-rewards-staged (profile ``staged``): propagation and controlling stages,
answer-verified detection and fidelity, structural format.

The circuit is a chain ``n1 = p & s1``, ``y = n1 & s2`` with the target fault ``sa0 p``
and an unused input ``u``. The stub simulator evaluates it in three-valued logic.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import atpgllm.llm.reward_funcs as reward_funcs
from atpgllm.llm.propagation_signals import attempt_signals, sensitize, SignalGraph
from atpgllm.llm.reward_funcs import (
    ATPG_GDPO_OBJECTIVE_KEYS,
    STAGED_OBJECTIVE_KEYS,
    STAGED_OBJECTIVE_WEIGHTS,
    active_reward_objectives,
    train_scalar_from_reward_components,
    test_generation_grpo_reward as score_generation,
    vector_key,
)

PROMPT = "target sa0 p<|im_start|>assistant\n<think>\n"

DETECTS = "p:1,s1:1,s2:1,u:x"
DETECTS_NO_X = "p:1,s1:1,s2:1"
BLOCKED_LATE = "p:1,s1:1,s2:0,u:x"    # effect reaches n1, blocked at y
BLOCKED_EARLY = "p:1,s1:0,s2:1,u:x"   # effect stays on p, blocked at n1
INACTIVE = "p:0,s1:1,s2:1,u:x"        # the fault is not excited


@pytest.fixture(autouse=True)
def staged_profile(monkeypatch):
    monkeypatch.setenv("TMAX_REWARD_PROFILE", "staged")


# --- circuit -------------------------------------------------------------------------

def _and(a, b):
    if a == 0 or b == 0:
        return 0
    return 1 if a == 1 and b == 1 else "x"


def _bit(value):
    return "x" if value == "x" else int(value)


def _frame(vector):
    values = dict(vector_key(vector) or ())
    p, s1, s2 = (_bit(values.get(net, "x")) for net in ("p", "s1", "s2"))
    n1 = _and(p, s1)
    good = {"p": p, "s1": s1, "s2": s2, "u": _bit(values.get("u", "x")), "n1": n1, "y": _and(n1, s2)}
    n1_bad = _and(0, s1)
    bad = dict(good, p=0, n1=n1_bad, y=_and(n1_bad, s2))
    nets = list(good)
    return pd.DataFrame(
        {"Good Machine": [good[n] for n in nets], "Bad Machine": [bad[n] for n in nets],
         "PIs": [n in ("p", "s1", "s2", "u") for n in nets], "POs": [n == "y" for n in nets]},
        index=nets,
    )


def _lut(function, arity):
    return [int(function(*[(index >> bit) & 1 for bit in range(arity)])) for index in range(2 ** arity)]


AND2 = _lut(lambda a, b: a & b, 2)


def _netlist():
    return SimpleNamespace(
        input_nets=["p", "s1", "s2", "u"], output_nets=["y"],
        instructions=[
            ("gate", "n1", AND2, ("A", "B"), {"A": "p", "B": "s1"}, {"p", "s1"}),
            ("gate", "y", AND2, ("A", "B"), {"A": "n1", "B": "s2"}, {"n1", "s2"}),
        ],
    )


# --- trajectories --------------------------------------------------------------------

def _call(vector):
    return {
        "arguments": {"input_vector": vector, "output_vector": "y=1", "fault": "sa0 p", "doc_id": "d"},
        "result": _frame(vector)[["Good Machine", "Bad Machine"]].to_json(),
    }


def _call_turn():
    return ("Drive the fault.\n</think>\n\nCalling the tool.\n"
            "<tool_call>\n<function=fault_simulation_tool>\n</function>\n</tool_call>")


def _answer(vector, output="y:1", faults="sa0 p"):
    return (f'</think>Done. INPUT_VECTOR:"{vector}" EXPECTED_OUTPUT:"{output}" '
            f'DETECTED_FAULTS:"{faults}"')


def _trace(call_vectors, answer):
    return {"turns": [_call_turn() for _ in call_vectors] + [answer],
            "calls": [_call(v) for v in call_vectors]}


def _extract(label):
    def extract(text):
        return [part.split('"', 1)[0] for part in text.split(f'{label}:"')[1:]]
    return extract


def _score(trace):
    simulated = []

    def simulator(input_vector, expected_output, fault, netlist, gate_func, **kwargs):
        simulated.append(input_vector)
        return _frame(input_vector), {}

    rows = score_generation(
        [PROMPT], [""], netlists=[_netlist()], tool_trace=[trace],
        fault_fn=lambda *_a, **_k: [("sa0", "p")], simulation_fn=lambda _: [],
        input_vector_fn=_extract("INPUT_VECTOR"), expected_output_fn=_extract("EXPECTED_OUTPUT"),
        detected_faults_fn=_extract("DETECTED_FAULTS"), lib_gate_funcs={}, fault_sim=simulator,
    )
    return rows[0], simulated


# --- profile and contract --------------------------------------------------------------

def test_staged_profile_exposes_seven_objectives():
    assert active_reward_objectives() == (STAGED_OBJECTIVE_KEYS, STAGED_OBJECTIVE_WEIGHTS)
    row, _ = _score(_trace([DETECTS], _answer(DETECTS)))
    # GDPO selects the objective columns by name, so the order of the row does not matter.
    assert {k for k in row if not k.endswith("_logonly")} == set(STAGED_OBJECTIVE_KEYS)


def test_v4_profile_is_untouched(monkeypatch):
    monkeypatch.delenv("TMAX_REWARD_PROFILE")
    assert active_reward_objectives()[0] == ATPG_GDPO_OBJECTIVE_KEYS
    row, _ = _score(_trace([DETECTS], _answer(DETECTS)))
    assert tuple(k for k in row if not k.endswith("_logonly")) == ATPG_GDPO_OBJECTIVE_KEYS


def test_provenance_names_the_contract(monkeypatch):
    from atpgllm.training.simulator_provenance import runtime_provenance
    staged = runtime_provenance()
    assert (staged["reward_contract"], staged["profile"]) == ("atpg-rewards-staged", "staged")
    monkeypatch.delenv("TMAX_REWARD_PROFILE")
    v4 = runtime_provenance()
    assert (v4["reward_contract"], v4["profile"]) == ("atpg-rewards-v4", "full")


def test_scalar_fallback_uses_the_staged_weights():
    row = dict.fromkeys(STAGED_OBJECTIVE_KEYS, 1.0)
    assert train_scalar_from_reward_components(row) == pytest.approx(sum(STAGED_OBJECTIVE_WEIGHTS))


# --- signals ---------------------------------------------------------------------------

def _signals(vector, activated=True, detected=False):
    frame = _frame(vector)
    return attempt_signals(SignalGraph(_netlist()), frame["Good Machine"].to_dict(),
                           frame["Bad Machine"].to_dict(), "p", activated, detected)


def test_propagation_grows_with_the_distance_the_effect_travels():
    assert _signals(INACTIVE, activated=False)["propagation"] == 0.0
    early, late = _signals(BLOCKED_EARLY), _signals(BLOCKED_LATE)
    assert early["propagation"] == pytest.approx(1 / 3)
    assert late["propagation"] == pytest.approx(2 / 3)
    assert early["depth"] < late["depth"] < 1.0
    assert _signals(DETECTS, detected=True)["propagation"] == 1.0


def test_controlling_counts_correct_side_inputs_of_the_reached_gates():
    assert _signals(BLOCKED_EARLY)["controlling"] == 0.0           # s1=0 blocks n1
    assert _signals(BLOCKED_LATE)["controlling"] == pytest.approx(1 / 2)   # s1 right, s2 wrong
    assert _signals(DETECTS, detected=True)["controlling"] == 1.0


def test_sensitization_is_convex_and_anchored():
    assert sensitize(0.0) == 0.0 and sensitize(1.0) == 1.0
    assert sensitize(0.5) < 0.5 * sensitize(1.0)


def _single_gate(lut, arity, values):
    """One gate with the fault effect on input A; ``values`` are the side-input values."""
    syms = tuple("ABCDEF"[:arity])
    nets = {s: f"i{s}" for s in syms}
    netlist = SimpleNamespace(
        output_nets=["o"],
        instructions=[("gate", "o", lut, syms, nets, set(nets.values()))],
    )
    good = {"iA": 1, "o": 0}
    bad = {"iA": 0, "o": 0}
    for sym, value in zip(syms[1:], values):
        good[nets[sym]] = bad[nets[sym]] = value
    return attempt_signals(SignalGraph(netlist), good, bad, "iA", True, False)["controlling"]


def test_side_inputs_of_a_three_input_and():
    and3 = _lut(lambda a, b, c: a & b & c, 3)
    assert _single_gate(and3, 3, (1, 1)) == 1.0
    assert _single_gate(and3, 3, (1, 0)) == pytest.approx(1 / 2)
    assert _single_gate(and3, 3, (0, 0)) == 0.0


def test_complex_cell_needs_the_exact_condition():
    # o = ~((A & B) | C): the effect on A passes only with B=1 and C=0.
    aoi = _lut(lambda a, b, c: 1 - ((a & b) | c), 3)
    assert _single_gate(aoi, 3, (1, 0)) == 1.0
    assert _single_gate(aoi, 3, (1, 1)) == pytest.approx(1 / 2)
    assert _single_gate(aoi, 3, (0, 1)) == 0.0


def test_xor_passes_the_effect_for_any_definite_side_value():
    xor = _lut(lambda a, b: a ^ b, 2)
    assert _single_gate(xor, 2, (0,)) == 1.0
    assert _single_gate(xor, 2, (1,)) == 1.0
    assert _single_gate(xor, 2, ("x",)) == 0.0


# --- reward ----------------------------------------------------------------------------

def test_first_try_success_is_best():
    row, simulated = _score(_trace([DETECTS], _answer(DETECTS)))
    for key in STAGED_OBJECTIVE_KEYS:
        assert row[key] == 1.0, key
    assert row["answer_detects_logonly"] == 1.0
    assert len(simulated) == 1     # the answer reuses the call; only the last call is simulated again


def test_failed_attempts_are_ranked_by_how_far_the_effect_got():
    scores = {name: _score(_trace([vector], _answer(vector)))[0]
              for name, vector in (("inactive", INACTIVE), ("early", BLOCKED_EARLY), ("late", BLOCKED_LATE))}
    assert all(row["detection"] == 0.0 for row in scores.values())
    assert scores["inactive"]["propagation"] == 0.0
    assert 0.0 < scores["early"]["propagation"] < scores["late"]["propagation"] < 1.0
    assert scores["early"]["controlling"] == 0.0 < scores["late"]["controlling"] < 1.0


def test_a_later_detection_gives_the_stages_their_maximum():
    row, _ = _score(_trace([BLOCKED_LATE, DETECTS], _answer(DETECTS)))
    assert row["detection"] == pytest.approx(1 / 2)
    assert row["propagation"] == row["controlling"] == row["activation"] == 1.0


def test_a_retry_that_does_worse_costs_nothing_on_the_stages():
    alone, _ = _score(_trace([BLOCKED_LATE], _answer(BLOCKED_LATE)))
    worse, _ = _score(_trace([BLOCKED_LATE, BLOCKED_EARLY], _answer(BLOCKED_LATE)))
    inactive, _ = _score(_trace([BLOCKED_LATE, INACTIVE], _answer(BLOCKED_LATE)))
    for row in (worse, inactive):
        for key in ("activation", "propagation", "controlling"):
            assert row[key] == pytest.approx(alone[key]), key
        assert row["uniqueness"] == 1.0


def test_the_best_attempt_counts_whatever_its_position():
    first, _ = _score(_trace([BLOCKED_LATE, BLOCKED_EARLY], _answer(BLOCKED_LATE)))
    second, _ = _score(_trace([BLOCKED_EARLY, BLOCKED_LATE], _answer(BLOCKED_LATE)))
    for key in ("activation", "propagation", "controlling"):
        assert first[key] == pytest.approx(second[key]), key
    assert first["propagation"] == pytest.approx(sensitize(2 / 3))
    assert first["controlling"] == pytest.approx(sensitize(1 / 2))


def test_repeating_a_vector_gains_no_stage_and_loses_uniqueness():
    once, _ = _score(_trace([BLOCKED_LATE], _answer(BLOCKED_LATE)))
    twice, _ = _score(_trace([BLOCKED_LATE, BLOCKED_LATE], _answer(BLOCKED_LATE)))
    assert twice["propagation"] == pytest.approx(once["propagation"])
    assert twice["uniqueness"] == 0.5 < once["uniqueness"]


def test_no_call_has_no_signal():
    row, _ = _score({"turns": [_answer(DETECTS)], "calls": []})
    assert row["propagation"] == row["controlling"] == row["detection"] == 0.0


def test_a_detecting_call_with_a_wrong_answer_is_not_detection():
    row, _ = _score(_trace([DETECTS], _answer(INACTIVE)))
    assert row["detected_by_call_logonly"] == 1.0
    assert row["answer_detects_logonly"] == 0.0
    assert row["detection"] == 0.0 and row["fidelity"] == 0.0
    assert row["propagation"] == 1.0    # the work itself still counts


def test_omitting_an_input_is_the_same_as_writing_x():
    omitted, simulated = _score(_trace([DETECTS], _answer(DETECTS_NO_X)))
    written, _ = _score(_trace([DETECTS], _answer(DETECTS)))
    for key in STAGED_OBJECTIVE_KEYS:
        assert omitted[key] == written[key] == 1.0, key
    assert len(simulated) == 1


def test_an_answer_with_other_care_bits_is_judged_on_its_own_simulation():
    row, simulated = _score(_trace([DETECTS], _answer("p:1,s1:1,s2:1,u:0")))
    assert row["detection"] == 1.0 and row["fidelity"] == 1.0
    assert len(simulated) == 2


def test_an_answer_naming_an_unknown_input_does_not_detect():
    row, _ = _score(_trace([DETECTS], _answer("p:1,s1:1,s2:1,zz:1")))
    assert row["detection"] == 0.0


def test_expected_output_is_checked_against_the_answer_simulation():
    row, _ = _score(_trace([DETECTS], _answer(DETECTS, output="y:0")))
    assert row["detection"] == 1.0
    assert row["fidelity"] == pytest.approx(1 / 2)


def test_extra_fault_claims_need_evidence_in_the_table():
    # n1 changes from 1 to 0 under the fault, so sa0 n1 is supported and sa1 n1 is not.
    row, _ = _score(_trace([DETECTS], _answer(DETECTS, faults="sa0 p, sa0 n1, sa1 n1")))
    assert row["claims_count_logonly"] == 3.0
    assert row["claims_valid_logonly"] == pytest.approx(2 / 3)
    assert row["fidelity"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert row["format"] == 1.0


def test_the_target_fault_must_be_listed():
    row, _ = _score(_trace([DETECTS], _answer(DETECTS, faults="sa0 n1")))
    assert row["claims_valid_logonly"] == 0.0
    assert row["fidelity"] == pytest.approx(1 / 2)
    assert row["format"] == pytest.approx(7 / 8)


def test_format_ignores_vector_content():
    all_x = "p:x,s1:x,s2:x,u:x"
    row, _ = _score(_trace([DETECTS], _answer(all_x)))
    assert row["format"] == 1.0
    assert row["detection"] == 0.0


def test_format_needs_reasoning_and_a_well_formed_call():
    trace = _trace([DETECTS], _answer(DETECTS))
    trace["turns"][0] = "</think>\n<tool_call>\n<function=fault_simulation_tool>\n</function>\n</tool_call>"
    trace["calls"][0]["arguments"] = {"input_vector": DETECTS}
    row, _ = _score(trace)
    assert row["format"] == pytest.approx(6 / 8)
    assert row["detection"] == 1.0


def test_a_fault_on_a_primary_output_has_nothing_left_to_control():
    netlist = SimpleNamespace(output_nets=["y"], instructions=[("assign_net", "z", "y")])
    graph = SignalGraph(netlist)
    signals = attempt_signals(graph, {"y": 1}, {"y": 0}, "y", True, False)
    assert signals["propagation"] == 1.0 and signals["controlling"] == 1.0


def test_a_net_the_structure_cannot_follow_scores_zero():
    # A tie-cell net feeds nothing in the compiled netlist (its readers see a constant).
    graph = SignalGraph(_netlist())
    signals = attempt_signals(graph, {"tie": 0}, {"tie": 1}, "tie", True, False)
    assert signals == {"propagation": 0.0, "controlling": 0.0, "depth": 0.0}


def test_a_ports_only_netlist_gets_its_structure_from_the_verilog_text():
    """The TetraMAX backend's netlist has no gates, so the Verilog is compiled once."""
    from atpgllm.llm.propagation_signals import signal_graph
    from atpgllm.training.reward_function_factory import RewardFunctionFactory

    gates = RewardFunctionFactory("sim_config.json").gate_funcs
    verilog = """module t (a, b, c, d, y);
 input a, b, c, d;
 output y;
 wire n1, n2;
 AOI21xp5_ASAP7_75t_R g1 (.A1(a), .A2(b), .B(c), .Y(n1));
 NAND2xp5_ASAP7_75t_R g2 (.A(n1), .B(d), .Y(n2));
 INVx1_ASAP7_75t_R g3 (.A(n2), .Y(y));
endmodule
"""
    ports_only = SimpleNamespace(netlist=verilog, input_nets=["a", "b", "c", "d"], output_nets=["y"])
    assert signal_graph(ports_only) is None            # no gate library, nothing to compile
    graph = signal_graph(ports_only, gates)
    assert graph is signal_graph(ports_only, gates)    # cached on the netlist
    assert sorted(graph.cone("n1")) == ["n1", "n2", "y"]
    good = {"a": 1, "b": 1, "c": 0, "d": 0, "n1": 0, "n2": 1, "y": 0}
    bad = dict(good, n1=1, n2=1)
    blocked = attempt_signals(graph, good, bad, "n1", True, False)
    assert blocked["propagation"] == pytest.approx(1 / 3)
    assert blocked["controlling"] == 0.0               # d=0 blocks the NAND


def test_an_answer_with_a_non_binary_care_bit_does_not_reach_the_simulator():
    row, simulated = _score(_trace([DETECTS], _answer("p:2,s1:1,s2:1")))
    assert row["detection"] == 0.0 and row["simulator_error_logonly"] == 0.0
    assert len(simulated) == 1
