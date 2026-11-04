"""The reasoning audit and the gate by gate renderer on one small hand built circuit."""
import json

import pytest

NETLIST = """module t ( a, b, c, y );
input a, b, c;
output y;
wire n1;
AND2x2_ASAP7_75t_R U1 ( .A(a), .B(b), .Y(n1) );
OR2x2_ASAP7_75t_R U2 ( .A(n1), .B(c), .Y(y) );
endmodule
"""
GOOD = {"a": 1, "b": 1, "c": 0, "n1": 1, "y": 1}
BAD = {"a": 1, "b": 1, "c": 0, "n1": 0, "y": 0}
FAULT = "sa0 n1"


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setenv("FAULT_SIM_BACKEND", "fast")


def audit(text, good=GOOD):
    from atpgllm.training.reasoning_audit import audit_text
    return audit_text(text, NETLIST, FAULT, good, BAD)


def test_true_and_false_claims_about_internal_nets():
    assert audit("We need n1 = 1 here.")["claim_truth"] == 1.0
    assert audit("We need n1 = 0 here.")["claim_truth"] == 0.0
    assert audit("Drive n1 to 1, and y is 0.")["claim_truth"] == 0.5


def test_a_claim_about_the_faulty_machine_is_checked_against_it():
    assert audit("Good y = 1, faulty y = 0.")["claim_truth"] == 1.0
    assert audit("Good y = 1, faulty y = 1.")["claim_truth"] == 0.5


def test_restating_primary_inputs_earns_nothing():
    result = audit("a = 1, b = 1, c = 0.")
    assert result["claims"] == 0 and "claim_truth" not in result


def test_free_text_with_nothing_checkable_is_ignored():
    result = audit("Let me think about how to excite this fault carefully.")
    assert result["claims"] == 0
    assert not {"claim_truth", "implication_truth", "structural_validity"} & set(result)


def test_one_true_claim_cannot_reach_a_full_score():
    assert audit("n1 = 1")["claim_score"] == pytest.approx(1 / 3)
    assert audit("n1 = 1 and y = 1, so n1 is 1")["claim_score"] == 1.0


def test_implications_follow_the_truth_table_whatever_the_vector_does():
    assert audit("U1 gives n1 = 1 when a = 1 and b = 1.", good={})["implication_truth"] == 1.0
    assert audit("U1 gives n1 = 1 when a = 0.", good={})["implication_truth"] == 0.0
    assert audit("U1 gives n1 = 1 when a = 1.", good={})["implication_truth"] == 0.5


def test_invented_names_lower_structural_validity():
    assert audit("U1 and n1 matter.")["structural_validity"] == 1.0
    assert audit("U1 and U99 matter.")["structural_validity"] == 0.5


def slot(call_outputs, final='DETECTED_FAULTS: "sa0 n1"', good=GOOD, bad=BAD):
    return {"final_answer": final, "messages": [
        {"role": "assistant", "content": "Set n1 to 1.", "tool_calls": [{"function": {
            "name": "fault_simulation_tool", "arguments": {"input_vector": {"a": 1}, "output_vector": call_outputs}}}]},
        {"role": "tool", "content": json.dumps({"Good Machine": good, "Bad Machine": bad})}]}


def test_output_prediction_needs_both_accuracy_and_coverage():
    from atpgllm.training.reasoning_audit import audit_slot
    ref = {"netlist": NETLIST, "fault": FAULT}
    assert audit_slot(slot({"y": 1}), ref)["output_prediction"] == 1.0
    assert audit_slot(slot({"y": 0}), ref)["output_prediction"] == 0.0
    assert audit_slot(slot({"y": "x"}), ref)["output_prediction"] == 0.0


def test_verdict_truth_catches_a_false_claim_of_detection():
    from atpgllm.training.reasoning_audit import audit_slot
    ref = {"netlist": NETLIST, "fault": FAULT}
    assert audit_slot(slot({"y": 1}), ref)["verdict_truth"] == 1.0
    undetected = dict(BAD, y=1)
    assert audit_slot(slot({"y": 1}, bad=undetected), ref)["verdict_truth"] == 0.0
    assert audit_slot(slot({"y": 1}, final="no detection", bad=undetected), ref)["verdict_truth"] == 1.0


def test_site_target_and_whether_the_vector_realized_it():
    from atpgllm.training.reasoning_audit import audit_slot
    ref = {"netlist": NETLIST, "fault": FAULT}
    result = audit_slot(slot({"y": 1}), ref)
    assert result["site_target_correct"] == 1.0 and result["site_target_realized"] == 1.0
    result = audit_slot(slot({"y": 1}, good=dict(GOOD, n1=0)), ref)
    assert result["site_target_correct"] == 1.0 and result["site_target_realized"] == 0.0


ROW = {"fault": FAULT, "netlist": NETLIST, "snapshot": json.dumps({"Good Machine": GOOD, "Bad Machine": BAD})}


def test_rendered_trace_passes_the_audit_and_a_corruption_fails():
    from atpgllm.training.reasoning_trace import render_trace
    text = render_trace(ROW, seed=1)
    result = audit(text)
    assert result["claim_truth"] == 1.0 and result["implication_truth"] == 1.0
    assert "n1 = 1" in text
    corrupted = text.replace("n1 = 1", "n1 = 0")
    worse = audit(corrupted)
    assert worse["claim_truth"] < 1.0 and worse["implication_truth"] < 1.0


def test_phrasing_varies_but_claims_do_not():
    from atpgllm.training.reasoning_trace import render_trace
    texts = {render_trace(ROW, seed=seed) for seed in range(12)}
    assert len(texts) > 1
    assert all(audit(text)["claim_truth"] == 1.0 for text in texts)


def test_no_trace_when_the_row_does_not_activate_the_fault():
    from atpgllm.training.reasoning_trace import render_trace
    good = dict(GOOD, n1=0)
    assert render_trace({**ROW, "snapshot": {"Good Machine": good, "Bad Machine": BAD}}) is None


def test_conclusions_about_inputs_come_after_the_derivation():
    from atpgllm.training.reasoning_audit import structure
    from atpgllm.training.reasoning_trace import render_trace
    split = {"activating": ["a", "b"], "propagating": ["c"]}
    good = {"a": 1, "b": 1, "c": 0, "d": "x", "n1": 1, "y": 1}
    netlist = NETLIST.replace("input a, b, c;", "input a, b, c, d;").replace("( a, b, c, y )", "( a, b, c, d, y )")
    row = {"fault": FAULT, "netlist": netlist, "snapshot": {"Good Machine": good, "Bad Machine": dict(good, n1=0, y=0)}}
    for seed in range(8):
        text = render_trace(row, seed=seed, split=split)
        lines = text.splitlines()
        assert "n1" in lines[0] and "a = 1" not in lines[0]
        assert "U1" in lines[1]
        assert "1 inputs" in text or "remaining 1" in text or "other 1" in text
        assert audit(text, good=good)["claim_truth"] == 1.0


def test_reading_a_negative_result_names_what_went_wrong():
    from atpgllm.training.reasoning_trace import read_result
    x = {"Good Machine": dict(GOOD, n1="x"), "Bad Machine": dict(BAD, n1=0)}
    assert "not excited" in read_result(x, FAULT, NETLIST) and "n1 = x" in read_result(x, FAULT, NETLIST)
    wrong = {"Good Machine": dict(GOOD, n1=0), "Bad Machine": dict(BAD)}
    assert "same as the stuck value" in read_result(wrong, FAULT, NETLIST)
    blocked_good = dict(GOOD, c=1, y=1)
    blocked = {"Good Machine": blocked_good, "Bad Machine": dict(blocked_good, n1=0)}
    text = read_result(blocked, FAULT, NETLIST)
    assert "U2" in text and "c = 1" in text and "No output differs" in text

