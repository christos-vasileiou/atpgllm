"""Dataset v5 full-scan rows through the consumer paths (needs data_preprocessing seq-v5).

The fixture is a real v5 pilot row (s114702_tx_ds_se, 2 scan flops whose QN nets are
also primary outputs). The model reads the scan netlist; inputs are PIs + flop QN
nets, outputs are POs + flop D nets. data_preprocessing simulates the internal cut
view and returns model names only.
"""
import asyncio
import json
from pathlib import Path

import pytest

from atpgllm.training._paths import ensure_data_preprocessing_on_path

ensure_data_preprocessing_on_path()
scan_view = pytest.importorskip("scan_view", reason="needs data_preprocessing seq-v5 (scan_view.py)")

ROW = json.loads((Path(__file__).with_name("fixtures") / "v5_scan_row.json").read_text())
SCAN_NAMES = ("test_si", "test_se", "test_so")


def _gates():
    from atpgllm.training.tools import resolve_sim_config_path
    return json.loads(resolve_sim_config_path().read_text())


def _fast_model():
    from fault_sim import prepare_netlist
    from atpgllm.training.reward_function_factory import RewardFunctionFactory as RF
    return prepare_netlist(ROW["netlist"], _gates(), RF.DECL_RE, RF.NAME_RE)


def test_prompt_renders_chain_table_and_hint():
    from atpgllm.training.conversation import ConversationExample
    messages = ConversationExample.prompt_messages_from_record(dict(ROW))
    system, user = messages[0]["content"], messages[1]["content"]
    assert "full-scan sequential circuit" in system
    assert "Scan chain (test_si -> test_so):" in user and "QN=d" in user
    assert "{scan_chain}" not in user and "{netlist}" not in user
    assert "SDFHx1_ASAP7_75t_R" in user


def test_scan_model_uses_model_names(monkeypatch):
    monkeypatch.setenv("FAULT_SIM_BACKEND", "fast")
    model = _fast_model()
    assert isinstance(model, scan_view.ScanModel)
    assert set(json.loads(ROW["input_vector"])) == set(model.input_nets)
    assert set(json.loads(ROW["expected_output"])) == set(model.output_nets)
    assert not set(SCAN_NAMES) & set(model.input_nets + model.output_nets)
    assert not set(model.clock_ports) & set(model.input_nets)


def test_row_fault_detected_and_outputs_match(monkeypatch):
    monkeypatch.setenv("FAULT_SIM_BACKEND", "fast")
    from fault_sim import fast_fault_sim
    model = _fast_model()
    frame = fast_fault_sim(json.loads(ROW["input_vector"]), json.loads(ROW["expected_output"]),
                           ROW["fault"], model, _gates())
    assert not any(str(n).startswith("v5_") for n in frame.index)
    good, bad = frame["Good Machine"], frame["Bad Machine"]
    for net, value in json.loads(ROW["expected_output"]).items():
        if value != "x":
            assert str(good[net]) == str(value), net
    assert any(good[n] in (0, 1) and bad[n] in (0, 1) and good[n] != bad[n] for n in model.output_nets)


def test_tool_handler_output_has_no_internal_names(monkeypatch):
    monkeypatch.setenv("FAULT_SIM_BACKEND", "fast")
    from atpgllm.training.tools import fault_simulation_tool_handler
    out = asyncio.run(fault_simulation_tool_handler(
        json.loads(ROW["input_vector"]), json.loads(ROW["expected_output"]), ROW["fault"],
        "doc", ROW["netlist"]))
    values = json.loads(out)["Good Machine"]
    assert set(json.loads(ROW["input_vector"])) <= set(values)
    assert not any(n.startswith("v5_") or n in SCAN_NAMES for n in values)


def test_tool_rejects_scan_ports(monkeypatch):
    monkeypatch.setenv("FAULT_SIM_BACKEND", "fast")
    from atpgllm.training.tools import fault_simulation_tool_handler
    vector = dict(json.loads(ROW["input_vector"]), test_se=0)
    out = asyncio.run(fault_simulation_tool_handler(vector, json.loads(ROW["expected_output"]), ROW["fault"],
                                                    "doc", ROW["netlist"]))
    assert out.startswith("Tool execution failed")


def test_signal_graph_never_crosses_a_flop(monkeypatch):
    monkeypatch.setenv("FAULT_SIM_BACKEND", "tetramax")  # ScanModel without compiled gates inside
    from fault_sim import prepare_netlist
    from atpgllm.llm.propagation_signals import signal_graph
    model = prepare_netlist(ROW["netlist"])
    graph = signal_graph(model, _gates())
    assert graph is not None
    produced = {record[0] for record in graph.records}
    qn = {c["QN"] for c in model.scan_cells}
    assert not qn & produced  # QN nets are sources, never driven through a flop
    assert not any(str(r[0]).startswith("v5_") for r in graph.records)
