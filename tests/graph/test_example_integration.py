"""
End-to-end example: Verilog netlist → PyG graph → node embeddings → DAG-GIN.

Demonstrates the full pipeline from raw structural netlist text to
graph-level and per-node embeddings ready for Q-Former / LLM integration.
"""

import json
import sys
from pathlib import Path

import torch
from atpgllm.graph import GateAttributeVocab, AttributeDecompositionEncoder, DAGGINEncoder
from atpgllm.graph.netlist_parser import parse_verilog_to_graph, parsed_to_pyg

# -- 1. Load gate function library and build attribute vocabulary ----------

def test_example_integration():

    from atpgllm.training._paths import resolve_sim_config_path
    with open(resolve_sim_config_path()) as f:
        gate_funcs = json.load(f)["gate_funcs"]

    vocab = GateAttributeVocab(gate_funcs)
    vocab.report_collapse()

    # -- 2. Parse a real synthesised netlist -----------------------------------

    netlist1 = r"""
module qkt_macro_handler_unit_add_21sx17u_22s_1 ( in2, in1, out1 );
input [20:0] in2;
input [16:0] in1;
output [21:0] out1;
wire \add_x_1/n49 , \add_x_1/n48 , \add_x_1/n47 , \add_x_1/n46 ,
\add_x_1/n45 , \add_x_1/n44 , \add_x_1/n43 , \add_x_1/n42 ,
\add_x_1/n39 , \add_x_1/n38 , \add_x_1/n37 , \add_x_1/n36 ,
\add_x_1/n35 , \add_x_1/n34 , \add_x_1/n33 , \add_x_1/n32 ,
\add_x_1/n31 , \add_x_1/n30 , \add_x_1/n29 , \add_x_1/n28 ,
\add_x_1/n27 , \add_x_1/n26 , \add_x_1/n25 , \add_x_1/n24 ,
\add_x_1/n23 , \add_x_1/n18 , \add_x_1/n17 , \add_x_1/n16 ,
\add_x_1/n15 , \add_x_1/n14 , \add_x_1/n13 , \add_x_1/n12 ,
\add_x_1/n11 , \add_x_1/n10 , \add_x_1/n9 , \add_x_1/n8 ,
\add_x_1/n7 , \add_x_1/n6 , \add_x_1/n5 , \add_x_1/n4 , \add_x_1/n3 ,
n2, n3, n4, n5, n6, n7;
FAx1_ASAP7_75t_R \add_x_1/U51 ( .A(\add_x_1/n17 ), .B(\add_x_1/n18 ), .CI(
\add_x_1/n39 ), .CON(\add_x_1/n38 ), .SN(out1[1]) );
FAx1_ASAP7_75t_R \add_x_1/U50 ( .A(in2[2]), .B(in1[2]), .CI(\add_x_1/n38 ),
.CON(\add_x_1/n37 ), .SN(\add_x_1/n49 ) );
FAx1_ASAP7_75t_R \add_x_1/U46 ( .A(\add_x_1/n15 ), .B(\add_x_1/n16 ), .CI(
\add_x_1/n37 ), .CON(\add_x_1/n36 ), .SN(out1[3]) );
FAx1_ASAP7_75t_R \add_x_1/U45 ( .A(in2[4]), .B(in1[4]), .CI(\add_x_1/n36 ),
.CON(\add_x_1/n35 ), .SN(\add_x_1/n48 ) );
FAx1_ASAP7_75t_R \add_x_1/U41 ( .A(\add_x_1/n13 ), .B(\add_x_1/n14 ), .CI(
\add_x_1/n35 ), .CON(\add_x_1/n34 ), .SN(out1[5]) );
FAx1_ASAP7_75t_R \add_x_1/U40 ( .A(in2[6]), .B(in1[6]), .CI(\add_x_1/n34 ),
.CON(\add_x_1/n33 ), .SN(\add_x_1/n47 ) );
FAx1_ASAP7_75t_R \add_x_1/U36 ( .A(\add_x_1/n11 ), .B(\add_x_1/n12 ), .CI(
\add_x_1/n33 ), .CON(\add_x_1/n32 ), .SN(out1[7]) );
FAx1_ASAP7_75t_R \add_x_1/U35 ( .A(in2[8]), .B(in1[8]), .CI(\add_x_1/n32 ),
.CON(\add_x_1/n31 ), .SN(\add_x_1/n46 ) );
FAx1_ASAP7_75t_R \add_x_1/U31 ( .A(\add_x_1/n9 ), .B(\add_x_1/n10 ), .CI(
\add_x_1/n31 ), .CON(\add_x_1/n30 ), .SN(out1[9]) );
FAx1_ASAP7_75t_R \add_x_1/U30 ( .A(in2[10]), .B(in1[10]), .CI(\add_x_1/n30 ), .CON(\add_x_1/n29 ), .SN(\add_x_1/n45 ) );
FAx1_ASAP7_75t_R \add_x_1/U26 ( .A(\add_x_1/n7 ), .B(\add_x_1/n8 ), .CI(
\add_x_1/n29 ), .CON(\add_x_1/n28 ), .SN(out1[11]) );
FAx1_ASAP7_75t_R \add_x_1/U25 ( .A(in2[12]), .B(in1[12]), .CI(\add_x_1/n28 ), .CON(\add_x_1/n27 ), .SN(\add_x_1/n44 ) );
FAx1_ASAP7_75t_R \add_x_1/U21 ( .A(\add_x_1/n5 ), .B(\add_x_1/n6 ), .CI(
\add_x_1/n27 ), .CON(\add_x_1/n26 ), .SN(out1[13]) );
FAx1_ASAP7_75t_R \add_x_1/U20 ( .A(in2[14]), .B(in1[14]), .CI(\add_x_1/n26 ), .CON(\add_x_1/n25 ), .SN(\add_x_1/n43 ) );
FAx1_ASAP7_75t_R \add_x_1/U16 ( .A(\add_x_1/n3 ), .B(\add_x_1/n4 ), .CI(
\add_x_1/n25 ), .CON(\add_x_1/n24 ), .SN(out1[15]) );
FAx1_ASAP7_75t_R \add_x_1/U15 ( .A(in2[16]), .B(in1[16]), .CI(\add_x_1/n24 ), .CON(\add_x_1/n23 ), .SN(\add_x_1/n42 ) );
INVxp33_ASAP7_75t_R U2 ( .A(in2[17]), .Y(n3) );
NOR2xp33_ASAP7_75t_R U3 ( .A(\add_x_1/n23 ), .B(n3), .Y(n7) );
NAND2xp33_ASAP7_75t_R U4 ( .A(n7), .B(in2[18]), .Y(n6) );
INVxp33_ASAP7_75t_R U5 ( .A(in2[19]), .Y(n2) );
NOR2xp33_ASAP7_75t_R U6 ( .A(n6), .B(n2), .Y(n5) );
AOI21xp33_ASAP7_75t_R U7 ( .A1(n6), .A2(n2), .B(n5), .Y(out1[19]) );
AOI21xp33_ASAP7_75t_R U8 ( .A1(\add_x_1/n23 ), .A2(n3), .B(n7), .Y(out1[17])
);
INVxp33_ASAP7_75t_R U9 ( .A(\add_x_1/n42 ), .Y(out1[16]) );
INVxp33_ASAP7_75t_R U10 ( .A(\add_x_1/n43 ), .Y(out1[14]) );
INVxp33_ASAP7_75t_R U11 ( .A(\add_x_1/n44 ), .Y(out1[12]) );
INVxp33_ASAP7_75t_R U12 ( .A(in2[20]), .Y(n4) );
NOR2xp33_ASAP7_75t_R U13 ( .A(n5), .B(n4), .Y(out1[21]) );
INVxp33_ASAP7_75t_R U14 ( .A(in1[11]), .Y(\add_x_1/n8 ) );
INVxp33_ASAP7_75t_R U15 ( .A(in2[11]), .Y(\add_x_1/n7 ) );
INVxp33_ASAP7_75t_R U16 ( .A(in1[13]), .Y(\add_x_1/n6 ) );
INVxp33_ASAP7_75t_R U17 ( .A(in2[13]), .Y(\add_x_1/n5 ) );
INVxp33_ASAP7_75t_R U18 ( .A(in1[15]), .Y(\add_x_1/n4 ) );
INVxp33_ASAP7_75t_R U19 ( .A(in2[15]), .Y(\add_x_1/n3 ) );
INVxp33_ASAP7_75t_R U20 ( .A(\add_x_1/n45 ), .Y(out1[10]) );
INVxp33_ASAP7_75t_R U21 ( .A(in1[9]), .Y(\add_x_1/n10 ) );
INVxp33_ASAP7_75t_R U22 ( .A(in2[9]), .Y(\add_x_1/n9 ) );
INVxp33_ASAP7_75t_R U23 ( .A(\add_x_1/n46 ), .Y(out1[8]) );
INVxp33_ASAP7_75t_R U24 ( .A(in1[7]), .Y(\add_x_1/n12 ) );
INVxp33_ASAP7_75t_R U25 ( .A(in2[7]), .Y(\add_x_1/n11 ) );
INVxp33_ASAP7_75t_R U26 ( .A(\add_x_1/n47 ), .Y(out1[6]) );
INVxp33_ASAP7_75t_R U27 ( .A(in1[5]), .Y(\add_x_1/n14 ) );
INVxp33_ASAP7_75t_R U28 ( .A(in2[5]), .Y(\add_x_1/n13 ) );
INVxp33_ASAP7_75t_R U29 ( .A(\add_x_1/n48 ), .Y(out1[4]) );
INVxp33_ASAP7_75t_R U30 ( .A(in1[3]), .Y(\add_x_1/n16 ) );
INVxp33_ASAP7_75t_R U31 ( .A(in2[3]), .Y(\add_x_1/n15 ) );
INVxp33_ASAP7_75t_R U32 ( .A(\add_x_1/n49 ), .Y(out1[2]) );
NAND2xp33_ASAP7_75t_R U33 ( .A(in2[0]), .B(in1[0]), .Y(\add_x_1/n39 ) );
INVxp33_ASAP7_75t_R U34 ( .A(in1[1]), .Y(\add_x_1/n18 ) );
INVxp33_ASAP7_75t_R U35 ( .A(in2[1]), .Y(\add_x_1/n17 ) );
AO21x1_ASAP7_75t_R U36 ( .A1(n5), .A2(n4), .B(out1[21]), .Y(out1[20]) );
OA21x2_ASAP7_75t_R U37 ( .A1(in2[0]), .A2(in1[0]), .B(\add_x_1/n39 ), .Y(
out1[0]) );
OA21x2_ASAP7_75t_R U38 ( .A1(n7), .A2(in2[18]), .B(n6), .Y(out1[18]) );
endmodule
"""

    # -- 3. Parse netlist → graph → PyG Data ----------------------------------

    parsed = parse_verilog_to_graph(netlist1, gate_funcs)
    data = parsed_to_pyg(parsed, vocab=vocab)

    print(f"Parsed {len(parsed.gates)} gates, {data.edge_index.size(1)} edges")
    print(f"  gate_attrs:       {data.gate_attrs.shape}")
    print(f"  structural_feats: {data.structural_feats.shape}")

    # -- 4. Build models -------------------------------------------------------

    node_encoder = AttributeDecompositionEncoder.from_vocab(vocab, out_dim=256)
    dag_gin = DAGGINEncoder(in_dim=256, hidden_dim=256, num_layers=6)

    total_params = sum(p.numel() for p in node_encoder.parameters()) + \
                   sum(p.numel() for p in dag_gin.parameters())
    print(f"  Total encoder params: {total_params:,}")

    # -- 5. Forward pass -------------------------------------------------------

    # Encode node attributes + structural features → initial embeddings
    x = node_encoder(data.gate_attrs, data.structural_feats)

    # Fake a single-graph batch (in real training, PyG DataLoader handles this)
    batch = torch.zeros(data.x.size(0), dtype=torch.long)

    # DAG-GIN: bidirectional message passing → node + graph embeddings
    out = dag_gin(x, data.edge_index, batch)

    node_embs = out["node_embs"]   # [53, 256] — per-node, for Q-Former cross-attention
    graph_embs = out["graph_embs"] # [1, 1024] — graph-level, for contrastive losses

    print(f"\n  node_embs:  {node_embs.shape}  (per-gate embeddings)")
    print(f"  graph_embs: {graph_embs.shape}  (whole-circuit embedding)")

    # -- 6. Inspect what the model sees per gate --------------------------------

    print("\nPer-gate breakdown (first 10):")
    print(f"  {'Gate':30s} {'Type':25s} {'FwdD':>5s} {'BwdD':>5s} {'InD':>5s} {'OutD':>5s}")
    print(f"  {'-'*30} {'-'*25} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")
    for i in range(min(10, len(parsed.gates))):
        g = parsed.gates[i]
        sf = data.structural_feats[i]
        attrs = vocab.get_attrs(g.cell)
        print(f"  {g.inst:30s} {attrs.logic_family + '/' + str(attrs.input_count) + 'in':25s}"
              f" {sf[0]:5.2f} {sf[1]:5.2f} {sf[2]:5.2f} {sf[3]:5.2f}")

    # -- 7. These embeddings plug into Stage 2 (Q-Former → Qwen2.5) -----------

    print("\n--- Next steps ---")
    print(f"  node_embs [{node_embs.shape[0]} gates × {node_embs.shape[1]}d]")
    print(f"    → Q-Former cross-attention (32 queries attend to {node_embs.shape[0]} gate tokens)")
    print(f"    → Linear projection (256 → 3584 for Qwen2.5-7B)")
    print(f"    → Prepend as soft prompt tokens to Qwen2.5 input")
