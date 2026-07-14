# Graph-Text Multimodal Architecture for Circuit ATPG

## Overview

This document describes the graph encoder pipeline that transforms structural
gate-level netlists into fixed-dimensional embeddings compatible with large
language models.  The architecture is designed for Automatic Test Pattern
Generation (ATPG) where the LLM (Qwen2.5-7B, fine-tuned with GRPO) must
reason about circuit topology, fault propagation, and signal controllability.

The pipeline maps an arbitrary-size netlist graph G = {V, E} to:
- **Per-node embeddings** `[N, 256]` for cross-modal attention (Q-Former)
- **Graph-level embeddings** `[1, 1024]` for contrastive/matching losses

The entire encoder is **invariant to node count** — it produces the same
embedding dimensionality whether the netlist has 1 gate or 10,000.


## Architecture Diagram

```
Verilog Netlist (text)
        |
        v
+-----------------------+
| Netlist Parser        |   netlist_parser.py
| Regex-based extraction|   Handles escaped names: \add_x_1/n36, \*Logic1*, s7[3]
| of gate instances,    |   Builds directed edges: driver → sink (DAG)
| net connections,      |   Computes structural features (Kahn's algorithm)
| I/O declarations      |
+----------+------------+
           |
           v
+----------+------------+       +---------------------------+
| Gate Attribute Vocab  |<------| sim_config.json           |
| gate_features.py      |       | 202 ASAP7 cell types      |
|                       |       | Boolean functions per cell |
| Cell name parsing:    |       +---------------------------+
|   AOI333xp33_ASAP7... |
|   → 8 attribute indices|
+-----------+-----------+
            |
            v
+-----------+-----------+
| Attribute Decomp.     |   node_encoder.py
| Encoder (nn.Module)   |
|                       |
| 8 learned embeddings  |   [N, 8] int → [N, 128] embedded
| + 4 structural feats  |   [N, 4] float (depth, degree)
| → MLP fusion          |   → [N, 256] initial node embedding
+-----------+-----------+
            |
            v
+-----------+-----------+
| DAG-GIN Encoder       |   dag_gin.py
| 6 bidirectional layers|
|                       |
| Forward GIN (fanin)   |   Controllability path
| Backward GIN (fanout) |   Observability path
| Fusion MLP per layer  |
| Residual + JK concat  |
|                       |
| → node_embs [N, 256]  |   For Q-Former cross-attention
| → graph_embs [1, 1024]|   For contrastive losses
+-----------+-----------+
            |
            v
    [Future: Q-Former]
    32 learnable queries cross-attend to N gate tokens
    → [1, 32, 768] → Linear → [1, 32, 3584]
    → Prepend as soft prompts to Qwen2.5-7B
```


## Component 1: Gate Attribute Decomposition

**File:** `gate_features.py` (pure Python, no PyTorch dependency)

### Problem

ASAP7 contains 202 cell names, but most are drive-strength variants of ~30
functionally distinct types.  For example, `NAND2xp33`, `NAND2xp5`, `NAND2x1`,
`NAND2x1p5`, `NAND2x2` all implement the same boolean function (~A) | (~B)
and differ only in electrical drive.

A monolithic embedding table (one vector per cell name) would:
- Waste 202 × 256 = 51,712 parameters learning redundant representations
- Give `NAND2xp33` and `NAND2x2` completely independent embeddings despite
  sharing logic family, input count, complementation, and sequential status
- Fail on unseen cell names from other PDK variants (ASAP7 LVT, different
  technology nodes)

### Solution: Attribute Decomposition

Each cell is decomposed into 8 categorical/binary attributes, each with its
own small learned embedding:

| # | Attribute             | Classes | Examples                              | Why it matters for ATPG             |
|---|-----------------------|---------|---------------------------------------|-------------------------------------|
| 0 | `logic_family`        | 23      | AND, OR, AOI, DFF, FA, TIE, ...      | Determines boolean behaviour and fault sensitisation conditions |
| 1 | `input_count`         | 7       | 1, 2, 3, 4, 5, 6+                    | More inputs = harder controllability; more potential fault masking |
| 2 | `output_complemented` | 2       | True (NAND, NOR, AOI) / False (AND)   | Inverted output changes fault effect polarity (s-a-0 vs s-a-1 detection) |
| 3 | `is_sequential`       | 2       | DFF, LATCH vs combinational           | Sequential boundaries break fault propagation; require scan chains |
| 4 | `drive_strength`      | 11      | TINY(<0.5), X1, X2, X4, X16+         | Higher drive = more fanout capacity; affects timing-aware ATPG |
| 5 | `num_outputs`         | 3       | 1 (most), 2 (FA: carry+sum), special  | Multi-output gates create coupled fault effects |
| 6 | `tristate`            | 2       | True / False                          | Placeholder for future PDKs with tristate buffers |
| 7 | `is_clock_related`    | 2       | ICG, CKINVDC vs normal                | Clock gates require special ATPG handling (launch/capture) |

**Total embedding parameters:** 52 vectors × 16 dims = **832 parameters**.

### Why not monolithic embeddings?

| Criterion                | Monolithic (202×256)    | Attribute Decomposition (52×16 + MLP) |
|--------------------------|------------------------|---------------------------------------|
| Parameters               | 51,712                 | 51,648 (similar)                      |
| Sharing across variants  | None                   | 7/8 attributes shared for drive variants |
| Unseen gate types        | Falls back to UNK      | Meaningful embedding from attributes  |
| Interpretability         | Opaque                 | Each dimension has semantic meaning   |
| PDK portability          | Requires retraining    | Attributes transfer directly          |

### Cell Name Parsing

ASAP7 cell names follow the pattern `<family><structure><drive>_ASAP7_75t_R`:

```
AOI333xp33_ASAP7_75t_R
│  │  │
│  │  └── drive strength: x0.33 → TINY bucket
│  └───── structure digits: 333 → 3+3+3 = 9 inputs
└──────── family prefix: AOI → AND-OR-Invert, complemented=True
```

The parser handles all ASAP7 naming conventions including compound gates
(`A2O1A1Ixp33`), clock variants (`CKINVDCx10`), and sequential elements
(`DFFASRHQNx1`, `SDFHx4`).

Input count is derived primarily from the boolean function in `sim_config.json`
(counting unique variable names like A, B, CI), with heuristic fallback for
sequential elements whose functions only reference internal state (IQ/IQN).


## Component 2: Structural Features (Replacing RWPE)

**File:** `netlist_parser.py`, function `compute_structural_features`

### Why not RWPE (Random Walk Positional Encoding)?

RWPE computes the probability of a random walk returning to its starting node
after k steps.  This is the standard positional encoding for general graphs.

**RWPE is degenerate on DAGs.** Circuit netlists are directed acyclic graphs.
A random walk on a DAG follows forward edges and never returns — the return
probability is zero for all nodes at all step counts.  RWPE would produce a
zero vector for every node, providing no positional information whatsoever.

### Why not Laplacian PE?

Laplacian eigenvectors are another standard graph PE.  They work on DAGs but
suffer from:
- Sign ambiguity (eigenvectors are defined up to ±1)
- Ordering ambiguity (eigenvalue multiplicity)
- O(N^3) computation for eigendecomposition on large netlists
- No direct connection to ATPG-relevant circuit properties

### Solution: DAG-native structural features

Four features computed per node in O(N + E) via Kahn's algorithm:

| Feature              | Computation                          | ATPG Meaning                          |
|----------------------|--------------------------------------|---------------------------------------|
| `forward_depth`      | Longest path from any PI (normalised 0-1) | Approximates combinational controllability — deeper gates are harder to drive to specific values |
| `backward_depth`     | Longest path to any PO (normalised 0-1)   | Approximates combinational observability — gates farther from outputs are harder to observe |
| `log_in_degree`      | log2(1 + fanin count)                | Higher fanin = more inputs to sensitise for fault propagation |
| `log_out_degree`     | log2(1 + fanout count)               | Higher fanout creates reconvergent structures that complicate ATPG |

These four features are a continuous relaxation of **SCOAP measures**
(Sandia Controllability/Observability Analysis Program), which are the
foundation of classical ATPG heuristics like PODEM and FAN.

Forward/backward depth uses Kahn's topological sort with longest-path
relaxation.  Cycles (which shouldn't exist in synthesised combinational logic
but may appear in feedback paths) are handled gracefully — nodes remaining
after the topological traversal are assigned max_depth + 1.


## Component 3: Node Feature Encoder

**File:** `node_encoder.py`

### Architecture

```
gate_attrs [N, 8] (int64)          structural_feats [N, 4] (float32)
     |                                       |
     v                                       |
8 × nn.Embedding(n_classes, 16)              |
     |                                       |
     v                                       |
 concat → [N, 128]                           |
     |                                       |
     +------------------+--------------------+
                        |
                        v
                   concat → [N, 132]
                        |
                        v
              Linear(132, 128) + LayerNorm + GELU
                        |
                        v
              Linear(128, 256) + LayerNorm
                        |
                        v
                   x_init [N, 256]
```

### Why LayerNorm instead of BatchNorm in the encoder MLP?

The encoder MLP operates on individual nodes before any graph-level
aggregation.  LayerNorm normalises per-feature (independent of batch
statistics), which is more stable when:
- Batch sizes vary (single graphs during inference)
- Node counts per graph differ wildly (1 to 10,000+)

BatchNorm is used later inside the GIN layers (following the original GIN
paper convention) where it normalises over the full node batch from all
graphs in a mini-batch.


## Component 4: DAG-Aware Bidirectional GIN

**File:** `dag_gin.py`

### Why GIN over other GNN architectures?

| Architecture       | Expressiveness        | Complexity  | Why rejected / chosen                      |
|--------------------|-----------------------|-------------|--------------------------------------------|
| **GCN / GraphConv** | Below 1-WL           | O(\|E\|)    | Mean aggregation cannot distinguish 2 neighbours from 5 — critical failure for fanout-sensitive ATPG |
| **GAT**            | Attention-weighted    | O(\|E\|)    | Attention helps but doesn't guarantee injectivity; GIN is provably more expressive |
| **Graph Transformer** | Beyond 1-WL        | O(N^2)      | Quadratic in node count. At N=10,000 gates, requires 100M attention entries per layer — kills VRAM on H100 |
| **GIN (chosen)**   | Maximal 1-WL         | O(\|E\|)    | Provably most expressive among message-passing GNNs. Sum aggregation is injective for distinguishing multisets. |

**Key insight from the GIN paper (Xu et al., 2019):** GIN with sum aggregation
and MLP update achieves the discriminative power of the Weisfeiler-Lehman graph
isomorphism test.  For circuit netlists, this means the encoder can distinguish:
- Different reconvergent fanout structures
- Gates with 2 vs 5 identical predecessors (mean aggregation cannot)
- Structurally distinct sub-circuits that produce the same I/O behaviour

### Why bidirectional?

Standard GNNs treat graphs as undirected.  Circuit netlists are **directed
acyclic graphs** with distinct information flow directions:

**Forward direction (fanin → node):** Encodes what a gate's output depends on.
In ATPG terms, this is the **controllability** path — "what inputs do I need to
set to drive this gate to a specific value?"

**Backward direction (node → fanout):** Encodes where a gate's output
propagates to.  In ATPG terms, this is the **observability** path — "if there's
a fault here, through which paths can I observe it at a primary output?"

Each DAGGINLayer has separate forward and backward GIN convolutions with
independent learned parameters, fused by an MLP:

```
h_fwd = GIN_fwd(h, edge_index)          # aggregate from predecessors
h_bwd = GIN_bwd(h, edge_index.flip(0))  # aggregate from successors
h_new = MLP_fuse(concat(h_fwd, h_bwd))  # combine both directions
h     = h + h_new                        # residual connection
```

### Why 6 layers?

Typical synthesised circuits have 5-15 levels of combinational logic between
flip-flop boundaries.  With 6 GIN layers, each node's receptive field covers
6 hops in each direction (12 total), sufficient to capture the combinational
cone of influence for the majority of gates.  JumpingKnowledge then preserves
representations from all depths (1-hop local patterns through 6-hop global
context).

### Why JumpingKnowledge (concat mode)?

Without JK, only the final layer's representations are used.  Deep GNN layers
tend to over-smooth, losing fine-grained local structure.  JK concatenation
preserves all intermediate representations:

```
h_jk = concat(h_0, h_1, h_2, h_3, h_4, h_5, h_6)  → [N, 7 × 256 = 1792]
node_embs = Linear(1792, 256)                         → [N, 256]
```

Shallow layers (h_0, h_1) capture local gate-level patterns.  Deep layers
(h_5, h_6) see multi-level logic cones.  The linear projection learns which
depths matter for each downstream task.

### Why residual connections?

GIN already has a self-connection via the `(1 + epsilon) * h` term, but this is
**inside** the MLP — after transformation.  The external residual `h = h + h_new`
preserves the **pre-transformation** representation, providing gradient shortcuts
for training 6-layer deep networks.  Without residuals, gradients must flow
through 6 × (forward_MLP + backward_MLP + fuse_MLP) = 18 sequential MLPs.

### Graph-level readout

Four pooling operations (following BRIDGES) collapse variable-size node
embeddings to a fixed graph vector:

```
graph_embs = concat(
    mean_pool(node_embs),   # average gate representation
    max_pool(node_embs),    # most activated features
    sum_pool(node_embs),    # size-sensitive aggregation
    min_pool(node_embs),    # least activated features
) → [B, 4 × 256 = 1024]
```

Sum pooling is intentionally included because it provides **graph-size
sensitivity** — the sum over 10 nodes differs from the sum over 1,000, giving
the model implicit information about circuit complexity.


## Component 5: Netlist Parser

**File:** `netlist_parser.py`

Regex-based Verilog structural netlist parser that handles ASAP7 synthesised
output including:

| Net Name Pattern       | Example                | How it's handled                          |
|------------------------|------------------------|-------------------------------------------|
| Simple wire            | `n35`                  | Matched by `\w+`                          |
| Bus indexing           | `s7[3]`, `out1[1]`     | Brackets pass through `[^)]+?`            |
| Escaped identifier     | `\add_x_1/n36`         | Matched by `\\[^\s]+` (backslash + non-whitespace) |
| Escaped bus            | `\y[80]`               | Same escaped pattern                      |
| Logic constant         | `\*Logic1*`            | Detected by LOGIC_VALUE regex, converted to `"1"` |
| Multi-line connections | `.CI(\n\add_x_1/n39 )` | `[\s\S]+?` in gate pattern spans newlines |

The parser produces:
- `gate_attrs [N, 8]` — attribute indices via `GateAttributeVocab`
- `structural_feats [N, 4]` — depth/degree via `compute_structural_features`
- `edge_index [2, E]` — directed edges (driver → sink)
- Backward-compatible `x [N, 1]` dummy features


## Parameter Budget

| Component                      | Parameters   | Memory (bf16) |
|--------------------------------|-------------|---------------|
| Gate attribute embeddings      | 832         | 1.6 KB        |
| Node encoder MLP               | 50,816      | 99 KB         |
| DAG-GIN (6 layers + JK)       | 2,907,660   | 5.5 MB        |
| **Total graph encoder**        | **2,959,308** | **~5.7 MB** |
| Q-Former (future, BERT-based) | ~130M       | ~260 MB       |
| Projection to Qwen (future)   | ~2.8M       | ~5.4 MB       |
| Qwen2.5-7B (4-bit quantised)  | ~7B         | ~4 GB         |

The graph encoder adds < 0.04% overhead to the total model size.


## File Inventory

```
libatpgllm/
├── atpgllm/
│   ├── llm/                     Causal LM training (SFT / GRPO)
│   ├── graph/                   Graph modality (this package)
│   │   ├── __init__.py          Public API re-exports
│   │   ├── gate_features.py     Attribute decomposition vocabulary (pure Python)
│   │   ├── node_encoder.py      Attribute embedding + structural features → node emb
│   │   ├── dag_gin.py           DAG-aware bidirectional GIN encoder
│   │   ├── netlist_parser.py    Verilog → PyG graph (with structural features)
│   │   ├── models_stage1.py     Stage 1 model (NetlistGNN + Q-Former)
│   │   ├── losses_stage1.py     GTC / GTM / GTG losses
│   │   ├── train_stage1.py      Stage 1 training loop
│   │   ├── stage2_model.py      Soft-prompt bridge to causal LMs
│   │   ├── dataset.py           HF / design-description datasets
│   │   ├── ARCHITECTURE.md      This document
│   │   └── scripts/             CLI entrypoints (python -m atpgllm.graph.scripts.*)
│   └── multimodal/              Reserved: future llm ↔ graph integration
└── tests/
    └── graph/
        ├── example_integration.py
        └── data/                Precomputed design descriptions (not shipped in wheel)
```


## Next Steps

1. **Upgrade Q-Former cross-attention** — replace single-token graph attention
   (`[B, 1, 768]`) with per-node attention (`[B, N, 768]`), letting 32
   learnable queries selectively attend to different circuit structures.

2. **LLM alignment projection** — Linear layer from Q-Former output (768) to
   Qwen2.5-7B embedding dimension (3584).

3. **Stage 1 pre-training** — Train graph encoder + Q-Former with
   GTC/GTM/GTG losses on (netlist, text description) pairs.

4. **Stage 2 fine-tuning** — Freeze graph encoder, train projection +
   Qwen2.5 LoRA on graph-conditioned ATPG tasks.

5. **Stage 3 GRPO** — Freeze all graph components, train only Qwen2.5 LoRA
   with fault simulation reward. Graph tokens prepended as soft prompts.
