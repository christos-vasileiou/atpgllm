# Graph-Text Multimodal Architecture for Circuit ATPG

## Overview

This document is the executable contract for the graph-conditioned ATPG
pipeline. It transforms a structural gate-level netlist plus a target stuck-at
fault into fixed-dimensional embeddings, aligns those embeddings with text,
and injects them into the current QLoRA causal-LM path for SFT and GRPO.

The implementation separates inference inputs from supervision. The netlist
and target fault produce graph features. Ground-truth propagation gates,
backtrack gates, snapshots, expected outputs, and detected faults remain labels
or reward inputs; they are never graph-encoder inputs.

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
|   → A attribute indices|
+-----------+-----------+
            |
            v
+-----------+-----------+
| Attribute Decomp.     |   node_encoder.py
| Encoder (nn.Module)   |
|                       |
| A learned embeddings  |   [N, A] int → [N, A*d_attr] embedded
| + structure + fault   |   [N, 9] float (depth, degree, target)
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
    Q-Former (implemented)
    32 learnable queries cross-attend to N gate tokens
    → [B, 32, d_q] → MLP → [B, 32, d_LM]
    → prepend as soft prompts to the causal LM
```

The parser also attaches five target-fault features to each node:
`is_fault_site`, `drives_fault_net`, `reads_fault_net`, `stuck_at_zero`, and
`stuck_at_one`. A PI fault therefore conditions its sink gates even when no
gate drives the target net.


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

Each cell can be decomposed into eight categorical/binary attributes. The
checkpointed production contract currently enables the four entries in
`ATTRIBUTE_NAMES`: `logic_family`, `input_count`, `is_sequential`, and
`num_outputs`. Changing this ordered list intentionally changes the vocabulary
fingerprint and invalidates incompatible checkpoints.

| # | Attribute             | Classes | Examples                              | Why it matters for ATPG             |
|---|-----------------------|---------|---------------------------------------|-------------------------------------|
| 0 | `logic_family`        | 23      | AND, OR, AOI, DFF, FA, TIE, ...      | Determines boolean behaviour and fault sensitisation conditions |
| 1 | `input_count`         | 7       | 1, 2, 3, 4, 5, 6+                    | More inputs = harder controllability; more potential fault masking |
| 2 | `output_complemented` | 3       | UNKNOWN / True / False                | Inverted output changes fault effect polarity (s-a-0 vs s-a-1 detection) |
| 3 | `is_sequential`       | 3       | UNKNOWN / DFF,LATCH / combinational   | Sequential boundaries break fault propagation; require scan chains |
| 4 | `drive_strength`      | 11      | TINY(<0.5), X1, X2, X4, X16+         | Higher drive = more fanout capacity; affects timing-aware ATPG |
| 5 | `num_outputs`         | 3       | 1 (most), 2 (FA: carry+sum), special  | Multi-output gates create coupled fault effects |
| 6 | `tristate`            | 3       | UNKNOWN / True / False                | Placeholder for future PDKs with tristate buffers |
| 7 | `is_clock_related`    | 3       | UNKNOWN / ICG,CKINVDC / normal        | Clock gates require special ATPG handling (launch/capture) |

Every enabled attribute has an explicit index-0 UNKNOWN value. Unknown cell
types are retained by the parser, using common output-pin names as a
conservative connectivity fallback. `GateAttributeVocab.to_dict()` serializes
labels, active ordering, cells, and indices; its SHA-256 fingerprint is checked
at every checkpoint transition.

### Why not monolithic embeddings?

| Criterion                | Monolithic (202×256)    | Attribute decomposition |
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
gate_attrs [N, A] (int64)          structure [N, 4] + fault [N, 5]
     |                                       |
     v                                       |
A × nn.Embedding(n_classes, d_attr)           |
     |                                       |
     v                                       |
 concat → [N, A*d_attr]                      |
     |                                       |
     +------------------+--------------------+
                        |
                        v
                   concat → [N, 132]
                        |
                        v
              Linear(A*d_attr+9, 128) + LayerNorm + GELU
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
- `gate_attrs [N, A]` — versioned attribute indices via `GateAttributeVocab`
- `structural_feats [N, 4]` — depth/degree via `compute_structural_features`
- `edge_index [2, E]` — directed edges (driver → sink)
- `fault_feats [N, 5]` — target location/direction and stuck-at polarity
- Backward-compatible `x [N, 1]` dummy features

Dataset records additionally carry `propagation_mask`, `backtrack_mask`, and
`discrepancy_mask` with per-graph validity flags. The discrepancy mask compares
Good Machine and Bad Machine values on each gate's output nets. These tensors
supervise encoder pretraining only.


## Shape Contracts

- Graph batch: PyG `Batch` with `gate_attrs [N,A]`,
  `structural_feats [N,4]`, optional `fault_feats [N,5]`,
  `edge_index [2,E]`, and `batch [N]`.
- DAG encoder: `node_embs [N,d_g]`, `graph_embs [B,4*d_g]`.
- Q-Former: densifies nodes with a valid-node mask and returns
  `query_embs [B,Q,d_q]`; padded nodes are masked in cross-attention.
- LM projector: `query_embs -> graph_prefix [B,Q,d_LM]`.
- SFT input: `[graph_prefix, token_embeddings]`; graph and prompt labels are
  `-100`, so only assistant tokens contribute to causal-LM loss.
- GRPO: grouped completions are generated from the same graph prefix. Policy,
  old-policy, and frozen-reference log probabilities all include their
  corresponding graph prefix.


## File Inventory

```
libatpgllm/
├── atpgllm/
│   ├── llm/                     Causal LM training (SFT / GRPO)
│   ├── graph/                   Graph modality (this package)
│   │   ├── __init__.py          Public API re-exports
│   │   ├── gate_features.py     Attribute decomposition vocabulary (pure Python)
│   │   ├── fault_context.py     Target-fault inputs + non-leaking node labels
│   │   ├── node_encoder.py      Attribute embedding + structural features → node emb
│   │   ├── dag_gin.py           DAG-aware bidirectional GIN encoder
│   │   ├── netlist_parser.py    Verilog → PyG graph (with structural features)
│   │   ├── pretrain.py          Stage-A ATPG node objectives
│   │   ├── checkpoints.py       Versioned stage-transition contract
│   │   ├── models_stage1.py     Stage 1 model (NetlistGNN + Q-Former)
│   │   ├── losses_stage1.py     GTC / GTM / GTG losses
│   │   ├── train_stage1.py      Stage 1 training loop
│   │   ├── stage2_model.py      Soft-prompt bridge to causal LMs
│   │   ├── dataset.py           HF / design-description datasets
│   │   ├── ARCHITECTURE.md      This document
│   │   └── scripts/             CLI entrypoints (python -m atpgllm.graph.scripts.*)
│   ├── training/                Conversation, datasets, GRPO trainers, tools, rewards
│   │   └── data/                Package data (sim_config.json)
│   └── multimodal/              Soft-prefix model, loading, SFT data, GRPO loss
├── scripts/
│   ├── train/                   training_code.py + Slurm launchers + configs/
│   ├── eval/                    evaluate_model.py + checkpoint eval wrappers
│   └── dataset/                 Dataset filter / token analysis CLIs
├── experiments/                 Non-packaged legacy / notebooks / scratch
└── tests/                       Pytest only
    ├── unit/
    ├── integration/
    └── graph/
        ├── test_example_integration.py
        └── data/                Precomputed design descriptions (not shipped in wheel)
```


## Executable Stages and Session Boundaries

Run from `libatpgllm/` after activating the project environment.

### Stage A — target-conditioned DAG encoder pretraining

```bash
activate && python -m atpgllm.graph.scripts.train_graph_encoder \
  --dataset chrivasileiou/asap7-language-of-test-v2 \
  --output-dir runs/graph_pretrain
```

The netlist and target fault are inputs. Three node heads supervise fault
propagation, ATPG backtracking, and Good/Bad Machine discrepancy masks. The
checkpoint exports the DAG encoder; auxiliary heads do not cross into the LLM.

### Stage B — Q-Former and graph/text alignment

```bash
activate && python -m atpgllm.graph.scripts.train_stage1 \
  --sim-config atpgllm/training/data/sim_config.json \
  --graph-ckpt runs/graph_pretrain/graph_pretrain_final.pt \
  --output-dir runs/graph_alignment --graph-policy frozen
```

This stage trains per-node Q-Former cross-attention and graph/text projectors
with GTC, GTM, and GTG. `--graph-policy` is `frozen`, `last_layer`, or `full`.
The default freezes the Stage-A encoder. Precomputed per-design descriptions
avoid contradictory captions for one circuit.

### Stage C — graph-conditioned QLoRA SFT

```bash
activate && python scripts/train/multimodal_training.py --method sft \
  --alignment-ckpt runs/graph_alignment/stage1_final.pt \
  --output-dir runs/multimodal_sft --graph-policy full
```

The rendered prompt replaces the duplicated Verilog body with a stable
`<GRAPH_CONTEXT>` payload while retaining its `doc_id`; topology is supplied by
the graph prefix. The target fault remains in user text and graph features.
Answer-only labels preserve the current conversation target. `full` jointly
updates the DAG encoder, Q-Former, LM projector, and LoRA adapter; `qformer`,
`last_layer`, and `frozen` provide explicit ablations.

### Stage D — graph-conditioned QLoRA GRPO

```bash
activate && python scripts/train/multimodal_training.py --method grpo \
  --alignment-ckpt runs/graph_alignment/stage1_final.pt \
  --sft-ckpt runs/multimodal_sft/multimodal_sft_final.pt \
  --output-dir runs/multimodal_grpo --graph-policy full
```

GRPO uses the existing fault-simulation reward factory. A frozen SFT LoRA
adapter and frozen SFT graph stack form the reference policy; a separate policy
adapter plus the selected graph modules are optimized jointly. The local
rollout loop is correctness-first and does not use vLLM.

The same commands are captured in sourceable configs:

```bash
bash scripts/train/run_graph_roadmap.sh scripts/train/configs/graph_pretrain.conf
bash scripts/train/run_graph_roadmap.sh scripts/train/configs/graph_alignment.conf
bash scripts/train/run_graph_roadmap.sh scripts/train/configs/multimodal_sft.conf
bash scripts/train/run_graph_roadmap.sh scripts/train/configs/multimodal_grpo.conf
```

## Hyperparameter Search

The executable HPO hierarchy is documented in `docs/GRAPH_HPO.md` and
configured by `scripts/train/configs/hpo_graph_pipeline.yaml`. It uses
PostgreSQL-backed Optuna grouped multivariate TPE with constant-liar sampling
and Hyperband pruning.

Ordinary trials execute one stage only:

```text
Stage-A trial -> graph_pretrain checkpoint
promoted Stage-A checkpoint -> Stage-B trial -> alignment checkpoint
promoted Stage-B checkpoint -> planned Stage-C SFT screen
promoted SFT checkpoint -> planned Stage-D GRPO pilot
```

Stage-A pruning uses held-out propagation/discrepancy/backtrack node metrics.
Stage-B pruning uses graph/text retrieval, matching average precision, and GTG
validation loss. Stage C/D remain explicit top-K launch plans until a
graph-aware simulator evaluator can select them from scheduled held-out
evaluation; training reward is not an HPO objective.

## Checkpoint Contract and Resume

Every `.pt` session boundary has format `atpgllm.graph-stack` version 1 and
stores stage, parent stage, step, vocabulary payload/fingerprint, architecture,
module states, optimizer state, and launch arguments. Allowed transitions are:

```text
graph_pretrain -> graph_text_alignment -> multimodal_sft -> multimodal_grpo
```

Each stage may also resume its own stage. Use `--resume` for optimizer/session
recovery; use the preceding stage's dedicated checkpoint argument for a fresh
transition. Legacy graph checkpoints are rejected because they cannot prove
attribute-order and shape compatibility.

## Current Limitations

- The parser targets synthesized structural netlists. Unknown cells are
  retained using conventional output-pin names, but libraries with unusual pin
  naming need an explicit gate-function entry.
- Sequential feedback is tolerated by structural-depth fallback, but this is
  not a scan-chain/timing model.
- Stage-D generation is single-prompt grouped sampling without vLLM, DDP, or
  tool-call continuation. It preserves graph gradients but is slower than the
  text-only TRL/vLLM path.
- Full 7B QLoRA SFT/GRPO requires CUDA, bitsandbytes, model access, and enough
  device memory. CPU tests cover parser, vocabulary, masks, shapes, checkpoint
  compatibility, SFT prefix labels, and the GRPO objective.
