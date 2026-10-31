# Scaling techniques for encoding large, variable-size synthesized netlists for bounded-context LLMs (focus: ATPG)

Research notes, compiled 2026-10-31. Coverage 2019–2026.

**Access caveat:** In this session the network policy blocked direct fetches of arxiv.org, openreview, ACM/IEEE, university PDF hosts, and similar sites. Paper details below come from (a) web-search result summaries and snippets of the primary sources, (b) GitHub READMEs, which were reachable, and (c) **local measurements** I made with open tools (Yosys and the Llama-3 and Mistral tokenizers). Numbers taken only from search summaries are marked "(search summary)". Verify them against the PDFs before they are relied on. Publication status: **[PR]** = peer-reviewed venue, **[PP]** = preprint or venue not verified.

**Local measurement method, used in several sections ("Local measurement")**
- Benchmarks: the EPFL combinational suite from [lsils/benchmarks on GitHub](https://github.com/lsils/benchmarks) (adder, i2c, arbiter, voter, mem_ctrl).
- Synthesis: Yosys, via the [yowasp-yosys](https://pypi.org/project/yowasp-yosys/) WebAssembly build (`synth -flatten`), then mapped to the [Nangate45 open cell library](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/tree/master/flow/platforms/nangate45) with `abc -liberty` and written out with `write_verilog -noattr`.
- Compression counts: Yosys `aigmap` (AND-node count from the ASCII AIGER header), `abc -lut 6` and `abc -lut 4`.
- Tokenizers: the Llama-3 128K-vocab BPE ([tokenizer.model from meta-llama/llama-models](https://github.com/meta-llama/llama-models/tree/main/models/llama3)), loaded with tiktoken, and the Mistral Tekken tokenizer (`tekken_240911.json`, bundled in [mistral-common](https://pypi.org/project/mistral-common/)).
- These are my own measurements, not published results. They are reproducible with the tools listed.

---

## Q1. Cone- and window-based decomposition (fan-in/fan-out cones, per-fault cones, k-hop neighborhoods, sliding windows, local subgraph sampling)

### Takeaway
Cone and window decomposition is the most widely used way to bound per-sample size in ML-for-EDA. Inductive models trained on small cones or small circuits transfer to much larger designs when the task is mostly local:
- DETECTive transfers to designs about 29x larger than its training designs.
- DeepGate4 was trained on circuits averaging about 15K gates and evaluated at 400K–1.6M gates.
- Gamora and GROOT train on 8-bit multipliers and infer on 32-bit to 1024-bit multipliers (up to 134M nodes).

I found **no published evidence that an LLM (text transformer) trained on netlist cones generalizes to full designs for ATPG**. All the transfer evidence is from GNNs or graph transformers.

### Cited Findings
**Fault-centric and test-specific ML**
- **DETECTive** uses a GNN to predict stuck-at test patterns directly, with no classical ATPG search (search summary):
  - It injects the fault into the circuit graph, finds activation and propagation paths, and builds a "cone embedding" for each primary input.
  - It reports up to 100% pattern accuracy on synthetic designs and up to 95% on realistic designs.
  - It reports that learned knowledge transfers to designs about 29x larger than the training designs.
  - The model is inductive, so it applies to unseen graphs without retraining.
  - Venue and exact circuit sizes not verified — [iris.polito.it record](https://iris.polito.it/retrieve/handle/11583/2990442/742165) **[PP / venue unverified]**
- **InF-ATPG** (Huawei) partitions circuits into **fanout-free regions (FFRs)** and uses them as the basic units (search summary):
  - Working at the FFR level shortens the RL decision sequence compared with gate-by-gate methods such as SmartATPG.
  - A "QGNN" encodes the state using logic values and SCOAP controllability/observability.
  - A DQN chooses which fan-in to backtrace through.
  - Reported results: backtracks fall 55.06% versus traditional ATPG and 38.31% versus the prior ML baseline, and fault coverage improves — [arXiv 2512.00079](https://arxiv.org/abs/2512.00079) **[PP]**
- **DEFT** (CMU) is a differentiable ATPG (search summary):
  - A reparameterization aligns a continuous objective with discrete fault-detection semantics.
  - A custom CUDA kernel handles forward/backward passes on deep circuit graphs, with gradient normalization against vanishing gradients.
  - Against a leading commercial tool, pattern count falls 27.3% on average and up to 75.9%. v1 also reports +21.1% and +48.9% HTD fault detection on two industrial benchmarks at the same pattern budget.
  - One search summary said DEFT duplicates the fault site's fan-out cone into fault-free and faulty copies. A second search found no fan-out-cone discussion in DEFT, so this detail is unconfirmed — [arXiv 2512.23746](https://arxiv.org/abs/2512.23746) **[PP]**
- **Neural backtrace for FAN/PODEM** (Zhang & Davoodi, 2024) (search summary):
  - A neural network combines several testability metrics to make backtrace decisions, where classic ATPG uses a single metric.
  - A hybrid variant builds and applies the model only at selected circuit levels.
  - [NSF PAR copy](https://par.nsf.gov/servlets/purl/10541552) **[PR, venue not verified in session]**
- **GCN testability analysis** (Ma et al., DAC 2019) (search summary):
  - A GCN classifies observation-point candidates, starting from SCOAP node features.
  - The authors say the GCN "is scalable to process a graph containing millions of nodes and edges efficiently."
  - Versus a commercial testability flow: 11% fewer observation points and 6% fewer test patterns at similar fault coverage.
  - [NVIDIA Research page](https://research.nvidia.com/publication/2019-06_high-performance-graph-convolutional-networks-applications-testability-analysis); [CUHK PDF](https://www.cse.cuhk.edu.hk/~byu/papers/C85-DAC2019-GCN-Test.pdf) **[PR: DAC 2019]**
- A GNN for gate-level testability prediction in the presence of unknown (X) values proposes a new graph representation of gate-level netlists — [ASP-DAC 2025, DOI 10.1145/3658617.3697753](https://dl.acm.org/doi/pdf/10.1145/3658617.3697753) **[PR]**
- Classic (non-ML) SAT-based ATPG keeps constraints local to the **fault's fan-out cone**, and finds the cone boundary and internal vertices by graph traversal (search summary) — [JEIT 2026, "Optimizing SAT-Based ATPG Systems"](https://www.jeit.ac.cn/en/article/doi/10.11999/JEIT260025) **[PR, Chinese journal]**

**Cone and window models in representation learning**
- **DeepGate4** splits a large circuit graph into small **cones** and encodes them **level by level**, which gives sub-linear memory:
  - Training used cone depth k=8 and stride δ=6 (search summary).
  - Offline "historical embeddings" initialize nodes in the overlap region between cones before each mini-batch (search summary).
  - Trained on circuits averaging about 15K gates; evaluated on circuits up to 400K gates.
  - From 400K to 1.6M gates, inference memory stays roughly constant, while baseline GNN memory grows linearly.
  - Versus DeepGate3 with the same partitioning, it reports 77.9% lower inference time and 92.7% lower memory on ITC99 (secondary summary, chatpaper).
  - [arXiv 2502.01681](https://arxiv.org/abs/2502.01681); [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/29131); [code](https://github.com/zyzheng17/DeepGate4-ICLR-25) **[PR: ICLR 2025]**
- **HOGA** (hop-wise gated attention) precomputes **k-hop features** for each node:
  - A gated self-attention then works on those per-node features alone, without touching the graph topology during training.
  - Nodes can therefore be trained independently (embarrassingly parallel). Training time falls almost linearly as compute grows.
  - Search summaries report a 46.76% error reduction for QoR prediction (OpenABC-D) and +10.0% accuracy on functional-block reasoning over unseen technology-mapped netlists versus GNNs. I matched these figures to the abstract wording from memory, so check them.
  - [DAC 2024 PDF](https://www.csl.cornell.edu/~zhiruz/pdfs/hoga-dac2024.pdf); [code](https://github.com/cornell-zhang/HOGA) **[PR: DAC 2024]**
- **Gamora** reasons over AIGs with a GNN, where message passing mimics symbolic propagation and structural hashing (search summary):
  - Accuracy is about 100% on CSA multipliers, >97% on Booth multipliers, and >92% after technology mapping.
  - The abstract says it scales to Boolean networks with more than 33M nodes. A later version claims billions of nodes, so the two versions conflict.
  - It runs up to six orders of magnitude faster than ABC.
  - It claims to generalize "from simple to complex" networks. The repo workflow trains on mult8 and tests on mult32.
  - [arXiv 2303.08256](https://arxiv.org/abs/2303.08256); [repo](https://github.com/Yu-Utah/Gamora) **[PR: DAC 2023]**
- **SAT-based ATPG gap:** The prompt mentioned SAT and timing prediction. Apart from PreRoutGNN (Q2), I did not get sources in this session showing cone- or k-hop-sampled SAT GNNs generalizing to larger instances.

**Text-transformer (LLM-style) evidence on netlist chunks**
- **BERT-HIT** treats gate-level netlists as text for hardware-Trojan detection. Its abstract does not say how long netlists are split (search summary) — [ISQED 2026 program entry](https://isqed.org/English/Archives/2026/Technical_Sessions/148.html) **[PR]**
- An earlier deep-learning Trojan detector uses **block-based** extraction: netlist blocks are extracted, translated with NLP, classified, and combined by a final voter. This is a non-LLM precedent for chunk-then-vote — [QUB Pure record](https://pure.qub.ac.uk/en/publications/deep-learning-based-hardware-trojan-detection-with-block-based-ne/) **[PR]**

### Inferences
- **A proposed fault-centric encoding for ATPG.** The minimal "cone of influence" for detecting a stuck-at fault at site f is the union of three parts:
  - the fan-in cone of f (activation and justification);
  - the fan-out cone of f up to observable POs or scan cells (propagation);
  - the fan-in cones of side inputs along the sensitized paths.

  This follows ATPG theory, and DETECTive's "activation and propagation path" cone embedding and the fan-out-cone locality in SAT-ATPG are partial versions of it. For logic deep in a large design these cones can grow very large. Every work above therefore truncates them, by depth (DeepGate4 k=8), by region (FFRs in InF-ATPG), or by hop count (HOGA). Each also passes **summary features at the cut**, such as SCOAP in Ma et al. 2019 and InF-ATPG, or historical embeddings in DeepGate4.
- **For an LLM tokenizer, the same approach would look like this.** Serialize a depth-bounded cone around the fault site. Replace each cut-off frontier signal with a compact boundary token carrying CC0/CC1/CO (SCOAP) or a learned embedding. This is a design suggestion that follows from the cited works; it has not been tested with an LLM.
- **Why generalization holds where it does.** Small-to-large transfer (Gamora, GROOT, HOGA, DETECTive, DeepGate4) holds for tasks whose decision depends on a bounded neighborhood.
  - ATPG justification across reconvergent fan-out is less local. This is likely why the RL and backtrace works (InF-ATPG, neural backtrace) keep a classical search loop and use ML only for decision guidance.
  - For an ATPG LLM, the safer architecture is **"LLM proposes or guides, the classical engine (PODEM/SAT) verifies"**, with cones as the input unit.

### Gaps
- I found no paper in which an LLM or text transformer is trained on fan-in or fan-out cones and evaluated for length or size generalization to full designs on test or ATPG tasks.
- I found no published LLM-based ATPG system for gate-level netlists, as of the searches run on 2026-10-31.
- DETECTive's exact training and test gate counts and its venue could not be verified because the host was blocked.

---

## Q2. Graph partitioning to chunk netlists (METIS, hMETIS/KaHyPar, min-cut, register-boundary, FFR) and how cross-partition dependencies are handled

### Takeaway
Partitioning with explicit boundary repair is well proven for GNN-scale netlists. Four mechanisms recur for handling dependencies across partitions:
1. **Halo/padding**: add k levels of neighbors around each partition (PreRoutGNN).
2. **Edge re-growth**: restore the edges that cross partitions, about 10% of edges (GROOT).
3. **Historical or cached boundary embeddings**: overlapping cones initialized from offline embeddings (DeepGate4).
4. **Virtual or super nodes**: summary nodes per partition or level of a hierarchy (DE-HNN, DeepSeq3).

Register boundaries (DeepSeq3) and fanout-free regions (InF-ATPG, ctrw) are functionally meaningful cut points that suit test applications.

### Cited Findings
- **GROOT**:
  - Uses **METIS** to partition large EDA graphs and runs GraphSAGE node classification on each partition.
  - "Edge re-growth" restores the edges from each partition to its boundary nodes outside it, so message passing can cross partitions. The authors estimate boundary edges at about 10% of edges, and say recovery adds little inference cost.
  - On a 1024-bit CSA multiplier (134,103,040 nodes and 268,140,544 edges at batch size 16), it reports 59.38% less memory and 99.96% accuracy.
  - Accuracy stays at 100% up to 16 partitions and falls beyond that. Re-growth recovers up to 12.62% accuracy on a 32-bit multiplier.
  - It also uses custom GPU kernels for high-degree and low-degree nodes. The model is trained on an 8-bit multiplier and run on larger ones.
  - [arXiv 2511.18297](https://arxiv.org/abs/2511.18297); [Stevens record](https://researchwith.stevens.edu/en/publications/groot-graph-edge-re-growth-and-partitioning-for-the-verification-/) **[PR: ICCAD 2025]**
- **PreRoutGNN** (pre-route timing):
  - An **order-preserving partition** splits the circuit graph into subgraphs, and each subgraph is **padded with neighboring pin levels equal to the GNN depth**, so no receptive field is lost.
  - It reports slack-prediction R² of 0.93 versus 0.59 for the prior state of the art, over 21 real circuits.
  - It notes that earlier wirelength work partitioned the graph and merged cluster embeddings into node features, but kept a limited receptive field inside each cluster.
  - [arXiv 2403.00012](https://arxiv.org/abs/2403.00012); [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29653) **[PR: AAAI 2024]**
- **Cluster-size study:** A study cited in Z. Xie's thesis partitioned each netlist into clusters, with the cluster count set by cell count divided by values from 100 to 3000. It trained GNNs on designs of about 4K to 800K nets. I attributed this from the search snippet only — [arXiv 2206.03032](https://arxiv.org/abs/2206.03032) **[PP/thesis]**
- **DE-HNN**:
  - Models the netlist as a **directed hypergraph**, with each net a directed hyperedge from driver to sinks.
  - Adds a **hierarchy of virtual nodes** to capture long-range interactions, because one virtual node becomes a bottleneck.
  - Task: post-route wirelength and congestion from the netlist.
  - [arXiv 2404.00477](https://arxiv.org/abs/2404.00477); [PMLR v238](https://proceedings.mlr.press/v238/luo24a.html) **[PR: AISTATS 2024]**
  - I could not confirm in this session which partitioner builds the virtual-node hierarchy.
- **DeepGate4** splits the graph into overlapping cones and processes them in logic-level order, so each gate is handled once. Overlap nodes are initialized from offline historical embeddings. The authors say this update scheme works with any graph transformer — [arXiv 2502.01681](https://arxiv.org/abs/2502.01681) **[PR: ICLR 2025]**
- **DeepSeq3 ("Beyond Flat Netlist")** — a **register-boundary** decomposition (search summary):
  - It splits the circuit into combinational subgraphs separated by flip-flops, plus a higher-level **Super-Node Graph** for register-transfer structure.
  - A dual GNN learns both levels.
  - Supervision comes from state-transition matrices plus reachability pre-training.
  - It reports 18% less BMC solving time. Its predecessors DeepSeq and DeepSeq2 work at bit level and struggle on large circuits.
  - [arXiv 2608.28188](https://arxiv.org/abs/2608.28188) (Aug 2026) **[PP]**
- **InF-ATPG** uses **fanout-free regions** as the partition unit for RL-guided ATPG — [arXiv 2512.00079](https://arxiv.org/abs/2512.00079) **[PP]**
- **ctrw** (Circuit Transformer rewriting) scales a small-circuit transformer to full designs. It runs on **fanout-free windows**, sub-structures with one or more outputs, from a recent rewriting framework, and optimizes each window. Evaluated on the IWLS 2023 contest benchmarks — [arXiv 2406.04699](https://arxiv.org/abs/2406.04699) **[PR: IWLS 2024 workshop]**

### Inferences
- **Scan designs make register-boundary partitioning natural for ATPG.** With full scan, sequential ATPG becomes combinational ATPG on the logic between scan cells (pseudo-PIs/POs; standard DFT practice).
  - The natural chunks are therefore the per-scan-cell or per-PO **combinational fan-in cones**, and register-to-register super-nodes, as in DeepSeq3, carry the global context.
- **A pattern to borrow for an LLM.**
  1. Partition with METIS or hMETIS (min-cut), or along FFRs and scan boundaries.
  2. Add a k-level halo, as PreRoutGNN does.
  3. Represent the edges you cut with explicit boundary tokens or summary embeddings, as GROOT re-growth and DeepGate4 historical embeddings do.
  - GROOT's data point (about 10% boundary edges; accuracy holds to 16 partitions and then degrades) is a useful prior for chunk count versus fidelity.

### Gaps
- I found no paper that uses hMETIS or KaHyPar hypergraph partitioning specifically to chunk netlists for a **language model**. All the partition-plus-ML evidence uses GNNs or graph transformers.
- I found no quantitative comparison of boundary-handling strategies (halo vs. re-growth vs. virtual nodes) on the same netlist task.

---

## Q3. Structural compression and abstraction (AIG, strash/rewrite/fraig, cut/LUT mapping, word-level/functional blocks, NPN classes, retained hierarchy) and token reduction

### Takeaway
Re-encoding a flat netlist at a coarser granularity cuts node count by a predictable factor. In my local Yosys/ABC measurements on EPFL benchmarks:
- Standard cells ≈ **0.53–0.63x** the AIG AND-node count.
- 6-LUTs ≈ **0.16–0.27x** AND nodes (mem_ctrl: 0.22).
- 4-LUTs ≈ **0.23–0.37x** AND nodes.

Coarse nodes carry more information each: a cell type, a 16-bit truth table for a 4-LUT, or an NPN class ID. NPN canonicalization keeps the vocabulary small for small cuts: 222 classes for 4 inputs and 616,126 for 5 inputs. I found **no published LLM that tokenizes netlists by NPN class or by functional subcircuit**. The closest work uses truth tables as node features (ShortCircuit), or identifies functional blocks with GNNs (Gamora, GNN-RE).

### Cited Findings
- **Local measurement: node counts at different abstraction levels** (EPFL benchmarks after Yosys `synth -flatten`, method in header)

| design | AIG AND nodes | Nangate45 cells | 6-LUTs | 4-LUTs | cells/AND | LUT6/AND |
|---|---|---|---|---|---|---|
| adder | 1,020 | 645 | 272 | 339 | 0.63 | 0.27 |
| i2c | 1,095 | 639 | 274 | 411 | 0.58 | 0.25 |
| arbiter | 11,839 | 6,592 | 2,718 | 4,042 | 0.56 | 0.23 |
| voter | 10,239 | 5,985 | 1,629 | 2,364 | 0.58 | 0.16 |
| mem_ctrl | 41,809 | 21,983 | 9,046 | (not run) | 0.53 | 0.22 |

  Tools: [yowasp-yosys](https://pypi.org/project/yowasp-yosys/), [EPFL benchmarks](https://github.com/lsils/benchmarks), [Nangate45 lib](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/tree/master/flow/platforms/nangate45)
- **NPN class counts:** OEIS A000370 lists 2, 4, 14, 222, 616,126 NPN classes for n = 1..5 inputs.
  - 4-input: 65,536 functions fall into 222 classes. 5-input: about 4.29 billion functions fall into 616,126 classes.
  - Knuth found minimum-cost implementations for all but 6 of the 616,126 five-input classes.
  - Sources: [EPFL "Classifying Functions with Exact Synthesis" (ISMVL 2017)](https://msoeken.github.io/papers/2017_ismvl.pdf) **[PR]**; [OEIS A000370 mirror](https://dcmpx.remotevs.com/org/oeis/SL/A000370)
- Synthesis tools already use NPN canonical forms for **cut matching**: functions in one NPN class can be built by the same circuit plus input/output inversions — [CIRCT NPNClass.h docs](https://circt.llvm.org/doxygen/NPNClass_8h_source.html)
- **Functional and word-level abstraction with ML:**
  - Gamora recovers adder and multiplier structure (XOR/MAJ/half-adder roles) from flat AIGs and mapped netlists — [arXiv 2303.08256](https://arxiv.org/abs/2303.08256) **[PR: DAC 2023]**
  - GNN-RE treats each gate as a node and identifies module or sub-circuit boundaries in flattened netlists; DAGNN-RE predicts each gate's functional class (search summary) — [GNN-RE TCAD PDF](https://bpb-us-e1.wpmucdn.com/wp.nyu.edu/dist/4/4483/files/2021/09/alrahis21_GNN-RE_TCAD_compressed.pdf) **[PR: IEEE TCAD]**
  - A thesis notes that exhaustive sub-circuit enumeration "cannot scale up to circuits containing thousands of logic cells" (search summary) — [NTU thesis](https://tdr.lib.ntu.edu.tw/handle/123456789/8353?locale=en)
  - A March 2026 preprint learns netlist representations from LLM-generated (possibly flawed) RTL. It evaluates sub-circuit boundary identification and component classification from operator level up to IP level — [arXiv 2603.09161](https://arxiv.org/abs/2603.09161) **[PP]**
- **Truth tables as node tokens:** ShortCircuit represents each logic node by its **truth table** and predicts the next AND node.
  - v2 reports success on 98% of 8-input test truth tables (v1: 84.6%) and circuits 18.62% smaller than ABC (another listing says 14.61%).
  - The test set is 500 truth tables extracted from 20 real circuits.
  - [arXiv 2408.09858](https://arxiv.org/abs/2408.09858) **[PP]**
- **Large AIG dataset:** OpenABC-D covers 29 open IPs, each synthesized with 1,500 random 20-step ABC recipes (rewrite/refactor/balance). It keeps every intermediate AIG with node-count, depth, area, and delay labels. The full dataset is 1.4 TB; the ML-ready subset is about 19 GB — [OpenABC README](https://github.com/NYU-MLDA/OpenABC) **[PR: dataset]**

### Inferences
- **LUT tokens: fewer nodes, but each costs more.** Mapping to 6-LUTs cuts node count by about 4–6x versus the AIG. Each LUT then needs a 64-bit truth table (16 hex characters, roughly 6–10 BPE tokens) plus up to 6 fan-in references, so the **token** saving is far smaller than the node saving.
- **NPN-class tokens recover more of the saving.** Using 4-input cuts or 4-LUTs with NPN-class IDs as single vocabulary tokens (222 classes), plus a short permutation/negation code, would keep most of the node reduction (about 3–4x fewer nodes than an AIG) at about one function token per node. This is an untested design proposal.
- **Cell-type tokens are already compact.** A standard-cell netlist is already a functional compression of about 1.6–1.9x versus an AIG. Keeping cell types as tokens (about 60–150 cell types per library) is almost free.
- **ATPG compatibility:**
  - Stuck-at faults are defined on cell pins of the mapped netlist. Abstraction to AIG or LUT changes the fault universe.
  - Any abstraction used for the LLM therefore needs a **pin/fault back-mapping**. Faults on internal LUT nets or merged AIG nodes map to cell-pin faults only partially.
  - Fault-equivalence collapsing (standard ATPG practice) is a separate, fault-level compression that helps here.

### Gaps
- I found no LLM or transformer work that tokenizes gate-level netlists by NPN class, by enumerated cuts, or by recognized functional blocks (adders, muxes).
- I did not obtain published ABC statistics for strash→fraig→resyn2 node reductions on industrial netlists in this session. EPFL benchmarks are already AIG-optimized, so my numbers mostly reflect cell or LUT re-encoding, not redundancy removal.
- I could not quantify the effect of abstraction on fault coverage, that is, how faithfully faults map back to cell pins.

---

## Q4. Hierarchical and multi-resolution encoding (summaries of submodules composed upward, coarsening, hierarchical transformers)

### Takeaway
Two families are active:
1. **Graph-level hierarchy**: super-node graphs at register boundaries (DeepSeq3), hierarchical virtual nodes (DE-HNN), hierarchical transformers over computation steps (TRACE), and level-wise cone composition (DeepGate4).
2. **LLM text-level hierarchy**: module-by-module summaries composed into a design-level description, mainly for RTL (CROP, VeriGraphi, MAHL, CodeV).

No work I found applies LLM tree-of-summaries to **flattened gate-level** netlists, where the module boundaries are gone. Boundary recovery (GNN-RE-style) would be a prerequisite.

### Cited Findings
- **DeepSeq3**: a two-level hierarchy — combinational subgraphs between flip-flops, and a Super-Node Graph of register-transfer structure — learned with a dual GNN; 18% less BMC time (search summary) — [arXiv 2608.28188](https://arxiv.org/abs/2608.28188) **[PP, Aug 2026]**
- **TRACE ("Learning to Compute on Circuit Graphs")**:
  - A **hierarchical transformer** treats each computation step as an ordered sequence and applies itself recursively along graph dependencies.
  - "Function shift learning" predicts the gap between the true global function and a local approximation.
  - One encoder covers RTL, AIG, and post-mapping netlists, evaluated on ITC, OpenCores, ISCAS'89, ForgeEDA, and DeepCircuitX.
  - The repo is named TRACE_DAC26 and the DeepGate4 README announces it (June 2026). I did not confirm the DAC 2026 acceptance independently.
  - [arXiv 2509.21886](https://arxiv.org/abs/2509.21886); [DeepGate4 README news](https://github.com/zyzheng17/DeepGate4-ICLR-25) **[PP / DAC'26 claimed]**
- **DE-HNN**: a hierarchy of virtual nodes over the netlist hypergraph for long-range interactions — [PMLR v238](https://proceedings.mlr.press/v238/luo24a.html) **[PR: AISTATS 2024]**
- **Generic coarsening hierarchies (non-circuit):**
  - Hierarchical Scalable Graph Transformer merges nodes through a surjective coarsening map, applied recursively — [arXiv 2305.02866](https://arxiv.org/abs/2305.02866) **[PR: IJCAI 2023 per my knowledge; not verified in session]**
  - SHAKE-GNN builds coarsened graph hierarchies from random-forest partitions — [arXiv 2509.22100](https://arxiv.org/abs/2509.22100) **[PP]**
  - Multilevel graph embedding runs expensive embeddings on coarsened levels — [Quiring & Vassilevski](https://pdxscholar.library.pdx.edu/mth_fac/296) **[PR]**
- **LLM module summaries for RTL:**
  - **CROP** iteratively summarizes each RTL module into a structured description (I/O, interactions with other modules, optimizations), because raw RTL of large designs can exceed the LLM context — [arXiv 2507.02128](https://arxiv.org/abs/2507.02128) **[PP]**
  - **VeriGraphi** (April 2026) builds a knowledge graph of module hierarchy, ports, and inter-module dependencies, with per-module summaries. It names lost context across modules as a core LLM failure mode — [arXiv 2604.14550](https://arxiv.org/abs/2604.14550) **[PP]**
  - **MAHL** does hierarchical description and then leaf-first RTL generation, ordered by module dependency analysis (search summary) — [emergentmind topic](https://www.emergentmind.com/topics/mahl.md) **[PP]**
  - **CodeV** uses multi-level summarization to build description–code pairs — [alphaXiv overview 2407.10424](https://www.alphaxiv.org/overview/2407.10424) **[PP]**
- **Functional-block identification as a hierarchy builder:** Gamora (functional blocks in flat netlists) and GNN-RE (module boundaries) can rebuild a hierarchy from a flattened netlist — [Gamora](https://arxiv.org/abs/2303.08256), [GNN-RE](https://bpb-us-e1.wpmucdn.com/wp.nyu.edu/dist/4/4483/files/2021/09/alrahis21_GNN-RE_TCAD_compressed.pdf)

### Inferences
- **A three-tier encoding for an ATPG LLM:**
  - (i) **Gate tier:** serialized local cones around fault sites (Q1).
  - (ii) **Block tier:** embeddings or summaries of FFRs, recognized functional blocks, or combinational blocks between scan cells. These come from a graph encoder such as DeepGate4/TRACE-style models, or from LLM-written summaries.
  - (iii) **Design tier:** a super-node graph of scan cells, register groups, and module boundaries (DeepSeq3-style).

  The LLM attends over the tier-(iii) and tier-(ii) tokens plus the full-detail tier-(i) tokens for the fault in question. This combines the cited techniques and has not been validated.
- **Keep the hierarchy when exporting the netlist.** Industrial ATPG netlists are often hierarchical, and the module hierarchy usually survives synthesis unless the netlist is flattened. Keeping it (no `-flatten`) gives free tier boundaries. This is a general EDA observation, not a cited result.

### Gaps
- I found no study measuring how much ATPG-relevant information (controllability, observability, reconvergence) survives hierarchical summarization.
- I found no multi-resolution LLM (text) encoder for gate-level netlists.

---

## Q5. Retrieval-augmented and agentic netlist access (tools instead of ingestion)

### Takeaway
Agentic EDA systems pass tool calls and **structured queries** to Yosys, OpenROAD/OpenLane, simulators, or report databases instead of putting whole artifacts in the prompt. The strongest evidence at industrial scale is NVIDIA's Timing Analysis Agent:
- A single timing report can hold more than 16,000 paths, and multi-report context exceeds LLM token limits.
- The agent writes Python to query a structured report database instead.
- It reaches 98% and 90% pass rates on industrial designs.

I found no published agent that answers ATPG questions through fan-in-cone or netlist-graph query tools on million-gate designs. The pattern itself is shown to work for timing reports and for flow orchestration.

### Cited Findings
- **Timing Analysis Agent (NVIDIA):**
  - One max or xtalk_max report can contain more than 16,000 timing paths for an industrial design. Multi-report analysis "exceeds the token limits" of current LLMs.
  - The agent builds a report-retrieval plan from a Timing Debug Relation Graph. Expert Report Agents **generate Python code to query a structured report database** (nested dictionaries).
  - Pass rates: 98% single-report, 90% multi-report on industrial designs; more than 46% better than other RAG techniques.
  - [arXiv 2504.11502](https://arxiv.org/abs/2504.11502) **[PP]**
- **MCP4EDA:**
  - An MCP server that gives an LLM access to Yosys synthesis, Icarus Verilog simulation, OpenLane place-and-route, GTKWave, and KLayout.
  - The LLM inspects tool outputs and revises the synthesis TCL.
  - Reported 15–30% timing-closure improvement and 10–20% area reduction versus default flows.
  - [arXiv 2507.19570](https://arxiv.org/abs/2507.19570) **[PP]**
  - A separate community "EDA MCP server" exposes Yosys synthesis to gate-level netlists, Icarus simulation, and OpenLane — [agentpedia listing](https://agentpedia.codes/mcp/eda-mcp-server)
- **ChatEDA:** an LLM ("AutoMage") with EDA tools as executors, covering RTL→GDSII task decomposition, script generation, and execution. Fine-tuned on 1,500 self-generated samples, which were not released.
  - Described in [EDA Corpus, arXiv 2405.06676](https://arxiv.org/abs/2405.06676) **[venue of ChatEDA not verified in session]**
- **ORAssistant:** a RAG conversational assistant for OpenROAD, judged by an LLM against Gemini 1.5 Flash and GPT-4o baselines. It covers Q&A only and has no evaluation at design scale — [arXiv 2410.03845](https://arxiv.org/abs/2410.03845) **[PP]**
- **VerilogCoder (NVIDIA):**
  - Planning uses a Task and Circuit Relation Graph (TCRG) linking tasks to signals and transitions.
  - An **AST-based waveform-tracing tool** traces mismatching signals back through the code.
  - Reports 94.2% correct Verilog on VerilogEval-Human v2, 33.9% above prior state of the art.
  - Search summaries said AAAI 2024; the arXiv date (Aug 2024) points to AAAI 2025. Verify.
  - [NVIDIA Research page](https://research.nvidia.com/index.php/publication/2024-08_verilogcoder-autonomous-verilog-coding-agents-graph-based-planning-and-abstract) **[PR: AAAI]**
- **Knowledge-graph retrieval of netlist fragments:**
  - **AMSnet-KG** (analog) stores netlists in a Neo4j graph schema. It retrieves precise subgraphs from structure and explicit triplets rather than dense embeddings — [arXiv 2411.13560](https://arxiv.org/abs/2411.13560) **[PP]**
  - **BRIDGES** builds dataflow and netlist graphs for LLMs, with tasks such as function retrieval and description — [arXiv 2504.05180](https://arxiv.org/abs/2504.05180) **[PP]**
- **Repository-scale RTL retrieval:** RTLRepoCoder fine-tunes on Verilog repositories at a **10K context** and uses **retrieval** for longer inputs. It reports state of the art on RTL-Repo — [arXiv 2504.08862](https://arxiv.org/abs/2504.08862) **[PP]**

### Inferences
- **A concrete tool API for an ATPG LLM:**
  - `fanin_cone(sig, depth)` and `fanout_cone(sig, depth)`
  - `scoap(sig)`
  - `implications(assignments)` (call the ATPG engine's implication procedure)
  - `simulate(pattern)` and `fault_sim(pattern, faults)`
  - `path_to_observable(fault)`

  These can be backed by a netlist database (Yosys JSON, PyVerilog, or an in-memory graph). The Timing Analysis Agent's LLM-writes-queries pattern carries over directly.
- **Agentic access removes the context-size ceiling but adds latency.** It works best combined with an **encoder** (Q1–Q4) that gives the LLM a compressed global view, so it knows *where* to query.

### Gaps
- I found no agent evaluation on gate-level netlists of 10^5–10^6+ gates that measures accuracy versus netlist size.
- I found no LLM-plus-ATPG-engine tool loop in the literature, as of 2026-10-31.
- I could not check in this session whether MCP4EDA or ChatEDA ever pass netlists to the LLM directly.

---

## Q6. Long-context architectures, SSM/Mamba, linear-attention graph transformers, and tokens per gate

### Takeaway
Measured cost of a synthesized standard-cell netlist (Llama-3 tokenizer):
- Raw Yosys Verilog: **about 50–59 tokens per gate**.
- One-line compact Verilog with short net names: **about 25–36**.
- `.bench`-style `n = NAND(a,b)`: **about 12–21**.
- Typed edge list: **about 15–20**.
- Costs rise slowly with design size, because longer net IDs need more number tokens.

At about 20 tokens per gate, a 128K window holds about 6K gates and a 1M window about 50K gates. A 1M-gate netlist needs about 20M tokens even in the leanest text format. Effective context is also often well below the advertised length (RULER).

Linear and sparse attention graph transformers (DeepGate4) handle 400K–1.6M-gate AIGs at roughly constant inference memory. **I found no Mamba/SSM model applied to netlists or circuits.**

### Cited Findings
- **Local measurement: tokens per gate** (Nangate45-mapped EPFL netlists; "gates" = cell instances; method in header)

| design | cells | raw Yosys file tok/gate (Llama-3 / Tekken) | instance text only | compact 1-line Verilog | .bench-style | ASCII AIGER tok per AND |
|---|---|---|---|---|---|---|
| adder | 645 | 58.6 / 70.9 | 40.5 | 24.6 / 33.6 | 12.5 / 18.9 | 7.9 |
| i2c | 639 | 52.0 / 68.9 | 41.0 | 27.6 / 37.6 | 13.5 / 20.9 | 7.5 |
| arbiter | 6,592 | 53.2 / 72.7 | 46.0 | 31.8 / 43.6 | 17.0 / 25.8 | 8.6 |
| voter | 5,985 | 50.0 / 68.1 | 40.3 | 29.0 / 38.8 | 15.6 / 22.8 | 8.9 |
| mem_ctrl | 21,983 | 51.7 / 71.8 | 44.1 | 32.6 / 46.0 | 17.4 / 27.4 | 8.8 |

  - Example compact line: `OAI22_X1 U10991 (.A1(n13340), .A2(n13482), .B1(n13483), .B2(n13341), .ZN(n13501));`
  - Example .bench-style line: `n13501 = OAI22(n13340, n13482, n13483, n13341)`
  - **mem_ctrl (about 22K cells) as raw Yosys Verilog = 1,137,501 Llama-3 tokens (2.49 MB)**. A 22K-gate netlist already exceeds a 1M-token window in its native text form.
  - AIGER costs about 8 tokens per AND, but there are about 1.6–1.9x more ANDs than cells, so about 12–17 tokens per equivalent cell.
  - Tools: [Llama-3 tokenizer](https://github.com/meta-llama/llama-models/tree/main/models/llama3), [mistral-common Tekken](https://pypi.org/project/mistral-common/), [yowasp-yosys](https://pypi.org/project/yowasp-yosys/), [EPFL benchmarks](https://github.com/lsils/benchmarks)
- **Local measurement: scaling with net-ID width.** I remapped the arbiter netlist's nets to random IDs drawn from ranges of 2N:

| scale of IDs (N) | compact Verilog (Llama-3 / Tekken) | .bench (Llama-3 / Tekken) | typed edge list "NAND2 out in1 in2" (Llama-3 / Tekken) |
|---|---|---|---|
| 10K | 33.4 / 47.4 | 18.6 / 29.6 | 15.6 / 26.6 |
| 100K | 33.7 / 52.3 | 18.8 / 33.5 | 15.8 / 30.5 |
| 1M | 35.8 / 57.4 | 20.8 / 37.5 | 17.9 / 34.6 |
| 10M | 38.4 / 62.2 | 22.5 / 41.4 | 19.6 / 38.5 |

  - Llama-3 groups digits in threes, so cost grows slowly with ID width.
  - Tekken splits digits one per token, so it is about 1.5–2x costlier on netlists.
- **Effective vs. advertised context:** RULER evaluated 17 open models that all claim 32K or more. Only about half held satisfactory performance at 32K, against a threshold of 85.6% (Llama2-7B at 4K). Nearly all models fell below the threshold before their claimed length. Vanilla needle-in-a-haystack hides large drops on multi-hop tracing and aggregation tasks, which resemble tracing a netlist path — [NVIDIA RULER repo](https://github.com/nvidia/RULER) **[PR: COLM 2024 per my knowledge]**
- **Long-context degradation on hardware code:**
  - RTL-Repo has 4,000+ Verilog samples paired with full repository context. Context ranges from about 2K to 64K tokens per one summary, or 2K to 128K per another, so the two disagree. Performance degrades substantially as context grows — [arXiv 2405.17378](https://arxiv.org/abs/2405.17378); [alphaXiv overview](https://www.alphaxiv.org/overview/2405.17378) **[PP]**
  - SPICED (analog netlists, Trojan detection) cites GPT-3.5-turbo's 16,385-token limit as a constraint — [arXiv 2408.16018](https://arxiv.org/abs/2408.16018) **[PP]**
  - NetTAG reports that a general LLM (o1-preview) struggles with gate-level function identification (search summary; NetTAG itself is out of scope here) — [arXiv 2504.09260](https://arxiv.org/abs/2504.09260)
- **Advertised windows in 2026:** frontier models list about 1M tokens; Llama 4 Scout lists 10M, but its long-context benchmarks were reported at 128K. These come from secondary blog aggregators, so treat the specific numbers as approximate — [morphllm comparison](https://www.morphllm.com/llm-context-window-comparison)
- **Sparse/linear attention on netlists — DeepGate4:**
  - A GAT-based **sparse transformer with global virtual edges** gives linear time and memory per mini-batch.
  - A CUDA inference kernel exploits AIG sparsity.
  - Memory stays roughly constant from 400K to 1.6M gates.
  - It reports +15.5% and +31.1% over the next-best models on ITC99 and EPFL.
  - The repo benchmarks it against GraphGPS, Exphormer, DAGformer, and PolarGate.
  - [arXiv 2502.01681](https://arxiv.org/abs/2502.01681); [repo README](https://github.com/zyzheng17/DeepGate4-ICLR-25) **[PR: ICLR 2025]**
- **HOGA**: per-node hop-wise attention, which avoids full-graph attention and parallelizes training across nodes — [cornell-zhang/HOGA](https://github.com/cornell-zhang/HOGA) **[PR: DAC 2024]**
- **SSM/Mamba:** searches for Mamba or state-space models applied to netlist representation found only **hardware accelerators for Mamba** (e.g., MARCA, ICCAD'24) and complexity theory placing Mamba in the same circuit-complexity class as transformers. Nothing applies an SSM to netlists — [MARCA arXiv 2409.11440](https://arxiv.org/abs/2409.11440); [Mamba circuit-complexity limits, PMLR v280](https://proceedings.mlr.press/v280/chen25b.html)

### Inferences
- **Back-of-envelope sizing, Llama-3-like tokenizer, leanest text (about 18–22 tokens per gate):**
  - 32K tokens ≈ 1.5K gates.
  - 128K ≈ 6K gates.
  - 1M ≈ 45–55K gates.
  - 10M ≈ 0.5M gates.
  - With raw synthesized Verilog (about 52 tokens per gate) these shrink about 2.5x; 128K ≈ 2.4K gates.
  - A 1M-gate design needs about 20M tokens (lean) to about 50M tokens (raw). That is beyond any 2026 window and far beyond RULER-style effective lengths.
  - **Whole-netlist ingestion is only viable at the "hundreds to low tens of thousands of gates" end** of the user's range. Million-gate designs need decomposition (Q1/Q2), compression (Q3), hierarchy (Q4), or tools (Q5).
- **Cheap format wins worth about 3x:**
  - Rename nets to short sequential IDs.
  - Drop drive-strength suffixes (`_X1`) and pin names where the pin order is canonical.
  - Remove wire declarations.
  - Use one line per gate in `.bench`/edge-list form.
- **Graph-native tokens beyond that:** Larger gains require **graph-native tokens**: one token or embedding per gate from a graph encoder (DeepGate4/HOGA-style), injected as soft tokens. That gives 1 token per gate instead of about 20, or fewer with cone or block pooling.
- **SSMs may suit topologically ordered netlist streams.** Their linear-time recurrence over a topologically sorted gate stream fits combinational DAGs, where each gate depends only on earlier gates in topological order. But an SSM compresses history into a fixed state and cannot retrieve exact arbitrary earlier signals, which a netlist's long-range fan-in references require. This is an untested hypothesis with no circuit evidence found.

### Gaps
- I found no published tokens-per-gate measurements for netlists in the literature. The numbers above are my own measurements and should be labeled that way.
- I found no evaluation of 128K–1M-context LLMs on gate-level netlist tasks as a function of netlist size.
- I found no Mamba/SSM or linear-attention **language model** work on netlists. Only graph-transformer evidence exists (DeepGate4).
- Tokenizers for proprietary frontier models (Claude, GPT, Gemini) were not measured. Expect the same order of magnitude, 15–60 tokens per gate depending on format.

---

## Q7. Circuit-specific sequence representations (Circuit Transformer, ShortCircuit, generative logic synthesis, AIG serialization order, length generalization)

### Takeaway
Next-gate generative transformers over AIGs work only on **small** circuits:
- Circuit Transformer: 8-input, 2-output circuits; 88M parameters.
- ShortCircuit: 8-input truth tables.

Both scale to real designs **only by windowing**. ctrw applies Circuit Transformer to fanout-free windows inside IWLS 2023 benchmarks. The key serialization ideas are:
- Circuit Transformer: a memoryless depth-first trajectory with an equivalence-preserving masked decoder.
- ShortCircuit: truth-table node features.

I found no published result showing **length generalization** of these sequence models to larger circuits.

### Cited Findings
- **Circuit Transformer** (Li et al., UCL/Huawei Noah's Ark):
  - Encodes an AIG as a **memoryless depth-first traversal trajectory** and predicts the next gate.
  - An equivalence-preserving decoder masks any token that would break equivalence to the target function ("cutoff properties").
  - An 88M-parameter model trained on 8-input, 2-output circuits, combined with MCTS, beats ABC `resyn2` while keeping strict equivalence.
  - [arXiv 2403.13838](https://arxiv.org/abs/2403.13838); [ICLR 2025 poster](https://www.iclr.cc/virtual/2025/poster/28560); [ICLR proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/2f75a57e9c71e8369da0150ea769d5a2-Paper-Conference.pdf); [code](https://github.com/snowkylin/circuit-transformer) **[PR: ICLR 2025]**
- **ctrw** ("Logic Synthesis with Generative Deep Neural Networks"):
  - A rewriting operator that runs Circuit Transformer on **fanout-free windows**.
  - Two-stage training, an MCTS reward based on AND-count change, and integration with DAG-aware rewriting.
  - Evaluated on IWLS 2023 contest benchmarks. I did not obtain per-benchmark numbers.
  - [arXiv 2406.04699](https://arxiv.org/abs/2406.04699) **[PR: IWLS 2024 workshop]**
- **ShortCircuit:**
  - Nodes are represented by **truth tables**; a transformer predicts the next AND node; an AlphaZero variant searches.
  - v2: 98% success on 8-input test truth tables and 18.62% smaller than ABC. v1 and another listing give 84.6% and 14.61%, so the versions conflict.
  - [arXiv 2408.09858](https://arxiv.org/abs/2408.09858) **[PP]**
- **TRACE** treats each computation step as an **ordered sequence** within a hierarchical transformer, applied recursively along the DAG. This is a hybrid of sequence and graph that avoids flat serialization — [arXiv 2509.21886](https://arxiv.org/abs/2509.21886) **[PP / DAC'26 claimed in repo]**
- **Local measurement: ASCII AIGER as a serialization** — one line per AND (`lhs rhs0 rhs1`, literals = 2·var + complement) costs about 7.5–8.9 Llama-3 tokens per AND node on EPFL designs (Q6 table). It is a topological-order serialization: AIGER requires an AND's inputs to have smaller indices in binary form — [EPFL benchmarks](https://github.com/lsils/benchmarks) for data; the AIGER rule is standard format knowledge

### Inferences
- **Choosing an AIG serialization:**
  - **Topological order with explicit references** (AIGER/.bench-like) is compact. But each gate references arbitrary earlier IDs, which needs long-range attention or retrieval.
  - **DFS/tree trajectories** (Circuit Transformer) localize references, but duplicate or need back-pointers for shared, reconvergent nodes. Reconvergent fan-out is exactly what makes ATPG hard.
  - For ATPG, keep explicit node IDs so reconvergence stays visible, and order gates by topological level inside each cone.
- **The repeated scaling recipe.** Generative transformers are reliable on windows of about 8 inputs and tens of gates. Large-design applications (ctrw) keep them local and let a classical framework do the global composition. An ATPG LLM should expect the same split: local reasoning by the model, global bookkeeping by tools.

### Gaps
- I found no study of length or size generalization (train on N-gate sequences, test on 10N) for Circuit Transformer, ShortCircuit, or any other AIG sequence model.
- I found no comparison of serialization orders (topological vs. DFS vs. BFS-by-level) for netlist LLMs.
- I could not retrieve ctrw's per-benchmark numbers or window-size parameters, because arXiv was blocked.
