# Circuit Graph Encoders (Circuit Foundation Models) and Their Alignment with LLMs for Bounded-Token Netlist Encoding

> Source-access caveat: in this research session, direct fetches of arxiv.org, proceedings.iclr.cc and alphaxiv.org were blocked by the sandbox network policy (DNS/CONNECT refused), and the GitHub API was not enabled for the relevant repos. Every finding below therefore comes from web-search result snippets and search-engine summaries of the primary sources (arXiv abstracts/HTML, ICLR/DAC/ASP-DAC PDFs, author pages, GitHub READMEs), plus some third-party review sites (themoonlight.io, Lacuna, paperswithcode mirrors). Numbers that came only from third-party summaries are marked as such. Before relying on an exact number, check it against the primary PDF. Allowing arxiv.org and proceedings.iclr.cc in the session network settings would make that possible.
>
> Peer-review status key: **[PR]** = peer-reviewed venue confirmed by a search result; **[PP]** = arXiv preprint only, or venue not confirmed in this session.

## Q1. Circuit graph encoders for gate-level netlists / AIGs: architecture, training objectives, and maximum demonstrated scale

### Takeaway
The AIG/netlist encoder lineage runs from GNNs that learn signal probabilities (DeepGate, 2022), through functionality-aware objectives (DeepGate2's truth-table-distance loss, FGNN/FGNN2's contrastive functional distance, PolarGate's bipolar embeddings), to graph transformers built for scale (DeepGate3, then DeepGate4). DeepGate4 [PR, ICLR 2025] has the strongest scale evidence among the learned functional encoders. It is trained on circuits averaging about 15K gates, still performs well at 400K gates, and its inference memory reportedly stays flat from 400K to 1.6M gates, where GNN baselines grow linearly. Gamora [PP/venue unverified] reports the largest raw size (more than 33M nodes), but for a narrower symbolic-reasoning task (adder/multiplier extraction). None of these encoders outputs a bounded token set by itself. They produce one embedding per gate, plus a pooled graph embedding in some cases.

### Cited Findings

**DeepGate (v1, 2022) — the origin of the cure-lab/CUHK line**
- DeepGate "utilizes attention-based aggregation function and skip connections to capture the logic computations" in combinational circuits, and it was used to improve test point insertion and Boolean satisfiability. — [DeepTPI paper (cites DeepGate)](https://arxiv.org/pdf/2206.06975); [cure-lab/DeepGate GitHub](https://github.com/cure-lab/DeepGate)
- The DeepGate line labels AIG nodes using logic-simulation-derived probability of each signal being logic '1' (signal probability). The sister work DeepSAT uses the same supervision, with "efficient logic simulation on the AIG circuits [to estimate] the probability of signals in the AIG being logic '1'". — [DeepSAT arXiv 2205.13745](https://arxiv.org/pdf/2205.13745)

**DeepGate2 (functionality-aware) [PP in this session; venue not confirmed]**
- Its training signal is the Hamming distance between truth tables of sampled gate pairs. These are incomplete truth tables obtained by fast logic simulation, and the loss makes embedding distances track truth-table differences. It uses "a scalable loss function and a one-round GNN" that trains roughly an order of magnitude faster than DeepGate. — [DeepGate2 arXiv 2305.16373](https://arxiv.org/abs/2305.16373)
- The public code trains in two stages. Stage 1 predicts logic-1 probability, and Stage 2 predicts pairwise truth-table differences. — [cure-lab/DeepGate2 GitHub](https://github.com/cure-lab/DeepGate2)
- The PolarGate authors report that DeepGate and DeepGate2 are significantly slower than other models because they rely on truth-table simulation. — [PolarGate ICCAD 2024 paper](https://www.cse.cuhk.edu.hk/~byu/papers/C233-ICCAD2024-PolarGate.pdf)

**DeepGate3 (2024) [PP in this session; venue not confirmed]**
- Architecture: a pretrained DeepGate2 GNN produces initial gate embeddings. A **Refine Transformer** then captures long-range dependencies among gates, and a **Pooling Transformer** aggregates gate-level embeddings into graph-level embeddings. The authors contrast the Pooling Transformer with the average pooling used by Gamora and HOGA. — [DeepGate3 arXiv 2407.11095](https://arxiv.org/pdf/2407.11095); [Moonlight review of DeepGate3](https://www.themoonlight.io/en/review/deepgate3-towards-scalable-circuit-representation-learning)
- A citing paper says DeepGate3 "proves the data scaling ability of circuit representation learning models", meaning performance improves with more training data. — [DeepGate3 ResearchGate listing / citing works](https://www.researchgate.net/publication/390635324_DeepGate3_Towards_Scalable_Circuit_Representation_Learning)
- DeepGate3's dense attention runs out of memory on million-gate circuits. DeepGate4 reports a **92.7% memory reduction vs. DeepGate3 on ITC99**. — [DeepGate4 arXiv v3 HTML](https://arxiv.org/html/2502.01681v3)

**DeepGate4 (ICLR 2025) [PR]**
- Problem statement: existing models "face significant challenges in scaling to large circuits due to limitations like over-squashing in graph neural networks and the quadratic complexity of transformer-based models." — [DeepGate4 arXiv 2502.01681](https://arxiv.org/abs/2502.01681)
- Three components:
  1. An update strategy tailored to circuit graphs (level-by-level graph partitioning and updating). It "reduce[s] memory complexity to sub-linear" and works with any graph transformer.
  2. A GAT-based sparse transformer with global and local structural encodings for AIGs.
  3. A fused inference CUDA kernel.
  — [DeepGate4 arXiv abstract](https://arxiv.org/abs/2502.01681); [Lacuna summary](https://lacuna.tiptreesystems.com/work/deepgate4-efficient-and-effective-representation-learning-for-circuit-design-at/wrk_80f074c713c818f4013e813519e3f6ca)
- Scale, verified as closely as possible:
  - Trained on circuits averaging about **15K gates**, it still performs well on circuits up to **400K gates**.
  - **Inference memory stays constant from 400K to 1.6M gates, while GNN baselines grow linearly.**
  - Prior approaches needed 30–40 GB of GPU memory to train on ITC99, whose largest graph has about 140K gates.
  - PNA and HOGA-5 run out of memory on the large circuits.
  — [DeepGate4 arXiv v3 HTML](https://arxiv.org/html/2502.01681v3); [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/d75f561eaaf2cb754bc8d7e36d8af362-Paper-Conference.pdf)
- Accuracy: overall loss is 15.5% (ITC99) and 31.1% (EPFL) lower than the next-best model. — [DeepGate4 arXiv abstract](https://arxiv.org/abs/2502.01681)
- Efficiency of Fused-DeepGate4. The reported numbers differ between paper versions:
  - One version: 35.1% less runtime and 46.8% less memory. — [paperswithcode mirror](https://paperswithcode.co/paper/2502.01681)
  - v3: 41.3% less runtime and 51.3% less memory on ITC99; 28.2% less runtime and 32.5% less memory on EPFL. — [DeepGate4 arXiv v3 HTML](https://arxiv.org/html/2502.01681v3)
- Training objectives: DeepGate4 reuses DeepGate2's gate-level tasks (logic-1 probability and pairwise truth-table distance) and adds graph-level objectives, including graph edit distance. — [DeepGate4 arXiv PDF](https://arxiv.org/pdf/2502.01681)
- Code: github.com/zyzheng17/DeepGate4-ICLR-25. — [DeepGate4 arXiv abstract](https://arxiv.org/abs/2502.01681)
- **Correction to the user's premise:** the paper does not claim a "linear-complexity graph transformer". It claims a *sub-linear-memory* partitioned update strategy combined with a sparse (GAT-based) transformer. The measured result is roughly *constant* inference memory between 400K and 1.6M gates. — [DeepGate4 arXiv v3 HTML](https://arxiv.org/html/2502.01681v3)

**HOGA (DAC 2024) [PR]**
- "Less is More: Hop-Wise Graph Attention for Scalable and Generalizable Learning on Circuits" (Deng et al., Cornell). Hop-wise aggregated features are computed for each node *before training*. A gated self-attention module then turns them into node representations without touching the graph topology, so training parallelizes and distributes easily. It was evaluated on QoR prediction and functional reasoning. — [HOGA arXiv 2403.01317](https://arxiv.org/abs/2403.01317); [DAC 2024 PDF](https://www.csl.cornell.edu/~zhiruz/pdfs/hoga-dac2024.pdf); [cornell-zhang/HOGA](https://github.com/cornell-zhang/HOGA)
- A third-party summary (not verified against the paper) reports a 46.76% lower QoR estimation error than GCN. — [Lacuna summary of HOGA](https://lacuna.tiptreesystems.com/work/less-is-more-hop-wise-graph-attention-for-scalable-and-generalizable-learning-on/wrk_3b762ee36550fbbe25fe539a8d9f2049)
- HOGA-5 ran out of memory on the large circuits in DeepGate4's experiments. — [DeepGate4 ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/d75f561eaaf2cb754bc8d7e36d8af362-Paper-Conference.pdf)

**PolarGate (ICCAD 2024) [PR]**
- Introduces a bipolar embedding space for logical states in AIGs. A 9-layer PolarGate has 62.1% lower average prediction MAE than DeepGate2 and runs 85.6% faster than DeepGate/DeepGate2. No million-node experiment was found. — [PolarGate ICCAD 2024 paper](https://www.cse.cuhk.edu.hk/~byu/papers/C233-ICCAD2024-PolarGate.pdf); [PolarGate slides](http://www.cse.cuhk.edu.hk/~byu/papers/C233-ICCAD2024-PolarGate-slides.pdf)

**Gamora [venue not confirmed in this session]**
- A graph-learning approach to symbolic reasoning on large Boolean networks (adder/XOR/MAJ extraction for arithmetic reverse engineering). With GPU acceleration it scales to Boolean networks with **more than 33 million nodes**, with speedups of up to six orders of magnitude. Accuracy is close to 100% on CSA multipliers and above 97% on Booth-encoded multipliers, and stays above 92% after complex technology mapping. "Billion-node scalability" appears as a stated contribution, not a measured result. — [Gamora arXiv 2303.08256](https://arxiv.org/pdf/2303.08256)

**GNN-RE (TCAD 2022) [PR]**
- Represents a flattened gate-level netlist as a graph, finds subcircuit boundaries, and classifies subcircuits by function. The authors report **98.82%** average accuracy in mapping individual gates to modules. — [GNN-RE, NYU Scholars](https://nyuscholars.nyu.edu/en/publications/gnn-re-graph-neural-networks-for-reverse-engineering-of-gate-leve); [ResearchGate](https://www.researchgate.net/publication/354429669_GNN-RE_Graph_Neural_Networks_for_Reverse_Engineering_of_Gate-Level_Netlists)

**FGNN / FGNN2 (CUHK, Bei Yu group)**
- FGNN, "Functionality Matters in Netlist Representation Learning", appeared at DAC 2022 [PR]. — [ResearchGate](https://www.researchgate.net/publication/362900499_Functionality_matters_in_netlist_representation_learning); [GNN4IC paper list](https://github.com/DfX-NYUAD/GNN4IC)
- FGNN2 is described as "a novel circuit representation learning framework that utilizes a contrastive scheme to effectively extract generic functionality knowledge", with "a novel contrastive loss function to capture the relative functional distance between different circuits". Its venue (possibly TCAD) was not confirmed. — [search result summary of FGNN2 abstract, via GNN4IC list](https://github.com/DfX-NYUAD/GNN4IC)

**Sequential encoders: DeepSeq, DeepSeq2, MOSS, DeepSeq3**
- DeepSeq (2023) [PP in this session] — [arXiv 2302.13608](https://arxiv.org/abs/2302.13608)
- DeepSeq2 (ASP-DAC 2025) [PR]: maps sequential circuits into three separate embedding spaces (structure, function, and sequential behavior). It uses a DAG-GNN that avoids DeepSeq's recursive propagation, which "significantly reduces execution times and improves model scalability". It is evaluated on power estimation and reliability analysis. — [DeepSeq2 arXiv 2411.00530](https://arxiv.org/abs/2411.00530); [ACM DL](https://dl.acm.org/doi/pdf/10.1145/3658617.3697594)
- MOSS (DAC 2025) [PR]: multimodal GNN+LLM. An LLM fine-tuned on RTL produces embeddings that enrich the features of D-flip-flop nodes, and the LLM also supplies a global circuit embedding. It reports **95.2% accuracy in arrival-time prediction**. — [MOSS, ACM DL](https://dl.acm.org/doi/10.1109/DAC63849.2025.11132437)
- DeepSeq3 ("Beyond Flat Netlist", arXiv 2608.28188, 2026) [PP]: a hierarchical two-level representation. The lower level is fine-grained combinational subgraphs partitioned at flip-flops. The upper level is a **Super-Node Graph** that models register-transfer structure, and a dual GNN learns both levels. The authors say MOSS and the earlier DeepSeq family "cannot handle large-scale circuits". — [DeepSeq3 arXiv HTML 2608.28188](https://arxiv.org/html/2608.28188)

**Other 2025–2026 encoder successors**
- MGVGA (ICLR 2025) [PR]: Masked Gate Modeling masks gates *in latent space* so the circuit stays logically equivalent, then reconstructs the masked gates' attributes. Verilog-AIG Alignment "employs LLMs as teachers" to align AIGs with Verilog. It was evaluated on QoR prediction and logic-equivalence identification against DeepGate2. Code: github.com/wuhy68/MGVGA. — [MGVGA arXiv 2502.12732](https://arxiv.org/pdf/2502.12732); [ICLR 2025 poster page](https://iclr.cc/virtual/2025/poster/29473); [CUHK PDF](https://www.cse.cuhk.edu.hk/~byu/papers/C261-ICLR2025-MGVGA.pdf)
- MixGate ("Alignment Unlocks Complementarity", arXiv 2509.20968) [PP]: Shi et al., with Qiang Xu (the DeepGate group). Training is staged: an Equivalence Alignment Loss first builds a shared space across views, then multiview masked modeling runs. The paper argues that "functional alignment is a necessary precondition to unlock the power of multiview self-supervision". — [arXiv 2509.20968](https://arxiv.org/abs/2509.20968)
- Also identified but not characterized, because only titles were retrieved:
  - DeepCell: self-supervised multiview fusion (arXiv 2502.06816). — [awesomepapers listing](https://awesomepapers.io/graph-learning/papers/2502.06816)
  - TRACE: Learning to Compute on Circuit Graphs (arXiv 2509.21886). — [arXiv PDF](https://arxiv.org/pdf/2509.21886)
  - FuncGNN, which includes a large-AIG evaluation table with PolarGate. — [ACM DL](https://dl.acm.org/doi/pdf/10.1145/3779445)
  - Relational AIG-GCN (arXiv 2508.11991). — [arXiv HTML](https://arxiv.org/html/2508.11991v1)
  - Path-aware multi-view circuit learning for technology mapping (arXiv 2601.14286). — [arXiv PDF](https://arxiv.org/pdf/2601.14286)
- "Wrong Code, Right Structure" (arXiv 2603.09161, March 2026) [PP]: trains netlist GNNs (GraphSAINT, with the netlist as an undirected gate graph) on netlists synthesized from *imperfect* LLM-generated RTL, filtered by cell count and by structural similarity to a golden reference. A third-party summary reports **93.79% F1-macro** on operator-level sub-circuit identification and **68.35% F1** on CPU boundary classification (vs. 58.28% for rule-based augmentation). — [Moonlight review](https://www.themoonlight.io/en/review/wrong-code-right-structure-learning-netlist-representations-from-imperfect-llm-generated-rtl); [arXiv HTML](https://arxiv.org/html/2603.09161)

**HKUST (Zhiyao Xie group) encoders**
- CircuitEncoder (ASP-DAC 2025) [PR]: pre-trained with intra-stage contrastive learning and inter-stage (RTL-to-netlist) contrastive alignment. Fine-tuning uses lightweight MLP or tree models. The slides claim results comparable to supervised baselines with 50% of the data. — [HKUST group page](https://computer-arch.hkust.edu.hk/publication/asp-da2025-a-self-supervised-pre-trained-and-cros/); [slides](https://fangwenji.github.io/uploads/CircuitEncoder_slides.pdf); [GitHub](https://github.com/hkust-zhiyao/CircuitEncoder)
- CircuitFusion (ICLR 2025) [PR]: three unimodal encoders (RTL code, graph, and natural-language summary) feed a multimodal fusion encoder. An auxiliary netlist encoder is used only during pre-training and discarded at inference. Implementation-aware contrastive alignment of RTL and netlist sub-circuits uses **7,166 aligned RTL/netlist sub-circuit pairs**, with four self-supervised tasks. It produces register-level embeddings. — [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/7ffb43adf37b3eeaba559098bc084cc6-Paper-Conference.pdf); [arXiv 2505.02168](https://arxiv.org/html/2505.02168); [GitHub](https://github.com/hkust-zhiyao/CircuitFusion)
- NetTAG (DAC 2025) [PR]: covered under Q2 because it is the encoder used by GenEDA.

### Inferences
- For an ATPG-oriented LLM pipeline, DeepGate4 is the most credible *scalable gate-level* encoder. Its partitioned, level-by-level update fits AIG topological levels and keeps per-gate embeddings at very large scale. It does not, however, by itself produce a bounded token set. Some pooling or resampling stage on top of it is needed.
- HOGA's precomputed hop-wise features (no message passing at train time) make it the easiest to distribute and stream over arbitrarily large netlists. DeepGate4 found that deep HOGA variants (HOGA-5) still run out of memory on very large circuits, likely because the stored per-node hop features grow with depth.

### Gaps
- Exact absolute numbers for DeepGate4 (GB of memory, seconds per circuit, size of the training set) were not visible in the retrieved snippets. They are in Section 4.4 / the efficiency tables of the [arXiv v3 HTML](https://arxiv.org/html/2502.01681v3).
- Peer-reviewed venues for DeepGate2, DeepGate3, Gamora and FGNN2 were not confirmed in this session. They may have appeared at ICCAD/DAC/TCAD, but this is unverified.
- No scale numbers (largest circuit evaluated) were found for PolarGate, CircuitFusion, NetTAG, MGVGA or MOSS.
- The contributions of TRACE and DeepCell could not be retrieved beyond their titles.

## Q2. How encoders are connected to LLMs (GenEDA, NetTAG, BRIDGES, MOSS, MGVGA, others): connectors, token budgets, alignment losses, data, tasks, accuracy

### Takeaway
Two connection patterns dominate. The first is a **projector/connector** (an MLP in GenEDA; a "cross-modal projector" trained with BLIP-2-style contrastive, matching and grounded-generation losses in BRIDGES) that maps encoder embeddings into the LLM's input-embedding space as "circuit tokens". The second is **prediction-as-text** (GenEDA's frozen-LLM paradigm), where the encoder's functional predictions are serialized into the prompt. Both yield large gains on netlist-to-function and netlist-to-RTL tasks. However, no retrieved source stated an explicit fixed query-token count (such as 32/64/256 Q-Former queries) for any circuit model. BRIDGES pools into fixed-size embeddings, which is bounded by construction. GenEDA's exact token budget per netlist was not visible.

### Cited Findings

**NetTAG (DAC 2025) [PR]: the encoder side**
- Each gate of a synthesized gate-level netlist becomes a node in a **text-attributed graph**. The node text holds the gate's functional logic expression and physical characteristics. Per-gate formulas are extracted symbolically with PySMT, which "avoids expensive truth-table simulation". — [NetTAG arXiv 2504.09260](https://arxiv.org/abs/2504.09260v1); [DAC'25 PDF](https://zhiyaoxie.com/files/DAC25_NetTAG.pdf)
- Encoding happens in two stages. ExprLLM, an LLM-based text encoder, embeds each gate's expression text into a node embedding. TAGFormer, a graph transformer, then refines those embeddings using global netlist structure. The model is pre-trained with self-supervised objectives and aligned cross-stage with RTL-text and post-layout-graph embeddings. — [NetTAG arXiv](https://arxiv.org/abs/2504.09260v1); [GitHub hkust-zhiyao/NetTAG](https://github.com/hkust-zhiyao/NetTAG)
- NetTAG "consistently outperforms each task-specific method on four largely different functional and physical tasks and surpasses state-of-the-art AIG encoders". DeepGate3 and FGNN are among the baselines. — [NetTAG paper](https://arxiv.org/pdf/2504.09260)
- The authors argue that graph-only netlist encoders underweight gate semantics, and that netlist text alone is "flattened" and lacks context. — [NetTAG arXiv](https://arxiv.org/pdf/2504.09260)

**GenEDA (arXiv 2504.09485; v1 13 Apr 2025, v2 6 Aug 2025, retitled "Towards Generative Netlist Functional Reasoning via Cross-Modal Circuit Encoder-Decoder Alignment") [PP; venue not confirmed]**
- Authors: Wenji Fang, Jing Wang, Yao Lu, Shang Liu, Zhiyao Xie (HKUST). It claims to be the first generative foundation model that aligns pre-trained netlist encoders with LLM decoders. Code and benchmarks are at hkust-zhiyao/GenEDA. — [GenEDA arXiv](https://arxiv.org/abs/2504.09485)
- **Paradigm 1, embedding-based alignment (trainable open-source LLM):** an MLP "modality connector" maps netlist-graph embeddings into the LLM's latent space, so the LLM "process[es] circuit graphs as specialized tokens". The paper calls the connector a "circuit tokenizer". Training has two steps: (1) pre-train the connector for alignment; (2) fine-tune connector and decoder end to end. — [GenEDA arXiv HTML](https://arxiv.org/html/2504.09485); [Lacuna summary](https://lacuna.tiptreesystems.com/work/geneda-towards-generative-netlist-functional-reasoning-via-cross-modal-circuit/wrk_ffa5918efca8b3a2e4c8faec5fd191da)
- **Paradigm 2, prediction-based alignment (frozen commercial LLMs such as GPT-4o and DeepSeek-V3):** the encoder's gate/bit functional predictions are fed to the LLM *as text*. — [GenEDA arXiv HTML](https://arxiv.org/html/2504.09485)
- **Tasks:** reverse generation of high-level functionality from *bit-blasted* netlists. Two are functional-specification generation tasks and one is RTL reconstruction. — [GenEDA arXiv](https://arxiv.org/abs/2504.09485)
- **Results**, from search summaries of the paper (numbers vary across versions):
  - Specification tasks: GenEDA-7B scores **GPT Score 0.62**, against 0.21 for GPT-4o and 0.19 for DeepSeek-V3.
  - RTL reconstruction:
    - Without encoder predictions, GPT-4o "generates only bit-level operations". Standalone GPT-4o and DeepSeek-V3 stay **below 20% functional pass rate**, and GPT-4o alone reaches 13%.
    - DeepSeek-V3 with GenEDA reaches 96–97% syntax pass, 88% GPT Score, and **60% or 69% functional pass**. The figure differs between versions.
  - Ablation: replacing the high-quality encoder drops syntax from 96% to 91% and **functional accuracy from 69% to 40%**.
  — [GenEDA arXiv PDF](https://arxiv.org/pdf/2504.09485); [aimodels.fyi summary](https://www.aimodels.fyi/papers/arxiv/geneda-unleashing-generative-reasoning-netlist-via-multimodal); [Lacuna summary](https://lacuna.tiptreesystems.com/work/geneda-unleashing-generative-reasoning-on-netlist-via-multimodal-encoder-decoder/wrk_a6088fd0339926a428155109a06eef82)

**BRIDGES (CMU; arXiv 2504.05180; listed for ICLAD 2025) [PP / workshop-level; venue listing unverified]**
- Framework: a data-generation workflow, a **cross-modal projector** that "encodes graph representations into text-compatible prompts" so that unmodified LLMs can consume graph data, and an evaluation suite. The graph encoder is **NetlistGNN**, a message-passing GNN for directed heterogeneous netlist graphs, and it pools into fixed-size embeddings. — [BRIDGES alphaXiv/arXiv](https://www.alphaxiv.org/abs/2504.05180); [arXiv PDF](https://arxiv.org/pdf/2504.05180)
- **Alignment losses:** graph-text contrastive learning, graph-text matching, and graph-grounded text generation. — [Moonlight review of BRIDGES](https://www.themoonlight.io/en/review/bridges-bridging-graph-modality-and-large-language-models-within-eda-tasks)
- **Data:** more than 500,000 graph instances and **1.5 billion tokens**. — [BRIDGES](https://www.alphaxiv.org/abs/2504.05180)
- **Results:** 2–10x improvements over text-only baselines on design retrieval, type prediction and function-description perplexity, with **under 1% added model weights and under 30% added runtime**. It improves about 40.5% over Llama3-8B and about 20% over GPT-4o, both given the netlist as text. — [BRIDGES](https://www.alphaxiv.org/abs/2504.05180); [Semantic Scholar](https://www.semanticscholar.org/paper/BRIDGES:-Bridging-Graph-Modality-and-Large-Language-Li-Zou/35dc166c939cd7af0a9b9d37a4d6770019f9c5d3)

**MOSS (DAC 2025) [PR]**
- The fusion runs the other way: LLM embeddings (from an LLM fine-tuned on RTL) are injected into the GNN as flip-flop node features and as a global circuit embedding. The GNN is not projected into the LLM. One later paper describes MOSS as running a foundation LLM over post-mapping netlists. This conflicts with the abstract's emphasis on RTL and was not reconciled. — [MOSS ACM DL](https://dl.acm.org/doi/10.1109/DAC63849.2025.11132437); [DeepSeq3 arXiv HTML](https://arxiv.org/html/2608.28188)

**MGVGA (ICLR 2025) [PR]**
- An LLM serves as a *teacher* to align AIG embeddings with Verilog-code embeddings. This is cross-modal alignment that improves the graph encoder, not an LLM front-end. — [MGVGA arXiv](https://arxiv.org/pdf/2502.12732)

**VeriDistill ("The Graph's Apprentice", IJCAI 2025) [PR]**
- Knowledge distillation runs from a GNN trained on LUT graphs into an LLM that reads raw Verilog to predict area and delay. The GNN regularizes the LLM's embeddings toward its own. Removing either the LLM representations or the distillation step drops performance below prior baselines. It was benchmarked on OpenABCD. Huawei Noah's Ark Lab. — [arXiv HTML 2411.00843](https://arxiv.org/html/2411.00843v1); [IJCAI 2025 record](https://www.ijcai.org/proceedings/2025/bibtex/1033)

**LLSM (ASP-DAC 2025) [PR]**
- "The text length of some circuits is far larger than the maximum input an LLM can take." LLSM therefore splits circuits into modules, summarizes each with an LLM, and embeds the summaries. This is a hierarchical text-compression alternative to graph tokens. — [LLSM ASP-DAC 2025 PDF](https://www.aspdac.com/aspdac2025/archive/pdf/6E-1.pdf)

**CircuitFusion / CircuitEncoder [PR]**
- These are encoders only. They align RTL, graph, natural-language summary and netlist modalities contrastively, but no LLM-decoder coupling was reported in the retrieved material. — [CircuitFusion ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/7ffb43adf37b3eeaba559098bc084cc6-Paper-Conference.pdf); [CircuitEncoder slides](https://fangwenji.github.io/uploads/CircuitEncoder_slides.pdf)

**General graph-LLM bridges that circuit work borrows from (context for designing a bounded-token interface)**
- GraphToken (Google, arXiv 2402.05862): a learned graph-prompt encoder trained through a *frozen* LLM by optimizing answer perplexity. It gains up to 73 percentage points on GraphQA. Whether it emits a fixed token count was not confirmed from snippets. — [arXiv 2402.05862](https://arxiv.org/abs/2402.05862v1)
- GraphTranslator translates graph embeddings into *query tokens* (Q-Former-like). TEA-GLM projects a representation into a **fixed number** of learnable graph-token embeddings through a linear projector. — [Revisiting Graph-Tokenizing LLMs, arXiv 2605.03514](https://arxiv.org/pdf/2605.03514)
- LLaGA (ICML 2024) [PR]: node-level sequences of fixed length, ordered by structural position, with placeholders for empty slots. Only the projector is trained and the LLM stays frozen. There are two templates: a detailed neighborhood view and a hop-embedding summary. — [PMLR v235](https://proceedings.mlr.press/v235/chen24bh.html); [Moonlight review](https://www.themoonlight.io/review/llaga-large-language-and-graph-assistant)

### Inferences
- BRIDGES' three losses (graph-text contrastive, graph-text matching, graph-grounded generation) are the graph analogue of BLIP-2's Q-Former stage-1 objectives (ITC/ITM/ITG). This is the closest published circuit-specific evidence for a Q-Former-style resampler. Combined with fixed-size pooling, it gives a bounded token budget regardless of netlist size. This is an inference from the loss names; the snippets did not show its exact token count or whether learnable queries are used.
- GenEDA's prediction-based paradigm is the most deployable route for frozen commercial LLMs. The encoder's work is reduced to a compact textual summary. The analogue in an ATPG pipeline would be a GNN predicting per-node controllability/observability or hard-to-detect faults and passing the top-k as text.

### Gaps
- **Token budget:** no retrieved source states how many LLM tokens GenEDA or BRIDGES use per netlist, or whether GenEDA passes per-node, per-output-bit or one pooled embedding. This must be checked in the GenEDA paper's method section and code.
- The training corpus sizes for GenEDA alignment (number of netlists, gate-count range) and the LLM backbone behind "GenEDA-7B" (Qwen or Llama variant) were not visible.
- No circuit paper explicitly using BLIP-2 Q-Former learnable queries (such as 32/64 queries) over a netlist encoder was found.

## Q3. Evidence that encoder+LLM beats raw-text netlist prompting as circuit size grows; ablations on graph-token count vs. circuit size

### Takeaway
There is qualitative and aggregate evidence that raw-text netlist prompting degrades with context length, and that encoder-augmented LLMs do much better. BRIDGES found that GPT-4o and Claude 3.5 Sonnet fail on basic netlist properties "particularly as context length increases". GenEDA shows standalone frontier LLMs below 20% functional pass vs. up to about 69% with encoder predictions. No retrieved study plots accuracy against gate count for both approaches, and none ablates the number of graph tokens against circuit size. This is an open gap the user's project could fill.

### Cited Findings
- In BRIDGES' preliminary experiments, GPT-4o and Claude 3.5 Sonnet failed to predict basic circuit properties (bit-widths, arithmetic types) from netlist files, "particularly as context length increases". The authors give two causes: linearized graphs are hard for LLMs to interpret, and large graphs produce contexts too long for accurate reasoning. — [BRIDGES](https://www.alphaxiv.org/abs/2504.05180)
- With graph encoding, BRIDGES gains about 40.5% over Llama3-8B and about 20% over GPT-4o given netlist text, plus 2–10x on retrieval, type prediction and description perplexity. — [BRIDGES](https://www.alphaxiv.org/abs/2504.05180)
- In GenEDA, standalone GPT-4o and DeepSeek-V3 stay below 20% functional pass on netlist-to-RTL reconstruction and emit bit-level code. With GenEDA's encoder predictions they reach 60–69% functional pass (the figure varies by version). — [GenEDA arXiv PDF](https://arxiv.org/pdf/2504.09485)
- LLSM: circuit text can exceed the LLM's maximum input, which motivates module splitting and summarization. — [LLSM ASP-DAC 2025](https://www.aspdac.com/aspdac2025/archive/pdf/6E-1.pdf)
- Outside EDA, "Talk Like a Graph" found LLM performance suffers when graphs are presented as sequential text. — [arXiv 2310.04560](https://arxiv.org/pdf/2310.04560)
- The HKUST circuit-foundation-model survey recommends hybrids at the netlist stage: an LLM for textual gate attributes and a GNN for global structural dependencies. — [Survey of Circuit Foundation Model, arXiv 2504.03711](https://arxiv.org/html/2504.03711v1); [PDF](https://zhiyaoxie.com/files/Survey_CFM.pdf)

### Inferences
- In the current evidence, the encoder carries the *size scaling* and the LLM carries the *generation*. DeepGate4 shows encoders can run at 1.6M gates with flat memory, while the LLM-side papers (GenEDA, BRIDGES) were evaluated on module- and block-scale netlists. Encoder+LLM at 100K–1M gates has not been demonstrated end to end in any retrieved paper.

### Gaps
- No published curve of accuracy vs. gate count comparing raw-text prompting with graph tokens was found.
- No ablation of graph-token count (such as 8/32/128 tokens) against netlist size was found for any circuit model.
- The maximum netlist size in the GenEDA and BRIDGES evaluation sets was not visible in snippets.

## Q4. Known failure modes: over-squashing, loss of gate identity, inability to reference specific nets/gates, reconvergent fanout, sequential elements

### Takeaway
Documented failure modes:
- Over-squashing in deep message-passing GNNs, and quadratic attention blow-ups (both motivating DeepGate4).
- Out-of-memory in multi-hop and dense models (PNA, HOGA-5, DeepGate3) at hundreds of thousands to millions of gates.
- Simulation cost and exponential truth-table growth in functionality-supervised encoders.
- LLM collapse to bit-level, non-abstracted output without encoder guidance.
- Graph-LLMs that do not robustly "understand" their graph tokens (over-sensitivity or insensitivity to prompt changes).
- Sequential encoders (DeepSeq family, MOSS) that reportedly do not scale to large sequential circuits.

No retrieved work specifically measures loss of per-gate identity after pooling, or the ability of an LLM to point to a specific net or gate. For ATPG fault-site addressing this is the critical, unaddressed gap.

### Cited Findings
- **Over-squashing / quadratic complexity:** DeepGate4 names GNN over-squashing and transformer quadratic complexity as the barriers to scaling to large circuits. — [DeepGate4 arXiv](https://arxiv.org/abs/2502.01681). Over-squashing is defined as long-range information dissemination being hindered, which hurts tasks that depend on long-distance interactions. — [Over-squashing survey](https://ar5iv.labs.arxiv.org/html/2308.15568)
- **Memory/OOM:** PNA and HOGA-5 run out of memory on large circuits. Dense attention (DeepGate3) runs out of memory at millions of gates. Prior methods needed 30–40 GB to train on ITC99 (largest graph about 140K gates). — [DeepGate4 arXiv v3](https://arxiv.org/html/2502.01681v3); [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/d75f561eaaf2cb754bc8d7e36d8af362-Paper-Conference.pdf)
- **Simulation dependence:** DeepGate and DeepGate2 are slow because they rely on truth-table simulation. — [PolarGate](https://www.cse.cuhk.edu.hk/~byu/papers/C233-ICCAD2024-PolarGate.pdf). NetTAG replaces truth tables with symbolic per-gate formulas (PySMT) to avoid expensive simulation. — [NetTAG](https://arxiv.org/abs/2504.09260v1)
- **LLM abstraction failure:** without encoder predictions, GPT-4o emits only bit-level operations when reconstructing RTL from netlists. — [GenEDA](https://arxiv.org/pdf/2504.09485)
- **Graph-token understanding:** a systematic study finds graph-tokenizing LLMs either over-sensitive to rephrasing (GraphGPT, TEA-GLM) or over-insensitive to content-level changes (LLaGA, GOFA). Graph tokens still preserve task-relevant information. — [Revisiting Graph-Tokenizing LLMs, arXiv 2605.03514](https://arxiv.org/pdf/2605.03514); [Moonlight review](https://www.themoonlight.io/review/revisiting-graph-tokenizing-large-language-models-a-systematic-evaluation-of-graph-token-understanding)
- **Sequential elements:**
  - The DeepSeq3 authors state that MOSS and the earlier DeepSeq family "cannot handle large-scale circuits". They address this by partitioning at flip-flops plus a super-node register-transfer graph. — [DeepSeq3 arXiv 2608.28188](https://arxiv.org/html/2608.28188)
  - DeepSeq2 removed DeepSeq's recursive propagation for scalability. — [DeepSeq2](https://arxiv.org/abs/2411.00530)
  - MOSS specifically enriches D-flip-flop node features. — [MOSS](https://dl.acm.org/doi/10.1109/DAC63849.2025.11132437)
- **Cross-stage dependence:** a competing paper (TopoRTL) criticizes CircuitFusion for relying on cross-stage netlist data for topology information and for focusing on behavioral equivalence over topology. This is a competitor's claim. — [TopoRTL PDF](http://www.shichuan.org/doc/221.pdf)
- **Reconvergence-aware design:** DeepGate uses skip connections in its aggregation to capture logic computation. — [DeepTPI citing DeepGate](https://arxiv.org/pdf/2206.06975). Reconvergent fanout is the classic source of signal correlation that breaks independent-probability estimates; the snippets did not show DeepGate's explicit treatment of it.

### Inferences
- Any pooling or resampling to a fixed number of tokens destroys gate addressability by design. For ATPG (fault sites at specific nets), the LLM would need either a separate, addressable channel for node embeddings (as in LLaGA) or a tool-call/pointer interface back to the encoder's per-node embeddings and IDs.
- The partition-by-flip-flop plus super-node design in DeepSeq3 maps naturally onto scan-based ATPG. Under full scan, flip-flops become pseudo-PIs/POs and the combinational cones are the natural units for per-cone tokens.

### Gaps
- No retrieved paper quantifies identity loss (for example, gate-retrieval accuracy from pooled tokens) or reconvergent-fanout error in learned encoders. DeepGate v1's explicit reconvergence handling could not be verified from the snippets.
- No circuit-LLM paper was found that evaluates the LLM's ability to name or point to specific gates or nets.

## Q5. Node-level vs. graph-level tokens; methods that keep per-node or per-cone embeddings addressable by the LLM

### Takeaway
Circuit encoders natively produce **per-gate (node-level)** embeddings: the DeepGate family, PolarGate, HOGA, NetTAG/TAGFormer. Graph-level summaries are added by pooling: DeepGate3's Pooling Transformer, mean pooling in Gamora and HOGA, fixed-size pooling in BRIDGES. In current circuit-LLM bridges, what reaches the LLM is pooled embeddings mapped by a projector (BRIDGES, likely GenEDA's embedding path) or textual predictions (GenEDA's prediction path). No circuit-specific work implements pointer-style, ID-addressable node tokens. The closest general-graph template is LLaGA's per-node, structure-ordered sequences. The closest circuit-specific hierarchical template is DeepSeq3's two-level (subgraph plus super-node) representation.

### Cited Findings
- DeepGate-family models "primarily utilize GNNs to encode circuit netlists into gate-level embeddings". — [DeepSAT / DeepGate context](https://arxiv.org/pdf/2205.13745)
- DeepGate3 has an explicit two-level output: Refine Transformer gate embeddings, and Pooling Transformer graph embeddings (vs. average pooling in Gamora and HOGA). — [DeepGate3 review](https://www.themoonlight.io/en/review/deepgate3-towards-scalable-circuit-representation-learning)
- DeepGate4 trains on both gate-level tasks (logic-1 probability, truth-table distance) and graph-level tasks (including graph edit distance), so it supports both granularities. — [DeepGate4 arXiv PDF](https://arxiv.org/pdf/2502.01681)
- NetTAG: per-gate text-attributed node embeddings (ExprLLM), refined by TAGFormer, plus netlist-level embeddings aligned with RTL and layout. — [NetTAG](https://arxiv.org/abs/2504.09260v1)
- BRIDGES pools NetlistGNN outputs into fixed-size embeddings before the projector. — [BRIDGES](https://www.alphaxiv.org/abs/2504.05180)
- An LLM4graph survey taxonomy separates graph-to-token granularities: Node2token, Pairwise Nodes2token, Group-aware Nodes2token, and Holistic Nodes2token. — [LLM4graph survey summary](https://chatpaper.com/paper/95858)
- LLaGA: each node's neighborhood becomes a fixed-length, structure-ordered sequence, and "each sequence position uniquely corresponds to a relative structural position". This keeps node-level tokens addressable while each node's context stays bounded. — [LLaGA review](https://www.themoonlight.io/review/llaga-large-language-and-graph-assistant); [PMLR](https://proceedings.mlr.press/v235/chen24bh.html)
- DeepSeq3: lower-level combinational subgraphs partitioned at flip-flops, and an upper-level Super-Node Graph. — [DeepSeq3](https://arxiv.org/html/2608.28188)
- CircuitFusion produces register-level embeddings (one per sequential cone or sub-circuit). — [CircuitEncoder repo description of CircuitFusion](https://github.com/hkust-zhiyao/CircuitEncoder)

### Inferences
- A plausible bounded-yet-addressable design for an ATPG LLM:
  1. A DeepGate4-class encoder produces per-gate embeddings.
  2. Gates are grouped per output cone, or per scan-cell cone as in DeepSeq3 and CircuitFusion's register-level units.
  3. A fixed K tokens per cone come from a Q-Former-style resampler trained with BRIDGES-style contrastive, matching and generation losses.
  4. A top-level budget of M cone tokens is selected by relevance (for example, cones containing the target fault).
  5. Exact fault-site identity is preserved by putting gate IDs in text alongside the tokens, or by letting the LLM call tools that fetch individual node embeddings.

  This is a synthesis of the cited components; no paper implements it end to end.

### Gaps
- No circuit work with pointer-network or ID-indexed references from LLM output to encoder nodes was found.
- No study compares node-level and pooled tokens on a gate-localization task.

## Q6. Relevance to ATPG / testability / SAT guidance (flagged for the user's LLM-for-ATPG project)

### Takeaway
DeepGate-family embeddings have been used directly for **test point insertion** (DeepTPI, ITC 2022), and the same group's **DeepSAT** solves SAT on AIGs via a conditional generative DAGNN supervised by logic-simulation signal probabilities. GNNs are also being used for **testability analysis with X-sources** (ASP-DAC 2025). No retrieved work (2021–2026) couples an LLM with a circuit encoder to generate or guide ATPG patterns at gate level. The LLM-for-ATPG space with graph tokens appears open.

### Cited Findings
- **DeepTPI (ITC 2022) [PR]:**
  - First RL-based test point insertion. Pre-trained DeepGate node embeddings form the RL agent's state, and the objective is to maximize LBIST test coverage. The authors frame TPI as node-level classification that benefits from DeepGate pre-training. — [DeepTPI arXiv 2206.06975](https://arxiv.org/pdf/2206.06975); [cure-lab/DeepTPI](https://github.com/cure-lab/DeepTPI)
  - The DeepGate repo ships a `./tpi` downstream directory. — [cure-lab/DeepGate](https://github.com/cure-lab/DeepGate)
- **DeepSAT [PP in this session]:**
  - Converts SAT instances to logic-synthesis-optimized AIGs and models solutions as a product of conditional Bernoulli distributions using a DAGNN with two polarity prototypes, supervised by simulated signal probabilities.
  - It reports significant gains over learning-based SAT baselines, especially on larger or distribution-shifted instances.
  - Author lists differ across mirrors.
  — [DeepSAT arXiv 2205.13745](https://arxiv.org/pdf/2205.13745)
- **GNN testability with unknowns (ASP-DAC 2025) [PR]:** a GAT with edge features learns the impact of X-source inputs on testability analysis. It predicts testability problems and does not generate patterns. — [ACM DL 10.1145/3658617.3697753](https://dl.acm.org/doi/10.1145/3658617.3697753)
- An arXiv paper titled "Enhancing Test Efficiency through Automated ATPG-Aware ..." (2505.19418) appeared in results. Its content was not retrieved. — [arXiv 2505.19418](https://arxiv.org/pdf/2505.19418)
- No 2025–2026 work showing an LLM generating or guiding ATPG patterns on gate-level netlists was found. — [search outcome; see the GNN testability listing above](https://dl.acm.org/doi/10.1145/3658617.3697753)

### Inferences
- DeepGate-style **logic-1 probability** heads are a learned analogue of COP/SCOAP controllability. Truth-table-distance and equivalence objectives (DeepGate2, MixGate) relate to fault equivalence and dominance reasoning. DeepGate4 at 1.6M gates is the strongest candidate backbone for producing per-node testability features at industrial scale.
- Since ATPG for a single stuck-at fault reduces to SAT on a miter, DeepSAT's AIG-based conditional-Bernoulli formulation is the closest published neural method to direct test-vector generation.

### Gaps
- No quantitative DeepTPI coverage-gain numbers or DeepSAT solve-rate numbers were retrieved.
- No paper was found that uses LLMs with circuit graph tokens for ATPG, fault-site reasoning, or controllability/observability prediction. This should be treated as an open research gap, not as evidence of infeasibility.
