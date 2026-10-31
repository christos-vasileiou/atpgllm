# LLM-for-Netlist Systems: Encoding Methods and Scaling Past the Context Window

> **Method and limits of this dossier (read first).** I did this research on 2026-10-31. In this session, direct page fetches of arxiv.org and research.nvidia.com failed: the hosts are outside the session's allowed network domains. GitHub repos other than christos-vasileiou/atpgllm could not be read through the GitHub tool either. Every fact below therefore comes from web-search result extracts. Those extracts quote the primary papers (arXiv HTML/PDF, IEEE/ACM listings, GitHub READMEs) or, where marked, secondary summaries. I did not read any paper end to end. Numbers that come only from secondary summaries are labelled as such. To verify a figure, re-fetch the cited URL once arxiv.org is allowed.
>
> **Labels used:** [DIGITAL-GL] = digital, gate-level / post-synthesis netlist. [DIGITAL-AIG] = And-Inverter Graph. [ANALOG/SPICE] = transistor-level SPICE netlists or schematics. [PEER] = peer-reviewed venue confirmed. [ARXIV] = preprint only, as far as I could confirm.

---

## Q1. The five named systems: what each is, who published it, and how it gives the netlist to the LLM

### Takeaway
GenEDA is the only one of the five named systems that is a true LLM-reasons-over-gate-level-netlist system with a scalable encoding. It feeds an LLM graph-encoder embeddings (from NetTAG) through an MLP connector as soft tokens, or it adds encoder predictions to the netlist text for a frozen LLM. LLM4Netlist *generates* small digital netlists (reported as BLIF) from natural language. Auto-SPICE/Masala-CHAI, Netlistify and LogicCraft are all analog or transistor-level. Netlistify does not use an LLM at all; it is a CNN/DETR vision pipeline. None of the four non-GenEDA systems shows a way to ingest large gate-level netlists.

### Cited Findings

**GenEDA [DIGITAL-GL] [PEER: ICCAD 2025]**
- Identity: arXiv 2504.09485 (v1 Apr 2025, v2 Aug 2025). The v1 title is "GenEDA: Unleashing Generative Reasoning on Netlist via Multimodal Encoder-Decoder Aligned Foundation Model". The later title is "GenEDA: Towards Generative Netlist Functional Reasoning via Cross-Modal Circuit Encoder-Decoder Alignment". Authors are Wenji Fang, Jing Wang, Yao Lu, Shang Liu and Zhiyao Xie (HKUST), and v2 states it was accepted at ICCAD'25. — [arXiv abs](https://arxiv.org/abs/2504.09485); [arXiv v2](https://arxiv.org/abs/2504.09485v2)
- The ICCAD 2025 DOI is 10.1109/ICCAD66269.2025.11240762. — [ResearchGate listing](https://www.researchgate.net/publication/397819047_GenEDA_Towards_Generative_Netlist_Functional_Reasoning_via_Cross-Modal_Circuit_Encoder-Decoder_Alignment)
- Core claim: it is the first framework to align pre-trained circuit (netlist graph) encoders with LLM decoders in a shared latent space. Earlier circuit models were either standalone encoders for prediction or decoders for generation. — [arXiv abs](https://arxiv.org/abs/2504.09485)
- **Encoding route 1 (embedding-based alignment, for trainable LLMs).** NetTAG is the graph encoder. An MLP "modality connector" maps the graph embeddings into the LLM latent space, so they act as soft tokens. The decoders are DeepSeek-Coder 1B and 7B. Training has two stages: (i) pre-train the connector to align embeddings, then (ii) fine-tune connector and decoder end to end. — [arXiv HTML](https://arxiv.org/html/2504.09485); [aimodels.fyi summary](https://www.aimodels.fyi/papers/arxiv/geneda-unleashing-generative-reasoning-netlist-via-multimodal)
- **Encoding route 2 (prediction-based alignment, for frozen commercial LLMs).** The encoder's gate-function predictions are written onto the netlist text as annotations, and the frozen LLM (GPT-4o or DeepSeek-V3 through their APIs) reads that text. Without these annotations, GPT-4o mostly emits bit-level operations. The gate-function encoder is reported at about 97% gate-classification accuracy. — [arXiv HTML](https://arxiv.org/html/2504.09485); [arXiv v1 HTML](https://arxiv.org/html/2504.09485v1)
- Tasks: generate high-level functionality (natural-language specifications and exact RTL) from low-level, bit-blasted netlists. Task 3 turns the GNN-RE gate-level arithmetic-function dataset into a generative benchmark with golden RTL and testbenches. — [arXiv v1 HTML](https://arxiv.org/html/2504.09485v1); [arXiv PDF](https://arxiv.org/pdf/2504.09485)
- Data (Table II as extracted). ITC99 has ~4k circuits, avg **15k tokens**, avg **1,025 gates**. OpenCores has ~55k circuits, avg **9k tokens**, avg **173 gates**. A later version adds Chipyard: 20k circuits, avg 24k tokens, gate count not given. VexRiscv is also listed as a source. Netlists are synthesized with Synopsys Design Compiler and the NanGate 45nm library (stated in the companion NetTAG paper, which uses the same sources). — [arXiv HTML](https://arxiv.org/html/2504.09485); [NetTAG arXiv HTML](https://arxiv.org/html/2504.09260)
- Results. Commercial LLMs (GPT-4o, DeepSeek-V3) alone reach **<20% functional success** on the arithmetic RTL-reconstruction benchmark. GenEDA + DeepSeek-V3 is best. One version reports **97% syntax / 88% GPT-score / 60% functional pass**; another reports **96% syntax / 88% GPT-score / 69% functional pass**. These two versions conflict, so state which one you cite. On specification reasoning (Tasks 1 and 2), GenEDA-7B reaches a GPT Score of 0.62. — [arXiv PDF](https://arxiv.org/pdf/2504.09485); [arXiv HTML](https://arxiv.org/html/2504.09485)
- Code and benchmarks: github.com/hkust-zhiyao/GenEDA (from a search-result description; I could not access the repo contents). — [search result citing arXiv v2](https://arxiv.org/html/2504.09485v2)

**LLM4Netlist [DIGITAL, generation direction] [PEER: IEEE JETCAS 2025]**
- Identity: "LLM4Netlist: LLM-Enabled Step-Based Netlist Generation From Natural Language Description". IEEE JETCAS 2025, DOI 10.1109/JETCAS.2025.3568548. Authors: Kailiang Ye, Qingyu Yang, Zheng Lu, Heng Yu, Tianxiang Cui, Ruibin Bai, Linlin Shen (Univ. of Nottingham Ningbo China et al.). I found no arXiv ID. — [UNNC research portal](https://research.nottingham.edu.cn/en/publications/llm4netlist-llm-enabled-step-based-netlist-generation-from-natura/); [Bohrium listing](https://www.bohrium.com/paper-details/llm4netlist-llm-enabled-step-based-netlist-generation-from-natural-language-description/1144908897933328393-3966)
- Representation: it *generates* netlist code as plain text directly from natural-language functional descriptions. It does not ingest an existing netlist. It has four parts: a step-based prompt-construction module, a fine-tuned LLM, a code-confidence estimator and a feedback loop. The authors name scarce netlist data, the need for tailored fine-tuning, and effective generation methods as the challenges. Training/test data and processing code are linked on GitHub. — [UNNC portal](https://research.nottingham.edu.cn/en/publications/llm4netlist-llm-enabled-step-based-netlist-generation-from-natura/)
- Output format: a secondary source says it produces **BLIF** netlists. — [search summary (secondary)](https://research.nottingham.edu.cn/en/publications/llm4netlist-llm-enabled-step-based-netlist-generation-from-natura/)
- Scaling, according to a third party (ArtNet, arXiv 2510.13582): LLM4Netlist reaches SOTA functional correctness on its benchmarks but "still faces challenges in scaling to highly complex or large-scale circuits", and inference time grows significantly with circuit size because of iterative decoding. — [ArtNet arXiv HTML](https://arxiv.org/html/2510.13582v1)
- Scope, according to a third party (CircuitLM): it stays focused on synthesizable digital logic. — [CircuitLM arXiv HTML](https://arxiv.org/html/2601.04505v1)

**Auto-SPICE, renamed Masala-CHAI in later versions [ANALOG/SPICE] [ARXIV; peer-review status not confirmed]**
- Identity: arXiv 2411.14299, by Bhandari, Bhat, He, Garg, Rahmani and Karri (NYU). Version 1 is titled Auto-SPICE. Version 5 renames it "Masala-CHAI: A Large-Scale SPICE Netlist Dataset for Analog Circuits by Harnessing AI". — [arXiv abs v1](https://arxiv.org/abs/2411.14299v1); [arXiv HTML v5](https://arxiv.org/html/2411.14299v5)
- Representation: an LLM-based pipeline (GPT-4o) that produces SPICE netlists from schematic *images*. It has three steps: label the circuits, tune the prompt, verify the netlist. GPT-4o alone does not extract accurate SPICE netlists. It handles 2-terminal devices well but has trouble getting the other connectivity right. v5 adds a fully automated, annotation-free flow that combines LLMs with custom-trained deep networks. — [arXiv HTML v1](https://arxiv.org/html/2411.14299v1); [arXiv HTML v5](https://arxiv.org/html/2411.14299v5)
- Scale: v1 covers roughly 2,100 schematics, mostly from analog textbooks. The final version has **7,500 schematics from 10 textbooks**. Models fine-tuned on Masala-CHAI and used inside AnalogCoder gain **46% in Pass@1**, and reach >40% Pass@1 on telescopic-cascode amplifiers and bandgap references. The paper says it is open-sourced. — [arXiv PDF](https://arxiv.org/pdf/2411.14299); [themoonlight review (secondary)](https://www.themoonlight.io/review/masala-chai-a-large-scale-spice-netlist-dataset-for-analog-circuits-by-harnessing-ai)

**Netlistify [ANALOG/SPICE schematics; NOT an LLM system] [PEER: MLCAD 2025]**
- Identity: "Netlistify: Transforming Circuit Schematics into Netlists with Deep Learning". MLCAD 2025, DOI 10.1109/MLCAD65511.2025.11189145. Authors: Chun-Yen Huang, Hsuan-I Chen, Hao-Wen Ho, Pei-Hsin Kang, Mark Po-Hung Lin, Wen-Hao Liu, Haoxing Ren (NYCU + NVIDIA). I found no arXiv ID. It won the Best Artifact Award at MLCAD'25. — [ACM DL](https://dl.acm.org/doi/10.1109/MLCAD65511.2025.11189145); [NVIDIA EDA research page](https://research.nvidia.com/labs/electronic-design-automation/); [NetlistBench reference list](https://arxiv.org/html/2608.12197)
- Representation: a vision pipeline. YOLOv8 detects components, ResNet classifies orientation, and a modified DETR transformer detects wires and connectivity. The output is a netlist. It ships a dataset of 100,000 schematic images plus pre-trained models. **No LLM is involved.** — [GitHub NYCU-AI-EDA/Netlistify](https://github.com/NYCU-AI-EDA/Netlistify)
- Results: against AMSNet, connectivity accuracy is +2.4%, recall +20.7% and F1 +12.4%. A secondary source gives ~93.33% accuracy on printed diagrams, ~85.33% on hand-drawn ones, and 100% on synthetic analog circuits. — [ResearchGate](https://www.researchgate.net/publication/396395178_Netlistify_Transforming_Circuit_Schematics_into_Netlists_with_Deep_Learning); [Scribd copy (secondary)](https://www.scribd.com/document/980616123/netlistify-mlcad25)

**LogicCraft [TRANSISTOR-LEVEL standard-cell; not gate-level ingestion] [PEER: AICAS 2025]**
- Identity: "LogicCraft: LLM-Assisted Optimization of Netlist to Layout for Complex Custom Standard Cell Designs". AICAS 2025 (Bordeaux, 28–30 Apr 2025), pp. 1–5, DOI 10.1109/AICAS64808.2025.11173139. Authors: Minwei Lin, Duy-Hieu Bui, Chao Wang and nine co-authors (East China Normal University). I found no arXiv ID. — [ECNU portal](https://pure.ecnu.edu.cn/en/publications/logiccraft-llm-assisted-optimization-of-netlist-to-layout-for-com/); [dblp AICAS 2025](https://dblp.org/db/conf/aicas/aicas2025.html)
- Representation: an LLM translates and optimizes Boolean truth tables into custom pull-up networks for complex standard cells. Later work describes this as producing SPICE netlists of standard cells. The LLM's input is a small truth table, not a gate-level netlist. — [ECNU portal (abstract)](https://pure.ecnu.edu.cn/en/publications/logiccraft-llm-assisted-optimization-of-netlist-to-layout-for-com/); [TOPCELL arXiv 2604.14237](https://arxiv.org/html/2604.14237v1)

### Inferences
- If the end user's notion of "LLM for netlists" means *reasoning over existing post-synthesis gate-level netlists*, GenEDA is the only one of the five that qualifies. The other four are (a) generation from NL (LLM4Netlist), (b) image-to-SPICE analog extraction (Auto-SPICE/Masala-CHAI, Netlistify), or (c) transistor-level cell synthesis (LogicCraft).
- GenEDA's two routes are a usable template for the user: a soft-token connector when the LLM can be fine-tuned, and a "graph-encoder predictions written into the text" route when only a frozen API LLM is available.

### Gaps
- Exact soft-token count per circuit in GenEDA (one token per gate, one per cone, or a pooled vector?), the connector's dimensions, and the decoder context length were not in any extract I could reach.
- LLM4Netlist's benchmark names, circuit sizes, base LLM, and pass rates were not retrievable. The full IEEE paper (DOI above) is needed.
- Whether Masala-CHAI was later accepted at a peer-reviewed venue could not be confirmed.
- LogicCraft's LLM, prompting scheme and quantitative results were not retrievable (only the abstract was reachable).

---

## Q2. Other closely related systems (digital gate-level first, then AIG, RTL-adjacent, security, and analog)

### Takeaway
The best-supported scalable encodings come from the HKUST Xie group. NetTAG (text-attributed graph: one LLM-encoded expression per gate plus a graph transformer, with register-cone chunking), CircuitFusion (multimodal, per register-cone) and GenEDA (encoder → LLM) form one lineage. DeepGate4 adds GNN-side evidence of scaling to 400K–1.6M gates. Raw-text LLM approaches on gate-level netlists, such as Gate-Breaker (netlist-to-RTL) and NtNDet (Trojans), report results only on small benchmark circuits. I found no LLM system that does ATPG on gate-level netlists.

### Cited Findings

**NetTAG [DIGITAL-GL] [PEER: DAC 2025]**
- arXiv 2504.09260 (12 Apr 2025), "NetTAG: A Multimodal RTL-and-Layout-Aligned Netlist Foundation Model via Text-Attributed Graph". Authors: Wenji Fang et al., Zhiyao Xie (HKUST). DAC 2025, San Francisco, 22–25 Jun 2025. Earlier working titles include "NetEncoder" and "Towards Netlist Foundation Model". — [arXiv abs](https://arxiv.org/abs/2504.09260v1); [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11133349/); [HKUST portal](https://researchportal.hkust.edu.hk/en/publications/nettag-a-multimodal-rtl-and-layout-aligned-netlist-foundation-mod/); [DAC PDF](https://zhiyaoxie.com/files/DAC25_NetTAG.pdf)
- **Encoding.** The netlist is a text-attributed graph. Each gate's text attributes are its symbolic logic expression and its physical characteristics. The expressions are generated algorithmically with PySMT rather than by truth-table simulation. **ExprLLM** (an LLM-based text encoder) embeds each gate's text. **TAGFormer** (a graph transformer) then refines the gate embeddings using global structure. Training is self-supervised and aligned with RTL and layout embeddings. — [arXiv HTML](https://arxiv.org/html/2504.09260v1); [emergentmind topic page](https://www.emergentmind.com/topics/nettag)
- **Size handling.** Large sequential circuits are split into **register cones**: each register is backtraced through all of its driving logic up to other registers, giving a sub-circuit that captures the register's full state transition. The data is 313k circuit netlists and 100k sub-circuit cones after chunking, as extracted. — [arXiv PDF](https://arxiv.org/pdf/2504.09260); [DAC PDF](https://zhiyaoxie.com/files/DAC25_NetTAG.pdf)
- **Context limit.** In the paper, ExprLLM is built on LLM2Vec with Llama-3.1-8B and takes **8k max input tokens**. The repo config sets max_seq_length 8192. The README names Sheared-LLaMA-1.3B as the default, which conflicts with the paper's 8B. — [arXiv PDF](https://arxiv.org/pdf/2504.09260); [GitHub README](https://github.com/hkust-zhiyao/NetTAG/blob/main/README.md)
- **Results and scaling.** It beats task-specific methods on four functional and physical tasks and outperforms SOTA AIG encoders. Scaling ExprLLM from BERT-110M to Llama-3.1 1.3B and 8B improves all four tasks, and so does growing the data from 25% to 100%. Code and pre-trained model are released. — [arXiv PDF](https://arxiv.org/pdf/2504.09260); [GitHub](https://github.com/hkust-zhiyao/NetTAG)

**CircuitFusion [DIGITAL: RTL + graph + summary, with a netlist encoder used in pre-training] [PEER: ICLR 2025]**
- arXiv 2505.02168. Authors: Fang, Liu, Wang, Xie. Described as the first multimodal, implementation-aware circuit encoder. It has three unimodal encoders (graph, summary, code) and a cross-attention fusion encoder that fuses around the summary. An auxiliary netlist encoder is used only in pre-training. — [ICLR proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/7ffb43adf37b3eeaba559098bc084cc6-Paper-Conference.pdf); [arXiv HTML](https://arxiv.org/html/2505.02168)
- Size handling: the unit is the **register-cone sub-circuit**. The graph encoder outputs a sequence of node embeddings for each sub-circuit. The code encoder is a frozen **NV-Embed-V1** with **32K max input tokens**. Code is public. — [arXiv HTML](https://arxiv.org/html/2505.02168); [GitHub](https://github.com/hkust-zhiyao/CircuitFusion)
- The precursor is CircuitEncoder (ASP-DAC 2025): a self-supervised, pre-trained, cross-stage-aligned circuit encoder. — [GitHub](https://github.com/hkust-zhiyao/CircuitEncoder); [PDF](https://zhiyaoxie.com/files/ASPDAC25_CircuitEncoder.pdf)

**DeepCircuitX (dataset) [DIGITAL; RTL→AIG→post-mapping netlist→layout] [ARXIV]**
- arXiv 2502.18297. A repository-level dataset from 4,000+ RTL projects with AIGs, post-mapping netlists, floorplan and layout data, plus chain-of-thought annotations describing functionality and structure at several levels. DeepCell (arXiv 2502.06816) uses ForgeEDA (arXiv 2505.02016) and DeepCircuitX. — [DeepCircuitX PDF](https://arxiv.org/pdf/2502.18297); [DeepCell PDF](https://arxiv.org/pdf/2502.06816); [ForgeEDA PDF](https://arxiv.org/pdf/2505.02016)

**DeepGate4 [DIGITAL-AIG; GNN/graph transformer, not an LLM; reference point for encoder scalability] [ICLR 2025 per the arXiv header; not separately confirmed]**
- arXiv 2502.01681. Trained on circuits averaging about **15K gates**, it generalizes to circuits of about **400K gates**. From 400K to **1.6M gates**, memory stays constant while GNN baselines grow linearly. The 800K and 1.6M test cases are the EPFL Image_Processing circuit duplicated 2× and 4×. The method partitions the AIG into overlapping fixed-depth cones and uses a GAT-based sparse transformer with a fused CUDA kernel. Evaluation uses EPFL, ITC99 and OpenABC-D. The authors' "millions/billions" extrapolation is a projection. — [arXiv HTML v3](https://arxiv.org/html/2502.01681v3); [alphaXiv](https://www.alphaxiv.org/abs/2502.01681)

**MGVGA [DIGITAL-AIG; LLM as teacher] [PEER: ICLR 2025]**
- arXiv 2502.12732, by Haoyuan Wu, Haisheng Zheng, Yuan Pu, Bei Yu (CUHK). Masked Gate Modeling masks gates in latent space, which preserves logical equivalence. Verilog-AIG Alignment uses an LLM as teacher: masked gates are reconstructed under the constraint of equivalent Verilog, so the GNN learns function from the LLM. It is reported to beat SOTA on logic-synthesis tasks such as QoR prediction. Code: github.com/wuhy68/MGVGA. — [OpenReview](https://openreview.net/forum?id=US9k5TXVLZ); [arXiv PDF](https://arxiv.org/pdf/2502.12732); [ICLR poster](https://iclr.cc/virtual/2025/poster/29473)

**Circuit Transformer [DIGITAL-AIG generation; a transformer, not a pre-trained LLM] [PEER: ICLR 2025]**
- arXiv 2403.13838 (Xihan Li et al.). Encoding: the circuit is serialized as a memory-less **depth-first traversal trajectory**, and the model predicts the next gate. A masking layer at the end of the decoder blocks any token that would break logical equivalence with the current "cutoff" circuit. The input AIG is encoded and the output AIG is built node by node. The model has 88M parameters. With MCTS it improves significantly over ABC resyn2 while keeping strict equivalence. The follow-up "ctrw" rewriting operator is arXiv 2406.04699. — [arXiv abs](https://arxiv.org/abs/2403.13838); [ICLR 2025 PDF](https://arxiv.org/pdf/2403.13838); [ctrw PDF](https://arxiv.org/pdf/2406.04699)

**"Wrong Code, Right Structure" [DIGITAL-GL representation learning; LLMs used as a data source] [ARXIV, Mar 2026]**
- arXiv 2603.09161, by Siyang Cai, Cangyuan Li, Yinhe Han, Ying Wang. Netlists synthesized from *functionally buggy* LLM-generated RTL are still structurally close to the correct design, so they serve as training data for netlist-representation models. A netlist-level filter scores each candidate against a golden netlist (cosine similarity, per a machine review), and RTL-level "architecture voting" adds diversity. It is reported to generalize better to unseen architectures of the same function. — [arXiv HTML](https://arxiv.org/html/2603.09161); [Pith review](https://pith.science/paper/2603.09161)

**Gate-Breaker [DIGITAL-GL; raw-text netlist → LLM → RTL] [PEER: IPCCC 2025]**
- An LLM-powered tool that reverse-engineers gate-level netlists into RTL. It is built on VerilogEval and SimEval: the LLM receives a synthesized gate-level netlist and must produce RTL that passes the original testbench. Scoring uses functional pass rate plus AST and CFG similarity. — [GitHub NMSU-PEARL/Gate-Breaker](https://github.com/NMSU-PEARL/Gate-Breaker); [IEEE Xplore](https://ieeexplore.ieee.org/document/11304682/)
- Results: 4 LLMs on **156 circuits**. **OpenAI o3-mini** is best at **75.1% overall success**, dropping to **53.4% on the 4th (most complex) quartile**. Where it succeeds, AST similarity is 0.924 and CFG similarity 0.871. The quartiles are defined by code complexity; I could not confirm they correspond to netlist size. — [IEEE Xplore abstract](https://ieeexplore.ieee.org/document/11304682/)
- A third-party pilot replication found no significant differences between its prompt variants. — [drpaj12/loki PR #49](https://github.com/drpaj12/loki/pull/49)

**NtNDet [DIGITAL-GL; netlist → natural-language text → BERT/GPT-2/T5] [PEER: Expert Systems with Applications 2025]**
- By Kuang, Quan, Xie, Cai, Chen, Li. **Netlist-to-Natural-Language (NtN)** converts gate-level netlists into a natural-language-like format so pre-trained LMs (BERT, GPT-2, T5) can classify them, with self-attention capturing dependencies across the netlist. The authors call it the first use of large pre-trained models for gate-level hardware-Trojan detection. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S095741742500288X)
- Results on Trust-Hub, TRIT-TC and TRIT-TS: +5.27% precision, +3.06% TPR, +0.01% TNR and +3.17% F1 over the prior SOTA (margins only; absolute values not retrieved). — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S095741742500288X)

**Other hardware-security works related to Trojans and gate-level netlists**
- GHOST (arXiv 2412.02816) uses LLMs (GPT-4, Gemini-1.5-pro, Llama-3-70B) to *insert* Trojans, with GPT-4 best. It works at RTL, not on gate-level netlists. — [arXiv PDF](https://arxiv.org/pdf/2412.02816)
- TrojanLoC (arXiv 2512.00591) uses RTL-fine-tuned LLM embeddings with lightweight classifiers: 0.99 F1 at module level and 0.92 macro-F1 at line level. RTL only. — [arXiv PDF](https://arxiv.org/pdf/2512.00591)
- The ICCAD 2025 CAD Contest Problem A asks for gate-level Trojan detection and localization on flattened netlists (10 RTL Trojan types). LoRD (arXiv 2609.17922, Sep 2026) is a heuristic built on subgraph signatures, with **no ML or LLM**. It averages ~2.957/3 and beats a transformer-based ML baseline and the top-5 contest teams. — [arXiv PDF](https://arxiv.org/pdf/2609.17922)
- ImprovedBiGCN (a contest entry) is a bidirectional GNN that labels each gate: 93.3% circuit-level accuracy, honorable mention. — [GitHub](https://github.com/HahhaNa/Hardware-Trojan-Detection-on-Gate-Level-Netlists)
- Structural manipulability (arXiv 2607.16245, Karn and Sinanoglu, Jun 2026) uses a GNN to regress a topology-derived per-node score (path participation, k-core, symmetry, centrality). A TrustHub case study is included. No LLM. — [arXiv abs](https://arxiv.org/abs/2607.16245)

**RTL- and metric-adjacent benchmarks**
- RTLLM designs range from roughly **121 to 2,435 cells** after synthesis. It is a generation benchmark, not one for understanding netlists. — [RTLLM PDF](https://arxiv.org/pdf/2308.05345)
- MetRex (arXiv 2411.03471) has LLMs reason about post-synthesis metrics. The authors note that LLMs are weak at numerical prediction. — [arXiv PDF](https://www.arxiv.org/pdf/2411.03471)
- ChipBench (arXiv 2601.21448, 2026): GPT-4o reaches about 10% pass@1 on its own set, versus about 65% on RTLLM V2 and VerilogEval V2. — [arXiv PDF](https://arxiv.org/pdf/2601.21448)
- ChipNeMo (NVIDIA, arXiv 2311.00176) evaluates an engineering-assistant chatbot, EDA script generation and bug summarization. **Netlist ingestion is not one of its tasks.** — [arXiv PDF](https://arxiv.org/pdf/2311.00176)

**Analog/SPICE (brief) [ANALOG/SPICE]**
- AMSNet (arXiv 2405.09045) is an automatically converted dataset of schematics paired with SPICE netlists (AMSNet-1.0, described as preliminary). — [arXiv HTML](https://arxiv.org/html/2405.09045v2)
- Schemato (arXiv 2411.13899) is an LLM for netlist-to-schematic conversion (LTSpice). — [arXiv HTML](https://arxiv.org/html/2411.13899)
- Netlist Babel Fish (arXiv 2504.10240) converts netlist formats with RAG. — [arXiv HTML](https://arxiv.org/html/2504.10240v5)
- NetlistBench (arXiv 2608.12197, Aug 2026) measures LLM reliability at recognizing and manipulating SPICE netlists. It concludes that netlist reliability is a distinct bottleneck. — [arXiv HTML](https://arxiv.org/html/2608.12197)
- The CIRCUIT benchmark (arXiv 2502.07980) has 510 analog QA pairs. GPT-4-turbo is best but struggles with complex topologies even when given the netlist. — [arXiv HTML](https://arxiv.org/html/2502.07980v1)
- The survey "LLMs for EDA: Future or Mirage?" (ACM TODAES 2025) reports **GPT-4o at 48.04% global accuracy** on circuit-reasoning QA and notes it is weak on netlist topologies. — [ACM DL](https://dl.acm.org/doi/10.1145/3736167)
- LLM-SPICEMixer/IGEL (arXiv 2608.23317, Aug 2026) adds an LLM proposal operator to genetic SPICE-netlist search: +8.4% median training reward and +8.8% median test reward. — [arXiv PDF](https://arxiv.org/pdf/2608.23317)

**Surveys that list netlist-level LLM work**
- Pan et al., "A Survey of Research in LLMs for EDA", ACM TODAES 2025 (arXiv 2501.09655). — [ACM DL](https://dl.acm.org/doi/10.1145/3715324); [arXiv PDF](https://arxiv.org/pdf/2501.09655)
- Z. Xie, "Foundation AI Models for VLSI Circuit Design and EDA" (survey/slides), covering netlist encoders and their downstream tasks. — [PDF](https://zhiyaoxie.com/files/Survey_CFM.pdf)
- "The Dawn of Agentic EDA" (arXiv 2512.23189). — [arXiv HTML](https://arxiv.org/html/2512.23189v2)
- "Agentic EDA: A Handoff Perspective" (arXiv 2606.19795). — [arXiv PDF](https://arxiv.org/pdf/2606.19795)
- SoK on hardware reverse engineering (arXiv 2603.17883) lists **partitioning** as the first netlist-RE stage, before module identification. — [arXiv PDF](https://arxiv.org/pdf/2603.17883)

### Inferences
- The *register cone* (sequential: a register back to registers and PIs) and the *fixed-depth fan-in cone* (DeepGate4) are the recurring partition units in systems that work at scale. The partition is decided by circuit structure, not by token budget.
- Raw-text gate-level LLM approaches (Gate-Breaker, NtNDet) are evaluated on VerilogEval-scale or Trust-Hub-scale circuits. They show feasibility for small netlists but not for netlists larger than the context window.
- No public LLM system does netlist reasoning through tool calls or a graph database for large netlists. That design pattern exists for code repositories (CodexGraph, NAACL 2025) but not yet, as far as I found, for netlists. — [CodexGraph PDF](https://aclanthology.org/2025.naacl-long.7.pdf)

### Gaps
- MasterRTL. Searches returned nothing about it. From background knowledge, not verified in this session: it is an ICCAD 2023 HKUST work on pre-synthesis PPA estimation over a bit-level "simple operator graph", and it is not an LLM system. Verify before citing.
- I found no paper by the name "NetlistGPT". LLM-based functional-block identification on gate-level netlists appears only inside GenEDA's tasks and Gate-Breaker; I found no standalone system.
- I found no LLM-based logic-locking attack or analysis on gate-level netlists (searches returned only pre-LLM ML attacks such as SAIL, SnapShot and DeepAttack).
- DeepCircuitX's and CircuitFusion's quantitative results and design-size statistics were not retrievable.
- Gate-Breaker's per-quartile netlist sizes (gates or tokens) and its prompt-truncation behavior were not retrievable.

---

## Q3. Largest netlist demonstrated, token counts per gate, and what happens beyond the limit

### Takeaway
No LLM-in-the-loop netlist system I found shows end-to-end reasoning on a single netlist with tens of thousands of gates or more. The largest LLM-paired gate-level data I found is GenEDA/NetTAG's ITC99 set (avg ~1K gates, ~15K tokens) and Chipyard (avg ~24K tokens). Those sizes are handled by register-cone chunking with an 8K-token per-gate-text encoder. Only the non-LLM DeepGate4 encoder shows 400K–1.6M gates, and it does so by cone partitioning. Raw-text approaches report no behavior beyond the context window.

### Cited Findings
- GenEDA/NetTAG data: ITC99 averages **1,025 gates and 15k tokens** per circuit; OpenCores averages **173 gates and 9k tokens**; Chipyard averages **24k tokens**. — [GenEDA arXiv HTML](https://arxiv.org/html/2504.09485)
- NetTAG's ExprLLM takes **8k max input tokens** per text attribute, and large sequential designs are chunked into register cones (313k netlists → 100k cones, as extracted). — [NetTAG arXiv PDF](https://arxiv.org/pdf/2504.09260); [GitHub README](https://github.com/hkust-zhiyao/NetTAG/blob/main/README.md)
- CircuitFusion's code modality uses NV-Embed-V1 with a **32K-token** maximum, and the sub-circuit unit is the register cone. — [arXiv HTML](https://arxiv.org/html/2505.02168)
- DeepGate4 (no LLM) trains on circuits averaging ~15K gates, generalizes to ~400K gates, and keeps memory constant from 400K to 1.6M gates, though the 1.6M case is a 4× duplicated EPFL circuit. — [arXiv HTML v3](https://arxiv.org/html/2502.01681v3)
- Gate-Breaker's success falls from 75.1% overall to 53.4% on the most complex quartile, a falloff with complexity. The size of the largest netlist is not stated in the extract. — [IEEE Xplore](https://ieeexplore.ieee.org/document/11304682/)
- LLM4Netlist, per a third party: hard to scale to large circuits, and inference time grows significantly with circuit size because of iterative decoding. — [ArtNet arXiv HTML](https://arxiv.org/html/2510.13582v1)
- The RTLLM reference designs run 121–2,435 cells after synthesis, a typical "LLM benchmark-scale" range. — [RTLLM PDF](https://arxiv.org/pdf/2308.05345)
- The classical non-LLM ATPG generator CONTEST reached 100% test efficiency on ISCAS circuits with up to 38,000 nodes, a scale reference for test generation. The search engine attributed this to the IEEE paper "Deterministic test generation for non-classical faults on the gate level"; I have not checked that attribution. — [IEEE Xplore 485343](https://ieeexplore.ieee.org/document/485343)

### Inferences
- **Tokens per gate (derived from GenEDA's Table II averages):** about **15 tokens/gate** for ITC99 (15k/1,025) and about **52 tokens/gate** for OpenCores (9k/173). The large difference implies a fixed per-module overhead (port and wire declarations), a different verbosity of the text form, or that "tokens" counts a different serialization (for example, expression attributes) than raw gate instances. Treat 15–50 tokens per gate as a rough planning range for structural Verilog. At that rate, a 128K-token window holds roughly 2.5K–8.5K gates of raw netlist text, and a 1M-token window roughly 20K–65K gates. These are extrapolations, not measured results.
- For ATPG-scale designs (ITC99 b17–b22, OpenABC-D, industrial blocks), raw-text whole-netlist ingestion is not proven feasible by any work found. The proven routes chunk by register or fan-in cone, or compress to embeddings.

### Gaps
- No system reports what happens when a single register cone, or a single gate's PySMT expression, exceeds the 8K ExprLLM limit (truncation versus further splitting). This is undocumented in the extracts.
- No paper reports a measured accuracy-versus-netlist-size curve for an LLM consuming raw gate-level text.
- GenEDA's maximum (not average) circuit size was not retrievable.

---

## Q4. Which systems handle variable netlist size by design, and which truncate or restrict to small circuits

### Takeaway
Systems that handle variable size *by design*: NetTAG (register-cone chunking plus per-gate text embedding plus a graph transformer), GenEDA (inherits NetTAG embeddings as soft tokens, or annotated text for frozen LLMs), CircuitFusion (register-cone sub-circuits), and, on the non-LLM side, DeepGate4 (overlapping fixed-depth cones). Systems restricted to small inputs: Gate-Breaker (VerilogEval-scale), NtNDet (Trust-Hub-scale), LLM4Netlist (scaling flagged as weak), Circuit Transformer (small AIGs; exact size not retrieved), and all of the analog/SPICE works (single schematics).

### Cited Findings
- NetTAG chunks by register cone: each register is backtraced to other registers, giving a sub-circuit that captures its complete state transition. — [NetTAG arXiv PDF](https://arxiv.org/pdf/2504.09260)
- CircuitFusion uses the register-cone sub-circuit as its representation unit, and its graph encoder emits per-node embedding sequences. — [arXiv HTML](https://arxiv.org/html/2505.02168)
- GenEDA projects NetTAG graph embeddings into the LLM through an MLP connector (trainable LLMs), or annotates gate-function predictions onto the netlist text (frozen LLMs). — [arXiv HTML](https://arxiv.org/html/2504.09485)
- DeepGate4 partitions into overlapping fixed-depth cones and uses sparse attention; its memory stays constant from 400K to 1.6M gates. — [arXiv HTML v3](https://arxiv.org/html/2502.01681v3); [liner review (secondary)](https://liner.com/review/deepgate4-efficient-and-effective-representation-learning-for-circuit-design-at)
- Gate-Breaker passes the synthesized netlist directly as LLM input text, using VerilogEval-derived problems. — [GitHub](https://github.com/NMSU-PEARL/Gate-Breaker)
- NtNDet converts the netlist into NL-like text for BERT, GPT-2 or T5, whose context windows are short (the paper's segmentation strategy was not retrievable). — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S095741742500288X)
- Pre-LLM reverse-engineering practice: partitioning is the first stage of netlist RE (SoK). GNN-RE classifies each gate from its local neighborhood. — [SoK PDF](https://arxiv.org/pdf/2603.17883); [GNN-RE PDF](https://www.researchgate.net/profile/Johann-Knechtel/publication/354429669_GNN-RE_Graph_Neural_Networks_for_Reverse_Engineering_of_Gate-Level_Netlists/links/613b5606f07b08309084c894/GNN-RE-Graph-Neural-Networks-for-Reverse-Engineering-of-Gate-Level-Netlists.pdf)

### Inferences
- **Methods with real feasibility evidence for variable-size post-synthesis netlists, in ranked order:**
  1. Cone partitioning plus a graph encoder plus an LLM via soft tokens (NetTAG → GenEDA). This is peer-reviewed (DAC'25, ICCAD'25), released, and tested on synthesized ITC99/OpenCores/Chipyard netlists, though at moderate average size.
  2. Cone-partitioned GNN or graph transformer alone (DeepGate4), demonstrated to 1.6M gates without an LLM. This is a candidate encoder for the soft-token route.
  3. Annotated or compressed text for frozen LLMs (GenEDA's prediction route), which reduces how much raw text the LLM must reason over but still needs the text to fit in context.
  4. Raw structural Verilog text, demonstrated only on small circuits (Gate-Breaker, NtNDet).
- **Methods with no proof of feasibility found:** hierarchical LLM summarization of netlist partitions, and RAG or tool-calling agents over netlist graph databases.
- **Relevance to ATPG (inference):** under full scan, flip-flops become pseudo-PI/PO, and each fault's test depends only on the fan-in and fan-out cones that reach observation points. That matches the register-cone and fan-in-cone partitioning NetTAG and DeepGate4 use, which makes cone-level chunking a natural fit for LLM-assisted ATPG.

### Gaps
- No ablation found comparing chunking strategies (cone versus min-cut versus hierarchical module) for LLM netlist reasoning quality.
- How GenEDA aggregates many cone embeddings into one LLM prompt for a whole design (all cones as tokens versus pooling) was not retrievable.

---

## Q5. Quantitative evidence of feasibility and reproducibility

### Takeaway
The strongest quantitative evidence for LLM reasoning over gate-level netlists is GenEDA, which raises GPT-4o/DeepSeek-V3 from under 20% to 60–69% functional pass on arithmetic RTL reconstruction from bit-blasted netlists, and Gate-Breaker, with o3-mini at 75.1% netlist-to-RTL success on 156 small circuits. Both are peer-reviewed, and both have public repos. I found no result on the ISCAS85/89 or EPFL suites for an LLM-in-the-loop netlist system. The large-benchmark evidence (EPFL, ITC99, OpenABC-D) comes from non-LLM encoders such as DeepGate4.

### Cited Findings
- GenEDA: GPT-4o and DeepSeek-V3 alone score under 20% functional success. GenEDA + DeepSeek-V3 scores 97% syntax / 88% GPT-score / 60% functional in one version, and 96% / 88% / 69% in another. GenEDA-7B reaches a GPT Score of 0.62 on the spec tasks. Code: hkust-zhiyao/GenEDA. — [arXiv PDF](https://arxiv.org/pdf/2504.09485); [arXiv HTML](https://arxiv.org/html/2504.09485)
- NetTAG beats task-specific methods and SOTA AIG encoders on four tasks, and gains from model and data scale. Code and model are public. — [GitHub](https://github.com/hkust-zhiyao/NetTAG); [arXiv PDF](https://arxiv.org/pdf/2504.09260)
- Gate-Breaker: o3-mini 75.1% overall, 53.4% on Q4, AST 0.924, CFG 0.871. Harness public. — [IEEE Xplore](https://ieeexplore.ieee.org/document/11304682/); [GitHub](https://github.com/NMSU-PEARL/Gate-Breaker)
- NtNDet improves on prior SOTA on Trust-Hub, TRIT-TC and TRIT-TS (margins as above). I did not confirm a code release. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S095741742500288X)
- Masala-CHAI (analog): +46% Pass@1 in AnalogCoder, >40% Pass@1 on hard analog blocks, open-sourced. — [arXiv PDF](https://arxiv.org/pdf/2411.14299)
- Netlistify (analog, no LLM): +2.4% connectivity accuracy, +20.7% recall and +12.4% F1 versus AMSNet. Code, models and a 100k-image dataset are public, and it won the Best Artifact award. — [GitHub](https://github.com/NYCU-AI-EDA/Netlistify); [ResearchGate](https://www.researchgate.net/publication/396395178_Netlistify_Transforming_Circuit_Schematics_into_Netlists_with_Deep_Learning)
- DeepGate4 (no LLM) is evaluated on EPFL, ITC99 and OpenABC-D, with scaling to 1.6M gates. — [arXiv HTML v3](https://arxiv.org/html/2502.01681v3)
- Circuit Transformer (88M, no pre-trained LLM) beats resyn2 with MCTS while keeping strict equivalence. — [arXiv abs](https://arxiv.org/abs/2403.13838)
- Survey evidence: GPT-4o reaches 48.04% global accuracy on circuit-reasoning QA and is weak on netlist topologies. — [ACM DL](https://dl.acm.org/doi/10.1145/3736167)

### Inferences
- **Peer-reviewed versus arXiv-only:** GenEDA (ICCAD'25), NetTAG (DAC'25), CircuitFusion, MGVGA and Circuit Transformer (all ICLR'25), Gate-Breaker (IPCCC'25), NtNDet (ESWA'25), LLM4Netlist (JETCAS'25), Netlistify (MLCAD'25) and LogicCraft (AICAS'25) are peer-reviewed. DeepCircuitX, ForgeEDA, "Wrong Code, Right Structure", the 2026 structural-manipulability paper, LoRD, NetlistBench, LLM-SPICEMixer and Masala-CHAI are arXiv-only as far as I confirmed.
- **Reproducibility:** public repos exist for GenEDA, NetTAG, CircuitFusion, CircuitEncoder, MGVGA, Gate-Breaker and Netlistify (and Masala-CHAI says it is open). That is enough to reproduce the soft-token route on new netlists.

### Gaps
- No ISCAS85/89 or EPFL result for an LLM-in-the-loop netlist reasoning system was found.
- Absolute NtNDet metrics, and LLM4Netlist metrics, were not retrievable.

---

## Q6. Systems that touch testability, fault analysis, ATPG or DFT (relevance to the user's LLM-for-ATPG project)

### Takeaway
I found **no published LLM system that performs ATPG, fault simulation, or DFT insertion on gate-level netlists** with reported fault coverage. Adjacent work falls into three groups: LLM *functional* test or testbench generation at RTL (VerilogReader, ChatTest), non-LLM learned ATPG (DEFT, a differentiable ATPG), and gate-level Trojan detection, which is a security-oriented "anomalous logic" analysis. A public dataset pipeline, "ASAP7 Language of Test", which turns RTL into gate-level netlists with ATPG patterns and fault simulation as LLM training data, appears to belong to the same GitHub owner as the user's atpgllm repo.

### Cited Findings
- VerilogReader (arXiv 2406.04373), "LLM-Aided Hardware Test Generation", generates functional tests at RTL, not structural ATPG on netlists. — [arXiv PDF](https://arxiv.org/pdf/2406.04373)
- An agentic-EDA survey (arXiv 2606.19795) cites ChatTest (coverage-enhanced testbench generation with LLMs) and LLM-based assertion generation among 2026 DAC/DATE verification works. Neither is gate-level ATPG. — [arXiv PDF](https://arxiv.org/pdf/2606.19795)
- DEFT (arXiv 2512.23746) is a differentiable ATPG that relaxes the circuit graph into continuous optimization. With X-bits taken into account, its patterns use 19.3% fewer 0/1 bits and detect 35% more faults. It uses no LLM. — [arXiv PDF](https://arxiv.org/pdf/2512.23746)
- The "ASAP7 Language of Test" repo (github.com/christos-vasileiou) is a pipeline: RTL → gate-level structural netlists → ATPG patterns → fault simulation → LLM training data for fault-aware test generation. It is a dataset pipeline, not a demonstrated LLM test generator. — [GitHub](https://github.com/christos-vasileiou/asap7-language-of-test)
- Gate-level Trojan detection works (NtNDet with LMs; LoRD, ImprovedBiGCN and structural manipulability without LLMs) are the closest gate-level "fault-like anomaly" analyses. — [NtNDet](https://www.sciencedirect.com/science/article/abs/pii/S095741742500288X); [LoRD](https://arxiv.org/pdf/2609.17922); [ImprovedBiGCN](https://github.com/HahhaNa/Hardware-Trojan-Detection-on-Gate-Level-Netlists); [manipulability](https://arxiv.org/abs/2607.16245)
- For non-LLM baselines on ISCAS-85, a PODEM implementation reports 98.08% weighted stuck-at coverage. — [GitHub rohan-js/podem-atpg-iscas85](https://github.com/rohan-js/podem-atpg-iscas85)

### Inferences
- An LLM-for-ATPG project would be first or near-first in the published record. The closest transferable encoding evidence is NetTAG/GenEDA's cone-chunked, embedding-as-soft-token approach. NetTAG's per-gate symbolic-expression attribute is also a natural carrier for controllability and observability information such as SCOAP-like features (an inference, not demonstrated).
- Baselines a reviewer would expect: PODEM/FAN (open source) on ISCAS85/89 and ITC99, and DEFT as the learned-ATPG comparator.

### Gaps
- I found no evidence of LLMs being used for DFT scan insertion, test-point insertion, or fault diagnosis on netlists. The searches returned only conventional DFT literature.
- Whether any 2026 DAC/ITC/ETS/VTS paper (not indexed by my searches) uses LLMs for gate-level ATPG remains unverified. Re-search those venues directly when full web access is available.
