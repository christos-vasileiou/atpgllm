# General (Domain-Agnostic) Graph-to-LLM Encoding Methods and Their Transferability to Large Gate-Level Netlists

*How these notes were sourced (read first):* This session could not reach arxiv.org, openreview.net, aclanthology.org, proceedings.neurips.cc, semanticscholar.org, huggingface.co or paperswithcode.com; the egress proxy refused them with 403. The facts below come from (a) web-search result extracts of the primary papers, whose arXiv, proceedings and ACL Anthology URLs are cited as the canonical sources, and (b) official GitHub READMEs fetched from raw.githubusercontent.com. Full-paper tables were **not** read directly. Numbers marked "(secondary)" came only from third-party summaries and should be checked against the PDF before they go into a final report. Venue labels: **[PR]** = peer-reviewed conference or journal; **[WS]** = workshop; **[PP]** = preprint. Netlist-applicability judgments are my own inferences and are labeled as such.

---

## Key Question 1 — Text serialization: how does accuracy degrade with graph size, and which formats are the most token-efficient and accurate?

### Takeaway
The standard text-serialization benchmarks test tiny graphs: ≤50 nodes in GraphArena, ≤160 in GrAlgoBench, and roughly 5–20 in GraphQA (not verified). Only 2026 code-vs-text studies go up to 10k nodes. Even at these sizes, accuracy falls steeply with node count. Non-reasoning frontier models drop to about 5–7% on NP-complete tasks over "large" graphs of only 10–50 nodes (DeepSeek-R1 reaches 43%), reasoning models fall below 50% beyond about 120 nodes, and text-only reasoning collapses at 5k–10k nodes, where code or tool execution still works. The choice of format (adjacency or incident list vs. edge list vs. GML/GraphML) moves accuracy by roughly 5–60 points, but no format wins everywhere. Compact adjacency- or incident-style lists are the most consistent choice for both accuracy and token cost. No general benchmark evaluates Verilog-like serialization.

### Cited Findings

**Talk like a Graph / GraphQA (Fatemi, Halcrow, Perozzi; ICLR 2024 [PR])**
- First systematic study of graph-to-text encoding functions for LLMs. Introduces the GraphQA benchmark: edge existence, node degree, node count, edge count, connected nodes, cycle check. Choosing the right encoding improved graph-reasoning performance by **4.8% to 61.8%, depending on the task** — [arXiv 2310.04560](https://arxiv.org/abs/2310.04560); [ICLR 2024 proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/bf72f65f30eedf5d48da6980ee02b589-Paper-Conference.pdf); [Google Research blog](https://research.google/blog/talk-like-a-graph-encoding-graphs-for-large-language-models/)
- Performance depends on three things: the encoding method, the task, and **the structure (generator) of the graph**. Graphs came from Erdős–Rényi, Barabási–Albert, SBM, path, star and complete generators (generator list is secondary) — [arXiv 2310.04560](https://arxiv.org/abs/2310.04560); [Lacuna summary (secondary)](https://lacuna.tiptreesystems.com/work/talk-like-a-graph-encoding-graphs-for-large-language-models/wrk_1584d944d33e4e76248f8f7c23ed3ecf)
- The **incident encoding** lists each node's neighbors. It beats adjacency (edge) encoding on node-degree and connected-nodes tasks because the relevant facts sit close together in the text. Integer node labels helped degree counting, while human-name labels helped connectivity tasks (secondary) — [Emergent Mind summary](https://www.emergentmind.com/papers/2310.04560); [arXiv 2310.04560](https://arxiv.org/abs/2310.04560)
- GraphQA zero-shot baseline numbers as re-reported in a later paper: edge existence 49.0, node count 24.2, cycle check 82.0 — [arXiv 2409.08864](https://arxiv.org/pdf/2409.08864)
- Code: [google-research/talk-like-a-graph](https://github.com/google-research/talk-like-a-graph)

**NLGraph (Wang et al., NeurIPS 2023 [PR])**
- 8 tasks. Standard set has 5,902 problems and the extended set 29,370. Each task is split into easy/medium/hard by **number of nodes and density**. The benefit of advanced prompting and in-context learning **shrinks on more complex graph problems**. On shortest path, accuracy falls steadily as the number of nodes on the path grows — [arXiv 2305.10037](https://arxiv.org/abs/2305.10037); [NeurIPS 2023 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/622afc4edf2824a1b6aaf5afe153fa93-Paper-Conference.pdf); [repo](https://github.com/Arthur-Heng/NLGraph)

**GPT4Graph (Guo et al., 2023 [PP])**
- Compares adjacency list, edge list, GML and GraphML inputs. Neither markup format wins consistently. 1-shot GML vs GraphML: size detection 54.50 vs 25.00, edge detection 50.45 vs 62.05 (the extracted tables were partly garbled). Self-format-explanation and self-summarization prompts helped — [arXiv 2305.15066](https://arxiv.org/html/2305.15066)

**GraphArena (Tang et al., ICLR 2025 [PR])**
- 10 tasks (4 polynomial, 6 NP-complete) on real graphs sampled by random walk with restart from DBLP (1.35M nodes), OpenFlights, social networks, DBpedia and PubChemQC. 500 small + 500 large problems per task, 10,000 in total. Outputs are classified as correct, suboptimal, hallucinatory or missing — [arXiv 2407.00379](https://arxiv.org/abs/2407.00379); [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/77fa8253adfc8b33209639f3e9985741-Paper-Conference.pdf)
- **"Large" is still tiny:** Neighbor/Distance tasks use 4–19 nodes (small) vs 20–50 (large). Component/Diameter/MCP/MIS/MVC use 4–14 vs 15–30. MCS/GED/TSP use 4–9 vs 10–20 — [arXiv 2407.00379v2](https://arxiv.org/html/2407.00379v2); [GraphThought (follows GraphArena specs)](https://arxiv.org/html/2502.11607)
- Accuracy from the official leaderboard (P = polynomial, NP = NP-complete):

  | Model | P small | P large | NP small | NP large |
  |---|---|---|---|---|
  | DeepSeek-R1 | 0.976 | 0.877 | 0.877 | **0.431** |
  | Claude-3.5-Sonnet (2024-10) | 0.822 | 0.587 | 0.478 | **0.072** |
  | GPT-4o (2024-08) | 0.769 | 0.435 | 0.473 | **0.063** |
  | Llama-3-70B | 0.612 | 0.316 | 0.368 | 0.047 |
  | Llama-3-8B | 0.285 | 0.094 | 0.202 | 0.019 |
  — [GraphArena official README](https://github.com/squareRoot3/GraphArena)
- Hallucination rises with graph size and task complexity, even for top models — [arXiv 2407.00379](https://arxiv.org/abs/2407.00379)
- Format: on GPT-4o the **edge list was more stable than the adjacency matrix**, and the adjacency list was sometimes slightly better than the edge list. The authors chose the edge list as their default — [arXiv 2407.00379](https://arxiv.org/pdf/2407.00379) (via search extract)

**GrAlgoBench — "Exposing Weaknesses of Large Reasoning Models through Graph Algorithm Problems" (ICLR 2026 [PR])**
- Large reasoning models get worse as graphs grow. **Qwen3-32B falls from 87.0% (31–50 nodes) to 47.0% (121–160 nodes)**, and most models are below 50% once graphs pass about 120 nodes. Errors are mostly in execution and memory. The benchmark stops at 160 nodes — [arXiv 2602.06319](https://arxiv.org/html/2602.06319v1); [paper note (ICLR 2026)](https://en.papernotes.org/ICLR2026/llm_reasoning/exposing_weaknesses_of_large_reasoning_models_through_graph_algorithm_problems/)

**GraphOmni (arXiv 2504.12764; reported as ICLR 2026 [PR], venue only from search extract)**
- Compares 7 serializations: Adjacency List, Adjacency Set, Edge List, Edge Set, Adjacency Matrix, GMaL, GMoL. **Adjacency List and Adjacency Set generally beat the more complex formats (GMaL, GMoL)**. Edge Set is best on cycle detection, and GMaL helps some closed-source models. Average *output* tokens are close across formats: ~484 for AL, ~494 for EL, ~505 for ES, ~461 for GMaL — [arXiv 2504.12764](https://arxiv.org/html/2504.12764v5)

**Graph Serializations for Reasoning LLMs (Yu, Bornea, Sil, Kirchhoff; Findings of EMNLP 2026 [PR])**
- Tests 15 text formats across 9 frontier LLM configurations. Format barely matters for knowledge extraction but **matters a lot for structural reasoning**: one search summary reports swings of up to about 28 points (secondary). **No single format dominates**, so the authors propose choosing the serialization per model, query and task at run time. They note their findings may not carry over to learned graph tokens, tools or Graph-RAG. "Compact formats such as adjacency and structured achieve high accuracy at low cost" — [arXiv 2610.22734](https://arxiv.org/abs/2610.22734); [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.558/)

**Robustness to serialization details**
- *Lost in Serialization* (ICML 2026 Workshop on Graph Foundation Models [WS]) splits a serialization into node labeling, edge encoding/ordering, and surface syntax. Larger models that were not fine-tuned are more robust. One secondary summary says fine-tuning buys tolerance to relabeling but not to edge order or format — [arXiv 2511.10234](https://arxiv.org/abs/2511.10234); [Pith summary (secondary)](https://pith.science/paper/2511.10234)
- LLMs often call a relabeled isomorphic graph "different", which suggests they react to label conventions rather than structure — [arXiv 2606.09484 [PP]](https://arxiv.org/html/2606.09484)
- In small LMs, adjacency-list serialization generally lowers error compared with edge lists — [arXiv 2603.06635 [PP]](https://arxiv.org/html/2603.06635v2)
- GraCoRe (COLING 2025 [PR]): comprehension and reasoning "degrade significantly" on graphs with many nodes and edges — [ACL Anthology PDF](https://aclanthology.org/2025.coling-main.531.pdf)

**Scaling past the context window (2026)**
- GraphGym, "Unified Multi-Dimensional Benchmark for Complex Graph Reasoning" (arXiv 2608.12391 [PP]): 202 composite tasks at **10, 100, 1,000 and 10,000 nodes**. All reasoning modes degrade as size grows. **Code-based reasoning beats text-based reasoning at every size**, and file-based code (graph loaded from a file) holds up better than inline code. The appendix quantifies how much serialized graph fits in a context window before truncation — [arXiv 2608.12391](https://arxiv.org/html/2608.12391)
- GraphSkill (arXiv 2603.06620 [PP]): ComplexGraph-L uses **5k- and 10k-node graphs**. Zero-shot coding reached 64.4 (DeepSeek) and 56.7 (Llama-70B) on composite tasks, with text-only methods far lower. Prior benchmarks "do not test graphs larger than the context window" — [arXiv 2603.06620](https://arxiv.org/pdf/2603.06620)

### Inferences
- **Token arithmetic (my estimate).** G-Retriever's WebQSP graphs average 1,371 nodes and **100,627 tokens** as text (see KQ2), about 73 tokens per node for an attributed graph. An untyped edge list like `n123 -> n456` costs roughly 5–8 BPE tokens per edge. A gate-level netlist with average fan-in about 2 therefore costs roughly 10–20 tokens per gate in a minimal edge list, plus the gate type: **~10^5–2×10^5 tokens for 10^4 gates, ~10^7–2×10^7 for 10^6 gates, ~10^8+ for 10^7 gates.** A 10^4-gate netlist is already at the edge of 128k–1M-token windows. At 10^6–10^7 gates, full-context serialization is ruled out by token count alone, before any reasoning degradation.
- **Accuracy is the binding constraint long before the context fills.** The evidence (GraphArena ≤50 nodes; GrAlgoBench below 50% past ~120 nodes) means LLM "mental execution" of graph algorithms such as path tracing, reachability or cut finding is unreliable even at 10^2 nodes. ATPG-like tasks (justification and propagation over fan-in/fan-out cones, which resemble NP-hard search) map onto GraphArena's NP category, where non-reasoning frontier models score about 5–7% on graphs of only 10–50 nodes.
- **Format for netlists.** The incident/adjacency-list result transfers directly. A per-gate line `g42 = NAND(g17, g23)` is an incident encoding keyed on fan-in. It is also Verilog/BENCH-like, which LLMs have seen in pre-training. That makes it plausibly the most token-efficient and accurate text format for gate-level logic, but **no general benchmark has tested it** (see Gaps). Fan-out is implicit and must be recovered by search, which hurts fault-propagation reasoning; adding fan-out lists doubles the tokens.
- **Netlist applicability verdict — text serialization:** Viable only for local windows (a fault-site cone of tens to a few hundred gates) or as the input to *code execution*. It is not viable as a whole-design representation. It keeps per-net addressability, because every net has a name, and that is its main advantage over pooled soft tokens.

### Gaps
- Exact GraphQA graph sizes (believed to be about 5–20 nodes) and per-encoding/per-model tables in Talk like a Graph could not be read: arXiv and proceedings were blocked.
- NLGraph's node ranges for easy/medium/hard and its per-difficulty accuracy tables were not retrieved.
- No study found that measures **input**-token cost per format across sizes. GraphOmni reports only output tokens.
- **No general (non-EDA) benchmark evaluates Verilog/BENCH-style serialization** against edge, adjacency or incident lists. That comparison would need the circuit-specific literature, which another researcher covers.
- GraphGym's (2608.12391) actual 1k/10k-node accuracy numbers were not visible in the search extracts.

---

## Key Question 2 — Learned graph tokens and soft prompts: tokens per graph or node, and the largest graph demonstrated

### Takeaway
Learned-token methods compress structure into a few soft tokens (one or a handful per graph, or per ego-subgraph). This produces large gains on small synthetic graphs: GraphToken lifted GraphQA node count from about 25% to about 99.6% (secondary), and GraphLLM cut context by 96.45%. But they have only ever been shown on (i) small whole graphs (GraphQA/NLGraph scale) or (ii) **sampled ego-subgraphs** of large graphs: ogbn-arxiv at 169k nodes, ogbn-products at 2.4M, WebQSP retrieved to about 18 nodes. Several 2025–2026 audits (GTEval, "When Graph Tokens Sink", "A Graph Talks, But Who's Listening?") find that LLMs only partly use graph tokens and lean heavily on text. None of these methods gives a mechanism for an LLM to *name* a specific node or net out of millions.

### Cited Findings

**GraphToken — "Let Your Graph Do the Talking" (Perozzi et al., 2024 [PP])**
- A trainable GNN encoder maps the graph to soft-prompt tokens for a **frozen** LLM; only the encoder is trained. On GraphQA it reports gains of **up to 73 percentage points** over text baselines. Node count reached 99.6% vs 25.3% for the best few-shot text baseline, and cycle check 95.6% vs 83.2% for plain soft prompting (task numbers secondary). No single GNN architecture was best on every task — [arXiv 2402.05862](https://arxiv.org/abs/2402.05862); [Lacuna summary (secondary)](https://lacuna.tiptreesystems.com/work/let-your-graph-do-the-talking-encoding-structured-data-for-llms/wrk_656c10690c2ea55cb84223f5f39849e7); [Emergent Mind (secondary)](https://www.emergentmind.com/papers/2402.05862)
- Code is shared with Talk like a Graph — [google-research/talk-like-a-graph](https://github.com/google-research/talk-like-a-graph)

**GraphLLM (Chai et al., 2023 [PP])**
- A graph transformer is fused into the LLM through **graph-enhanced prefix tuning**, trained end-to-end. Compared with Graph2Text, it reports **96.45% shorter context**, **+54.44% average accuracy** and **3.42× faster inference** on four synthetic tasks: substructure counting, maximum triplet sum, shortest path, bipartite matching. It needs one 80 GB GPU with LLaMA-2-7B — [arXiv 2310.05845](https://arxiv.org/abs/2310.05845); [repo](https://github.com/mistyreed63849/Graph-LLM)

**LLaGA (Chen et al., ICML 2024 [PR])**
- Two parameter-free templates turn a node's neighborhood into a **fixed-length node sequence**. The *Neighborhood Detail* template walks a sampled computation tree level by level; the *Hop-Field Overview* template uses one summarized hop embedding per hop. A projector maps the sequence into token space: a 2-layer MLP for the general model, linear for the classification experts — [arXiv 2402.08170](https://arxiv.org/abs/2402.08170); [ICML 2024 poster](https://icml.cc/virtual/2024/poster/34739); [repo](https://github.com/VITA-Group/LLaGA)
- Evaluated on ogbn-arxiv, **ogbn-products**, PubMed and Cora. The released data files are named `sampled_2_10_*`, consistent with 2-hop sampling and 10 neighbors per hop — [LLaGA README](https://github.com/VITA-Group/LLaGA)

**GraphGPT (Tang et al.; usually cited as SIGIR 2024 [PR], venue not confirmed this session)**
- Dual-stage graph instruction tuning. A projector aligns GNN features to graph tokens, text-graph grounding is pre-aligned, and CoT distillation is added. The "lightweight" training script runs on two 24 GB RTX 3090s. Tasks are node classification and link prediction on Arxiv and PubMed — [GraphGPT README](https://github.com/HKUDS/GraphGPT); [arXiv 2310.13023](https://arxiv.org/abs/2310.13023)

**G-Retriever (He et al., NeurIPS 2024 [PR])**
- RAG over textual graphs. A Prize-Collecting Steiner Tree subgraph is retrieved, then encoded by a GNN into a soft prompt and also textualized. On WebQSP, the graph shrinks from **1,371 nodes / 100,627 tokens to 18 nodes / 610 tokens (−99%)**, and training time drops 67% — [arXiv 2402.07630](https://arxiv.org/abs/2402.07630); [NeurIPS 2024 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/efaf1c9726648c8ba363a5c927440529-Paper-Conference.pdf); [author blog](https://medium.com/@xxhe/graph-retrieval-augmented-generation-rag-beb19dc30424); [T-Retriever (same baseline numbers)](https://arxiv.org/pdf/2601.04945)
- Hallucination: the authors manually checked 100 responses. Secondary sources disagree on the headline: "valid responses 8% → 62%" vs "−54% hallucination". Check Table 5 — [alphaxiv overview (secondary)](https://www.alphaxiv.org/overview/2402.07630v3)
- Now built into PyG 2.6 — [G-Retriever README](https://github.com/XiaoxinHe/G-Retriever)

**InstructGLM (Ye et al.)**
- Uses shallow, task-specific graph features as graph embeddings, mapped to graph tokens by a lightweight MLP projector (as described by a 2026 evaluation paper) — [arXiv 2605.03514](https://arxiv.org/pdf/2605.03514)

**GraphTranslator (Zhang et al., WWW 2024 [PR])**
- GraphSAGE embeddings pass through a **Q-Former-based Translator** that emits query tokens, which act as **prefix tokens to a frozen LLM**. A "Producer" LLM textualizes node and neighbor information to create alignment data. Evaluated on zero-shot node classification and graph QA — [arXiv 2402.07197](https://arxiv.org/abs/2402.07197); [README](https://github.com/alibaba/GraphTranslator); [description in 2605.03514](https://arxiv.org/html/2605.03514v1)
- The same Q-Former "graph-query-token" pattern is the template for "Graph Q-Former"-style projectors — [arXiv 2605.03514](https://arxiv.org/html/2605.03514v1)

**TEA-GLM (Wang et al., NeurIPS 2024 [PR])**
- Pre-trains a GNN whose representations align with the LLM's token embeddings, then maps them to graph tokens with a linear projector for zero-shot cross-dataset and cross-task transfer — [NeurIPS 2024 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/0b77d3a82b59e9d9899370b378087faf-Paper-Conference.pdf); [arXiv 2408.14512](https://arxiv.org/html/2408.14512v3)

**GOFA (Kong et al., ICLR 2025 [PR])**
- **Interleaves randomly initialized GNN layers inside a frozen LLM.** Each node's or edge's text is followed by K **memory tokens**, and their outputs serve as node and edge features for the GNN. Pre-trained on graph-level next-word prediction, QA and structural tasks, it is strong zero-shot. Pre-training took **4 days on 4× A100-80GB** — [arXiv 2407.09709](https://arxiv.org/abs/2407.09709); [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/652c104b5b0652a03684efeaf805463b-Paper-Conference.pdf); [GOFA README](https://github.com/JiaruiFeng/GOFA)

**UniGraph (He & Hooi, KDD 2025 [PR])**
- A cross-domain foundation model for text-attributed graphs (DeBERTa + GAT), self-supervised, evaluated on 11 datasets from 5 domains — [ACM DL](https://dl.acm.org/doi/pdf/10.1145/3690624.3709277); [arXiv 2402.13630](https://arxiv.org/html/2402.13630v3)

**LangGFM ([PP]; ICLR 2025 venue not confirmed)**
- Uses an LLM alone as the graph foundation model and feeds graphs as **standard text formats (GraphML, GML, JSON, Markdown)**. Established formats beat custom ones. Self-supervised objectives are a topology autoencoder and a feature-masked autoencoder — [Moonlight review (secondary)](https://www.themoonlight.io/en/review/langgfm-a-large-language-model-alone-can-be-a-powerful-graph-foundation-model); [Graph Foundation Models survey](https://arxiv.org/pdf/2505.15116)

**2025–2026 audits and successors**
- **GTEval — "Revisiting Graph-Tokenizing LLMs" (arXiv 2605.03514 [PP], 2026):** across six representative graph-token LLMs, models **do not fully understand graph tokens**. They are over- or under-sensitive to instruction changes and **depend heavily on text**. Graph tokens keep task-relevant information and draw attention, but how much it is used varies. Extra instruction tuning only partly helps — [arXiv 2605.03514](https://arxiv.org/abs/2605.03514)
- **"When Graph Tokens Sink" (arXiv 2606.03712 [PP], June 2026):** pruning, repositioning and swapping experiments show that graph "sink" tokens (biased to early positions) are **not** the most important tokens for structure or semantics. This points to weaknesses in how current graph tokens are built, placed and aligned. In TEA-GLM the model often attends to tokens 2–4 rather than sink tokens 0–1 — [arXiv 2606.03712](https://arxiv.org/abs/2606.03712)
- **"A Graph Talks, But Who's Listening?" (Findings of ACL 2026 [PR]):** on the synthetic CLEGR benchmark, **soft-prompted LLMs given only node text match graph-language models with full GNN backbones**, and GLMs do not beat them on reasoning. Structure does matter on Cora and CiteSeer — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1624/); [arXiv 2508.20583](https://arxiv.org/html/2508.20583)
- **Agentic Graph Token (AGT) reasoning (arXiv 2608.00542 [PP], Aug 2026):** replaces "static single-shot" graph tokens with **on-demand graph-token materialization interleaved with reasoning**. Reports large gains and zero-shot transfer across 7 graph domains (numbers not verified) — [arXiv 2608.00542](https://arxiv.org/abs/2608.00542)
- Reconstructive graph instruction tuning for graph-tokenizing LLMs (arXiv 2603.01385 [PP]) — [awesomepapers listing](https://awesomepapers.io/graph-learning/papers/2603.01385)
- Survey framing "Graph2text vs Graph2token" — [arXiv 2501.01124](https://arxiv.org/pdf/2501.01124)

### Inferences
- **Scale actually demonstrated:** the largest *graphs* touched are ogbn-products (2.4M nodes) for LLaGA and GraphGPT-style models. The LLM never sees the whole graph, only a sampled 2-hop ego-subgraph per query (about 111 nodes if 2 hops × 10 neighbors, inferred from LLaGA's file names). The largest *whole graph* given to an LLM through soft tokens is GraphQA/NLGraph scale (tens of nodes), or a retrieved subgraph of about 18 nodes (G-Retriever). **No learned-token method has fed a 10^4+ node graph as a whole.**
- **Token budget patterns:** (a) graph-level pooled: one or a few tokens per graph (GraphToken, G-Retriever's GNN prompt); (b) node-centric: one projected token per node in a fixed-size ego sequence (LLaGA, GraphGPT, TEA-GLM, InstructGLM); (c) Q-Former: a fixed set of query tokens per node or graph (GraphTranslator); (d) memory tokens: K per node or edge inside the LLM (GOFA). Pattern (d) costs O(N·K) tokens of LLM compute and does not scale to 10^6 nodes.
- **Netlist applicability (my judgments):**
  - *GraphToken / pooled soft prompts:* **Low.** Squashing a 10^5–10^7-gate netlist into a few vectors erases per-net identity, so you cannot ask "which nets are untestable" or "give a test pattern for a fault on net n123". It could serve as a global "design summary" token alongside other inputs.
  - *GraphLLM (prefix from a graph transformer):* **Low–medium.** It is only as scalable as its graph transformer (see KQ6) and has no per-net output head.
  - *LLaGA / GraphGPT / TEA-GLM (ego-subgraph tokens):* **Medium for node-local tasks** such as gate-type or SCOAP-like testability classification of one net, because the template naturally centers on a "fault site". Weaknesses: undirected k-hop templates ignore fan-in/fan-out direction and logic depth, and 2-hop/10-neighbor sampling truncates the deep cones ATPG needs. A netlist variant would need separate fan-in and fan-out cone templates with level (depth) indices.
  - *G-Retriever:* **Medium–high as a pattern.** Retrieve the relevant cone (PCST, or simply the structural fan-in/fan-out cone of the fault), then textualize it *and* soft-prompt it. The textualization keeps net names, so answers can cite nets. It has only been shown at about 10^3 nodes per graph.
  - *GraphTranslator / Q-Former:* **Low–medium.** It needs natural-language "Producer" data, and netlists have no node text beyond gate type.
  - *GOFA:* **Low** at netlist scale. Per-node LLM compute is prohibitive for 10^6 nodes, though the interleaved GNN-in-LLM idea is attractive for small cones.
  - *LangGFM:* same as text serialization (KQ1).
  - **Cross-cutting caution:** the 2026 audits suggest that much of the reported gain comes from text, not graph tokens. Netlists have almost **no** node text (only gate types), so graph-token methods would have to carry structure by themselves, which is exactly the part these models are weakest at. This is a real risk for the transfer.
- Cross-reference (not evaluated here): circuit-specific encoders (DeepGate family, NetTAG, GenEDA) belong to the other researcher. They are the natural "GNN encoder" to put behind a LLaGA- or G-Retriever-style projector.

### Gaps
- Exact soft-token counts per graph for GraphToken, GraphGPT, InstructGLM and TEA-GLM, and the exact K for GOFA, could not be checked: the PDFs were blocked.
- GraphToken's numbers (73 pp, 99.6%, 95.6%) are from secondary summaries only.
- InstructGLM venue (believed Findings of EACL 2024) and LangGFM venue were not confirmed.
- "Graph Q-Former" as a standalone named paper was not found. The Q-Former projector pattern is documented through GraphTranslator.
- No learned-token method has been evaluated on directed acyclic graphs with typed, text-free nodes like netlists.

---

## Key Question 3 — Tokenization by substructure (motif/BPE, hierarchical, VQ codebooks, node-ID schemes): how compressive?

### Takeaway
Three families exist. (1) **Discrete VQ/RVQ codebooks** (GQT, Dr.E, VQGraph) give each node a short hierarchical code of 3 RVQ levels in GQT's follow-up setup. That is compact per node, but still O(N) tokens for the graph. (2) **Substructure/BPE tokenizers** (GraphTokenizer, ICLR 2026; motif/NPE for molecules) serialize the graph reversibly in O(|E|) and merge frequent labeled-edge patterns, which suits graphs with small label alphabets and repetitive motifs, as netlists have. (3) **Whole-graph "K graph words"** (GraphsGPT; K = 1–8) and **anchor-based node IDs** (NT-LLM; NodePiece uses 1–10% of nodes as anchors). All of them have been demonstrated mostly on molecules and small or medium graphs. None reports compression ratios on graphs of 10^5+ nodes.

### Cited Findings
- **GQT — Learning Graph Quantized Tokenizers (Wang et al., ICLR 2025 [PR]):** trains the tokenizer separately from the Transformer using multi-task graph self-supervision. **Residual Vector Quantization** produces hierarchical discrete tokens, with "significantly reduced memory requirements" and better generalization. With token modulation it is state of the art on **20 of 22 benchmarks, including large-scale homophilic and heterophilic datasets** — [arXiv 2410.13798](https://arxiv.org/abs/2410.13798); [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/28351); [mlanthology](https://mlanthology.org/iclr/2025/wang2025iclr-learning/)
  - A follow-up sets "the number of codebooks to three for GQT", i.e., about 3 discrete codes per node — [arXiv 2510.12369 (hierarchical quantized tokenization) [PP]](https://arxiv.org/pdf/2510.12369)
- **Dr.E — Multi-View Structural Graph Wordification (Liu et al., AAAI 2025 [PR]):** a dual-residual RQ-VAE with a GNN encoder whose **codebook is the LLM's own vocabulary**, so each node is represented by *real words* taken from different hop views — [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/34652/36807); [arXiv 2406.15504](https://arxiv.org/abs/2406.15504); [discrete-tokenization survey (TPAMI 2026)](https://arxiv.org/pdf/2507.22920)
- **VQGraph (ICLR 2024 [PR]):** a graph VQ-VAE that tokenizes nodes by neighborhood structure, used for GNN→MLP distillation — [PDF](https://cs.stanford.edu/people/jure/pubs/rethinking-iclr24.pdf)
- Survey of quantized graph representation learning and its links to LLMs — [arXiv 2502.00681](https://arxiv.org/pdf/2502.00681)
- **GraphTokenizer — "Graph Tokenization for Bridging Graphs and Transformers" (ICLR 2026 [PR]):**
  - Pipeline: count frequent labeled-edge patterns, then serialize the graph **reversibly and deterministically** with a frequency-guided Eulerian circuit ("Feuler", **O(|E|)**) so frequent substructures sit next to each other, then run **standard BPE**, then feed off-the-shelf BERT or GTE — [README](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers); [arXiv 2603.11099](https://arxiv.org/abs/2603.11099)
  - Serializers supported: Feuler and FCPP (reversible and deterministic); Eulerian and CPP (reversible only); canonical SMILES (molecules only); **DFS/BFS/topological order (neither reversible nor deterministic)** — [README](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers)
  - State of the art on 14 graph classification and regression datasets. For example, GT+GTE scores molhiv AUC 87.4 vs GraphGPS 78.5, Exphormer 82.3 and GraphMamba 81.2; ZINC MAE 0.131 vs GCN+ 0.116. Scaling from BERT-small to GTE gives consistent gains — [README results table](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers)
- **Node Pair Encoding / motif-BPE for molecules:** BPE-like merging of adjacent motifs, with frequent ring units seeded as indivisible vocabulary — [Graph Diffusion Transformers (arXiv 2510.08744)](https://arxiv.org/pdf/2510.08744); [arXiv 2605.15354](https://arxiv.org/pdf/2605.15354)
- **GraphsGPT — "A Graph is Worth K Words" (Gao, Dong et al., ICML 2024 [PR]):** a Graph2Seq encoder maps a whole graph to **K learnable "graph words"** and a decoder reconstructs the graph (an information-equivalence objective). Pre-trained on **100M molecules**, with checkpoints for **K = 1, 2, 4, 8**. State of the art on 8 of 9 classification and regression tasks — [README](https://github.com/A4Bio/GraphsGPT); [ICML poster](https://icml.cc/virtual/2024/poster/32612); [mlanthology](https://mlanthology.org/icml/2024/gao2024icml-graph/)
- **NT-LLM — anchor-based node tokenizer (Ji et al., arXiv 2410.10743 [PP]):** picks anchor nodes greedily up to a coverage threshold and encodes each node by its **distances to the anchors**, pre-trained with a rank-preserving objective, then prompt-tuning or LoRA. It is pitched as an alternative to costly GNN tokenizers — [arXiv 2410.10743](https://arxiv.org/abs/2410.10743); [OpenReview](https://openreview.net/forum?id=OcGPaztkjp)
  - Precedent: NodePiece (KGs) uses anchors as a vocabulary and reports **1–10% of nodes** are enough anchors in many cases — [NodePiece explainer](https://towardsdatascience.com/nodepiece-tokenizing-knowledge-graphs-6dd2b91847aa/)
- Graph Optimization Foundation Model, "Tokenizing Graph via a Language-Model Paradigm" (arXiv 2509.24256 [PP]) — [arXiv](https://arxiv.org/html/2509.24256)

### Inferences
- **Compression ratios (my estimates):**
  - RVQ codes give about **3 tokens per node**, roughly 3–7× fewer than an edge-list text serialization at 10–20 tokens per gate. Still O(N): about 3×10^6 tokens for 10^6 gates.
  - BPE over an Euler serialization starts at about 2|E| symbols. Merges then compress repeated labeled patterns. On netlists with a 5–10-type alphabet and heavy reuse (XOR trees, adders, MUX trees, scan chains) compression could be large, but **it has not been measured**.
  - GraphsGPT's K = 8 words per graph is extreme compression and **lossy for addressing**; it was only shown for molecules of tens of atoms.
- **Netlist applicability (my judgments):**
  - *GraphTokenizer (Feuler + BPE):* **Medium–high, the most promising general idea.** Netlists are labeled graphs with a tiny alphabet and many repeated motifs, which is exactly what BPE exploits. Reversibility keeps every net recoverable. Open issues: (i) Eulerian serialization of a directed DAG must keep edge direction and pin order; (ii) sequence length is still Θ(|E|) before merges, so 10^7 gates means about 10^7+ symbols; (iii) all demos are graph-level classification and regression on small graphs.
  - *GQT / RVQ codes:* **Medium.** Per-node codes give a compact "word" for each net's local structural role, usable as a node-ID-plus-structure token inside a retrieved cone. Codes are not unique per node, so a separate ID is still needed for addressing.
  - *Dr.E (vocabulary-as-codebook):* **Low–medium.** It produces human-readable tokens, but those words carry no meaning for gates.
  - *NT-LLM anchors:* **Medium for addressability.** Distance-to-anchor vectors to primary inputs, primary outputs or scan flops are close cousins of classic circuit features (logic level, distance to PI/PO, SCOAP-like observability distance). Anchor selection on a DAG should use directed distances.
  - *GraphsGPT K-words:* **Low** for net-level ATPG, though plausibly useful as a global design embedding.
  - *Node-ID tokens* (adding vocabulary entries per node, as InstructGLM-style approaches do): **Not viable** at 10^6–10^7 nets. The vocabulary explodes and IDs do not transfer across designs. Hierarchical names (`u_core/u_alu/n123`) tokenized by BPE are the practical node-ID scheme.

### Gaps
- No paper found reporting graph-tokenizer compression ratios (tokens per node or per edge) on graphs above about 10^5 nodes.
- GQT's exact tokens per node in the original paper, and its exact memory reduction, were not retrieved.
- "Hierarchical graph tokens" for very large graphs (multi-level coarsening into tokens) were not found as a mature LLM-facing method beyond the hierarchical RVQ codes above.
- GraphBPE (Shen & Póczos, 2024) and other earlier BPE-on-graph work were not checked this session.

---

## Key Question 4 — Structural and positional encodings that let a transformer read topology from a sequence; which work for DAGs and circuits?

### Takeaway
For DAGs, the best-supported approach is **depth (level) positional encoding plus a reachability-masked attention bias**: the DAG Transformer (NeurIPS 2023), which beat both plain transformers and DAG-specific GNNs and runs in time linear in total neighborhood size. For general directed graphs, **Magnetic-Laplacian eigenvector PEs** (ICML 2023) and their **multi-q** extension (ICLR 2025) are the principled spectral options. Multi-q explicitly targets "walk profiles" relevant to circuit performance prediction. Plain topological-order sequencing is weak because orders are not unique and lose dependencies. All of these were demonstrated on graphs of hundreds to thousands of nodes (code ASTs, citation DAGs, sorting networks). Eigenvector PEs need eigendecomposition, which is costly at 10^6–10^7 nodes.

### Cited Findings
- **DAG Transformer — "Transformers over Directed Acyclic Graphs" (Luo, Thost, Shi; NeurIPS 2023 [PR]):** DAG attention = a **reachability-based attention mask** (each node attends only to its ancestors and descendants) plus a **depth-based positional encoding**. It sits on top of any transformer. Runtime is linear in the sum of reachability-neighborhood sizes, far below quadratic. **On every dataset tested it beat the original transformers and the best DAG-specific GNNs.** Datasets: ogbg-code2 (source-code graphs), NA (neural architectures), self-citation, and Cora/Citeseer/Pubmed — [arXiv 2210.13148](https://arxiv.org/abs/2210.13148); [NeurIPS 2023 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/94e85561a342de88b559b72c9b29f638-Paper-Conference.pdf); [repo README](https://github.com/LUOyk1999/DAGformer)
- **Magnetic-Laplacian PE and directional random-walk PE — "Transformers Meet Directed Graphs" (Geisler et al., ICML 2023 [PR]):** direction-aware generalization of Laplacian eigenvector PEs. It beats baselines on reachability and directed-distance prediction and on correctness testing of **sorting networks**, while staying competitive on undirected tasks. On source code (ogbg-code2), SAT++ with MagLap beat prior state of the art by **2.85 points (14.7% relative)**. The paper notes that **near-sequential DAGs can have huge numbers of topological sorts**, which undermines sinusoidal PEs based on a topological order — [arXiv 2302.00049](https://arxiv.org/abs/2302.00049); [PMLR PDF](https://proceedings.mlr.press/v202/geisler23a/geisler23a.pdf); [ICML poster](https://icml.cc/virtual/2023/poster/24931)
- **Multi-q Magnetic Laplacian PE — "What Are Good Positional Encodings for Directed Graphs?" (Huang et al., ICLR 2025 [PR]):** shows existing PEs cannot represent **walk profiles** (directed walk-count generalizations). Concatenating eigenvectors from several phase parameters q provably expresses them. The authors motivate it with **program analysis and circuit performance prediction** — [arXiv 2407.20912](https://arxiv.org/abs/2407.20912); [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/28146); [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/7422317f84e8c83e4c1ad2ff87e1e88e-Paper-Conference.pdf)
- **PathLapPE (arXiv 2610.04657 [PP], Oct 2026):** spectral PE from path Laplacians of a directed path complex, with node- and edge-level features for directional higher-order structure. It needs **no tuning of the directionality hyperparameter q** and is competitive with MagLap in runtime and accuracy — [arXiv 2610.04657](https://arxiv.org/abs/2610.04657)
- **PACE (Dong et al., 2022):** a parallelizable DAG computation encoder. It notes that **topological orders of a DAG are often not unique** and that standard sequence PEs capture position in the sequence but **not node dependencies** — [arXiv 2203.10304](https://arxiv.org/pdf/2203.10304)
- For DAGs, PEs based on **partial orderings** can be applied directly as an alternative to spectral PEs — [arXiv 2506.16404 (directed graph generation, [PP]; ICLR 2026 per proceedings listing)](https://arxiv.org/pdf/2506.16404)
- "Transformers provably learn DAGs via kernel-guided mutual information" (arXiv 2510.25542 [PP]) is theoretical support that transformers can learn DAG structure — [arXiv](https://arxiv.org/pdf/2510.25542)
- Cross-reference only (circuit-specific, covered elsewhere): DeepGate3 observes that topological and canonical orderings **cannot tell isomorphic graphs apart** and uses learned positional embeddings on AIGs — [arXiv 2407.11095](https://arxiv.org/pdf/2407.11095). A generative "Circuit Transformer" turns an AIG into a trajectory for a transformer — [arXiv 2406.04699](https://arxiv.org/pdf/2406.04699)
- Graph Linearization (LLM-specific): when a graph *must* be a token sequence, the ordering and relabeling strategy matters. See "Lost in Serialization" in KQ1 — [arXiv 2511.10234](https://arxiv.org/abs/2511.10234)

### Inferences
- **The best match for netlists is DAG attention: depth PE plus reachability mask.** *Depth* is the circuit's **logic level**, a standard and cheap (O(|E|)) feature. The *reachability mask* is the union of a gate's **fan-in cone** (needed for justification) and **fan-out cone** (needed for fault propagation), which are exactly the regions ATPG and fault simulation reason over. Costs and risks: cone sizes in deep or high-fan-out logic (clock and reset trees, buses) can approach O(N) per node, so the mask needs caps such as depth-bounded cones or sampling.
- **MagLap / multi-q MagLap:** **medium.** It is principled for direction and reconvergent fan-out (walk profiles relate to reconvergence, a key source of ATPG difficulty). But a global eigendecomposition on 10^6–10^7 nodes is expensive and sign- and basis-ambiguous; practical use would be per cone or per partition. PathLapPE removes q-tuning but has no circuit results yet.
- **Topological-order sequencing alone:** **low**, because orders are non-unique, irreversible and lose structure (GraphTokenizer and PACE both make this point). It becomes viable when paired with explicit level IDs and explicit fan-in references (e.g., `g42@L7 = NAND(g17,g23)`), which turns ordering into a canonical textual netlist.
- **Laplacian and random-walk PEs (undirected):** **low–medium.** They ignore the direction that defines controllability vs observability. RWPE return probabilities in a DAG are trivially zero unless direction is symmetrized.
- **For LLMs specifically:** none of these PEs has been injected into a *pretrained LLM's* sequence to describe a netlist. In LLM practice, the "PE" for a gate would be explicit tokens such as level, PI/PO distance and cone sizes, or soft vectors added to node tokens (LLaGA's hop embeddings play this role).

### Gaps
- The DAG Transformer's exact ogbg-code2 and NA numbers and speedups were not retrieved, and the sorting-network numbers for MagLap were not visible.
- No 2025–2026 general-domain paper found that evaluates DAG PEs on AIG or netlist benchmarks. That is circuit-specific work, cross-referenced to the other researcher.
- PACE venue (believed ICML 2022) not confirmed this session.
- No evidence found on eigenvector PE quality or cost on directed graphs with 10^6+ nodes.

---

## Key Question 5 — Retrieval-augmented and agentic graph access: evidence that it scales where full-context fails

### Takeaway
This is the only family with direct evidence of working **beyond the context window**. Tool and code-execution approaches stay usable at 1k–10k nodes where text reasoning collapses: ComplexGraph-L at 5k/10k nodes, GraphGym up to 10k, GraphAgent-Reasoner up to 1,000 nodes while fine-tuned text models fail at 500+. Retrieval cuts context by about 99% (G-Retriever). Community summaries compress 1M-token corpora (GraphRAG). Knowledge-graph agents explore huge KGs through neighbor APIs (Think-on-Graph, StructGPT, Graph-CoT). Because the LLM never has to "hold" the graph, these methods keep **exact node identity** while touching only tiny subgraphs. That makes them the closest general analogue to an LLM driving an ATPG or fault-simulation engine.

### Cited Findings
- **G-Retriever (NeurIPS 2024 [PR]):** PCST subgraph retrieval takes WebQSP from **1,371 nodes / 100,627 tokens to 18 nodes / 610 tokens** — [arXiv 2402.07630](https://arxiv.org/abs/2402.07630); [secondary overview](https://www.alphaxiv.org/overview/2402.07630v3). T-Retriever (hierarchical tree retrieval, arXiv 2601.04945 [PP]) cuts tokens a further **up to 84.16%** relative to G-Retriever — [arXiv 2601.04945](https://arxiv.org/pdf/2601.04945)
- **GraphRAG — "From Local to Global" (Edge et al., Microsoft, arXiv 2404.16130 [PP]):** an LLM builds an entity graph, Leiden communities are summarized hierarchically, and queries are answered by map-reduce over summaries. Targets sensemaking over **~1M-token corpora**. Against vector RAG it wins **72–83% on comprehensiveness and 62–82% on diversity** (LLM-judged). Root-level (C0) summaries use **about 2–3% of the tokens** of the most detailed approach and still win 72%/62% — [arXiv 2404.16130](https://arxiv.org/html/2404.16130v2); [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/). Caveat: a blog cites a 2025 audit claiming the win rates fall after correcting for position and length bias. **Unverified** — [beancount.io (unverified)](https://beancount.io/bean-labs/research-logs/2026/06/04/graphrag-local-to-global-query-focused-summarization)
- **Think-on-Graph (Sun et al., ICLR 2024 [PR]):** the LLM agent runs **iterative beam search over the KG** to find promising multi-hop paths. ToG-2.0 (arXiv 2407.10805) mixes KG and unstructured retrieval — [ToG README](https://github.com/GasolSun36/ToG); [arXiv 2307.07697](https://arxiv.org/abs/2307.07697); [ToG-2 arXiv](https://arxiv.org/abs/2407.10805)
- **StructGPT (Jiang et al., 2023; commonly cited as EMNLP 2023):** a general framework for LLM reasoning over structured data (KGs, tables, databases) through iterative interface calls — [arXiv 2305.09645](https://arxiv.org/abs/2305.09645); [README](https://github.com/RUCAIBox/StructGPT)
- **Graph-ToolFormer (Zhang, 2023 [PP]):** teaches an LLM to emit **API calls to external graph-reasoning tools and models**, with training data built by prompting ChatGPT. It was slightly weaker on generating KG-reasoning API calls — [arXiv 2304.11116](https://arxiv.org/abs/2304.11116); [README](https://github.com/jwzhanggy/Graph_Toolformer)
- **Graph-CoT (Jin et al., Findings of ACL 2024 [PR]):** each iteration does LLM reasoning, then an LLM–graph interaction, then graph execution, using functions such as `RetrieveNode(text)` and `NeighborCheck(nodeID, type)` (some versions add `NodeFeature` and `NodeDegree`). GRBench has 1,740 QA pairs over 10 domain graphs. With GPT-3.5 it reports GPT4score 36.29 vs 23.09 for the best baseline (secondary) — [ACL Anthology PDF](https://aclanthology.org/2024.findings-acl.11.pdf); [arXiv 2404.07103](https://arxiv.org/abs/2404.07103); [repo](https://github.com/PeterGriffinJin/Graph-CoT). GraphDancer (arXiv 2602.02518 [PP]) trains LLMs to explore graphs with these actions and transfers to unseen GRBench domains — [arXiv](https://arxiv.org/pdf/2602.02518)
- **ProGraph — "Can LLMs Analyze Graphs like Professionals?" (NeurIPS 2024 D&B [PR]):** 512 problems that require *code* written against graph libraries. Best model about **36%**. Document retrieval (closed models) or fine-tuning on LLM4Graph code (open models) adds **11–32 points absolute**. Open models went from 12–24% to 45–47% after tuning — [NeurIPS 2024 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/ff417c3993894694e88ffc4d3f53d28b-Paper-Datasets_and_Benchmarks_Track.pdf); [repo](https://github.com/BUPT-GAMMA/ProGraph); [slides](https://neurips.cc/media/neurips-2024/Slides/97519_mwijEee.pdf)
- **GraphAgent-Reasoner (arXiv 2410.05130 [PP]):** decomposes problems into **node-centric sub-tasks spread across many agents**, following distributed graph computation. Near-perfect on polynomial-time GraphInstruct tasks. Scales to **1,000 nodes** and failed on only 2 of the 1,000-node test samples, while **fine-tuned GraphWiz could not handle 500+ nodes because of context limits** — [arXiv 2410.05130](https://arxiv.org/pdf/2410.05130)
- **GraphChain (NeurIPS 2025 [PR]):** the LLM chains graph tools from 45 NetworkX APIs. *Progressive Graph Distillation* (RL) learns tool sequences that trade task relevance against compression of intermediate state, so results fit in context. *Structure-aware Test-Time Adaptation* uses a lightweight adapter on a frozen LLM — [NeurIPS 2025 abstract](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0afacd536648265c03a0bd86dabb0e1a-Abstract-Conference.html); [arXiv 2511.00457](https://arxiv.org/html/2511.00457v1)
- **GraphCogent (arXiv 2508.12379 [PP]):** a multi-agent design modeled on working memory, with Sensory (subgraph sampling and format standardization), Buffer (multi-format index) and Execution (tool calling plus generation) modules. The Graph4real benchmark has 21 tasks at **up to 10× the graph scale of prior benchmarks**. A Llama-3.1-8B-based GraphCogent reportedly beats DeepSeek-R1 (671B) by 50%. A secondary source instead says +20% accuracy with −80% tokens on in-toolset tasks and −30% on out-of-toolset tasks; check the paper — [arXiv 2508.12379](https://arxiv.org/pdf/2508.12379); [alphaxiv overview](https://www.alphaxiv.org/overview/2508.12379/metadatav2)
- **GraphDC (arXiv 2605.06671 [PP]):** divide-and-conquer agents (partition, local agent per subgraph, master agent merges boundary information). Gains are largest on the biggest graphs tested, **80–100 nodes** — [arXiv 2605.06671](https://arxiv.org/abs/2605.06671)
- **Code-based vs text-based at scale:** GraphGym at 10/100/1k/10k nodes shows code beats text at every size and file-loaded graphs beat inline ones ([arXiv 2608.12391](https://arxiv.org/html/2608.12391)). GraphSkill's ComplexGraph-L at **5k/10k nodes** shows zero-shot coding at 64.4 (DeepSeek) and 56.7 (Llama-70B), with text-only "much lower" ([arXiv 2603.06620](https://arxiv.org/pdf/2603.06620))
- Related 2026 work on LLM graph analytics: LoReC ([arXiv 2604.17897](https://arxiv.org/html/2604.17897v1)), GraphSeek ([arXiv 2602.11052](https://arxiv.org/pdf/2602.11052)), and GraphRAG-Bench ([arXiv 2506.02404](https://arxiv.org/pdf/2506.02404))

### Inferences
- **Strongest transfer to netlists (my judgment).** Netlists already sit in mature graph databases and engines (EDA databases, ATPG and fault simulators, SAT solvers). An LLM that calls tools such as `fanin_cone(net, depth)`, `fanout_cone`, `level(net)`, `scoap(net)`, `simulate(pattern)`, `atpg(fault)` and `implication(net=val)` follows the **GraphChain / Graph-CoT / ProGraph** pattern. It keeps **exact per-net addressability at 10^7 nodes**, because the LLM only sees names and small results. The evidence (GraphAgent-Reasoner at 1k nodes; code at 10k; G-Retriever −99% tokens) consistently says this route scales where full context fails.
- **Retrieval unit for netlists:** G-Retriever's PCST is designed for semantic relevance. For fault analysis the right retrieval unit is structural and deterministic: the fan-in cone of the fault site (for justification) plus its fan-out cone to observable outputs (for propagation), bounded by depth or by gate count. This "cone-RAG" is a direct analogue of G-Retriever. Its textualization, an incident-style gate list, fits in a few thousand tokens for cones of about 10^2–10^3 gates.
- **GraphRAG-style community summaries:** **low–medium.** Hierarchical module or partition summaries could support design-level Q&A ("what does this block do"). They are lossy, though, and win rates were judged by LLMs, so they are unsuitable for exact fault-level tasks.
- **Multi-agent decomposition (GraphAgent-Reasoner, GraphDC):** maps onto partition-based ATPG and fault simulation, but message passing between agents is far slower than native engines. Agents should orchestrate tools, not reimplement them.
- **Risk:** ProGraph's 36% baseline shows that LLMs do not reliably write correct graph code out of the box. Domain API documentation (retrieval) and fine-tuning on tool traces gave +11–32 points, which suggests an EDA tool API needs its own docs or fine-tuning corpus.

### Gaps
- Think-on-Graph and StructGPT benchmark numbers were not retrieved, and KG sizes reached per query (e.g., Freebase scale) were not confirmed this session.
- No general study found that measures tool-augmented LLM accuracy on graphs above 10^5 nodes. The largest seen is 10k nodes (GraphSkill, GraphGym), although the tools themselves obviously scale further.
- GraphCogent's headline numbers conflict between sources.
- The independent GraphRAG bias audit could not be verified.

---

## Key Question 6 — Long-context and subquadratic sequence models on graphs (Graph-Mamba, GMN, SGFormer, NodeFormer, Exphormer, Polynormer): scale and accuracy

### Takeaway
Linear-time graph transformers and SSMs scale to **millions of nodes and up to 0.1B nodes**: SGFormer on ogbn-papers100M (111M nodes, one GPU), NodeFormer at 2M, Polynormer on ogbn-products (2.45M) and pokec (1.63M). Mamba-style models (Graph-Mamba, GMN) show linear memory scaling and gains on long-range benchmarks. **None of these is an LLM.** They are task-specific graph encoders, which makes them the realistic "front end" for a hybrid (encoder → per-net embeddings → retrieval or soft tokens → LLM). They are not a way for an LLM to read a netlist directly.

### Cited Findings
- **SGFormer (Wu et al., NeurIPS 2023 [PR]):** **one** linear global attention layer plus a GCN, with no positional encodings, no pre-processing and no extra losses. Scales to **ogbn-papers100M (0.1B nodes) on a single GPU**, "two orders of magnitude larger than the largest demonstration of NodeFormer". Up to **37×/141× training/inference speedup** over scalable Transformers on medium graphs — [arXiv 2306.10759](https://arxiv.org/abs/2306.10759); [NeurIPS 2023 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/cc57fac10eacadb3b72a907ac48f9a98-Paper-Conference.pdf); extended as "Single-Layer Graph Transformers with Approximation-Free Linear Complexity" ([arXiv 2409.09007](https://arxiv.org/pdf/2409.09007)); ogbn-papers100M has 111M nodes ([OGB paper](https://arxiv.org/pdf/2005.00687))
- **NodeFormer (Wu et al., NeurIPS 2022 [PR]):** a kernelized Gumbel-Softmax operator learns latent graphs per layer with **linear complexity**. "The largest demonstration … is the graph with **2M nodes**" — [arXiv 2306.08385](https://arxiv.org/pdf/2306.08385); [repo](https://github.com/qitianwu/NodeFormer)
- **Exphormer (Shirzad et al., ICML 2023 [PR]):** sparse attention built from **real edges + expander graphs + virtual nodes**, linear in graph size. It outperforms BigBird and Performer and sometimes dense attention. It scales to larger graphs than earlier graph transformers (evaluation includes ogbn-arxiv, about 169k nodes) — [PMLR PDF](https://proceedings.mlr.press/v202/shirzad23a/shirzad23a.pdf); [README](https://github.com/hamed1375/Exphormer); [arXiv 2303.06147](https://arxiv.org/abs/2303.06147)
- **Polynormer (Deng et al., arXiv 2403.01232; ICLR 2024 venue not confirmed this session):** linear local-to-global attention that is expressive for equivariant polynomials. Accuracy: **ogbn-products (2,449,029 nodes, 61.9M edges) 83.82%**, **pokec (1,632,803 nodes) 86.10%**, ogbn-arxiv 73.46%. Products and pokec need **mini-batch training by random partitioning** because they do not fit in GPU memory — [arXiv 2403.01232](https://arxiv.org/abs/2403.01232); [README](https://github.com/cornell-zhang/Polynormer); [large_graph_exp](https://github.com/cornell-zhang/Polynormer/tree/master/large_graph_exp)
- **Graph-Mamba (Wang, Tsepa, Ma, Wang; arXiv 2402.00789 [PP]):** a Mamba block with **input-dependent node selection** plus graph-centric **node prioritization and permutation** to turn the graph into a sequence. On 10 benchmarks it beats state of the art on long-range graph prediction "with a fraction of the computational cost in both FLOPs and GPU memory" — [README](https://github.com/bowang-lab/Graph-Mamba); [arXiv 2402.00789](https://arxiv.org/abs/2402.00789)
- **Graph Mamba Networks, GMN (Behrouz & Hashemi, KDD 2024 [PR]):** selective SSMs over hierarchical neighborhood sequences, linear time, with significantly better memory efficiency than quadratic graph transformers on **ogbn-arxiv and MalNet-Tiny** — [arXiv 2402.08678](https://arxiv.org/abs/2402.08678); [Emergent Mind (secondary)](https://www.emergentmind.com/papers/2402.08678)
- **Comparison point:** on graph-level tasks the 2026 GraphTokenizer (BPE + plain BERT/GTE) beats GraphMamba and Exphormer on most of 9 reported datasets (e.g., molhiv 87.4 vs 81.2 vs 82.3). So off-the-shelf sequence models on a good serialization are competitive with specialized linear graph transformers at small graph scale — [GraphTokenizer README](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers)
- "Classic GNNs are strong baselines" (NeurIPS 2024 D&B): tuned classic GNNs rival graph transformers on node classification, a caution against assuming global attention is necessary — [NeurIPS 2024 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/b10ed15ff1aa864f1be3a75f1ffc021b-Paper-Datasets_and_Benchmarks_Track.pdf)
- Related 2025: GraphTARIF (linear graph transformer with augmented rank and better focus) — [arXiv 2510.10631](https://arxiv.org/html/2510.10631v2); survey of graph transformers — [arXiv 2502.16533](https://arxiv.org/pdf/2502.16533); OpenGT benchmark — [arXiv 2506.04765](https://arxiv.org/pdf/2506.04765)

### Inferences
- **Scale fit (my judgment):** SGFormer, Polynormer and NodeFormer have run at 10^6–10^8 nodes, which covers the 10^4–10^7-gate netlist range in raw size. They were shown on undirected, homophilous or heterophilous node classification with rich features. Netlists are directed, text-free (gate type only) and have functionally rigid (logical) semantics, so accuracy transfer is unproven.
- **Directionality gap:** all of these use undirected or symmetric attention by default. For netlists they should be combined with KQ4's directed encodings (level PE, reachability masks, MagLap) or with DAG-ordered recurrence. Mamba's sequential scan over a **topological or level order** is a natural fit for a DAG, since signal flow is causal. Graph-Mamba's "node prioritization" ordering would then be replaced by the circuit's levelization, but this has not been evaluated in general literature.
- **Role in an LLM pipeline:** use one of these models (or a circuit GNN from the cross-referenced DeepGate/NetTAG line) as a **netlist-scale encoder that produces one embedding per net**. The LLM then reads (a) retrieved per-net embeddings as soft tokens (LLaGA-like) next to (b) textual net names and gate lines, and calls (c) tools for exact work. This combines the only families with scale evidence: linear encoders (KQ6) and retrieval/tools (KQ5).

#### Cross-method applicability matrix for gate-level netlists (my synthesis)

| Family / method | Largest scale demonstrated | Tokens to LLM | Keeps per-net addressability? | Handles DAG direction / types? | Netlist verdict |
|---|---|---|---|---|---|
| Text serialization (GraphQA, NLGraph, GraphArena, GraphOmni) | ≤50 nodes (GraphArena); ≤160 (GrAlgoBench); text collapses at 5k–10k | ~10–20 per gate (edge list) | Yes (names) | Yes, if explicit (`g=NAND(a,b)`) | Local cones only |
| GraphToken / pooled soft prompt | GraphQA-size graphs | few per graph | **No** | Depends on GNN | Low |
| GraphLLM prefix | small synthetic graphs | prefix (−96.45% vs text) | No | Graph transformer | Low–medium |
| LLaGA / GraphGPT / TEA-GLM / InstructGLM | ego-subgraphs of ogbn-products (2.4M) | ~1 per sampled node | Center node only | Undirected templates | Medium (node-local) |
| GraphTranslator (Q-Former) | Taobao-scale graph via embeddings (scale not verified) | fixed query tokens | No | Needs node text | Low–medium |
| GOFA (GNN inside LLM) | TAG datasets | K memory tokens per node/edge | Yes, but costly | Text-dependent | Low at scale |
| G-Retriever (retrieve + soft prompt + text) | 1,371 → 18 nodes; 100k → 610 tokens | ~600 tokens | Yes (textualized) | Retrieval unit must be cones | Medium–high (as "cone-RAG") |
| GQT (RVQ codes) | "large-scale" node benchmarks | ~3 codes per node | Needs separate ID | Encoder-dependent | Medium |
| GraphTokenizer (Feuler + BPE) | small graph-level datasets | Θ(\|E\|) before merges | Yes (reversible) | Labels yes; direction needs adaptation | Medium–high (promising) |
| GraphsGPT K-words | molecules (100M pre-training) | 1–8 per graph | No | n/a | Low |
| NT-LLM anchors | TAG benchmarks | per node | Partially | Should use directed distance | Medium |
| DAG attention (depth PE + reachability) | code ASTs / citation DAGs | n/a (encoder) | Yes | **Native** | High (as encoder / PE) |
| MagLap / multi-q / PathLapPE | sorting networks, code graphs | n/a | Yes | Directed | Medium (eigendecomposition cost) |
| Tool / agentic (GraphChain, Graph-CoT, ProGraph, GAR) | 1k (GAR), 10k (GraphSkill/GraphGym) for LLM eval; tools unbounded | small results only | **Yes (exact)** | Yes (via tools) | **Highest** |
| SGFormer / Polynormer / NodeFormer / GMN / Graph-Mamba | up to 0.1B nodes (SGFormer) | n/a (not LLMs) | Per-node embeddings | Undirected by default | High as front-end encoder |

### Gaps
- SGFormer's ogbn-papers100M accuracy, NodeFormer's 2M-node dataset name (believed Amazon2M) and Exphormer's exact largest graph were not confirmed.
- Graph-Mamba (Wang et al.) and GMN exact FLOPs and memory numbers and largest graph sizes were not retrieved. GMN is shown only up to ogbn-arxiv scale in what was found.
- No general-domain study applies these linear graph transformers or SSMs to directed acyclic, typed-gate graphs, or pairs them with an LLM at million-node scale.
- No published "graph-LLM" system was found that couples a million-node linear graph encoder with an LLM through per-node soft tokens and retrieval. This combination is my proposal, not established practice.
