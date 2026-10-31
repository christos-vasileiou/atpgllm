# Cone-bounded encoding tames variable netlist size

No published system has a language model read a large synthesized gate-level netlist whole, and the evidence says none should try. Every method with real feasibility evidence limits what the model sees by circuit structure, not by design size. The best-supported recipe has three layers. First, **decompose by cone**: fault, register or fan-in cones. This is formally justified because a stuck-at test depends only on the fault's sensitization cone, and because chain-of-thought theorems prove a constant-depth transformer can simulate any size-T circuit in T steps and can execute DPLL-style search. Second, **optionally compress each cone with a circuit graph encoder**, whose output reaches the LLM as soft tokens or as predictions written into the prompt. GenEDA used this to lift GPT-4o and DeepSeek-V3 from under 20% to 60–69% functional pass on netlist-to-RTL, and DeepGate4-class encoders run at 400K–1.6M gates with flat memory. Third, **give the LLM tool access to the full netlist and verify every answer with a classical engine**; every ML-for-test result that reached industrial scale works this way. Of the systems named in the question, only GenEDA ingests gate-level netlists. LLM4Netlist generates small netlists from text. Auto-SPICE (now Masala-CHAI), Netlistify and LogicCraft work on analog schematics or transistor-level cells, and Netlistify uses no LLM at all. Raw text does not scale. Structural Verilog measured 50–59 tokens per gate, so a 22K-gate design overflows a 1M-token window. LLM reasoning over serialized graphs also collapses within the first hundred or so nodes, well before the window fills. No published LLM performs gate-level ATPG with reported fault coverage, so the atpgllm project would be first. For it, I recommend this design: serialize each fault's cone as compact, levelized gate lines with SCOAP boundary annotations; add a per-cone graph prefix that never carries fault-site identity alone; fall back to tool calls when a cone overflows; and verify every pattern by fault simulation, with PODEM/SAT as the fallback.

## Only GenEDA among the named systems reads gate-level netlists

The five named systems address three different problems, and only one addresses the problem in the question.

**GenEDA** (HKUST, ICCAD 2025) is the only one in which an LLM reasons over existing post-synthesis netlists with an encoding built for scale ([arXiv](https://arxiv.org/abs/2504.09485); [ICCAD DOI listing](https://www.researchgate.net/publication/397819047_GenEDA_Towards_Generative_Netlist_Functional_Reasoning_via_Cross-Modal_Circuit_Encoder-Decoder_Alignment)). It offers two routes into the LLM.

- **Trainable decoders** (DeepSeek-Coder 1B and 7B): an MLP "modality connector" projects NetTAG graph embeddings into the LLM's latent space as soft tokens. The connector is pre-trained first, then fine-tuned end to end with the decoder.
- **Frozen commercial LLMs**: the encoder's gate-function predictions, about 97% accurate per gate, are written into the netlist text as annotations ([arXiv HTML](https://arxiv.org/html/2504.09485)).

The second route matters most here. **Without the annotations, GPT-4o and DeepSeek-V3 stay below 20% functional pass on arithmetic RTL reconstruction** and emit only bit-level operations. With them, DeepSeek-V3 reaches 96–97% syntax pass and **60% or 69% functional pass**; the paper's versions disagree on which figure applies. Swapping in a weaker encoder drops functional pass from 69% to 40% ([arXiv PDF](https://arxiv.org/pdf/2504.09485)).

GenEDA's data shows the scale actually exercised: ITC99 circuits average **1,025 gates and 15K tokens**, OpenCores average 173 gates and 9K tokens, and a Chipyard set averages 24K tokens ([arXiv HTML](https://arxiv.org/html/2504.09485)).

The other four systems do not touch the size problem.

- **LLM4Netlist** (IEEE JETCAS 2025) works in the opposite direction. It generates netlist text, reportedly BLIF, from natural-language descriptions. Its pipeline is step-based prompting, a fine-tuned LLM, a code-confidence estimator and a feedback loop ([UNNC portal](https://research.nottingham.edu.cn/en/publications/llm4netlist-llm-enabled-step-based-netlist-generation-from-natura/)). A later paper notes it "still faces challenges in scaling to highly complex or large-scale circuits" ([ArtNet](https://arxiv.org/html/2510.13582v1)).
- **Auto-SPICE**, renamed **Masala-CHAI** in later versions, is a GPT-4o pipeline that turns analog schematic images into SPICE netlists. It covers 7,500 textbook schematics and adds 46% Pass@1 when used to fine-tune AnalogCoder ([arXiv PDF](https://arxiv.org/pdf/2411.14299)).
- **Netlistify** (MLCAD 2025 Best Artifact) also converts schematics, using YOLOv8, ResNet and a modified DETR. **No language model is involved** ([GitHub](https://github.com/NYCU-AI-EDA/Netlistify)).
- **LogicCraft** (AICAS 2025) has an LLM translate a small Boolean truth table into a custom pull-up network for a standard cell ([ECNU portal](https://pure.ecnu.edu.cn/en/publications/logiccraft-llm-assisted-optimization-of-netlist-to-layout-for-com/)).

The related systems are more informative.

- **NetTAG** (DAC 2025) is the encoder behind GenEDA. It treats the netlist as a text-attributed graph. Each gate's symbolic expression, extracted with PySMT, is embedded by an LLM-based text encoder ("ExprLLM", **8K-token maximum input**), and a graph transformer ("TAGFormer") then refines the gate embeddings. Large sequential designs are first split into **register cones**: each register is traced back through its driving logic to other registers ([arXiv PDF](https://arxiv.org/pdf/2504.09260); [GitHub](https://github.com/hkust-zhiyao/NetTAG)).
- **BRIDGES** projects pooled NetlistGNN embeddings into an LLM's input space. It was trained on 500K graph instances and 1.5B tokens, and improves 2–10× on retrieval, type prediction and description perplexity while adding under 1% to model weights. It also gives the clearest warning about raw text: **GPT-4o and Claude 3.5 Sonnet failed to predict basic properties from netlist files "particularly as context length increases"** ([BRIDGES](https://www.alphaxiv.org/abs/2504.05180)).
- **Gate-Breaker** gives raw synthesized netlists directly to the LLM. OpenAI o3-mini reaches **75.1% netlist-to-RTL success on 156 VerilogEval-scale circuits**, falling to 53.4% on the most complex quartile ([IEEE Xplore](https://ieeexplore.ieee.org/document/11304682/)).
- **LLSM** (ASP-DAC 2025) splits circuits into modules and summarizes each with an LLM because "the text length of some circuits is far larger than the maximum input an LLM can take" ([ASP-DAC PDF](https://www.aspdac.com/aspdac2025/archive/pdf/6E-1.pdf)).

| System | Domain and task | How the netlist reaches the model | Largest scale shown | Status |
|---|---|---|---|---|
| GenEDA | Digital gate-level; netlist to spec/RTL | NetTAG embeddings as soft tokens via MLP, or encoder predictions written into text for frozen LLMs | ITC99 average ~1K gates / 15K tokens; Chipyard average 24K tokens | ICCAD 2025 |
| LLM4Netlist | Digital; text to netlist | LLM emits netlist text (reported BLIF) | Small circuits; scaling flagged as weak | JETCAS 2025 |
| Auto-SPICE / Masala-CHAI | Analog; schematic image to SPICE | GPT-4o reads images | Single textbook schematics (7,500 total) | arXiv |
| Netlistify | Analog; schematic image to netlist | No LLM (YOLOv8, ResNet, DETR) | 100K schematic images | MLCAD 2025 |
| LogicCraft | Transistor-level cells | LLM reads a small truth table | One cell at a time | AICAS 2025 |
| NetTAG | Digital gate-level encoder | Per-gate expression text (8K tokens) plus graph transformer | Register cones of synthesized designs | DAC 2025 |
| BRIDGES | Digital netlist graphs | Pooled GNN embedding through a projector | 500K graph instances | arXiv |
| CircuitFusion | RTL, graph and summary encoder | Register-cone sub-circuits; 32K-token code encoder | Register cones | ICLR 2025 |
| Gate-Breaker | Gate-level to RTL | Raw netlist text | 156 VerilogEval-scale circuits | IPCCC 2025 |
| NtNDet | Gate-level Trojan detection | Netlist rewritten as natural-language-like text for BERT/GPT-2/T5 | Trust-Hub scale | ESWA 2025 |
| Circuit Transformer / ctrw | AIG optimization | Depth-first gate trajectory with an equivalence-masked decoder; applied to fanout-free windows | 8-input, 2-output circuits | ICLR 2025 / IWLS 2024 |

Sources for rows not cited above: CircuitFusion ([arXiv HTML](https://arxiv.org/html/2505.02168)), NtNDet ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S095741742500288X)), Circuit Transformer ([arXiv](https://arxiv.org/abs/2403.13838)) and ctrw ([arXiv](https://arxiv.org/abs/2406.04699)).

The pattern across the table is consistent. **Every system that handles variable size by design cuts the netlist at structural boundaries**: register cones in NetTAG, CircuitFusion and GenEDA, and fanout-free windows in ctrw. Every raw-text system reports results only on benchmark-scale circuits. No LLM-in-the-loop system demonstrates end-to-end reasoning on one netlist of tens of thousands of gates. GenEDA's exact soft-token budget per netlist, and how it combines many cone embeddings for a whole design, are not stated in the accessible sources.

## A 22K-gate netlist already overflows a million-token window

Token cost explains why raw ingestion fails at ATPG scale. The research team measured it directly. EPFL benchmarks were synthesized with Yosys, mapped to Nangate45, and tokenized with the Llama-3 128K-vocabulary tokenizer ([EPFL benchmarks](https://github.com/lsils/benchmarks); [yowasp-yosys](https://pypi.org/project/yowasp-yosys/); [Llama-3 tokenizer](https://github.com/meta-llama/llama-models/tree/main/models/llama3)). These are the team's own measurements, not published figures.

| Design | Cells | Raw Yosys Verilog (tokens/gate) | Compact one-line Verilog | `.bench`-style `n = NAND(a,b)` | ASCII AIGER (tokens per AND) |
|---|---|---|---|---|---|
| adder | 645 | 58.6 | 24.6 | 12.5 | 7.9 |
| i2c | 639 | 52.0 | 27.6 | 13.5 | 7.5 |
| arbiter | 6,592 | 53.2 | 31.8 | 17.0 | 8.6 |
| voter | 5,985 | 50.0 | 29.0 | 15.6 | 8.9 |
| mem_ctrl | 21,983 | 51.7 | 32.6 | 17.4 | 8.8 |

The mem_ctrl netlist, about 22K cells, came to **1,137,501 tokens as raw Yosys output**.

- **Leaner formats help, but cost stays linear in gate count.** `.bench`-style lines save roughly 3×, yet per-gate cost still rises slowly with design size because longer net IDs need more number tokens.
- **The tokenizer matters.** Mistral's Tekken tokenizer splits digits one per token and costs 1.5–2× more on netlists ([mistral-common](https://pypi.org/project/mistral-common/)).
- **The planning arithmetic is unforgiving.** At about 20 tokens per gate, a 128K window holds about 6K gates. A 1M-gate design needs about 20M tokens even in the leanest text, and about 50M as raw Verilog.
- **GenEDA's statistics agree.** Its averages imply roughly 15–52 tokens per gate ([arXiv HTML](https://arxiv.org/html/2504.09485)).

Standard test benchmarks span this whole range:

- ISCAS-85 c7552 has 3,512 gates ([CVUT](https://ddd.fit.cvut.cz/www/prj/Benchmarks/ISCAS_Desc/c7552.html)).
- ITC99 b19 has about 98K gates and 6.6K flip-flops ([ITC'99 notes](https://www.cerc.utexas.edu/itc99-benchmarks/bendoc1.html)).
- EPFL hypotenuse has **214,335 AND nodes and 24,801 logic levels** ([IWLS'15](https://si2.epfl.ch/%7Edemichel/publications/archive/2015/IWLS15.pdf)).
- Industrial testability graphs exceed 1M nodes ([Ma et al., DAC 2019](https://www.cse.cuhk.edu.hk/~byu/papers/C85-DAC2019-GCN-Test.pdf)).

Information theory sets a floor that no clever format can beat. The number of fan-in-2 circuits with s gates is at most 2^(2s log s + 2s) ([TIFR notes](https://www.tcs.tifr.res.in/~ramprasad/courses/2026-complexity/all_lectures.pdf)). Shannon's counting argument implies a matching worst-case lower bound ([Trevisan](https://theory.stanford.edu/~trevisan/cs278-08/lecture02.pdf)). Describing an s-gate circuit therefore takes **Θ(s log s) bits**. That has two consequences:

- **Even a perfect serialization is huge.** A 10^6-gate netlist needs about 42 Mbit. At no more than about 17 bits per token from a 128K vocabulary, that is roughly 2.5M tokens.
- **A single embedding cannot hold a large netlist.** One 4,096-dimensional fp16 vector holds 65,536 bits, enough to index circuits of only about 2,600 gates losslessly in the worst case.

So **any fixed-size whole-netlist embedding is necessarily lossy beyond a few thousand gates**. It can be faithful only to a task-specific question, such as whether a given pattern detects a given fault.

Accuracy, not window length, is the binding constraint, and it fails first.

- **NP-hard tasks on tiny graphs (GraphArena).** On the "large" NP-complete problems, whose graphs have only 10–30 nodes, GPT-4o scores **6.3%** and Claude 3.5 Sonnet 7.2%, while DeepSeek-R1 reaches 43.1% ([GraphArena leaderboard](https://github.com/squareRoot3/GraphArena); [size ranges](https://arxiv.org/html/2407.00379v2)).
- **Reasoning models degrade with size (GrAlgoBench).** Qwen3-32B falls from 87% on 31–50-node graphs to **47% on 121–160-node graphs** ([arXiv](https://arxiv.org/html/2602.06319v1)).
- **Effective context is shorter than advertised (RULER).** Only about half of 17 long-context models held acceptable performance at 32K tokens, and simple needle-retrieval tests hide large drops on multi-hop tracing, the closest analogue to following a netlist path ([RULER](https://github.com/nvidia/RULER)).
- **Format choice matters, but only so much.** It shifts graph-reasoning performance by **4.8% to 61.8%** depending on the task, and incident-style lists, a node with its neighbors, beat edge lists on local queries ([Talk like a Graph](https://arxiv.org/abs/2310.04560)). A gate line `g42 = NAND(g17, g23)` is exactly an incident encoding keyed on fan-in.
- **Code beats text once graphs grow.** Code-based reasoning outperforms text-based reasoning at every size from 10 to 10,000 nodes ([GraphGym](https://arxiv.org/html/2608.12391)). On 5K–10K-node graphs, zero-shot coding reached 64.4 (DeepSeek) where text-only methods scored far lower ([GraphSkill](https://arxiv.org/pdf/2603.06620)).

Structural abstraction buys a constant factor, not a change in scaling. In the same local measurements, standard cells counted **0.53–0.63×** the AIG AND-node count, and 6-LUTs counted 0.16–0.27×. Each LUT, however, needs a 64-bit truth table, so most of the node saving disappears in tokens. NPN canonicalization keeps the function vocabulary small: 65,536 four-input functions fall into **222 classes** ([Soeken et al., ISMVL 2017](https://msoeken.github.io/papers/2017_ismvl.pdf)). No published LLM tokenizes netlists by NPN class. For ATPG, abstraction has a further cost: stuck-at faults live on cell pins, so any AIG or LUT re-encoding changes the fault universe and needs an explicit back-mapping.

## Theory proves step-wise, cone-local processing and rules out one-shot encodings

The formal literature gives two kinds of result: limits on single-pass models, and positive constructions for step-wise ones.

**The limit.** Transformers with log precision lie in uniform TC0 ([Merrill and Sabharwal, TACL 2023](https://arxiv.org/abs/2207.00729v4)). Circuit evaluation is P-complete. A fixed-depth transformer therefore cannot faithfully evaluate arbitrary deep netlists in one forward pass unless TC0 = P. Simulating a single fault is circuit evaluation, so the same bound applies to fault simulation.

**The positive results.** These are the strongest formal proofs of feasibility in the whole literature.

- **Simulation.** Constant-depth transformers given **T chain-of-thought steps can compute any function computable by a Boolean circuit of size T** ([Li, Liu, Zhou and Ma, ICLR 2024](https://arxiv.org/abs/2402.12875)). Polynomial-length chain of thought gives exactly the class P ([Merrill and Sabharwal, ICLR 2024](https://arxiv.org/pdf/2310.07923.pdf)). An LLM that walks a cone in topological order and writes down each gate's value is therefore provably sufficient, at about one reasoning token per gate.
- **Search.** A decoder-only transformer **can decide 3-SAT through a chain of thought that is trace-equivalent to DPLL** ([Pan et al., ICML 2025](https://proceedings.mlr.press/v267/pan25d.html)). Single-fault ATPG reduces to Circuit-SAT on a fault miter, and PODEM is a DPLL-like search over primary-input assignments. The same construction is therefore a feasibility proof for an LLM that runs ATPG with an explicit backtracking trace. Two qualifications apply: ATPG is NP-complete, so trace length is exponential in the worst case, and Pan et al.'s learned models generalized only to a limited extent to problems longer than those seen in training.

The structural results explain which subproblems need care.

- **Reachability needs depth.** Logarithmic depth is necessary and sufficient for graph connectivity ([Sanford et al., NeurIPS 2024](https://arxiv.org/abs/2405.18512v1)). Questions like "is there a sensitizable path from the fault to an output" need model depth that grows with netlist size, or chain of thought, or a tool.
- **Message passing has hard limits.** Some message-passing GNN tasks are impossible unless depth × width exceeds a polynomial in graph size ([Loukas, ICLR 2020](https://arxiv.org/pdf/1907.03199)). Exponentially growing receptive fields squeezed into fixed-size vectors cause over-squashing ([Topping et al., ICLR 2022](https://arxiv.org/pdf/2111.14522)). Standard GNNs are no more discriminative than the 1-WL test ([Xu et al., ICLR 2019](https://arxiv.org/abs/1810.00826)). The last point means two fault sites with identical local gate types but different reconvergent fan-out can receive identical embeddings unless the encoder adds unique IDs or simulation-derived features.
- **Small-to-large transfer is not guaranteed.** GNNs can fit small graphs and fail on large ones when local structure shifts with size ([Yehudai et al., ICML 2021](https://proceedings.mlr.press/v139/yehudai21a.html)). Looped transformers extrapolate only on tasks with a known iterative solution ([Fan et al., ICLR 2025](https://arxiv.org/html/2409.15647v2)).

The decisive argument for cone decomposition comes from test theory, not ML. A stuck-at fault test needs three regions of the netlist:

1. the fault site's fan-in cone, for activation;
2. its fan-out cone up to observable outputs, for propagation;
3. the fan-in cones of side inputs along the sensitized paths.

Under full scan, flip-flops become pseudo-primary inputs and outputs, so every cone ends at a scan cell. **The information an ATPG step needs therefore scales with cone size, not design size.** Real synthesized netlists reuse library cells and repeated structures, which adds practical compression on top of that. Cones can still be large. The 24,801-level EPFL hypotenuse shows that deep arithmetic produces cones far beyond any window. And **no published distribution of fault-cone sizes on ISCAS, ITC99 or EPFL exists**, so the per-fault context budget is currently unknown.

Taken together, these results validate one design and invalidate another. The validated design walks a bounded cone step by step, keeping intermediate state outside the model. The invalidated one predicts a test for a whole netlist from a single compressed encoding. Nothing in the literature supports that second design, and both the TC0 bound and the Θ(s log s) bound argue against it.

## Four encoding families carry real evidence past the window

### Structural decomposition is the best-proven size control

Partitioned graph encoders hold the strongest empirical scale record, and each handles the cut boundary explicitly.

- **DeepGate4** (ICLR 2025) splits AIGs into overlapping fixed-depth cones (reportedly depth 8, stride 6). It processes them level by level and initializes overlap nodes from cached "historical" embeddings. Trained on circuits averaging about 15K gates, it performs well at about 400K gates, and **its inference memory stays flat from 400K to 1.6M gates while GNN baselines grow linearly** ([arXiv v3](https://arxiv.org/html/2502.01681v3)). Its loss is 15.5% lower than the next-best model on ITC99 and 31.1% lower on EPFL ([arXiv](https://arxiv.org/abs/2502.01681)). The 1.6M-gate case is an EPFL circuit duplicated four times, not a distinct design.
- **GROOT** (ICCAD 2025) partitions with METIS and adds "edge re-growth" to restore the roughly 10% of edges that cross partitions. On a 1024-bit multiplier with **134M nodes** it reports 99.96% accuracy, and accuracy holds at 100% up to 16 partitions before degrading ([arXiv](https://arxiv.org/abs/2511.18297)).
- **PreRoutGNN** pads each partition with neighbor levels equal to the GNN depth, so no receptive field is lost. It reaches slack-prediction R² of 0.93 against 0.59 for the prior state of the art ([arXiv](https://arxiv.org/abs/2403.00012)).
- **DeepSeq3** partitions sequential circuits at flip-flops and links the pieces through a register-transfer "super-node" graph ([arXiv](https://arxiv.org/html/2608.28188)). This is the hierarchy that full-scan ATPG already implies.

Test-specific models confirm that cone-local learning transfers across sizes.

- **DETECTive** predicts stuck-at patterns from fault-centered cone embeddings. It reports **up to 95% pattern accuracy on realistic designs** and transfers to designs **29× larger** than those it was trained on ([PoliTo](https://iris.polito.it/handle/11583/2990442)).
- **InF-ATPG** uses fanout-free regions as its decision unit and reports 55.06% fewer backtracks than traditional ATPG ([arXiv](https://arxiv.org/abs/2512.00079)).

The gap is that **all of this evidence comes from GNNs and graph transformers**. No paper trains an LLM on netlist cones and measures how it generalizes to larger cones.

### Encoder-to-LLM alignment raises accuracy but pools away gate identity

Graph-encoder tokens give the largest measured accuracy gains for LLMs on netlists. GenEDA's jump from under 20% to 60–69% and BRIDGES' 2–10× gains are the circuit evidence. In general graph work, GraphToken reports gains of up to 73 percentage points on small GraphQA graphs ([arXiv](https://arxiv.org/abs/2402.05862)). G-Retriever offers a retrieval-first variant: it shrinks a WebQSP graph from **1,371 nodes and 100,627 tokens to 18 nodes and 610 tokens** before soft-prompting it and also writing it out as text ([NeurIPS 2024](https://arxiv.org/abs/2402.07630)).

Three cautions limit what these methods can carry.

- **They have never been shown at netlist scale.** No learned-token method has fed a whole graph of 10^4 or more nodes to an LLM. LLaGA reaches 2.4M-node graphs only by sampling small ego-subgraphs per query ([ICML 2024](https://proceedings.mlr.press/v235/chen24bh.html)).
- **LLMs make limited use of graph tokens.** A 2026 audit finds that graph-token LLMs "do not fully understand graph tokens" and lean heavily on accompanying text ([GTEval](https://arxiv.org/abs/2605.03514)). Another finds that soft-prompted LLMs given only node text match full graph-language models on synthetic reasoning ([Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1624/)). Netlist nodes carry almost no text beyond the gate type, so the graph tokens would have to carry the structure on their own, which is exactly where these models are weakest.
- **Pooling loses addressability.** Pooling to a fixed K tokens destroys the ability to name a specific net, and no circuit paper ablates token count against circuit size.

The safe reading is that soft graph tokens work as a **hint channel**, a learned summary of controllability, function and context. The fault site, side inputs and primary-input names must stay in addressable text. GenEDA's second route, writing encoder predictions into text, is the deployable form for frozen models.

### Tool access keeps exact identity at any size

Tool-driven access is the only family evaluated beyond the context window in industrial EDA. NVIDIA's Timing Analysis Agent works on industrial designs whose single reports hold **more than 16,000 timing paths**. Instead of ingesting the reports, it has expert agents write Python queries against a structured report database. It reaches **98% single-report and 90% multi-report pass rates** ([arXiv](https://arxiv.org/abs/2504.11502)).

General graph results point the same way.

- **Agents scale where fine-tuned text models stop.** GraphAgent-Reasoner splits problems into node-centric sub-tasks and handles 1,000-node graphs, while fine-tuned GraphWiz could not process graphs of 500+ nodes ([arXiv](https://arxiv.org/pdf/2410.05130)).
- **Tool code needs domain support.** ProGraph shows LLMs write correct graph-library code only about 36% of the time out of the box. Documentation retrieval or fine-tuning on tool traces adds 11–32 points ([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/ff417c3993894694e88ffc4d3f53d28b-Paper-Datasets_and_Benchmarks_Track.pdf)), so an EDA tool API needs its own documentation or training corpus.
- **No netlist-specific precedent exists.** No published agent queries a gate-level netlist graph for ATPG. The closest pattern is graph-database code retrieval for software repositories ([CodexGraph](https://aclanthology.org/2025.naacl-long.7.pdf)).

### Classical-engine hybrids are the only route proven at industrial test scale

Every ML-for-test result at industrial scale keeps a complete engine and gives the model a bounded, per-decision input. The model's input size is therefore independent of netlist size, and the engine guarantees fault coverage.

- **Testability analysis.** Ma et al. classify difficult-to-observe nodes on graphs with **more than 1M nodes and 2M edges**, starting from SCOAP features. Against a commercial tool they use 11% fewer observation points and 6% fewer test patterns at similar coverage ([DAC 2019](https://www.cse.cuhk.edu.hk/~byu/papers/C85-DAC2019-GCN-Test.pdf)).
- **Fault-parallel ATPG.** FPGNN-ATPG reports a **7.56× speedup and 14.13% fewer patterns at 100% fault coverage** on 10 industrial instances ([GLSVLSI 2023](https://dl.acm.org/doi/10.1145/3583781.3590250)).
- **Backtrace guidance.** HybMT reports a 56.6% speedup over a commercial ATPG tool ([arXiv](https://ar5iv.labs.arxiv.org/html/2207.11312)). Neural backtrace selection inside PODEM reduces backtracks when it combines several testability measures ([Roy, Millican and Agrawal, VLSID 2021](https://www.eng.auburn.edu/~agrawvd/TALKS/VLSID21/From_Proceedings.pdf)).

### Hierarchical summaries, new vocabularies and SSMs remain unproven for exact test tasks

Several appealing ideas lack the evidence needed for fault-level work.

- **Module summaries** are demonstrated for RTL understanding (LLSM; CROP, [arXiv](https://arxiv.org/abs/2507.02128)) and for corpus sensemaking. GraphRAG's root-level summaries use 2–3% of the tokens of its most detailed setting, but they are judged by LLMs ([arXiv](https://arxiv.org/html/2404.16130v2)). Summaries are lossy by construction, and no study measures how much controllability or reconvergence information survives them.
- **Reversible tokenizers** exist but are untested on netlists. GraphTokenizer serializes a graph reversibly as a frequency-guided Eulerian circuit and then applies BPE, setting state of the art on 14 small-graph datasets ([arXiv](https://arxiv.org/abs/2603.11099)). Netlists' tiny gate alphabet and repeated motifs suit this scheme, but no netlist result exists.
- **DAG-aware attention** is the best positional design for circuits, but only as an encoder. The DAG Transformer combines a depth positional encoding with a reachability attention mask and beat both plain transformers and DAG-specific GNNs on every dataset tested ([NeurIPS 2023](https://arxiv.org/abs/2210.13148)). In circuit terms, depth is logic level and the reachability mask is the union of fan-in and fan-out cones. It has never been injected into a pretrained LLM.
- **State-space models** have no circuit evidence at all. Searches found Mamba accelerators, but no SSM applied to netlists.

| Methodology | Formal support | Strongest empirical evidence | Exact fault-site identity | Verdict for ATPG |
|---|---|---|---|---|
| Per-fault or register-cone decomposition | Fault locality; T-step CoT simulates size-T circuits | DeepGate4 15K→400K gates, flat memory to 1.6M; DETECTive 29× transfer; GROOT 134M nodes | Yes | Primary unit |
| Classical engine plus learned local decisions | Completeness from the engine | Ma et al. >1M nodes; FPGNN 100% coverage | Yes | Mandatory safety net |
| Tool and agentic netlist access | Moves computation outside the TC0 limit | Timing agent 98%/90% on industrial designs; code reasoning at 10K nodes | Yes, exact | Overflow path |
| Compact text serialization | Sufficient when processed step by step | Format shifts performance 4.8–61.8%; collapse beyond ~120 nodes | Yes | Inside cones only |
| Graph encoder to soft tokens or annotated text | None; lossless fixed tokens capped at ~thousands of gates | GenEDA <20%→60–69%; BRIDGES 2–10× | No, when pooled | Auxiliary hint channel |
| Hierarchical LLM summaries | None | RTL-level only; LLM-judged | No | Not for exact test |
| Single fixed whole-netlist embedding | Ruled out beyond ~2,600 gates lossless | None at scale | No | Reject |
| NPN, BPE or RVQ vocabularies; SSMs | None | Small graphs, or nothing on netlists | Varies | Research bets |

## A cone-bounded, tool-backed, simulator-verified design fits atpgllm

The project's current graph path already follows the strongest pattern in the literature. Described in `docs/GRAPH_MODALITY_REVIEW_20260919.md`, it runs encoder, then Q-Former, then projector, then a soft prefix beside a text prompt, with the fault and interface names kept in text. The evidence above points to five changes that make that path size-independent.

**1. Make the per-fault sensitization cone the unit of both text and graph.** The cone is the fault site's fan-in, its fan-out to scan cells or primary outputs, and the side-input fan-ins. Serialize it in levelized topological order as `.bench`-style lines with short sequential net IDs and explicit level tags, for example `g42@L7 = NAND(g17, g23)`. This is the incident encoding that general graph benchmarks favor. It keeps reconvergence visible through explicit references, and it matches the depth-plus-reachability structure of the DAG Transformer. Annotate each gate with SCOAP values. Replace every signal cut at the budget boundary with a compact boundary token carrying its CC0, CC1 and CO. That is the per-gate feature set behind Ma et al. and InF-ATPG, and it applies GenEDA's predictions-into-text route.

At the measured 13–20 tokens per gate, an 8K-token cone budget covers roughly 400–600 gates and 32K covers roughly 1.6–2.5K. Keep standard cells, not AIG or LUT nodes, as the tokens, so the stuck-at fault universe stays on cell pins.

**2. Keep the graph prefix, but compute it per cone and treat it as a hint.** A DeepGate4-class encoder can embed the whole netlist once with flat memory. The Q-Former should then pool per-cone embeddings, plus at most a few global summary tokens. Fault-site identity must never depend on pooled tokens alone, because both the GTEval audit and the information bound say they cannot carry it. The repository's planned comparison of a pooling baseline against Q-Former query budgets of 16 to 128 is the right experiment. Add BRIDGES-style contrastive, matching and grounded-generation losses as an alignment variant. Embeddings conditioned on the fault cannot be cached per netlist; only fault-independent embeddings can.

**3. Route oversized cones to tools rather than truncating silently.** When a cone exceeds the budget, which will happen on deep arithmetic and wide-fanout nets, the LLM should call functions instead of reading the cone:

- `fanin_cone(net, depth)` and `fanout_cone(net, depth)`;
- `scoap(net)`;
- `implications(assignments)`, backed by the ATPG engine's implication procedure;
- `fault_sim(pattern, faults)`.

This is the Timing Analysis Agent's pattern. It sidesteps the reachability-depth limit and keeps exact net names. Expect to need documentation retrieval or fine-tuning on tool traces, given ProGraph's 36% baseline.

**4. Verify every proposed pattern by fault simulation and fall back to PODEM or SAT.** Single-pass prediction cannot be exact in general, and the best learned pattern predictor reaches at most about 95% on realistic designs. The project's simulator-backed reward already supplies this verification during GRPO. It should also run at inference, which turns an approximate model into a sound test generator.

**5. Run the experiments that would themselves be the feasibility proof.** Three are needed:

1. Measure the distribution of stuck-at cone sizes (gates and depth) across ISCAS-85/89, ITC99 b14–b22 and EPFL. This has never been published, and it fixes the context budget directly.
2. Report detection rate and fault coverage against cone size and depth, not design size.
3. Train on small cones and test on cones 10–30× larger, mirroring the DETECTive and DeepGate4 protocols. Compare text-only, text with graph prefix, and text with tools at matched compute.

These steps separate what is validated from what is novel. The validated parts are cone decomposition, classical verification, encoder-assisted LLMs on circuits averaging about 1K gates, and tool access on graphs of up to about 10K nodes. The novel parts are an LLM generating stuck-at patterns with measured coverage, cone-trained LLMs extrapolating to larger cones, and SCOAP boundary tokens as a lossy cut summary. Those three are the project's research contribution, and they need the controlled evidence above before they can be claimed.

## Conclusion

The "highly variable netlist size problem" is the wrong frame for ATPG. The quantity that varies and matters is the size and depth of each fault's cone, and nobody has measured its distribution on standard benchmarks. A longer context window does not address the real bottleneck either. LLM structural reasoning degrades at 10^2 nodes, the information-theoretic floor makes whole-design encodings either huge or lossy, and the TC0 bound forbids single-pass exactness. Investment should therefore go into step-wise and tool-backed processing of bounded cones, not into fitting bigger netlists into bigger windows.

The literature also implies a division of labor. The graph encoder carries scale (DeepGate4-class models already handle millions of gates). Addressable text carries identity. Classical engines carry correctness. The LLM's job is the local, heuristic part: choosing objectives, backtrace directions and candidate assignments within a cone. The opening for atpgllm is unusually clean. The first LLM to report verified stuck-at fault coverage, with accuracy plotted against cone size, would be a first-of-its-kind result, and the project's existing simulator-verified training loop already provides the measurement apparatus such a claim requires.
