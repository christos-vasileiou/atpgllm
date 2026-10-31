# Feasibility of Encoding Large Boolean Circuits for Sequence/Graph Models, and ML/LLM Precedents in ATPG, Fault Simulation and Testability

Legend used throughout: **[PROOF]** = formal theorem in the cited work; **[EMPIRICAL]** = measured result reported by authors; **[CLAIM]** = author/secondary claim not independently checked; **[PR]** = peer-reviewed venue; **[PREPRINT]** = arXiv / unreviewed; **[REPO]** = GitHub project, unreviewed. URLs marked "(prior knowledge)" are well-known arXiv records cited from memory because arxiv.org could not be fetched in this session (search results were reachable; direct fetches of arxiv.org and iris.polito.it failed with DNS errors and must be allowed in the session network settings to re-check them).

## Q1. Theory: what expressivity results say about transformers and GNNs on circuit/graph problems, and what they imply for evaluating depth-d / size-n circuits with a fixed context or fixed-size embedding

### Takeaway
No fixed-depth transformer or GNN can evaluate, let alone find tests for, arbitrary circuits in a single forward pass. Log-precision transformers sit in uniform TC0, while circuit evaluation is P-complete. Feasibility *is* formally proven when the model runs step by step: a constant-depth transformer with T chain-of-thought (CoT) tokens can simulate any circuit of size T, and a decoder-only transformer can provably execute DPLL-style search for SAT via CoT. So the proven route to "netlist larger than context" is algorithmic: per-step or per-cone processing with intermediate state, not a single holistic encoding.

### Cited Findings
- **[PROOF][PR] Transformers without CoT are in uniform TC0.** Merrill & Sabharwal prove that transformers with O(log n) precision can be simulated by logspace-uniform constant-depth threshold circuits, i.e., they solve only problems in uniform TC0. Under L ≠ P they "cannot even accurately solve linear equalities or check membership in an arbitrary context-free grammar with empty productions." The authors *speculate*, without proof, that any architecture as parallelizable as the transformer has similar limits. — [The Parallelism Tradeoff, TACL 2023, arXiv:2207.00729](https://arxiv.org/abs/2207.00729v4); [TACL page](https://transacl.org/index.php/tacl/article/view/4305)
- **[PROOF][PR] CoT length determines power.** Merrill & Sabharwal show that a logarithmic number of intermediate steps adds little, that linear steps (projected pre-norm) recognize all regular languages, and that polynomial steps (generalized pre-norm) let decoder transformers solve exactly the class P. — [The Expressive Power of Transformers with Chain of Thought, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1f59721c106ea80f613299039112f651-Abstract-Conference.html); [arXiv:2310.07923](https://arxiv.org/pdf/2310.07923.pdf)
- **[PROOF][PR] T CoT steps simulate a size-T Boolean circuit.** Li, Liu, Zhou & Ma prove that constant-depth transformers with constant-bit precision and O(log n) embedding size, given T CoT steps, can compute any function computable by a Boolean circuit of size T (Theorem 3.3). Polynomial T gives P/poly. Without CoT, constant-bit-precision constant-depth transformers are limited to AC0. — [Chain of Thought Empowers Transformers to Solve Inherently Serial Problems, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3309b4112c9f04a993f2bbdd0274bba1-Abstract-Conference.html); [arXiv:2402.12875](https://arxiv.org/abs/2402.12875)
- **[PROOF][PR] Transformers can execute SAT search via CoT.** Pan, Ganesh, Abernethy, Esposo & Lee prove that a decoder-only transformer can decide 3-SAT with backtracking and deduction through CoT. Correctness is shown by trace equivalence to DPLL, and the construction is compiled to a PyTorch model ("PARAT") for empirical checking. — [Can Transformers Reason Logically? A Study in SAT Solving, ICML 2025 (PMLR v267)](https://proceedings.mlr.press/v267/pan25d.html); [arXiv:2410.07432](https://arxiv.org/pdf/2410.07432)
- **[PROOF][PR] Graph-reasoning hierarchy for transformers.** Sanford et al. separate 9 graph/algorithmic tasks into classes solvable in different depth/width/extra-token regimes. Logarithmic depth is *necessary and sufficient* for tasks like graph connectivity, while single-layer transformers with small embeddings solve contextual retrieval tasks. Empirically, transformers outperform specialized GNNs on many GraphQA tasks. — [Understanding Transformer Reasoning Capabilities via Graph Algorithms, NeurIPS 2024](https://neurips.cc/virtual/2024/poster/96239); [arXiv:2405.18512](https://arxiv.org/abs/2405.18512v1); [Google Research blog](https://research.google/blog/understanding-transformer-reasoning-capabilities-via-graph-algorithms/)
- **[PROOF][PR] GNN depth × width lower bounds.** Loukas shows that message-passing GNNs are Turing universal under sufficient depth, width, node attributes and layer expressiveness. Several problems are impossible unless depth × width exceeds a polynomial of graph size, and the dependence remains for apparently simple tasks and under approximation. These are worst-case expressivity bounds, not trainability results. — [What graph neural networks cannot learn: depth vs width, ICLR 2020](https://iclr.cc/virtual/2020/poster/1736); [arXiv:1907.03199](https://arxiv.org/pdf/1907.03199)
- **[PROOF/EMPIRICAL][PR] Over-squashing.** Alon & Yahav introduced "over-squashing": exponentially growing receptive fields compressed into fixed-size node vectors create bottlenecks (ICLR 2021, [arXiv:2006.05205](https://arxiv.org/abs/2006.05205), prior knowledge). Topping et al. formalize it via the Jacobian of node representations and show negatively curved edges (Balanced Forman curvature) drive it. They note no consensus on the right fix (rewiring or otherwise). — [Understanding over-squashing and bottlenecks on graphs via curvature, ICLR 2022](https://mlanthology.org/iclr/2022/topping2022iclr-understanding); [arXiv:2111.14522](https://arxiv.org/pdf/2111.14522)
- **[PROOF][PR] WL limits.** Standard message-passing GNNs are at most as discriminative as the 1-WL isomorphism test. GIN attains that bound (Xu et al., ICLR 2019, [arXiv:1810.00826](https://arxiv.org/abs/1810.00826), prior knowledge), and k-GNNs that match k-WL are a higher-order remedy (Morris et al., AAAI 2019, [arXiv:1810.02244](https://arxiv.org/abs/1810.02244), prior knowledge).
- **[PROOF] ATPG hardness.** ATPG is NP-complete, which is why backtracking dominates runtime (stated as background in DETECTive). — [DETECTive record, PoliTo](https://iris.polito.it/handle/11583/2990442)

### Inferences
- **Single pass vs. chain of thought.** The circuit value problem (evaluating a circuit) is P-complete, a standard complexity fact. Combined with the TC0 bound, a fixed-depth, log-precision transformer cannot faithfully evaluate *arbitrary* depth-d netlists in one forward pass unless TC0 = P. Fault simulation of a single fault is circuit evaluation, so the same limit applies. The Li et al. construction gives the matching positive result: about one CoT token per gate is *sufficient*. So an LLM that "fault-simulates" an n-gate cone needs Θ(n) sequential tokens in the generic construction. This is a feasibility proof for an LLM that walks the netlist in topological order, emitting the values as it goes.
- **Test generation as search.** Finding a test is NP-complete, so no polynomial-size CoT is expected in the worst case. Pan et al. prove a transformer can *execute* DPLL, and by the same argument PODEM, which is a DPLL-like search over PI assignments. That is a feasibility proof for an "LLM-as-ATPG-engine with explicit backtracking trace" design, but trace length is exponential in the worst case.
- **Path-sensitization needs depth.** Sanford et al.'s log-depth necessity for connectivity suggests that reachability-style subproblems need model depth that grows as Θ(log n) with netlist size. Examples are "is there a sensitizable path from fault site to a PO" and transitive-fanin/fanout (TFI/TFO) cone extraction. A fixed-depth LLM therefore has a ceiling unless it uses CoT or external tools (e.g., a graph library that extracts the cone).
- **Message-passing GNNs need rounds equal to logic depth.** For a message-passing GNN to propagate a PI value to a PO, it needs at least as many rounds as the circuit's logic depth: the receptive field grows one hop per round. EPFL "hypotenuse" has 24,801 logic levels (see Q6), so plain message passing is infeasible there. DAG-recurrent encoders that sweep in topological order (DeepGate-style, DAG-RNN) avoid this but are inherently sequential, in line with the parallelism tradeoff.
- **WL indistinguishability.** Circuits with identical local gate-type neighbourhoods can compute different functions (e.g., via reconvergent fanout), so 1-WL-bounded encoders can map functionally different fault sites to identical embeddings. Unique node IDs, positional encodings, or simulation-derived features (e.g., signal probabilities or random-pattern simulation signatures) are needed to break the symmetry.

### Gaps
- No theorem found that specifically bounds transformer performance on *stuck-at ATPG* or *fault simulation* as a function of cone size/depth; the implications above are derived from general results.
- Sanford, Hsu & Telgarsky ("Transformers, parallel computation, and logarithmic depth", ICML 2024) and Feng et al. ("Towards Revealing the Mystery behind Chain of Thought", NeurIPS 2023, [arXiv:2305.15408](https://arxiv.org/abs/2305.15408), prior knowledge) are relevant but were not re-verified in this session.

## Q2. Information-theoretic bounds: bits needed to describe an n-gate circuit, and what compression is possible

### Takeaway
An s-gate fan-in-2 circuit takes Θ(s log s) bits to describe. The upper bound is ≤ 2s·log s + 2s bits, and a matching worst-case lower bound follows from Shannon's counting argument. So lossless worst-case encoding of a 10^5–10^6-gate netlist needs about 3.5–42 Mbit, i.e., at least ~0.2M–2.5M maximal-entropy tokens, and a single fixed-size embedding vector can losslessly hold only a few thousand gates in the worst case. Anything beyond that must be lossy or task-specific, e.g., per-fault cones or testability summaries.

### Cited Findings
- **[PROOF] Upper bound.** The number of circuits of size s is at most 2^{O(s log s)} (for s ≥ n), by encoding each circuit as a compact binary string. Comparing with the 2^{2^n} Boolean functions gives Shannon's lower bound (some functions need ~2^n/(4n) gates). — [Trevisan, CS278 Lecture 2](https://theory.stanford.edu/~trevisan/cs278-08/lecture02.pdf)
- **[PROOF] Explicit constant.** The number of fan-in-2 circuits of size s is at most 2^{2s log s + 2s}. — [TIFR complexity course notes, 2026](https://www.tcs.tifr.res.in/~ramprasad/courses/2026-complexity/all_lectures.pdf). Rao's count gives (3s²+n)^s ≤ (4s²)^s < 2^{3s log s} for n > 4. — [Rao, UW CSE531 Lecture 4](https://homes.cs.washington.edu/~anuprao/pubs/CSE531Sp2020/lecture4.pdf)
- **[PROOF] Matching size bound (Shannon–Lupanov).** The count can be written as 2^{O(S log(n+S))}, and every n-input function has a circuit of size O(2^n/n). — [Williams, MIT notes](https://people.csail.mit.edu/rrw/week2.pdf)

### Inferences
- **Worst-case lower bound (my derivation).** Every n-input function has a circuit of size s ≈ c·2^n/n, and there are 2^{2^n} = 2^{Θ(s log s)} such functions at that size. So even an encoding that preserves only the *function*, not the structure, needs Ω(s log s) bits in the worst case. Together with the upper bound, describing an s-gate circuit costs Θ(s log s) bits, i.e., about log₂ s to 2·log₂ s bits per gate.
- **Concrete sizes (upper bound 2s·log₂ s + 2s, my arithmetic):**
  - c7552 (3,512 gates): ~90 kbit.
  - 10^4 gates: ~0.29 Mbit.
  - 10^5 gates: ~3.5 Mbit.
  - 10^6 gates: ~42 Mbit.
- **Token budget.** A token from a 128k vocabulary carries at most ~17 bits. A maximally efficient lossless serialization of a 10^6-gate netlist therefore needs on the order of 2.5M tokens. Real structural-Verilog text is far less efficient: in my estimate, tens of tokens per gate instance, which is an assumption not measured here.
- **Fixed embedding.** One 4,096-dim fp16 vector holds 65,536 bits, which by the upper bound losslessly indexes circuits of only ~2,600 gates in the worst case. A "fixed-size netlist embedding" for large designs is therefore necessarily lossy. It can be faithful only for a task-specific quotient, e.g., "does vector v detect fault f", which is a much smaller target.
- **Task-specific compression.** For a single stuck-at fault, only the fault site's transitive fan-out to observation points, plus the transitive fan-in of those points, matters. That is standard ATPG cone reasoning, stated here without a URL. So the information an ATPG model needs scales with *cone* size, not design size. Real synthesized netlists are also far from worst-case entropy: they reuse library cells and repeated structures, as the AIG preprocessing in Q5 exploits. That gives practical, but not guaranteed, compression headroom.

### Gaps
- No published measurement found of tokens-per-gate for structural Verilog / BLIF / AIGER under common LLM tokenizers.
- No published measurement of fault-cone size distributions on ITC99/EPFL/industrial designs was found in this session; such statistics would directly bound the per-fault context needed.

## Q3. Length/size generalization: do models trained on small circuits/graphs work on larger ones?

### Takeaway
Theory gives *no guarantee*: GNNs can provably fit small graphs and fail on larger ones when local structure shifts. Transformers length-generalize poorly unless the task has a simple iterative (RASP-L / looped) form. Several circuit-domain results nevertheless report useful extrapolation. DeepGate4 trains on ~15K-gate circuits and is evaluated at 400K–1.6M gates; DETECTive trains small and applies to 29× larger circuits. Iterative models (NeuroSAT, looped transformers) extrapolate by running more steps.

### Cited Findings
- **[PROOF+EMPIRICAL][PR] GNN size generalization can fail.** Yehudai et al. show that when local structures ("d-patterns", WL-inspired) differ between small and large graphs, GNNs can converge to minima that fit small graphs but fail on large ones. Size generalization deteriorates as d-pattern discrepancy grows, and a self-supervised local-structure task is proposed as a remedy. — [From Local Structures to Size Generalization in GNNs, ICML 2021](https://proceedings.mlr.press/v139/yehudai21a.html); [arXiv:2010.08853](https://arxiv.org/pdf/2010.08853)
- **[EMPIRICAL][PR] Looped transformers.** Standard transformers have fixed depth. Looped transformers with an adaptive number of iterations of one decoder block significantly improve length generalization on tasks with known iterative RASP-L solutions (Copy, Parity, Addition). — [Fan et al., Looped Transformers for Length Generalization, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/25cc3adf8c85f7c70989cb8a97a691a7-Abstract-Conference.html); [arXiv:2409.15647](https://arxiv.org/html/2409.15647v2). The RASP-L conjecture (transformers length-generalize when a short RASP-L program exists) comes from Zhou et al., ICLR 2024, [arXiv:2310.16028](https://arxiv.org/abs/2310.16028) (prior knowledge).
- **[EMPIRICAL][PR] SAT-via-CoT transformers.** Trained models generalize out of distribution to problem sizes seen in training, but only to a limited extent to longer problems. — [Pan et al., ICML 2025](https://proceedings.mlr.press/v267/pan25d.html)
- **[EMPIRICAL][PR] DeepGate4 scale-up.** A model trained on circuits averaging 15K gates performs well on 400K-gate circuits. Its memory stays constant from 400K to 1.6M gates, while GNN memory grows linearly, thanks to a partitioning plus update strategy giving sub-linear memory. Reported gains are +15.5% (ITC99) and +31.1% (EPFL) over the next-best model. — [DeepGate4, ICLR 2025, arXiv:2502.01681](https://arxiv.org/pdf/2502.01681); [summary](https://liner.com/review/deepgate4-efficient-and-effective-representation-learning-for-circuit-design-at)
- **[EMPIRICAL][PR] DETECTive transfer.** Trained on small circuits and applied to circuits with up to 29× more gates. — [DETECTive, PoliTo repository](https://iris.polito.it/handle/11583/2990442)
- **[EMPIRICAL][PR] NeuroSAT and DeepSAT.**
  - NeuroSAT can solve problems "substantially larger and more difficult than it ever saw during training by simply running for more iterations" (ICLR 2019, [arXiv:1802.03685](https://arxiv.org/abs/1802.03685), prior knowledge).
  - DeepSAT claims significant gains over learning-based SAT baselines "especially when generalized to SAT instances that are relatively large or with diverse distributions", attributed to AIG normalization. — [arXiv:2205.13745](https://arxiv.org/abs/2205.13745v2) [PREPRINT; later version titled "On EDA-Driven Learning for SAT Solving"]

### Inferences
- **What extrapolates.** The successful cases are (a) iterative or recurrent computation whose step count can grow at inference (NeuroSAT, looped transformers, DAG-recurrent encoders), and (b) partitioned or local encoders whose input statistics are size-invariant (DeepGate4 cones, DETECTive). Both suit an ATPG-LLM that processes one fault cone or one decision at a time. Synthesized netlists from a fixed cell library are plausibly closer to "local structure size-invariant" than social graphs, but Yehudai et al.'s failure mode applies if cone depth or fan-in statistics shift with design size.
- **Train on a cone-size curriculum.** Train on a broad *cone-size* distribution rather than a *design-size* distribution, and report accuracy against cone size and depth.

### Gaps
- No study found that reports LLM (decoder) accuracy on gate-level evaluation or ATPG as a function of netlist size or depth (see Q6).
- DeepGate4's "performs well" at 400K gates is task-specific (signal-probability-style representation tasks), not ATPG; exact metrics were not retrieved.

## Q4. ML/LLM for ATPG, fault simulation, testability and DFT: how they encoded the netlist and handled large designs

### Takeaway
Every ML-for-test result that scales to industrial size keeps a classical engine (PODEM, deterministic ATPG, commercial DFT flow) and uses the model for *local decisions* from *per-gate features* (SCOAP/COP, gate type, levels) or for graph-level testability classification. The netlist is never serialized wholesale into a model context. Pure-prediction approaches (DETECTive) have reached up to 95% pattern accuracy on realistic designs. I found no peer-reviewed LLM that generates stuck-at patterns directly from a gate-level netlist with measured fault coverage as of this search.

### Cited Findings
**Backtrace guidance and decision learning (encoding = per-gate feature vectors, local decisions inside PODEM)**
- **[EMPIRICAL][PR] Roy, Millican & Agrawal (VLSID 2021; Auburn).** A neural network picks the backtrace fanin in PODEM from multiple testability measures (e.g., COP, SCOAP) plus gate-type features. Training labels come from PODEM runs: "success" if the backtrace leads to a test, "failure" if it causes a backtrack. Single-feature networks match conventional PODEM, while multi-feature networks reduce backtracks and CPU time. — [VLSID'21 proceedings PDF](https://www.eng.auburn.edu/~agrawvd/TALKS/VLSID21/From_Proceedings.pdf); [ITC'20 material](https://www.eng.auburn.edu/~agrawvd/TALKS/ITC20/PID6639785.pdf); [Auburn dissertation "Toward Zero Backtracks…"](https://etd.auburn.edu/handle/10415/7836?show=full)
- **[EMPIRICAL][PREPRINT, venue unverified] HybMT.** A meta-predictor chooses per net between a neural network (HybNN) and an SVR regressor. It reports a 56.6% speedup over a commercial ATPG tool and 126.4% over the prior ML-based state of the art. — [arXiv:2207.11312](https://ar5iv.labs.arxiv.org/html/2207.11312)
- **[EMPIRICAL][PR] MLCAD 2024, "Efficient and Effective Neural Networks for ATPG".** Applies the network only to selected circuit levels and adds a lookup technique to cut per-gate inference cost. This directly addresses the overhead of calling a model at every backtrace. — [NSF PAR record](https://par.nsf.gov/servlets/purl/10541552)
- **[EMPIRICAL][PR] Zeng & Cui, ITC-Asia 2024.** Falls back to traditional heuristics when the ANN's input-score difference is below a threshold, and reports this is "better suited to large circuits". — [search summary; ITC-Asia 2024, as reported in search results](https://par.nsf.gov/servlets/purl/10541552)
- **[EMPIRICAL][PR] KNN-based ATPG (JCAD 2023).** Replaces the backtrace heuristic with k-nearest-neighbour selection to reduce backtracks on ISCAS85/89 and ITC99. — [JCAD](https://www.jcad.cn/en/article/doi/10.3724/SP.J.1089.2023.20030)

**Direct pattern prediction / ML-assisted deterministic ATPG**
- **[EMPIRICAL][PR] DETECTive (PoliTo; ACM 2024 per search; exact venue not verified).** Graph deep learning learns fault characteristics and *predicts* a test pattern for a designated stuck-at fault, avoiding backtracking entirely.
  - Speed: up to 15× faster than academic and up to 2× faster than commercial tools.
  - Pattern accuracy: up to 100% on synthetic and up to 95% on realistic designs.
  - Transfer: trained small, applied to designs with up to 29× more gates.
  - The authors acknowledge a predicted pattern may fail to detect the fault, so verification by fault simulation remains necessary.
  - [PoliTo record](https://iris.polito.it/handle/11583/2990442); [related thesis](https://webthesis.biblio.polito.it/27650/)
- **[EMPIRICAL][PR] FPGNN-ATPG (ACM GLSVLSI 2023).** A GNN-based fault classifier feeds a fault-driven deterministic generator in a fault-parallel ATPG. On 10 industrial instances (8 cores) it reports a 7.56× average speedup and 14.13% fewer patterns at 100% fault coverage. — [ACM DL 10.1145/3583781.3590250](https://dl.acm.org/doi/10.1145/3583781.3590250). The same page cites Zhao, Yang & Shan (ISEDA 2024), a graph autoencoder for fault prediction in TPG; details not retrieved.

**Testability analysis and test-point insertion (encoding = whole netlist as a graph, node = gate)**
- **[EMPIRICAL][PR] Ma et al., DAC 2019 (CUHK + NVIDIA).**
  - Encoding: the netlist is a directed graph with node = gate and initial node attributes = SCOAP values.
  - Task: imbalanced classification of difficult-to-observe nodes, used to drive iterative observation-point insertion.
  - Scale: industrial designs, each graph > 1M nodes and > 2M edges.
  - Result vs. a commercial tool: similar fault coverage with 11% fewer observation points and 6% fewer test patterns.
  - [Paper PDF](https://www.cse.cuhk.edu.hk/~byu/papers/C85-DAC2019-GCN-Test.pdf); [NVIDIA page](https://research.nvidia.com/publication/2019-06_high-performance-graph-convolutional-networks-applications-testability-analysis); [patent US10657306](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10657306)
- **[EMPIRICAL][PR] DeepTPI, ITC 2022 (CUHK + HiSilicon).**
  - Method: the first reinforcement-learning TPI. The netlist becomes a graph, each action inserts a test point after a gate, and a GNN + DQN ("Graph-DQN") reuses pretrained general node embeddings and adds testability-aware attention.
  - Target: LBIST coverage limited by random-pattern-resistant faults.
  - Result: reports significantly better coverage than a commercial DFT tool across circuit scales.
  - [arXiv:2206.06975](https://arxiv.org/pdf/2206.06975); [code](https://ar5iv.labs.arxiv.org/html/2206.06975)

**Learned fault simulation / fault-effect propagation**
- **[PREPRINT] ST-GCN for silent data corruption (arXiv 2509.06289).** Spatio-temporal GNN predicting propagation of stuck-at, transition and delay faults on ISCAS'89 over 5- and 10-cycle horizons. It compares structural-testability inputs vs. simulation-derived inputs and measures error against ground-truth fault simulation plus graph-conversion time. Final numbers were not retrieved. — [arXiv:2509.06289](https://arxiv.org/pdf/2509.06289)

**LLMs in test/DFT (sparse; mostly adjacent tasks)**
- **[PR, venue details not retrieved] LLMCov.** LLM-aided resolution of scan-insertion design-rule violations to improve DfT coverage in open-source designs; quantitative results not visible in snippets. — [IEEE Xplore 11141575](https://ieeexplore.ieee.org/abstract/document/11141575)
- **[PR] Schwachhofer et al., DFT 2024.** LLM-based optimization for *system-level test* program generation (cited in an LLM-for-EDA survey). — [LLMs for EDA survey, arXiv:2508.20030](https://arxiv.org/pdf/2508.20030)
- **[CLAIM] DFT is a gap.** An MDPI scoping review lists design-for-test as an open research gap for LLMs in hardware design and verification. — [Electronics 14(1):120](https://www.mdpi.com/2079-9292/14/1/120)
- **[REPO] ASAP7 "Language of Test" (appears to be the end user's own project).** Pipeline: RTL → gate-level structural netlist → ATPG patterns → fault simulation → LLM training data for fault-aware test generation, with prompts pairing netlist + target fault and ATPG-derived outputs. — [github.com/christos-vasileiou/asap7-language-of-test](https://github.com/christos-vasileiou/asap7-language-of-test)
- **[REPO] GNN-RL-ATPG.** Gate-level netlists of Ibex and PicoRV32 as graphs with per-gate features, a PPO agent making ATPG decisions, and a PODEM baseline. Unreviewed. — [github.com/mridulabeshi/GNN-RL-ATPG](https://github.com/mridulabeshi/GNN-RL-ATPG)
- **[CLAIM] Background.** ATPG analyzes the netlist, identifies fault conditions and searches for input assignments that propagate fault effects to observable outputs; patterns are typically delivered as IEEE 1450 STIL. — [IEEE TechNav](https://technav.ieee.org/topic/automatic-test-pattern-generation/)

### Inferences
- **Established precedent: local decisions inside a classical engine.** The ML model's input is a bounded per-gate or per-decision feature vector (SCOAP/COP, gate type, level), so model input size is independent of netlist size. Measured speedups exist (HybMT 56.6% vs. commercial; FPGNN 7.56×), and fault coverage is guaranteed by the engine. An LLM can be slotted into the same role (backtrace/objective selection, fault ordering, compaction hints) with the same scaling guarantee.
- **Second precedent: whole-graph GNN classification.** Testability/TPI models (Ma DAC'19 at > 1M nodes; DeepTPI) produce node-level predictions that steer a commercial flow. They scale because GCN cost is linear in edges and computation is sparse. This is not a sequence-context approach.
- **Third precedent: per-fault pattern prediction (DETECTive).** This is closest to "LLM emits the test". It shows graph models can learn fault-to-pattern mappings that transfer to 29× larger circuits, but reported accuracy (≤ 95% realistic) means a fault-simulation verification loop is mandatory. That also fits the theory in Q1: a single-pass predictor cannot be exact in general.
- **Recommended LLM-ATPG architecture under these precedents.**
  1. A classical tool extracts the fault's sensitization cone (TFO of the site plus TFI of reached outputs).
  2. The cone is serialized in topological order with per-gate testability annotations (SCOAP/COP), possibly truncated or abstracted beyond k levels.
  3. The LLM proposes PI assignments or decisions.
  4. A fault simulator verifies; failures fall back to, or are repaired by, PODEM/SAT.

### Gaps
- No peer-reviewed paper found (2023–Oct 2026) in which an LLM directly generates stuck-at or transition patterns from gate-level netlists with reported fault coverage. Searches of ITC/VTS/ETS/ATS-oriented terms returned only adjacent LLM work (system-level test, scan DRC, verification). This may reflect search-index limits; checking IEEE Xplore ITC 2025/VTS 2026 proceedings directly is recommended.
- Specific items named in the brief, "LLM4DFT" and "HybridATPG", could not be located under those names. HybMT (above) may be what "HybridATPG" refers to.
- DETECTive's exact venue, graph encoding details and per-benchmark sizes could not be retrieved (host iris.polito.it blocked in this session).

## Q5. SAT/LEC analogies: neural Circuit-SAT and LLM-guided SAT/equivalence checking (scale and encoding)

### Takeaway
ATPG for one fault reduces to Circuit-SAT on the fault miter. The neural SAT line shows that (a) circuit (DAG/AIG) encodings generalize better than CNF encodings, (b) iterative message passing extrapolates by running longer, and (c) the most useful role at industrial scale is *guiding* a CDCL solver, e.g., phase selection on LEC miters, not replacing it. LLMs reasoning directly over SAT instances produce unreliable traces. A transformer can provably emulate DPLL, but learned versions length-generalize poorly.

### Cited Findings
- **[EMPIRICAL][PR] Amizadeh, Matusevych & Weimer (ICLR 2019).** A DAG neural network embeds the circuit, trained end-to-end with a differentiable, unsupervised, RL-like objective. It reports superior out-of-sample generalization vs. NeuroSAT, and the framework is "mostly suitable for circuit inputs with DAG structure". — [Learning to Solve Circuit-SAT, ICLR 2019](https://iclr.cc/virtual/2019/poster/750); [PDF](https://asgweb-production.azurewebsites.net/applied-sciences/uploads/publications/91/learning_to_solve_circuit_sat_an_unsupervised_differentiable_approach.pdf)
- **[EMPIRICAL][PREPRINT] DeepSAT.** Converts SAT instances to optimized AIGs via logic synthesis to shrink distribution gaps. It is supervised with simulation-derived per-node logic-1 probabilities, uses a DAGNN with polarity prototypes (reference config: 10 rounds, hidden size 128), and claims stronger generalization to larger instances. — [arXiv:2205.13745](https://arxiv.org/abs/2205.13745v2); [SATGL config](https://satgl.readthedocs.io/en/latest/user_guide/model/sat/deepsat.html)
- **[PREPRINT] Circuit-aware SAT solving.** Guides CDCL (CaDiCaL) via conditional probabilities for phase selection and clause filtering, evaluated on logic-equivalence-checking miter instances. — [arXiv:2508.04235](https://arxiv.org/pdf/2508.04235)
- **[PREPRINT] Logic Optimization Meets SAT.** Preprocesses Circuit-SAT with logic optimization and an RL agent, evaluated on industrial LEC-derived instances (per search summary). — [arXiv:2403.19446](https://arxiv.org/html/2403.19446)
- **[PREPRINT] ReVEAL.** A GNN infers multiplier architecture, followed by SAT-based combinational equivalence checking. — [arXiv:2512.22260](https://arxiv.org/pdf/2512.22260)
- **[EMPIRICAL][PR] SATBench (EMNLP 2025).** LLM reasoning traces on SAT problems are often unreliable. — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1716.pdf)
- **[PROOF+EMPIRICAL][PR] Pan et al. (ICML 2025).** Transformer-DPLL equivalence via CoT, with limited length generalization when learned (see Q1/Q3). — [PMLR](https://proceedings.mlr.press/v267/pan25d.html)

### Inferences
- **Circuit-native encodings.** For an ATPG-LLM, the analogy favours circuit-native encodings (AIG / topologically ordered gate lists with signal-probability annotations) over CNF/Tseitin serializations. The demonstrated scalable integration is "learned guidance + exact solver" (phase/decision hints on the fault miter), which matches the Q4 ATPG precedents.
- **Neural solvers are verifiers' assistants.** Neural SAT solvers alone have not been shown to solve industrial-size Circuit-SAT/LEC instances. Every scalable result in this search keeps a CDCL or ATPG engine for completeness.

### Gaps
- Exact instance sizes (variables/gates) at which NeuroSAT, DeepSAT and circuit-aware guidance were evaluated were not retrieved (arxiv.org fetch blocked).
- No paper found using an LLM to guide a SAT solver specifically on ATPG miters; "SolSearch" (LLM-evolved SAT strategies) appeared only as an unverified snippet.

## Q6. Empirical scaling: benchmark size ranges, and LLM accuracy vs. netlist size

### Takeaway
Standard test benchmarks span about 6 gates (c17) to about 98K gates / 6.6K flip-flops (ITC99 b19). EPFL arithmetic circuits reach 214K AIG nodes and 24,801 levels, its "more than ten million" set reaches about 60M nodes, and industrial GCN work handles more than 1M nodes. I found no published curve of *LLM* accuracy versus gate-level netlist size or depth. The nearest evidence shows sharp accuracy drops as circuit-reasoning difficulty rises.

### Cited Findings
- **ISCAS-85.** c7552 has 3,512 gates (some sources 3,513), 207 PIs and 108 POs. c432 = 160, c880 = 383, c1355 = 546, c1908 = 880 gates. Sources disagree for c2670 (1,193 vs. 1,269) and c6288 (2,406 vs. 2,416). — [c7552 high-level model](https://ddd.fit.cvut.cz/www/prj/Benchmarks/ISCAS_Desc/c7552.html); [ISCAS85 table](https://ddd.fit.cvut.cz/www/prj/Benchmarks/iscas85.pdf)
- **ISCAS-89.** s38584 maps to 6,724 library cells in the IWLS 2005 synthesized version, which is not comparable to the original gate count. The original-table column headers were garbled in retrieval. — [IWLS 2005 table](https://ddd.fit.cvut.cz/www/prj/Benchmarks/IWLS2005.pdf)
- **ITC'99.** b17 = 3 × b15; b18 and b19 = 2 × b14 + 2 × b17. Flip-flops are about 1,415 (b17), 3,296 (b18) and 6,594 (b19), and the largest is described as about 98,000 gates and 6,600 FFs. Column interpretation in the IWLS table is uncertain. — [ITC'99 notes](https://www.cerc.utexas.edu/itc99-benchmarks/bendoc1.html); [IWLS 2005 table](https://ddd.fit.cvut.cz/www/prj/Benchmarks/IWLS2005.pdf)
- **EPFL.** 23 circuits:
  - hypotenuse: 256 PIs, 128 POs, 214,335 AND nodes, 24,801 levels.
  - multiplier: 27,062 AND nodes, 274 levels.
  - Arithmetic set: about 0.37M nodes in total.
  - Three "very large" circuits with more than 10M gates each (about 60M nodes total).
  - LUT-6 mapped counts (e.g., 44,635 for hypotenuse) are a different metric.
  - [IWLS'15 paper](https://si2.epfl.ch/%7Edemichel/publications/archive/2015/IWLS15.pdf); [EPFL benchmarks page](https://epfl.ch/labs/lsi/page-102566-en-html/benchmarks); [ShortCircuit table, arXiv:2408.09858](https://arxiv.org/pdf/2408.09858)
- **OpenABC-D.** 870,000 AIGs from 1,500 synthesis runs (recipe length 20) of 29 open IPs (MIT-CEP, IWLS, OpenROAD, OpenPiton…), made with yosys-abc and Nangate 15nm; 1.4 TB. Per-IP node counts were not retrieved. — [arXiv:2110.11292](https://arxiv.org/abs/2110.11292); [GitHub](https://www.github.com/NYU-MLDA/OpenABC)
- **ForgeEDA.** 1,189 real-world repositories (6 or 7 categories, depending on version) with post-mapping and placed netlists. The ">300K circuits" figure is a secondary claim. — [arXiv:2505.02016](https://arxiv.org/abs/2505.02016)
- **DeepCircuitX.** More than 4,000 repository-level RTL projects with synthesized netlists and PPA. — [arXiv:2502.18297](https://arxiv.org/pdf/2502.18297)
- **Scale achieved by ML models on netlists.**
  - Ma et al. DAC'19: > 1M nodes / > 2M edges per industrial graph. — [PDF](https://www.cse.cuhk.edu.hk/~byu/papers/C85-DAC2019-GCN-Test.pdf)
  - DeepGate4: trained at ~15K gates, evaluated to 1.6M gates. — [arXiv:2502.01681](https://arxiv.org/pdf/2502.01681)
- **Nearest "LLM accuracy vs. difficulty" evidence.**
  - CIRCUIT (analog, netlist-included) [PREPRINT/OpenReview]: GPT-4o reaches 48.04% overall, falling from 49.4% at level 1 to 18.5% at level 5. — [arXiv:2502.07980](https://arxiv.org/html/2502.07980v1)
  - Operator-logic chains at depth 1–10 (not netlists) [PREPRINT]: some models stay ≥ 99%, but at depth 5 accuracy varies from 98% to 44% by operator for GPT-4o. — [arXiv:2604.13065](https://arxiv.org/html/2604.13065)

### Inferences
- **Context ceiling.** Even the largest classic ATPG benchmark (b19, ~10^5 gates) needs ~3.5 Mbit (~0.2M ideal tokens) to describe losslessly (Q2). Its structural-Verilog text is likely far larger, so "whole netlist in context" is out of reach for ITC99-scale designs, let alone industrial (> 10^6) designs. Per-fault cones of ISCAS/ITC circuits are plausibly within context, but this is unmeasured (see Gaps).
- **Benchmark mix for scaling curves.**
  - ISCAS85/89 for small-cone sanity checks.
  - ITC99 b14–b22 for mid-size sequential (full-scan) designs.
  - EPFL arithmetic (deep: 274–24,801 levels) for depth stress.
  - OpenABC-D / ForgeEDA for synthesis-diverse training.
  - Report accuracy and fault coverage against cone gates and cone depth.

### Gaps
- No study found that measures LLM accuracy (logic simulation, fault detection, or test generation) against gate-level netlist size or depth. This is a clear open gap and a publishable contribution.
- Authoritative original gate counts for s38584 and ITC99 b14–b22 could not be confirmed. Commonly quoted values (e.g., s38584 ≈ 19K gates) were not verified in this session.

## Q7. Synthesis: which methodologies for netlists larger than the context window have the strongest "proofs of feasibility"?

### Takeaway
Ranked by strength of evidence:
1. **Classical engine + learned local decisions.** Proven in practice at industrial scale (Ma DAC'19 > 1M nodes; FPGNN 100% FC; HybMT vs. commercial). Complete by construction.
2. **Per-fault cone extraction + LLM step-wise simulation/search via CoT.** Formal expressivity proofs: T CoT steps ⇒ size-T circuit simulation, and CoT DPLL ⇒ SAT/ATPG search. Practical risk is length generalization.
3. **Partitioned or hierarchical graph-transformer encodings.** Empirical scale-up to 1.6M gates with constant memory (DeepGate4), but no task-faithfulness guarantee.
4. **Single fixed-size whole-netlist embedding.** Information-theoretically impossible to be lossless beyond a few thousand gates. Viable only for coarse, task-specific summaries.

### Cited Findings
- The T-step CoT → size-T circuit theorem. — [Li et al., ICLR 2024](https://arxiv.org/abs/2402.12875)
- P-exactness of polynomial CoT. — [Merrill & Sabharwal, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1f59721c106ea80f613299039112f651-Abstract-Conference.html)
- DPLL-via-CoT proof and limited learned length generalization. — [Pan et al., ICML 2025](https://proceedings.mlr.press/v267/pan25d.html)
- Θ(s log s) description length. — [Trevisan](https://theory.stanford.edu/~trevisan/cs278-08/lecture02.pdf); [TIFR notes](https://www.tcs.tifr.res.in/~ramprasad/courses/2026-complexity/all_lectures.pdf)
- Partitioned graph transformer: constant memory from 400K to 1.6M gates. — [DeepGate4](https://arxiv.org/pdf/2502.01681)
- Industrial-scale GCN testability. — [Ma et al. DAC 2019](https://www.cse.cuhk.edu.hk/~byu/papers/C85-DAC2019-GCN-Test.pdf)
- ML-guided ATPG with complete coverage. — [FPGNN-ATPG](https://dl.acm.org/doi/10.1145/3583781.3590250)
- ML-guided backtrace speedup over a commercial tool. — [HybMT](https://ar5iv.labs.arxiv.org/html/2207.11312)
- Pattern prediction with 29× size transfer but ≤ 95% accuracy. — [DETECTive](https://iris.polito.it/handle/11583/2990442)
- Log-depth necessity for connectivity-type tasks. — [Sanford et al.](https://arxiv.org/abs/2405.18512v1)
- Depth × width ≥ poly(n) for GNNs on some tasks. — [Loukas](https://arxiv.org/pdf/1907.03199)
- Size-generalization failure mode. — [Yehudai et al.](https://proceedings.mlr.press/v139/yehudai21a.html)

### Inferences
- **Fault-local decomposition removes the context problem in the common case.** The information an ATPG step needs is bounded by the fault's sensitization cone, not the design (Q2). Cone extraction is cheap and exact with classical tools, so it carries no ML risk.
- **Remaining risk is cones that are themselves larger than context.** Wide-fanout signals, multipliers and deep arithmetic are examples, e.g., EPFL's 24,801-level hypotenuse. Proven-feasible handling options:
  - (a) Step-wise CoT over a topologically ordered cone with an external state store (Li et al. guarantee), or a tool-call loop where the LLM queries the netlist graph for neighbours on demand.
  - (b) Fall back to PODEM/SAT with LLM-supplied hints (completeness kept).
  - (c) Hierarchical abstraction, e.g., replacing fully-specified sub-cones by their SCOAP/COP summaries. This is lossy; its faithfulness would need empirical validation.
- **Every LLM-proposed pattern must be fault-simulated.** Single-pass prediction cannot be exact in general (TC0 vs. P; ATPG NP-complete), and empirical pattern predictors top out around 95% on realistic designs. Fault simulation is cheap and exact, which turns an approximate model into a sound system.
- **Evaluation protocol that would itself be a feasibility proof.**
  - Report fault coverage, test-generation success rate, and verified-pattern rate against cone size and depth, on ISCAS85 → ITC99 → EPFL → OpenABC-D designs.
  - Train on small cones and test on 10–30× larger ones, mirroring the DETECTive and DeepGate4 protocols.

### Gaps
- No end-to-end LLM-ATPG system with published fault coverage on ITC99/EPFL-scale designs was found. The feasibility case therefore rests on (i) expressivity theorems, (ii) classical-engine-plus-ML precedents, and (iii) graph-model scale-up evidence, not on a direct LLM precedent.
- No published distribution of stuck-at fault cone sizes (gates, depth) across standard benchmarks was located. Computing it is a cheap, high-value first experiment for sizing context budgets.
