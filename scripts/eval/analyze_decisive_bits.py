#!/usr/bin/env python3
"""Decisive-bit analysis of a GRPO checkpoint against uniform random vectors.

    python scripts/eval/analyze_decisive_bits.py CHECKPOINT TESTSET [--out DIR]
    python scripts/eval/analyze_decisive_bits.py --results MODEL_SLOTS --random RANDOM_SLOTS [--out DIR]

TESTSET is a frozen eval manifest (.json) or a Hugging Face dataset whose
``test`` split is sampled into DIR/manifest.json. Random and single_completion runs go
through eval_grpo_policy_checkpoints.sh (its environment variables still apply)
and are skipped when DIR already holds their slots. --results runs no
evaluation: it reports existing *.slots.jsonl files that recorded the same eval
manifest. Bins are always recomputed here (exhaustive up to 16 inputs, else
65,536 random vectors), so both modes bin identically. The report holds Table 1
(decisive bits / inputs, P(act) and P(prop given act), random vs model), Table 2
(agreement on decisive vs other bits), a Method section explaining both, and an
appendix of outcomes and all-zeros answers.
"""
import argparse
import itertools
import json
import math
import os
import re
import statistics as st
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from fault_difficulty import DEFAULT_CONFIG, Analyzer, Circuit, difficulty_bin, identity, load_manifest, raw_netlist

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BINS = ["easy", "medium", "hard", "very_hard", "unresolved_zero_hits", "proven_undetectable", "unknown"]
BIN_NAMES = dict(easy="Easy", medium="Medium", hard="Hard", very_hard="Very hard", unresolved_zero_hits="Random never detects",
                 proven_undetectable="Proven undetectable", unknown="Unsupported")
MIN_PROBLEMS = 5  # hide agreement cells averaged over fewer problems
DIFFICULTY_SAMPLES, DIFFICULTY_EXACT_MAX_INPUTS, DIFFICULTY_SEED = 65536, 16, 1729


def resolve_checkpoint(path):
    for candidate in (Path(path), REPO_ROOT / path, REPO_ROOT / "runs" / path):
        if candidate.is_dir():
            candidate = candidate.resolve()
            while candidate.name in ("policy", "combined"):
                candidate = candidate.parent
            return candidate
    sys.exit(f"error: checkpoint not found: {path}")


def run_eval(checkpoint, method, out, testset, manifest):
    if list(out.glob(f"*_passatk_{method}_*.slots.jsonl")):
        print(f"[{method}] reusing slots in {out}")
        return
    env = dict(os.environ, EVAL_RESULTS_DIR=str(out), SAMPLING_METHOD=method, EVAL_MANIFEST=str(manifest))
    if not testset.endswith(".json"):
        env["EVAL_DATASET"] = testset
    for key, value in dict(NUM_COMPLETIONS="16", EVAL_PROMPT_BATCH_SIZE="64",
                           DIFFICULTY_RANDOM_SAMPLES="65536", DIFFICULTY_EXACT_MAX_INPUTS="16").items():
        env.setdefault(key, value)
    print(f"[{method}] evaluating {checkpoint}")
    subprocess.run([str(SCRIPT_DIR / "eval_grpo_policy_checkpoints.sh"), str(checkpoint)], env=env, check=True)


def one(out, pattern):
    files = sorted(out.glob(pattern))
    if len(files) != 1:
        sys.exit(f"error: expected one {pattern} in {out}, found {len(files)}")
    return files[0]


def slots_config(path):
    return json.loads(open(path).readline())["config"]


def recorded_manifest(path):
    manifest = slots_config(path).get("eval_manifest")
    if not manifest:
        sys.exit(f"error: {path} recorded no eval manifest (pre-manifest run)")
    manifest = Path(manifest)
    return (manifest if manifest.is_absolute() else REPO_ROOT / manifest).resolve()


def vector_of(slot):
    if slot.get("vector"):
        return slot["vector"]
    if slot.get("answer_source") == "random":
        m = re.search(r'INPUT_VECTOR: "(.*?)"', slot["completion"])
        return {k.strip(): int(v) for k, v in (t.split(":") for t in m.group(1).split(","))}
    return None


def finite(x):
    return x is not None and x != "" and not math.isnan(float(x))


def flip_lanes(circuit, fault, vector):
    """Lane 0 simulates the vector, lane i+1 the vector with input i flipped."""
    stuck, net = re.fullmatch(r"sa([01])\s+(.+)", fault.strip()).groups()
    names = circuit.meta.input_nets
    mask = (1 << (len(names) + 1)) - 1
    values = {circuit.alias[n]: (mask if int(vector[n]) else 0) ^ (1 << (i + 1)) for i, n in enumerate(names)}
    good = circuit.simulate(values, mask)
    bad = circuit.simulate(values, mask, (circuit.alias[net], int(stuck)))
    detected = 0
    for o in circuit.outputs:
        detected |= good[o] ^ bad[o]
    return names, [(detected >> lane) & 1 for lane in range(len(names) + 1)]


def load_run(path, bin_of, circuits):
    """Per-bin counters, per-problem activation/detection and vectors."""
    s = defaultdict(Counter)
    per_problem = defaultdict(lambda: [0, 0, 0])  # finished with activation, activated, detected
    vectors = defaultdict(list)                   # (bits, decisive set or None, all-zeros)
    for line in open(path):
        slot = json.loads(line)
        p = slot["evaluation_problem_id"]
        if p not in bin_of:
            sys.exit(f"error: {path} holds a problem that is not in its eval manifest")
        c = s[bin_of[p]]
        c["slots"] += 1
        v = vector_of(slot)
        if not v:
            c["novec"] += 1
            continue
        rc = slot["reward_components"]
        zeros = all(int(x) == 0 for x in v.values())
        c["fin"] += 1
        c["det"] += rc["detection"]
        c["zeros"] += zeros
        polarity = slot["fault"].split()[0]
        c[f"{polarity}_fin"] += 1
        kind = f"{polarity}_{'zeros' if zeros else 'other'}"
        c[kind] += 1
        c[f"{kind}_det"] += rc["detection"]
        if finite(rc.get("activation")):
            c[f"{kind}_act_n"] += 1
            c[f"{kind}_act"] += rc["activation"]
        if finite(rc.get("activation")):
            q = per_problem[p]
            q[0] += 1
            q[1] += rc["activation"]
            q[2] += rc["detection"]
        circuit = circuits.get(p)
        if circuit is None or any(n not in v for n in circuit.meta.input_nets):
            continue
        names, lanes = flip_lanes(circuit, slot["fault"], v)
        bits = tuple(int(v[n]) for n in names)
        decisive = {i for i in range(len(names)) if not lanes[i + 1]} if lanes[0] else None
        vectors[p].append((bits, decisive, zeros))
    return s, per_problem, vectors


def agreement(vectors):
    """Per problem: how often the run's other finished vectors (all-zeros excluded) repeat each detecting
    vector's decisive bits and its other bits. Kept only when both kinds of bits were compared."""
    out = {}
    for p, vs in vectors.items():
        vs = [x for x in vs if not x[2]]
        a = [0, 0, 0, 0]
        for (bits, decisive, _), (other, _, _) in itertools.permutations(vs, 2):
            if decisive is None:
                continue
            for i, (x, y) in enumerate(zip(bits, other)):
                j = 0 if i in decisive else 2
                a[j] += x == y
                a[j + 1] += 1
        if a[1] and a[3]:
            out[p] = (a[0] / a[1], a[2] / a[3])
    return out


def decisive_bits(records, circuits, random_vectors):
    """Per problem: (mean decisive bits over distinct detecting vectors, inputs). Vectors come from the
    dataset label, when it detects, and the random run; neither depends on the checkpoint."""
    out = {}
    for p, circuit in circuits.items():
        found = {bits: len(decisive) for bits, decisive, _ in random_vectors.get(p, []) if decisive is not None}
        label = records[p].get("input_vector")
        label = json.loads(label) if isinstance(label, str) else label
        if label and all(n in label for n in circuit.meta.input_nets):
            names, lanes = flip_lanes(circuit, records[p]["fault"], label)
            if lanes[0]:
                found[tuple(int(label[n]) for n in names)] = sum(not x for x in lanes[1:])
        if found:
            out[p] = (st.mean(found.values()), len(circuit.meta.input_nets))
    return out


def diversity(vectors, pids):
    unique, hamming = [], []
    for p in pids:
        vs = [x[0] for x in vectors.get(p, [])]
        if len(vs) < 2:
            continue
        unique.append(len(set(vs)))
        pairs = list(itertools.combinations(vs, 2))
        hamming.append(sum(sum(x != y for x, y in zip(a, b)) / len(a) for a, b in pairs) / len(pairs))
    return (float(st.mean(unique)) if unique else float("nan")), (st.mean(hamming) if hamming else float("nan"))


def table(header, rows):
    fmt = lambda x: "-" if isinstance(x, float) and math.isnan(x) else f"{x:.3f}" if isinstance(x, float) else str(x)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def ratio(a, b):
    return a / b if b else float("nan")


def report(model_slots, random_slots, manifest, path):
    records = {identity(r): r for r in load_manifest(manifest)[0]}
    analyzer = Analyzer(samples=DIFFICULTY_SAMPLES, exact_max_inputs=DIFFICULTY_EXACT_MAX_INPUTS, seed=DIFFICULTY_SEED)
    difficulty = {p: analyzer.analyze(r) for p, r in records.items()}
    bin_of = {p: next((b for b in BINS if difficulty_bin(r).startswith(b)), difficulty_bin(r)) for p, r in difficulty.items()}
    bins = [b for b in BINS if b in bin_of.values()] + sorted(set(bin_of.values()) - set(BINS))
    gates = json.loads(DEFAULT_CONFIG.read_text())["gate_funcs"]
    circuits = {}
    for p, r in records.items():
        try:
            circuits[p] = Circuit(raw_netlist(r), gates)
        except (ValueError, KeyError, TypeError):
            pass
    model = slots_config(model_slots)["sampling_method"]
    runs = {"random": load_run(random_slots, bin_of, circuits), model: load_run(model_slots, bin_of, circuits)}
    pids = {b: [p for p in bin_of if bin_of[p] == b] for b in bins}

    sections = [f"# Decisive-bit analysis\n\nCheckpoint: `{slots_config(model_slots)['adapter']}`  \n"
                f"Test set: `{manifest}` ({len(records)} problems)  \n"
                f"Model run ({model}): `{model_slots}`  \nRandom run: `{random_slots}`  \n"
                f"Bins: p = P(one random vector detects), exhaustive up to {DIFFICULTY_EXACT_MAX_INPUTS} "
                f"inputs, else {DIFFICULTY_SAMPLES:,} random vectors."]

    name = lambda b: BIN_NAMES.get(b, b)
    m = lambda x: st.mean(x) if x else float("nan")
    adapter = Path(slots_config(model_slots)["adapter"])
    ckpt = next((part for part in reversed(adapter.parts) if part.startswith("checkpoint-")), adapter.name)
    ckpt = f"{ckpt} ({model})"

    # Table 1: every column is a mean of per-problem values; each random/model pair uses the same problems.
    dec = decisive_bits(records, circuits, runs["random"][2])
    _, per_problem, _ = runs[model]
    rows, counts = [], defaultdict(list)
    for b in bins:
        d = [dec[p] for p in pids[b] if p in dec]
        act, act_ref, prop, prop_ref = [], [], [], []
        for p in pids[b]:
            n, a, det = per_problem.get(p, (0, 0, 0))
            ref = difficulty[p]
            if n and finite(ref.get("activation_probability")):
                act.append(a / n)
                act_ref.append(float(ref["activation_probability"]))
            if a and finite(ref.get("conditional_observability")):
                prop.append(det / a)
                prop_ref.append(float(ref["conditional_observability"]))
        k, n_in = m([x[0] for x in d]), m([x[1] for x in d])
        rows.append([f"{name(b)} ({len(pids[b])})", f"{k:.1f} / {n_in:.1f} ({k / n_in:.0%})" if d else "-",
                     m(act_ref), m(act), m(prop_ref), m(prop)])
        for column, values in (("decisive bits", d), ("P(act)", act), ("P(prop given act)", prop)):
            counts[column].append(f"{name(b)} {len(values)}/{len(pids[b])}")
    sections.append(f"## Table 1. p = P(act) × P(prop given act): random vectors vs {ckpt}\n\n" + table(
        ["Bin (problems)", "Decisive bits / inputs", "Random P(act)", "Model P(act)",
         "Random P(prop given act)", "Model P(prop given act)"], rows)
        + "\n\nProblems behind each average: " + "; ".join(f"{c}: {', '.join(v)}" for c, v in counts.items()) + ".")

    # Table 2: per-problem agreement on decisive / other bits, averaged over the bin.
    agree = {run: agreement(vectors) for run, (_, _, vectors) in runs.items()}
    cell = lambda a, ps: (lambda v: f"{m([x[0] for x in v]):.2f} / {m([x[1] for x in v]):.2f} ({len(v)})"
                          if len(v) >= MIN_PROBLEMS else "-")([a[p] for p in ps if p in a])
    rows = [[name(b), cell(agree["random"], pids[b]), cell(agree[model], pids[b])] for b in bins]
    sections.append("## Table 2. Agreement on decisive bits / on other bits\n\n" + table(
        ["Bin", "Random", ckpt], rows)
        + f"\n\nEach cell: decisive-bit agreement / other-bit agreement (problems averaged). '-' marks fewer than "
          f"{MIN_PROBLEMS} problems. An ideal generator scores near 1.00 on decisive bits and 0.50 on the rest; "
          "independent random vectors score about 0.50 on both.")

    sections.append(f"""## Method

1. **Problems and bins.** A problem is one (netlist, fault) pair from the test set; the fault names its net and
   stuck-at value. Its bin comes from p, the probability that one uniform random input vector detects the fault,
   computed with the difficulty analyzer (all input vectors when the circuit has at most
   {DIFFICULTY_EXACT_MAX_INPUTS} inputs, else {DIFFICULTY_SAMPLES:,} random vectors): Easy p ≥ 1/4, Medium 1/16 to 1/4,
   Hard 1/256 to 1/16, Very hard below 1/256, Random never detects when no sampled vector detects it.
2. **Per problem, then per bin.** Every number is first computed for each problem and then averaged over the
   problems of its bin. Each random/model pair of columns is averaged over the same problems (counts under Table 1).
3. **Decisive bits / inputs.** A detecting vector's decisive bits are the inputs whose single flip makes the fault
   undetected; this is a lower bound on the bits that matter. For each problem, decisive bits are averaged over
   its distinct detecting vectors: the dataset's label vector (when it detects) and the random run's detecting
   vectors. Both are independent of the checkpoint. Problems with neither are left out. The cell shows mean decisive
   bits / mean inputs over those problems, and their ratio.
4. **Random P(act) and P(prop given act).** From the same analyzer simulation as the bins: the share of random
   vectors that activate the fault (set the faulty net opposite to its stuck value), and, among those, the share
   that also propagate it to a primary output. Per problem, their product is p. This reference is the
   analyzer's simulation, not the random run's 16 vectors.
5. **Model P(act) and P(prop given act).** From the checkpoint's own answers ({model_slots.name}): per problem,
   activated answers / answers with an input vector, and detected answers / activated answers, using the
   evaluator's fault simulation of each answer. Answers without an input vector are excluded. Model P(prop given
   act) only covers problems the model activated at least once.
6. **Agreement (Table 2).** For each detecting vector of a run, compare it bit by bit with the run's other answers
   to the same problem that have an input vector, all-zeros answers excluded. Agreement on decisive bits = matching
   decisive bits / decisive bits compared; the same on the remaining bits. Averaged per problem, then over the bin.
   The decisive bits here are those of the run's own detecting vector. The Random column uses the random run's
   vectors, which are independent of each other, so its agreement is about 0.50 on both kinds of bits.""")

    rows = []
    for run, (s, _, vectors) in runs.items():
        for b in bins:
            c = s[b]
            u, h = diversity(vectors, pids[b])
            rows.append([run, name(b), len(pids[b]), ratio(c["novec"], c["slots"]), ratio(c["zeros"], c["fin"]),
                         ratio(c["det"], c["fin"]), ratio(c["det"], c["slots"]), u, h])
    sections.append("## Appendix A. Outcomes per bin\n\n" + table(
        ["Run", "Bin", "Problems", "No vector", "All-zeros (finished)", "P(det) finished",
         "P(det) all slots", "Unique vectors / problem", "Mean pairwise Hamming"], rows)
        + "\n\nRatios here are pooled over the answers (slots) of the bin, not averaged per problem.")

    rows = []
    for run, (s, _, _) in runs.items():
        t = Counter()
        for b in bins:
            t.update(s[b])
        for pol in ("sa0", "sa1"):
            rows.append([run, pol, ratio(t[f"{pol}_zeros"], t[f"{pol}_fin"]),
                         ratio(t[f"{pol}_zeros_act"], t[f"{pol}_zeros_act_n"]), ratio(t[f"{pol}_zeros_det"], t[f"{pol}_zeros"]),
                         ratio(t[f"{pol}_other_act"], t[f"{pol}_other_act_n"]), ratio(t[f"{pol}_other_det"], t[f"{pol}_other"])])
    sections.append("## Appendix B. All-zeros answers by fault polarity\n\n" + table(
        ["Run", "Fault", "All-zeros share (finished)", "All-zeros P(act)", "All-zeros P(det)",
         "Other P(act)", "Other P(det)"], rows) + "\n\nPooled over all answers of the run.")

    text = "\n\n".join(sections) + "\n"
    path.write_text(text)
    print(text)
    print(f"Report saved to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", help="GRPO checkpoint directory (checkpoint-N, or its policy/)")
    parser.add_argument("testset", nargs="?", help="frozen eval manifest (.json) or Hugging Face dataset (test split)")
    parser.add_argument("--results", metavar="MODEL_SLOTS",
                        help="report an existing evaluation (its *.slots.jsonl) instead of running one")
    parser.add_argument("--random", metavar="RANDOM_SLOTS",
                        help="with --results: an existing random-baseline *.slots.jsonl on the same manifest")
    parser.add_argument("--out", help="output directory (default: runs/decisive_bits_<experiment>_<checkpoint>_<testset>; "
                                      "with --results, the directory of MODEL_SLOTS)")
    args = parser.parse_args()

    if args.results:
        if args.checkpoint or args.testset:
            parser.error("--results reports an existing evaluation; do not pass CHECKPOINT or TESTSET")
        if not args.random:
            parser.error("--results requires --random")
        model_slots, random_slots = Path(args.results).resolve(), Path(args.random).resolve()
        if slots_config(random_slots)["sampling_method"] != "random" or slots_config(model_slots)["sampling_method"] == "random":
            parser.error("--results takes a model run and --random a random run")
        manifest = recorded_manifest(model_slots)
        if recorded_manifest(random_slots) != manifest:
            parser.error("--results and --random were evaluated on different manifests")
        out = Path(args.out).resolve() if args.out else model_slots.parent
        out.mkdir(parents=True, exist_ok=True)
        report(model_slots, random_slots, manifest, out / model_slots.name.replace(".slots.jsonl", ".decisive_bits.md"))
        return
    if not (args.checkpoint and args.testset) or args.random:
        parser.error("pass CHECKPOINT and TESTSET to run evaluations, or --results and --random to report existing ones")

    checkpoint = resolve_checkpoint(args.checkpoint)
    tag = Path(args.testset).stem if args.testset.endswith(".json") else args.testset.replace("/", "_")
    out = Path(args.out or REPO_ROOT / "runs" / f"decisive_bits_{checkpoint.parent.name}_{checkpoint.name}_{tag}").resolve()
    if args.testset.endswith(".json") and not Path(args.testset).is_file():
        sys.exit(f"error: manifest not found: {args.testset}")
    testset = str(Path(args.testset).resolve()) if args.testset.endswith(".json") else args.testset
    manifest = Path(testset) if testset.endswith(".json") else out / "manifest.json"
    out.mkdir(parents=True, exist_ok=True)
    for method in ("random", "single_completion"):
        run_eval(checkpoint, method, out, testset, manifest)
    if os.environ.get("DRY_RUN") == "1":
        return
    model_slots = one(out, "*_passatk_single_completion_*.slots.jsonl")
    report(model_slots, one(out, "*_passatk_random_*.slots.jsonl"), manifest,
           out / model_slots.name.replace(".slots.jsonl", ".decisive_bits.md"))


if __name__ == "__main__":
    main()
