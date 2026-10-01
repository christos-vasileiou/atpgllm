"""CPU-only combinational fault testability, sampling, and stratified reporting.

SCOAP is a structural heuristic; packed exhaustive/Monte Carlo simulation measures
P(activation), P(detection | activation), and P(detection) without an independence
assumption. Only declared scalar-net stuck-at stems are supported. See
 docs/FAULT_DIFFICULTY_EVALUATION.md for assumptions and interpretation.
"""
from __future__ import annotations

from collections import Counter, defaultdict, OrderedDict
from functools import lru_cache
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import sys

VERSION = "fault-difficulty-v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "atpgllm/training/data/sim_config.json"
INF = float("inf")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def raw_netlist(record):
    value = record.get("netlist", "")
    return value.get("netlist", "") if isinstance(value, dict) else value


def identity(record):
    return digest([raw_netlist(record), str(record.get("fault", "")).strip()])


@lru_cache(maxsize=1024)
def forcing_cubes(lut):
    """All minimal ternary cubes forcing each output, including don't-cares.

    Full assignments alone give the wrong CC0 for AND gates. Enumerating local
    cubes supports arbitrary small ASAP7 cells, including AOI/OAI and muxes.
    """
    n = (len(lut) - 1).bit_length()
    cubes = [[], []]
    for cube in itertools.product((-1, 0, 1), repeat=n):
        values = {lut[i] for i in range(len(lut))
                  if all(v == -1 or (i >> j) & 1 == v for j, v in enumerate(cube))}
        if len(values) == 1:
            cubes[next(iter(values))].append(cube)
    return tuple(tuple(c for c in group if not any(
        other != c and all(a == -1 or a == b for a, b in zip(other, c)) for other in group))
        for group in cubes)


def cube_cost(cubes, costs):
    return min((sum(costs[j][v] for j, v in enumerate(c) if v != -1) for c in cubes), default=INF)


def apply_lut(lut, args, mask):
    result = 0
    for i, value in enumerate(lut):
        if value:
            term = mask
            for j, arg in enumerate(args):
                term &= arg if (i >> j) & 1 else mask ^ arg
            result |= term
    return result


def wilson(successes, trials):
    if not trials:
        return None
    p, z = successes / trials, 1.959963984540054
    den = 1 + z*z/trials
    mid = (p + z*z/(2*trials))/den
    half = z*math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/den
    return [max(0., mid-half), min(1., mid+half)]


def finite(value):
    return value if math.isfinite(value) else None


@lru_cache(maxsize=2048)
def compile_cell(gate, output, expression):
    from fault_sim import _compile_gate_port
    return _compile_gate_port(gate, output, {gate: {output: expression}})


class Circuit:
    """Strict scalar structural parser using the project's cell truth tables.

    Direct wire aliases are collapsed (zero gate depth), matching physical stem
    semantics. Unsupported syntax is rejected, never silently skipped.
    """
    def __init__(self, text, gates):
        sys.path.insert(0, str(ROOT.parent / "data_preprocessing")) if str(ROOT.parent / "data_preprocessing") not in sys.path else None
        from tetramax_backend import SimulationNetlist
        meta = SimulationNetlist(text)
        self.meta = meta
        clean = re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S)
        self.alias = {n: n for n in meta.all_nets}
        for group in meta.net_aliases:
            for n in group:
                self.alias[n] = sorted(group)[0]
        self.inputs = list(dict.fromkeys(self.alias[n] for n in meta.input_nets))
        self.outputs = list(dict.fromkeys(self.alias[n] for n in meta.output_nets))
        if len(self.inputs) != len(meta.input_nets):
            raise ValueError("Aliased primary inputs impose unsupported input constraints")
        def net(n):
            n = n.strip()
            if n in ("0", "1", "1'b0", "1'b1", "1'h0", "1'h1", "*Logic0*", "*Logic1*"):
                return int(n[-1]) if n[-1] in '01' else int(n[-2])
            if n not in self.alias:
                raise ValueError(f"Undeclared/unsupported scalar connection {n}")
            return self.alias[n]
        clean = re.sub(r'\bmodule\s+(?:\\\S+|\w+)\s*\([^;]*?\)\s*;', '', clean, count=1, flags=re.S)
        clean = re.sub(r'\bendmodule\b', '', clean)
        clean = re.sub(r'\b(?:input|output|wire|reg)\b[^;]*;', '', clean)
        ops = []
        self.gate_count = 0
        for statement in clean.split(';'):
            statement = statement.strip()
            if not statement:
                continue
            assign = re.fullmatch(r'assign\s+([^=]+)=([^=]+)', statement)
            if assign:
                dest, src = map(net, assign.groups())
                if not isinstance(dest, str):
                    raise ValueError("Constant destination")
                if isinstance(src, int):
                    ops.append((dest, (), (src,), 0))
                elif dest != src:
                    raise ValueError("Unresolved alias")
                continue
            m = re.fullmatch(r'(\\\S+|\w+)\s+(\\\S+|\w+)\s*\((.*)\)', statement, re.S)
            if not m or m[1] not in gates or re.search(r'DFF|LATCH|SDFF', m[1], re.I):
                raise ValueError(f"Unsupported cell/statement: {statement[:100]}")
            gate, _, connections = m.groups()
            pairs = re.findall(r'\.(\w+)\s*\(\s*([^()]+?)\s*\)', connections)
            residue = re.sub(r'\.\w+\s*\(\s*[^()]+?\s*\)', '', connections)
            if residue.replace(',', '').strip() or len(dict(pairs)) != len(pairs):
                raise ValueError("Unsupported or duplicate cell pins")
            pins = dict(pairs)
            outputs = set(pins) & set(gates[gate])
            if not outputs:
                raise ValueError("Cell has no recognized connected output")
            self.gate_count += 1
            for output in sorted(outputs):
                lut, symbols = compile_cell(gate, output, gates[gate][output])
                if not isinstance(lut, list) or len(symbols) > 8 or any(s not in pins for s in symbols):
                    raise ValueError("Unsupported cell truth table or missing pin")
                mapped = [net(pins[s]) for s in symbols]
                deps = tuple(dict.fromkeys(x for x in mapped if isinstance(x, str)))
                table = []
                for i in range(1 << len(deps)):
                    bits = {n: (i >> j) & 1 for j, n in enumerate(deps)}
                    index = sum((bits[x] if isinstance(x, str) else x) << j for j, x in enumerate(mapped))
                    table.append(lut[index])
                dest = net(pins[output])
                if not isinstance(dest, str):
                    raise ValueError("Constant cell destination")
                ops.append((dest, deps, tuple(table), 1))
        producers = [op[0] for op in ops]
        if len(set(producers)) != len(producers) or set(producers) & set(self.inputs):
            raise ValueError("Multiple drivers or driven primary input")
        known = set(self.inputs)
        self.ops = []
        while ops:
            ready = [op for op in ops if set(op[1]) <= known]
            if not ready:
                raise ValueError("Cycle or undriven cell input")
            for op in ready:
                self.ops.append(op)
                known.add(op[0])
                ops.remove(op)
        if not set(self.outputs) <= known:
            raise ValueError("Undriven primary output")
        self.cc = {n: (1, 1) for n in self.inputs}
        self.depth = dict.fromkeys(self.inputs, 0)
        self.support = {n: {n} for n in self.inputs}
        self.fanout = Counter()
        self.reconvergent_gates = 0
        for out, deps, lut, cost in self.ops:
            costs = [self.cc[n] for n in deps]
            self.cc[out] = tuple(cube_cost(c, costs) + cost for c in forcing_cubes(lut))
            self.depth[out] = max((self.depth[n] for n in deps), default=0) + cost
            self.support[out] = set().union(*(self.support[n] for n in deps))
            self.reconvergent_gates += int(any(self.support[a] & self.support[b] for a, b in itertools.combinations(deps, 2)))
            self.fanout.update(deps)
        self.co = dict.fromkeys(known, INF)
        self.distance = dict.fromkeys(known, INF)
        for n in self.outputs:
            self.co[n] = self.distance[n] = 0
        for out, deps, lut, cost in reversed(self.ops):
            for j, n in enumerate(deps):
                others = deps[:j] + deps[j+1:]
                derivative = []
                for i in range(1 << len(others)):
                    lo = (i & ((1 << j)-1)) | ((i >> j) << (j+1))
                    derivative.append(lut[lo] ^ lut[lo | (1 << j)])
                sensitization = cube_cost(forcing_cubes(tuple(derivative))[1], [self.cc[x] for x in others])
                self.co[n] = min(self.co[n], self.co[out] + sensitization + cost)
                self.distance[n] = min(self.distance[n], self.distance[out] + cost)

    def simulate(self, values, mask, fault=None):
        values = dict(values)
        if fault and fault[0] in values:
            values[fault[0]] = mask if fault[1] else 0
        for out, deps, lut, _ in self.ops:
            values[out] = (mask if fault[1] else 0) if fault and out == fault[0] else apply_lut(lut, [values[n] for n in deps], mask)
        return values

    def characterize(self, fault, *, samples, exact_max_inputs, seed):
        m = re.fullmatch(r'sa([01])\s+(.+)', fault.strip())
        if not m or m[2] not in self.alias:
            raise ValueError("Expected a declared net stem sa0/sa1; pin branches unsupported")
        stuck, original = int(m[1]), m[2]
        site = self.alias[original]
        if site not in self.cc:
            raise ValueError("Undriven fault location")
        exact = len(self.inputs) <= exact_max_inputs
        trials = (1 << len(self.inputs)) if exact else samples
        mask = (1 << trials) - 1
        rng = random.Random(seed)
        values = {n: (sum(((i >> j) & 1) << i for i in range(trials)) if exact else rng.getrandbits(trials))
                  for j, n in enumerate(self.inputs)}
        good = self.simulate(values, mask)
        bad = self.simulate(values, mask, (site, stuck))
        activation_mask = good[site] if stuck == 0 else mask ^ good[site]
        detection_mask = 0
        for n in self.outputs:
            detection_mask |= good[n] ^ bad[n]
        if detection_mask & (mask ^ activation_mask):
            raise AssertionError("Detection without activation")
        a, d = activation_mask.bit_count(), detection_mask.bit_count()
        p = d/trials
        cc = self.cc[site][1-stuck]
        return dict(status="ok", fault_type=f"sa{stuck}", fault_location=original,
            location_class="primary_input" if site in self.inputs else "primary_output" if site in self.outputs else "internal",
            alias_group=sorted(n for n in self.alias if self.alias[n] == site),
            num_inputs=len(self.meta.input_nets), num_outputs=len(self.meta.output_nets),
            num_output_signals=len(self.outputs), gate_count=self.gate_count,
            circuit_depth=max(self.depth[n] for n in self.outputs), site_depth=self.depth[site],
            distance_to_output=finite(self.distance[site]), fanin_support=len(self.support[site]),
            fanout=self.fanout[site], reconvergent_gates=self.reconvergent_gates,
            cc0=finite(self.cc[site][0]), cc1=finite(self.cc[site][1]), activation_cost=finite(cc),
            co=finite(self.co[site]), scoap_cost=finite(cc+self.co[site]),
            structural_unreachable=not math.isfinite(cc+self.co[site]),
            probability_method="exhaustive" if exact else "monte_carlo", random_trials=trials,
            random_activations=a, random_detections=d, activation_probability=a/trials,
            conditional_observability=d/a if a else None,
            random_detection_probability=p,
            random_detection_ci95=[p,p] if exact else wilson(d,trials),
            activation_ci95=[a/trials]*2 if exact else wilson(a,trials),
            conditional_observability_ci95=([d/a]*2 if exact else wilson(d,a)) if a else None,
            difficulty_bits=-math.log2(p) if p else None,
            zero_detection_status=("proven_undetectable_exhaustive" if exact else "unresolved_zero_hits") if not d else None,
            expected_random_vectors=1/p if p else None)


class Analyzer:
    def __init__(self, config=DEFAULT_CONFIG, samples=4096, exact_max_inputs=12, seed=1729):
        if samples < 1 or not 0 <= exact_max_inputs <= 16:
            raise ValueError("Require samples >= 1 and 0 <= exact_max_inputs <= 16")
        path = Path(config)
        self.gates = json.loads(path.read_text())["gate_funcs"]
        self.config_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.samples, self.exact_max_inputs, self.seed = samples, exact_max_inputs, seed
        self.cache = OrderedDict()

    def analyze(self, record):
        text = raw_netlist(record)
        cid = digest(text)
        base = dict(problem_id=identity(record), circuit_id=cid, module_name=record.get("module_name", ""),
                    fault=record.get("fault", ""), version=VERSION)
        try:
            if cid not in self.cache:
                self.cache[cid] = Circuit(text, self.gates)
                if len(self.cache) > 128:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(cid)
            result = self.cache[cid].characterize(base["fault"], samples=self.samples,
                exact_max_inputs=self.exact_max_inputs, seed=self.seed ^ int(cid[:16],16))
            return {**base, **result}
        except (ValueError, KeyError, TypeError) as exc:
            match = re.fullmatch(r"sa([01])\s+(.+)", str(base["fault"]).strip())
            return {**base, "status": "unsupported", "error": str(exc),
                    "fault_type": f"sa{match[1]}" if match else "unknown",
                    "fault_location": match[2] if match else "unknown"}

    def provenance(self):
        return dict(version=VERSION, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            gate_config_sha256=self.config_hash, random_samples=self.samples,
            exact_max_inputs=self.exact_max_inputs, seed=self.seed,
            backend="independent packed Boolean simulation; physical alias stems; not native TetraMAX",
            constraints="combinational, unconstrained independent binary PIs, all POs observed, single net stem stuck-at")


def difficulty_bin(row):
    if row.get("status") != "ok":
        return "unknown"
    if row.get("zero_detection_status"):
        return row["zero_detection_status"]
    p = row["random_detection_probability"]
    return "easy_p>=0.25" if p >= .25 else "medium_0.0625<=p<0.25" if p >= .0625 else "hard_0.00390625<=p<0.0625" if p >= 1/256 else "very_hard_p<0.00390625"


def select_records(stream, limit=512, *, mode="uniform_faults", seed=42, pool_size=4096, eligible=None, analyzer=None):
    """Reservoir over unique eligible (netlist text, fault) pairs; entire stream.

    Stratification balances (polarity, location class, empirical difficulty) in
    the reservoir. Its distribution is NOT a population detection estimate.
    """
    if limit == 0 or limit < -1 or pool_size < 1 or (limit > pool_size and mode == "stratified"):
        raise ValueError("Require limit=-1 or positive, positive pool_size, and stratified limit <= pool_size")
    if mode not in ("uniform_faults", "stratified", "legacy_prefix"):
        raise ValueError("Unknown selection mode")
    rng, seen, pool = random.Random(seed), set(), []
    counts = Counter()
    capacity = None if limit == -1 else (limit if mode != "stratified" else pool_size)
    for record in stream:
        counts["source_rows"] += 1
        key = digest(raw_netlist(record)) if mode == "legacy_prefix" else identity(record)
        if key in seen:
            counts["duplicate_rows"] += 1
            continue
        seen.add(key)
        if eligible is not None and not eligible(record):
            counts["ineligible_rows"] += 1
            continue
        counts["eligible_unique_problems"] += 1
        n = counts["eligible_unique_problems"]
        if capacity is None or len(pool) < capacity:
            pool.append(dict(record))
        elif mode == "legacy_prefix":
            break
        else:
            j = rng.randrange(n)
            if j < capacity:
                pool[j] = dict(record)
        if mode == "legacy_prefix" and capacity and len(pool) == capacity:
            break
    if not pool:
        raise ValueError("No eligible evaluation problems")
    audit = dict(mode=mode, seed=seed, counts=dict(counts), candidate_pool_size=len(pool),
        source_scan_complete=mode != "legacy_prefix", deduplication="netlist" if mode == "legacy_prefix" else "netlist+fault",
        population_estimate=mode == "uniform_faults")
    if mode == "stratified":
        if analyzer is None:
            raise ValueError("Stratified selection requires an analyzer")
        groups = defaultdict(list)
        for record in pool:
            row = analyzer.analyze(record)
            record["_difficulty"] = row
            groups[(row.get("fault_type", "unknown"), row.get("location_class", "unknown"), difficulty_bin(row))].append(record)
        audit["candidate_strata"] = {"|".join(k): len(v) for k,v in sorted(groups.items())}
        keys = sorted(groups)
        rng.shuffle(keys)
        for group in groups.values():
            rng.shuffle(group)
        selected = []
        while keys and (limit == -1 or len(selected) < limit):
            for key in list(keys):
                if limit != -1 and len(selected) == limit:
                    break
                selected.append(groups[key].pop())
                if not groups[key]:
                    keys.remove(key)
        pool = selected
    rng.shuffle(pool) if mode != "legacy_prefix" else None
    audit["selected_problems"] = len(pool)
    audit["selected_ids_sha256"] = digest([identity(r) for r in pool])
    return pool, audit


def save_manifest(path, records, selection):
    examples = [{k:v for k,v in r.items() if k != "_difficulty"} for r in records]
    payload = dict(version=VERSION, examples=examples, examples_sha256=digest(examples), selection=selection)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(payload, out, indent=2)
    return payload


def load_manifest(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("version") != VERSION or digest(payload["examples"]) != payload["examples_sha256"]:
        raise ValueError("Evaluation manifest version/checksum mismatch")
    if not payload["examples"]:
        raise ValueError("Empty evaluation manifest")
    if len({identity(r) for r in payload["examples"]}) != len(payload["examples"]):
        raise ValueError("Duplicate problems in evaluation manifest")
    return payload["examples"], payload["selection"]


def outcome_counts(rewards, n):
    def acc(r, key):
        return r.get(key, r.get(key+"_logonly", 0)) == 1
    counts = Counter(dict(D=0, failed_slots=max(0,n-len(rewards))))
    for r in rewards[:n]:
        detected = acc(r, "fault_detected_by_pred_input_vector_acc")
        failed = r.get("search_failure_logonly",0) > 0 or r.get("simulator_error_logonly",0) > 0
        counts.update(dict(D=int(detected and not failed), failed_slots=int(failed)))
    return dict(counts)


def pass_k(n, c, k):
    return 1. if n-c < k else 1-math.prod(1-k/j for j in range(n-c+1,n+1))


def numeric_bin(value, edges):
    if value is None:
        return "unknown_or_infinite"
    low = 0
    for high in edges:
        if value <= high:
            return str(high) if low == high else f"{low}..{high}"
        low = high+1
    return f">{edges[-1]}"


def strata(row):
    return dict(circuit=row["circuit_id"], fault_type=row.get("fault_type","unknown"),
        fault_location=row["circuit_id"]+":"+row.get("fault_location", row.get("fault","unknown")),
        location_class=row.get("location_class","unknown"), difficulty=difficulty_bin(row),
        site_depth=numeric_bin(row.get("site_depth"),(0,1,2,4,8,16)),
        circuit_depth=numeric_bin(row.get("circuit_depth"),(0,1,2,4,8,16)),
        activation_cost=numeric_bin(row.get("activation_cost"),(1,2,4,8,16,32,64)),
        observability_cost=numeric_bin(row.get("co"),(0,1,2,4,8,16,32,64)),
        scoap_cost=numeric_bin(row.get("scoap_cost"),(1,2,4,8,16,32,64)),
        gate_count=numeric_bin(row.get("gate_count"),(1,2,4,8,16,32,64,128,256)),
        num_inputs=numeric_bin(row.get("num_inputs"),(0,1,2,4,8,16,32,64,128)),
        num_outputs=numeric_bin(row.get("num_outputs"),(0,1,2,4,8,16,32,64,128)),
        distance_to_output=numeric_bin(row.get("distance_to_output"),(0,1,2,4,8,16)),
        activation_probability=probability_bin(row.get("activation_probability")),
        conditional_observability=probability_bin(row.get("conditional_observability")))


def probability_bin(p):
    if p is None:
        return "unknown"
    if p == 0:
        return "0"
    for low, high in ((0,1/256),(1/256,1/16),(1/16,1/4),(1/4,1/2),(1/2,1)):
        if p <= high:
            return f"({low},{high}]"
    raise ValueError("Invalid probability")


def bootstrap_mean_ci(values, seed=1729, repeats=1000):
    # One value per circuit; this is not an interval across training runs/families.
    if len(values) < 2:
        return None
    import numpy as np
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    # Bound working memory even when MAX_EVAL_SAMPLES=-1 is used.
    batch_size = max(1, min(100, 1_000_000 // len(values)))
    means = []
    for start in range(0, repeats, batch_size):
        means.extend(rng.choice(values, size=(min(batch_size,repeats-start),len(values))).mean(axis=1).tolist())
    means.sort()
    return [means[int(.025*repeats)], means[int(.975*repeats)]]


def summarize(rows, k_values=(1,)):
    """Problem micro and circuit macro averages; missing/failed slots stay in n."""
    def aggregate(group):
        measured = [r for r in group if r.get("status") == "ok"]
        scored = [r for r in group if r.get("num_completions",0) > 0]
        result = dict(problems=len(group), circuits=len({r["circuit_id"] for r in group}),
            characterized=len(measured), unsupported=len(group)-len(measured),
            scored_problems=len(scored), slots=sum(r.get("num_completions",0) for r in scored),
            failed_slots=sum(r.get("outcomes",{}).get("failed_slots",0) for r in scored))
        usage = Counter()
        for row in group:
            usage.update(row.get("saved_search_usage", {}))
        result["search_usage"] = dict(usage)
        if measured:
            result["random_single_vector_detection"] = sum(r["random_detection_probability"] for r in measured)/len(measured)
        for metric in ("D",):
            for k in k_values:
                eligible = [r for r in scored if r["num_completions"] >= k]
                if not eligible:
                    continue
                vals = [pass_k(r["num_completions"], r["outcomes"][metric], k) for r in eligible]
                circuits = defaultdict(list)
                for r, v in zip(eligible, vals):
                    circuits[r["circuit_id"]].append(v)
                circuit_means = [sum(v)/len(v) for v in circuits.values()]
                result[f"{metric}_pass@{k}"] = dict(problem_mean=sum(vals)/len(vals),
                    circuit_macro=sum(circuit_means)/len(circuit_means),
                    circuit_macro_bootstrap_ci95=bootstrap_mean_ci(circuit_means), problems=len(eligible))
                paired = [(r,v) for r,v in zip(eligible,vals) if r.get("status") == "ok"]
                if metric == "D" and paired:
                    random_mean = sum(1-(1-r["random_detection_probability"])**k for r,v in paired)/len(paired)
                    result[f"uniform_random_{k}_vectors"] = random_mean
                    result[f"D_lift_over_random_{k}_vectors"] = sum(v for r,v in paired)/len(paired)-random_mean
                    result[f"paired_problems@{k}"] = len(paired)
                    gaps = defaultdict(list)
                    for r,v in paired:
                        gaps[r["circuit_id"]].append(v-(1-(1-r["random_detection_probability"])**k))
                    macro_gaps = [sum(v)/len(v) for v in gaps.values()]
                    result[f"D_macro_lift_over_random_{k}_vectors"] = sum(macro_gaps)/len(macro_gaps)
                    result[f"D_macro_lift_bootstrap_ci95@{k}"] = bootstrap_mean_ci(macro_gaps)
        return result
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        labels = strata(row)
        for dimension in ("num_inputs", "num_outputs"):
            labels[dimension + "_difficulty"] = labels[dimension] + "|" + labels["difficulty"]
        labels["input_output_counts"] = labels["num_inputs"] + "|" + labels["num_outputs"]
        labels["type_location_difficulty"] = "|".join(labels[k] for k in ("fault_type","location_class","difficulty"))
        for dimension, label in labels.items():
            groups[dimension][label].append(row)
    return dict(version=VERSION,
        outcome_metric="D: good and faulty circuits differ at at least one primary output for the applied input vector; predicted expected outputs do not affect detection",
        overall=aggregate(rows),
        by={d:{label:aggregate(group) for label,group in sorted(g.items())} for d,g in groups.items()},
        interpretation="Random k means k independent uniform PI vectors, not k search completions. Search/tool budgets must be compared separately. SCOAP is heuristic. Zero Monte Carlo hits are not proof of untestability. Scoring semantics/backend are declared in provenance. Intervals bootstrap circuits, not design families or training seeds; lift intervals condition on estimated random probabilities.")


def write_report(prefix, rows, provenance, k_values=(1,)):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Accept historical rows but publish only the requested detection outcome.
    rows = [{**r, "outcomes": {key: value for key, value in r["outcomes"].items()
             if key in ("D", "failed_slots")}} if "outcomes" in r else dict(r) for r in rows]
    report = {"provenance": provenance, **summarize(rows,k_values)}
    prefix.with_suffix(".json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    # Easy to hard; unresolved and unsupported last. Do not invent a finite
    # difficulty for zero hits. Numeric metrics remain available for re-sorting.
    ordered = sorted(rows,key=lambda r:(r.get("status") != "ok", r.get("random_detection_probability",-1)*-1,
                                        r.get("scoap_cost") or INF, r["problem_id"]))
    prefix.with_suffix(".problems.jsonl").write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in ordered))
    flat = []
    for rank, row in enumerate(ordered,1):
        item = {k:v for k,v in row.items() if not isinstance(v,(dict,list))}
        item.update(difficulty_rank=rank,difficulty_bin=difficulty_bin(row))
        n = row.get("num_completions",0)
        for key,value in row.get("outcomes",{}).items():
            item[key+"_rate" if key != "failed_slots" else key] = value/n if n and key != "failed_slots" else value
        flat.append(item)
    with prefix.with_suffix(".problems.csv").open("w") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in flat for k in row)))
        writer.writeheader(); writer.writerows(flat)
    return report
