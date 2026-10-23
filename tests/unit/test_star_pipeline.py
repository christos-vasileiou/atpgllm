"""rationalize.py's selection and write loop, then build_sft.py, with a scripted LLM.

The simulator is replaced by the dataset's stored TetraMAX snapshot (the row's
pattern detects its fault), so this runs on CPU without TetraMAX seats.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
V4 = REPO.parent / "data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.v4"


@pytest.mark.skipif(not (V4 / "train").is_dir(), reason="needs the local v4 dataset")
def test_rationalization_job_path_writes_judged_slots_and_sft_examples(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "star"))
    import rationalize as R
    from atpgllm.training.rationalization import care_bits
    from atpgllm.training.reward_function_factory import RewardFunctionFactory
    from atpgllm.training._paths import resolve_sim_config_path
    from atpgllm.training.sampling_strategies import make_strategy
    from atpgllm.training.search_types import GenerationResult, SearchConfig
    from atpgllm.training.search_verifier import Verifier
    try:
        tok = transformers.AutoTokenizer.from_pretrained("ibm-granite/granite-4.2-8b")
    except OSError:
        pytest.skip("Granite tokenizer unavailable")
    OUT = tmp_path
    problems, taken = R.prepare_problems(
        V4,
        {"hard": 1, "random_never_detects": 1}, 1729, tok, 8192)
    by_doc = {hashlib.sha256(row["netlist"].encode()).hexdigest()[:16]: row for row, _, _ in problems}

    def call_text(row, doc):
        return ("Tracing the cone.</think>\n<tool_call>\n<function=fault_simulation_tool>\n"
                f"<parameter=input_vector>\n{json.dumps(json.loads(row['input_vector']))}\n</parameter>\n"
                f"<parameter=output_vector>\n{json.dumps(json.loads(row['expected_output']))}\n</parameter>\n"
                f"<parameter=fault>\n{row['fault']}\n</parameter>\n<parameter=doc_id>\n{doc}\n</parameter>\n"
                "</function>\n</tool_call>")

    class Scripted:
        temperature, top_p, max_tokens, context_limit = 1.0, 1.0, 4096, 32768
        tokenizer = tok
        def generate_requests(self, requests):
            out = []
            for r in requests:
                doc = next(d for d in by_doc if d in r.prompt)
                row = by_doc[doc]
                if "<tool_response>" not in r.prompt:
                    text = call_text(row, doc)
                else:
                    bits = ", ".join(f"{k}: {v}" for k, v in care_bits(row).items())
                    outs = ", ".join(f"{k}: {v}" for k, v in json.loads(row["expected_output"]).items() if v != "x")
                    leak = " As the hint said." if "Private note" in r.prompt and row["difficulty"] == "hard" else ""
                    text = (f"The outputs differ.{leak}</think>\n\nINPUT_VECTOR: \"{bits}\"\n"
                            f"EXPECTED_OUTPUT: \"{outs}\"\nDETECTED_FAULTS: \"{row['fault']}\"")
                out.append(GenerationResult(text, tuple(tok.encode(text, add_special_tokens=False)), "stop", None, 0))
            return out

    class SnapshotVerifier(Verifier):
        def simulate(self, problem, vector, context):
            row = by_doc[problem.doc_id]
            frame = pd.DataFrame(json.loads(row["snapshot"]))
            # The columns tetramax_backend.as_frame adds next to the machine values.
            frame["PIs"] = frame.index.isin(problem.input_nets)
            frame["POs"] = frame.index.isin(problem.output_nets)
            return frame, {"tetramax_available": True, "tetramax_detected": True,
                           "tetramax_indeterminate": False, "backend": "tetramax"}

    verifier = SnapshotVerifier(RewardFunctionFactory(config_path=str(resolve_sim_config_path())))
    strategy = make_strategy("single_completion", Scripted(), verifier, num_completions=2, use_tools=True,
                             max_tool_rounds=4, seed=1729, search_config=SearchConfig(max_generated_tokens=24576))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.json").write_text(json.dumps({"base_model": "ibm-granite/granite-4.2-8b"}))
    with open(OUT / "slots.jsonl", "a") as sink:
        R.sample_and_write(strategy, problems, ["none", "partial"], tok, sink, seed=1729, batch_problems=64)
    slots = [json.loads(line) for line in open(OUT / "slots.jsonl")]
    assert len(slots) == 2 * 2 * 2 and all(s["status"] == "FINAL" and s["facts"]["detected"] for s in slots)
    # The scripted model mentions the hint only on the hinted hard problem.
    assert {(s["hint_mode"], s["difficulty"]) for s in slots if s["facts"]["leak"]} == {("partial", "hard")}
    subprocess.run([sys.executable, str(REPO / "scripts/star/build_sft.py"), str(OUT)], check=True,
                   capture_output=True, env={**__import__("os").environ, "PYTHONPATH": str(REPO)})
    report = json.loads((OUT / "report.json").read_text())
    examples = [json.loads(line) for line in open(OUT / "sft.jsonl")]
    assert report["sft_examples"] == len(examples) == 1  # two identical RND answers, one kept
    assert examples[0]["star"]["difficulty"] == "random_never_detects"
    assert all("Private note" not in str(m["content"]) for m in examples[0]["messages"])
