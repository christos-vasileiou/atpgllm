"""Stratified selection of the frozen evaluation set and its protocol."""
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from atpgllm.training.fixed_eval import (digest, family_of, fixed_eval_protocol, load_or_create_manifest,
                                         select_stratified)

LEVELS = ("easy", "hard")


def rows(families, netlists=3, faults=10, level="easy"):
    return [{"difficulty": level, "module_name": f"{family}_{n}", "netlist_id": f"{level}-{family}-{n}", "fault": f"sa0 n{i}"}
            for family in families for n in range(netlists) for i in range(faults)]


def test_family_of_groups_generator_siblings():
    assert family_of("snax_req_handler_equal_1u_22_1") == "snax_req_handler_equal"
    assert family_of("snax_req_handler_equal_1u_13_1") == "snax_req_handler_equal"
    assert family_of("wideexpr_00956") == family_of("wideexpr_00881") == "wideexpr"
    assert family_of("top_module") == "top_module"


def test_quota_family_cap_and_determinism():
    families = [f"f{chr(97 + i)}" for i in range(10)]
    pool = rows(families) + rows(families, level="hard")
    chosen = select_stratified(pool, {"easy": 20, "hard": 20}, 1, seen_families=families[:5])
    assert Counter(r["difficulty"] for r in chosen) == {"easy": 20, "hard": 20}
    assert len({(r["netlist_id"], r["fault"]) for r in chosen}) == 40
    for level in LEVELS:
        counts = Counter(family_of(r["module_name"]) for r in chosen if r["difficulty"] == level)
        assert max(counts.values()) <= 3  # 15% of 20
        assert sum(counts[f] for f in families[:5]) == 10  # half from seen families
    assert chosen == select_stratified(pool, {"easy": 20, "hard": 20}, 1, seen_families=families[:5])
    assert chosen != select_stratified(pool, {"easy": 20, "hard": 20}, 2, seen_families=families[:5])


def test_netlists_of_a_family_take_turns():
    chosen = select_stratified(rows(["fa", "fb"], netlists=4), {"easy": 8}, 3)
    assert Counter(r["netlist_id"] for r in chosen).most_common(1)[0][1] <= 1


def test_short_group_and_cap_give_way_before_the_quota_does():
    pool = rows(["seen"], netlists=1, faults=30) + rows(["only"], netlists=1, faults=4)
    chosen = select_stratified(pool, {"easy": 20}, 1, seen_families=["seen"])
    assert len(chosen) == 20
    with pytest.raises(ValueError, match="eligible problems"):
        select_stratified(rows(["fa"], netlists=1, faults=3), {"easy": 5}, 1)


def test_rejected_rows_are_skipped():
    pool = rows(["fa", "fb"])
    chosen = select_stratified(pool, {"easy": 6}, 1, eligible=lambda r: not r["fault"].endswith("0"))
    assert len(chosen) == 6 and all(not r["fault"].endswith("0") for r in chosen)


TOKENIZER = SimpleNamespace(get_vocab=lambda: {"a": 1}, special_tokens_map={"eos": "x"}, chat_template="t")
ARGS = dict(dataset="d", split="validation", size=250, seed=1729, tokenizer=TOKENIZER, max_prompt_length=8192,
            max_completion_length=24576, max_model_len=32768, generations=1, per_device_batch_size=1, world_size=2,
            holdout_policy="exclude_evaluation_circuits", rolling_buffer=True)


def test_selection_is_absent_from_the_legacy_protocol():
    assert "selection" not in fixed_eval_protocol(**ARGS)
    assert fixed_eval_protocol(**ARGS, selection="stratified_family_v2")["selection"] == "stratified_family_v2"


def test_stored_legacy_manifest_has_exactly_the_shared_protocol_keys():
    path = Path(__file__).resolve().parents[3] / "libatpgllm/runs/grpo_granite_4.2_8b_v4_a100_tetramax/fixed_eval_manifest.json"
    if not path.exists():
        pytest.skip("stopped GDPO run not present")
    stored = json.loads(path.read_text())["protocol"]
    assert set(fixed_eval_protocol(**ARGS)) == set(stored)


def test_built_manifest_is_accepted_and_a_changed_protocol_refused(tmp_path):
    protocol = fixed_eval_protocol(**ARGS, selection="stratified_family_v2")
    examples = [{"prompt": "p", "fault": "sa0 a", "netlist": "n", "_fixed_eval_id": "i"}]
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"version": 1, "protocol": protocol, "examples": examples,
                                "examples_sha256": digest(examples)}))
    never = lambda: pytest.fail("a frozen manifest must not be selected again")
    assert load_or_create_manifest(path, protocol, never)["examples"] == examples
    with pytest.raises(ValueError, match="protocol changed"):
        load_or_create_manifest(path, fixed_eval_protocol(**ARGS), never)


def test_builder_defaults_match_the_after_rft_config():
    repo = Path(__file__).resolve().parents[2]
    conf = (repo / "scripts/train/configs/grpo_granite_4.2_8b_v4_a100_tetramax_after_rft.conf").read_text()
    builder = (repo / "scripts/rft/build_eval_manifest.py").read_text()

    def value(key):
        import re
        return re.search(rf"^{key}=(.*)$", conf, re.M).group(1).strip()

    assert value("FIXED_EVAL_SIZE") == "250" and value("FIXED_EVAL_SELECTION") == "stratified_family_v2"
    assert f'selection="{value("FIXED_EVAL_SELECTION")}"' in builder
    assert f'"--dataset", default="{value("TRAIN_DATASET")}"' in builder
    assert f'"--split", default="{value("FIXED_EVAL_SPLIT")}"' in builder
    for key, flag in (("MAX_PROMPT_LENGTH", "max_prompt_length"), ("MAX_COMPLETION_LENGTH", "max_completion_length"),
                      ("MAX_MODEL_LEN", "max_model_len"), ("FIXED_EVAL_SEED", "seed")):
        assert f'"--{flag}", type=int, default={value(key)}' in builder
    assert f'"--backend", default="{value("FAULT_SIM_BACKEND")}"' in builder
