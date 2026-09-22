"""Read a bounded training shard from the pinned private HF dataset revision."""
import hashlib
import json
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
revision = json.loads((OUT / "dataset_metadata.json").read_text())["sha"]
manifest = json.loads((REPO / "runs/grpo_granite_4.2_8b/fixed_eval_manifest.json").read_text())
by_net = {e["netlist"]["netlist"]: e for e in manifest["examples"]}
by_problem = {(e["netlist"]["netlist"], e["fault"]): e for e in manifest["examples"]}
fs = HfFileSystem()
path = f"datasets/chrivasileiou/asap7-language-of-test-v2@{revision}/data/train-00000-of-00043.parquet"
with fs.open(path, "rb", block_size=1024 * 1024) as stream:
    file = pq.ParquetFile(stream)
    columns = file.schema_arrow.names
    print(json.dumps({"rows": file.metadata.num_rows, "columns": columns}), flush=True)
    chosen = [k for k in ("netlist", "fault", "input_vector", "expected_output", "reasoning_content", "answer_content", "system_content", "user_content", "module_name", "number_of_gates", "gates") if k in columns]
    rows = file.read(columns=chosen).to_pylist()
counts = Counter()
matches, problem_matches, representatives = {}, {}, []
reasoning, answers = Counter(), Counter()
for index, row in enumerate(rows):
    net = row["netlist"]
    if net in by_net:
        matches.setdefault(by_net[net]["_fixed_eval_id"], index)
    if (net, row["fault"]) in by_problem:
        key = by_problem[net,row["fault"]]["_fixed_eval_id"]
        problem_matches.setdefault(key, []).append({"row":index,"input_vector":row.get("input_vector"),"expected_output":row.get("expected_output")})
    reasoning[row.get("reasoning_content", "")] += 1
    answers[row.get("answer_content", "")] += 1
    if len(representatives) < 3:
        representatives.append(row)
result = {"revision":revision,"shard":path,"num_rows":len(rows), "columns":columns,
          "eval_circuits_found_in_train_shard":len(matches), "eval_exact_problems_found_in_train_shard":len(problem_matches),
          "circuit_matches":matches,"problem_matches":problem_matches,
          "unique_reasoning_templates":len(reasoning),"unique_answer_templates":len(answers),
          "top_reasoning_templates":reasoning.most_common(3),"top_answer_templates":answers.most_common(3),
          "representative_rows":representatives}
(OUT / "dataset_audit.json").write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ("circuit_matches","problem_matches","top_reasoning_templates","top_answer_templates","representative_rows")},indent=2))
