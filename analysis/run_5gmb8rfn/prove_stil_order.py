"""Independent Boolean proof of two corrupt labels and their STIL mapping cause.

Uses only Python's standard library, original TetraMAX files, and downloaded
dataset examples. Does not call the project's simulator.
"""
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
cases = json.loads((OUT / "reference_label_discrepancies.json").read_text())

def ex102(iv):
    out = {f"out3[{i}]": 1 - (iv[f"a[{i}]"] & iv[f"b[{i}]"]) for i in range(9)}
    out["out1"] = 1 - int(all(iv[f"a[{i}]"] for i in range(9)))
    out["out2"] = 1 - int(all(iv[f"b[{i}]"] for i in range(9)))
    return out

def clic(iv):
    p,s,t = (iv[k] for k in ("interrupt_plic_in","interrupt_soft_in","interrupt_time_in"))
    i,b,e = (iv[k] for k in ("exception_illegal_instruction","exception_breakpoint","exception_ecall_mmode"))
    out={f"{prefix}_code[{j}]":0 for prefix in ("interrupt","exception") for j in range(8)}
    out.update({"interrupt_out":p|s|t, "exception_out":i|b|e,
                "interrupt_code[0]":p|s|t, "interrupt_code[1]":p|s|t,
                "interrupt_code[2]":t & (1-s), "interrupt_code[3]":p & (1-t) & (1-s),
                "exception_code[0]":(1-i)&(e|b), "exception_code[1]":i|b|e,
                "exception_code[3]":e & (1-i) & (1-b)})
    return out

results=[]
for row, directory, fn in [(20451,"4077_ex_102_test_vector_and",ex102),(2038,"4128_core_c1_clic",clic)]:
    case=next(r for r in cases if r["row"]==row)
    path=ROOT / "data/freeset/out.freeset.asap7sc7p5t_28.rvt.tt" / directory / "simulation.stil"
    source=path.read_text()
    def order(name):
        group=re.search(r'"'+name+r'"\s*=\s*\x27(.*?)\x27',source,re.S)[1]
        return re.findall(r'"(.*?)"',group)
    pis,pos=order("_pi"),order("_po")
    patterns=re.findall(r'"pattern (\d+)": Call "capture" \{\s*"_pi"=([01]+); "_po"=([HL]+);',source)
    # The exported JSON retains the erroneous declaration/ascending-bit order.
    raw_input="".join(map(str,case["input_vector"].values()))
    raw_output="".join(map(str,case["expected_output_label"].values()))
    index,pi_bits,po_bits=next((n,iv,ov) for n,iv,ov in patterns if iv==raw_input and ov.translate(str.maketrans("HL","10"))==raw_output)
    corrected_inputs=dict(zip(pis,map(int,pi_bits)))
    corrected_outputs=dict(zip(pos,map(int,po_bits.translate(str.maketrans("HL","10")))))
    manual=fn(case["input_vector"])
    assert manual!=case["expected_output_label"]
    assert manual==case["simulated_good"]
    assert fn(corrected_inputs)==corrected_outputs
    results.append({"dataset_train_shard_row":row,"example_id":case["example_id"],"module":directory,
        "stil_path":str(path),"pattern":int(index),"stil_pi_order":pis,"stil_po_order":pos,
        "dataset_input_order":list(case["input_vector"]),"dataset_output_order":list(case["expected_output_label"]),
        "incorrect_label_bits":{k:{"label":v,"boolean_truth":manual[k]} for k,v in case["expected_output_label"].items() if v!=manual[k]},
        "dataset_label_disagrees_with_boolean_logic":True,"stil_order_repairs_all_outputs":True,
        "corrected_input_vector":corrected_inputs,"corrected_expected_output":corrected_outputs})
(OUT / "stil_order_proof.json").write_text(json.dumps(results,indent=2))
print(json.dumps([{k:r[k] for k in ("module","dataset_train_shard_row","pattern","incorrect_label_bits","stil_order_repairs_all_outputs")} for r in results],indent=2))
