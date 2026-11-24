"""Apply the current plan filter to the failed first call of every row in a built retry set.

A synthetic retry row starts from a dataset row whose reasoning spells out the
detecting vector, while its first call carries a perturbed one, so the builder
keeps only the plan of that reasoning (``strip_assignment_lines``). When the
filter changes, the rows can be brought up to date without confirming their
paths again with TetraMAX: only the text of the first call changes, that call
is context (never trained), and the new filter drops a superset of the lines
the old one dropped. The thinking opener that the old filter could drop with
the first line is put back.

    python scripts/retry/restrip_plans.py --retry runs/v5r_schedule/retry/retry.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atpgllm.training.retry_trajectories import strip_assignment_lines  # noqa: E402


def restrip(content: str) -> str:
    """The first call's text under the current filter, with its thinking opener."""
    if "</think>" in content and not content.startswith("<think>"):
        content = "<think>" + content
    return strip_assignment_lines(content)


def restrip_row(row: dict, stats: Counter) -> dict:
    index = next(i for i, m in enumerate(row["messages"]) if m["role"] == "assistant")
    if row.get("assistant_train") and row["assistant_train"][0]:
        raise SystemExit("the first call of a retry row is trained; it must stay context only")
    old = row["messages"][index].get("content") or ""
    new = restrip(old)
    stats["rows"] += 1
    stats["changed"] += new != old
    stats["opener_restored"] += not old.startswith("<think>")
    stats["lines_dropped"] += old.count("\n") - new.count("\n")
    row["messages"][index]["content"] = new
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--retry", required=True, help="retry.jsonl written by build_retry_sft")
    parser.add_argument("--keep", default=None,
                        help="where the rows before this pass are kept (default: <retry>.before_restrip)")
    args = parser.parse_args()
    source = Path(args.retry)
    keep = Path(args.keep) if args.keep else source.with_name(source.name + ".before_restrip")
    if keep.exists():
        raise SystemExit(f"{keep} exists: this set was re-stripped already, start from that file")
    os.replace(source, keep)
    stats = Counter()
    with open(keep) as rows, open(source, "w") as sink:
        for line in rows:
            sink.write(json.dumps(restrip_row(json.loads(line), stats)) + "\n")
    report = source.with_name("report.json")
    if report.exists():
        data = json.loads(report.read_text())
        data["restrip"] = dict(stats, kept_as=str(keep.name))
        report.write_text(json.dumps(data, indent=1))
    print(json.dumps(dict(stats)))


if __name__ == "__main__":
    main()
