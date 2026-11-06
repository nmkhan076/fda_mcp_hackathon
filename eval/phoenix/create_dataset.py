"""Validate the local golden dataset and optionally register it in Phoenix."""
import argparse
import csv
from pathlib import Path
import pprint
DATA = Path(__file__).with_name("golden_fda_eval.csv")

def load_rows():
    with DATA.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"case_id", "query", "ground_truth_answer", "source_locator", "scope", "expected_tool", "category"}
    assert rows and required.issubset(rows[0]), "Invalid golden dataset columns"
    assert all(r["query"] and r["ground_truth_answer"] for r in rows)
    assert all(r["scope"] in {"in_scope", "out_of_scope"} for r in rows)
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload-phoenix", action="store_true")
    ap.add_argument("--dataset-name", default="fda-eval-v02")
    args = ap.parse_args()
    rows = load_rows()
    for i, row in enumerate(rows, start=1):
        print(f"\n{'=' * 60}")
        print(f"Evaluation Question {i} --> {row['query']}")
        # pprint(row)
    print(f"Read {len(rows)} evaluation questions")
    if args.upload_phoenix:
        from phoenix.client import Client
        Client().datasets.create_dataset(
            name=args.dataset_name,
            inputs=[{"query": r["query"], "category": r["category"], "case_id": r["case_id"]} for r in rows],
            outputs=[{"ground_truth_answer": r["ground_truth_answer"], "source_locator": r["source_locator"], "scope": r["scope"], "expected_tool": r["expected_tool"]} for r in rows],
        )
        print("Uploaded to Phoenix")

