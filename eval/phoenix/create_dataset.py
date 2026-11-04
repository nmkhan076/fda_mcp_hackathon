"""Validate the local golden dataset and optionally register it in Phoenix."""
import argparse
import csv
from pathlib import Path

DATA = Path(__file__).with_name("golden_fda_eval.csv")

def load_rows():
    with DATA.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"query", "responses", "scope", "expected_tool", "category"}
    assert rows and required.issubset(rows[0]), "Invalid golden dataset"
    assert all(r["query"] and r["responses"] for r in rows)
    assert all(r["scope"] in {"in_scope", "out_of_scope"} for r in rows)
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload-phoenix", action="store_true")
    ap.add_argument("--dataset-name", default="fda-eval-v01")
    args = ap.parse_args()
    rows = load_rows()
    print(f"Validated {len(rows)} evaluation questions")
    if args.upload_phoenix:
        from phoenix.client import Client
        client = Client()
        client.datasets.create_dataset(
            name=args.dataset_name,
            inputs=[{"query": r["query"], "category": r["category"]} for r in rows],
            outputs=[{"responses": r["responses"], "scope": r["scope"], "expected_tool": r["expected_tool"]} for r in rows],
        )
        print("Dataset uploaded to Phoenix")
