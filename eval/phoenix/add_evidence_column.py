"""Add evidence_file column to an existing evaluation CSV without modifying it."""
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='Path to existing evaluation CSV')
args = parser.parse_args()
source = Path(args.dataset)
try:
    with source.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
except UnicodeDecodeError:
    with source.open('r', newline='', encoding='cp1252') as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
if not rows:
    raise ValueError('CSV contains no rows')
if 'evidence_file' not in columns:
    columns.append('evidence_file')
for row in rows:
    row.setdefault('evidence_file', '')
dest = source.with_name(source.stem + '_with_evidence.csv')
if dest.exists():
    raise FileExistsError(f'Output exists; refusing to overwrite: {dest}')
with dest.open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
print(f'Created {dest} with {len(rows)} cases. Fill evidence_file for each case before running all three modes.')
