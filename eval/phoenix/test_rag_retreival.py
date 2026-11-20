"""Offline retrieval smoke test; does not call Databricks or MCP."""
import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'src'))
from example_server.rag.fda_rag import search_pdf_file

RAG_IDS = {'7', '8', '23', '24', '25', '26', '32', '33', '34', '35'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=str(HERE / 'golden_fda_eval_with_evidence.csv'))
    parser.add_argument('--pdf', default=str(HERE / 'evidence' / 'opioid_label_updates.pdf'))
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    with open(args.dataset, encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r.get('case_id', '').strip() in RAG_IDS]
    if len(rows) != 10:
        raise SystemExit(f'Expected 10 RAG questions, found {len(rows)}. Check dataset: {args.dataset}')
    out = HERE / 'results' / 'rag_retrieval_audit.csv'
    out.parent.mkdir(exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id','query','rank','page','table_heading','score','chunk_id','passage_preview'])
        writer.writeheader()
        for row in rows:
            matches = search_pdf_file(args.pdf, row['query'], args.top_k)
            print(f"case {row['case_id']}: {len(matches)} matches, pages {[m['page'] for m in matches]}")
            for rank, m in enumerate(matches, 1):
                writer.writerow({'case_id':row['case_id'], 'query':row['query'], 'rank':rank,
                    'page':m['page'], 'table_heading':m['table_heading'], 'score':m['score'],
                    'chunk_id':m['chunk_id'], 'passage_preview':m['text'][:350]})
    print('Saved', out)

if __name__ == '__main__':
    main()
