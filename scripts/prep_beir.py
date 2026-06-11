#!/usr/bin/env python3
"""prep_beir.py — fetch BeIR/scifact (corpus + queries + test qrels) and write a
compact testdata/beir-scifact/scifact.json the Go BEIR harness reads. SciFact: 5183
corpus docs, 300 test queries, binary qrels. Cross-referenceable: a standard BEIR
dataset + nDCG@10, so aikit's retrieval quality can be compared to published numbers.
"""
import csv, json, sys
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parent.parent / "testdata" / "beir-scifact" / "scifact.json"

def main() -> int:
    corpus = pd.read_parquet(hf_hub_download("BeIR/scifact", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset"))
    queries = pd.read_parquet(hf_hub_download("BeIR/scifact", "queries/queries-00000-of-00001.parquet", repo_type="dataset"))
    qrels_path = hf_hub_download("BeIR/scifact-qrels", "test.tsv", repo_type="dataset")

    cmap = {str(r["_id"]): (str(r.get("title", "") or "") + " " + str(r["text"])).strip() for _, r in corpus.iterrows()}
    qmap = {str(r["_id"]): str(r["text"]) for _, r in queries.iterrows()}
    qrels = {}
    with open(qrels_path) as f:
        rd = csv.reader(f, delimiter="\t"); next(rd)  # header: query-id corpus-id score
        for qid, did, score in rd:
            qrels.setdefault(qid, {})[did] = int(score)
    qmap = {q: t for q, t in qmap.items() if q in qrels}  # test queries only

    OUT.write_text(json.dumps({"dataset": "BeIR/scifact (test)", "corpus": cmap, "queries": qmap, "qrels": qrels}))
    sys.stderr.write(f"[prep_beir] wrote {OUT} — {len(cmap)} docs, {len(qmap)} test queries, {sum(len(v) for v in qrels.values())} judgments\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
