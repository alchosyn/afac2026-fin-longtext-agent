# -*- coding: utf-8 -*-
"""Eyeball the retrieval layer: run a few queries against one document with the two-field
BM25 (body 0.7 + heading path 0.3) and print the top units with their path and a preview.

    python tools/probe_retrieval.py fr_byd_2024 "货币资金 2024年12月31日 余额" "公司法定代表人"
    python tools/probe_retrieval.py ins_01 --top 5 "责任免除 身故 不承担保险责任"
"""
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.retrieval import bm25_scores, get_doc  # noqa: E402


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_ids", help="comma separated document ids")
    ap.add_argument("queries", nargs="+")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--body-only", action="store_true")
    args = ap.parse_args(argv)
    doc_ids = args.doc_ids.split(",")
    for q in args.queries:
        print(f"\n===== {q} =====")
        for rank, (uid, score) in enumerate(bm25_scores(q, doc_ids, header=not args.body_only)[:args.top], 1):
            doc = get_doc(uid.split("#")[0])
            u = doc.by_id[uid]
            extra = f" caption={u.get('caption', '')[:30]!r}" if u["type"] == "table" and u.get("caption") else ""
            print(f"[{rank}] {score:6.2f} {uid.split('#')[1]:<6} {u['type']:<9} p{u['page']:<4} {u['path'][:48]}{extra}")
            print("       " + u["body"][:160].replace("\n", " / "))


if __name__ == "__main__":
    main(sys.argv[1:])
