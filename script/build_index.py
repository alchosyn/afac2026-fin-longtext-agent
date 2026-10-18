# -*- coding: utf-8 -*-
"""Build the indexes of documents over their chunks.jsonl:

  * the inverted index  processed_data/<doc_id>/bm25.json  (agent.bm25: body and path
    fields, postings + chunk lengths, the sha1 of the chunks it was built from) -- seconds
    for every document
  * with --vectors, the bge-m3 vectors  vectors_bge_m3.npz + .json  (agent.retrieval
    .build_vectors), incrementally: only chunks whose encoded text changed are encoded --
    one to two chunks a second on the CPU

The indexes are consumed by agent.retrieval; a missing or stale bm25.json is rebuilt in
memory there with a warning, a stale vector file is refused.

Usage:
    python script/build_index.py                       # bm25.json for every document
    python script/build_index.py fr_byd_2024 ins_01    # some
    python script/build_index.py --vectors             # bm25.json and vectors for every document (hours)
    python script/build_index.py --vectors --qids reg_a_008 reg_a_015   # the documents of these questions
    <cuda venv python> script/build_index.py --vectors-only --device cuda --force   # every document on the GPU
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.bm25 import build_index, save_index  # noqa: E402
from agent.retrieval import Encoder, build_vectors, get_doc  # noqa: E402
from parsing.common import PROC, load_cards  # noqa: E402


def doc_ids_of(qids):
    out = []
    for path in (ROOT / "questions" / "group_a").glob("*_questions.json"):
        for q in json.load(open(path, encoding="utf-8")):
            if q["qid"] in qids:
                out += [d for d in q["doc_ids"] if d not in out]
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_ids", nargs="*")
    ap.add_argument("--qids", nargs="*", help="documents of these questions")
    ap.add_argument("--vectors", action="store_true", help="also build / refresh the bge-m3 vectors")
    ap.add_argument("--vectors-only", action="store_true", help="build / refresh the vectors, leave bm25.json alone")
    ap.add_argument("--device", default="cpu", help="encoder device: cpu, or cuda (plain transformers, fp16)")
    ap.add_argument("--force", action="store_true", help="re-encode every chunk")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    cards = load_cards()
    doc_ids = args.doc_ids or (doc_ids_of(args.qids) if args.qids else
                               [c["doc_id"] for c in cards if (PROC / c["doc_id"] / "chunks.jsonl").exists()])

    t0 = time.time()
    n_chunks = terms = size = 0
    for d in ([] if args.vectors_only else doc_ids):
        doc = get_doc(d)
        index = build_index(doc)
        path = save_index(index)
        n_chunks += len(doc.units)
        terms += len(index["fields"]["body"]["postings"])
        size += path.stat().st_size
        if not args.quiet:
            print(f"{d[:34]:34} {len(doc.units):5} chunks  {len(index['fields']['body']['postings']):6} terms  {path.stat().st_size / 1e3:7.0f} KB")
    if not args.vectors_only:
        print(f"bm25: {len(doc_ids)} documents, {n_chunks} chunks, {terms} body terms, {size / 1e6:.1f} MB, {time.time() - t0:.0f}s")

    if args.vectors or args.vectors_only:
        t1 = time.time()
        enc = Encoder(device=args.device)
        print(f"encoder: {enc.backend} from {enc.model_path} ({time.time() - t1:.0f}s to load)")
        for d in doc_ids:
            t2 = time.time()
            meta = build_vectors(d, enc, force=args.force)
            print(f"{d[:34]:34} {meta['count']:5} chunks  encoded {meta['encoded']:5}  reused {meta['reused']:5}  {time.time() - t2:.0f}s")
        print(f"vectors: {len(doc_ids)} documents, {time.time() - t1:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
