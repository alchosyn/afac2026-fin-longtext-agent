# -*- coding: utf-8 -*-
"""Write the document header into every chunk of processed_data/<doc_id>/chunks.jsonl:
each record gets a `header` field (parsing.header: doc_id, domain, title, doc_type, company,
year and the fields of the document's domain, from processed_data/doc_cards.json).

Nothing else in a record changes. script/build_chunks.py writes the header itself when it
builds chunks, so this script is for chunk files that exist already and for refreshing the
headers after script/build_doc_cards.py has been run again. The inverted indexes record the
sha1 of chunks.jsonl: run script/build_index.py afterwards.

Usage:
    python script/add_chunk_headers.py                 # every document that has a card
    python script/add_chunk_headers.py ins_05 fc_text12
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsing.common import PROC          # noqa: E402
from parsing.header import load_headers  # noqa: E402


def main(argv):
    headers = load_headers()
    done = changed = chunks = 0
    for doc_id in argv or headers:
        path = PROC / doc_id / "chunks.jsonl"
        if doc_id not in headers or not path.exists():
            print(f"{doc_id}: no card or no chunks.jsonl, skipped")
            continue
        records = [json.loads(l) for l in path.open(encoding="utf-8")]
        if any(r.get("header") != headers[doc_id] for r in records):
            for r in records:
                r["header"] = headers[doc_id]
            with path.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            changed += 1
        done += 1
        chunks += len(records)
    print(f"{done} documents, {chunks} chunks carry a header; {changed} chunk files rewritten")


if __name__ == "__main__":
    main(sys.argv[1:])
