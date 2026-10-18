# -*- coding: utf-8 -*-
"""Chunk the structure layer of documents: processed_data/<doc_id>/elements.jsonl
-> processed_data/<doc_id>/chunks.jsonl + chunks.meta.json (parsing.chunk), with the
post-processing (dedup under one heading, clause cross-references) and gate 3 (members,
parents, coverage against the dropped list, token length against the encoder's limit).

chunks.meta.json holds the sha1 of the elements.jsonl the chunks were made from (agent.docview
refuses stale chunks), the chunking parameters, the counters, the ids of the structure
records dropped on purpose (by reason) and the problems gate 3 found.

Usage:
    python script/build_chunks.py                       # every document of processed_data/docs.jsonl
    python script/build_chunks.py fr_byd_2024 ins_01    # some
    python script/build_chunks.py --text-cap 600        # other parameters (see parsing.chunk)
    python script/build_chunks.py --no-tokens           # skip the token-length check (no tokenizer load)
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsing.chunk import TABLE_CAP, TEXT_CAP, check_chunks, chunk_body, chunk_doc  # noqa: E402
from parsing.common import PROC, load_cards  # noqa: E402
from parsing.header import load_headers  # noqa: E402

SIZES = ["<100", "100-400", "400-800", "800-1200", ">1200"]
TOKENIZER_PATH = Path(r"D:\models\bge-m3")
_TOK = None
_HEADERS = None


def headers():
    global _HEADERS
    if _HEADERS is None:
        _HEADERS = load_headers()
    return _HEADERS


def file_sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def token_lengths(chunks):
    """{chunk id: bge-m3 tokens of its body}, or None when the tokenizer is not available."""
    global _TOK
    if _TOK is None:
        try:
            from transformers import AutoTokenizer
            _TOK = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
        except Exception as e:                                   # noqa: BLE001
            print(f"[build_chunks] no tokenizer ({e}): token-length check skipped", file=sys.stderr)
            _TOK = False
    if not _TOK:
        return None
    bodies = [chunk_body(c) for c in chunks]
    enc = _TOK(bodies, add_special_tokens=False)["input_ids"] if bodies else []
    return {c["id"]: len(t) for c, t in zip(chunks, enc)}


def build(doc_id, card=None, tokens=True, **params):
    src = PROC / doc_id / "elements.jsonl"
    records = [json.loads(l) for l in src.open(encoding="utf-8")]
    card = card or {}
    chunks, stats, dropped = chunk_doc(records, doc_id, kind=card.get("kind"), title=card.get("title", ""), **params)
    lengths = token_lengths(chunks) if tokens else None
    problems = check_chunks(chunks, records, dropped, lengths)
    header = headers().get(doc_id)                               # the document header on every chunk (parsing.header)
    if header:
        for c in chunks:
            c["header"] = header
    out = PROC / doc_id / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    meta = {"doc_id": doc_id, "structure": str(src.relative_to(ROOT)).replace("\\", "/"), "structure_sha1": file_sha1(src),
            "params": {"text_cap": params.get("text_cap", TEXT_CAP), "table_cap": params.get("table_cap", TABLE_CAP), **params},
            "count": len(chunks), "stats": dict(stats), "dropped": dropped,
            "tokens_max": max(lengths.values()) if lengths else None, "problems": problems}
    (PROC / doc_id / "chunks.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return records, chunks, stats, problems


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_ids", nargs="*")
    ap.add_argument("--text-cap", type=int)
    ap.add_argument("--table-cap", type=int)
    ap.add_argument("--no-tokens", action="store_true", help="skip the token-length check")
    ap.add_argument("--quiet", action="store_true", help="only the summary")
    args = ap.parse_args(argv)
    params = {k: v for k, v in (("text_cap", args.text_cap), ("table_cap", args.table_cap)) if v}
    cards = {c["doc_id"]: c for c in load_cards()}
    doc_ids = args.doc_ids or [d for d in cards if (PROC / d / "elements.jsonl").exists()]

    total, by_dom, sizes, problems_all = Counter(), defaultdict(Counter), defaultdict(Counter), []
    for d in doc_ids:
        card = cards.get(d, {})
        records, chunks, stats, problems = build(d, card, tokens=not args.no_tokens, **params)
        dom = card.get("domain") or "?"
        units = sum(1 for r in records if r["type"] != "heading")
        total.update(stats)
        by_dom[dom]["docs"] += 1
        by_dom[dom]["elements"] += units
        by_dom[dom]["chunks"] += len(chunks)
        by_dom[dom]["refs"] += stats.get("ref_links", 0) + stats.get("fig_ref_links", 0)
        for k, v in stats.items():
            if k.startswith("size_"):
                sizes[dom][k.split("_", 2)[2]] += v
        problems_all += [f"{d}: {p}" for p in problems]
        if not args.quiet:
            kinds = {k[6:]: v for k, v in stats.items() if k.startswith("chunk_")}
            steps = {k: v for k, v in stats.items() if not k.startswith(("chunk_", "size_"))}
            print(f"{d[:34]:34} {units:5} elements -> {len(chunks):5} chunks {kinds} | {steps}"
                  + (f" | PROBLEMS {len(problems)}" if problems else ""))

    print(f"\n{'domain':20}{'docs':>5}{'elements':>10}{'chunks':>8}{'refs':>7}  " + "".join(f"{s:>10}" for s in SIZES))
    for dom, c in sorted(by_dom.items()):
        n = sum(sizes[dom].values()) or 1
        print(f"{dom:20}{c['docs']:5}{c['elements']:10}{c['chunks']:8}{c['refs']:7}  " + "".join(f"{sizes[dom][s] / n:>10.0%}" for s in SIZES))
    steps = {k: v for k, v in total.items() if not k.startswith(("chunk_", "size_")) and k != "ref_scope_depth"}
    print(f"\nsteps over all documents: {steps}")
    if problems_all:
        print(f"\ngate 3: {len(problems_all)} problems")
        for p in problems_all[:30]:
            print("   " + p)
    else:
        print("\ngate 3: no problems")


if __name__ == "__main__":
    main(sys.argv[1:])
