# -*- coding: utf-8 -*-
"""Repair + chunk one MinerU-parsed document: processed_data/<doc_id>/mineru.json + the PDF
-> processed_data/<doc_id>/elements.jsonl, and the document card in processed_data/docs.jsonl.

The work is done by the parsing package (parsing.repair, parsing.structure); this script only
resolves paths from the card, writes the files and prints the counters. Run quality gate 1
first (script/audit_doc.py): its audit.json is read here and the text blocks it reported as
missing headings (head_as_text) are promoted, and the ones merged into a text block
(head_merged) are split out, before the tree is built.

Usage:
    python script/build_elements_mineru.py pack2_text01
    python script/build_elements_mineru.py fc_text01 --pdf raw/financial_contracts/text01.pdf --domain financial_contracts
    python script/build_elements_mineru.py fr_byd_2024 --pdf raw/financial_reports/annual_byd_2024_report.PDF --domain financial_reports --title 比亚迪2024年年度报告
    python script/build_elements_mineru.py fc_text01 --no-audit          # ignore audit.json (no heading promotion)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsing.structure import Builder  # noqa: E402
from parsing.common import FIXED_LEVELS, PROC, clean, load_cards, load_mineru, save_cards, txt  # noqa: E402
from parsing.native import load_native, load_ocr, scan_pages  # noqa: E402
from parsing.repair import caption_headings_from_audit, hidden_captions_from_audit, merged_headings_from_audit, promotions_from_audit  # noqa: E402
from parsing.rerun import image_table_reruns  # noqa: E402


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_id")
    ap.add_argument("--pdf", help="PDF path (default: the card's source_pdf)")
    ap.add_argument("--domain", help="domain for a new card")
    ap.add_argument("--title", help="title for the card / root heading (default: MinerU's first doc_title)")
    ap.add_argument("--mineru", help="MinerU JSON (default: processed_data/<doc_id>/mineru.json)")
    ap.add_argument("--audit", help="audit.json of quality gate 1 (default: processed_data/<doc_id>/audit.json when present)")
    ap.add_argument("--no-audit", action="store_true", help="build without the audit findings")
    ap.add_argument("--no-rerun", action="store_true", help="do not run MinerU again on images captioned 表N (parsing.rerun)")
    args = ap.parse_args(argv)

    cards = load_cards()
    card = next((c for c in cards if c["doc_id"] == args.doc_id), None)
    pdf = args.pdf or (card or {}).get("source_pdf")
    if not pdf:
        sys.exit("no PDF: pass --pdf or add a card first")
    mineru_path = Path(args.mineru) if args.mineru else PROC / args.doc_id / "mineru.json"
    data = load_mineru(mineru_path)
    title = args.title or (card or {}).get("title") or next(
        (clean(txt(b)) for p in data["pages"] for b in p["blocks"] if b["type"] == "doc_title"), args.doc_id)

    audit_path = None if args.no_audit else Path(args.audit) if args.audit else PROC / args.doc_id / "audit.json"
    promote, caption_heads, merged, hidden_caps = {}, {}, {}, {}
    if audit_path and audit_path.exists():
        report = json.loads(audit_path.read_text(encoding="utf-8"))
        promote = promotions_from_audit(report)
        caption_heads = caption_headings_from_audit(report)
        merged = merged_headings_from_audit(report)
        hidden_caps = hidden_captions_from_audit(report)

    import pymupdf
    scan = load_ocr(args.doc_id, pymupdf.open(str(ROOT / pdf)), scan_pages(pymupdf.open(str(ROOT / pdf))))
    rerun_stats, rerun_log = Counter(), []
    image_tables = {} if args.no_rerun else image_table_reruns(args.doc_id, pymupdf.open(str(ROOT / pdf)), data, rerun_stats, rerun_log)
    builder = Builder(args.doc_id, ROOT / pdf, mineru_path, promote=promote, caption_heads=caption_heads, merged=merged,
                      native=load_native(args.doc_id), scan=scan, hidden_caps=hidden_caps, image_tables=image_tables)
    records = builder.build(title)
    out = PROC / args.doc_id / "elements.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    new_card = dict(card or {})
    new_card.update({"doc_id": args.doc_id, "domain": args.domain or new_card.get("domain") or "unknown", "title": title,
                     "issuer": new_card.get("issuer", ""), "date": new_card.get("date", ""), "source_pdf": str(pdf).replace("\\", "/"),
                     "mineru": str(mineru_path.relative_to(ROOT)).replace("\\", "/"), "chunks": str(out.relative_to(ROOT)).replace("\\", "/")})
    if card:
        cards[cards.index(card)] = new_card
    else:
        cards.append(new_card)
    save_cards(cards)

    types = Counter(r["type"] for r in records)
    levels = Counter(r.get("level") for r in records if r["type"] == "heading")
    print(f"{args.doc_id}: {len(records)} records {dict(types)} | heading levels {dict(sorted(levels.items()))}")
    print(f"   audit: {audit_path if audit_path and audit_path.exists() else 'none'} -> {sum(len(v) for v in promote.values())} blocks to promote, "
          f"{sum(len(v) for v in caption_heads.values())} table captions / footnotes that are headings, "
          f"{sum(len(c) for v in merged.values() for c in v.values())} headings to split out of text blocks, "
          f"{sum(len(v) for v in hidden_caps.values())} hidden table captions")
    print(f"   pages: {len(scan)} scan pages OCR'd ({dict(__import__('collections').Counter(v['reason'] for v in scan.values()))}), "
          f"native.json {'used' if builder.native else 'missing (PyMuPDF words)'}")
    so = builder.style_order
    moved = {k: v for k, v in so["levels"].items() if v != FIXED_LEVELS.get(k)}
    print(f"   heading levels: {'fixed table (styles mixed)' if so['fallback'] else (moved or 'fixed table')}"
          f" | evidence {[(a, b, s, c) for a, b, s, c in so['edges']]}"
          f"{' | undecided ' + str(so['undecided']) if so['undecided'] else ''}")
    print(f"   tables by status: {dict(builder.table_status)}")
    if rerun_stats or rerun_log:
        print(f"   images captioned 表N re-run through MinerU: {dict(rerun_stats)}{' | ' + '; '.join(rerun_log) if rerun_log else ''}")
    print(f"   steps: {dict(builder.stats)}")
    print(f"   -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
