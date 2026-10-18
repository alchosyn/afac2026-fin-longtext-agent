# -*- coding: utf-8 -*-
"""Quality gate 1 for one MinerU-parsed document: build the heading list (parsing.outline:
PDF bookmarks, else the printed table of contents) into processed_data/<doc_id>/docs.json, run
parsing.audit.Audit with it, write the report to processed_data/<doc_id>/audit.json and print a
summary with the findings per code. The audit reads processed_data/<doc_id>/native.json (export it
first with script/export_native.py in the MinerU venv; without it the coverage check falls back to
PyMuPDF words) and OCRs the scan pages once into processed_data/<doc_id>/ocr.json.

    python script/audit_doc.py fr_byd_2024
    python script/audit_doc.py fr_byd_2024 --no-ocr          # skip OCR of raster images
    python script/audit_doc.py fr_byd_2024 --show 8          # findings per code to print
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pymupdf  # noqa: E402

from parsing.audit import Audit  # noqa: E402
from parsing.common import load_mineru  # noqa: E402
from parsing.outline import build_outline  # noqa: E402

PROC = ROOT / "processed_data"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_ids", nargs="+")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args(argv)
    cards = {json.loads(l)["doc_id"]: json.loads(l) for l in (PROC / "docs.jsonl").open(encoding="utf-8")}
    for doc_id in args.doc_ids:
        card = cards[doc_id]
        mineru = card.get("mineru") or f"processed_data/{doc_id}/mineru.json"
        t0 = time.time()
        outline = build_outline(doc_id, pymupdf.open(str(ROOT / card["source_pdf"])), load_mineru(ROOT / mineru))
        (PROC / doc_id / "docs.json").write_text(json.dumps(outline, ensure_ascii=False, indent=1), encoding="utf-8")
        report = Audit(doc_id, ROOT / card["source_pdf"], ROOT / mineru, ocr=not args.no_ocr, outline=outline).run()
        out = PROC / doc_id / "audit.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        s = report["summary"]
        print(f"\n===== {doc_id}: {s['pages']} pages ({s['pages_without_text_layer']} without text layer), "
              f"char coverage {s['char_coverage']:.3%} ({s['chars_uncovered']} uncovered, {'native.json' if s['native'] else 'PyMuPDF words'}), "
              f"scan pages {s['scan_pages']} (OCR chars {s['ocr_chars']}: {s['ocr_chars_outside']} outside, {s['ocr_chars_in_figures']} in figures, "
              f"{s['ocr_chars_in_dropped']} in dropped blocks), tables with a hidden caption {s['tables_with_hidden_caption']}, headings {s['headings']}, "
              f"toc entries {s['toc_entries']}, ruled grids {s['ruled_grids']}, "
              f"table candidates ok/rejected {s['table_candidates_ok']}/{s['table_candidates_rejected']}  [{time.time() - t0:.0f}s]")
        print(f"  heading list: {s['outline_source']}, {s['outline_entries']} headings ({s['outline_unmatched']} unmatched), "
              f"styles {outline.get('styles')}" + (f", page offset {outline['offset']}" if outline.get('offset') else ""))
        print("  counts:", report["counts"])
        by = {}
        for f in report["findings"]:
            by.setdefault(f["code"], []).append(f)
        for code, fs in sorted(by.items()):
            print(f"  -- {code} ({len(fs)})")
            for f in fs[:args.show]:
                extra = {k: v for k, v in f.items() if k not in ("code", "page", "block", "text")}
                print(f"     p{f['page']:<4} #{str(f['block']):<4} {f['text'][:50]!r} {extra if extra else ''}")
        print(f"  -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
