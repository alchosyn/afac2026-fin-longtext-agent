# -*- coding: utf-8 -*-
"""Bring the figure records of existing structures up to the current rules
(parsing.figures.describe_figure, the same call parsing.structure makes on a new build), then
rebuild the documents' chunks and BM25 indexes:

  * every figure record, captioned or not, bitmap or vector, is cut out of the PDF with its
    MinerU box -> processed_data/<doc_id>/figures/<local id>.png, path in `image`
  * labels and `numbers` (every number written in the figure) from the text layer, or OCR on a
    bitmap / a scan page (cached in processed_data/<doc_id>/figure_text.json)
  * a figure without a MinerU caption gets `title_candidate` (its first line, where it came
    from, the hand-review verdict of processed_data/figure_title_review.json); a "title"
    verdict makes the line its caption (`caption_from` text_layer_first_line / ocr_first_line)

Only figure records change; headings, paragraphs and tables are left as they are, so no
structure rebuild (and no MinerU re-run) is needed. A figure without a box is reported and
skipped.

Usage:
    python script/render_figures.py                              # every document with figures
    python script/render_figures.py pack2_text01 fc_text09
    python script/render_figures.py --candidates out.json        # dry run: write the first-line candidates, change nothing
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsing.common import PROC, load_cards  # noqa: E402
from parsing.figures import FigureTextCache, describe_figure  # noqa: E402
from parsing.native import scan_pages  # noqa: E402
from parsing.structure import figure_record_fields  # noqa: E402

FIGURE_FIELDS = ("caption_from", "numbers", "image", "title_candidate")


def update_doc(card, dry_run=False, candidates=None):
    doc_id = card["doc_id"]
    src = PROC / doc_id / "elements.jsonl"
    records = [json.loads(l) for l in src.open(encoding="utf-8")]
    figures = [r for r in records if r["type"] == "figure"]
    stats = Counter(figures=len(figures))
    if not figures or not card.get("source_pdf"):
        return stats
    pdf = pymupdf.open(str(ROOT / card["source_pdf"]))
    scans = scan_pages(pdf)
    cache = FigureTextCache(doc_id)
    for r in figures:
        if not (r.get("bbox") and r.get("page")):
            stats["no_box"] += 1
            continue
        # the caption the parser gave: MinerU builds keep it in `caption`, the legacy build of
        # pack2_text07 only in `text`; a first-line caption from an earlier run is not one
        mineru_caption = ""
        if r.get("caption_from", "mineru") == "mineru":
            mineru_caption = r.get("caption", r.get("text")) or ""
            if mineru_caption in ("图片", "图表"):
                mineru_caption = ""
        info = describe_figure(pdf[r["page"] - 1], r["bbox"], doc_id, r["page"], r["id"].split("#", 1)[1],
                               caption=mineru_caption, text_layer=r["page"] not in scans, cache=cache, stats=stats)
        fields = figure_record_fields(info)
        if r.get("labels") is not None and fields["labels"] != r["labels"]:
            stats["labels_changed"] += 1
        if candidates is not None and info["title_candidate"]:
            candidates.append({"doc_id": doc_id, "id": r["id"], "page": r["page"], "kind": r.get("kind"),
                               "scan_page": r["page"] in scans, **info["title_candidate"],
                               "image": info["image"], "n_lines": None})
        if not dry_run:
            for k in FIGURE_FIELDS:
                r.pop(k, None)
            r.update(fields)
            r["text"] = fields["caption"] or ("图表" if r.get("kind") == "chart" else "图片")
    cache.save()
    if not dry_run:
        with src.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return stats


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_ids", nargs="*")
    ap.add_argument("--candidates", help="dry run: write the first-line candidates of the uncaptioned figures here")
    ap.add_argument("--no-chunks", action="store_true", help="update elements.jsonl only")
    args = ap.parse_args(argv)
    cards = {c["doc_id"]: c for c in load_cards()}
    doc_ids = args.doc_ids or [d for d in cards if (PROC / d / "elements.jsonl").exists()]
    dry = bool(args.candidates)
    cands = [] if dry else None
    total = Counter()
    touched = []
    for d in doc_ids:
        s = update_doc(cards[d], dry_run=dry, candidates=cands)
        total.update(s)
        if s["figures"]:
            touched.append(d)
            print(f"{d[:34]:34} {dict(s)}", flush=True)
    if dry:
        Path(args.candidates).write_text(json.dumps(cands, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"\n{len(cands)} candidates -> {args.candidates}")
    elif not args.no_chunks and touched:
        from script.build_chunks import build as build_chunks
        from agent.bm25 import build_index, save_index
        from agent.retrieval import get_doc
        for d in touched:
            _, chunks, _, problems = build_chunks(d, card=cards[d])
            total["figure_chunks"] += sum(1 for c in chunks if c["type"] == "figure")
            total["figure_chunks_with_numbers"] += sum(1 for c in chunks if c["type"] == "figure" and c.get("numbers"))
            total["chunk_problems"] += len(problems)
            save_index(build_index(get_doc(d)))                # first load of the document in this process: the new chunks
    size = sum(p.stat().st_size for d in touched for p in (PROC / d / "figures").glob("*.png") if (PROC / d / "figures").exists())
    print(f"\ntotal: {dict(total)} | {size / 1e6:.0f} MB of PNG under processed_data/*/figures")


if __name__ == "__main__":
    main(sys.argv[1:])
