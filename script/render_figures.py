# -*- coding: utf-8 -*-
"""Cut the captioned figures of a structure out of its PDF and record the files:
processed_data/<doc_id>/elements.jsonl figure records get `image` =
processed_data/<doc_id>/figures/<local id>.png (parsing.figures), then the chunks are rebuilt
so the figure chunks carry the pointer.

The structure builder (parsing.structure) does this itself for every new build; this script
is for structures built before 2026-10-01 or by the legacy builders. A figure without a box
or with a degenerate one is reported and skipped.

Usage:
    python script/render_figures.py                     # every document
    python script/render_figures.py pack2_text01 ins_02
    python script/render_figures.py --force             # rewrite existing PNGs
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
from parsing.figures import save_figure  # noqa: E402
from script.build_chunks import build as build_chunks  # noqa: E402


def render_doc(card, force=False):
    doc_id = card["doc_id"]
    src = PROC / doc_id / "elements.jsonl"
    records = [json.loads(l) for l in src.open(encoding="utf-8")]
    figures = [r for r in records if r["type"] == "figure" and r.get("caption", r.get("text")) not in ("", "图片", "图表", None)]
    stats = Counter(figures=len(figures))
    if not figures:
        return stats
    pdf = pymupdf.open(str(ROOT / card["source_pdf"]))
    changed = False
    for r in figures:
        if not (r.get("bbox") and r.get("page")):
            stats["no_box"] += 1
            continue
        rel = save_figure(pdf[r["page"] - 1], r["bbox"], doc_id, r["id"].split("#", 1)[1], force=force)
        if rel is None:
            stats["degenerate_box"] += 1
            continue
        stats["saved"] += 1
        if r.get("image") != rel:
            r["image"] = rel
            changed = True
    if changed:
        with src.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["structure_updated"] = 1
        _, chunks, _, problems = build_chunks(doc_id)
        stats["figure_chunks_with_image"] = sum(1 for c in chunks if c["type"] == "figure" and c.get("image"))
        stats["problems"] = len(problems)
    return stats


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_ids", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cards = {c["doc_id"]: c for c in load_cards()}
    doc_ids = args.doc_ids or [d for d in cards if (PROC / d / "elements.jsonl").exists()]
    total = Counter()
    for d in doc_ids:
        s = render_doc(cards[d], args.force)
        total.update(s)
        if s["figures"]:
            print(f"{d[:34]:34} {dict(s)}")
    size = sum(p.stat().st_size for d in doc_ids for p in (PROC / d / "figures").glob("*.png") if (PROC / d / "figures").exists())
    print(f"\ntotal: {dict(total)} | {size / 1e6:.0f} MB of PNG under processed_data/*/figures")


if __name__ == "__main__":
    main(sys.argv[1:])
