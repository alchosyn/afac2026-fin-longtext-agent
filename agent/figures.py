# -*- coding: utf-8 -*-
"""Figure images for the answering model.

A figure's chunk holds its caption, the naming words inside it and the source line, the
numbers written in it (`numbers`: read off the text layer or OCR, never generated; not part of
the BM25 text) and a pointer `image` to the PNG the structure build cut out of the PDF
(parsing.figures, processed_data/<doc>/figures/; every figure has one since 2026-10-28).
When such a chunk is evidence, the judge gets the figure itself: the file is read and
attached to the call as a PNG (agent.llm.Qwen.chat images=...). A record without the
pointer (a structure built before 2026-10-01 and not yet passed through
script/render_figures.py) is cut out of the PDF on the spot. Tested 2026-09-24 on 10
hand-checked chart claims: text only 1/10 (the model answers 无法判断), with the crop 9/10.

    from agent.figures import render_unit, evidence_images
    png = render_unit(doc.by_id["pack2_text01#c0009"])        # bytes
    images = evidence_images(windows)                          # [(evidence id, unit id, png)] <= MAX_IMAGES
"""
import sys
from pathlib import Path

import pymupdf

from .retrieval import get_doc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from parsing.figures import render_figure  # noqa: E402

MAX_IMAGES = 3        # crops per judge call
_PDFS = {}


def _pdf(doc_id):
    if doc_id not in _PDFS:
        _PDFS[doc_id] = pymupdf.open(str(ROOT / get_doc(doc_id).card["source_pdf"]))
    return _PDFS[doc_id]


def render_unit(unit):
    """PNG bytes of a figure chunk or structure record: the saved file when the record
    points at one, else the box cut from the PDF page; None when neither is possible (a
    legacy record without a box)."""
    doc_id = unit["id"].split("#")[0]
    path = unit.get("image") or get_doc(doc_id).element_of(unit).get("image")
    if path and (ROOT / path).exists():
        return (ROOT / path).read_bytes()
    rec = get_doc(doc_id).element_of(unit)
    if not (rec.get("bbox") and rec.get("page")):
        return None
    return render_figure(_pdf(doc_id)[rec["page"] - 1], rec["bbox"])


def evidence_images(windows, max_images=MAX_IMAGES):
    """Images of the figure units among the evidence windows, in evidence order:
    [(evidence id "E<n>", unit id, png bytes)], at most max_images."""
    out = []
    for i, (doc, units, uids) in enumerate(windows, 1):
        for u in units:
            if u["type"] == "figure" and len(out) < max_images:
                png = render_unit(u)
                if png:
                    out.append((f"E{i}", u["id"], png))
    return out
