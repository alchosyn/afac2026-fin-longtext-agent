# -*- coding: utf-8 -*-
"""MinerU run again on one region of a page: the layout model labels by page context, so a
small table under a heading and a paragraph can come out as an image on the full page and as a
table when the region is parsed alone (pack2_text03 p19 表3, 4.6 pt text in a 10.6 pt page:
measured 2026-09-30 by removing one page element at a time). The re-run is its own filter:
the timeline infographics the author also calls tables (p18 表2, p20 表4) come back as images
again, so no second model is needed to decide.

Which images are tried: image / chart blocks whose caption starts with 表N (3 of 614 figure
blocks on 20 docs). The region (the image body plus PAD) is written as a one-page PDF under
processed_data/<doc_id>/crops/ and parsed by mineru-kit (the MinerU 4.0.4 venv, see the memory
"parsing-toolchain-environment"; ~15 s per crop, mostly model loading, and it sometimes fails
with a "PDF render executor" error that a retry clears). The middle JSON is cached next to the
PDF, so a rebuild never runs MinerU twice for the same region; delete the crops folder to
redo. Run through subprocess, not a shell: bash mangles the Windows path.

    image_table_reruns(doc_id, pdf, data) -> {page: {image block index: table_body HTML}}
"""
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import pymupdf

from .common import FIGURE_TYPES, PROC, clean, spans, txt

MINERU_KIT = os.environ.get("MINERU_KIT") or str(Path(
    r"D:\Temp\claude\D--long-text-economic-qustions-public-dataset-upload\cecefb31-f71d-421b-a866-7465c14904cd\scratchpad"
    r"\mineru_venv\Scripts\mineru-kit.exe"))
TABLE_CAPTION = re.compile(r"^表\s*[A-Z]?\d")
PAD = 4.0              # pt around the image body
TIMEOUT = 900          # s per mineru-kit call
ATTEMPTS = 2


def crop_pdf(pdf, pno, bbox, path):
    """The region bbox (page fractions) of page pno, plus PAD, as a one-page PDF at path; the
    text layer and drawings come along (pymupdf show_pdf_page)."""
    page = pdf[pno - 1]
    W, H = page.rect.width, page.rect.height
    clip = pymupdf.Rect(bbox[0] * W - PAD, bbox[1] * H - PAD, bbox[2] * W + PAD, bbox[3] * H + PAD) & page.rect
    out = pymupdf.open()
    cp = out.new_page(width=clip.width, height=clip.height)
    cp.show_pdf_page(cp.rect, pdf, pno - 1, clip=clip)
    out.save(str(path))
    out.close()


def rerun_region(pdf, pno, bbox, stem, cache_dir, log=None):
    """MinerU's middle JSON for the region, from the cache or a fresh run; None when mineru-kit
    produced nothing in ATTEMPTS tries."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_json = cache_dir / f"{stem}.json"
    if out_json.exists():
        return json.loads(out_json.read_text(encoding="utf-8"))
    pdf_path = cache_dir / f"{stem}.pdf"
    crop_pdf(pdf, pno, bbox, pdf_path)
    cmd = [MINERU_KIT, "parse", str(pdf_path), "-o", str(out_json), "-f", "middle_json", "--tier", "standard",
           "--ocr-mode", "auto", "--pages", "all"]
    for _ in range(ATTEMPTS):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, encoding="utf-8", errors="replace")
        except (subprocess.TimeoutExpired, OSError) as e:          # noqa: PERF203
            if log is not None:
                log.append(f"{stem}: {type(e).__name__}")
            continue
        if out_json.exists():
            return json.loads(out_json.read_text(encoding="utf-8"))
        if log is not None:
            log.append(f"{stem}: exit {r.returncode} {(r.stderr or r.stdout or '')[-300:]}")
    return None


def table_html_of(result):
    """The table_body HTML of the first table block in a middle JSON, or None."""
    if not result:
        return None
    for pg in result.get("pages", []):
        for b in pg.get("blocks", []):
            if b.get("type") == "table":
                body = spans(b, "_body")
                html = txt(body[0]) if body else ""
                if "<td" in html:
                    return html
    return None


def image_table_reruns(doc_id, pdf, data, stats=None, log=None):
    """{page: {image block index: table_body HTML}} for the figure blocks of a document whose
    caption names a table and whose region MinerU returns as a table when parsed alone. stats
    (a Counter) gets image_table_candidates / image_table_rerun_table / image_table_rerun_image /
    image_table_rerun_failed."""
    stats = stats if stats is not None else Counter()
    out = {}
    cache_dir = PROC / doc_id / "crops"
    for pg in data["pages"]:
        pno = pg["page_idx"] + 1
        for b in pg["blocks"]:
            if b["type"] not in FIGURE_TYPES:
                continue
            cap = clean("".join(txt(c) for c in spans(b, "_caption")))
            if not TABLE_CAPTION.match(cap):
                continue
            stats["image_table_candidates"] += 1
            body = spans(b, "_body")
            bbox = body[0]["bbox"] if body else b["bbox"]
            result = rerun_region(pdf, pno, bbox, f"p{pno}_b{b['index']}", cache_dir, log)
            if result is None:
                stats["image_table_rerun_failed"] += 1
                continue
            html = table_html_of(result)
            if html:
                out.setdefault(pno, {})[b["index"]] = html
                stats["image_table_rerun_table"] += 1
            else:
                stats["image_table_rerun_image"] += 1
    return out
