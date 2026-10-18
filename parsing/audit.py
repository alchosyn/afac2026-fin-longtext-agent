# -*- coding: utf-8 -*-
"""Quality gate 1: checks of a MinerU layout result against the PDF, before any repair.

Every check reports findings, none of them changes anything. A finding is a dict
    {"code", "page", "block" (MinerU index or None), "text" (snippet), ...extra}
and the audit also returns counts per code plus document-level rates.

Coverage depends on the page kind (parsing.native.page_kind): a scan is a page with an image over
70 % of it and no text layer, or with text painted under that image (the contracts' signature
pages: the signed scan laid over the Word page, whose draft text stays underneath and never lines
up). Digital pages are checked with the characters MinerU itself reads (native.json; PyMuPDF words
when it was not exported), scan pages with RapidOCR lines (ocr.json); scan pages skip the raster
check and the text-layer table checks.

Codes (what each one catches, with the case that motivated it):
  cov_text         characters of the text layer inside no block, digital pages (ins_01 p9: the
                   受益人 clauses, 442 characters; annual report p170: a whole column dropped)
  cov_ocr          OCR lines of a scan page inside no block (fc_text01 p302-304: the authorised
                   agreements list, p284 the business licence fields)
  tab_caption_hidden  a table whose title / date / unit lines sit above it in a figure or dropped
                   block (fr_cscec_2025 p146-162: every audited statement's title under a seal,
                   labelled image, and "人民币千元" labelled header); the repair makes them the caption
  cov_raster       a raster image no image / chart / table block covers  (annual report: product photos)
  cov_vector       a cluster of vector drawings no figure covers     (noisy; charts glued to neighbours)
  drop_header      header / footer text seen on one page only and long  (body text labelled as furniture)
  drop_index       an index block that is no table of contents (common.is_toc_index: not after a 目录
                   title and fewer than half of its entries end in a page number): a two-column
                   alignment taken for a TOC (fr_cscec_2025 p207-208 tax-rate table, ins_01 p18-19
                   definitions); parsing.repair.restore_false_index turns it back into a table / text
  head_as_text     a text block that is a missing heading: the title of a heading-list entry
                   (confidence bookmark / toc: annual report p102, p108, contract p113-119), a
                   第N章/第N节 line labelled text (medium), a number that fills a gap (high)
  head_merged      a heading-list title inside a longer text block, opening it before a numbered
                   item or closing it after a sentence end (ins_03 p5 "...化学污染。第十条 保险期间与不保证续保")
  head_in_caption  a table whose caption or footnote is a heading-list title (annual p155 "8、公司现金流量表"
                   as caption; contract p125 the next section's （四） heading as footnote)
  head_in_table    a heading-list title that sits inside a table body (annual p237 "与投资活动有关的现金")
  head_outline_missing  a heading-list entry with no block carrying its title on the target page ± 1
  head_gap         a numbered heading sequence skips a number, or repeats / goes backwards
  head_false       a title block that reads like a sentence or sits inside a table
  head_split       two title blocks side by side, or a bare number next to a title (number split off)
  cap_gap          图N / 表N caption numbers skip or repeat
  cap_orphan       a text block that starts like a caption (图N / 表N)
  cap_bad          a caption that is the running header, a checkbox line or a heading
  tab_candidate    a run of text blocks that projects to a grid (ok=True: a missed table)
  tab_ruled_missed a ruled grid in the PDF drawings without a table block
  img_maybe_table  an image whose OCR / text-layer words form a grid

Headings are checked in this order once every page has been scanned:
  1. heading list (parsing.outline: bookmarks, else the printed TOC -> docs.json): every entry
     must have a title block on its page ± 1; a text block, a merged block or a table caption
     carrying the title is reported for repair (outline_checks)
  2. a short 第N章 / 第N节 / 附录 text block is a heading (keyword_promotions)
  3. the headings as the repair will leave them (title blocks the chunker keeps + the blocks
     of 1 and 2) give the document's level table (common.infer_style_levels)
  4. a stack of those headings by level gives every numbered heading its parent; numbers under
     one parent and of one style must be consecutive, and a gap is searched among the short
     numbered text blocks of the page range (heading_sequences)

Usage:
    from parsing.audit import Audit
    a = Audit(doc_id, pdf_path, mineru_path, ocr=True)          # outline=build_outline(...) to pass docs.json
    report = a.run()          # {"summary": {...}, "counts": {...}, "findings": [...], "style_order": {...}}
native (parsing.native.load_native) and scan OCR ({page: {"reason", "lines"}}, parsing.native.load_ocr)
are loaded by run() when not passed.
"""
import re
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

from .common import (CONTINUED, LIST_ITEM, NORM_HEAD, STYLE_RX, TOC_LINE, Leveler, bookmark_covers, bookmark_is_block, heading_number,
                     index_entries, infer_style_levels, is_toc_index, load_mineru, looks_like_sentence, norm_heading, sentence_signals,
                     style_of)
from .native import NORM as NORM_TEXT
from .native import (OCR_MIN_SCORE, PAGE_RECALL, bigram_recall, hidden_table_captions, load_native, load_ocr, locate,
                     native_lines, page_grams, scan_pages, scan_table_bands, uncovered_runs)
from .outline import build_outline
from .tables import NUM_CELL, find_table_candidates, group_lines

DROP = {"header", "footer", "page_number", "index"}
FURNITURE = {"header", "footer", "page_number", "aside_text"}
TITLE_TYPES = {"doc_title", "paragraph_title"}
HIDDEN = re.compile(r"\[\s*Table_[A-Za-z_]*\]?|\[Table_[A-Za-z_]*")
CJK_SPACE = re.compile(r"(?<=[一-鿿，。、；：（）])\s+(?=[一-鿿，。、；：（）])")
SENT_PUNCT = re.compile(r"[。，；：;:]")
CAPTION = re.compile(r"^(图表|图|表)\s*([A-Z]?\d{1,3}(?:[-.]\d{1,2})?)\s*[:：.．、]?")
CHECKBOX = re.compile(r"[☑□■☐]")
KEYWORD_HEAD = re.compile(r"^(第[一二三四五六七八九十百\d]+\s*[章节部分篇]|附\s*录|附\s*则|前\s*言|引\s*言)")
SENT_BOUNDARY = tuple("。；;！？!?")
# numbering styles, levels and numbers come from parsing.common (STYLES, heading_number): the
# audit had its own copy until 2026-09-27, which lacked 第N条 and ① and kept 一、 at the level of
# 第N节, so ① gaps were never searched and the parent stack differed from the chunker's


# ----------------------------------------------------------------------------- helpers
def txt(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return txt(x.get("content", ""))
    if isinstance(x, list):
        return "".join(txt(i) for i in x)
    return ""


def clean(s):
    return CJK_SPACE.sub("", HIDDEN.sub("", s)).strip()


def numbered(text):
    """Does the text start with a heading numbering style?"""
    return style_of(text)[0] != "none"


def number_key(text):
    """(style, prefix, n) of a numbered heading: prefix is the dotted number without its last
    component (2.3 -> "2"), "" for the other styles; (None, None, None) when unnumbered."""
    style, n = heading_number(text)
    if style == "none" or n is None:
        return None, None, None
    prefix = ""
    if style == "dot":
        prefix = ".".join(STYLE_RX["dot"].match(text).group(0).rstrip(".").split(".")[:-1])
    return style, prefix, n


def merged_span(text, title):
    """(start, end) of a heading-list title (normalised) inside a longer text block, or None.
    The title must stand on its own there: it opens the block or follows a sentence end, and
    it closes the block or is followed by a numbered item. Cases: ins_03 p5 "...化学污染。第十条
    保险期间与不保证续保" (closes the block), fc_text02 p110 "（三）2024 年度会计政策...的影响1、重要
    会计政策变更情况无。" (opens it before an item). The characters norm_heading drops may sit
    anywhere between the title's characters."""
    if len(title) < 4:
        return None
    sep = f"[{NORM_HEAD.pattern[1:-1]}]*"
    rx = re.compile(sep.join(re.escape(ch) for ch in title))
    for m in rx.finditer(text):
        s, e = m.start(), m.end()
        while s > 0 and NORM_HEAD.match(text[s - 1]):
            s -= 1                                          # the "（" of "（三）" belongs to the title
        while e < len(text) and text[e] in "）)":
            e += 1
        head = text[:s].rstrip()
        rest = text[e:].lstrip()
        if not head and not rest:
            return None                                     # the whole block: head_as_text, not a merge
        if (not head or head.endswith(SENT_BOUNDARY)) and (not rest or numbered(rest)):
            return s, e
    return None


def cover(a, b):
    """share of rect a covered by rect b"""
    i = a & b
    return 0.0 if not i.is_valid or i.is_empty else i.get_area() / max(1e-6, a.get_area())


def sub_bboxes(b):
    out = [b["bbox"]]
    for c in b.get("content", []):
        if isinstance(c, dict) and "bbox" in c:
            out.append(c["bbox"])
    return out


def grid_of_boxes(boxes, gap=12.0, tol=None):
    """(n_lines, median columns per line, numeric share) of text boxes (x0, y0, x1, y1, text)."""
    if not boxes:
        return 0, 0, 0.0
    if tol is None:
        hs = sorted(b[3] - b[1] for b in boxes)
        tol = max(2.0, 0.6 * hs[len(hs) // 2])
    boxes = sorted(boxes, key=lambda b: ((b[1] + b[3]) / 2, b[0]))
    lines = []
    for b in boxes:
        yc = (b[1] + b[3]) / 2
        if lines and abs(yc - lines[-1][0]) <= tol:
            lines[-1][1].append(b)
        else:
            lines.append([yc, [b]])
    cols = []
    for _, bs in lines:
        bs.sort(key=lambda b: b[0])
        c, last = 1, bs[0][2]
        for b in bs[1:]:
            if b[0] - last > gap:
                c += 1
            last = max(last, b[2])
        cols.append(c)
    cols.sort()
    numeric = sum(1 for b in boxes if NUM_CELL.match(b[4].replace(" ", ""))) / len(boxes)
    return len(lines), cols[len(cols) // 2], numeric


# ----------------------------------------------------------------------------- the audit
class Audit:
    def __init__(self, doc_id, pdf_path, mineru_path, ocr=True, outline=None, native=None, scan_ocr=None):
        self.doc_id = doc_id
        self.outline = outline                              # the heading list (docs.json); built in run() when None
        self.native = native                                # {page: native.json page}; loaded in run() when None
        self.scan_ocr = scan_ocr                            # {page: {"reason", "lines"}} of the scan pages; read in run() when None
        self.pdf = pymupdf.open(str(pdf_path))
        self.data = load_mineru(mineru_path)
        self.findings = []
        self.counts = Counter()
        self.totals = Counter()
        self.ocr = None
        if ocr:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self.ocr = RapidOCR()
            except Exception:
                self.ocr = None

    def add(self, code, page, block=None, text="", **extra):
        f = {"code": code, "page": page, "block": block, "text": text[:80]}
        f.update(extra)
        self.findings.append(f)
        self.counts[code] += 1

    # ---- per page --------------------------------------------------------------
    def page_checks(self, pg):
        pno = pg["page_idx"] + 1
        page = self.pdf[pg["page_idx"]]
        W, H = page.rect.width, page.rect.height
        blocks = pg["blocks"]
        has_layer = len(page.get_text().strip()) >= 30
        scan = pno in self.scan_ocr
        has_text = has_layer and not scan                   # the text-layer checks (tables, grids, image words) skip scans
        self.totals["pages"] += 1
        if not has_layer:
            self.totals["pages_without_text_layer"] += 1
        rects = {id(b): pymupdf.Rect(b["bbox"][0] * W, b["bbox"][1] * H, b["bbox"][2] * W, b["bbox"][3] * H) for b in blocks}
        figs = [b for b in blocks if b["type"] in ("image", "chart")]
        tables = [b for b in blocks if b["type"] == "table"]
        content = [b for b in blocks if b["type"] not in DROP]
        text_of = {id(b): clean(txt(b)) for b in content}

        # dropped-type blocks: collect for the document-level checks
        prev_content = None                                 # the block before an index block (common.is_toc_index)
        for b in blocks:
            t = clean(txt(b))
            if b["type"] in ("header", "footer", "page_number") and t:
                self.furniture_text[t].add(pno)
                self.furniture_blocks.append((pno, b["index"], b["type"], t))
            elif b["type"] == "index":
                lines = index_entries(b)
                toc = sum(1 for l in lines if TOC_LINE.search(l))
                if not is_toc_index(b, prev_content):
                    self.add("drop_index", pno, b["index"], lines[0], n_lines=len(lines), toc_lines=toc)
                else:
                    self.toc_entries += len(lines)
            if b["type"] not in DROP:
                prev_content = b

        # A1 text inside no block (sub-block bboxes count: captions sit outside the body)
        if scan:
            info = self.scan_ocr[pno]
            self.totals["scan_pages"] += 1
            self.totals[f"scan_pages_{info['reason']}"] += 1
            lines = [l for l in info.get("lines") or [] if l[5] >= OCR_MIN_SCORE]
            where, outside = Counter(), []
            on_page = page_grams(blocks)
            for x0, y0, x1, y1, t, _ in lines:
                n = len(NORM_TEXT.sub("", t))
                if not n:
                    continue
                kind, _, btext = locate((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, blocks)
                if kind == "text" and bigram_recall(t, btext) < 0.5:
                    kind = "text_differs"
                elif kind == "outside" and bigram_recall(t, None, on_page) >= PAGE_RECALL:
                    kind = "outside_in_page_text"           # MinerU has it, in a box that does not reach here
                where[kind] += n
                if kind == "outside":
                    outside.append(t)
            for kind, n in where.items():
                self.totals[f"ocr_chars_{kind}"] += n
            if outside:
                self.add("cov_ocr", pno, None, " / ".join(outside[:6]), n=where["outside"], lines=len(outside))
            caption_lines = [tuple(l[:5]) for l in lines]
        elif self.native is not None and pno in self.native:
            runs, total, missed = uncovered_runs(self.native[pno], blocks)
            self.totals["chars"] += total
            self.totals["chars_uncovered"] += missed
            if missed:
                self.add("cov_text", pno, None, " / ".join(r[4] for r in runs[:6]), n=missed, source="native")
            caption_lines = native_lines(self.native[pno])
        elif has_layer:                                     # no native.json: PyMuPDF words
            boxes = [bb for b in blocks for bb in sub_bboxes(b)]
            words = [w for w in page.get_text("words") if w[4].strip()]
            missed = [w[4] for w in words if not any(bb[0] - 0.005 <= (w[0] + w[2]) / 2 / W <= bb[2] + 0.005 and
                                                     bb[1] - 0.005 <= (w[1] + w[3]) / 2 / H <= bb[3] + 0.005 for bb in boxes)]
            self.totals["chars"] += sum(len(w[4]) for w in words)
            self.totals["chars_uncovered"] += sum(len(t) for t in missed)
            if missed:
                self.add("cov_text", pno, None, " / ".join(missed[:6]), n=sum(len(t) for t in missed), source="pymupdf_words")
            caption_lines = [tuple(w[:5]) for w in words]
        else:
            caption_lines = []
        # tables whose title / date / unit lines sit above them in a figure or dropped block; on a
        # scan page the band above each table is read again at a higher resolution
        hidden = hidden_table_captions(blocks, caption_lines, W, H)
        if scan:
            for idx, band_lines in scan_table_bands(page, blocks).items():
                cap = hidden_table_captions(blocks, [tuple(l[:5]) for l in band_lines], W, H).get(idx)
                if cap and len(cap) >= len(hidden.get(idx, "")):
                    hidden[idx] = cap
        for idx, cap in hidden.items():
            self.add("tab_caption_hidden", pno, idx, cap, caption=cap, source="ocr" if scan else "native")

        # A2 raster images
        rasters = []
        for img in ([] if scan else page.get_images(full=True)):      # a scan page is one image: nothing to match
            for r in page.get_image_rects(img[0]):
                r = r & page.rect
                if r.is_empty or r.get_area() < 1500 or min(r.width, r.height) < 25 or max(r.width, r.height) / max(1, min(r.width, r.height)) > 8:
                    continue                                    # tiny, or a decorative strip
                if r.get_area() >= 0.7 * W * H:
                    self.totals["full_page_images"] += 1
                    continue
                if r.y1 <= 0.08 * H or r.y0 >= 0.92 * H or any(cover(r, rects[id(b)]) >= 0.5 for b in blocks if b["type"] in FURNITURE):
                    continue
                rasters.append(r)
        for r in rasters:
            self.totals["rasters"] += 1
            if not any(cover(r, rects[id(b)]) >= 0.5 for b in figs + tables):
                self.add("cov_raster", pno, None, f"{int(r.width)}x{int(r.height)}pt at y={r.y0 / H:.2f}")

        # A3 vector clusters (drawings outside tables, grouped)
        drawings = page.get_drawings()
        drects = [d["rect"] for d in drawings if not (d["rect"].width < 1.5 and d["rect"].height < 1.5)
                  and d["rect"].get_area() < 0.5 * W * H and not any(cover(d["rect"], rects[id(t)]) >= 0.8 for t in tables)]
        for grp in self._clusters(drects):
            if len(grp) < 8:
                continue
            U = pymupdf.Rect(drects[grp[0]])
            for i in grp[1:]:
                U |= drects[i]
            if U.width < 80 or U.height < 50 or len(page.get_text("words", clip=U)) < 5:
                continue
            self.totals["vector_clusters"] += 1
            if sum(cover(U, rects[id(b)]) for b in figs) < 0.5 and not any(cover(U, rects[id(t)]) >= 0.5 for t in tables):
                self.add("cov_vector", pno, None, f"{len(grp)} objs {int(U.width)}x{int(U.height)}pt at y={U.y0 / H:.2f}")

        # captions of figures and tables
        for b in figs + tables:
            cap = clean("".join(txt(c) for c in b.get("content", []) if isinstance(c, dict) and str(c.get("type", "")).endswith("_caption")))
            if not cap:
                self.totals[f"{b['type']}_without_caption"] += 1
                continue
            m = CAPTION.match(cap)
            if m:
                self.captions[m.group(1)].append((m.group(2), pno, b["index"], b["type"]))
            if cap in self.running_headers:
                self.add("cap_bad", pno, b["index"], cap, reason="running_header")
            elif CHECKBOX.search(cap):
                self.add("cap_bad", pno, b["index"], cap, reason="checkbox")
            elif not m and numbered(cap) and len(cap) <= 40:
                self.add("cap_bad", pno, b["index"], cap, reason="heading_style")
        for b in content:
            t = text_of[id(b)]
            if b["type"] == "text" and CAPTION.match(t) and len(t) <= 80 and not SENT_PUNCT.search(t[:12]):
                self.add("cap_orphan", pno, b["index"], t)

        # tables: candidates among text blocks, ruled grids without a table block
        if has_text:
            for c in find_table_candidates(content, page, W, H, text_of):
                self.totals["table_candidates"] += 1
                if not c["ok"]:
                    self.totals["table_candidates_rejected"] += 1
                    if not c["trig"] and c["why"].startswith("shape 1x"):
                        continue                                # a run of short lines that never formed columns: not worth a line
                first = content[c["run"][0]]
                self.add("tab_candidate", pno, first["index"], text_of[id(first)], ok=c["ok"], why=c["why"], trigger=c["trig"],
                         blocks=[content[k]["index"] for k in c["run"]])
            try:
                found = page.find_tables(strategy="lines").tables
            except Exception:
                found = []
            for t in found:
                if t.row_count < 2 or t.col_count < 2:
                    continue
                R = pymupdf.Rect(t.bbox)
                words = [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words", clip=R) if w[4].strip()]
                if R.width < 60 or R.height < 15 or len(words) < 4 or grid_of_boxes(words)[1] < 2:
                    continue                                    # shaded list bands come back as "tables" with one column of text
                self.totals["ruled_grids"] += 1
                if any(cover(R, rects[id(b)]) >= 0.5 or cover(rects[id(b)], R) >= 0.5 for b in tables):
                    continue
                inside_img = any(cover(R, rects[id(b)]) >= 0.5 for b in figs)
                self.add("tab_ruled_missed", pno, None, f"{t.row_count}x{t.col_count} at y={R.y0 / H:.2f}", in_image=inside_img)

        # images: a grid of text inside?
        for b in figs:
            R = rects[id(b)]
            if b["type"] == "chart" or R.get_area() < 0.015 * W * H or R.y1 <= 0.08 * H:
                continue                                        # a chart's axis labels always form a column; only image blocks are checked
            raster = sum(cover(R, r) for r in rasters) >= 0.5
            boxes = []
            if raster and self.ocr is not None:
                pix = page.get_pixmap(clip=R, dpi=150)
                import numpy as np
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
                res, _ = self.ocr(img)
                for r in res or []:
                    xs = [p[0] for p in r[0]]
                    ys = [p[1] for p in r[0]]
                    boxes.append((min(xs), min(ys), max(xs), max(ys), r[1]))
                gap = 12.0 * 150 / 72
            elif has_text:
                boxes = [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words", clip=R) if w[4].strip()]
                gap = 12.0
            if len(boxes) < 9:
                continue
            n_lines, n_cols, numeric = grid_of_boxes(boxes, gap=gap)
            if n_lines >= 3 and (n_cols >= 3 or (n_cols == 2 and n_lines >= 4 and numeric >= 0.5)):
                self.add("img_maybe_table", pno, b["index"], " ".join(x[4] for x in boxes[:6]), source="ocr" if raster else "text_layer",
                         lines=n_lines, cols=n_cols, numeric=round(numeric, 2))

        # headings and heading-like text blocks (collected; sequences checked at document level)
        titles = [b for b in content if b["type"] in TITLE_TYPES]
        for b in titles:
            t = text_of[id(b)]
            self.headings.append((pno, b["index"], t, b["type"]))
            if t.endswith(("。", "；", ";")) or (not numbered(t) and looks_like_sentence(t)):
                self.add("head_false", pno, b["index"], t, reason="sentence")
            c = rects[id(b)]
            cx, cy = (c.x0 + c.x1) / 2, (c.y0 + c.y1) / 2
            if any(r.x0 <= cx <= r.x1 and r.y0 - 0.01 * H <= cy <= r.y1 for r in (rects[id(tb)] for tb in tables)):
                self.add("head_false", pno, b["index"], t, reason="inside_table")
            for o in titles:
                if o is not b and abs(o["bbox"][1] - b["bbox"][1]) < 0.01 and o["bbox"][0] > b["bbox"][2]:
                    self.add("head_split", pno, b["index"], t + " | " + text_of[id(o)], reason="side_by_side")
            for o in blocks:
                if o["type"] in ("aside_text", "text") and re.fullmatch(r"\d{1,3}", clean(txt(o))) and abs(o["bbox"][1] - b["bbox"][1]) < 0.01 and o["bbox"][2] <= b["bbox"][0] + 0.01:
                    self.add("head_split", pno, b["index"], clean(txt(o)) + " " + t, reason="number_split")
        for i, b in enumerate(content):
            if b["type"] != "text":
                continue
            t = text_of[id(b)]
            if len(t) > 40 or SENT_PUNCT.search(t):
                continue
            if KEYWORD_HEAD.match(t) or numbered(t):
                self.text_heading_like.append((pno, b["index"], t))   # short, numbered, no sentence punctuation

    @staticmethod
    def _clusters(rects, gap=8.0):
        parent = list(range(len(rects)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        exp = [pymupdf.Rect(r.x0 - gap, r.y0 - gap, r.x1 + gap, r.y1 + gap) for r in rects]
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if exp[i].intersects(exp[j]):
                    parent[find(i)] = find(j)
        groups = defaultdict(list)
        for i in range(len(rects)):
            groups[find(i)].append(i)
        return list(groups.values())

    # ---- document level ----------------------------------------------------------
    def outline_checks(self):
        """Step 1: every entry of the heading list (docs.json) is a heading the author listed.
        An entry with no title block carrying its title on the target page (± 1: a heading at a
        page edge is often listed one page off) is searched among the other blocks of those
        pages: a text block that is the title -> head_as_text (confidence = the list's source;
        parsing.repair promotes it), a text block that holds the title next to other text ->
        head_merged (the repair splits the block), a table whose caption or footnote is the
        title -> head_in_caption (the repair moves it out of the table), a table whose body
        holds the title -> head_in_table (report only), nothing at all -> head_outline_missing.
        Figure / table lines of a printed TOC are not headings and are skipped."""
        src = self.outline.get("source", "none")
        heads = [e for e in self.outline.get("entries", []) if e["kind"] == "heading" and e.get("page")]
        self.totals["outline_entries"] = len(heads)
        self.totals["outline_unmatched"] = 0
        pages = {pg["page_idx"] + 1: pg["blocks"] for pg in self.data["pages"]}
        reported = set()
        for e in heads:
            page, t, title, level = e["page"], e["norm"], e["title"], e["level"]
            near = [(p, b) for p in (page, page + 1, page - 1) for b in pages.get(p, [])]
            if any(b["type"] in TITLE_TYPES and bookmark_covers(norm_heading(clean(txt(b))), t) for _, b in near):
                continue
            found = False
            for p, b in near:
                if b["type"] == "text":
                    bt = clean(txt(b))
                    if bookmark_is_block(norm_heading(bt), t):
                        if (p, b["index"]) not in reported:
                            self.add("head_as_text", p, b["index"], bt, confidence=src, level=level, entry=title)
                            reported.add((p, b["index"]))
                        found = True
                    else:
                        span = merged_span(bt, t)
                        if span:
                            if (p, b["index"]) not in reported:
                                self.add("head_merged", p, b["index"], bt[span[0]:span[1]], heading=bt[span[0]:span[1]],
                                         start=span[0], end=span[1], confidence=src, level=level, entry=title)
                                reported.add((p, b["index"]))
                            found = True
                elif b["type"] == "table":
                    for c in b.get("content", []):
                        ctype = str(c.get("type", "")) if isinstance(c, dict) else ""
                        if ctype in ("table_caption", "table_footnote") and bookmark_is_block(norm_heading(clean(txt(c))), t):
                            self.add("head_in_caption", p, b["index"], clean(txt(c)), part=ctype, confidence=src, level=level, entry=title)
                            found = True
                            break
                    if not found and len(t) >= 4 and t in norm_heading(clean(txt(b))):
                        self.add("head_in_table", p, b["index"], title, confidence=src, level=level)
                        found = True
                if found:
                    break
            if not found:
                self.totals["outline_unmatched"] += 1
                self.add("head_outline_missing", page, None, title, source=src, level=level)

    def keyword_promotions(self):
        """Step 2: a short text block opening with 第N章 / 第N节 / 附录 / 前言 is a heading."""
        reported = {(f["page"], f["block"]) for f in self.findings if f["code"] in ("head_as_text", "head_merged")}
        for p, i, s in self.text_heading_like:
            if (p, i) not in reported and KEYWORD_HEAD.match(s):
                self.add("head_as_text", p, i, s, confidence="medium")

    def heading_items(self):
        """The headings as parsing.repair and the chunker will leave them, in reading order:
        (page, block, order within the block, text, is doc_title). Title blocks the chunker
        demotes on their wording (unnumbered and sentence-like, or a numbered list line with
        sentence wording) are left out unless promoted; text blocks of steps 1-2, titles split
        out of merged blocks and table captions / footnotes that are headings are added."""
        items = []
        for p, i, t, btype in self.headings:
            if btype == "paragraph_title" and ((not numbered(t) and looks_like_sentence(t)) or (LIST_ITEM.match(t) and sentence_signals(t))):
                continue
            items.append((p, i, 0, t, btype == "doc_title"))
        for f in self.findings:
            if f["code"] == "head_as_text":
                items.append((f["page"], f["block"], 0, f["text"], False))
            elif f["code"] == "head_merged":
                items.append((f["page"], f["block"], 0, f["heading"], False))
            elif f["code"] == "head_in_caption":
                items.append((f["page"], f["block"], -1 if f["part"] == "table_caption" else 1, f["text"], False))
        items.sort(key=lambda x: (x[0], x[1], x[2]))
        return items

    def heading_sequences(self):
        """Steps 3-4: the level table is learned from the headings of heading_items (numbering
        restarts, common.infer_style_levels); a stack of the headings by level (common.Leveler,
        as the chunker builds its tree) gives every heading its parent. Numbered headings under
        one parent and of one style must be consecutive; a gap is searched among the short
        numbered text blocks of the page range (head_as_text, confidence high), a number that
        repeats or goes backwards is reported. "五、重要会计政策（续）" is the running repeat of a
        heading, not a new one: it takes no place in the stack and no number."""
        items = self.heading_items()
        self.style_order = infer_style_levels([x[3] for x in items])
        leveler = Leveler(self.style_order["levels"])
        promoted = {(f["page"], f["block"]) for f in self.findings if f["code"] in ("head_as_text", "head_merged")}
        stack = []                                          # [(level, item number)]
        last = {}                                           # (parent, style, prefix) -> (n, page)
        for k, (pno, idx, _, t, is_doc_title) in enumerate(items):
            level = leveler.level(t, is_doc_title)
            if CONTINUED.search(t):
                continue
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent = stack[-1][1] if stack else "root"
            stack.append((level, k))
            style, prefix, n = number_key(t)
            if style is None:
                continue
            key = (parent, style, prefix)
            if n == 1 or key not in last:
                last[key] = (n, pno)
                continue
            prev, prev_page = last[key]
            if n > prev + 1:
                missing = list(range(prev + 1, n))
                found = [(p, i, s) for p, i, s in self.text_heading_like
                         if prev_page <= p <= pno and (p, i) not in promoted and number_key(s)[:2] == (style, prefix) and number_key(s)[2] in missing]
                self.add("head_gap", pno, idx, t, style=style, missing=missing, pages=[prev_page, pno], candidates=found[:3])
                for p, i, s in found:
                    self.add("head_as_text", p, i, s, confidence="high", fills=missing)
                    promoted.add((p, i))
            elif n <= prev:
                self.add("head_gap", pno, idx, t, style=style, missing=[], pages=[prev_page, pno], reason="repeat_or_backwards")
            last[key] = (n, pno)
        self.totals["headings_checked"] = len(items)

    def caption_sequences(self):
        for kind, items in self.captions.items():
            nums = []
            for num, pno, idx, btype in items:
                m = re.match(r"^[A-Z]?(\d+)(?:[-.](\d+))?$", num)
                if not m:
                    continue
                nums.append(((int(m.group(1)), int(m.group(2)) if m.group(2) else None), pno, idx))
            simple = [(a, p, i) for (a, b), p, i in nums if b is None]
            seen = Counter(a for a, _, _ in simple)
            for a, c in seen.items():
                if c > 1:
                    self.add("cap_gap", 0, None, f"{kind}{a}", reason="duplicate", count=c)
            if simple:
                present = {a for a, _, _ in simple}
                missing = sorted(set(range(1, max(present) + 1)) - present)
                if missing:
                    self.add("cap_gap", 0, None, f"{kind} missing {missing[:10]}", reason="missing", missing=missing)
            self.totals[f"captions_{kind}"] += len(items)

    def furniture_checks(self):
        for pno, idx, btype, t in self.furniture_blocks:
            if len(self.furniture_text[t]) == 1 and len(t) > 15 and not re.search(r"\d+\s*of\s*\d+|第\s*\d+\s*页|共\s*\d+\s*页", t):
                self.add("drop_header", pno, idx, t, block_type=btype)

    def run(self):
        self.furniture_text = defaultdict(set)
        self.furniture_blocks = []
        self.toc_entries = 0
        self.captions = defaultdict(list)
        self.headings = []
        self.text_heading_like = []
        # the running header: header texts seen on 3+ pages (captions equal to one are wrong)
        cnt = Counter()
        for pg in self.data["pages"]:
            for b in pg["blocks"]:
                if b["type"] == "header":
                    cnt[clean(txt(b))] += 1
        self.running_headers = {t for t, c in cnt.items() if c >= 3 and t}
        if self.native is None:
            self.native = load_native(self.doc_id)
        if self.scan_ocr is None:
            self.scan_ocr = load_ocr(self.doc_id, self.pdf, scan_pages(self.pdf))
        for pg in self.data["pages"]:
            self.page_checks(pg)
        if self.outline is None:
            self.outline = build_outline(self.doc_id, self.pdf, self.data)
        self.outline_checks()
        self.keyword_promotions()
        self.heading_sequences()
        self.caption_sequences()
        self.furniture_checks()
        chars = max(1, self.totals["chars"])
        ocr_total = sum(v for k, v in self.totals.items() if k.startswith("ocr_chars_"))
        summary = {"doc_id": self.doc_id, "pages": self.totals["pages"], "pages_without_text_layer": self.totals["pages_without_text_layer"],
                   "native": self.native is not None,
                   "char_coverage": round(1 - self.totals["chars_uncovered"] / chars, 5), "chars_uncovered": self.totals["chars_uncovered"],
                   "scan_pages": self.totals["scan_pages"], "ocr_chars": ocr_total, "ocr_chars_outside": self.totals["ocr_chars_outside"],
                   "ocr_chars_in_figures": self.totals["ocr_chars_figure"], "ocr_chars_in_dropped": self.totals["ocr_chars_dropped"],
                   "tables_with_hidden_caption": self.counts["tab_caption_hidden"],
                   "headings": len(self.headings), "toc_entries": self.toc_entries, "ruled_grids": self.totals["ruled_grids"],
                   "outline_source": self.outline.get("source", "none"), "outline_entries": self.totals["outline_entries"],
                   "outline_unmatched": self.totals["outline_unmatched"],
                   "table_candidates_ok": self.totals["table_candidates"] - self.totals["table_candidates_rejected"],
                   "table_candidates_rejected": self.totals["table_candidates_rejected"]}
        return {"summary": summary, "counts": dict(self.counts), "totals": dict(self.totals), "findings": self.findings,
                "style_order": self.style_order}
