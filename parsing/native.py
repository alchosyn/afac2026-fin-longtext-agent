# -*- coding: utf-8 -*-
"""Native PDF data, page kinds and OCR for the coverage checks (quality gate 1) and the text
recovery (parsing.repair).

native.json (script/export_native.py, run with the MinerU venv): the characters MinerU itself
reads through docvortex / pypdfium2, with their boxes, grouped in lines and spans; coordinates in
PDF points of the displayed page (rotation applied), the same frame as MinerU's block fractions.

Page kinds. A page is a scan when one image covers at least SCAN_IMAGE_SHARE of it and the page
either has no text layer (fewer than MIN_TEXT_CHARS characters) or paints some of its text before
that image, i.e. under it. The second case is the contracts' signature pages: the signed and
stamped printout is scanned and laid over the Word page, whose draft text stays underneath and
differs from what is visible (fc_text02 p236: "公司法定代表人签名" and an empty "年 月 日" under the
scan's "法定代表人（签字）" and "2026 年 4 月 16 日"; the scan is also shifted, so that text never lines
up with MinerU's boxes). Text painted over the image is real text and the page stays digital:
slide backgrounds (60 research pages), designed annual-report pages, page numbers stamped on a
scan. Counted over raw/ (2026-09-27): contracts 103 image-only + 112 text-under + 24 text on both
sides + 9 text-over; annual reports 58 + 2 + 2 + 6; regulatory 5 image-only; research 2 + 0 + 3 + 60.

OCR. A scan page is read with RapidOCR at OCR_DPI; its lines are cached in
processed_data/<doc_id>/ocr.json as [x0, y0, x1, y1, text, score] in points of the displayed page.
"""
import json
import re
from collections import Counter
from pathlib import Path

from .common import PROC

SCAN_IMAGE_SHARE = 0.7
MIN_TEXT_CHARS = 30
OCR_DPI = 150
OCR_MIN_SCORE = 0.5
TOL = 0.005                    # page fraction: tolerance of the inside-a-block test
COL_GAP = 25.0                 # pt: a gap this wide between two uncovered characters splits the run
DROP = {"header", "footer", "page_number", "index"}
FIGURE = {"image", "chart"}
NORM = re.compile(r"[\s，。、：:；;（）()《》“”\"'_\-—.·【】]")
_OCR = []


def ocr_engine():
    """The shared RapidOCR engine (None when rapidocr_onnxruntime is missing)."""
    if not _OCR:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR.append(RapidOCR())
        except Exception:                                   # noqa: BLE001
            _OCR.append(None)
    return _OCR[0]


# ----------------------------------------------------------------------------- page kinds
def page_kind(page):
    """("scan", "no_text_layer" | "text_under_image") or ("digital", None | "text_over_image")."""
    import pymupdf
    area = page.rect.get_area()
    full = any(not r.is_empty and (r & page.rect).get_area() >= SCAN_IMAGE_SHARE * area
               for img in page.get_images(full=True) for r in page.get_image_rects(img[0]))
    if not full:
        return "digital", None
    if len(page.get_text().strip()) < MIN_TEXT_CHARS:
        return "scan", "no_text_layer"
    log = page.get_bboxlog()
    big = next((i for i, (t, r) in enumerate(log) if "image" in t and pymupdf.Rect(r).get_area() >= SCAN_IMAGE_SHARE * area), None)
    if big is None:
        return "digital", None
    if any("text" in t for t, _ in log[:big]):
        return "scan", "text_under_image"
    return "digital", "text_over_image"


def scan_pages(pdf):
    """{page number (1-based): reason} of the scan pages of a document."""
    out = {}
    for page in pdf:
        kind, reason = page_kind(page)
        if kind == "scan":
            out[page.number + 1] = reason
    return out


# ----------------------------------------------------------------------------- native data
def load_native(doc_id):
    """native.json of a document as {page number: page dict}, or None when it was not exported."""
    path = PROC / doc_id / "native.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {p["page"]: p for p in data["pages"]}


def block_boxes(blocks):
    boxes = [b["bbox"] for b in blocks]
    boxes += [c["bbox"] for b in blocks for c in b.get("content", []) if isinstance(c, dict) and "bbox" in c]
    return boxes


def inside(cx, cy, boxes):
    return any(b[0] - TOL <= cx <= b[2] + TOL and b[1] - TOL <= cy <= b[3] + TOL for b in boxes)


def _containers(cx, cy, boxes):
    """[(block, sub-block or None)] of every box holding the point, smallest first."""
    hits = []
    for bb, block, sub in boxes:
        if bb[0] - TOL <= cx <= bb[2] + TOL and bb[1] - TOL <= cy <= bb[3] + TOL:
            hits.append(((bb[2] - bb[0]) * (bb[3] - bb[1]), id(sub) if sub is not None else 0, block, sub))
    hits.sort(key=lambda h: (h[0], h[1]))
    return [(h[2], h[3]) for h in hits]


def uncovered_runs(npage, blocks, margin=None):
    """Characters of a native page that no block carries, as word-like runs (x0, y0, x1, y1, text)
    in points; a run of rotated text (line rotation != 0, e.g. a landscape table printed turned on
    a portrait page: fr_cscec_2025 has 3,317 such lines) carries a sixth element "rotated" and is
    kept in the native reading order. A character counts as carried when its box centre lies in a
    block or sub-block AND, for a text, title, table, caption or dropped block, the run of the
    line's characters in that box is found in the block's text (character-bigram recall >= 0.5):
    a box that merely overlaps the end of a neighbouring line does not carry it (fr_catl_2024 p59:
    the second half of "说明：……公司报告期内盈利且母公司可供股东分配利润为正但未提出现金红利分配预案" fell in
    the box of the "□适用 ☑不适用" line below and was lost). Characters in a figure or table body count
    as carried: the figure's words become its labels, and the chunker projects a table's cells from
    the text layer inside its box (MinerU's own cell text is often scrambled, so it is no reference).
    A stretch is checked against every box it lies in, not only the smallest (ins_03 p5: the 第十条
    title the heading repair split out of its block shares that block's box), and a stretch no box
    carries is checked against the blocks that carry the rest of its line: the tail of a line that
    runs past its paragraph's box is not recovered a second time (ins_01 p16 "吞咽的状态。"). A check
    against the whole page's text was tried and dropped: annual-report pages repeat their terms, so
    table-side lines like "企业所得税" or "持股5%以上股东……出借股份情况" (fr_byd_2024 p185, p128) and the
    header of the recovered 资本负债率 table (p255) all looked carried and were lost. Uncovered runs of a horizontal line are broken
    at a carried character or at a gap wider than COL_GAP. Spaces PDFium generates between runs
    (spans of render mode None: "持股 5%以上股东") are left out. margin (page fraction) skips the
    furniture bands at the top and bottom. Returns (runs, n_chars, n_uncovered)."""
    from .common import clean, txt
    W, H = npage["size"]
    boxes = [(b["bbox"], b, None) for b in blocks]
    boxes += [(c["bbox"], b, c) for b in blocks for c in b.get("content", []) if isinstance(c, dict) and "bbox" in c]
    texts = {}

    def grams_of(block, sub):
        key = id(sub if sub is not None else block)
        if key not in texts:
            texts[key] = text_grams(clean(txt(sub if sub is not None else block)))
        return texts[key]

    runs, total, missed = [], 0, 0
    for line in npage["lines"]:
        rotated = abs(line.get("rotation") or 0) > 0.01
        seq = []                                            # [char, box, state]; state: True carried, False not, None space
        for sp in line["spans"]:
            generated = sp.get("mode") is None
            for ch, cb in zip(sp["text"], sp.get("chars") or []):
                if not ch.strip():
                    if not generated:
                        seq.append([ch, cb, None])
                    continue
                cx, cy = (cb[0] + cb[2]) / 2 / W, (cb[1] + cb[3]) / 2 / H
                if margin is not None and (cy < margin or cy > 1 - margin):
                    continue
                total += 1
                seq.append([ch, cb, tuple(_containers(cx, cy, boxes))])
        # stretches: consecutive characters with the same smallest box (or none)
        stretches, i = [], 0
        while i < len(seq):
            if seq[i][2] is None:
                i += 1
                continue
            key = seq[i][2][0] if seq[i][2] else None
            j = i
            while j < len(seq) and (seq[j][2] is None or ((seq[j][2][0] if seq[j][2] else None) == key)):
                j += 1
            conts = []
            for c in seq[i:j]:
                if c[2]:
                    conts += [x for x in c[2] if all(x[0] is not y[0] or x[1] is not y[1] for y in conts)]
            stretches.append([i, j, "".join(c[0] for c in seq[i:j] if c[2] is not None), conts, None])
            i = j

        def verdict(seg, conts):
            for block, sub in conts:
                sub_type = str(sub.get("type", "")) if sub is not None else ""
                if block["type"] in FIGURE | {"table"} and not sub_type.endswith(("_caption", "_footnote")):
                    return True                             # figure labels and table cells are read from the text layer later
                if bigram_recall(seg, None, grams_of(block, sub)) >= 0.5:
                    return True
            return False

        for st in stretches:                                # 1. against every box the stretch lies in
            st[4] = verdict(st[2], st[3])
        line_blocks = [c for st in stretches if st[4] for c in st[3]]
        for st in stretches:                                # 2. the tail of a line beyond its own block's box
            if not st[4] and line_blocks:
                st[4] = verdict(st[2], line_blocks)
        for i, j, _, _, carried in stretches:
            for k in range(i, j):
                if seq[k][2] is not None:
                    seq[k][2] = carried
        cur = None
        for ch, cb, state in seq:
            if state is None:
                if cur is not None:
                    cur[4] += " "
                continue
            if state:
                if cur is not None:
                    runs.append(cur)
                    cur = None
                continue
            missed += 1
            if cur is not None and not rotated and cb[0] - cur[2] > COL_GAP:
                runs.append(cur)
                cur = None
            if cur is None:
                cur = [cb[0], cb[1], cb[2], cb[3], ch] + (["rotated"] if rotated else [])
            else:
                cur[0], cur[1] = min(cur[0], cb[0]), min(cur[1], cb[1])
                cur[2], cur[3] = max(cur[2], cb[2]), max(cur[3], cb[3])
                cur[4] += ch
        if cur is not None:
            runs.append(cur)
    out = []
    for r in runs:
        text = r[4].strip()
        if text:
            out.append((r[0], r[1], r[2], r[3], text) + tuple(r[5:]))
    return out, total, missed


def native_lines(npage):
    """[(x0, y0, x1, y1, text)] of the native lines of a page."""
    out = []
    for line in npage["lines"]:
        text = "".join(sp["text"] for sp in line["spans"]).strip()
        if text:
            out.append(tuple(line["bbox"]) + (text,))
    return out


# ----------------------------------------------------------------------------- where a line falls
def locate(cx, cy, blocks):
    """(kind, block, text) of the smallest block or sub-block box holding the point (page fractions):
    kind is "text" (any content block or caption), "figure", "dropped" or "outside"."""
    best = None
    for b in blocks:
        for bb, t, obj in [(b["bbox"], b["type"], b)] + [(c["bbox"], str(c.get("type", "")), c) for c in b.get("content", [])
                                                         if isinstance(c, dict) and "bbox" in c]:
            if bb[0] - TOL <= cx <= bb[2] + TOL and bb[1] - TOL <= cy <= bb[3] + TOL:
                a = (bb[2] - bb[0]) * (bb[3] - bb[1])
                if best is None or a < best[0]:
                    best = (a, t, b, obj)
    if best is None:
        return "outside", None, ""
    from .common import clean, txt
    _, t, block, obj = best
    kind = "dropped" if block["type"] in DROP else "figure" if block["type"] in FIGURE and t in FIGURE | {"image_body", "chart_body"} else "text"
    return kind, block, clean(txt(obj))


def bigram_recall(a, b, grams_b=None):
    """Share of a's character bigrams found in b (both normalised): same content >= 0.5.
    grams_b: b's bigram set when the caller reuses it (see text_grams); b may then be None."""
    a = NORM.sub("", a)
    if grams_b is None:
        b = NORM.sub("", b)
        grams_b = {b[i:i + 2] for i in range(len(b) - 1)}
    if not a:
        return 1.0
    if len(a) == 1:
        return float(a in b) if b is not None else float(any(a in g for g in grams_b))
    grams = [a[i:i + 2] for i in range(len(a) - 1)]
    return sum(g in grams_b for g in grams) / len(grams)


PAGE_RECALL = 0.8              # a stretch found this well in the page's MinerU text is carried, wherever its box is


def text_grams(text):
    t = NORM.sub("", text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def page_grams(blocks):
    """Bigram set of all the text MinerU put on a page (every block and sub-block)."""
    from .common import clean, txt
    return text_grams("".join(clean(txt(b)) for b in blocks))


# ----------------------------------------------------------------------------- OCR
def ocr_page(page, dpi=OCR_DPI, clip=None):
    """[[x0, y0, x1, y1, text, score]] of the RapidOCR lines of a rendered page (or of the clip
    rectangle, in points of the displayed page), in points of the displayed page."""
    import numpy as np
    import pymupdf
    engine = ocr_engine()
    if engine is None:
        return []
    rect = page.rect if clip is None else pymupdf.Rect(clip) & page.rect
    if rect.is_empty:
        return []
    pix = page.get_pixmap(dpi=dpi, clip=None if clip is None else rect)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    res, _ = engine(img)
    sx, sy = rect.width / pix.width, rect.height / pix.height
    out = []
    for box, text, score in res or []:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        out.append([round(rect.x0 + min(xs) * sx, 2), round(rect.y0 + min(ys) * sy, 2), round(rect.x0 + max(xs) * sx, 2),
                    round(rect.y0 + max(ys) * sy, 2), str(text), round(float(score), 3)])
    return out


def load_ocr(doc_id, pdf, scans, recompute=False):
    """{page: {"reason", "lines"}} of a document's scan pages, from processed_data/<doc_id>/ocr.json;
    pages missing there are read now and the file is updated."""
    path = PROC / doc_id / "ocr.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() and not recompute else {}
    pages = {int(k): v for k, v in data.get("pages", {}).items()}
    changed = False
    for pno, reason in sorted(scans.items()):
        if pno not in pages or pages[pno].get("dpi") != OCR_DPI:
            pages[pno] = {"reason": reason, "dpi": OCR_DPI, "lines": ocr_page(pdf[pno - 1])}
            changed = True
        pages[pno]["reason"] = reason
    pages = {p: v for p, v in pages.items() if p in scans}
    if changed or set(pages) != {int(k) for k in data.get("pages", {})}:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"doc_id": doc_id, "engine": "rapidocr_onnxruntime", "pages": {str(k): v for k, v in sorted(pages.items())}},
                                   ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return pages


def ocr_uncovered(lines, blocks, W, H, margin=None):
    """OCR lines (score >= OCR_MIN_SCORE) whose centre lies in no block and whose text is not in
    the page's MinerU text (recall < PAGE_RECALL), as word-like tuples."""
    boxes = block_boxes(blocks)
    on_page = page_grams(blocks)
    out = []
    for x0, y0, x1, y1, text, score in lines:
        if score < OCR_MIN_SCORE or not NORM.sub("", text):
            continue
        cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
        if margin is not None and (cy < margin or cy > 1 - margin):
            continue
        if not inside(cx, cy, boxes) and bigram_recall(text, None, on_page) < PAGE_RECALL:
            out.append((x0, y0, x1, y1, text))
    return out


# ----------------------------------------------------------------------------- numbers of a scan-page table
NUMBER = re.compile(r"\d[\d,，]*(?:\.\d+)?")
MIN_DIGITS = 3                 # a number this long is checked; shorter ones (years' tails, row numbers) are noise


def numbers_of(text, min_digits=MIN_DIGITS):
    return [n.replace(",", "").replace("，", "") for n in NUMBER.findall(text) if sum(ch.isdigit() for ch in n) >= min_digits]


def check_scan_numbers(rows, lines, bbox_pt, W, H):
    """A scan-page table's numbers against an independent reading: every number of MIN_DIGITS
    digits or more in MinerU's cells is looked for among the numbers of the RapidOCR lines
    (score >= OCR_MIN_SCORE) whose centre lies in the table box. Nothing is changed: the table
    keeps MinerU's cells (its VLM read the 23 scan-page tables well: 820 of 821 numbers agreed
    on 2026-09-30, the one difference being handwriting in a stamp) and the record gets the
    count, so a wrong number can be traced. Returns {"numbers", "confirmed", "unconfirmed":
    [numbers no OCR line carries], "ocr_extra": OCR numbers no cell carries}."""
    x0, y0, x1, y1 = bbox_pt
    tol = 0.01 * max(W, H)
    pool = Counter()
    for l in lines:
        if l[5] < OCR_MIN_SCORE:
            continue
        cx, cy = (l[0] + l[2]) / 2, (l[1] + l[3]) / 2
        if x0 - tol <= cx <= x1 + tol and y0 - tol <= cy <= y1 + tol:
            pool.update(numbers_of(l[4]))
    cell_numbers = [n for row in rows for c in row for n in numbers_of(c)]
    confirmed, unconfirmed = 0, []
    for n in cell_numbers:
        if pool[n] > 0:
            pool[n] -= 1
            confirmed += 1
        else:
            unconfirmed.append(n)
    return {"numbers": len(cell_numbers), "confirmed": confirmed, "unconfirmed": unconfirmed[:20], "ocr_extra": sum(pool.values())}


# ----------------------------------------------------------------------------- titles hidden next to a table
TABLE_TITLE = re.compile(r"表\s*([（(]\s*续\s*[)）])?\s*$|表\s*[（(]\s*续")
DATE_LINE = re.compile(r"^\d{4}\s*年(\s*\d{1,2}\s*月(\s*\d{1,2}\s*日)?|度)?")
UNIT_LINE = re.compile(r"单位|人民币\s*.{0,2}元|^(千|万|百万|亿)?元$")
COMPANY_LINE = re.compile(r"(公司|集团)$")
TITLE_BAND = 100.0             # pt above a table searched for its title lines


def hidden_table_captions(blocks, lines, W, H):
    """{table block index: caption text} for tables with title lines above them that MinerU put in
    a figure block (a seal pressed on the title: fr_cscec_2025 p146-162, all 17 audited statements)
    or in a dropped block (the unit "人民币千元" labelled header on the same pages). lines are
    (x0, y0, x1, y1, text) in points; only lines inside figure or dropped blocks, within
    TITLE_BAND above the table and below the nearest content block above it, overlapping the
    table horizontally, and reading like a statement title, a date, a unit or a company name are
    used, and at least one must be a title or a unit."""
    out = {}
    for t in blocks:
        if t["type"] != "table":
            continue
        tx0, ty0, tx1 = t["bbox"][0] * W, t["bbox"][1] * H, t["bbox"][2] * W
        upper = ty0 - TITLE_BAND
        for b in blocks:
            if b is t or b["type"] in DROP or b["type"] in FIGURE:
                continue
            bx0, by0, bx1, by1 = b["bbox"][0] * W, b["bbox"][1] * H, b["bbox"][2] * W, b["bbox"][3] * H
            if by1 <= ty0 + 1 and min(bx1, tx1) - max(bx0, tx0) > 0:
                upper = max(upper, by1)
        picked = []
        for x0, y0, x1, y1, text in lines:
            if y1 > ty0 + 2 or y0 < upper - 2 or min(x1, tx1 + 0.05 * W) - max(x0, tx0 - 0.05 * W) <= 0:
                continue
            kind, _, _ = locate((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, blocks)
            if kind not in ("figure", "dropped"):
                continue
            s = text.strip()
            if TABLE_TITLE.search(s) or DATE_LINE.search(s) or UNIT_LINE.search(s) or COMPANY_LINE.search(s):
                picked.append((y0, x0, s))
        if any(TABLE_TITLE.search(s) or UNIT_LINE.search(s) for _, _, s in picked):
            picked.sort(key=lambda x: (bool(UNIT_LINE.search(x[2])), x[0], x[1]))        # company, title, date, then the unit
            out[t["index"]] = " ".join(s for _, _, s in picked)
    return out


BAND_OCR_DPI = 200             # a seal over the title defeats OCR at 150 dpi (fr_cscec_2025 p147 read only "续")


def scan_table_bands(page, blocks):
    """{table index: OCR lines of the band above the table, read again at BAND_OCR_DPI}, for the
    tables of a scan page that have a figure or dropped block in that band."""
    W, H = page.rect.width, page.rect.height
    out = {}
    for t in blocks:
        if t["type"] != "table":
            continue
        tx0, ty0, tx1 = t["bbox"][0] * W, t["bbox"][1] * H, t["bbox"][2] * W
        band = (max(0, tx0 - 0.05 * W), max(0, ty0 - TITLE_BAND), min(W, tx1 + 0.05 * W), ty0 + 2)
        near = [b for b in blocks if (b["type"] in DROP or b["type"] in FIGURE) and b["bbox"][3] * H > band[1]
                and b["bbox"][1] * H < band[3] and min(b["bbox"][2] * W, band[2]) - max(b["bbox"][0] * W, band[0]) > 0]
        if near:
            out[t["index"]] = [l for l in ocr_page(page, dpi=BAND_OCR_DPI, clip=band) if l[5] >= OCR_MIN_SCORE]
    return out
