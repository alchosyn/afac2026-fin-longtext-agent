# -*- coding: utf-8 -*-
"""Figure images and figure text of the structure layer.

Every figure record (MinerU image or chart block after the repairs, captioned or not, bitmap or
vector) is cut out of the PDF page with MinerU's box at FIGURE_DPI and a small pad and saved
as processed_data/<doc_id>/figures/<local id>.png (2026-10-28; before that only captioned
figures were cut). The record carries the path (field `image`, relative to the repository
root), the figure chunk copies the pointer, and a retrieved figure chunk hands the file to the
multimodal judge (agent.figures).

The text inside a figure (figure_text) is the PDF's own text layer in the box, or, when the box
has none or the page is a scan (its layer is the draft under the scan), RapidOCR on a
TEXT_DPI rendering; OCR results are cached in processed_data/<doc_id>/figure_text.json. From
it come, by rules only (nothing is generated from the image):
  * tokens -> the naming words of the figure (parsing.structure.label_words)
  * numbers -> every number written in the figure, in reading order (top to bottom, left to
    right): axis ticks, data labels, years, as written ("14,000", "25.0%", "(10,453)")
  * the first line -> the caption of a figure MinerU gave none, when it is a title: the
    verdicts are hand-reviewed in processed_data/figure_title_review.json (title / not_title);
    a first line not in the review is kept as an unreviewed candidate, never as a caption

    from parsing.figures import describe_figure, FigureTextCache
    cache = FigureTextCache("pack2_text01")
    info = describe_figure(page, bbox, "pack2_text01", 20, "f29", caption="", text_layer=True, cache=cache)
    cache.save()
"""
import json
import re
from pathlib import Path

import pymupdf

from .common import PROC

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DPI = 150      # the judge read 9 of 10 hand-checked charts at this resolution (2026-09-24)
PAD_PT = 4            # points of page around the figure box, so edge labels are not clipped
MIN_SIDE = 4          # points: a thinner box is a parsing artefact, not a figure
TEXT_DPI = 200        # chart labels are small: OCR renders the figure at this resolution
OCR_MIN_SCORE = 0.5   # RapidOCR confidence below which a line is noise
LEGEND_GLYPHS = re.compile(r"[■□●○◆◇▲△▪▫◼◻]+")
NUMBER = re.compile(r"(?<![0-9A-Za-z.])\(?[-−–]?\d[\d,]*(?:\.\d+)?\)?%?")   # "2018-2022" is two years, not 2018 and -2022
CONTENT = re.compile(r"[A-Za-z一-鿿]")
NUMBER_ONLY = re.compile(r"[-−–+]?\(?[\d,.]+\)?%?")
REVIEW_PATH = PROC / "figure_title_review.json"


# ----------------------------------------------------------------------------- the image
def figure_rect(page, bbox, pad=PAD_PT):
    """The clip rectangle in page points of a box given in page fractions (the MinerU
    builds) or in page points (the legacy builds: integers, pymupdf4llm's coordinates)."""
    W, H = page.rect.width, page.rect.height
    x0, y0, x1, y1 = bbox
    if max(bbox) > 1.5:                                   # points already
        return pymupdf.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad) & page.rect
    return pymupdf.Rect(x0 * W - pad, y0 * H - pad, x1 * W + pad, y1 * H + pad) & page.rect


def render_figure(page, bbox, dpi=FIGURE_DPI, pad=PAD_PT):
    """PNG bytes of the box cut from the page; None when the box is degenerate."""
    rect = figure_rect(page, bbox, pad)
    if rect.is_empty or rect.width < MIN_SIDE or rect.height < MIN_SIDE:
        return None
    return page.get_pixmap(dpi=dpi, clip=rect).tobytes("png")


def figure_path(doc_id, local_id):
    return PROC / doc_id / "figures" / f"{local_id}.png"


def save_figure(page, bbox, doc_id, local_id, dpi=FIGURE_DPI, pad=PAD_PT, force=False):
    """Write the figure's PNG unless it exists (force=True rewrites). Returns the path
    relative to the repository root, or None when the box is degenerate."""
    path = figure_path(doc_id, local_id)
    if not path.exists() or force:
        png = render_figure(page, bbox, dpi, pad)
        if png is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
    return str(path.relative_to(ROOT)).replace("\\", "/")


# ----------------------------------------------------------------------------- the text
class FigureTextCache:
    """OCR results of a document's figures: processed_data/<doc_id>/figure_text.json,
    {"<page>:<box>": [[x0, y0, x1, y1, text], ...]} in RapidOCR's order."""

    def __init__(self, doc_id):
        self.path = PROC / doc_id / "figure_text.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.dirty = False

    @staticmethod
    def key(pno, bbox):
        return f"{pno}:" + ",".join(f"{x:.4f}" for x in bbox)

    def get(self, pno, bbox):
        return self.data.get(self.key(pno, bbox))

    def put(self, pno, bbox, items):
        self.data[self.key(pno, bbox)] = items
        self.dirty = True

    def save(self):
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
            self.dirty = False


def ocr_items(page, rect):
    """[[x0, y0, x1, y1, text]] of the RapidOCR lines of the rectangle (page points), lines
    below OCR_MIN_SCORE dropped, in RapidOCR's order."""
    from .native import ocr_engine
    import numpy as np
    engine = ocr_engine()
    if engine is None or rect.is_empty:
        return []
    pix = page.get_pixmap(dpi=TEXT_DPI, clip=rect)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    res, _ = engine(img)                                  # with the angle classifier: vertical labels read the right way up
    # the classifier also turns some horizontal number strings upside down ("16000" -> "00091",
    # pack2_text01 图29); horizontal text in a document is never upside down, so a wide box
    # takes the reading without the classifier (same detection, same boxes)
    plain = {tuple((round(p[0]), round(p[1])) for p in box): (text, score)
             for box, text, score in (engine(img, use_cls=False)[0] or [])}
    sx, sy = rect.width / pix.width, rect.height / pix.height
    out = []
    for box, text, score in res or []:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        key = tuple((round(p[0]), round(p[1])) for p in box)
        if max(xs) - min(xs) >= max(ys) - min(ys) and key in plain:
            text, score = plain[key]
        if float(score) < OCR_MIN_SCORE:
            continue
        out.append([round(rect.x0 + min(xs) * sx, 1), round(rect.y0 + min(ys) * sy, 1),
                    round(rect.x0 + max(xs) * sx, 1), round(rect.y0 + max(ys) * sy, 1), str(text)])
    return out


def visual_lines(items):
    """Items [x0, y0, x1, y1, text] grouped into lines (vertical centres within half a line
    height), top to bottom; each line is its items left to right."""
    lines = []
    for it in sorted(items, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        cy, h = (it[1] + it[3]) / 2, max(1.0, it[3] - it[1])
        for ln in reversed(lines[-3:]):
            if abs(cy - ln["cy"]) <= max(2.0, 0.5 * min(h, ln["h"])):
                ln["items"].append(it)
                break
        else:
            lines.append({"cy": cy, "h": h, "items": [it]})
    out = []
    for ln in sorted(lines, key=lambda l: l["cy"]):
        ws = sorted(ln["items"], key=lambda w: w[0])
        out.append({"text": join_items(ws, ln["h"]), "items": ws, "h": ln["h"]})
    return out


def join_items(ws, h):
    """The items of one line as text: a space where the gap is wider than 0.6 line height."""
    text = ws[0][4] if ws else ""
    for a, b in zip(ws, ws[1:]):
        text += (" " if b[0] - a[2] > 0.6 * h else "") + b[4]
    return text.strip()


def figure_text(page, bbox, pno, text_layer=True, cache=None, stats=None):
    """The text inside a figure box: {"source": "text_layer" | "ocr" | None, "tokens": [...]
    (the words in the order the labels have always been read: the layer's own order, or
    RapidOCR's lines split at legend glyphs and spaces), "lines": visual_lines}.
    text_layer=False on a scan page (its layer is the draft under the scan)."""
    rect = figure_rect(page, bbox, pad=0)
    if text_layer:
        words = page.get_text("words", clip=rect)
        if words:
            items = [[w[0], w[1], w[2], w[3], w[4]] for w in words]
            return {"source": "text_layer", "tokens": [w[4] for w in words], "lines": visual_lines(items)}
    items = cache.get(pno, bbox) if cache is not None else None
    if items is None:
        items = ocr_items(page, rect)
        if stats is not None:
            stats["figure_ocr"] += 1
        if cache is not None:
            cache.put(pno, bbox, items)
    tokens = [tok for it in items for tok in LEGEND_GLYPHS.split(it[4]) for tok in tok.split()]
    return {"source": "ocr" if items else None, "tokens": tokens, "lines": visual_lines(items)}


def figure_numbers(lines):
    """Every number written in the figure, reading order, as written; unicode minus signs
    become "-"."""
    out = []
    for ln in lines:
        for it in ln["items"]:
            for m in NUMBER.findall(it[4]):
                out.append(m.replace("−", "-").replace("–", "-"))
    return out


def first_line(lines):
    """The top line of the figure that has a letter or a CJK character, without the number
    items at its two ends: an axis tick on the title's line ("500 周频公募权益产品新成立份额
    (亿份)", pack2_text08 p5) is not part of the title; a line of numbers only is skipped."""
    for ln in lines:
        ws = list(ln["items"])
        while ws and NUMBER_ONLY.fullmatch(ws[0][4].strip()):
            ws.pop(0)
        while ws and NUMBER_ONLY.fullmatch(ws[-1][4].strip()):
            ws.pop()
        text = join_items(ws, ln["h"])
        if CONTENT.search(text):
            return text
    return ""


# ----------------------------------------------------------------------------- the review
_REVIEW = {}


def title_verdict(doc_id, pno, text):
    """The hand review of a first line: "title", "not_title", or "unreviewed"."""
    if "items" not in _REVIEW:
        data = json.loads(REVIEW_PATH.read_text(encoding="utf-8")) if REVIEW_PATH.exists() else {"items": []}
        _REVIEW["items"] = {(r["doc_id"], r["page"], r["text"]): r["verdict"] for r in data["items"]}
    return _REVIEW["items"].get((doc_id, pno, text), "unreviewed")


def describe_figure(page, bbox, doc_id, pno, local_id, caption="", text_layer=True, cache=None, stats=None):
    """What the structure layer records about one figure besides its box and MinerU caption:
    {"tokens", "numbers", "image", "caption", "caption_from", "title_candidate"} (see the
    module docstring). The PNG is written every time (force)."""
    ft = figure_text(page, bbox, pno, text_layer=text_layer, cache=cache, stats=stats)
    out = {"tokens": ft["tokens"], "numbers": figure_numbers(ft["lines"]), "caption": caption,
           "caption_from": "mineru" if caption else None, "title_candidate": None,
           "image": save_figure(page, bbox, doc_id, local_id, force=True)}
    if not caption:
        line = first_line(ft["lines"])
        if line:
            verdict = title_verdict(doc_id, pno, line)
            out["title_candidate"] = {"text": line, "source": ft["source"], "verdict": verdict}
            if verdict == "title":
                out["caption"] = line
                out["caption_from"] = f"{ft['source']}_first_line"
    if stats is not None:
        stats["figure_image_saved"] += bool(out["image"])
        stats["figure_with_numbers"] += bool(out["numbers"])
        if out["title_candidate"]:
            stats[f"figure_first_line_{out['title_candidate']['verdict']}"] += 1
    return out
