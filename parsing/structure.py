# -*- coding: utf-8 -*-
"""The structure layer of a repaired MinerU document: heading tree, paragraphs, tables and figures
-> the records of processed_data/<doc_id>/elements.jsonl.

Record shape (same for every route, so docview / retrieval work unchanged):
    heading    id h0 / h1 / h2 ...          text, level
    paragraph  id p001 ...                  text
    table      id t01 ...                   text (markdown), caption, source, rows, pages, table_status
    figure     id f01 ...                   text (caption or 图表/图片), caption, labels, source
every record: parent (enclosing heading), page, bbox (0-1 page fractions), ocr (page had no text layer)

Rules:
  * header / footer / page_number / index blocks are dropped
  * text blocks quality gate 1 reported as missing headings (head_as_text in audit.json) are
    promoted to titles before the tree is built (parsing.repair.promote_headings), a table
    caption / footnote it found to be a heading-list title (head_in_caption) becomes a title block
    next to the table, and a text block it found to hold a heading-list title next to other text
    (head_merged) is cut around it (split_merged_headings); a title gate 1 promoted is never demoted by the rules below (annual report
    p102 "2、公司资产或项目存在盈利预测，且..." is a bookmarked heading with two commas)
  * titles get a level from their numbering style (第N节 > 一、 > （一） > 1、 > （1）; 1. > 1.1. > 1））
    with a style stack, doc_title is level 1; the level of each style comes from the document's own
    table (common.infer_style_levels: 一、 > 1. in the annual reports, （一） > 1. in pack2_text02, the
    fixed table where the document gives no evidence); a "title" that reads like a sentence becomes a paragraph
    (unnumbered: longer than 40 chars or with sentence wording; numbered: sentence wording only, since a
    numbered heading can be long);
    a numbered title that continues a list written as paragraphs in the current section (the section
    holds a line numbered n-1 in the same style and none numbered n or above, and n does not extend
    the numbering of the would-be parent's headings) is a list item labelled title and becomes a paragraph
  * text: continues_prev blocks joined to the previous paragraph; list items （一） 1. ① appended
    to the paragraph that introduces them; text-layer words the layout model left outside every
    block arrive as text blocks from parsing.repair.recover_text (recovered_from = "text_layer")
  * tables: MinerU rows x text-layer projection columns, cell text from the text layer
    (parsing.tables); a grid used to lay out prose (<= 2 columns of long text) becomes paragraphs;
    an uncaptioned table takes the preceding short sentence ending with a colon as caption;
    in a conflict table (the projection could not rebuild it, MinerU's cells stay) a cell whose
    wrapped lines MinerU read side by side is re-read from the text layer when exactly one stack
    of words has its characters (tables.unscramble_cells, 2026-10-01); a scan-page table keeps
    MinerU's cells and records how many of its numbers an independent RapidOCR reading confirms
    (record field ocr_check, native.check_scan_numbers); an image MinerU re-read as a table on
    the cropped region (parsing.rerun, rerun_from = "image") is reconciled like any table but
    never demoted to paragraphs
  * block repairs before the tree (parsing.repair): fake formulas back to text, the printed TOC
    boxed as a table back to an index block, missed borderless tables recovered from runs of
    punctuation-free text blocks (every text block in the region a member, bullet / checkbox
    grids rejected)
  * cross-page tables: the first content block of a page that is an uncaptioned table is merged
    into the last table of the previous page when the header row repeats, or the column layout
    matches, or MinerU flagged continues_prev; the merged table keeps chaining page after page
    (annual report p27-33 R&D projects, p75-95 pollutant tables; before 2026-09-24 the check
    compared the first page of the merged table, so no table ever spanned more than two pages)
  * figures: caption + the naming words inside the figure (legend entries, axis categories,
    year labels; no tick values) as labels + the source footnote; the words come from the text
    layer, or from RapidOCR on a 200 dpi rendering when the figure is a bitmap (2026-09-26);
    nothing is generated from the image (competition rule: no preprocessing-stage summaries), the figure
    is cropped from the PDF at answer time; a figure without a caption (seals, signatures,
    photos) stays in elements.jsonl but is not a retrieval unit (agent.docview.Doc.indexable)
  * pages without a text layer and scan pages (parsing.native.page_kind: an image over 70 % of the
    page with no text layer or with the text under it) skip the text-layer repairs, take their
    table cells from MinerU and their figure labels from OCR, recover uncovered text from their
    RapidOCR lines, and mark their records ocr=True
"""
import re
from collections import Counter

import pymupdf

from .common import (CONTINUED, DROP, FIGURE_TYPES, LIST_ITEM, TEXT_TYPES, TITLE_TYPES, Leveler, clean, heading_number,
                     infer_style_levels, load_mineru, looks_like_sentence, sentence_signals, spans, style_of, txt)
from .figures import save_figure
from .native import check_scan_numbers, ocr_engine
from .repair import repair_page
from .tables import (header_rows_of_html, html_grid, is_layout_table, norm, project, reconcile, rows_to_paragraphs,
                     same_columns, to_markdown, unscramble_cells)

OCR_MIN_CHARS = 30
CAPTION_MAX = 60


YEAR_LABEL = re.compile(r"^(?:19|20)\d{2}(?:[-/.]\d{1,2}|[QH][1-4]|[A-Z])?$")     # 2019, 2015-01, 2025Q3, 2024E
OCR_DPI = 200            # chart labels are small: render the figure at this resolution before OCR
OCR_MIN_SCORE = 0.5      # RapidOCR confidence below which a line is noise
LEGEND_GLYPHS = re.compile(r"[■□●○◆◇▲△▪▫◼◻]+")


def label_words(tokens, max_words):
    """The naming words among tokens: two characters or more, containing a CJK or Latin
    letter and not mostly digits, or a year-like label; duplicates dropped, in order."""
    out, seen = [], set()
    for t in tokens:
        t = t.strip()
        if len(t) < 2 or t in seen:
            continue
        if not YEAR_LABEL.match(t):
            if not re.search(r"[一-鿿 A-Za-z]", t):
                continue
            digits = sum(ch.isdigit() or ch in "%.,-" for ch in t)
            if digits / len(t) > 0.5:
                continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_words:
            break
    return out


def figure_labels(page, bbox_pt, max_words=60, stats=None, text_layer=True):
    """Words inside a figure that name something: legend entries (series names), axis
    categories and titles, and year-like axis labels (2019, 2015-01, 2025Q3). Taken from the
    text layer; when the region has none (a bitmap: photo, scan, chart pasted as an image)
    the figure is rendered at OCR_DPI and read with RapidOCR (lines below OCR_MIN_SCORE
    dropped), then filtered the same way.
    Axis tick values (0%, 20%, 800,000) are dropped: on the research report's 35 charts they
    are the only numbers in 31 (only 图40 prints data labels), and a stored "50%" is a false
    BM25 hit. Charts whose axis labels are drawn as paths (图3, 图4, 图7: no year anywhere in
    the page's text layer) used to yield no year; OCR now reads them off the rendering. The
    values of a chart are never stored: at answer time the figure is cropped from the PDF
    and shown to the model (see the figure record's page and bbox)."""
    x0, y0, x1, y1 = bbox_pt
    rect = pymupdf.Rect(x0, y0, x1, y1)
    words = [w[4] for w in page.get_text("words", clip=rect)] if text_layer else []     # a scan's layer is the draft under it
    if words:
        return label_words(words, max_words)
    engine = ocr_engine()
    if engine is None or rect.is_empty:
        return []
    if stats is not None:
        stats["figure_ocr"] += 1
    import numpy as np
    pix = page.get_pixmap(dpi=OCR_DPI, clip=rect)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    res, _ = engine(img)
    # legend colour squares come back as ■ □ ● glyphs glued to the entries ("■邮购渠道■代理店渠道",
    # 图17): split on them and drop them
    tokens = [tok for box, text, score in (res or []) if float(score) >= OCR_MIN_SCORE
              for tok in LEGEND_GLYPHS.split(str(text)) for tok in tok.split()]
    labels = label_words(tokens, max_words)
    if stats is not None and labels:
        stats["figure_ocr_with_labels"] += 1
    return labels


class Builder:
    def __init__(self, doc_id, pdf_path, mineru_path, promote=None, caption_heads=None, merged=None, native=None, scan=None,
                 hidden_caps=None, image_tables=None):
        self.doc_id = doc_id
        self.pdf = pymupdf.open(str(pdf_path))
        self.data = load_mineru(mineru_path)
        self.promote = promote or {}                     # {page: {block index: snippet}} from parsing.repair.promotions_from_audit
        self.caption_heads = caption_heads or {}         # {page: {table index: (part, snippet)}} from parsing.repair.caption_headings_from_audit
        self.merged = merged or {}                       # {page: {text index: [(start, end, heading)]}} from parsing.repair.merged_headings_from_audit
        self.native = native or {}                       # {page: native.json page} (parsing.native.load_native)
        self.scan = scan or {}                           # {page: {"reason", "lines"}} of the scan pages (parsing.native.load_ocr)
        self.hidden_caps = hidden_caps or {}             # {page: {table index: caption}} from parsing.repair.hidden_captions_from_audit
        self.image_tables = image_tables or {}           # {page: {image index: table HTML}} from parsing.rerun.image_table_reruns
        self.page_scan = False
        self.last_num = {}                               # (parent heading id, numbering style) -> number of its last heading
        self.records = []
        self.counters = Counter()
        self.headings = []                               # [(level, id)]
        self.style_order = infer_style_levels(self.title_texts())    # the document's own level table
        self.leveler = Leveler(self.style_order["levels"])
        self.stats = Counter()
        self.table_status = Counter()

    def title_texts(self):
        """Every heading text of the document in order, before any repair: MinerU's title
        blocks, the text blocks gate 1 asked to promote, and the headings it found merged
        into a text block."""
        out = []
        for pg in self.data["pages"]:
            wanted = self.promote.get(pg["page_idx"] + 1) or {}
            merged = self.merged.get(pg["page_idx"] + 1) or {}
            for b in pg["blocks"]:
                if b["type"] in TITLE_TYPES or (b["type"] == "text" and b.get("index") in wanted):
                    t = clean(txt(b))
                    if t:
                        out.append(t)
                elif b["type"] == "text" and b.get("index") in merged:
                    out += [h for _, _, h in sorted(merged[b["index"]])]
        return out

    # ---- ids and parents
    def new_id(self, kind):
        self.counters[kind] += 1
        n = self.counters[kind]
        local = {"heading": f"h{n}", "paragraph": f"p{n:03d}", "table": f"t{n:02d}", "figure": f"f{n:02d}"}[kind]
        return f"{self.doc_id}#{local}"

    def parent_for_heading(self, level):
        while self.headings and self.headings[-1][0] >= level:
            self.headings.pop()
        return self.headings[-1][1] if self.headings else f"{self.doc_id}#h0"

    def current_heading(self):
        return self.headings[-1][1] if self.headings else f"{self.doc_id}#h0"

    def peek_parent(self, level):
        """The heading a new heading of this level would be attached to, without popping the stack."""
        for lvl, hid in reversed(self.headings):
            if lvl < level:
                return hid
        return f"{self.doc_id}#h0"

    def continues_list(self, text):
        """A numbered title that is really the next item of a list written as paragraphs in
        the current section. Two conditions: the would-be parent already has a heading of
        the same style numbered n or higher (so the title does not extend the heading
        sequence), and the current section holds a line numbered n-1 in that style and
        none numbered n or above (so it does extend a list).
        Demoted: research report p11 "2.鼓励金融混业..." (parent's headings reach 2., the
        section holds "1.银行主导..."); contract p125 "2、会计估计变更" and p126 "3、..."
        (parent's headings reach 3、, the section under （5） lists 1、 2、 as paragraphs).
        Kept: annual report p130 "5、其他持股..." after the paragraph "4、公司控股股东..."
        (a missed heading: the parent's headings stop at 3、, 5 extends them), and p125
        "2、限售股份变动情况" (its section already holds a "2、公司于..." line)."""
        if not LIST_ITEM.match(text) or CONTINUED.search(text):
            return False
        style, n = heading_number(text)
        if n is None or n < 2:
            return False
        last = self.last_num.get((self.peek_parent(self.leveler.lookup(text)), style))
        if last is None or n > last:
            return False
        seen = set()
        for r in reversed(self.records):                 # direct units of the current section
            if r["type"] == "heading":
                break
            if r["type"] != "paragraph":
                continue
            for line in r["text"].splitlines():
                s, k = heading_number(line)
                if s == style and k is not None:
                    seen.add(k)
        return (n - 1) in seen and not any(k >= n for k in seen)

    def last_unit(self):
        return self.records[-1] if self.records and self.records[-1]["type"] != "heading" else None

    # ---- record makers
    def add_heading(self, text, level, page, bbox):
        parent = self.parent_for_heading(level)
        style, n = heading_number(text)
        if n is not None and not CONTINUED.search(text):
            self.last_num[(parent, style)] = n
        rec = {"id": self.new_id("heading"), "type": "heading", "parent": parent, "page": page, "text": text,
               "level": level, "bbox": bbox}
        self.records.append(rec)
        self.headings.append((level, rec["id"]))
        return rec

    def add_paragraph(self, text, page, bbox, kind="paragraph_body"):
        rec = {"id": self.new_id("paragraph"), "type": "paragraph", "parent": self.current_heading(), "page": page,
               "text": text, "bbox": bbox}
        self.records.append(rec)
        self.stats[kind] += 1
        return rec

    # ---- the document
    def build(self, title):
        self.records.append({"id": f"{self.doc_id}#h0", "type": "heading", "parent": None, "page": 1, "text": title, "level": 0})
        for pg in self.data["pages"]:
            pno = pg["page_idx"] + 1
            page = self.pdf[pg["page_idx"]]
            W, H = page.rect.width, page.rect.height
            self.page_scan = pno in self.scan
            ocr = self.page_scan or len(page.get_text().strip()) < OCR_MIN_CHARS
            first_content = True
            page_records = []
            blocks = repair_page(pg["blocks"], page, W, H, self.stats, promote=self.promote.get(pno),
                                 caption_heads=self.caption_heads.get(pno), merged=self.merged.get(pno), ocr=ocr,
                                 native=self.native.get(pno), ocr_lines=(self.scan.get(pno) or {}).get("lines"),
                                 hidden_caps=self.hidden_caps.get(pno), image_tables=self.image_tables.get(pno))
            for b in blocks:
                btype = b["type"]
                if btype in DROP:
                    continue
                before = len(self.records)
                if btype in TITLE_TYPES:
                    self.handle_title(b, pno)
                elif btype in TEXT_TYPES:
                    self.handle_text(b, pno)
                elif btype in FIGURE_TYPES:
                    self.handle_figure(b, pno, page, W, H)
                elif btype == "table":
                    self.handle_table(b, pno, page, W, H)
                else:
                    self.stats[f"unknown_type_{btype}"] += 1
                    continue
                new = self.records[before:]
                if new and first_content:
                    new[0]["_first_on_page"] = True
                    first_content = False
                for r in new:
                    if ocr:
                        r["ocr"] = True
                page_records.extend(new)
            if page_records:
                page_records[-1]["_last_on_page"] = True
        self.merge_cross_page_tables()
        for r in self.records:
            for k in ("_first_on_page", "_last_on_page", "_continues_prev", "_ranges", "_list"):
                r.pop(k, None)
        return self.records

    # ---- one block
    def handle_title(self, b, pno):
        text = clean(txt(b))
        if not text:
            return
        if b["type"] == "paragraph_title" and not b.get("promoted"):     # gate 1 decided a promoted title: never demoted
            if style_of(text)[0] == "none" and looks_like_sentence(text):
                self.stats["title_demoted_to_paragraph"] += 1
                return self.add_paragraph(text, pno, b["bbox"])
            if LIST_ITEM.match(text) and sentence_signals(text):
                self.stats["title_demoted_to_paragraph"] += 1
                return self.handle_text(b, pno)
            if self.continues_list(text):
                self.stats["title_demoted_list_item"] += 1
                return self.handle_text(b, pno)
        level = self.leveler.level(text, b["type"] == "doc_title")
        self.add_heading(text, level, pno, b["bbox"])

    def handle_text(self, b, pno):
        text = clean(txt(b))
        if not text:
            return
        last = self.last_unit()
        if b.get("continues_prev") and last and last["type"] == "paragraph":
            joiner = "" if re.search(r"[一-鿿，、]$", last["text"]) or re.match(r"^[一-鿿]", text) else " "
            last["text"] += joiner + text
            self.stats["paragraph_joined_across_pages"] += 1
            return
        if LIST_ITEM.match(text) and last and last["type"] == "paragraph" and (last["text"].rstrip().endswith(("：", ":")) or last.get("_list")):
            last["text"] += "\n" + text
            last["_list"] = True
            self.stats["list_item_merged"] += 1
            return
        self.add_paragraph(text, pno, b["bbox"])

    def handle_figure(self, b, pno, page, W, H):
        caption = clean("".join(txt(c) for c in spans(b, "_caption")))
        source = clean(" ".join(txt(c) for c in spans(b, "_footnote")))
        body = spans(b, "_body")
        bbox = body[0]["bbox"] if body else b["bbox"]
        labels = figure_labels(page, (bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H), stats=self.stats,
                               text_layer=not self.page_scan)
        rec = {"id": self.new_id("figure"), "type": "figure", "parent": self.current_heading(), "page": pno,
               "text": caption or ("图表" if b["type"] == "chart" else "图片"), "caption": caption, "bbox": bbox,
               "labels": labels, "source": source, "kind": b["type"]}
        if caption:                                      # a captioned figure is cut out of the PDF and kept as a file
            image = save_figure(page, bbox, self.doc_id, rec["id"].split("#", 1)[1], force=True)
            if image:
                rec["image"] = image
                self.stats["figure_image_saved"] += 1
        self.records.append(rec)
        self.stats["figure"] += 1

    def handle_table(self, b, pno, page, W, H):
        caption = clean("".join(txt(c) for c in spans(b, "_caption")))
        source = clean(" ".join(txt(c) for c in spans(b, "_footnote")))
        body = spans(b, "_body")
        bbox = body[0]["bbox"] if body else b["bbox"]
        html = txt(body[0]) if body else ""
        proj = {"lines": [], "seps": [], "ranges": [], "n_words": 0} if self.page_scan else             project(page, (bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H))     # a scan's layer is not its table
        recovered = bool(b.get("recovered_from"))
        rerun = bool(b.get("rerun_from"))                    # an image MinerU re-read as a table: reconciled, never demoted
        bbox_pt = (bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H)
        ocr_check = None
        if recovered:
            rows, status = html_grid(html)[0], "recovered"   # the grid came from the projection and was validated already
        else:
            rows, status, meta = reconcile(html, proj)
            if self.page_scan and status == "no_text_layer":
                status = "scan_page"
            if status == "conflict" and not self.page_scan:  # MinerU's cells stay: re-read the interleaved ones
                rows, checked, restored, ambiguous = unscramble_cells(rows, page, bbox_pt)
                self.stats["cells_unscramble_checked"] += checked
                self.stats["cells_unscrambled"] += restored
                self.stats["cells_unscramble_ambiguous"] += ambiguous
            if self.page_scan and (self.scan.get(pno) or {}).get("lines"):
                ocr_check = check_scan_numbers(rows, self.scan[pno]["lines"], bbox_pt, W, H)
                self.stats["scan_numbers_checked"] += ocr_check["numbers"]
                self.stats["scan_numbers_confirmed"] += ocr_check["confirmed"]
                self.stats["scan_numbers_unconfirmed"] += len(ocr_check["unconfirmed"])
        self.table_status[status] += 1
        if not rows:
            return
        if not recovered and not rerun and is_layout_table(rows):
            self.stats["layout_table_to_paragraphs"] += 1
            for para in rows_to_paragraphs(rows):
                self.add_paragraph(clean(para), pno, bbox, kind="paragraph_from_layout_table")
            return
        if not caption:
            last = self.last_unit()
            if last and last["type"] == "paragraph" and last["page"] == pno and len(last["text"]) <= CAPTION_MAX \
                    and last["text"].rstrip().endswith(("：", ":", "如下", "见下表")):
                caption = last["text"]
                self.stats["caption_from_preceding_sentence"] += 1
        if b.get("_label"):
            caption = (caption + " " + b["_label"]).strip() if caption else b["_label"]
        rec = {"id": self.new_id("table"), "type": "table", "parent": self.current_heading(), "page": pno,
               "text": to_markdown(rows), "caption": caption, "source": source, "bbox": bbox, "rows": rows,
               "pages": [pno], "table_status": status, "n_cols": max(len(r) for r in rows),
               "header_rows": header_rows_of_html(html) if html and not recovered else 1,   # from the HTML spans, for the chunker
               "row_pages": [[0, len(rows) - 1, pno]],                                     # [first row, last row, page], extended when pages merge
               "_continues_prev": bool(b.get("continues_prev")), "_ranges": proj["ranges"]}
        if recovered:
            rec["recovered_from"] = b["recovered_from"]
        if rerun:
            rec["rerun_from"] = b["rerun_from"]
        if ocr_check is not None:
            rec["ocr_check"] = ocr_check
        self.records.append(rec)

    # ---- cross-page tables
    def merge_cross_page_tables(self):
        out = []
        for rec in self.records:
            prev = out[-1] if out else None
            if (rec["type"] == "table" and rec.get("_first_on_page") and not rec.get("caption")
                    and prev is not None and prev["type"] == "table" and prev.get("_last_on_page")
                    and prev["pages"][-1] == rec["page"] - 1):        # last page of prev, so a table chains over any number of pages
                a, b = prev["rows"], rec["rows"]
                header_repeat = bool(a) and bool(b) and [norm(c) for c in a[0]] == [norm(c) for c in b[0]] and any(norm(c) for c in b[0])
                same_layout = len(a[0]) == len(b[0]) and same_columns(prev.get("_ranges") or [], rec.get("_ranges") or [])
                if header_repeat or same_layout or rec.get("_continues_prev"):
                    body = b[1:] if header_repeat else b
                    if body and not norm(body[0][0]) and a and len(body[0]) == len(a[-1]):       # a row split by the page break
                        a[-1] = [(x + y) if not norm(y) or not norm(x) else (x + y) for x, y in zip(a[-1], body[0])]
                        body = body[1:]
                        self.stats["split_row_joined"] += 1
                    if body:
                        prev.setdefault("row_pages", [[0, len(a) - 1, prev["page"]]]).append([len(a), len(a) + len(body) - 1, rec["page"]])
                    prev["rows"] = a + body
                    prev["pages"].append(rec["page"])
                    prev["text"] = to_markdown(prev["rows"])
                    prev["_last_on_page"] = rec.get("_last_on_page", False)
                    if rec.get("source") and not prev.get("source"):
                        prev["source"] = rec["source"]
                    self.stats["table_merged_across_pages"] += 1
                    continue
            out.append(rec)
        self.records = out
