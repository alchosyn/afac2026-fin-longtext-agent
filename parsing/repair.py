# -*- coding: utf-8 -*-
"""Block-level repair of a MinerU page: fixes applied to the block list in memory, before
chunking. Output is still a list of MinerU-shaped blocks; mineru.json is never rewritten.

Implemented:
  * promote_headings -- text blocks that quality gate 1 reported as missing headings
    (head_as_text findings of audit.json: a numbered heading that fills a gap in the
    sequence, a 第N章 / 第N节 / 附录 line labelled text, or the title of a PDF bookmark)
    become paragraph_title blocks, marked promoted so the chunker never demotes them.
    The gap search and the bookmark match need the whole document, so they stay in
    parsing.audit; the repair only consumes its (page, block) list and checks the snippet
    still matches the block.
  * split_heading_captions -- a table whose caption or footnote gate 1 found to be a
    heading-list title (head_in_caption) gets a paragraph_title block next to it: before the
    table for a caption (which stays with the table), after it for a footnote (which is
    removed from the table: it is the next section's heading, not a note).
  * split_merged_headings -- a text block that gate 1 found to hold a heading-list title next
    to other text (head_merged: ins_03 p5 第十条 closing the previous clause's block) is cut
    into the text before, a paragraph_title block, and the text after. MinerU keeps no line
    positions inside a block, so the cut is by character offset in the cleaned text and the
    pieces keep the block's bbox; only the first piece keeps continues_prev.
  * recover_text -- words of the text layer that lie inside no block (the audit's cov_word
    finding, recomputed here so the repair needs no audit.json) become text blocks: the
    words are grouped into lines, the lines into paragraphs, and each paragraph is inserted
    before the first block below it in the same column, marked recovered_from = "text_layer".
    Words in the page's top / bottom 8 % are left alone (page furniture, which MinerU drops
    anyway) and fragments shorter than 4 characters are dropped (signature pages: "年 月 日",
    a page number). On the 15 first-three-per-category documents MinerU left 0.19 % of the
    text-layer words outside every block (2026-09-25): mostly the signature and declaration
    pages of the contracts, twice a whole region of body text (ins_01 p2 TOC, p9 clauses),
    and a few title / unit / note lines hugging annual-report tables. This step runs before
    recover_tables, so a missed borderless table comes back as text blocks first and is
    then tried as a table like any other run of text blocks.
    Since 2026-09-27 the uncovered text comes from native.json on digital pages (the characters
    MinerU itself reads; PyMuPDF words when it was not exported) and from RapidOCR lines on scan
    pages (parsing.native.page_kind), where the text layer is never used: it is missing, or it
    is the Word draft under the signed scan (fc_text01 p264 / p281, fc_text02 p236 / p249 had
    every sentence twice, once from MinerU's reading of the scan and once from the draft, with
    the draft's wording "公司法定代表人签名" and an empty date).
  * attach_hidden_captions -- a table whose title, date and unit lines gate 1 found above it in a
    figure or dropped block (tab_caption_hidden) gets them as its caption (fr_cscec_2025 p146-162:
    the seal over each audited statement's title made MinerU box the title as an image, and
    "人民币千元" became a header, so the 17 statements had no caption and all sat under 审计报告（续）).
  * recover_tables -- runs of text blocks that are really a borderless table become one
    synthetic table block (table_body HTML from the text-layer projection), carrying
    recovered_from = the MinerU indices it replaced. Since 2026-10-01 the only recall signal is
    the run of punctuation-free text blocks, every text block inside the run's region is a
    member (no duplicate paragraph beside the table), and a grid with a cell that is only a
    bullet or checkbox is rejected (tables.validate_grid).
  * restore_fake_formulas -- equation blocks and inline formula spans that are plain text in
    LaTeX clothing ("14.0\\%", "7 . 2 \\%", "亿元", "\\text {公司最近三年及一期期间费用情况}") become
    text; real formulas (fractions, powers, subscripts) are left as MinerU wrote them.
  * restore_toc_tables -- a captionless table right after a 目录 title (ins_02 p2, fr_cscec_2025
    p138) becomes an index block filled line by line from the text layer.
  * restore_false_index -- an index block that is no table of contents (common.is_toc_index: not
    after a 目录 title and fewer than half of its entries end in a page number; 8 of 37 index
    blocks on 2026-10-01) becomes a table when its entries split at a wide gap into one grid
    (fr_cscec_2025 p207-208 tax rates), else one text block per entry (ins_01 p1 reading guide,
    p18-19 definitions, ins_02 p1 guide bullets; fr_byd_2024 p159 is then recovered as a table
    by recover_tables). Before, the chunker dropped them with the real TOCs.
  * replace_image_tables -- an image captioned 表N whose region MinerU, run again on the crop
    (parsing.rerun), returns as a table becomes that table (pack2_text03 p19 表3; 表2 / 表4 on
    p18 / p20 come back as images and stay figures).

Planned (the gate-1 findings they will consume): expand figure boxes to the raster image,
drop captions that are the running header / a checkbox line, attach a recovered line that
hugs a table as its caption / footnote instead of a paragraph.
"""
import re

from .common import DROP, FIGURE_TYPES, LIST_ITEM, TITLE_TYPES, TOC_TITLE, clean, index_entries, is_toc_index, spans, txt
from .native import OCR_MIN_SCORE, native_lines, ocr_uncovered, uncovered_runs
from .tables import find_table_candidates, group_lines, join_words, rows_to_html, strip_label_rows, validate_grid

MARGIN = 0.08          # page fraction: words above / below this line are page furniture
MIN_CHARS = 4          # a recovered paragraph shorter than this is a fragment, not text
PARA_GAP = 1.5         # a vertical gap larger than this many line heights starts a new paragraph
COL_GAP = 25.0         # pt: a horizontal gap this wide splits a visual line into two columns
TOL = 0.005            # page fraction: tolerance of the inside-a-block test (as in the audit)


def promotions_from_audit(report):
    """{page: {block index: snippet}} of the text blocks gate 1 found to be headings
    (head_as_text, high and medium confidence: the medium ones all start with a
    chapter / section keyword)."""
    out = {}
    for f in report.get("findings", []):
        if f["code"] == "head_as_text" and f.get("block") is not None:
            out.setdefault(f["page"], {})[f["block"]] = f["text"]
    return out


def promote_headings(blocks, wanted, stats):
    """Text blocks listed in wanted (index -> audit snippet) become paragraph_title blocks.
    A block whose text no longer matches the snippet is left alone and counted: the audit
    was run on a different mineru.json."""
    if not wanted:
        return blocks
    out = []
    for b in blocks:
        snippet = wanted.get(b.get("index"))
        if snippet is not None and b["type"] == "text":
            if clean(txt(b))[:80] == snippet:
                b = dict(b, type="paragraph_title", promoted=True)
                stats["heading_promoted"] += 1
            else:
                stats["heading_promote_mismatch"] += 1
        out.append(b)
    return out


def caption_headings_from_audit(report):
    """{page: {table block index: (part, snippet)}} of the tables whose caption or footnote
    gate 1 found to be a bookmark heading (head_in_caption)."""
    out = {}
    for f in report.get("findings", []):
        if f["code"] == "head_in_caption" and f.get("block") is not None:
            out.setdefault(f["page"], {})[f["block"]] = (f["part"], f["text"])
    return out


def split_heading_captions(blocks, wanted, stats):
    """Tables listed in wanted (index -> (part, audit snippet)) get a paragraph_title block
    carrying the caption / footnote text: before the table for a caption (annual report p155
    "8、公司现金流量表"), after it for a footnote, which is also dropped from the table
    (contract p125: the next section's "（四）2025年1-6月..." heading). A table whose sub-block
    no longer matches the snippet is left alone and counted."""
    if not wanted:
        return blocks
    out = []
    for b in blocks:
        hit = wanted.get(b.get("index"))
        if hit is None or b["type"] != "table":
            out.append(b)
            continue
        part, snippet = hit
        sub = next((c for c in b.get("content", []) if isinstance(c, dict) and c.get("type") == part
                    and clean(txt(c))[:80] == snippet), None)
        if sub is None:
            stats["heading_caption_mismatch"] += 1
            out.append(b)
            continue
        title = {"type": "paragraph_title", "index": b["index"], "bbox": sub.get("bbox", b["bbox"]),
                 "content": [{"type": "text", "content": clean(txt(sub))}], "promoted": True, "from": part}
        stats["heading_from_" + part] += 1
        if part == "table_caption":
            out.extend([title, b])
        else:
            out.append(dict(b, content=[c for c in b["content"] if c is not sub]))
            out.append(title)
    return out


def merged_headings_from_audit(report):
    """{page: {text block index: [(start, end, heading)]}} of the heading-list titles gate 1
    found inside a longer text block (head_merged); offsets are in clean(txt(block))."""
    out = {}
    for f in report.get("findings", []):
        if f["code"] == "head_merged" and f.get("block") is not None:
            out.setdefault(f["page"], {}).setdefault(f["block"], []).append((f["start"], f["end"], f["heading"]))
    return out


def split_merged_headings(blocks, wanted, stats):
    """Text blocks listed in wanted (index -> [(start, end, heading)]) are cut at the headings:
    text before -> text block, heading -> promoted paragraph_title, text after -> text block.
    A block whose text no longer carries the heading at the offsets is left alone and counted."""
    if not wanted:
        return blocks
    out = []
    for b in blocks:
        cuts = wanted.get(b.get("index"))
        if not cuts or b["type"] != "text":
            out.append(b)
            continue
        text = clean(txt(b))
        cuts = sorted(cuts)
        if any(text[s:e] != h for s, e, h in cuts):
            stats["heading_merged_mismatch"] += 1
            out.append(b)
            continue
        pos, first = 0, True
        for s, e, h in cuts:
            before = text[pos:s].strip()
            if before:
                piece = dict(b, content=[{"type": "text", "content": before}], split_from=b.get("index"))
                if not first:
                    piece.pop("continues_prev", None)
                out.append(piece)
            out.append({"type": "paragraph_title", "index": b["index"], "bbox": b["bbox"], "promoted": True, "from": "merged",
                        "content": [{"type": "text", "content": h}]})
            stats["heading_split_from_text"] += 1
            pos, first = e, False
        after = text[pos:].strip()
        if after:
            piece = dict(b, content=[{"type": "text", "content": after}], split_from=b.get("index"))
            piece.pop("continues_prev", None)
            out.append(piece)
    return out


def recover_tables(blocks, page, W, H, stats):
    """Replace runs of text blocks that are really a borderless table by a synthetic table
    block. Two signals propose candidates: a trigger sentence ("...如下:", "见下表")
    followed by text blocks, and two or more consecutive text blocks without sentence
    punctuation. The region is the run's vertical extent over the full text width; the
    grid must pass validate_grid (see tables.find_table_candidates)."""
    content = [b for b in blocks if b["type"] not in DROP]
    if len(content) < 2:
        return blocks
    text_of = {id(b): clean(txt(b)) for b in content}
    replace = {}                                        # id of first block of a run -> synthetic block
    drop = set()
    for cand in find_table_candidates(content, page, W, H, text_of):
        stats["table_candidates"] += 1
        if not cand["ok"]:
            stats["table_candidate_rejected"] += 1
            continue
        members = [content[k] for k in cand["run"]]
        if any(m.get("rotated") for m in members):          # the projection assumes horizontal lines (fr_cscec_2025 p127)
            stats["table_candidate_rotated_skipped"] += 1
            continue
        stats["table_recovered"] += 1
        bbox = cand["bbox"]
        first = members[0]
        replace[id(first)] = {"type": "table", "index": first["index"], "bbox": bbox,
                              "content": [{"type": "table_body", "index": first["index"], "bbox": bbox, "content": rows_to_html(cand["rows"])}],
                              "recovered_from": [m["index"] for m in members], "_label": cand["label"], "_why": cand["why"]}
        drop.update(id(m) for m in members)
    if not replace:
        return blocks
    out = []
    for b in blocks:
        if id(b) in replace:
            out.append(replace[id(b)])
        elif id(b) not in drop:
            out.append(b)
    return out


def uncovered_words(blocks, page, W, H):
    """Words of the page's text layer whose centre lies inside no block or sub-block, outside
    the furniture margins. Each word is a pymupdf tuple (x0, y0, x1, y1, text, ...) in points."""
    boxes = [b["bbox"] for b in blocks]
    boxes += [c["bbox"] for b in blocks for c in b.get("content", []) if isinstance(c, dict) and "bbox" in c]
    out = []
    for w in page.get_text("words"):
        if not w[4].strip():
            continue
        cx, cy = (w[0] + w[2]) / 2 / W, (w[1] + w[3]) / 2 / H
        if cy < MARGIN or cy > 1 - MARGIN:
            continue
        if any(bb[0] - TOL <= cx <= bb[2] + TOL and bb[1] - TOL <= cy <= bb[3] + TOL for bb in boxes):
            continue
        out.append(w)
    return out


def split_columns(line):
    """A visual line -> its runs of words separated by a column-wide gap (two-column pages:
    group_lines joins the columns' words that share a baseline)."""
    runs, cur = [], [line[0]]
    for w in line[1:]:
        if w[0] - cur[-1][2] > COL_GAP:
            runs.append(cur)
            cur = [w]
        else:
            cur.append(w)
    runs.append(cur)
    return runs


def paragraphs_of(words):
    """Uncovered words -> paragraphs [{"text", "x0", "y0", "x1", "y1"}] in points. A line joins
    the last paragraph of its column when the gap is small and it does not start a list item."""
    paras = []
    for line in group_lines(words):
        for run in split_columns(line):
            x0, y0 = min(w[0] for w in run), min(w[1] for w in run)
            x1, y1 = max(w[2] for w in run), max(w[3] for w in run)
            text = join_words([w[4] for w in run])
            h = max(1.0, y1 - y0)
            target = None
            if not LIST_ITEM.match(text):
                for p in reversed(paras[-4:]):                    # the last paragraph of this column
                    if min(x1, p["x1"]) - max(x0, p["x0"]) > 0:
                        if 0 <= y0 - p["y1"] <= PARA_GAP * max(h, p["h"]):
                            target = p
                        break
            if target is None:
                paras.append({"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "h": h})
                continue
            joiner = "" if re.search(r"[一-鿿，、；：（]$", target["text"]) or re.match(r"^[一-鿿）]", text) else " "
            target["text"] += joiner + text
            target["x0"], target["y0"] = min(target["x0"], x0), min(target["y0"], y0)
            target["x1"], target["y1"] = max(target["x1"], x1), max(target["y1"], y1)
            target["h"] = max(target["h"], h)
    return paras


def insert_by_position(blocks, new):
    """Insert a synthetic block into the reading order: after the last content block that
    shares its row (vertical overlap over half the smaller height) and lies to its left --
    the other cells of a borderless table row MinerU half-covered (annual report p204
    "注册及实收股本 | 9,117,198 | 2,909,266") stay one run for recover_tables --, else before the
    first content block below it in the same column (x ranges overlap), else at the end.
    It takes the neighbour's index: the list order is what the chunker follows, the index
    only names the block in findings."""
    nb = new["bbox"]
    nh = max(1e-6, nb[3] - nb[1])
    if new.get("rotated"):
        # turned text (a landscape table printed on a portrait page) reads column by column, left to
        # right: before the first block to its right that it overlaps vertically (fr_cscec_2025 p123:
        # the unit line "单位：人民币" left of the bond table, which must stay the page's last block
        # for the cross-page merge with p124)
        for i, b in enumerate(blocks):
            bb = b["bbox"]
            if b["type"] not in DROP and bb[0] >= nb[2] - TOL and min(bb[3], nb[3]) - max(bb[1], nb[1]) > 0:
                new["index"] = b.get("index", i)
                blocks.insert(i, new)
                return
        new["index"] = blocks[-1].get("index", len(blocks)) if blocks else 0
        blocks.append(new)
        return
    row_mate = None
    for i, b in enumerate(blocks):
        bb = b["bbox"]
        if b["type"] in DROP:
            continue
        overlap = min(bb[3], nb[3]) - max(bb[1], nb[1])
        # a row mate is a block of comparable height: a vertical strip down the margin (ins_01 p2
        # aside_text) overlaps every row and must not swallow the page's blocks in reverse order
        if overlap > 0.5 * nh and bb[3] - bb[1] <= 3 * nh and bb[2] <= nb[0] + TOL:
            row_mate = i
    if row_mate is not None:
        new["index"] = blocks[row_mate].get("index", row_mate)
        blocks.insert(row_mate + 1, new)
        return
    for i, b in enumerate(blocks):
        bb = b["bbox"]
        if b["type"] not in DROP and bb[1] >= nb[3] - TOL and min(bb[2], nb[2]) - max(bb[0], nb[0]) > 0:
            new["index"] = b.get("index", i)
            blocks.insert(i, new)
            return
    new["index"] = blocks[-1].get("index", len(blocks)) if blocks else 0
    blocks.append(new)


def recover_text(blocks, page, W, H, stats, words=None, source="text_layer"):
    """Text no block covers -> text blocks in reading order (see the module docstring). words
    are word-like tuples (x0, y0, x1, y1, text) in points: the uncovered runs of native.json
    (digital pages), the uncovered OCR lines (scan pages), or by default PyMuPDF's uncovered
    words. Returns a new block list; the input list is not modified."""
    if words is None:
        words = uncovered_words(blocks, page, W, H)
    if not words:
        return blocks
    out = list(blocks)
    flat = [w for w in words if len(w) < 6]
    turned = [w for w in words if len(w) >= 6]              # rotated runs, in native reading order
    paras = sorted(paragraphs_of(flat), key=lambda p: (p["y0"], p["x0"])) if flat else []
    for w in turned:                                        # consecutive rotated runs of one column form a paragraph
        last = paras[-1] if paras and paras[-1].get("rotated") else None
        if last and min(w[2], last["x1"]) - max(w[0], last["x0"]) > -3:
            last["text"] += w[4]
            last["x0"], last["y0"] = min(last["x0"], w[0]), min(last["y0"], w[1])
            last["x1"], last["y1"] = max(last["x1"], w[2]), max(last["y1"], w[3])
        else:
            paras.append({"text": w[4], "x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3], "h": w[3] - w[1], "rotated": True})
    for p in paras:
        text = clean(p["text"])
        if len(text.replace(" ", "")) < MIN_CHARS:
            stats["text_fragment_dropped"] += 1
            continue
        block = {"type": "text", "index": None,
                 "bbox": [round(p["x0"] / W, 3), round(p["y0"] / H, 3), round(p["x1"] / W, 3), round(p["y1"] / H, 3)],
                 "content": [{"type": "text", "content": text}], "recovered_from": source}
        if p.get("rotated"):
            block["rotated"] = True                         # turned text: no borderless-table projection
        insert_by_position(out, block)
        stats["text_recovered" if source != "ocr" else "text_recovered_ocr"] += 1
        stats["text_recovered_chars" if source != "ocr" else "text_recovered_ocr_chars"] += len(text)
    return out


def hidden_captions_from_audit(report):
    """{page: {table block index: caption}} of the tables whose title lines gate 1 found in a
    figure or dropped block above them (tab_caption_hidden)."""
    out = {}
    for f in report.get("findings", []):
        if f["code"] == "tab_caption_hidden" and f.get("block") is not None:
            out.setdefault(f["page"], {})[f["block"]] = f["caption"]
    return out


def attach_hidden_captions(blocks, wanted, stats):
    """Tables listed in wanted (index -> caption) get the caption as a table_caption sub-block:
    fr_cscec_2025 p146 "中国建筑股份有限公司 合并资产负债表 2025年12月31日 人民币千元", whose title
    lines MinerU put in the seal's image block and a header block. A table that already has a
    caption gets the parts its caption lacks appended."""
    if not wanted:
        return blocks
    out = []
    for b in blocks:
        cap = wanted.get(b.get("index"))
        if cap is None or b["type"] != "table":
            out.append(b)
            continue
        subs = [c for c in b.get("content", []) if isinstance(c, dict) and c.get("type") == "table_caption"]
        have = clean("".join(txt(c) for c in subs))
        extra = " ".join(part for part in cap.split(" ") if part and part not in have)
        if not extra:
            out.append(b)
            continue
        new_cap = {"type": "table_caption", "bbox": b["bbox"], "content": [{"type": "text", "content": (have + " " + extra).strip()}],
                   "recovered_from": "hidden_caption"}
        content = [c for c in b.get("content", []) if not (isinstance(c, dict) and c.get("type") == "table_caption")]
        out.append(dict(b, content=[new_cap] + content))
        stats["table_caption_from_hidden_lines"] += 1
    return out


# ----------------------------------------------------------------------------- fake formulas
FORMULA_WRAP = re.compile(r"\\(?:text|mathrm|mathbf|mathit|mathbb|textbf|operatorname)\s*\{([^{}]*)\}")
FORMULA_SYMBOL = re.compile(r"\\(leqslant|geqslant|leq|geq|le|ge|times|%|,|;|!|quad|qquad)(?![A-Za-z])")
SYMBOL_TEXT = {"leqslant": "≤", "geqslant": "≥", "leq": "≤", "geq": "≥", "le": "≤", "ge": "≥", "times": "×", "%": "%",
               ",": "", ";": "", "!": "", "quad": " ", "qquad": " "}
SPACED_DIGITS = re.compile(r"(?<=[\d.])\s+(?=[\d.%])")
PLAIN_FORMULA = re.compile(r"^[\d.,%≤≥×+\-–—/\s一-鿿，、。：；（）()]+$")
LATEX_LEFT = re.compile(r"[\\{}^_=<>$&]")


def plain_formula(latex):
    """The ordinary text a fake formula stands for, or None for a real one. MinerU labels a
    few plain strings as formulas (2026-09-30 count on 20 docs: 2 of 4 equation blocks, 18 of
    37 inline spans): "14.0\\%" and "7 . 2 \\%" (percentages), "亿元" and
    "\\text {公司最近三年及一期期间费用情况}" (words), "\\leqslant" (one symbol). Those come back as
    text: \\text{} wrappers removed, a few symbols translated, digit spacing collapsed. Anything
    that still carries LaTeX structure (\\frac, ^, _, =, braces: ins_03's blood counts
    "1 0 0 \\times 1 0 ^ { 9 }", pack2_text03's consumption functions, ins_01 p2's hallucinated
    "\\therefore m = \\frac{3}{11}") stays a formula."""
    s = latex
    for _ in range(3):
        s = FORMULA_WRAP.sub(lambda m: m.group(1), s)
    s = FORMULA_SYMBOL.sub(lambda m: SYMBOL_TEXT[m.group(1)], s).replace("~", " ")
    if LATEX_LEFT.search(s):
        return None
    s = re.sub(r"\s+", " ", SPACED_DIGITS.sub("", s)).strip()
    if not s or not PLAIN_FORMULA.match(s) or not re.search(r"[\d一-鿿≤≥×]", s):
        return None
    return s


def restore_fake_formulas(blocks, stats):
    """Equation blocks and inline formula spans whose LaTeX is plain text (plain_formula)
    become text; the block keeps its index and box."""
    out = []
    for b in blocks:
        if b["type"] == "equation":
            plain = plain_formula(txt(b))
            if plain is not None:
                b = dict(b, type="text", content=[{"type": "text", "content": plain}], formula_restored=True)
                stats["formula_block_restored"] += 1
        elif isinstance(b.get("content"), list) and any(isinstance(c, dict) and c.get("type") == "equation_inline" for c in b["content"]):
            content = []
            for c in b["content"]:
                if isinstance(c, dict) and c.get("type") == "equation_inline":
                    plain = plain_formula(txt(c))
                    if plain is not None:
                        c = dict(c, type="text", content=plain)
                        stats["formula_inline_restored"] += 1
                    else:
                        stats["formula_inline_kept"] += 1
                content.append(c)
            b = dict(b, content=content)
        out.append(b)
    return out


# ----------------------------------------------------------------------------- the printed TOC boxed as a table
TOC_GAP = 40.0         # pt: the gap between two TOC columns (ins_02 p2: 48 pt; the tab after a clause number is 27 pt)


def _split_runs(chars):
    """[(x0, y0, x1, y1, text)] of a line's characters [(box, ch)], cut at a gap wider than
    TOC_GAP: a two-column TOC prints two entries on one baseline (ins_02 p2 "第七条 责任免除" and
    "第八条 其他免责条款"), which native.json and PyMuPDF both keep as one line."""
    runs, cur = [], None
    for box, ch in chars:
        if cur is not None and box[0] - cur[2] > TOC_GAP:
            runs.append(cur)
            cur = None
        if cur is None:
            cur = [box[0], box[1], box[2], box[3], ch]
        else:
            cur[0], cur[1], cur[2], cur[3] = min(cur[0], box[0]), min(cur[1], box[1]), max(cur[2], box[2]), max(cur[3], box[3])
            cur[4] += ch
    if cur is not None:
        runs.append(cur)
    return [(r[0], r[1], r[2], r[3], r[4].strip()) for r in runs if r[4].strip()]


def page_lines(page, native=None, ocr_lines=None):
    """[(x0, y0, x1, y1, text)] of a page's text lines in points, in reading order (top to
    bottom, left to right): the OCR lines of a scan page, else native.json's lines, else
    PyMuPDF's; a line is cut at a column-wide gap (_split_runs)."""
    if ocr_lines:
        lines = [(l[0], l[1], l[2], l[3], l[4]) for l in ocr_lines if l[5] >= OCR_MIN_SCORE]
    elif native is not None:
        lines = []
        for ln in native["lines"]:
            chars = [(cb, ch) for sp in ln["spans"] for ch, cb in zip(sp["text"], sp.get("chars") or []) if ch.strip()]
            lines += _split_runs(chars) if chars else [l for l in native_lines({"lines": [ln]})]
    else:
        lines = []
        for blk in page.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                chars = [(tuple(sp["bbox"]), sp["text"]) for sp in ln["spans"] if sp["text"].strip()]
                lines += _split_runs(chars)
    return sorted(lines, key=lambda l: (round(l[1]), l[0]))


def restore_toc_tables(blocks, lines_fn, W, H, stats):
    """A captionless table right after a 目录 title is the printed table of contents boxed as a
    grid (ins_02 p2 "条款目录", fr_cscec_2025 p138 the audit report's 目录 with page ranges: the
    only two among 2,051 table blocks, 2026-09-30). It becomes an index block with one entry
    per text line inside the box, in reading order (lines_fn() -> page_lines), so the TOC
    readers (parsing.outline, the audit's drop_index check) see it like any printed TOC and the
    chunker drops it like one. MinerU's grid is not used: ins_02's rows glue two columns of
    clause titles into one cell."""
    out, prev, lines = [], None, None
    for b in blocks:
        if (b["type"] == "table" and prev is not None and prev["type"] in TITLE_TYPES
                and TOC_TITLE.search(clean(txt(prev))) and not spans(b, "_caption")):
            if lines is None:
                lines = lines_fn()
            bx0, by0, bx1, by1 = b["bbox"][0] * W, b["bbox"][1] * H, b["bbox"][2] * W, b["bbox"][3] * H
            inside = [clean(t) for x0, y0, x1, y1, t in lines
                      if bx0 - 2 <= (x0 + x1) / 2 <= bx1 + 2 and by0 - 2 <= (y0 + y1) / 2 <= by1 + 2 and clean(t)]
            if inside:
                b = {"type": "index", "index": b["index"], "bbox": b["bbox"], "recovered_from": "toc_table",
                     "content": [{"type": "text", "content": [{"type": "text", "content": t}]} for t in inside]}
                stats["toc_table_restored"] += 1
                stats["toc_table_lines"] += len(inside)
        if b["type"] not in DROP:
            prev = b
        out.append(b)
    return out


# ----------------------------------------------------------------------------- an index block that is no TOC
COLUMN_GAP = re.compile(r"\s{2,}|\t")      # MinerU writes a run of spaces between the columns of an index entry


def restore_false_index(blocks, stats):
    """An index block that is not a table of contents (common.is_toc_index: not after a 目录
    title, fewer than half of its entries end in a page number) is a two-column alignment the
    layout model took for a TOC: 8 of the 37 index blocks on 2026-10-01 (fr_cscec_2025 p207-208
    the 公司名称 / 实际税率 tax-rate table, fr_byd_2024 p159 the materiality criteria, ins_01 p1 the
    reading guide, p18-19 the disease definitions, ins_02 p1 the guide bullets). The chunker
    drops index blocks, so each one was a page of content lost (ins_01 p18 had no record at all).
    No text layer is needed: MinerU has already joined the wrapped lines of every entry and
    writes a run of spaces between columns. When every entry splits at such a gap into the same
    two or more columns and the grid passes tables.validate_grid, the block becomes a table
    (p207-208: 21 x 2 and 22 x 2, numeric column); otherwise one text block per entry, with the
    block's bbox (MinerU keeps no entry positions), marked recovered_from = "index". Those text
    blocks then go through recover_text / recover_tables like any other: p159 comes back as a
    10 x 2 table from the text-layer projection (its columns are a single space apart in the
    entries), the guides and definitions stay paragraphs and the chunker's list rules apply.
    A true TOC is left alone and dropped as before."""
    out, prev = [], None
    for b in blocks:
        if b["type"] == "index" and not is_toc_index(b, prev):
            entries = index_entries(b)
            raw = [txt(c) for c in b.get("content", []) if isinstance(c, dict)]        # before clean(): it removes the
            raw = [r for r in raw if r.strip()] or [txt(b)]                             # gap between two CJK cells
            rows = [[c for c in (clean(x) for x in COLUMN_GAP.split(r.strip())) if c] for r in raw]
            widths = {len(r) for r in rows}
            grid, ok, why = [], False, "columns"
            if len(rows) >= 2 and len(widths) == 1 and min(widths) >= 2:
                grid, _ = strip_label_rows(rows)
                ok, why = validate_grid(grid, allow_text_pairs=prev is not None and prev["type"] in TITLE_TYPES)
            if ok:
                out.append({"type": "table", "index": b["index"], "bbox": b["bbox"],
                            "content": [{"type": "table_body", "index": b["index"], "bbox": b["bbox"], "content": rows_to_html(grid)}],
                            "recovered_from": "index", "_why": why})
                stats["false_index_table"] += 1
            else:
                for e in entries:
                    out.append({"type": "text", "index": b["index"], "bbox": b["bbox"],
                                "content": [{"type": "text", "content": e}], "recovered_from": "index"})
                stats["false_index_text"] += 1
                stats["false_index_text_blocks"] += len(entries)
            prev = out[-1]
            continue
        if b["type"] not in DROP:
            prev = b
        out.append(b)
    return out


# ----------------------------------------------------------------------------- a table boxed as an image
def replace_image_tables(blocks, wanted, stats):
    """Image blocks listed in wanted (index -> table_body HTML, from parsing.rerun: MinerU run
    again on the cropped region) become table blocks: the caption and footnote sub-blocks keep
    their text as table_caption / table_footnote, the body takes the HTML, and rerun_from =
    "image" tells the chunker to reconcile it with the text layer like any table but never to
    demote it to paragraphs (pack2_text03 p19 表3: 4 x 3 cells of multi-line text)."""
    if not wanted:
        return blocks
    out = []
    for b in blocks:
        html = wanted.get(b.get("index")) if b["type"] in FIGURE_TYPES else None
        if not html:
            out.append(b)
            continue
        body = spans(b, "_body")
        content = [dict(c, type="table_caption") for c in spans(b, "_caption")]
        content.append({"type": "table_body", "index": b["index"], "bbox": body[0]["bbox"] if body else b["bbox"], "content": html})
        content += [dict(c, type="table_footnote") for c in spans(b, "_footnote")]
        out.append({"type": "table", "index": b["index"], "bbox": b["bbox"], "content": content, "rerun_from": "image"})
        stats["image_table_rerun"] += 1
    return out


def repair_page(blocks, page, W, H, stats, promote=None, caption_heads=None, merged=None, ocr=False, native=None,
                ocr_lines=None, hidden_caps=None, image_tables=None):
    """All block-level repairs of one page, in order: fake formulas back to text, images that
    are tables, heading promotion, heading captions, merged headings and hidden table captions
    (need no text layer), the TOC boxed as a table, then text recovery and table recovery. ocr
    marks a page whose text layer is not to be used (no layer, or a scan whose layer is the
    draft under the image): table recovery is skipped and text is recovered from ocr_lines (the
    page's RapidOCR lines) when given. On a digital page text is recovered from native (its
    native.json page), else from PyMuPDF words. image_tables: {image block index: table HTML}
    of this page (parsing.rerun.image_table_reruns)."""
    blocks = restore_fake_formulas(blocks, stats)
    blocks = replace_image_tables(blocks, image_tables, stats)
    blocks = promote_headings(blocks, promote, stats)
    blocks = split_heading_captions(blocks, caption_heads, stats)
    blocks = split_merged_headings(blocks, merged, stats)
    blocks = attach_hidden_captions(blocks, hidden_caps, stats)
    blocks = restore_toc_tables(blocks, lambda: page_lines(page, native, ocr_lines if ocr else None), W, H, stats)
    blocks = restore_false_index(blocks, stats)
    if ocr:
        if ocr_lines:
            blocks = recover_text(blocks, page, W, H, stats, words=ocr_uncovered(ocr_lines, blocks, W, H, margin=MARGIN), source="ocr")
        return blocks
    words = uncovered_runs(native, blocks, margin=MARGIN)[0] if native is not None else None
    blocks = recover_text(blocks, page, W, H, stats, words=words)
    blocks = recover_tables(blocks, page, W, H, stats)
    return blocks
