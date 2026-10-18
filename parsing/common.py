# -*- coding: utf-8 -*-
"""Shared pieces of the MinerU pipeline: block helpers, text cleaning, heading numbering
styles, and the document cards (processed_data/docs.jsonl)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"

# MinerU block types: what the builder drops, and how the rest is dispatched
DROP = {"header", "footer", "page_number", "index"}
TITLE_TYPES = {"doc_title", "paragraph_title"}
TEXT_TYPES = {"text", "aside_text", "page_footnote", "equation", "code", "list", "ref_text"}
FIGURE_TYPES = {"image", "chart"}

# the printed table of contents (parsing.audit drop_index, parsing.outline, parsing.repair)
TOC_LINE = re.compile(r"(\.{3,}|…+|·{3,})\s*\d{1,3}\s*$|\s\d{1,3}\s*$")      # "第一节 风险提示....15", "董事长致辞 2"
TOC_TITLE = re.compile(r"目\s*录\s*$")                                       # a title ending in 目录 (条款目录, 图表目录)

HIDDEN = re.compile(r"\[\s*Table_[A-Za-z_]*\]?|\[Table_[A-Za-z_]*")          # Word template tags left in the text
CJK_SPACE = re.compile(r"(?<=[一-鿿，。、；：（）])\s+(?=[一-鿿，。、；：（）])")
LIST_ITEM = re.compile(r"^\s*(（[一二三四五六七八九十]+）|\([一二三四五六七八九十]+\)|[①②③④⑤⑥⑦⑧⑨⑩]|\d{1,2}\s*[、.．）)]\s*(?!\d)|[•●■◆▪✿❖])")
SENT_END = tuple("。！？；!?;")
CLAUSE_WORDS = re.compile(r"如果|若您|若被|应当|可以|不得|不能|须|视为|自.{1,12}起|以下情形|下列")

# numbering styles and the heading level each one denotes. Two families: the Chinese
# outline 第N章 > 第N节 > 一、 > （一） > 1、 > （1） > ① and the dotted outline 1. > 1.1 > 1.1.1 > 1）,
# plus 第N条 (insurance clauses ins_03: 第N节 > 第N条, bookmarked as levels 1 > 2; without the
# style the 条 headings inherited the 节 level and sat beside their sections). 章 and 节 are
# separate styles since 2026-09-26: the contracts' annexed 债券持有人会议规则 nests 第N节 under
# 第N章, and one merged style made every 第二节 a "backwards" number after 第二章. The chain
# starts at 2 for 章 with 节 at 3, so 一、 (4) never shares a level with 第N节.
# Fixed levels (not order of appearance): a 第N节 heading that first appears after some 一、
# headings must still sit above them. Per document the levels are then adjusted by
# infer_style_levels below.
STYLES = [("zhang", re.compile(r"^第[一二三四五六七八九十百\d]+\s*[章部分篇]"), 2),
          ("jie", re.compile(r"^第[一二三四五六七八九十百\d]+\s*节"), 3),
          ("tiao", re.compile(r"^第[一二三四五六七八九十百\d]+\s*条"), 4),
          ("cn", re.compile(r"^[一二三四五六七八九十]+、"), 4),
          ("cnparen", re.compile(r"^（[一二三四五六七八九十]+）|^\([一二三四五六七八九十]+\)"), 5),
          ("dot", re.compile(r"^\d+(?:\.\d+)+\.?(?=\s|[^\d.])"), None),          # 1.1: one level under the document's "1."
          ("ar_dot", re.compile(r"^\d{1,2}\s*[.．]\s*(?!\d)"), 2),                  # 1.  top level of a research report
          ("ar_cn", re.compile(r"^\d{1,2}\s*、\s*(?!\d)"), 6),                     # 1、 contracts / annual report notes
          ("arparen", re.compile(r"^（\d{1,2}）|^\(\d{1,2}\)"), 7),
          ("arbr", re.compile(r"^\d{1,2}）"), 7),
          ("circ", re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩]"), 8)]


# ----------------------------------------------------------------------------- blocks
def txt(x):
    """All text of a MinerU block / sub-block / span list, concatenated."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return txt(x.get("content", ""))
    if isinstance(x, list):
        return "".join(txt(i) for i in x)
    return ""


def spans(block, suffix):
    """Sub-blocks of a table / figure block whose type ends with suffix (_caption, _body, _footnote)."""
    return [c for c in block.get("content", []) if isinstance(c, dict) and str(c.get("type", "")).endswith(suffix)]


def clean(s):
    s = HIDDEN.sub("", s)
    s = CJK_SPACE.sub("", s)
    return s.strip()


def index_entries(block):
    """The entries of an index block, cleaned, empty ones dropped (MinerU writes one sub-block
    per entry, wrapped lines already joined); the block's whole text when it has no sub-blocks."""
    lines = [clean(txt(c)) for c in block.get("content", []) if isinstance(c, dict)]
    return [l for l in lines if l] or [clean(txt(block))]


def is_toc_index(block, prev):
    """Is this index block a table of contents? Yes when the content block before it (prev: the
    last block that is not header / footer / page_number / index) is a title ending in 目录, or
    when at least half of its entries end in a page number (TOC_LINE: dot leaders and a number,
    or a bare number after a space). Measured 2026-10-01 on the 37 index blocks of 20 documents:
    every true TOC passes (18 first pages by the title, among them pack2_text07's chapter TOCs
    without page numbers and fr_cscec_2025 p3 without leaders; 11 continuation pages with 100 %
    numbered lines), while the 8 false ones (fr_cscec_2025 p207-208 tax-rate table whose "15%"
    ends in no digit, fr_byd_2024 p159 materiality table, ins_01 p1 reading guide, p18-19 disease
    definitions numbered at the line start, ins_02 p1 guide bullets ending in 第N条) have at most
    3 of 31 numbered lines. A TOC that is neither after a 目录 title nor numbered fails and is
    restored to text: a few noise paragraphs, no loss."""
    if prev is not None and prev.get("type") in TITLE_TYPES and TOC_TITLE.search(clean(txt(prev))):
        return True
    lines = index_entries(block)
    return sum(1 for l in lines if TOC_LINE.search(l)) >= len(lines) / 2


def load_mineru(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for pg in data.get("pages", []):
        pg.setdefault("blocks", [])            # MinerU writes a blank page without the key (the standards' empty versos)
    return data


# ----------------------------------------------------------------------------- headings
def style_of(text):
    """(style name, level) of a heading's numbering, or ("none", None)."""
    for name, rx, level in STYLES:
        m = rx.match(text)
        if m:
            if name == "dot":
                return name, 2 + m.group(0).count(".")
            return name, level
    return "none", None


def sentence_signals(text):
    """Wording no heading has: ends a sentence, two or more commas, or reads like a clause
    (conditional / deontic wording)."""
    return text.endswith(SENT_END) or text.count("，") >= 2 or bool(CLAUSE_WORDS.search(text))


def looks_like_sentence(text):
    """A 'title' that is really a sentence or a list item: long, or carrying sentence_signals.
    Length alone only counts for unnumbered titles: a numbered heading can be long (annual
    report p130 "4、公司控股股东或第一大股东...达到 80%", 47 chars; contract p46 "（三）发行人持股
    比例未达 50%...", 52 chars), so the builder tests numbered ones with sentence_signals."""
    return len(text) > 40 or sentence_signals(text)


CONTINUED = re.compile(r"[（(]\s*续\s*[)）]\s*$")            # "五、重要会计政策（续）": the running repeat of a heading
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"
CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
NUM_TOKEN = re.compile(r"^\s*第?\s*[（(]?\s*([一二三四五六七八九十百]+|\d{1,3})")
STYLE_RX = {name: rx for name, rx, _ in STYLES}


def cn2int(s):
    if s.isdigit():
        return int(s)
    total, num = 0, 0
    for ch in s:
        if ch in CN_DIGITS:
            num = CN_DIGITS[ch]
        elif ch == "十":
            total += (num or 1) * 10
            num = 0
        elif ch == "百":
            total += (num or 1) * 100
            num = 0
    return total + num


def heading_number(text):
    """(style, n) of the numbering a heading or list line starts with; for a dotted number
    (2.3.) n is its last component. ("none", None) when there is no numbering."""
    style, _ = style_of(text)
    if style == "none":
        return style, None
    if style == "circ":
        return style, CIRCLED.index(text[0]) + 1
    if style == "dot":
        head = STYLE_RX["dot"].match(text).group(0).rstrip(".")
        return style, int(head.split(".")[-1])
    m = NUM_TOKEN.match(text)
    return style, (cn2int(m.group(1)) if m else None)


# ----------------------------------------------------------------------------- document level table
# The fixed levels in STYLES hold for the standard chain (第N节 > 一、 > （一） > 1、 > （1） > ①), but
# the position of the arabic forms varies by document: "1." is the top level of a research
# report and sits under 一、 / （一） in the annual reports and pack2_text02; 1） sits under （1）
# in the contracts and the CATL report while the table puts both at level 6. Measured on the 15
# first-three-per-category documents (2026-09-26): every pair of the standard chain restarts in
# one direction only (reverse 0-2), the arabic forms are the only deviations. So each document
# learns its own table from its heading sequence: style B sits below style A when B's numbering
# restarts at 1 right after an A heading and (almost) never the other way round.
MIN_SUPPORT = 2          # A -> B restarts that decide the pair when no B -> A restart contradicts
MIN_SUPPORT_MIXED = 3    # ... when some do; A -> B must then also be RATIO times B -> A
RATIO = 3
FIXED_LEVELS = {name: lvl for name, _, lvl in STYLES if lvl is not None}


def heading_sequence(texts):
    """[(style, n)] of the numbered headings of a document in order: (续) repeats, numbered
    titles that read like sentences (the chunker demotes those) and dotted numbers (1.1: the
    depth is in the number itself, see Leveler) are left out."""
    out = []
    for t in texts:
        if CONTINUED.search(t):
            continue
        style, n = heading_number(t)
        if style in ("none", "dot") or n is None:
            continue
        if LIST_ITEM.match(t) and sentence_signals(t):
            continue
        out.append((style, n))
    return out


def infer_style_levels(texts):
    """The heading level of every numbering style in one document, learned from the document
    itself. Evidence: consecutive numbered headings of different styles A then B with B's number
    1 count one A -> B restart. A pair is decided as "A above B" when A -> B >= MIN_SUPPORT and
    B -> A == 0, or A -> B >= MIN_SUPPORT_MIXED and >= RATIO x B -> A; otherwise it stays
    undecided. Levels start from FIXED_LEVELS and every decided pair whose levels disagree
    pushes the lower style under the upper one (propagated), so the standard chain keeps its
    fixed levels and only the styles the document places differently move; a cycle (the
    document mixes its styles) falls back to the fixed table.
    Returns {"levels": {style: level}, "edges": [(above, below, restarts, reverse)],
             "undecided": [(a, b, restarts, reverse)], "fallback": bool}."""
    from collections import Counter
    seq = heading_sequence(texts)
    restart = Counter()
    for (a, _), (b, nb) in zip(seq, seq[1:]):
        if a != b and nb == 1:
            restart[(a, b)] += 1
    edges, undecided = [], []
    for a, b in sorted({tuple(sorted(k)) for k in restart}):
        f, r = restart[(a, b)], restart[(b, a)]
        hi, lo, s, c = (a, b, f, r) if f >= r else (b, a, r, f)
        if (c == 0 and s >= MIN_SUPPORT) or (s >= MIN_SUPPORT_MIXED and s >= RATIO * c):
            edges.append((hi, lo, s, c))
        else:
            undecided.append((hi, lo, s, c))
    levels = dict(FIXED_LEVELS)
    fallback = False
    for _ in range(len(levels) + 1):
        changed = False
        for hi, lo, _, _ in edges:
            if levels[hi] >= levels[lo]:
                levels[lo] = levels[hi] + 1
                changed = True
        if not changed:
            break
    else:
        fallback, levels = True, dict(FIXED_LEVELS)
    return {"levels": levels, "edges": edges, "undecided": undecided, "fallback": fallback}


class Leveler:
    """Heading levels: doc_title is 1, a numbered title gets the level of its style from the
    document's own table (infer_style_levels; the fixed table without one), a dotted number
    (1.1, 1.1.1) sits that many levels under the document's "1." level, an unnumbered title
    sits at the level of the previous heading (its sibling). Shared by the chunker (the tree)
    and quality gate 1 (the parent stack of the continuity check), so both see one hierarchy."""

    def __init__(self, levels=None):
        self.levels = dict(levels) if levels else dict(FIXED_LEVELS)
        self.current = 2

    def lookup(self, text):
        """Level a numbered title would get, without moving the sibling level; None if unnumbered."""
        style, lvl = style_of(text)
        if lvl is None:
            return None
        if style == "dot":
            return self.levels.get("ar_dot", FIXED_LEVELS["ar_dot"]) + (lvl - 2)
        return self.levels.get(style, lvl)

    def level(self, text, is_doc_title):
        if is_doc_title:
            self.current = 1
            return 1
        lvl = self.lookup(text)
        if lvl is None:                      # unnumbered: a sibling of the previous heading, never above level 2
            self.current = max(self.current, 2)
            return self.current
        self.current = lvl
        return lvl


# ----------------------------------------------------------------------------- bookmarks
# The PDF outline (bookmarks) is written by the authoring tool from the heading styles, so
# every entry is a heading the author styled as one, with its level and target page. It is
# not complete (unstyled headings have no entry) and not every file has one: of the four
# MinerU-parsed documents the contract (239 entries) and the annual report (448) do, the
# research report and the insurance terms do not.
NORM_HEAD = re.compile(r"[\s、．.:：（）()#]")


def norm_heading(s):
    """A heading for comparison with a bookmark: whitespace, list punctuation and the '#'
    a Word export leaves in some bookmark titles removed."""
    return NORM_HEAD.sub("", s or "")


def load_bookmarks(pdf):
    """{target page: [(level, title, normalised title)]} of the PDF outline; {} without one."""
    out = {}
    for level, title, page in pdf.get_toc():
        if page > 0 and title and title.strip():
            out.setdefault(page, []).append((level, title.strip(), norm_heading(title)))
    return out


def bookmark_covers(text, bookmark):
    """Does a title block's normalised text carry this normalised bookmark title? Equal, or
    one contains the other and the shorter is not a fragment (4 characters or more)."""
    if not text or not bookmark:
        return False
    if text == bookmark:
        return True
    short, long_ = (text, bookmark) if len(text) <= len(bookmark) else (bookmark, text)
    return len(short) >= 4 and short in long_


def bookmark_is_block(text, bookmark):
    """Is a block's normalised text this bookmark's heading and nothing else? The block may
    carry a short tail (a colon, a page number) or be a truncated form of the title, but not
    a paragraph that merely quotes it."""
    if not text or not bookmark:
        return False
    if text == bookmark:
        return True
    if bookmark in text:
        return len(text) <= len(bookmark) + 6
    return text in bookmark and len(text) >= max(4, int(0.6 * len(bookmark)))


# ----------------------------------------------------------------------------- cards
def source_for(doc_id):
    """(domain, source PDF relative path) from the doc_id naming convention, for documents
    that have no card yet; (None, None) when the id follows no known convention.
        fc_textNN       raw/financial_contracts/textNN.pdf
        fr_<name>_<yr>  raw/financial_reports/annual_<name>_<yr>_report.(PDF|pdf)
        ins_NN          raw/insurance/<N>.pdf
        pack2_textNN    raw/research/pack2_textNN.pdf
        csrc_NNNN_attN  raw/regulatory/attachments/csrc_NNNN_attN.pdf"""
    m = re.fullmatch(r"fc_(text\d+)", doc_id)
    if m:
        return "financial_contracts", f"raw/financial_contracts/{m.group(1)}.pdf"
    m = re.fullmatch(r"fr_([a-z0-9]+)_(\d{4})", doc_id)
    if m:
        hits = sorted((ROOT / "raw/financial_reports").glob(f"annual_{m.group(1)}_{m.group(2)}_report.*"))
        return "financial_reports", (str(hits[0].relative_to(ROOT)).replace("\\", "/") if hits else None)
    m = re.fullmatch(r"ins_(\d+)", doc_id)
    if m:
        return "insurance", f"raw/insurance/{int(m.group(1))}.pdf"
    if re.fullmatch(r"pack2_text\d+", doc_id):
        return "research", f"raw/research/{doc_id}.pdf"
    if re.fullmatch(r"csrc_\d{4}_att\d", doc_id):
        return "regulatory", f"raw/regulatory/attachments/{doc_id}.pdf"
    return None, None


def load_cards():
    path = PROC / "docs.jsonl"
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def save_cards(cards):
    with (PROC / "docs.jsonl").open("w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
