# -*- coding: utf-8 -*-
"""Retrieval units (chunks) over the structure layer processed_data/<doc_id>/elements.jsonl
-> processed_data/<doc_id>/chunks.jsonl (written, with chunks.meta.json, by script/build_chunks.py).

Three layers, each kept on disk:
    mineru.json      the layout blocks, untouched
    elements.jsonl   the structure: heading tree, paragraphs, whole tables, figures, with page
                     and bbox (parsing.structure); the layer every citation points at
    chunks.jsonl     the retrieval units; what BM25, the dense encoder and the judge see
A chunk never copies what the structure already holds (bbox, table rows, quality marks): it
points back at the structure records it was made from (members) and the code looks them up.
Headings are not chunks; a chunk carries the id of its heading (parent) and the heading path
is prepended to the index text at retrieval time (agent.retrieval.unit_text).

Rules (agreed 2026-10-01):
  * clause (第N条 of a regulation or an insurance policy): one chunk however short, its 款
    and 项 inside; longer than TEXT_CAP it is split at its 款 (the newlines), balanced, and
    every piece starts with the 第N条 label. An article is a clause record of the structure
    (legacy regulatory builds) or, where the structure holds paragraphs only (the MinerU
    route, 2026-10-03), the paragraph that opens with 第N条 plus the paragraphs after it
    under the same heading, up to the next article, heading, table or figure; "第十条修改为",
    "第十条第二款", "第十条、第十一条" cite an article and open none; read only in a document
    with at least CLAUSE_MIN such records. A whole article MinerU labelled as a title (the
    heading "第六条证券交易所依法建立程序化交易报告制度。": a sentence after the label) opens
    an article in the same way, a 款 / 项 of the open article labelled as a title (it reads
    as a sentence and is no 章 / 节 / 条 heading) is one more line of it, and the chunks of
    what the structure hangs under such a heading take its own heading as parent; a 第N条
    heading that is only a title ("第五条 保险责任", its text below) keeps its section as it is
  * paragraphs: consecutive paragraphs under one heading are packed in document order into
    chunks of about TEXT_CAP characters; a heading, table, figure or clause closes the pack;
    a tail shorter than TEXT_MIN joins the previous text chunk of the section; one paragraph
    longer than TEXT_CAP is split at sentence ends, balanced (the boundary nearest to each
    target within WINDOW; fallback ；/ newline, then ，, then a hard cut); a reference list
    "[1] ... [2] ..." or a code table "代码编号：..." without sentence ends is split at its
    own markers; a list introduced by a line ending with ： repeats that line in every piece
  * tables: one chunk when the markdown is at most TABLE_CAP characters; longer, split at row
    boundaries, balanced to about TABLE_CAP, cutting before a group row (only the first cell
    filled, "一、流动资产") or after a subtotal row (合计 / 小计) when one lies within
    TABLE_WINDOW of the target; every piece repeats the header rows (header_rows of the
    record, from the HTML spans; else the first row, plus a title row) and the group row it
    is inside; a row longer than the cap is a piece by itself; a tail shorter than TABLE_MIN
    stays with the previous piece
  * figures: one chunk when captioned (caption, the words inside the figure, the source line),
    with `image`, the path of the PNG the structure build cut out of the PDF (parsing.figures),
    which a retrieved figure chunk hands to the multimodal judge, and `numbers`, every number
    written in the figure; the caption is MinerU's, or the figure's first line when the hand
    review called it a title (`caption_from` text_layer_first_line / ocr_first_line); an
    uncaptioned figure (seal, signature, photo, logo) is not a chunk
  * noise that never becomes a chunk: checkbox boilerplate (□适用 √不适用, □是 √否) and the
    private-use glyphs of icon fonts; a standalone unit line (单位：元) right before a table
    joins the table's caption instead of being a paragraph

Post-processing (agreed 2026-10-01), in chunk_doc after the walk:
  * dedup: a chunk whose caption + text + source equals an earlier chunk's under the same
    heading is dropped (its members count as dropped for the coverage check); the same
    text under different headings is kept, it is different evidence
  * clause cross-references: in a law / regulation / standard, every 第N条 (本办法第N条,
    第N条至第M条, 第N条、第M条, 第N条第M款, 前条) mentioned in a chunk's text is resolved to the
    clause chunks of that article -> `refs` (chunk ids, document order); the scope of an
    article number is the whole document, or in a compilation the top-level section under
    which numbers are unique; 《某法》第N条 names another document -> `refs_external` (the
    text, for a later cross-document pass); a citation resolves only inside the citing
    chunk's own scope, so from a decision's or an explanation's section (no clauses of its
    own) every 第N条 goes to refs_external: there it names the document amended or explained
  * figure / table citations (2026-10-03): a numbered label in a text or clause chunk (见表1,
    如图3 所示, 应符合表2, 图表 12) is resolved to the table / figure chunks of the same
    document whose caption starts with that label (every piece of a split table) and their
    ids join `refs`; 数据表200余张, 代表1个 and the like are not citations; unnumbered
    mentions (如下图所示) are not resolved
  * ids are assigned after dedup, so they stay consecutive
  * gate 3 (check_chunks): parents and members exist, no empty chunk, table rows inside the
    table, coverage (every content element is in a chunk or in the dropped list), and with
    token lengths given, no chunk over the encoder's limit

Record:
    id        <doc_id>#c0001 ...                    in document order
    type      text | clause | table | figure
    parent    id of the enclosing heading (a structure record)
    members   ids of the structure records the chunk was made from, in order
    pages     [first page, last page] of the members
    text      the chunk's own text (text / clause: the paragraphs joined by newlines;
              table: markdown of the header rows and the piece's rows; figure: 图内文字：...)
    caption, source           table and figure only
    image, numbers            figure only: the PNG path; every number written in the figure
                              (axis ticks, data labels, years; reading order, as written), when any
    caption_from              figure only, when the caption is the figure's first line
    rows [lo, hi], part [k, n]    a table piece: body rows lo..hi (0-based, header excluded)
    span [lo, hi], part [k, n]    a text / clause piece: character span of the member's text
    refs                      ids of the chunks this one cites: clauses and numbered figures /
                              tables (see above), in document order, when any
    refs_external             citations of other documents' articles, as text

    from parsing.chunk import chunk_doc, check_chunks
    chunks, stats, dropped = chunk_doc(records, "fr_byd_2024", kind=card.get("kind"))
    problems = check_chunks(chunks, records, dropped)
"""
import math
import re
from collections import Counter, defaultdict

from .common import cn2int

TEXT_CAP = 800        # characters a text chunk aims at (about 512 bge-m3 tokens: 1.57 chars / token measured)
TEXT_MIN = 100        # a shorter tail joins the previous chunk of its section
WINDOW = 200          # a sentence end this far from the balanced target is taken
TABLE_CAP = 1000      # markdown characters of a table piece (about 650 tokens; 1024 is the encoder's hard limit)
TABLE_WINDOW = 300    # a group / subtotal row this far from the target is preferred as the cut
TABLE_MIN = 100

SENT_END = re.compile(r"[。！？][”’）」』]*")
SOFT_END = re.compile(r"[；\n]")
COMMA = re.compile(r"，")
REF_ITEM = re.compile(r"(?<!\S)\[\d{1,3}\]")           # "[12] GA/T 517—2004 ..." reference lists
CODE_ITEM = re.compile(r"代码编号[:：]")                 # code tables flattened into one paragraph
CLAUSE_LABEL = re.compile(r"^第[一二三四五六七八九十百零〇两\d]+条(?:之[一二三四五六七八九十]+)?")
# a paragraph that opens an article: "第十条 ..."; "第十条修改为 / 第十条中的 / 第十条第二款 / 第十条、第十一条"
# cite one and stay ordinary text (the test of the legacy regulatory builder)
CLAUSE_OPEN = re.compile(r"^第(?P<a>[一二三四五六七八九十百零〇两\d]+)条(?:之[一二三四五六七八九十]+)?"
                         r"(?!修改为|修改如下|删去|增加[一二两三]|改为|作为|中的|中“|中规定|中有关|中第|[前后]增加"
                         r"|第[一二三四五六七八九十百零〇两\d]+[款项]|的|之|至第|和第|或第|或者第|[、，；：）》”])")
CLAUSE_MIN = 3        # paragraphs opening an article a document needs before its paragraphs are read as clauses


def article_as_title(text):
    """A heading that is a whole article: opens with 第N条 and goes on as a sentence (a full
    stop inside, a colon or semicolon at the end, or more than 40 characters). "第五条 保险责任"
    is the title of an article whose text stands below it and stays a heading."""
    m = CLAUSE_OPEN.match(text)
    if not m:
        return False
    return reads_as_sentence(text[m.end():].strip())


STRUCT_TITLE = re.compile(r"^第[一二三四五六七八九十百零〇两\d]+[编章节条]|^附\s*[则件录]")


def reads_as_sentence(text):
    """Wording a section title does not have: a comma or a full stop inside, a colon or
    semicolon at the end, or more than 40 characters."""
    return "，" in text or "。" in text or text.endswith(("：", ":", "；", ";")) or len(text) > 40


def line_as_title(text):
    """A line of an article MinerU labelled as a title ("（六）法律、行政法规和国务院有关部门
    规定的其他情形。"): it reads as a sentence and is no 编 / 章 / 节 / 条 / 附则 heading."""
    return not STRUCT_TITLE.match(text) and reads_as_sentence(text)
LEAD_IN = re.compile(r"[:：]\s*$")
PUA = re.compile(r"[-]")                   # icon-font glyphs ( is a ticked box)
CHECKBOX = re.compile(r"^[□■√☑☐✓✔\s]*(适用|不适用|是|否)[□■√☑☐✓✔\s]*(适用|不适用|是|否)?[□■√☑☐✓✔\s]*$")
UNIT_LINE = re.compile(r"^(?:单位|币种|金额单位|计量单位)[:：]?\s*[^\s，。；]{1,12}$")
SEP_LINE = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+\s*$")
SUBTOTAL = re.compile(r"合计|小计|总计")


def clean_text(s):
    return re.sub(r"[ \t]+", " ", PUA.sub("", s or "")).strip()


def is_checkbox(text):
    return bool(CHECKBOX.match(clean_text(text)))


# ----------------------------------------------------------------------------- balanced cutting
def balanced_cuts(length, cap, window, levels, min_tail):
    """Cut points of a text of `length` characters into pieces of about `cap`: the number of
    pieces is ceil(remaining / cap), recomputed after every cut; each cut is the candidate
    boundary nearest to the balanced target within `window`, taken from the first level
    (sorted candidate positions) that has one, else a hard cut at the target. A final piece
    shorter than min_tail is folded into the previous one. Returns [(lo, hi), ...]."""
    spans, lo = [], 0
    while length - lo > cap:
        n = math.ceil((length - lo) / cap)
        target = lo + round((length - lo) / n)
        cut = None
        for cands in levels:
            near = [p for p in cands if lo < p < length and abs(p - target) <= window]
            if near:
                cut = min(near, key=lambda p: abs(p - target))
                break
        if cut is None:
            cut = target
        spans.append((lo, cut))
        lo = cut
    if lo < length:
        if spans and length - lo < min_tail:
            spans[-1] = (spans[-1][0], length)
        else:
            spans.append((lo, length))
    return spans


def text_levels(text):
    """Candidate boundaries of a prose paragraph, best first: sentence ends; ；and newlines;
    commas. A paragraph without sentence ends that is a reference list or a flattened code
    table gets its item starts as the first level."""
    ends = [m.end() for m in SENT_END.finditer(text)]
    levels = [ends, [m.end() for m in SOFT_END.finditer(text)], [m.end() for m in COMMA.finditer(text)]]
    if len(ends) < 2:
        for rx in (REF_ITEM, CODE_ITEM):
            starts = [m.start() for m in rx.finditer(text)]
            if len(starts) >= 3:
                levels.insert(0, starts)
                break
    return levels


def split_paragraph(text, cap=TEXT_CAP, window=WINDOW, min_tail=TEXT_MIN):
    """[(lo, hi, piece text)] of a paragraph longer than cap. A list whose first line ends
    with ： repeats that line at the top of every later piece."""
    spans = balanced_cuts(len(text), cap, window, text_levels(text), min_tail)
    lead = ""
    first = text.split("\n", 1)[0]
    if "\n" in text and LEAD_IN.search(first) and len(first) <= 60:
        lead = first.strip()
    out = []
    for k, (lo, hi) in enumerate(spans):
        piece = text[lo:hi].strip()
        if k and lead and not piece.startswith(lead):
            piece = lead + "\n" + piece
        out.append((lo, hi, piece))
    return out


def split_clause(text, cap=TEXT_CAP, min_tail=TEXT_MIN):
    """[(lo, hi, piece text)] of a clause longer than cap: cut at its 款 (newlines) only,
    every later piece starting with the 第N条 label."""
    newlines = [m.end() for m in re.finditer(r"\n", text)]
    spans = balanced_cuts(len(text), cap, max(cap, 1), [newlines], min_tail)
    m = CLAUSE_LABEL.match(text)
    label = m.group(0) if m else ""
    out = []
    for k, (lo, hi) in enumerate(spans):
        piece = text[lo:hi].strip()
        if k and label and not piece.startswith(label):
            piece = label + " " + piece
        out.append((lo, hi, piece))
    return out


# ----------------------------------------------------------------------------- tables
def md_cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_group_row(line):
    cells = md_cells(line)
    return len(cells) > 1 and bool(cells[0]) and not any(cells[1:])


def is_subtotal_row(line):
    cells = md_cells(line)
    return bool(cells) and bool(SUBTOTAL.search(cells[0]))


def table_parts(text):
    """(prefix lines, header lines, body lines, suffix lines, header_rows guess) of a
    markdown table: the first pipe line and the separator are the header; lines before the
    first pipe line and after the last one are kept with every piece."""
    lines = text.split("\n")
    pipes = [i for i, l in enumerate(lines) if l.startswith("|")]
    if not pipes:
        return lines, [], [], [], 1
    first, last = pipes[0], pipes[-1]
    prefix, suffix = lines[:first], lines[last + 1:]
    grid = [l for l in lines[first:last + 1] if l.startswith("|")]
    header = grid[:1]
    rest = grid[1:]
    if rest and SEP_LINE.match(rest[0]):
        header.append(rest[0])
        rest = rest[1:]
    guess = 2 if header and sum(1 for c in md_cells(header[0]) if c) == 1 and len(rest) > 1 else 1
    return prefix, header, rest, suffix, guess


def split_table(text, header_rows=None, cap=TABLE_CAP, window=TABLE_WINDOW, min_tail=TABLE_MIN):
    """[(lo, hi, piece markdown)] of a table whose markdown is longer than cap: body rows
    lo..hi (0-based) with the header rows and the current group row repeated. Returns a
    single piece covering every row when the body fits in cap."""
    prefix, header, body, suffix, guess = table_parts(text)
    h = header_rows or guess
    header = header + body[:max(0, h - 1)]
    body = body[max(0, h - 1):]
    if not body:
        return [(0, -1, text)]
    overhead = sum(len(l) + 1 for l in prefix + header + suffix)     # repeated in every piece: comes off the budget
    budget = max(cap // 2, cap - overhead)
    size = [len(l) + 1 for l in body]
    pieces, lo, group = [], 0, None
    while lo < len(body):
        rest = size[lo:]
        total = sum(rest)
        if total <= budget * 1.1:                           # a 10 % overshoot beats a split into two halves
            j = len(rest)
        else:
            n = math.ceil(total / budget)
            target = total / n
            acc, best, near = 0, None, []
            for j, s in enumerate(rest):                    # a cut before row lo + j
                if j:
                    if acc <= budget and (best is None or abs(acc - target) < abs(best[0] - target)):
                        best = (acc, j)
                    if abs(acc - target) <= window:
                        if is_group_row(body[lo + j]):
                            near.append((0, abs(acc - target), j))
                        elif is_subtotal_row(body[lo + j - 1]):
                            near.append((1, abs(acc - target), j))
                acc += s
                if acc > target + window:
                    break
            j = min(near)[2] if near else best[1] if best else 1
            if sum(rest[j:]) < min_tail:                    # a tiny tail stays with this piece
                j = len(rest)
        rows = body[lo:lo + j]
        head = [group] if group and not is_group_row(rows[0]) else []
        pieces.append((lo, lo + j - 1, "\n".join(prefix + header + head + rows + suffix)))
        for line in rows:
            if is_group_row(line):
                group = line
        lo += j
    return pieces


# ----------------------------------------------------------------------------- the document
def pages_of(recs):
    pages = []
    for r in recs:
        pages += r.get("pages") or ([r["page"]] if r.get("page") else [])
    return [min(pages), max(pages)] if pages else [None, None]


def table_pages(rec, lo, hi):
    """Pages of body rows lo..hi from the record's row_pages (absolute row index = body index
    + header rows), else the whole table's pages."""
    rp, h = rec.get("row_pages"), rec.get("header_rows") or 1
    if rp and lo <= hi:
        pages = [p for a, b, p in rp if a <= hi + h and b >= lo + h]
        if pages:
            return [min(pages), max(pages)]
    return pages_of([rec])


class Chunker:
    def __init__(self, doc_id, text_cap=TEXT_CAP, text_min=TEXT_MIN, window=WINDOW, table_cap=TABLE_CAP,
                 table_window=TABLE_WINDOW, table_min=TABLE_MIN):
        self.doc_id = doc_id
        self.text_cap, self.text_min, self.window = text_cap, text_min, window
        self.table_cap, self.table_window, self.table_min = table_cap, table_window, table_min
        self.chunks, self.stats = [], Counter()
        self.dropped = defaultdict(list)                        # reason -> ids of structure records left out on purpose
        self.buf = []                                           # [(record, text)] paragraphs waiting to be packed
        self.buf_parent = None
        self.clause = []                                        # [(record, text)] paragraphs of the article being read
        self.reparent = {}                                      # a whole article labelled as a title -> the heading it sits under

    def par(self, rec):
        """The heading a record's chunk hangs under: its parent, or that parent's own parent
        when the parent is a whole article MinerU labelled as a title (run)."""
        return self.reparent.get(rec["parent"], rec["parent"])

    def params(self):
        return {"text_cap": self.text_cap, "text_min": self.text_min, "window": self.window,
                "table_cap": self.table_cap, "table_window": self.table_window, "table_min": self.table_min}

    def new_id(self):
        return f"{self.doc_id}#c{len(self.chunks) + 1:04d}"

    def emit(self, ctype, parent, members, text, pages=None, **extra):
        rec = {"id": self.new_id(), "type": ctype, "parent": parent, "members": [m["id"] for m in members],
               "pages": pages or pages_of(members), "text": text}
        rec.update(extra)
        self.chunks.append(rec)
        self.stats[f"chunk_{ctype}"] += 1
        return rec

    # ---- paragraphs
    def buf_len(self):
        return sum(len(t) for _, t in self.buf) + max(0, len(self.buf) - 1)

    def flush(self):
        if not self.buf:
            return
        last = self.chunks[-1] if self.chunks else None
        if (self.buf_len() < self.text_min and last and last["type"] == "text" and last["parent"] == self.buf_parent
                and "part" not in last):
            last["members"] += [r["id"] for r, _ in self.buf]
            last["text"] += "\n" + "\n".join(t for _, t in self.buf)
            last["pages"] = pages_of([r for r, _ in self.buf] + [last])
            self.stats["tail_joined"] += 1
        else:
            if len(self.buf) > 1:
                self.stats["paragraphs_packed"] += len(self.buf)
            self.emit("text", self.buf_parent, [r for r, _ in self.buf], "\n".join(t for _, t in self.buf))
        self.buf, self.buf_parent = [], None

    def add_paragraph(self, rec, text):
        if len(text) > self.text_cap:
            self.flush()
            pieces = split_paragraph(text, self.text_cap, self.window, self.text_min)
            for k, (lo, hi, piece) in enumerate(pieces, 1):
                self.emit("text", self.par(rec), [rec], piece, span=[lo, hi], part=[k, len(pieces)])
            self.stats["paragraph_split"] += 1
            self.stats["paragraph_pieces"] += len(pieces)
            return
        if self.buf and (self.par(rec) != self.buf_parent or self.buf_len() + 1 + len(text) > self.text_cap):
            self.flush()
        self.buf.append((rec, text))
        self.buf_parent = self.par(rec)

    # ---- the other kinds
    def add_clause(self, members, text):
        """One article: a clause record of the structure, or the paragraphs of an article the
        structure holds as paragraphs (close_clause)."""
        self.flush()
        parent = self.par(members[0])
        if len(text) <= self.text_cap:
            self.emit("clause", parent, members, text)
            return
        pieces = split_clause(text, self.text_cap, self.text_min)
        for k, (lo, hi, piece) in enumerate(pieces, 1):
            self.emit("clause", parent, members, piece, span=[lo, hi], part=[k, len(pieces)])
        self.stats["clause_split"] += 1
        self.stats["clause_pieces"] += len(pieces)

    def close_clause(self):
        if not self.clause:
            return
        self.stats["clause_from_paragraphs"] += 1
        self.stats["clause_paragraphs"] += len(self.clause)
        members, text = [r for r, _ in self.clause], "\n".join(t for _, t in self.clause)
        self.clause = []
        self.add_clause(members, text)

    def add_table(self, rec, extra_caption=""):
        self.flush()
        caption = clean_text(rec.get("caption") or "")
        if is_checkbox(caption):
            caption, self.stats["caption_checkbox_dropped"] = "", self.stats["caption_checkbox_dropped"] + 1
        if extra_caption:
            caption = (caption + " " + extra_caption).strip() if caption else extra_caption
        source = clean_text(rec.get("source") or "")
        text = rec["text"]
        if len(text) <= self.table_cap:
            self.emit("table", self.par(rec), [rec], text, caption=caption, source=source)
            return
        pieces = split_table(text, rec.get("header_rows"), self.table_cap, self.table_window, self.table_min)
        if len(pieces) == 1:
            self.emit("table", self.par(rec), [rec], text, caption=caption, source=source)
            return
        for k, (lo, hi, piece) in enumerate(pieces, 1):
            self.emit("table", self.par(rec), [rec], piece, pages=table_pages(rec, lo, hi), caption=caption,
                      source=source, rows=[lo, hi], part=[k, len(pieces)])
        self.stats["table_split"] += 1
        self.stats["table_pieces"] += len(pieces)

    def add_figure(self, rec):
        self.flush()
        caption = clean_text(rec.get("caption", rec.get("text")) or "")
        if not caption or caption in ("图片", "图表"):
            self.stats["figure_uncaptioned_skipped"] += 1
            self.dropped["figure_uncaptioned"].append(rec["id"])
            return
        labels = [x for x in (rec.get("labels") or rec.get("legend") or []) if x != caption]
        if rec.get("caption_from", "mineru") != "mineru":                    # the first line is the caption: its words are not labels too
            labels = [x for x in labels if x not in caption or re.fullmatch(r"(?:19|20)\d{2}\S{0,3}", x)]
        text = ("图内文字：" + "、".join(labels)) if labels else ""
        extra = {"image": rec["image"]} if rec.get("image") else {}        # pointer to the PNG the structure build saved
        if rec.get("numbers"):
            extra["numbers"] = rec["numbers"]                              # every number written in the figure, reading order
        if rec.get("caption_from") and rec["caption_from"] != "mineru":
            extra["caption_from"] = rec["caption_from"]
        self.emit("figure", self.par(rec), [rec], text, caption=caption, source=clean_text(rec.get("source") or ""), **extra)

    # ---- the walk
    def run(self, records):
        # a standalone unit line (单位：元) right before a table joins the table's caption
        attach = {}
        content = [i for i, r in enumerate(records) if r["type"] != "heading"]
        for k, i in enumerate(content):
            r = records[i]
            if r["type"] == "paragraph" and UNIT_LINE.match(clean_text(r["text"])):
                nxt = next((records[j] for j in content[k + 1:] if not (records[j]["type"] == "paragraph" and is_checkbox(records[j]["text"]))), None)
                if nxt is not None and nxt["type"] == "table" and nxt["parent"] == r["parent"]:
                    attach[nxt["id"]] = clean_text(r["text"])
                    attach[r["id"]] = None
        # articles the structure holds as paragraphs (the MinerU route has no clause records):
        # read only in a document with at least CLAUSE_MIN records opening one
        articles = sum(1 for r in records if (r["type"] == "paragraph" and CLAUSE_OPEN.match(clean_text(r["text"])))
                       or (r["type"] == "heading" and article_as_title(clean_text(r["text"])))) >= CLAUSE_MIN
        for r in records:
            t = r["type"]
            if t == "heading":
                text = clean_text(r["text"])
                if articles and article_as_title(text):
                    # a whole article MinerU labelled as a title ("第六条证券交易所依法建立程序化交易
                    # 报告制度。"): it opens an article like a paragraph would, and what the structure
                    # hangs under it (its 款, the articles after it up to the next heading) goes
                    # under its own heading
                    self.close_clause()
                    self.reparent[r["id"]] = self.par(r)
                    self.clause = [(r, text)]
                    self.stats["clause_from_title"] += 1
                elif self.clause and line_as_title(text) and self.par(r) == self.par(self.clause[0][0]):
                    # a 款 / 项 of the open article labelled as a title: one more line of the article
                    self.reparent[r["id"]] = self.par(r)
                    self.clause.append((r, text))
                    self.stats["clause_line_from_title"] += 1
                else:
                    self.close_clause()
                    self.flush()
                continue
            if t != "paragraph":
                self.close_clause()
            if t == "paragraph":
                if r["id"] in attach:
                    self.stats["unit_line_attached"] += 1
                    self.dropped["unit_line_attached"].append(r["id"])
                    continue
                text = clean_text(r["text"])
                if not text:
                    self.stats["paragraph_empty_dropped"] += 1
                    self.dropped["empty"].append(r["id"])
                elif is_checkbox(text):
                    self.stats["checkbox_dropped"] += 1
                    self.dropped["checkbox"].append(r["id"])
                elif articles and CLAUSE_OPEN.match(text):          # 第N条 ...: a new article
                    self.close_clause()
                    self.clause = [(r, text)]
                elif self.clause and self.par(r) == self.par(self.clause[0][0]):
                    self.clause.append((r, text))                   # its 款 and 项, up to the next article or heading
                else:
                    self.close_clause()
                    self.add_paragraph(r, text)
            elif t == "clause":
                self.add_clause([r], clean_text(r["text"]))
            elif t == "table":
                self.add_table(r, attach.get(r["id"]) or "")
            elif t == "figure":
                self.add_figure(r)
            else:
                self.stats[f"unknown_type_{t}"] += 1
        self.close_clause()
        self.flush()
        for c in self.chunks:
            size = len(c["text"]) + len(c.get("caption") or "")
            bucket = "<100" if size < 100 else "100-400" if size < 400 else "400-800" if size < 800 else "800-1200" if size < 1200 else ">1200"
            self.stats[f"size_{c['type']}_{bucket}"] += 1
        return self.chunks


# ----------------------------------------------------------------------------- post-processing
def chunk_body(c):
    return "\n".join(x for x in (c.get("caption"), c["text"], c.get("source")) if x)


def dedup(chunks):
    """Drop a chunk whose body equals an earlier chunk's under the same heading (the same
    text under another heading is other evidence and stays). Returns (kept, dropped)."""
    seen, kept, dropped = set(), [], []
    for c in chunks:
        key = (c["parent"], chunk_body(c))
        if key in seen:
            dropped.append(c)
        else:
            seen.add(key)
            kept.append(c)
    return kept, dropped


def renumber(chunks, doc_id):
    for n, c in enumerate(chunks, 1):
        c["id"] = f"{doc_id}#c{n:04d}"
    return chunks


CN = "[一二三四五六七八九十百零〇两]+"
ARTICLE_REF = re.compile(rf"(?P<ctx>《[^《》]{{1,60}}》|本(?:办法|规定|法|细则|指引|准则|规则|条例|意见|通知|决定|实施细则))?"
                         rf"\s*第(?P<a>{CN})条(?:至第(?P<b>{CN})条)?")
PREV_ARTICLE = re.compile(r"前条")
# the document kind is not consulted: a citation resolves inside the citing chunk's own scope when that scope holds clauses


def link_clauses(chunks, records, doc_title="", kind=None, stats=None):
    """Clause cross-references (module docstring): sets `refs` (chunk ids of the cited
    articles, document order) and `refs_external` (citations of other documents, as text)
    on the clause and text chunks that have any. The scope of an article number is the
    shallowest heading level under which the numbers are unique (the document, or the
    top-level sections of a compilation); where the heading tree separates the numbers only
    at chapter level or not at all (a compilation whose regulation titles are not headings
    of one level), the scope is the run of articles up to the next restart of the numbering.
    Returns the stats."""
    stats = Counter() if stats is None else stats
    by_id = {r["id"]: r for r in records}
    chain_cache = {}

    def chain(hid):                                   # root ... heading
        if hid not in chain_cache:
            out, h = [], hid
            while h and h in by_id:
                out.append(h)
                h = by_id[h]["parent"]
            chain_cache[hid] = list(reversed(out))
        return chain_cache[hid]

    def ancestor(hid, depth):
        ch = chain(hid)
        return ch[min(depth, len(ch) - 1)] if ch else None

    def article(c):                                   # the record's number, or the label of an article read from paragraphs
        a = by_id[c["members"][0]].get("article")
        if not isinstance(a, int):
            m = CLAUSE_OPEN.match(c["text"])
            a = cn2int(m.group("a")) if m else None
        return a

    clauses = [(c, article(c)) for c in chunks if c["type"] == "clause"]
    clauses = [(c, a) for c, a in clauses if isinstance(a, int)]
    depth, unique = 0, True
    if clauses:
        for depth in range(max(len(chain(c["parent"])) for c, _ in clauses)):
            groups = defaultdict(list)
            for c, a in clauses:
                if "part" not in c or c["part"][0] == 1:  # one entry per article, its pieces aside
                    groups[ancestor(c["parent"], depth)].append(a)
            unique = all(len(g) == len(set(g)) for g in groups.values())
            if unique:
                break
    runs = None
    if clauses and (depth >= 2 or not unique):
        # the heading tree does not separate the regulations of a compilation (their titles
        # are not all headings of one level, so the numbers are unique only chapter by chapter
        # and a citation of another chapter's article would find nothing): the scope is the
        # run of articles up to the next restart of the numbering (run 0, what stands before
        # the first article, holds no clause: the items of a 修改决定 cite other documents)
        runs, run, last = {}, 0, None
        for c in chunks:
            if c["type"] == "clause" and ("part" not in c or c["part"][0] == 1):
                a = article(c)
                if isinstance(a, int):
                    if last is None or a <= last:
                        run += 1
                    last = a
            runs[c["id"]] = run
        stats["ref_scope_runs"] = run

    def scope_of(c):
        return ("run", runs[c["id"]]) if runs is not None else ancestor(c["parent"], depth)

    index = defaultdict(list)
    for c, a in clauses:
        index[(scope_of(c), a)].append(c["id"])
    scopes_with_clauses = {s for s, _ in index}
    title = (doc_title or "").strip("《》")
    for c in chunks:
        if c["type"] not in ("clause", "text"):
            continue
        text = c["text"]
        own = article(c) if c["type"] == "clause" else None
        scope = scope_of(c)
        in_doc = scope in scopes_with_clauses          # a citation from inside a regulation's body names its own
        wanted, external = [], []                      # article; from a decision / explanation section it names
        for m in ARTICLE_REF.finditer(text):           # the document it amends or explains
            if m.start() == 0 or text[m.start() - 1] == "\n":   # a clause label at the start of a line, not a citation
                continue
            ctx = m.group("ctx") or ""
            try:
                a = cn2int(m.group("a"))
                b = cn2int(m.group("b")) if m.group("b") else a
            except Exception:
                continue
            if not (0 < a <= b <= a + 50):
                continue
            named = ctx.strip("《》")
            foreign = ctx.startswith("《") and not (named in title or title in named)
            if foreign or not in_doc:
                external.append(m.group(0).strip())
            else:
                wanted += range(a, b + 1)
        if own and in_doc and PREV_ARTICLE.search(text):
            wanted.append(own - 1)
        refs, unresolved = [], 0
        for n in dict.fromkeys(wanted):
            if n == own:
                continue
            ids = index.get((scope, n))
            if ids:
                refs += [i for i in ids if i != c["id"] and i not in refs]
            else:
                unresolved += 1
        if refs:
            c["refs"] = refs
            stats["chunks_with_refs"] += 1
            stats["ref_links"] += len(refs)
        if external:
            c["refs_external"] = list(dict.fromkeys(external))
            stats["refs_external"] += len(c["refs_external"])
        stats["refs_unresolved"] += unresolved
    if clauses:
        stats["ref_scope_depth"] = depth
    return stats


FIG_LABEL = (r"(附?图表|附?图|附?表)\s*([A-Za-z]?\d{1,3}(?:[-.．－]\d{1,3})*)"
             r"(?![\d.．])(?!\s*[年月日%％万亿元个张项份次])")                 # 表 2022 年末… is a caption without a number
FIG_CAPTION = re.compile(r"^\s*" + FIG_LABEL)
FIG_CITE = re.compile(FIG_LABEL)
FIG_WORD_BEFORE = re.compile(r"(?:数据|代|发|报|列|仪|地|视|截|试|企|意|蓝)$")    # 数据表200余张, 代表1个, 地图1张: no citation


def link_figures(chunks, stats=None):
    """Figure / table citations (module docstring): a numbered label (见表1, 如图3 所示,
    应符合表2, 图表 12) in a text or clause chunk is resolved to the table or figure chunks
    of the same document whose caption starts with that label, and their ids join `refs`.
    A table the structure holds page by page ("表 3 …", then "表 3 …（续）" on the following
    pages: a captioned table is never merged across pages) is one target, all its parts;
    a label otherwise carried by several tables resolves to the nearest one in document
    order."""
    stats = Counter() if stats is None else stats
    order = {c["id"]: k for k, c in enumerate(chunks)}
    labelled = defaultdict(lambda: defaultdict(list))            # (kind, number) -> structure record -> its chunks
    continued = set()                                            # structure records captioned （续）
    for c in chunks:
        if c["type"] in ("table", "figure"):
            m = FIG_CAPTION.match(c.get("caption") or "")
            if m:
                labelled[(m.group(1), m.group(2))][c["members"][0]].append(c["id"])
                if "续" in c["caption"]:
                    continued.add(c["members"][0])
    if not labelled:
        return stats
    for targets in labelled.values():
        parts = list(targets)
        if len(parts) > 1 and all(p in continued for p in parts[1:]):
            whole = [i for p in parts for i in targets[p]]
            for p in parts:
                targets[p] = whole
    for c in chunks:
        if c["type"] not in ("clause", "text"):
            continue
        text, found = c["text"], []
        for m in FIG_CITE.finditer(text):
            if m.group(1) in ("图", "表") and FIG_WORD_BEFORE.search(text[max(0, m.start() - 2):m.start()]):
                continue
            targets = labelled.get((m.group(1), m.group(2)))
            if not targets:
                stats["fig_refs_unresolved"] += 1
                continue
            ids = min(targets.values(), key=lambda v: abs(order[v[0]] - order[c["id"]]))
            found += [i for i in ids if i not in found]
        if found:
            refs = c.get("refs", [])
            c["refs"] = sorted(dict.fromkeys(refs + found), key=order.get)
            stats["chunks_with_fig_refs"] += 1
            stats["fig_ref_links"] += len(found)
    return stats


def chunk_doc(records, doc_id, kind=None, title="", **params):
    """(chunks, stats, dropped) of one document's structure records: the walk, then dedup,
    consecutive ids, the clause cross-references and the figure / table citations. `dropped`
    maps a reason to the ids of the structure records left out on purpose (the coverage check
    reads it)."""
    ch = Chunker(doc_id, **params)
    chunks = ch.run(records)
    kept, gone = dedup(chunks)
    for c in gone:
        ch.dropped["dedup"] += c["members"]
    ch.stats["dedup_dropped"] += len(gone)
    renumber(kept, doc_id)
    link_clauses(kept, records, title, kind, ch.stats)
    link_figures(kept, ch.stats)
    return kept, ch.stats, dict(ch.dropped)


TOKEN_CAP = 1024            # bge-m3 passage limit in agent.retrieval
HEADER_ALLOWANCE = 64       # tokens the title | notice | heading path line adds in front of a chunk


def check_chunks(chunks, records, dropped=None, token_lengths=None, token_cap=TOKEN_CAP):
    """Gate 3: every member and parent exists, no empty chunk, no duplicate id, a table
    piece's rows are inside its table, every ref points at a chunk, every content record of
    the structure is in a chunk or in `dropped`, and (when token_lengths {chunk id: tokens}
    is given) no chunk plus its header line exceeds the encoder's limit. Returns the
    problems (empty = fine)."""
    by_id = {r["id"]: r for r in records}
    ids = {c["id"] for c in chunks}
    problems, seen, covered = [], set(), set()
    for c in chunks:
        if c["id"] in seen:
            problems.append(f"{c['id']}: duplicate id")
        seen.add(c["id"])
        covered.update(c["members"])
        if c["parent"] not in by_id or by_id[c["parent"]]["type"] != "heading":
            problems.append(f"{c['id']}: parent {c['parent']} is not a heading of the structure")
        for m in c["members"]:
            if m not in by_id:
                problems.append(f"{c['id']}: member {m} not in the structure")
        if not (c["text"] or c.get("caption")):
            problems.append(f"{c['id']}: empty")
        if c["type"] == "table" and "rows" in c:
            n = len(by_id.get(c["members"][0], {}).get("rows") or [])
            if n and c["rows"][1] >= n:
                problems.append(f"{c['id']}: rows {c['rows']} beyond the table's {n} rows")
        for r in c.get("refs", []):
            if r not in ids:
                problems.append(f"{c['id']}: ref {r} is not a chunk")
        if token_lengths and token_lengths.get(c["id"], 0) + HEADER_ALLOWANCE > token_cap:
            problems.append(f"{c['id']}: {token_lengths[c['id']]} tokens + header exceed the encoder's {token_cap}")
    if dropped is not None:
        left_out = {i for ids_ in dropped.values() for i in ids_}
        for r in records:
            if r["type"] != "heading" and r["id"] not in covered and r["id"] not in left_out:
                problems.append(f"{r['id']}: {r['type']} in no chunk and not dropped on purpose")
    return problems
