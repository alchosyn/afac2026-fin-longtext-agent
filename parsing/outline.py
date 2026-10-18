# -*- coding: utf-8 -*-
"""The heading list of one document, processed_data/<doc_id>/docs.json: the headings its author
listed, from the PDF bookmarks or, for a file without bookmarks, from the printed table of
contents. Quality gate 1 matches every entry against the layout blocks (parsing.audit,
Audit.outline_checks) and reports the listed headings no title block carries.

Sources, the first one present wins:
  bookmark  the PDF outline (pymupdf get_toc): title, level, target page. The authoring tool
            writes it from the heading styles, down to every styled level (fc_text01 239
            entries, fr_byd_2024 448), so it lists more than the printed TOC does (fc_text01's
            TOC: 62 lines) and adds nothing when both exist.
  toc       the printed table of contents: the lines of MinerU's index blocks that end in a
            page number ("第一节 风险提示及说明....15"); an index block whose lines mostly do not
            is no TOC (insurance definitions, the audit's drop_index). A printed TOC is cut at
            a fixed depth (fr_catl_2024 lists only 第N节, fc_text02 down to 一、, pack2_text02
            down to （一）), and lists figures and tables of research reports as well
            (pack2_text03: 63 of 79 lines; kind figure / table, not matched as headings).
            MinerU keeps no position per line, so an entry's level is the rank of its
            numbering style among the TOC's styles (unnumbered: 1). A line without a page
            number is the first part of a wrapped entry and is joined to the next line.
            The printed page number becomes a PDF page through an offset learned from the
            entries a title block carries word for word on exactly one page (anchors): an
            entry uses its own anchor, else the nearest anchor before it in the TOC (an
            annex that restarts the page numbers changes the offset from there on), else the
            one after it, else the most common offset.
  none      neither (the csrc attachments, ins_01).

docs.json: {"doc_id", "source", "entries": [...], "styles": numbering styles of the heading
entries, shallowest first ("jie", "cn" for fc_text02: where the list is cut),
"depth": deepest entry level, and for a toc: "toc_pages", "wrapped" (joined lines),
"offset": {"default", "anchors", "values"}}
entry: {"title", "norm", "kind" (heading / figure / table), "level", "style",
        "page" (PDF page, 1-based; None when no offset is known), "printed_page" (toc only)}
"""
import re
from collections import Counter

from .common import TITLE_TYPES, TOC_LINE, clean, norm_heading, style_of, txt

TOC_ENTRY = re.compile(r"^(?P<title>.*?\S)\s*(?:\.{2,}|…+|·{2,}|\s)\s*(?P<page>\d{1,4})\s*$")
LIST_CAPTION = re.compile(r"^(图表|图|表)\s*[A-Z]?\d")


def index_lines(data):
    """[(page, line)] of the index blocks that are a table of contents (at least half of the
    lines end in a page number)."""
    out = []
    for pg in data["pages"]:
        for b in pg["blocks"]:
            if b["type"] != "index":
                continue
            lines = [clean(txt(c)) for c in b.get("content", []) if isinstance(c, dict)]
            lines = [l for l in lines if l] or [clean(txt(b))]
            if sum(1 for l in lines if TOC_LINE.search(l)) >= len(lines) / 2:
                out += [(pg["page_idx"] + 1, l) for l in lines]
    return out


def entry(title, level, page, **extra):
    style, _ = style_of(title)
    m = LIST_CAPTION.match(title)
    kind = ("table" if m.group(1) == "表" else "figure") if m else "heading"
    e = {"title": title, "norm": norm_heading(title), "kind": kind, "level": level, "style": style, "page": page}
    e.update(extra)
    return e


def from_bookmarks(pdf):
    return [entry(title.strip(), level, page) for level, title, page in pdf.get_toc() if page > 0 and title and title.strip()]


def from_toc(data):
    """(entries, toc pages, wrapped lines) of the printed TOC; page is still the printed number."""
    entries, pages, pending, wrapped = [], [], "", 0
    for pno, line in index_lines(data):
        m = TOC_ENTRY.match(line)
        if not m:
            pending += line
            continue
        title = (pending + m.group("title")).strip()
        if pending:
            wrapped += 1
        pending = ""
        if pno not in pages:
            pages.append(pno)
        entries.append(entry(title, None, None, printed_page=int(m.group("page"))))
    ranks = sorted({lvl for e in entries if e["kind"] == "heading" for lvl in [style_of(e["title"])[1]] if lvl is not None})
    for e in entries:
        lvl = style_of(e["title"])[1]
        e["level"] = 1 + ranks.index(lvl) if lvl is not None else 1
    return entries, pages, wrapped


def learn_offsets(entries, data, toc_pages):
    """Turn printed page numbers into PDF pages (see the module docstring)."""
    last_toc = max(toc_pages) if toc_pages else 0
    where = {}
    for pg in data["pages"]:
        pno = pg["page_idx"] + 1
        if pno <= last_toc:
            continue
        for b in pg["blocks"]:
            if b["type"] in TITLE_TYPES:
                where.setdefault(norm_heading(clean(txt(b))), set()).add(pno)
    heads = [e for e in entries if e["kind"] == "heading" and e.get("printed_page") is not None]
    anchor = {}
    for k, e in enumerate(heads):
        pages = where.get(e["norm"], set())
        if e["norm"] and len(pages) == 1:
            anchor[k] = next(iter(pages)) - e["printed_page"]
    values = Counter(anchor.values())
    default = values.most_common(1)[0][0] if values else None
    for k, e in enumerate(heads):
        off = anchor.get(k)
        if off is None:
            before = [anchor[j] for j in anchor if j < k]
            after = [anchor[j] for j in sorted(anchor) if j > k]
            off = before[-1] if before else after[0] if after else default
        e["page"] = e["printed_page"] + off if off is not None else None
    for e in entries:                                   # figure / table list lines: the most common offset
        if e["kind"] != "heading" and e.get("printed_page") is not None and default is not None:
            e["page"] = e["printed_page"] + default
    return {"default": default, "anchors": len(anchor), "values": dict(values.most_common(5))}


def build_outline(doc_id, pdf, data):
    """The heading list of a document: bookmarks, else the printed TOC, else empty."""
    out = {"doc_id": doc_id, "source": "none", "entries": []}
    entries = from_bookmarks(pdf)
    if entries:
        out.update(source="bookmark", entries=entries)
    else:
        entries, pages, wrapped = from_toc(data)
        if entries:
            out.update(source="toc", entries=entries, toc_pages=pages, wrapped=wrapped,
                       offset=learn_offsets(entries, data, pages))
    heads = [e for e in out["entries"] if e["kind"] == "heading"]
    levels = {}
    for e in heads:
        lvl = style_of(e["title"])[1]
        if lvl is not None:
            levels.setdefault(e["style"], lvl)
    out["styles"] = sorted(levels, key=levels.get)
    out["depth"] = max((e["level"] for e in heads), default=0)
    return out
