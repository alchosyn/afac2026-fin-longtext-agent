# -*- coding: utf-8 -*-
"""Document views built at load time from the two layers of processed_data/<doc_id>/:

  elements.jsonl   the structure: headings, paragraphs, whole tables, figures, with page and
                   bbox -- the layer of navigation and citation
  chunks.jsonl     the retrieval units (parsing.chunk): clauses, packed paragraphs, table
                   pieces with their header, captioned figures; each points back at the
                   structure records it was made from (members)

Views:
  * navigation view  -- every heading (full chain), the caption of every figure and table,
                        the first sentence of every paragraph (from the structure)
  * section view     -- the full content of one heading from the structure
  * units            -- the chunks, each with depth, path (heading chain) and body (the text
                        BM25 and the judge see: caption + text + source)
Nothing is summarised or generated; every line is copied from the files.

The chunks are read from chunks.jsonl when chunks.meta.json says they were made from the
current elements.jsonl; otherwise (missing or stale) they are chunked in memory with the
default parameters and a warning is printed: run script/build_chunks.py to persist them.
"""
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"
SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from parsing.chunk import chunk_doc  # noqa: E402


def load_cards():
    cards = {}
    for line in (PROC / "docs.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        cards[c["doc_id"]] = c
    return cards


def first_sentence(text, limit=60):
    s = SENT_SPLIT.split(text.strip(), maxsplit=1)[0].strip()
    return s[:limit]


def local_id(full_id):
    return full_id.split("#", 1)[1] if "#" in full_id else full_id


def load_chunks(doc_id, records, card=None):
    """The chunks of a document: chunks.jsonl when fresh, else chunked in memory."""
    path, meta_path = PROC / doc_id / "chunks.jsonl", PROC / doc_id / "chunks.meta.json"
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        current = hashlib.sha1((PROC / doc_id / "elements.jsonl").read_bytes()).hexdigest()
        if meta.get("structure_sha1") == current:
            return [json.loads(l) for l in path.open(encoding="utf-8")]
        print(f"[docview] chunks of {doc_id} are stale (elements.jsonl changed): chunking in memory, "
              f"run script/build_chunks.py {doc_id}", file=sys.stderr)
    else:
        print(f"[docview] no chunks.jsonl for {doc_id}: chunking in memory, run script/build_chunks.py {doc_id}", file=sys.stderr)
    card = card or {}
    chunks, _, _ = chunk_doc(records, doc_id, kind=card.get("kind"), title=card.get("title", ""))
    return chunks


class Doc:
    def __init__(self, doc_id, card=None):
        self.doc_id = doc_id
        self.card = card or load_cards()[doc_id]
        self.title = self.card["title"]
        self.records = [json.loads(l) for l in (PROC / doc_id / "elements.jsonl").open(encoding="utf-8")]
        self.chunks = load_chunks(doc_id, self.records, self.card)
        self.by_id = {r["id"]: r for r in self.records}
        self.by_id.update({c["id"]: c for c in self.chunks})
        self.children = defaultdict(list)                    # heading id -> structure records
        for r in self.records:
            self.children[r["parent"]].append(r)
        self.chunk_children = defaultdict(list)              # heading id -> chunks
        self._chunks_of = defaultdict(list)                  # structure id -> chunks made from it
        for c in self.chunks:
            self.chunk_children[c["parent"]].append(c)
            for m in c["members"]:
                self._chunks_of[m].append(c)
        self.headings = [r for r in self.records if r["type"] == "heading"]
        for r in self.records:
            r["depth"] = self._depth(r)
            r["path"] = self._path(r)
        self.units = self.chunks
        for u in self.units:
            u["page"] = u["pages"][0]
            u["depth"] = self._depth(u)
            u["path"] = self._path(u)
            u["body"] = self.unit_text(u)

    # ---- structure
    @staticmethod
    def indexable(rec):
        """A structure record that stands for content: any non-heading record except a
        figure without a caption (seals, signatures, photos, logos)."""
        if rec["type"] == "heading":
            return False
        if rec["type"] == "figure":
            return bool(rec.get("caption", rec.get("text")))
        return True

    def _depth(self, rec):
        d, p = 0, rec["parent"]
        while p:
            d += 1
            p = self.by_id[p]["parent"] if p in self.by_id else None
        return d

    def _path(self, rec):
        parts, p = [], rec["parent"]
        while p and p in self.by_id:
            h = self.by_id[p]
            if local_id(h["id"]) != "h0":
                parts.append(h["text"])
            p = h["parent"]
        return " > ".join(reversed(parts))

    def section_units(self, hid, subtree=False):
        """The chunks under one heading (its own, or with subtree=True those of its sub-sections too)."""
        out = list(self.chunk_children.get(hid, []))
        if subtree:
            for r in self.children.get(hid, []):
                if r["type"] == "heading":
                    out.extend(self.section_units(r["id"], True))
        return out

    def section_elements(self, hid, subtree=False):
        """The content records of the structure under one heading."""
        out = []
        for r in self.children.get(hid, []):
            if r["type"] == "heading":
                if subtree:
                    out.extend(self.section_elements(r["id"], True))
            elif self.indexable(r):
                out.append(r)
        return out

    def chunks_of(self, element_id):
        """The chunks made from one structure record (a packed paragraph: one; a split table: several)."""
        return list(self._chunks_of.get(element_id, []))

    def element_of(self, unit):
        """The first structure record behind a chunk (its figure, its table, its first paragraph)."""
        if "members" in unit:
            return self.by_id[unit["members"][0]]
        return unit

    # ---- text of one unit
    @staticmethod
    def unit_text(u):
        """The text BM25 and the judge see. A chunk: caption, text and source joined. A
        structure record (section views): the same assembly from its own fields."""
        if "members" in u:
            return "\n".join(x for x in (u.get("caption"), u["text"], u.get("source")) if x)
        if u["type"] in ("paragraph", "clause"):
            return u["text"]
        if u["type"] == "table":
            body = ((u.get("caption") or "") + "\n" + u["text"]).strip()
        else:
            body = u["text"]
            labels = u.get("labels") or u.get("legend")          # legend: field name of builds before 2026-09-24
            if labels:
                body += " 图内文字：" + "、".join(labels)
        if u.get("source"):
            body += "\n" + u["source"]
        return body

    def unit_header(self, u):
        head = f"{self.doc_id} 《{self.title}》 第{u['page']}页"
        if u["path"]:
            head += " | " + u["path"]
        return head

    # ---- views
    def nav_view(self, max_first_sentence=60):
        lines = []
        for h in self.headings:
            lid = local_id(h["id"])
            if lid == "h0":
                lines.append(f"[{lid}] 《{h['text']}》")
                continue
            lines.append(f"{'  ' * (h['depth'] - 1)}[{lid}] {h['text']} (p{h['page']})")
            pad = "  " * h["depth"] + "- "
            for u in self.section_elements(h["id"]):
                if u["type"] in ("paragraph", "clause"):
                    lines.append(pad + first_sentence(u["text"], max_first_sentence))
                elif u["type"] == "table":
                    lines.append(pad + "[表] " + (u.get("caption") or first_sentence(u["text"].split("\n")[0], 50)))
                else:
                    lines.append(pad + "[图] " + u["text"])
        return "\n".join(lines)

    def section_body(self, hid, subtree=None, max_chars=6000):
        """(heading record, text) of one heading from the structure: its paragraphs, tables
        and figures. subtree=None: descend into sub-sections only when the heading itself
        holds fewer than 3 records."""
        h = self.by_id[hid]
        units = self.section_elements(hid, False)
        if subtree or (subtree is None and len(units) < 3):
            units = self.section_elements(hid, True)
        return h, "\n\n".join(self.unit_text(u) for u in units)[:max_chars]

    def section_view(self, hid, subtree=None, max_chars=6000):
        """Full content of one heading as text, with a header line."""
        h, body = self.section_body(hid, subtree, max_chars)
        head = f"{self.doc_id} 《{self.title}》 第{h['page']}页起 | {h['path'] + ' > ' if h['path'] else ''}{h['text']}"
        return head + "\n" + body

    def grep(self, pattern):
        rx = re.compile(pattern)
        return [u for u in self.units if rx.search(u["body"])]
