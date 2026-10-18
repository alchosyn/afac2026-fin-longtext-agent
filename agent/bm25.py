# -*- coding: utf-8 -*-
"""The inverted index of a document's chunks, on disk as processed_data/<doc_id>/bm25.json,
and BM25 scoring over the merged indexes of a question's documents.

Per document, two fields (agent.retrieval scores body * 0.7 + path * 0.3):
    body   one entry per chunk: its caption + text + source (docview's unit body)
    path   one entry per heading that holds chunks: document title | notice | heading path.
           The chunks under a heading share its entry (heading_of: chunk -> heading), so a
           path counts once in N and in the document frequencies however many chunks stand
           under it (until 2026-10-03 every chunk carried its own copy of the path)
Each field is a postings table  term -> [[local id, term frequency], ...]  plus the token
count of every entry; the tokens come from agent.claims.tokenize (jieba words, lower-cased,
function words dropped, numbers without thousands separators, English tokens). The file
records the sha1 of the chunks.jsonl it was built from: a stale or missing file is rebuilt
in memory with a warning (seconds), script/build_index.py persists it.

At query time the indexes of the question's documents are merged: postings are unioned
with the ids prefixed by their document, N and document frequencies add up, so the IDF is
computed over exactly the question's documents (as the in-memory rank_bm25 index did
before 2026-10-01). The body scoring is BM25Okapi's as rank_bm25 implements it (k1 1.5,
b 0.75, negative IDFs floored at 0.25 x the mean IDF); only the chunks that contain a
query term are touched. The path field uses the same formula with an IDF that cannot be
negative (MergedIndex._idf), scores the headings and hands every chunk its heading's score.

    from agent.bm25 import doc_index, MergedIndex
    idx = MergedIndex([(doc, doc_index(doc)) for doc in docs])   # loads / builds the documents' bm25.json
    scores = idx.scores(tokenize(query), "body")       # {full chunk id: score}, chunks containing a query term
    scores = idx.scores(tokenize(query), "path")       # {full chunk id: score of its heading's path}
"""
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .claims import tokenize
from .docview import local_id

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"
INDEX_NAME = "bm25.json"
TOKENIZER = "agent.claims.tokenize/2026-10-stopwords"   # bump when tokenize changes: every index is then rebuilt
K1 = 1.5
B = 0.75
EPSILON = 0.25
FIELDS = ("body", "path")
FORMAT = 2            # layout of bm25.json; 2 (2026-10-03): the path field has one entry per heading, not per chunk


def chunks_sha1(doc_id):
    path = PROC / doc_id / "chunks.jsonl"
    return hashlib.sha1(path.read_bytes()).hexdigest() if path.exists() else None


def postings_table(texts):
    """{n, doc_len, postings} over (local id, text) pairs."""
    postings, lengths = defaultdict(list), {}
    for lid, text in texts:
        toks = tokenize(text)
        lengths[lid] = len(toks)
        for term, tf in Counter(toks).items():
            postings[term].append([lid, tf])
    return {"n": len(lengths), "doc_len": lengths, "postings": dict(postings)}


def build_index(doc):
    """The index of one loaded document (agent.docview.Doc). body: postings and lengths over
    its chunks. path: postings and lengths over the headings that hold chunks, the path text
    of a heading indexed once, plus heading_of, the heading of every chunk."""
    from .retrieval import path_text                  # late import: retrieval imports this module
    paths = {}                                        # heading -> the path text its chunks share
    for u in doc.units:
        paths.setdefault(local_id(u["parent"]), path_text(doc, u))
    body = postings_table((local_id(u["id"]), u["body"]) for u in doc.units)
    path = postings_table(paths.items())
    path["heading_of"] = {local_id(u["id"]): local_id(u["parent"]) for u in doc.units}
    return {"doc_id": doc.doc_id, "chunks_sha1": chunks_sha1(doc.doc_id), "tokenizer": TOKENIZER, "format": FORMAT,
            "fields": {"body": body, "path": path}}


def save_index(index):
    path = PROC / index["doc_id"] / INDEX_NAME
    path.write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path


def doc_index(doc, persist=False):
    """The document's index from bm25.json when it matches the current chunks, tokenizer and
    layout, else built now (and written when persist=True)."""
    path = PROC / doc.doc_id / INDEX_NAME
    if path.exists():
        index = json.loads(path.read_text(encoding="utf-8"))
        if (index.get("chunks_sha1") == chunks_sha1(doc.doc_id) and index.get("tokenizer") == TOKENIZER
                and index.get("format") == FORMAT):
            return index
        print(f"[bm25] index of {doc.doc_id} is stale: rebuilding in memory, run script/build_index.py {doc.doc_id}", file=sys.stderr)
    else:
        print(f"[bm25] no {INDEX_NAME} for {doc.doc_id}: building in memory, run script/build_index.py {doc.doc_id}", file=sys.stderr)
    index = build_index(doc)
    if persist:
        save_index(index)
    return index


class MergedIndex:
    """The indexes of several documents merged: one postings table per field over full ids
    (chunks for body, headings for path), N and average length over all of them, and the
    heading of every chunk."""

    def __init__(self, docs):
        self.fields = {}
        self.heading_of = {doc.doc_id + "#" + c: doc.doc_id + "#" + h
                           for doc, index in docs for c, h in index["fields"]["path"]["heading_of"].items()}
        for field in FIELDS:
            postings, lengths = defaultdict(list), {}
            for doc, index in docs:
                f = index["fields"][field]
                prefix = doc.doc_id + "#"
                for lid, n in f["doc_len"].items():
                    lengths[prefix + lid] = n
                for term, plist in f["postings"].items():
                    postings[term].extend((prefix + lid, tf) for lid, tf in plist)
            n = len(lengths)
            self.fields[field] = {"n": n, "avgdl": (sum(lengths.values()) / n) if n else 0.0,
                                  "doc_len": lengths, "postings": postings, "idf": self._idf(postings, n, field)}

    @staticmethod
    def _idf(postings, n, field):
        """body: rank_bm25's BM25Okapi IDF, log((N - df + 0.5) / (df + 0.5)), negatives floored
        at EPSILON x mean. path: log(1 + (N - df + 0.5) / (df + 0.5)), which cannot be negative.
        The heading paths have a small vocabulary and a document's title is in every path of
        that document, so the mean IDF, and with it the floor, can be negative there: a claim
        naming a document then took points off that document's chunks (-0.30 on the gold
        clause of reg_a_008 A, 2026-10-03)."""
        if field == "path":
            return {term: math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5)) for term, plist in postings.items()}
        idf, negative = {}, []
        for term, plist in postings.items():
            df = len(plist)
            v = math.log(n - df + 0.5) - math.log(df + 0.5)
            idf[term] = v
            if v < 0:
                negative.append(term)
        if idf:
            floor = EPSILON * (sum(idf.values()) / len(idf))
            for term in negative:
                idf[term] = floor
        return idf

    def scores(self, query_tokens, field="body"):
        """{full chunk id: BM25 score}. body: over the chunks that contain at least one query
        token. path: the headings are scored and every chunk gets the score of its heading
        (chunks of a heading whose path holds no query token are left out). rank_bm25 counts
        a repeated query token once per occurrence: so do we."""
        f = self.fields[field]
        out = defaultdict(float)
        avgdl, k1, b = f["avgdl"], K1, B
        for q in query_tokens:
            idf = f["idf"].get(q)
            if idf is None:
                continue
            for cid, tf in f["postings"][q]:
                dl = f["doc_len"][cid]
                out[cid] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
        if field == "path":
            return {cid: out[hid] for cid, hid in self.heading_of.items() if hid in out}
        return out

    def chunk_ids(self, field="body"):
        return list(self.fields[field]["doc_len"])
