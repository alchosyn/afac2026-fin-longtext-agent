# -*- coding: utf-8 -*-
"""Unit-level retrieval over processed_data/<doc_id>/elements.jsonl: BM25, bge-m3 dense
vectors, and a BM25 + vector hybrid fused by reciprocal rank.

Every non-heading record of a document (a clause 第N条 in regulatory texts, a paragraph /
table / figure elsewhere) is one retrieval unit. BM25 and the encoder see the same text:
a short header (document title | notice | chapter path) followed by the unit's original
text, which is never rewritten.

Vectors are computed once by experiments/archive/vectors/build_vectors_bge_m3.py and stored next to the
document as vectors_bge_m3.npz (unit ids + float32 L2-normalised vectors) plus
vectors_bge_m3.json (model, dimension, header flag, a sha1 per unit text).

    from agent.retrieval import Encoder, search
    enc = Encoder()                                   # loads D:/models/bge-m3 on CPU
    res = search("hybrid", "撤并分支机构应当提前7日报告", doc_ids, encoder=enc, top_k=5)
    res["merged"]      -> [(unit_id, score), ...] over all documents
    res["per_doc"]     -> {doc_id: [(unit_id, score), ...]}
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .bm25 import MergedIndex, doc_index
from .claims import tokenize
from .docview import Doc, load_cards

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"
MODEL_NAME = "BAAI/bge-m3"
MODEL_PATH = Path(r"D:\models\bge-m3")
VEC_STEM = "vectors_bge_m3"
PASSAGE_MAX_LENGTH = 1024
QUERY_MAX_LENGTH = 512
RRF_K = 60

_DOCS = {}
_CARDS = None
_BM25 = {}


def get_doc(doc_id):
    global _CARDS
    if doc_id not in _DOCS:
        if _CARDS is None:
            _CARDS = load_cards()
        _DOCS[doc_id] = Doc(doc_id, _CARDS[doc_id])
    return _DOCS[doc_id]


# ----------------------------------------------------------------------------- unit text
def doc_header(doc):
    """document title | notice (the decree / announcement number, without the repeated title)."""
    title = doc.card.get("title") or doc.doc_id
    notice = (doc.card.get("notice") or "").replace(f"《{title}》", "")
    notice = re.sub(r"[【】]", "", notice).strip()
    return "｜".join(x for x in (title, notice) if x)


def unit_text(doc, unit, header=True):
    """Text BM25's header field sees: header line + the chunk's body (caption, text, source)."""
    if not header:
        return unit["body"]
    head = doc_header(doc)
    if unit.get("path"):
        head += "｜" + unit["path"]
    return head + "\n" + unit["body"]


def lean_table(md):
    """A table's markdown for the encoder: separator line dropped, pipes removed, cells
    joined by one space. Same rows and cells, 37 % fewer tokens (measured on the 2,175
    tables, 2026-10-01), so a TABLE_CAP piece stays near 512 tokens."""
    out = []
    for line in md.split("\n"):
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
                continue
            out.append(" ".join(c for c in cells if c))
        else:
            out.append(line)
    return "\n".join(out)


def vector_text(doc, unit, header=True):
    """Text the dense encoder sees: as unit_text, with a table's markdown made lean."""
    if unit["type"] != "table":
        return unit_text(doc, unit, header)
    body = "\n".join(x for x in (unit.get("caption"), lean_table(unit["text"]), unit.get("source")) if x)
    if not header:
        return body
    head = doc_header(doc)
    if unit.get("path"):
        head += "｜" + unit["path"]
    return head + "\n" + body


def text_sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- encoder
class Encoder:
    """bge-m3 dense encoder: FlagEmbedding's BGEM3FlagModel, falling back to
    sentence-transformers (same CLS pooling and normalisation) if FlagEmbedding is absent.
    device="cuda" uses plain transformers in fp16 with the same CLS pooling (the CUDA venv
    has no FlagEmbedding); texts are sorted by length and batched by a token budget."""

    def __init__(self, model_path=MODEL_PATH, device="cpu"):
        self.model_path = str(model_path)
        self.device = device
        if device.startswith("cuda"):
            import torch
            from transformers import AutoModel, AutoTokenizer
            self.backend = "transformers-fp16"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path, dtype=torch.float16).to(device).eval()
            return
        try:
            from FlagEmbedding import BGEM3FlagModel
            self.backend = "FlagEmbedding"
            self.model = BGEM3FlagModel(self.model_path, use_fp16=False, devices=device,
                                        passage_max_length=PASSAGE_MAX_LENGTH, query_max_length=QUERY_MAX_LENGTH)
        except ImportError:
            from sentence_transformers import SentenceTransformer
            self.backend = "sentence-transformers"
            self.model = SentenceTransformer(self.model_path, device=device)

    def encode(self, texts, max_length=PASSAGE_MAX_LENGTH, batch_size=16):
        if isinstance(texts, str):
            texts = [texts]
        if self.backend == "transformers-fp16":
            vecs = self._encode_torch(texts, max_length)
        elif self.backend == "FlagEmbedding":
            out = self.model.encode(texts, batch_size=batch_size, max_length=max_length,
                                    return_dense=True, return_sparse=False, return_colbert_vecs=False)
            vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
        else:
            vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                     convert_to_numpy=True).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)

    def _encode_torch(self, texts, max_length, token_budget=16384):
        import torch
        lengths = [len(x) for x in self.tokenizer(texts, truncation=True, max_length=max_length)["input_ids"]]
        order = sorted(range(len(texts)), key=lambda k: -lengths[k])
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        start = 0
        while start < len(order):
            size = max(1, min(64, token_budget // lengths[order[start]]))
            batch = order[start:start + size]
            enc = self.tokenizer([texts[k] for k in batch], truncation=True, max_length=max_length,
                                 padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                cls = self.model(**enc).last_hidden_state[:, 0].float()
            out[batch] = torch.nn.functional.normalize(cls, dim=-1).cpu().numpy()
            start += size
        return out

    def encode_query(self, query):
        return self.encode([query], max_length=QUERY_MAX_LENGTH)[0]


# ----------------------------------------------------------------------------- vector store
def vector_paths(doc_id):
    return PROC / doc_id / f"{VEC_STEM}.npz", PROC / doc_id / f"{VEC_STEM}.json"


def build_vectors(doc_id, encoder, header=True, force=False):
    """Encode the units of one document and write the .npz / .json pair, incrementally: a
    unit whose encoded text has the sha1 the existing pair recorded keeps its vector, only
    new or changed units are encoded (force=True encodes everything). Returns the meta,
    with "encoded" and "reused" counts."""
    doc = get_doc(doc_id)
    units = doc.units
    texts = [vector_text(doc, u, header) for u in units]
    shas = [text_sha1(t) for t in texts]
    npz, meta_path = vector_paths(doc_id)
    old = {}
    if not force and npz.exists() and meta_path.exists():
        meta0 = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta0.get("model") == MODEL_NAME and meta0.get("header") == header and meta0.get("passage_max_length") == PASSAGE_MAX_LENGTH:
            data = np.load(npz, allow_pickle=False)
            rows = {str(i): k for k, i in enumerate(data["ids"])}
            old = {x["id"]: data["vectors"][rows[x["id"]]] for x in meta0["units"] if x["id"] in rows}
            old = {x["id"]: (x["sha1"], old[x["id"]]) for x in meta0["units"] if x["id"] in old}
    todo = [k for k, u in enumerate(units) if old.get(u["id"], (None,))[0] != shas[k]]
    dim = next((v.shape[0] for _, v in old.values()), None)
    fresh = encoder.encode([texts[k] for k in todo]) if todo else np.zeros((0, dim or 1024), dtype=np.float32)
    vecs = np.zeros((len(units), fresh.shape[1]), dtype=np.float32)
    todo_set = set(todo)
    for pos, k in enumerate(todo):
        vecs[k] = fresh[pos]
    for k, u in enumerate(units):
        if k not in todo_set:
            vecs[k] = old[u["id"]][1]
    ids = np.array([u["id"] for u in units])
    np.savez(npz, ids=ids, vectors=vecs)
    meta = {"doc_id": doc_id, "model": MODEL_NAME, "model_path": encoder.model_path, "backend": encoder.backend,
            "dim": int(vecs.shape[1]), "count": int(vecs.shape[0]), "header": header,
            "passage_max_length": PASSAGE_MAX_LENGTH, "encoded": len(todo), "reused": len(units) - len(todo),
            "units": [{"id": u["id"], "sha1": s} for u, s in zip(units, shas)]}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def load_vectors(doc_id, check=True):
    """(ids, vectors, meta) for one document; raises if the stored texts no longer match."""
    npz, meta_path = vector_paths(doc_id)
    if not npz.exists():
        raise FileNotFoundError(f"no vectors for {doc_id}: run experiments/archive/vectors/build_vectors_bge_m3.py")
    data = np.load(npz, allow_pickle=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ids = [str(x) for x in data["ids"]]
    if check:
        doc = get_doc(doc_id)
        current = {u["id"]: text_sha1(vector_text(doc, u, meta["header"])) for u in doc.units}
        stored = {x["id"]: x["sha1"] for x in meta["units"]}
        if current != stored:
            raise ValueError(f"vectors of {doc_id} are stale (elements changed): rebuild them")
    return ids, data["vectors"], meta


# ----------------------------------------------------------------------------- scoring
def vector_scores(query, doc_ids, encoder):
    """[(unit_id, cosine)] over all units of doc_ids, best first."""
    q = encoder.encode_query(query)
    out = []
    for d in doc_ids:
        ids, vecs, _ = load_vectors(d)
        sims = vecs @ q
        out.extend(zip(ids, (float(s) for s in sims)))
    return sorted(out, key=lambda x: -x[1])


BODY_WEIGHT = 0.7      # weight of the unit's own text in the two-field BM25 score
PATH_WEIGHT = 0.3      # weight of its context field: document title | notice | heading path


def path_text(doc, unit):
    """The unit's context field: document title | notice | heading path."""
    head = doc_header(doc)
    return head + ("｜" + unit["path"] if unit.get("path") else "")


def bm25_index(doc_ids):
    """The merged inverted index (agent.bm25.MergedIndex) over the chunks of doc_ids, both
    fields; each document's index comes from its bm25.json (built in memory when stale).
    Built once per document set; IDF is computed over exactly these documents."""
    key = tuple(doc_ids)
    if key not in _BM25:
        docs = [get_doc(d) for d in doc_ids]
        _BM25[key] = MergedIndex([(doc, doc_index(doc)) for doc in docs])
    return _BM25[key]


def bm25_scores(query, doc_ids, header=True, body_weight=BODY_WEIGHT, path_weight=PATH_WEIGHT):
    """[(unit_id, score)] best first over the chunks of all doc_ids, every chunk listed
    (zero for those sharing no token with the query), in document order among equals.
    header=True (default): two-field score  body * 0.7 + heading path * 0.3  on the raw
    BM25 scores of each field. Raw, not normalised: normalising each field to its best
    score hands every unit of a document whose title matches the query the same bonus and
    flattens the ranking (gold rank 1 -> 56 on reg_a_008 A in the check of 2026-09-21);
    the raw sum keeps the body ranking and lets the path break ties and lift units whose
    own text omits the subject their heading states. The path score is computed once per
    heading and shared by the chunks under it (agent.bm25, 2026-10-03).
    header=False: the body field alone."""
    q = tokenize(query)
    idx = bm25_index(doc_ids)
    ids = idx.chunk_ids("body")
    b = idx.scores(q, "body")
    if not header:
        return sorted(((i, b.get(i, 0.0)) for i in ids), key=lambda x: -x[1])
    p = idx.scores(q, "path")
    return sorted(((i, body_weight * b.get(i, 0.0) + path_weight * p.get(i, 0.0)) for i in ids), key=lambda x: -x[1])


def rrf_fuse(rankings, k=RRF_K):
    """Reciprocal rank fusion of several best-first [(unit_id, score)] lists."""
    fused = {}
    for ranking in rankings:
        for rank, (uid, _) in enumerate(ranking, 1):
            fused[uid] = fused.get(uid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


def hybrid_scores(query, doc_ids, encoder, k=RRF_K):
    return rrf_fuse([bm25_scores(query, doc_ids), vector_scores(query, doc_ids, encoder)], k=k)


def search(method, query, doc_ids, encoder=None, top_k=5):
    """method: 'bm25' | 'vector' | 'hybrid'. Returns the full ranking plus top_k merged and per document."""
    if method == "bm25":
        ranking = bm25_scores(query, doc_ids)
    elif method == "vector":
        ranking = vector_scores(query, doc_ids, encoder)
    elif method == "hybrid":
        ranking = hybrid_scores(query, doc_ids, encoder)
    else:
        raise ValueError(method)
    per_doc = {d: [x for x in ranking if x[0].startswith(d + "#")][:top_k] for d in doc_ids}
    return {"ranking": ranking, "merged": ranking[:top_k], "per_doc": per_doc}


def rank_of(ranking, unit_id, doc_id=None):
    """1-based rank of unit_id in a best-first ranking, optionally restricted to one document."""
    pos = 0
    for uid, _ in ranking:
        if doc_id is not None and not uid.startswith(doc_id + "#"):
            continue
        pos += 1
        if uid == unit_id:
            return pos
    return None
