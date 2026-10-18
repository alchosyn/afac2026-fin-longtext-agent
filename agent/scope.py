# -*- coding: utf-8 -*-
"""Which chunks a B-list question is searched in, and the entity short names it uses.

A B question carries its domain but no doc_ids. Every chunk carries a header (parsing.header)
with its document's domain, title, entity and year, so the search scope is a filter on the
headers; as the header is the same for all chunks of a document, the filter is worked out
once per document on the cards (processed_data/doc_cards.json):

    domain     always
    names      when the question text (stem + options) names a document: its title, entity
               (company, product, regulation), a party or an entity short name of its own
               (processed_data/doc_terms.json), and for annual reports the year
    nothing named -> every chunk of the domain (no document is guessed from the content)

    scope(question)        -> {"docs": [...], "named": {doc_id: the string that named it}}
    short_names(text, ...) -> the rows 简称 = 全称 of the scope's documents that stand in the text
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"
QDIR = ROOT / "upload_b" / "upload_b" / "question_b"
SUFFIX = re.compile(r"(股份有限公司|有限责任公司|有限公司)$")
BRANDS = {"平安": ["平安"], "众安": ["众安"], "中国人寿": ["国寿", "中国人寿"], "太平洋": ["太保", "太平洋"]}
GENERIC_INS = ["重大疾病保险", "医疗保险", "意外伤害保险", "养老年金保险", "终身寿险", "养老保险", "责任保险", "商业保险", "保险",
               "团体", "个人", "专属", "互联网", "平安产险", "平安", "众安", "国寿", "太保", "（", "）"]
GENERIC_REG = ["中华人民共和国", "上市公司", "证券公司", "期货公司", "金融机构", "证券期货", "监督管理", "管理办法", "信息披露", "实施细则",
               "规定", "办法", "指引", "准则", "管理", "证券", "基金", "公司", "关于", "的", "《", "》", "—"]
TERM_DOMAINS = ("financial_contracts", "financial_reports")      # where an entity short name belongs to one document's subject
_CARDS = _TERMS = None


def cards():
    global _CARDS
    if _CARDS is None:
        _CARDS = json.loads((PROC / "doc_cards.json").read_text(encoding="utf-8"))["cards"]
    return _CARDS


def terms():
    """{domain: {short name: [(doc_id, row)]}} without the names three or more documents define."""
    global _TERMS
    if _TERMS is None:
        data = json.loads((PROC / "doc_terms.json").read_text(encoding="utf-8"))
        _TERMS = defaultdict(lambda: defaultdict(list))
        for d, rows in data["docs"].items():
            for r in rows:
                if not r["generic"]:
                    _TERMS[data["domain"][d]][r["term"]].append((d, r))
    return _TERMS


def load_b_questions():
    out = []
    for f in sorted(QDIR.glob("*")):
        text = f.read_text(encoding="utf-8-sig")
        out += [json.loads(l) for l in text.splitlines() if l.strip()] if f.suffix == ".jsonl" else json.loads(text)
    return out


def norm(s):
    return re.sub(r"\s+", "", s or "")


def short(name):
    return SUFFIX.sub("", norm(name))


def lcs(a, b):
    """Longest common substring of a and b."""
    best, prev = "", [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > len(best):
                    best = a[i - cur[j]:i]
        prev = cur
    return best


def distinct(s, generic):
    for g in generic:
        s = s.replace(g, "")
    return s


def question_text(q):
    return norm(q["question"] + " " + " ".join((q.get("options") or {}).values()))


def named(q, text=None):
    """{doc_id: the string of the question that named it}: the documents of the question's
    domain whose title, entity or year the question text names."""
    text = text or question_text(q)
    pool = [c for c in cards() if c["domain"] == q["domain"] and not c.get("duplicate_of")]
    hit = {}
    if q["domain"] == "financial_reports":
        groups = defaultdict(list)
        for c in pool:
            groups[c["company"]].append(c)
        for company, docs in groups.items():
            name = next((n for n in (docs[0]["stock_name"], short(company), norm(company)) if n and n in text), None)
            if name:
                hit.update({c["doc_id"]: name for c in docs})
    elif q["domain"] == "insurance":
        found = defaultdict(list)                                  # matched string -> cards
        for c in pool:
            m = lcs(text, norm(c["product"]))
            if m == norm(c["product"]) or len(distinct(m, GENERIC_INS)) >= 3:
                found[m].append(c)
        for m, group in found.items():
            if len(group) > 1:                                     # one product name at two insurers: the brand in front decides
                branded = []
                for c in group:
                    tokens = next((v for k, v in BRANDS.items() if k in c["company"]), [])
                    spots = [text[max(0, i.start() - 10):i.end()] for i in re.finditer(re.escape(m), text)]
                    if any(t in s for t in tokens for s in spots):
                        branded.append(c)
                group = branded or group
            hit.update({c["doc_id"]: m for c in group})
    elif q["domain"] == "financial_contracts":
        for c in pool:
            names = [norm(c["company"]), short(c["company"])] + [norm(a) for a in c.get("company_aliases", [])]
            for k in ("counterparty", "target_company", "guarantor"):
                if c.get(k) and c[k] != "无担保":
                    names += [short(x) for x in re.split(r"[、，,]", c[k]) if len(short(x)) >= 3]
            names.append(re.sub(r"^[\d年度]+", "", norm(c["title"])))
            name = next((n for n in names if n and len(n) >= 3 and n in text), None)
            if name:
                hit[c["doc_id"]] = name
    elif q["domain"] == "regulatory":
        for c in pool:
            for name in [c["name"]] + c.get("former_names", []) + [c["title"]]:
                core = norm(name).replace("中华人民共和国", "")
                m = core if core in text else lcs(text, core)
                if m == core or len(distinct(m, GENERIC_REG)) >= 5:
                    hit[c["doc_id"]] = m
                    break
    else:
        for c in pool:
            if c.get("subject") and c["subject"] in text:
                hit[c["doc_id"]] = c["subject"]
    if q["domain"] in TERM_DOMAINS:                                # an entity short name the document itself defines
        ids = {c["doc_id"] for c in pool}
        for term, owners in terms()[q["domain"]].items():
            if len(term) >= 3 and term in text:
                for d, _ in owners:
                    if d in ids:
                        hit.setdefault(d, term)
    if q["domain"] == "financial_reports" and hit:                 # the years the text mentions, per company
        years = set(re.findall(r"20\d{2}", text))
        by_id = {c["doc_id"]: c for c in pool}
        groups = defaultdict(list)
        for d in hit:
            groups[by_id[d]["company"]].append(d)
        keep = set()
        for docs in groups.values():
            keep |= set([d for d in docs if str(by_id[d]["year"]) in years] or docs)
        hit = {d: f"{n} {by_id[d]['year']}" for d, n in hit.items() if d in keep}
    return hit


def scope(q):
    """The documents whose chunks the question is searched in: the named ones, else the whole domain."""
    hit = named(q)
    domain = [c["doc_id"] for c in cards() if c["domain"] == q["domain"] and not c.get("duplicate_of")]
    return {"docs": [d for d in domain if d in hit] or domain, "named": hit}


def short_names(text, domain, docs=None, limit=8):
    """Rows {"term", "full", "doc_id"}: the entity short names standing in `text`, the longest
    first, of the documents `docs` (None: of every document of the domain)."""
    text = norm(text)
    hits = [(d, r) for term, owners in terms()[domain].items() if term in text for d, r in owners if docs is None or d in docs]
    hits.sort(key=lambda x: -len(x[1]["term"]))
    out, seen = [], set()
    for d, r in hits:
        if (r["term"], r["full"]) not in seen:
            seen.add((r["term"], r["full"]))
            out.append({"term": r["term"], "full": r["full"], "doc_id": d})
    return out[:limit]
