# -*- coding: utf-8 -*-
"""Probe: route B-list questions (no doc_ids) to documents with processed_data/doc_cards.json.

Two routes, no model call:
    names    the question text (stem + options) names a card of its domain
               reports     company / stock name, then the years the text mentions
               insurance   a distinctive part of the product name; the insurer's brand word in
                           front of it decides between insurers that share a product name
               contracts   company, its short names, the parties, an entity defined in one
                           document's 释义 table only, the title
               regulatory  the regulation's name (current or former), or a long distinctive
                           part of it
               research    the company a report is about
    content  BM25 over every chunk of the domain, one query per option (stem + option): the
             documents of the top chunks, scored by reciprocal rank
The answer is the names route when it finds anything, else the content route; research
questions take both (their stems seldom name a report).

Usage:
    python tools/probe_route.py                       # all 100 questions, one line each
    python tools/probe_route.py fc_b_013 res_b_006    # some questions, with the matched names
    python tools/probe_route.py --gold gold.json ...  # {qid: [doc ids]}: also compare
"""
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.retrieval import bm25_scores   # noqa: E402

CARDS = json.loads((ROOT / "processed_data" / "doc_cards.json").read_text(encoding="utf-8"))["cards"]
QDIR = ROOT / "upload_b" / "upload_b" / "question_b"
SUFFIX = re.compile(r"(股份有限公司|有限责任公司|有限公司)$")
BRANDS = {"平安": ["平安"], "众安": ["众安"], "中国人寿": ["国寿", "中国人寿"], "太平洋": ["太保", "太平洋"]}
GENERIC_INS = ["重大疾病保险", "医疗保险", "意外伤害保险", "养老年金保险", "终身寿险", "养老保险", "责任保险", "商业保险", "保险",
               "团体", "个人", "专属", "互联网", "平安产险", "平安", "众安", "国寿", "太保", "（", "）"]
GENERIC_REG = ["中华人民共和国", "上市公司", "证券公司", "期货公司", "金融机构", "证券期货", "监督管理", "管理办法", "信息披露", "实施细则",
               "规定", "办法", "指引", "准则", "管理", "证券", "基金", "公司", "关于", "的", "《", "》", "—"]
ROLE_TERMS = {"交易对方", "标的公司", "标的资产", "交易标的", "上市公司", "发行人", "发行主体", "本公司", "担保人", "保证人", "主承销商",
              "受托管理人", "实际控制人", "控股股东", "评级机构", "独立财务顾问"}
CONTENT_TOP, CONTENT_SHARE, CONTENT_MAX = 10, 0.05, 3      # top chunks per query; a document needs this share of the best score; documents kept


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


def load_questions():
    out = []
    for f in sorted(QDIR.glob("*")):
        text = f.read_text(encoding="utf-8-sig")
        out += [json.loads(l) for l in text.splitlines() if l.strip()] if f.suffix == ".jsonl" else json.loads(text)
    return out


# ----------------------------------------------------------------------------- names
def contract_terms(cards):
    """Entity short names defined in one contract only: {term: doc_id}."""
    owners = defaultdict(set)
    for c in cards:
        for d in c.get("definitions", []):
            full = re.search(r"[一-龥（）()]{2,30}?(?:公司|银行|集团|事务所)", d["meaning"])
            if full:
                for t in d["terms"]:                               # a short name shares characters with the full name; a role word does not
                    if len(t) >= 3 and t not in ROLE_TERMS and len(set(t) & set(full.group(0))) >= 2:
                        owners[norm(t)].add(c["doc_id"])
    return {t: next(iter(ds)) for t, ds in owners.items() if len(ds) == 1}


def by_names(q, text):
    """{doc_id: the string of the question that named it}."""
    cards = [c for c in CARDS if c["domain"] == q["domain"] and not c.get("duplicate_of")]
    hit = {}
    if q["domain"] == "financial_reports":
        years = set(re.findall(r"20\d{2}", text))
        groups = defaultdict(list)
        for c in cards:
            groups[c["company"]].append(c)
        for company, docs in groups.items():
            name = next((n for n in (docs[0]["stock_name"], short(company), norm(company)) if n and n in text), None)
            if name:
                picked = [c for c in docs if str(c["year"]) in years] or docs
                hit.update({c["doc_id"]: f"{name} {c['year']}" for c in picked})
    elif q["domain"] == "insurance":
        found = defaultdict(list)                                  # matched string -> cards
        for c in cards:
            m = lcs(text, norm(c["product"]))
            if m == norm(c["product"]) or len(distinct(m, GENERIC_INS)) >= 3:
                found[m].append(c)
        for m, group in found.items():
            if len(group) > 1:                                     # same product name at two insurers: the brand in front decides
                branded = []
                for c in group:
                    tokens = next((v for k, v in BRANDS.items() if k in c["company"]), [])
                    spots = [text[max(0, i.start() - 10):i.end()] for i in re.finditer(re.escape(m), text)]
                    if any(t in s for t in tokens for s in spots):
                        branded.append(c)
                group = branded or group
            hit.update({c["doc_id"]: m for c in group})
    elif q["domain"] == "financial_contracts":
        unique = contract_terms(cards)
        for c in cards:
            names = [norm(c["company"]), short(c["company"])] + [norm(a) for a in c.get("company_aliases", [])]
            for k in ("counterparty", "target_company", "guarantor"):
                if c.get(k) and c[k] != "无担保":
                    names += [short(x) for x in re.split(r"[、，,]", c[k]) if len(short(x)) >= 3]
            names.append(re.sub(r"^[\d年度]+", "", norm(c["title"])))
            names += [t for t, d in unique.items() if d == c["doc_id"]]
            name = next((n for n in names if n and len(n) >= 3 and n in text), None)
            if name:
                hit[c["doc_id"]] = name
    elif q["domain"] == "regulatory":
        for c in cards:
            for name in [c["name"]] + c.get("former_names", []) + [c["title"]]:
                core = norm(name).replace("中华人民共和国", "")
                m = core if core in text else lcs(text, core)
                if m == core or len(distinct(m, GENERIC_REG)) >= 5:
                    hit[c["doc_id"]] = m
                    break
    else:
        for c in cards:
            if c.get("subject") and c["subject"] in text:
                hit[c["doc_id"]] = c["subject"]
    return hit


# ----------------------------------------------------------------------------- content
def by_content(q):
    """{doc_id: score}: documents of the top chunks, the whole domain searched."""
    docs = [c["doc_id"] for c in CARDS if c["domain"] == q["domain"] and not c.get("duplicate_of")]
    stem = q["question"].strip()
    queries = [stem + " " + v for v in (q.get("options") or {}).values()] or [stem]
    score = Counter()
    for query in queries:
        for rank, (uid, s) in enumerate(bm25_scores(query, docs)[:CONTENT_TOP], 1):
            if s > 0:
                score[uid.split("#")[0]] += 1.0 / rank
    best = max(score.values(), default=0)
    return {d: round(s, 2) for d, s in score.most_common(CONTENT_MAX) if s >= CONTENT_SHARE * best}


def route(q):
    text = norm(q["question"] + " " + " ".join((q.get("options") or {}).values()))
    names = by_names(q, text)
    need_content = not names or q["domain"] == "research"
    content = by_content(q) if need_content else {}
    return {"names": names, "content": content, "docs": list(dict.fromkeys(list(names) + list(content)))}


def main(argv):
    gold = {}
    if "--gold" in argv:
        gold = json.loads(Path(argv[argv.index("--gold") + 1]).read_text(encoding="utf-8"))
        argv = [a for k, a in enumerate(argv) if a != "--gold" and argv[k - 1] != "--gold"]
    questions = [q for q in load_questions() if not argv or q["qid"] in argv]
    titles = {c["doc_id"]: c["title"] for c in CARDS}
    tally = Counter()
    for q in questions:
        r = route(q)
        how = "names" if r["names"] and not r["content"] else "names+content" if r["names"] else "content"
        line = f"{q['qid']:10s} {q['type']} {how:13s} {len(r['docs'])} docs"
        if q["qid"] in gold:
            want = set(gold[q["qid"]])
            missing, extra = want - set(r["docs"]), set(r["docs"]) - want
            tally["all found"] += not missing
            tally["exact"] += not missing and not extra
            line += f" | gold {len(want)}: missing {len(missing)}, extra {len(extra)}"
        print(line)
        if argv:
            for d in r["docs"]:
                via = f"named by “{r['names'][d]}”" if d in r["names"] else f"content {r['content'][d]}"
                mark = ("" if not gold.get(q["qid"]) else " ✓" if d in gold[q["qid"]] else " ✗")
                print(f"      {d[:24]:24s} {titles[d][:30]:30s} {via}{mark}")
            for d in set(gold.get(q["qid"], [])) - set(r["docs"]):
                print(f"      MISSING {d[:24]} {titles[d][:30]}")
    if gold:
        n = sum(1 for q in questions if q["qid"] in gold)
        print(f"\n{n} questions with gold: every gold document routed {tally['all found']}, exactly the gold set {tally['exact']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
