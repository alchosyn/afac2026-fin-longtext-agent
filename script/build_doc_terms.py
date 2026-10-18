# -*- coding: utf-8 -*-
"""Entity short names of the documents: processed_data/doc_terms.json.

Per document, the short names its own text gives to companies and other organisations, read
by rules (no model call):

    释义表      the rows  | 简称、简称 | 指 | 全称 |  of a contract (from doc_cards.json)
    以下简称    全称（以下简称“简称”）

A row is kept only when it names an entity: the full form is the name of an organisation
(…公司, …银行, …集团, …事务所, …委员会 …) and the short name is made of its characters (陕国投
/ 陕西省国际信托…, not a role word such as 实际控制人). Concept definitions (X是指…) and
short names of other things (本次可转债, 转股价格) are not collected: on the B questions only
the entity names helped routing and rewriting (2026-10-04).

Every row: {"term", "full", "source", "chunk", "generic"}. `generic` marks a short name that
three or more documents of the domain define (中国证监会, 深交所): it does not tell documents
apart and is left out of the reverse table (term -> documents) that lookup() reads.

Usage:
    python script/build_doc_terms.py            # every document -> processed_data/doc_terms.json
    python script/build_doc_terms.py fc_text12  # print the table of some documents
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"
OUT = PROC / "doc_terms.json"
ENTITY = re.compile(r"^[一-龥（）()A-Za-z·\d]{2,40}?(?:股份有限公司|有限责任公司|有限公司|（有限合伙）|合伙企业|事务所|银行|集团|委员会|交易所|协会|基金会|研究院)$")
ABBR = re.compile(r"[（(](?:以下|下文|下称)?(?:合并|统一|分别)?(?:简称|称为|称)[:：]?\s*[“\"「]?([^”\"」）)，。、；《》]{2,16})[”\"」]?[^）)]{0,10}[）)]")
ROLES = {"公司", "本公司", "发行人", "发行主体", "上市公司", "本集团", "集团", "本行", "本企业", "该公司", "该机构", "我公司", "交易对方", "标的公司",
         "担保人", "保证人", "控股股东", "实际控制人", "主承销商", "受托管理人", "债券受托管理人", "评级机构", "资信评级机构", "保荐机构", "保荐人",
         "独立财务顾问", "律师", "发行人律师", "会计师", "发行人会计师", "审计机构", "评估机构", "资产评估机构"}


def entity_name(text):
    """The organisation name a full form opens with, or None: the part before the first comma or bracketed remark."""
    head = re.split(r"[，,；;]|（(?:曾用名|原名|以下)", text.strip())[0].strip()
    return head if ENTITY.match(head) else None


def full_before(before, short):
    """The organisation name that stands right before （以下简称 short）: it starts where the
    short name's first character stands and spells the short name in order (美的集团 in 我们审计了
    美的集团股份有限公司), else at the first character the two share (深圳爱佩仪 / 爱佩仪光电技术（深圳）有限公司)."""
    tail = re.split(r"[，。；：、“”\n|]", before.rstrip(" 　"))[-1][-40:]
    for start in range(len(tail) - 1, -1, -1):
        if tail[start] == short[0]:
            it = iter(tail[start:])
            if all(ch in it for ch in short) and ENTITY.match(tail[start:]):
                return tail[start:]
    start = next((i for i, ch in enumerate(tail) if ch in short), None)
    if start is not None and ENTITY.match(tail[start:]):
        return tail[start:]
    return None


def rows_of(doc_id, card):
    rows, seen = [], set()

    def add(term, full, source, chunk=None):
        term = term.strip(" 　“”\"")
        full = full and re.sub(r"^(?:子公司|孙公司|全资子公司|控股子公司|参股公司|公司|与|和|及|由|对|向|为)+(?=[一-龥])", "", full)
        if (full and ENTITY.match(full) and 2 <= len(term) <= 16 and term != full and term not in ROLES
                and len(set(term) & set(full)) >= 2 and term not in seen):
            seen.add(term)
            rows.append({"term": term, "full": full, "source": source, "chunk": chunk})

    for d in card.get("definitions", []):
        for t in d["terms"]:
            add(t, entity_name(d["meaning"]), "释义表")
    path = PROC / doc_id / "chunks.jsonl"
    for line in path.open(encoding="utf-8") if path.exists() else []:
        c = json.loads(line)
        for m in ABBR.finditer(c["text"]):
            add(m.group(1), full_before(c["text"][max(0, m.start() - 60):m.start()], m.group(1)), "以下简称", c["id"])
    return rows


def build():
    cards = json.loads((PROC / "doc_cards.json").read_text(encoding="utf-8"))["cards"]
    docs = {c["doc_id"]: rows_of(c["doc_id"], c) for c in cards}
    domain = {c["doc_id"]: c["domain"] for c in cards}
    df = defaultdict(Counter)                                      # domain -> short name -> documents defining it
    for d, rows in docs.items():
        for r in rows:
            df[domain[d]][r["term"]] += 1
    for d, rows in docs.items():
        for r in rows:
            r["generic"] = df[domain[d]][r["term"]] >= 3
    return cards, docs


def load():
    """(tables {doc_id: rows}, reverse {domain: {short name: [(doc_id, row)]}}), the generic names left out."""
    data = json.loads(OUT.read_text(encoding="utf-8"))
    reverse = defaultdict(lambda: defaultdict(list))
    for d, rows in data["docs"].items():
        for r in rows:
            if not r["generic"]:
                reverse[data["domain"][d]][r["term"]].append((d, r))
    return data["docs"], reverse


def lookup(text, domain, reverse, docs=None, limit=8):
    """Rows whose short name stands in `text`, the longest names first; `docs` restricts them to these documents."""
    text = re.sub(r"\s+", "", text)
    hits = [(d, r) for term, owners in reverse[domain].items() if term in text for d, r in owners if docs is None or d in docs]
    hits.sort(key=lambda x: -len(x[1]["term"]))
    out, seen = [], set()
    for d, r in hits:
        if (r["term"], r["full"]) not in seen:
            seen.add((r["term"], r["full"]))
            out.append({"doc_id": d, **r})
    return out[:limit]


def main(argv):
    cards, docs = build()
    if argv:
        for d in argv:
            for r in docs[d]:
                print(f"{'(generic) ' if r['generic'] else ''}{r['term']} = {r['full']}  [{r['source']}]")
        return
    domain = {c["doc_id"]: c["domain"] for c in cards}
    OUT.write_text(json.dumps({"meta": {"source": "rules (script/build_doc_terms.py): entity short names from 释义表 and 以下简称; no model call"},
                               "domain": domain, "docs": docs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{sum(map(len, docs.values()))} entity short names in {sum(bool(v) for v in docs.values())} of {len(docs)} documents -> {OUT.relative_to(ROOT)}")
    for dom in dict.fromkeys(domain.values()):
        rows = [r for d, v in docs.items() if domain[d] == dom for r in v]
        by = Counter(r["source"] for r in rows)
        print(f"  {dom}: {len(rows)} rows ({by['释义表']} 释义表, {by['以下简称']} 以下简称), {sum(r['generic'] for r in rows)} generic")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
