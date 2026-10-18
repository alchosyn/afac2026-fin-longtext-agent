# -*- coding: utf-8 -*-
"""Document cards for routing a question to its documents: processed_data/doc_cards.json.

One card per document of processed_data/docs.jsonl, filled by rules from the title, the
existing card and the first pages of the document (regular expressions and table cells; no
model call, every value is text of the document or of its card):

    all          doc_id, domain, title, doc_type, company, year, date
    regulatory   kind, name (the regulation's current name), former_names, amends, issuer,
                 notice, notice_no, effective_date, repeals, companions
    reports      stock_code, stock_name
    contracts    company_aliases, stock_code, issuer, guarantor (bonds), counterparty,
                 target_company, target (reorganisations), definitions (the 释义 table)
    insurance    product, product_type, version
    research     broker, report_type, subject, stock_code, abstract (the first-page summary,
                 verbatim), toc (top-level entries of the bookmarks / printed contents)

`warnings` names the fields a rule could not fill or whose value failed its check: those
cards are the ones to review by hand.

Usage:
    python script/build_doc_cards.py                 # every document -> processed_data/doc_cards.json
    python script/build_doc_cards.py fc_text12 ...   # print the cards of these documents, write nothing
"""
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.docview import Doc, load_cards   # noqa: E402

PROC = ROOT / "processed_data"
OUT = PROC / "doc_cards.json"
COMPANY = r"[一-龥（）()A-Za-z]{2,30}?(?:股份有限公司|有限责任公司|有限公司)"


# ----------------------------------------------------------------------------- text helpers
def first_pages(doc, pages):
    """Headings and chunks of the first `pages` pages, one text."""
    heads = [h["text"] for h in doc.headings if h.get("page", 1) <= pages]
    return "\n".join(heads + [u["body"] for u in doc.units if u["page"] <= pages])


def find(pattern, text, group=1):
    m = re.search(pattern, text)
    return m.group(group).strip() if m else None


def table_rows(text):
    """Cells of the markdown table rows of a text, separator rows left out."""
    rows = []
    for line in text.split("\n"):
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells if c):
                rows.append(cells)
    return rows


def labelled(doc, pages, labels, value, check):
    """The value that belongs to a label: the text right after the label, else the cell
    under it in the next table row (header row above, values below)."""
    text = first_pages(doc, pages)
    for label in labels:
        for m in re.finditer(label + r"[^0-9A-Za-z一-龥\n]{0,14}(" + value + ")", text):
            if check(m.group(1)):
                return m.group(1)
    for u in doc.units:
        if u["page"] > pages:
            break
        rows = table_rows(u["body"])
        for head, below in zip(rows, rows[1:]):
            for label in labels:                                   # a header cell may carry the table's caption before the label
                k = next((i for i, cell in enumerate(head) if cell.endswith(label)), None)
                if k is not None and k < len(below) and check(below[k]):
                    return below[k]
    return None


CN_DIGIT = {c: i for i, c in enumerate("〇一二三四五六七八九")}
CN_DIGIT.update({"○": 0, "零": 0, "O": 0, "0": 0})


def cn_int(s):
    """十二 -> 12, 二十 -> 20, 8 -> 8."""
    s = s.strip()
    if s.isdigit():
        return int(s)
    if "十" in s:
        a, b = s.split("十")
        return (CN_DIGIT.get(a, 1) if a else 1) * 10 + (CN_DIGIT.get(b, 0) if b else 0)
    return CN_DIGIT[s]


def year_month(s):
    """二〇一八年十二月 / 2026年 4 月 -> 2018-12 / 2026-04."""
    m = re.search(r"([二2][〇○零O0\d][一二三四五六七八九〇○零\d]{2})\s*年\s*([一二三四五六七八九十\d]{1,2})\s*月", s or "")
    if not m:
        return None
    y = m.group(1) if m.group(1).isdigit() else "".join(str(CN_DIGIT[c]) for c in m.group(1))
    return f"{y}-{cn_int(m.group(2)):02d}"


def iso_date(s):
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", s or "")
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None


# ----------------------------------------------------------------------------- regulatory
KIND_LABEL = {"law": "法律", "regulation": "规章及规范性文件", "decision": "修改决定", "explanation": "说明",
              "standard": "行业标准", "other": "其他", "scanned": "扫描件"}
EFFECTIVE = re.compile(r"自\s*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|公布之日|发布之日|印发之日)\s*起\s*(?:施行|实施|执行)")
OWN_EFFECTIVE = re.compile(r"本(办法|规定|准则|指引|细则|规则|决定|法|条例|通知|意见|适用意见|标准|解释|制度|公告)\s*自\s*"
                           r"(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|公布之日|发布之日|印发之日)\s*起\s*(?:施行|实施|执行)")


def regulatory(card, doc, cards):
    title, kind = card["title"], card.get("kind")
    out = {"doc_type": KIND_LABEL.get(kind, kind), "company": None, "kind": kind, "name": title, "former_names": [],
           "amends": None, "issuer": card.get("issuer") or None, "notice": card.get("notice") or None,
           "notice_no": find(r"(第\s*\d+\s*号(?:令|公告)?|〔\d{4}〕\s*第?\s*\d+\s*号)", card.get("notice") or "", 1),
           "date": card.get("date") or None, "effective_date": None, "repeals": [], "companions": []}
    out["year"] = int(out["date"][:4]) if out["date"] else None
    texts = [u["body"] for u in doc.units]
    amended = find(r"关于修改[《〈](.+?)[》〉]的决定", title)
    if amended:
        out["amends"] = out["name"] = amended
        renamed = find(r"将[《〈]" + re.escape(amended) + r"[》〉]更名为[《〈](.+?)[》〉]", "\n".join(texts[:3]))
        if renamed:
            out["name"], out["former_names"] = renamed, [amended]
    elif kind == "explanation":
        out["name"] = find(r"[《〈](.+?)[》〉](?!的决定)", title) or title
    if kind != "explanation":
        # the article "本办法自 … 起施行" may stand before the annexes; a decision takes effect by "本决定自 …"
        own = [(i, m) for i, t in enumerate(texts) for m in OWN_EFFECTIVE.finditer(t)]
        if len({i for i, _ in own}) >= 3:                          # several regulations in one document: no single date
            out["compilation"] = True
        else:
            pick = next(((i, m) for i, m in own if m.group(1) == "决定"), None) if kind == "decision" else None
            pick = pick or (own[-1] if own else None)
            where = texts[-8:]
            if pick:
                out["effective_date"] = iso_date(pick[1].group(2)) or out["date"]   # 自公布之日起施行: the publication date
                where = [texts[pick[0]]] + texts[-8:]
            else:
                m = next((EFFECTIVE.search(t) for t in texts[-8:] + texts[:3] if EFFECTIVE.search(t)), None)
                if m:
                    out["effective_date"] = iso_date(m.group(1)) or out["date"]
            for t in where:
                for sentence in re.split(r"[。；\n]", t):
                    if "废止" in sentence:
                        out["repeals"] += [x for x in re.findall(r"《([^《》]+)》", sentence) if x not in (out["name"], title)]
            out["repeals"] = list(dict.fromkeys(out["repeals"]))
    if card.get("notice_id"):
        out["companions"] = [d for d, c in cards.items() if c.get("notice_id") == card["notice_id"] and d != card["doc_id"]]
    need = ["name", "date"] + (["effective_date"] if kind in ("law", "regulation", "decision") and not out.get("compilation") else [])
    return out, need


# ----------------------------------------------------------------------------- annual reports
def report(card, doc):
    title = card["title"]
    bad = re.compile(r"股票|证券|代码|简称|交易所|不适用|^[A-Z]股$")
    out = {"doc_type": "年度报告", "company": find(rf"^({COMPANY})", title), "year": int(find(r"(20\d{2})\s*年", title) or 0) or None,
           "date": None,
           "stock_code": labelled(doc, 20, ["股票代码", "证券代码", "股份代号", "A股股票代码"], r"\d{5,6}",
                                  lambda v: bool(re.fullmatch(r"\d{5,6}", v))),
           "stock_name": labelled(doc, 20, ["股票简称", "证券简称", "A股简称"], r"[一-龥A-Za-z]{2,10}",
                                  lambda v: bool(re.fullmatch(r"[一-龥A-Za-z]{2,10}", v)) and not bad.search(v))}
    return out, ["company", "year", "stock_code", "stock_name"]


# ----------------------------------------------------------------------------- contracts
ROLE_WORDS = {"发行人", "发行主体", "上市公司", "本公司", "公司", "本集团", "集团", "本企业", "企业"}
NO_GUARANTEE = re.compile(r"无担保|不提供担保|未提供担保|不设担保|未设置?担保|无增信|增信措施(?:情况)?\s*[:：]\s*无")


def definitions(doc, pages=30):
    """The rows  | 简称、简称 | 指 | 全称 |  of the 释义 table, in order, repeated header rows once."""
    out, seen = [], set()
    for u in doc.units:
        if u["page"] > pages:
            break
        for cells in table_rows(u["body"]):
            if len(cells) >= 3 and cells[1] == "指" and cells[0] and cells[2] and cells[0] not in seen:
                seen.add(cells[0])
                out.append({"terms": [t.strip() for t in re.split(r"[、/，,]|或", cells[0]) if t.strip()], "meaning": cells[2]})
    return out


def contract(card, doc):
    title = card["title"]
    cover, front = first_pages(doc, 2), first_pages(doc, 40)
    defs = definitions(doc)

    def role(keys):
        return next((d for d in defs if set(d["terms"]) & keys), None)

    if "可转换公司债券" in title:
        doc_type = "可转换公司债券募集说明书"
    elif "重大资产重组报告书" in title:
        doc_type = "重大资产重组报告书"
    elif "购买资产" in title and "报告书" in title:
        doc_type = "发行股份购买资产报告书"
    elif "公司债券" in title:
        doc_type = "公司债券募集说明书"
    elif "债券" in title or "募集说明书" in front[:3000]:
        doc_type = "债券募集说明书"
    else:
        doc_type = None
    main = role({"发行人", "发行主体", "上市公司"}) or role({"本公司", "公司"})
    issuer = find(rf"({COMPANY})", main["meaning"]) if main else None
    date = year_month(cover)
    out = {"doc_type": doc_type, "company": find(rf"^({COMPANY})", title) or issuer, "date": date,
           "year": int(find(r"(20\d{2})\s*年", title) or (date or "0")[:4]) or None,
           "company_aliases": [t for t in main["terms"] if t not in ROLE_WORDS] if main else [],
           "stock_code": find(r"(?:股票代码|证券代码)[^0-9]{0,6}(\d{6})", (main["meaning"] if main else "") + "\n" + cover)}
    need = ["doc_type", "company", "year"]
    if doc_type and "报告书" in doc_type:                           # reorganisations: the listed company, the other side, what is bought
        out["listed_company"] = issuer
        for field, keys in (("counterparty", {"交易对方"}), ("target_company", {"标的公司"}),
                            ("target", {"交易标的", "标的资产", "拟购买资产", "标的股权"})):
            d = role(keys)
            out[field] = d["meaning"] if d else None
        need += ["listed_company", "counterparty", "target"]
    else:                                                          # bonds: the issuer and who guarantees them
        out["issuer"] = issuer
        d = role({"担保人", "保证人"})
        named = [x for x in re.findall(r"(?:本期|本次)债券由([^，。、\n]{2,30}?)提供[^。\n]{0,24}担保", "\n".join(u["body"] for u in doc.units))
                 if not re.search(r"担保人|第三方", x)]
        if d:
            out["guarantor"] = d["meaning"]
        elif named:
            out["guarantor"] = Counter(named).most_common(1)[0][0]
        elif NO_GUARANTEE.search(front):
            out["guarantor"] = "无担保"
        else:
            out["guarantor"] = None
        need += ["issuer", "guarantor"]
    out["definitions"] = defs
    return out, need + ["definitions"]


# ----------------------------------------------------------------------------- insurance
VERSION = re.compile(r"[（(][^（）()]*(?:版|款)[^（）()]*[）)]|[A-Z]\s*款")
PRODUCT_TYPE = ["重大疾病保险", "医疗保险", "意外伤害保险", "养老年金保险", "终身寿险", "养老保险", "家庭财产综合保险", "家庭财产保险",
                "责任保险", "特种车商业保险"]


def insurance(card, doc):
    title = card["title"]
    company = find(rf"^({COMPANY})", title)
    rest = title[len(company):] if company else title
    version = [re.sub(r"\s+", "", v) for v in VERSION.findall(rest)]
    product = re.sub(r"\s+", " ", re.sub(r"(?:利益|示范)?条款\s*$", "", VERSION.sub("", rest).strip())).strip()
    out = {"doc_type": "保险条款", "company": company or find(r"([一-龥]{2,12}保险(?:股份)?有限公司)", first_pages(doc, 10)),
           "year": int(find(r"(20\d{2})", "".join(version)) or 0) or None, "date": None,
           "product": product, "product_type": next((p for p in PRODUCT_TYPE if p in re.sub(r"\s", "", title)), None),
           "version": "".join(version) or None}
    return out, ["company", "product", "product_type"]


# ----------------------------------------------------------------------------- research
REPORT_TYPE = re.compile(r"首次覆盖|公司深度|公司研究|深度研究报告|行业深度|行业研究|行业周报|周报|行业月报|月报|策略|专题|深度报告|白皮书")
ABSTRACT_HEAD = re.compile(r"投资要点|核心观点|报告导读|报告摘要|投资摘要|本周观点|核心结论|主要观点|摘要")
CONTACT = re.compile(r"执业|证书编号|登记编[号码]|SAC|邮箱|@|电话|联系人")
GENERIC = {"行业", "公司", "证券", "研究", "金融", "策略", "专题", "宏观"}


def research(card, doc):
    front = first_pages(doc, 2)
    subject, code = None, None
    m = re.search(r"([一-龥A-Za-z]{2,10})\s*[（(]\s*(\d{4,6})(?:\.[A-Z]{2})?\s*[）)]", front)
    if m and m.group(1) not in GENERIC:
        subject, code = m.group(1), m.group(2)
    else:
        named = find(r"([一-龥]{2,8}?)(?:首次覆盖报告|深度研究报告|公司深度报告)", front)
        subject = named if named and named not in GENERIC and not named.endswith("行业") else None
    texts = [u for u in doc.units if u["page"] <= 3 and u["type"] == "text"]
    summary = [u for u in texts if ABSTRACT_HEAD.search((u["path"] or "").split(" > ")[-1])]
    if not summary:                                                 # no summary heading: the long paragraphs of the first page that has any
        long = [u for u in texts if len(u["body"]) >= 150 and len(CONTACT.findall(u["body"])) < 2]
        summary = [u for u in long if u["page"] == long[0]["page"]] if long else []
    toc = []
    outline = PROC / doc.doc_id / "docs.json"
    if outline.exists():
        entries = [e for e in json.loads(outline.read_text(encoding="utf-8"))["entries"] if e.get("kind", "heading") == "heading"]
        top = min((e["level"] for e in entries), default=None)
        toc = [e["title"] for e in entries if e["level"] == top]
    date = card.get("date") or None
    out = {"doc_type": "研究报告", "company": subject, "year": int(date[:4]) if date else None, "date": date,
           "broker": card.get("issuer") or None, "report_type": find(REPORT_TYPE, front, 0), "subject": subject, "stock_code": code,
           "abstract": "\n".join(u["body"] for u in summary)[:2000] or None, "toc": toc}
    company_report = bool(out["report_type"]) and re.search(r"首次覆盖|公司", out["report_type"])
    return out, ["broker", "date", "abstract", "toc"] + (["subject"] if company_report else [])


# ----------------------------------------------------------------------------- all documents
def build(doc_id, cards):
    card = cards[doc_id]
    doc = Doc(doc_id, card)
    domain = card["domain"]
    if domain == "regulatory":
        fields, need = regulatory(card, doc, cards)
    elif domain == "financial_reports":
        fields, need = report(card, doc)
    elif domain == "financial_contracts":
        fields, need = contract(card, doc)
    elif domain == "insurance":
        fields, need = insurance(card, doc)
    else:
        fields, need = research(card, doc)
    out = {"doc_id": doc_id, "domain": domain, "title": card["title"]}
    for key in ("doc_type", "company", "year", "date"):
        out[key] = fields.pop(key, None)
    out.update(fields)
    out["warnings"] = [k for k in need if not out.get(k)]
    return out


def main(argv):
    cards = load_cards()
    if argv:
        for d in argv:
            print(json.dumps(build(d, cards), ensure_ascii=False, indent=1))
        return
    out = [build(d, cards) for d in cards]
    by_id = {c["doc_id"]: c for c in out}
    for c in out:                                                   # an explanation carries the name of the text it explains
        if c.get("kind") == "explanation":
            text = next((by_id[d] for d in c["companions"] if by_id[d].get("kind") != "explanation"), None)
            if text:
                c["name"], c["former_names"] = text["name"], text["former_names"]
    seen = {}
    for c in out:                                                   # the same report stored twice
        key = (c["domain"], re.sub(r"\W", "", c["title"]), c.get("date"))
        if c["domain"] == "research" and key in seen:
            c["duplicate_of"] = seen[key]
        seen.setdefault(key, c["doc_id"])
    meta = {"documents": len(out), "by_domain": dict(Counter(c["domain"] for c in out)),
            "source": "rules over the title, processed_data/docs.jsonl and the first pages (script/build_doc_cards.py); no model call"}
    OUT.write_text(json.dumps({"meta": meta, "cards": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(out)} cards -> {OUT.relative_to(ROOT)}")
    for domain in meta["by_domain"]:
        rows = [c for c in out if c["domain"] == domain]
        warned = [c for c in rows if c["warnings"]]
        print(f"  {domain}: {len(rows)} cards, {len(warned)} with warnings {dict(Counter(w for c in warned for w in c['warnings']))}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
