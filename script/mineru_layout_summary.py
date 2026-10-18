# -*- coding: utf-8 -*-
"""Register MinerU outputs and summarise their layout analysis.

For every processed_data/<doc_id>/mineru.json (or only the doc_ids given) two things happen:
  * the document card in processed_data/docs.jsonl gets its `mineru` field; a document
    without a card gets one (domain and source PDF from the doc_id naming convention,
    title from MinerU's first doc_title block)
  * one summary row: pages, pages without a text layer, block counts per MinerU type,
    tables with / without caption, tables MinerU flagged as continuing the previous page
The rows go to reports/mineru_layout_summary.json and .md.

    python script/mineru_layout_summary.py                 # every doc with a mineru.json
    python script/mineru_layout_summary.py ins_02 ins_03   # a subset

A document without a card gets its domain and source PDF from the doc_id naming
convention (parsing.common.source_for).
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from parsing.common import PROC, clean, load_cards, load_mineru, save_cards, source_for, txt  # noqa: E402

DROPPED = ("header", "footer", "page_number", "index")
TEXTY = ("text", "aside_text", "page_footnote", "equation", "code", "list", "ref_text")
DOMAIN_LABEL = {"financial_contracts": "金融合同", "financial_reports": "年报", "insurance": "保险条款",
                "regulatory": "监管法规", "research": "研报"}


def mineru_title(data):
    """MinerU's first doc_title, without the registration number an insurance clause
    appends ("...条款注册号:C000179...")."""
    for p in data["pages"]:
        for b in p["blocks"]:
            if b["type"] == "doc_title":
                t = re.split(r"\s*注册号", clean(txt(b)))[0].strip()
                if t:
                    return t
    return ""


def summarise(doc_id, card, data, mineru_path):
    pdf = pymupdf.open(str(ROOT / card["source_pdf"]))
    no_text = sum(1 for p in pdf if len(p.get_text().strip()) < 30)
    blocks = [b for p in data["pages"] for b in p["blocks"]]
    types = Counter(b["type"] for b in blocks)
    tables = [b for b in blocks if b["type"] == "table"]
    subs = lambda b, suffix: [c for c in b.get("content", []) if isinstance(c, dict) and str(c.get("type", "")).endswith(suffix)]
    levels = Counter(b.get("level") for b in blocks if b["type"] in ("doc_title", "paragraph_title"))
    return {"doc_id": doc_id, "domain": card["domain"], "title": card["title"], "source_pdf": card["source_pdf"],
            "mineru": str(mineru_path.relative_to(ROOT)).replace("\\", "/"),
            "tier": data.get("extensions", {}).get("mineru", {}).get("tier"),
            "parse_mode": data.get("extensions", {}).get("mineru", {}).get("parse_mode"),
            "pages": len(data["pages"]), "pdf_pages": len(pdf), "pages_without_text_layer": no_text,
            "blocks": len(blocks), "types": dict(sorted(types.items())),
            "titles": types["doc_title"] + types["paragraph_title"], "title_levels": {str(k): v for k, v in sorted(levels.items(), key=lambda x: (x[0] is None, x[0]))},
            "text_blocks": sum(types[t] for t in TEXTY), "dropped_blocks": sum(types[t] for t in DROPPED),
            "tables": len(tables), "tables_with_caption": sum(1 for b in tables if subs(b, "_caption")),
            "tables_with_footnote": sum(1 for b in tables if subs(b, "_footnote")),
            "tables_continues_prev": sum(1 for b in tables if b.get("continues_prev")),
            "figures": types["image"] + types["chart"], "images": types["image"], "charts": types["chart"],
            "bytes": mineru_path.stat().st_size}


def markdown(rows):
    head = ["doc_id", "类别", "页数", "无文本层页", "块总数", "标题", "正文块", "表格(有题注/续页)", "图片/图表", "页眉页脚页码", "目录", "档位"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        t = r["types"]
        out.append("| " + " | ".join(str(x) for x in (
            r["doc_id"], DOMAIN_LABEL.get(r["domain"], r["domain"]), r["pages"], r["pages_without_text_layer"], r["blocks"],
            r["titles"], r["text_blocks"], f"{r['tables']} ({r['tables_with_caption']}/{r['tables_continues_prev']})",
            f"{r['images']}/{r['charts']}", t.get("header", 0) + t.get("footer", 0) + t.get("page_number", 0),
            t.get("index", 0), f"{r['tier']}/{r['parse_mode']}")) + " |")
    return "\n".join(out)


def main(argv):
    wanted = set(argv)
    cards = load_cards()
    by_id = {c["doc_id"]: c for c in cards}
    rows, changed = [], 0
    for mineru_path in sorted(PROC.glob("*/mineru.json")):
        doc_id = mineru_path.parent.name
        if wanted and doc_id not in wanted:
            continue
        data = load_mineru(mineru_path)
        rel = str(mineru_path.relative_to(ROOT)).replace("\\", "/")
        card = by_id.get(doc_id)
        if card is None:
            domain, pdf = source_for(doc_id)
            if not pdf or not (ROOT / pdf).exists():
                print(f"{doc_id}: no card and no source PDF found by naming convention -> skipped")
                continue
            card = {"doc_id": doc_id, "domain": domain, "title": mineru_title(data) or doc_id, "issuer": "", "date": "",
                    "source_pdf": pdf, "mineru": rel}
            cards.append(card)
            by_id[doc_id] = card
            changed += 1
            print(f"{doc_id}: new card ({domain}, {pdf}) title={card['title']!r}")
        elif card.get("mineru") != rel:
            card["mineru"] = rel
            changed += 1
            print(f"{doc_id}: card updated with mineru={rel}")
        if not card.get("source_pdf"):
            print(f"{doc_id}: card has no source_pdf -> summary skipped")
            continue
        rows.append(summarise(doc_id, card, data, mineru_path))
    if changed:
        save_cards(cards)
    order = {d: i for i, d in enumerate(DOMAIN_LABEL)}
    rows.sort(key=lambda r: (order.get(r["domain"], 9), r["doc_id"]))
    (ROOT / "reports" / "mineru_layout_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "reports" / "mineru_layout_summary.md").write_text(markdown(rows) + "\n", encoding="utf-8")
    print(f"\n{len(rows)} documents, {changed} cards changed\n")
    print(markdown(rows))
    for r in rows:
        print(f"  {r['doc_id']}: {r['types']}")


if __name__ == "__main__":
    main(sys.argv[1:])
