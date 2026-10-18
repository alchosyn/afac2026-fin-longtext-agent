# -*- coding: utf-8 -*-
"""The header every chunk carries, so that a chunk says by itself which document it is from
and can be filtered on these fields:

    {"doc_id", "domain", "title", "doc_type", "company", "year", "date", + the fields of the domain}

taken from the document's card in processed_data/doc_cards.json (script/build_doc_cards.py).
The long card fields (the 释义 table, a report's summary and contents) stay on the card.
header_text() is the header as one line, for an index field or a prompt."""
import json

from .common import PROC

CARDS = PROC / "doc_cards.json"
COMMON = ("doc_id", "domain", "title", "doc_type", "company", "year", "date")
BY_DOMAIN = {
    "regulatory": ("kind", "name", "former_names", "issuer", "notice_no", "effective_date"),
    "financial_reports": ("stock_code", "stock_name"),
    "financial_contracts": ("company_aliases", "stock_code", "issuer", "guarantor", "listed_company", "counterparty",
                            "target_company", "target"),
    "insurance": ("product", "product_type", "version"),
    "research": ("broker", "report_type", "subject", "stock_code"),
}
NOT_TEXT = ("doc_id", "domain", "kind")          # identifiers, no words a query would use


def header_of(card):
    """The header of a document's chunks: the card's short fields that have a value."""
    keys = COMMON + BY_DOMAIN.get(card["domain"], ())
    return {k: card[k] for k in keys if card.get(k) not in (None, "", [])}


def load_headers():
    """{doc_id: header} for the documents that have a card; {} without doc_cards.json."""
    if not CARDS.exists():
        return {}
    return {c["doc_id"]: header_of(c) for c in json.loads(CARDS.read_text(encoding="utf-8"))["cards"]}


def header_text(header):
    """标题｜公司｜年份｜… : every value once, lists joined by 、."""
    parts = []
    for k, v in header.items():
        if k in NOT_TEXT:
            continue
        v = "、".join(v) if isinstance(v, list) else str(v)
        if v and v not in parts:
            parts.append(v)
    return "｜".join(parts)
