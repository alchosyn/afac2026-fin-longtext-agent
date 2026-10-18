# -*- coding: utf-8 -*-
"""Compare the BM25 retrieval of the pipeline with and without the stop-word list of
agent.claims (STOPWORDS) on ten hand-labelled claims, two per domain.

The claims are those of tools/compare_expansion.py. Their gold is restated here by
content: a regular expression that the body of a gold chunk matches (the sentence, clause
or table row that decides the claim), so it survives a rebuild of the structure or of the
chunks; every chunk of the claim's documents that matches is gold (checked against the
text on 2026-10-03).

Both settings run the pipeline's retrieval (split_claims, two-field BM25 over the
question's documents, top 3 per sub-claim) on indexes built in memory, so the bm25.json
files on disk are neither read nor written. Reported per claim and setting: the best rank
of a gold chunk over the sub-claims, whether one is among the pipeline's hits, the number
of chunks with a positive score (the candidates the postings give) and the hits.

    python tools/compare_stopwords.py            # prints the table, writes output/retrieval/compare_stopwords.json
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "script"))

from agent import bm25_pipeline as bp      # noqa: E402
from agent import claims as cl             # noqa: E402
from agent import retrieval as rt          # noqa: E402
from agent.bm25 import build_index         # noqa: E402
from compare_expansion import all_claims   # noqa: E402

OUT = ROOT / "output" / "retrieval" / "compare_stopwords.json"

# claim id -> pattern of the chunks that decide it
GOLD = {
    "reg_a_008-A": r"停止担任董事和高级管理人员职务",                                      # 第二十六条：7日内报告
    "reg_a_015-B": r"未在规定期限内披露年度报告和中期报告|未按照《证券法》规定在规定期限内报送有关报告",   # 第二十一条 / 第五十五条
    "res_a_001-B": r"2005-2018\s*年实现复合增速\s*9\.9\s*%",
    "res_a_001-C": r"金融信创市场规模预计接近\s*2500\s*亿",
    "fc-2": r"(发行金额|发行规模|募集资金)不超过(人民币)?\s*10\s*亿元",
    "fc-8": r"66\.15",                                                                     # 2023 年末的资产负债率：正文两处、指标表两处
    "fin-1": r"研发投入占营业收入比例 \| 6\.97%",
    "fin-6": r"营业收入[^\n]{0,20}777,102,455",                                            # 主要会计数据、营业收入构成、利润表
    "ins-4": r"2\s*年内自杀",
    "ins-7": r"犹豫期为\s*20\s*日|有\s*20\s*日的犹豫期",
}
SETTINGS = (("off", frozenset()), ("on", cl.STOPWORDS))


def gold_chunks(claim):
    rx = re.compile(GOLD[claim["id"]])
    return [u["id"] for d in claim["doc_ids"] for u in rt.get_doc(d).units if rx.search(u["body"])]


def run(claim, gold):
    subs = cl.split_claims(claim["statement"])
    best, candidates = None, 0
    for sub in subs:
        ranking = [uid for uid, s in rt.bm25_scores(sub, claim["doc_ids"]) if s > 0]
        candidates = max(candidates, len(ranking))
        ranks = [ranking.index(g) + 1 for g in gold if g in ranking]
        if ranks and (best is None or min(ranks) < best):
            best = min(ranks)
    hits = [uid for uid, _, _, _ in bp.retrieve(subs, claim["doc_ids"])]
    return {"rank": best, "covered": any(h in gold for h in hits), "candidates": candidates,
            "hits": hits, "query": [cl.tokenize(s) for s in subs]}


def short(uid):
    return uid.split("#")[0][:14] + "#" + uid.split("#")[1]


def main():
    claims = [c for c in all_claims() if c["id"] in GOLD]
    rt.doc_index = build_index                     # indexes from the tokenizer of the moment, not from bm25.json
    rows = []
    for c in claims:
        gold = gold_chunks(c)
        if not gold:
            raise KeyError(f"{c['id']}: no chunk matches its gold pattern")
        rows.append({"id": c["id"], "domain": c["domain"], "statement": c["statement"], "docs": len(c["doc_ids"]),
                     "chunks": sum(len(rt.get_doc(d).units) for d in c["doc_ids"]),
                     "gold": [g.split("#")[1] for g in gold], "_gold": gold, "_claim": c})
    for name, words in SETTINGS:
        cl.STOPWORDS = words
        rt._BM25.clear()
        for r in rows:
            r[name] = run(r["_claim"], r["_gold"])
    cl.STOPWORDS = SETTINGS[1][1]

    print(f"{'claim':<13}{'chunks':>7}  {'gold':<24}{'rank off':>9}{'rank on':>9}{'cand off':>10}{'cand on':>9}  top-3 changed")
    for r in rows:
        a, b = r["off"], r["on"]
        changed = "same" if a["hits"] == b["hits"] else "order" if set(a["hits"]) == set(b["hits"]) else "yes"
        print(f"{r['id']:<13}{r['chunks']:>7}  {','.join(r['gold'])[:22]:<24}{str(a['rank']):>9}{str(b['rank']):>9}"
              f"{a['candidates']:>10}{b['candidates']:>9}  {changed}")
    summary = {}
    n = len(rows)
    print(f"\n{'setting':<9}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'MRR':>7}{'in pipeline hits':>18}{'mean candidates':>17}")
    for name, _ in SETTINGS:
        ranks = [r[name]["rank"] for r in rows]
        s = {f"hit@{k}": sum(1 for x in ranks if x and x <= k) for k in (1, 3, 5)}
        s["mrr"] = round(sum(1 / x for x in ranks if x) / n, 3)
        s["covered"] = sum(1 for r in rows if r[name]["covered"])
        s["mean_candidates"] = round(sum(r[name]["candidates"] for r in rows) / n, 1)
        summary[name] = s
        print(f"{name:<9}{s['hit@1']:>5}/{n}{s['hit@3']:>5}/{n}{s['hit@5']:>5}/{n}{s['mrr']:>7}{s['covered']:>16}/{n}{s['mean_candidates']:>17}")

    print("\nhits (gold marked *)")
    for r in rows:
        print(f"\n{r['id']}: {r['statement']}")
        for name, _ in SETTINGS:
            marked = [("*" if h in r["_gold"] else "") + h.split("#")[1] for h in r[name]["hits"]]
            print(f"  {name:<4}{' '.join(marked):<28}query: {' '.join(r[name]['query'][0])}")

    for r in rows:
        del r["_gold"], r["_claim"]
        for name, _ in SETTINGS:
            r[name]["hits"] = [short(h) for h in r[name]["hits"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"stopwords": sorted(SETTINGS[1][1]), "summary": summary, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
