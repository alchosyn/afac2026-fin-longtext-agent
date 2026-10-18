# -*- coding: utf-8 -*-
"""Compare the two context strategies of the BM25 pipeline on hand-labelled claims:
    window   the hit plus one unit before and after it in its section (the default)
    section  the whole leaf section, trimmed around the hits (bm25_pipeline.fit_section)

Eight claims per domain. Regulatory and research use the options of two group-A questions
each (all documents parsed). Financial contracts, annual reports and insurance have one
MinerU-parsed document each, so their claims are written against that document alone
(half of them deliberately wrong: the number of the neighbouring year or of the other
subject, which is what the table labels have to settle). Gold verdicts and gold units were
checked against the source text on 2026-09-23.

Retrieval (BM25 hits) is identical for both strategies; only the evidence handed to the
judge differs. The claim-analysis call does not see the documents, so it is made once per
claim and shared. Reported per claim: gold, verdict under each strategy, whether a gold
unit is inside the evidence, evidence characters and judge tokens.

    python tools/compare_expansion.py --dry          # retrieval and evidence only, no API calls
    python tools/compare_expansion.py                # both strategies with the model
    python tools/compare_expansion.py --domain insurance --mode section

Writes output/retrieval/compare_expansion.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import bm25_pipeline as bp                       # noqa: E402
from agent.claims import analyze_claim, judge, split_claims  # noqa: E402
from agent.docview import load_cards                         # noqa: E402

OUT = ROOT / "output" / "retrieval" / "compare_expansion.json"
MODES = ("window", "section")


def card_id(prefix):
    """The card whose doc_id starts with prefix (the strict_v3 ids carry the full title)."""
    ids = [d for d in load_cards() if d.startswith(prefix)]
    if len(ids) != 1:
        raise KeyError(f"{prefix}: {ids}")
    return ids[0]


def question_claims(domain, qid, gold_by_letter, gold_units):
    """One claim per option of a group-A question; gold_units: letter -> [(doc prefix, local id)]."""
    qs = json.load(open(ROOT / "questions" / "group_a" / f"{domain}_questions.json", encoding="utf-8"))
    q = next(x for x in qs if x["qid"] == qid)
    docs = [card_id(d) for d in q["doc_ids"]]
    out = []
    for letter, text in q["options"].items():
        units = [f"{card_id(p)}#{u}" for p, u in gold_units.get(letter, [])]
        out.append({"id": f"{qid}-{letter}", "domain": domain, "statement": text.strip(), "doc_ids": docs,
                    "gold": gold_by_letter[letter], "gold_units": units})
    return out


PREFIX = {"financial_contracts": "fc", "financial_reports": "fin", "insurance": "ins"}


def single_doc_claims(domain, doc, rows):
    return [{"id": f"{PREFIX[domain]}-{i + 1}", "domain": domain, "statement": s, "doc_ids": [doc], "gold": g,
             "gold_units": [f"{doc}#{u}" for u in units]} for i, (s, g, units) in enumerate(rows)]


def all_claims():
    claims = []
    claims += question_claims("regulatory", "reg_a_008", {"A": "正确", "B": "错误", "C": "错误", "D": "错误"},
                              {"A": [("strict_v3_016", "a26")], "B": [("strict_v3_009", "a30")],
                               "C": [("strict_v3_016", "a27")], "D": [("strict_v3_009", "a13")]})
    claims += question_claims("regulatory", "reg_a_015", {"A": "错误", "B": "错误", "C": "正确", "D": "错误"},
                              {"A": [("csrc_0009_att1", "a12")], "B": [("csrc_0009_att1", "a21"), ("csrc_0009_att1", "a55")],
                               "C": [("csrc_0023_att1", "a10")], "D": [("csrc_0009_att1", "a12")]})
    claims += question_claims("research", "res_a_001", {"A": "正确", "B": "正确", "C": "正确", "D": "错误"},
                              {"A": [("pack2_text01", "p020")], "B": [("pack2_text01", "p018"), ("pack2_text01", "p035")],
                               "C": [("pack2_text17", "t04")], "D": [("pack2_text17", "p067")]})
    claims += question_claims("research", "res_a_002", {"A": "正确", "B": "正确", "C": "正确", "D": "错误"},
                              {"A": [("pack2_text11", "p067"), ("pack2_text11", "p006")], "B": [("pack2_text01", "p020")],
                               "C": [("pack2_text11", "p038")], "D": []})           # D: 金融信创 is in text17, not in this question's documents
    claims += single_doc_claims("financial_contracts", "fc_text01", [
        ("第一份文档的发行人名称为广东省广晟控股集团有限公司", "正确", ["p005"]),
        ("本期债券发行金额上限为 10 亿元", "正确", ["p003"]),
        ("发行人主体信用评级为 AAA", "正确", ["p006"]),
        ("文档包含发行人将及时、公平地履行信息披露义务的明确承诺条款", "正确", ["p008", "p607"]),
        ("文档明确给出了发行人合并口径资产负债率的具体数值", "正确", ["p018", "p060", "t115"]),
        ("文档列出了发行人最近三年的净利润数据", "正确", ["p019"]),
        ("发行人最近一期末合并口径资产负债率为 70.02%", "错误", ["p018"]),                     # 70.02% is the parent-company figure; consolidated 68.06%
        ("2024 年末发行人资产负债率为 66.15%", "错误", ["p060", "t115"]),                        # 2024: 67.89%; 66.15% is 2023
    ])
    claims += single_doc_claims("financial_reports", "fr_byd_2024", [
        ("比亚迪 2024 年研发投入占营业收入比例为 6.97%", "正确", ["t31"]),
        ("比亚迪 2024 年研发投入占营业收入的比例为 6.63%", "错误", ["t31"]),                     # 6.63% is 2023
        ("比亚迪 2024 年归属于上市公司股东的净利润为 300.4 亿元", "错误", ["t05"]),               # 402.5 亿; 300.4 亿 is 2023
        ("比亚迪 2024 年经营活动产生的现金流量净额较上年同期有所增长", "错误", ["t05", "t32"]),    # -21.37%
        ("比亚迪 2024 年经营活动产生的现金流量净额低于其营业收入的一半", "正确", ["t04", "t05"]),
        ("比亚迪 2024 年营业收入为 7,771.02 亿元", "正确", ["t04"]),
        ("比亚迪 2024 年归属于上市公司股东的净利润同比增长 34.00%", "正确", ["t05"]),
        ("比亚迪 2024 年研发投入资本化的金额为 3.43 亿元", "错误", ["t31"]),                    # 9.66 亿; 3.43 亿 is 2023
    ])
    claims += single_doc_claims("insurance", "ins_01", [
        ("平安智盈金生允许保单贷款，最高为现金价值的 80%", "错误", []),                          # no such clause in the policy
        ("平安智盈金生条款给出了不同保单年度现金价值的比例公式", "正确", ["p150"]),
        ("平安智盈金生允许在养老保险金开始领取日前变更养老保险金开始领取年龄和领取方式", "正确", ["p016"]),
        ("平安智盈金生将被保险人自合同成立之日起 2 年内自杀列为责任免除", "正确", ["p071"]),
        ("平安智盈金生投保人在犹豫期内解除合同的，退还全部已交保险费", "正确", ["p126"]),
        ("平安智盈金生：被保险人因酒后驾驶机动车导致失能失智护理状态属于责任免除范围", "正确", ["p071", "p074"]),
        ("平安智盈金生的犹豫期为 15 日", "错误", ["p126"]),                                       # 20 日
        ("平安智盈金生投保人在犹豫期后解除合同的，保险公司自收到通知之日起 60 日内退还现金价值", "错误", ["p127"]),  # 30 日
    ])
    return claims


def run_claim(claim, modes, llm, dry):
    sub = split_claims(claim["statement"])
    hits = bp.retrieve(sub, claim["doc_ids"])
    res = {"id": claim["id"], "domain": claim["domain"], "statement": claim["statement"], "gold": claim["gold"],
           "gold_units": claim["gold_units"], "hits": [u.split("#")[1] for u, _, _, _ in hits], "modes": {}}
    analysis = None
    for mode in modes:
        bp.EXPAND = mode
        records, trace, _ = bp.evidence_records(hits, claim["statement"])
        units = [f"{b['doc']}#{u}" for b in trace for u in b["units"]]
        gold_in = [g for g in claim["gold_units"] if any(x.split("#")[1] == g.split("#")[1] and g.startswith(x.split("#")[0]) for x in units)]
        m = {"evidence_units": sum(len(b["units"]) for b in trace), "evidence_chars": sum(len(r["text"]) for r in records),
             "gold_covered": bool(gold_in) if claim["gold_units"] else None, "sections": [b["section"][-40:] for b in trace],
             "verdict": None, "reason": "", "tokens": 0}
        if not dry and llm and records:
            if analysis is None:
                analysis = analyze_claim(llm, claim["statement"], claim["id"] + "-")
            before = llm.usage["total_tokens"]
            verdict, data = judge(llm, claim["statement"], analysis, records, f"{claim['id']}-{mode}-judge")
            m["verdict"] = "错误" if verdict == "证据不足" else verdict      # the pipeline's default for unsupported claims
            m["raw_verdict"] = verdict
            m["reason"] = str(data.get("reason", ""))[:160]
            m["tokens"] = llm.usage["total_tokens"] - before
        res["modes"][mode] = m
    bp.EXPAND = "window"
    return res


def summarize(results, modes):
    domains = sorted({r["domain"] for r in results}, key=lambda d: [r["domain"] for r in results].index(d))
    print(f"\n{'domain':<20}" + "".join(f"{m + ' acc':>13}{m + ' cov':>13}{m + ' chars':>15}" for m in modes))
    for dom in domains + ["all"]:
        rows = [r for r in results if dom == "all" or r["domain"] == dom]
        cells = ""
        for m in modes:
            judged = [r for r in rows if r["modes"][m]["verdict"]]
            acc = sum(1 for r in judged if r["modes"][m]["verdict"] == r["gold"])
            cov_rows = [r for r in rows if r["modes"][m]["gold_covered"] is not None]
            cov = sum(1 for r in cov_rows if r["modes"][m]["gold_covered"])
            chars = sum(r["modes"][m]["evidence_chars"] for r in rows) / max(1, len(rows))
            cells += f"{(str(acc) + '/' + str(len(judged))) if judged else '-':>13}{str(cov) + '/' + str(len(cov_rows)):>13}{chars:>15.0f}"
        print(f"{dom:<20}{cells}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--domain")
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--ids", nargs="*")
    args = ap.parse_args(argv)
    modes = [args.mode] if args.mode else list(MODES)
    claims = [c for c in all_claims() if (not args.domain or c["domain"] == args.domain) and (not args.ids or c["id"] in args.ids)]

    llm = None
    if not args.dry:
        from agent.llm import Qwen
        llm = Qwen(log_path=ROOT / "logs" / f"compare_expansion_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")

    results = []
    for c in claims:
        r = run_claim(c, modes, llm, args.dry)
        results.append(r)
        line = f"{r['id']:<14} gold={r['gold']} hits={','.join(r['hits'][:4])}"
        for m in modes:
            x = r["modes"][m]
            cov = {True: "cov", False: "MISS", None: "n/a"}[x["gold_covered"]]
            line += f" | {m}: {x['verdict'] or '-'} {cov} {x['evidence_units']}u/{x['evidence_chars']}c"
        print(line)
    summarize(results, modes)
    if not args.dry:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {OUT}\n-> {llm.log_path}  tokens {llm.usage}")


if __name__ == "__main__":
    main(sys.argv[1:])
