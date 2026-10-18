# -*- coding: utf-8 -*-
"""ReAct (react_relevance.py --mask none for choice questions, react_compact.py for calculation
questions) against react_plan.py on ten B questions drawn 2026-10-06 (two per domain, seed
20261011, none run before): answers against our own gold, steps, tokens.

    python flows/compare_sample10.py

Writes reports/sample10_compare.md. Gold, set by hand from the documents:
    fin_b_011  ABC  fr_chinamobile_2025#c0009/c0010: 338,931 / 1,050,187 = 32.27% (disclosed 32.3%);
                    fr_cscec_2025#c0014: 2.718 x 41,320,390,444 / 10 = 112.31 亿元; fr_cmb_2025#c0005:
                    2.016 per share = 20.16 per 10; fr_midea_2025#c0192: 5 + 38 = 43 per 10, not 48
    fin_b_008  AC   fr_cscec_2024#c0015 24.29%, fr_cscec_2025#c0014 28.75% -> +4.46 points; 2025
                    cash flow / revenue 20,537,132 / 2,082,141,811 = 0.99% (not over 1%); EPS 1.11 ->
                    0.94 -15.32%, net profit 46,187,099 -> 39,069,002 -15.41%; profit fell 7.12 bn,
                    cash flow rose 4.76 bn -> D wrong
    fc_b_010   BCD  fc_text10#c1108/c1112: shares first, cash when the shares do not suffice (A
                    wrong); c1106: formula on cumulative promised minus actual revenue x 交易作价 x
                    39.61%
    fc_b_007   B    fc_text05#c0008 word for word; 安克 (fc_text04) has no capacity-absorption risk at
                    all (A, D wrong); 普联 does not mention it, but the reason C gives is not in the
                    text (C judged wrong, doubtful)
    ins_b_012  BCD  地震 excluded in ins_11#c0009, ins_12#c0009, ins_14#c0008; not in ins_04
    ins_b_001  333.20  ins_01#c0042 year 8: 80 + 12 x 75% = 89; ins_02#c0042/c0043 year 3: 90 x 98% =
                    88.2; ins_15#c0020 / ins_16#c0034 cash value 70 and 86
    reg_b_004  2026年6月30日  银行卡清算机构管理办法 第十二条: 自受理之日起90日内; 4月1日 + 90 days
    reg_b_021  B    金融机构客户受益所有人识别管理办法 第二十五条: 无法作出准确判断的，不得采取简化或者豁免措施
    res_b_017  C    pack2_text17#c0007 bank 信创 from the periphery to the core; B, D contradicted or
                    unsupported; A's "与ASIC定制服务直接相关" is not in the reports (judged wrong, doubtful)
    res_b_002  AC   pack2_text16#c0007 种源育种, 深加工; pack2_text05#c0060 透明工厂, 产品配方, c0085
                    线下旗舰店; B and D unsupported
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = {"fin_b_011": "ABC", "fin_b_008": "AC", "fc_b_010": "BCD", "fc_b_007": "B", "ins_b_012": "BCD",
        "ins_b_001": "333.20", "reg_b_004": "2026年6月30日", "reg_b_021": "B", "res_b_017": "C", "res_b_002": "AC"}
CALC = {"ins_b_001", "reg_b_004"}
COLS = (("react", "ReAct（逐选项）"), ("rplan", "ReAct + 需求清单"))


def load(qid, mode):
    if mode == "rplan":
        o = json.loads((ROOT / "output" / "react_plan" / f"react_plan_{qid}.json").read_text(encoding="utf-8"))
        return {"answer": o["answer"], "steps": o["steps"], "usage": o["usage"], "seconds": o["seconds"],
                "max_in": max(r["usage"]["prompt_tokens"] for r in o["trace"]), "calls": o["calls"], "md": f"react_plan_{qid}.md"}
    if qid in CALC:
        o = json.loads((ROOT / "output" / "react_compact" / f"react_compact_{qid}.json").read_text(encoding="utf-8"))
        return {"answer": (o.get("final") or {}).get("answer") or "", "steps": o["steps"], "usage": o["usage"], "seconds": o["seconds"],
                "max_in": max(r["usage"]["prompt_tokens"] for r in o["trace"]), "calls": None, "md": f"react_compact_{qid}.md"}
    o = json.loads((ROOT / "output" / "react_relevance" / f"react_relevance_{qid}_none.json").read_text(encoding="utf-8"))
    return {"answer": o["answer"] or "", "steps": o["steps"], "usage": o["usage"], "seconds": o["seconds"],
            "max_in": max(r["usage"]["prompt_tokens"] for r in o["trace"]), "calls": o["calls"], "md": f"react_relevance_{qid}_none.md"}


def judge(qid, answer):
    """(strict, lenient): strict = the string the submission needs; lenient = same value (numbers at
    two decimals, dates without leading zeros)."""
    g, a = GOLD[qid], str(answer or "").strip()
    if a == g:
        return True, True
    if qid in CALC:
        try:
            return False, round(float(a), 2) == round(float(g), 2)
        except ValueError:
            return False, re.sub(r"(?<=年|月)0", "", a) == g
    return False, False


def main():
    L = ["# 10 道 B 榜题：ReAct（逐选项）vs ReAct + 需求清单", "",
         "抽样：五个领域各 2 道、此前都没跑过（seed 20261011）；7 道多选、1 道单选、2 道计算。每种做法各跑一次，模型 qwen3.6-plus，温度 0，思考模式关闭。"
         "标准答案是我们按原文定的，依据见 flows/compare_sample10.py 文件头；fc_b_007 的 C、res_b_017 的 A 存疑。", "",
         "“严格”按提交格式比对字符串；“宽松”只看数值或日期是否一致。", "",
         "| 题 | 标准答案 | " + " | ".join(f"{n} 答案 | 步数 | tokens" for _, n in COLS) + " | token 比 |",
         "|---|---|" + "---|---|---|" * len(COLS) + "---|"]
    tot = {m: {"strict": 0, "lenient": 0, "tokens": 0, "steps": 0} for m, _ in COLS}
    for qid in GOLD:
        runs = {m: load(qid, m) for m, _ in COLS}
        cells = []
        for m, _ in COLS:
            o = runs[m]
            s, l = judge(qid, o["answer"])
            tot[m]["strict"] += s
            tot[m]["lenient"] += l
            tot[m]["tokens"] += o["usage"]["total_tokens"]
            tot[m]["steps"] += o["steps"]
            mark = "✓" if s else ("≈" if l else "✗")
            cells.append(f"{o['answer'] or '（空）'} {mark} | {o['steps']} | {o['usage']['total_tokens']:,}")
        ratio = runs["rplan"]["usage"]["total_tokens"] / runs["react"]["usage"]["total_tokens"]
        L.append(f"| [{qid}]({runs['rplan']['md']}) | {GOLD[qid]} | " + " | ".join(cells) + f" | {ratio:.0%} |")
    n = len(GOLD)
    L += ["", "| 合计 | " + " | ".join(n_ for _, n_ in COLS) + " |", "|---|---|---|",
          "| 答对（严格） | " + " | ".join(f"{tot[m]['strict']} / {n}" for m, _ in COLS) + " |",
          "| 答对（宽松） | " + " | ".join(f"{tot[m]['lenient']} / {n}" for m, _ in COLS) + " |",
          "| 总 tokens | " + " | ".join(f"{tot[m]['tokens']:,}" for m, _ in COLS) + " |",
          "| 平均每题 tokens | " + " | ".join(f"{tot[m]['tokens'] / n:,.0f}" for m, _ in COLS) + " |",
          "| 平均步数 | " + " | ".join(f"{tot[m]['steps'] / n:.1f}" for m, _ in COLS) + " |",
          f"| token 消耗率（清单 / 逐选项） | {tot['rplan']['tokens'] / tot['react']['tokens']:.0%} |  |"]
    out = ROOT / "reports" / "sample10_compare.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
