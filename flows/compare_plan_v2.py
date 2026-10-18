# -*- coding: utf-8 -*-
"""Compare the need-list flow v2 (plan_fill.py) with ReAct on six B questions not run before
(2 multiple choice, 2 single choice, 2 calculation): answers against our own gold, model calls,
tokens. ReAct is react_relevance.py --mask none for the choice questions and react_compact.py
(default: chunks the model cited stay, the others become placeholders) for the calculation
questions, which react_relevance.py does not take.

    python flows/compare_plan_v2.py

Writes reports/plan_compare_v2.md. Gold, set by hand on 2026-10-06 from the documents:
    fin_b_009  BCD  研发费用 (fr_catl_2025#c0085, fr_midea_2025#c0128, income statements agree):
                    CATL 18,606,756 -> 22,146,581 千元 (+19.02%, revenue +17.04%), ratio 5.14% -> 5.23%;
                    Midea 16,232,771 -> 17,787,624 (+9.58%), ratio 3.99% -> 3.90% (-0.09 points);
                    2025 gap 5.23 - 3.90 = 1.33. A wrong (Midea's ratio fell)
    fc_b_012   BCD  fc_text09#c0076: trigger is 85% of the conversion price, not 80% -> A wrong;
                    B, C, D as written there
    fc_b_016   C    fc_text13#c0032 / c0033: 1,950 亿元 for 债务结构调整
    reg_b_020  B    csrc_0027_att1#c0046 第三十二条: 不得将分类评价结果用于广告、宣传、营销等商业目的
    fin_b_018  2.58；7.65  fr_midea_2025: 资产负债率 61.17%, 加权平均净资产收益率 19.70%;
                    1 / (1 - 0.6117) = 2.5753, 19.70 / 2.5753 = 7.6495
    ins_b_018  60   ins_01#c0009: after the pension start the death benefit is the account value at
                    the start minus the pensions paid, zero when they reach it: 0 + (150 - 90) = 60
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = {"fin_b_009": {"answer": "BCD", "verdicts": {"A": "错误", "B": "正确", "C": "正确", "D": "正确"}},
        "fc_b_012": {"answer": "BCD", "verdicts": {"A": "错误", "B": "正确", "C": "正确", "D": "正确"}},
        "fc_b_016": {"answer": "C", "verdicts": {"A": "错误", "B": "错误", "C": "正确", "D": "错误"}},
        "reg_b_020": {"answer": "B", "verdicts": {"A": "错误", "B": "正确", "C": "错误", "D": "错误"}},
        "fin_b_018": {"parts": ["2.58", "7.65"]},
        "ins_b_018": {"parts": ["60"]}}
COLS = (("react", "ReAct"), ("plan", "需求清单流程 v2"), ("rplan", "ReAct + 清单 + 遮蔽"))


def parts_of(answer):
    return [p.strip() for p in re.split(r"[；;]", str(answer or "")) if p.strip()]


def same_part(a, b):
    """Numbers compared at two decimals (a trailing % or 万元 ignored), anything else as text."""
    na, nb = (re.sub(r"[%,\s万元亿]", "", x) for x in (a, b))
    try:
        return round(float(na), 2) == round(float(nb), 2)
    except ValueError:
        return re.sub(r"\s", "", a) == re.sub(r"\s", "", b)


def load(qid, mode):
    if mode == "rplan":
        o = json.loads((ROOT / "output" / "react_plan" / f"react_plan_{qid}.json").read_text(encoding="utf-8"))
        return {"answer": o["answer"], "verdicts": o["verdicts"], "calls": o["steps"], "usage": o["usage"],
                "max_in": max(r["usage"]["prompt_tokens"] for r in o["trace"]), "chunks": len(o["store"]), "seconds": o["seconds"],
                "needs": f"{sum(1 for n in o['needs'].values() if n['value'])} / {len(o['needs'])}",
                "rounds": o["calls"]["search"], "stop": "，".join(f"{k} {v}" for k, v in o["calls"].items() if v), "md": f"react_plan_{qid}.md"}
    if mode == "plan":
        o = json.loads((ROOT / "output" / "plan_fill" / f"plan_fill_{qid}.json").read_text(encoding="utf-8"))
        return {"answer": o["answer"], "verdicts": o["verdicts"], "calls": o["n_calls"], "usage": o["usage"],
                "max_in": max(c["usage"]["prompt_tokens"] for c in o["calls"]), "chunks": len(o["store"]), "seconds": o["seconds"],
                "needs": f"{sum(1 for n in o['needs'].values() if n['value'])} / {len(o['needs'])}"
                         + (f"（追加 {sum(1 for n in o['needs'].values() if n['origin'] != '读题')}）" if any(n["origin"] != "读题" for n in o["needs"].values()) else ""),
                "rounds": len(o["rounds"]), "stop": o["stop"], "md": f"plan_fill_{qid}.md"}
    calc = "parts" in GOLD[qid]
    if calc:
        o = json.loads((ROOT / "output" / "react_compact" / f"react_compact_{qid}.json").read_text(encoding="utf-8"))
        return {"answer": (o.get("final") or {}).get("answer", ""), "verdicts": {}, "calls": o["steps"], "usage": o["usage"],
                "max_in": max(r["usage"]["prompt_tokens"] for r in o["trace"]), "seconds": o["seconds"],
                "chunks": len({u for r in o["trace"] for c in r["calls"] if c.get("chunks") for u in c["chunks"]}),
                "needs": "—", "rounds": "—", "stop": "—", "md": f"react_compact_{qid}.md"}
    o = json.loads((ROOT / "output" / "react_relevance" / f"react_relevance_{qid}_none.json").read_text(encoding="utf-8"))
    return {"answer": o["answer"], "verdicts": {k: c["verdict"] for k, c in o["ledger"]["conclusions"].items()},
            "calls": o["steps"], "usage": o["usage"], "max_in": max(r["usage"]["prompt_tokens"] for r in o["trace"]), "seconds": o["seconds"],
            "chunks": len({u for r in o["trace"] for c in r["calls"] if c["chunks"] for u in c["chunks"]}),
            "needs": "—", "rounds": "—", "stop": o["answered_by"], "md": f"react_relevance_{qid}_none.md"}


def right(qid, o):
    g = GOLD[qid]
    if "parts" in g:
        got = parts_of(o["answer"])
        return len(got) == len(g["parts"]) and all(same_part(a, b) for a, b in zip(got, g["parts"]))
    return o["answer"] == g["answer"]


def main():
    L = ["# 需求清单流程 v2、ReAct + 清单 + 遮蔽 vs ReAct（6 道新题）", "",
         "同一模型（qwen3.6-plus，温度 0，思考模式关闭），每道题每种做法各跑一次。这 6 道题此前都没有跑过，流程也没有按它们调过。"
         "标准答案是我们按原文和计算定的，依据见 flows/compare_plan_v2.py 文件头。", "",
         "ReAct：选择题用 react_relevance.py --mask none（原文全文一直留在上下文），计算题用 react_compact.py 默认模式（模型引用过的块保留全文，其余换成占位符）。"
         "“模型调用”对 ReAct 是轮数（每轮重发整段对话），对清单流程是单次调用的次数（上下文由代码重新拼）。", "",
         "ReAct + 清单 + 遮蔽（react_plan.py）：模型每步自己选工具（plan / search / fill / compute），每步的上下文是代码按状态拼的快照，"
         "fill 判为不相关或已取完值的原文移出上下文。它带着清单流程 v2 之后的两处修正（计算题答案部分由代码从题面定；ready 只看数据够不够），"
         "并在这 6 道题上改过两处代码问题（工具参数的形状、跨条目的摘录核对），所以对它来说这 6 道题不算全新。", ""]
    tot = {m: {"tokens": 0, "calls": 0, "right": 0} for m, _ in COLS}
    qids = list(GOLD)
    for qid in qids:
        g = GOLD[qid]
        runs = {m: load(qid, m) for m, _ in COLS}
        gold = g.get("answer") or "；".join(g["parts"])
        L += [f"## {qid}（标准答案 {gold}）", "", "| 指标 | " + " | ".join(n for _, n in COLS) + " |", "|---|---|---|---|"]

        def line(name, f):
            L.append(f"| {name} | " + " | ".join(str(f(runs[m])) for m, _ in COLS) + " |")
        line("答案", lambda o: f"**{o['answer'] or '（空）'}** {'✓' if right(qid, o) else '✗'}")
        if "verdicts" in g:
            line("选项结论对了几个", lambda o: f"{sum(o['verdicts'].get(k) == v for k, v in g['verdicts'].items())} / 4")
            line("各选项结论", lambda o: " ".join(f"{k}{o['verdicts'].get(k) or '—'}" for k in g["verdicts"]))
        line("模型调用", lambda o: o["calls"])
        line("输入 tokens", lambda o: o["usage"]["prompt_tokens"])
        line("输出 tokens", lambda o: o["usage"]["completion_tokens"])
        line("总 tokens", lambda o: o["usage"]["total_tokens"])
        line("总 tokens / ReAct", lambda o: f"{o['usage']['total_tokens'] / runs['react']['usage']['total_tokens']:.0%}")
        line("单次调用最大输入", lambda o: o["max_in"])
        line("看过的原文块", lambda o: o["chunks"])
        line("需求取到 / 总数", lambda o: o["needs"])
        line("检索轮数（search 次数）", lambda o: o["rounds"])
        line("怎么结束的 / 工具调用", lambda o: o["stop"])
        line("耗时（秒）", lambda o: o["seconds"])
        L += ["", "完整过程：" + "、".join(f"[{n}]({runs[m]['md']})" for m, n in COLS), ""]
        for m, _ in COLS:
            tot[m]["tokens"] += runs[m]["usage"]["total_tokens"]
            tot[m]["calls"] += runs[m]["calls"]
            tot[m]["right"] += right(qid, runs[m])
    L += ["## 合计", "", "| 指标 | " + " | ".join(n for _, n in COLS) + " |", "|---|---|---|---|",
          "| 总 tokens | " + " | ".join(f"{tot[m]['tokens']}（{tot[m]['tokens'] / tot['react']['tokens']:.0%}）" for m, _ in COLS) + " |",
          "| 模型调用 | " + " | ".join(str(tot[m]["calls"]) for m, _ in COLS) + " |",
          "| 答对题数 | " + " | ".join(f"{tot[m]['right']} / {len(qids)}" for m, _ in COLS) + " |"]
    out = ROOT / "reports" / "plan_compare_v2.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
