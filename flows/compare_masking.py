# -*- coding: utf-8 -*-
"""Compare the two context strategies of react_relevance.py (none / relevance) on the questions
run with both: tokens, steps, answers against our own gold, the relevance judgements.

    python flows/compare_masking.py [fin_b_002 fc_b_006]

Writes reports/masking_compare.md. The B list has no official answers; the gold here was set
by hand on 2026-10-06 from the documents:
    fin_b_002  BCD  2024 revenue 777,102,455,000 (sum of the quarters), net profit 40,254,346,000,
                    operating cash flow 133,453,873,000; 2025: 803,964,958,000 / 32,619,022,000 /
                    59,135,544,000. A: margin 5.18% -> 4.06%, relative fall 21.68% (1.12 is the
                    point difference) -> wrong; B -18.97%, C -55.69%, D 17.17% -> 7.36%, 9.82 points
    fc_b_006   B    普联软件 independent directors: "本人及本人配偶、父母、子女将不参与" (fc_text11#c0154),
                    so "未明确其配偶、父母、子女" is wrong; A, C, D agree with fc_text04#c0166 and
                    fc_text05#c1196
    fin_b_003  ABCD key-figure tables fr_catl_2024#c0015, fr_catl_2025#c0019, fr_midea_2024#c0014,
                    fr_midea_2025#c0014: revenue +17.04% / +12.11%; operating cash flow over revenue
                    CATL 26.79% -> 31.44%, Midea 14.86% -> 11.69%; 2025 gap 19.75 points; basic EPS
                    +39.38% / +6.62%, gap 32.76 points
    fc_b_008   ABC  A, B, C are word for word fc_text04#c0927, fc_text11#c0065, fc_text05#c0095; D wrong:
                    only 普联 (c0063) and 本川 (c0093) say holders may not ask for early repayment or a
                    guarantee after a buy-back reduction, 安克 has no such clause, and 本川's
                    exceptions add 用于转换本次可转债
    ins_b_002  BC   ins_02#c0011: max(age ratio x sum assured, account value), 160% from 18 to 41,
                    140% from 41 to 61 -> A 160万 not 130万, B 140万; ins_01#c0009: before the pension
                    start the policy account value -> C 76万; ins_15#c0014: max(premiums - annuities
                    paid, cash value) = max(70, 80) -> D 80万 not 70万
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = {"fin_b_002": {"answer": "BCD", "verdicts": {"A": "错误", "B": "正确", "C": "正确", "D": "正确"}},
        "fc_b_006": {"answer": "B", "verdicts": {"A": "正确", "B": "错误", "C": "正确", "D": "正确"}},
        "fin_b_003": {"answer": "ABCD", "verdicts": {"A": "正确", "B": "正确", "C": "正确", "D": "正确"}},
        "fc_b_008": {"answer": "ABC", "verdicts": {"A": "正确", "B": "正确", "C": "正确", "D": "错误"}},
        "ins_b_002": {"answer": "BC", "verdicts": {"A": "错误", "B": "正确", "C": "正确", "D": "错误"}}}
MODES = (("none", "不遮蔽"), ("relevance", "相关性遮蔽"))


def load(qid, mode):
    return json.loads((ROOT / "output" / "react_relevance" / f"react_relevance_{qid}_{mode}.json").read_text(encoding="utf-8"))


def main(qids):
    L = ["# 上下文策略对比：不遮蔽 vs 相关性遮蔽", "",
         "同一模型（qwen3.6-plus，温度 0，思考模式关闭），同一套登记工具和作答规则，只改检索原文在上下文里的去留。"
         "每道题每种策略各跑一次；温度 0 下同一配置重跑也可能走不同的路，单次结果只能看方向。", "",
         "标准答案是我们自己按原文和计算定的（B 榜没有官方答案），依据见 flows/compare_masking.py 文件头。", ""]
    rows = []
    for qid in qids:
        g = GOLD[qid]
        runs = {m: load(qid, m) for m, _ in MODES}
        base = runs["none"]["usage"]["total_tokens"]
        L += [f"## {qid}（{runs['none']['type']}，标准答案 {g['answer']}）", "", runs["none"]["question"], "",
              "| 指标 | " + " | ".join(n for _, n in MODES) + " |", "|---|---|---|"]
        def line(name, f):
            L.append(f"| {name} | " + " | ".join(str(f(runs[m])) for m, _ in MODES) + " |")
        line("答案", lambda o: f"**{o['answer'] or '（空）'}** {'✓' if o['answer'] == g['answer'] else '✗'}")
        line("选项结论对了几个", lambda o: f"{sum(o['ledger']['conclusions'].get(k, {}).get('verdict') == v for k, v in g['verdicts'].items())} / {len(g['verdicts'])}")
        line("各选项结论", lambda o: " ".join(f"{k}{o['ledger']['conclusions'].get(k, {}).get('verdict', '—')}" for k in g["verdicts"]))
        line("轮数（模型调用次数）", lambda o: o["steps"])
        line("输入 tokens", lambda o: o["usage"]["prompt_tokens"])
        line("输出 tokens", lambda o: o["usage"]["completion_tokens"])
        line("总 tokens", lambda o: o["usage"]["total_tokens"])
        line("总 tokens / 不遮蔽", lambda o: f"{o['usage']['total_tokens'] / base:.0%}")
        line("平均每轮输入", lambda o: round(o["usage"]["prompt_tokens"] / o["steps"]))
        line("单轮最大输入", lambda o: max(r["usage"]["prompt_tokens"] for r in o["trace"]))
        line("工具调用", lambda o: "，".join(f"{k} {v}" for k, v in o["calls"].items() if v))
        line("检索返回的块（去重）", lambda o: len({u for r in o["trace"] for c in r["calls"] if c["chunks"] for u in c["chunks"]}))
        line("被判不相关、移入证据库", lambda o: len({j["chunk"] for j in o["judgments"] if j["relevance"] == "不相关"}) if o["judgments"] else "—")
        line("判断：相关 / 不相关 / 拿不准", lambda o: " / ".join(str(sum(j["relevance"] == x for j in o["judgments"])) for x in ("相关", "不相关", "拿不准")) if o["judgments"] else "—")
        line("judge 调用参数（结束时的字符数）", lambda o: o["trace"][-1]["context"]["composition"].get("judge 参数", 0) or "—")
        line("耗时（秒）", lambda o: o["seconds"])
        steps = max(o["steps"] for o in runs.values())
        L += ["", "每轮输入 tokens：", "", "| 轮 | " + " | ".join(n for _, n in MODES) + " |", "|---|---|---|"]
        for i in range(steps):
            L.append(f"| {i + 1} | " + " | ".join(str(runs[m]["trace"][i]["usage"]["prompt_tokens"]) if i < runs[m]["steps"] else "" for m, _ in MODES) + " |")
        rel = runs["relevance"]
        if rel["judgments"]:
            L += ["", "相关性遮蔽这一轮的判断：", "", "| 步 | 块 | 判断 | 针对 | 理由 | 后来被结论引用 |", "|---|---|---|---|---|---|"]
            cited = {e for c in rel["ledger"]["conclusions"].values() for e in c["evidence"]}
            L += [f"| {j['step']} | `{j['chunk']}` | {j['relevance']} | {j['for']} | {j['reason']} | {'是' if j['chunk'] in cited else ''} |" for j in rel["judgments"]]
        crashed = ROOT / "logs" / "relevance_demo" / f"{qid}_relevance_crashed.log"
        if crashed.exists():                        # a first relevance run that died in calc (string expressions, fixed since)
            steps_seen = re.findall(r"^--- 第 (\d+) 步（输入 (\d+)，输出 (\d+)，累计 (\d+)）", crashed.read_text(encoding="utf-8"), re.M)
            calls = re.findall(r"^调用：(\w+)", crashed.read_text(encoding="utf-8"), re.M)
            L += ["", f"另有一次相关性遮蔽的运行在第 {steps_seen[-1][0]} 步因脚本错误中断（模型给 calc 传了字符串表达式，已修复后重跑，上表是重跑的结果）。"
                  f"中断时它已经用了 {steps_seen[-1][3]} tokens，还没有登记任何结论；单轮输入从 {steps_seen[0][1]} 涨到 {max(int(s[1]) for s in steps_seen)}；"
                  "调用：" + "，".join(f"{n} {calls.count(n)}" for n in dict.fromkeys(calls)) + "。日志：logs/relevance_demo/" + crashed.name]
        L += ["", "完整过程（每一步模型写的文字、全部调用和参数、返回的原文、系统动作）：", ""]
        L += [f"- {n}：[react_relevance_{qid}_{m}.md](react_relevance_{qid}_{m}.md)" for m, n in MODES]
        L += [""]
        rows.append((qid, runs))
    L += ["## 合计", "", "| 指标 | " + " | ".join(n for _, n in MODES) + " |", "|---|---|---|"]
    tot = {m: sum(r[m]["usage"]["total_tokens"] for _, r in rows) for m, _ in MODES}
    L.append("| 总 tokens | " + " | ".join(f"{tot[m]}（{tot[m] / tot['none']:.0%}）" for m, _ in MODES) + " |")
    L.append("| 轮数 | " + " | ".join(str(sum(r[m]["steps"] for _, r in rows)) for m, _ in MODES) + " |")
    L.append("| 答对题数 | " + " | ".join(f"{sum(r[m]['answer'] == GOLD[q]['answer'] for q, r in rows)} / {len(rows)}" for m, _ in MODES) + " |")
    L.append("| 选项结论对的个数 | " + " | ".join(
        f"{sum(r[m]['ledger']['conclusions'].get(k, {}).get('verdict') == v for q, r in rows for k, v in GOLD[q]['verdicts'].items())} / "
        f"{sum(len(GOLD[q]['verdicts']) for q, _ in rows)}" for m, _ in MODES) + " |")
    out = ROOT / "reports" / "masking_compare.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:] or ["fin_b_002", "fc_b_006"])
