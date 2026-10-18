# -*- coding: utf-8 -*-
"""Compare the need-list flow (plan_fill.py) with the two ReAct runs of react_relevance.py
(none / relevance) on the questions run with all three: answers against the gold of
compare_masking.py, model calls, tokens.

    python flows/compare_plan.py [fin_b_002 fc_b_006 fin_b_003 fc_b_008 ins_b_002]

Writes reports/plan_compare.md.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "flows"))
from compare_masking import GOLD  # noqa: E402

COLS = (("none", "ReAct 不遮蔽"), ("relevance", "ReAct 相关性遮蔽"), ("plan", "需求清单流程"))


def load(qid, mode):
    if mode == "plan":
        o = json.loads((ROOT / "output" / "plan_fill" / f"plan_fill_{qid}.json").read_text(encoding="utf-8"))
        return {"answer": o["answer"], "verdicts": o["verdicts"], "calls": o["n_calls"], "usage": o["usage"],
                "max_in": max(c["usage"]["prompt_tokens"] for c in o["calls"]), "chunks": len(o["store"]),
                "masked": len(o["masked"]), "seconds": o["seconds"],
                "needs": f"{sum(1 for n in o['needs'].values() if n['value'])} / {len(o['needs'])}",
                "rounds": len(o["rounds"]), "raw": o}
    o = json.loads((ROOT / "output" / "react_relevance" / f"react_relevance_{qid}_{mode}.json").read_text(encoding="utf-8"))
    return {"answer": o["answer"], "verdicts": {k: c["verdict"] for k, c in o["ledger"]["conclusions"].items()},
            "calls": o["steps"], "usage": o["usage"], "max_in": max(r["usage"]["prompt_tokens"] for r in o["trace"]),
            "chunks": len({u for r in o["trace"] for c in r["calls"] if c["chunks"] for u in c["chunks"]}),
            "masked": len({j["chunk"] for j in o["judgments"] if j["relevance"] == "不相关"}) if o["judgments"] else "—",
            "seconds": o["seconds"], "needs": "—", "rounds": "—", "raw": o}


def main(qids):
    L = ["# 需求清单流程 vs ReAct", "",
         "同一模型（qwen3.6-plus，温度 0，思考模式关闭）。每道题每种做法各跑一次。标准答案是我们按原文和计算定的，依据见 flows/compare_masking.py 文件头。", "",
         "“模型调用”对 ReAct 是轮数（每轮把整段对话重发一次），对需求清单流程是单次调用的次数（每次的上下文由代码重新拼，不累积）。", ""]
    tot = {m: {"tokens": 0, "calls": 0, "right": 0, "opts": 0, "n": 0} for m, _ in COLS}
    for qid in qids:
        g = GOLD[qid]
        runs = {m: load(qid, m) for m, _ in COLS}
        base = runs["none"]["usage"]["total_tokens"]
        L += [f"## {qid}（标准答案 {g['answer']}）", "", "| 指标 | " + " | ".join(n for _, n in COLS) + " |", "|---|---|---|---|"]

        def line(name, f):
            L.append(f"| {name} | " + " | ".join(str(f(runs[m])) for m, _ in COLS) + " |")
        right = lambda o: sum(o["verdicts"].get(k) == v for k, v in g["verdicts"].items())
        line("答案", lambda o: f"**{o['answer'] or '（空）'}** {'✓' if o['answer'] == g['answer'] else '✗'}")
        line("选项结论对了几个", lambda o: f"{right(o)} / {len(g['verdicts'])}")
        line("各选项结论", lambda o: " ".join(f"{k}{o['verdicts'].get(k) or '—'}" for k in g["verdicts"]))
        line("模型调用", lambda o: o["calls"])
        line("输入 tokens", lambda o: o["usage"]["prompt_tokens"])
        line("输出 tokens", lambda o: o["usage"]["completion_tokens"])
        line("总 tokens", lambda o: o["usage"]["total_tokens"])
        line("总 tokens / ReAct 不遮蔽", lambda o: f"{o['usage']['total_tokens'] / base:.0%}")
        line("单次调用最大输入", lambda o: o["max_in"])
        line("看过的原文块", lambda o: o["chunks"])
        line("判为不相关、移出的块", lambda o: o["masked"])
        line("需求取到 / 总数", lambda o: o["needs"])
        line("检索轮数", lambda o: o["rounds"])
        line("耗时（秒）", lambda o: o["seconds"])
        L += ["", f"完整过程：[ReAct 不遮蔽](react_relevance_{qid}_none.md)、[ReAct 相关性遮蔽](react_relevance_{qid}_relevance.md)、"
              f"[需求清单流程](plan_fill_{qid}.md)", ""]
        for m, _ in COLS:
            o = runs[m]
            tot[m]["tokens"] += o["usage"]["total_tokens"]
            tot[m]["calls"] += o["calls"]
            tot[m]["right"] += o["answer"] == g["answer"]
            tot[m]["opts"] += right(o)
            tot[m]["n"] += len(g["verdicts"])
    L += ["## 合计", "", "| 指标 | " + " | ".join(n for _, n in COLS) + " |", "|---|---|---|---|",
          "| 总 tokens | " + " | ".join(f"{tot[m]['tokens']}（{tot[m]['tokens'] / tot['none']['tokens']:.0%}）" for m, _ in COLS) + " |",
          "| 模型调用 | " + " | ".join(str(tot[m]["calls"]) for m, _ in COLS) + " |",
          "| 答对题数 | " + " | ".join(f"{tot[m]['right']} / {len(qids)}" for m, _ in COLS) + " |",
          "| 选项结论对的个数 | " + " | ".join(f"{tot[m]['opts']} / {tot[m]['n']}" for m, _ in COLS) + " |"]
    out = ROOT / "reports" / "plan_compare.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:] or ["fin_b_002", "fc_b_006", "fin_b_003", "fc_b_008", "ins_b_002"])
