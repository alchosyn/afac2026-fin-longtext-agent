# -*- coding: utf-8 -*-
"""ReAct demo on one B-list question (a calculation question, or a choice / true-false question
whose options are judged against the documents): Qwen decides every step itself, with two
tools, until it gives the answer.

    search   the project's BM25 chunk search over the documents the question names; a query
             that names a company and a year is searched in that report alone (agent.search.narrow)
    calc     exact arithmetic (agent.tools.calc)

    python flows/react_demo.py fin_b_014 [--steps 12] [--top 5] [--tag run2]
    python flows/react_demo.py res_b_010 --steps 20

Prints the trace and writes output/react_demo/react_demo_<qid>[_<tag>].json: every step with what the model
wrote, its tool calls, the tool results and the token usage of the call.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.docview import Doc, load_cards                   # noqa: E402
from agent.llm import Qwen                                  # noqa: E402
from agent.retrieval import bm25_scores                     # noqa: E402
from agent.scope import load_b_questions, scope             # noqa: E402
from agent.search import narrow                             # noqa: E402
from agent.rerank import Reranker                           # noqa: E402
from agent.tools import CALC_SCHEMA, run_tool               # noqa: E402

MAX_BODY = 1200         # characters of one chunk shown to the model
POOL, KEEP = 20, 8      # --rerank: BM25 candidates the cross-encoder ranks, chunks it returns
SEARCH_SCHEMA = {"type": "function", "function": {
    "name": "search",
    "description": ("在本题涉及的文档原文中做关键词检索，返回最相关的若干个原文块（表格或段落），每块带编号、所属文档和章节位置。"
                    "查询里写上公司简称和年份（如“比亚迪 2024年 营业收入”）时，只在该公司该年度的报告里找。"
                    "一次只查一件事（一个指标、一个事实）效果最好；没找到就换说法（指标全称、所在表格的名称、原文可能的用词）再查。"),
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "检索词，例如“比亚迪 2025年 经营活动产生的现金流量净额”"},
        "top": {"type": "integer", "description": "返回几块，默认 5，最大 8"}},
        "required": ["query"]}}}
SYSTEM_CHOICE = """你是金融文档问答助手，用工具一步步回答一道{kind}：逐个核实选项，选出与文档一致的。
本题涉及的文档：
{docs}
规则：
1. 每个选项的判断都必须以 search 返回的原文为依据，不能凭常识或记忆。一个选项涉及多个行业或公司时，每一个都要找到对应的原文。
2. 每次调用工具前，先用一两句话写明：已经核实了什么、还缺什么、这一步要做什么。
3. 题面用类别或代称（“某龙头”“一家公司”）时，先找出原文里对应的具体对象。
4. 原文明确支持的选项判“正确”；与原文矛盾，或说法超出原文（范围更大、程度更绝对、因果不同）的判“错误”；找不到任何相关原文的写“无依据”，不要猜。
5. 需要算术时用 calc。
6. 核实完所有选项后，不再调用工具，只输出一个 JSON：
{{"answer": "正确选项的字母，按顺序连写", "options": {{"A": {{"verdict": "正确|错误|无依据", "evidence": ["依据的块编号"], "reason": "一句话依据"}}}}}}"""
SYSTEM = """你是金融文档问答助手，用工具一步步回答一道计算题。
本题涉及的文档：
{docs}
规则：
1. 题目需要的每个数字都必须来自 search 返回的原文，不能凭记忆或估计；找不到就换查询再找。
2. 每次调用工具前，先用一两句话写明：已经拿到了什么、还缺什么、这一步要做什么。
3. 取数时核对指标名称、期间、口径（合并或母公司）和单位；优先用表格里的精确金额，不用正文里四舍五入的数。
4. 所有算术都用 calc 完成，中间结果不要四舍五入。
5. 数字齐全并算完后，不再调用工具，只输出一个 JSON：
{{"answer": "按题目要求的格式", "inputs": [{{"name": "指标（公司、期间）", "value": "原文数值", "unit": "单位", "chunk": "来源块编号"}}], "steps": "计算过程"}}"""


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("qid")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--rerank", action="store_true", help=f"search = BM25 top {POOL}, reranked by the cross-encoder, the best {KEEP} returned")
    args = ap.parse_args(argv)
    q = next(x for x in load_b_questions() if x["qid"] == args.qid)
    sc, cards, docs = scope(q), load_cards(), {}

    def unit(uid):
        d = uid.split("#")[0]
        docs.setdefault(d, Doc(d, cards[d]))
        return docs[d].by_id[uid]

    rr = Reranker() if args.rerank else None

    def search(a):
        where, text = narrow(q, str(a.get("query", "")), sc)
        if rr:                                              # BM25 top POOL, the cross-encoder picks the KEEP it returns
            pool = [u for u, s in bm25_scores(text, where)[:POOL] if s > 0]
            hits = rr.top(str(a.get("query", "")), pool, KEEP)
        else:
            hits = [(u, s) for u, s in bm25_scores(text, where)[:min(int(a.get("top") or args.top), 8)] if s > 0]
        if not hits:
            return "没有命中，换一个查询。"
        out = []
        for u, _ in hits:
            x = unit(u)
            out.append(f"[{u}]《{cards[u.split('#')[0]]['title']}》{x['path'] or ''}\n{x['body'][:MAX_BODY]}")
        return "\n\n".join(out)

    titles = "\n".join(f"- 《{cards[d]['title']}》" for d in sc["docs"])
    if not sc["named"]:                                             # nothing named: the whole domain is searched
        titles = ("（题面没有点名文档，search 在下面这些文档里找）\n" + titles if len(sc["docs"]) <= 20
                  else f"（题面没有点名文档，search 在本领域的 {len(sc['docs'])} 份文档里找）")
    options = "".join(f"\n{k}. {v.strip()}" for k, v in (q.get("options") or {}).items())
    system = SYSTEM_CHOICE.format(kind=q["type"], docs=titles) if options else SYSTEM.format(docs=titles)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": q["question"].strip() + options}]
    llm, trace, usage, t0, final = Qwen(log_path=ROOT / "logs" / f"react_demo_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"), [], [0, 0], time.time(), None
    where = sc["docs"] if sc["named"] else f"whole domain ({len(sc['docs'])})"
    print(f"{q['qid']} {q['question'].strip()}{options}\n文档：{where}\n")
    for step in range(1, args.steps + 2):
        last = step > args.steps                                    # out of steps: one call without tools
        if last:
            messages.append({"role": "user", "content": "步数已用完。不再调用工具，根据已有信息直接输出最终 JSON。"})
        resp = llm.client.chat.completions.create(
            model=llm.model, messages=messages, temperature=0, max_tokens=1500, extra_body={"enable_thinking": False},
            **({} if last else {"tools": [SEARCH_SCHEMA, CALC_SCHEMA]}))
        msg = resp.choices[0].message
        usage[0] += resp.usage.prompt_tokens
        usage[1] += resp.usage.completion_tokens
        calls = msg.tool_calls or []
        row = {"step": step, "content": msg.content or "", "calls": [],
               "usage": {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}}
        print(f"--- 第 {step} 步（输入 {resp.usage.prompt_tokens} / 输出 {resp.usage.completion_tokens} tokens）")
        if msg.content:
            print("想法：" + msg.content.strip())
        if not calls:
            final = msg.content or ""
            trace.append(row)
            break
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
        for c in calls:
            try:
                a = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                a = {}
            result = search(a) if c.function.name == "search" else run_tool(c.function.name, c.function.arguments, None)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
            row["calls"].append({"name": c.function.name, "arguments": a, "result": result})
            print(f"调用：{c.function.name} {json.dumps(a, ensure_ascii=False)}")
            print("结果：" + (" | ".join(l.split("\n")[0][:90] for l in result.split("\n\n")) if c.function.name == "search" else result.replace("\n", "；")))
        trace.append(row)
    out = {"qid": q["qid"], "question": q["question"].strip(), "docs": sc["docs"] if sc["named"] else None, "model": llm.model,
           "final": final, "steps": len(trace), "searches": sum(c["name"] == "search" for r in trace for c in r["calls"]),
           "calcs": sum(c["name"] == "calc" for r in trace for c in r["calls"]),
           "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)},
           "seconds": round(time.time() - t0, 1), "trace": trace}
    if rr:
        out["rerank"] = {"pool": POOL, "keep": KEEP, "pairs": rr.pairs}
        rr.close()
    path = ROOT / "output" / "react_demo" / f"react_demo_{q['qid']}{'_' + args.tag if args.tag else ''}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{out['steps']} 步，search {out['searches']} 次，calc {out['calcs']} 次，{out['seconds']} 秒，"
          f"tokens 输入 {usage[0]} + 输出 {usage[1]} = {sum(usage)} -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
