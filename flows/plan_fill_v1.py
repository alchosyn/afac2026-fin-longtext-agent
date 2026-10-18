# -*- coding: utf-8 -*-
"""Answer one B-list choice question from a list of data needs instead of a ReAct loop (demo,
2026-10-06). Every model call is a single call whose context the code builds afresh; nothing
accumulates from call to call.

    0 plan     one call reads the question and lists the data needs: what value (company,
               period, scope), number or passage, unit, two or three search queries, the
               options that use it
    1 search   code only: every query of every unmet need through agent.search.narrow + BM25,
               top TOP_PER_QUERY each, merged round-robin so that each need gets its best chunk
               first, duplicates and chunks set aside earlier dropped, at most MAX_NEW new chunks
    2 fill     one call over the unmet needs and the chunks in play: which needs each chunk
               serves (none: 不相关), the value of every need found (copied from the chunk; the
               code checks it stands there), new queries for the needs not found (recall poor).
               Chunks that serve no unmet need leave the context (不相关 ones are masked into
               the evidence store, the others are done with)
    3 check    all needs filled -> one call writes Python over the filled values that works
               out every option and sets its verdict; the code runs it and picks the options
               by the direction of the stem (agent.decide). Needs still open -> back to 1 with
               the new queries
    4 limit    after MAX_ROUNDS fill calls with needs still open the final call is made anyway:
               options resting on a missing value are 无依据, a single-choice question is
               answered by elimination when that leaves one option, and the missing values are
               named

    python flows/plan_fill_v1.py fin_b_003   (the version of the 5-question run of 2026-10-06; plan_fill.py is v2)

Writes output/plan_fill/plan_fill_<qid>.json (every call: system, user, response, usage; retrieval;
needs; code and its output) and .md (the same as a readable record).
"""
import json
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.decide import caveat, choose, direction          # noqa: E402
from agent.docview import Doc, load_cards                   # noqa: E402
from agent.llm import Qwen, parse_json                      # noqa: E402
from agent.retrieval import bm25_scores                     # noqa: E402
from agent.scope import load_b_questions, scope             # noqa: E402
from agent.search import narrow                             # noqa: E402

MAX_BODY = 1200             # characters of one chunk shown to the model
TOP_PER_QUERY = 3           # BM25 hits kept per query
QUERIES_PER_NEED = 2        # queries of one need run in one round
MAX_NEW = 40                # new chunks per round (24 cut the third hit of every query on fin_b_003, 2026-10-06)
MAX_ROUNDS = 3              # fill calls
VERDICTS = ("正确", "错误", "无依据")

SYSTEM_PLAN = """你是金融文档问答助手。先读题，列出判断各选项需要从文档里取得的数据（需求清单），之后系统按清单检索原文。
本题涉及的文档：
{docs}
要求：
1. 只列文档里能直接读到的值：财务报表和指标表里的原始数（营业收入、净利润等），表格直接给出的比率或增减幅度，或条款、承诺的原文。需要计算才能得到的值（比率、增幅、差额）不要列，由后面的代码计算。
2. 每项写明主体（公司或产品）、期间、口径（合并或母公司等），一个值只列一次，注明被哪些选项用到。题面已经给出的数不用列。
3. kind：数值 或 原文。unit 按原文的单位写（年报表格常用千元、元），不确定就留空，不要自己定一个单位；原文需求写清要找哪一段规定。
4. queries：2 到 3 个检索词。第一个写“公司简称或产品名 年份 指标或条款名”（如“宁德时代 2025年 营业收入”“国寿增益宝 身故保险金”），后面的换成表名、章节名或同义说法（如“主要会计数据 营业收入”“合并现金流量表 经营活动产生的现金流量净额”）。不要写“年度报告”“年报”“条款”这类每一页都有的词。
只输出一个 JSON：{{"needs": [{{"id": "n1", "what": "...", "kind": "数值", "unit": "千元", "queries": ["...", "..."], "options": ["A", "B"]}}]}}"""

SYSTEM_FILL = """你是金融文档问答助手。系统按数据需求检索到了一批原文块，请一次完成三件事：
1. chunks：逐块写出它能满足哪些需求编号；一个都满足不了的写空列表（系统会把它移出上下文）。
2. values：找到的需求逐个填值。数值照抄原文里的数字（保留千分位和小数），不换算单位，单位照抄原文表头或正文（如 千元）；原文需求照抄关键句（不超过 200 字）。核对主体、期间、口径（合并或母公司）和单位，对不上的不填。系统会核对所填的值是否真在该块里。
3. missing：这批块里找不到的需求，写原因，并给 2 个换了说法的新检索词（换表名、科目名、章节名或同义说法）。
只输出一个 JSON：{"chunks": {"块编号": ["n1"]}, "values": [{"need": "n1", "value": "...", "unit": "...", "chunk": "块编号", "note": "期间、口径等说明"}], "missing": [{"need": "n2", "reason": "...", "queries": ["...", "..."]}]}"""

SYSTEM_FINAL = """你是金融文档问答助手。判断各选项需要的数据已经从原文取到（系统已核对与原文一致），请写一段 Python 代码，一次把所有选项算完并给出结论。
代码约定：
- V 是 dict：需求编号 -> 值。数值需求是 float（单位见表），原文需求是摘录字符串。代码里不要手写这些数，用 V["n1"] 取用。
- 计算的中间结果存进 dict R（键用中文名），每个选项的结论写进 dict verdicts，取值 "正确" / "错误" / "无依据"，一句话理由写进 dict why。
- 结论只表示选项的说法与原文和计算结果是否一致。题干问的是正确还是错误的选项都不影响结论，选哪些由系统按题干决定。
- 选项里带“约”的数，按选项写到的小数位把计算结果四舍五入后比较；百分数和“个百分点”注意乘 100。
- 依赖“没找到的数据”的选项判 "无依据"，除非其余数据已足以判断。
- 原文类的选项：读摘录判断，直接给 verdicts 赋值，并在 why 里写明依据哪句原文。
只输出一个 ```python 代码块，不要输出别的。"""


def squash(s):
    return re.sub(r"[\s|]+", "", str(s))


def loose(s):
    """Letters, digits and CJK only: a copied passage matches across punctuation and spacing."""
    return re.sub(r"[^\w]", "", str(s))


def number(s):
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(s))
    return (m.group(0), float(m.group(0).replace(",", ""))) if m else (None, None)


def run_code(code, values, timeout=20):
    """Run the model's code in a fresh interpreter with V bound; ({R, verdicts, why, stdout}, error)."""
    wrapper = ("import json, math, sys\nV = json.loads(sys.stdin.read())\nR, verdicts, why = {}, {}, {}\n"
               + code + "\nprint('<<<OUT>>>' + json.dumps({'R': {str(k): (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) "
               "for k, v in R.items()}, 'verdicts': verdicts, 'why': why}, ensure_ascii=False))\n")
    try:
        p = subprocess.run([sys.executable, "-c", wrapper], input=json.dumps(values, ensure_ascii=False), capture_output=True,
                           text=True, encoding="utf-8", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "代码运行超时"
    if p.returncode != 0 or "<<<OUT>>>" not in p.stdout:
        return None, (p.stderr or p.stdout)[-1500:]
    out, printed = p.stdout.split("<<<OUT>>>")[-1], p.stdout.split("<<<OUT>>>")[0]
    res = json.loads(out)
    res["stdout"] = printed
    return res, None


def main(argv):
    qid = argv[0]
    q = next(x for x in load_b_questions() if x["qid"] == qid)
    options = q.get("options") or {}
    sc, cards, docs = scope(q), load_cards(), {}
    asked, clause, word = direction(q["question"])
    llm = Qwen(log_path=ROOT / "logs" / f"plan_fill_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    t0 = time.time()
    calls, rounds = [], []
    store, active, masked, done_with = {}, [], set(), set()

    def unit(uid):
        d = uid.split("#")[0]
        docs.setdefault(d, Doc(d, cards[d]))
        return docs[d].by_id[uid]

    def chunk_text(u):
        s = store[u]
        return f"[{u}]《{s['title']}》{s['path']}\n{s['text']}"

    def call(stage, system, user, max_tokens):
        content, usage = llm.chat(system, user, max_tokens=max_tokens, tag=f"{qid}:{stage}")
        calls.append({"stage": stage, "system": system, "user": user, "response": content, "usage": usage})
        print(f"--- 调用 {len(calls)}：{stage}（输入 {usage['prompt_tokens']}，输出 {usage['completion_tokens']}）")
        return content

    stem = q["question"].strip() + "".join(f"\n{k}. {v.strip()}" for k, v in options.items())
    titles = "\n".join(f"- 《{cards[d]['title']}》" for d in sc["docs"][:20]) + (f"\n（共 {len(sc['docs'])} 份）" if len(sc["docs"]) > 20 else "")
    print(f"{qid} {stem}\n方向：选判为“{asked}”的选项\n")

    # ------------------------------------------------------------------ 0 plan
    plan = parse_json(call("plan", SYSTEM_PLAN.format(docs=titles), stem, 2000))
    needs = {}
    for i, n in enumerate(plan.get("needs") or [], 1):
        nid = str(n.get("id") or f"n{i}")
        needs[nid] = {"id": nid, "what": str(n.get("what", "")), "kind": "原文" if str(n.get("kind")) == "原文" else "数值",
                      "unit": str(n.get("unit") or ""), "options": [str(o) for o in n.get("options") or []],
                      "queries": [str(x) for x in n.get("queries") or []][:3], "value": None, "history": []}
    print("需求清单：\n" + "\n".join(f"  {n['id']} {n['what']}（{n['kind']}，{n['unit']}）选项 {''.join(n['options'])}｜{n['queries']}" for n in needs.values()))
    open_ = lambda: [n for n in needs.values() if n["value"] is None]

    # ------------------------------------------------------------------ 1-3 search / fill
    for rnd in range(1, MAX_ROUNDS + 1):
        unmet = open_()
        if not unmet:
            break
        # 1 search: all queries of all unmet needs in one batch, merged round-robin
        lists, log = [], []
        for n in unmet:
            for query in n["queries"][:QUERIES_PER_NEED]:
                where, text = narrow(q, query, sc)
                got = [u for u, s in bm25_scores(text, where)[:TOP_PER_QUERY] if s > 0]
                lists.append(got)
                log.append({"need": n["id"], "query": query, "searched": text.strip(), "docs": len(where), "chunks": got})
        new = []
        for pos in range(TOP_PER_QUERY):
            for got in lists:
                if pos < len(got) and got[pos] not in new and got[pos] not in active and got[pos] not in masked and got[pos] not in done_with:
                    new.append(got[pos])
        new = new[:MAX_NEW]
        for u in new:
            x = unit(u)
            store[u] = {"title": cards[u.split("#")[0]]["title"], "path": x["path"] or "", "text": x["body"][:MAX_BODY], "type": x["type"], "round": rnd}
        in_play = active + new
        print(f"第 {rnd} 轮检索：{len(log)} 个查询，新块 {len(new)}，带入上一轮仍相关的块 {len(active)}")
        row = {"round": rnd, "queries": log, "new": new, "carried": list(active), "in_play": in_play}
        rounds.append(row)
        if not in_play:
            row["note"] = "没有新块可看"
            break
        # 2 fill
        user = (f"题目：\n{stem}\n\n还没取到的数据需求：\n"
                + "\n".join(f"- {n['id']}｜{n['what']}｜{n['kind']}" + (f"｜单位 {n['unit']}" if n["unit"] else "") + f"｜用于选项 {'、'.join(n['options'])}"
                            + (f"｜上一轮：{n['history'][-1]}" if n["history"] else "") for n in unmet)
                + f"\n\n原文块（共 {len(in_play)} 块）：\n\n" + "\n\n".join(chunk_text(u) for u in in_play))
        fill = parse_json(call(f"fill{rnd}", SYSTEM_FILL, user, 3000))
        serves = {str(k).strip("[] "): [str(x) for x in v or []] for k, v in (fill.get("chunks") or {}).items()}
        checks = []
        for v in fill.get("values") or []:
            nid, ch = str(v.get("need", "")), str(v.get("chunk", "")).strip("[] ")
            if nid not in needs or needs[nid]["value"] is not None:
                continue
            n = needs[nid]
            if ch not in store:
                ok, why = False, f"块 {ch} 不在本轮原文里"
            elif n["kind"] == "数值":
                text, val = number(v.get("value"))
                ok = bool(text) and squash(text) in squash(store[ch]["text"])
                why = "" if ok else f"块 {ch} 里没有数值 {v.get('value')}"
            else:
                ok = len(loose(v.get("value"))) >= 4 and loose(v.get("value")) in loose(store[ch]["text"])
                why = "" if ok else f"摘录不在块 {ch} 里"
            checks.append({"need": nid, "value": v.get("value"), "unit": v.get("unit"), "chunk": ch, "ok": ok, "why": why})
            if ok:
                n["value"] = {"text": str(v.get("value")), "number": number(v.get("value"))[1] if n["kind"] == "数值" else None,
                              "unit": str(v.get("unit") or n["unit"]), "chunk": ch, "note": str(v.get("note") or ""), "round": rnd}
            else:
                n["history"].append(f"填了 {v.get('value')}（{ch}），核对未通过：{why}")
        poor = []
        for m in fill.get("missing") or []:
            nid = str(m.get("need", ""))
            if nid in needs and needs[nid]["value"] is None:
                qs = [str(x) for x in m.get("queries") or []][:3]
                needs[nid]["history"].append(f"没找到：{m.get('reason', '')}")
                if qs:
                    needs[nid]["queries"] = qs
                poor.append({"need": nid, "reason": str(m.get("reason", "")), "queries": qs})
        still = {n["id"] for n in open_()}
        for n in open_():                               # a need neither filled nor reported: search its queries again
            if not any(p["need"] == n["id"] for p in poor):
                poor.append({"need": n["id"], "reason": "填值调用没有提到", "queries": n["queries"]})
        # context: keep the chunks that serve a need still open
        keep = [u for u in in_play if set(serves.get(u, [])) & still]
        irrelevant = [u for u in in_play if not serves.get(u)]
        used = [u for u in in_play if u not in keep and u not in irrelevant]
        masked.update(irrelevant)
        done_with.update(used)
        active = keep
        row.update({"serves": serves, "checks": checks, "missing": poor, "kept": keep, "masked": irrelevant, "used": used,
                    "filled": [c["need"] for c in checks if c["ok"]]})
        print(f"  填值：通过 {sum(c['ok'] for c in checks)}，未通过 {sum(not c['ok'] for c in checks)}；仍缺 {sorted(still)}；"
              f"不相关移出 {len(irrelevant)}，已用完移出 {len(used)}，保留 {len(keep)}")

    # ------------------------------------------------------------------ 3/4 final
    missing = open_()
    V = {nid: (n["value"]["number"] if n["kind"] == "数值" else n["value"]["text"]) for nid, n in needs.items() if n["value"]}
    table = "\n".join(f"| {nid} | {n['what']} | {n['value']['text']} | {n['value']['unit']} | {n['value']['chunk']} | {'、'.join(n['options'])} |"
                      for nid, n in needs.items() if n["value"] and n["kind"] == "数值")
    quotes = "\n".join(f"- {nid}｜{n['what']}（{n['value']['chunk']}，用于选项 {'、'.join(n['options'])}）：“{n['value']['text']}”"
                       for nid, n in needs.items() if n["value"] and n["kind"] == "原文")
    user = (f"题目：\n{stem}\n\n已取到的数值（V[编号] 是去掉千分位的 float，单位按表）：\n| 编号 | 内容 | 原文数值 | 单位 | 来源块 | 用于选项 |\n|---|---|---|---|---|---|\n{table or '（无）'}\n\n"
            f"已取到的原文摘录（V[编号] 是摘录字符串）：\n{quotes or '（无）'}\n\n"
            f"没找到的数据：\n" + ("\n".join(f"- {n['id']}｜{n['what']}｜用于选项 {'、'.join(n['options'])}" for n in missing) or "（无）"))
    final, error, code = None, None, ""
    for attempt in range(2):
        content = call("final" if attempt == 0 else "final_retry", SYSTEM_FINAL,
                       user if attempt == 0 else user + f"\n\n上一次的代码：\n```python\n{code}\n```\n运行出错：\n{error}\n请改正后重新输出完整代码。", 3000)
        m = re.search(r"```(?:python)?\s*\n(.*?)```", content, re.S)
        code = m.group(1) if m else content
        final, error = run_code(code, V)
        if final and not error:
            bad = [k for k in options if final["verdicts"].get(k) not in VERDICTS]
            if not bad:
                break
            error, final = f"verdicts 缺少或取值不对的选项：{bad}（每个选项都要给 正确 / 错误 / 无依据）", None
        print("代码出错：" + str(error)[:300])
    verdicts = {k: (final["verdicts"].get(k) if final else None) for k in options}
    knowns = [v for v in V.values() if isinstance(v, float)]
    literals = sorted({float(x.replace(",", "")) for x in re.findall(r"\d[\d,]{4,}(?:\.\d+)?", code)
                       if not any(abs(float(x.replace(",", "")) - k) < 1e-6 * max(1, abs(k)) for k in knowns)})
    chosen, unfounded = choose(asked, verdicts)
    how = "代码按题干方向从各选项结论里挑出判为“" + asked + "”的选项"
    if q["type"] == "单选题" and not chosen and len(unfounded) == 1 and missing:
        chosen, how = unfounded, how + "；没有选项符合，按排除法选唯一缺数据的选项"
    answer = "".join(chosen)
    note = caveat(asked, [k for k in unfounded if k not in chosen])
    if missing:
        note = (note + "；" if note else "") + "没找到的值：" + "、".join(f"{n['id']} {n['what']}" for n in missing)
    print(f"\n结论：{verdicts}\n答案：{answer}（{how}）" + (f"\n注意：{note}" if note else ""))
    out = {"qid": qid, "type": q["type"], "question": q["question"].strip(), "options": options, "model": llm.model,
           "direction": {"asked": asked, "clause": clause, "word": word}, "needs": needs, "rounds": rounds,
           "code": code, "code_output": final, "code_error": error, "literals_in_code": literals,
           "verdicts": verdicts, "answer": answer, "answered_by": how, "caveat": note,
           "calls": calls, "n_calls": len(calls), "usage": llm.usage, "seconds": round(time.time() - t0, 1),
           "store": store, "masked": sorted(masked), "done_with": sorted(done_with)}
    base = ROOT / "output" / "plan_fill" / f"plan_fill_{qid}"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    base.with_suffix(".md").write_text(report(out), encoding="utf-8")
    u = llm.usage
    print(f"{len(calls)} 次调用，tokens {u['prompt_tokens']} + {u['completion_tokens']} = {u['total_tokens']}，{out['seconds']} 秒 -> {base.relative_to(ROOT)}.json / .md")


def report(o):
    """The run as a Markdown record: every call in full, what the code did between them."""
    u = o["usage"]
    L = [f"# {o['qid']}：需求清单流程作答记录", "",
         f"模型 {o['model']}，温度 0，思考模式关闭。{o['n_calls']} 次模型调用，{o['seconds']} 秒；"
         f"tokens 输入 {u['prompt_tokens']} + 输出 {u['completion_tokens']} = {u['total_tokens']}。",
         f"答案：**{o['answer'] or '（空）'}**（{o['answered_by']}）。" + (f"注意：{o['caveat']}。" if o["caveat"] else ""), "",
         "## 题目", "", "```text", o["question"] + "".join(f"\n{k}. {v.strip()}" for k, v in o["options"].items()), "```", "",
         "## 需求清单（最终状态）", "", "| 编号 | 内容 | 类型 | 用于选项 | 取到的值 | 来源块 | 第几轮 | 经过 |", "|---|---|---|---|---|---|---|---|"]
    for n in o["needs"].values():
        v = n["value"]
        L.append(f"| {n['id']} | {n['what']} | {n['kind']} | {'、'.join(n['options'])} | {(v['text'][:60] + ' ' + v['unit']) if v else '**没找到**'} | "
                 f"{('`' + v['chunk'] + '`') if v else ''} | {v['round'] if v else ''} | {'；'.join(n['history'])} |")
    L += ["", "## 逐步过程", ""]
    calls = iter(o["calls"])

    def show(c):
        return [f"### 调用：{c['stage']}（输入 {c['usage']['prompt_tokens']}，输出 {c['usage']['completion_tokens']}）", "",
                "<details><summary>系统提示</summary>", "", "```text", c["system"], "```", "", "</details>", "",
                "**发给模型的内容：**", "", "```text", c["user"], "```", "", "**模型输出（原样）：**", "", "```text", c["response"], "```", ""]
    L += show(next(calls))
    for r in o["rounds"]:
        L += [f"### 第 {r['round']} 轮检索（代码，不调用模型）", "", "| 需求 | 检索词 | 实际检索的文本 | 文档数 | 返回 |", "|---|---|---|---|---|"]
        L += [f"| {x['need']} | {x['query']} | {x['searched']} | {x['docs']} | {', '.join(x['chunks'])} |" for x in r["queries"]]
        L += ["", f"去重后新块 {len(r['new'])} 个：{', '.join(r['new']) or '无'}；带入上一轮仍相关的块 {len(r['carried'])} 个：{', '.join(r['carried']) or '无'}。", ""]
        if "serves" not in r:
            L += [r.get("note", ""), ""]
            continue
        L += show(next(calls))
        L += ["**代码核对填值：**", ""] + [f"- {c['need']} = {c['value']} {c['unit'] or ''}（{c['chunk']}）→ {'通过' if c['ok'] else '未通过：' + c['why']}" for c in r["checks"]]
        L += ["", "**召回不好、改写检索词的需求：**", ""] + ([f"- {m['need']}：{m['reason']} → {m['queries']}" for m in r["missing"]] or ["- 无"])
        L += ["", f"**上下文：** 不相关、移入证据库 {len(r['masked'])} 块（{', '.join(r['masked']) or '无'}）；已取完值、移出 {len(r['used'])} 块；"
              f"仍服务于未满足需求、保留到下一轮 {len(r['kept'])} 块（{', '.join(r['kept']) or '无'}）。", ""]
    for c in calls:
        L += show(c)
    L += ["## 代码运行结果", ""]
    if o["code_output"]:
        L += ["| 中间结果 | 值 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in o["code_output"]["R"].items()]
        L += ["", "| 选项 | 结论 | 理由 |", "|---|---|---|"] + [f"| {k} | {o['verdicts'].get(k)} | {o['code_output']['why'].get(k, '')} |" for k in o["options"]]
    else:
        L += ["代码没有跑通：", "", "```text", str(o["code_error"]), "```"]
    if o["literals_in_code"]:
        L += ["", f"注意：代码里直接写了不在清单里的数 {o['literals_in_code']}。"]
    return "\n".join(L)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
