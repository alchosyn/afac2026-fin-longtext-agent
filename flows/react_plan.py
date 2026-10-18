# -*- coding: utf-8 -*-
"""ReAct over a list of data needs, with evidence masking (demo, 2026-10-06). The model decides
every step which tool to call; the tools work on the need list in batches; the context of every
step is a snapshot the code builds from the state, not the transcript of the conversation.

    tools      plan     add needs (what value, number or passage, unit, queries, the options or
                        answer parts using it); for a calculation question whose answer parts the
                        code could not read from the stem, also the parts
               search   the queries of a batch of needs, BM25 through agent.search.narrow, top
                        TOP_PER_QUERY per query, merged round-robin, duplicates dropped
               fill     which needs every chunk in the context serves, the values found (the code
                        checks each stands in its chunk), new queries for needs not found, for
                        every target (option / answer part) whether the data is enough to decide
                        it, new needs for what is lacking
               compute  Python over the filled values V (about() and fmt() predefined) that works
                        out every target: verdicts for a choice question, answers for a calculation
                        question; the code runs it and checks it is complete
    masking    after every fill: a chunk that serves no need leaves the context (不相关, into the
               evidence store), a chunk whose needs are all filled leaves it too (its values stay
               in the list), a chunk serving a need still open stays. A search never brings back
               a chunk set aside after the need it searches for existed
    snapshot   system prompt + one user message: question, answer parts, the need list with
               states, the targets with states, the chunks in play in full, the last actions as
               one line each, what the tools of the last step returned, the steps left
    answer     choice: the options picked by the direction of the stem (agent.decide), by
               elimination for a single-choice question with no option chosen and one option
               without data; calculation: the answer parts as the code formatted them
    limits     --steps; at the limit only compute is offered and only compute is run

    python flows/react_plan.py fin_b_009 [--steps 12]

Writes output/react_plan/react_plan_<qid>.json (every step: the snapshot sent, what the model wrote, every
call and reply, what the code did) and .md (the same as a readable record).
"""
import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "flows"))

from agent.decide import caveat, choose, direction          # noqa: E402
from agent.docview import Doc, load_cards                   # noqa: E402
from agent.llm import Qwen                                  # noqa: E402
from agent.retrieval import bm25_scores                     # noqa: E402
from agent.scope import load_b_questions, scope             # noqa: E402
from agent.search import narrow                             # noqa: E402
from plan_fill import (MAX_BODY, MAX_NEW, QUERIES_PER_NEED, QUERY_RULES, TOP_PER_QUERY,  # noqa: E402
                       VERDICTS, loose, number, run_code)

RECENT = 8                  # actions listed in the snapshot


def fn(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


NEED_ITEM = {"type": "object", "properties": {
    "what": {"type": "string", "description": "主体（公司、产品或法规）、期间、口径和指标或条款"},
    "kind": {"type": "string", "enum": ["数值", "原文"]},
    "unit": {"type": "string", "description": "按原文的单位（年报表格常用千元），不确定留空"},
    "queries": {"type": "array", "items": {"type": "string"}, "description": "2 到 3 个检索词，写法见系统提示"},
    "targets": {"type": "array", "items": {"type": "string"}, "description": "用到它的选项字母或答案部分编号"}},
    "required": ["what", "kind", "queries", "targets"]}
PLAN = fn("plan", "往需求清单里加需求（第一步列全，之后缺什么再追加）。",
          {"needs": {"type": "array", "items": NEED_ITEM},
           "parts": {"type": "array", "description": "仅当系统提示计算题答案部分未定时填写",
                     "items": {"type": "object", "properties": {"what": {"type": "string"}, "format": {"type": "string"}}, "required": ["what"]}}},
          ["needs"])
SEARCH = fn("search", "按需求批量检索原文，结果去重后放进上下文。可以给需求换检索词。",
            {"needs": {"type": "array", "items": {"type": "object", "properties": {
                "need": {"type": "string", "description": "需求编号"},
                "queries": {"type": "array", "items": {"type": "string"}, "description": "可选：换用的检索词"}}, "required": ["need"]}}},
            ["needs"])
FILL = fn("fill", "看完原文后一次处理：每块满足哪些需求、填值、没找到的写新检索词、各目标数据够不够、缺的写进 new_needs。"
          "chunks 里没有列出、也没被填值引用的块，系统按不相关移出上下文。",
          {"chunks": {"type": "array", "description": "上下文里的每一块都列出", "items": {"type": "object", "properties": {
              "chunk": {"type": "string"}, "needs": {"type": "array", "items": {"type": "string"}}}, "required": ["chunk", "needs"]}},
           "values": {"type": "array", "items": {"type": "object", "properties": {
               "need": {"type": "string"}, "value": {"type": "string", "description": "数值照抄，不换算单位；原文照抄关键句，不超过 200 字"},
               "unit": {"type": "string"}, "chunk": {"type": "string"}, "note": {"type": "string"}}, "required": ["need", "value", "chunk"]}},
           "missing": {"type": "array", "items": {"type": "object", "properties": {
               "need": {"type": "string"}, "reason": {"type": "string"}, "queries": {"type": "array", "items": {"type": "string"}}}, "required": ["need", "reason"]}},
           "targets": {"type": "array", "items": {"type": "object", "properties": {
               "target": {"type": "string", "description": "选项字母或答案部分编号"},
               "ready": {"type": "boolean", "description": "数据够不够做出判断或算出结果"},
               "uses": {"type": "array", "items": {"type": "string"}}, "lacking": {"type": "string"}}, "required": ["target", "ready"]}},
           "new_needs": {"type": "array", "items": NEED_ITEM}},
          ["chunks", "values", "targets"])
COMPUTE = fn("compute", "写 Python 代码，用取到的值一次算完所有目标，系统运行并作答。",
             {"code": {"type": "string", "description": "Python 代码，约定见系统提示"}}, ["code"])

SYSTEM = """你是金融文档问答助手，用工具回答一道{kind}。每一步系统都会把当前状态（需求清单、各{word}的状态、上下文里的原文块、最近的动作）重新发给你，你决定下一步调用哪个工具。
本题涉及的文档：
{docs}
工具：plan（列需求、追加需求）、search（按需求批量检索）、fill（看原文：判断相关性、填值、检查各{word}的数据够不够）、compute（写代码一次算完并作答）。
做法：
1. 先 plan 一次列全需求：每个{word}都要有需求覆盖；比较类说法（“均”“都”“完全一致”“高于”）每个被比较的主体各列一项；只列文档里能直接读到的值或原文，需要计算的值不列。
2. 再 search 一次查所有需求；看到原文后 fill 一次处理所有块、所有需求、所有{word}。
3. 还有{word}的数据不够，就只 search 它缺的需求（可以换检索词）再 fill；都够了就 compute。
4. fill 里的 ready 只看数据够不够：数据和说法对不上也算够（结论是错误）；一条一般性规定已经能否定的说法也算够。
5. 判断只依据原文和计算结果。fill 判为不相关（或没有列出）和已取完值的原文块会移出上下文，取到的值留在清单里。
6. 需求的 unit 按原文的单位写（年报表格常用千元），不确定留空；{query_rules}
compute 的代码约定：
- V 是 dict：需求编号 -> 值（数值是去掉千分位的 float，单位见清单；原文是摘录字符串）。不要在代码里手写这些数。
- 中间结果存进 dict R；不要先四舍五入再计算。已定义 about(x, stated)：计算值 x 与题面写的数 stated（字符串，如 "9.82"、"17.17%"）在 stated 的小数位上是否一致；fmt(x, n=2)：格式化为 n 位小数。百分数和“个百分点”注意乘 100。
{compute_rules}"""
CHOICE_RULES = """- 每个选项的结论写进 dict verdicts（"正确" / "错误" / "无依据"），一句话理由写进 dict why。结论只表示说法与原文和计算是否一致，题干问正确还是错误的选项都不影响结论，选哪些由系统按题干决定。带“约”的数用 about() 比较；依赖没找到的数据又无法判断的选项判 "无依据"。"""
CALC_RULES = """- 按答案部分的顺序把结果写进 list answers，每个部分一个值（一个字符串），理由写进 dict why。格式：百分数带 % 并保留两位小数；天数、笔数、金额、分值等只填数字、保留两位小数；日期写成“YYYY年M月D日”；排序用英文“>”且前后不加空格；题目另有要求的按题目。"""


def items(v, key=None):
    """A tool argument as a list of dicts, whatever shape the model sent: a JSON string, a dict
    keyed by `key` ({"c0019": ["n1"]} -> [{"chunk": "c0019", "needs": ["n1"]}]), a list of
    JSON strings or of bare ids."""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return []
    if isinstance(v, dict):
        if key is None:
            return [v]
        return [({key: k, **x} if isinstance(x, dict) else {key: k, "needs": x if isinstance(x, list) else [x]}) for k, x in v.items()]
    out = []
    for x in v or []:
        if isinstance(x, str):
            try:
                x = json.loads(x)
            except json.JSONDecodeError:
                x = {key or "value": x}
        if isinstance(x, dict):
            out.append(x)
    return out


def answer_parts(question):
    """The answer parts a calculation question asks for, read from its wording, or None when only
    the model can tell ("…分别是多少" over several periods)."""
    text = re.sub(r"\s+", "", question)
    rule = "；".join(re.findall(r"[^。；]*(?:保留|只填写|不带|以[^。；]*计)[^。；]*", text))
    m = re.search(r"答案格式为[“\"]([^”\"]+)[”\"]", text)
    if m:
        return [{"id": f"p{i}", "what": w, "format": rule} for i, w in enumerate(re.split(r"[；;]", m.group(1)), 1) if w]
    last = re.split(r"[。？?]", text.rstrip("。？?"))[-1] if text else ""
    if re.search(r"分别(?:是|为|约为)?多少", text) and not re.search(r"合计|总计|差额|几笔|多少笔|之和", last):
        return None
    return [{"id": "p1", "what": "题目所求的结果", "format": rule}]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("qid")
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args(argv)
    q = next(x for x in load_b_questions() if x["qid"] == args.qid)
    options = q.get("options") or {}
    is_calc = not options
    word = "答案部分" if is_calc else "选项"
    sc, cards, docs = scope(q), load_cards(), {}
    asked, clause, dword = direction(q["question"]) if not is_calc else (None, None, None)
    llm = Qwen(log_path=ROOT / "logs" / f"react_plan_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    t0 = time.time()

    needs, store, inplay = {}, {}, []
    masked, done_with, aside = set(), set(), {}
    parts = answer_parts(q["question"]) if is_calc else None
    targets = ({p["id"]: p for p in parts} if parts else {}) if is_calc else {k: {"id": k, "what": v.strip()} for k, v in options.items()}
    status = {t: {"ready": False, "uses": [], "lacking": "还没有检查"} for t in targets}
    actions, trace, last_replies = [], [], []
    usage = [0, 0]
    final = None

    def unit(uid):
        d = uid.split("#")[0]
        docs.setdefault(d, Doc(d, cards[d]))
        return docs[d].by_id[uid]

    def add_need(n, step):
        what = str(n.get("what", "")).strip()
        if not what:
            return None, "缺少 what"
        dup = next((m for m in needs.values() if loose(m["what"]) == loose(what)), None)
        if dup:
            return None, f"与 {dup['id']} 重复"
        nid = f"n{len(needs) + 1}"
        tg = [str(o).strip().upper() if not is_calc else str(o).strip() for o in n.get("targets") or []]
        needs[nid] = {"id": nid, "what": what, "kind": "原文" if str(n.get("kind")) == "原文" else "数值", "unit": str(n.get("unit") or ""),
                      "targets": tg, "queries": [str(x) for x in n.get("queries") or []][:3], "value": None, "history": [], "since": step}
        for t in tg:
            if t in status and status[t]["ready"]:
                status[t] = {"ready": False, "uses": [], "lacking": f"追加了需求 {nid}"}
        return nid, ""

    def check_value(nid, v, ch):
        n = needs[nid]
        if ch not in store:
            return False, f"块 {ch} 没有检索到过"
        if n["kind"] == "数值":                          # the number among the chunk's numbers, thousands separators or not (fin_b_009)
            _, val = number(v)
            ok = val is not None and any(abs(val - x) < 1e-9 * max(1, abs(x)) for x in
                                         (float(t.replace(",", "")) for t in re.findall(r"-?\d[\d,]*(?:\.\d+)?", store[ch]["text"])))
            return ok, "" if ok else f"块 {ch} 里没有数值 {v}"
        # sentence by sentence: a quote may skip the "1." / "2." between the items it joins (ins_b_018, 2026-10-06)
        segs = [loose(x) for x in re.split(r"[。；;！？\n]", str(v)) if len(loose(x)) >= 4]
        bad = [x for x in segs if x not in loose(store[ch]["text"])]
        ok = bool(segs) and not bad
        return ok, "" if ok else f"摘录不在块 {ch} 里（对不上的一句：{bad[0][:30] if bad else '摘录太短'}）"

    # ------------------------------------------------------------------ tools
    def t_plan(a, step):
        out, added = [], []
        if is_calc and not targets and a.get("parts"):
            for i, p in enumerate(items(a["parts"]), 1):
                targets[f"p{i}"] = {"id": f"p{i}", "what": str(p.get("what", "")), "format": str(p.get("format", ""))}
                status[f"p{i}"] = {"ready": False, "uses": [], "lacking": "还没有检查"}
            out.append(f"答案部分：{'、'.join(f'{k} {t['what']}' for k, t in targets.items())}")
        for n in items(a.get("needs")):
            nid, why = add_need(n, step)
            (added.append(nid) if nid else out.append(f"没有加入“{str(n.get('what', ''))[:30]}”：{why}"))
        unknown = sorted({t for nid in added for t in needs[nid]["targets"] if t not in targets})
        if unknown:
            out.append(f"不认识的选项或部分：{unknown}")
        return f"清单加入 {len(added)} 项：{'、'.join(added) or '无'}" + ("；" + "；".join(out) if out else ""), f"plan：加入需求 {'、'.join(added) or '无'}"

    def t_search(a, step):
        lists, log, asked_needs = [], [], []
        for item in items(a.get("needs"), "need"):
            nid = str(item.get("need", "")).strip()
            if nid not in needs:
                continue
            n = needs[nid]
            if item.get("queries"):
                n["queries"] = [str(x) for x in item["queries"]][:3]
            asked_needs.append(nid)
            for query in n["queries"][:QUERIES_PER_NEED]:
                where, text = narrow(q, query, sc)
                got = [u for u, s in bm25_scores(text, where)[:TOP_PER_QUERY] if s > 0]
                lists.append([u for u in got if aside.get(u, 0) <= n["since"]])
                log.append({"need": nid, "query": query, "searched": text.strip(), "chunks": got})
        new = []
        for pos in range(TOP_PER_QUERY):
            for got in lists:
                if pos < len(got) and got[pos] not in new and got[pos] not in inplay:
                    new.append(got[pos])
        new = new[:MAX_NEW]
        for u in new:
            masked.discard(u)
            done_with.discard(u)
            aside.pop(u, None)
            if u not in store:
                x = unit(u)
                store[u] = {"title": cards[u.split("#")[0]]["title"], "path": x["path"] or "", "text": x["body"][:MAX_BODY], "type": x["type"], "step": step}
            inplay.append(u)
        # a need whose queries bring nothing new twice in a row is given up: 原文未见 (ins_b_005 searched C three times for a clause
        # the policy does not have, 2026-10-06)
        exhausted = []
        for nid in asked_needs:
            n = needs[nid]
            hits = {u for x in log if x["need"] == nid for u in x["chunks"] if aside.get(u, 0) <= n["since"]}
            if n["value"] is None and not (hits & set(inplay)):
                n["dry"] = n.get("dry", 0) + 1
                if n["dry"] >= 2 and not n.get("exhausted"):
                    n["exhausted"] = True
                    n["history"].append(f"第 {step} 步：连续两次检索没有新原文，标为原文未见")
                    exhausted.append(nid)
            else:
                n["dry"] = 0
        for k, s in status.items():                     # a target lacking only given-up needs can be decided now (无依据 for that part)
            own = [n for n in needs.values() if k in n["targets"] and n["value"] is None]
            if not s["ready"] and own and all(n.get("exhausted") for n in own):
                s.update(ready=True, lacking="原文未见：" + "、".join(n["id"] for n in own))
        reply = (f"检索需求 {'、'.join(asked_needs) or '无'}，{len(log)} 个查询，新放进上下文 {len(new)} 块"
                 + ("" if new else "（召回的都是看过的原文）")
                 + (f"；{'、'.join(exhausted)} 连续两次没有新原文，标为原文未见，用到它的{word}按现有数据判断（缺的部分判无依据）" if exhausted else ""))
        return reply, f"search {'、'.join(asked_needs)}：新块 {len(new)}" + (f"，{'、'.join(exhausted)} 原文未见" if exhausted else ""), log

    def t_fill(a, step):
        serves = {str(x.get("chunk", "")).strip("[] "): [str(n) for n in x.get("needs") or []] for x in items(a.get("chunks"), "chunk")}
        lines, checks = [], []
        for v in items(a.get("values")):
            nid, ch = str(v.get("need", "")).strip(), str(v.get("chunk", "")).strip("[] ")
            if nid not in needs or needs[nid]["value"] is not None:
                continue
            ok, why = check_value(nid, v.get("value"), ch)
            checks.append({"need": nid, "value": v.get("value"), "chunk": ch, "ok": ok, "why": why})
            if ok:
                needs[nid]["value"] = {"text": str(v.get("value")), "number": number(v.get("value"))[1] if needs[nid]["kind"] == "数值" else None,
                                       "unit": str(v.get("unit") or needs[nid]["unit"]), "chunk": ch, "note": str(v.get("note") or ""), "step": step}
            else:
                needs[nid]["history"].append(f"第 {step} 步填了 {v.get('value')}，核对未通过：{why}")
        for m in items(a.get("missing")):
            nid = str(m.get("need", "")).strip()
            if nid in needs and needs[nid]["value"] is None:
                needs[nid]["history"].append(f"第 {step} 步没找到：{m.get('reason', '')}")
                if m.get("queries"):
                    needs[nid]["queries"] = [str(x) for x in m["queries"]][:3]
        added = []
        for n in items(a.get("new_needs")):
            nid, why = add_need(n, step)
            if nid:
                added.append(nid)
        for t in items(a.get("targets"), "target"):
            k = str(t.get("target", "")).strip()
            k = k.upper() if not is_calc else k
            if k in status:
                status[k] = {"ready": bool(t.get("ready")), "uses": [str(x) for x in t.get("uses") or []], "lacking": str(t.get("lacking") or "")}
        for k, s in status.items():                     # ready on a need without a value is not ready
            if s["ready"] and any(u in needs and needs[u]["value"] is None for u in s["uses"]):
                s["ready"], s["lacking"] = False, "用到的需求还没有取到值"
        # masking
        open_ids = {n["id"] for n in needs.values() if n["value"] is None}
        for c in checks:                                # a chunk a value cites serves that need, listed or not
            serves.setdefault(c["chunk"], [])
            if c["need"] not in serves[c["chunk"]]:
                serves[c["chunk"]].append(c["need"])
        unjudged = [u for u in inplay if u not in serves]   # seen and not listed: 不相关 (fin_b_009 kept 26 of them, 2026-10-06)
        for u in unjudged:
            serves[u] = []
        irrelevant = [u for u in inplay if not serves[u]]
        used = [u for u in inplay if serves[u] and not (set(serves[u]) & open_ids)]
        for u in irrelevant:
            masked.add(u)
        for u in used:
            done_with.add(u)
        for u in irrelevant + used:
            aside[u] = step
            inplay.remove(u)
        ok_n, bad_n = sum(c["ok"] for c in checks), [c for c in checks if not c["ok"]]
        lines.append(f"填值通过 {ok_n} 项" + (f"；未通过：" + "；".join(f"{c['need']} {c['why']}" for c in bad_n) if bad_n else ""))
        if added:
            lines.append(f"追加需求 {'、'.join(added)}")
        lines.append(f"不相关移出 {len(irrelevant)} 块" + (f"（其中没有列出的 {len(unjudged)} 块）" if unjudged else "") + f"，已取完值移出 {len(used)} 块，留在上下文 {len(inplay)} 块")
        not_ready = [k for k, s in status.items() if not s["ready"]]
        lines.append("各" + word + "的数据都够了，可以 compute" if not not_ready else f"数据还不够的{word}：{'、'.join(not_ready)}")
        summary = f"fill：填值 {ok_n} 项，追加 {len(added)}，移出 {len(irrelevant) + len(used)} 块，还不够 {'、'.join(not_ready) or '无'}"
        return "；".join(lines), summary, {"checks": checks, "added": added, "masked": irrelevant, "used": used, "unjudged": unjudged, "serves": serves}

    def t_compute(a, step):
        nonlocal final
        code = str(a.get("code") or "")
        V = {nid: (n["value"]["number"] if n["kind"] == "数值" else n["value"]["text"]) for nid, n in needs.items() if n["value"]}
        if re.search(r"^\s*V\s*(?:\[[^\]]*\]\s*)?(?:=|\.update\()", code, re.M):     # ins_b_005 rewrote V from a cut-off snapshot
            return ("代码没有通过：不能给 V 赋值或修改 V，取到的值系统已放在 V 里，直接用 V[\"n1\"] 读取。", "compute：出错",
                    {"code": code, "error": "rewrote V", "literals": []})
        res, error = run_code(code, V)
        if res and not error:
            if is_calc:
                bad = [p for p in res["answers"] if re.search(r"\d\s*(?:[;；，、]|,\s)\s*\d", p)]
                if not targets:
                    error = "答案部分还没有定：先用 plan 的 parts 列出"
                elif len(res["answers"]) != len(targets):
                    error = f"answers 有 {len(res['answers'])} 个部分，应为 {len(targets)} 个（{'、'.join(targets)}）"
                elif bad:
                    error = f"每个部分只能有一个值：{bad}"
            else:
                miss = [k for k in options if res["verdicts"].get(k) not in VERDICTS]
                if miss:
                    error = f"verdicts 缺少或取值不对的选项：{miss}"
        knowns = [v for v in V.values() if isinstance(v, float)]
        literals = sorted({float(x.replace(",", "")) for x in re.findall(r"\d[\d,]{4,}(?:\.\d+)?", code)
                           if not any(abs(float(x.replace(",", "")) - k) < 1e-6 * max(1, abs(k)) for k in knowns)})
        if error:
            return f"代码没有通过：{error}", "compute：出错", {"code": code, "error": error, "literals": literals}
        final = {"code": code, "result": res, "literals": literals, "step": step}
        return "代码运行通过，系统据此作答。", "compute：完成", {"code": code, "result": res, "literals": literals}

    TOOLS = {"plan": t_plan, "search": t_search, "fill": t_fill, "compute": t_compute}

    # ------------------------------------------------------------------ snapshot
    stem = q["question"].strip() + "".join(f"\n{k}. {v.strip()}" for k, v in options.items())
    titles = "\n".join(f"- 《{cards[d]['title']}》" for d in sc["docs"][:20]) + (f"\n（共 {len(sc['docs'])} 份，题面没有点名文档）" if len(sc["docs"]) > 20 else "")
    system = SYSTEM.format(kind=q["type"], word=word, docs=titles, query_rules=QUERY_RULES, compute_rules=CALC_RULES if is_calc else CHOICE_RULES)

    def snapshot(step, left):
        L = [f"题目：\n{stem}", ""]
        if is_calc:
            L += (["答案各部分（系统按题面定好，compute 按这个顺序写 answers）："] + [f"- {t['id']}｜{t['what']}｜{t.get('format', '')}" for t in targets.values()]
                  if targets else ["答案部分：系统没能从题面定下，请在 plan 的 parts 里按题目顺序列出。"]) + [""]
        if needs:
            L += ["需求清单：", "| 编号 | 内容 | 类型 | 单位 | 用于 | 状态 | 检索词 |", "|---|---|---|---|---|---|---|"]
            for n in needs.values():
                st = (f"已取到 {n['value']['text'][:400]} {n['value']['unit']}（{n['value']['chunk']}）" if n["value"]   # [:80] cut ins_b_005's D clause
                      else "未取到" + (f"：{n['history'][-1]}" if n["history"] else ""))
                L.append(f"| {n['id']} | {n['what']} | {n['kind']} | {n['unit']} | {'、'.join(n['targets'])} | {st} | {' / '.join(n['queries'])} |")
        else:
            L += ["需求清单：还没有，先用 plan 列出。"]
        L += ["", f"各{word}的状态：" + ("；".join(f"{k} " + (f"够（用 {'、'.join(s['uses'])}）" if s["ready"] else f"不够（{s['lacking']}）") for k, s in status.items()) or "未定")]
        L += ["", f"上下文里的原文块（{len(inplay)} 块；已移出 {len(masked) + len(done_with)} 块，存在证据库里）：", ""]
        L += [f"[{u}]《{store[u]['title']}》{store[u]['path']}\n{store[u]['text']}\n" for u in inplay] or ["（无）", ""]
        L += ["最近的动作：", *(f"- {x}" for x in actions[-RECENT:])] if actions else ["最近的动作：无"]
        if last_replies:
            L += ["", "上一步工具返回：", *(f"- {x}" for x in last_replies)]
        L += ["", f"这是第 {step} 步，还剩 {left} 步" + ("。步数已用完：现在只能调用 compute，用已取到的数据作答，缺数据的写无依据或无法计算。" if left == 0 else "。")]
        return "\n".join(L)

    # ------------------------------------------------------------------ loop
    print(f"{q['qid']}（{q['type']}）{stem}\n" + (f"方向：选判为“{asked}”的选项\n" if not is_calc else f"答案部分：{[t['what'] for t in targets.values()] or '未定'}\n"))
    step, idle = 0, 0
    while final is None and step < args.steps + 1:
        step += 1
        left = args.steps - step
        wrap = left <= 0
        user = snapshot(step, max(left, 0))
        tools = [COMPUTE] if wrap else [PLAN, SEARCH, FILL, COMPUTE]
        for attempt in range(3):
            try:
                resp = llm.client.chat.completions.create(model=llm.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                                          temperature=0, max_tokens=4000, extra_body={"enable_thinking": False}, tools=tools)
                break
            except Exception as e:                  # noqa: BLE001 - transient API errors
                if attempt == 2:
                    raise
                print(f"API 出错，重试：{e}")
                time.sleep(5)
        msg = resp.choices[0].message
        usage[0] += resp.usage.prompt_tokens
        usage[1] += resp.usage.completion_tokens
        row = {"step": step, "snapshot": user, "content": msg.content or "", "calls": [],
               "usage": {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens, "cumulative": sum(usage)},
               "inplay": len(inplay)}
        trace.append(row)
        print(f"--- 第 {step} 步（输入 {resp.usage.prompt_tokens}，输出 {resp.usage.completion_tokens}，累计 {sum(usage)}）上下文原文 {len(inplay)} 块")
        if msg.content and msg.content.strip():
            print("模型：" + msg.content.strip()[:400])
        last_replies = []
        calls = msg.tool_calls or []
        if not calls:
            idle += 1
            last_replies = ["你没有调用工具。请调用 plan / search / fill / compute 之一。"]
            actions.append(f"第 {step} 步：没有调用工具")
            if idle >= 2:
                break
            continue
        for c in calls:
            name = c.function.name
            try:
                a = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError as e:
                a, name = {}, name + "（参数不是合法 JSON）"
                last_replies.append(f"{c.function.name} 的参数不是合法 JSON：{e}")
                row["calls"].append({"name": c.function.name, "arguments": c.function.arguments, "reply": last_replies[-1]})
                continue
            if name not in TOOLS or (wrap and name != "compute"):
                reply, summary, extra = f"{name} 现在不能用。", f"{name}：被拒", None
            else:
                out = TOOLS[name](a, step)
                reply, summary, extra = out[0], out[1], (out[2] if len(out) > 2 else None)
            last_replies.append(f"{name}：{reply}")
            actions.append(f"第 {step} 步 {summary}")
            row["calls"].append({"name": name, "arguments": a, "reply": reply, "detail": extra})
            print(f"调用：{name} {json.dumps(a, ensure_ascii=False)[:300]}\n返回：{reply}")
            if final:
                break
    # ------------------------------------------------------------------ answer
    missing = [n for n in needs.values() if n["value"] is None]
    note, verdicts, answers = "", {}, None
    if is_calc:
        answers = final["result"]["answers"] if final else []
        answer = "；".join(answers)
        how = "代码按题目要求的格式算出各部分" if final else "没有算出结果"
    else:
        verdicts = {k: (final["result"]["verdicts"].get(k) if final else None) for k in options}
        chosen, unfounded = choose(asked, verdicts)
        how = f"代码按题干方向从各选项结论里挑出判为“{asked}”的选项" if final else "没有算出结论"
        if q["type"] == "单选题" and not chosen and len(unfounded) == 1 and missing:
            chosen, how = unfounded, how + "；没有选项符合，按排除法选唯一缺数据的选项"
        answer = "".join(chosen)
        note = caveat(asked, [k for k in unfounded if k not in chosen])
    if missing:
        note = (note + "；" if note else "") + "没找到的值：" + "、".join(f"{n['id']} {n['what']}" for n in missing)
    print(f"\n答案：{answer}（{how}）" + (f"\n注意：{note}" if note else ""))
    out = {"qid": q["qid"], "type": q["type"], "question": q["question"].strip(), "options": options, "model": llm.model, "system": system,
           "direction": {"asked": asked, "clause": clause, "word": dword}, "targets": targets, "target_status": status, "needs": needs,
           "verdicts": verdicts, "answers": answers, "answer": answer, "answered_by": how, "caveat": note, "final": final,
           "steps": len(trace), "calls": {n: sum(c["name"] == n for r in trace for c in r["calls"]) for n in TOOLS},
           "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)},
           "seconds": round(time.time() - t0, 1), "store": store, "masked": sorted(masked), "done_with": sorted(done_with), "trace": trace}
    base = ROOT / "output" / "react_plan" / f"react_plan_{q['qid']}"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    base.with_suffix(".md").write_text(report(out), encoding="utf-8")
    print(f"{out['steps']} 步，调用 {out['calls']}，tokens {usage[0]} + {usage[1]} = {sum(usage)}，{out['seconds']} 秒 -> {base.relative_to(ROOT)}.json / .md")


def report(o):
    """The run as a Markdown record: every snapshot sent, everything the model wrote and called."""
    u = o["usage"]
    L = [f"# {o['qid']}：ReAct + 需求清单 + 证据遮蔽 作答记录", "",
         f"{o['type']}。模型 {o['model']}，温度 0，思考模式关闭。{o['steps']} 步，工具调用：" + "，".join(f"{k} {v}" for k, v in o["calls"].items()) + f"；{o['seconds']} 秒。",
         f"tokens：输入 {u['prompt_tokens']} + 输出 {u['completion_tokens']} = {u['total_tokens']}。",
         f"答案：**{o['answer'] or '（空）'}**（{o['answered_by']}）。" + (f"注意：{o['caveat']}。" if o["caveat"] else ""), "",
         "## 系统提示", "", "```text", o["system"], "```", "",
         "## 需求清单（最终状态）", "", "| 编号 | 内容 | 类型 | 用于 | 取到的值 | 来源块 | 经过 |", "|---|---|---|---|---|---|---|"]
    for n in o["needs"].values():
        v = n["value"]
        L.append(f"| {n['id']} | {n['what']} | {n['kind']} | {'、'.join(n['targets'])} | {(v['text'][:60] + ' ' + v['unit']) if v else '**没找到**'} | "
                 f"{('`' + v['chunk'] + '`') if v else ''} | {'；'.join(n['history'])} |")
    L += ["", "## 逐步过程", "", "每一步发给模型的是代码按当前状态拼出的快照（下面“发给模型的快照”原样照录），不含之前的对话。", ""]
    for r in o["trace"]:
        L += [f"### 第 {r['step']} 步（输入 {r['usage']['prompt_tokens']}，输出 {r['usage']['completion_tokens']}，累计 {r['usage']['cumulative']}；上下文原文 {r['inplay']} 块）", "",
              "<details><summary>发给模型的快照</summary>", "", "```text", r["snapshot"], "```", "", "</details>", "",
              "**模型写的文字：**", "", "```text", r["content"].strip() or "（无）", "```", ""]
        for i, c in enumerate(r["calls"], 1):
            args = c["arguments"] if isinstance(c["arguments"], str) else json.dumps(c["arguments"], ensure_ascii=False, indent=1)
            L += [f"**调用 {i}：{c['name']}**", "", "```json", args, "```", "", f"返回：{c['reply']}", ""]
            d = c.get("detail")
            if c["name"] == "search" and d:
                L += ["| 需求 | 检索词 | 实际检索的文本 | 返回 |", "|---|---|---|---|"] + [f"| {x['need']} | {x['query']} | {x['searched']} | {', '.join(x['chunks'])} |" for x in d] + [""]
            if c["name"] == "fill" and d:
                L += [f"- 核对：" + ("；".join(f"{x['need']} {'通过' if x['ok'] else '未通过（' + x['why'] + '）'}" for x in d["checks"]) or "无"),
                      f"- 不相关移出：{', '.join(d['masked']) or '无'}；已取完值移出：{', '.join(d['used']) or '无'}；没判断的：{', '.join(d['unjudged']) or '无'}", ""]
    if o["final"]:
        res = o["final"]["result"]
        L += ["## 代码运行结果", "", "```python", o["final"]["code"], "```", "", "| 中间结果 | 值 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in res["R"].items()]
        if o["answers"] is not None:
            L += ["", "答案各部分：" + "；".join(o["answers"])] + [f"- {k}：{v}" for k, v in res["why"].items()]
        else:
            L += ["", "| 选项 | 结论 | 理由 |", "|---|---|---|"] + [f"| {k} | {o['verdicts'].get(k)} | {res['why'].get(k, '')} |" for k in o["options"]]
        if o["final"]["literals"]:
            L += ["", f"注意：代码里直接写了不在清单里的数 {o['final']['literals']}。"]
    return "\n".join(L)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
