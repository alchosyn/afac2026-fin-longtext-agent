# -*- coding: utf-8 -*-
"""Answer one B-list question (choice or calculation) from a list of data needs instead of a
ReAct loop (demo v2, 2026-10-06; plan_fill_v1.py is the first version). Every model call is a
single call whose context the code builds afresh; nothing accumulates from call to call.

    targets    what has to be decided: the options of a choice question, the answer parts of a
               calculation question (listed by the plan call, in the order the question asks)
    0 plan     one call reads the question and lists the data needs: what value (subject,
               period, scope), number or passage, unit, two or three search queries, the
               targets that use it
    1 search   code only: the queries of the unmet needs of the targets not yet decidable,
               through agent.search.narrow + BM25, top TOP_PER_QUERY each, merged round-robin,
               duplicates and chunks set aside earlier dropped, at most MAX_NEW new chunks
    2 fill     one call over those needs, the needs already filled and the chunks in play:
               which needs each chunk serves (none: 不相关), the value of every need found (the
               code checks it stands in the chunk), new queries for needs not found, and for
               every target whether it can be decided now; a target that cannot names what it
               lacks, and what the list does not have yet is added to it as new needs
    3 check    code: every target decidable -> final call. Otherwise back to 1 for the needs of
               the targets still open; stop early when a round brings no new chunk
    4 final    one call writes Python over the filled values (V) that works out every target
               (about() compares a computed value with a stated one at its precision); the
               code runs it. Choice: the options are picked by the direction of the stem
               (agent.decide); a single-choice question with no option chosen and one option
               without data is answered by elimination. Calculation: the answer parts as the
               code formatted them. Missing values are named.

    python flows/plan_fill.py fin_b_018

Writes output/plan_fill/plan_fill_<qid>.json (every call: system, user, response, usage; retrieval;
needs; targets; code and its output) and .md (the same as a readable record).
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

QUERY_RULES = ("queries：2 到 3 个检索词。第一个写“公司简称或产品名 年份 指标或条款名”（如“宁德时代 2025年 营业收入”“国寿增益宝 身故保险金”），"
               "后面的换成表名、章节名或同义说法（如“主要会计数据 营业收入”“合并现金流量表 经营活动产生的现金流量净额”）。"
               "不要写“年度报告”“年报”“条款”这类每一页都有的词。")

SYSTEM_PLAN = """你是金融文档问答助手。先读题，列出需要从文档里取得的数据（需求清单），之后系统按清单检索原文。
本题涉及的文档：
{docs}
{targets_rule}
要求：
1. 只列文档里能直接读到的值：财务报表和指标表里的原始数（营业收入、净利润等），表格直接给出的比率或增减幅度，或条款、规定、承诺的原文。需要计算才能得到的值（比率、增幅、差额）不要列，由后面的代码计算。题面已经给出的数不用列。
2. 每个{target_word}都要有需求覆盖它的说法；比较类的说法（“均”“都”“完全一致”“高于”），每个被比较的主体各列一项。
3. 每项写明主体（公司、产品或法规）、期间、口径（合并或母公司等），一个值只列一次，注明被哪些{target_word}用到。
4. kind：数值 或 原文。unit 按原文的单位写（年报表格常用千元、元），不确定就留空，不要自己定一个单位；原文需求写清要找哪一段规定。
5. {query_rules}
只输出一个 JSON：{output}"""
PLAN_TARGETS_CHOICE = "要判断的是题目的每个选项。"
PLAN_TARGETS_CALC = ("这是计算题。先按题目要求的顺序列出答案的各部分（parts：编号 p1、p2…，写明算什么、格式要求），"
                     "题目只要一个结果就只列 p1。")
PLAN_OUT_CHOICE = '{"needs": [{"id": "n1", "what": "...", "kind": "数值", "unit": "千元", "queries": ["...", "..."], "targets": ["A", "B"]}]}'
PLAN_OUT_CALC = ('{"parts": [{"id": "p1", "what": "...", "format": "保留两位小数，不带单位"}], '
                 '"needs": [{"id": "n1", "what": "...", "kind": "数值", "unit": "千元", "queries": ["...", "..."], "targets": ["p1"]}]}')

SYSTEM_FILL = """你是金融文档问答助手。系统按数据需求检索到了一批原文块，请一次完成四件事：
1. chunks：逐块写出它能满足哪些还缺的需求编号；一个都满足不了的写空列表（系统会把它移出上下文）。
2. values：找到的需求逐个填值。数值照抄原文里的数字（保留千分位和小数），不换算单位，单位照抄原文表头或正文（如 千元）；原文需求照抄关键句（不超过 200 字）。核对主体、期间、口径（合并或母公司）和单位，对不上的不填。系统会核对所填的值是否真在该块里。
3. missing：这批块里找不到的需求，写原因，并给 2 个换了说法的新检索词（换表名、科目名、章节名或同义说法）。
4. targets：对每个{target_word}，结合已取到的值和这批原文，判断现在能不能做出判断或算出结果（只判断数据够不够，不用给结论）。ready 为 true 时写 uses（用到的需求编号）；为 false 时写 lacking（还缺什么），清单里没有的写进 new_needs（{query_rules}）；如果这批原文里就有，同时填 value、unit、chunk。
只输出一个 JSON：{{"chunks": {{"块编号": ["n1"]}}, "values": [{{"need": "n1", "value": "...", "unit": "...", "chunk": "块编号", "note": "期间、口径等说明"}}], "missing": [{{"need": "n2", "reason": "...", "queries": ["...", "..."]}}], "targets": {{"{example}": {{"ready": true, "uses": ["n1"]}}}}, "new_needs": [{{"what": "...", "kind": "原文", "unit": "", "queries": ["...", "..."], "targets": ["{example}"], "value": "", "chunk": ""}}]}}"""

FINAL_COMMON = """代码约定：
- V 是 dict：需求编号 -> 值。数值需求是 float（单位见表），原文需求是摘录字符串。代码里不要手写这些数，用 V["n1"] 取用。
- 计算的中间结果存进 dict R（键用中文名）。中间结果不要先四舍五入，算到底再比较或格式化。
- 已经定义好两个函数：about(x, stated) 判断计算值 x 与题面写的数 stated（字符串，如 "9.82"、"17.17%"）是否在 stated 的小数位上一致；fmt(x, n=2) 把 x 格式化为 n 位小数的字符串。百分数和“个百分点”注意乘 100。"""
SYSTEM_FINAL_CHOICE = """你是金融文档问答助手。判断各选项需要的数据已经从原文取到（系统已核对与原文一致），请写一段 Python 代码，一次把所有选项算完并给出结论。
""" + FINAL_COMMON + """
- 每个选项的结论写进 dict verdicts，取值 "正确" / "错误" / "无依据"，一句话理由写进 dict why。
- 结论只表示选项的说法与原文和计算结果是否一致。题干问的是正确还是错误的选项都不影响结论，选哪些由系统按题干决定。
- 选项里带“约”的数用 about() 比较。依赖“没找到的数据”的选项判 "无依据"，除非其余数据已足以判断。
- 原文类的选项：读摘录判断，直接给 verdicts 赋值，并在 why 里写明依据哪句原文。
只输出一个 ```python 代码块，不要输出别的。"""
SYSTEM_FINAL_CALC = """你是金融文档问答助手。计算题需要的数据已经从原文取到（系统已核对与原文一致），请写一段 Python 代码算出答案。
""" + FINAL_COMMON + """
- 按题目要求的顺序把答案各部分写进 list answers（每部分一个字符串），说明写进 dict why（键用部分编号）。
- 格式：百分数带 % 并保留两位小数；天数、笔数、金额、分值等只填数字、保留两位小数；日期写成“YYYY年M月D日”；排序用英文“>”，前后不加空格；题目对格式另有要求的按题目。
- 依赖“没找到的数据”的部分写 "无法计算"。
只输出一个 ```python 代码块，不要输出别的。"""

HELPERS = '''
def about(x, stated):
    s = str(stated).replace(",", "").replace("%", "").strip()
    d = len(s.split(".")[1]) if "." in s else 0
    return abs(float(x) - float(s)) <= 0.5 * 10 ** (-d) + 1e-9

def fmt(x, n=2):
    return f"{float(x):.{n}f}"
'''


def squash(s):
    return re.sub(r"[\s|]+", "", str(s))


def loose(s):
    """Letters, digits and CJK only: a copied passage matches across punctuation and spacing."""
    return re.sub(r"[^\w]", "", str(s))


def number(s):
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(s))
    return (m.group(0), float(m.group(0).replace(",", ""))) if m else (None, None)


def run_code(code, values, timeout=20):
    """Run the model's code in a fresh interpreter with V, about() and fmt(); (result, error)."""
    wrapper = ("import json, math, sys\nV = json.loads(sys.stdin.read())\nR, verdicts, why, answers = {}, {}, {}, []\n" + HELPERS
               + code + "\nprint('<<<OUT>>>' + json.dumps({'R': {str(k): (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) "
               "for k, v in R.items()}, 'verdicts': verdicts, 'why': {str(k): str(v) for k, v in why.items()}, "
               "'answers': [str(a) for a in answers]}, ensure_ascii=False))\n")
    try:
        p = subprocess.run([sys.executable, "-c", wrapper], input=json.dumps(values, ensure_ascii=False), capture_output=True,
                           text=True, encoding="utf-8", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "代码运行超时"
    if p.returncode != 0 or "<<<OUT>>>" not in p.stdout:
        return None, (p.stderr or p.stdout)[-1500:]
    res = json.loads(p.stdout.split("<<<OUT>>>")[-1])
    res["stdout"] = p.stdout.split("<<<OUT>>>")[0]
    return res, None


def main(argv):
    qid = argv[0]
    q = next(x for x in load_b_questions() if x["qid"] == qid)
    options = q.get("options") or {}
    is_calc = not options
    word = "答案部分" if is_calc else "选项"
    sc, cards, docs = scope(q), load_cards(), {}
    asked, clause, dword = direction(q["question"]) if not is_calc else (None, None, None)
    llm = Qwen(log_path=ROOT / "logs" / f"plan_fill_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    t0 = time.time()
    calls, rounds = [], []
    store, active, masked, done_with = {}, [], set(), set()
    aside = {}                                      # chunk -> round in which it was set aside (masked or done with)

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
    titles = "\n".join(f"- 《{cards[d]['title']}》" for d in sc["docs"][:20]) + (f"\n（共 {len(sc['docs'])} 份，题面没有点名文档，检索在这些文档里进行）" if len(sc["docs"]) > 20 else "")
    print(f"{qid}（{q['type']}）{stem}\n" + (f"方向：选判为“{asked}”的选项\n" if not is_calc else ""))

    needs = {}

    def add_need(n, origin):
        what = str(n.get("what", "")).strip()
        if not what or any(loose(what) == loose(m["what"]) for m in needs.values()):
            return None
        nid = f"n{len(needs) + 1}"
        needs[nid] = {"id": nid, "what": what, "kind": "原文" if str(n.get("kind")) == "原文" else "数值", "unit": str(n.get("unit") or ""),
                      "targets": [str(o).strip().upper() if not is_calc else str(o).strip() for o in n.get("targets") or n.get("options") or []],
                      "queries": [str(x) for x in n.get("queries") or []][:3], "value": None, "history": [], "origin": origin,
                      "since": int(re.sub(r"\D", "", origin) or 0)}
        return nid

    # ------------------------------------------------------------------ 0 plan
    plan = parse_json(call("plan", SYSTEM_PLAN.format(docs=titles, targets_rule=PLAN_TARGETS_CALC if is_calc else PLAN_TARGETS_CHOICE,
                                                      target_word=word, query_rules=QUERY_RULES,
                                                      output=PLAN_OUT_CALC if is_calc else PLAN_OUT_CHOICE), stem, 2500))
    if is_calc:
        parts = [{"id": str(p.get("id") or f"p{i}"), "what": str(p.get("what", "")), "format": str(p.get("format", ""))}
                 for i, p in enumerate(plan.get("parts") or [], 1)] or [{"id": "p1", "what": "题目所求", "format": ""}]
        targets = {p["id"]: p for p in parts}
    else:
        targets = {k: {"id": k, "what": v.strip()} for k, v in options.items()}
    for n in plan.get("needs") or []:
        add_need(n, "读题")
    status = {t: {"ready": False, "uses": [], "lacking": ""} for t in targets}
    print("需求清单：\n" + "\n".join(f"  {n['id']} {n['what']}（{n['kind']}，{n['unit']}）→ {'、'.join(n['targets'])}｜{n['queries']}" for n in needs.values()))
    unfilled = lambda: [n for n in needs.values() if n["value"] is None]
    open_targets = lambda: [t for t, s in status.items() if not s["ready"]]

    def to_search():
        """The unmet needs of the targets not yet decidable (all unmet needs before the first fill)."""
        ot = set(open_targets())
        return [n for n in unfilled() if not n["targets"] or set(n["targets"]) & ot]

    # ------------------------------------------------------------------ 1-3 search / fill / check
    stop = ""
    for rnd in range(1, MAX_ROUNDS + 1):
        wanted = to_search()
        if not open_targets():
            stop = "所有目标都已能判断"
            break
        if not wanted and not active:
            stop = "还不能判断的目标没有可检索的需求"
            break
        lists, log = [], []
        for n in wanted:
            for query in n["queries"][:QUERIES_PER_NEED]:
                where, text = narrow(q, query, sc)
                got = [u for u, s in bm25_scores(text, where)[:TOP_PER_QUERY] if s > 0]
                lists.append([u for u in got if aside.get(u, 0) <= n["since"]])   # set aside after the need existed: judged against it
                log.append({"need": n["id"], "query": query, "searched": text.strip(), "docs": len(where), "chunks": got})
        new = []
        for pos in range(TOP_PER_QUERY):
            for got in lists:
                if pos < len(got) and got[pos] not in new and got[pos] not in active:
                    new.append(got[pos])
        new = new[:MAX_NEW]
        for u in new:
            masked.discard(u)
            done_with.discard(u)
            aside.pop(u, None)
            if u in store:
                continue
            x = unit(u)
            store[u] = {"title": cards[u.split("#")[0]]["title"], "path": x["path"] or "", "text": x["body"][:MAX_BODY], "type": x["type"], "round": rnd}
        in_play = active + new
        row = {"round": rnd, "searched_needs": [n["id"] for n in wanted], "queries": log, "new": new, "carried": list(active), "in_play": in_play}
        rounds.append(row)
        print(f"第 {rnd} 轮检索：需求 {[n['id'] for n in wanted]}，{len(log)} 个查询，新块 {len(new)}，带入 {len(active)}")
        if rnd > 1 and not new and not active:
            stop = "这一轮检索没有新的原文块，也没有带入的块"
            row["note"] = stop
            break
        # 2 fill
        filled = [n for n in needs.values() if n["value"]]
        user = (f"题目：\n{stem}\n\n" + (("答案各部分：\n" + "\n".join(f"- {t['id']}｜{t['what']}｜{t['format']}" for t in targets.values()) + "\n\n") if is_calc else "")
                + "已取到的数据：\n" + ("\n".join(f"- {n['id']}｜{n['what']}｜{n['value']['text'][:80]} {n['value']['unit']}｜用于 {'、'.join(n['targets'])}" for n in filled) or "（无）")
                + "\n\n还缺的数据需求：\n" + ("\n".join(f"- {n['id']}｜{n['what']}｜{n['kind']}" + (f"｜单位 {n['unit']}" if n["unit"] else "")
                                                   + f"｜用于 {'、'.join(n['targets'])}" + (f"｜上一轮：{n['history'][-1]}" if n["history"] else "") for n in unfilled()) or "（无）")
                + f"\n\n原文块（共 {len(in_play)} 块）：\n\n" + "\n\n".join(chunk_text(u) for u in in_play))
        example = "p1" if is_calc else "A"
        fill = parse_json(call(f"fill{rnd}", SYSTEM_FILL.format(target_word=word, query_rules=QUERY_RULES, example=example), user, 3500))
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
                text, _ = number(v.get("value"))
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
                needs[nid]["history"].append(f"第 {rnd} 轮没找到：{m.get('reason', '')}")
                if qs:
                    needs[nid]["queries"] = qs
                poor.append({"need": nid, "reason": str(m.get("reason", "")), "queries": qs})
        added = []
        for nn in fill.get("new_needs") or []:
            nid = add_need(nn, f"第 {rnd} 轮追加")
            if not nid:
                continue
            added.append(nid)
            ch = str(nn.get("chunk") or "").strip("[] ")
            if nn.get("value") and ch in store:         # found in this batch already: checked like any value
                n = needs[nid]
                if n["kind"] == "数值":
                    text, _ = number(nn.get("value"))
                    ok = bool(text) and squash(text) in squash(store[ch]["text"])
                else:
                    ok = len(loose(nn.get("value"))) >= 4 and loose(nn.get("value")) in loose(store[ch]["text"])
                checks.append({"need": nid, "value": nn.get("value"), "unit": nn.get("unit"), "chunk": ch, "ok": ok, "why": "" if ok else f"追加需求所填的值不在块 {ch} 里"})
                if ok:
                    n["value"] = {"text": str(nn.get("value")), "number": number(nn.get("value"))[1] if n["kind"] == "数值" else None,
                                  "unit": str(nn.get("unit") or n["unit"]), "chunk": ch, "note": "追加时填入", "round": rnd}
        for t, s in (fill.get("targets") or {}).items():
            t = str(t).strip().upper() if not is_calc else str(t).strip()
            if t in status:
                status[t] = {"ready": bool(s.get("ready")), "uses": [str(x) for x in s.get("uses") or []], "lacking": str(s.get("lacking") or "")}
        for t, s in status.items():                     # a target whose own needs are still empty is not ready, whatever was said
            own = [n for n in needs.values() if t in n["targets"]]
            if s["ready"] and any(n["value"] is None and n["id"] in s["uses"] for n in own):
                s["ready"], s["lacking"] = False, "用到的需求还没有取到值"
        row.update({"serves": serves, "checks": checks, "missing": poor, "added": added,
                    "targets": {t: dict(s) for t, s in status.items()}, "filled": [c["need"] for c in checks if c["ok"]]})
        # context: keep the chunks that serve a need still searched for
        still = {n["id"] for n in to_search()}
        keep = [u for u in in_play if set(serves.get(u, [])) & still]
        irrelevant = [u for u in in_play if not serves.get(u)]
        used = [u for u in in_play if u not in keep and u not in irrelevant]
        masked.update(irrelevant)
        done_with.update(used)
        aside.update({u: rnd for u in irrelevant + used})
        active = keep
        row.update({"kept": keep, "masked": irrelevant, "used": used})
        print(f"  填值：通过 {sum(c['ok'] for c in checks)}，未通过 {sum(not c['ok'] for c in checks)}；追加需求 {added}；"
              f"还不能判断 {open_targets()}；不相关移出 {len(irrelevant)}，已用完移出 {len(used)}，保留 {len(keep)}")
    else:
        stop = f"到达 {MAX_ROUNDS} 轮上限" if open_targets() else "所有目标都已能判断"
    if not stop:
        stop = "所有目标都已能判断"
    print(f"停止：{stop}")

    # ------------------------------------------------------------------ 4 final
    missing = unfilled()
    V = {nid: (n["value"]["number"] if n["kind"] == "数值" else n["value"]["text"]) for nid, n in needs.items() if n["value"]}
    table = "\n".join(f"| {nid} | {n['what']} | {n['value']['text']} | {n['value']['unit']} | {n['value']['chunk']} | {'、'.join(n['targets'])} |"
                      for nid, n in needs.items() if n["value"] and n["kind"] == "数值")
    quotes = "\n".join(f"- {nid}｜{n['what']}（{n['value']['chunk']}，用于 {'、'.join(n['targets'])}）：“{n['value']['text']}”"
                       for nid, n in needs.items() if n["value"] and n["kind"] == "原文")
    user = (f"题目：\n{stem}\n\n" + (("答案各部分（按顺序写进 answers）：\n" + "\n".join(f"- {t['id']}｜{t['what']}｜{t['format']}" for t in targets.values()) + "\n\n") if is_calc else "")
            + f"已取到的数值（V[编号] 是去掉千分位的 float，单位按表）：\n| 编号 | 内容 | 原文数值 | 单位 | 来源块 | 用于 |\n|---|---|---|---|---|---|\n{table or '（无）'}\n\n"
            f"已取到的原文摘录（V[编号] 是摘录字符串）：\n{quotes or '（无）'}\n\n"
            "没找到的数据：\n" + ("\n".join(f"- {n['id']}｜{n['what']}｜用于 {'、'.join(n['targets'])}" for n in missing) or "（无）"))
    final, error, code = None, None, ""
    for attempt in range(2):
        content = call("final" if attempt == 0 else "final_retry", SYSTEM_FINAL_CALC if is_calc else SYSTEM_FINAL_CHOICE,
                       user if attempt == 0 else user + f"\n\n上一次的代码：\n```python\n{code}\n```\n运行出错：\n{error}\n请改正后重新输出完整代码。", 3000)
        m = re.search(r"```(?:python)?\s*\n(.*?)```", content, re.S)
        code = m.group(1) if m else content
        final, error = run_code(code, V)
        if final and not error:
            bad = (["answers 是空的"] if is_calc and not final["answers"] else []) if is_calc else [k for k in options if final["verdicts"].get(k) not in VERDICTS]
            if not bad:
                break
            error, final = (f"answers 是空的，要按顺序写入 {len(targets)} 个部分" if is_calc else
                            f"verdicts 缺少或取值不对的选项：{bad}（每个选项都要给 正确 / 错误 / 无依据）"), None
        print("代码出错：" + str(error)[:300])
    knowns = [v for v in V.values() if isinstance(v, float)]
    literals = sorted({float(x.replace(",", "")) for x in re.findall(r"\d[\d,]{4,}(?:\.\d+)?", code)
                       if not any(abs(float(x.replace(",", "")) - k) < 1e-6 * max(1, abs(k)) for k in knowns)})
    note = ""
    if is_calc:
        verdicts = {}
        answer = "；".join(final["answers"]) if final else ""
        how = "代码按题目要求的格式算出各部分"
    else:
        verdicts = {k: (final["verdicts"].get(k) if final else None) for k in options}
        chosen, unfounded = choose(asked, verdicts)
        how = "代码按题干方向从各选项结论里挑出判为“" + asked + "”的选项"
        if q["type"] == "单选题" and not chosen and len(unfounded) == 1 and missing:
            chosen, how = unfounded, how + "；没有选项符合，按排除法选唯一缺数据的选项"
        answer = "".join(chosen)
        note = caveat(asked, [k for k in unfounded if k not in chosen])
    if missing:
        note = (note + "；" if note else "") + "没找到的值：" + "、".join(f"{n['id']} {n['what']}" for n in missing)
    print(f"\n" + (f"结论：{verdicts}\n" if not is_calc else "") + f"答案：{answer}（{how}）" + (f"\n注意：{note}" if note else ""))
    out = {"qid": qid, "type": q["type"], "question": q["question"].strip(), "options": options, "model": llm.model,
           "direction": {"asked": asked, "clause": clause, "word": dword}, "targets": targets, "target_status": status,
           "needs": needs, "rounds": rounds, "stop": stop, "code": code, "code_output": final, "code_error": error,
           "literals_in_code": literals, "verdicts": verdicts, "answers": final["answers"] if (final and is_calc) else None,
           "answer": answer, "answered_by": how, "caveat": note,
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
    L = [f"# {o['qid']}：需求清单流程作答记录（v2）", "",
         f"{o['type']}。模型 {o['model']}，温度 0，思考模式关闭。{o['n_calls']} 次模型调用，{o['seconds']} 秒；"
         f"tokens 输入 {u['prompt_tokens']} + 输出 {u['completion_tokens']} = {u['total_tokens']}。停止原因：{o['stop']}。",
         f"答案：**{o['answer'] or '（空）'}**（{o['answered_by']}）。" + (f"注意：{o['caveat']}。" if o["caveat"] else ""), "",
         "## 题目", "", "```text", o["question"] + "".join(f"\n{k}. {v.strip()}" for k, v in o["options"].items()), "```", "",
         "## 需求清单（最终状态）", "", "| 编号 | 来源 | 内容 | 类型 | 用于 | 取到的值 | 来源块 | 第几轮 | 经过 |", "|---|---|---|---|---|---|---|---|---|"]
    for n in o["needs"].values():
        v = n["value"]
        L.append(f"| {n['id']} | {n['origin']} | {n['what']} | {n['kind']} | {'、'.join(n['targets'])} | {(v['text'][:60] + ' ' + v['unit']) if v else '**没找到**'} | "
                 f"{('`' + v['chunk'] + '`') if v else ''} | {v['round'] if v else ''} | {'；'.join(n['history'])} |")
    L += ["", "## 各目标的数据是否够用（最后一次填值时的判断）", "", "| 目标 | 内容 | 能判断 | 用到 / 还缺 |", "|---|---|---|---|"]
    L += [f"| {t} | {o['targets'][t]['what'][:60]} | {'是' if s['ready'] else '否'} | {'、'.join(s['uses']) if s['ready'] else s['lacking']} |" for t, s in o["target_status"].items()]
    L += ["", "## 逐步过程", ""]
    calls = iter(o["calls"])

    def show(c):
        return [f"### 调用：{c['stage']}（输入 {c['usage']['prompt_tokens']}，输出 {c['usage']['completion_tokens']}）", "",
                "<details><summary>系统提示</summary>", "", "```text", c["system"], "```", "", "</details>", "",
                "**发给模型的内容：**", "", "```text", c["user"], "```", "", "**模型输出（原样）：**", "", "```text", c["response"], "```", ""]
    L += show(next(calls))
    for r in o["rounds"]:
        L += [f"### 第 {r['round']} 轮检索（代码，不调用模型）", "", f"检索的需求：{'、'.join(r['searched_needs']) or '无'}", "",
              "| 需求 | 检索词 | 实际检索的文本 | 文档数 | 返回 |", "|---|---|---|---|---|"]
        L += [f"| {x['need']} | {x['query']} | {x['searched']} | {x['docs']} | {', '.join(x['chunks'])} |" for x in r["queries"]]
        L += ["", f"去重后新块 {len(r['new'])} 个：{', '.join(r['new']) or '无'}；带入上一轮仍需要的块 {len(r['carried'])} 个：{', '.join(r['carried']) or '无'}。", ""]
        if "serves" not in r:
            L += [r.get("note", ""), ""]
            continue
        L += show(next(calls))
        L += ["**代码核对填值：**", ""] + ([f"- {c['need']} = {c['value']} {c['unit'] or ''}（{c['chunk']}）→ {'通过' if c['ok'] else '未通过：' + c['why']}" for c in r["checks"]] or ["- 无"])
        L += ["", "**召回不好、改写检索词的需求：**", ""] + ([f"- {m['need']}：{m['reason']} → {m['queries']}" for m in r["missing"]] or ["- 无"])
        L += ["", f"**追加的需求：** {'、'.join(r['added']) or '无'}", "",
              "**各目标是否能判断：** " + "；".join(f"{t} {'能' if s['ready'] else '不能（' + s['lacking'] + '）'}" for t, s in r["targets"].items()), "",
              f"**上下文：** 不相关、移入证据库 {len(r['masked'])} 块（{', '.join(r['masked']) or '无'}）；已取完值、移出 {len(r['used'])} 块；"
              f"仍服务于还要查的需求、保留到下一轮 {len(r['kept'])} 块（{', '.join(r['kept']) or '无'}）。", ""]
    for c in calls:
        L += show(c)
    L += ["## 代码运行结果", ""]
    if o["code_output"]:
        L += ["| 中间结果 | 值 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in o["code_output"]["R"].items()]
        if o["answers"] is not None:
            L += ["", "答案各部分：" + "；".join(o["answers"]), ""] + [f"- {k}：{v}" for k, v in o["code_output"]["why"].items()]
        else:
            L += ["", "| 选项 | 结论 | 理由 |", "|---|---|---|"] + [f"| {k} | {o['verdicts'].get(k)} | {o['code_output']['why'].get(k, '')} |" for k in o["options"]]
    else:
        L += ["代码没有跑通：", "", "```text", str(o["code_error"]), "```"]
    if o["literals_in_code"]:
        L += ["", f"注意：代码里直接写了不在清单里的数 {o['literals_in_code']}。"]
    return "\n".join(L)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
