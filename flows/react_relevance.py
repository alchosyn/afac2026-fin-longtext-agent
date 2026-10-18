# -*- coding: utf-8 -*-
"""ReAct on one B-list choice question with two context strategies for the returned chunks
(demo, 2026-10-06; the ledger tools, the checks and the answer step are those of
react_conclude.py, only what happens to the chunk text differs).

    none       every chunk keeps its full text where it first came back; a chunk returned
               again is not sent a second time
    relevance  one more tool, judge: the model judges every chunk that came back against the
               option or target it is working on (相关 / 不相关 / 拿不准); the judgements are
               written to a JSON file after every round. Before the next request the chunks
               judged 不相关 are replaced by a one-line placeholder and their text stays only in
               the evidence store (read brings it back); 相关 and 拿不准 keep the full text.
               A chunk the model has seen but not judged is kept in full (as 拿不准) and the
               model is reminded once.

    python flows/react_relevance.py fin_b_002 --mask relevance [--steps 20] [--budget 200000]

Writes output/react_relevance/react_relevance_<qid>_<mask>.json (every step: what the model wrote, every call,
every reply, what the system did, the context sent), .md (the same as a readable record),
_store.json, _ledger.json and, for relevance, _judgments.json.
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
from agent.tools import calc, render_calc                   # noqa: E402
from react_conclude import (CALC, CONCLUDE, MAX_BODY, READ, RECORD, SEARCH, TARGET_CAP,  # noqa: E402
                            VERDICTS, fn, number, same, squash)

LABELS = ("相关", "不相关", "拿不准")
JUDGE = fn("judge", "判断检索或重读返回的原文块与当前要核实的选项或目标是否相关，每块一条。系统把判断写入判断文件；"
           "判为不相关的块在下一次请求前从上下文移入证据库，只留一行占位符（需要时可用 read 取回）；相关和拿不准的保留全文。"
           "可以和 search、record、calc、conclude 放在同一步。",
           {"judgments": {"type": "array", "description": "每个块一条判断",
                          "items": {"type": "object", "properties": {
                              "chunk": {"type": "string", "description": "块编号"},
                              "relevance": {"type": "string", "enum": list(LABELS)},
                              "for": {"type": "string", "description": "和哪个选项或目标有关；不相关的写当前在查的目标"},
                              "reason": {"type": "string", "description": "一句话理由"}},
                              "required": ["chunk", "relevance", "for", "reason"]}}}, ["judgments"])

SYSTEM = """你是金融文档问答助手，用工具逐个核实一道{kind}的选项。
本题涉及的文档：
{docs}
工具：{tools}。
规则：
1. 判断只能依据工具返回的原文和计算结果，不能凭常识或记忆。
2. {context}
3. 取数时核对指标名称、期间、口径（合并或母公司）和单位；优先用表格里的精确数字。表格里直接给出的增减比例等，可以直接登记使用。要用于计算的数先用 record 登记，系统当场核对。
4. 一个选项判完就用 conclude 登记，不要攒到最后。原文或计算结果支持的判“正确”；与之矛盾，或说法超出原文的判“错误”；查不到相关原文的判“无依据”，不要猜。
5. conclude 的结论只表示这个选项的说法（连同题干给出的主体和话题）与原文是否一致。每个选项单独判，不和其他选项比较。题干要找的是正确的还是错误的选项，都不影响结论：说法与原文一致就判“正确”，哪怕题干问的是哪项错误。选哪些选项由系统按题干决定。
6. 一步里可以同时调用多个工具；多个选项要算的内容尽量放在同一次 calc 里。
7. 所有选项都登记了结论并通过系统核对后，系统会根据登记结果结束作答；你不需要输出最终答案。"""
CONTEXT = {
    "none": "检索返回的原文全文会一直保留在上下文里，同一块再次返回时不重复发送。",
    "relevance": "每次 search 或 read 返回原文后，下一步先用 judge 对返回的每一块判断它与当前要核实的选项或目标是否相关："
                 "相关、不相关、拿不准。判为不相关的块在下一次请求前移出上下文，只留一行占位符，原文存在证据库里，需要时用 read 取回；"
                 "相关和拿不准的块保留全文。没有判断的块按拿不准保留。",
}
TOOL_NAMES = {"none": "search（检索原文块）、read（按块编号重读）、record（登记取到的数，系统当场核对）、calc（精确计算）、conclude（登记一个选项的判断）",
              "relevance": "search（检索原文块）、judge（判断返回的块是否相关）、read（按块编号重读移入证据库的块）、"
                           "record（登记取到的数，系统当场核对）、calc（精确计算）、conclude（登记一个选项的判断）"}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("qid")
    ap.add_argument("--mask", choices=("none", "relevance"), required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--budget", type=int, default=200000)
    args = ap.parse_args(argv)
    q = next(x for x in load_b_questions() if x["qid"] == args.qid)
    options = q.get("options") or {}
    sc, cards, docs = scope(q), load_cards(), {}
    base = ROOT / "output" / "react_relevance" / f"react_relevance_{q['qid']}_{args.mask}"
    base.parent.mkdir(parents=True, exist_ok=True)

    def unit(uid):
        d = uid.split("#")[0]
        docs.setdefault(d, Doc(d, cards[d]))
        return docs[d].by_id[uid]

    store, counts = {}, {}
    first, home = {}, {}                            # turn where a chunk first came back / where its full text is now (None: masked)
    status, judgments, reminded = {}, [], set()     # 待判断 / 相关 / 不相关 / 拿不准, every judge entry, chunks reminded about
    records, calcs, conclusions = [], {}, {}        # the ledger
    shown = set()
    in_question = {v for _, v in (number(x) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", q["question"] + " ".join(options.values())))}
    head = lambda u: f"[{u}]《{store[u]['title']}》{store[u]['path']}"
    full = lambda u: head(u) + "\n" + store[u]["text"]
    ref = lambda u: head(u) + " ——（全文在上下文另一处，未重复发送）"

    def stub(u):
        j = next((x for x in reversed(judgments) if x["chunk"] == u and x["relevance"] == "不相关"), None)
        taken = [r for r in records if r["chunk"] == u and r["ok"]]
        s = head(u) + " ——【不相关｜已移入证据库】" + (f"第 {j['step']} 步判为不相关：{j['reason']}" if j else "")
        if taken:
            s += "；已从中取数：" + "；".join(f"{r['name']} = {r['value']} {r['unit']}" for r in taken)
        return s + "。需要原文时用 read 取回。"

    def show(u, i, compress):
        h = home.get(u) if compress else first.get(u)
        return full(u) if h == i else ref(u) if h is not None else stub(u)

    def render(compress=True):
        out = []
        for i, t in enumerate(turns):
            if "chunks" in t:
                out.append({"role": "tool", "tool_call_id": t["tool_call_id"], "content": "\n\n".join(show(u, i, compress) for u in t["chunks"])})
            else:
                out.append({k: t[k] for k in ("role", "content", "tool_calls", "tool_call_id") if k in t})
        return out

    def composition():
        c = {"系统提示": 0, "题目": 0, "系统消息": 0, "模型写的文字": 0, "judge 参数": 0, "其他工具调用参数": 0,
             "原文块全文": 0, "占位符和引用": 0, "工具回复": 0}
        for i, t in enumerate(turns):
            if t["role"] == "system":
                c["系统提示"] += len(t["content"])
            elif t["role"] == "user":
                c["题目" if i == 1 else "系统消息"] += len(t["content"])
            elif t["role"] == "assistant":
                c["模型写的文字"] += len(t.get("content") or "")
                for x in t.get("tool_calls") or []:
                    c["judge 参数" if x["function"]["name"] == "judge" else "其他工具调用参数"] += len(x["function"]["arguments"]) + len(x["function"]["name"])
            elif "chunks" in t:
                for u in t["chunks"]:
                    s = show(u, i, True)
                    c["原文块全文" if home.get(u) == i else "占位符和引用"] += len(s)
            else:
                c["工具回复"] += len(t.get("content") or "")
        return c

    size = lambda msgs: sum(len(m.get("content") or "") + sum(len(x["function"]["arguments"]) + len(x["function"]["name"])
                                                               for x in m.get("tool_calls") or []) for m in msgs)

    # ------------------------------------------------------------------ tools
    def do_search(a, step, events):
        target = re.sub(r"\s+", "", str(a.get("target") or "未说明"))
        if counts.get(target, 0) >= TARGET_CAP:
            events.append(f"检索被拒：目标“{a.get('target')}”已检索 {TARGET_CAP} 次")
            return None, f"目标“{a.get('target')}”已检索 {TARGET_CAP} 次，本次不再检索：按查不到处理，继续别的目标。"
        counts[target] = counts.get(target, 0) + 1
        where, text = narrow(q, str(a.get("query", "")), sc)
        got = [u for u, s in bm25_scores(text, where)[:min(int(a.get("top") or args.top), 8)] if s > 0]
        events.append(f"检索：查询“{a.get('query')}”在 {len(where)} 份文档里搜“{text.strip()}”，返回 {len(got)} 块")
        return (got, None) if got else (None, "没有命中，换一个查询。")

    def do_read(a, step, events):
        u = str(a.get("chunk", "")).strip("[] ")
        if u not in store:
            return None, f"没有编号为 {u} 的块：read 只能重读检索返回过的块。"
        if home.get(u) is not None:
            events.append(f"重读 {u}：全文仍在上下文里，没有重发")
            return None, f"块 {u} 的全文仍在上下文里（见上文），没有重发。"
        events.append(f"重读 {u}：从证据库取回，重新发送全文")
        return [u], None

    def do_judge(a, step, events):
        ok, bad = [], []
        for j in a.get("judgments") or []:
            u, lab = str(j.get("chunk", "")).strip("[] "), str(j.get("relevance", "")).strip()
            if u not in store or home.get(u) is None:
                bad.append(f"{u}（不在上下文的全文里）")
                continue
            if lab not in LABELS:
                bad.append(f"{u}（relevance 只能是 {' / '.join(LABELS)}）")
                continue
            status[u] = lab
            judgments.append({"step": step, "chunk": u, "relevance": lab, "for": str(j.get("for", "")), "reason": str(j.get("reason", ""))})
            ok.append((u, lab))
        by = {lab: [u for u, x in ok if x == lab] for lab in LABELS}
        events.append("相关性判断：" + "；".join(f"{lab} {len(v)} 块" for lab, v in by.items() if v) + (f"；无效 {bad}" if bad else ""))
        waiting = [u for u, s in status.items() if s == "待判断" and home.get(u) is not None]
        reply = ("已写入判断文件：" + "；".join(f"{lab} {'、'.join(v)}" for lab, v in by.items() if v)
                 + ("。判为不相关的块在下一次请求前移入证据库。" if by["不相关"] else "。")
                 + (f"无效：{'、'.join(bad)}。" if bad else "")
                 + (f"还没有判断的块：{'、'.join(waiting)}。" if waiting else ""))
        return None, reply

    def do_record(a, step, events):
        name, ch, unit_ = str(a.get("name", "")).strip(), str(a.get("chunk", "")).strip("[] "), str(a.get("unit", "")).strip()
        text, value = number(a.get("value"))
        ok = bool(text) and ch in store and squash(text) in squash(store[ch]["text"])
        records[:] = [r for r in records if r["name"] != name]
        records.append({"name": name, "value": text or str(a.get("value")), "number": value, "unit": unit_, "chunk": ch, "ok": ok, "step": step})
        events.append(f"登记取数：{name} = {text} {unit_}（{ch}）→ 代码核对：{'已核实' if ok else '不一致'}")
        if ok:
            return None, f"已核实并登记：{name} = {text} {unit_}。"
        why = f"没有编号为 {ch} 的块" if ch not in store else f"块 {ch} 里没有数值 {a.get('value')}"
        return None, f"不一致，未通过核对：{why}。请核对数值和块编号后重新 record。"

    def do_calc(a, step, events):
        exprs = a.get("expressions") or {}
        exprs = {"结果": exprs} if isinstance(exprs, str) else exprs if isinstance(exprs, dict) else {}
        variables = a.get("variables") if isinstance(a.get("variables"), dict) else {}
        a = {**a, "expressions": exprs, "variables": variables}
        res = calc(exprs, variables)
        known = [r["number"] for r in records if r["ok"]] + list(shown) + list(in_question) + [c["value"] for c in calcs.values()]
        lits = [number(x)[1] for e in exprs.values() for x in re.findall(r"\d[\d,]*(?:\.\d+)?", str(e))]
        lits += [float(v) for v in variables.values() if isinstance(v, (int, float))]
        loose = sorted({x for x in lits if len(re.sub(r"\D", "", f"{x:.10g}")) >= 5 and not any(same(x, k) for k in known)})
        for name, v in res["results"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                calcs[name] = {"value": float(v), "expression": str((a.get("expressions") or {}).get(name)), "step": step, "loose": loose}
                shown.add(float(f"{round(v, 6):g}"))
        text = render_calc(res)
        events.append(f"计算：{len(res['results'])} 个结果入账" + ("；出错：" + res["error"] if "error" in res else "")
                      + (f"；表达式里有未登记的数 {loose}" if loose else ""))
        if loose:
            text += f"\n提示：表达式里的数 {', '.join(f'{x:g}' for x in loose)} 还没有用 record 登记来源。登记后重新计算，结论才能通过核对。"
        return None, text

    def do_conclude(a, step, events):
        k, verdict = str(a.get("option", "")).strip().upper()[:1], str(a.get("verdict", "")).strip()
        ev = [str(e).strip("[] ") for e in a.get("evidence") or []]
        if k not in options:
            return None, f"没有选项 {a.get('option')}，可选：{'、'.join(options)}。"
        if verdict not in VERDICTS:
            return None, f"verdict 只能是 {' / '.join(VERDICTS)}。"
        problems = []
        for e in ev:
            if e in calcs and calcs[e]["loose"]:
                problems.append(f"计算结果 {e} 用到了未登记的数 {calcs[e]['loose']}")
            elif e not in store and e not in calcs:
                problems.append(f"依据“{e}”既不是检索返回过的块，也不是 calc 返回过的结果名")
        if verdict != "无依据" and not any(e in store or e in calcs for e in ev):
            problems.append("判为正确或错误必须给出至少一条依据（块编号或 calc 结果名）")
        conclusions[k] = {"verdict": verdict, "evidence": ev, "reason": str(a.get("reason", "")), "ok": not problems, "problems": problems, "step": step}
        missing = [x for x in options if x not in conclusions or not conclusions[x]["ok"]]
        events.append(f"登记结论：选项 {k} = {verdict}，依据 {ev} → 代码核对：{'通过' if not problems else '未通过（' + '；'.join(problems) + '）'}")
        if problems:
            return None, f"已登记选项 {k} = {verdict}，但未通过核对：{'；'.join(problems)}。请补上依据后重新 conclude。"
        return None, f"已登记并通过核对：选项 {k} = {verdict}。" + (f"还没有通过核对的结论的选项：{'、'.join(missing)}。" if missing else "所有选项都已有通过核对的结论。")

    TOOLS = {"search": do_search, "read": do_read, "record": do_record, "calc": do_calc, "conclude": do_conclude}
    if args.mask == "relevance":
        TOOLS["judge"] = do_judge
    SCHEMAS = [SEARCH] + ([JUDGE] if args.mask == "relevance" else []) + [READ, RECORD, CALC, CONCLUDE]

    def ledger_text():
        L = ["已核实的取数："] + [f"- {r['name']} = {r['value']} {r['unit']}（{r['chunk']}）" for r in records if r["ok"]]
        L += ["计算结果："] + [f"- {n} = {c['value']:g}（{c['expression']}）" for n, c in calcs.items()]
        L += ["各选项结论："] + [f"- {k}：{c['verdict']}｜依据 {'、'.join(c['evidence']) or '无'}｜{c['reason']}" for k, c in sorted(conclusions.items())]
        return "\n".join(L)

    asked, clause, word = direction(q["question"])
    picked = lambda: choose(asked, {k: conclusions[k]["verdict"] if k in conclusions else None for k in options})

    def consistent():
        chosen, unfounded = picked()
        if unfounded:
            return f"选项 {'、'.join(unfounded)} 判为无依据（无依据的选项不会被选）"
        if q["type"] == "单选题" and len(chosen) != 1:
            return f"按题干应选判为“{asked}”的选项，单选题应当恰好一个，现在是 {len(chosen)} 个"
        return "" if chosen else f"按题干应选判为“{asked}”的选项，现在一个也没有"

    def write_judgments():
        if args.mask == "relevance":
            Path(str(base) + "_judgments.json").write_text(json.dumps(
                {"qid": q["qid"], "judgments": judgments, "status": status}, ensure_ascii=False, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ loop
    titles = "\n".join(f"- 《{cards[d]['title']}》" for d in sc["docs"])
    if not sc["named"]:
        titles = ("（题面没有点名文档，search 在下面这些文档里找）\n" + titles if len(sc["docs"]) <= 20
                  else f"（题面没有点名文档，search 在本领域的 {len(sc['docs'])} 份文档里找）")
    system = SYSTEM.format(kind=q["type"], docs=titles, tools=TOOL_NAMES[args.mask], context=CONTEXT[args.mask])
    listed = "".join(f"\n{k}. {v.strip()}" for k, v in options.items())
    turns = [{"role": "system", "content": system}, {"role": "user", "content": q["question"].strip() + listed}]
    llm = Qwen(log_path=ROOT / "logs" / f"react_relevance_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    trace, usage, t0 = [], [0, 0], time.time()
    answer, how, nudges, reviewed, wrap_up = None, None, 0, False, False
    print(f"{q['qid']} [{args.mask}] {q['question'].strip()}{listed}\n方向：选判为“{asked}”的选项\n")
    step = 0
    while answer is None:
        step += 1
        if (step > args.steps or sum(usage) >= args.budget) and not wrap_up:
            wrap_up = True
            missing = [k for k in options if k not in conclusions or not conclusions[k]["ok"]]
            turns.append({"role": "user", "content": f"步数或 token 预算已用完。请现在就用 conclude 给出这些选项的判断：{'、'.join(missing)}；没有依据的写“无依据”。"})
        sent = render()
        state = {"full": [u for u, h in home.items() if h is not None], "masked": [u for u, h in home.items() if h is None],
                 "composition": composition(), "chars_sent": size(sent), "chars_none": size(render(False)), "store": len(store)}
        for attempt in range(3):
            try:
                resp = llm.client.chat.completions.create(model=llm.model, messages=sent, temperature=0, max_tokens=1500,
                                                          extra_body={"enable_thinking": False}, tools=[CONCLUDE] if wrap_up else SCHEMAS)
                break
            except Exception as e:                  # noqa: BLE001 - transient API errors
                if attempt == 2:
                    raise
                print(f"API 出错，重试：{e}")
                time.sleep(5)
        msg = resp.choices[0].message
        usage[0] += resp.usage.prompt_tokens
        usage[1] += resp.usage.completion_tokens
        calls = msg.tool_calls or []
        events = []
        row = {"step": step, "content": msg.content or "", "reasoning": getattr(msg, "reasoning_content", None), "calls": [],
               "context": state, "events": events, "sent": sent,
               "usage": {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens, "cumulative": sum(usage)}}
        trace.append(row)
        print(f"--- 第 {step} 步（输入 {resp.usage.prompt_tokens}，输出 {resp.usage.completion_tokens}，累计 {sum(usage)}）"
              f" 全文块 {len(state['full'])}，占位符块 {len(state['masked'])}")
        if msg.content:
            print("模型：" + msg.content.strip())
        turns.append({"role": "assistant", "content": msg.content or ""} | ({"tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]} if calls else {}))
        for c in calls:
            try:
                a = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                a = {}
            name = c.function.name
            got, reply = TOOLS[name](a, step, events) if name in TOOLS else (None, f"没有名为 {name} 的工具。")
            turn = {"role": "tool", "tool_call_id": c.id}
            if got:
                turn["chunks"] = got
                i = len(turns)
                for u in got:
                    if u not in store:
                        x = unit(u)
                        store[u] = {"title": cards[u.split("#")[0]]["title"], "path": x["path"] or "", "text": x["body"][:MAX_BODY],
                                    "type": x["type"], "first_step": step, "returned": []}
                        first[u] = i
                    store[u]["returned"].append({"step": step, "tool": name, "query": a.get("query"), "target": a.get("target")})
                    if home.get(u) is None:             # new, or back from the store: full text here, to be judged again
                        home[u] = i
                        status[u] = "待判断"
                        reminded.discard(u)
            else:
                turn["content"] = reply
            turns.append(turn)
            row["calls"].append({"name": name, "arguments": a, "chunks": got, "reply": reply})
            print(f"调用：{name} {json.dumps(a, ensure_ascii=False)}")
            print("返回：" + (", ".join(got) if got else str(reply).replace("\n", "；")))
        # ---- context: chunks judged 不相关 leave the context before the next request
        if args.mask == "relevance":
            gone = [u for u, h in home.items() if h is not None and status.get(u) == "不相关"]
            for u in gone:
                home[u] = None
            row["masked_now"] = gone
            if gone:
                events.append(f"上下文：{len(gone)} 块判为不相关，换成占位符，原文移入证据库：{'、'.join(gone)}")
            seen = [u for u, h in home.items() if h is not None and status.get(u) == "待判断"
                    and store[u]["returned"][-1]["step"] < step and u not in reminded]
            if seen and not wrap_up:
                reminded.update(seen)
                turns.append({"role": "user", "content": f"这些块还没有判断相关性：{'、'.join(seen)}（暂按拿不准保留全文）。请用 judge 补上判断。"})
                events.append(f"提醒补判：{seen}")
            write_judgments()
        done = all(k in conclusions and conclusions[k]["ok"] for k in options)
        row["ledger"] = {"records_ok": sum(r["ok"] for r in records), "calcs": len(calcs),
                         "conclusions": {k: (c["verdict"], c["ok"]) for k, c in sorted(conclusions.items())}, "done": done}
        for e in events:
            print("系统：" + e)
        if done or wrap_up:
            why = consistent() if done else "到达上限"
            if (done and not why) or reviewed or wrap_up:
                answer = "".join(picked()[0])
                how = f"代码按题干方向从登记的结论里挑出判为“{asked}”的选项" + ("（到达上限）" if wrap_up else "")
                events.append(f"作答：{how}，答案 {answer or '（空）'}")
                print("系统：" + events[-1])
            else:
                reviewed = True
                turns.append({"role": "user", "content": f"所有选项都已有通过核对的结论。但{why}，请确认或修改（用 conclude），确认无误就直接回复“确认”。\n" + ledger_text()})
                events.append("系统把登记内容交还给模型：" + why)
                print("系统：" + events[-1])
        elif not calls:
            nudges += 1
            wrap_up = nudges > 2
            missing = [k for k in options if k not in conclusions or not conclusions[k]["ok"]]
            turns.append({"role": "user", "content": f"还没有通过核对的结论的选项：{'、'.join(missing)}。请继续核实，并用 conclude 登记；不要直接作答。"})
            events.append(f"模型没有调用工具，系统提醒还缺结论的选项：{missing}")
    # ------------------------------------------------------------------ record
    unfounded = picked()[1]
    ledger = {"records": records, "calcs": calcs, "conclusions": conclusions, "unfounded": unfounded,
              "caveat": caveat(asked, unfounded), "answer": answer, "answered_by": how}
    out = {"qid": q["qid"], "mask": args.mask, "type": q["type"], "question": q["question"].strip(), "options": options,
           "model": llm.model, "system": system, "direction": {"asked": asked, "clause": clause, "word": word},
           "answer": answer, "answered_by": how, "steps": len(trace),
           "calls": {n: sum(c["name"] == n for r in trace for c in r["calls"]) for n in TOOLS}, "targets": counts,
           "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)},
           "seconds": round(time.time() - t0, 1), "ledger": ledger, "judgments": judgments, "status": status,
           "trace": trace, "final_context": render()}
    base.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(str(base) + "_store.json").write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(str(base) + "_ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8")
    write_judgments()
    base.with_suffix(".md").write_text(report(out, store), encoding="utf-8")
    print(f"\n答案：{answer}（{how}）；{out['steps']} 步，tokens {usage[0]} + {usage[1]} = {sum(usage)} -> {base.relative_to(ROOT)}.json / .md")


def report(o, store):
    """The run as a Markdown record: everything the model wrote and called, what the system did."""
    u = o["usage"]
    L = [f"# {o['qid']}（{'相关性遮蔽' if o['mask'] == 'relevance' else '不遮蔽'}）作答记录", "",
         f"模型 {o['model']}，温度 0，思考模式关闭。共 {o['steps']} 步，{o['seconds']} 秒。工具调用："
         + "，".join(f"{k} {v} 次" for k, v in o["calls"].items()) + "。",
         f"答案：**{o['answer']}**（{o['answered_by']}）。",
         f"tokens：输入 {u['prompt_tokens']} + 输出 {u['completion_tokens']} = {u['total_tokens']}。", "",
         "## 给模型的系统提示", "", "```text", o["system"], "```", "", "## 题目", "", "```text",
         o["question"] + "".join(f"\n{k}. {v.strip()}" for k, v in o["options"].items()), "```", "", "## 逐步过程", "",
         "每一步是一次模型调用。“模型写的文字”是这一步模型输出的全部正文（思考模式关闭，没有单独的推理内容）；"
         "“调用”是它发出的全部工具调用和参数，原样照录；“系统”是代码做的事。", ""]
    for r in o["trace"]:
        c = r["context"]
        comp = "，".join(f"{k} {v}" for k, v in c["composition"].items() if v)
        L += [f"### 第 {r['step']} 步", "",
              f"- 发给模型：输入 {r['usage']['prompt_tokens']} tokens；上下文构成（字符数）：{comp}。",
              f"- 此时带全文的块 {len(c['full'])} 个：{', '.join(c['full']) or '无'}；占位符块 {len(c['masked'])} 个：{', '.join(c['masked']) or '无'}。",
              f"- 模型输出 {r['usage']['completion_tokens']} tokens；累计 {r['usage']['cumulative']}。", ""]
        if r.get("reasoning"):
            L += ["**模型的推理内容：**", "", "```text", r["reasoning"].strip(), "```", ""]
        L += ["**模型写的文字：**", "", "```text", r["content"].strip() or "（无）", "```", ""]
        for i, call in enumerate(r["calls"], 1):
            L += [f"**调用 {i}：{call['name']}**", "", "```json", json.dumps(call["arguments"], ensure_ascii=False, indent=1), "```", ""]
            if call["chunks"]:
                for x in call["chunks"]:
                    L += [f"- 返回 `{x}`　《{store[x]['title']}》　{store[x]['path']}", "", "  ```text"] + ["  " + line for line in store[x]["text"].split("\n")] + ["  ```", ""]
            else:
                L += ["返回：", "", "```text", str(call["reply"]), "```", ""]
        L += ["**系统：**", ""] + [f"- {e}" for e in r["events"]] + [""]
    if o["judgments"]:
        L += ["## 相关性判断（判断文件的内容）", "", "| 步 | 块 | 判断 | 针对 | 理由 |", "|---|---|---|---|---|"]
        L += [f"| {j['step']} | `{j['chunk']}` | {j['relevance']} | {j['for']} | {j['reason']} |" for j in o["judgments"]]
    g = o["ledger"]
    L += ["", "## 登记簿", "", "| 步 | 指标 | 数值 | 单位 | 来源块 | 核对 |", "|---|---|---|---|---|---|"]
    L += [f"| {r['step']} | {r['name']} | {r['value']} | {r['unit']} | `{r['chunk']}` | {'已核实' if r['ok'] else '不一致'} |" for r in g["records"]]
    L += ["", "| 步 | 计算结果 | 值 | 表达式 |", "|---|---|---|---|"]
    L += [f"| {c['step']} | {n} | {c['value']:g} | `{c['expression']}` |" for n, c in g["calcs"].items()]
    L += ["", "| 选项 | 结论 | 登记于 | 依据 | 理由 | 核对 |", "|---|---|---|---|---|---|"]
    L += [f"| {k} | {c['verdict']} | 第 {c['step']} 步 | {'、'.join(c['evidence'])} | {c['reason']} | {'通过' if c['ok'] else '；'.join(c['problems'])} |"
          for k, c in sorted(g["conclusions"].items())]
    L += ["", "## token 明细", "", "| 步 | 输入 | 输出 | 累计 | 全文块 | 占位符块 |", "|---|---|---|---|---|---|"]
    L += [f"| {r['step']} | {r['usage']['prompt_tokens']} | {r['usage']['completion_tokens']} | {r['usage']['cumulative']} | "
          f"{len(r['context']['full'])} | {len(r['context']['masked'])} |" for r in o["trace"]]
    L += ["", "## 结束时的完整上下文", ""]
    for m in o["final_context"]:
        L += [f"**[{m['role']}]**", ""]
        if m.get("content"):
            L += ["```text", m["content"], "```", ""]
        for c in m.get("tool_calls") or []:
            L += [f"`工具调用 {c['function']['name']} {c['function']['arguments']}`", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
