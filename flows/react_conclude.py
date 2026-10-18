# -*- coding: utf-8 -*-
"""ReAct on one B-list choice question with a ledger the model fills through tools and the code
checks (the design settled on 2026-10-05; react_demo.py is the bare loop, react_compact.py the
version that read the model's notes out of its text).

    tools      search   BM25 through agent.search.narrow; `target` says what it is for (cap 3)
               read     a chunk returned before, by id
               record   a number taken from a chunk -> the code checks that it stands there
               calc     agent.tools.calc; every result goes into the ledger; numbers of five or
                        more digits in the expressions must be recorded numbers, earlier results
                        or numbers of the question
               conclude the verdict on one option with its evidence (chunk ids, calc names)
                        -> the code checks that the evidence exists and is itself checked
    ledger     records, calc results, conclusions -> output/react_conclude/react_conclude_<qid>_ledger.json
    done       after every round the code works out one boolean: every option has a
               conclusion that passed the check
    answer     three steps kept apart (agent.decide): the stem gives the direction (the clause
               in which a judgement word asks, "以下说法错误的是" / "哪些判断不正确", says
               whether the 正确 or the 错误 options are wanted; no such clause -> 正确), a
               conclusion only says whether the option agrees with the documents, the code
               picks the options whose verdict is the one asked for. 无依据 is chosen in
               neither direction; asked for 错误, such an option may be the one to choose, and
               the run records that as a caveat.
               done and consistent (single choice: exactly one option with the verdict asked
               for; multiple choice: at least one; no 无依据) -> the code writes the answer
               from the ledger, no further model call. Otherwise the ledger is handed back to
               the model once. --final model: the ledger is always handed back and the model
               writes the answer
    context    a chunk's text is in the context for the one step after it came back, then a
               one-line placeholder stays: chunk id, section, the checked numbers taken from
               it and the options it is evidence for; the model's text, its calls and the
               replies to record / calc / conclude stay
    limits     3 searches per target, --budget tokens, --steps steps; at a limit the model
               gets one call with conclude alone, then the code answers from the ledger

    python flows/react_conclude.py fin_b_002 [--final model] [--steps 16] [--budget 90000]

Writes output/react_conclude/react_conclude_<qid>.json (every step: what was sent, what the model wrote, every
call, every reply, what the system did), .md (the same as a readable record), _store.json (all
chunks) and _ledger.json.
"""
import argparse
import json
import math
import re
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
from agent.rerank import Reranker                           # noqa: E402
from agent.tools import CALC_SCHEMA, calc, render_calc      # noqa: E402

MAX_BODY = 1200             # characters of one chunk shown to the model
POOL, KEEP = 20, 8          # --rerank: BM25 candidates the cross-encoder ranks, chunks it returns
TARGET_CAP = 3              # searches for one target
CHARS_PER_TOKEN = 1.57      # document text, measured on the chunks (2026-10-01)
VERDICTS = ("正确", "错误", "无依据")


def fn(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


SEARCH = fn("search", "在本题涉及的文档原文中做关键词检索，返回最相关的若干个原文块（表格或段落），每块带编号、所属文档和章节位置。"
            "查询里写上公司简称和年份（如“比亚迪 2024年 营业收入”）时，只在该公司该年度的报告里找。没找到就换说法再查。",
            {"query": {"type": "string", "description": "检索词"},
             "target": {"type": "string", "description": "这次是为哪个选项、哪个数字查的；同一个 target 最多检索 3 次"},
             "top": {"type": "integer", "description": "返回几块，默认 5，最大 8"}}, ["query", "target"])
READ = fn("read", "按块编号重新读取一个之前检索返回过、已经归档的原文块。",
          {"chunk": {"type": "string", "description": "块编号，例如 fr_byd_2025#c0019"}}, ["chunk"])
RECORD = fn("record", "登记从原文块里取到的一个数。系统当场核对这个数是否在该块里，回复“已核实”或“不一致”。"
            "一步里可以登记多个数，也可以和 search、calc、conclude 放在同一步。",
            {"name": {"type": "string", "description": "指标，写明公司和期间，例如“比亚迪 2025年 营业收入”"},
             "value": {"type": "string", "description": "原文里的数值，照抄，例如 803,964,958,000.00"},
             "unit": {"type": "string", "description": "单位，例如 元、千元、%"},
             "chunk": {"type": "string", "description": "来源块编号"}}, ["name", "value", "unit", "chunk"])
CONCLUDE = fn("conclude", "登记一个选项的判断。一个选项判完就登记；可以和下一个选项的 search、calc 放在同一步。"
              "系统会核对依据是否存在：块编号必须是检索返回过的块，计算结果名必须是 calc 返回过的。",
              {"option": {"type": "string", "description": "选项字母"},
               "verdict": {"type": "string", "enum": list(VERDICTS)},
               "evidence": {"type": "array", "items": {"type": "string"}, "description": "依据：块编号，和（或）用到的 calc 结果名"},
               "reason": {"type": "string", "description": "一句话理由"}}, ["option", "verdict", "evidence", "reason"])
CALC = json.loads(json.dumps(CALC_SCHEMA))
CALC["function"]["description"] += ("结果名和变量名不能以数字开头。每次调用相互独立，上一次的结果名不能在下一次直接引用。"
                                    "表达式里的数应当是已经用 record 登记过的数；多个选项的计算尽量放在同一次调用里。")
SYSTEM = """你是金融文档问答助手，用工具逐个核实一道{kind}的选项。
本题涉及的文档：
{docs}
工具：search（检索原文块）、read（按块编号重读已归档的块）、record（登记取到的数，系统当场核对）、calc（精确计算）、conclude（登记一个选项的判断）。
规则：
1. 判断只能依据工具返回的原文和计算结果，不能凭常识或记忆。
2. 检索结果的全文只在紧接着的下一步可见，之后归档成一行说明（块编号、位置、系统核实过的取数）。所以读到要用的数，当步就用 record 登记；一张表里各选项用得到的数一次登记完。已登记并核实的数可以直接使用，不需要重读。
3. 取数时核对指标名称、期间、口径（合并或母公司）和单位；优先用表格里的精确数字。表格里直接给出的增减比例等，可以直接登记使用。
4. 一个选项判完就用 conclude 登记，不要攒到最后。原文或计算结果支持的判“正确”；与之矛盾，或说法超出原文的判“错误”；查不到相关原文的判“无依据”，不要猜。
5. conclude 的结论只表示这个选项的说法（连同题干给出的主体和话题）与原文是否一致。每个选项单独判，不和其他选项比较。题干要找的是正确的还是错误的选项，都不影响结论：说法与原文一致就判“正确”，哪怕题干问的是哪项错误。选哪些选项由系统按题干决定。
6. 一步里可以同时调用多个工具；多个选项要算的内容尽量放在同一次 calc 里。
7. {finish}"""
FINISH_CODE = "所有选项都登记了结论并通过系统核对后，系统会根据登记结果结束作答；你不需要输出最终答案。"
FINISH_MODEL = "所有选项都登记了结论并通过系统核对后，系统会把登记的内容交还给你，由你输出最终答案。"


def squash(s):
    return re.sub(r"[\s|]+", "", str(s))


def number(s):
    """The first number in a string as written and as a float, or (None, None)."""
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(s))
    return (m.group(0), float(m.group(0).replace(",", ""))) if m else (None, None)


def same(a, b):
    """a and b are the same number up to a power of ten (元 / 千元 / 亿元, % / fraction)."""
    if not a or not b:
        return a == b
    k = round(math.log10(abs(a / b)))
    return abs(k) <= 8 and abs(a / b / 10 ** k - 1) < 1e-6


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("qid")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--budget", type=int, default=90000)
    ap.add_argument("--final", choices=("code", "model"), default="code")
    ap.add_argument("--rerank", action="store_true", help=f"search = BM25 top {POOL}, reranked by the cross-encoder, the best {KEEP} returned")
    args = ap.parse_args(argv)
    q = next(x for x in load_b_questions() if x["qid"] == args.qid)
    options = q.get("options") or {}
    if not options:
        raise SystemExit("this script takes choice questions; calculation questions: flows/react_compact.py")
    sc, cards, docs = scope(q), load_cards(), {}

    def unit(uid):
        d = uid.split("#")[0]
        docs.setdefault(d, Doc(d, cards[d]))
        return docs[d].by_id[uid]

    store, home, counts = {}, {}, {}
    records, calcs, conclusions = [], {}, {}        # the ledger
    shown = set()                                   # numbers calc has displayed
    asked = {v for _, v in (number(x) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", q["question"] + " ".join(options.values())))}
    head = lambda u: f"[{u}]《{store[u]['title']}》{store[u]['path']}"
    full = lambda u: head(u) + "\n" + store[u]["text"]
    ref = lambda u: head(u) + " ——（全文见上文，未重复发送）"

    def stub(u):
        taken = [r for r in records if r["chunk"] == u and r["ok"]]
        used = "".join(f"；是选项 {k} 结论的依据" for k, c in sorted(conclusions.items()) if u in c["evidence"])
        if not taken:
            return head(u) + (" ——【已归档】已读过" + used + "。" if used else " ——【已归档｜未取数】已读过，没有从中取数。")
        return (head(u) + " ——【已归档｜已核实】已从中取数，系统已核对与原文一致："
                + "；".join(f"{r['name']} = {r['value']} {r['unit']}" for r in taken) + used + "。无需重读。")

    def render(compress=True):
        out = []
        for i, t in enumerate(turns):
            if "chunks" not in t:
                out.append({k: t[k] for k in ("role", "content", "tool_calls", "tool_call_id") if k in t})
                continue
            parts = [full(u) if not compress or home.get(u) == i else ref(u) if home.get(u) is not None else stub(u) for u in t["chunks"]]
            out.append({"role": "tool", "tool_call_id": t["tool_call_id"], "content": "\n\n".join(parts)})
        return out

    def composition():
        """Characters of the context as sent now, by what they are."""
        c = {"系统提示": 0, "题目": 0, "系统消息": 0, "模型写的文字": 0, "工具调用参数": 0, "原文块全文": 0, "占位符": 0, "工具回复": 0}
        for i, t in enumerate(turns):
            if t["role"] == "system":
                c["系统提示"] += len(t["content"])
            elif t["role"] == "user":
                c["题目" if i == 1 else "系统消息"] += len(t["content"])
            elif t["role"] == "assistant":
                c["模型写的文字"] += len(t.get("content") or "")
                c["工具调用参数"] += sum(len(x["function"]["arguments"]) + len(x["function"]["name"]) for x in t.get("tool_calls") or [])
            elif "chunks" in t:
                for u in t["chunks"]:
                    if home.get(u) == i:
                        c["原文块全文"] += len(full(u))
                    else:
                        c["占位符"] += len(ref(u) if home.get(u) is not None else stub(u))
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
        if rr:                                              # BM25 top POOL, the cross-encoder picks the KEEP it returns
            pool = [u for u, s in bm25_scores(text, where)[:POOL] if s > 0]
            got = [u for u, _ in rr.top(str(a.get("query", "")), pool, KEEP)]
            moved = [f"{u.split('#')[1]}（BM25 第 {pool.index(u) + 1}）" for u in got]
        else:
            got = [u for u, s in bm25_scores(text, where)[:min(int(a.get("top") or args.top), 8)] if s > 0]
        events.append(f"检索：查询“{a.get('query')}”经筛选后在 {len(where)} 份文档（{', '.join(where[:3])}{'…' if len(where) > 3 else ''}）里搜"
                      f"“{text.strip()}”，返回 {len(got)} 块" + (f"；BM25 前 {len(pool)} 经重排后取前 {len(got)}：{'、'.join(moved)}" if rr else ""))
        return (got, None) if got else (None, "没有命中，换一个查询。")

    def do_read(a, step, events):
        u = a.get("chunk")
        if u not in store:
            return None, f"没有编号为 {u} 的块：read 只能重读检索返回过的块。"
        if home.get(u) is not None:
            events.append(f"重读 {u}：全文仍在上下文里，没有重发")
            return None, f"块 {u} 的全文仍在上下文里（见上文），没有重发。"
        events.append(f"重读 {u}：从证据库取出，重新发送全文")
        return [u], None

    def do_record(a, step, events):
        name, ch, unit_ = str(a.get("name", "")).strip(), str(a.get("chunk", "")).strip("[] "), str(a.get("unit", "")).strip()
        text, value = number(a.get("value"))
        ok = bool(text) and ch in store and squash(text) in squash(store[ch]["text"])
        records[:] = [r for r in records if r["name"] != name]
        records.append({"name": name, "value": text or str(a.get("value")), "number": value, "unit": unit_, "chunk": ch, "ok": ok, "step": step})
        events.append(f"登记取数：{name} = {text} {unit_}（{ch}）→ 代码核对：{'数值在该块原文里，已核实' if ok else '该块里找不到这个数，不一致'}")
        if ok:
            return None, f"已核实并登记：{name} = {text} {unit_}。"
        why = f"没有编号为 {ch} 的块" if ch not in store else f"块 {ch} 里没有数值 {a.get('value')}"
        return None, f"不一致，未通过核对：{why}。请核对数值和块编号后重新 record。"

    def do_calc(a, step, events):
        res = calc(a.get("expressions") or {}, a.get("variables") or {})
        known = [r["number"] for r in records if r["ok"]] + list(shown) + list(asked) + [c["value"] for c in calcs.values()]
        lits = [number(x)[1] for e in (a.get("expressions") or {}).values() for x in re.findall(r"\d[\d,]*(?:\.\d+)?", str(e))]
        lits += [float(v) for v in (a.get("variables") or {}).values() if isinstance(v, (int, float))]
        loose = sorted({x for x in lits if len(re.sub(r"\D", "", f"{x:.10g}")) >= 5 and not any(same(x, k) for k in known)})
        for name, v in res["results"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                calcs[name] = {"value": float(v), "expression": str((a.get("expressions") or {}).get(name)), "step": step, "loose": loose}
                shown.add(float(f"{round(v, 6):g}"))
        text = render_calc(res)
        events.append(f"计算：{len(res['results'])} 个结果入账" + ("；出错：" + res["error"] if "error" in res else "")
                      + (f"；表达式里有未登记的数 {loose}" if loose else "；表达式里的数都是已核实的取数、已有的计算结果或题面里的数"))
        if loose:
            text += f"\n提示：表达式里的数 {', '.join(f'{x:g}' for x in loose)} 还没有用 record 登记来源。登记后重新计算，结论才能通过核对。"
        return None, text

    def do_conclude(a, step, events):
        k, verdict = str(a.get("option", "")).strip().upper()[:1], str(a.get("verdict", "")).strip()
        ev = [str(e).strip("[] ") for e in a.get("evidence") or []]
        problems = []
        if k not in options:
            return None, f"没有选项 {a.get('option')}，可选：{'、'.join(options)}。"
        if verdict not in VERDICTS:
            return None, f"verdict 只能是 {' / '.join(VERDICTS)}。"
        for e in ev:
            if e in store:
                continue
            if e in calcs:
                if calcs[e]["loose"]:
                    problems.append(f"计算结果 {e} 用到了未登记的数 {calcs[e]['loose']}")
            else:
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

    def ledger_text():
        L = ["已核实的取数："] + [f"- {r['name']} = {r['value']} {r['unit']}（{r['chunk']}）" for r in records if r["ok"]]
        L += ["计算结果："] + [f"- {n} = {c['value']:g}（{c['expression']}）" for n, c in calcs.items()]
        L += ["各选项结论："] + [f"- {k}：{c['verdict']}｜依据 {'、'.join(c['evidence']) or '无'}｜{c['reason']}" + ("" if c["ok"] else "｜未通过核对：" + "；".join(c["problems"]))
                            for k, c in sorted(conclusions.items())]
        return "\n".join(L)

    asked, clause, word = direction(q["question"])  # the verdict the stem asks for

    def picked():
        """(options chosen, options in neither direction) from the ledger as it stands."""
        return choose(asked, {k: conclusions[k]["verdict"] if k in conclusions else None for k in options})

    def consistent():
        chosen, unfounded = picked()
        if unfounded:
            return f"选项 {'、'.join(unfounded)} 判为无依据（无依据的选项不会被选）"
        if q["type"] == "单选题" and len(chosen) != 1:
            return f"按题干应选判为“{asked}”的选项，单选题应当恰好一个，现在是 {len(chosen)} 个"
        if not chosen:
            return f"按题干应选判为“{asked}”的选项，现在一个也没有"
        return ""

    # ------------------------------------------------------------------ loop
    titles = "\n".join(f"- 《{cards[d]['title']}》" for d in sc["docs"])
    if not sc["named"]:
        titles = ("（题面没有点名文档，search 在下面这些文档里找）\n" + titles if len(sc["docs"]) <= 20
                  else f"（题面没有点名文档，search 在本领域的 {len(sc['docs'])} 份文档里找）")
    system = SYSTEM.format(kind=q["type"], docs=titles, finish=FINISH_CODE if args.final == "code" else FINISH_MODEL)
    listed = "".join(f"\n{k}. {v.strip()}" for k, v in options.items())
    turns = [{"role": "system", "content": system}, {"role": "user", "content": q["question"].strip() + listed}]
    llm = Qwen(log_path=ROOT / "logs" / f"react_conclude_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    trace, usage, t0 = [], [0, 0], time.time()
    answer, how, nudges, reviewed, wrap_up = None, None, 0, False, False
    rr = Reranker() if args.rerank else None
    asks = f"问句“{clause}”" + (f"里的“{word}”" if word else "没有匹配到判断词，按默认") + f" → 选判为“{asked}”的选项"
    print(f"{q['qid']} {q['question'].strip()}{listed}\n文档：{sc['docs'] if sc['named'] else 'whole domain'}\n方向：{asks}\n")
    step = 0
    while answer is None:
        step += 1
        over = step > args.steps or sum(usage) >= args.budget
        if over and not wrap_up:
            wrap_up = True
            missing = [k for k in options if k not in conclusions or not conclusions[k]["ok"]]
            turns.append({"role": "user", "content": f"步数或 token 预算已用完。请现在就用 conclude 给出这些选项的判断：{'、'.join(missing)}；没有依据的写“无依据”。"})
        sent = render()
        comp = composition()
        state = {"visible": [u for u, h in home.items() if h is not None], "composition": comp, "chars_sent": size(sent),
                 "chars_uncompressed": size(render(False)), "store": len(store)}
        tools = [CONCLUDE] if wrap_up else [SEARCH, READ, RECORD, CALC, CONCLUDE]
        resp = llm.client.chat.completions.create(model=llm.model, messages=sent, temperature=0, max_tokens=1500,
                                                  extra_body={"enable_thinking": False}, tools=tools)
        msg = resp.choices[0].message
        usage[0] += resp.usage.prompt_tokens
        usage[1] += resp.usage.completion_tokens
        calls = msg.tool_calls or []
        state["tokens_uncompressed_est"] = round(resp.usage.prompt_tokens + (state["chars_uncompressed"] - state["chars_sent"]) / CHARS_PER_TOKEN)
        events = []
        row = {"step": step, "content": msg.content or "", "calls": [], "context": state, "events": events, "sent": sent if step == 1 else None,
               "usage": {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens, "cumulative": sum(usage)}}
        trace.append(row)
        print(f"--- 第 {step} 步（输入 {resp.usage.prompt_tokens}，不压缩约 {state['tokens_uncompressed_est']}；输出 {resp.usage.completion_tokens}；累计 {sum(usage)}）"
              f" 带全文的块 {len(state['visible'])}，证据库 {len(store)}")
        if msg.content:
            print("模型：" + msg.content.strip())
        if calls:
            turns.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
        else:
            turns.append({"role": "assistant", "content": msg.content or ""})
        first = len(turns)
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
                for u in got:
                    if u not in store:
                        x = unit(u)
                        store[u] = {"title": cards[u.split("#")[0]]["title"], "path": x["path"] or "", "text": x["body"][:MAX_BODY],
                                    "type": x["type"], "first_step": step, "returned": []}
                    store[u]["returned"].append({"step": step, "tool": name, "query": a.get("query"), "target": a.get("target")})
            else:
                turn["content"] = reply
            turns.append(turn)
            row["calls"].append({"name": name, "arguments": a, "chunks": got, "reply": reply})
            print(f"调用：{name} {json.dumps(a, ensure_ascii=False)}")
            print("返回：" + (", ".join(u.split('#')[0][-12:] + '#' + u.split('#')[1] for u in got) if got else str(reply).replace("\n", "；")))
        # ---- context: only what this step returned stays in full
        if calls:
            latest = [u for t in turns[first:] for u in t.get("chunks", [])]
            new, back, gone = [], [], []
            for i in range(first, len(turns)):
                for u in turns[i].get("chunks", []):
                    if home.get(u) is None:
                        (new if store[u]["first_step"] == step and u not in new else back).append(u)
                        home[u] = i
            for u, h in home.items():
                if h is not None and u not in latest:
                    home[u] = None
                    gone.append(u)
            row["archive"] = {"new": new, "restored": [u for u in back if u not in new], "archived": gone}
            if new or gone or back:
                events.append(f"上下文：新入库 {len(new)} 块（下一步可见全文）；归档 {len(gone)} 块（换成占位符）"
                              + (f"；从证据库重发 {len([u for u in back if u not in new])} 块" if [u for u in back if u not in new] else ""))
        # ---- the boolean
        done = all(k in conclusions and conclusions[k]["ok"] for k in options)
        row["ledger"] = {"records_ok": sum(r["ok"] for r in records), "records_bad": sum(not r["ok"] for r in records), "calcs": len(calcs),
                         "conclusions": {k: (c["verdict"], c["ok"]) for k, c in sorted(conclusions.items())}, "done": done}
        events.append(f"本轮结束判断：已核实取数 {row['ledger']['records_ok']} 个，计算结果 {len(calcs)} 个，"
                      f"结论 {''.join(k + ('✓' if c['ok'] else '✗') for k, c in sorted(conclusions.items())) or '无'} → 全部选项已核实 = {done}")
        for e in events:
            print("系统：" + e)
        # ---- what next
        if args.final == "model" and reviewed and not calls:          # the model's own answer after the ledger was handed back
            final = parse_json(msg.content or "")
            answer, how = str(final.get("answer") or "".join(picked()[0])), "模型在拿到登记内容后输出"
        elif done or wrap_up:
            why = consistent() if done else "到达上限"
            if (done and not why and args.final == "code") or reviewed or wrap_up:
                answer = "".join(picked()[0])
                how = f"代码按题干方向从登记的结论里挑出判为“{asked}”的选项" + ("（到达上限）" if wrap_up else "")
                events.append(f"作答：{how}，答案 {answer or '（空）'}")
                print("系统：" + events[-1])
            else:
                reviewed = True
                text = ("所有选项都已有通过核对的结论。" + (f"但{why}，请确认或修改（用 conclude），确认无误就直接回复“确认”。" if why else "请根据下面登记的内容输出最终答案，只输出一个 JSON：{\"answer\": \"题干要求选出的选项字母\"}")
                        + "\n" + ledger_text())
                turns.append({"role": "user", "content": text})
                events.append("系统把登记内容交还给模型：" + (why or "由模型作答"))
                print("系统：" + events[-1])
        elif not calls:
            nudges += 1
            missing = [k for k in options if k not in conclusions or not conclusions[k]["ok"]]
            if nudges > 2:
                wrap_up = True
            turns.append({"role": "user", "content": f"还没有通过核对的结论的选项：{'、'.join(missing)}。请继续核实，并用 conclude 登记；不要直接作答。"})
            events.append(f"模型没有调用工具，系统提醒还缺结论的选项：{missing}")
            print("系统：" + events[-1])
    # ------------------------------------------------------------------ record
    saved = sum(r["context"]["tokens_uncompressed_est"] - r["usage"]["prompt_tokens"] for r in trace)
    unfounded = picked()[1]                         # no conclusion or 无依据: chosen in neither direction
    note = caveat(asked, unfounded)
    asking = {"asked": asked, "clause": clause, "word": word}
    ledger = {"records": records, "calcs": calcs, "conclusions": conclusions, "done": all(k in conclusions and conclusions[k]["ok"] for k in options),
              "direction": asking, "unfounded": unfounded, "caveat": note, "answer": answer, "answered_by": how}
    out = {"qid": q["qid"], "type": q["type"], "question": q["question"].strip(), "options": options, "docs": sc["docs"] if sc["named"] else None,
           "model": llm.model, "system": system, "final_mode": args.final, "direction": asking, "direction_text": asks,
           "unfounded": unfounded, "caveat": note, "answer": answer, "answered_by": how, "steps": len(trace),
           "calls": {n: sum(c["name"] == n for r in trace for c in r["calls"]) for n in TOOLS}, "targets": counts, "ledger": ledger,
           "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)},
           "uncompressed_total_est": sum(usage) + saved, "seconds": round(time.time() - t0, 1), "trace": trace, "final_context": render()}
    if rr:
        out["rerank"] = {"pool": POOL, "keep": KEEP, "pairs": rr.pairs}
        rr.close()
    base = ROOT / "output" / "react_conclude" / f"react_conclude_{q['qid']}{'_rerank' if rr else ''}"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(str(base) + "_store.json").write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(str(base) + "_ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8")
    base.with_suffix(".md").write_text(report(out, store), encoding="utf-8")
    print(f"\n答案：{answer}（{how}）" + (f"\n注意：{note}" if note else "")
          + f"\n{out['steps']} 步，调用 {out['calls']}，{out['seconds']} 秒；tokens 输入 {usage[0]} + 输出 {usage[1]} = {sum(usage)}"
          f"（同一过程不压缩约 {out['uncompressed_total_est']}） -> {base.relative_to(ROOT)}.json / .md / _store.json / _ledger.json")


def report(o, store):
    """The run as a Markdown record: what the model thought, what the system did, what it cost."""
    u = o["usage"]
    L = [f"# {o['qid']}：带登记工具的 ReAct 作答记录", "",
         f"模型 {o['model']}，温度 0。共 {o['steps']} 步，{o['seconds']} 秒。工具调用：" + "，".join(f"{k} {v} 次" for k, v in o["calls"].items()) + "。",
         f"方向：{o['direction_text']}。",
         f"答案：**{o['answer']}**（{o['answered_by']}）。" + (f"注意：{o['caveat']}。" if o["caveat"] else ""),
         f"tokens：输入 {u['prompt_tokens']} + 输出 {u['completion_tokens']} = {u['total_tokens']}；同一过程不归档的估算 {o['uncompressed_total_est']}。", "",
         "## 给模型的系统提示", "", "```text", o["system"], "```", "", "## 题目", "", "```text",
         o["question"] + "".join(f"\n{k}. {v.strip()}" for k, v in o["options"].items()), "```", "", "## 逐步过程", "",
         "每一步是一次模型调用。“模型的思考”是模型这一步输出的全部文字；“系统”是代码做的事。", ""]
    for r in o["trace"]:
        c = r["context"]
        comp = "，".join(f"{k} {v}" for k, v in c["composition"].items() if v)
        L += [f"### 第 {r['step']} 步", "",
              f"- 发给模型：输入 {r['usage']['prompt_tokens']} tokens（不归档约 {c['tokens_uncompressed_est']}）。上下文构成（字符数）：{comp}。",
              f"- 此时带全文的原文块 {len(c['visible'])} 个：{', '.join(c['visible']) or '无'}；证据库 {c['store']} 块。",
              f"- 模型输出 {r['usage']['completion_tokens']} tokens；累计 {r['usage']['cumulative']}。", ""]
        if r["content"].strip():
            L += ["**模型的思考：**", "", "```text", r["content"].strip(), "```", ""]
        for i, call in enumerate(r["calls"], 1):
            L += [f"**模型调用 {i}：{call['name']}**　`{json.dumps(call['arguments'], ensure_ascii=False)}`", ""]
            if call["chunks"]:
                for x in call["chunks"]:
                    L += [f"- 返回 `{x}`　《{store[x]['title']}》　{store[x]['path']}", "", "  ```text"] + ["  " + line for line in store[x]["text"].split("\n")] + ["  ```", ""]
            else:
                L += ["返回：", "", "```text", str(call["reply"]), "```", ""]
        L += ["**系统：**", ""] + [f"- {e}" for e in r["events"]] + [""]
    g = o["ledger"]
    L += ["## 登记簿（最终状态）", "", "### 取数", "", "| 步 | 指标 | 数值 | 单位 | 来源块 | 核对 |", "|---|---|---|---|---|---|"]
    L += [f"| {r['step']} | {r['name']} | {r['value']} | {r['unit']} | `{r['chunk']}` | {'已核实' if r['ok'] else '不一致'} |" for r in g["records"]]
    L += ["", "### 计算结果", "", "| 步 | 名称 | 值 | 表达式 | 表达式里未登记的数 |", "|---|---|---|---|---|"]
    L += [f"| {c['step']} | {n} | {c['value']:g} | `{c['expression']}` | {c['loose'] or '无'} |" for n, c in g["calcs"].items()]
    L += ["", "### 结论", "", "| 选项 | 结论 | 登记于 | 依据 | 理由 | 核对 |", "|---|---|---|---|---|---|"]
    L += [f"| {k} | {c['verdict']} | 第 {c['step']} 步 | {'、'.join(c['evidence'])} | {c['reason']} | {'通过' if c['ok'] else '；'.join(c['problems'])} |"
          for k, c in sorted(g["conclusions"].items())]
    L += ["", "## 证据库", "", "| 块 | 类型 | 章节位置 | 入库步 | 被返回的记录 |", "|---|---|---|---|---|"]
    for x, s in store.items():
        rec = "；".join(f"第{y['step']}步 {y['tool']}（{y.get('target') or ''}）" for y in s["returned"])
        L.append(f"| `{x}` | {s['type']} | {s['path'][-40:]} | {s['first_step']} | {rec} |")
    L += ["", "## token 明细", "", "| 步 | 输入 | 不归档估算 | 输出 | 累计 |", "|---|---|---|---|---|"]
    L += [f"| {r['step']} | {r['usage']['prompt_tokens']} | {r['context']['tokens_uncompressed_est']} | {r['usage']['completion_tokens']} | {r['usage']['cumulative']} |" for r in o["trace"]]
    tot = {}
    for r in o["trace"]:
        for k, v in r["context"]["composition"].items():
            tot[k] = tot.get(k, 0) + v
    allc = sum(tot.values()) or 1
    L += ["", "各步发送内容按类别累加（字符数，不含工具说明）：", "", "| 类别 | 字符数 | 占比 |", "|---|---|---|"]
    L += [f"| {k} | {v} | {100 * v / allc:.0f}% |" for k, v in sorted(tot.items(), key=lambda x: -x[1]) if v]
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
