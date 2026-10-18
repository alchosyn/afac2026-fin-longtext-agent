# -*- coding: utf-8 -*-
"""ReAct with context compression on one B-list question, a calculation question or a choice
question whose options are judged one by one (the design talked through on 2026-10-05;
flows/react_demo.py is the loop without it).

    loop       Qwen decides every step; tools: search (BM25 through agent.search.narrow), read
               (a chunk seen before, by id), calc (agent.tools)
    notes      after reading a result the model writes what it took from it: value as printed,
               unit, chunk id, the row or sentence copied; or what it looked for and did not find
    compress   a result is in the context in full for the one step after it came back. Then the
               chunks the model has cited by id in what it wrote stay, the others leave the
               context and a one-line placeholder (chunk id, document, section) stays; what
               the model wrote and its tool calls stay as they are. --strict: cited chunks
               leave too (the first version, 2026-10-05: on fin_b_014 the model then re-read
               the same two chunks in turn, 7 reads, 85,322 tokens against 38,849 without
               any compression)
               --archive: --strict, but every 【已确认】 line the model writes is checked against
               its chunk at once, the placeholder of that chunk then reads 已归档｜已核实 with
               the values taken from it, and when the calculation is done the tool result
               also carries all checked values with their source rows
    reuse      a chunk that comes back, or is asked for with read, while its text is still in
               the context is not sent again; one that had left is sent again from the store
    store      every chunk ever returned, with the step, query and target that returned it
               -> output/react_compact/react_compact_<qid>_store.json
    limits     TARGET_CAP searches per target, BUDGET tokens per question, --steps steps
    check      every number of the final answer must stand in the chunk it cites, and the chunk
               must be in the store; one round of feedback when that fails

    python flows/react_compact.py fin_b_014 [--steps 14] [--top 5]

Writes output/react_compact/react_compact_<qid>.json (every step with the state of the context, the final
context as sent, the checks) and output/react_compact/react_compact_<qid>.md (the same as a readable record).
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

from agent.docview import Doc, load_cards                   # noqa: E402
from agent.llm import Qwen, parse_json                      # noqa: E402
from agent.retrieval import bm25_scores                     # noqa: E402
from agent.scope import load_b_questions, scope             # noqa: E402
from agent.search import narrow                             # noqa: E402
from agent.tools import CALC_SCHEMA, run_tool               # noqa: E402

MAX_BODY = 1200             # characters of one chunk shown to the model
TARGET_CAP = 3              # searches for one target
BUDGET = 70000              # tokens of one question; beyond it the model has to answer
CHARS_PER_TOKEN = 1.57      # document text, measured on the chunks (2026-10-01)
SEARCH_SCHEMA = {"type": "function", "function": {
    "name": "search",
    "description": ("在本题涉及的文档原文中做关键词检索，返回最相关的若干个原文块（表格或段落），每块带编号、所属文档和章节位置。"
                    "查询里写上公司简称和年份（如“比亚迪 2024年 营业收入”）时，只在该公司该年度的报告里找。"
                    "一次只查一件事效果最好；没找到就换说法（指标全称、所在表格的名称、原文可能的用词）再查。"),
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "检索词，例如“比亚迪 2025年 经营活动产生的现金流量净额”"},
        "target": {"type": "string", "description": "这次是为哪个数字查的，例如“比亚迪2024年营业收入”；同一个 target 最多检索 3 次"},
        "top": {"type": "integer", "description": "返回几块，默认 5，最大 8"}},
        "required": ["query", "target"]}}}
READ_SCHEMA = {"type": "function", "function": {
    "name": "read",
    "description": "按块编号重新读取一个之前检索返回过、全文已移出上下文的原文块。",
    "parameters": {"type": "object", "properties": {"chunk": {"type": "string", "description": "块编号，例如 fr_byd_2025#c0019"}},
                   "required": ["chunk"]}}}
SYSTEM = """你是金融文档问答助手，用工具一步步回答一道计算题。
本题涉及的文档：
{docs}
工具：search（检索原文块）、read（按块编号重读一个块）、calc（精确计算）。
规则：
1. 题目需要的每个数字都必须来自工具返回的原文，不能凭记忆或估计。
2. 每次调用 search 都要在 target 里写明这次是为哪个数字查的；同一个 target 最多检索 3 次，仍找不到就记为未找到。
3. {keep}所以每次读完结果，先做笔记再做别的，一个数字一行：
   【已确认】指标（公司、期间）｜数值原文｜单位｜块编号｜该数字所在的原句或表格行（照抄）
   【未找到】要找什么｜查过哪份文档｜用过的查询
   已经写进【已确认】的数字不需要再查、再读；只有笔记里缺的内容才用 read 回看。
4. 取数时核对指标名称、期间、口径（合并或母公司）和单位；优先用表格里的精确金额，不用正文里四舍五入的数。
5. 所有算术用 calc 一次完成（需要的四舍五入用 round 写在同一次里），中间结果不四舍五入；calc 只用于算术，不要用它拼字符串或做确认。
6. 数字齐全并算完后，不再调用工具，只输出一个 JSON：
{{"answer": "按题目要求的格式", "inputs": [{{"name": "指标（公司、期间）", "value": "原文数值", "unit": "单位", "chunk": "来源块编号", "quote": "照抄的原句或表格行"}}], "steps": "计算过程"}}"""


SYSTEM_CHOICE = """你是金融文档问答助手，用工具一步步回答一道{kind}：逐个核实选项，选出与文档一致的。
本题涉及的文档：
{docs}
工具：search（检索原文块）、read（按块编号重读一个块）、calc（精确计算）。
规则：
1. 每个选项的判断都必须以工具返回的原文为依据，不能凭常识或记忆；选项里的数字要用原文数字核对，需要计算的用 calc。
2. 每次调用 search 都要在 target 里写明这次是为哪个选项、哪个数字查的；同一个 target 最多检索 3 次，仍找不到就记为未找到。
3. {keep}所以每次读完结果，先做笔记再做别的，一条一行：
   【已确认】指标（公司、期间）｜数值原文｜单位｜块编号
   【未找到】要找什么｜查过哪份文档｜用过的查询
   【结论】选项字母｜正确 / 错误 / 无依据｜依据的块编号｜一句话理由
   一张表里凡是各选项用得到的数字，要在这一步全部记下。已经写进【已确认】的数字不需要再查、再读；只有笔记里缺的内容才用 read 回看。
4. 取数时核对指标名称、期间、口径（合并或母公司）和单位；优先用表格里的精确数字。
5. 原文或计算结果支持的选项判“正确”；与原文或计算结果矛盾，或说法超出原文的判“错误”；找不到任何相关原文的写“无依据”，不要猜。
6. 算术用 calc（需要的四舍五入用 round 写在同一次里）；calc 只用于算术，不要用它拼字符串或做确认。
7. 所有选项都有【结论】后，不再调用工具，只输出一个 JSON：
{{"answer": "正确选项的字母，按顺序连写", "options": {{"A": {{"verdict": "正确|错误|无依据", "evidence": ["依据的块编号"], "reason": "一句话依据"}}}}}}"""


CALC = json.loads(json.dumps(CALC_SCHEMA))      # two things the model tripped over on fin_b_002 (2026-10-05), said up front
CALC["function"]["description"] += "结果名和变量名不能以数字开头（写“净利率2024”，不要写“2024净利率”）。每次调用相互独立，上一次调用的结果名不能在下一次直接引用。"
KEEP_CITED = "检索结果里，你在笔记中写了块编号的块会一直保留全文；没有被你引用的块在下一步之后移出上下文，只留块编号和位置。"
KEEP_STRICT = "检索结果的全文只在紧接着的下一步可见，之后会被移出上下文，只留块编号和位置。"
KEEP_ARCHIVE = ("检索结果的全文只在紧接着的下一步可见，之后会被归档：原处只留一行说明，写明块编号、位置，以及系统核对过的取数记录。"
                "标记为【已归档｜已核实】的取数已由系统对照原文核对，可直接作为依据。作答前系统会把全部已核实的取数（连同原文行）和各项结论重新给出。")


def squash(s):
    return re.sub(r"[\s|]+", "", str(s))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("qid")
    ap.add_argument("--steps", type=int, default=14)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--budget", type=int, default=BUDGET, help="tokens after which the model has to answer")
    ap.add_argument("--strict", action="store_true", help="cited chunks leave the context too")
    ap.add_argument("--archive", action="store_true",
                    help="--strict, with placeholders that carry the values taken from the chunk as checked by the code, "
                         "and all checked values with their source rows given again once the calculation is done")
    args = ap.parse_args(argv)
    args.strict = args.strict or args.archive
    q = next(x for x in load_b_questions() if x["qid"] == args.qid)
    sc, cards, docs = scope(q), load_cards(), {}

    def unit(uid):
        d = uid.split("#")[0]
        docs.setdefault(d, Doc(d, cards[d]))
        return docs[d].by_id[uid]

    store = {}          # chunk id -> title, path, text, first_step, returned [{step, tool, query, target}]
    home = {}           # chunk id -> index of the tool turn that holds its full text; None once it left the context
    counts = {}         # target -> searches run
    records = []        # values the model noted: name, value, unit, chunk, ok (the value stands in the chunk), step
    options = q.get("options") or {}
    conclusions = {}    # option letter -> verdict, chunks, reason, step (choice questions)
    head = lambda u: f"[{u}]《{store[u]['title']}》{store[u]['path']}"
    full = lambda u: head(u) + "\n" + store[u]["text"]
    ref = lambda u: head(u) + " ——（全文见上文，未重复发送）"

    def stub(u):
        """The line that stands for a chunk whose text left the context."""
        if not args.archive:
            return head(u) + " ——（全文已移出上下文，需要时用 read 重读）"
        taken = [r for r in records if r["chunk"] == u and r["ok"]]
        used = "".join(f"；是选项 {k} 结论的依据" for k, c in sorted(conclusions.items()) if u in c["chunks"])
        if not taken:
            return head(u) + (" ——【已归档】已读过" + used + "。" if used else " ——【已归档｜未取数】已读过，没有从中取数。")
        return (head(u) + " ——【已归档｜已核实】已从中取数，系统已核对与原文一致："
                + "；".join(f"{r['name']} = {r['value']} {r['unit']}" for r in taken) + used + "。无需重读。")

    def note(text, step):
        """The 【已确认】 lines of what the model wrote -> records, each checked against its chunk."""
        new = []
        for line in (text or "").splitlines():
            if "【结论】" in line:                             # option ｜ verdict ｜ chunk ids ｜ reason; the latest one counts
                f = [x.strip() for x in line.split("【结论】", 1)[1].split("｜")]
                if len(f) >= 2 and f[0][:1] in options:
                    conclusions[f[0][:1]] = {"verdict": f[1], "chunks": [u for u in store if len(f) > 2 and u in f[2]],
                                             "reason": f[3] if len(f) > 3 else "", "step": step}
                continue
            if "【已确认】" not in line:
                continue
            f = [x.strip() for x in line.split("【已确认】", 1)[1].split("｜")]
            if len(f) < 4:
                continue
            name, value, unit_, ch = f[0], f[1], f[2], f[3].strip("[]【】 ")
            if any(r["name"] == name and r["chunk"] == ch and r["value"] == value for r in records):
                continue
            m = re.search(r"-?\d[\d,]*(?:\.\d+)?", value)            # the number alone: the model may write the unit after it
            value = m.group(0) if m else value
            ok = ch in store and squash(value) in squash(store[ch]["text"])
            records.append({"name": name, "value": value, "unit": unit_, "chunk": ch, "ok": ok, "step": step})
            new.append(records[-1])
        return new

    def rows_of(u, value):
        """The header row of the chunk's table and the lines holding the value, long lines cut around it."""
        lines = store[u]["text"].split("\n")
        out = [l for l in lines[:2] if l.startswith("|") and "---" not in l][:1]
        for l in lines:
            if squash(value.replace("%", "")) in squash(l) and l not in out:
                i = l.find(value)
                out.append(l if len(l) <= 300 else "…" + l[max(0, i - 80): i + len(value) + 80] + "…")
        return out

    def recap():
        """All checked values with their source rows and the conclusions noted so far, given once
        before the final answer."""
        L = ["", "【作答前复核｜系统提供】以下取数均已由系统对照原文核对，数值一致，可直接作为依据："]
        for i, r in enumerate([r for r in records if r["ok"]], 1):
            L.append(f"{i}. {r['name']} = {r['value']} {r['unit']} ｜ 来源 {r['chunk']}《{store[r['chunk']]['title']}》{store[r['chunk']]['path'].split(' > ')[-1]}")
            L += ["   原文：" + l for l in rows_of(r["chunk"], r["value"])]
        if conclusions:
            L.append("你记录的各选项结论：")
            L += [f"{k}：{c['verdict']}｜依据 {'、'.join(c['chunks']) or '（未写块编号）'}｜{c['reason']}" for k, c in sorted(conclusions.items())]
            missing = [k for k in options if k not in conclusions]
            L.append(("还没有【结论】的选项：" + "、".join(missing) + "。" if missing else "四个选项都已有结论。")
                     + "计算结果见上文各次 calc 的返回。现在可以直接输出最终 JSON，不需要重读原文或重新计算。")
            return "\n".join(L)
        return "\n".join(L + ["计算已完成，结果见上。现在可以直接输出最终 JSON，不需要重读原文或重新计算。"])

    def render(compress=True):
        """The message list as sent. compress=False: what the loop without compression would send."""
        out = []
        for i, t in enumerate(turns):
            if "chunks" not in t:
                out.append({k: t[k] for k in ("role", "content", "tool_calls", "tool_call_id") if k in t})
                continue
            parts = [full(u) if not compress or home.get(u) == i else ref(u) if home.get(u) is not None else stub(u) for u in t["chunks"]]
            out.append({"role": "tool", "tool_call_id": t["tool_call_id"], "content": "\n\n".join(parts)})
        return out

    size = lambda msgs: sum(len(m.get("content") or "") + sum(len(c["function"]["arguments"]) + len(c["function"]["name"])
                                                               for c in m.get("tool_calls") or []) for m in msgs)
    titles = "\n".join(f"- 《{cards[d]['title']}》" for d in sc["docs"])
    if not sc["named"]:
        titles = f"（题面没有点名文档，search 在本领域的 {len(sc['docs'])} 份文档里找）"
    keep = KEEP_ARCHIVE if args.archive else KEEP_STRICT if args.strict else KEEP_CITED
    listed = "".join(f"\n{k}. {v.strip()}" for k, v in options.items())
    system = SYSTEM_CHOICE.format(kind=q["type"], docs=titles, keep=keep) if options else SYSTEM.format(docs=titles, keep=keep)
    turns = [{"role": "system", "content": system}, {"role": "user", "content": q["question"].strip() + listed}]
    llm = Qwen(log_path=ROOT / "logs" / f"react_compact_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    trace, usage, t0, final, fed_back, sent, recapped = [], [0, 0], time.time(), None, False, None, False
    print(f"{q['qid']} {q['question'].strip()}{listed}\n文档：{sc['docs'] if sc['named'] else 'whole domain'}\n")
    step = 0
    while True:
        step += 1
        forced = step > args.steps or sum(usage) >= args.budget
        if forced:
            turns.append({"role": "user", "content": "步数或 token 预算已用完。不再调用工具，根据笔记直接输出最终 JSON。"})
        sent = render()
        state = {"visible": [u for u, h in home.items() if h is not None], "chars_sent": size(sent), "chars_uncompressed": size(render(False)),
                 "store": len(store)}
        resp = llm.client.chat.completions.create(
            model=llm.model, messages=sent, temperature=0, max_tokens=1500, extra_body={"enable_thinking": False},
            **({} if forced else {"tools": [SEARCH_SCHEMA, READ_SCHEMA, CALC]}))
        msg = resp.choices[0].message
        usage[0] += resp.usage.prompt_tokens
        usage[1] += resp.usage.completion_tokens
        calls = msg.tool_calls or []
        state["tokens_uncompressed_est"] = round(resp.usage.prompt_tokens + (state["chars_uncompressed"] - state["chars_sent"]) / CHARS_PER_TOKEN)
        row = {"step": step, "content": msg.content or "", "calls": [], "context": state, "events": {},
               "usage": {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}}
        row["records"] = note(msg.content, step)
        trace.append(row)
        print(f"--- 第 {step} 步（输入 {resp.usage.prompt_tokens}，不压缩约 {state['tokens_uncompressed_est']}；输出 {resp.usage.completion_tokens}）"
              f" 上下文里有全文的块 {len(state['visible'])} 个，证据库 {len(store)} 个")
        if msg.content:
            print("模型：" + msg.content.strip())
        if not calls:
            if args.archive and not recapped and not forced and (conclusions or any(r["ok"] for r in records)):
                recapped = True                              # the model wants to answer: first give it its takings back
                row["recap"] = recap()
                row["events"] = {"masked": [u for u, h in home.items() if h is not None]}
                for u in row["events"]["masked"]:
                    home[u] = None
                turns.append({"role": "assistant", "content": msg.content or ""})
                turns.append({"role": "user", "content": row["recap"].strip()})
                print("作答前补回：" + row["recap"].replace("\n", "；")[:600])
                continue
            final = parse_json(msg.content or "")
            problems = check(final, store)
            row["check"] = problems
            if (problems["hard"] or "answer" not in final) and not fed_back and not forced:
                fed_back = True                              # one round of feedback, tools still on
                turns.append({"role": "assistant", "content": msg.content or ""})
                turns.append({"role": "user", "content": "最终 JSON 没有通过核对：" + "；".join(problems["hard"] or ["不是合法的 JSON"])
                              + "。请用 read 或 search 重新核对后再输出完整的 JSON。"})
                print("核对未通过，反馈给模型：", problems["hard"])
                continue
            break
        turns.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
        first = len(turns)
        for c in calls:
            try:
                a = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                a = {}
            turn, got, name = {"role": "tool", "tool_call_id": c.id}, None, c.function.name
            if name == "search":
                target = re.sub(r"\s+", "", str(a.get("target") or "未说明"))
                if counts.get(target, 0) >= TARGET_CAP:
                    turn["content"] = f"目标“{a.get('target')}”已检索 {TARGET_CAP} 次，本次不再检索：记为【未找到】，继续下一个目标或直接作答。"
                else:
                    counts[target] = counts.get(target, 0) + 1
                    where, text = narrow(q, str(a.get("query", "")), sc)
                    got = [u for u, s in bm25_scores(text, where)[:min(int(a.get("top") or args.top), 8)] if s > 0]
                    if not got:
                        turn["content"], got = "没有命中，换一个查询。", None
            elif name == "read":
                got = [a.get("chunk")] if a.get("chunk") in store else None
                if not got:
                    turn["content"] = f"没有编号为 {a.get('chunk')} 的块：read 只能重读检索返回过的块。"
                elif home.get(got[0]) is not None:               # still in the context: nothing is sent again
                    turn["content"], got = f"块 {got[0]} 的全文仍在上下文里（见上文），没有重发。", None
                    row.setdefault("read_in_context", []).append(a.get("chunk"))
            else:
                turn["content"] = run_tool(name, c.function.arguments, None)
                if args.archive and not options and not recapped and "错误" not in turn["content"] and any(r["ok"] for r in records):
                    recapped = True
                    row["recap"] = recap()
                    turn["content"] += "\n" + row["recap"]
            if got:
                turn["chunks"] = got
                for u in got:
                    if u not in store:
                        x = unit(u)
                        store[u] = {"title": cards[u.split("#")[0]]["title"], "path": x["path"] or "", "text": x["body"][:MAX_BODY],
                                    "first_step": step, "returned": []}
                    store[u]["returned"].append({"step": step, "tool": name, "query": a.get("query"), "target": a.get("target")})
            turns.append(turn)
            row["calls"].append({"name": name, "arguments": a, "chunks": got, "content": turn.get("content")})
        # which chunks stay in the context: those this step returned; the rest leaves
        ev = {"new": [], "restored": [], "reused": [], "masked": []}
        latest = [u for t in turns[first:] for u in t.get("chunks", [])]
        for i in range(first, len(turns)):
            for u in turns[i].get("chunks", []):
                if home.get(u) is None:
                    (ev["new"] if store[u]["first_step"] == step and u not in ev["new"] + ev["restored"] else ev["restored"]).append(u)
                    home[u] = i
                elif home[u] < first and u not in ev["reused"]:
                    ev["reused"].append(u)
        written = "" if args.strict else "".join(t.get("content") or "" for t in turns if t["role"] == "assistant")
        for u, h in home.items():
            if h is not None and u not in latest and u not in written:      # not returned just now, not cited by the model
                home[u] = None
                ev["masked"].append(u)
        ev["kept_cited"] = [u for u, h in home.items() if h is not None and u not in latest]
        row["events"] = ev
        for c in row["calls"]:
            print(f"调用：{c['name']} {json.dumps(c['arguments'], ensure_ascii=False)}")
            print("返回：" + (", ".join(u.split('#')[0][-12:] + '#' + u.split('#')[1] for u in c["chunks"]) if c["chunks"] else c["content"].replace("\n", "；")))
        print(f"证据：新入库 {len(ev['new'])}，仍在上下文未重发 {len(ev['reused'])}，从证据库重发 {len(ev['restored'])}，移出上下文 {len(ev['masked'])}")
    # ------------------------------------------------------------------ record
    saved = sum(r["context"]["tokens_uncompressed_est"] - r["usage"]["prompt_tokens"] for r in trace)
    out = {"qid": q["qid"], "question": q["question"].strip(), "docs": sc["docs"] if sc["named"] else None, "model": llm.model, "system": system,
           "final": final, "final_raw": trace[-1]["content"], "check": trace[-1].get("check"), "steps": len(trace),
           "searches": sum(c["name"] == "search" and c["chunks"] is not None for r in trace for c in r["calls"]),
           "reads": sum(c["name"] == "read" for r in trace for c in r["calls"]), "calcs": sum(c["name"] == "calc" for r in trace for c in r["calls"]),
           "targets": counts, "records": records, "options": options, "conclusions": conclusions, "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)},
           "uncompressed_total_est": sum(usage) + saved, "seconds": round(time.time() - t0, 1), "trace": trace,
           "final_context": sent + [{"role": "assistant", "content": trace[-1]["content"]}]}
    base = ROOT / "output" / "react_compact" / f"react_compact_{q['qid']}{'_archive' if args.archive else '_strict' if args.strict else ''}"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(str(base) + "_store.json").write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    base.with_suffix(".md").write_text(report(out, store), encoding="utf-8")
    print(f"\n答案：{(final or {}).get('answer')}；核对：{out['check']}\n{out['steps']} 步，search {out['searches']}，read {out['reads']}，calc {out['calcs']}，"
          f"{out['seconds']} 秒；tokens 输入 {usage[0]} + 输出 {usage[1]} = {sum(usage)}（同一过程不压缩约 {out['uncompressed_total_est']}）"
          f" -> {base.relative_to(ROOT)}.json / .md / _store.json")


def check(final, store):
    """hard: a cited chunk that no tool returned, or a value that does not stand in its chunk.
    soft: a quote that is not verbatim in the chunk."""
    hard, soft = [], []
    opts = (final or {}).get("options")
    if isinstance(opts, dict):                               # a choice question: evidence ids, and answer = the options judged 正确
        for k, o in opts.items():
            for e in o.get("evidence") or []:                # "calc" and the like are a basis too, only a chunk id has to exist
                if e not in store:
                    (hard if "#c" in str(e) else soft).append(f"选项 {k}：依据 {e} 不是工具返回过的块")
        right = "".join(k for k in sorted(opts) if str(opts[k].get("verdict")).strip() == "正确")
        if (final or {}).get("answer") != right:
            hard.append(f"answer 写的是 {final.get('answer')}，各选项里判为正确的是 {right}")
        return {"hard": hard, "soft": soft, "options": len(opts)}
    for i in (final or {}).get("inputs") or []:
        name, ch, v = i.get("name"), i.get("chunk"), str(i.get("value", "")).replace("%", "").strip()
        if ch not in store:
            hard.append(f"{name}：块 {ch} 不是工具返回过的块")
        elif squash(v) not in squash(store[ch]["text"]):
            hard.append(f"{name}：数值 {i.get('value')} 不在块 {ch} 里")
        elif i.get("quote") and squash(i["quote"]) not in squash(store[ch]["text"]):
            soft.append(f"{name}：原句与块 {ch} 的原文不完全一致")
    return {"hard": hard, "soft": soft, "inputs": len((final or {}).get("inputs") or [])}


def report(o, store):
    """The run as a Markdown record."""
    short = lambda u: u.split("#")[1] if "#" in u else u
    L = [f"# {o['qid']}：带上下文压缩的 ReAct 作答记录", "",
         f"模型 {o['model']}，温度 0。共 {o['steps']} 步，search {o['searches']} 次，read {o['reads']} 次，calc {o['calcs']} 次，{o['seconds']} 秒。",
         f"tokens：输入 {o['usage']['prompt_tokens']} + 输出 {o['usage']['completion_tokens']} = {o['usage']['total_tokens']}；"
         f"同一过程不压缩的估算 {o['uncompressed_total_est']}。", "",
         "## 给模型的系统提示", "", "```text", o["system"], "```", "", "## 题目", "", "```text",
         o["question"] + "".join(f"\n{k}. {v.strip()}" for k, v in (o.get("options") or {}).items()), "```", "", "## 逐步过程", ""]
    for r in o["trace"]:
        c = r["context"]
        L += [f"### 第 {r['step']} 步", "",
              f"- 这一步发给模型的上下文：输入 {r['usage']['prompt_tokens']} tokens（不压缩约 {c['tokens_uncompressed_est']}）；"
              f"其中带全文的证据块 {len(c['visible'])} 个：{', '.join(c['visible']) or '无'}；证据库已有 {c['store']} 个块。",
              f"- 模型输出 {r['usage']['completion_tokens']} tokens。", ""]
        if r["content"].strip():
            L += ["**模型写的：**", "", "```text", r["content"].strip(), "```", ""]
        for i, call in enumerate(r["calls"], 1):
            L += [f"**调用 {i}：{call['name']}**　参数 `{json.dumps(call['arguments'], ensure_ascii=False)}`", ""]
            if call["chunks"] is None:
                L += ["返回：", "", "```text", call["content"] or "", "```", ""]
                continue
            for u in call["chunks"]:
                how = ("新块，全文发送并入库" if u in r["events"].get("new", []) else "仍在上下文，未重发" if u in r["events"].get("reused", [])
                       else "从证据库取出，重新发送全文")
                L += [f"- `{u}`　《{store[u]['title']}》　{store[u]['path']}　（{how}）", ""]
                if how != "仍在上下文，未重发":
                    L += ["  ```text"] + ["  " + line for line in store[u]["text"].split("\n")] + ["  ```", ""]
        if r.get("recap") and not r["calls"]:
            L += ["**模型准备作答，系统先补回（作为一条用户消息）：**", "", "```text", r["recap"].strip(), "```", ""]
        ev = r["events"]
        if ev:
            L += [f"**这一步之后的压缩：** 移出上下文、换成占位符的块 {len(ev['masked'])} 个：{', '.join(ev['masked']) or '无'}。", ""]
        if "check" in r:
            L += [f"**代码核对：** {json.dumps(r['check'], ensure_ascii=False)}", ""]
    if o.get("records"):
        L += ["## 取数记录（模型写下、代码逐条对照原文核对）", "", "| 步 | 指标 | 数值 | 单位 | 来源块 | 核对 |", "|---|---|---|---|---|---|"]
        L += [f"| {r['step']} | {r['name']} | {r['value']} | {r['unit']} | `{r['chunk']}` | {'一致' if r['ok'] else '不一致'} |" for r in o["records"]] + [""]
    if o.get("conclusions"):
        L += ["## 各选项结论（模型在过程中记录的）", "", "| 选项 | 结论 | 记录于 | 依据块 | 理由 |", "|---|---|---|---|---|"]
        L += [f"| {k} | {c['verdict']} | 第 {c['step']} 步 | {'、'.join('`' + u + '`' for u in c['chunks'])} | {c['reason']} |"
              for k, c in sorted(o["conclusions"].items())] + [""]
    L += ["## 证据库（每个块何时入库、被哪次检索返回）", "", "| 块 | 章节位置 | 入库步 | 被返回的记录 |", "|---|---|---|---|"]
    for u, s in store.items():
        rec = "；".join(f"第{x['step']}步 {x['tool']}（{x.get('target') or x.get('query') or ''}）" for x in s["returned"])
        L.append(f"| `{u}` | {s['path'][-40:]} | {s['first_step']} | {rec} |")
    L += ["", "## 最后一次调用时的完整上下文", ""]
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
