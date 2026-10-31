# -*- coding: utf-8 -*-
"""选择题策略 v2（2026-10-30，方案见 reports/option_rounds_v2_plan.md）：固定轮次换成工具调用主循环，
每次请求重新拼上下文。只做选择题（单选、多选）和判断题。flows/option_rounds.py 是 v1，留作对照。

    第 1 次请求  输入 system prompt + 题目。模型在一条回复里同时调用 update_plan（建立 plan：把题干和
                 每个选项改写成一条原子陈述，再为它列出需求清单，即槽位）和若干个 search；代码先执行
                 update_plan，再执行 search，检索结果对应到刚建好的陈述和需求上
    之后每次     模型按工具返回的结果自己决定调用 update_plan（填写槽位、下结论）/ search（补查）/
                 finish（认为已经完成），可以在同一条回复里调用多个；代码按回复里的顺序执行
    结束         finish：代码检查每个选项是否都有结论（正确或错误），能就作答，不能就把还缺的项作为
                 工具结果返回（“B 选项的 catl23 仍然缺失；C 选项为 unknown”），模型继续。
                 护栏 AUTO_FINISH：任意一次 update_plan 之后答案已经能确定，就直接结束，同一条回复里
                 排在后面的调用不再执行。主循环最多 MAX_REQUESTS 次请求
    上下文       每次请求只有 system 和 user 两条消息，不追加对话历史。user：题目、plan 的当前视图、
                 证据池、已经用过的检索词（10-30 加，同 v1 的重判提示）、一行摘要（上一轮调用了哪些工具、
                 结果如何）
    需求的池子   （10-30 改）每个需求（槽位）一个池子：search 的 for 换算成需求（槽位就是它自己，原子
                 换成它的槽位，选项字母换成这个选项所有原子的槽位，对不上的名字自成一个需求），检索到的
                 块进这些需求的池子。查重按需求：一块对这次检索服务的某个需求是新的就给出，对这些需求都
                 看过才不再给出（Seen 每个需求一份）。之前从证据池里放掉、又为新的需求检索到的块，重新给出
    证据池       （10-30 改）已有结论的选项所引用的块 + 还没完成的需求的池子 + 上一轮 update_plan 引用的块
                 + 本轮 search 检索到的块。需求完成 = 用到它的选项都有了结论（正确或错误）：槽位有值不算，
                 记录里有填了值、选项仍 unknown、又为同一需求检索的情况（fin_b_008、ins_a_018）
    原子陈述     （10-30 改）每个选项由模型把“题干 + 选项”改写成一条原子陈述（一句能直接和原文核对的
                 完整判断；候选答案写成“这个选项满足题干的条件”，括号里的说明只当提示，不写进“正确/错误”），
                 再列出需求清单（槽位）；结论判的是这条陈述与原文是否一致，按题干方向挑选项照旧。起因：
                 ins_a_011 问“哪些产品可以赔付”，选项“众安白血病医疗险（仅保障白血病复发，不赔骨折）”
                 单独判是对的，被选上了。它替换了之前“option_role 分类 + 代码拼接‘题干——选项’+ 候选类
                 一律挑正确”的做法：分类器把 ins_a_011 判成陈述，强制按候选拼接后模型仍按括号说明判
    沿用 v1      search()（narrow + BM25，前 TOP_PER_QUERY 块，同一批检索轮转合并，MAX_NEW 上限，用过的
                 检索词跳过）、作答规则（JUDGE_RULES，unknown 写 need 和 lean）、倾向兜底（没写 lean 的在
                 主循环之外单独问一次）、按题干方向挑选项、单选题取唯一有原文结论的那项、判断题当作一条表述 T

回复里连续的几个 search 作为一批执行（v1 一轮的检索也是一批：轮转合并，新块上限按一次请求算，重新给出
的块也算在上限里）。槽位的值写成“缺失”“未找到”这类词的，按未填处理（ins_a_018 照抄过 plan 视图里的“缺失”）。

    python flows/option_rounds_v2.py fc_b_006 [fc_b_010 ...] [--no-auto-finish]

每题写 output/option_rounds_v2/option_rounds_v2_<qid>.json（每次请求发出的上下文、模型的工具调用原样、
执行结果、证据池）和 .md（同样内容的可读记录）。solve(q, spent) 的返回值有 answer / usage / caveat，
和 v1 一致。
"""
import json
import re
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "flows"))

from agent.decide import caveat, choose, direction          # noqa: E402
from agent.llm import Qwen, parse_json                      # noqa: E402
from agent.retrieval import bm25_scores, get_doc            # noqa: E402
from agent.scope import cards, load_b_questions, scope      # noqa: E402
from agent.search import narrow, statements                 # noqa: E402
from option_rounds import (JUDGE_RULES, MAX_BODY, MAX_NEW, MAX_SEARCHES, OVERLAP, QUERY_RULE, SYSTEM_LEAN,  # noqa: E402
                           TOP_PER_QUERY, UNKNOWN, VERDICTS, Seen, fill_in, norm_query, norm_verdict, parse_answers)

NAME = "option_rounds_v2"
MAX_REQUESTS = 6            # requests of the main loop
AUTO_FINISH = True          # 护栏: end as soon as an update_plan makes the answer certain
MAX_TOKENS = 4000           # output of one request


# ---------------------------------------------------------------------- tools
def fn(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


ANSWER = {"type": "object", "properties": {
    "cite": {"type": "array", "items": {"type": "string"}, "description": "判断所依据的块编号；unknown 的也写已经找到的部分依据，没有就写空列表"},
    "calc": {"type": "string", "description": "需要计算的：用到的原文数、算式和结果；不需要计算写空字符串"},
    "reason": {"type": "string", "description": "一句话：依据哪句原文或哪个数、与说法是否一致"},
    "verdict": {"type": "string", "enum": ["正确", "错误", "unknown"]},
    "need": {"type": "string", "description": "unknown 时：还缺什么（哪个主体、哪个期间、哪项数据或哪一条规定）"},
    "lean": {"type": "string", "enum": ["正确", "错误"], "description": "unknown 时：假如最后还是找不到，你倾向于判什么"}},
    "required": ["cite", "calc", "reason", "verdict"]}
UPDATE_PLAN = fn("update_plan", "建立 plan、填写槽位、给<W>下结论。字段都可选：原子和槽位按编号新建或改写，只改写写出来的字段；answers 按<W>写结论。", {
    "atoms": {"type": "array", "description": "原子陈述：每个<W>一条，由题干和这个<W>合成的一句完整判断", "items": {"type": "object", "properties": {
        "id": {"type": "string", "description": "<W>字母，如 <K1>"},
        "text": {"type": "string", "description": "原子陈述：一句能直接拿去和原文核对的完整判断，写法见系统提示"},
        "slots": {"type": "array", "items": {"type": "string"}, "description": "需求清单：判断这条陈述要从文档里读到的需求（槽位）编号"}},
        "required": ["id"]}},
    "slots": {"type": "array", "description": "需求（槽位）：判断陈述要从文档里读到的一项数据或一段原文", "items": {"type": "object", "properties": {
        "id": {"type": "string", "description": "编号自己起，短、可读，如 catl23_rev"},
        "what": {"type": "string", "description": "主体 期间 口径 指标或条款名"},
        "kind": {"type": "string", "enum": ["数值", "原文"]},
        "value": {"type": "string", "description": "数：照原文抄数字；原文：照抄关键句"},
        "unit": {"type": "string", "description": "数的单位（元、千元、万元、%……）"},
        "cite": {"type": "array", "items": {"type": "string"}, "description": "值所在的块编号"},
        "where": {"type": "string", "description": "取自哪一行哪一列，或哪几句"}},
        "required": ["id"]}},
    "answers": {"type": "object", "description": "键是<W>字母<TKEY>，值是这个<W>的结论", "additionalProperties": ANSWER}}, [])
SEARCH = fn("search", "按一个检索词在本题的文档里做关键词检索（BM25），返回最相关的 <TOP> 个原文块；for 里的需求都已经看过的块不再给出。", {
    "query": {"type": "string", "description": "关键词串，写法见系统提示"},
    "for": {"type": "array", "items": {"type": "string"}, "description": "这次检索服务的原子或槽位编号"}}, ["query", "for"])
FINISH = fn("finish", "认为答案已经可以确定时调用：系统检查每个<W>是否都有结论（正确或错误），没有就返回还缺的项。", {}, [])

UNKNOWN_RULE = """5. 判断不了的（unknown）：先写 need（还缺什么：哪个主体、哪个期间、哪项数据或哪一条规定），再写 lean（假如最后还是找不到，你倾向于判 "正确" 还是 "错误"），然后调用 search 补查，for 写它服务的原子或槽位。要换检索词：换表名、章节名、科目名、同义说法，或换一种主体、期间的写法，不要重复已经用过的检索词。"""

SYSTEM = """你是金融文档问答助手，用工具回答一道<KIND>。每次请求系统都会重新给出：题目、plan 的当前视图、证据池、上一轮工具调用的一行摘要，不保留对话历史。
本题涉及的文档：
<DOCS>
plan：先把题干和每个<W>合成一条原子陈述，再为它列出需求清单。
- 原子陈述：一句完整、能直接拿去和原文核对的判断，编号用<W>字母（如 <K1>）。<W>本身就是完整判断时，照<W>原文写，只补上题干里的主体、期间等；<W>是题干问题的候选答案（产品名、机构名、数值，或要接在题干后面才成立的短结论）时，把题干的问题和这个<W>合成一句，写成“这个<W>满足题干的条件”，如题干问“以下哪些产品可以赔付这次住院费用”、<W>是“X 医疗险”，就写“X 医疗险可以赔付这次住院费用”；题干问“某项金额是多少”、<W>是“10 亿元”，就写“某项金额为 10 亿元”。<W>里括号中的说明只是提示，不照抄进陈述：说明里写了结论（如“不赔”“未给公式”“不涉及”）的，那是要你去核对的提示，陈述仍写成满足题干条件的肯定句（如“X 医疗险可以赔付这次住院费用”），由原文来判它对不对。也不要把题干问的“正确”“错误”写进陈述。
- 需求清单：判断这条陈述要从文档里读到的每一项数据或原文，各是一个槽位（编号自己起，短、可读，如 catl23_rev）。
<W>的结论就是它的原子陈述与原文是否一致，写在 update_plan 的 answers 里。
工具：
- update_plan：建立 plan、填写槽位（value、unit、cite、where）、给<W>下结论（answers）。
- search：一个检索词一次调用，for 写它服务的原子或槽位编号。系统做关键词检索（BM25），每次返回最相关的 <TOP> 个原文块；查重按槽位，for 里的槽位都已经看过的块不再给出。
- finish：认为答案已经可以确定时调用；系统检查每个<W>都有结论（正确或错误），没有就返回还缺的项。
做法：
1. 第 1 次请求：在同一条回复里同时调用 update_plan（写出每个<W>的原子陈述和需求清单）和若干个 search。逐个<W>想清楚：判断它需要文档里的哪些数据或原文（报表数字、指标、条款、承诺、事实）。比较类的说法（“均”“都”“完全一致”“高于”），每个被比较的主体、每个期间各查一次；需要计算的值，查计算所需的原始数据。几个<W>要看的是同一处原文，就只写一个 search，for 里列出它们的原子或槽位。每个<W>至少有一个 search，全部 search 不超过 <MAX> 个。
2. 之后每次：根据证据池和上一轮的结果，自己决定调用哪些工具，可以在同一条回复里调用多个：用 update_plan 填写槽位、下结论；用 search 补查；认为已经完成就调用 finish。<AUTO>
3. 证据池：已有结论的<W>所引用的块、还没有结论的<W>的槽位检索到的块、上一轮 update_plan 引用过的块、本轮检索到的块。一个槽位用到的<W>都有了结论，为它检索到的块就不再给出（被结论引用的除外）。
4. <QUERY_RULE>
下结论（update_plan 的 answers，每个<W>一项；判断的是这个<W>的原子陈述与原文是否一致，说法即指这条陈述）：
<JUDGE_RULES>
<UNKNOWN_RULE>"""
AUTO_NOTE = "每次 update_plan 之后，系统发现每个<W>都有结论，就直接结束作答。"


def tool_list(word, k1, tf):
    fill = {"W": word, "K1": k1, "TOP": TOP_PER_QUERY, "TKEY": "（判断题是 T）" if tf else ""}
    return [json.loads(fill_in(json.dumps(t, ensure_ascii=False), **fill)) for t in (UPDATE_PLAN, SEARCH, FINISH)]


# ---------------------------------------------------------------------- parsing tool arguments
def loads_first(s):
    """The first JSON value in s: the model sometimes sends a field as a JSON string with a stray
    bracket after the value (fc_b_007). Raises ValueError when s does not start with one."""
    return json.JSONDecoder().raw_decode(s.strip())[0]


def as_list(v):
    """A list argument, also when the model sent it as a JSON string or a single object; None when
    it is a string that holds no JSON value."""
    if isinstance(v, str):
        if not v.strip():
            return []
        try:
            v = loads_first(v)
        except ValueError:
            return None
    if isinstance(v, dict):
        return [v]
    return v if isinstance(v, list) else []


def as_dict(v):
    """A dict argument, also when sent as a JSON string; None when it is a string that holds no JSON value."""
    if isinstance(v, str):
        if not v.strip():
            return {}
        try:
            v = loads_first(v)
        except ValueError:
            return None
    return v if isinstance(v, dict) else {}


def text_of(v):
    return "" if v is None else str(v).strip()


NO_VALUE = re.compile(r"^[（(]?(?:缺失|未填|未找到|未检索到|未查到|待查|待定|待补充|待确认|未知|暂无数据|无数据|n/?a|none|null|空|[-—–]+)[)）]?$", re.I)


def blank(v):
    """True for an empty slot value and for words that only say it is empty (“缺失”, “未找到”)."""
    return not v or bool(NO_VALUE.match(v.strip()))


def labels_in(v):
    return list(dict.fromkeys(re.findall(r"E\d+", json.dumps(v if v is not None else [], ensure_ascii=False).upper())))


# ---------------------------------------------------------------------- one question
def main(argv):
    qids = [a for a in argv if not a.startswith("--")]
    auto = "--no-auto-finish" not in argv
    qs = {x["qid"]: x for x in load_b_questions()}
    for qid in qids:
        solve(qs[qid], auto_finish=auto)


def solve(q, spent=None, auto_finish=None):
    auto = AUTO_FINISH if auto_finish is None else auto_finish
    qid, kind = q["qid"], q["type"]
    if not q.get("options"):
        raise ValueError(f"{qid} 是{kind}，这个策略只做选择题和判断题")
    tf = kind == "判断题"
    if tf:                                              # one statement, the options are 正确 / 错误
        targets = {"T": statements(q)[0][1]}
        word, asked, clause, dword = "表述", "正确", None, None
    else:
        targets = {k: v.strip() for k, v in q["options"].items()}
        word = "选项"
        asked, clause, dword = direction(q["question"])
    keys = list(targets)
    sc = scope(q)
    title = {c["doc_id"]: c["title"] for c in cards()}
    llm = Qwen(log_path=ROOT / "logs" / f"{NAME}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    usage = spent if spent is not None else [0, 0]      # the caller's list keeps the count when the run fails
    t0 = time.time()
    seen = defaultdict(Seen)                            # need -> the chunks seen for it
    store, used = {}, {}                                # store: label -> chunk shown; used: queries already run, in order
    serves = defaultdict(set)                           # label -> needs it was retrieved for
    asked_for = defaultdict(set)                        # label -> names the search calls wrote in for
    atoms, slots, answers = {}, {}, {}                  # the plan; answers: key -> latest answer
    history = []                                        # every answer of every request
    requests, calls_log = [], []

    stem = q["question"].strip() + ("" if tf else "".join(f"\n{k}. {v}" for k, v in targets.items()))
    if tf:
        stem += f"\n\n要判断的表述（记作 T）：{targets['T']}"
    many = len(sc["docs"]) > 20
    docs = "\n".join(f"- 《{title.get(d, d)}》" for d in sc["docs"][:20]) + (f"\n（共 {len(sc['docs'])} 份，题面没有点名文档，检索在这些文档里进行）" if many else "")
    k1 = keys[0]
    system = fill_in(SYSTEM, AUTO=AUTO_NOTE if auto else "", QUERY_RULE=QUERY_RULE, JUDGE_RULES=JUDGE_RULES, UNKNOWN_RULE=UNKNOWN_RULE)
    system = fill_in(system, KIND=kind, W=word, TOP=TOP_PER_QUERY, MAX=MAX_SEARCHES, DOCS=docs, K1=k1)
    tools = tool_list(word, k1, tf)
    print(f"{qid}（{kind}）{stem}\n" + ("" if tf else f"方向：选判为“{asked}”的选项\n"))

    # -------------------------------------------------------------- model call
    def chat(user, tag):
        for attempt in range(3):
            try:
                t1 = time.time()
                resp = llm.client.chat.completions.create(
                    model=llm.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0, max_tokens=MAX_TOKENS, tools=tools,
                    extra_body={"enable_thinking": False, "parallel_tool_calls": True})
                break
            except Exception as e:                      # noqa: BLE001 - transient API errors
                if attempt == 2:
                    raise
                print(f"API 出错，重试：{e}")
                time.sleep(5)
        choice = resp.choices[0]
        u = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
        u["total_tokens"] = u["prompt_tokens"] + u["completion_tokens"]
        usage[0] += u["prompt_tokens"]
        usage[1] += u["completion_tokens"]
        for k in llm.usage:
            llm.usage[k] += u[k]
        raw = [{"name": c.function.name, "arguments": c.function.arguments or ""} for c in (choice.message.tool_calls or [])]
        with llm.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": tag, "model": llm.model,
                                "seconds": round(time.time() - t1, 1), "usage": u, "system": system, "user": user,
                                "response": choice.message.content or "", "tool_calls": raw,
                                "finish_reason": choice.finish_reason}, ensure_ascii=False) + "\n")
        return choice.message.content or "", raw, choice.finish_reason, u

    # -------------------------------------------------------------- plan
    def option_of(aid):
        if tf:
            return "T"
        alone = [k for k in dict.fromkeys(re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", aid)) if k.upper() in keys]
        if len(alone) == 1:                             # B1, B_2, atom_B
            return alone[0].upper()
        return aid[:1].upper() if aid[:1].upper() in keys else None

    def determined():
        return all((answers.get(k) or {}).get("verdict") in VERDICTS for k in keys)

    def concluded(k):
        return (answers.get(k) or {}).get("verdict") in VERDICTS

    def needs_of(names):
        """The needs a search serves: a slot as it is, an atom by its slots, an option letter by the
        slots of its atoms; a name that leads to no slot is a need of its own ("*" when for is empty)."""
        out = []
        for x in names:
            if x in slots:
                out.append(x)
            elif x in atoms and atoms[x]["slots"]:
                out += atoms[x]["slots"]
            elif x.upper() in keys and len(x) == 1 and any(at["option"] == x.upper() and at["slots"] for at in atoms.values()):
                out += [s for at in atoms.values() if at["option"] == x.upper() for s in at["slots"]]
            else:
                out.append(x)
        return list(dict.fromkeys(out)) or ["*"]

    def options_of(nd):
        """The options a need serves, read from the plan as it is now."""
        if nd in atoms:
            return {atoms[nd]["option"]}
        users = {at["option"] for at in atoms.values() if nd in at["slots"]}
        if users:
            return users
        if len(nd) == 1 and nd.upper() in keys:
            return {nd.upper()}
        if nd not in slots and nd != "*" and option_of(nd):
            return {option_of(nd)}
        return set(keys)

    def need_open(nd):
        """A need is done when every option using it has a conclusion; a filled value is not enough."""
        return not all(concluded(k) for k in options_of(nd))

    def missing():
        out = []
        for k in keys:
            a = answers.get(k)
            if a and a["verdict"] in VERDICTS:
                continue
            empty = list(dict.fromkeys(s for at in atoms.values() if at["option"] == k for s in at["slots"]
                                       if not (slots.get(s) or {}).get("value")))
            if empty:
                out.append(f"{k} {word}的 {'、'.join(empty)} 仍然缺失")
            else:
                out.append(f"{k} {word}为 unknown" if a else f"{k} {word}还没有结论")
        return "；".join(out)

    def apply_update(a, n, cited):
        """Write one update_plan call into the plan; (summary, detail)."""
        notes, new_atoms, new_slots, filled, bad = [], [], [], [], []
        fields = {}
        for f, read in (("atoms", as_list), ("slots", as_list), ("answers", as_dict)):
            fields[f] = read(a.get(f))
            if fields[f] is None:                       # say so instead of dropping it silently
                notes.append(f"{f} 不是合法 JSON，没有收")
                fields[f] = [] if read is as_list else {}
        for x in fields["atoms"]:
            if not isinstance(x, dict) or not text_of(x.get("id")):
                continue
            aid = text_of(x["id"])
            opt = option_of(aid)
            if not opt:
                notes.append(f"原子 {aid} 的编号对不上{word}字母，没有收")
                continue
            at = dict(atoms.get(aid) or {"id": aid, "option": opt, "text": "", "slots": [], "request": n})
            if "text" in x:
                at["text"] = text_of(x["text"])
            if "slots" in x:
                at["slots"] = list(dict.fromkeys(text_of(s) for s in as_list(x["slots"]) or [] if text_of(s)))
            if aid not in atoms:
                new_atoms.append(aid)
            atoms[aid] = at
        for x in fields["slots"]:
            if not isinstance(x, dict) or not text_of(x.get("id")):
                continue
            sid = text_of(x["id"])
            s = dict(slots.get(sid) or {"id": sid, "what": "", "kind": "", "value": "", "unit": "", "cite": [], "where": "", "request": n})
            had = bool(s["value"])
            for f in ("what", "kind", "value", "unit", "where"):
                if f in x:
                    s[f] = text_of(x[f])
            if s["value"] and blank(s["value"]):        # “缺失”, “未找到”: not a value
                notes.append(f"槽位 {sid} 的值“{s['value']}”按未填处理")
                s["value"] = ""
            if "cite" in x:
                labs = labels_in(x["cite"])
                bad += [lab for lab in labs if lab not in store]
                s["cite"] = [lab for lab in labs if lab in store]
                cited.update(s["cite"])
            if s["value"] and not had:
                filled.append(sid)
            if sid not in slots:
                new_slots.append(sid)
            slots[sid] = s
        got = {}
        ans = fields["answers"]
        if ans:
            for v in ans.values():
                if isinstance(v, dict):
                    bad += [lab for lab in labels_in(v.get("cite")) if lab not in store]
            got = parse_answers({"answers": ans}, keys, set(store), n)
            for k, v in got.items():
                answers[k] = v
                history.append({"key": k, **v})
                cited.update(v["cite"])
            other = [k for k in ans if k not in keys]
            if other:
                notes.append(f"answers 里的 {'、'.join(map(str, other))} 不是本题的{word}，没有收")
        if bad:
            notes.append(f"引用里不认识的块编号 {'、'.join(dict.fromkeys(bad))} 已去掉")
        parts = []
        if new_atoms or new_slots:
            parts.append(f"新建 {len(new_atoms)} 个原子、{len(new_slots)} 个槽位")
        if filled:
            parts.append(f"填 {len(filled)} 个槽位（{'、'.join(filled)}）")
        if got:
            parts.append("结论 " + "、".join(f"{k} {v['verdict']}" for k, v in got.items()))
        summary = "update_plan（" + "；".join(parts + notes or ["没有改动"]) + "）"
        return summary, {"new_atoms": new_atoms, "new_slots": new_slots, "filled": filled, "answers": list(got), "notes": notes}

    # -------------------------------------------------------------- search (v1's, batched per run of search calls)
    def admit(needs, u):
        """(the needs for which the chunk is new, its key); the chunk is recorded as seen for them."""
        fresh, k = [], None
        for nd in needs:
            f, k = seen[nd].admit(u)
            if f:
                fresh.append(nd)
        return fresh, k

    def run_searches(batch, n, pool, budget):
        """Run a batch of search calls; (results, summary, got, new, skipped). Dedup per need: a hit is
        given when it is new for one of the needs its search serves. got: labels this batch returned
        (new, back again for a new need, or already in this request's pool)."""
        results, skipped = [], []
        for c in batch:
            a = c["args"]
            query = norm_query(a.get("query"))
            if not query:
                skipped.append("（空检索词）")
                continue
            if query in used or any(query == r["query"] for r in results):
                skipped.append(f"用过的「{query}」")
                continue
            if budget["searches"] >= MAX_SEARCHES:
                skipped.append(f"超过 {MAX_SEARCHES} 个的「{query}」")
                continue
            budget["searches"] += 1
            target = [text_of(x) for x in as_list(a.get("for")) or [] if text_of(x)]
            where, text = narrow(q, query, sc)
            hits = [u for u, score in bm25_scores(text, where)[:TOP_PER_QUERY] if score > 0]
            results.append({"query": query, "for": target, "needs": needs_of(target), "searched": text.strip(),
                            "docs": len(where), "hits": hits, "status": {}})
            used[query] = n
        label_of = {v["uid"]: k for k, v in store.items()}
        new, got = [], []
        for pos in range(TOP_PER_QUERY):
            for r in results:
                if pos >= len(r["hits"]):
                    continue
                u, needs = r["hits"][pos], r["needs"]

                def take(lab, status):
                    serves[lab].update(needs)
                    asked_for[lab].update(r["for"])
                    found.setdefault(lab, set()).update(r["for"])
                    got.append(lab)
                    r["status"][u] = (status, lab)

                if u in label_of:                       # given a label before: this request, or earlier
                    lab = label_of[u]
                    if lab in pool or store[lab]["request"] == n:   # in the context already, or new in this request
                        admit(needs, u)
                        take(lab, "shown" if lab in pool else "new")
                        continue
                    if budget["new"] >= MAX_NEW:
                        r["status"][u] = ("cut", None)
                        continue
                    fresh, _ = admit(needs, u)
                    if not fresh:                       # every need of this search has seen it
                        r["status"][u] = ("seen", None)
                        continue
                    budget["new"] += 1
                    take(lab, "back")                   # let go of earlier, new for one of these needs
                    continue
                if budget["new"] >= MAX_NEW:
                    r["status"][u] = ("cut", None)
                    continue
                fresh, k = admit(needs, u)
                if not fresh:                           # covered by chunks these needs have seen
                    r["status"][u] = ("seen", None)
                    continue
                lab = f"E{len(store) + 1}"
                x = get_doc(k["doc"]).by_id[u]
                store[lab] = {"uid": u, "doc": k["doc"], "title": title.get(k["doc"], k["doc"]), "pages": k["pages"],
                              "chars": k["chars"], "path": x.get("path") or "", "text": x["body"][:MAX_BODY], "request": n}
                label_of[u] = lab
                budget["new"] += 1
                new.append(lab)
                take(lab, "new")
        lines = []
        for r in results:
            st = list(r["status"].values())
            labs = [lab for t, lab in st if t in ("new", "shown", "back")]
            part = "、".join(dict.fromkeys(labs)) if labs else "没有新块"
            back = [lab for t, lab in st if t == "back"]
            if back:
                part += f"（{'、'.join(back)} 之前看过，为这次的槽位重新给出）"
            n_seen, n_cut = sum(t == "seen" for t, _ in st), sum(t == "cut" for t, _ in st)
            if n_seen:
                part += f"，{n_seen} 块这些槽位之前看过、不再给出"
            if n_cut:
                part += f"，{n_cut} 块超出本次上限、没有给出"
            if not r["hits"]:
                part = "没有命中"
            lines.append(f"「{r['query']}」→ {part}")
        if skipped:
            lines.append("跳过" + "、".join(skipped))
        summary = f"search {len(results)} 次（" + "；".join(lines) + "）"
        return results, summary, list(dict.fromkeys(got)), new, skipped

    # -------------------------------------------------------------- context
    def plan_view():
        if not atoms and not slots and not answers:
            return "还没有，用 update_plan 建立。"
        L = []
        for k in keys:
            a = answers.get(k)
            if a:
                head = f"{k}：{a['verdict']}｜依据 {'、'.join(a['cite']) or '无'}"
                if a["calc"]:
                    head += f"｜计算：{a['calc']}"
                if a["reason"]:
                    head += f"｜理由：{a['reason']}"
                if a["verdict"] == UNKNOWN and a["need"]:
                    head += f"｜还缺：{a['need']}"
                if a["verdict"] == UNKNOWN and a["lean"]:
                    head += f"｜倾向 {a['lean']}"
            else:
                head = f"{k}：还没有结论"
            L.append(head)
            for at in atoms.values():
                if at["option"] == k:
                    L.append(f"  陈述 {at['id']}：{at['text'] or '（未写）'}｜需求 {'、'.join(at['slots']) or '无'}")
        if slots:
            L += ["", "需求清单（槽位）：", "| 槽位 | 内容 | 类型 | 值 | 依据 | 出处 |", "|---|---|---|---|---|---|"]
            for s in slots.values():
                val = (f"{s['value']} {s['unit']}".strip() if s["value"] else "（未填）").replace("|", "/").replace("\n", " ")
                L.append(f"| {s['id']} | {s['what']} | {s['kind']} | {val} | {'、'.join(s['cite'])} | {s['where'].replace('|', '/')} |")
        return "\n".join(L)

    def evidence_of(lab):
        """The concluded options citing a chunk."""
        return [k for k in keys if concluded(k) and lab in answers[k]["cite"]]

    def show_chunk(lab, tagged):
        s = store[lab]
        p = s["pages"]
        page = f"第{p[0]}页" if p[0] == p[-1] else f"第{p[0]}-{p[-1]}页"
        head = f"[{lab}]《{s['title']}》{page}" + (f" | {s['path']}" if s["path"] else "")
        if tagged.get(lab):
            head += f" | 为 {'、'.join(sorted(tagged[lab]))} 检索到"
        ev = evidence_of(lab)
        if ev:
            head += f" | {'、'.join(ev)} 的依据"
        return head + "\n" + s["text"]

    def context(pool, parts, tagged, summary):
        L = [f"题目：\n{stem}", "", "当前 plan：", plan_view(), ""]
        if pool:
            L += [f"证据池（已有结论的依据 {parts['evidence']} 块 + 还没完成的槽位的块 {parts['open']} 块 + 本轮新给出的 {parts['new']} 块）：", ""]
            L += [show_chunk(lab, tagged) + "\n" for lab in pool]
        else:
            L += ["证据池：（没有原文块）", ""]
        if used:                                        # as v1's retry prompt (10-30)
            L.append(f"已经用过的检索词：{'；'.join(used)}")
        if summary:
            L.append(f"上一轮：{summary}")
        return "\n".join(L).rstrip()

    # -------------------------------------------------------------- main loop
    def next_pool_of(pool, cited, got):
        """Evidence of the concluded options + the chunks of needs still open + this request's citations
        + what this request's searches gave; (labels in label order, how many of each part)."""
        keep = {lab for lab in store if evidence_of(lab)}
        keep |= {lab for lab, nds in serves.items() if any(need_open(nd) for nd in nds)}
        keep |= set(cited) | set(got)
        labs = sorted(keep, key=lambda s: int(s[1:]))
        fresh = {lab for lab in got if lab not in pool}
        parts = {"new": len(fresh), "evidence": sum(1 for lab in labs if lab not in fresh and evidence_of(lab))}
        parts["open"] = len(labs) - parts["new"] - parts["evidence"]
        return labs, parts

    pool, parts_now, tagged, summary = [], {}, {}, ""
    stop = ""
    for n in range(1, MAX_REQUESTS + 1):
        user = context(pool, parts_now, tagged, summary)
        content, raw, finish_reason, u = chat(user, f"{qid}:r{n}")
        calls_log.append({"stage": f"r{n}", "usage": u})
        print(f"--- 第 {n} 次请求（输入 {u['prompt_tokens']}，输出 {u['completion_tokens']}）证据池 {len(pool)} 块，"
              f"工具调用：{'、'.join(c['name'] for c in raw) or '无'}" + (f"，finish_reason={finish_reason}" if finish_reason not in ("tool_calls", "stop") else ""))
        calls = []
        for c in raw:
            try:
                args = json.loads(c["arguments"] or "{}")
                calls.append({"name": c["name"], "args": args if isinstance(args, dict) else {}})
            except json.JSONDecodeError as e:
                calls.append({"name": c["name"], "error": str(e)})
        order = calls
        if n == 1:                                      # update_plan first, then the rest in the reply's order
            order = [c for c in calls if c["name"] == "update_plan"] + [c for c in calls if c["name"] != "update_plan"]
        found = {}                                      # label -> atoms / slots it was searched for, this request
        cited, got = set(), []
        parts, executed = [], []
        budget = {"searches": 0, "new": 0}
        ended = None
        i = 0
        while i < len(order):
            c = order[i]
            if "error" in c:
                parts.append(f"{c['name']}：参数不是合法 JSON，没有执行")
                executed.append({"name": c["name"], "result": parts[-1], "error": c["error"]})
                i += 1
                continue
            if c["name"] == "search":                   # a run of search calls is one batch
                batch = []
                while i < len(order) and order[i]["name"] == "search" and "error" not in order[i]:
                    batch.append(order[i])
                    i += 1
                results, s, labs, new, skipped = run_searches(batch, n, set(pool), budget)
                got += labs
                parts.append(s)
                executed.append({"name": "search", "calls": [b["args"] for b in batch], "result": s, "results": results, "new": new, "skipped": skipped})
                print(s)
                continue
            i += 1
            if c["name"] == "update_plan":
                try:
                    s, detail = apply_update(c["args"], n, cited)
                except Exception as e:                  # noqa: BLE001 - a failing call answers with its error
                    s, detail = f"update_plan：执行出错（{type(e).__name__}: {e}）"[:300], {"error": traceback.format_exc()[-1500:]}
                parts.append(s)
                executed.append({"name": "update_plan", "args": c["args"], "result": s, "detail": detail})
                print(s)
                if auto and determined():
                    ended = "护栏：update_plan 之后每个" + word + "都有结论"
            elif c["name"] == "finish":
                if determined():
                    ended = "finish 通过"
                    executed.append({"name": "finish", "result": "通过"})
                else:
                    s = "finish 未通过：" + missing()
                    parts.append(s)
                    executed.append({"name": "finish", "result": s})
                    print(s)
            else:
                parts.append(f"{c['name']}：没有这个工具")
                executed.append({"name": c["name"], "result": parts[-1]})
            if ended:
                break
        skipped_calls = [{"name": c["name"]} for c in order[i:]] if ended else []
        cited_list = sorted(cited, key=lambda s: int(s[1:]))
        next_pool, next_parts = next_pool_of(pool, cited_list, got)
        requests.append({"request": n, "user": user, "pool": pool, "content": content, "tool_calls": raw, "finish_reason": finish_reason,
                         "executed": executed, "not_executed": skipped_calls, "cited": cited_list, "got": list(dict.fromkeys(got)),
                         "next_pool": next_pool, "next_parts": next_parts, "usage": u, "ended": ended,
                         "open_needs": sorted({nd for nds in serves.values() for nd in nds if need_open(nd)})})
        if ended:
            stop = ended
            break
        summary = "；".join(parts) if parts else "没有调用工具"
        pool, parts_now = next_pool, next_parts
        tagged = {lab: asked_for[lab] for lab in next_pool if asked_for.get(lab)}
    else:
        stop = f"请求次数用完（{MAX_REQUESTS} 次）"
    print(f"停止：{stop}")

    # -------------------------------------------------------------- 兜底：仍为 unknown 的按倾向作答（v1）
    for k in keys:
        if k not in answers:
            answers[k] = {"verdict": UNKNOWN, "cite": [], "calc": "", "reason": "模型没有给出这个" + word + "的结论",
                          "need": "", "lean": "", "round": None}
    open_keys = [k for k in keys if answers[k]["verdict"] == UNKNOWN]
    no_lean = [k for k in open_keys if not answers[k]["lean"]]
    lean_call = None
    if no_lean:
        cited = sorted({c for a in answers.values() for c in a["cite"]}, key=lambda s: int(s[1:]))
        user = (f"题目：\n{stem}\n\n当前 plan：\n{plan_view()}\n\n要按倾向作答的{word}：{'、'.join(no_lean)}\n\n"
                f"之前引用过的原文块（{len(cited)} 块）：\n\n" + ("\n\n".join(show_chunk(lab, {}) for lab in cited) or "（无）"))
        content, u = llm.chat(fill_in(SYSTEM_LEAN, W=word, K1=k1), user, max_tokens=600, tag=f"{qid}:lean")
        usage[0] += u["prompt_tokens"]
        usage[1] += u["completion_tokens"]
        calls_log.append({"stage": "lean", "usage": u, "user": user, "response": content})
        data = parse_json(content)
        leans = data.get("lean") if isinstance(data.get("lean"), dict) else {}
        reasons = data.get("reason") if isinstance(data.get("reason"), dict) else {}
        lean_call = {k: {"lean": norm_verdict(leans.get(k)), "reason": str(reasons.get(k) or "")} for k in no_lean}
        for k in no_lean:
            if lean_call[k]["lean"] != UNKNOWN:
                answers[k]["lean"] = lean_call[k]["lean"]
    final = {k: (a["verdict"] if a["verdict"] != UNKNOWN else (a["lean"] or None)) for k, a in answers.items()}
    leaned = [k for k in open_keys if final[k]]

    # -------------------------------------------------------------- 选项（v1）
    notes = []
    if tf:
        want = final["T"]
        chosen = [k for k, v in q["options"].items() if want and v.strip() == want]
        how = f"表述判为“{want}”，选“{want}”那项" if want else "表述没有结论"
    else:
        chosen, unfounded = choose(asked, final)
        how = f"代码按题干方向挑出判为“{asked}”的选项"
        if kind == "单选题" and len(chosen) > 1:
            firm = [k for k in chosen if answers[k]["verdict"] != UNKNOWN]
            if len(firm) == 1:
                chosen, how = firm, how + "；单选题挑出了多个，取其中唯一有原文结论的那个"
        if kind == "单选题" and len(chosen) != 1:
            notes.append(f"单选题挑出了 {len(chosen)} 个选项")
        if unfounded:
            notes.append(caveat(asked, unfounded))
    answer = "".join(chosen)
    if leaned:
        notes.append(f"{word} " + "、".join(f"{k}（{final[k]}）" for k in leaned) + " 没有原文结论，按模型倾向作答")
    note = "；".join(notes)
    print(f"\n结论：{final}\n答案：{answer or '（空）'}（{how}）" + (f"\n注意：{note}" if note else ""))

    out = {"qid": qid, "type": kind, "question": q["question"].strip(), "options": q["options"], "targets": targets,
           "model": llm.model, "system": system, "tools": tools, "direction": {"asked": asked, "clause": clause, "word": dword},
           "settings": {"max_requests": MAX_REQUESTS, "auto_finish": auto, "top_per_query": TOP_PER_QUERY, "max_searches": MAX_SEARCHES,
                        "max_new": MAX_NEW, "max_body": MAX_BODY, "overlap": OVERLAP, "max_tokens": MAX_TOKENS},
           "requests": requests, "n_requests": len(requests), "stop": stop, "atoms": atoms, "slots": slots,
           "answers_by_option": answers, "history": history, "lean_call": lean_call, "verdicts": final, "leaned": leaned,
           "answer": answer, "answered_by": how, "caveat": note, "calls": calls_log, "n_calls": len(calls_log),
           "usage": dict(llm.usage), "seconds": round(time.time() - t0, 1),
           "store": store, "serves": {lab: sorted(v) for lab, v in serves.items()},
           "seen": {nd: {d: [list(w) for w in ws] for d, ws in s.windows.items()} for nd, s in seen.items()}}
    base = ROOT / "output" / NAME / f"{NAME}_{qid}"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    base.with_suffix(".md").write_text(report(out), encoding="utf-8")
    uu = llm.usage
    print(f"{len(requests)} 次请求{'（另有 1 次倾向调用）' if lean_call else ''}，看过 {len(store)} 块，tokens {uu['prompt_tokens']} + {uu['completion_tokens']} = "
          f"{uu['total_tokens']}，{out['seconds']} 秒 -> {base.relative_to(ROOT)}.json / .md")
    return out


# ---------------------------------------------------------------------- record
def report(o):
    """The run as a Markdown record: every request's context, the tool calls as sent, what the code did."""
    u = o["usage"]
    L = [f"# {o['qid']}：选择题策略 v2（工具调用主循环）作答记录", "",
         f"{o['type']}。模型 {o['model']}，温度 0，思考模式关闭，护栏 {'开' if o['settings']['auto_finish'] else '关'}。"
         f"{o['n_requests']} 次请求" + ("，另有 1 次倾向调用" if o["lean_call"] else "") + f"，{o['seconds']} 秒；"
         f"tokens 输入 {u['prompt_tokens']} + 输出 {u['completion_tokens']} = {u['total_tokens']}。停止原因：{o['stop']}。",
         f"答案：**{o['answer'] or '（空）'}**（{o['answered_by']}）。" + (f"注意：{o['caveat']}。" if o["caveat"] else ""), "",
         "## 题目", "", "```text", o["question"] + "".join(f"\n{k}. {v.strip()}" for k, v in o["options"].items()), "```", "",
         "## 最终结论", "", "| 选项 | 结论 | 来源 | 依据 | 理由 |", "|---|---|---|---|---|"]
    for k, a in o["answers_by_option"].items():
        src = f"第 {a['round']} 次请求" if a["verdict"] != UNKNOWN else ("模型倾向" if o["verdicts"].get(k) else "没有结论")
        L.append(f"| {k} | {o['verdicts'].get(k) or '—'} | {src} | {'、'.join(a['cite'])} | {(a['reason'] + (' 计算：' + a['calc'] if a['calc'] else '')).replace('|', '/')} |")
    L += ["", "## 系统提示", "", "<details><summary>展开</summary>", "", "```text", o["system"], "```", "", "</details>", "", "## 逐次请求", ""]
    for r in o["requests"]:
        L += [f"### 第 {r['request']} 次请求（输入 {r['usage']['prompt_tokens']}，输出 {r['usage']['completion_tokens']}，证据池 {len(r['pool'])} 块）", "",
              "<details><summary>发给模型的内容</summary>", "", "```text", r["user"], "```", "", "</details>", ""]
        if r["content"].strip():
            L += ["**模型文字：**", "", "```text", r["content"].strip(), "```", ""]
        L += ["**工具调用（原样）：**", ""]
        L += [f"- `{c['name']}` `{c['arguments']}`" for c in r["tool_calls"]] or ["（没有调用工具）"]
        if r["finish_reason"] not in ("tool_calls", "stop"):
            L += ["", f"finish_reason={r['finish_reason']}"]
        L += ["", "**执行：**", ""]
        for e in r["executed"]:
            L.append(f"- {e['result']}")
            if e["name"] == "search":
                for s in e["results"]:
                    got = "，".join(f"{x.split('#')[-1]}：{ {'new': '新 ', 'shown': '已在证据池 ', 'back': '重新给出 ', 'seen': '见过', 'cut': '超上限'}[t] }{lab or ''}"
                                   for x, (t, lab) in s["status"].items()) or "没有命中"
                    L.append(f"  - 「{s['query']}」（为 {'、'.join(s['for']) or '—'}）实际检索「{s['searched']}」，{s['docs']} 份文档：{got}")
        if r["not_executed"]:
            L.append(f"- 结束后没有执行：{'、'.join(c['name'] for c in r['not_executed'])}")
        L += ["", f"引用：{'、'.join(r['cited']) or '无'}；本轮检索到：{'、'.join(r['got']) or '无'}；下一次的证据池：{'、'.join(r['next_pool']) or '无'}。"
              + (f"结束：{r['ended']}。" if r["ended"] else ""), ""]
    if o["lean_call"]:
        c = next(x for x in o["calls"] if x["stage"] == "lean")
        L += ["### 兜底：按倾向作答", "", "```text", c["response"], "```", ""]
    L += ["## 最终 plan", "", "| 原子 | 选项 | 说法 | 槽位 |", "|---|---|---|---|"]
    L += [f"| {a['id']} | {a['option']} | {a['text'].replace('|', '/')} | {'、'.join(a['slots'])} |" for a in o["atoms"].values()]
    L += ["", "| 槽位 | 内容 | 类型 | 值 | 依据 | 出处 |", "|---|---|---|---|---|---|"]
    L += [f"| {s['id']} | {s['what']} | {s['kind']} | {(s['value'] + ' ' + s['unit']).strip().replace('|', '/') or '缺失'} | {'、'.join(s['cite'])} | {s['where'].replace('|', '/')} |"
          for s in o["slots"].values()]
    L += ["", "## 看过的原文块", "", "| 编号 | 块 | 页 | 字符区间 | 请求 |", "|---|---|---|---|---|"]
    L += [f"| {lab} | {s['uid']} | {s['pages'][0]}-{s['pages'][-1]} | {s['chars'][0]}-{s['chars'][1]} | {s['request']} |" for lab, s in o["store"].items()]
    return "\n".join(L)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
