# -*- coding: utf-8 -*-
"""计算题策略（2026-10-31）：选择题策略 v2（flows/option_rounds_v2.py，方案 reports/option_rounds_v2_plan.md）
的主循环搬到填空计算题上。方案和改动见 reports/calc_rounds_plan.md。

    和 v2 一样   工具调用主循环（update_plan / search / finish），第 1 次请求同时建 plan 和检索、代码先执行
                 update_plan；每次请求只有 system 和 user 两条消息，user 是题目、plan 视图、证据池、已经用过的
                 检索词、一行摘要；需求（槽位）各一个池子，查重按需求；证据池 = 已算出的部分引用的块 + 还没完成
                 的需求的块 + 上一轮引用的块 + 本轮检索到的块；护栏 AUTO_FINISH；最多 MAX_REQUESTS 次请求；
                 search() 原样（narrow + BM25，前 TOP_PER_QUERY 块，同一批轮转合并，MAX_NEW 上限，用过的词跳过）
    选项 -> 答案部分   p1、p2…：题面写了“答案格式为‘…；…’”或只要一个结果的，代码按 react_plan.answer_parts
                 定好；“分别是多少”这类，由模型在第 1 次请求里列出。每个部分写 what、unit（输出单位）、slots
    槽位         数值（文档里的数）/ 原文（条款、规则原文）/ 题给（题面给出的数或日期）。value、unit 照原文或
                 题面抄；代码把数统一换算成以 1 为单位（agent/units.py：23% -> 0.23，13 万元 -> 130000）放进
                 算式变量，换算不了的单位（元/10股）保持原值并标“未换算”。题给的值要在题面里找得到
    结论 -> 算式   answers 每个部分：cite、formula、reason、status（算得出 / unknown），unknown 的再写 need、
                 fallback。formula 是 Python：一个表达式，或几行、最后赋给 ans；槽位编号就是变量名。算式里
                 不写数字（0–10、12、100、1000、10000、365 这类常数除外），题面的数也要建成题给槽位。
                 代码运行算式（另起进程，超时 20 秒），结果按部分的输出单位换算、按题目的小数位四舍五入
    检索词       （10-31，你定的）第一批检索由代码按需求做（need_queries）：模型给每个要查文档的槽位写 stem
                 （它在题面里对应的句子，照抄，代码核对是题面原文），每个需求一条检索词 = 主体名 + 它的 stem +
                 题面里没被任何需求摘走的句子（开头、条件、问句）的分词；没写 stem 的需求退回按主体切
                 （segment_queries）。第 1 次请求里模型只建 plan，它写的 search 不执行。第 2 次请求起，模型自己
                 写检索词补查（提示里建议先照抄证据池原文）
    finish 检查  每个部分的算式都能算出结果；没有就返回“p1 的 zy_cv 仍然缺失”“p1 的算式没有通过：…”“p2 为 unknown”
    兜底         结束时还没算出的部分：运行模型写的 fallback（假设算式，可以写数字）；没写的，在主循环之外再问
                 一次假设算式（不计入 MAX_REQUESTS）；都不行写“无法计算”

    python flows/calc_rounds.py ins_b_019 [res_b_005 ...] [--no-auto-finish]

每题写 output/calc_rounds/calc_rounds_<qid>.json 和 .md。solve(q, spent) 返回 answers（每个部分一个字符串）、
usage、caveat，main.py 可以直接接。
"""
import ast
import json
import os
import re
import subprocess
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

from agent.claims import tokenize                           # noqa: E402
from agent.llm import Qwen, parse_json                      # noqa: E402
from agent.retrieval import bm25_scores, get_doc            # noqa: E402
from agent.scope import cards, load_b_questions, scope      # noqa: E402
from agent.search import narrow                             # noqa: E402
from agent.units import NO_SIGN, factor_of, render, render_scaled, to_base   # noqa: E402
from option_rounds import MAX_BODY, MAX_NEW, MAX_SEARCHES, OVERLAP, TOP_PER_QUERY, Seen, fill_in, norm_query  # noqa: E402
from option_rounds_v2 import AUTO_FINISH, MAX_REQUESTS, MAX_TOKENS, as_dict, as_list, blank, fn, labels_in, text_of  # noqa: E402
from react_plan import answer_parts                         # noqa: E402

NAME = "calc_rounds"
UNKNOWN = "unknown"
READY = "算得出"
KINDS = ("数值", "原文", "题给")
CONSTS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 100, 1000, 10000, 365, 366, 360, 0.5,
          10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8, 10 ** 12, 0.01, 0.001, 0.0001,
          1e-5, 1e-6, 1e-7, 1e-8, 1e-12}                                          # the unit factors (换成万、亿……)
CONST_WORDS = "0 到 10、12、100、1000、10000、365 和换算单位用的 1e6、1e8"
OK_MODULES = {"datetime", "math"}
ALLOWED = {"math", "date", "datetime", "timedelta", "to_date", "V", "abs", "round", "min", "max", "sum", "sorted", "len", "float", "int",
           "str", "range", "enumerate", "zip", "dict", "list", "tuple", "set", "any", "all", "reversed", "map", "filter", "pow",
           "divmod", "isinstance", "ans", "True", "False", "None"}
RUN_TIMEOUT = 20

# ---------------------------------------------------------------------- tools
ANSWER = {"type": "object", "properties": {
    "cite": {"type": "array", "items": {"type": "string"}, "description": "算式用到的值所在的块编号；unknown 的也写已经找到的部分依据，没有就写空列表"},
    "formula": {"type": "string", "description": "Python 算式：一个表达式，或几行、最后赋给 ans；槽位编号当变量名，写法见系统提示"},
    "reason": {"type": "string", "description": "一句话：用了哪些原文和题给的数、怎么算"},
    "status": {"type": "string", "enum": [READY, UNKNOWN]},
    "need": {"type": "string", "description": "unknown 时：还缺什么（哪个主体、哪个期间、哪项数据或哪一条规定）"},
    "fallback": {"type": "string", "description": "unknown 时：假如最后还是找不到，按什么假设算（Python 算式，缺的数可以直接写数字，也按以 1 为单位：70 万元写 700000，5% 写 0.05；在 reason 里说明假设）"}},
    "required": ["cite", "formula", "reason", "status"]}
UPDATE_PLAN = fn("update_plan", "建立 plan、填写槽位、给答案部分写算式。字段都可选：部分和槽位按编号新建或改写，只改写写出来的字段；answers 按部分编号写算式。", {
    "parts": {"type": "array", "description": "答案部分：每个部分一个值", "items": {"type": "object", "properties": {
        "id": {"type": "string", "description": "部分编号，如 p1"},
        "what": {"type": "string", "description": "这个部分算什么"},
        "unit": {"type": "string", "description": "可以不写：输出单位由系统按题面定"},
        "slots": {"type": "array", "items": {"type": "string"}, "description": "需求清单：算这个部分要用的槽位编号"}},
        "required": ["id"]}},
    "slots": {"type": "array", "description": "需求（槽位）：算式要用的一个数，或一段规则原文", "items": {"type": "object", "properties": {
        "id": {"type": "string", "description": "编号自己起，要能当 Python 变量名，如 catl25_rev"},
        "what": {"type": "string", "description": "主体 期间 口径 指标或条款名；题给的写题面的说法"},
        "kind": {"type": "string", "enum": list(KINDS)},
        "value": {"type": "string", "description": "照原文或题面抄：数写数字，不换算单位；原文照抄关键句"},
        "unit": {"type": "string", "description": "照原文写全（元、千元、万元、亿元、%、万辆、kWh……）"},
        "cite": {"type": "array", "items": {"type": "string"}, "description": "值所在的块编号；题给的不写"},
        "where": {"type": "string", "description": "取自哪一行哪一列，或哪几句"},
        "stem": {"type": "array", "items": {"type": "string"},
                 "description": "这个槽位在题面里对应的句子，照抄题面原文，讲哪个主体、哪种情形就摘哪一句；题给槽位摘它的数所在的那一句"}},
        "required": ["id"]}},
    "calc": {"type": "array", "description": "中间量：几处要用的中间结果（如比率、增速、合计），只写算式，系统运行后记下原始结果（不四舍五入）",
             "items": {"type": "object", "properties": {
                 "id": {"type": "string", "description": "编号自己起，要能当 Python 变量名，如 catl_ocf_rate"},
                 "what": {"type": "string", "description": "这个中间量是什么"},
                 "formula": {"type": "string", "description": "Python 算式，可以用槽位、别的中间量和答案部分的编号"}},
                 "required": ["id", "formula"]}},
    "answers": {"type": "object", "description": "键是部分编号（如 p1），值是这个部分的算式", "additionalProperties": ANSWER}}, [])
SEARCH = fn("search", "按一个检索词在本题的文档里做关键词检索（BM25），返回最相关的 <TOP> 个原文块；for 里的需求都已经看过的块不再给出。", {
    "query": {"type": "string", "description": "关键词串，写法见系统提示"},
    "for": {"type": "array", "items": {"type": "string"}, "description": "这次检索服务的槽位或答案部分编号"}}, ["query", "for"])
FINISH = fn("finish", "认为答案已经可以确定时调用：系统检查每个答案部分的算式都能算出结果，没有就返回还缺的项。", {}, [])

PARTS_FIXED = "答案部分系统已按题面定好，parts 里只能用这些编号，给每个部分写 unit（输出单位）和 slots：\n<LIST>"
PARTS_OPEN = "这道题要先列出答案的各部分：按题目要求的顺序编号 p1、p2…，每个部分一个值，写在 parts 里。"

SYSTEM = """你是金融文档问答助手，用工具回答一道计算题。每次请求系统都会重新给出：题目、plan 的当前视图、证据池、上一轮工具调用的一行摘要，不保留对话历史。
本题涉及的文档：
<DOCS>
<PARTS>
plan：每个答案部分写清算什么（what），再列出算它要用的需求清单（槽位）。输出单位由系统按题面定，plan 视图里会列出。
- 槽位：算式要用的每一个数、每一段规则原文各是一个槽位。编号自己起，短、可读，要能当 Python 变量名（字母、数字、下划线，不以数字开头，如 catl25_rev）。类型：
  - 数值：从文档里读到的数；
  - 原文：从文档里读到的条款、规则原文（value 照抄关键句）；
  - 题给：题面给出的一个数或一个日期，照抄题面原样（如 60、2026年3月27日），不用检索；要算的（如 60+10）拆成几个题给槽位，写进算式。题给的数不用查，但它们在题面里的名字（如“累计所交保险费”）正适合写进检索词去查规则。
  按条款、规定计算的量（退还、赔付、领取的金额，期限、日期），题面给的数只是规则的输入：为规则建原文槽位去查，规则里的比例、天数建数值槽位。
  value 照原文或题面抄，不要自己换算：数只写数字（“下降 18.97%”写 -18.97），unit 照原文写全（看行名、表头或“单位：”，如 元、千元、万元、亿元、%、万辆、kWh）。系统把数统一换算成以 1 为单位放进算式：23% 是 0.23，13 万元是 130000（元）；plan 视图里列出每个槽位换算后的值，换算不了的单位（如 元/10股）保持原值，标“未换算”。
- 算式（formula）：Python，用槽位编号当变量名，如 (bev26 - bev25) / bev25。算式里不写数字：题面的数建成题给槽位，原文里的数（包括条款里的比例、天数）建成数值槽位，只有 <CONSTS> 这类常数可以直接写。按以 1 为单位算，比率就是小数（0.2227，不要乘 100），系统按题面的输出单位换算并按题目要求保留小数（已经乘了 100、除以 10000 换成题目单位的，系统也认得，不会再换一次）；中间不要四舍五入。排序、日期这类可以写几行，最后把结果赋给 ans：排序的结果是“公司>公司”这样的字符串；日期用 to_date(题给的日期槽位) 得到 date，再用 timedelta 推算，结果可以是 date。算式里不要 round，保留几位小数由系统做。
- 中间量（calc）：几处要用的中间结果（比率、增速、合计等）写成中间量 {id, what, formula}，只写算式、不写数，系统运行后把原始结果（不四舍五入）记在 plan 的中间量表里。后面的算式可以直接用中间量的编号，也可以用前面答案部分的编号：如 p3 = p2 - p1，用的是 p1、p2 没有四舍五入的原始值。算出来的数不要填进槽位，槽位只放原文和题面的数。
工具：
- update_plan：建立 plan（parts、slots）、填写槽位（value、unit、cite、where）、写中间量（calc）、给答案部分写算式（answers）。
- search：一个检索词一次调用，for 写它服务的槽位或答案部分编号。系统做关键词检索（BM25），每次返回最相关的 <TOP> 个原文块；查重按槽位，for 里的槽位都已经看过的块不再给出。
- finish：认为答案已经可以确定时调用；系统检查每个答案部分的算式都能算出结果，没有就返回还缺的项。
做法：
1. 第 1 次请求：只调用 update_plan，写出每个答案部分和它的需求清单（题给槽位直接填值），不用写 search。逐个部分想清楚：算它需要文档里的哪些数据或规则原文；比较、排序类的，每个被比较的主体、每个期间各列一个槽位。槽位的 what 写上主体的完整名称（如“平安智盈金生累计所交保险费”）。每个槽位都写 stem：它在题面里对应的那一句，照抄题面原文，讲哪个主体、哪种情形就摘哪一句；题给槽位摘它的数所在的那一句。系统接着按需求做第一批检索：每个要从文档里查的需求（数值、原文）用它的 stem 加上题面里没被任何槽位摘走的句子（如交代事件的开头、问句）的分词去查。
2. 之后每次：根据证据池和上一轮的结果，自己决定调用哪些工具，可以在同一条回复里调用多个：用 update_plan 填写槽位、写算式；用 search 补查；认为已经完成就调用 finish。<AUTO>
3. 证据池：已算出的部分所引用的块、还没算出的部分的槽位检索到的块、上一轮 update_plan 引用过的块、本轮检索到的块。一个槽位用到的部分都算出了，为它检索到的块就不再给出（被引用的除外）。
4. <QUERY_RULE>
写算式（update_plan 的 answers，每个部分一项）：
1. 只依据给出的原文块和题面，不用自己的知识补数据。核对主体、期间、口径（合并或母公司）和单位。
2. 按 cite、formula、reason、status 的顺序写：先写 cite（算式用到的值所在的块编号），再写 formula，再用 reason 一句话说明用了哪些原文和题给的数、怎么算，最后写 status：算式用到的槽位都有值、能算出结果，写“算得出”，系统运行算式；还算不出写 unknown。
3. 算不出的（unknown）：先写 need（还缺什么：哪个主体、哪个期间、哪项数据或哪一条规定），再写 fallback（假如最后还是找不到，按什么假设算：一个算式，缺的数可以直接写数字，也按以 1 为单位，70 万元写 700000、5% 写 0.05，在 reason 里说明假设），然后调用 search 补查，for 写它服务的槽位。要换检索词：换表名、章节名、科目名、同义说法，或换一种主体、期间的写法，不要重复已经用过的检索词。"""
AUTO_NOTE = "每次 update_plan 之后，系统发现每个答案部分都算出了结果，就直接结束作答。"
# 10-31, the user's rule: the first batch of searches uses only words of the question (BM25 matches
# words, and the question's names are the clauses' own: 累计所交保险费 found the formula that
# 解除 / 退还 missed); later searches are the model's own. v1 / v2 keep QUERY_RULE.
CALC_QUERY_RULE = ("补查的检索词（query）由你自己写，是用空格隔开的几个词，不是问句。主体写完整名称（如“平安智盈金生”“比亚迪”），"
                   "系统按它只查这个主体的文档；期间照题面写（如“2025年”“第12个保单年度”）。先照抄证据池原文里的说法"
                   "（如条款写“扣除相应的退保费用”，就用“退保费用”），还查不到再换表名、章节名、科目名或同义说法。"
                   "一条检索词只查一个主体。不要写“年度报告”“年报”“条款”这类每一页都有的词。")

SYSTEM_GUESS = """你是金融文档问答助手。下面这些答案部分检索多次后仍算不出。请结合 plan 里已有的槽位值和原文，给每个这样的部分写一个假设算式：Python，可以用槽位、中间量和已算出的答案部分的编号当变量名，缺的数按你的判断直接写数字，也按以 1 为单位（70 万元写 700000，5% 写 0.05），在 reason 里说明假设。
只输出一个 JSON：{"fallback": {"p1": "算式"}, "reason": {"p1": "一句话"}}"""


def tool_list():
    return [json.loads(fill_in(json.dumps(t, ensure_ascii=False), TOP=TOP_PER_QUERY)) for t in (UPDATE_PLAN, SEARCH, FINISH)]


# ---------------------------------------------------------------------- formulas
def check_formula(code, slots, literals_ok=False, out_unit=None, known=()):
    """(slots used, problems) for one formula: syntax, names that are neither slots, nor known
    (intermediate values and answer parts, 10-31), nor allowed; empty slots, numbers written into
    it (unless literals_ok), round(), a converted slot converted again, a ratio result times 100,
    imports."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return [], [f"语法错误：{e.msg}（第 {e.lineno} 行）"]
    nodes = list(ast.walk(tree))
    local = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
    local |= {a.arg for a in nodes if isinstance(a, ast.arg)}
    local |= {(a.asname or a.name).split(".")[0] for n in nodes if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    loads = list(dict.fromkeys(n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)))
    keys = [n.slice.value for n in nodes if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "V"
            and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)]
    # a name the formula assigns itself is its own variable, also when a slot has that id (fin_b_016: max_div)
    used = list(dict.fromkeys([x for x in loads if x in slots and x not in local] + [k for k in keys if k in slots]))
    problems = []
    known = set(known)
    unknown = [x for x in loads if x not in slots and x not in known and x not in ALLOWED and x not in local] + [k for k in keys if k not in slots]
    if unknown:
        problems.append(f"不认识的名字 {'、'.join(dict.fromkeys(unknown))}（不是槽位、中间量或答案部分的编号）")
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "round" for n in nodes):
        problems.append("算式里不要四舍五入（round）：中间量和结果都保留原始数，保留几位小数由系统按题目要求做")
    empty = [x for x in used if not slots[x]["value"]]
    if empty:
        problems.append(f"槽位 {'、'.join(empty)} 还没有值")
    if not literals_ok:
        lits = [n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)]
        bad = list(dict.fromkeys(x for x in lits if x not in CONSTS))
        if bad:
            hint = "；日期不要手写年月日，用 to_date(题给的日期槽位) 得到" if re.search(r"\bdate(time)?\s*\(", code) else ""
            problems.append(f"算式里写了数字 {'、'.join(str(x) for x in bad)}：题面的数建成题给槽位，原文的数建成数值槽位{hint}")
    # converting again what the code already converted: a slot of 1% (0.01) divided by 100, a slot
    # of 万元 times 10000 (10-31: ins_b_019 wrote rate / 100 on a slot already 0.01)
    # in any product / quotient chain (10-31: fin_b_020 wrote net_profit * dividend_ratio / 100,
    # the slot inside the chain); the factor applied again is the slot's own one (÷ 100 on a 0.01
    # slot, × 1000 on a 千元 slot); × 100 on a percent slot is left alone, it may be the output's
    # a slot in the denominator takes the factor the other way round (fin_b_017: ebitda / (rate / 100))
    again = []
    for leaves in chains(tree):
        effects = [const_effect(leaf, inv) for leaf, inv in leaves]
        for leaf, inv in leaves:
            name = leaf.id if isinstance(leaf, ast.Name) else None
            c = (slots.get(name) or {}).get("conv") if name else None
            if not c or c.get("number") is None or not c.get("converted") or c["factor"] == 1:
                continue
            target = float(c["factor"]) if not inv else 1 / float(c["factor"])
            if any(same_factor(e, target) for e in effects):
                again.append(f"{name}（已是 {c['base']:.12g}）")
    if again:
        problems.append(f"槽位 {'、'.join(dict.fromkeys(again))} 已经换算成以 1 为单位，不要再乘除换算系数")
    top = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        top = tree.body[-1].value
    elif tree.body and isinstance(tree.body[-1], ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ans" for t in tree.body[-1].targets):
        top = tree.body[-1].value
    if not literals_ok and isinstance(top, ast.Constant):   # reg_b_003 wrote ans = '2026年3月30日'
        problems.append("结果不能直接写成一个常量，要用槽位算出来（日期用 datetime 推算）")
    # (10-31) a result the formula already converted to the question's unit (× 100, ÷ 1e4 …) is no
    # longer refused: the code reads the formula and does not convert it again (formula_scaled)
    # importing datetime / math is harmless (they are given anyway); anything else is refused
    # (10-31: reg_b_003 / 004 / 018 wrote "from datetime import ..." six requests in a row)
    bad_import = any((isinstance(n, ast.Import) and any(a.name.split(".")[0] not in OK_MODULES for a in n.names))
                     or (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] not in OK_MODULES) for n in nodes)
    if bad_import or any(isinstance(n, ast.Attribute) and n.attr.startswith("__") for n in nodes):
        problems.append("算式里只能 import datetime 和 math，也不能用双下划线属性")
    text_used = [x for x in used if (slots.get(x) or {}).get("kind") == "原文"]
    if text_used:                                        # reg_b_014 multiplied a clause text by a count
        problems.append(f"{'、'.join(text_used)} 是原文槽位（文字），不能拿来算：算式要用的数另建数值槽位"
                        "（如 deduct_warning，value 0.5，unit 分）")
    if "ans" not in local:
        try:
            ast.parse(code, mode="eval")
        except SyntaxError:
            problems.append("多行算式最后要把结果赋给 ans")
    return used, problems


RUNNER = r'''
import json, math, sys
from datetime import date, datetime, timedelta
import re
data = json.loads(sys.stdin.read())
out = {}


def to_date(s):
    """2026年3月27日 / 2026-03-27 -> date."""
    m = re.search(r"(\d{4})\s*[年/.-]\s*(\d{1,2})\s*[月/.-]\s*(\d{1,2})", str(s))
    if not m:
        raise ValueError(f"认不出日期：{s}")
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


shared = dict(data["V"])          # slot values, then every result in order: later formulas use earlier ones unrounded
rescale = data.get("rescale") or {}   # part -> the output scale its formula applied itself: later formulas get base units
for pid, code in data["formulas"].items():
    ns = {"math": math, "date": date, "datetime": datetime, "timedelta": timedelta, "to_date": to_date, "V": dict(shared), **shared}
    try:
        try:
            ans = eval(compile(code, "<formula>", "eval"), ns)
        except SyntaxError:
            exec(compile(code, "<formula>", "exec"), ns)
            ans = ns["ans"]
        keep = ans
        if isinstance(ans, (date, datetime)):
            ans = f"{ans.year}年{ans.month}月{ans.day}日"
        if isinstance(ans, bool) or not isinstance(ans, (int, float, str)):
            raise TypeError(f"结果要是一个数或一个字符串，得到的是 {type(ans).__name__}")
        if isinstance(ans, float) and (math.isnan(ans) or math.isinf(ans)):
            raise ValueError("结果不是有限的数")
        base = ans
        if pid in rescale and isinstance(ans, (int, float)):   # the formula already converted to the output unit
            base = ans / rescale[pid]
            keep = base
        shared[pid] = keep
        out[pid] = {"ok": True, "ans": ans, "base": base}
    except Exception as e:
        out[pid] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
print("<<<OUT>>>" + json.dumps(out, ensure_ascii=False))
'''


def run_formulas(formulas, values, rescale=None):
    """{name: {"ok", "ans", "base" | "error"}}: the formulas run in order in one fresh interpreter
    with the slot values; each result is passed on to the later ones (divided back by its scale
    first when the formula converted to the output unit itself: rescale)."""
    if not formulas:
        return {}
    try:
        p = subprocess.run([sys.executable, "-c", RUNNER], input=json.dumps({"formulas": formulas, "V": values, "rescale": rescale or {}}, ensure_ascii=False),
                           capture_output=True, text=True, encoding="utf-8", timeout=RUN_TIMEOUT,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    except subprocess.TimeoutExpired:
        return {k: {"ok": False, "error": "算式运行超时"} for k in formulas}
    if "<<<OUT>>>" not in p.stdout:
        err = (p.stderr or p.stdout)[-500:]
        return {k: {"ok": False, "error": err} for k in formulas}
    return json.loads(p.stdout.split("<<<OUT>>>")[-1])


SEG_SPLIT = r"[；;。？?：:]"
FILLER = {"万元", "多少", "合计", "四份", "合同", "分别", "可", "答案", "只", "填写数字", "某人", "持有", "若", "则", "为", "是"}


def segment_queries(stem, named):
    """The first search batch, built by the code (10-31, the user's rule): the stem cut at ；：。？;
    for every subject the question names (named: doc -> short name), the subject + the words of the
    pieces that mention it + the words of every piece that names no subject (the opening, the
    question asked); numbers and filler words dropped. A stem that names no subject gives one query
    of all its words. [(query, subject or "")]; tested on the 11 rule questions in
    reports/calc_query_tokens.md."""
    # annual-report short names carry the year ("宁德时代 2025"): match the name without it (10-31,
    # fin_b_019 found only one of its three companies)
    names = sorted({re.sub(r"\s*\d{4}$", "", n) for n in named.values()} - {""}, key=len, reverse=True)
    names = [n for n in names if len(n) >= 3 and n in stem]
    pieces = [p for p in re.split(SEG_SPLIT, stem) if p.strip()]
    common = [p for p in pieces if not any(n in p for n in names)]

    def words(text):
        return [w for w in dict.fromkeys(tokenize(text)) if not re.fullmatch(r"[\d.]+%?", w) and w not in FILLER]

    if not names:
        return [(" ".join(words(stem)), "")]
    out = []
    for n in names:
        own = [p for p in pieces if n in p]
        text = " ".join(p.replace(n, " ") for p in own) + " " + " ".join(common)
        out.append((n + " " + " ".join(words(text)), n))
    return out


MAG = r"(万亿|千亿|百亿|十亿|亿|千万|百万|十万|万|千)?(元|美元|港元|人|股|辆|户|家|笔|吨)"
PCT_WORDS = r"百分点|百分数|%|％|增速|增幅|增长率|增值率|毛利率|净利率|收益率|占比|比重|比例|偏差|降幅|涨幅"


def unit_of_text(t, whole, strict=False):
    """(scale applied to a base-unit result, % sign) a piece of the question asks for, "text" for
    a ranking or a date, None when it says nothing. strict (the ask of a question whose parts have
    names): only units written out ("多少个百分点", "以亿元计"), since the ask covers every part."""
    if strict:
        if re.search(r"多少\s*个?\s*百分点", t):
            return (100, "")
        m = re.search(r"以\s*" + MAG + r"\s*计|单位[:：]\s*" + MAG + r"|多少\s*" + MAG, t)
        if m:
            f, _, ok = factor_of("".join(x for x in m.groups() if x))
            return (1 / float(f), "") if ok else (1, "")
        return None
    if re.search(r">|排序|哪一天|何时|日期|哪天", t):
        return "text"
    m = re.search(r"以\s*" + MAG + r"\s*计|单位[:：]\s*" + MAG + r"|多少\s*" + MAG + r"|为多少\s*(万亿|亿|千万|百万|万|千)\s*(元|人)", t)
    if m:
        f, _, ok = factor_of("".join(x for x in m.groups() if x))
        return (1 / float(f), "") if ok else (1, "")
    if re.search(r"百分点", t):
        return (100, "")
    if re.search(r"以\s*百分数计|百分比|%|％", t) or re.search(PCT_WORDS, t):
        return (100, "" if NO_SIGN.search(whole) or re.search(r"不带\s*(%|％|单位)", t) else "%")
    return None


def stem_units(question, part_list):
    """{part id: (scale, sign) | None (text part)}: every answer part's output unit read from the
    question (10-31, instead of the unit the model declares): the part's own name, then the 前者 /
    后者 / 均 clauses after 答案格式为“…”, then the ask. All 40 parts of the 26 B fill-in questions
    right (in-sample, reports/calc_rounds_10x3.md section 8)."""
    q = re.sub(r"\s+", "", question)
    fmt = re.search(r"答案格式为[“\"][^”\"]*[”\"](.*?)(?:。|$)", q)
    rule = fmt.group(1) if fmt else ""
    clauses = {"前者": "", "后者": "", "均": ""}
    for c in re.split(r"[，,；;]", rule):
        for k in clauses:
            if c.startswith(k) or (k == "均" and "均" in c):
                clauses[k] += c
    sents = re.split(r"(?<=[。？?])", q)
    ask = next((s for s in reversed(sents) if re.search(r"多少|何时|哪一天|几笔|几|哪", s)), None) \
        or next((s for s in reversed(sents) if re.search(r"计算", s)), q)
    named = [p for p in part_list if p.get("what") and p["what"] != "题目所求的结果" and p.get("fixed")]
    strict = len(part_list) > 1 and bool(named)
    out = {}
    for i, p in enumerate(part_list):
        name = p.get("what") if p.get("fixed") and p.get("what") != "题目所求的结果" else ""
        own = clauses["前者"] if i == 0 and len(part_list) == 2 else (clauses["后者"] if i == 1 and len(part_list) == 2 else "")
        u = None
        for piece in (name, own, clauses["均"]):
            if piece:
                u = unit_of_text(piece, q)
                if u:
                    break
        if u is None:
            u = unit_of_text(ask, q, strict=strict) or (1, "")
        out[p["id"]] = None if u == "text" else u
    return out


def unit_label(u):
    """How a stem unit reads in the plan view."""
    if u is None:
        return "文字（排序、日期）"
    scale, sign = u
    if scale == 100:
        return "%" if sign else "百分数 / 个百分点（不带 %）"
    names = {1e-4: "万", 1e-8: "亿", 1e-6: "百万", 1e-3: "千", 1e-12: "万亿"}
    return next((v for k, v in names.items() if abs(scale - k) < 1e-15), "原数")


def chain_leaves(node, inv=False):
    """The leaves of a product / quotient chain with their position: [(leaf node, in the
    denominator)]. Walks Mult and Div only (a / (b * 1e4) puts 1e4 in the denominator)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        return chain_leaves(node.left, inv) + chain_leaves(node.right, inv != isinstance(node.op, ast.Div))
    return [(node, inv)]


def chain_nodes(node):
    """The Mult / Div nodes that make up one chain (not those inside its leaves)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        return [node] + chain_nodes(node.left) + chain_nodes(node.right)
    return []


def chains(tree):
    """Every maximal product / quotient chain of a formula, as its leaf list; a chain inside a leaf
    of another (rate / 100 in acct * (1 - rate / 100)) is a chain of its own."""
    out, inner = [], set()
    for n in ast.walk(tree):
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Div)) and id(n) not in inner:
            inner.update(id(m) for m in chain_nodes(n))
            out.append(chain_leaves(n))
    return out


def const_effect(leaf, inv):
    """The factor a constant leaf applies (its value, or 1 / value in a denominator), else None."""
    if isinstance(leaf, ast.Constant) and isinstance(leaf.value, (int, float)) and not isinstance(leaf.value, bool) and leaf.value:
        v = float(leaf.value)
        return 1 / v if inv else v
    return None


def same_factor(a, b):
    return a is not None and b is not None and abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1e-300)


def formula_scaled(code, scale):
    """True when the formula itself applies the output scale: × 100 for a percent, ÷ 1e4 / ÷ 1e8 /
    ÷ 1e6 (or × their inverse) for 万 / 亿 / 百万, anywhere in it, numerator or denominator (10-31:
    fin_b_015 wrote × 100 in a middle line, res_b_012 round(x / 10000, 1) and x / (y * 1e4); the
    code then converted a second time)."""
    if not code or not scale or scale == 1:
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(same_factor(const_effect(leaf, inv), scale) for leaves in chains(tree) for leaf, inv in leaves)


def calc_value_text(v):
    """An unrounded result for the plan view: up to 15 significant digits, text as it is."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:.15g}"
    return str(v)


def squeeze(text):
    return re.sub(r"\s+", "", str(text or ""))


def in_question(quote, stem):
    """True when a quoted sentence is a piece of the question (spaces ignored, trailing punctuation
    of the quote allowed)."""
    q = squeeze(quote).rstrip("，,。；;？?：:")
    return len(q) >= 4 and q in squeeze(stem)


def need_queries(stem, needs, named, claimed=()):
    """The first search batch by need (10-31, the user's rule "需求对应的那句话的分词 + 总结句"): for every
    need to look up (slot id, its description, its quoted sentences of the question) one query = the
    subject names it is about + the words of its own sentences + the words of every piece of the
    question that no slot quoted, 题给 slots included (`claimed`: their quotes); that leaves the
    summary pieces: the opening that states the event, the question asked. Numbers and filler words
    dropped. [(query, [slot ids])], identical queries merged. Needs without a quote are left out.
    (reg_b_014: its opening "基准分为100分" and condition "两项…各1次" are 题给 sentences and pulled both
    clauses to rank 4-7; ins_b_019's opening "犹豫期后退保或解除" quotes no number and is what finds
    the 退还 clauses.)"""
    pieces = [p for p in re.split(SEG_SPLIT, stem) if p.strip()]
    quoted = [squeeze(t).rstrip("，,。；;？?：:") for _, _, qs in needs for t in qs]
    quoted += [squeeze(t).rstrip("，,。；;？?：:") for t in claimed]
    common = [p for p in pieces if not any(t in squeeze(p) or squeeze(p) in t for t in quoted)]
    names = sorted({re.sub(r"\s*\d{4}$", "", n) for n in named.values()} - {""}, key=len, reverse=True)
    out = {}
    for sid, what, qs in needs:
        if not qs:
            continue
        own = " ".join(qs)
        subj = [n for n in names if len(n) >= 3 and n in (what or "")] or [n for n in names if len(n) >= 3 and n in own]
        subj = [n for n in subj if not any(n != m and n in m for m in subj)]       # 智盈金生 inside 平安智盈金生
        text = own + " " + " ".join(common)
        for n in subj:                                   # the subject whole in front; its jieba fragments are no use
            text = text.replace(n, " ")
        words = [w for w in dict.fromkeys(tokenize(text)) if not re.fullmatch(r"[\d.]+%?", w) and w not in FILLER]
        query = " ".join(subj + words)
        out.setdefault(query, []).append(sid)
    return list(out.items())


def digits_in(text):
    return re.sub(r"(?<=\d)[,，](?=\d)", "", re.sub(r"\s+", "", text))


def in_stem(value, stem):
    """True when a 题给 value can be found in the question: its number (thousands separators
    dropped) or, for text, the text itself."""
    v = digits_in(str(value))
    s = digits_in(stem)
    nums = re.findall(r"\d+(?:\.\d+)?", v)
    if len(nums) == 1 and re.fullmatch(r"[-+−–]?" + re.escape(nums[0]) + r"\D{0,8}", v):
        return re.search(r"(?<![\d.])" + re.escape(nums[0]) + r"(?![\d])", s) is not None
    return v in s


# ---------------------------------------------------------------------- one question
OUT_TAG = ""                # --tag=r1 -> output/calc_rounds/r1/ (repeated runs side by side)


def main(argv):
    global OUT_TAG
    qids = [a for a in argv if not a.startswith("--")]
    auto = "--no-auto-finish" not in argv
    OUT_TAG = next((a.split("=", 1)[1] for a in argv if a.startswith("--tag=")), "")
    qs = {x["qid"]: x for x in load_b_questions()}
    for qid in qids:
        solve(qs[qid], auto_finish=auto)


def solve(q, spent=None, auto_finish=None):
    auto = AUTO_FINISH if auto_finish is None else auto_finish
    qid, kind = q["qid"], q["type"]
    if q.get("options"):
        raise ValueError(f"{qid} 是{kind}，这个策略只做填空计算题")
    question = q["question"].strip()
    fixed = answer_parts(question)
    parts = {}                                          # id -> {id, what, unit, slots, format, fixed, request}
    for p in fixed or []:
        parts[p["id"]] = {"id": p["id"], "what": p["what"], "unit": "", "slots": [], "format": p.get("format") or "", "fixed": True, "request": 0}
    sc = scope(q)
    title = {c["doc_id"]: c["title"] for c in cards()}
    llm = Qwen(log_path=ROOT / "logs" / f"{NAME}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    usage = spent if spent is not None else [0, 0]      # the caller's list keeps the count when the run fails
    t0 = time.time()
    seen = defaultdict(Seen)                            # need -> the chunks seen for it
    store, used = {}, {}                                # store: label -> chunk shown; used: queries already run, in order
    serves = defaultdict(set)                           # label -> needs it was retrieved for
    asked_for = defaultdict(set)                        # label -> names the search calls wrote in for
    slots, answers = {}, {}                             # the plan; answers: part -> latest answer
    searched = set()                                    # parts / slots / needs some search ran for
    calcs = {}                                          # intermediate values: id -> {id, what, formula, result, request}
    history, requests, calls_log = [], [], []

    many = len(sc["docs"]) > 20
    docs = "\n".join(f"- 《{title.get(d, d)}》" for d in sc["docs"][:20]) + (f"\n（共 {len(sc['docs'])} 份，题面没有点名文档，检索在这些文档里进行）" if many else "")
    if fixed:
        plist = "\n".join(f"- {p['id']}：{p['what']}" + (f"（格式：{p['format']}）" if p.get("format") else "") for p in fixed)
        parts_note = fill_in(PARTS_FIXED, LIST=plist)
    else:
        parts_note = PARTS_OPEN
    system = fill_in(SYSTEM, AUTO=AUTO_NOTE if auto else "", QUERY_RULE=CALC_QUERY_RULE, PARTS=parts_note)
    system = fill_in(system, TOP=TOP_PER_QUERY, MAX=MAX_SEARCHES, DOCS=docs, CONSTS=CONST_WORDS)
    tools = tool_list()
    print(f"{qid}（{kind}）{question}\n答案部分：{'、'.join(parts) if parts else '由模型列出'}\n")

    # -------------------------------------------------------------- model call (v2's)
    def chat(user, tag):
        for attempt in range(6):                        # 10-31: a network blip killed most of 60 parallel runs with 3 x 5 s
            try:
                t1 = time.time()
                resp = llm.client.chat.completions.create(
                    model=llm.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0, max_tokens=MAX_TOKENS, tools=tools,
                    extra_body={"enable_thinking": False, "parallel_tool_calls": True})
                break
            except Exception as e:                      # noqa: BLE001 - transient API errors
                if attempt == 5:
                    raise
                print(f"API 出错，{5 * 2 ** attempt} 秒后重试：{e}")
                time.sleep(5 * 2 ** attempt)
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

    # -------------------------------------------------------------- values and formulas
    def values():
        """The formula's variables: numbers in base units for 数值 / 题给 slots holding one number,
        the text as it is otherwise; empty slots are left out."""
        out = {}
        for sid, s in slots.items():
            if not s["value"]:
                continue
            c = s.get("conv")
            out[sid] = c["base"] if c and c["number"] is not None and s["kind"] != "原文" else s["value"]
        return out

    def evaluate(keys_to_run=None, literals_ok=False, field="formula"):
        """Check and run the formulas of the intermediate values (calcs) and of these parts, in the
        order their names depend on each other, in one interpreter: a later formula sees earlier
        results unrounded (10-31, the user's rule). {part: result}; a result is {"ok", "ans",
        "text"} or {"ok": False, "error"}; every calc gets its own result as well. Parts not run
        here that already have a result pass it on as a value (the fallback round uses this)."""
        part_codes = {}
        for pid in keys_to_run if keys_to_run is not None else list(parts):
            code = ((answers.get(pid) or {}).get(field) or "").strip()
            if code:
                part_codes[pid] = code
        calc_codes = {cid: c["formula"] for cid, c in calcs.items() if (c.get("formula") or "").strip()}
        extra = {pid: answers[pid]["result"].get("base", answers[pid]["result"]["ans"]) for pid in parts
                 if pid not in part_codes and ((answers.get(pid) or {}).get("result") or {}).get("ok")}
        known = set(calc_codes) | set(parts)
        items = {**{("calc", c): code for c, code in calc_codes.items()}, **{("part", p): code for p, code in part_codes.items()}}
        problems, deps = {}, {}
        for (kind, name), code in items.items():
            _, pr = check_formula(code, slots, literals_ok and kind == "part", parts[name]["unit"] if kind == "part" else None, known)
            if pr:
                problems[(kind, name)] = "；".join(pr)
            try:
                used = {n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name)}
            except SyntaxError:
                used = set()
            deps[(kind, name)] = {("calc", u) if u in calc_codes else ("part", u) for u in used if u in known and u != name}
        # order: a formula after every formula it names; one that names something unavailable fails with that name
        order, done, failed = [], set(), dict(problems)
        pending = [k for k in items if k not in failed]
        while pending:
            progress = False
            for k in list(pending):
                ds = deps[k]
                missing = [d for d in ds if d not in done and not (d[0] == "part" and d[1] in extra)]
                if any(d in failed or (d not in items and not (d[0] == "part" and d[1] in extra)) for d in missing):
                    bad = [d[1] for d in missing if d in failed or d not in items]
                    failed[k] = f"用到的 {'、'.join(bad)} 还没有算出"
                    pending.remove(k)
                    progress = True
                elif not missing:
                    order.append(k)
                    done.add(k)
                    pending.remove(k)
                    progress = True
            if not progress:                            # a cycle
                for k in pending:
                    failed[k] = "算式互相引用，形成了循环"
                break
        # output units from the question; a part whose formula (or a calc it uses) already applies its
        # output scale is divided back to base units, so the code does not convert it twice
        su = stem_units(question, list(parts.values()))

        def calc_chain(k, seen_=None):
            seen_ = seen_ or set()
            codes = [items.get(k, "")]
            for d in deps.get(k, ()):
                if d[0] == "calc" and d not in seen_:
                    seen_.add(d)
                    codes += calc_chain(d, seen_)
            return codes

        rescale = {}
        for k in order:
            if k[0] == "part" and su.get(k[1]):
                scale = su[k[1]][0]
                if scale != 1 and formula_scaled("\n".join(calc_chain(k)), scale):
                    rescale[k[1]] = scale
        ran = run_formulas({k[1]: items[k] for k in order}, {**values(), **extra}, rescale) if order else {}
        res = {}
        for k in items:
            kind, name = k
            r = ran.get(name) if k in done else {"ok": False, "error": failed.get(k, "没有运行")}
            if kind == "calc":
                calcs[name]["result"] = r
                continue
            if r["ok"]:
                u = su.get(name)
                r["unit"] = unit_label(u)
                r["self_scaled"] = name in rescale
                if isinstance(r["ans"], str) or u is None:
                    r["text"] = render(r["ans"], "", question, parts[name].get("format", ""))
                else:
                    r["text"] = render_scaled(r.get("base", r["ans"]), u[0], u[1], question, parts[name].get("format", ""))
            res[name] = r
        return res

    def checked(k):
        """A part counts as done only after some search ran for it, for one of its slots or for a
        slot its formula uses; a part on given numbers only needs one search in the question
        (10-31: ins_b_019 took every number as given and never searched the clauses; fin_b_016's
        差额 part used the slots of the ranking part)."""
        if not searched:
            return False
        code = (answers.get(k) or {}).get("formula") or ""
        names = set(parts[k]["slots"]) | set(check_formula(code, slots)[0] if code.strip() else [])
        if any(k in parts_of(nd) for nd in searched) or names & searched:
            return True
        return not any((slots.get(s) or {}).get("kind") in ("数值", "原文") for s in names)

    def concluded(k):
        a = answers.get(k) or {}
        return a.get("status") == READY and bool((a.get("result") or {}).get("ok")) and checked(k)

    def determined():
        return bool(parts) and all(concluded(k) for k in parts)

    def needs_of(names):
        """The needs a search serves: a slot as it is, a part by its slots; a name that leads to no
        slot is a need of its own ("*" when for is empty)."""
        out = []
        for x in names:
            if x in slots:
                out.append(x)
            elif x in parts and parts[x]["slots"]:
                out += parts[x]["slots"]
            else:
                out.append(x)
        return list(dict.fromkeys(out)) or ["*"]

    def parts_of(nd):
        if nd in parts:
            return {nd}
        users = {pid for pid, p in parts.items() if nd in p["slots"]}
        return users or set(parts)

    def need_open(nd):
        """A need is done when every part using it has a result; a filled value is not enough."""
        ps = parts_of(nd)
        return not ps or not all(concluded(k) for k in ps)

    def missing():
        if not parts:
            return "还没有列出答案部分"
        out = []
        for k, p in parts.items():
            if concluded(k):
                continue
            a = answers.get(k)
            empty = [s for s in p["slots"] if not (slots.get(s) or {}).get("value")]
            res = (a or {}).get("result") or {}
            if empty:
                out.append(f"{k} 的 {'、'.join(empty)} 仍然缺失")
            elif not checked(k):
                out.append(f"{k} 还没有检索过原文")
            elif a and a.get("status") == READY and res.get("error"):
                out.append(f"{k} 的算式没有通过：{res['error']}")
            elif a:
                out.append(f"{k} 为 unknown")
            else:
                out.append(f"{k} 还没有算式")
        return "；".join(out)

    def apply_update(a, n, cited):
        """Write one update_plan call into the plan and run the formulas; (summary, detail)."""
        notes, new_parts, new_slots, filled, bad = [], [], [], [], []
        fields = {}
        for f, read in (("parts", as_list), ("slots", as_list), ("calc", as_list), ("answers", as_dict)):
            fields[f] = read(a.get(f))
            if fields[f] is None:                       # say so instead of dropping it silently
                notes.append(f"{f} 不是合法 JSON，没有收")
                fields[f] = [] if read is as_list else {}
        for x in fields["parts"]:
            if not isinstance(x, dict) or not text_of(x.get("id")):
                continue
            pid = text_of(x["id"]).lower()
            if pid not in parts:
                if fixed or not re.fullmatch(r"p\d+", pid):
                    notes.append(f"部分 {pid} 不是本题的答案部分，没有收" if fixed else f"部分编号 {pid} 要写成 p1、p2…，没有收")
                    continue
                parts[pid] = {"id": pid, "what": "", "unit": "", "slots": [], "format": "", "fixed": False, "request": n}
                new_parts.append(pid)
            p = parts[pid]
            if "what" in x and not p["fixed"]:
                p["what"] = text_of(x["what"])
            if "unit" in x:
                p["unit"] = text_of(x["unit"])
            if "slots" in x:
                p["slots"] = list(dict.fromkeys(text_of(s) for s in as_list(x["slots"]) or [] if text_of(s)))
        for x in fields["slots"]:
            if not isinstance(x, dict) or not text_of(x.get("id")):
                continue
            sid = text_of(x["id"])
            if not sid.isidentifier() or sid in ALLOWED or sid in parts or sid.startswith("_"):
                notes.append(f"槽位编号 {sid} 不能当变量名（或和部分编号、内置名字重名），没有收，换一个")
                continue
            s = dict(slots.get(sid) or {"id": sid, "what": "", "kind": "", "value": "", "unit": "", "cite": [], "where": "", "stem": [],
                                        "conv": None, "request": n})
            had = bool(s["value"])
            for f in ("what", "kind", "value", "unit", "where"):
                if f in x:
                    s[f] = text_of(x[f])
            if "stem" in x:                             # the need's sentence(s) of the question, verbatim
                quotes = [text_of(t) for t in (as_list(x["stem"]) if not isinstance(x["stem"], str) or x["stem"].strip().startswith("[")
                                                else [x["stem"]]) or [] if text_of(t)]
                ok = [t for t in quotes if in_question(t, question)]
                if len(ok) < len(quotes):
                    notes.append(f"槽位 {sid} 的 stem 有 {len(quotes) - len(ok)} 句不是题面原文，没有收")
                s["stem"] = ok
            if s["kind"] not in KINDS:
                s["kind"] = "数值" if s["kind"] in ("数", "数字") else (s["kind"] or "数值")
            if s["value"] and blank(s["value"]):        # “缺失”, “未找到”: not a value
                notes.append(f"槽位 {sid} 的值“{s['value']}”按未填处理")
                s["value"] = ""
            if s["value"] and s["kind"] == "题给" and not in_stem(s["value"], question):
                notes.append(f"题给槽位 {sid} 的值“{s['value']}”在题面里找不到，没有收：题给槽位只写题面里原样出现的一个数或日期，"
                             "要算的拆成几个题给槽位写进算式，文档里的数改成数值槽位")
                s["value"] = ""
            if "cite" in x:
                labs = labels_in(x["cite"])
                bad += [lab for lab in labs if lab not in store]
                s["cite"] = [lab for lab in labs if lab in store]
                cited.update(s["cite"])
            s["conv"] = to_base(s["value"], s["unit"]) if s["value"] and s["kind"] != "原文" else None
            if s["value"] and not had:
                filled.append(sid)
            if sid not in slots:
                new_slots.append(sid)
            slots[sid] = s
        new_calcs = []
        for x in fields["calc"]:                        # intermediate values: formulas only, the code keeps the results
            if not isinstance(x, dict) or not text_of(x.get("id")):
                continue
            cid = text_of(x["id"])
            if not cid.isidentifier() or cid in ALLOWED or cid in parts or cid in slots or cid.startswith("_"):
                notes.append(f"中间量编号 {cid} 不能当变量名（或和槽位、部分编号、内置名字重名），没有收，换一个")
                continue
            c = dict(calcs.get(cid) or {"id": cid, "what": "", "formula": "", "result": None, "request": n})
            if "what" in x:
                c["what"] = text_of(x["what"])
            if "formula" in x:
                c["formula"] = text_of(x["formula"])
            if cid not in calcs:
                new_calcs.append(cid)
            calcs[cid] = c
        got = []
        ans = fields["answers"]
        for k, v in (ans or {}).items():
            pid = str(k).strip().lower()
            if pid not in parts or not isinstance(v, dict):
                notes.append(f"answers 里的 {k} 不是本题的答案部分，没有收")
                continue
            labs = labels_in(v.get("cite"))
            bad += [lab for lab in labs if lab not in store]
            status = READY if text_of(v.get("status")) in (READY, "可计算", "能算出", "ready", "完成") else UNKNOWN
            rec = {"cite": [lab for lab in labs if lab in store], "formula": text_of(v.get("formula")),
                   "reason": text_of(v.get("reason")), "status": status,
                   "need": text_of(v.get("need")) if status == UNKNOWN else "",
                   "fallback": text_of(v.get("fallback")) if status == UNKNOWN else "", "round": n, "result": None}
            answers[pid] = rec
            cited.update(rec["cite"])
            got.append(pid)
        # run every formula marked 算得出: slot values may have changed since it was written
        results = evaluate([k for k, v in answers.items() if v["status"] == READY])
        for k, r in results.items():
            answers[k]["result"] = r
        for k in got:
            history.append({"part": k, **{f: answers[k][f] for f in ("cite", "formula", "reason", "status", "need", "fallback", "round")},
                            "result": answers[k]["result"]})
        if bad:
            notes.append(f"引用里不认识的块编号 {'、'.join(dict.fromkeys(bad))} 已去掉")
        conv_notes = [f"{sid} {s['conv']['note']}" for sid, s in slots.items()            # text values (dates) are fine as text
                      if sid in filled and s.get("conv") and s["conv"]["note"] and s["conv"]["number"] is not None]
        bits = []
        if new_parts or new_slots:
            bits.append(f"新建 {len(new_parts)} 个部分、{len(new_slots)} 个槽位")
        for cid in [x.get("id") for x in fields["calc"] if isinstance(x, dict)]:
            c = calcs.get(text_of(cid))
            if c:
                r = c.get("result") or {}
                bits.append(f"中间量 {c['id']} = {calc_value_text(r.get('ans'))}" if r.get("ok") else f"中间量 {c['id']} 没有算出：{r.get('error', '还没有运行')}")
        if filled:
            bits.append(f"填 {len(filled)} 个槽位（{'、'.join(filled)}）")
        for k in got:
            r = answers[k]["result"]
            if answers[k]["status"] != READY:
                bits.append(f"{k} unknown")
            elif r and r["ok"] and not checked(k):
                bits.append(f"{k} 算出 {r['text']}，但还没有为它检索过原文，不算完成：每个答案部分至少检索一次，"
                            "核对题面的数在条款、原文里怎么用")
            elif r and r["ok"]:
                bits.append(f"{k} 算出 {r['text']}")
            elif r:
                bits.append(f"{k} 的算式没有通过：{r['error']}")
        for k, r in results.items():                    # an earlier formula whose outcome changed with this call's slots
            if k not in got:
                bits.append(f"{k} 重新运行：" + (f"算出 {r['text']}" if r["ok"] else f"没有通过（{r['error']}）"))
        summary = "update_plan（" + "；".join(bits + conv_notes + notes or ["没有改动"]) + "）"
        return summary, {"new_parts": new_parts, "new_slots": new_slots, "filled": filled, "answers": got,
                         "results": results, "notes": notes + conv_notes}

    # -------------------------------------------------------------- search (v2's)
    def admit(needs, u):
        fresh, k = [], None
        for nd in needs:
            f, k = seen[nd].admit(u)
            if f:
                fresh.append(nd)
        return fresh, k

    def run_searches(batch, n, pool, budget):
        results, skipped, dropped_all = [], [], []
        for c in batch:
            a = c["args"]
            query = norm_query(a.get("query"))
            raw, dropped = query, []
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
            results.append({"query": query, "raw_query": raw, "dropped": dropped, "for": target, "needs": needs_of(target),
                            "searched": text.strip(), "docs": len(where), "hits": hits, "status": {}})
            searched.update(target + results[-1]["needs"])
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
                    got.append(lab)
                    r["status"][u] = (status, lab)

                if u in label_of:
                    lab = label_of[u]
                    if lab in pool or store[lab]["request"] == n:
                        admit(needs, u)
                        take(lab, "shown" if lab in pool else "new")
                        continue
                    if budget["new"] >= MAX_NEW:
                        r["status"][u] = ("cut", None)
                        continue
                    fresh, _ = admit(needs, u)
                    if not fresh:
                        r["status"][u] = ("seen", None)
                        continue
                    budget["new"] += 1
                    take(lab, "back")
                    continue
                if budget["new"] >= MAX_NEW:
                    r["status"][u] = ("cut", None)
                    continue
                fresh, k = admit(needs, u)
                if not fresh:
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

    def first_batch():
        """The code's first search calls: one per need from its quoted sentence of the question +
        the pieces no need quoted (need_queries); needs to look up without a quote fall back to
        segment_queries (by subject), which also serves when no need quoted anything."""
        named = sc.get("named") or {}
        lookups = [(sid, s["what"], s.get("stem") or []) for sid, s in slots.items() if s["kind"] in ("数值", "原文")]
        given = [t for s in slots.values() if s["kind"] == "题给" for t in s.get("stem") or []]
        calls = [{"name": "search", "args": {"query": query, "for": sids}, "how": "需求"}
                 for query, sids in need_queries(question, lookups, named, given)]
        unquoted = [sid for sid, _, qs in lookups if not qs]
        if unquoted or not calls:
            for query, subj in segment_queries(question, named):
                target = [sid for sid in unquoted if subj and subj in (slots[sid]["what"] or "")] or unquoted or list(parts)
                calls.append({"name": "search", "args": {"query": query, "for": target}, "how": "主体"})
        return calls[:MAX_SEARCHES]

    # -------------------------------------------------------------- context
    def slot_value_text(s):
        if not s["value"]:
            return "（未填）", ""
        orig = f"{s['value']} {s['unit']}".strip()
        c = s.get("conv")
        if s["kind"] == "原文" or not c:
            return orig, "（原文）"
        if c["number"] is None:
            return orig, "（文字，原样给出）"
        base = c["base"]
        shown = f"{base:.12g}" + (f"（{c['base_unit']}）" if c.get("base_unit") else "")
        if not c["converted"]:
            shown += f"（单位 {c['unit']} 未换算）"
        return orig, shown

    def plan_view():
        if not parts and not slots and not answers:
            return "还没有，用 update_plan 建立。"
        L = []
        su = stem_units(question, list(parts.values()))
        for k, p in parts.items():
            a = answers.get(k)
            head = f"{k}（{p['what'] or '未写'}｜输出单位（系统按题面定）{unit_label(su.get(k))}）"
            if a:
                r = a.get("result") or {}
                if a["status"] == READY and r.get("ok"):
                    raw = "" if isinstance(r.get("ans"), str) else \
                        f"（以 1 为单位的原始值 {calc_value_text(r.get('base', r['ans']))}，别的算式用 {k} 引用的就是它）"
                    head += f"：算出 {r['text']}{raw}" + ("" if checked(k) else "（还没有检索过原文，不算完成）")
                elif a["status"] == READY:
                    head += f"：算式没有通过（{r.get('error', '')}）"
                else:
                    head += "：unknown"
                if a["formula"]:
                    head += f"｜算式 {a['formula']}"
                head += f"｜依据 {'、'.join(a['cite']) or '无'}"
                if a["reason"]:
                    head += f"｜理由：{a['reason']}"
                if a["status"] == UNKNOWN and a["need"]:
                    head += f"｜还缺：{a['need']}"
                if a["status"] == UNKNOWN and a["fallback"]:
                    head += f"｜假设算式 {a['fallback']}"
            else:
                head += "：还没有算式"
            L.append(head)
            L.append(f"  需求 {'、'.join(p['slots']) or '无'}")
        if calcs:
            L += ["", "中间量（系统运行算式得到的原始值，没有四舍五入；算式里用编号引用）：", "| 中间量 | 是什么 | 算式 | 值 |", "|---|---|---|---|"]
            for c in calcs.values():
                r = c.get("result") or {}
                val = calc_value_text(r["ans"]) if r.get("ok") else f"（没有算出：{r.get('error', '还没有运行')}）"
                cell2 = lambda t: str(t or "").replace("|", "/").replace("\n", " ⏎ ")   # noqa: E731
                L.append(f"| {c['id']} | {cell2(c['what'])} | {cell2(c['formula'])} | {cell2(val)} |")
        if slots:
            L += ["", "需求清单（槽位；算式里的变量就是“算式里的值”这一列，已经以 1 为单位，比率已是小数）：","| 槽位 | 内容 | 类型 | 原文或题面的值 | 算式里的值 | 依据 | 出处 |", "|---|---|---|---|---|---|---|"]
            for s in slots.values():
                orig, shown = slot_value_text(s)
                cell = lambda t: str(t).replace("|", "/").replace("\n", " ")   # noqa: E731
                L.append(f"| {s['id']} | {cell(s['what'])} | {s['kind']} | {cell(orig)} | {cell(shown)} | {'、'.join(s['cite'])} | {cell(s['where'])} |")
        return "\n".join(L)

    def evidence_of(lab):
        return [k for k in parts if concluded(k) and lab in answers[k]["cite"]]

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

    def context(pool, pparts, tagged, summary):
        L = [f"题目：\n{question}", "", "当前 plan：", plan_view(), ""]
        if pool:
            L += [f"证据池（已算出部分的依据 {pparts['evidence']} 块 + 还没完成的槽位的块 {pparts['open']} 块 + 本轮新给出的 {pparts['new']} 块）：", ""]
            L += [show_chunk(lab, tagged) + "\n" for lab in pool]
        else:
            L += ["证据池：（没有原文块）", ""]
        if used:
            L.append(f"已经用过的检索词：{'；'.join(used)}")
        if summary:
            L.append(f"上一轮：{summary}")
        return "\n".join(L).rstrip()

    def next_pool_of(pool, cited, got):
        keep = {lab for lab in store if evidence_of(lab)}
        keep |= {lab for lab, nds in serves.items() if any(need_open(nd) for nd in nds)}
        keep |= set(cited) | set(got)
        labs = sorted(keep, key=lambda s: int(s[1:]))
        fresh = {lab for lab in got if lab not in pool}
        pp = {"new": len(fresh), "evidence": sum(1 for lab in labs if lab not in fresh and evidence_of(lab))}
        pp["open"] = len(labs) - pp["new"] - pp["evidence"]
        return labs, pp

    # -------------------------------------------------------------- main loop (v2's)
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
        model_first = []
        if n == 1:                                      # update_plan first; the first search batch is the code's (below)
            model_first = [c for c in calls if c["name"] == "search"]
            order = [c for c in calls if c["name"] == "update_plan"] + [c for c in calls if c["name"] not in ("update_plan", "search")]
        cited, got = set(), []
        bits, executed = [], []
        budget = {"searches": 0, "new": 0}
        ended = None
        i = 0
        while i < len(order):
            c = order[i]
            if "error" in c:
                bits.append(f"{c['name']}：参数不是合法 JSON，没有执行")
                executed.append({"name": c["name"], "result": bits[-1], "error": c["error"]})
                i += 1
                continue
            if c["name"] == "search":                   # a run of search calls is one batch
                batch = []
                while i < len(order) and order[i]["name"] == "search" and "error" not in order[i]:
                    batch.append(order[i])
                    i += 1
                results, s, labs, new, skipped = run_searches(batch, n, set(pool), budget)
                got += labs
                bits.append(s)
                executed.append({"name": "search", "calls": [b["args"] for b in batch], "result": s, "results": results, "new": new, "skipped": skipped})
                print(s)
                continue
            i += 1
            if c["name"] == "update_plan":
                try:
                    s, detail = apply_update(c["args"], n, cited)
                except Exception as e:                  # noqa: BLE001 - a failing call answers with its error
                    s, detail = f"update_plan：执行出错（{type(e).__name__}: {e}）"[:300], {"error": traceback.format_exc()[-1500:]}
                bits.append(s)
                executed.append({"name": "update_plan", "args": c["args"], "result": s, "detail": detail})
                print(s)
                if auto and determined():
                    ended = "护栏：update_plan 之后每个答案部分都算出了结果"
            elif c["name"] == "finish":
                if determined():
                    ended = "finish 通过"
                    executed.append({"name": "finish", "result": "通过"})
                else:
                    s = "finish 未通过：" + missing()
                    bits.append(s)
                    executed.append({"name": "finish", "result": s})
                    print(s)
            else:
                bits.append(f"{c['name']}：没有这个工具")
                executed.append({"name": c["name"], "result": bits[-1]})
            if ended:
                break
        if n == 1 and not ended:                        # the code's first batch: each subject's piece of the stem + the common pieces
            batch = first_batch()
            results, s, labs, new, skipped = run_searches(batch, n, set(pool), budget)
            got += labs
            s = "系统按题面做的第一批检索：" + s + (f"；这次回复里的 {len(model_first)} 个 search 没有执行（第一批由系统按题面做，之后补查可以自己写）" if model_first else "")
            bits.append(s)
            executed.append({"name": "search", "by": "system", "calls": [b["args"] for b in batch], "result": s, "results": results,
                             "new": new, "skipped": skipped, "model_searches_not_run": [c.get("args") for c in model_first]})
            print(s)
        skipped_calls = [{"name": c["name"]} for c in order[i:]] if ended else []
        cited_list = sorted(cited, key=lambda s: int(s[1:]))
        next_pool, next_parts = next_pool_of(pool, cited_list, got)
        requests.append({"request": n, "user": user, "pool": pool, "content": content, "tool_calls": raw, "finish_reason": finish_reason,
                         "executed": executed, "not_executed": skipped_calls, "cited": cited_list, "got": list(dict.fromkeys(got)),
                         "next_pool": next_pool, "next_parts": next_parts, "usage": u, "ended": ended})
        if ended:
            stop = ended
            break
        summary = "；".join(bits) if bits else "没有调用工具"
        pool, parts_now = next_pool, next_parts
        tagged = {lab: asked_for[lab] for lab in next_pool if asked_for.get(lab)}
    else:
        stop = f"请求次数用完（{MAX_REQUESTS} 次）"
    print(f"停止：{stop}")

    # -------------------------------------------------------------- 兜底：没算出的部分按假设算式
    final, how = {}, {}
    open_keys = [k for k in parts if not concluded(k)]
    for k in parts:
        if concluded(k):
            final[k], how[k] = answers[k]["result"]["text"], "算式"
    with_fb = [k for k in open_keys if (answers.get(k) or {}).get("fallback")]
    for k, r in evaluate(with_fb, literals_ok=True, field="fallback").items():
        if r["ok"]:
            final[k], how[k] = r["text"], "模型写的假设算式"
        answers[k]["fallback_result"] = r
    guess_call = None
    no_fb = [k for k in open_keys if k not in final]
    if no_fb:
        cited = sorted({c for a in answers.values() for c in a["cite"]} | {c for s in slots.values() for c in s["cite"]}, key=lambda s: int(s[1:]))
        user = (f"题目：\n{question}\n\n当前 plan：\n{plan_view()}\n\n要写假设算式的答案部分：{'、'.join(no_fb)}\n\n"
                f"之前引用过的原文块（{len(cited)} 块）：\n\n" + ("\n\n".join(show_chunk(lab, {}) for lab in cited) or "（无）"))
        content, u = llm.chat(SYSTEM_GUESS, user, max_tokens=800, tag=f"{qid}:guess")
        usage[0] += u["prompt_tokens"]
        usage[1] += u["completion_tokens"]
        calls_log.append({"stage": "guess", "usage": u, "user": user, "response": content})
        data = parse_json(content)
        fbs = data.get("fallback") if isinstance(data.get("fallback"), dict) else {}
        reasons = data.get("reason") if isinstance(data.get("reason"), dict) else {}
        guess_call = {k: {"fallback": text_of(fbs.get(k)), "reason": text_of(reasons.get(k))} for k in no_fb}
        for k in no_fb:
            answers.setdefault(k, {"cite": [], "formula": "", "reason": "", "status": UNKNOWN, "need": "", "fallback": "", "round": None, "result": None})
            answers[k]["fallback"] = guess_call[k]["fallback"]
        for k, r in evaluate(no_fb, literals_ok=True, field="fallback").items():
            if r["ok"]:
                final[k], how[k] = r["text"], "补问的假设算式"
            answers[k]["fallback_result"] = r
    for k in parts:
        if k not in final:
            final[k], how[k] = "无法计算", "没有算出"
    answers_list = [final[k] for k in parts]
    notes = []
    guessed = [k for k in parts if how[k] != "算式"]
    if guessed:
        notes.append("、".join(f"{k}（{how[k]}）" for k in guessed) + " 没有按原文算出")
    note = "；".join(notes)
    print(f"\n答案：{'；'.join(answers_list)}" + (f"\n注意：{note}" if note else ""))

    out = {"qid": qid, "type": kind, "question": question, "model": llm.model, "system": system, "tools": tools,
           "settings": {"max_requests": MAX_REQUESTS, "auto_finish": auto, "top_per_query": TOP_PER_QUERY, "max_searches": MAX_SEARCHES,
                        "max_new": MAX_NEW, "max_body": MAX_BODY, "overlap": OVERLAP, "max_tokens": MAX_TOKENS},
           "parts_fixed": bool(fixed), "parts": parts, "slots": slots, "calcs": calcs, "values": values(),
           "requests": requests, "n_requests": len(requests), "stop": stop,
           "answers_by_part": answers, "history": history, "guess_call": guess_call, "how": how,
           "answers": answers_list, "answer": "；".join(answers_list), "caveat": note, "calls": calls_log, "n_calls": len(calls_log),
           "usage": dict(llm.usage), "seconds": round(time.time() - t0, 1),
           "store": store, "serves": {lab: sorted(v) for lab, v in serves.items()},
           "seen": {nd: {d: [list(w) for w in ws] for d, ws in s.windows.items()} for nd, s in seen.items()}}
    base = ROOT / "output" / NAME / OUT_TAG / f"{NAME}_{qid}"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    base.with_suffix(".md").write_text(report(out), encoding="utf-8")
    uu = llm.usage
    print(f"{len(requests)} 次请求{'（另有 1 次补问假设算式）' if guess_call else ''}，看过 {len(store)} 块，tokens {uu['prompt_tokens']} + {uu['completion_tokens']} = "
          f"{uu['total_tokens']}，{out['seconds']} 秒 -> {base.relative_to(ROOT)}.json / .md")
    return out


# ---------------------------------------------------------------------- record
def report(o):
    """The run as a Markdown record: every request's context, the tool calls as sent, what the code did."""
    u = o["usage"]
    cell = lambda t: str(t or "").replace("|", "/").replace("\n", " ")   # noqa: E731
    L = [f"# {o['qid']}：计算题策略（工具调用主循环）作答记录", "",
         f"{o['type']}。模型 {o['model']}，温度 0，思考模式关闭，护栏 {'开' if o['settings']['auto_finish'] else '关'}。"
         f"{o['n_requests']} 次请求" + ("，另有 1 次补问假设算式" if o["guess_call"] else "") + f"，{o['seconds']} 秒；"
         f"tokens 输入 {u['prompt_tokens']} + 输出 {u['completion_tokens']} = {u['total_tokens']}。停止原因：{o['stop']}。",
         f"答案：**{o['answer']}**。" + (f"注意：{o['caveat']}。" if o["caveat"] else ""), "",
         "## 题目", "", "```text", o["question"], "```", "",
         "## 最终结果", "", "| 部分 | 算什么 | 输出单位 | 结果 | 来源 | 算式 | 依据 | 理由 |", "|---|---|---|---|---|---|---|---|"]
    for k, p in o["parts"].items():
        a = o["answers_by_part"].get(k) or {}
        formula = a.get("formula") if o["how"][k] == "算式" else a.get("fallback")
        L.append(f"| {k} | {cell(p['what'])} | {cell(p['unit'])} | {o['answers'][list(o['parts']).index(k)]} | {o['how'][k]} | `{cell(formula)}` | "
                 f"{'、'.join(a.get('cite') or [])} | {cell(a.get('reason'))} |")
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
                    cut = f"（模型写的是「{s['raw_query']}」，去掉了不在题面里的 {'、'.join(s['dropped'])}）" if s.get("dropped") else ""
                    L.append(f"  - 「{s['query']}」{cut}（为 {'、'.join(s['for']) or '—'}）实际检索「{s['searched']}」，{s['docs']} 份文档：{got}")
        if r["not_executed"]:
            L.append(f"- 结束后没有执行：{'、'.join(c['name'] for c in r['not_executed'])}")
        L += ["", f"引用：{'、'.join(r['cited']) or '无'}；本轮检索到：{'、'.join(r['got']) or '无'}；下一次的证据池：{'、'.join(r['next_pool']) or '无'}。"
              + (f"结束：{r['ended']}。" if r["ended"] else ""), ""]
    if o["guess_call"]:
        c = next(x for x in o["calls"] if x["stage"] == "guess")
        L += ["### 兜底：补问假设算式", "", "```text", c["response"], "```", ""]
    L += ["## 最终 plan", "", "| 部分 | 算什么 | 输出单位 | 槽位 |", "|---|---|---|---|"]
    L += [f"| {p['id']} | {cell(p['what'])} | {cell(p['unit'])} | {'、'.join(p['slots'])} |" for p in o["parts"].values()]
    if o.get("calcs"):
        L += ["", "| 中间量 | 是什么 | 算式 | 原始值 |", "|---|---|---|---|"]
        for c in o["calcs"].values():
            r = c.get("result") or {}
            L.append(f"| {c['id']} | {cell(c['what'])} | `{cell(c['formula'])}` | {calc_value_text(r['ans']) if r.get('ok') else '没有算出：' + cell(r.get('error'))} |")
    L += ["", "| 槽位 | 内容 | 类型 | 值 | 算式里的值 | 依据 | 出处 |", "|---|---|---|---|---|---|---|"]
    for s in o["slots"].values():
        v = o["values"].get(s["id"])
        L.append(f"| {s['id']} | {cell(s['what'])} | {s['kind']} | {cell((s['value'] + ' ' + s['unit']).strip()) or '缺失'} | "
                 f"{cell(v if isinstance(v, str) else (f'{v:.12g}' if v is not None else ''))} | {'、'.join(s['cite'])} | {cell(s['where'])} |")
    L += ["", "## 看过的原文块", "", "| 编号 | 块 | 页 | 字符区间 | 请求 |", "|---|---|---|---|---|"]
    L += [f"| {lab} | {s['uid']} | {s['pages'][0]}-{s['pages'][-1]} | {s['chars'][0]}-{s['chars'][1]} | {s['request']} |" for lab, s in o["store"].items()]
    return "\n".join(L)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
