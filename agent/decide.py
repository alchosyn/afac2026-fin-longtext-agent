# -*- coding: utf-8 -*-
"""Choosing the answer of a single- or multiple-choice question from the per-option
verdicts (moved out of the v1 mcq / multi pipelines).

    direction(stem)          -> which verdict the stem asks for, "正确" or "错误"
    choose(asked, verdicts)  -> the options with that verdict, and the ones in neither direction
    pick / pick_multi        -> the comparison calls of the group A pipeline (agent.bm25_pipeline)

The three steps are kept apart (2026-10-05): the stem gives the direction, a statement is
judged only on whether it agrees with the documents, the code picks the options by the
direction.
"""
import re

from .llm import parse_json

# ----------------------------------------------------------------------------- what the stem asks for
# A stem asks for 错误 only through a word that judges the options as statements (错误,
# 不正确, 有误, 与原文不一致 ...) standing where a question asks ("…错误的是", "哪些判断不正确",
# "请选出有误的一项"). Two things look alike and are not that:
#   - the topic: 年龄错误 is a clause of every insurance product, 登记信息错误、不一致 a
#     case of a regulation; "以下哪些产品约定了年龄错误的处理方式" asks for 正确
#   - a negation that is part of what an option says about the documents (不属于, 不包括,
#     不承担, 产品不符合投保条件): it stays in the statement and is judged with it
_NO = "(?:不|并不|并非|未必)(?:完全|全|太|够|尽|相|是|算)?"
_HARD = "正确|准确|成立|恰当|合理|属实|妥当"                  # judge a statement whatever the subject is called
_SOFT = "符合|相符|一致"                                      # judge one only when the subject is a statement
WORD = re.compile(rf"(?P<fine>(?:没有|不存在|未见|无)(?:任何)?(?:错误|误))"            # 没有错误的是: asks for 正确
                  rf"|(?P<hard_no>{_NO}(?:{_HARD})|错误|有误|不妥|不对(?![称等外应付]))"
                  rf"|(?P<soft_no>{_NO}(?:{_SOFT})|不符)"
                  rf"|(?P<hard>{_HARD})|(?P<soft>{_SOFT})")
NOTE = re.compile(r"[（(][^（()）]*[)）]")                    # a bracketed note, innermost first
QUOTED = re.compile(r"“([^”]*)”|‘([^’]*)’|\"([^\"]*)\"|「([^」]*)」|《([^》]*)》")
STOCK = re.compile(r"正确答案|最佳答案|标准答案|符合题[意目](?:要求)?")   # "请选出符合题意的选项": says nothing
CLAUSE = re.compile(r"[，,。；;：:？?！!\n]")
PICK = re.compile(r"以下|下列|下述|下面|哪[一二两三四几]?[些项个种条]?")   # the words that point at the options
ORDER = re.compile(r"(?:选出|指出|找出|挑出|选择|勾选)(?:所有|全部|其中)?$")
BLANK = re.compile(r"[ \t　\xa0*]+|[_＿—\-]{2,}")           # spaces, emphasis marks, the blank to fill in
_SAID = "说法|表述|表达|判断|描述|陈述|论述|叙述|结论|分析|归纳|概括|理解|推断|推论|解读|观点|数据|数字|数值|选项|各项|项"
_LINK = "(?:中|里)?(?:是|为|都|均|有|存在)?(?:最|较|更|明显)?"
# what may stand between the pointing word and the judgement word: a noun for a statement or
# nothing, 是 / 最 / 存在, a comparison with the documents ("说法中与原文")
SAID = re.compile(rf"(?:(?P<pre>[^，]{{0,12}}?)(?:{_SAID}))?{_LINK}(?:与[^，]{{1,16}}?)?{_LINK}")
# any other subject of a hard word ("计算结果", "产品匹配", "金额排序") may not hold a verb
# (存在计算错误, 属于年龄错误, 导致披露不准确 are content) nor, unless 是 stands between
# ("哪些信息是准确的"), the head of a topic compound (年龄错误, 登记信息错误)
_VERB = r"属于|导致|造成|构成|存在|出现|发生|涉及|约定|规定|列明|写明|载明|明确|包含|包括|视为|认定|需要|可以|应当|承担|对(?![应比照])|把|被|了"
VERB = re.compile(_VERB)
CONTENT = re.compile(_VERB + r"|年龄|性别|信息|登记|披露|申报|操作")
COPULA = re.compile(r"(?:都|均)?(?:是|为)(?:最|较|更|明显)?$")
NOUN_END = re.compile(rf"(?:{_SAID})(?:最|较|更)?$")
# after the judgement word: "…的是", "监管要求的是", "的一项是", "描述是", "的有哪些"
TAIL = re.compile(rf"(?:[^的]{{0,12}}的){{0,2}}(?:{_SAID}|一项)?(?:是|有|为)(?:{PICK.pattern}|[一二两三四几][项个])?(?:选项|说法|表述)?")
# "哪些判断错误", "哪项是错误的", "选出错误的一项", "最准确地概括了…", "准确且有充分依据的"
END_HARD = re.compile(rf"的?(?:{_SAID}|一项|答案)?|地[^，]{{0,16}}|[且或并而][^，]{{0,12}}")
END_SOFT = re.compile(r"[^，]{0,16}")                                     # "哪项不符合监管要求"


def _said(text):
    """text names a statement (or nothing) as what the judgement word is said of."""
    m = SAID.fullmatch(text)
    return bool(m) and not VERB.search(m.group("pre") or "")


def _asks(clause, m):
    """Whether the judgement word m of a clause stands where a question asks for options."""
    before, after = clause[:m.start()], clause[m.end():]
    soft = m.lastgroup.startswith("soft")
    picks = list(PICK.finditer(before))
    head = before[picks[-1].end():] if picks else before          # between the pointing word and the judgement word
    last = head.split("的")[-1]
    if _said(head) or _said(last):
        subject = True
    elif soft:
        subject = False
    else:
        bare = COPULA.sub("", last)
        subject = len(bare) <= 14 and not (VERB if bare != last else CONTENT).search(head)
    if TAIL.fullmatch(after):                                     # 以下说法错误的是
        return subject
    if picks and soft:                                            # 哪项不符合监管要求
        return subject and bool(END_SOFT.fullmatch(after))
    if picks:                                                     # 以下哪些判断错误 / 哪项描述准确反映了…
        return subject and (bool(END_HARD.fullmatch(after)) or bool(NOUN_END.search(head)) and len(after) <= 24)
    return bool(ORDER.search(before)) and bool(END_HARD.fullmatch(after))    # 请选出错误的一项


def _quoted(q):
    """A quoted judgement word stays (说法“错误”的是); any other quoted text or title goes."""
    inner = next(g for g in q.groups() if g is not None)
    return inner if WORD.fullmatch(inner) else "□"


def direction(stem):
    """(asked, clause, word): the verdict the stem asks for, "正确" (the options agreeing with
    the documents are the answer) or "错误" (the ones that do not), the clause it was read
    from and the word that decided (None: nothing matched).

    The direction is read from the question alone: of all clauses of the stem, the last one
    in which a judgement word stands in asking position decides (_asks), and in it the last
    such word. So it does not matter whether the question is the last clause ("以下哪些判断
    错误？计算时使用原始金额，结果保留两位小数"), is cut by a comma ("下列说法中，错误的
    是") or holds the topic itself ("下列关于年龄错误处理的说法正确的是"). Bracketed notes,
    quoted text and titles are left out first, a quoted judgement word (说法“错误”的是)
    kept. A stem in which no judgement word asks ("下列产品明确列明该项的是") asks for 正确."""
    text = BLANK.sub("", stem.strip())
    while NOTE.search(text):
        text = NOTE.sub("", text)
    text = STOCK.sub("", QUOTED.sub(_quoted, text))
    clauses = [c.strip() for c in CLAUSE.split(text) if c.strip()]
    for clause in reversed(clauses):
        asking = [m for m in WORD.finditer(clause) if _asks(clause, m)]
        if asking:
            m = asking[-1]
            wrong = m.lastgroup in ("hard_no", "soft_no")
            return ("错误" if wrong else "正确"), clause, m.group(0)
    return "正确", (clauses[-1] if clauses else ""), None


def choose(asked, verdicts):
    """(chosen, unfounded) from {option: verdict}: the options whose verdict is `asked`, and
    the ones with any other value than 正确 / 错误 (无依据, 证据不足, None: no conclusion).
    An unfounded option was not found in the documents, so it is neither 正确 nor 错误 and is
    chosen in neither direction. What that costs differs: asked for 正确, it is left out like
    an option judged 错误; asked for 错误, it is left out although a statement that goes
    beyond the documents may be the very one to choose. The caller reports it (caveat)."""
    chosen = sorted(k for k, v in verdicts.items() if v == asked)
    unfounded = sorted(k for k, v in verdicts.items() if v not in ("正确", "错误"))
    return chosen, unfounded


def caveat(asked, unfounded):
    """What the unfounded options mean for the answer, as one sentence ("" when there are none)."""
    if not unfounded:
        return ""
    names = "、".join(unfounded)
    if asked == "错误":
        return f"选项 {names} 判为无依据或没有结论，没有选；题干问的是错误的选项，其中可能有该选而漏掉的"
    return f"选项 {names} 判为无依据或没有结论，没有选（题干问的是正确的选项，与判为错误的结果相同）"


# ----------------------------------------------------------------------------- comparison calls

SYSTEM_PICK = (
    "你是金融研报核验助手。这是一道单选题，四个选项中只有一个与文档一致。"
    "下面给出每个选项核验后的关键句、主体、时期、口径和初步结论，请只依据这些材料选出唯一正确的选项。"
    "数字、单位、年份、主体、口径、方向与关键句完全一致的选项才是正确选项；如果没有一个完全一致，选与关键句差异最小的那个。"
)

USER_PICK = """题干：{stem}

选项及核验材料：
{material}

先用不超过 100 字说明取舍，然后另起一行，只输出一个 JSON：
{{"answer": "A|B|C|D", "reason": "一句话结论"}}"""

SYSTEM_PICK_MULTI = (
    "你是金融研报核验助手。这是一道多选题，可能有一个或多个选项与文档一致。"
    "下面给出每个选项核验后的关键句、主体、时期、口径和初步结论，请只依据这些材料选出所有正确的选项。"
    "数字、单位、年份、主体、口径、方向与关键句完全一致的选项才算正确；如果没有一个完全一致，选与关键句差异最小的那个，至少选一个。"
)

USER_PICK_MULTI = """题干：{stem}

选项及核验材料：
{material}

先用不超过 120 字逐项说明取舍，然后另起一行，只输出一个 JSON：
{{"answer": "正确选项字母按 ABCD 顺序拼接，如 AC", "reason": "一句话结论"}}"""


def material(results):
    """The per-option verification material shown to the comparison call."""
    lines = []
    for letter in sorted(results):
        r = results[letter]
        lines.append(f"{letter}. {r['statement']}\n   初步结论：{r['verdict']}（第{r.get('resolved_at') or '-'}步）"
                     f"{'，' + r['data'].get('reason', '') if r.get('data') else ''}")
        for k in (r.get("data") or {}).get("key_sentences", [])[:3]:
            lines.append(f"   关键句[{k.get('evidence', '')}]“{k.get('quote', '')}” 主体：{k.get('subject', '')} "
                         f"时期：{k.get('period', '')} 口径：{k.get('metric', '')}")
    return "\n".join(lines)


def pick(llm, question, results):
    """Single choice: one comparison call over the candidates. Returns (letter, reason)."""
    content, _ = llm.chat(SYSTEM_PICK, USER_PICK.format(stem=question["question"], material=material(results)),
                          max_tokens=400, tag="pick")
    data = parse_json(content)
    letter = str(data.get("answer", "")).strip().upper()[:1]
    if letter not in question["options"]:
        m = re.search(r"[A-D]", content[-80:])
        letter = m.group(0) if m else "A"
    return letter, data.get("reason", "")


def pick_multi(llm, question, results, tag="pick"):
    """Multiple choice fallback when no option was judged 正确. Returns (letters, reason)."""
    content, _ = llm.chat(SYSTEM_PICK_MULTI, USER_PICK_MULTI.format(stem=question["question"], material=material(results)),
                          max_tokens=500, tag=tag)
    data = parse_json(content)
    letters = [c for c in str(data.get("answer", "")).upper() if c in question["options"]]
    return sorted(set(letters)) or [sorted(question["options"])[0]], str(data.get("reason", ""))
