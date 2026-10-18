# -*- coding: utf-8 -*-
"""Shared claim machinery of the answer pipelines: tokenisation, statement/claim
splitting, the claim-analysis call and the judgement call.

Moved out of the v1 pipelines (experiments/archive/agent_v1) so that bm25_pipeline and the
retrieval layer depend on one small module instead of on the retired cascades.
"""
import json
import re

import jieba

from .llm import parse_json

jieba.setLogLevel(40)

VERDICTS = ("正确", "错误", "证据不足")

# ----------------------------------------------------------------------------- tokenisation
RE_NUM_TOKEN = re.compile(r"\d+(?:\.\d+)?")
RE_EN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+\-]*")

# function words dropped from queries and indexes (2026-10-03): particles, conjunctions,
# prepositions and the copula among the most frequent tokens of the corpus (的 is in 72 % of
# the chunks, 和 36 %, 在 27 %). rank_bm25's IDF floor gives a term found in more than half
# of the chunks 0.25 x the mean IDF, more than a term found in a third of them gets.
# Kept on purpose: position words (内 / 后 / 中 / 时: 犹豫期内 vs 犹豫期后), negations
# (不 / 未) and modals (应当 / 可以 / 不得). bm25.TOKENIZER is bumped with every change here.
STOPWORDS = frozenset("的 了 之 所 和 与 及 或 并 而 且 以及 及其 或者 并且 而且 "
                      "为 是 在 对 于 以 由 向 将 被 把 从 等 其 也".split())


def tokenize(text):
    """jieba words (lower-cased, no punctuation, no STOPWORDS) plus normalised numbers and
    English tokens."""
    toks = []
    for t in jieba.lcut(text):
        t = t.strip().lower()
        if t and t not in STOPWORDS and not re.fullmatch(r"[\W_]+", t):
            toks.append(t)
    plain = text.replace(",", "").replace("，", "")
    toks += RE_NUM_TOKEN.findall(plain)
    toks += [m.lower() for m in RE_EN_TOKEN.findall(text)]
    return toks


# ----------------------------------------------------------------------------- statements
def statement_of(question):
    """The bare claim: drop the framing ("判断以下陈述是否正确：", "根据…报告，", "该说法是否正确？")."""
    s = question["question"].strip()
    s = re.sub(r"^(判断以下陈述是否正确|请判断|判断)[:：]?\s*", "", s)
    s = re.sub(r"^根据[^，,]{0,40}(报告|文档|信息|资料)[，,]\s*", "", s)
    s = re.sub(r"[，,]?\s*(该|上述|以上)?(说法|陈述|描述)是否正确[？?]?$", "", s)
    return s.strip()


SPLIT = re.compile(r"[；;]|，\s*且|，\s*同时[，,]?|；\s*同时[，,]?|，\s*并且")


def split_claims(statement):
    """Sub-claims of a statement, split at 、且 / ；同时 / 并且 / ；."""
    parts = [p.strip("，,。 ") for p in SPLIT.split(statement)]
    parts = [p for p in parts if len(p) >= 8]
    return parts or [statement]


# ----------------------------------------------------------------------------- call 0: analyse the claim
SYSTEM_ANALYZE = (
    "你是金融研报核验助手。只看待核对的陈述本身（此时不看文档）：说明它声称的主体、时期和指标口径，"
    "并给出最容易在研报原文中定位到对应句子的检索词。"
)

USER_ANALYZE = """陈述：{statement}

只输出一个 JSON：
{{"subject": "陈述明确写出的主体（公司/行业/市场/国家或地区），没写则填“未指明”", "period": "陈述明确写出的时期（年份/季度/区间），没写则填“未指明”", "metric": "指标名称及口径，保留限定词（如除客户资金、储能、零售口径）", "search": ["陈述中出现的每一个数字和百分比，按原样写（如 3.6%、1.56、4252.91、1000）", "公司、机构、产品、地区等专有名词", "2 到 4 个最具区分度的关键词或短语"]}}"""


def analyze_claim(llm, statement, tag=""):
    """Subject / period / metric stated by the claim, and its search terms."""
    content, _ = llm.chat(SYSTEM_ANALYZE, USER_ANALYZE.format(statement=statement), max_tokens=300, tag=tag + "analyze")
    data = parse_json(content)
    search = [str(s).strip() for s in (data.get("search") or []) if str(s).strip()]
    return {"subject": str(data.get("subject") or "未指明").strip(), "period": str(data.get("period") or "未指明").strip(),
            "metric": str(data.get("metric") or "").strip(), "search": search}


# ----------------------------------------------------------------------------- judgement
SYSTEM_JUDGE = (
    "你是金融研报核验助手。下面给出待核对的陈述、陈述自身声称的主体/时期/口径，以及从文档中检索到的证据"
    "（JSON 数组，每条含文档标题 doc_title、页码 page、小节 section 和原文 text）。只能依据这些证据判断，不得使用其他知识。"
    "步骤：1）在证据里找出与陈述直接相关的关键句（含陈述中的数字、指标、事件或比较对象的句子或表格行），"
    "逐句判断这句数据的主体（公司/行业/市场/国家或地区）、时期（年份/季度/区间）和指标口径。"
    "主体和时期要结合 doc_title、section、前后句和表头来读，不能只看这一句本身；年份常写在文档标题或小节标题里（如“2025年财报”）或段首。"
    "2）逐项对照主体、时期、指标口径、数字与单位、方向（增长/下降、自动/手动、美元/欧元等）。"
    "陈述有多个分句时，每个分句都要在证据里找到对应内容并单独核对：任一分句与证据不符（如证据说“支撑了保费的快速增长”，"
    "陈述改成“支撑了管理体系”）即为错误；某个分句的内容证据完全没有涉及时为证据不足。"
    "判错标准（实质性差异，任一处即为错误）：数字、单位或方向不一致；指标口径的限定词（如“除/剔除”“净/毛”“同比/环比”“累计/单季”"
    "“储能/动力”“零售/批发口径”）缺失或改变；陈述明确写出的主体或时期与关键句不一致；把行业、市场或他国的数据归到某家公司名下（或反之）。"
    "不算错误：陈述没有写主体或时期，而关键句所在上下文中的主体/时期唯一、且与陈述其余内容不冲突；不改变含义的同义改写；"
    "陈述写的年份是关键句时期区间的起点、终点或其中一年，且陈述引用的数字正是该时点的数字（如证据“1985年至2000年从10%提升至50%”，"
    "陈述“从1985年的10%快速提升”）；同一文档不同位置的数字略有出入时，只要有一处与陈述一致。"
    "“证据不足”只用于证据完全没有涉及陈述所说的数据或事项；如果关键句已经给出陈述所引用的数字或事项，"
    "只是主体、口径或时期与陈述不同，那就是陈述在文档中没有依据，结论是“错误”而不是“证据不足”。"
)

USER_JUDGE = """待判断的陈述：
{statement}
陈述自身声称的主体：{subject}；时期：{period}；指标口径：{metric}

证据：
{evidence}
{images_note}
只输出一个 JSON，不要输出其他文字：
{{"key_sentences": [{{"evidence": "E1", "quote": "关键句原文，完整的一句，不超过150字", "subject": "这句数据的主体", "period": "这句数据的时期，从 doc_title、section、前后句读出", "metric": "指标名称及口径，保留限定词", "basis": "判断主体和时期的依据"}}], "clauses": [{{"clause": "陈述的一个分句", "verdict": "一致|不一致|证据未涉及", "evidence": "E1"}}], "comparison": {{"subject": "一致|不一致|陈述未写且上下文唯一", "period": "一致|不一致|陈述未写且上下文唯一", "metric": "一致|不一致", "number": "一致|不一致", "direction": "一致|不一致"}}, "verdict": "正确|错误|证据不足", "reason": "一句话结论"{chart_field}}}"""

# added to the system prompt and the JSON when figure crops are attached (agent.figures); rules
# from the 2026-09-24 chart test: printed data labels are exact, unlabelled bars and points
# are estimates, an exact decimal the chart does not print cannot be confirmed, a year or
# category outside the chart is not evidence, and two bars closer than a tenth of the axis
# range cannot be told apart
SYSTEM_CHART = (
    "证据中标注为图表的条目附有从原文裁出的图片，按顺序排在本消息末尾；图内文字只列了图例和类目，数值要从图片上读。"
    "读图规则：1）图上印有数据标签的，以标签数值为准；2）没有数据标签的，按坐标轴估读，估读值与陈述数值的差不超过"
    "坐标轴量程的十分之一视为一致；陈述给出的是图上没有印出的精确小数值时，判“证据不足”；3）陈述涉及的年份或类目"
    "不在图的横轴或图例范围内时，判“证据不足”；4）比较两个柱或两条线的高低时，差距小于量程的十分之一视为无法分辨，"
    "判“证据不足”。把从图上读到并用于判断的每个数值列在 chart_readings 里。"
)
CHART_FIELD = ', "chart_readings": [{"evidence": "E2", "item": "读数对应的类目和时期", "value": "读到的数值", "printed": true}]'


def images_note(images):
    """The line telling the judge which evidence records come with a crop."""
    if not images:
        return ""
    return "\n以下证据是图表，其图片按此顺序附在本消息末尾：" + "、".join(f"{eid}（{uid.split('#')[1]}）" for eid, uid, _ in images) + "\n"


def evidence_json(records):
    return json.dumps(records, ensure_ascii=False, indent=1)


def judge(llm, statement, analysis, records, tag, images=None):
    """One call over the evidence records: key sentences with subject and period,
    the comparison with the claim, and the verdict. images: [(evidence id, unit id, png)]
    figure crops attached to the call (agent.figures.evidence_images). Returns (verdict, data)."""
    a = analysis or {}
    user = USER_JUDGE.format(statement=statement, subject=a.get("subject") or "未指明", period=a.get("period") or "未指明",
                             metric=a.get("metric") or "未指明", evidence=evidence_json(records),
                             images_note=images_note(images), chart_field=CHART_FIELD if images else "")
    system = SYSTEM_JUDGE + (SYSTEM_CHART if images else "")
    pngs = [png for _, _, png in (images or [])]
    content, _ = llm.chat(system, user, max_tokens=1200 if images else 1000, tag=tag, images=pngs)
    data = parse_json(content)
    if str(data.get("verdict", "")).strip() not in VERDICTS:
        content, _ = llm.chat(system, user + "\n\n上一次的回答没有给出合法的 JSON，这次只输出一个 JSON。",
                              max_tokens=1200 if images else 1000, tag=tag + "-retry", images=pngs)
        data = parse_json(content)
    verdict = str(data.get("verdict", "")).strip()
    keys = [k for k in (data.get("key_sentences") or []) if isinstance(k, dict) and k.get("quote")]
    if verdict not in VERDICTS:
        verdict, data = "证据不足", {"reason": "模型没有给出合法的结论", "comparison": {}}
    elif verdict == "正确" and not keys:
        verdict = "证据不足"                   # 正确 must rest on a quoted sentence
        data["reason"] = "判为正确但没有引用关键句：" + str(data.get("reason", ""))
    data["verdict"] = verdict
    data["key_sentences"] = keys
    data["chart_readings"] = [c for c in (data.get("chart_readings") or []) if isinstance(c, dict)]
    return verdict, data


def evidence_list(data):
    return [{"evidence": k.get("evidence"), "quote": k.get("quote"), "subject": k.get("subject"), "period": k.get("period"),
             "metric": k.get("metric"), "basis": k.get("basis")} for k in (data or {}).get("key_sentences", [])]


def letter_for(question, verdict):
    for k, v in question["options"].items():
        if verdict == "正确" and "正确" in v:
            return k
        if verdict == "错误" and "错误" in v:
            return k
    return "B"
