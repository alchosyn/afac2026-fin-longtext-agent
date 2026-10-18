# -*- coding: utf-8 -*-
"""Chunk-level evidence search for one statement of a B-list question (no doc_ids given).

    scope      the chunks whose header passes the question's filter (agent.scope): the domain,
               and the documents the question names; nothing named -> the whole domain
    rewrite    one Qwen call turns stem + statement into self-contained sub-claims, one entity
               or year each; the entity short names of the scope that stand in the text are
               handed to it as 简称 = 全称 (processed_data/doc_terms.json)
    retrieve   BM25 (agent.retrieval.bm25_scores: body 0.7 + heading path 0.3), twice: stem +
               statement as one query, and every rewritten sub-claim. A query that names some
               of the scope's documents is searched in those alone, without the names
               (narrow). The top TOP of each query are the candidates (measured 2026-10-04 on
               10 B questions: top 10 holds what top 20 holds)
    select     the top KEEP of each query, in BM25 order, are the evidence

    statements(question)                -> [(claim id, statement)]
    option_role(llm, question)          -> "statements" | "candidates" for a choice question
                                           (search() does not read it yet)
    search(question, statement, llm)    -> {"scope", "rewrite", "queries", "candidates", "evidence", "usage"}
"""
import hashlib
import re

from .claims import split_claims, statement_of
from .llm import parse_json
from .retrieval import bm25_scores
from .scope import cards, named, norm, scope, short, short_names

TOP = 10                # candidates per query
KEEP = 3                # evidence chunks per query
KINDS = {"多选题": "multi", "单选题": "mcq", "判断题": "tf", "计算题": "calc"}
STRIP = ("financial_reports", "financial_contracts", "insurance")   # domains whose document names leave the query text
_CARD = {}

SYSTEM = "你是金融文档检索助手。你的任务是把一道题里待核对的陈述改写成适合在文档原文中做关键词检索的陈述，不判断对错。"
USER = """题干：{question}
待核对的陈述：{statement}
涉及的文档：{titles}
{names}
把“待核对的陈述”改写成检索用的陈述，要求：
1. 自足：补上陈述里省略、但题干或文档名里有的主体（公司、产品、法规，用简称即可，不要照抄文档全名）、时间和话题，让它脱离题干也能看懂。
2. 用文档原文里更可能出现的正式说法替换口语、缩写和概括性说法（例如“归母净利润”写成“归属于上市公司股东的净利润”，“两融”写成“融资融券”）。
3. 陈述涉及多个主体或多个年份时，拆成多句，每句只讲一个主体或一个年份，句子之间用中文分号“；”隔开。
4. 陈述里需要计算才能得到的数字（差额、增幅、百分点变化、比例）不要写，改为写出计算所需的原始指标名称；文档里会原样出现的数字、日期、专有名词保留。
5. 不添加题目里没有的事实，不写“是否正确”“说法”等与文档内容无关的词。
{rule6}
只输出改写后的陈述，不要解释。"""
NAMES = "主体简称对照（摘自相关文档原文；左边是题面里出现的简称，右边是文档里的全称）：\n{rows}\n"
RULE6 = "6. 题面里的简称如果出现在主体简称对照里，改写时按对照写出全称；与这条陈述无关的对照不要用。\n"


def statements(q):
    """The statements of a question, each verified on its own: the options of a choice
    question, the statement of a true/false question, a calculation question as a whole."""
    kind = KINDS[q["type"]]
    if kind == "tf":
        s = re.sub(r"[，,。]?\s*该表述是否正确[？?]?$", "", statement_of(q).lstrip("题：").strip()).strip()
        return [(q["qid"] + "-T", s)]
    if kind == "calc":
        return [(q["qid"] + "-Q", re.sub(r"答案只填写数字。?$", "", q["question"].strip()).strip())]
    return [(f"{q['qid']}-{k}", v.strip()) for k, v in q["options"].items()]


SYSTEM_ROLE = "你是金融文档问答系统的题目分析助手。你的任务是判断一道选择题的选项与题干是什么关系，用来决定怎样到文档里检索；不回答题目，不判断选项对错。"
USER_ROLE = """题干：{question}
选项：
{options}

选项与题干的关系有两种：
- 陈述：每个选项自己就是一句完整的判断，有自己要核对的事项、数据或对象；题干只提供话题、范围、背景或共同的主体。检索时每个选项要单独查。
- 候选：题干提出一个具体的问题或情形，各选项只是它的候选答案：一个数值、一个名称（产品、文档、机构），或者省略了情形和对象、必须接在题干后面才成立的短结论。检索时只用题干查一次。

判断方法：
1. 把每个选项单独拿出来，只允许补上题干里的主体（公司、产品、法规的名称），看它是不是一句完整的、能直接拿去和原文核对的判断。
2. 各选项都是完整的判断，就是“陈述”。题干用问句写（问原因、影响、趋势、共同点等开放性问题），或者题干已经指明了文档和指标，都不改变这一点：只要选项各有各的内容，就要单独查。
3. 选项离开题干的问题或情形就读不懂，或者只是一个值或名称，才是“候选”。
4. 拿不准时归为“陈述”。

只输出一个 JSON：{{"reason": "一句话依据", "kind": "候选|陈述"}}"""
ROLES = {"候选": "candidates", "陈述": "statements"}


def option_role(llm, q, cache=None, tag=""):
    """(role, reason, usage) of a choice question, one Qwen call over stem + options:
    "candidates" when the stem asks one question and the options are its candidate answers,
    "statements" when every option is a claim of its own. A reply without a clear label is
    "statements", the cheaper mistake: a candidate searched on its own repeats the stem's
    search, a claim searched with the stem alone loses its own terms."""
    options = "\n".join(f"{k}. {v.strip()}" for k, v in q["options"].items())
    user = USER_ROLE.format(question=q["question"].strip(), options=options)
    key = hashlib.sha1(user.encode("utf-8")).hexdigest()
    if cache is not None and key in cache:
        return cache[key]["role"], cache[key]["reason"], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    content, usage = llm.chat(SYSTEM_ROLE, user, max_tokens=200, tag=tag + "role")
    data = parse_json(content)
    role = ROLES.get(str(data.get("kind", "")).strip(), "statements")
    reason = str(data.get("reason", "")).strip()
    if cache is not None:
        cache[key] = {"role": role, "reason": reason}
    return role, reason, usage


def rewrite(llm, q, statement, sc, cache=None, tag=""):
    """(rewritten statement, short-name rows, usage). `cache` {prompt sha1: text} keeps a rewrite
    across runs: at temperature 0 the model still words half of them differently each time."""
    filtered = bool(sc["named"])
    rows = short_names(q["question"] + " " + statement, q["domain"], sc["docs"] if filtered else None)
    title = {c["doc_id"]: c["title"] for c in cards()}
    titles = "；".join(f"《{title[d]}》" for d in sc["docs"]) if filtered else "（题面未指明具体文档）"
    user = USER.format(question=q["question"], statement=statement, titles=titles,
                       names=NAMES.format(rows="\n".join(f"- {r['term']}：{r['full']}" for r in rows)) if rows else "",
                       rule6=RULE6 if rows else "")
    key = hashlib.sha1(user.encode("utf-8")).hexdigest()
    if cache is not None and key in cache:
        return cache[key], rows, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    content, usage = llm.chat(SYSTEM, user, max_tokens=300, tag=tag + "rewrite")
    text = content.strip().strip("“”\"").strip()
    if cache is not None:
        cache[key] = text
    return text, rows, usage


def doc_names(doc_id, how):
    """The spellings a query uses for the subject of a document: the company in full, without
    its legal suffix, as stock name or alias; for an insurance product the string that named
    it (`how`, agent.scope.named)."""
    if not _CARD:
        _CARD.update({c["doc_id"]: c for c in cards()})
    c = _CARD[doc_id]
    if c["domain"] == "insurance":
        names = [how]
    else:
        names = [norm(c["company"]), short(c["company"]), c.get("stock_name"), *c.get("company_aliases", [])]
    return [n for n in names if n and len(n) >= 2]


def narrow(q, text, sc):
    """(documents, query text) one query is searched with. In a scope the question named, a
    query that itself names some of those documents (agent.scope.named: the entity, for annual
    reports also the year) is searched in them alone. The names of the documents searched are
    then taken out of the text: a name has done its work by selecting the documents, and left
    in the query it scores the chunks that spell it out (audit opinion, page headers,
    guarantee tables) above the tables holding the figures. Regulation titles and research
    subjects stay in the text: there the matched string is often the topic itself."""
    if not sc["named"]:
        return sc["docs"], text
    hit = named(q, norm(text))
    docs = [d for d in sc["docs"] if d in hit] or sc["docs"]
    if q["domain"] in STRIP:
        bare = text
        for n in sorted({n for d in docs for n in doc_names(d, hit.get(d) or sc["named"][d])}, key=len, reverse=True):
            bare = bare.replace(n, " ")
        text = bare if bare.strip() else text
    return docs, text


def search(q, statement, llm, top=TOP, keep=KEEP, cache=None, tag=""):
    """Candidates and evidence chunks of one statement. A query records "docs" and "searched"
    when narrow() changed where or with what text it was searched."""
    sc = scope(q)
    rewritten, rows, usage = rewrite(llm, q, statement, sc, cache, tag)
    stem = q["question"].strip() + " " + statement if KINDS[q["type"]] in ("multi", "mcq") else statement
    queries = [{"kind": "stem", "text": stem}] + [{"kind": "rewrite", "text": s} for s in split_claims(rewritten)]
    for query in queries:
        docs, text = narrow(q, query["text"], sc)
        if docs != sc["docs"]:
            query["docs"] = docs
        if text != query["text"]:
            query["searched"] = text
        query["top"] = [(u, round(s, 2)) for u, s in bm25_scores(text, docs)[:top] if s > 0]
    candidates = list(dict.fromkeys(u for query in queries for u, _ in query["top"]))
    evidence = list(dict.fromkeys(u for query in queries for u, _ in query["top"][:keep]))
    return {"scope": sc, "rewrite": rewritten, "short_names": rows, "queries": queries,
            "candidates": candidates, "evidence": evidence, "usage": usage}
