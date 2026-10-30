# -*- coding: utf-8 -*-
"""选择题作答的新策略（2026-10-30）：模型先规划检索，再逐选项作答，没有结论的选项补检索重判。
只做选择题（单选、多选）和判断题；计算题不走这里。

    第 1 轮  规划并发出检索：输入第 1 轮 prompt + 题目，模型直接输出若干个检索调用
             {"query", "options"}（options：这个调用服务于哪些选项）。代码逐个调用跑
             agent.search.narrow + BM25，每个调用取前 TOP_PER_QUERY 块
    第 2 轮  逐选项作答：输入第 2 轮 prompt + 题目 + 第 1 轮检索到的全部块。模型逐选项输出
             verdict（正确 / 错误 / unknown）、cite（引用的块）、calc、reason；unknown 的
             先写 need（还缺什么）和 lean（找不到时的倾向），再给新的检索调用
    第 3 轮起 补检索并重判：上下文只有之前作答引用过的块、新检索到的块、之前的作答结果；
             模型只判还没有结论的选项，仍判不了就再写 need、换检索词。最多重试 MAX_RETRIES 次
    兜底     重试用完或提前结束时仍为 unknown 的选项，取模型最后一次写的 lean（倾向）；
             没写 lean 的，再调用一次只问倾向
    选项     代码按题干方向（agent.decide.direction）从各选项结论里挑：题干问正确就选判为
             正确的，问错误就选判为错误的；判断题判为正确选“正确”那项

本地去重（Seen）：本题已见块的集合，跨轮保留。键是（文档、页码起止、字符起止），字符位置是
文档全文坐标：elements.jsonl 按阅读顺序以换行相接，块的区间由它的 members 定，切开的段落和
条款再按 span 取子区间，切开的表格按 rows 在整表区间里按行数比例取。一个块被同一文档里已见
窗口覆盖的比例达到 OVERLAP 就算见过，不再放进上下文；不是新块的不进上下文，也不再给编号。
之前没被引用、已从上下文删掉的块仍留在集合里。某一轮去重后没有新块，就直接结束重试。

    python flows/option_rounds.py fc_b_006 [fc_b_010 ...]

每题写 output/option_rounds/option_rounds_<qid>.json（每次调用的 system / user / 输出 / tokens，
每轮的检索和去重，各选项每轮的作答）和 .md（同样内容的可读记录）。solve(q, spent) 的返回值
有 answer / usage / caveat，和 main.py 读 react_plan.solve 的字段一致。
"""
import json
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.decide import caveat, choose, direction          # noqa: E402
from agent.llm import Qwen, parse_json                      # noqa: E402
from agent.retrieval import bm25_scores, get_doc            # noqa: E402
from agent.scope import cards, load_b_questions, scope      # noqa: E402
from agent.search import narrow, statements                 # noqa: E402

NAME = "option_rounds"
TOP_PER_QUERY = 3           # BM25 hits kept per search call
MAX_SEARCHES = 12           # search calls run in one round
MAX_NEW = 40                # new chunks put into the context in one round
MAX_BODY = 1200             # characters of one chunk shown to the model
MAX_RETRIES = 3             # rounds of 补检索并重判 after round 2
OVERLAP = 0.5               # share of a chunk covered by seen windows that makes it seen
UNKNOWN = "unknown"
VERDICTS = ("正确", "错误")

# ---------------------------------------------------------------------- prompts
QUERY_RULE = ("query 是关键词串，不是问句，也不要照抄整个<W>：写“主体简称或产品名 年份或期间 指标名或条款名”"
              "（如“宁德时代 2025年 营业收入”“国寿增益宝 身故保险金”），用原文里可能的叫法；同一项内容可以再用表名、章节名或同义说法"
              "另写一个调用（如“主要会计数据 营业收入”“合并现金流量表 经营活动产生的现金流量净额”）。"
              "不要写“年度报告”“年报”“条款”这类每一页都有的词。")

SYSTEM_PLAN = """你是金融文档问答助手。这一步只规划检索，不作答。系统会按你给出的检索调用在原文里做关键词检索（BM25），每个调用返回最相关的 <TOP> 个原文块，下一步你再依据这些原文逐个<W>作答。
本题涉及的文档：
<DOCS>
要求：
1. 逐个<W>想清楚：判断它需要文档里的哪些数据或原文（报表数字、指标、条款、承诺、事实）。比较类的说法（“均”“都”“完全一致”“高于”），每个被比较的主体、每个期间各查一次；需要计算的值，查计算所需的原始数据。
2. 每个检索调用写 query 和 options（它服务于哪些<W>）。几个<W>要看的是同一处原文，就只写一个调用，options 里列出这几个<W>。
3. <QUERY_RULE>
4. 每个<W>至少有一个检索调用，全部调用不超过 <MAX> 个。
只输出一个 JSON：{"searches": [{"query": "宁德时代 2025年 研发费用", "options": ["<K1>", "<K2>"]}, {"query": "...", "options": ["<K1>"]}]}"""

JUDGE_RULES = """1. 只依据给出的原文块判断，不用自己的知识补数据。verdict 取：
   "正确"：说法与原文一致（需要计算的，按原文的数算出的结果与说法一致）；
   "错误"：说法与原文矛盾，或者原文对同一件事写的是别的内容；
   "unknown"：给出的原文不足以下结论（缺数据、缺条款，或主体、期间对不上）。
2. 结论只表示说法与原文是否一致。题干问的是正确还是错误的选项都不影响结论，选哪些由系统按题干决定；每个<W>单独判断，不要拿<W>互相比较来定结论。
3. 核对主体、期间、口径（合并或母公司）和单位。需要计算的写 calc：用到的原文数、算式和结果；说法里带“约”的数按它写出的小数位比较。不需要计算的 calc 写空字符串。
4. 每个<W>按 cite、calc、reason、verdict 的顺序写：先写 cite（判断所依据的块编号；unknown 的也写已经找到的部分依据，没有就写空列表），再写 calc，再用 reason 一句话说明依据哪句原文或哪个数、与说法是否一致，最后按 reason 写 verdict。"""

UNKNOWN_RULE_FIRST = """5. verdict 为 unknown 的<W>：先写 need（还缺什么：哪个主体、哪个期间、哪项数据或哪一条规定），再写 lean（假如最后还是找不到，你倾向于判 "正确" 还是 "错误"）。然后在 searches 里为这些<W>给出新的检索调用，每个调用写明 options。<QUERY_RULE>换表名、章节名、科目名或同义说法，不要重复已经检索过的词。没有 unknown 时 searches 写空列表。"""
UNKNOWN_RULE_RETRY = """5. 仍然判断不了的（unknown）：先写 need（还缺什么），再写 lean（假如最后还是找不到，你倾向于判 "正确" 还是 "错误"），然后在 searches 里给出新的检索调用，每个调用写明 options。要换检索词：换表名、章节名、科目名、同义说法，或换一种主体、期间的写法，不要重复“已经用过的检索词”。<QUERY_RULE>都有结论时 searches 写空列表。"""

OUT_JUDGE = ('{"answers": {"<K1>": {"cite": ["E2"], "calc": "", "reason": "...", "verdict": "正确"}, '
             '"<K2>": {"cite": ["E5"], "calc": "", "reason": "...", "verdict": "unknown", "need": "...", "lean": "错误"}}, '
             '"searches": [{"query": "...", "options": ["<K2>"]}]}')

SYSTEM_JUDGE = """你是金融文档问答助手。系统已按检索调用在文档里检索到一批原文块（每块开头是块编号、文档名和页码），请逐个<W>作答。
""" + JUDGE_RULES + "\n" + UNKNOWN_RULE_FIRST + """
只输出一个 JSON，answers 里每个<W>都要有：""" + OUT_JUDGE

SYSTEM_RETRY = """你是金融文档问答助手。这道题已经作答过，还有<W>没有结论。下面给出：之前的作答结果，之前作答时引用过的原文块，按上次的检索调用新检索到的原文块（之前看过的块不会再给出）。
只判断“还没有结论的<W>”，已有结论的不再作答。
""" + JUDGE_RULES + "\n" + UNKNOWN_RULE_RETRY + """
只输出一个 JSON，answers 里只写还没有结论的<W>：""" + OUT_JUDGE

SYSTEM_LEAN = """你是金融文档问答助手。下面这些<W>检索多次后仍没有结论。请结合已有原文和你自己的判断，按你的倾向给每个这样的<W>判 "正确" 或 "错误"，不能写 unknown。
只输出一个 JSON：{"lean": {"<K1>": "错误"}, "reason": {"<K1>": "一句话"}}"""


def fill_in(template, **kw):
    for k, v in kw.items():
        template = template.replace(f"<{k}>", str(v))
    return template


# ---------------------------------------------------------------------- 本地去重
class Seen:
    """The chunks this question has seen, as windows (doc, pages, chars) over the document's
    full text: elements.jsonl in reading order joined by newlines. A chunk is seen when the
    windows of its document cover OVERLAP of its own span."""

    def __init__(self, overlap=OVERLAP):
        self.overlap = overlap
        self.windows = defaultdict(list)            # doc -> [(lo, hi, uid)]
        self._layout = {}                           # doc -> ({element id: (lo, hi)}, {table element id: rows})

    def layout(self, doc):
        if doc.doc_id not in self._layout:
            off, pos = {}, 0
            for r in doc.records:
                n = len(r.get("text") or "")
                off[r["id"]] = (pos, pos + n)
                pos += n + 1
            rows = {}
            for c in doc.chunks:
                if c.get("rows"):
                    m = c["members"][0]
                    rows[m] = max(rows.get(m, 0), c["rows"][1] + 1)
            self._layout[doc.doc_id] = (off, rows)
        return self._layout[doc.doc_id]

    def key(self, uid):
        """{"doc", "pages", "chars"} of one chunk."""
        doc = get_doc(uid.split("#")[0])
        c = doc.by_id[uid]
        off, rows = self.layout(doc)
        spans = [off[m] for m in c["members"] if m in off] or [(0, len(c["text"]))]
        lo, hi = spans[0][0], spans[-1][1]
        if c.get("span"):                           # a piece of a long paragraph or clause
            lo, hi = lo + c["span"][0], min(hi, lo + c["span"][1])
        elif c.get("rows") and rows.get(c["members"][0]):   # a piece of a table: its rows' share
            n, (a, b) = rows[c["members"][0]], c["rows"]
            lo, hi = lo + (hi - lo) * a // n, lo + (hi - lo) * (b + 1) // n
        return {"doc": doc.doc_id, "pages": list(c["pages"]), "chars": [lo, max(hi, lo + 1)]}

    def covered(self, k):
        lo, hi = k["chars"]
        segs = sorted((max(lo, a), min(hi, b)) for a, b, _ in self.windows[k["doc"]] if a < hi and b > lo)
        got, end = 0, lo
        for a, b in segs:
            a = max(a, end)
            if b > a:
                got, end = got + b - a, b
        return got / (hi - lo)

    def admit(self, uid):
        """(True, key) and the window recorded when the chunk is new; (False, key) when seen."""
        k = self.key(uid)
        if self.covered(k) >= self.overlap:
            return False, k
        self.windows[k["doc"]].append((k["chars"][0], k["chars"][1], uid))
        return True, k


# ---------------------------------------------------------------------- parsing the model's output
def letters(x, allowed):
    """Option keys named in x (a list, "A、B", "AB"), kept in allowed."""
    if isinstance(x, (list, tuple)):
        x = " ".join(str(v) for v in x)
    return [k for k in dict.fromkeys(re.findall(r"[A-Z]", str(x or "").upper())) if k in allowed]


def norm_query(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def parse_searches(data, allowed, used):
    """([{"query", "options"}], skipped): the search calls of a reply for the allowed options,
    at most MAX_SEARCHES; a query already run (used) or repeated is skipped. A call naming no
    allowed option serves all of them."""
    out, skipped = [], []
    raw = data.get("searches") if isinstance(data, dict) else None
    for s in raw if isinstance(raw, list) else []:
        if isinstance(s, str):
            s = {"query": s}
        if not isinstance(s, dict):
            continue
        query = norm_query(s.get("query"))
        if not query:
            continue
        if query in used or any(query == x["query"] for x in out):
            skipped.append(query)
            continue
        out.append({"query": query, "options": letters(s.get("options"), allowed) or sorted(allowed)})
    return out[:MAX_SEARCHES], skipped + [x["query"] for x in out[MAX_SEARCHES:]]


def norm_verdict(v):
    v = str(v or "").strip()
    if v.startswith("正确") or v in ("对", "一致", "true", "True"):
        return "正确"
    if v.startswith("错误") or v in ("错", "不正确", "不一致", "false", "False"):
        return "错误"
    return UNKNOWN


def parse_answers(data, keys, labels, rnd):
    """{key: answer} for the keys the reply answered; an answer is {verdict, cite, calc,
    reason, need, lean, round}. Cites outside the context are dropped."""
    got = data.get("answers") if isinstance(data, dict) else None
    out = {}
    for k in keys:
        a = got.get(k) if isinstance(got, dict) else None
        if not isinstance(a, dict):
            continue
        verdict = norm_verdict(a.get("verdict"))
        lean = norm_verdict(a.get("lean"))
        out[k] = {"verdict": verdict,
                  "cite": [c for c in dict.fromkeys(re.findall(r"E\d+", json.dumps(a.get("cite") or [], ensure_ascii=False).upper())) if c in labels],
                  "calc": str(a.get("calc") or "").strip(), "reason": str(a.get("reason") or "").strip(),
                  "need": str(a.get("need") or "").strip() if verdict == UNKNOWN else "",
                  "lean": lean if verdict == UNKNOWN and lean != UNKNOWN else "", "round": rnd}
    return out


# ---------------------------------------------------------------------- one question
def main(argv):
    qids = [a for a in argv if not a.startswith("--")]
    qs = {x["qid"]: x for x in load_b_questions()}
    for qid in qids:
        solve(qs[qid])


def solve(q, spent=None):
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
    calls, rounds = [], []
    seen, store, used = Seen(), {}, set()               # store: label -> chunk shown; used: queries already run
    answers = {}                                        # key -> latest answer
    history = []                                        # every answer of every round

    def call(stage, system, user, max_tokens):
        content, u = llm.chat(system, user, max_tokens=max_tokens, tag=f"{qid}:{stage}")
        usage[0] += u["prompt_tokens"]
        usage[1] += u["completion_tokens"]
        calls.append({"stage": stage, "system": system, "user": user, "response": content, "usage": u})
        print(f"--- 调用 {len(calls)}：{stage}（输入 {u['prompt_tokens']}，输出 {u['completion_tokens']}）")
        return content

    def ask_json(stage, system, user, max_tokens, ok):
        """One call; once more when the reply holds no usable JSON (cut off, prose only)."""
        data = {}
        for attempt in range(2):
            data = parse_json(call(stage if attempt == 0 else stage + "_again", system,
                                   user if attempt == 0 else user + "\n\n（上一次的输出不是完整的 JSON。请重新输出完整的 JSON，reason 和 calc 写短一些。）", max_tokens))
            if ok(data):
                return data
        return data

    def search(searches, rnd, context=()):
        """Run the search calls; (results, new labels). Hits merged round-robin over the calls,
        checked against the seen set in that order, at most MAX_NEW new chunks. A hit already
        seen is "shown" when it stays in the context (cited before), else "seen" (not shown again)."""
        results = []
        for s in searches:
            where, text = narrow(q, s["query"], sc)
            hits = [u for u, score in bm25_scores(text, where)[:TOP_PER_QUERY] if score > 0]
            results.append({**s, "searched": text.strip(), "docs": len(where), "hits": hits, "status": {}})
            used.add(s["query"])
        label_of = {v["uid"]: k for k, v in store.items()}
        new = []
        for pos in range(TOP_PER_QUERY):
            for r in results:
                if pos >= len(r["hits"]):
                    continue
                u = r["hits"][pos]
                if u in label_of:                       # given a label before: this round, or earlier
                    lab = label_of[u]
                    r["status"][u] = (("shown", lab) if lab in context else ("new", lab) if store[lab]["round"] == rnd else ("seen", None))
                    continue
                if len(new) >= MAX_NEW:
                    r["status"][u] = ("cut", None)
                    continue
                fresh, k = seen.admit(u)
                if not fresh:
                    r["status"][u] = ("seen", None)
                    continue
                lab = f"E{len(store) + 1}"
                x = get_doc(k["doc"]).by_id[u]
                store[lab] = {"uid": u, "doc": k["doc"], "title": title.get(k["doc"], k["doc"]), "pages": k["pages"],
                              "chars": k["chars"], "path": x.get("path") or "", "text": x["body"][:MAX_BODY], "round": rnd}
                label_of[u] = lab
                new.append(lab)
                r["status"][u] = ("new", lab)
        return results, new

    def show_chunk(lab):
        s = store[lab]
        p = s["pages"]
        page = f"第{p[0]}页" if p[0] == p[-1] else f"第{p[0]}-{p[-1]}页"
        return f"[{lab}]《{s['title']}》{page}" + (f" | {s['path']}" if s["path"] else "") + "\n" + s["text"]

    def search_lines(results):
        lines = []
        for i, r in enumerate(results, 1):
            st = list(r["status"].values())
            parts = []
            for tag, name in (("new", "新块"), ("shown", "已在上文")):
                labs = [lab for t, lab in st if t == tag]
                if labs:
                    parts.append(name + " " + "、".join(labs))
            n_seen, n_cut = sum(t == "seen" for t, _ in st), sum(t == "cut" for t, _ in st)
            if n_seen:
                parts.append(f"{n_seen} 块之前看过、不再给出")
            if n_cut:
                parts.append(f"{n_cut} 块超出本轮上限、没有给出")
            lines.append(f"q{i}「{r['query']}」（{'、'.join(r['options'])}）→ " + ("；".join(parts) or "没有命中"))
        return "\n".join(lines)

    def answer_lines(only=None):
        out = []
        for k in keys:
            a = answers.get(k)
            if only and k not in only:
                continue
            if not a:
                out.append(f"{k}：还没有作答")
                continue
            cite = "、".join(a["cite"]) or "无"
            line = f"{k}：{'没有结论（unknown）' if a['verdict'] == UNKNOWN else a['verdict']}（第 {a['round']} 轮，依据 {cite}）{a['reason']}"
            if a["calc"]:
                line += f"；计算：{a['calc']}"
            if a["verdict"] == UNKNOWN and a["need"]:
                line += f"；还缺：{a['need']}"
            out.append(line)
        return "\n".join(out)

    def record(rnd, data, asked_keys):
        got = parse_answers(data, asked_keys, set(store), rnd)
        for k in asked_keys:
            a = got.get(k) or {"verdict": UNKNOWN, "cite": (answers.get(k) or {}).get("cite", []), "calc": "",
                               "reason": "模型没有给出这个" + word + "的作答", "need": "", "lean": "", "round": rnd}
            answers[k] = a
            history.append({"key": k, **a})
        return [k for k in keys if answers[k]["verdict"] == UNKNOWN]

    stem = q["question"].strip() + ("" if tf else "".join(f"\n{k}. {v}" for k, v in targets.items()))
    if tf:
        stem += f"\n\n要判断的表述（记作 T）：{targets['T']}"
    many = len(sc["docs"]) > 20
    docs = "\n".join(f"- 《{title.get(d, d)}》" for d in sc["docs"][:20]) + (f"\n（共 {len(sc['docs'])} 份，题面没有点名文档，检索在这些文档里进行）" if many else "")
    k1, k2 = keys[0], keys[1] if len(keys) > 1 else keys[0]
    common = {"W": word, "TOP": TOP_PER_QUERY, "MAX": MAX_SEARCHES, "DOCS": docs, "K1": k1, "K2": k2}
    print(f"{qid}（{kind}）{stem}\n" + ("" if tf else f"方向：选判为“{asked}”的选项\n"))

    # ------------------------------------------------------------------ 第 1 轮：规划并发出检索
    system = fill_in(fill_in(SYSTEM_PLAN, QUERY_RULE=QUERY_RULE), **common)
    plan = ask_json("plan", system, stem, 1500, lambda d: isinstance(d.get("searches"), list) and d["searches"])
    searches, skipped = parse_searches(plan, set(keys), used)
    lacking = [k for k in keys if not any(k in s["options"] for s in searches)]
    for k in lacking:                                   # an option without a search call: its own text as the query
        searches.append({"query": norm_query(targets[k]), "options": [k], "added_by": "代码（模型没有为它写检索调用）"})
    results, new = search(searches, 1)
    rounds.append({"round": 1, "stage": "plan", "searches": results, "skipped": skipped, "new": new, "context": list(new)})
    print(f"第 1 轮：{len(searches)} 个检索调用，新块 {len(new)}" + (f"；代码补检索的{word} {lacking}" if lacking else ""))
    print(search_lines(results))

    # ------------------------------------------------------------------ 第 2 轮：逐选项作答
    system = fill_in(fill_in(SYSTEM_JUDGE, QUERY_RULE=QUERY_RULE), **common)
    user = (f"题目：\n{stem}\n\n检索调用与结果：\n{search_lines(results)}\n\n原文块（共 {len(new)} 块）：\n\n"
            + ("\n\n".join(show_chunk(lab) for lab in new) or "（没有检索到原文块）"))
    data = ask_json("judge", system, user, 4000, lambda d: isinstance(d.get("answers"), dict) and d["answers"])
    open_keys = record(2, data, keys)
    pending, skipped = parse_searches(data, set(open_keys), used)
    rounds.append({"round": 2, "stage": "judge", "context": list(new), "answered": keys, "open": open_keys,
                   "next_searches": pending, "skipped": skipped})
    print(f"第 2 轮作答：\n{answer_lines()}")

    # ------------------------------------------------------------------ 第 3 轮起：补检索并重判
    stop, retries = "", 0
    system = fill_in(fill_in(SYSTEM_RETRY, QUERY_RULE=QUERY_RULE), **common)
    while True:
        open_keys = [k for k in keys if answers[k]["verdict"] == UNKNOWN]
        if not open_keys:
            stop = f"所有{word}都有结论"
            break
        if retries >= MAX_RETRIES:
            stop = f"重试用完（{MAX_RETRIES} 次）"
            break
        if not pending:
            stop = "模型没有给出新的检索调用"
            break
        retries += 1
        rnd = 2 + retries
        cited = sorted({c for a in answers.values() for c in a["cite"]}, key=lambda s: int(s[1:]))
        results, new = search(pending, rnd, set(cited))
        row = {"round": rnd, "stage": "retry", "searches": results, "new": new}
        rounds.append(row)
        print(f"第 {rnd} 轮检索：{len(pending)} 个检索调用，新块 {len(new)}")
        print(search_lines(results))
        if not new:
            stop = "去重后没有新原文块"
            row["context"] = []
            break
        context = cited + [lab for lab in new if lab not in cited]
        user = (f"题目：\n{stem}\n\n之前的作答：\n{answer_lines()}\n\n还没有结论的{word}：{'、'.join(open_keys)}\n\n"
                f"已经用过的检索词：{'；'.join(sorted(used))}\n\n上次的检索调用与结果：\n{search_lines(results)}\n\n"
                f"原文块（之前引用过的 {len(cited)} 块，新检索到的 {len(new)} 块）：\n\n" + "\n\n".join(show_chunk(lab) for lab in context))
        data = ask_json(f"retry{retries}", system, user, 3000, lambda d: isinstance(d.get("answers"), dict) and d["answers"])
        still = record(rnd, data, open_keys)
        pending, skipped = parse_searches(data, set(still), used)
        row.update({"context": context, "cited_before": cited, "answered": open_keys, "open": still,
                    "next_searches": pending, "skipped": skipped})
        print(f"第 {rnd} 轮作答：\n{answer_lines(open_keys)}")
    print(f"停止：{stop}")

    # ------------------------------------------------------------------ 兜底：仍为 unknown 的按倾向作答
    open_keys = [k for k in keys if answers[k]["verdict"] == UNKNOWN]
    no_lean = [k for k in open_keys if not answers[k]["lean"]]
    lean_call = None
    if no_lean:
        cited = sorted({c for a in answers.values() for c in a["cite"]}, key=lambda s: int(s[1:]))
        user = (f"题目：\n{stem}\n\n之前的作答：\n{answer_lines()}\n\n要按倾向作答的{word}：{'、'.join(no_lean)}\n\n"
                f"之前引用过的原文块（{len(cited)} 块）：\n\n" + ("\n\n".join(show_chunk(lab) for lab in cited) or "（无）"))
        data = ask_json("lean", fill_in(SYSTEM_LEAN, **common), user, 600, lambda d: isinstance(d.get("lean"), dict))
        leans = data.get("lean") if isinstance(data.get("lean"), dict) else {}
        reasons = data.get("reason") if isinstance(data.get("reason"), dict) else {}
        lean_call = {k: {"lean": norm_verdict(leans.get(k)), "reason": str(reasons.get(k) or "")} for k in no_lean}
        for k in no_lean:
            if lean_call[k]["lean"] != UNKNOWN:
                answers[k]["lean"] = lean_call[k]["lean"]
    final = {k: (a["verdict"] if a["verdict"] != UNKNOWN else (a["lean"] or None)) for k, a in answers.items()}
    leaned = [k for k in open_keys if final[k]]

    # ------------------------------------------------------------------ 选项
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
           "model": llm.model, "direction": {"asked": asked, "clause": clause, "word": dword},
           "settings": {"top_per_query": TOP_PER_QUERY, "max_searches": MAX_SEARCHES, "max_new": MAX_NEW,
                        "max_body": MAX_BODY, "max_retries": MAX_RETRIES, "overlap": OVERLAP},
           "rounds": rounds, "retries": retries, "stop": stop, "answers_by_option": answers, "history": history,
           "lean_call": lean_call, "verdicts": final, "leaned": leaned, "answer": answer, "answered_by": how, "caveat": note,
           "calls": calls, "n_calls": len(calls), "usage": dict(llm.usage), "seconds": round(time.time() - t0, 1),
           "store": store, "seen": {d: [list(w) for w in ws] for d, ws in seen.windows.items()}}
    base = ROOT / "output" / NAME / f"{NAME}_{qid}"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    base.with_suffix(".md").write_text(report(out), encoding="utf-8")
    u = llm.usage
    print(f"{len(calls)} 次调用，重试 {retries} 次，看过 {len(store)} 块，tokens {u['prompt_tokens']} + {u['completion_tokens']} = {u['total_tokens']}，"
          f"{out['seconds']} 秒 -> {base.relative_to(ROOT)}.json / .md")
    return out


# ---------------------------------------------------------------------- record
def report(o):
    """The run as a Markdown record: every call in full, the searches and the dedup between them."""
    u = o["usage"]
    L = [f"# {o['qid']}：选择题新策略（检索规划 → 逐选项作答 → 补检索重判）作答记录", "",
         f"{o['type']}。模型 {o['model']}，温度 0，思考模式关闭。{o['n_calls']} 次模型调用，重试 {o['retries']} 次，{o['seconds']} 秒；"
         f"tokens 输入 {u['prompt_tokens']} + 输出 {u['completion_tokens']} = {u['total_tokens']}。停止原因：{o['stop']}。",
         f"答案：**{o['answer'] or '（空）'}**（{o['answered_by']}）。" + (f"注意：{o['caveat']}。" if o["caveat"] else ""), "",
         "## 题目", "", "```text", o["question"] + "".join(f"\n{k}. {v.strip()}" for k, v in o["options"].items()), "```", "",
         "## 最终结论", "", "| 选项 | 结论 | 来源 | 依据 | 理由 |", "|---|---|---|---|---|"]
    for k, a in o["answers_by_option"].items():
        src = f"第 {a['round']} 轮原文" if a["verdict"] != UNKNOWN else ("模型倾向" if o["verdicts"].get(k) else "没有结论")
        L.append(f"| {k} | {o['verdicts'].get(k) or '—'} | {src} | {'、'.join(a['cite'])} | {(a['reason'] + (' 计算：' + a['calc'] if a['calc'] else '')).replace('|', '/')} |")
    L += ["", "## 逐轮过程", ""]
    calls = {c["stage"]: c for c in o["calls"]}

    def show(stage):
        out = []
        for st in (stage, stage + "_again"):
            c = calls.get(st)
            if c:
                out += [f"### 调用：{st}（输入 {c['usage']['prompt_tokens']}，输出 {c['usage']['completion_tokens']}）", "",
                        "<details><summary>系统提示</summary>", "", "```text", c["system"], "```", "", "</details>", "",
                        "<details><summary>发给模型的内容</summary>", "", "```text", c["user"], "```", "", "</details>", "",
                        "**模型输出（原样）：**", "", "```text", c["response"], "```", ""]
        return out

    def searches(r):
        out = ["| 调用 | 检索词 | 服务选项 | 实际检索的文本 | 文档数 | 返回（去重结果） |", "|---|---|---|---|---|---|"]
        for i, s in enumerate(r["searches"], 1):
            got = "，".join(f"{u.split('#')[-1]}：{ {'new': '新 ', 'shown': '已在上文 ', 'seen': '见过', 'cut': '超上限'}[t] }{lab or ''}"
                           for u, (t, lab) in s["status"].items()) or "没有命中"
            out.append(f"| q{i} | {s['query']} | {'、'.join(s['options'])}{'（' + s['added_by'] + '）' if s.get('added_by') else ''} | {s['searched']} | {s['docs']} | {got} |")
        return out + [""]

    retry = 0
    for r in o["rounds"]:
        if r["stage"] == "plan":
            L += ["### 第 1 轮：规划并发出检索", ""] + show("plan") + ["**代码检索：**", ""] + searches(r)
            L += [f"新块 {len(r['new'])} 个：{'、'.join(r['new']) or '无'}。" + (f"跳过重复的检索词：{'；'.join(r['skipped'])}。" if r["skipped"] else ""), ""]
        elif r["stage"] == "judge":
            L += ["### 第 2 轮：逐选项作答", ""] + show("judge")
            L += [f"仍为 unknown：{'、'.join(r['open']) or '无'}；下一轮的检索调用 {len(r['next_searches'])} 个。", ""]
        else:
            retry += 1
            L += [f"### 第 {r['round']} 轮：补检索并重判（重试 {retry}）", "", "**代码检索：**", ""] + searches(r)
            if not r["new"]:
                L += ["去重后没有新原文块，结束重试。", ""]
                continue
            L += [f"上下文：之前引用过的 {len(r['cited_before'])} 块（{'、'.join(r['cited_before']) or '无'}）+ 新块 {len(r['new'])} 块（{'、'.join(r['new'])}）。", ""]
            L += show(f"retry{retry}") + [f"仍为 unknown：{'、'.join(r['open']) or '无'}；下一轮的检索调用 {len(r['next_searches'])} 个。", ""]
    if o["lean_call"]:
        L += ["### 兜底：按倾向作答", ""] + show("lean")
    L += ["## 看过的原文块", "", "| 编号 | 块 | 页 | 字符区间 | 轮 |", "|---|---|---|---|---|"]
    L += [f"| {lab} | {s['uid']} | {s['pages'][0]}-{s['pages'][-1]} | {s['chars'][0]}-{s['chars'][1]} | {s['round']} |" for lab, s in o["store"].items()]
    return "\n".join(L)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
