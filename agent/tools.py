# -*- coding: utf-8 -*-
"""Tools for the models, in OpenAI function-calling format (the Bailian endpoint takes the
same `tools` list), and the plain Python functions behind them for the workflow to call.

  grep -- search the original text of the question's documents, unit by unit (clause,
          paragraph, table, figure); regex or literal; counts or matching snippets
  calc -- exact arithmetic over named expressions with a safe AST evaluator (never eval)

Model side:
    ctx = ToolContext(question["doc_ids"])          # the documents are D1, D2, ... in prompts
    client.chat.completions.create(..., tools=TOOL_LIST)
    text = run_tool(call.function.name, call.function.arguments, ctx)   # role="tool" content

Workflow side:
    grep("受益所有人身份信息", doc_ids, literal=True)["hits"]
    calc({"身故保险金": "max(1.6 * 已交保费, 现金价值)"}, {"已交保费": 100, "现金价值": 80})
"""
import ast
import json
import math
import operator as op
import time

import regex

from .retrieval import get_doc

MAX_PATTERN = 200         # characters of a grep pattern
SEARCH_TIMEOUT = 2.0      # seconds for one grep over all units of the question
MAX_HITS = 30             # snippets one grep may return
MAX_EXPR = 500            # characters of one calc expression
MAX_POWER = 100           # largest exponent calc accepts


# ----------------------------------------------------------------------------- context
class ToolContext:
    """The documents a tool call may touch (those of the current question), with the short
    aliases D1, D2, ... that the prompts and the tool outputs use instead of long ids."""

    def __init__(self, doc_ids):
        self.doc_ids = list(doc_ids)
        self.alias = {f"D{i}": d for i, d in enumerate(self.doc_ids, 1)}
        self.short = {d: a for a, d in self.alias.items()}

    def resolve(self, doc):
        """D1 / a full doc id / None -> a doc id of the question, or raise ValueError."""
        if not doc:
            return None
        if doc in self.alias:
            return self.alias[doc]
        if doc in self.doc_ids:
            return doc
        raise ValueError(f"文档 {doc} 不属于本题，可选：{', '.join(self.alias)}")

    def describe(self):
        """One line per document for the system prompt."""
        return "\n".join(f"{a}：《{get_doc(d).title}》" for a, d in self.alias.items())


# ----------------------------------------------------------------------------- grep
def literal_pattern(text):
    """A literal phrase as a regex that tolerates spaces the PDF parser left between characters."""
    return r"\s*".join(regex.escape(c) for c in text if not c.isspace())


def _table_rows(text, rx):
    """Caption, header, separator and source lines of a markdown table, plus the matching rows."""
    rows = text.split("\n")
    start = next((i for i, r in enumerate(rows) if r.startswith("|")), None)
    if start is None:
        return None
    keep = rows[:start + 2]
    data = [r for r in rows[start + 2:] if r.startswith("|")]
    hit = [r for r in data if rx.search(r)]
    keep += hit if hit else data[:5] + (["…"] if len(data) > 5 else [])
    keep += [r for r in rows[start + 2:] if not r.startswith("|")]
    return "\n".join(keep)


def _snippet(unit, match, rx, window):
    text = unit["body"]
    if unit["type"] == "table":
        rows = _table_rows(text, rx)
        if rows is not None:
            return rows[:1500]
    s, e = match.span()
    return ("…" if s > window else "") + text[max(0, s - window):e + window] + ("…" if e + window < len(text) else "")


_PAGED = {}


def _paged(doc_id):
    """False for documents without page information (TXT sources put every unit on page 1)."""
    if doc_id not in _PAGED:
        _PAGED[doc_id] = len({u["page"] for u in get_doc(doc_id).records}) > 1
    return _PAGED[doc_id]


def grep(pattern, doc_ids, literal=False, doc=None, mode="content", window=80, max_hits=10):
    """Search the units of doc_ids (optionally one of them, by alias or id).
    Returns {"pattern", "total", "per_doc", "hits", "shown", "truncated"} or {"error"}.
    Each hit: unit_id (full), doc_id, unit (local id), doc_title, section, page, type,
    match (the matched text), text (snippet; table: header + matching rows)."""
    ctx = doc_ids if isinstance(doc_ids, ToolContext) else ToolContext(doc_ids)
    if not pattern or not str(pattern).strip():
        return {"error": "pattern 为空"}
    if len(pattern) > MAX_PATTERN:
        return {"error": f"pattern 过长（超过 {MAX_PATTERN} 字符）"}
    try:
        scope = [ctx.resolve(doc)] if doc else ctx.doc_ids
    except ValueError as e:
        return {"error": str(e)}
    source = literal_pattern(pattern) if literal else pattern
    try:
        rx = regex.compile(source, regex.IGNORECASE)
    except regex.error as e:
        return {"error": f"正则无法编译：{e}；只想按字面搜索时请设 literal=true"}

    deadline = time.monotonic() + SEARCH_TIMEOUT
    found, per_doc = [], {}
    for d in scope:
        n = 0
        for u in get_doc(d).units:
            left = deadline - time.monotonic()
            try:
                if left <= 0:
                    raise TimeoutError
                m = rx.search(u["body"], timeout=left)
            except TimeoutError:
                return {"error": f"搜索超过 {SEARCH_TIMEOUT:g} 秒，请换一个更简单的正则"}
            if m:
                n += 1
                found.append((d, u, m))
        per_doc[ctx.short[d]] = n

    max_hits = max(1, min(int(max_hits or 10), MAX_HITS))
    shown = found if mode == "count" else found[:max_hits]
    hits = []
    for d, u, m in shown:
        h = {"unit_id": u["id"], "doc_id": d, "doc": ctx.short[d], "unit": u["id"].split("#", 1)[1],
             "doc_title": get_doc(d).title, "section": u["path"] or "", "page": u["page"] if _paged(d) else None,
             "type": u["type"],
             "match": m.group(0)}
        if mode != "count":
            h["text"] = _snippet(u, m, rx, max(10, min(int(window or 80), 300)))
        hits.append(h)
    return {"pattern": pattern, "literal": bool(literal), "total": len(found), "per_doc": per_doc,
            "hits": hits, "shown": len(hits), "truncated": len(found) > len(hits)}


def render_grep(res):
    """Compact text of a grep result for a role="tool" message."""
    if "error" in res:
        return "错误：" + res["error"]
    dist = "，".join(f"{a} {n} 条" for a, n in res["per_doc"].items())
    head = f"「{res['pattern']}」共命中 {res['total']} 条（{dist}）"
    if res["total"] == 0:
        return head + "。可以换同义说法、去掉修饰词，或改用 literal=true。"
    if all("text" not in h for h in res["hits"]):                       # mode=count
        units = {}
        for h in res["hits"]:
            units.setdefault(h["doc"], []).append(h["unit"])
        return head + "\n" + "\n".join(f"{a}：{', '.join(u[:50])}" for a, u in units.items())
    lines = [head + (f"，以下是前 {res['shown']} 条；命中太多时请换更具体的词" if res["truncated"] else "")]
    for h in res["hits"]:
        page = f" | 第{h['page']}页" if h["page"] else ""
        lines.append(f"[{h['doc']} {h['unit']} | {h['section'] or '（无小节）'}{page}] " + h["text"].replace("\n", " / "))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- calc
_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow}
_UNARY = {ast.USub: op.neg, ast.UAdd: op.pos, ast.Not: op.not_}
_CMP = {ast.Lt: op.lt, ast.LtE: op.le, ast.Gt: op.gt, ast.GtE: op.ge, ast.Eq: op.eq, ast.NotEq: op.ne}
_FUNCS = {"max": max, "min": min, "round": round, "abs": abs, "sum": sum}


def _eval(node, names):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id in ("True", "False"):
            return node.id == "True"
        raise ValueError(f"未定义的变量：{node.id}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _eval(node.left, names), _eval(node.right, names)
        if isinstance(node.op, ast.Pow) and abs(b) > MAX_POWER:
            raise ValueError(f"指数过大：{b}")
        return _BIN[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand, names))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, names)
        for o, c in zip(node.ops, node.comparators):
            if type(o) not in _CMP:
                raise ValueError("不支持的比较运算")
            right = _eval(c, names)
            if not _CMP[type(o)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, names) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.IfExp):
        return _eval(node.body, names) if _eval(node.test, names) else _eval(node.orelse, names)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, names) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a, names) for a in node.args])
    raise ValueError(f"不支持的写法：{ast.unparse(node)[:60]}")


def calc(expressions, variables=None):
    """expressions: one expression, or {name: expression} evaluated in order; each result is
    usable by the later expressions under its name. variables: known numbers.
    Returns {"results": {name: value}} or {"results": <those computed so far>, "error": ...}."""
    if isinstance(expressions, str):
        expressions = {"结果": expressions}
    names = {}
    for k, v in (variables or {}).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return {"results": {}, "error": f"变量 {k} 不是数字：{v!r}"}
        names[str(k)] = v
    results = {}
    for name, expr in expressions.items():
        expr = str(expr).strip()
        try:
            if len(expr) > MAX_EXPR:
                raise ValueError(f"表达式过长（超过 {MAX_EXPR} 字符）")
            value = _eval(ast.parse(expr, mode="eval").body, names)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("结果不是有限数")
        except ZeroDivisionError:
            return {"results": results, "error": f"{name}：除数为 0"}
        except SyntaxError:
            return {"results": results, "error": f"{name}：表达式有语法错误（百分数请写成小数，如 3.6% 写 0.036）：{expr}"}
        except (ValueError, TypeError) as e:
            return {"results": results, "error": f"{name}：{e}"}
        names[str(name)] = results[name] = value
    return {"results": results}


def _fmt(v):
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{round(v, 6):g}" if abs(v) < 1e15 else f"{v:.6e}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def render_calc(res):
    lines = [f"{k} = {_fmt(v)}" for k, v in res["results"].items()]
    if "error" in res:
        lines.append("错误：" + res["error"])
    return "\n".join(lines)


# ----------------------------------------------------------------------------- tool list
GREP_SCHEMA = {"type": "function", "function": {
    "name": "grep",
    "description": ("在本题给定的文档原文中搜索，逐条匹配条款、段落和表格，返回命中位置和原文片段。"
                    "pattern 默认是正则表达式；只想按字面搜索时设 literal=true（会自动容忍字间空格）。"
                    "命中很多时先用 mode=count 看分布，再换更具体的词。用来核对原文是否确有某句话、某个数字，或查找证据中缺失的内容。"),
    "parameters": {"type": "object", "properties": {
        "pattern": {"type": "string", "description": "要搜索的词、短语或正则，例如 '1000\\s*美元' 或 '提前\\d+日'"},
        "literal": {"type": "boolean", "description": "true 表示按字面搜索，不解析正则符号；默认 false"},
        "doc": {"type": "string", "description": "只在某一篇文档里搜，填 D1、D2 等；不填则搜本题所有文档"},
        "mode": {"type": "string", "enum": ["content", "count"],
                 "description": "content 返回原文片段（默认）；count 只返回命中数和位置"},
        "window": {"type": "integer", "description": "命中处前后各取多少字，默认 80，最大 300"},
        "max_hits": {"type": "integer", "description": f"最多返回几条片段，默认 10，最大 {MAX_HITS}"}},
        "required": ["pattern"]}}}

CALC_SCHEMA = {"type": "function", "function": {
    "name": "calc",
    "description": ("精确计算，不要心算。expressions 是 {结果名: 表达式}，按顺序计算，后面的表达式可以引用前面的结果名。"
                    "支持 + - * / // % **、比较（> >= < <= == !=）、and/or/not、`x if 条件 else y`、"
                    "max/min/round/abs/sum。百分数写成小数（3.6% 写 0.036），金额先统一单位。"
                    "变量名可以用中文，但不能含标点或空格。"),
    "parameters": {"type": "object", "properties": {
        "expressions": {"type": "object", "additionalProperties": {"type": "string"},
                        "description": "例如 {\"身故保险金\": \"max(1.6 * 已交保费, 现金价值)\", \"差额\": \"身故保险金 - 已交保费\"}"},
        "variables": {"type": "object", "additionalProperties": {"type": "number"},
                      "description": "已知数，例如 {\"已交保费\": 100, \"现金价值\": 80}"}},
        "required": ["expressions"]}}}


def _run_grep(args, ctx):
    return render_grep(grep(args.get("pattern", ""), ctx, literal=bool(args.get("literal")), doc=args.get("doc"),
                            mode=args.get("mode") or "content", window=args.get("window") or 80,
                            max_hits=args.get("max_hits") or 10))


def _run_calc(args, ctx):
    return render_calc(calc(args.get("expressions") or {}, args.get("variables") or {}))


TOOLS = {"grep": (GREP_SCHEMA, _run_grep), "calc": (CALC_SCHEMA, _run_calc)}
TOOL_LIST = [schema for schema, _ in TOOLS.values()]


def run_tool(name, arguments, ctx):
    """Execute one tool call from the model; arguments is the JSON string (or dict) the model
    produced. Always returns text for a role="tool" message, never raises."""
    if name not in TOOLS:
        return f"错误：没有名为 {name} 的工具，可用：{', '.join(TOOLS)}"
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
    except json.JSONDecodeError as e:
        return f"错误：参数不是合法的 JSON：{e}"
    try:
        return TOOLS[name][1](args, ctx)
    except Exception as e:                                   # a tool must never crash the loop
        return f"错误：{name} 执行失败：{type(e).__name__}: {e}"
