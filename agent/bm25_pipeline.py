# -*- coding: utf-8 -*-
"""Body-only BM25 pipeline for true/false, single- and multiple-choice questions.

Per claim (the statement of a true/false question, or one option of a choice question):

  1. split at 、且 / ；同时 / 并且 / ； into sub-claims (tf_v2.split_claims)
  2. every sub-claim is one two-field BM25 query (body 0.7 + heading path 0.3, see
     retrieval.bm25_scores) with one index over all documents of the question; the merged
     top 3 with a positive score are kept, whichever document they come from
  3. the hits become evidence records, one per stretch of a section, each carrying the
     document title | notice and the heading path of the section. A chunk already packs its
     neighbouring paragraphs, so by default a hit stands alone (CONTEXT = 0) and is extended
     only where something says the evidence continues elsewhere (2026-10-03):
       * what a hit cites comes along: the clauses it names (依照第十二条) and the numbered
         tables / figures it points at (见表1), at most REFS_MAX, the ones sharing the most
         words with the claim first (chunk field `refs`, parsing.chunk)
       * hits of one section are one record: adjacent ones always, hits lying apart
         together with the chunks between them when the stretch from the first to the last
         holds at most SECTION_MAX characters (MERGE_SECTION)
     EXPAND = "section" (experiments) takes the whole leaf section instead, trimmed around
     the hits to about SECTION_MAX characters; uncited, unhit figures are skipped either way
  4. one call analyses the claim and one judgement call reads the records
     (claims.analyze_claim / judge); a figure among the evidence (hit or cited) is cut out
     of the PDF and attached to the judgement call as an image, at most three per call
     (agent.figures, 2026-09-26)
  5. a verdict of 证据不足 is tried once more with every hit widened by RETRY_CONTEXT
     neighbours on each side in its section, when that adds anything (2026-10-03)

No vectors, no rerank. Choice questions are decided from the per-option verdicts as in
mcq_pipeline / multi_pipeline, without their step 3.
"""
from .claims import analyze_claim, evidence_list, judge, letter_for, split_claims, statement_of, tokenize
from .decide import pick, pick_multi
from .figures import evidence_images
from .retrieval import bm25_scores, doc_header, get_doc

TOP_K = 3               # units kept per sub-claim, merged over the documents of the question
CONTEXT = 0             # units taken before and after a hit in the same section (EXPAND = "window"); 1 until
                        # 2026-10-01, when the units were single paragraphs: a chunk now packs its neighbours itself
EXPAND = "window"       # "window": CONTEXT neighbours; "section": the whole leaf section (tools/compare_expansion.py)
SECTION_MAX = 3000      # characters a section record may hold before it is trimmed around the hits
MAX_UNIT_CHARS = 1500   # text of one unit handed to the judge
MERGE_SECTION = True    # hits of one section lying apart become one record with the chunks between them (<= SECTION_MAX)
RETRY_CONTEXT = 1       # neighbours added on each side of every hit when the first verdict is 证据不足 (0: no second try)


# ----------------------------------------------------------------------------- retrieval
def retrieve(claims, doc_ids, top_k=TOP_K):
    """[(unit_id, sub-claim index, rank, score)]: the top_k units of every sub-claim by
    body-only BM25, first occurrence kept when two sub-claims hit the same unit."""
    hits, seen = [], set()
    for i, claim in enumerate(claims):
        ranking = [(uid, s) for uid, s in bm25_scores(claim, doc_ids) if s > 0]
        for rank, (uid, score) in enumerate(ranking[:top_k], 1):
            if uid not in seen:
                seen.add(uid)
                hits.append((uid, i, rank, score))
    return hits


def clip(text, qtokens, limit=MAX_UNIT_CHARS):
    """One unit's text for the judge, about `limit` characters at most. A long table keeps
    its caption, header row, source line and the data rows sharing the most words with
    the claim, in table order."""
    if len(text) <= limit:
        return text
    rows = text.split("\n")
    start = next((i for i, r in enumerate(rows) if r.startswith("|")), None)
    if start is None or len(rows) < start + 3:
        return text[:limit]
    head = rows[:start + 2]                                   # caption lines, header, separator
    data = [(i, r) for i, r in enumerate(rows) if i >= start + 2 and r.startswith("|")]
    tail = [r for i, r in enumerate(rows) if i >= start + 2 and not r.startswith("|")]
    budget = limit - sum(len(r) + 1 for r in head + tail)
    keep = set()
    for i, r in sorted(data, key=lambda x: -len(qtokens & set(tokenize(x[1])))):
        if len(r) + 1 <= budget:
            keep.add(i)
            budget -= len(r) + 1
    return "\n".join(head + [r for i, r in data if i in keep] + tail)


def windows(hits, qtokens=None, context=None):
    """Hits as evidence stretches: [(doc, [units], [hit unit ids])], ordered by the best hit
    of each. Every hit is widened by `context` neighbours in its section (default CONTEXT);
    overlapping or adjacent windows of one section are one stretch, and so are the windows
    of one section lying apart, with the chunks between them, when the stretch from the
    first to the last holds at most SECTION_MAX characters (MERGE_SECTION). The chunks the
    hits cite are appended (with_refs; qtokens, the claim's tokens, rank them)."""
    context = CONTEXT if context is None else context
    groups = {}                                               # (doc_id, parent) -> [(lo, hi, order, uid)]
    for order, (uid, _, _, _) in enumerate(hits):
        doc = get_doc(uid.split("#")[0])
        u = doc.by_id[uid]
        siblings = doc.section_units(u["parent"])
        k = next((j for j, x in enumerate(siblings) if x is u), None)
        if k is None:
            lo, hi = None, None
        elif EXPAND == "section":
            lo, hi = 0, len(siblings) - 1
        else:
            lo, hi = max(0, k - context), min(len(siblings) - 1, k + context)
        groups.setdefault((doc.doc_id, u["parent"]), []).append((lo, hi, order, uid))
    out = []
    for (doc_id, parent), spans in groups.items():
        doc = get_doc(doc_id)
        siblings = doc.section_units(parent)
        merged = []
        for lo, hi, order, uid in sorted(spans, key=lambda x: (x[0] is None, x[0] or 0)):
            if lo is None:                                    # not found among its siblings: the unit alone
                merged.append([None, None, order, [uid]])
            elif merged and merged[-1][0] is not None and lo <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], hi)
                merged[-1][2] = min(merged[-1][2], order)
                merged[-1][3].append(uid)
            else:
                merged.append([lo, hi, order, [uid]])
        placed = [m for m in merged if m[0] is not None]
        if MERGE_SECTION and len(placed) > 1:                 # windows of one section lying apart
            lo, hi = placed[0][0], max(m[1] for m in placed)
            uids = [x for m in placed for x in m[3]]
            between = [x for x in siblings[lo:hi + 1] if x["id"] in uids or x["type"] != "figure"]
            if sum(min(len(x["body"]), MAX_UNIT_CHARS) for x in between) <= SECTION_MAX:
                merged = [[lo, hi, min(m[2] for m in placed), uids]] + [m for m in merged if m[0] is None]
        for lo, hi, order, uids in merged:
            if lo is None:
                units = [doc.by_id[uids[0]]]
            else:
                units = [x for x in siblings[lo:hi + 1] if x["id"] in uids or x["type"] != "figure"]
                if EXPAND == "section":
                    units = fit_section(units, uids)
            units = with_refs(doc, units, uids, qtokens)
            out.append((order, doc, units, uids))
    return [(doc, units, uids) for _, doc, units, uids in sorted(out, key=lambda x: x[0])]


REFS_MAX = 2            # cited chunks appended to a stretch after its hits (chunk field refs, parsing.chunk)


def with_refs(doc, units, uids, qtokens=None, limit=REFS_MAX):
    """The stretch plus the chunks its hit units cite: clauses ("依照第十二条的规定处以罚款"
    is no evidence without 第十二条) and numbered tables / figures (见表1), at most `limit`,
    those already in the stretch skipped. When more are cited than fit, the ones sharing the
    most words with the claim go first (the right piece of a split table), then document
    order."""
    present = {u["id"] for u in units}
    cited = []
    for u in units:
        if u["id"] in uids:
            cited += [r for r in u.get("refs", []) if r not in present and r not in cited and r in doc.by_id]
    if qtokens and len(cited) > limit:
        overlap = {r: len(qtokens & set(tokenize(doc.by_id[r]["body"]))) for r in cited}
        cited = sorted(cited, key=lambda r: -overlap[r])      # stable: document order among equals
    return units + [doc.by_id[r] for r in cited[:limit]]


def fit_section(units, uids, cap=None):
    """A whole section for the judge, trimmed when it is longer than cap characters: the
    span from the first to the last hit, grown outwards one unit at a time while it fits,
    plus the section's opening unit when that is short (the sentence or label line that
    introduces what follows: "...情况如下:", "单位:元", "2024 年")."""
    cap = SECTION_MAX if cap is None else cap
    if sum(len(u["body"]) for u in units) <= cap:
        return units
    idx = [i for i, u in enumerate(units) if u["id"] in uids]
    lo, hi = min(idx), max(idx)
    size = sum(len(u["body"]) for u in units[lo:hi + 1])
    grown = True
    while grown:
        grown = False
        if lo > 0 and size + len(units[lo - 1]["body"]) <= cap:
            lo -= 1
            size += len(units[lo]["body"])
            grown = True
        if hi < len(units) - 1 and size + len(units[hi + 1]["body"]) <= cap:
            hi += 1
            size += len(units[hi]["body"])
            grown = True
    chosen = units[lo:hi + 1]
    if lo > 0 and len(units[0]["body"]) <= 200:
        chosen = [units[0]] + chosen
    return chosen


def evidence_records(hits, statement, context=None):
    """Judge records (id, doc_title, page, section, text), a compact trace of each, and the
    figure crops among them (agent.figures.evidence_images)."""
    qtokens = set(tokenize(statement))
    stretches = windows(hits, qtokens, context)
    records, trace = [], []
    for i, (doc, units, uids) in enumerate(stretches, 1):
        records.append({"id": f"E{i}", "doc_title": doc_header(doc), "page": units[0]["page"],
                        "section": units[0]["path"] or "",
                        "text": "\n".join(clip(u["body"], qtokens) for u in units)})
        trace.append({"id": f"E{i}", "doc": doc.doc_id[:14], "units": [u["id"].split("#")[1] for u in units],
                      "hits": [x.split("#")[1] for x in uids], "section": units[0]["path"] or ""})
    return records, trace, evidence_images(stretches)


def image_trace(images):
    return [{"evidence": eid, "unit": uid.split("#")[1], "bytes": len(png)} for eid, uid, png in images]


# ----------------------------------------------------------------------------- one claim
def verify(statement, doc_ids, llm=None, dry=False, tag=""):
    """Retrieval, claim analysis and the judgement for one claim -> claim record. A verdict
    of 证据不足 is judged once more on the hits widened by RETRY_CONTEXT neighbours, when
    the wider evidence differs; the record keeps the first evidence and, under "retry", the
    wider one with its verdict."""
    claims = split_claims(statement)
    hits = retrieve(claims, doc_ids)
    records, trace, images = evidence_records(hits, statement)
    r = {"statement": statement, "claims": claims,
         "hits": [{"unit": u.split("#")[0][:14] + "#" + u.split("#")[1], "claim": f"S{i + 1}", "rank": rank,
                   "score": round(score, 2)} for u, i, rank, score in hits],
         "evidence_blocks": trace, "images": image_trace(images),
         "analysis": None, "verdict": "证据不足", "resolved_at": None, "data": {}}
    if dry or not llm or not records:
        return r
    analysis = analyze_claim(llm, statement, tag)
    verdict, data = judge(llm, statement, analysis, records, tag + "judge", images=images)
    r.update({"analysis": analysis, "verdict": verdict, "data": data})
    if verdict == "证据不足" and RETRY_CONTEXT:
        wide, wide_trace, wide_images = evidence_records(hits, statement, context=RETRY_CONTEXT)
        if [t["units"] for t in wide_trace] != [t["units"] for t in trace]:
            verdict2, data2 = judge(llm, statement, analysis, wide, tag + "judge-wide", images=wide_images)
            r["retry"] = {"evidence_blocks": wide_trace, "images": image_trace(wide_images), "verdict": verdict2}
            if verdict2 != "证据不足":
                r.update({"verdict": verdict2, "data": data2})
    return r


# ----------------------------------------------------------------------------- questions
def answer_tf(question, llm=None, dry=False):
    statement = statement_of(question).lstrip("题：").strip()
    r = verify(statement, question["doc_ids"], llm, dry, tag=f"{question['qid']}-")
    res = {"qid": question["qid"], "statement": statement, "claim": r, "answer": "", "verdict": ""}
    if dry:
        return res
    verdict = r["verdict"]
    res["defaulted"] = verdict == "证据不足"
    if res["defaulted"]:
        verdict = "错误"                       # nothing in the evidence supports the statement
    res.update({"verdict": verdict, "answer": letter_for(question, verdict), "reason": r["data"].get("reason", ""),
                "evidence": evidence_list(r["data"])})
    return res


def _verify_options(question, llm, dry):
    return {letter: verify(text.strip(), question["doc_ids"], llm, dry, tag=f"{question['qid']}-{letter}-")
            for letter, text in question["options"].items()}


def answer_mcq(question, llm=None, dry=False):
    results = _verify_options(question, llm, dry)
    res = {"qid": question["qid"], "options": results, "answer": "", "how": ""}
    if dry:
        return res
    correct = [l for l, r in results.items() if r["verdict"] == "正确"]
    undecided = [l for l, r in results.items() if r["verdict"] == "证据不足"]
    if len(correct) == 1:
        answer, how = correct[0], "唯一判为正确的选项"
    elif len(correct) > 1:
        answer, reason = pick(llm, question, {l: results[l] for l in correct})
        how = "多个选项判为正确，比较后选定：" + reason
    elif len(undecided) == 1:
        answer, how = undecided[0], "其余选项均被证据否定，剩下唯一未被否定的选项"
    else:
        answer, reason = pick(llm, question, results)
        how = "没有选项被判为正确，比较后选定：" + reason
    res["answer"], res["how"] = answer, how
    return res


def answer_multi(question, llm=None, dry=False):
    results = _verify_options(question, llm, dry)
    res = {"qid": question["qid"], "options": results, "answer": "", "how": ""}
    if dry:
        return res
    correct = [l for l, r in results.items() if r["verdict"] == "正确"]
    if correct:
        res["how"] = "判为正确的选项集合"
    else:
        correct, reason = pick_multi(llm, question, results, tag=f"{question['qid']}-pick")
        res["how"] = "没有选项被判为正确，比较后选定：" + reason
    res["answer"] = "".join(sorted(set(correct)))
    return res


HANDLERS = {"tf": answer_tf, "mcq": answer_mcq, "multi": answer_multi}
