# -*- coding: utf-8 -*-
"""Run the chunk-level evidence search (agent.search) on B-list questions: retrieval only, no
judgement call. One Qwen call per statement for the rewrite; rewrites are kept in
output/retrieval/rewrite_cache_b.json so that a second run searches with the same sub-claims.

Usage:
    python script/search_b.py fc_b_013 res_b_006      # some questions
    python script/search_b.py --domain insurance      # the 20 questions of a domain
    python script/search_b.py --all

Writes output/retrieval/search_b.json: per statement the scope, the rewrite, every query with its top
chunks, the candidates and the evidence.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.llm import Qwen                                  # noqa: E402
from agent.scope import load_b_questions                    # noqa: E402
from agent.search import KEEP, TOP, search, statements      # noqa: E402

OUT = ROOT / "output" / "retrieval" / "search_b.json"
CACHE = ROOT / "output" / "retrieval" / "rewrite_cache_b.json"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("qids", nargs="*")
    ap.add_argument("--domain")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--top", type=int, default=TOP)
    ap.add_argument("--keep", type=int, default=KEEP)
    args = ap.parse_args(argv)
    questions = [q for q in load_b_questions() if args.all or q["qid"] in args.qids or q["domain"] == args.domain]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    llm = Qwen(log_path=ROOT / "logs" / f"search_b_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    rows, t0 = [], time.time()
    for q in questions:
        for cid, statement in statements(q):
            r = search(q, statement, llm, args.top, args.keep, cache, tag=cid + "-")
            rows.append({"claim": cid, "qid": q["qid"], "domain": q["domain"], "type": q["type"], "statement": statement, **r})
            CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            sc = r["scope"]
            where = f"{len(sc['docs'])} named" if sc["named"] else f"whole domain ({len(sc['docs'])})"
            print(f"{cid:13s} {where:18s} {len(r['queries'])} queries  {len(r['candidates']):2d} candidates  {len(r['evidence']):2d} evidence | {r['rewrite'][:70]}")
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    n = max(len(rows), 1)
    print(f"\n{len(questions)} questions, {len(rows)} statements in {time.time() - t0:.0f} s; rewrite calls {llm.calls}, tokens {llm.usage['total_tokens']}; "
          f"candidates {sum(len(r['candidates']) for r in rows) / n:.1f}, evidence {sum(len(r['evidence']) for r in rows) / n:.1f} per statement -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
