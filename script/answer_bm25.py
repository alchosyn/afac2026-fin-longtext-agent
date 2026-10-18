# -*- coding: utf-8 -*-
"""Answer the group A questions of one domain with the body-only BM25 pipeline
(agent/bm25_pipeline.py): per option / sub-claim, BM25 over the units' own text, merged
top 3, each hit widened to its neighbours in the same section, one judgement call.

Usage:
    python script/answer_bm25.py regulatory                  # tf, mcq and multi
    python script/answer_bm25.py research --format tf        # one format
    python script/answer_bm25.py regulatory --dry            # retrieval only, no API calls
    python script/answer_bm25.py regulatory reg_a_008        # a subset

Writes output/group_a/answer_bm25_<domain>.csv (summary row first) and
output/group_a/evidence_bm25_<domain>.json; every model call is logged under logs/bm25_<domain>_*.jsonl.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.bm25_pipeline import HANDLERS  # noqa: E402
from agent.docview import load_cards      # noqa: E402

OUT_DIR = ROOT / "output" / "group_a"


def print_claim(r, indent, dry):
    pad = " " * indent
    for i, c in enumerate(r["claims"]):
        if len(r["claims"]) > 1:
            print(f"{pad}S{i + 1} {c[:60]}")
    print(f"{pad}hits " + "  ".join(f"{h['claim']}#{h['rank']} {h['unit'].split('#')[1]}({h['score']})" for h in r["hits"]))
    for b in r["evidence_blocks"]:
        print(f"{pad}{b['id']} {b['doc']} {','.join(b['units'])}  | {b['section'][:50]}")
    if not dry:
        print(f"{pad}=> {r['verdict']}  {(r.get('data') or {}).get('reason', '')[:90]}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=("regulatory", "research"))
    ap.add_argument("qids", nargs="*")
    ap.add_argument("--format", default="tf,mcq,multi", help="comma separated: tf, mcq, multi")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args(argv)
    formats = [f for f in args.format.split(",") if f in HANDLERS]
    questions = [q for q in json.load(open(ROOT / "questions" / "group_a" / f"{args.domain}_questions.json", encoding="utf-8"))
                 if q["answer_format"] in formats]
    if args.qids:
        questions = [q for q in questions if q["qid"] in args.qids]

    llm = None
    if not args.dry:
        from agent.llm import Qwen
        llm = Qwen(log_path=ROOT / "logs" / f"bm25_{args.domain}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")

    processed = set(load_cards())
    rows, details = [], []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for q in questions:
        missing = [d for d in q["doc_ids"] if d not in processed]
        if missing:                           # e.g. penalty decisions published as HTML only, never parsed
            print(f"\n{q['qid']}  WARNING: documents not in processed_data, answering without them: {missing}")
            q = dict(q, doc_ids=[d for d in q["doc_ids"] if d in processed])
        if llm:
            llm.reset()
        res = HANDLERS[q["answer_format"]](q, llm, dry=args.dry)
        usage = dict(llm.usage) if llm else {k: 0 for k in totals}
        res["usage"], res["answer_format"], res["missing_docs"] = usage, q["answer_format"], missing
        details.append(res)
        for k in totals:
            totals[k] += usage[k]

        if q["answer_format"] == "tf":
            print(f"\n{q['qid']}  {res['statement'][:80]}")
            print_claim(res["claim"], 3, args.dry)
        else:
            print(f"\n{q['qid']}  {q['question'][:80]}")
            for letter, r in res["options"].items():
                print(f"   {letter}. {r['statement'][:70]}")
                print_claim(r, 6, args.dry)
        if not args.dry:
            how = res.get("how") or res.get("reason", "")
            print(f"   answer={res['answer']}{'（证据不足，默认判错）' if res.get('defaulted') else ''} "
                  f"tokens={usage['total_tokens']} calls={llm.calls}  {how[:90]}")
            rows.append([q["qid"], res["answer"], usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]])

    if args.dry:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    answer_path = OUT_DIR / f"answer_bm25_{args.domain}.csv"
    evidence_path = OUT_DIR / f"evidence_bm25_{args.domain}.json"
    with answer_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        w.writerow(["summary", "", totals["prompt_tokens"], totals["completion_tokens"], totals["total_tokens"]])
        w.writerows(rows)
    evidence_path.write_text(json.dumps(details, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n{len(rows)} questions, tokens {totals}\n-> {answer_path}\n-> {evidence_path}\n-> {llm.log_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
