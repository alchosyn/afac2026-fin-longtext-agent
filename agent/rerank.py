# -*- coding: utf-8 -*-
"""Cross-encoder reranking of BM25 candidates (bge-reranker-v2-m3, D:\\models).

The default Python has a CPU build of torch (0.5-2 s per pair); the scoring therefore runs in a
worker process (agent/rerank_worker.py) started with a Python that has a CUDA build: the path
in RERANK_PYTHON, by default the venv built for dots.ocr (about 20 pairs per second on the
RTX 4050, 1.5 GB of VRAM). The worker stays up for the life of the Reranker.

    rr = Reranker()
    rr.top("比亚迪 2024年 营业收入", chunk_ids, keep=8)   -> [(chunk id, score)] best first
    rr.close()

A passage is retrieval.vector_text: title | notice | heading path, then the chunk with lean
tables, cut by the worker at 1,024 tokens.
"""
import json
import os
import subprocess
from pathlib import Path

from .retrieval import get_doc, vector_text

MODEL = os.environ.get("RERANK_MODEL", r"D:\models\bge-reranker-v2-m3")
PYTHON = os.environ.get("RERANK_PYTHON", r"D:\Temp\claude\D--long-text-economic-qustions-public-dataset-upload"
                        r"\afee3180-a77f-4528-93fe-42fecc566c17\scratchpad\dots_venv\Scripts\python.exe")


class Reranker:
    def __init__(self, python=PYTHON, model=MODEL):
        if not Path(python).exists():
            raise SystemExit(f"没有找到带 CUDA 的 Python：{python}（用环境变量 RERANK_PYTHON 指定）")
        self.proc = subprocess.Popen([python, str(Path(__file__).with_name("rerank_worker.py")), model], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
        self.device = self._read()["ready"]
        self.pairs = 0
        self._text = {}

    def _read(self):
        while True:                                  # skip anything a library prints before the JSON line
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("reranker worker stopped")
            if line.lstrip().startswith("{"):
                return json.loads(line)

    def scores(self, query, passages):
        self.proc.stdin.write(json.dumps({"query": query, "passages": passages}, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        self.pairs += len(passages)
        return self._read()["scores"]

    def passage(self, uid):
        if uid not in self._text:
            doc = get_doc(uid.split("#")[0])
            self._text[uid] = vector_text(doc, doc.by_id[uid])
        return self._text[uid]

    def top(self, query, uids, keep=8):
        """The `keep` best of the chunks `uids` for `query`: [(chunk id, score)], best first."""
        if not uids:
            return []
        s = self.scores(query, [self.passage(u) for u in uids])
        return sorted(zip(uids, s), key=lambda x: -x[1])[:keep]

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
