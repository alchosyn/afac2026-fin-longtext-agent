# -*- coding: utf-8 -*-
"""bge-reranker-v2-m3 as a line server, started by agent.rerank with a Python that has torch and
transformers (a CUDA build where there is one). One JSON object per line:

    stdin   {"query": "...", "passages": ["...", ...]}
    stdout  {"scores": [raw logit per passage]}

    python agent/rerank_worker.py <model dir>
"""
import json
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 1024      # at 512 a fifth of the chunks are cut, at 1024 one in a thousand (2026-10-03)
BATCH = 16

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(sys.argv[1])
model = AutoModelForSequenceClassification.from_pretrained(
    sys.argv[1], torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
print(json.dumps({"ready": device}), flush=True)
for line in sys.stdin:
    x = json.loads(line)
    passages, scores = x["passages"], [0.0] * len(x["passages"])
    order = sorted(range(len(passages)), key=lambda i: len(passages[i]))        # similar lengths share a batch
    with torch.inference_mode():
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            enc = tok([x["query"]] * len(idx), [passages[j] for j in idx], padding=True, truncation="only_second",
                      max_length=MAX_LEN, return_tensors="pt")
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.view(-1).float().cpu().tolist()
            for j, s in zip(idx, logits):
                scores[j] = round(s, 4)
    print(json.dumps({"scores": scores}), flush=True)
