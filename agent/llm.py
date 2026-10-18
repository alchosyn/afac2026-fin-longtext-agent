# -*- coding: utf-8 -*-
"""Qwen client (Bailian OpenAI-compatible endpoint) with per-question token accounting
and a JSONL log of every call. A call may carry PNG images (figure crops, see
agent.figures); qwen3.6-plus reads them through the same endpoint."""
import base64
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 百炼国内版: https://dashscope.aliyuncs.com/compatible-mode/v1
# 百炼国际版 (QwenCloud, key 以 sk-ws- 开头): https://dashscope-intl.aliyuncs.com/compatible-mode/v1


def load_dotenv(path=ROOT / ".env"):
    """KEY=VALUE lines from the project's .env file (never committed); the process
    environment wins when both are set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()
DEFAULT_MODEL = os.environ.get("QWEN_MODEL", "qwen3.6-plus")
BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


def api_key():
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("DASHSCOPE_API_KEY 未设置：请在项目根目录的 .env 文件里写一行 DASHSCOPE_API_KEY=<你的key>，"
                         "或用 setx DASHSCOPE_API_KEY 设置环境变量")
    return key


class Qwen:
    def __init__(self, model=DEFAULT_MODEL, log_path=None):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=api_key(), base_url=BASE_URL)
        self.log_path = Path(log_path) if log_path else ROOT / "logs" / f"qwen_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.calls = 0

    def reset(self):
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.calls = 0

    def chat(self, system, user, max_tokens=800, tag="", retries=2, images=None):
        """One call. images: PNG bytes to attach to the user message (figure crops), sent as
        image_url data URIs after the text; the log records their count and sizes, not the bytes."""
        t0 = time.time()
        content_parts = user
        if images:
            content_parts = [{"type": "text", "text": user}]
            for png in images:
                b64 = base64.b64encode(png).decode("ascii")
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        for attempt in range(retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": content_parts}],
                    temperature=0,
                    max_tokens=max_tokens,
                    extra_body={"enable_thinking": False},
                )
                break
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(3 * (attempt + 1))
        content = resp.choices[0].message.content or ""
        usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        for k in self.usage:
            self.usage[k] += usage[k]
        self.calls += 1
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": tag, "model": self.model,
                                "seconds": round(time.time() - t0, 1), "usage": usage,
                                "images": [len(p) for p in (images or [])],
                                "system": system, "user": user, "response": content}, ensure_ascii=False) + "\n")
        return content, usage


def _balanced_block(content, start):
    """Return the text of the {...} object starting at `start`, honouring string
    literals, or None if the braces never close."""
    depth, in_str, esc = 0, False, False
    for j in range(start, len(content)):
        ch = content[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return content[start:j + 1]
    return None


def parse_json(content):
    """The outermost JSON object in the reply (the model may reason in prose before
    it, and the object may nest lists of objects). Longest parseable block wins."""
    best = None
    i = content.find("{")
    while i >= 0:
        block = _balanced_block(content, i)
        if block:
            for candidate in (block, re.sub(r"\s*\n\s*", " ", block)):
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and (best is None or len(block) > best[0]):
                    best = (len(block), obj)
                break
        i = content.find("{", i + 1)
    return best[1] if best else {}
