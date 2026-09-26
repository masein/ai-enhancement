"""12i.1: a fake OpenRouter — its models list, each model's providers, chat
completions with usage and cost, and (12i.2) embeddings — at service.llm._http,
so the service's own client runs unchanged and nothing leaves the machine. A
request to any other address goes to the real _http, which no test here reaches.

    fake = FakeOpenRouter.install(monkeypatch)
    ... fake.chat[0]["provider"] == {"order": ["inference-net"], "allow_fallbacks": False}
"""

from __future__ import annotations

import json
import re

BASE = "https://openrouter.test/api/v1"
KEY = "sk-or-test-not-a-real-key"


def _m(mid, version, name, pin, pout, out=("text",), inp=("text",), ctx=1048576):
    return {"id": mid, "canonical_slug": version, "name": name, "context_length": ctx,
            "pricing": {"prompt": pin, "completion": pout},
            "architecture": {"input_modalities": list(inp), "output_modalities": list(out)}}


MODELS = [
    _m("deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash-20260910",
       "DeepSeek: DeepSeek V4.1 Flash", "0.00000014", "0.00000042"),
    _m("z-ai/glm-5.3", "z-ai/glm-5.3-20260816", "Z.ai: GLM 5.3", "0.0000014", "0.0000044"),
    _m("z-ai/glm-5.3-flash", "z-ai/glm-5.3-flash-20260826", "Z.ai: GLM 5.3 Flash",
       "0.00000004", "0.0000005"),
    _m("openai/gpt-6-luna", "openai/gpt-6-luna-20260922", "OpenAI: GPT-6 Luna",
       "0.0000001", "0.0000005"),
    _m("qwen/qwen3.5-72b-instruct", "qwen/qwen3.5-72b-instruct-20260701",
       "Qwen: Qwen3.5 72B Instruct", "0.0000003", "0.0000009"),
    # not offered: an image-only model, and a free one
    _m("acme/painter-1", "acme/painter-1", "Acme: Painter", "0.000001", "0.000002",
       out=("image",)),
    _m("acme/free-chat:free", "acme/free-chat", "Acme: Free Chat", "0", "0"),
]
ENDPOINTS = [
    {"provider_name": "InferenceNet", "tag": "inference-net", "quantization": "fp8",
     "status": 0, "pricing": {"prompt": "0.0000001", "completion": "0.0000004"}},
    {"provider_name": "Morph", "tag": "morph/fp8", "quantization": "fp8", "status": 0,
     "pricing": {"prompt": "0.00000008", "completion": "0.0000003"}},
]


def default_reply(req: dict) -> str:
    """a judge's reply to the board's prompts: a pass for an Everyday rubric,
    every criterion met for a criteria prompt, else a 3 of 4"""
    user = req["messages"][-1]["content"]
    if "RUBRIC" in user and '"pass"' in user:
        return json.dumps({"pass": True, "reason": "it does what the rubric asks"})
    ids = re.findall(r'^\s*-\s*"?([a-z][a-z0-9_]*)"?\s*[:(]', user, re.M)
    if '"criteria"' in user and ids:
        flags = re.findall(r'^\s*-\s*flag\s+"?([a-z][a-z0-9_]*)"?', user, re.M)
        return json.dumps({"criteria": {i: 1.0 for i in ids if i not in flags},
                           "flags": {f: False for f in flags}, "justification": "covers it"})
    return json.dumps({"score": 3, "justification": "mostly right"})


def embedding(text: str, dims: int = 256) -> list[float]:
    """12i.2: a word-count vector, so texts sharing most of their words sit
    close (a near-duplicate's cosine is well over 0.9) and unrelated ones don't"""
    import hashlib
    v = [0.0] * dims
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dims] += 1.0
    return v


class FakeOpenRouter:
    def __init__(self, real):
        self.real = real
        self.calls: list[tuple[str, str]] = []      # (method, path)
        self.chat: list[dict] = []                   # every chat completion's body
        self.embedded: list[dict] = []               # every embeddings request's body
        self.reply = default_reply
        self.cost = 0.001                            # dollars a completion reports

    @classmethod
    def install(cls, monkeypatch, key: str = KEY) -> "FakeOpenRouter":
        from service import config, llm
        fake = cls(llm._http)
        monkeypatch.setattr(llm, "_http", fake)
        monkeypatch.setattr(config, "OPENROUTER_BASE_URL", BASE)
        monkeypatch.setattr(config, "OPENROUTER_API_KEY", key)
        monkeypatch.setattr(llm.LocalOpenAI, "BACKOFF", (0, 0, 0))
        return fake

    def __call__(self, method, url, headers, body=None, timeout=60.0):
        if not url.startswith(BASE):
            return self.real(method, url, headers, body, timeout)
        path = url[len(BASE):]
        self.calls.append((method, path))
        assert headers.get("authorization") == f"Bearer {KEY}" or path == "/models"
        if path == "/models":
            return 200, json.dumps({"data": MODELS}).encode()
        m = re.fullmatch(r"/models/(.+)/endpoints", path)
        if m:
            return 200, json.dumps({"data": {"id": m.group(1), "endpoints": ENDPOINTS}}).encode()
        if path == "/chat/completions":
            req = json.loads(body)
            self.chat.append(req)
            return 200, json.dumps({
                "choices": [{"message": {"content": self.reply(req)}, "finish_reason": "stop"}],
                "provider": "InferenceNet",
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100,
                          "cost": self.cost}}).encode()
        if path == "/embeddings":
            req = json.loads(body)
            self.embedded.append(req)
            return 200, json.dumps({
                "data": [{"index": i, "embedding": embedding(t)} for i, t in enumerate(req["input"])],
                "usage": {"prompt_tokens": 10 * len(req["input"]), "cost": 0.00001}}).encode()
        raise AssertionError(f"the fake OpenRouter has no {method} {path}")
