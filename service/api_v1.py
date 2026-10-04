"""16b.3: Use as an API — an OpenAI-compatible address a colleague calls from
their own code, to try a model. It is a shared 32 GB card and benchmark runs
come first: this is for trying a model, not for serving an app.

- `GET /v1/models`, `POST /v1/chat/completions` (streamed and not); `model`
  is the board's model id.
- Behind it, the Playground's engine (chat.ENGINE): a served model is asked
  through its server (the board keeps its key), a Hugging Face or uploaded
  folder model through the Playground's loader and its placement rules. Not
  offered: a base model (no chat template), a GGUF file with no server, a
  model from OpenRouter (its cost isn't counted for chats), and thinking rows
  (the model's own id is asked).
- **Keys:** one for each person, made in the dashboard with the board's write
  token, shown once, stored as a sha256, with requests, tokens and a "last
  used" time, and Revoke. The shared write token is not an API key.
- **Taking turns:** a model that would load while a run holds the GPU is 503
  with Retry-After and one plain line (a small one answers on the CPU, as in
  the Playground; a served model answers unless the run is testing it); one
  request at a time for each model — another waits API_WAIT_S, then 429; a
  reply at most API_MAX_TOKENS, a conversation at most what the model reads;
  an idle model is unloaded after CHAT_IDLE_UNLOAD_S, as in the Playground.
- **Kept:** counts only, for each key. Prompts and replies are not stored, and
  an API call is never a benchmark result."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time

from . import chat, config, db

KEY_PREFIX = "ebk_"
_model_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()

NOT_OFFERED = {
    "base": "a base model, with no chat template: it isn't asked as a chat",
    "gguf": "a GGUF file with no server here: register its llama-server under Add a model "
            "▸ Running on a server",
    "openrouter": "a model from OpenRouter: a chat's cost isn't counted against the month",
    "own code": "it runs its own model code, which chat does not run",
}


class ApiError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error",
                 retry_after: int | None = None):
        super().__init__(message)
        self.status, self.kind, self.retry_after = status, kind, retry_after

    def body(self) -> dict:
        return {"error": {"message": str(self), "type": self.kind, "code": self.status}}


# ---------------------------------------------------------------------------
# keys: one for each person, shown once, kept as a sha256
# ---------------------------------------------------------------------------

def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def make_key(who: str) -> dict:
    who = (who or "").strip()[:80]
    if not who:
        raise ApiError(422, "Whose key: a name")
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    kid = db.apikey_add(who, key[:len(KEY_PREFIX) + 6], _hash(key))
    return {"id": kid, "who": who, "key": key, "prefix": key[:len(KEY_PREFIX) + 6],
            "line": "Copy it now: it is shown once, and the board keeps only its hash."}


def revoke(kid: int, by: str) -> dict:
    row = db.apikey_get(kid)
    if not row:
        raise ApiError(404, "No such key.")
    by = (by or "").strip()
    if by.lower() != row["who"].lower() and not config.is_owner(by):
        raise ApiError(403, f"Only {row['who']}, whose key it is, revokes it.")
    db.apikey_revoke(kid)
    return {**db.apikey_get(kid), "line": "Revoked: it answers nothing from now on."}


def auth(header: str) -> dict:
    """the key a request carries — Authorization: Bearer … — or ApiError 401"""
    key = (header or "").removeprefix("Bearer ").strip()
    if not key:
        raise ApiError(401, "No API key: send Authorization: Bearer <your key>. Make one on a "
                            "model's page ▸ Use as an API.", "authentication_error")
    row = db.apikey_by_hash(_hash(key))
    if not row:
        raise ApiError(401, "That API key isn't one of this board's.", "authentication_error")
    if row["revoked_at"]:
        raise ApiError(401, "That API key was revoked.", "authentication_error")
    return row


# ---------------------------------------------------------------------------
# what can be asked
# ---------------------------------------------------------------------------

def offered() -> list[dict]:
    return [m for m in chat.board_models() if m["chat"]]


def why_not(model_id: str) -> str:
    """'' when the model can be asked through the API, else why not"""
    row = next((m for m in chat.board_models() if m["id"] == model_id), None)
    if model_id.endswith(" · thinking"):
        return "a thinking row: ask the model itself; thinking is its server's to decide"
    if not row:
        return f"{model_id} isn't a model on this board"
    return "" if row["chat"] else NOT_OFFERED.get(row["why_not"], row["why_not"] or "not here")


def models_list() -> dict:
    return {"object": "list", "data": [{"id": m["id"], "object": "model", "created": 0,
                                        "owned_by": "board"} for m in offered()]}


# ---------------------------------------------------------------------------
# a reply
# ---------------------------------------------------------------------------

def _words(messages: list[dict]) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):                 # OpenAI's parts: the text ones
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        n += len(str(c or "").split())
    return n


def _messages(body: dict) -> list[dict]:
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ApiError(400, "messages: a list of {role, content}")
    out = []
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant"):
            raise ApiError(400, "each message is {role: system | user | assistant, content}")
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        out.append({"role": m["role"], "content": str(c or "")})
    return out


def _settings(body: dict, row: dict) -> dict:
    s = chat.effective({}, row)
    s["system"] = ""                             # the request carries its own, if any
    want = body.get("max_tokens") or body.get("max_completion_tokens")
    try:
        cap = int(want) if want else config.API_MAX_TOKENS
    except (TypeError, ValueError):
        raise ApiError(400, "max_tokens: a number") from None
    s["max_gen_toks"] = max(1, min(cap, config.API_MAX_TOKENS))
    if body.get("temperature") is not None:
        try:
            s["temperature"] = max(0.0, min(2.0, float(body["temperature"])))
        except (TypeError, ValueError):
            raise ApiError(400, "temperature: a number") from None
    return s


def _model_lock(model_id: str) -> threading.Lock:
    with _locks_lock:
        return _model_locks.setdefault(model_id, threading.Lock())


def begin(body: dict, key: dict) -> tuple[chat.Stream, dict, threading.Lock]:
    """a reply under way — or ApiError: not offered, too long, busy (429), or
    a run holds the GPU (503)"""
    model_id = str(body.get("model") or "").strip()
    why = why_not(model_id)
    if why:
        raise ApiError(404, f"{model_id or 'model'}: not offered through the API — {why}.",
                       "invalid_request_error")
    row = chat.model_row(model_id)
    messages = _messages(body)
    settings = _settings(body, row)
    fits = chat.context_words(row, settings)
    if _words(messages) > fits:
        raise ApiError(400, f"Too long for {row['name']}: about {fits:,} words fit beside a "
                            f"{settings['max_gen_toks']}-token reply.", "context_length_exceeded")
    device, why = chat.ENGINE.place(row)
    if not device:
        raise ApiError(503, why, "server_error", retry_after=60)
    # one request at a time for each model: another waits a moment, then 429
    lock = _model_lock(model_id)
    t0 = time.time()
    while True:
        held = chat.ENGINE.loaded.get(model_id)
        if (not held or not held.busy.locked()) and lock.acquire(blocking=False):
            break
        if time.time() - t0 >= config.API_WAIT_S:
            raise ApiError(429, f"{row['name']} is answering another request: try again in a "
                                "moment. One request at a time for each model.",
                           "rate_limit_error", retry_after=10)
        time.sleep(0.1)
    done: dict = {}

    def on_done(reply: dict | None, why: str = "") -> dict | None:
        done.update(reply=reply, why=why)        # kept in memory for this request, never stored
        return reply
    try:
        st = chat.ENGINE.start({"id": f"api-{key['id']}"}, messages, settings, row, on_done)
    except Exception:
        lock.release()
        raise
    return st, {"row": row, "settings": settings, "messages": messages, "done": done}, lock


def _events(st: chat.Stream, timeout: float = 900.0):
    """the stream's events as they come, until it ends"""
    i, t0 = 0, time.time()
    while True:
        with st.cond:
            while i >= len(st.events) and not st.done:
                if time.time() - t0 > timeout:
                    st.stop.set()
                    return
                st.cond.wait(0.5)
            batch = st.events[i:]
            i = len(st.events)
            ended = st.done
        yield from batch
        if ended and i >= len(st.events):
            return


def _usage(ctx: dict, text: str, reply: dict | None) -> dict:
    prompt = max(1, round(sum(len(m["content"]) for m in ctx["messages"]) / 4))
    out = (reply or {}).get("tokens") or max(1, round(len(text) / 4)) if text else 0
    return {"prompt_tokens": prompt, "completion_tokens": int(out),
            "total_tokens": prompt + int(out)}


def _finish(ev: dict) -> str:
    reply = ev.get("reply") or {}
    return "length" if reply.get("cut") and reply.get("cut") not in ("stopped",) else "stop"


def complete(st: chat.Stream, ctx: dict, key: dict, lock: threading.Lock) -> dict:
    """the whole reply, OpenAI's way"""
    try:
        text, think, finish = "", "", "stop"
        for ev in _events(st):
            t = ev.get("t")
            if t == "text":
                text += ev["d"]
            elif t == "think":
                think += ev["d"]
            elif t == "reset":
                text, think = ev.get("text", ""), ev.get("think", "")
            elif t == "refused":
                raise ApiError(503, ev.get("why") or "not now", "server_error", retry_after=60)
            elif t == "error":
                raise ApiError(502 if "server" in (ev.get("why") or "") else 500,
                               ev.get("why") or "the model failed", "server_error")
            elif t == "done":
                finish = _finish(ev)
                reply = ev.get("reply") or {}
                text, think = reply.get("text", text), reply.get("thinking", think)
        usage = _usage(ctx, text, ctx["done"].get("reply"))
        db.apikey_used(key["id"], usage["total_tokens"])
        msg = {"role": "assistant", "content": text}
        if think:
            msg["reasoning_content"] = think
        return {"id": "chatcmpl-" + st.id[:24], "object": "chat.completion",
                "created": int(time.time()), "model": ctx["row"]["id"],
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": usage}
    finally:
        lock.release()


def stream(st: chat.Stream, ctx: dict, key: dict, lock: threading.Lock):
    """server-sent events, OpenAI's chunks: its thinking as reasoning_content"""
    cid, created, mid = "chatcmpl-" + st.id[:24], int(time.time()), ctx["row"]["id"]

    def chunk(delta: dict, finish: str | None = None, **more) -> str:
        return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk",
                                      "created": created, "model": mid,
                                      "choices": [{"index": 0, "delta": delta,
                                                   "finish_reason": finish}], **more},
                                     ensure_ascii=False) + "\n\n"
    text = ""
    try:
        yield chunk({"role": "assistant", "content": ""})
        for ev in _events(st):
            t = ev.get("t")
            if t == "text":
                text += ev["d"]
                yield chunk({"content": ev["d"]})
            elif t == "think":
                yield chunk({"reasoning_content": ev["d"]})
            elif t in ("refused", "error"):
                yield "data: " + json.dumps(ApiError(503 if t == "refused" else 500,
                                                     ev.get("why") or "failed").body()) + "\n\n"
                return
            elif t == "done":
                usage = _usage(ctx, text, ctx["done"].get("reply"))
                db.apikey_used(key["id"], usage["total_tokens"])
                yield chunk({}, _finish(ev), usage=usage)
        yield "data: [DONE]\n\n"
    finally:
        if not st.done:                          # the caller went away: the reply stops
            st.stop.set()
        lock.release()
