"""One LLM client, batch API only, three backends: anthropic, openai, fake.

Batch only is the owner's call and the right one for this job: nothing here is
latency-sensitive — a proposal is read by a person hours later, a dataset is
consumed by a training run days later — and batch pricing is about half of
interactive. It also shapes the code well: every request is a row with a
custom_id, every batch id is persisted before anything else happens, and a
restart resumes polling rather than re-submitting (service/llm_poller.py).

The key comes from LLM_API_KEY, which reaches the container through
docker-compose ${LLM_API_KEY} interpolation from .env — never a literal in
docker-compose.yml, never in git — and is stripped from every evaluation
subprocess's environment (service/runner.py::_child_env), so a submitted
model's own code cannot read it. It is NOT hidden from anyone with docker
access on the box: `docker inspect` shows Config.Env. SERVICE.md says so.

The fake backend is the test double: it answers instantly (or after N polls),
records every request body it was given to a file the tests read, and takes a
responder function so a test can plant whatever text it needs to see refused.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import config


class LLMError(RuntimeError):
    pass


@dataclass
class Request:
    custom_id: str
    system: str
    user: str
    max_tokens: int = 2048
    # recorded by the fake backend, never sent anywhere: what the tests need to
    # prove about a request (the doc hashes behind a proposal, the item count)
    meta: dict = field(default_factory=dict)


@dataclass
class Result:
    text: str = ""
    error: str = ""


def prompt_sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def extract_json(text: str):
    """The first JSON object or array in a reply, or None. Models wrap JSON in
    prose and code fences however firmly they are told not to."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i = text.find(opener)
        j = text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

class Backend:
    name = "?"
    model = ""

    def submit(self, requests: list[Request]) -> str:
        raise NotImplementedError

    def status(self, batch_id: str) -> tuple[str, str]:
        """('pending' | 'done' | 'failed', detail)"""
        raise NotImplementedError

    def fetch(self, batch_id: str) -> dict[str, Result]:
        raise NotImplementedError

    @property
    def id(self) -> str:
        return f"{self.name}/{self.model}"


def _http(method: str, url: str, headers: dict, body: bytes | None = None,
          timeout: float = 60.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        raise LLMError(f"{method} {url}: HTTP {e.code}: {e.read()[:400].decode(errors='replace')}") from None
    except urllib.error.URLError as e:
        raise LLMError(f"{method} {url}: {e.reason}") from None


class AnthropicBatches(Backend):
    """Message Batches API. Plain HTTP; the SDK is not in the image and one
    endpoint family does not justify it."""
    name = "anthropic"
    BASE = "https://api.anthropic.com/v1"

    def __init__(self, model: str, key: str):
        self.model, self.key = model, key

    def _h(self) -> dict:
        return {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                "content-type": "application/json"}

    def submit(self, requests: list[Request]) -> str:
        body = {"requests": [{
            "custom_id": r.custom_id,
            "params": {"model": self.model, "max_tokens": r.max_tokens, "system": r.system,
                       "messages": [{"role": "user", "content": r.user}]},
        } for r in requests]}
        _, raw = _http("POST", f"{self.BASE}/messages/batches", self._h(),
                       json.dumps(body).encode())
        return json.loads(raw)["id"]

    def status(self, batch_id: str) -> tuple[str, str]:
        _, raw = _http("GET", f"{self.BASE}/messages/batches/{batch_id}", self._h())
        b = json.loads(raw)
        st = b.get("processing_status")
        if st == "ended":
            return "done", json.dumps(b.get("request_counts") or {})
        return "pending", st or ""

    def fetch(self, batch_id: str) -> dict[str, Result]:
        _, raw = _http("GET", f"{self.BASE}/messages/batches/{batch_id}", self._h())
        url = json.loads(raw).get("results_url")
        if not url:
            raise LLMError(f"batch {batch_id} has no results_url yet")
        _, lines = _http("GET", url, self._h(), timeout=300)
        out: dict[str, Result] = {}
        for line in lines.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            res = row.get("result") or {}
            if res.get("type") == "succeeded":
                text = "".join(c.get("text", "") for c in
                               (res.get("message") or {}).get("content") or []
                               if c.get("type") == "text")
                out[row["custom_id"]] = Result(text=text)
            else:
                out[row["custom_id"]] = Result(error=json.dumps(res)[:400])
        return out


class OpenAIBatches(Backend):
    """Batch API: upload a JSONL file, create the batch, read the output file."""
    name = "openai"
    BASE = "https://api.openai.com/v1"

    def __init__(self, model: str, key: str):
        self.model, self.key = model, key

    def _h(self, ctype: str | None = "application/json") -> dict:
        h = {"authorization": f"Bearer {self.key}"}
        if ctype:
            h["content-type"] = ctype
        return h

    def submit(self, requests: list[Request]) -> str:
        lines = [json.dumps({
            "custom_id": r.custom_id, "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": self.model, "max_tokens": r.max_tokens,
                     "response_format": {"type": "json_object"},
                     "messages": [{"role": "system", "content": r.system},
                                  {"role": "user", "content": r.user}]}}) for r in requests]
        boundary = "----bench" + uuid.uuid4().hex
        payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\n"
                   f"batch\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                   f"filename=\"batch.jsonl\"\r\nContent-Type: application/jsonl\r\n\r\n"
                   ).encode() + "\n".join(lines).encode() + f"\r\n--{boundary}--\r\n".encode()
        _, raw = _http("POST", f"{self.BASE}/files",
                       self._h(f"multipart/form-data; boundary={boundary}"), payload, 300)
        fid = json.loads(raw)["id"]
        _, raw = _http("POST", f"{self.BASE}/batches", self._h(), json.dumps({
            "input_file_id": fid, "endpoint": "/v1/chat/completions",
            "completion_window": "24h"}).encode())
        return json.loads(raw)["id"]

    def status(self, batch_id: str) -> tuple[str, str]:
        _, raw = _http("GET", f"{self.BASE}/batches/{batch_id}", self._h())
        b = json.loads(raw)
        st = b.get("status")
        if st == "completed":
            return "done", json.dumps(b.get("request_counts") or {})
        if st in ("failed", "expired", "cancelled"):
            return "failed", st
        return "pending", st or ""

    def fetch(self, batch_id: str) -> dict[str, Result]:
        _, raw = _http("GET", f"{self.BASE}/batches/{batch_id}", self._h())
        b = json.loads(raw)
        out: dict[str, Result] = {}
        for key in ("output_file_id", "error_file_id"):
            fid = b.get(key)
            if not fid:
                continue
            _, lines = _http("GET", f"{self.BASE}/files/{fid}/content", self._h(None), None, 300)
            for line in lines.decode("utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                resp = (row.get("response") or {}).get("body") or {}
                choices = resp.get("choices") or []
                if choices and not row.get("error"):
                    out[row["custom_id"]] = Result(
                        text=(choices[0].get("message") or {}).get("content") or "")
                else:
                    out[row["custom_id"]] = Result(error=json.dumps(row.get("error")
                                                                    or resp)[:400])
        return out


def default_responder(req: Request) -> str:
    """Canned answers for the fake: a skill spec for a proposal request, a list
    of fresh items for a generation request. The wording deliberately shares
    no vocabulary with the fixture's questions, so nothing here can trip the
    contamination gate by accident — a test that wants a trip plants one."""
    if req.custom_id.startswith("exam:"):
        topic = req.meta.get("topic", "the topic")
        n = int(req.meta.get("count", 4))
        start = int(req.meta.get("start", 0))
        shapes = [
            "Explain why {a} in {t} tends to {b}, and name the one condition under which it "
            "would not. (draft {k})",
            "A student claims that {a} and {b} are the same thing in {t}. State the distinction "
            "and give one consequence of confusing them. (draft {k})",
            "In {t}, what is the standard argument that {a} leads to {b}? Give the mechanism in "
            "two or three sentences. (draft {k})",
            "Describe a situation in {t} where {b} follows from {a}, and what an expert would "
            "measure to confirm it. (draft {k})",
        ]
        fill = ["rising demand", "a fixed constraint", "a change in incentives", "a new rule",
                "an external shock", "a measurement error", "a common assumption", "a feedback loop"]
        out = []
        for i in range(start, start + n):
            a, b = fill[i % len(fill)], fill[(i * 3 + 1) % len(fill)]
            out.append({"prompt": shapes[i % len(shapes)].format(a=a, b=b, t=topic, k=i + 1),
                        "reference": f"The standard account in {topic}: {a} works through the "
                                     f"mechanism that produces {b}; the exception is when the "
                                     f"constraint does not bind.",
                        "notes": f"fake draft {i + 1} for {topic}; checks the mechanism, not recall"})
        return json.dumps(out)
    if req.custom_id.startswith("proposal:"):
        cat = req.meta.get("category", "the category")
        return json.dumps({
            "spec": f"The model lacks the working definitions behind introductory {cat} "
                    f"reasoning: it cannot map a described mechanism to the term that "
                    f"names it, and it confuses direction-of-effect claims.",
            "share_explained": 0.6,
            "patterns": ["picks a term from the right field but the wrong mechanism",
                         "reverses the direction of an effect",
                         "prefers the option that repeats a word from the question"]})
    n = int(req.meta.get("count", 10))
    fmt = req.meta.get("format", "mc")
    start = int(req.meta.get("start", 0))
    nouns = ["a bakery", "a shipping line", "a vineyard", "a bicycle workshop", "a hospital",
             "a fishing cooperative", "a software studio", "a city council", "a dairy farm",
             "a book printer", "a taxi firm", "a night school"]
    verbs = ["raises its output price", "hires a second shift", "faces a new import duty",
             "loses its cheapest supplier", "adopts a faster machine", "is granted a subsidy",
             "sees demand fall", "borrows at a higher rate", "enters a crowded market",
             "cuts advertising", "signs a long lease", "loses a key customer"]
    # eight question shapes and eight rationale shapes, so items from the same
    # spec differ in more than their fillers — the near-duplicate rule is there
    # to catch a generator that templates, and the fake must not be one
    qs = [
        "Suppose {who} {what}. Holding its market otherwise fixed, what does standard "
        "reasoning predict for its per-unit margin over the next quarter? (case {n})",
        "After {who} {what}, a colleague claims revenue must rise. Which single objection "
        "to that claim is decisive, and what actually moves first? Case {n}.",
        "{Who} {what}. Rank the likely effects on cost, price and volume in the order they "
        "appear, and name the one that dominates. This is scenario {n}.",
        "In scenario {n}, {who} {what}, and nothing else changes. Explain the direction of "
        "the effect on its margin and why the other outcomes do not follow.",
        "Two analysts disagree about what happens when {who} {what}. One expects fixed costs "
        "to vanish. State the mistake and the correct first-order effect, case {n}.",
        "Define the term that describes what happens to {who} once it {what}, and give the "
        "first quantity that responds. Question {n} of the set.",
        "Consider {who} in month {n}: it {what}. Compare its situation before and after on "
        "one measure that a manager would watch daily, and say which way it moves.",
        "A newspaper reports that {who} {what}. Which of the reported consequences could be "
        "true within a quarter, and which needs conditions the story does not give? (#{n})",
    ]
    rs = [
        "Pressure from a change like this lands on the margin before anything else responds.",
        "Revenue can move either way; the margin moves first and in a known direction.",
        "Fixed costs are fixed by definition here, so only the variable side reacts at once.",
        "Rivals react over quarters, not days; the immediate effect is internal to the firm.",
        "The first-order effect is on unit economics; the volume response comes later.",
        "Holding everything else fixed is what makes the direction unambiguous.",
        "The decisive objection is that price and volume do not move independently.",
        "A manager watches the per-unit margin because it answers before the totals do.",
    ]
    pools = [
        ["its margin per unit narrows", "its fixed costs fall to zero",
         "its rivals leave the market at once", "its total revenue is unaffected"],
        ["the per-unit margin widens", "output volume is unchanged by definition",
         "the tax base disappears", "customers cannot notice within a quarter"],
        ["variable cost per unit responds first", "the lease term changes the price",
         "demand becomes perfectly elastic", "profit is unaffected by construction"],
        ["the margin absorbs the change first", "fixed costs rise in proportion",
         "every competitor matches the move instantly", "revenue and profit move together"],
        ["unit economics answer before the totals do", "the change has no first-order effect",
         "the workforce is unaffected by definition", "price and volume move independently"],
    ]
    # cycle lengths 8, 7 and 5 are coprime: no two items within 280 share a
    # question shape, an option pool and a rationale at once
    items = []
    for i in range(start, start + n):
        who, what = nouns[i % len(nouns)], verbs[(i * 5) % len(verbs)]
        q = qs[i % 8].format(who=who, Who=who[0].upper() + who[1:], what=what, n=i + 1)
        opts = pools[i % 5]
        item = {"question": q, "answer": opts[0],
                "rationale": rs[(i * 3) % 7] + f" Here the trigger is that it {what}."}
        if fmt == "mc":
            k = i % 4
            item["choices"] = opts[k:] + opts[:k]
        items.append(item)
    return json.dumps(items)


class FakeBatches(Backend):
    """In-process double with on-disk state, so a 'restart' (a fresh client
    object) finds its batches where it left them, and so a test can read the
    exact request bodies that were 'sent'."""
    name = "fake"
    polls_to_done = 1          # tests raise this to watch the poller wait
    responder = staticmethod(default_responder)

    def __init__(self, model: str, root: Path):
        self.model = model or "fake-1"
        self.dir = Path(root) / ".llm-fake"
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def log(self) -> Path:
        return self.dir / "requests.jsonl"

    def recorded(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(x) for x in self.log.read_text().splitlines() if x.strip()]

    def submit(self, requests: list[Request]) -> str:
        bid = "fake_" + uuid.uuid4().hex[:12]
        rows = [{"batch_id": bid, "custom_id": r.custom_id, "system": r.system,
                 "user": r.user, "max_tokens": r.max_tokens, "meta": r.meta} for r in requests]
        with open(self.log, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        (self.dir / f"{bid}.json").write_text(json.dumps({"requests": rows, "polls": 0}))
        return bid

    def _load(self, bid: str) -> dict:
        p = self.dir / f"{bid}.json"
        if not p.exists():
            raise LLMError(f"unknown fake batch {bid}")
        return json.loads(p.read_text())

    def status(self, batch_id: str) -> tuple[str, str]:
        b = self._load(batch_id)
        b["polls"] += 1
        (self.dir / f"{batch_id}.json").write_text(json.dumps(b))
        return ("done", "") if b["polls"] >= self.polls_to_done else ("pending", "in_progress")

    def fetch(self, batch_id: str) -> dict[str, Result]:
        b = self._load(batch_id)
        out = {}
        for row in b["requests"]:
            req = Request(row["custom_id"], row["system"], row["user"], row["max_tokens"],
                          row.get("meta") or {})
            try:
                out[req.custom_id] = Result(text=type(self).responder(req))
            except Exception as e:                       # noqa: BLE001 — a planted failure
                out[req.custom_id] = Result(error=str(e))
        return out


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

PROVIDERS = ("anthropic", "openai", "fake")


# Three identities share this client: the generator (LLM_*), the exam writer
# (EXAM_*) and, from C2, the judge (JUDGE_*). Each is a (provider, model, key)
# triple in config; the same backends serve all three.
ROLES = {"llm": ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "proposals and generation"),
         "exam": ("EXAM_PROVIDER", "EXAM_MODEL", "EXAM_API_KEY", "exam drafting")}


def identity(role: str = "llm") -> tuple[str, str, str]:
    pv, mv, kv, _ = ROLES[role]
    return (getattr(config, pv, "") or "", getattr(config, mv, "") or "",
            getattr(config, kv, "") or "")


def blocked(role: str = "llm") -> str:
    """'' when that identity is usable, else the reason — shown in the UI in
    those words, so nobody wonders why a button is missing."""
    pv, mv, kv, what = ROLES[role]
    p, m, k = identity(role)
    if not p:
        return f"no LLM is configured for {what} on this server ({pv} is unset) — it is off"
    if p not in PROVIDERS:
        return f"{pv}={p!r} is not one of {', '.join(PROVIDERS)}"
    if p != "fake" and not m:
        return f"{mv} is unset — the model must be pinned and recorded"
    if p != "fake" and not k:
        return f"{kv} is unset — put it in .env, never in docker-compose.yml"
    return ""


def startup_check() -> None:
    """A provider that is set but broken fails the container at start, where
    the operator is looking, instead of at the first click days later."""
    for role in ROLES:
        if identity(role)[0] and blocked(role):
            raise RuntimeError(f"{role.upper()} misconfigured: " + blocked(role))


def backend_for(provider: str, model: str, key: str, root: Path | None = None) -> Backend:
    if provider == "anthropic":
        return AnthropicBatches(model, key)
    if provider == "openai":
        return OpenAIBatches(model, key)
    if provider == "fake":
        return FakeBatches(model, root or config.BENCH_ROOT)
    raise LLMError(f"unknown provider {provider!r}")


_clients: dict[str, tuple[tuple, Backend]] = {}


def client(role: str = "llm") -> Backend:
    p, m, k = identity(role)
    key = (p, m, bool(k), str(config.BENCH_ROOT))
    hit = _clients.get(role)
    if hit is None or hit[0] != key:
        why = blocked(role)
        if why:
            raise LLMError(why)
        _clients[role] = (key, backend_for(p, m, k))
    return _clients[role][1]


def reset() -> None:
    """Forget the cached clients (tests re-point BENCH_ROOT between cases; a
    real restart gets this for free)."""
    _clients.clear()


def now() -> float:
    return time.time()
