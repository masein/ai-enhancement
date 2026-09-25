"""One LLM client, batch API only, four backends: anthropic, openai, local, fake.

Batch only is the owner's call and the right one for this job: nothing here is
latency-sensitive — a proposal is read by a person hours later, a dataset is
consumed by a training run days later — and batch pricing is about half of
interactive. It also shapes the code well: every request is a row with a
custom_id, every batch id is persisted before anything else happens, and a
restart resumes polling rather than re-submitting (service/llm_poller.py).

`local` is a vLLM server on the deploy box. It has no batch API, so it keeps
the batch INTERFACE and fulfils it with ordinary chat completions on a worker
thread (LocalOpenAI). Its model id is whatever someone typed at launch — not
dated, changeable with no version to check — so everything a local identity
produces is stamped provisional (local_mark), with no flag to turn it off.

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

import fcntl
import hashlib
import http.client
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import config


class LLMError(RuntimeError):
    """`status` is the HTTP status when there was one — None for a timeout, a
    refused connection, or anything that never reached a response. Callers
    that retry branch on it; everything else reads only the message."""

    def __init__(self, message: str = "", status: int | None = None):
        super().__init__(message)
        self.status = status


class LocalUnreachable(LLMError):
    """Nothing answered at LOCAL_BASE_URL. Transient on a box where vLLM
    restarts: the poller tries again next tick instead of failing the batch."""


@dataclass
class Request:
    custom_id: str
    system: str
    user: str
    max_tokens: int = 2048
    # recorded by the fake backend, never sent anywhere: what the tests need to
    # prove about a request (the doc hashes behind a proposal, the item count)
    meta: dict = field(default_factory=dict)
    # the caller parses the reply as JSON: openai and local put
    # response_format=json_object on the wire, anthropic has no such field and
    # relies on the prompt. extract_json stays the fallback either way
    json: bool = False


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
    # try whichever bracket opens FIRST. Taking the object first would read a
    # one-element array of objects as the object itself — which is exactly what
    # a request for a single item returns, and losing it is silent.
    pairs = [("{", "}"), ("[", "]")]
    pairs.sort(key=lambda p: (text.find(p[0]) if p[0] in text else len(text) + 1))
    for opener, closer in pairs:
        i = text.find(opener)
        j = text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


def extract_array(text: str) -> list | None:
    """The JSON array in a reply. JSON mode (response_format=json_object) may
    only produce an object, so a model asked for an array under it answers
    {"documents": [...]}: an object whose ONE list value is the array reads as
    that array. A bare object is still not an array."""
    obj = extract_json(text)
    if isinstance(obj, dict):
        lists = [v for v in obj.values() if isinstance(v, list)]
        obj = lists[0] if len(lists) == 1 else None
    return obj if isinstance(obj, list) else None


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


def _counted(counts: dict, fallback: str = "") -> str:
    """"40/130 done" from a provider's request counts, whatever it calls them.
    A batch in flight that can only say "in_progress" tells the person nothing
    about whether to wait or go and have lunch."""
    if not isinstance(counts, dict) or not counts:
        return fallback
    done = sum(int(counts.get(k) or 0) for k in ("succeeded", "completed", "errored",
                                                 "failed", "canceled", "cancelled", "expired"))
    total = int(counts.get("total") or 0) or sum(int(v or 0) for v in counts.values()
                                                 if isinstance(v, (int, float)))
    if not total:
        return fallback
    bad = sum(int(counts.get(k) or 0) for k in ("errored", "failed", "expired"))
    return f"{done}/{total} done" + (f", {bad} failed" if bad else "")


def _http(method: str, url: str, headers: dict, body: bytes | None = None,
          timeout: float = 60.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        raise LLMError(f"{method} {url}: HTTP {e.code}: {e.read()[:400].decode(errors='replace')}",
                       status=e.code) from None
    except urllib.error.URLError as e:
        raise LLMError(f"{method} {url}: {e.reason}") from None
    except TimeoutError:
        raise LLMError(f"{method} {url}: no response within {timeout:g}s") from None
    except (OSError, http.client.HTTPException) as e:   # a dropped connection mid-read
        raise LLMError(f"{method} {url}: {e!r}") from None


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
        counts = b.get("request_counts") or {}
        if st == "ended":
            return "done", _counted(counts, json.dumps(counts))
        # the counts are there while it runs too, and they are what a person
        # waiting on a judge batch actually wants to know
        return "pending", _counted(counts, st or "")

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

    @staticmethod
    def body(model: str, r: Request) -> dict:
        """One request's body. JSON mode only when the caller asked for it:
        OpenAI rejects a JSON-mode request whose prompt does not mention JSON."""
        b = {"model": model, "max_tokens": r.max_tokens}
        if r.json:
            b["response_format"] = {"type": "json_object"}
        b["messages"] = [{"role": "system", "content": r.system},
                         {"role": "user", "content": r.user}]
        return b

    def submit(self, requests: list[Request]) -> str:
        lines = [json.dumps({
            "custom_id": r.custom_id, "method": "POST", "url": "/v1/chat/completions",
            "body": self.body(self.model, r)}) for r in requests]
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
        counts = b.get("request_counts") or {}
        if st == "completed":
            return "done", _counted(counts, json.dumps(counts))
        if st in ("failed", "expired", "cancelled"):
            return "failed", st
        return "pending", _counted(counts, st or "")

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


_DOC_SUBJECTS = ["a regional bakery", "a shipping cooperative", "a teaching hospital",
                 "a municipal water board", "a family vineyard", "a second-hand bookshop",
                 "a night school", "a ferry operator"]
_DOC_MOVES = ["raises its prices", "signs a long lease", "loses its largest customer",
              "adopts a faster machine", "faces a new duty", "hires a second shift",
              "opens a second site", "changes its supplier"]
# Twelve paragraph shapes, five to a document, rotated so that two documents
# rarely share more than one. A real generator told to vary register does
# this; the fake has to as well, or the near-duplicate rule would collapse
# every set it writes and the tests would be measuring the wrong thing.
_DOC_PARAS = [
    "Start with what is held fixed. When {who} {move}, nothing else about it changed: the "
    "same suppliers, the same customers, the same building, the same people. That assumption "
    "carries the argument, and it is worth stating out loud rather than leaving implicit, "
    "because the moment it fails everything downstream fails with it.",
    "Trace the first effect before the large one. The quantity that responds first is the one "
    "closest to the change — usually a margin, occasionally a volume, rarely a total. Totals "
    "are sums of parts that move at different speeds, which is why they are the last place to "
    "look and the first place people look.",
    "A worked case helps. Suppose {who} {move} in the second week of a quarter. By the fourth "
    "week two things have moved and three have not; naming which is which, before opening the "
    "figures, is the exercise. Prediction first, then arithmetic, is the order that catches "
    "a wrong mechanism.",
    "The common error is reasoning backwards from the outcome one expected. If the numbers "
    "fall where a rise was predicted, the discipline is to ask which assumption was wrong, "
    "not which cell was mistyped. Direction of effect is a claim about mechanism, and a "
    "mechanism can be stated, checked, and found wanting.",
    "Say what would settle it. For {who} the measurable thing is the per-unit figure over the "
    "following weeks, set against the same figure beforehand with the seasonal pattern "
    "removed. If it moves as the mechanism predicts, the account survives; if it does not, "
    "something else was doing the work.",
    "Distinguish the rule from its purpose. A rule written for one situation is applied in "
    "another, and the two come apart precisely where the reasoning gets interesting. When "
    "{who} {move}, ask what the constraint was for before asking whether it binds.",
    "Two quantities moving together are not thereby related by cause. The third thing that "
    "moved both is usually unglamorous — a season, a price index, a holiday — and it is "
    "almost always cheaper to find than the elaborate story that does without it.",
    "Scale matters more than sign in practice. An effect in the right direction and two "
    "orders of magnitude too small is, for any decision anyone has to take, no effect. Saying "
    "roughly how large, before saying which way, is the harder and more useful half.",
    "Consider the counterexample deliberately. If {who} {move} and the expected consequence "
    "does not follow, what would have had to be true? Naming that condition converts a "
    "confident claim into a testable one, which is the only kind worth arguing about.",
    "Reference case: a firm in a crowded market has less room than the same firm alone in "
    "one, and the difference shows up in how quickly a change is matched. The number of "
    "competitors is not decoration on the problem; it is part of the mechanism.",
    "Write the assumption list before the conclusion. Three or four lines are enough, and the "
    "reader who disagrees can then say which line they reject rather than disputing the "
    "conclusion in general. Arguments that hide their premises cannot be corrected.",
    "In the end, the account has to name a quantity, a direction, a rough size and a way of "
    "being wrong. An explanation missing any of those is a story. The habit of supplying all "
    "four is most of what separates useful reasoning from fluent reasoning.",
]


def _fake_document(i: int) -> dict:
    """A prose training document, long enough and varied enough to pass the
    parser and the near-duplicate rule — the shape a real generator returns.
    Every paragraph carries the case number and two case-specific figures, so
    two documents in a long set do not share 5-word windows the way a
    templated fake otherwise would."""
    who = _DOC_SUBJECTS[i % len(_DOC_SUBJECTS)]
    move = _DOC_MOVES[(i * 3 + i // len(_DOC_SUBJECTS)) % len(_DOC_MOVES)]
    n, week, pct = i + 1, 2 + (i * 5) % 11, 3 + (i * 7) % 17
    picks = [(i * 7 + k * (2 + i % 4)) % len(_DOC_PARAS) for k in range(4 + i % 3)]
    paras = []
    for j, k in enumerate(picks):
        body = _DOC_PARAS[k].format(who=who, move=move)
        tail = (f" In case {n} the figure to watch is the {pct} per cent gap that opened in "
                f"week {week}, and paragraph {j + 1} of note {n} is where that shows.")
        paras.append(body + tail)
    paras.insert(0, f"Case {n}. This note works through what changes, and what does not, when "
                    f"{who} {move} in week {week} of the quarter, and why the order of the "
                    f"steps is the point rather than the arithmetic.")
    return {"title": f"{who.title()} and the first-order effect (case {n})",
            "text": "\n\n".join(paras)}


def default_responder(req: Request) -> str:
    """Canned answers for the fake: a skill spec for a proposal request, a list
    of fresh items for a generation request. The wording deliberately shares
    no vocabulary with the fixture's questions, so nothing here can trip the
    contamination gate by accident — a test that wants a trip plants one."""
    if req.custom_id.startswith("everyday:"):
        # 12a: the judged Everyday questions, answered as the stub would
        from everyday import stub_reply                # scripts/, on sys.path in the service
        return stub_reply(req.user)
    if req.custom_id.startswith(("judge:", "canary:")):
        try:
            from judge import StubGrader              # scripts/, on sys.path in the service
            return StubGrader.reply(req.user)         # single-score or per-criterion, as asked
        except Exception:                             # noqa: BLE001 — a fixed grade beats a crash
            return json.dumps({"score": 2, "justification": "fake grade"})
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
        cat = req.meta.get("topic") or req.meta.get("category") or "the topic"
        return json.dumps({
            "spec": f"The model lacks the working definitions behind introductory {cat} "
                    f"reasoning: it cannot map a described mechanism to the term that "
                    f"names it, and it confuses direction-of-effect claims.",
            "share_explained": 0.6,
            "patterns": ["picks a term from the right field but the wrong mechanism",
                         "reverses the direction of an effect",
                         "prefers the option that repeats a word from the question"]})
    n = int(req.meta.get("count", 10))
    fmt = req.meta.get("format", "doc")
    start = int(req.meta.get("start", 0))
    if fmt == "doc":
        return json.dumps([_fake_document(i) for i in range(start, start + n)])
    if fmt == "chat":
        return json.dumps([_fake_chat(i) for i in range(start, start + n)])
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


def _fake_chat(i: int) -> dict:
    """12g.2: one chat example that passes its own checks — six shapes and
    varied fillers, so the near-duplicate rule does not collapse a set"""
    things = ["mangoes", "bus tickets", "paper cups", "phone chargers", "tomato plants",
              "library books", "bike lights", "coffee pods", "hair ties", "spare keys", "stamps"]
    names = ["priya", "tomas", "wen", "aisha", "leo", "marta", "kofi", "yuki", "sam", "ines"]
    a, b, t, who = 3 + i % 17, 2 + (i * 7) % 13, things[i % len(things)], names[(i * 3) % len(names)]
    shapes = [
        ({"user": f"if i have {a} {t} and {who} gives me {b} more hw many is that",
          "assistant": f"{a + b} {t}."}, [{"type": "number", "value": a + b, "tolerance": 0}]),
        ({"user": f"{who} owes me {a * 10} and paid {b} back, how much left?? #{i}",
          "assistant": f"{a * 10 - b} left to pay."},
         [{"type": "number", "value": a * 10 - b, "tolerance": 0}]),
        ({"user": f"one line only pls: remind {who} to bring {t} on day {i + 1}",
          "assistant": f"Hi {who.title()}, please bring the {t} on day {i + 1}."},
         [{"type": "line_count", "n": 1}, {"type": "max_words", "n": 25}]),
        ({"user": f"can u tell me what {who} thinks about {t} (case {i})",
          "assistant": f"I can't know what {who.title()} thinks about {t} — you'd have to ask them."},
         [{"type": "admits_limit"}]),
        ({"user": f"list {t} and {things[(i + 4) % len(things)]} in that order, nothing else {i}",
          "assistant": f"{t}\n{things[(i + 4) % len(things)]}"},
         [{"type": "in_order", "values": [t, things[(i + 4) % len(things)]]},
          {"type": "max_words", "n": 8}]),
        ({"user": f"no numbers pls, just say if {a} {t} is more than {b} for {who}",
          "assistant": "Yes, it is more." if a > b else "No, it is not more."},
         [{"type": "no_digits"}]),
    ]
    item, checks = shapes[i % len(shapes)]
    return {**item, "checks": checks}


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
                 "user": r.user, "max_tokens": r.max_tokens, "json": r.json,
                 "meta": r.meta} for r in requests]
        with open(self.log, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        self._save(bid, {"requests": rows, "polls": 0})
        return bid

    # the service's poller thread and a test's own tick() both poll, and a
    # poll writes: a reader caught a half-written file and raised where the
    # LLM's own errors are reported. Written aside and moved into place
    def _save(self, bid: str, state: dict) -> None:
        p = self.dir / f"{bid}.json"
        tmp = p.with_suffix(f".json.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, p)

    def _load(self, bid: str) -> dict:
        p = self.dir / f"{bid}.json"
        if not p.exists():
            raise LLMError(f"unknown fake batch {bid}")
        return json.loads(p.read_text())

    def status(self, batch_id: str) -> tuple[str, str]:
        b = self._load(batch_id)
        b["polls"] += 1
        self._save(batch_id, b)
        n = len(b["requests"])
        if b["polls"] >= self.polls_to_done:
            return "done", _counted({"succeeded": n, "total": n})
        # the fake one counts too, so the page's progress line is exercised
        done = min(n, round(n * b["polls"] / max(1, self.polls_to_done)))
        return "pending", _counted({"succeeded": done, "total": n}, "in_progress")

    def fetch(self, batch_id: str) -> dict[str, Result]:
        b = self._load(batch_id)
        out = {}
        for row in b["requests"]:
            req = Request(row["custom_id"], row["system"], row["user"], row["max_tokens"],
                          row.get("meta") or {}, bool(row.get("json")))
            try:
                out[req.custom_id] = Result(text=type(self).responder(req))
            except Exception as e:                       # noqa: BLE001 — a planted failure
                out[req.custom_id] = Result(error=str(e))
        return out


# ---------------------------------------------------------------------------
# local: a vLLM server behind the batch interface
# ---------------------------------------------------------------------------

LOCAL_REASON = "{} by a local model — not a pinned benchmark"
# base url -> {served id: weights id}, filled when a client is constructed, so
# a provenance record can name the weights without another network call
_SERVED: dict[str, dict[str, str]] = {}
_OOM = re.compile(r"out of memory|\boom\b", re.I)
_BATCH_ID = re.compile(r"local_[0-9a-f]{12}")


def local_mark(provider: str, model: str, verb: str, base_url: str | None = None) -> dict:
    """{} for every provider but `local`. For `local`, the mark every artefact
    it produced carries — always, with no flag to turn it off. A local
    server's model id is whatever someone typed at launch: not a dated id,
    changeable with no version to check, and a 4B model is not an instrument
    anyone publishes from. `verb` says what it did: graded, drafted,
    proposed, generated."""
    if provider != "local":
        return {}
    base = (base_url or config.LOCAL_BASE_URL).rstrip("/")
    out = {"provisional": True, "provisional_reason": LOCAL_REASON.format(verb),
           "base_url": base, "served_model": model}
    weights = _SERVED.get(base, {}).get(model)
    if weights:
        out["weights"] = weights
    return out


def provisional(backend: Backend, verb: str) -> dict:
    """local_mark for the backend that did the work."""
    return local_mark(backend.name, backend.model, verb, getattr(backend, "base", None))


class LocalOpenAI(Backend):
    """A local OpenAI-compatible server (vLLM) presented as a batch backend.

    vLLM has no /v1/batches, so this is not OpenAIBatches with another base
    URL — that would 404 on the first submit. submit() writes the batch under
    BENCH_ROOT/llm_batches/local/<id>/ and returns at once; a worker thread
    fulfils it with ordinary chat completions, at most LOCAL_CONCURRENCY at a
    time because the card is shared, and appends each result to disk as it
    lands. status() and fetch() read the disk, so the callers and the poller
    drive it exactly as they drive a real batch — and a restart resumes where
    the last process stopped instead of re-running what was done.

    One request failing never fails the batch. A 5xx, a timeout, a refused
    connection or anything that reads like CUDA OOM is retried after 2, 8 and
    30 seconds, then recorded as that request's error; transient pressure on
    a shared card is expected. A 4xx is recorded at once — a malformed request
    or an unknown model id does not fix itself in 30 seconds.

        requests.jsonl   the batch, written whole before submit() returns
        results.jsonl    one line per finished request, appended as it lands
        worker.lock      flock'd by the live worker; the kernel frees it when
                         the process dies, which is how a restart knows to resume
    """
    name = "local"
    BACKOFF = (2, 8, 30)          # seconds before each retry; the tests set zeros

    def __init__(self, model: str, key: str, root: Path, base_url: str | None = None,
                 concurrency: int | None = None, max_tokens: int | None = None,
                 timeout: float | None = None, role: str = "llm"):
        self.model, self.key = model, key
        self.role = role
        self.base = (base_url or config.LOCAL_BASE_URL).rstrip("/")
        self.concurrency = max(1, int(concurrency or config.LOCAL_CONCURRENCY))
        # the cap belongs to the role: only generation needs the headroom
        self.max_tokens = int(max_tokens or config.local_max_tokens(role))
        self.timeout = float(timeout or config.LOCAL_TIMEOUT_S)
        self.dir = Path(root) / "llm_batches" / "local"
        self.served_models = self._check_served()

    @property
    def cap_name(self) -> str:
        """Which knob capped this reply, so a truncation names the thing to
        raise rather than a knob that may not be the one in force."""
        own = {"llm": config.LOCAL_MAX_TOKENS_LLM, "judge": config.LOCAL_MAX_TOKENS_JUDGE,
               "exam": config.LOCAL_MAX_TOKENS_EXAM}.get(self.role, 0)
        if own:
            return f"LOCAL_MAX_TOKENS_{self.role.upper()}"
        return "LOCAL_MAX_TOKENS" if config.LOCAL_MAX_TOKENS else \
            f"LOCAL_MAX_TOKENS_{self.role.upper()} (default)"

    def _h(self) -> dict:
        h = {"content-type": "application/json"}
        if self.key:              # vLLM ignores it unless launched with --api-key
            h["authorization"] = f"Bearer {self.key}"
        return h

    def _check_served(self) -> list[str]:
        """GET /models once, at construction: "vLLM is up but serves ['chat'],
        not 'gemma'" now beats a 404 on the first click."""
        try:
            _, raw = _http("GET", f"{self.base}/models", self._h(), timeout=10)
        except LLMError as e:
            if e.status is not None:
                raise LLMError(f"the local model server at {self.base} refused GET /models: {e}",
                               status=e.status) from None
            inside = (" This process is in a container, where localhost is the container "
                      "itself: reach the box through the host gateway "
                      "(LOCAL_BASE_URL=http://host.docker.internal:8000/v1, which is what "
                      "docker-compose.yml sets)." if Path("/.dockerenv").exists() else "")
            raise LocalUnreachable(
                f"nothing is answering at {self.base} ({e}).{inside} The vLLM server listens on "
                f"the deploy box's loopback only; from another machine, tunnel it first: "
                f"ssh -L 8000:localhost:8000 <box>") from None
        try:
            data = json.loads(raw).get("data") or []
        except (ValueError, AttributeError):
            raise LLMError(f"{self.base}/models did not return an OpenAI-style model list") from None
        served = {str(m["id"]): str(m.get("root") or "") for m in data
                  if isinstance(m, dict) and m.get("id")}
        if self.model not in served:
            raise LLMError(f"vLLM is up at {self.base} but serves {sorted(served)}, not "
                           f"{self.model!r} — set the model id to one of those")
        _SERVED[self.base] = {k: v for k, v in served.items() if v}
        return sorted(served)

    # -- the batch on disk ---------------------------------------------------

    def _bdir(self, batch_id: str) -> Path:
        d = self.dir / batch_id
        if not _BATCH_ID.fullmatch(batch_id) or not (d / "requests.jsonl").exists():
            raise LLMError(f"unknown local batch {batch_id}")
        return d

    @staticmethod
    def _requests(d: Path) -> list[dict]:
        return [json.loads(x) for x in (d / "requests.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()]

    @staticmethod
    def _results(d: Path) -> dict[str, dict]:
        p = d / "results.jsonl"
        out: dict[str, dict] = {}
        if not p.exists():
            return out
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue          # a line cut short by a crash: that request runs again
            out[row["custom_id"]] = row
        return out

    def submit(self, requests: list[Request]) -> str:
        bid = "local_" + uuid.uuid4().hex[:12]
        d = self.dir / bid
        d.mkdir(parents=True)
        (d / "batch.json").write_text(json.dumps({
            "id": bid, "model": self.model, "base_url": self.base, "n": len(requests),
            "concurrency": self.concurrency, "max_tokens_cap": self.max_tokens,
            "submitted_at": time.time()}), encoding="utf-8")
        # whole or not at all: a batch id the caller persists must never point
        # at half a batch
        tmp = d / "requests.jsonl.tmp"
        tmp.write_text("".join(json.dumps({"custom_id": r.custom_id, "system": r.system,
                                           "user": r.user, "max_tokens": r.max_tokens,
                                           "json": r.json}) + "\n" for r in requests),
                       encoding="utf-8")
        tmp.replace(d / "requests.jsonl")
        self._ensure_worker(bid)
        return bid

    def status(self, batch_id: str) -> tuple[str, str]:
        d = self._bdir(batch_id)
        ids = [r["custom_id"] for r in self._requests(d)]
        res = self._results(d)
        got = [res[c] for c in ids if c in res]
        failed = [r for r in got if r.get("error")]
        cut = sum(1 for r in got if r.get("finish_reason") == "length")
        detail = (f"{len(got)}/{len(ids)} done" + (f", {len(failed)} failed" if failed else "")
                  + (f", {cut} cut off at {self.cap_name}={self.max_tokens}" if cut else ""))
        if len(got) < len(ids):
            self._ensure_worker(batch_id)   # after a restart: resume, never re-run
            return "pending", detail
        if ids and len(failed) == len(ids):
            return "failed", failed[0]["error"]
        return "done", detail

    def fetch(self, batch_id: str) -> dict[str, Result]:
        d = self._bdir(batch_id)
        ids = [r["custom_id"] for r in self._requests(d)]
        res = self._results(d)
        if any(c not in res for c in ids):
            self._ensure_worker(batch_id)
            raise LLMError(f"local batch {batch_id} is not finished: "
                           f"{sum(c in res for c in ids)}/{len(ids)} done")
        return {c: Result(text=res[c].get("text") or "", error=res[c].get("error") or "")
                for c in ids}

    # -- the worker ----------------------------------------------------------

    def _ensure_worker(self, batch_id: str) -> bool:
        """Start a worker unless one already holds this batch, in this process
        or another. False when one does."""
        fd = os.open(self.dir / batch_id / "worker.lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        threading.Thread(target=self._work, args=(batch_id, fd), name=f"llm-{batch_id}",
                         daemon=True).start()
        return True

    def _work(self, batch_id: str, fd: int) -> None:
        try:
            d = self.dir / batch_id
            done = self._results(d)
            out = d / "results.jsonl"
            if out.exists() and not out.read_bytes().endswith(b"\n"):
                with open(out, "a", encoding="utf-8") as fh:   # a crash cut the last line
                    fh.write("\n")                             # short; never glue onto it
            todo: queue.SimpleQueue = queue.SimpleQueue()
            n = 0
            for row in self._requests(d):
                if row["custom_id"] not in done:
                    todo.put(row)
                    n += 1
            write = threading.Lock()

            def drain() -> None:
                while True:
                    try:
                        row = todo.get_nowait()
                    except queue.Empty:
                        return
                    rec = self._complete(row)
                    with write, open(out, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")

            pool = [threading.Thread(target=drain, name=f"llm-{batch_id}-{i}", daemon=True)
                    for i in range(min(self.concurrency, n))]
            for t in pool:
                t.start()
            for t in pool:
                t.join()
        finally:
            os.close(fd)          # releases the flock

    @staticmethod
    def _transient(e: LLMError) -> bool:
        return e.status is None or e.status >= 500 or bool(_OOM.search(str(e)))

    def _complete(self, row: dict) -> dict:
        """One request, synchronously, retried while the failure looks
        transient. Always returns a record — text or error, never a raise."""
        body = {"model": self.model, "max_tokens": min(int(row["max_tokens"]), self.max_tokens),
                "messages": ([{"role": "system", "content": row["system"]}] if row["system"] else [])
                + [{"role": "user", "content": row["user"]}]}
        if row.get("json"):
            body["response_format"] = {"type": "json_object"}
        payload = json.dumps(body).encode()
        rec = {"custom_id": row["custom_id"], "text": "", "error": "", "attempts": 0}
        for wait in (*self.BACKOFF, None):
            rec["attempts"] += 1
            try:
                _, raw = _http("POST", f"{self.base}/chat/completions", self._h(), payload,
                               self.timeout)
            except LLMError as e:
                if wait is None or not self._transient(e):
                    rec["error"] = str(e)[:400]
                    return rec
                time.sleep(wait)
                continue
            try:
                choice = json.loads(raw)["choices"][0]
                rec["text"] = choice["message"].get("content") or ""
                rec["finish_reason"] = choice.get("finish_reason")
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                rec["error"] = f"unreadable reply from {self.base}: {raw[:200]!r}"
            return rec


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

PROVIDERS = ("anthropic", "openai", "local", "fake")
# providers that need no API key: the test double, and a server on the box's
# loopback that ignores one. Both still need a model id except the fake.
KEYLESS = ("fake", "local")


def needs_key(provider: str) -> bool:
    return provider not in KEYLESS


# Three identities share this client: the generator (LLM_*), the exam writer
# (EXAM_*) and, from C2, the judge (JUDGE_*). Each is a (provider, model, key)
# triple in config; the same backends serve all three.
ROLES = {"llm": ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "proposals and generation"),
         "exam": ("EXAM_PROVIDER", "EXAM_MODEL", "EXAM_API_KEY", "exam drafting"),
         "judge": ("JUDGE_PROVIDER", "JUDGE_MODEL", "JUDGE_API_KEY", "judging")}


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
    if p == "local" and not m:
        return (f"{mv} is unset — name the model the local server serves "
                f"(GET {config.LOCAL_BASE_URL}/models lists them)")
    if p != "fake" and not m:
        return f"{mv} is unset — the model must be pinned and recorded"
    if needs_key(p) and not k:
        return f"{kv} is unset — put it in .env, never in docker-compose.yml"
    return ""


def startup_check() -> None:
    """A provider that is set but broken fails the container at start, where
    the operator is looking, instead of at the first click days later."""
    for role in ROLES:
        if role == "judge":
            continue          # the judge's refusals are shown on the page, not fatal (judge.blocked)
        if identity(role)[0] and blocked(role):
            raise RuntimeError(f"{role.upper()} misconfigured: " + blocked(role))


def backend_for(provider: str, model: str, key: str, root: Path | None = None,
                role: str = "llm") -> Backend:
    if provider == "anthropic":
        return AnthropicBatches(model, key)
    if provider == "openai":
        return OpenAIBatches(model, key)
    if provider == "local":
        return LocalOpenAI(model, key, root or config.BENCH_ROOT, role=role)
    if provider == "fake":
        return FakeBatches(model, root or config.BENCH_ROOT)
    raise LLMError(f"unknown provider {provider!r}")


_clients: dict[str, tuple[tuple, Backend]] = {}


def client(role: str = "llm") -> Backend:
    """The backend for one identity, built once per process. For `local`,
    building it asks the server what it serves, so this can raise
    LocalUnreachable (try later) or LLMError (the configuration is wrong)."""
    p, m, k = identity(role)
    key = (p, m, bool(k), str(config.BENCH_ROOT), config.LOCAL_BASE_URL if p == "local" else "")
    hit = _clients.get(role)
    if hit is None or hit[0] != key:
        why = blocked(role)
        if why:
            raise LLMError(why)
        _clients[role] = (key, backend_for(p, m, k, role=role))
    return _clients[role][1]


def reset() -> None:
    """Forget the cached clients (tests re-point BENCH_ROOT between cases; a
    real restart gets this for free)."""
    _clients.clear()


def now() -> float:
    return time.time()
