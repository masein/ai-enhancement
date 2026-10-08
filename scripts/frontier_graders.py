"""17, stage 3: the Frontier benchmarks' graders — each one's prompt, word
for word (scripts/grader_prompts/), how it is filled in and how its reply is
read. No service import: service/frontier_grade.py sends them, on the server
only, after masein's Start.

- SimpleQA Verified: Google's grader prompt from its own starter code (the
  Kaggle notebook it publishes with the dataset); graded by gpt-4.1-2025-04-14
  in Google's runs. The share graded CORRECT is the score (Epoch's). 17c: a
  grade only in its exact form — a lone letter, a lone CORRECT, INCORRECT or
  NOT_ATTEMPTED, or the prompt's own "B: INCORRECT", with only punctuation
  or markup around it; anything else is no grade, asked again. Google's own
  reading (the first A, B or C anywhere) read "NOT_ATTEMPTED" and "As an
  AI…" as A, and 17b's whole words "not correct" as correct.
- Humanity's Last Exam: CAIS's judge prompt (hle_eval/run_judge_results.py,
  its typos and |\\%| as they are), judged by o3-mini-2025-01-31 in CAIS's
  runs, at its default reasoning effort and a 4,096-token cap. CAIS asks for
  structured output; its fields are read from the reply here, as text
  ("correct: yes") or JSON ("correct": "yes"), the last one given, its value
  alone ("correct: yes/no" isn't one). 17b: a reply nothing can be read from
  isn't a grade: asked again — 17c: its words never shown (the judge may
  quote HLE's question, which is gated).
- MATH Level 5: Epoch AI's equivalence prompt (its scorer.py), expression 1
  the key and 2 the answer as Epoch's code extracted it; "Yes" is right (17c:
  yes or no alone, punctuation or markup around it; anything else, asked
  again — it was a final "not equivalent"). Epoch
  named gemini-1.5-flash-002, which is retired. Asked only of the answers the
  code marks wrong (an answer the code can't read, Epoch scores wrong without
  asking).
- OTIS Mock AIME: Epoch extracts the final answer with a model that doesn't
  see the key, then matches it exactly; its prompt isn't published, so the
  one here is ours (otis_extract.txt). Asked only of the answers the code marks
  wrong or can't read. An integer is read; NONE is no final answer (wrong);
  anything else isn't a grade.

17b: each grader's reasoning is set, never left to the model's default
(`reasoning`, OpenRouter's switch, which a model that doesn't reason
ignores), and its cap sized for it — reasoning tokens count against the cap.
The owners' graders as their owners ran them: gpt-4.1 doesn't reason;
o3-mini at medium, its default, which CAIS kept; Epoch's MATH check,
gemini-1.5-flash-002, didn't reason, so neither does its suggested successor.
A chosen model that reasons is given REASONING_ROOM more, in case it can't be
switched off. An empty reply, or one cut at the cap, is never a grade.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "grader_prompts"
REASONING_ROOM = 2048       # 17b: a model that reasons when told not to still answers
OFF = {"enabled": False}

GRADERS: dict[str, dict] = {
    "simpleqa": {
        "label": "SimpleQA Verified's grader", "task": "simpleqa_epoch",
        "prompt": "simpleqa_google.txt", "prompt_words": "Google's grader prompt",
        "suggested": "openai/gpt-4.1", "owners": "gpt-4.1-2025-04-14",
        "why": "the model Google grades SimpleQA Verified with",
        "does": "grades every answer CORRECT, INCORRECT or NOT_ATTEMPTED",
        "reasoning": OFF, "reasoning_words": "reasoning off, as gpt-4.1 answers",
        "max_tokens": 16, "out_tokens": 2},
    "hle": {
        "label": "Humanity's Last Exam's judge", "task": "hle_text_cais",
        "prompt": "hle_judge.txt", "prompt_words": "CAIS's judge prompt",
        "suggested": "openai/o3-mini", "owners": "o3-mini-2025-01-31",
        "why": "the judge CAIS grades Humanity's Last Exam with",
        "does": "judges every answer against the key, yes or no",
        # o3-mini thinks before it answers: its reasoning tokens are billed, and
        # count against CAIS's own cap (max_completion_tokens=4096)
        "reasoning": {"effort": "medium"},
        "reasoning_words": "reasoning at medium effort, o3-mini's default, as CAIS ran it",
        "max_tokens": 4096, "out_tokens": 900},
    "math": {
        "label": "MATH Level 5's equivalence check", "task": "math_l5_epoch",
        "prompt": "math_equivalence.txt", "prompt_words": "Epoch AI's equivalence prompt",
        "suggested": "google/gemini-2.5-flash", "owners": "gemini-1.5-flash-002 (retired)",
        "why": "Epoch's check, gemini-1.5-flash-002, is retired: the same family's",
        "does": "a second look at the answers the code marks wrong: equivalent to the key?",
        "reasoning": OFF, "reasoning_words": "reasoning off, as gemini-1.5-flash-002 answered",
        "max_tokens": 16, "out_tokens": 2, "look": True},
    "otis": {
        "label": "OTIS Mock AIME's extractor", "task": "otis_aime_epoch",
        "prompt": "otis_extract.txt", "prompt_words": "our extraction prompt (Epoch's isn't "
                                                       "published)",
        "suggested": "google/gemini-2.5-flash", "owners": "not published",
        "why": "a small model that reads an answer well; Epoch's isn't named",
        "does": "a second look at the answers the code marks wrong or can't read: the final "
                "answer, read without the key",
        "reasoning": OFF, "reasoning_words": "reasoning off: it copies a number out",
        "max_tokens": 32, "out_tokens": 4, "look": True},
}
SLOT_OF = {g["task"]: s for s, g in GRADERS.items()}


def prompt_text(slot: str) -> str:
    return (PROMPTS / GRADERS[slot]["prompt"]).read_text(encoding="utf-8")


def prompt_sha(slot: str) -> str:
    """the grader's prompt, pinned: shown beside the scores it produced"""
    return hashlib.sha256(prompt_text(slot).encode("utf-8")).hexdigest()


def ask(slot: str, reasons: bool | None = None, structured: bool | None = None) -> dict:
    """17b: what is sent with each answer, and what the dry run counts —
    {max_tokens, reasoning, out_tokens, schema}. `reasons`: whether OpenRouter
    lists the chosen model as one that reasons. 17f: `structured`, whether it
    takes a JSON schema — HLE's judge is then asked for its four fields as
    JSON only, as CAIS asked o3-mini"""
    g = GRADERS[slot]
    cap = g["max_tokens"]
    if reasons and g["reasoning"] == OFF:
        cap += REASONING_ROOM
    return {"max_tokens": cap, "reasoning": dict(g["reasoning"]), "out_tokens": g["out_tokens"],
            "schema": HLE_SCHEMA if slot == "hle" and structured else None}


def _fill(t: str, values: dict) -> str:
    """17b: every placeholder filled in one pass — an answer holding
    "{correct_answer}" stays as it is, never the key"""
    return re.sub(r"\{(" + "|".join(map(re.escape, values)) + r")\}",
                  lambda m: values[m.group(1)], t)


def render(slot: str, item: dict, run: dict) -> str:
    """the grader's message for one answer: `item` the question as the dataset
    has it, `run` the answer as service/frontier.marks scored it"""
    t = prompt_text(slot)
    answer = _visible(run.get("answer") or "")
    if slot == "simpleqa":
        return _fill(t, {"question": item["question"], "target": item["answer"],
                         "predicted_answer": answer})
    if slot == "hle":
        return _fill(t, {"question": item["question"], "response": answer,
                         "correct_answer": item["answer"]})
    if slot == "math":
        return t % {"expression1": item["answer"], "expression2": run.get("read") or ""}
    if slot == "otis":
        return _fill(t, {"response": answer})
    raise KeyError(slot)


def think_split(text: str) -> tuple[str, list[str], bool]:
    """18b: (the text outside every thinking block, each block's thinking,
    whether one never closed) — every block wherever it sits, not only one
    at the start; a reply that begins inside its thinking (its template
    opened <think> in the prompt: a </think> before any <think>) has that
    first part as thinking; a block never closed runs to the end. Linear:
    found by position, never by a pattern that backtracks"""
    t = text or ""
    thoughts: list[str] = []
    c, o = t.find("</think>"), t.find("<think>")
    if c >= 0 and (o < 0 or c < o):
        thoughts.append(t[:c])
        t = t[c + len("</think>"):]
    out, i, unclosed = [], 0, False
    while True:
        o = t.find("<think>", i)
        if o < 0:
            out.append(t[i:])
            break
        out.append(t[i:o])
        c = t.find("</think>", o + len("<think>"))
        if c < 0:
            thoughts.append(t[o + len("<think>"):])
            unclosed = True
            break
        thoughts.append(t[o + len("<think>"):c])
        i = c + len("</think>")
    return "".join(out).replace("</think>", "").strip(), thoughts, unclosed


def _visible(text: str) -> str:
    """the answer the grader is given: the text outside every thinking block"""
    return think_split(text)[0]


_SQA_WORDS = {"A": "correct", "B": "incorrect", "C": "not attempted"}


_SQA_OF = {"correct": "A", "incorrect": "B", "not_attempted": "C"}


def _last(pattern: str, text: str):
    ms = list(re.finditer(pattern, text))
    return ms[-1] if ms else None


# 17c: a grade only in its exact expected form — a lone letter, word or yes/no,
# with only punctuation or markup around it ("**B**", "(A)", "Yes.")
_WRAP = re.compile(r"^[\s*_`\"'#>(\[]+|[\s*_`\"'.!:)\]]+$")


def lone(text: str) -> str:
    """the reply without the punctuation and markup around it"""
    return _WRAP.sub("", text or "")


def _close_of(text: str, i: int) -> int | None:
    """17i: where the object opening at `i` closes — quotes of either kind
    respected — or None when it never does (a reply cut off)"""
    depth, q, esc = 0, None, False
    for j in range(i, len(text)):
        c = text[j]
        if q:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == q:
                q = None
        elif c in "\"'" and re.search(r"[{\[,:]\s*$", text[max(i, j - 40):j]):
            q = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return None


def _close_braces(text: str, i: int) -> int | None:
    """17j: where the object opening at `i` closes, counting braces alone —
    for one whose quotes can't be trusted (a quote left unescaped inside a
    string, a quoted object inside its reasoning)"""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return None


def _in_strings(raw: str, fix) -> str:
    """17j: `raw` with `fix` applied to each character inside a double-quoted
    string — a raw newline there made the whole object unread"""
    out, q, esc = [], False, False
    for c in raw:
        if q:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                q = False
            else:
                c = fix(c)
        elif c == '"':
            q = True
        out.append(c)
    return "".join(out)


def _mend(raw: str) -> str:
    """17i: raw LaTeX backslashes ("\\alpha = 2") and trailing commas mended.
    17j: and a backslash before a u that isn't four hex digits ("\\uparrow",
    "\\underline"), a raw newline or tab inside a string, and a verdict left
    unquoted ("correct": yes)"""
    t = re.sub(r'\\(?![\\"/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", raw)
    t = _in_strings(t, lambda c: {"\n": "\\n", "\t": "\\t", "\r": "\\r"}.get(c, c))
    t = re.sub(r'("correct"\s*:\s*)(yes|no|true|false)\b', r'\1"\2"', t, flags=re.I)
    return re.sub(r",\s*([}\]])", r"\1", t)


def _lenient(raw: str):
    """17i: an object as a person reads it — JSON, else mended (_mend), else
    in single quotes (a Python literal); None when none of these reads"""
    import ast
    import json
    for t in (raw, _mend(raw)):
        try:
            got = json.loads(t)
            return got if isinstance(got, dict) else None
        except ValueError:
            continue
    try:
        got = ast.literal_eval(re.sub(r"\b(true|false|null)\b",
                                      lambda m: {"true": "True", "false": "False",
                                                 "null": "None"}[m.group(1)], raw))
        return got if isinstance(got, dict) else None
    except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
        return None


FIELDS = ("extracted_final_answer", "reasoning", "correct", "confidence")
# 17j: an object a verdict can be in opens with one of its fields' names — a
# brace in prose, LaTeX ("\left\{ … \right.") or code ("for (…) {") is none
# 18b: a name in quotes may follow on the next line; a bare name only on the
# same line — "S = {" ending a line, then a "reasoning:" line, is no object
_NAMES = "|".join(FIELDS)
_OPENS = re.compile(r"\{(?:\s*([\"'])(" + _NAMES + r")\1\s*|[ \t]*()(" + _NAMES
                    + r")[ \t]*):")
_KEY_AT = re.compile(r"[{,](?:\s*([\"'])(" + _NAMES + r")\1\s*|[ \t]*()(" + _NAMES
                     + r")[ \t]*):")


def _name(m) -> str:
    return m.group(2) or m.group(4)
AMBIGUOUS = "ambiguous"


def _by_fields(raw: str):
    """17j: an object that reads as nothing else, field by field, as a person
    reads it — each field's value runs to the next field's name (an
    apostrophe in a single-quoted object, a quote left unescaped inside a
    string). AMBIGUOUS when a field is named twice (an object quoted inside
    its reasoning): nothing in it is read; None when it holds no verdict"""
    keys = list(_KEY_AT.finditer(raw))
    names = [_name(m) for m in keys]
    if "correct" not in names:
        return None
    if len(set(names)) != len(names):
        return AMBIGUOUS
    end = raw.rstrip().rfind("}")
    out = {}
    for k, m in enumerate(keys):
        stop = keys[k + 1].start() if k + 1 < len(keys) else end
        v = raw[m.end():stop].strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[_name(m)] = v
    return out


def _json_objects(text: str) -> tuple[list[tuple[int, int, object]], bool]:
    """17f: every object in the reply that stands alone — (start, end,
    object), objects inside another one never counted. 17i: read as a person
    reads it (_lenient); and whether the reply ends cut. 17j: only an object
    that opens with a verdict field's name is one — it is cut only when such
    an object never closes; one whose quotes can't be trusted is read by its
    braces and field by field, and nothing quoted inside it is read (it was
    the verdict); the object is AMBIGUOUS when it can't be told apart"""
    out, i = [], 0
    while True:
        m = _OPENS.search(text, i)
        if not m:
            return out, False
        a = m.start()
        strict, loose = _close_of(text, a), _close_braces(text, a)
        if strict is None and loose is None:
            return out, True
        got, b = None, None
        for end in dict.fromkeys(x for x in (strict, loose) if x):
            got = _lenient(text[a:end])
            if isinstance(got, dict):
                b = end
                break
        if b is None:
            b = loose or strict
            got = _by_fields(text[a:b])
        if got is not None:
            out.append((a, b, got))
        i = b


def _outer_verdicts(text: str, found: list) -> list:
    """18b: an object around the verdict whose first key isn't a field's name
    ({"judgement": {…}, "correct": "no"}) — whole JSON with "correct" at its
    top. It is the verdict with every object inside it: one verdict, or
    AMBIGUOUS when they disagree"""
    import json
    dec = json.JSONDecoder()
    outer: list = []
    for k, m in enumerate(re.finditer(r'\{\s*"', text)):
        a = m.start()
        if k > 200 or any(x <= a < y for x, y, _ in outer):
            continue
        try:
            o, b = dec.raw_decode(text, a)
        except ValueError:
            continue
        if isinstance(o, dict) and "correct" in o and next(iter(o)) not in FIELDS:
            outer.append((a, b, o))
    for x, y, o in outer:
        inside = [f for f in found if x <= f[0] and f[1] <= y]
        says = {_yes_no(o.get("correct"))} | {
            _yes_no(f[2].get("correct")) for f in inside
            if isinstance(f[2], dict) and "correct" in f[2]}
        found = [f for f in found if f not in inside] + [(x, y, o if len(says) == 1
                                                          else AMBIGUOUS)]
    return sorted(found, key=lambda f: f[0])


def _quoted(line: str) -> bool:
    """18b: a line quoted in markdown (> …): the response's words, not the judge's"""
    return line.lstrip().startswith(">")


def whole_verdict(slot: str, text: str) -> bool:
    """18b point 19: a reply that reached its cap is graded only when it is
    one whole verdict object — nothing before or after it, nothing cut. A
    reply in the line form, or cut mid-reasoning after a quoted "correct:
    yes", is cut. The other graders read a lone grade: theirs stands"""
    if slot != "hle":
        return True
    import json
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip()).strip()
    if not t.startswith("{") or _close_of(t, 0) != len(t):
        return False
    try:
        o = json.loads(t)
    except ValueError:
        o = _lenient(t) if _OPENS.match(t) else None
    return isinstance(o, dict) and _yes_no(o.get("correct")) is not None


def _norm(t: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", str(t).lower()).split())


_FIELDS: dict = {}


def _field_text(field: str) -> str:
    """17i: the HLE judge prompt's own text for one field — what an example
    object copies from it"""
    if not _FIELDS:
        for line in prompt_text("hle").splitlines():
            m = re.match(r"\s*(extracted_final_answer|reasoning|correct|confidence)\s*:\s*(.+)",
                         line)
            if m:
                _FIELDS[m.group(1)] = _norm(m.group(2))
    return _FIELDS.get(field, "")


# 17j: a placeholder in angle brackets, "..." — and in square brackets only
# the prompt's own words ("[response]"): "[Verse]" is an answer
_PLACEHOLDER = re.compile(r"\[(?:response|correct_answer|question)\]|\.\.\.|…")
# 18b: an angle-bracket placeholder found by its brackets first, then its
# letters — the one pattern took 2.7 s on a 1,200-letter value
_ANGLE = re.compile(r"<([^<>]*+)>")


def _placeholder(v: str) -> bool:
    if _PLACEHOLDER.fullmatch(v):
        return True
    m = _ANGLE.fullmatch(v)
    return bool(m and re.search(r"[A-Za-z]{3}", m.group(1)))


def _template(field: str, v) -> bool:
    """17i: a field's value as an example writes it — the prompt's own text for
    THAT field, whole or from its start (three words at least), or a
    placeholder ("<the extracted answer>"). Never by what an answer looks
    like: "[0, 1]", "1939-1945", "<1, 0, 0>" and "number" are answers"""
    if field == "confidence":
        # a real one is a number; anything else is an example's ("<0-100>").
        # 100 is the prompt's own value for it ("Put 100 if there is no
        # confidence score available"): with every other field an example's,
        # the object is one
        num = re.fullmatch(r"\s*(\d{1,3}(?:\.\d+)?)\s*%?\s*", str(v)) \
            if not isinstance(v, bool) else None
        return num is None or float(num.group(1)) == 100
    if not isinstance(v, str):
        return False
    if _placeholder(v.strip()):
        return True
    n = _norm(v)
    own = _field_text(field)
    return bool(own) and len(n.split()) >= 3 and own.startswith(n)


def _example(o: dict) -> bool:
    """17i: an example object — every field but its verdict an example's
    value (_template); one real field makes it real. 17h took ANY field
    holding ANY fragment of the prompt, and threw real verdicts away. 17j:
    never one without its answer and its reasoning ({"correct": "no",
    "confidence": 100} is a terse verdict)"""
    if "extracted_final_answer" not in o or "reasoning" not in o:
        return False
    rest = [(k, v) for k, v in o.items() if k != "correct"]
    return all(_template(k, v) for k, v in rest)


_W = r"[\s*_`\"'#>-]*"


def _hle_object(text: str) -> tuple[str, dict | None, str]:
    """17g: the reply's verdict object — ("grade", it), ("conflict", None),
    or ("none", None) — and 17j: the reply's text outside its objects, where
    a verdict in the prompt's line form is read. Objects quoted the same
    twice are one; an example (_example) is none; a reply that ends cut, or
    an object that can't be told apart (AMBIGUOUS), is no grade. Objects
    that disagree are no grade — 17j: but an object alone on its line wins
    over one inside a sentence that holds nothing but its verdict (the
    response's own {"correct": "yes"}, quoted); a fuller one quoted inside a
    sentence is a second verdict. A reply in the prompt's line form has its
    verdict on its "correct:" line, never in an object it quotes inline; a
    "correct:" line that says otherwise than the object is a conflict"""
    import json
    found, cut = _json_objects(text)
    found = _outer_verdicts(text, found)
    outside, k = "", 0
    for a, b, _ in found:
        outside += text[k:a] + "\n"
        k = max(k, b)
    outside += text[k:]
    if cut or any(o is AMBIGUOUS for _, _, o in found):
        return "conflict", None, outside
    objs = [(a, b, o) for a, b, o in found if "correct" in o]
    real = [(a, b, o) for a, b, o in objs
            if _yes_no(o.get("correct")) is not None and not _example(o)]

    def by_itself(a: int, b: int) -> bool:
        """nothing but markup before it on its line, and after it"""
        end = text.find("\n", b)
        return not re.sub(r"[\s`*_>#-]|json", "", text[text.rfind("\n", 0, a) + 1:a]) \
            and not re.sub(r"[\s`*_.,;]", "", text[b:end if end >= 0 else len(text)])
    # 18c point 17: an object holding nothing but its verdict is the judge's
    # only when it ends the reply — {"correct": "yes"} on its line, then prose
    # saying the answer is wrong, is the response's, quoted. (The judge's
    # reasoning, then its verdict object last, stays its grade)
    def last(b: int) -> bool:
        return not re.sub(r"[\s`*_>#.,;:-]|json", "", text[b:])
    alone = [(a, b, o) for a, b, o in real if by_itself(a, b)
             and (set(o) - {"correct"} or last(b))]
    lines = bool(re.search(rf"(?im)^{_W}(extracted_final_answer|reasoning){_W}:", outside))
    # 18b: inside a sentence, an object that holds nothing but its verdict is
    # a quote (the response's own {"correct": "yes"}, or {correct: yes}) —
    # never the judge's verdict
    pool = alone or ([] if lines else [x for x in real if set(x[2]) - {"correct"}])
    if not pool:
        return "none", None, outside
    v_alone = {_yes_no(o["correct"]) for _, _, o in alone}
    others = [o for a, b, o in real if (a, b, o) not in alone
              and _yes_no(o["correct"]) not in v_alone]
    if alone and others and (len(v_alone) > 1 or any(set(o) - {"correct"} for o in others)):
        return "conflict", None, outside
    distinct = list({json.dumps(o, sort_keys=True, default=str): o
                     for _, _, o in pool}.values())
    verdicts = {_yes_no(o["correct"]) for o in distinct}
    if len(verdicts) > 1:
        return "conflict", None, outside
    said = {m.group(1).lower() for m in
            re.finditer(rf"(?im)^{_W}correct{_W}:{_W}(yes|no)\b", outside)
            if not _quoted(m.group(0))}
    if said <= verdicts:
        return "grade", distinct[-1], outside
    return ("conflict", None, outside) if alone else ("none", None, outside)


def _yes_no(v) -> str | None:
    """17f: a verdict's value — yes and no, and JSON's true and false"""
    if isinstance(v, bool):
        return "yes" if v else "no"
    w = str(v).strip().strip(".").lower()
    return {"yes": "yes", "no": "no", "true": "yes", "false": "no"}.get(w)


# 17f: CAIS's judge as CAIS ran it — OpenAI's structured outputs, its four
# fields — asked where the grader takes a JSON schema
HLE_SCHEMA = {"type": "object", "additionalProperties": False,
              "required": ["extracted_final_answer", "reasoning", "correct", "confidence"],
              "properties": {"extracted_final_answer": {"type": "string"},
                             "reasoning": {"type": "string"},
                             "correct": {"type": "string", "enum": ["yes", "no"]},
                             "confidence": {"type": "integer"}}}


def unread(text: str, show: bool = True) -> dict:
    """17b: a reply that isn't a grade — never kept as one; the next Start
    asks again. 17c: its words without the reply for a benchmark that is
    never shown (the judge may quote its question)"""
    return {"ok": None, "unread": "the grader's reply isn't a grade"
            + ((f": “{text.strip()[:80]}”" if text.strip() else ": it was empty") if show
               else ("" if text.strip() else ": it was empty")),
            "said": text.strip()[:200] if show else ""}


def read(slot: str, text: str, item: dict) -> dict:
    """the grader's reply, as its owners read it: {ok, words, said} — `said`
    what it answered, short — or 17b's unread(): {ok: None, unread}"""
    text = text or ""
    shown = slot not in ("hle", "math")            # 17c: never a gated question's words
    said = text.strip()[:200] if shown else ""
    if not text.strip():
        return unread(text, shown)
    if slot == "simpleqa":
        # 17c: a lone letter (as Google's reading takes it), a lone grade word,
        # or the prompt's own "B: INCORRECT" — never a word out of a sentence
        core = lone(text)
        m = re.fullmatch(r"([ABC])(?:\s*[:.)\-]\s*((?i:CORRECT|INCORRECT|NOT[ _]ATTEMPTED)))?",
                         core)
        w = re.fullmatch(r"(?i)(CORRECT|INCORRECT|NOT[ _]ATTEMPTED)", core)
        if m:
            g = m.group(1)
            if m.group(2) and _SQA_OF[re.sub(r"[ _]", "_", m.group(2).lower())] != g:
                return unread(text)
        elif w:
            g = _SQA_OF[re.sub(r"[ _]", "_", w.group(1).lower())]
        else:
            return unread(text)
        return {"ok": g == "A", "words": _SQA_WORDS[g], "said": said, "grade": g}
    if slot == "hle":
        # CAIS's fields: as JSON (its structured output), else each at the start
        # of its own line — 17d: the last such "correct:" line, its value
        # alone; never a "correct: no" inside the judge's reasoning, and a
        # carriage return isn't part of the value
        t = text.replace("\r", "")
        # 17g: two identical objects, an example before the real one, and an
        # object with a "Correct:" line after it are read; objects that say
        # otherwise are no grade
        how, fields, outside = _hle_object(t)
        if how == "conflict":
            return unread(text, shown)
        if fields is not None:
            ok = _yes_no(fields["correct"]) == "yes"
            ext = str(fields.get("extracted_final_answer") or "").strip()
            conf = re.fullmatch(r"\s*(\d{1,3})\s*%?\s*", str(fields.get("confidence", "")))
            return {"ok": ok, "words": ("the judge: correct" if ok else "the judge: incorrect")
                    + (f" · read as {ext[:60]}" if ext else ""), "said": said,
                    **({"confidence": int(conf.group(1))} if conf else {})}
        # 17j: the line form read from the text outside every object — the
        # "correct": "yes" line of an example printed over several lines,
        # alone, was taken for the verdict
        w = _W
        # 18b: a line quoted from the response (> correct: no) is not the
        # judge's verdict — and one that says otherwise than the judge's is a
        # second verdict: no grade. The judge's own last line stands (17d)
        found = list(re.finditer(rf"(?im)^{w}correct{w}:{w}(yes|no)[\s*_`\"'.]*$", outside))
        own = [x for x in found if not _quoted(x.group(0))]
        m = own[-1] if own else None
        if m and any(x.group(1).lower() != m.group(1).lower() for x in found
                     if _quoted(x.group(0))):
            return unread(text, shown)
        conf = _last(rf"(?im)^{w}confidence{w}:{w}(\d{{1,3}})\s*%?[\s*_`\"'.]*$", outside)
        ext = _last(rf"(?im)^{w}extracted_final_answer{w}:\s*(.+)$", outside)
        if not m:
            return unread(text, shown)
        ok = m.group(1).lower() == "yes"
        return {"ok": ok, "words": ("the judge: correct" if ok else "the judge: incorrect")
                + (f" · read as {ext.group(1).strip()[:60]}" if ext else ""),
                "said": said, **({"confidence": int(conf.group(1))} if conf else {})}
    if slot == "math":
        # 17c: yes or no, alone — "Yes." and "**Yes**" are yes; anything else
        # is no grade, where it was a final "not equivalent"
        core = lone(text).lower()
        if core not in ("yes", "no"):
            return unread(text, shown)
        ok = core == "yes"
        return {"ok": ok, "words": "equivalent" if ok else "not equivalent", "said": said}
    if slot == "otis":
        if re.fullmatch(r"\W*NONE\W*", text, re.I):
            return {"ok": False, "words": "no final answer", "said": said}
        m = re.fullmatch(r"\s*\**\s*(-?\d{1,4})\s*\**\.?\s*", text)
        if not m:
            return unread(text)
        got = int(m.group(1))
        ok = got == int(str(item["answer"]).strip())
        return {"ok": ok, "words": f"read as {got}", "said": said, "read": str(got)}
    raise KeyError(slot)
