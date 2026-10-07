"""17h: what the raw-run exports may write — by a list of what is allowed,
never by scrubbing what isn't. Three rounds of scrubbing by pattern each left
holes (a key as its own element of a list, an env string, a quoted value, a
question's middle); here everything not on a list is left out, and said.

- Fields: `pick(obj, schema)` keeps a field only when the schema names it,
  with a value of its kind; a string by its own rule (`TEXT` short plain
  words, `NAME` a file or model name, `ANY` text our code wrote).
- Launch flags and the environment: `flags()` keeps the llama-server flags in
  FLAGS, each value a plain token; `env()` the variables ENV_NAMES allows,
  never one whose name holds KEY, TOKEN, SECRET or PASS.
- Log lines: `log_lines()` keeps only lines the runner itself writes, in its
  fixed shapes (LOG_SHAPES), and leaves out any of those that holds an
  address, a URL, a user@host, a key's name or a quote of a gated question
  (`Questions`); with no list of the questions to check against, no log.
- Public: `public()` — a model the board was told is public (its page, or the
  import's --public-weights), and nothing else; `confirm()` lists what would
  go to public/ and waits for a typed yes.

What still goes through export_devicemark_raw.scrub, and why: the lines and
fields this lets through are our own, but they hold paths and host names our
code puts there (a home path in a "bundle /home/…" line, this server's name
in a header) — the scrub takes those, as before, and is the last pass, never
the first."""

from __future__ import annotations

import json
import re
import shlex
import sys

# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------

TEXT, NAME, ANY, SCALAR = "text", "name", "any", "scalar"
_KEY = re.compile(r"[\w.,+@ -]{1,80}")
_TEXT = re.compile(r"[\w .,:;()/+%·–—'’-]{0,200}", re.U)
_NAME = re.compile(r"[\w.+@-]{1,160}(/[\w.+@-]{1,160}){0,2}")


def _ok_str(v: str, kind: str) -> bool:
    if kind == ANY:
        return len(v) <= 2000
    if kind == NAME:
        return bool(_NAME.fullmatch(v))
    return bool(_TEXT.fullmatch(v))


def pick(obj, schema):
    """`obj` with only what `schema` allows: a dict schema names its fields
    ({"*": s} for any key), [s] a list of s, a type (int, float, bool) a
    value of it, TEXT/NAME/ANY a string; None when nothing is left"""
    if isinstance(schema, dict):
        if not isinstance(obj, dict):
            return None
        out = {}
        for k, v in obj.items():
            s = schema.get(k, schema.get("*"))
            if s is None or not isinstance(k, str) or not _KEY.fullmatch(k):
                continue
            got = pick(v, s)
            if got is not None:
                out[k] = got
        return out
    if isinstance(schema, list):
        if not isinstance(obj, list):
            return None
        return [x for x in (pick(v, schema[0]) for v in obj) if x is not None]
    if schema == SCALAR:
        if isinstance(obj, (bool, int, float)):
            return obj
        return obj if isinstance(obj, str) and _ok_str(obj, TEXT) else None
    if schema in (TEXT, NAME, ANY):
        return obj if isinstance(obj, str) and _ok_str(obj, schema) else None
    if schema is float:
        return obj if isinstance(obj, (int, float)) and not isinstance(obj, bool) else None
    if schema is int:
        return obj if isinstance(obj, int) and not isinstance(obj, bool) else None
    if schema is bool:
        return obj if isinstance(obj, bool) else None
    return None


# ---------------------------------------------------------------------------
# launch flags and the environment
# ---------------------------------------------------------------------------

# llama-server's flags that say how it answered: the context, the slots, the
# cache, the attention, the experts, the template, the reasoning, the rope —
# never one that names a key, a host, a port or a path
FLAGS = {"-c", "--ctx-size", "-np", "--parallel", "-ngl", "--n-gpu-layers", "--gpu-layers",
         "-fa", "--flash-attn", "-ctk", "--cache-type-k", "-ctv", "--cache-type-v",
         "--cpu-moe", "--n-cpu-moe", "-ncmoe", "-b", "--batch-size", "-ub", "--ubatch-size",
         "--jinja", "--no-jinja", "--reasoning-budget", "--reasoning-format",
         "--rope-scaling", "--rope-scale", "--rope-freq-base", "--yarn-orig-ctx",
         "--temp", "--top-p", "--top-k", "--min-p", "--seed", "-t", "--threads",
         "--no-mmap", "--mlock", "--swa-full", "--no-kv-offload", "--cont-batching",
         "--kv-unified", "-kvu", "--split-mode", "-sm", "-ts", "--tensor-split", "-mg",
         "--main-gpu", "--draft-max", "--draft-min", "--draft-p-min", "--spec-replace",
         "-ot", "--override-tensor", "--no-warmup", "--defrag-thold", "-dt"}
_VALUE = re.compile(r"[A-Za-z0-9_.,:=+*|^$\\-]{1,64}")
ENV_NAMES = re.compile(r"(LLAMA_MOE_[A-Z0-9_]+|LLAMA_ARG_[A-Z0-9_]+|GGML_[A-Z0-9_]+|"
                       r"CUDA_VISIBLE_DEVICES|OMP_NUM_THREADS)")
_SECRETISH = re.compile(r"KEY|TOKEN|SECRET|PASS|AUTH|CRED", re.I)


def _argv(x) -> list[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    try:
        return shlex.split(str(x or ""))
    except ValueError:
        return str(x or "").split()


_KEYISH = re.compile(r"[A-Za-z0-9_]{24,}|\bsk-|\bhf_")
_NUM = re.compile(r"-\d+(\.\d+)?")


def _plain(v: str) -> bool:
    return bool(_VALUE.fullmatch(v)) and not _SECRETISH.search(v) and not _KEYISH.search(v)


def flags(x) -> tuple[list[str], int]:
    """(the allowed flags, each with its value when it takes one; how many
    were left out) — from a list or a command line. A flag not in FLAGS goes
    with its value, whatever it is (a key as its own element of a list)"""
    out, left, args, k = [], 0, _argv(x), 0
    while k < len(args):
        a = args[k]
        if not a.startswith("-") or _NUM.fullmatch(a):
            left += 1                           # a value with no flag before it
            k += 1
            continue
        name, eq, inline = a.partition("=")
        nxt = args[k + 1] if k + 1 < len(args) else None
        takes = not eq and nxt is not None and (not nxt.startswith("-") or bool(_NUM.fullmatch(nxt)))
        val = inline if eq else (nxt if takes else "")
        if name in FLAGS and (not val or _plain(val)):
            out += [f"{name}={val}"] if eq else [name, *([val] if val else [])]
        else:
            left += 1
        k += 2 if takes else 1
    return out, left


def env(x) -> tuple[dict, int]:
    """(the allowed variables and their values; how many were left out) —
    from a dict or a "NAME=value …" string"""
    if isinstance(x, dict):
        items = [(str(k), str(v)) for k, v in x.items()]
    else:
        items = [tuple(t.split("=", 1)) if "=" in t else (t, "") for t in _argv(x)]
    out, left = {}, 0
    for k, v in items:
        if ENV_NAMES.fullmatch(k) and not _SECRETISH.search(k) and (not v or _plain(v)):
            out[k] = v
        else:
            left += 1
    return out, left


# ---------------------------------------------------------------------------
# log lines
# ---------------------------------------------------------------------------

# the runner's own lines: its headers, the Frontier run's, the import's, the
# DeviceMark run's and the service's, and a box's own log (a time, then its
# line). llama-server's start-up output and anything else are left out
LOG_SHAPES = [re.compile(p) for p in (
    r"===== .{1,300} =====",
    r"\[(frontier|import|devicemark|service)\] .{1,500}",
    r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d .{1,500}")]
# a kept shape's line still left out when it holds any of these: an address
# or URL, a user@host, a key's name, an ssh line
_RISKY = re.compile(r"https?://|\w@[\w.-]|\b(?:\d{1,3}\.){3}\d{1,3}\b|\[[0-9a-f:]+\]|"
                    r"\b(?:[0-9a-f]{1,4}:){3,7}[0-9a-f]{1,4}\b|[0-9a-f]{0,4}::[0-9a-f]{0,4}|"
                    r"api[-_]?key|\btoken\s*[:=]|access[-_ ]?token|hf_token|x-token|"
                    r"submit_token|bearer|secret|passw|authoriz|\bssh\b|\bscp\b|-o\s*port|"
                    r"hf://|-hf\s|\bdatasets/[\w.-]+/", re.I)
_SECTION = re.compile(r"===== llama-server, as it started =====")


def _norm(t: str) -> str:
    t = t.replace("\\n", " ").replace('\\"', '"').replace("\\t", " ").replace("\\u", " ")
    return " ".join(re.findall(r"\w+", t.lower()))


class Questions:
    """the gated benchmarks' questions, as runs of six words — a line quoting
    any six words of one in a row (its first line, its middle, a later line,
    JSON-escaped) is one that quotes it"""
    N = 6

    def __init__(self, texts: list[str]):
        self.grams: set = set()
        self.short: set = set()
        for t in texts:
            w = _norm(t).split()
            if len(w) < self.N:
                if len(" ".join(w)) >= 16:
                    self.short.add(" ".join(w))
                continue
            self.grams.update(" ".join(w[i:i + self.N]) for i in range(len(w) - self.N + 1))

    def quotes(self, line: str) -> bool:
        n = _norm(line)
        w = n.split()
        if any(" ".join(w[i:i + self.N]) in self.grams for i in range(len(w) - self.N + 1)):
            return True
        return any(s in n for s in self.short)


def log_lines(text: str, questions: Questions | None) -> tuple[list[str], dict]:
    """(the lines that may go out; counts of those left out and why). With
    `questions` None — the list couldn't be loaded — no line at all"""
    lines = text.splitlines()
    if questions is None:
        return [], {"no question list": len(lines)}
    out, left = [], {"not the runner's": 0, "an address, a key or a host": 0,
                     "quotes a question": 0}
    in_server = False
    for line in lines:
        s = line.rstrip()
        if s.startswith("====="):
            in_server = bool(_SECTION.search(s))
        if in_server or not s:
            if s and in_server:
                left["not the runner's"] += 1
            continue
        if not any(p.fullmatch(s) for p in LOG_SHAPES):
            left["not the runner's"] += 1
        elif _RISKY.search(s):
            left["an address, a key or a host"] += 1
        elif questions.quotes(s):
            left["quotes a question"] += 1
        else:
            out.append(s)
    return out, {k: v for k, v in left.items() if v}


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------

def public(model: str) -> bool:
    """a model the board was told is public — on its page, or by the import's
    --public-weights — and nothing else: not a name that looks public
    (teamacct/bonsai-2-27b), never a flag of the export's"""
    from service import db
    return model.removesuffix(" · thinking") in db.public_all()


def confirm(models: list[str], ask=None, say=print) -> bool:
    """the models a run would publish, listed; a typed yes, or nothing goes
    to public/"""
    if not models:
        return True
    ask = ask or input
    say("These would go to public/, for anyone to download:")
    for m in sorted(set(models)):
        say(f"  {m}")
    try:
        got = ask("Type yes to publish them (anything else keeps them in private/): ")
    except EOFError:
        got = ""
    return str(got).strip().lower() == "yes"


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False)


if __name__ == "__main__":                          # pragma: no cover
    sys.exit("a module: scripts/export_frontier_raw.py and export_devicemark_raw.py use it")
