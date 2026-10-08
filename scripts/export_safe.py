"""17h: what the raw-run exports may write — by a list of what is allowed,
never by scrubbing what isn't. Three rounds of scrubbing by pattern each left
holes (a key as its own element of a list, an env string, a quoted value, a
question's middle); here everything not on a list is left out, and said.

- Fields: `pick(obj, schema)` keeps a field only when the schema names it,
  with a value of its kind; a string by its own rule (`TEXT` short plain
  words, `NAME` a file or model name, `ANY` text our code wrote).
- Launch flags and the environment: `flags()` keeps the llama-server flags in
  FLAGS, each value a plain token (17i: TAKES take one, SWITCHES never);
  `env()` the variables ENV_NAMES allows, never one whose name holds KEY,
  TOKEN, SECRET or PASS — 17i: a value only for a name in ENV_VALUES.
- Log lines: `log_lines()` keeps only lines in one of LOG_SHAPES — 17i: each
  a fixed form with typed slots, none holding a "/" but the run's own model,
  a step's folder on the box and a path here, so none carries a repository —
  and leaves out any of those that holds an address, a URL, a user@host, a
  key's name, a private repository's name or a quote of a question from any
  gated or private set (`private_questions`, `Questions`); with a set that
  can't be loaded, no log.
- Public: `public()` — a model the board was told is public (its page, or the
  import's --public-weights), and nothing else, its name never; `confirm()`
  lists what would go to public/, each model's file, sha256 and source, and
  waits for a typed yes; `refused_public()` while SCRUB_HOSTS or
  SCRUB_ACCOUNTS is empty; `twin()` removes a private run's public folder.

What still goes through export_devicemark_raw.scrub, and why: the lines and
fields this lets through are our own, but they hold paths and host names our
code puts there (a home path in a "bundle /home/…" line, this server's name
in a header) — the scrub takes those, as before, and is the last pass, never
the first."""

from __future__ import annotations

import base64
import binascii
import html
import json
import os
import re
import shlex
import socket
import sys
import unicodedata
from pathlib import Path
import urllib.parse

# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------

TEXT, NAME, ANY, SCALAR, WHERE = "text", "name", "any", "scalar", "where"
_KEY = re.compile(r"[\w.,+@ -]{1,80}")
_TEXT = re.compile(r"[\w .,:;()/+%·–—'’-]{0,200}", re.U)
_NAME = re.compile(r"[\w.+@-]{1,160}(/[\w.+@-]{1,160}){0,2}")


# 17j: where a run ran, picked from a list — "this server", or "rented GPU"
# with the cards it knows by name and the boxes' plan labels (A3, A3-2). It
# was scrubbed but free: a box's label took any text
_GPU = re.compile(r"(?:(?:NVIDIA )?(?:GeForce )?RTX (?:PRO )?\d{4}(?: Ti| SUPER| D)?"
                  r"(?: Ada Generation| Blackwell(?: Server Edition| Workstation Edition)?)?|"
                  r"(?:NVIDIA )?RTX A\d{4}|(?:NVIDIA )?[ABHL]\d{2,3}S?(?: (?:PCIe|SXM\d?|NVL|"
                  r"\d{2,3}GB(?: HBM\de?)?))*|(?:NVIDIA )?GH\d{3}|(?:Tesla )?(?:V100|T4|P100))")
_LABEL = re.compile(r"[A-Z]{1,2}\d{1,3}(?:-\d{1,2})?")


def where_of(text) -> str:
    """17j: a run's where, as one of the forms on the list; anything else is
    'rented GPU' (a run here is always 'this server')"""
    t = str(text or "").strip()
    if t == "this server" or not t:
        return "this server"
    parts = [x.strip() for x in t.split(" · ")]
    if parts[0] != "rented GPU":
        return "rented GPU"
    out = ["rented GPU"]
    for x in parts[1:]:
        m = re.fullmatch(r"box(?:es)? (.+)", x)
        if m:
            labels = [y.strip() for y in m.group(1).split(",")]
            if labels and all(_LABEL.fullmatch(y) for y in labels):
                out.append(("boxes " if len(labels) > 1 else "box ") + ", ".join(labels))
            continue
        cards = [y.strip() for y in x.split(",")]
        if cards and all(_GPU.fullmatch(re.sub(r"^\d{1,2} × ", "", y)) for y in cards):
            out.append(", ".join(cards))
    return " · ".join(out)


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
    if schema == WHERE:
        return where_of(obj) if isinstance(obj, str) else None
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
# 17i: each with whether it takes a value — `--jinja hunter2` carried the next
# token out as --jinja's
TAKES = {"-c", "--ctx-size", "-np", "--parallel", "-ngl", "--n-gpu-layers", "--gpu-layers",
         "-fa", "--flash-attn", "-ctk", "--cache-type-k", "-ctv", "--cache-type-v",
         "--n-cpu-moe", "-ncmoe", "-b", "--batch-size", "-ub", "--ubatch-size",
         "--reasoning-budget", "--reasoning-format", "--rope-scaling", "--rope-scale",
         "--rope-freq-base", "--yarn-orig-ctx", "--temp", "--top-p", "--top-k", "--min-p",
         "--seed", "-t", "--threads", "--split-mode", "-sm", "-ts", "--tensor-split", "-mg",
         "--main-gpu", "--draft-max", "--draft-min", "--draft-p-min", "--spec-replace",
         "-ot", "--override-tensor", "--defrag-thold", "-dt"}
SWITCHES = {"--cpu-moe", "--jinja", "--no-jinja", "--no-mmap", "--mlock", "--swa-full",
            "--no-kv-offload", "--cont-batching", "--kv-unified", "-kvu", "--no-warmup"}
FLAGS = TAKES | SWITCHES
_VALUE = re.compile(r"[A-Za-z0-9_.,:=+*|^$\\-]{1,64}")
ENV_NAMES = re.compile(r"(LLAMA_MOE_[A-Z0-9_]+|LLAMA_ARG_[A-Z0-9_]+|GGML_[A-Z0-9_]+|"
                       r"CUDA_VISIBLE_DEVICES|OMP_NUM_THREADS)")
# 17i: the variables whose values may go out, one by one — how the model was
# asked (its routing, its context, its slots, its cache, its kernels); every
# other allowed name goes out as a name alone, its value withheld (LLAMA_ARG_
# HOST, _PORT, _ALIAS and _HF_FILE came out with theirs)
ENV_VALUES = {"LLAMA_MOE_ROUTE_MODE", "LLAMA_MOE_ROUTE_LOOKAHEAD", "LLAMA_ARG_CTX_SIZE",
              "LLAMA_ARG_N_PARALLEL", "LLAMA_ARG_N_GPU_LAYERS", "LLAMA_ARG_FLASH_ATTN",
              "LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V", "LLAMA_ARG_BATCH",
              "LLAMA_ARG_UBATCH", "LLAMA_ARG_THREADS", "LLAMA_ARG_JINJA", "LLAMA_ARG_N_CPU_MOE",
              "LLAMA_ARG_CPU_MOE", "LLAMA_ARG_REASONING_BUDGET", "LLAMA_ARG_REASONING_FORMAT",
              "GGML_CUDA_FORCE_CUBLAS", "GGML_CUDA_FORCE_MMQ", "GGML_CUDA_NO_PINNED",
              "GGML_CUDA_ENABLE_UNIFIED_MEMORY", "GGML_SCHED_MAX_COPIES",
              "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"}
_SECRETISH = re.compile(r"KEY|TOKEN|SECRET|PASS|AUTH|CRED", re.I)

# 17j: each value by its kind, not by its characters — a 23-character token
# passed as CUDA_VISIBLE_DEVICES. A flag or variable with no kind here goes out
# without its value
_INT = r"-?\d{1,9}"
_FLOAT = r"-?\d{1,9}(?:\.\d{1,9})?(?:e-?\d{1,3})?"
_SWITCH = r"(?i:on|off|auto|true|false|enabled|disabled|[01])"
_CACHE = r"f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1"
_TENSORS = r"(?:blk|ffn|exps|attn|token_embd|output|shexp)"
_OT_ONE = (rf"[\w.\\|()^$*+?\[\]-]{{0,60}}{_TENSORS}[\w.\\|()^$*+?\[\]-]{{0,60}}"
           r"=(?:CPU|CUDA\d{1,2}|CUDA_Host)")
KINDS = {
    **{f: _INT for f in ("-c", "--ctx-size", "-np", "--parallel", "--n-cpu-moe", "-ncmoe",
                         "-b", "--batch-size", "-ub", "--ubatch-size", "--reasoning-budget",
                         "--yarn-orig-ctx", "--top-k", "--seed", "-t", "--threads", "-mg",
                         "--main-gpu", "--draft-max", "--draft-min")},
    **{f: _INT + "|all|auto" for f in ("-ngl", "--n-gpu-layers", "--gpu-layers")},
    **{f: _FLOAT for f in ("--rope-scale", "--rope-freq-base", "--temp", "--top-p", "--min-p",
                           "--draft-p-min", "--defrag-thold", "-dt")},
    **{f: "on|off|auto" for f in ("-fa", "--flash-attn")},
    **{f: _CACHE for f in ("-ctk", "--cache-type-k", "-ctv", "--cache-type-v")},
    "--reasoning-format": "none|deepseek|deepseek-legacy|auto",
    "--rope-scaling": "none|linear|yarn",
    **{f: "none|layer|row" for f in ("-sm", "--split-mode")},
    **{f: r"\d{1,3}(?:\.\d{1,3})?(?:,\d{1,3}(?:\.\d{1,3})?){0,15}" for f in ("-ts",
                                                                          "--tensor-split")},
    **{f: rf"{_OT_ONE}(?:,{_OT_ONE}){{0,7}}" for f in ("-ot", "--override-tensor")},
    "LLAMA_MOE_ROUTE_MODE": "lookahead|default|off|none|topk|greedy|static",
    **{v: _INT for v in ("LLAMA_MOE_ROUTE_LOOKAHEAD", "LLAMA_ARG_CTX_SIZE", "LLAMA_ARG_N_PARALLEL",
                         "LLAMA_ARG_BATCH", "LLAMA_ARG_UBATCH", "LLAMA_ARG_THREADS",
                         "LLAMA_ARG_N_CPU_MOE", "LLAMA_ARG_REASONING_BUDGET",
                         "GGML_SCHED_MAX_COPIES", "OMP_NUM_THREADS")},
    "LLAMA_ARG_N_GPU_LAYERS": _INT + "|all|auto",
    "LLAMA_ARG_FLASH_ATTN": "on|off|auto",
    **{v: _CACHE for v in ("LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V")},
    "LLAMA_ARG_REASONING_FORMAT": "none|deepseek|deepseek-legacy|auto",
    **{v: _SWITCH for v in ("LLAMA_ARG_JINJA", "LLAMA_ARG_CPU_MOE", "GGML_CUDA_FORCE_CUBLAS",
                            "GGML_CUDA_FORCE_MMQ", "GGML_CUDA_NO_PINNED",
                            "GGML_CUDA_ENABLE_UNIFIED_MEMORY")},
    "CUDA_VISIBLE_DEVICES": r"\d{1,2}(?:,\d{1,2}){0,15}",
}


def of_kind(name: str, value: str) -> bool:
    """17j: a flag's or a variable's value is of its kind"""
    k = KINDS.get(name)
    return bool(k) and bool(re.fullmatch(rf"(?:{k})", value))


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
    with its value, whatever it is (a key as its own element of a list). 17i:
    a switch never takes the next token: it is a value with no flag before
    it, left out"""
    out, left, args, k = [], 0, _argv(x), 0
    while k < len(args):
        a = args[k]
        if not a.startswith("-") or _NUM.fullmatch(a):
            left += 1                           # a value with no flag before it
            k += 1
            continue
        name, eq, inline = a.partition("=")
        nxt = args[k + 1] if k + 1 < len(args) else None
        looks = nxt is not None and (not nxt.startswith("-") or bool(_NUM.fullmatch(nxt)))
        takes = not eq and looks and name not in SWITCHES
        val = inline if eq else (nxt if takes else "")
        if name in FLAGS and (not val or (_plain(val) and of_kind(name, val))) \
                and not (eq and name in SWITCHES):
            out += [f"{name}={val}"] if eq else [name, *([val] if val else [])]
        else:
            left += 1
        k += 2 if takes else 1
    return out, left


WITHHELD = "(set; its value withheld)"


def env(x) -> tuple[dict, int]:
    """(the allowed variables and their values; how many were left out) —
    from a dict or a "NAME=value …" string. 17i: a value only for a name in
    ENV_VALUES; any other allowed name by its name alone"""
    if isinstance(x, dict):
        items = [(str(k), str(v)) for k, v in x.items()]
    else:
        items = [tuple(t.split("=", 1)) if "=" in t else (t, "") for t in _argv(x)]
    out, left = {}, 0
    for k, v in items:
        if not ENV_NAMES.fullmatch(k) or _SECRETISH.search(k):
            left += 1
        elif k in ENV_VALUES and (not v or (_plain(v) and of_kind(k, v))):
            out[k] = v
        else:
            out[k] = WITHHELD
    return out, left


# ---------------------------------------------------------------------------
# log lines
# ---------------------------------------------------------------------------

# 17i: a line goes out only in one of these shapes, each a fixed form with
# typed slots — never "any text after a time" (a box wrote "fetching … from
# <account>/evalboard-private", and it went out). No slot holds a "/" but the
# run's own model (MODEL) and a step's folder on the box (BOX_PATH), so no
# shape carries a repository; nor an "@", a backslash or a quote
_T = r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d"
_W = r"[\w .,;:'’()×%·…–—+&=#?!<>\[\]*-]"           # words: never / @ \ "
_N = r"\d[\d,]*(?:\.\d+)?"
_HEX = r"[0-9a-f]{8,64}"
_FILE = r"[\w.+-]{1,200}"
MODEL, BOX_PATH, HERE_PATH = "⟨model⟩", "⟨box path⟩", "⟨path⟩"
_M, _P, _H = re.escape(MODEL), re.escape(BOX_PATH), re.escape(HERE_PATH)
LOG_SHAPES = {
    "a section header": rf"===== {_W}{{1,300}} =====",
    "a step's first line":
        rf"{_T} {_M} · thinking (?:on|off|auto) · [\w, ]{{1,300}}(?: · shard \d+ of \d+)? · "
        rf"in {_P}",
    "the file, the build and the GPU":
        rf"{_T} {_FILE} · {_N} GB · sha256 {_HEX} · llama\.cpp [\w.?-]{{1,40}} \([\w?]{{1,40}}\)"
        rf" · GPU {_W}{{1,80}} · {_N} slots of {_N} tokens",
    "a benchmark's questions": rf"{_T} {_W}{{1,80}}: {_N} of its {_N} questions here × {_N} runs?",
    "the server coming up": rf"{_T} (?:llama-server is up|waiting for llama-server to load the "
                            rf"model…) \({_N} s\)",
    "a sha256 worked out": rf"{_T} sha256 of {_FILE} \({_N} GB\)…",
    "the memory check": rf"{_T} (?:memory, from |memory not worked out: |{_N} slots of {_N} "
                        rf"tokens would leave ){_W}{{1,600}}",
    "the server's variables": rf"{_T} not given to llama-server: [A-Z0-9_, ]{{1,300}}, set in "
                              rf"this shell — {_W}{{1,300}}",
    "progress": rf"{_T} · {_W}{{1,300}}",
    "a DeviceMark task's progress": rf"{_T} [\w-]{{1,40}}  \d+/(?:\d+|\?) · {_W}{{1,300}}",
    "the parity questions": rf"{_T} parity {_N} of {_N}",
    "the parity file": rf"{_T} parity: {_N} answers · {_P} — frontier_fetch\.py fetches it, and "
                       r"compares it with the server's when given that \(--parity\)",
    "the run's end": rf"{_T} the run: (?:done|failed|canceled|stopped|running|queued)"
                     rf"(?: · {_W}{{0,600}})?",
    "a benchmark not whole": rf"{_T} [\w-]{{1,40}}: {_N} of {_N} answered — run the same "
                             r"command again to carry on",
    "no bundle": rf"{_T} no task is answered whole yet: no bundle",
    "the bundle": rf"{_T} bundle {_P}/{_FILE} · {_N} MB · sha256 {_HEX} · {_W}{{1,600}}",
    "DeviceMark's battery": rf"{_T} battery {_W}{{1,40}}: {_N} items · items sha256 {_HEX}",
    "DeviceMark's shard": rf"{_T} {_W}{{1,80}}: items {_N}, {_N}, {_N}, … of each task · "
                          rf"{_W}{{1,300}}",
    "DeviceMark's GPU": rf"{_T} GPU {_W}{{1,80}} \(driver {_W}{{1,40}}\) · torch {_W}{{1,300}}",
    "a DeviceMark run's first line":
        rf"{_T} {_M} · thinking (?:on|off|auto) · {_W}{{1,300}}",
    "the service's, the import's and DeviceMark's own lines":
        rf"\[(?:frontier|import|devicemark|service)\] (?:{_W}|{_M}|{_P}|{_H}){{1,800}}",
}
_SHAPES = [re.compile(x) for x in LOG_SHAPES.values()]
# a kept shape's line still left out when it holds any of these: an address
# or URL, a user@host, a key's name, an ssh line, a private repository's name.
# 17i: an IPv6 address in brackets has a colon — "===== [7] … =====" headers
# were dropped as addresses
_RISKY = re.compile(r"https?://|\w@[\w.-]|\b(?:\d{1,3}\.){3}\d{1,3}\b|\[[0-9a-f]*:[0-9a-f:]*\]|"
                    r"\b(?:[0-9a-f]{1,4}:){3,7}[0-9a-f]{1,4}\b|[0-9a-f]{0,4}::[0-9a-f]{0,4}|"
                    r"api[-_]?key|\btoken\s*[:=]|access[-_ ]?token|hf_token|x-token|"
                    r"submit_token|bearer|secret|passw|authoriz|\bssh\b|\bscp\b|-o\s*port|"
                    r"hf://|-hf\s|\bdatasets/[\w.-]+/|[\w.-]*-private\b|"
                    # 18b: a 40-character hex key with no prefix (a sha256's 64
                    # stand in their own slot); a home path URL-encoded
                    r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])|%2F(?:home|Users|root)(?:%2F|\b)",
                    re.I)
_SECTION = re.compile(r"===== llama-server, as it started =====")


def _norm(t: str) -> str:
    r"""a text as words, to compare: JSON's escapes read (\uXXXX — any accented
    or non-Latin question — \n, \t, \"), URL-encoding read, words joined by
    underscores split, case and width folded. 17i: each of those let a quote
    through the six-word rule"""
    t = html.unescape(t)                     # 17j: &nbsp; between its words, &#39;, &amp;
    t = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), t)
    t = t.replace("\\n", " ").replace('\\"', '"').replace("\\t", " ")
    if re.search(r"%[0-9a-fA-F]{2}", t):
        t = urllib.parse.unquote_plus(t)
    t = unicodedata.normalize("NFKD", t).casefold()
    t = "".join(c for c in t if not unicodedata.combining(c))    # é and e alike
    return " ".join(re.findall(r"[^\W_]+", t))


_B64 = re.compile(r"[A-Za-z0-9+/_-]{16,}={0,2}")


def _decoded(line: str) -> list[str]:
    """17j: the text a line's base64 runs hold, where it is text"""
    out = []
    for m in _B64.finditer(line):
        raw = m.group(0)
        for dec in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                t = dec(raw + "=" * (-len(raw) % 4)).decode("utf-8")
            except (binascii.Error, ValueError, UnicodeDecodeError):
                continue
            if t and sum(c.isprintable() or c.isspace() for c in t) >= 0.9 * len(t):
                out.append(t)
                break
    return out


class Questions:
    """the gated and private sets' questions, as runs of four words — a line
    quoting any four words of one in a row (its first line, its middle, a later
    line, JSON-escaped, URL-encoded, joined by underscores, 17j: with &nbsp;
    between them, or in base64; 18b: four, not five, and with a filler word
    between each) is one that quotes it. A line it catches wrongly is only
    left out"""
    N = 4

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
        for text in (line, *_decoded(line)):
            n = _norm(text)
            w = n.split()
            if any(" ".join(w[i:i + self.N]) in self.grams for i in range(len(w) - self.N + 1)):
                return True
            # 18b: a filler word between each of its words
            for k in (0, 1):
                v = w[k::2]
                if any(" ".join(v[i:i + self.N]) in self.grams
                       for i in range(len(v) - self.N + 1)):
                    return True
            if any(s in n for s in self.short):
                return True
        return False


def _slots(line: str, models: list[str]) -> str:
    """the line with the run's own model and a step's folder on the box put as
    their slots — the only places a "/" may stand"""
    for m in sorted({x for x in models if x}, key=len, reverse=True):
        line = re.sub(rf"(?<![\w./-]){re.escape(m)}(?![\w./-])", MODEL, line)
    line = re.sub(r"(?<![\w./~-])/workspace(?:/[\w.+-]{1,160}){0,5}(?=/[\w.+-]+\.tar\.gz)|"
                  r"(?<![\w./~-])/workspace(?:/[\w.+-]{1,160}){0,5}(?![\w./-])", BOX_PATH, line)
    # a path on this server, in a tagged line of its own (the scrub makes its
    # home ~): never a repository — a "-private" name is left out before this
    return re.sub(r"(?<![\w./~-])(?:~|/home/[\w.-]+|/Users/[\w.-]+|/root|/app|/data|/tmp)"
                  r"(?:/[\w.+-]{1,160}){0,8}(?![\w./-])", HERE_PATH, line)


def accounts() -> list[str]:
    """the Hugging Face accounts a private repository is kept under (17j:
    SCRUB_ACCOUNTS and those the board worked out) — a line naming one is left
    out, wherever it stands"""
    return scrub_names()["accounts"]


# 17j: free words in a kept line's slots carried things out — the importer's
# name in every imported run's header, an exception's text (a host that isn't
# in SCRUB_HOSTS: the judge's) and the server's own words after "counted
# wrong". These lines go out without them; a tagged line that still holds an
# exception's text is left out
_BY_NAME = re.compile(r"^(===== \[\d+\] imported .+? \(sha256 [0-9a-f]+\)) by [^:]{1,80}(: )")
_REASON = re.compile(
    r"^(\[frontier\] [\w-]+: question \S+, run \d+: the server failed on it twice"
    r"(?:, on \d+ separate runs)? — written as no answer, counted wrong) \(.*\)$")
_ASKED = re.compile(r"^(\[frontier\] .*?Nothing was asked) \(.*\)$")
_KEPT = re.compile(r"^(\[frontier\] .*? · the answers it gave are kept: the next run asks only "
                   r"the rest) \(.*\)$")
_STOPPED = re.compile(r"the server stopped answering \(.*\)(?=: \d[\d,]* question)")
# 18b: the stop reasons that held the server's words (a log written before
# 18b): "refused a question: HTTP 400 <its words>", "failed on 6 questions,
# asked … (HTTP 400 <its words>): stopped"
_REFUSED_Q = re.compile(r"(the server refused a question)(?:: (?:(?! · ).)*)")
_FAILED_ON = re.compile(r"(the server failed on (?:\d[\d,]* questions|every question it was "
                        r"asked \(\d[\d,]*\)), asked its own way and without its chat parsing) "
                        r"\([\s\S]*\)(?=: stopped)")
_RUN_END = re.compile(rf"^({_T} the run: (?:failed|canceled|stopped))(?: · .*)?$")
_EXCEPTION = re.compile(r"\b\w*(?:Error|Exception|Exit|Interrupt|Refused|Timeout)\b|"
                        r"Traceback|\bErrno\b|\bstatus code\b|Max retries|Errno|"
                        r"HTTPConnectionPool|NewConnectionError|getaddrinfo|\bresolve\b")
_TAGGED = re.compile(r"^\[(?:frontier|import|devicemark|service)\] ")


def _free_words_out(line: str) -> tuple[str, bool]:
    """(the line without its free words, whether it may go out at all)"""
    s = _BY_NAME.sub(r"\1\2", line)
    s = _REASON.sub(r"\1", s)
    s = _ASKED.sub(r"\1", s)
    s = _KEPT.sub(r"\1", s)
    s = _STOPPED.sub("the server stopped answering", s)
    s = _REFUSED_Q.sub(r"\1", s)
    s = _FAILED_ON.sub(r"\1", s)
    s = _RUN_END.sub(r"\1", s)
    # 18b: on every line with a free slot — a section's header and the run's
    # end too, not only a tagged line
    if _EXCEPTION.search(s):
        return s, False
    return s, True


# 18b: a line of the runner's own starts with one of these. A line below one
# whose bracket is still open is the rest of it (a reason with a newline in
# it — its first line went out): the two are read as one, out whole or left
# out whole. A library's lines below a closed one stay lines of their own
_STARTS = re.compile(rf"^(?:\[[a-z]+\] |===== |{_T} )")
JOIN_MAX = 20                            # lines a reason may run to


def _joined(lines: list[str]) -> list[str]:
    out: list[str] = []
    in_server, joined = False, 0
    for line in lines:
        s = line.rstrip()
        if s.startswith("====="):
            in_server = bool(_SECTION.search(s))
        if (out and s and not in_server and not _STARTS.match(s) and joined < JOIN_MAX
                and out[-1].count("(") > out[-1].count(")")):
            out[-1] += "\n" + s
            joined += 1
        else:
            out.append(s)
            joined = 0
    return out


def log_lines(text: str, questions: Questions | None,
              models: list[str] | None = None) -> tuple[list[str], dict]:
    """(the lines that may go out; counts of those left out and why). With
    `questions` None — a set couldn't be loaded — no line at all. `models`:
    the run's own model ids, the one slot a model's name may fill"""
    lines = _joined(text.splitlines())
    if questions is None:
        return [], {"no question list": len(lines)}
    acct = [re.compile(rf"(?<![\w-]){re.escape(a)}(?![\w-])", re.I) for a in accounts()]
    out, left = [], {"not in a shape on the list": 0, "an address, a key, a host or a "
                     "repository": 0, "quotes a question": 0,
                     "an exception's or a server's own words": 0}
    in_server = False
    for line in lines:
        s = line.rstrip()
        if s.startswith("====="):
            in_server = bool(_SECTION.search(s))
        if in_server or not s:
            if s and in_server:
                left["not in a shape on the list"] += 1
            continue
        s, free = _free_words_out(s)
        if _RISKY.search(s) or any(a.search(s) for a in acct):
            left["an address, a key, a host or a repository"] += 1
        elif not free:
            left["an exception's or a server's own words"] += 1
        elif not any(p.fullmatch(_slots(s, models or [])) for p in _SHAPES):
            left["not in a shape on the list"] += 1
        elif questions.quotes(s):
            left["quotes a question"] += 1
        else:
            out.append(s)
    return out, {k: v for k, v in left.items() if v}


# 17i: the benchmarks whose questions may be shown — every other Frontier set
# is gated or withheld
FRONTIER_SHOWN = ("mmlupro_tiger", "simpleqa_epoch", "arc_agi2_public")
_TEXT_FIELDS = ("question", "problem", "prompt", "text", "passage", "input", "context",
                "instruction", "stem", "A", "B", "C", "D")


def private_questions(say=None) -> Questions | None:
    """17i: every gated and private set on this server, always — whatever the
    run asked (a box's log holds every step on that box: a SimpleQA run's log
    quoted an HLE question): Frontier's gated and withheld benchmarks, the
    Everyday tasks' hidden half, the Knowledge exam's bank, Mobile-MMLU and
    Mobile-MMLU-Pro. None — no log at all — when one couldn't be loaded,
    said with `say` (it was said only in the README). 17j: and how many
    questions each set gave, said — one absent here gave none, and no word"""
    from service import config
    texts: list[str] = []
    missing: list[str] = []
    gave: list[tuple[str, int]] = []

    def take(name: str, rows) -> None:
        n = 0
        for r in rows or []:
            got = [str(r[k]) for k in _TEXT_FIELDS if isinstance(r, dict)
                   and isinstance(r.get(k), str)]
            texts.extend(got)
            n += bool(got)
        gave.append((name, n))
    import frontier as fb
    for t in fb.TASKS:
        if t in FRONTIER_SHOWN:
            continue
        try:
            take(fb.BENCH[t]["label"], fb.load(t, config.BENCH_ROOT))
        except Exception:                               # noqa: BLE001 — fail closed
            missing.append(fb.BENCH[t]["label"])
    loaders = []
    try:
        import everyday as ev
        loaders.append(("the Everyday tasks' hidden half", lambda: (
            ev._read_bank(ev.hidden_path()) if ev.hidden_path().exists() else [])))
    except ImportError:
        pass
    try:
        import exam_build as eb
        loaders.append(("the Knowledge exam's bank",
                        lambda: [r for rows in eb.load_bank(config.EXAM_DIR).values()
                                 for r in rows]))
    except ImportError:
        pass
    try:
        import mobile_mmlu as mm
        loaders.append(("Mobile-MMLU-Pro", mm.load))
        loaders.append(("Mobile-MMLU", mm.load_full))
    except ImportError:
        pass
    for name, load in loaders:
        try:
            take(name, load())
        except Exception:                               # noqa: BLE001 — fail closed
            missing.append(name)
    if say:
        say("the logs are checked against: " + ", ".join(
            f"{name} {n:,}" if n else f"{name} 0 (none on this server)" for name, n in gave))
    if missing:
        if say:
            say(f"no log: {', '.join(missing)} couldn't be loaded to check the log against "
                "(each log line is checked against every gated and private set) — every run "
                "is written without its log, and the export exits 1")
        return None
    return Questions(texts)


UNLOADED = object()                     # 17j: the sets not loaded yet (None: one couldn't be)


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------

def public(model: str) -> bool:
    """a model the board was told is public — on its page, or by the import's
    --public-weights — and nothing else: not a name that looks public
    (teamacct/bonsai-2-27b), never a flag of the export's. 17i: and never kept
    private by its name either (served/Qwen3.6-35B-A3B-BF16 is unsloth's file,
    unmodified, and public): the mark and the typed yes decide"""
    from service import db
    return model.removesuffix(" · thinking") in db.public_all()


def file_of(model: str) -> dict:
    """17i: what the typed-yes list says of a model — its file, that file's
    sha256 (for a GGUF in parts, its split identity) and where it came from,
    as the board recorded them; {} for a model with no file here"""
    from service import served
    model = model.removesuffix(" · thinking")
    try:
        rec = (served.get(model) if served.is_served(model) else None) or {}
    except Exception:                                   # noqa: BLE001 — said as not recorded
        rec = {}
    fs, gp = rec.get("file_sha256") or {}, rec.get("gguf_pin") or {}
    sha = gp.get("sha256") or fs.get("sha256") or ""
    name = gp.get("name") or fs.get("name") or (rec.get("pin") or {}).get("file") or ""
    src = fs.get("source") or (f"{rec['gguf_path']} on this server" if rec.get("gguf_path")
                               else "")
    from service import public_files as pf
    public = pf.words(model, pf.get(model))
    return {k: v for k, v in (("file", name), ("sha256", sha), ("source", src),
                              ("parts", len(fs.get("parts") or [])), ("public", public)) if v}


# ---------------------------------------------------------------------------
# 17j: the names the scrub removes — given, and worked out
# ---------------------------------------------------------------------------

NAME_MIN = 3
_HOST_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)*")
_ACCOUNT_NAME = re.compile(r"[A-Za-z0-9][\w.-]*")
# a host the scrub takes when the board's own settings name it: one that
# isn't a public name (a container's, the tailnet's, an internal one)
_OWN_HOST = re.compile(r"(?i)[a-z0-9-]+|.+\.(?:internal|local|localdomain|lan|home|corp|"
                       r"intranet|private|docker|ts\.net)")
_NOT_HOSTS = {"localhost", "host.docker.internal"}
_HF_REPO = re.compile(r"(?i)(?:\bhf://|\bhuggingface\.co/(?:models/|datasets/)?)"
                      r"([A-Za-z0-9][\w.-]*)/([\w.-]+)")
_names: dict = {}


def _given(var: str, ok: re.Pattern, what: str) -> tuple[list[str], list[str]]:
    """(the names an environment variable gives, what is wrong with it)"""
    raw = os.environ.get(var, "")
    names, bad = [], []
    for v in (x.strip() for x in raw.split(",")):
        if not v:
            continue
        if len(v) < NAME_MIN:
            bad.append(f'{var}: "{v}" is too short to be {what}')
        elif not ok.fullmatch(v):
            bad.append(f'{var}: "{v}" isn\'t {what}')
        else:
            names.append(v)
    if raw.strip() and not names and not bad:
        bad.append(f'{var}: "{raw}" gives no name')
    return names, bad


def _host(url: str) -> str:
    try:
        h = urllib.parse.urlsplit(url if "//" in url else "//" + url).hostname or ""
    except ValueError:
        return ""
    return h.lower()


def _worked_out() -> tuple[list[str], list[str]]:
    """(the dotted hosts the board's own settings name, the accounts of the
    repositories its models were fetched from but those of a checked public
    file) — names that add to SCRUB_HOSTS and SCRUB_ACCOUNTS, never stand for
    them. 18b: never a name from a request's Host header"""
    hosts: set[str] = set()
    urls = [v for k, v in os.environ.items() if k.endswith("_URL") and isinstance(v, str)]
    accts: set[str] = set()
    try:
        from service import config, db
        from service import public_files as pf
        urls.append(config.LOCAL_BASE_URL)
        recs = db.served_all() if config.DB_PATH.exists() else []
        urls += [str(r.get("base_url") or "") for r in recs]
        checked = {(x.get("repo") or "").lower() for m, x in db.public_files_all().items()
                   if pf.still_same(m, x)}
        for r in recs:
            for acct, repo in _HF_REPO.findall(json.dumps(r)):
                if f"{acct}/{repo}".lower().split("@")[0] not in checked:
                    accts.add(acct)
    except Exception:                                   # noqa: BLE001 — what was given alone
        pass
    for u in urls:
        h = _host(u)
        if h and h not in _NOT_HOSTS and _OWN_HOST.fullmatch(h) and not re.fullmatch(r"[\d.]+", h) \
                and "." in h:
            hosts.add(h)
    return sorted(hosts), sorted(accts)


def scrub_names() -> dict:
    """{hosts, accounts, server, problems} — the hosts and accounts the scrub
    removes: those given (SCRUB_HOSTS, SCRUB_ACCOUNTS), and those the board
    works out (this container's name, the dotted hosts its settings name;
    the accounts of the repositories its models were fetched from), which
    only add to them. `server`: this server's own names, as given — the
    container's own name isn't one. `problems`: why public/ is refused —
    18b: unless both variables are given (with neither, one request with a
    made-up Host header opened public/)"""
    me = socket.gethostname() or ""
    key = (os.environ.get("SCRUB_HOSTS"), os.environ.get("SCRUB_ACCOUNTS"), me,
           os.environ.get("BENCH_ROOT"), _stamp())
    if _names.get("key") == key:
        return _names["value"]
    hosts, bad_h = _given("SCRUB_HOSTS", _HOST_NAME, "a host's name")
    accts, bad_a = _given("SCRUB_ACCOUNTS", _ACCOUNT_NAME, "an account's name")
    own, found = _worked_out()
    server = [h for h in hosts if h.lower() != me.lower() and h.lower().split(".")[0] != me.lower()]
    problems = [*bad_h, *bad_a]
    if not hosts and not bad_h:
        problems.append("SCRUB_HOSTS isn't given: give it this server's own name and the "
                        "tailnet's name for it, SCRUB_HOSTS=\"$(hostname),<its tailnet name>\", "
                        "from the server's shell — the names the board works out only add to it")
    elif hosts and not server:
        problems.append("this server's own name isn't in SCRUB_HOSTS: give SCRUB_HOSTS="
                        "\"$(hostname)\" from the server's shell" + (
                            f" (the container's own name, {me}, isn't it)"
                            if me and me in hosts else ""))
    if not accts and not bad_a:
        problems.append("SCRUB_ACCOUNTS isn't given: give it the Hugging Face accounts the "
                        "private repositories are kept under — the accounts the board works "
                        "out only add to it")
    all_accts = sorted({*accts, *found}, key=str.lower)
    names = [x for h in [*hosts, *own, me] if h for x in (h, h.split(".")[0])]
    out = {"hosts": sorted({h for h in names if len(h) >= NAME_MIN}, key=str.lower),
           "accounts": [a for a in all_accts if len(a) >= 2],
           "server": server, "problems": problems}
    _names.update(key=key, value=out)
    return out


def _stamp() -> tuple:
    """what the worked-out names depend on, cheaply: the database's time"""
    try:
        from service import config
        return (config.DB_PATH.stat().st_mtime_ns if config.DB_PATH.exists() else 0,)
    except Exception:                                   # noqa: BLE001
        return ()


def refused_public() -> str:
    """17i: '' when public/ may be written; else why not. 17j: a value too
    short to be a name, a container's name for the server's, or an account
    no one gave and none found — each refuses, said"""
    got = scrub_names()
    if not got["problems"]:
        return ""
    return ("public/ refused: " + "; ".join(got["problems"]) + " (docs/REMOTE-RUNS.md § "
            "Publishing the raw runs). Everything goes to private/")


def will_remove() -> str:
    got = scrub_names()
    return (f"will remove: hosts {', '.join(got['hosts']) or 'none'}; "
            f"accounts {', '.join(got['accounts']) or 'none'}")


def confirm(models: list[str], ask=None, say=print) -> bool:
    """the models a run would publish, listed — 17i: each with its file, its
    sha256 and where it came from — and a typed yes, or nothing goes to
    public/. Closed stdin is a no, never a traceback"""
    if not models:
        return True
    say(will_remove())                      # 17j: what the scrub takes, above the list
    say("These would go to public/, for anyone to download:")
    for m in sorted(set(models)):
        f = file_of(m)
        say(f"  {m}" + (f" — {f['file']}" if f.get("file") else " — no file recorded")
            + (f" ({f['parts']} parts)" if f.get("parts") else "")
            + (f", sha256 {f['sha256']}" if f.get("sha256") else ", no sha256 recorded")
            + (f", {f['public']}" if f.get("public") else
               f", from {f['source']}" if f.get("source") else ", its source not recorded"))
        if not f.get("public"):
            # 17j: a team build marked public by mistake reads like a public one
            say("    CHECK: not shown to be a public file — give its Hugging Face repository "
                "and path on the model's page, where the board checks its sha256")
    if ask is None and (sys.stdin is None or sys.stdin.closed):
        say("")                             # 17j: closed stdin (<&-): a no, never a traceback
        return False
    ask = ask or input
    try:
        got = ask("Type yes to publish them (anything else keeps them in private/): ")
    except (EOFError, OSError, ValueError, RuntimeError, KeyboardInterrupt):
        say("")                             # closed stdin: "I/O operation on closed file"
        got = ""
    return str(got).strip().lower() == "yes"


def twin(dest: Path, say=print) -> None:
    """17i: a run written to private/ removes its public/ twin — clearing a
    model's mark left its earlier public folder, and the documented upload
    sends the whole folder — and says so"""
    import shutil
    other = dest.parent.parent / "public" / dest.name
    if dest.parent.name == "private" and other.exists():
        shutil.rmtree(other)
        say(f"{other} removed: this run is private now")


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False)


if __name__ == "__main__":                          # pragma: no cover
    sys.exit("a module: scripts/export_frontier_raw.py and export_devicemark_raw.py use it")
