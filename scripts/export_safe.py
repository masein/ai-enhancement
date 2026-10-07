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

import json
import os
import re
import shlex
import sys
import unicodedata
from pathlib import Path
import urllib.parse

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
        if name in FLAGS and (not val or _plain(val)) and not (eq and name in SWITCHES):
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
        elif k in ENV_VALUES and (not v or _plain(v)):
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
                    r"hf://|-hf\s|\bdatasets/[\w.-]+/|[\w.-]*-private\b", re.I)
_SECTION = re.compile(r"===== llama-server, as it started =====")


def _norm(t: str) -> str:
    r"""a text as words, to compare: JSON's escapes read (\uXXXX — any accented
    or non-Latin question — \n, \t, \"), URL-encoding read, words joined by
    underscores split, case and width folded. 17i: each of those let a quote
    through the six-word rule"""
    t = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), t)
    t = t.replace("\\n", " ").replace('\\"', '"').replace("\\t", " ")
    if re.search(r"%[0-9a-fA-F]{2}", t):
        t = urllib.parse.unquote_plus(t)
    t = unicodedata.normalize("NFKD", t).casefold()
    t = "".join(c for c in t if not unicodedata.combining(c))    # é and e alike
    return " ".join(re.findall(r"[^\W_]+", t))


class Questions:
    """the gated and private sets' questions, as runs of six words — a line
    quoting any six words of one in a row (its first line, its middle, a later
    line, JSON-escaped, URL-encoded, joined by underscores) is one that
    quotes it"""
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
    """SCRUB_ACCOUNTS: the Hugging Face accounts a private repository is kept
    under — a line naming one is left out, wherever it stands"""
    return [a.strip() for a in os.environ.get("SCRUB_ACCOUNTS", "").split(",")
            if len(a.strip()) >= 2]


def log_lines(text: str, questions: Questions | None,
              models: list[str] | None = None) -> tuple[list[str], dict]:
    """(the lines that may go out; counts of those left out and why). With
    `questions` None — a set couldn't be loaded — no line at all. `models`:
    the run's own model ids, the one slot a model's name may fill"""
    lines = text.splitlines()
    if questions is None:
        return [], {"no question list": len(lines)}
    acct = [re.compile(rf"(?<![\w-]){re.escape(a)}(?![\w-])", re.I) for a in accounts()]
    out, left = [], {"not in a shape on the list": 0, "an address, a key, a host or a "
                     "repository": 0, "quotes a question": 0}
    in_server = False
    for line in lines:
        s = line.rstrip()
        if s.startswith("====="):
            in_server = bool(_SECTION.search(s))
        if in_server or not s:
            if s and in_server:
                left["not in a shape on the list"] += 1
            continue
        if _RISKY.search(s) or any(a.search(s) for a in acct):
            left["an address, a key, a host or a repository"] += 1
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
    said with `say` (it was said only in the README)"""
    from service import config
    texts: list[str] = []
    missing: list[str] = []

    def take(rows) -> None:
        for r in rows or []:
            if isinstance(r, dict):
                texts.extend(str(r[k]) for k in _TEXT_FIELDS if isinstance(r.get(k), str))
    import frontier as fb
    for t in fb.TASKS:
        if t in FRONTIER_SHOWN:
            continue
        try:
            take(fb.load(t, config.BENCH_ROOT))
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
            take(load())
        except Exception:                               # noqa: BLE001 — fail closed
            missing.append(name)
    if missing:
        if say:
            say(f"no log: {', '.join(missing)} couldn't be loaded to check the log against "
                "(each log line is checked against every gated and private set)")
        return None
    return Questions(texts)


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
    return {k: v for k, v in (("file", name), ("sha256", sha), ("source", src),
                              ("parts", len(fs.get("parts") or []))) if v}


def refused_public() -> str:
    """17i: '' when public/ may be written; else why not — the scrub's names
    for this server's hosts and the private repository's accounts must be
    given, or a host or an account name reaches a public file"""
    gone = [v for v in ("SCRUB_HOSTS", "SCRUB_ACCOUNTS") if not os.environ.get(v, "").strip()]
    if not gone:
        return ""
    return (f"public/ refused: {' and '.join(gone)} {'is' if len(gone) == 1 else 'are'} empty — "
            "give SCRUB_HOSTS the server's host names and SCRUB_ACCOUNTS the Hugging Face "
            "accounts the private repositories are kept under (docs/REMOTE-RUNS.md § "
            "Publishing the raw runs). Everything goes to private/")


def confirm(models: list[str], ask=None, say=print) -> bool:
    """the models a run would publish, listed — 17i: each with its file, its
    sha256 and where it came from — and a typed yes, or nothing goes to
    public/. Closed stdin is a no, never a traceback"""
    if not models:
        return True
    ask = ask or input
    say("These would go to public/, for anyone to download:")
    for m in sorted(set(models)):
        f = file_of(m)
        say(f"  {m}" + (f" — {f['file']}" if f.get("file") else " — no file recorded")
            + (f" ({f['parts']} parts)" if f.get("parts") else "")
            + (f", sha256 {f['sha256']}" if f.get("sha256") else ", no sha256 recorded")
            + (f", from {f['source']}" if f.get("source") else ", its source not recorded"))
    try:
        got = ask("Type yes to publish them (anything else keeps them in private/): ")
    except (EOFError, OSError, ValueError, KeyboardInterrupt):
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
