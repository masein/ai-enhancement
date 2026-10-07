#!/usr/bin/env python3
"""15.4: each DeviceMark row's raw per-item run, ready to publish on Hugging
Face — as DeviceMark links every row of its board to its raw file
(huggingface.co/datasets/devicemark/results, raw/). Run inside the container:

    python scripts/export_devicemark_raw.py --run 212            one run's row
    python scripts/export_devicemark_raw.py --all                every DeviceMark row
    python scripts/export_devicemark_raw.py --link 212 <url> --by masein   its "raw" link
    python scripts/export_devicemark_raw.py --link 212 --unlink --by masein

One folder a row, under --out (default $BENCH_ROOT/raw-export), in public/ or
private/. 17h: public only for a model whose page marks its weights public
(DeviceMark's calibration models: mark them there), never an in-house build
(Qwen3.6), and only after a typed yes to the list the export prints;
--private keeps every row private. Upload is masein's step
(docs/REMOTE-RUNS.md § Publishing the raw runs).

  items.jsonl    a line an item: the test and its key; the prompt as sent; the
                 reply and the thinking; the parsed answer and the dataset's,
                 whether it is right, and IFEval's rules, strict and loose; the
                 tokens, and whether it ran out of room. Their field names where
                 ours mean the same: key, answer, capped, generated_tokens,
                 prompt_chars, content_chars, thinking_chars (and a served row's
                 prompt_tokens, decode_tok_s, wall_s)
  setup.json     the row's setup record
  scores.json    each test's score and interval, the composite and its interval,
                 and their summary's counts (n, capped_count, max_tokens, …)
  log.txt        the run's log
  recompute.py   scores.json again from items.jsonl, standard library only
  README.md      the protocol, the battery's sources and licences, the fields

Scrubbed before anything is written: keys and tokens, home paths, this host's
name and the tailnet's, private and tailnet addresses (scrub). DeviceMark rows
only: no other suite's run is ever read here — Everyday, the Knowledge exam,
the reasoning lab, Privacy Leakage and Mobile-MMLU-Pro never leave the server.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import devicemark as dm  # noqa: E402
import remote_bundle as rb  # noqa: E402
import restrictions  # noqa: E402

RAW_NAME = dm.RAW_NAME
NOT_HERE = ("only DeviceMark rows are exported: Everyday, the Knowledge exam, the reasoning lab, "
            "Privacy Leakage and Mobile-MMLU-Pro never leave the server")
IN_HOUSE = re.compile(r"qwen[\s_-]*3[._-]?6", re.I)
LICENCES = {"ifeval": ("google/IFEval", "Apache-2.0"),
            "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "MIT"),
            "math": ("HuggingFaceH4/MATH-500", "MIT")}


# ---------------------------------------------------------------------------
# the scrub
# ---------------------------------------------------------------------------

_KEYS = [re.compile(p) for p in (
    r"\bhf_[A-Za-z0-9]{20,}", r"\bsk-or-v1-[A-Za-z0-9]{20,}", r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{16,}",
    r"\bgh[pousr]_[A-Za-z0-9]{20,}", r"\bAKIA[0-9A-Z]{16}\b", r"\bxox[abprs]-[A-Za-z0-9\-]{10,}",
    r"\bAIza[0-9A-Za-z_\-]{30,}")]
# a key's value after its name, assigned (key=…, "token": "…") or a bearer's;
# key-shaped: 12 or more characters with a digit among them
_KEY_AFTER = re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password|passwd|"
                        r"authorization)(\"?\s*[:=]\s*\"?(?:bearer\s+)?)"
                        r"(?=[A-Za-z0-9._\-/+=]*\d)[A-Za-z0-9._\-/+=]{12,}")
_BEARER = re.compile(r"(?i)\b(bearer\s+)(?=[A-Za-z0-9._\-/+=]*\d)[A-Za-z0-9._\-/+=]{12,}")
_HOME = re.compile(r"/(?:home|Users)/[^/\s\"'`:;,)]+")
_ROOT_HOME = re.compile(r"(?<![\w.])/root(?=/|\b)")
_TAILNET = re.compile(r"\b[\w-]+(?:\.[\w-]+)*\.ts\.net\b", re.I)
_ADDR = re.compile(r"\b(?:100\.\d{1,3}|10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01])|127\.\d{1,3}"
                   r"|169\.254)\.\d{1,3}\.\d{1,3}\b")
_ADDR6 = re.compile(r"\bfd7a:115c:a1e0:[0-9a-f:]*", re.I)
_LOCAL_URL = re.compile(r"\b(https?://)(?!huggingface\.co\b)([A-Za-z0-9_-]+)(:\d+)?(?=[/\s\"']|$)")
# 17g: what the scrub missed — a box's address and port (ssh -p 41234
# root@203.0.113.77, ssh4.vast.ai), a key given as a flag (--api-key <value>,
# kept whole in a task's launch flags), x-token: and SUBMIT_TOKEN= with no
# digit, a URL's dotted internal host, and the Hugging Face path naming the
# account the builds are kept under
_PUBLIC_ADDR = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_PORT = re.compile(r"(?i)(\b(?:ssh|scp|rsync)\b[^\n]*?\s-[pP]\s*)\d{2,5}\b")
_USER_AT = re.compile(r"\b[a-z_][a-z0-9_-]{0,31}@(?=\[(?:host|address)\]|[\w-]+(?:\.[\w-]+)+)")
_RENTED_HOST = re.compile(r"\b(?:[\w-]+\.)+(?:vast\.ai|runpod\.io|runpod\.net|lambdalabs\.com|"
                          r"tensordock\.com|paperspace\.com)\b|(?<=@)[\w-]+(?:\.[\w-]+)+", re.I)
_FLAG_KEY = re.compile(r"(?i)(--(?:api[-_]?key|hf[-_]?token|token|key|password|secret)[=\s]+)"
                       r"(?!\[)[^\s\"']{4,}")
_NAMED_KEY = re.compile(r"(?i)\b(x-token|submit_token|openrouter_api_key|hf_token|"
                        r"huggingface_hub_token)(\"?\s*[:=]\s*\"?)(?!\[)[^\s\"',}]{4,}")
_INTERNAL_URL = re.compile(r"(?i)\b(https?://)(?:[\w-]+\.)+(?:internal|lan|local|localdomain|"
                           r"home|corp|intranet|private|docker)(?=[:/\s\"']|$)")
_HF_ACCOUNT = re.compile(r"(?i)(\bhf://|\bhuggingface\.co/(?:datasets/|models/)?|(?<![\w/])"
                         r"datasets/|(?<!\w)-hf\s+)(?!\[)([\w.-]+)/"
                         r"(?=[\w.-]*(?:private|evalboard))")
# 17h: the rest of what 17g's scrub let through — any IPv6 address, a port
# after an address or in -o Port=, -p before user@host with no ssh before it,
# a URL's user:password@ and any dotted host but Hugging Face's
_IPV6 = re.compile(r"(?<![\w:])(?=[0-9a-f:]*::|(?:[0-9a-f]{1,4}:){3})"
                   r"[0-9a-f]{0,4}(?::[0-9a-f]{0,4}){2,7}(?![\w:])", re.I)
_ADDR_PORT = re.compile(r"(\[(?:address|host)\]):\d{2,5}\b")
_O_PORT = re.compile(r"(?i)(-o\s*Port\s*=?\s*)\d{2,5}")
_P_USER = re.compile(r"(-p\s+)\d{2,5}(\s+[\w.-]+@)")
_URL_USER = re.compile(r"(?i)\b(https?://)[^\s/@:]+(?::[^\s/@]*)?@")
_ANY_HOST = re.compile(r"(?i)\b(https?://)(?!huggingface\.co\b|\[)([\w-]+(?:\.[\w-]+)+)")
WHAT_THE_SCRUB_REMOVES = (
    "API keys and tokens: the environment's secrets by value, and anything key-shaped "
    "(hf_…, sk-…, sk-or-v1-…, gh?_…, AKIA…, xox?-…, AIza…, a key-shaped value assigned to "
    "api_key, token, secret, password or authorization, or after bearer)",
    "home paths: /home/<user>, /Users/<user> and /root become ~, and BENCH_ROOT becomes "
    "$BENCH_ROOT",
    "host names: this machine's own and any in SCRUB_HOSTS, the tailnet's (*.ts.net), and a "
    "URL's dotless host (http://gemma-vllm:8000)",
    "private addresses: tailnet and CGNAT 100.x, 10.x, 172.16–31.x, 192.168.x, 127.x, 169.254.x "
    "and the tailnet's IPv6 (fd7a:115c:a1e0::)",
    "rented boxes: every IPv4 and IPv6 address, a port after an address or in an ssh line, "
    "the user before an address or host, a URL's user and password, any URL's host but "
    "Hugging Face's, and a rented GPU host's name (*.vast.ai, *.runpod.io, …)",
    "a key given as a flag (--api-key, --hf-token, --token …) or named x-token, SUBMIT_TOKEN, "
    "OPENROUTER_API_KEY or HF_TOKEN, whatever its value",
    "a URL's internal host (*.internal, *.lan, *.local, …), and the Hugging Face account a "
    "private repo is kept under (hf://<account>/…-private, and any in SCRUB_ACCOUNTS)")


def scrub(text: str, env: dict | None = None, hosts: list[str] | None = None) -> str:
    """text fit to publish: no keys, no home paths, no host names, no private
    addresses"""
    env = os.environ if env is None else env
    text = rb.scrub(text, env)
    for p in _KEYS:
        text = p.sub("[key withheld]", text)
    text = _FLAG_KEY.sub(lambda m: f"{m.group(1)}[key withheld]", text)
    text = _NAMED_KEY.sub(lambda m: f"{m.group(1)}{m.group(2)}[key withheld]", text)
    text = _KEY_AFTER.sub(lambda m: f"{m.group(1)}{m.group(2)}[key withheld]", text)
    text = _BEARER.sub(lambda m: f"{m.group(1)}[key withheld]", text)
    root = str(env.get("BENCH_ROOT") or "")
    if len(root) > 3:
        text = text.replace(root, "$BENCH_ROOT")
    text = _HOME.sub("~", text)
    text = _ROOT_HOME.sub("~", text)
    for h in sorted({h for h in (hosts if hosts is not None else _hosts()) if len(h) >= 3},
                    key=len, reverse=True):
        text = re.sub(rf"(?<![\w-]){re.escape(h)}(?![\w-])", "[host]", text, flags=re.I)
    # 17g: the Hugging Face accounts SCRUB_ACCOUNTS names, wherever they stand
    for acct in sorted({a.strip() for a in env.get("SCRUB_ACCOUNTS", "").split(",")
                        if len(a.strip()) >= 2}, key=len, reverse=True):
        text = re.sub(rf"(?<![\w-]){re.escape(acct)}(?=/)", "[account]", text, flags=re.I)
    text = _TAILNET.sub("[host]", text)
    text = _RENTED_HOST.sub("[host]", text)
    text = _INTERNAL_URL.sub(lambda m: f"{m.group(1)}[host]", text)
    text = _HF_ACCOUNT.sub(lambda m: f"{m.group(1)}[account]/", text)
    text = _URL_USER.sub(lambda m: f"{m.group(1)}[user]@", text)
    text = _ANY_HOST.sub(lambda m: f"{m.group(1)}[host]", text)
    text = _ADDR6.sub("[address]", text)
    text = _IPV6.sub("[address]", text)
    text = _ADDR.sub("[address]", text)
    text = _PUBLIC_ADDR.sub("[address]", text)
    text = _PORT.sub(lambda m: f"{m.group(1)}[port]", text)
    text = _O_PORT.sub(lambda m: f"{m.group(1)}[port]", text)
    text = _P_USER.sub(lambda m: f"{m.group(1)}[port]{m.group(2)}", text)
    text = _USER_AT.sub("[user]@", text)
    text = _ADDR_PORT.sub(lambda m: f"{m.group(1)}:[port]", text)
    return _LOCAL_URL.sub(lambda m: f"{m.group(1)}[host]{m.group(3) or ''}", text)


def _hosts() -> list[str]:
    """this machine's name, and any SCRUB_HOSTS names (comma-separated): the
    server's own, when the container's name isn't it"""
    names = [socket.gethostname() or ""] + os.environ.get("SCRUB_HOSTS", "").split(",")
    return [x for n in names if n.strip() for x in (n.strip(), n.strip().split(".")[0])]


def scrub_obj(obj, **kw):
    """a JSON value, every string in it scrubbed"""
    if isinstance(obj, str):
        return scrub(obj, **kw)
    if isinstance(obj, dict):
        return {scrub(k, **kw) if isinstance(k, str) else k: scrub_obj(v, **kw)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_obj(v, **kw) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# a row
# ---------------------------------------------------------------------------

def row_dirs(model: str, thinking: bool) -> list[Path]:
    from service import config
    safe = model.replace("/", "__")
    names = [safe + "__thinking", safe] if thinking else [safe]
    return [config.OUT_DIR / n for n in names if (config.OUT_DIR / n / dm.OUT_NAME).exists()]


def row_of_run(run_id: int) -> tuple[Path, dict]:
    """a DeviceMark run's row — refused, in words, for any other suite"""
    from service import db
    sub = db.get(run_id)
    if not sub:
        raise SystemExit(f"no run #{run_id}")
    if sub.get("suite") != "devicemark":
        raise SystemExit(f"#{run_id} is a {sub.get('suite')} run: {NOT_HERE}")
    dirs = row_dirs(sub["hf_id"], bool(sub.get("thinking")))
    if not dirs:
        raise SystemExit(f"#{run_id}'s row ({sub['hf_id']}) has no scored DeviceMark row yet")
    return dirs[0], sub


def all_rows() -> list[Path]:
    from service import config
    return [d for d in sorted(config.OUT_DIR.iterdir()) if d.is_dir()
            and (d / dm.OUT_NAME).exists()] if config.OUT_DIR.is_dir() else []


def model_of(row: Path) -> str:
    try:
        return json.loads((row / "model_meta.json").read_text(encoding="utf-8"))["model"] \
            .removesuffix(" · thinking")
    except (OSError, ValueError, KeyError):
        return row.name.removesuffix("__thinking").replace("__", "/", 1)


def public_by_default(row: Path, setup: dict) -> bool:
    """17h: public only for a model whose page marks its weights public — a
    name that looks public (teamacct/bonsai-2-27b) never is, and an in-house
    build, a setup served here or a checkpoint never is, marked or not"""
    import export_safe as es
    model = model_of(row)
    if IN_HOUSE.search(model) or IN_HOUSE.search(setup.get("based_on") or ""):
        return False
    return es.public(model)


# 17h: the setup record as it may go out, field by field — its launch's flags
# and environment by name (export_safe), never its address, its port or how
# it is served in words
SETUP = {"model": "name", "name": "text", "runtime": "text", "file": "name", "build": "scalar",
         "ctx": int, "quant": "name", "phone": bool, "lookahead": bool, "mtp": bool,
         "thinking": bool, "thinking_mode": "text", "battery": "text", "cap": int, "seed": int,
         "revision": "name", "dtype": "name", "where": "text", "scoring": "text",
         "reading": {"on": bool, "marks": ["text"], "opens": bool},
         "props": {"build": "scalar", "n_ctx": int, "total_slots": int, "speculative": int},
         "battery_hashes": {"*": "name"}, "runs": [int], "provisional": "text"}


def setup_out(setup: dict) -> tuple[dict, int]:
    """the setup that may go out, and how many launch flags or variables were
    left out"""
    import export_safe as es
    out = es.pick(setup, SETUP) or {}
    launch = setup.get("launch") if isinstance(setup.get("launch"), dict) else {}
    fl, l1 = es.flags(launch.get("flags"))
    ev, l2 = es.env(launch.get("env"))
    out["launch"] = {"flags": fl, "env": ev}
    return out, l1 + l2


def runs_of(row: Path) -> list[dict]:
    """the DeviceMark runs that made this row (its full battery), oldest first"""
    from service import db
    model, thinking = model_of(row), row.name.endswith("__thinking")
    return sorted((r for r in db.recent(5000) if r.get("suite") == "devicemark"
                   and r.get("hf_id") == model and bool(r.get("thinking")) == thinking
                   and (r.get("part") or "full") == "full"), key=lambda r: r["id"])


def log_of(row: Path, runs: list[dict]) -> str:
    from service import config
    out = []
    for r in runs:
        p = config.LOGS_DIR / f"service_{r['id']}_{r['hf_id'].replace('/', '__')}.log"
        if p.exists():
            out.append(f"===== run #{r['id']} · {r.get('status')} =====\n"
                       + p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(out) or "(no log on this server for this row)\n"


def item_record(s: dict, item: dict | None, reading: dict | None = None) -> dict:
    """one scored item, in DeviceMark's raw field names where ours mean the same.
    15.7: split as it was scored (the row's reading), where the cap fell"""
    if reading is not None and "scored_answer" in s:
        got = dm.read_reply(s, reading)
        thinking, answer = got["thinking"] or "", s["scored_answer"] or ""
    elif "answer" in s and s.get("text") is None:
        thinking, answer = s.get("thinking") or "", s.get("answer") or ""
    else:
        thinking, answer = dm.split_thinking(s.get("text") or s.get("answer") or "")
    prompt = (item or {}).get("text") or ""
    out = {"test": s["bench"], "key": str(s["key"]), "prompt": prompt,
           "prompt_chars": len(prompt), "thinking": thinking, "thinking_chars": len(thinking),
           "answer": answer, "content_chars": len(answer),
           "generated_tokens": s.get("gen_tokens"), "capped": bool(s.get("capped")),
           # 15.7: where the cap fell, and whether the thinking closed
           "cap_in": s.get("cap_in"), "thinking_closed": s.get("closed"),
           "answered": bool(s.get("answered")), "correct": bool(s.get("correct"))}
    for k in ("prompt_tokens", "decode_tok_s", "wall_s"):
        if s.get(k) is not None:
            out[k] = s[k]
    if s["bench"] == "ifeval":
        v = s.get("ifeval") or {}
        ids = list((item or {}).get("instruction_id_list") or [])
        out["ifeval"] = {m: {"prompt": bool((v.get(m) or {}).get("prompt")),
                             "instructions": [bool(x) for x in (v.get(m) or {}).get("inst") or []]}
                         for m in ("strict", "loose")}
        out["rules"] = [{"id": rid, "strict": st, "loose": lo} for rid, st, lo in
                        zip(ids, out["ifeval"]["strict"]["instructions"],
                            out["ifeval"]["loose"]["instructions"])]
    else:
        out.update(parsed=s.get("parsed"), parsed_how=s.get("how") or None, gold=s.get("gold"))
    return out


def scores_of(row: dict, items: list[dict]) -> dict:
    """the row's numbers, and their summary's counts for each test"""
    tests = {}
    for b in dm.BENCHES:
        got = [i for i in items if i["test"] == b]
        toks = [i["generated_tokens"] for i in got if isinstance(i["generated_tokens"], int)]
        bench = row["benches"][b]
        tests[b] = {**{k: bench.get(k) for k in ("n", "acc", "ci", "answered", "answered_pct",
                                                   "median_tokens", "acc_answered", "parts")
                       if k in bench},
                    "capped_count": sum(i["capped"] for i in got), "max_tokens": dm.CAP,
                    "mean_generated_tokens": round(sum(toks) / len(toks), 2) if toks else None,
                    "total_generated_tokens": sum(toks) if toks else None}
    return {"version": row.get("version"), "composite": row["composite"], "tests": tests,
            "n": row.get("n"), "answered_pct": row.get("answered_pct"),
            "median_tokens": row.get("median_tokens"), "time_frontier": row.get("time_frontier")}


def readme(model: str, thinking: bool, setup: dict, scores: dict, bat: dict) -> str:
    rev = bat.get("revisions") or {}
    src = "\n".join(f"- **{dm.LABEL[b]}**: [{name}](https://huggingface.co/datasets/{name}) at "
                    f"revision `{str(rev.get(name, ''))[:12]}`, {lic}" for b, (name, lic) in
                    LICENCES.items())
    comp = scores["composite"]
    return f"""# DeviceMark raw run: {model}{' · thinking' if thinking else ''}

Every item of one row on our board, as it ran: the prompt as sent, the reply and its thinking,
the parsed answer and whether it is right, IFEval's rules, the tokens. The same layout as
DeviceMark's raw files ([devicemark/results](https://huggingface.co/datasets/devicemark/results),
`raw/`) wherever a field means the same thing.

Composite **{100 * comp['value']:.1f}** (95% interval {100 * comp['ci'][0]:.1f}–{100 * comp['ci'][1]:.1f}),
the mean of IFEval {100 * scores['tests']['ifeval']['acc']:.1f}, MMLU-Pro
{100 * scores['tests']['mmlu_pro']['acc']:.1f} and MATH {100 * scores['tests']['math']['acc']:.1f}.

## The protocol ({dm.VERSION})

- 0-shot; the item's prompt as the one user message, in the model's own chat template.
- Greedy: temperature 0, seed {dm.SEED}.
- A cap of {dm.CAP:,} generated tokens, thinking included. No answer within the cap is wrong
  and stays in the denominator (`answered`, beside each test's score).
- Thinking {'on' if thinking else 'off'}{f" ({setup.get('thinking_mode')})" if setup.get('thinking_mode') else ''}.
- Run on {setup.get('runtime') or 'hf transformers (lm_eval)'}{', ' + setup['where'] if setup.get('where') else ''}.

## The battery

{src}

- IFEval: DeviceMark's own 300 keys.
- MMLU-Pro: our draw of their design, 14 a category × 14 categories.
- MATH-500: our draw, 100 across its 7 subjects.

## The scores

- **IFEval:** its official checkers. The score is the mean of prompt- and instruction-level,
  strict and loose.
- **MMLU-Pro:** the letter in the last `\\boxed{{}}`, else one of a short list of unambiguous
  phrasings (`parsed_how`).
- **MATH:** the last `\\boxed{{}}`, equal as maths (math-verify).
- **Intervals:** each test's is Wilson's 95%. The composite's is an item bootstrap: 2,000
  resamples within each test, seed 0.

## Recompute

```bash
python recompute.py
```

It reads `items.jsonl`, computes `scores.json` again from each item's verdict, and says
whether the two agree. It uses the standard library only.

## The files

- `items.jsonl`: one item a line.
  - DeviceMark's fields: `key`, `answer`, `capped`, `generated_tokens`, `prompt_chars`,
    `content_chars`, `thinking_chars`; a served row also has `prompt_tokens`,
    `decode_tok_s` and `wall_s`.
  - Ours: `test`, `prompt`, `thinking`, `answered`, `correct`; `parsed`, `parsed_how` and
    `gold` for MMLU-Pro and MATH; `ifeval` and `rules` (each instruction, strict and loose)
    for IFEval.
- `scores.json`: each test's score and interval, and the composite. DeviceMark's summary
  counts are beside them: `n`, `capped_count`, `max_tokens`, `mean_generated_tokens`,
  `total_generated_tokens`.
- `setup.json`: the run's setup.
- `log.txt`: the run's log, scrubbed of keys, paths, host names and private addresses.

Licences: IFEval Apache-2.0; MMLU-Pro MIT; MATH-500 MIT. The replies are the model's.
"""


RECOMPUTE = '''#!/usr/bin/env python3
"""scores.json again, from items.jsonl: each test's score and Wilson interval,
the composite and its item bootstrap (2,000 resamples within each test, seed
0) — the board's own arithmetic (scripts/devicemark.py), standard library only.

    python recompute.py
"""
import json
import math
import random
import sys
from pathlib import Path

TESTS = ("ifeval", "mmlu_pro", "math")
Z = 1.959963984540054
BOOT_N, BOOT_SEED = 2000, 0


def wilson(p, n):
    if n <= 0:
        return [0.0, 0.0]
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def ifeval_mean4(items):
    if not items:
        return 0.0
    parts = []
    for mode in ("strict", "loose"):
        parts.append(sum(i["ifeval"][mode]["prompt"] for i in items) / len(items))
        inst = [x for i in items for x in i["ifeval"][mode]["instructions"]]
        parts.append(sum(inst) / len(inst) if inst else 0.0)
    return sum(parts) / 4


def score(test, items):
    if test == "ifeval":
        return ifeval_mean4(items)
    return sum(bool(i["correct"]) for i in items) / len(items) if items else 0.0


def composite_ci(by):
    rng = random.Random(BOOT_SEED)
    vals = []
    for _ in range(BOOT_N):
        vals.append(sum(score(t, [rng.choice(v) for _ in v]) for t, v in by.items()) / len(by))
    vals.sort()
    return [round(vals[int(0.025 * (BOOT_N - 1))], 4),
            round(vals[int(math.ceil(0.975 * (BOOT_N - 1)))], 4)]


def recompute(items):
    by = {t: [i for i in items if i["test"] == t] for t in TESTS}
    tests = {t: {"n": len(v), "acc": round(score(t, v), 4), "ci": wilson(score(t, v), len(v))}
             for t, v in by.items()}
    comp = round(sum(tests[t]["acc"] for t in TESTS) / 3, 4)
    return {"composite": {"value": comp, "ci": composite_ci(by)}, "tests": tests}


def main():
    here = Path(__file__).resolve().parent
    items = [json.loads(x) for x in (here / "items.jsonl").read_text(encoding="utf-8").splitlines()
             if x.strip()]
    got = recompute(items)
    want = json.loads((here / "scores.json").read_text(encoding="utf-8"))
    bad = [] if got["composite"] == want["composite"] else ["composite"]
    bad += [t for t in TESTS if {k: want["tests"][t][k] for k in ("n", "acc", "ci")} != got["tests"][t]]
    print(json.dumps(got, indent=1))
    print("agrees with scores.json" if not bad else "differs from scores.json: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
'''


def refused(tasks) -> str:
    """14.4.4: '' when every task may leave the server; else why not — a
    non-commercial set never does, whatever the row"""
    return restrictions.stays_here(tasks)


def export_row(row: Path, out: Path, public: bool | None = None, runs: list[dict] | None = None,
               say=print, publish: bool = True) -> Path:
    """one row's folder, scrubbed, under out/public or out/private"""
    from service import config
    why = refused(dm.TASK.values())
    if why:
        raise SystemExit(f"{row.name}: {why} Nothing was exported.")
    data = json.loads((row / dm.OUT_NAME).read_text(encoding="utf-8"))
    if data.get("part", "full") != "full" or data.get("composite", {}).get("value") is None:
        raise SystemExit(f"{row.name} has no full battery scored: nothing to export")
    setup = data.get("setup") or {}
    scored = dm.read_items(row)
    if not scored:
        raise SystemExit(f"{row.name} has no per-item record ({dm.ITEMS_NAME})")
    try:
        items_src = dm.load_items(config.DM_ITEMS)
    except Exception:                               # noqa: BLE001 — prompts left out, said
        items_src = {}
    model, thinking = model_of(row), row.name.endswith("__thinking")
    items = [item_record(s, items_src.get((s["bench"], str(s["key"]))), setup.get("reading"))
             for s in scored]
    scores = scores_of(data, items)
    # 17g: never public for a model the board doesn't know as public, whatever
    # the flags — --private may still keep a public one private
    pub = public_by_default(row, setup) and public is not False and publish
    dest = out / ("public" if pub else "private") / row.name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    runs = runs_of(row) if runs is None else runs
    import export_safe as es
    full_setup, left = setup_out({**setup, "model": model, "battery": dm.VERSION,
                                  "battery_hashes": dm.battery_hashes(items_src)
                                  if items_src else None,
                                  "runs": [r["id"] for r in runs]})
    # 17h: the runner's own lines only (DeviceMark's battery is public: no
    # question is withheld here)
    lines, out_ = es.log_lines(log_of(row, runs), es.Questions([]))
    files = {
        "items.jsonl": "".join(json.dumps(scrub_obj(i), ensure_ascii=False) + "\n" for i in items),
        "setup.json": json.dumps(scrub_obj(full_setup), indent=1, ensure_ascii=False),
        "scores.json": json.dumps(scores, indent=1),
        "log.txt": scrub("\n".join(lines) + "\n"),
        "recompute.py": RECOMPUTE,
        "README.md": scrub(readme(model, thinking, full_setup, scores, dm.battery())
                           + "\nWritten by a list of what may go out, field by field"
                           + (f" ({left} launch flags or variables left out)" if left else "")
                           + "; the log holds only the runner's own lines"
                           + (": " + "; ".join(f"{n} left out ({w})" for w, n in out_.items())
                              if out_ else "") + ".\n")}
    for name, text in files.items():
        (dest / name).write_text(text, encoding="utf-8")
    say(f"{row.name}: {len(items)} items · composite {100 * scores['composite']['value']:.1f} · "
        f"{'public' if pub else 'private'} · {dest}")
    return dest


def set_link(row: Path, url: str | None, by: str) -> dict | None:
    """the row's published raw run, or none: the board's "raw" link"""
    p = row / RAW_NAME
    if not url:
        p.unlink(missing_ok=True)
        return None
    if not re.match(r"^https://[\w.-]+\.[a-z]{2,}/\S+$", url):
        raise ValueError("a raw run's link is an https URL (its folder on Hugging Face)")
    rec = {"url": url, "by": by, "at": time.time()}
    p.write_text(json.dumps(rec), encoding="utf-8")
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pick = ap.add_mutually_exclusive_group(required=True)
    pick.add_argument("--run", type=int, help="a DeviceMark run's id: its row")
    pick.add_argument("--all", action="store_true", help="every DeviceMark row")
    pick.add_argument("--link", nargs="+", metavar=("RUN", "URL"),
                      help="a run's row, and the URL of its published raw run")
    ap.add_argument("--unlink", action="store_true", help="with --link: take the link off")
    ap.add_argument("--by", default="", help="your name, with --link")
    ap.add_argument("--out", type=Path, default=None,
                    help="where the folders go (default $BENCH_ROOT/raw-export)")
    ap.add_argument("--private", action="store_true",
                    help="every row private, even a model marked public")
    ap.add_argument("--public", action="store_true",
                    help="17h: no effect — a model is public when its page marks its weights "
                         "public, and the export asks before it writes anything there")
    a = ap.parse_args(argv)
    from service import config, db
    db.init()
    if a.link:
        row, _ = row_of_run(int(a.link[0]))
        if not a.by.strip():
            ap.error("--by: your name, for the link's record")
        got = set_link(row, None if a.unlink else (a.link[1] if len(a.link) > 1 else None),
                       a.by.strip()[:80])
        print(f"{row.name}: " + (f"raw → {got['url']}" if got else "no raw link"))
        return 0
    out = a.out or (config.BENCH_ROOT / "raw-export")
    rows = [row_of_run(a.run)[0]] if a.run else all_rows()
    if a.public:
        print("--public: no effect — a model is public when its page marks its weights public")
    import export_safe as es
    would = [] if a.private else [model_of(r) for r in rows if public_by_default(r, json.loads(
        (r / dm.OUT_NAME).read_text(encoding="utf-8")).get("setup") or {})]
    publish = es.confirm(would)
    if would and not publish:
        print("not published: those rows go to private/")
    for row in rows:
        export_row(row, out, False if a.private else None, publish=publish)
    print(f"{len(rows)} row(s) under {out} — upload: docs/REMOTE-RUNS.md § Publishing the raw runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
