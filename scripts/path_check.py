"""31.1: the settings check of a runtime of expert paths (k3k6-2611, step2388's),
before any benchmark — what the file stores for each path, what the server
says it applied, and what it does, held to fb.PATHS. Nothing is worked around:
a value that doesn't match stops the box, and the check says which and where.

On a server started for one path (scripts/remote_gguf.py — before each step,
and alone with --check):

1. the file: its teraformer.paths.* entries, read from its header (the path
   list [3, 6], the default 3, and each path's sampling and thinking);
2. the server: its /props `expert_paths` (each path's params and thinking),
   and its start-up lines ("expert path k3 (default): temp = …");
3. two short questions on the path, each carrying the path and nothing that
   samples: a thinking block on k6, none on k3 — and each answer's
   generation_settings when the build returns them;
4. on k6, whether the presence penalty counts the prompt's tokens: one prompt
   of a word many times, its first token's probabilities read twice from
   /completion — before the samplers and after them. A token's log
   probability after them, less its log probability before them over the
   temperature, is the same for every token the penalty leaves alone; a
   penalty on the prompt's tokens would put those 0.75 below the others
   (divided by the temperature). When too few tokens come back on either
   side, it says so and the setting the server reports is what holds.

Standard library only, beside scripts/gguf_header.py."""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from pathlib import Path

import frontier as fb
import gguf_header

PREFIX = "teraformer.paths."
# the names a value may go by — the file's, the server's /props and its log
ALIASES = {"temp": "temperature", "temperature": "temperature", "top_p": "top_p",
           "top_k": "top_k", "min_p": "min_p", "presence_penalty": "presence_penalty",
           "penalty_present": "presence_penalty", "penalty_last_n": "penalty_last_n",
           "repeat_last_n": "penalty_last_n", "last_n": "penalty_last_n",
           "penalty_prompt_tokens": "penalty_prompt_tokens",
           "thinking": "thinking", "enable_thinking": "thinking", "reasoning": "thinking"}
# two short questions a path answers, each with as little room as it needs
QUESTIONS = ("What is 7 times 8? Reply with the number alone.",
             "Name the largest planet in the solar system, in one word.")
ROOM = {3: 256, 6: 6144}
# the probe: a word many times, then room for one more
PROBE_WORD = "river"
PROBE = ("A list of words, some of them many times over: "
         + ", ".join([PROBE_WORD] * 48 + ["stone", PROBE_WORD, "cloud", PROBE_WORD])
         + ". One more word for the list:")
TOL = 1e-4


def _name(k: str) -> str:
    return ALIASES.get(k.strip().lower().replace("-", "_").rsplit(".", 1)[-1], "")


def _value(v):
    """a value as the comparison reads it: a number, or a boolean"""
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "on", "yes"):
            return True
        if s in ("false", "off", "no"):
            return False
        try:
            return float(s) if re.search(r"[.e]", s) else int(s)
        except ValueError:
            return v
    return v


def normal(d: dict) -> dict:
    """{temperature, top_p, …, thinking} out of any of the names they go by;
    the rest left out"""
    out = {}
    for k, v in (d or {}).items():
        n = _name(str(k))
        if n:
            out[n] = _value(v)
    return out


def stored(gguf: str | Path) -> dict | None:
    """the file's own: {"keys": every teraformer.paths.* entry as read, "top_k":
    the path list, "default": the default path, "paths": {3: {…}, 6: {…}}} —
    None when its header can't be read"""
    raw = gguf_header.keys(gguf, PREFIX)
    if raw is None:
        return None
    paths: dict[int, dict] = {}
    for k, v in raw.items():
        m = re.match(r"^teraformer\.paths\.k(\d+)\.(?:sampling\.)?(.+)$", k)
        if m and _name(m.group(2)):
            # a float as the header keeps it (32 bits: 0.8 reads 0.800000011920929)
            paths.setdefault(int(m.group(1)), {})[_name(m.group(2))] = (
                round(v, 6) if isinstance(v, float) else _value(v))
    return {"keys": raw, "top_k": raw.get(PREFIX + "top_k"), "default": raw.get(PREFIX + "default"),
            "paths": paths}


def _path_id(item: dict) -> int | None:
    for k in ("path", "expert_path", "k", "top_k", "id", "n_expert_used", "experts"):
        v = item.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            return v
        if isinstance(v, str) and re.fullmatch(r"k?\d+", v.strip()):
            return int(v.strip().lstrip("k"))
    return None


def props_paths(props: dict) -> dict[int, dict]:
    """the server's /props `expert_paths`: {path: {"params": {…}, "thinking": …,
    "default": …}} — {} when it names none"""
    got = props.get("expert_paths") if isinstance(props, dict) else None
    items = []
    if isinstance(got, list):
        items = [(None, x) for x in got if isinstance(x, dict)]
    elif isinstance(got, dict):
        items = [(k, x) for k, x in got.items() if isinstance(x, dict)]
    out = {}
    for key, x in items:
        p = _path_id(x)
        if p is None and key is not None and re.fullmatch(r"k?\d+", str(key)):
            p = int(str(key).lstrip("k"))
        if p is None:
            continue
        params = normal(x.get("params") or x.get("sampling") or {})
        th = x.get("thinking", params.get("thinking"))
        if th is not None:
            params["thinking"] = bool(_value(th))
        out[p] = {"params": params, "default": x.get("default"), "raw": x}
    return out


LINE = re.compile(r"expert path k(\d+)(\s*\(default\))?\s*:\s*(.*)$")


def startup(text: str) -> dict[int, dict]:
    """the start-up lines "expert path k3 (default): temp = 0.800, …": {path:
    {"line", "default", "values"}} — the last of each in `text`"""
    out = {}
    for ln in (text or "").splitlines():
        m = LINE.search(ln)
        if not m:
            continue
        vals = dict(re.findall(r"([A-Za-z_][\w-]*)\s*=\s*(-?[\d.]+(?:e-?\d+)?|true|false|on|off)",
                               m.group(3)))
        out[int(m.group(1))] = {"line": m.group(0).strip()[:400], "default": bool(m.group(2)),
                                "values": normal(vals)}
    return out


def compare(want: dict, got: dict, where: str, slot_ctx: int | None = None,
            all_of: bool = True) -> list[str]:
    """each of `want` that `got` gives another way — or, `all_of`, doesn't
    give at all. penalty_last_n -1 is the whole context: the server may say
    the context it resolved it to"""
    bad = []
    for k, w in want.items():
        if k not in got:
            if all_of:
                bad.append(f"{where} gives no {fb.PATH_WORDS.get(k, k)}")
            continue
        v = got[k]
        if k == "thinking":
            ok = isinstance(v, (bool, int)) and bool(v) == w
        elif k == "penalty_last_n" and w == -1:
            ok = isinstance(v, (int, float)) and (v == -1 or (slot_ctx and v >= slot_ctx))
        else:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v - w) < TOL
        if not ok:
            bad.append(f"{where}: {fb.PATH_WORDS.get(k, k)} {v!r}, not {w!r}")
    return bad


def _post(base: str, path: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{path}: HTTP {e.code} {e.read()[:200]!r}") from None


def _get(base: str, path: str) -> dict:
    try:
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return {}


def answers(base: str, path: int) -> list[dict]:
    """the two questions on `path`, each carrying its messages, its room and the
    path alone: did it think, what it said, and its generation_settings when
    the build returns them"""
    out = []
    for q in QUESTIONS:
        body = {"messages": [{"role": "user", "content": q}], "max_tokens": ROOM[path],
                "expert_path": path}
        got = _post(base, "/v1/chat/completions", body)
        ch = (got.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content, think = msg.get("content") or "", msg.get("reasoning_content") or ""
        gs = got.get("generation_settings") or ch.get("generation_settings")
        out.append({"question": q, "sent": sorted(body),
                    "thinking": bool(think.strip()) or "</think>" in content,
                    "thinking_chars": len(think), "reply": content.strip()[:200],
                    "finish": ch.get("finish_reason"),
                    **({"generation_settings": normal(gs)} if isinstance(gs, dict) else {})})
    return out


def _probs(entry: dict) -> dict[int, float]:
    """{token id: log probability} from one /completion probabilities entry —
    after the samplers ("top_probs", "prob") or before them ("top_logprobs",
    "logprob")"""
    out = {}
    for x in entry.get("top_probs") or entry.get("top_logprobs") or []:
        if not isinstance(x, dict) or not isinstance(x.get("id"), int):
            continue
        if isinstance(x.get("logprob"), (int, float)):
            out[x["id"]] = float(x["logprob"])
        elif isinstance(x.get("prob"), (int, float)) and x["prob"] > 0:
            out[x["id"]] = math.log(x["prob"])
    return out


def prompt_penalty(base: str, path: int, want: dict) -> dict:
    """does the presence penalty count the prompt's tokens? (see the module's
    words) — {"verdict": "not counted" | "counted" | "inconclusive", …}"""
    pen, temp = float(want.get("presence_penalty") or 0), float(want["temperature"])
    try:
        ids = set(_post(base, "/tokenize", {"content": PROBE, "add_special": True})
                  .get("tokens") or [])
        got = {}
        for after in (True, False):
            r = _post(base, "/completion", {"prompt": PROBE, "n_predict": 1, "n_probs": 40,
                                            "post_sampling_probs": after, "expert_path": path,
                                            "cache_prompt": False, "seed": 7})
            got[after] = _probs((r.get("completion_probabilities") or [{}])[0])
    except (RuntimeError, OSError, ValueError, TypeError, KeyError, IndexError) as e:
        return {"verdict": "inconclusive", "why": f"the server didn't give probabilities: {e}"}
    both = [i for i in got[True] if i in got[False]]
    inside = [i for i in both if i in ids]
    outside = [i for i in both if i not in ids]
    words = {"prompt_tokens_kept": len(inside), "other_tokens_kept": len(outside),
             "after_samplers": len(got[True]), "before_samplers": len(got[False])}
    if not inside or not outside:
        return {"verdict": "inconclusive", **words,
                "why": "too few tokens came back on both sides of the prompt to tell"}
    # the temperature, as the tokens the penalty leaves alone show it: log
    # after = log before / T + a constant (the slope over them, two or more)
    slope = 1 / temp
    if len(outside) >= 2:
        xs, ys = [got[False][i] for i in outside], [got[True][i] for i in outside]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        var = sum((x - mx) ** 2 for x in xs)
        if var > 1e-9:
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
    res = {i: got[True][i] - slope * got[False][i] for i in both}
    shift = (sum(res[i] for i in inside) / len(inside)
             - sum(res[i] for i in outside) / len(outside))
    # in logit terms: what the prompt's tokens lost, over what the others did
    est = -shift / slope if slope > 0 else float("nan")
    verdict = ("counted" if est > pen / 2 else "not counted") if pen else "inconclusive"
    return {"verdict": verdict, **words, "penalty_seen": round(est, 3) + 0.0,
            "penalty_if_counted": pen, "temperature_seen": round(1 / slope, 3) if slope > 0 else None}


def check(base: str, path: int, gguf: str | Path, log_text: str,
          slot_ctx: int | None = None) -> dict:
    """the settings check on a server started for `path`: every reading, and
    "pass" with the mismatches (none on a pass)"""
    want = fb.PATHS[path]
    bad: list[str] = []
    file = stored(gguf)
    if file is None:
        bad.append("the file's metadata can't be read")
        file = {"keys": {}, "top_k": None, "default": None, "paths": {}}
    else:
        if file["top_k"] != sorted(fb.PATHS):
            bad.append(f"the file's path list (teraformer.paths.top_k) is {file['top_k']!r}, not "
                       f"{sorted(fb.PATHS)}")
        if file["default"] != fb.PATH_DEFAULT:
            bad.append(f"the file's default path (teraformer.paths.default) is "
                       f"{file['default']!r}, not {fb.PATH_DEFAULT}")
        for p, w in fb.PATHS.items():
            bad += compare(w, file["paths"].get(p, {}), f"the file's k{p}")
    props = _get(base, "/props")
    srv = props_paths(props)
    if not srv:
        bad.append("the server's /props names no expert paths")
    for p, w in fb.PATHS.items():
        if srv:
            bad += compare(w, (srv.get(p) or {}).get("params") or {}, f"the server's k{p} (/props)",
                           slot_ctx)
    lines = startup(log_text)
    for p, w in fb.PATHS.items():
        if p not in lines:
            bad.append(f"the server's start-up lines name no expert path k{p}")
            continue
        if lines[p]["default"] != (p == fb.PATH_DEFAULT):
            bad.append(f"the server's start-up line calls k{p} "
                       + ("the default" if lines[p]["default"] else "not the default"))
        # a line says some of its values: each it says must agree
        bad += compare(w, lines[p]["values"], f"the server's start-up line for k{p}",
                       slot_ctx, all_of=False)
    try:
        said = answers(base, path)
    except (RuntimeError, OSError, ValueError) as e:
        said = []
        bad.append(f"the two questions on k{path} weren't answered: {e}")
    for k, a in enumerate(said, 1):
        if a["thinking"] != want["thinking"]:
            bad.append(f"question {k} on k{path}: "
                       + ("no thinking block, where k6 thinks" if want["thinking"] else
                          "a thinking block, where k3 doesn't think"))
        if "generation_settings" in a:
            bad += compare(want, a["generation_settings"],
                           f"question {k}'s generation_settings", slot_ctx, all_of=False)
    probe = prompt_penalty(base, path, want) if want.get("presence_penalty") else None
    if probe and probe["verdict"] == "counted":
        bad.append(f"on k{path} the presence penalty counts the prompt's tokens: they came back "
                   f"about {probe.get('penalty_seen')} below the others")
    return {"path": path, "want": want, "file": file, "server": {
                "props": {p: {k: x[k] for k in ("params", "default")} for p, x in srv.items()},
                "startup": lines},
            "answers": said, "prompt_penalty": probe, "pass": not bad, "mismatches": bad}


def words(chk: dict) -> list[str]:
    """the check in lines, for the box's log"""
    p = chk["path"]
    out = [f"settings check, expert path k{p}: " + ("pass" if chk["pass"] else "FAILED")]
    f = (chk["file"]["paths"] or {}).get(p) or {}
    out.append(f"  the file's k{p}: {fb.path_words(p, f) if f else 'nothing stored'}")
    s = ((chk["server"]["props"] or {}).get(p) or {}).get("params") or {}
    out.append(f"  the server's k{p} (/props): {fb.path_words(p, s) if s else 'none'}")
    for q, ln in sorted((chk["server"]["startup"] or {}).items()):
        out.append(f"  start-up: {ln['line']}")
    for a in chk["answers"]:
        out.append(f"  {a['question'][:40]}… → {a['reply'][:40]!r}, "
                   + ("a thinking block" if a["thinking"] else "no thinking block")
                   + f", sent {', '.join(a['sent'])}")
    pr = chk.get("prompt_penalty")
    if pr:
        out.append(f"  the presence penalty and the prompt's tokens: {pr['verdict']}"
                   + (f" (they came back {pr['penalty_seen']} below the others, where "
                      f"{pr['penalty_if_counted']} would be counted; the temperature read "
                      f"{pr.get('temperature_seen')})" if "penalty_seen" in pr else
                      f" — {pr.get('why')}; the server's own setting holds"))
    out += [f"  ✗ {m}" for m in chk["mismatches"]]
    return out
