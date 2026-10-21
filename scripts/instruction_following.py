"""25: instruction following, checked by each benchmark's own code on the
reply alone — IFEval's checkers (scripts/ifeval_official, lm_eval v0.4.12's
copy, as DeviceMark's IFEval is checked) and IFBench's (scripts/ifbench_official,
AI2's, vendored at a commit). Each instruction strict and loose, as their own
evaluation code decides them: nothing here changes what they check.

The official code draws an instruction's unset arguments at random, and
langdetect guesses from a random start: both are seeded before each check,
as DeviceMark seeds them, so a reply checks the same every time, and the
process's random state is put back after. One reply at a time (the seed is
the process's). A verdict is kept by what it was asked of — a page reading
answers asks again — up to KEEP of them."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import threading

SEED = 0                                  # DeviceMark's (scripts/devicemark.py)
KEEP = 20_000
_LOCK = threading.Lock()
_KEPT: dict[str, dict] = {}


def _seed() -> None:
    import langdetect
    random.seed(SEED)
    langdetect.DetectorFactory.seed = SEED


def _ifeval(prompt: str, ids: list[str], kwargs: list[dict], reply: str) -> dict:
    from ifeval_official import utils as U
    inp = U.InputExample(key=0, instruction_id_list=list(ids), prompt=prompt,
                         kwargs=copy.deepcopy(kwargs))
    out = {}
    for name, fn in (("strict", U.test_instruction_following_strict),
                     ("loose", U.test_instruction_following_loose)):
        _seed()
        out[name] = [bool(x) for x in fn(inp, reply).follow_instruction_list]
    return out


def _ifbench(prompt: str, ids: list[str], kwargs: list[dict], reply: str) -> dict:
    from ifbench_official import evaluation_lib as E
    # as its run_eval asks: the reply found by its prompt, strict first (which
    # drops the unset arguments it then leaves for loose)
    inp = E.InputExample(key=0, instruction_id_list=list(ids), prompt=prompt,
                         kwargs=copy.deepcopy(kwargs))
    out = {}
    for name, fn in (("strict", E.test_instruction_following_strict),
                     ("loose", E.test_instruction_following_loose)):
        _seed()
        out[name] = [bool(x) for x in fn(inp, {prompt: reply}).follow_instruction_list]
    return out


CHECKERS = {"ifeval": _ifeval, "ifbench": _ifbench}


def check(kind: str, prompt: str, ids: list[str], kwargs: list[dict], reply: str) -> dict:
    """{"strict": [bool, …], "loose": [bool, …]}, one a instruction, for one
    reply (its thinking already taken off)"""
    key = hashlib.sha256(json.dumps([kind, prompt, ids, kwargs, reply], sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()
    with _LOCK:
        got = _KEPT.get(key)
        if got is None:
            import langdetect
            state, was = random.getstate(), langdetect.DetectorFactory.seed
            try:
                got = CHECKERS[kind](prompt, ids, kwargs, reply)
            finally:
                random.setstate(state)
                langdetect.DetectorFactory.seed = was
            if len(_KEPT) >= KEEP:
                _KEPT.clear()
            _KEPT[key] = got
    return {k: list(v) for k, v in got.items()}


def numbers(follows: list[dict]) -> dict:
    """the four numbers over a benchmark's answers, as IFEval's and IFBench's
    reports give them — prompt-level, the share of prompts whose every
    instruction was followed (with its standard error over prompts), and
    instruction-level, the share of every instruction asked — strict and loose"""
    out: dict = {"prompts": len(follows), "instructions": sum(len(f["strict"]) for f in follows)}
    for how in ("strict", "loose"):
        all_ = [1.0 if f[how] and all(f[how]) else 0.0 for f in follows]
        n = len(all_)
        mu = sum(all_) / n if n else None
        var = sum((x - mu) ** 2 for x in all_) / (n - 1) if n > 1 else 0.0
        out[f"prompt_{how}"] = {"v": mu, "se": (var / n) ** 0.5 if n else None}
        each = [x for f in follows for x in f[how]]
        out[f"inst_{how}"] = {"v": sum(each) / len(each) if each else None}
    return out
