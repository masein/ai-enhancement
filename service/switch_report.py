"""32: the report "Switching mode mid-chat" (eval_tasks/reports/switching-mode.json,
kind "switch") — the switch experiment of brief 32: step2388 on the team's
research build, a chat switching between k3 and k6 with the memory cleared
(arm A, the team's default) or kept (arm B, keep_cache_across_paths).

Internal: no row of the board's tables, nothing in the whitepaper. Every
number is the summary the import keeps ($BENCH_ROOT/switch/summary.json,
scripts/switch_box.py summarise), read when the page asks, never worked out
here. No question text: counts, scores and memory only. Nothing here starts a
run, an import or a grading batch."""

from __future__ import annotations

import json
import time

from . import config

KIND = "switch"
NOTES = [
    "Arm A is the team's default: on a path change the server clears the slot's memory and the "
    "new path reads the whole chat again. Arm B sends keep_cache_across_paths: the new path "
    "continues from the memory the other path computed. Both arms see the same text in every "
    "turn — arm A's earlier replies — with the same seed, so only the memory differs.",
    "What arm B keeps: the chat template drops each earlier reply's thinking, so the server rolls "
    "back to the start of the previous question. The previous reply is always read by the new "
    "path; the memory kept is the chat before the previous question.",
    "A turn counts only when its memory was what the arm says: arm A cleared at a switch (the "
    "server's \"clearing\" line, nothing reused), arm B kept it (its \"keeping\" line, and at "
    "least the chat up to the previous question reused). Turns that weren't are counted apart, "
    "not scored.",
    "Turns 1 and 2 run on k3 with the same memory in both arms: a placebo, their difference "
    "should be noise.",
    "Loops use the board's rule (the last 200 characters before the end repeat at least three "
    "times in the last 20,000). Turns 1–3 are scored by the board's own scorers; turn 4 by each "
    "fact exactly, turn 5 by the value of (a × b) / (a + b).",
    "McNemar's exact test is on the chats one arm got right and the other didn't. Turn 3 asks "
    "each GPQA Diamond question up to three times; its clustered line tests by question.",
]


def root():
    return config.BENCH_ROOT / "switch"


def _read(name: str) -> dict | None:
    try:
        got = json.loads((root() / name).read_text(encoding="utf-8"))
        return got if isinstance(got, dict) else None
    except (OSError, ValueError):
        return None


def _imports() -> list[dict]:
    return [x for x in (_read("imports.json") or {}).get("imports") or [] if isinstance(x, dict)]


def _changed() -> float | None:
    try:
        return (root() / "summary.json").stat().st_mtime
    except OSError:
        return None


def listing_line() -> dict:
    """the card's lines: what is in, and when its numbers last changed"""
    from .reports import when
    s = _read("summary.json") or {}
    chats = ((s.get("part2") or {}).get("chats")) or 0
    jobs = ((s.get("part1") or {}).get("jobs")) or 0
    at = _changed()
    return {"line": f"step2388 on the team's research build · {chats:,} chats and {jobs:,} drift "
                    "runs imported" if chats or jobs else
                    "step2388 on the team's research build · nothing imported yet",
            "changed": when(at) if at else "", "changed_at": at}


def report(rep: dict) -> dict:
    """the report's payload: the summary as kept, and its imports"""
    from .reports import when
    s = _read("summary.json")
    at = _changed()
    return {"id": rep["id"], "kind": KIND, "title": rep.get("title") or rep["id"],
            "what": rep.get("what") or "", "internal": True, "notes": NOTES,
            "part2": (s or {}).get("part2"), "part1": (s or {}).get("part1"),
            "imports": [{k: x.get(k) for k in ("bundle", "what", "part", "chats", "at", "gpu")}
                        for x in _imports()],
            "sources": {"ours": {"at": at, "words": f"as of the last import, {when(at)}"
                                 if at else "nothing imported yet"}},
            "read_at": time.time()}
