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
    "Run as the apps run it (Kevin, 5 Nov): each turn the whole chat rendered again, the memory "
    "reused as far as the new prompt extends it, every earlier reply's thinking kept in the chat "
    "(--reasoning-preserve), 4 checkpoints a slot (--ctx-checkpoints 4), no prompt cache in host "
    "memory.",
    "Arm A is the team's default: on a path change the server clears the slot's memory and the "
    "new path reads the whole chat again, earlier thinking included. Arm B sends "
    "keep_cache_across_paths: the new path continues from the memory the other path computed, "
    "and reads only the new message. Both arms see the same text in every turn — arm A's earlier "
    "replies, thinking included — with the same seed. Before each turn arm A's previous reply is "
    "replayed in arm B's slot on the path that wrote it, so only the memory differs.",
    "The facts: a and b are read by k3 (turns 1–2) and used on k6 (turn 5); the courier and the "
    "gate are read by k6 (turn 3) and recalled on k3 (turn 4).",
    "A turn counts only when its memory was what the arm says: nothing at turn 1; later, arm A's "
    "to the new question on the same path and nothing after a switch (the server's \"clearing\" "
    "line), arm B's to the end of the previous reply (its \"keeping\" line at a switch). Turns "
    "that weren't are counted apart, not scored.",
    "Round trip: after each reply, arm A's slot reads the chat to the next question on the same "
    "path, as the apps' next request does. Token for token means the reply came back as the model "
    "wrote it and nothing was read again; otherwise the memory rolled back to the latest of its 4 "
    "checkpoints before the difference.",
    "Turn 1 is a placebo, both memories empty; turn 2 nearly one (the reply decoded in arm A, "
    "replayed in arm B).",
    "Loops use the board's rule (the last 200 characters before the end repeat at least three "
    "times in the last 20,000). Turns 1–3 are scored by the board's own scorers; turn 4 by each "
    "fact exactly, turn 5 by the value of (a × b) / (a + b).",
    "McNemar's exact test is on the chats one arm got right and the other didn't. Turn 3 asks "
    "each GPQA Diamond question up to three times; its clustered line tests by question.",
    "Not tested: chats saved to disk and restored (the apps' slot files).",
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
