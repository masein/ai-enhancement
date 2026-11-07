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
    "Run as the apps run it (Kevin, 5 Nov; Kevin and Roohi's review, 5 Nov): each turn the whole "
    "chat sent again, each earlier reply's content alone — never its thinking "
    "(--reasoning-preserve writes an empty think block where it was) — the memory reused as far "
    "as the new prompt extends it, 4 checkpoints a slot (--ctx-checkpoints 4), no prompt cache in "
    "host memory, and a conversation capped at 16,384 tokens: each turn's limit what is left.",
    "What the memory holds, the apps' way: after a k3 reply the next prompt extends it to the "
    "reply's end; after a k6 reply it rolls back to the start of that turn's question, the "
    "thinking not being in the history. So k6 borrows memory k3 built (turns 3 and 5), and what "
    "k3 borrows back after the k6 turn (turn 4) is k3's own — but where reply 2 didn't come back "
    "as written, which rolls turn 3 back to before question 2 and has k6 read that exchange "
    "again. Turn 4 is shown both ways.",
    "Arm A is the team's default: on a path change the server clears the slot's memory and the "
    "new path reads the whole chat again. Arm B sends keep_cache_across_paths: the new path "
    "continues from the memory the other path computed. Both arms see the same text in every "
    "turn — arm A's earlier replies' content — with the same seed. Before each turn arm A's "
    "previous reply is replayed in arm B's slot as arm A's server generated it (its token ids, "
    "thinking included) on the path that wrote it; the turn's request then rolls arm B's memory "
    "back as the apps' would.",
    "The facts are all read by k3: a in turn 1; b, the courier and the gate in turn 2. a and b "
    "are used on k6 in turn 5 — memory one path built and the other uses. The courier and the "
    "gate are recalled on k3 in turn 4, after the k6 turn, where the apps keep only k3's memory: "
    "a control.",
    "A turn counts only when its memory was what the arm says: nothing at turn 1; later, arm A's "
    "to the new question on the same path and nothing after a switch (the server's \"clearing\" "
    "line), arm B's where arm A's probe found the apps' next request brings it (its \"keeping\" "
    "line at a switch). Turns that weren't are counted apart, not scored.",
    "Round trip: after each reply, arm A's slot reads the chat to the next question on the same "
    "path, as the apps' next request does. As the apps' history makes it: a k3 reply kept to its "
    "end, a k6 reply rolled back to the start of its question. Otherwise the memory came back "
    "elsewhere: a k3 reply whose end the template trims rolls back to a checkpoint before it.",
    "Turn 3 asks GPQA Diamond and MATH Level 5 questions whose k6 replies in plan U all finished "
    "within 10,240 tokens, so most finish in the room the chat leaves (about 13,000); those that "
    "still ran out are counted.",
    "Turn 1 is a placebo, both memories empty; turn 2 nearly one (the reply decoded in arm A, "
    "replayed in arm B).",
    "Loops use the board's rule (the last 200 characters before the end repeat at least three "
    "times in the last 20,000). Turns 1–3 are scored by the board's own scorers; turn 4 by each "
    "fact exactly, turn 5 by the value of (a × b) / (a + b).",
    "McNemar's exact test is on the chats one arm got right and the other didn't. Turn 3 asks a "
    "question more than once (each GPQA Diamond one up to three times); its clustered line tests "
    "by question.",
    "Part 1's main sizes are about 1k and 4k tokens of memory, and the largest memory that fits "
    "the apps' 16,384 with the last question and the 512-token reference reply (about 15,500; "
    "each chat's oldest exchanges dropped until it fits); 48k is a stress test, beyond "
    "the apps' 16,384. Its histories are step2388's k3 replies, which the apps keep to their end: "
    "k3→k6 is a memory the apps hold; k6→k3 isn't (no k6 memory outlives its turn) and bounds "
    "what keeping k6's would do.",
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
