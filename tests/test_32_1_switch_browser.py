"""32.1, 32.2 and 32.4 in a real browser, at 1280 and 400 px, light and dark:
Reports ▸ "Switching mode mid-chat", internal — empty until plan S is
imported, then its tiles, Part 2 by turn and arm (B − A with McNemar's p,
loops, ran out, no answer, length, cache_n, prompt_n and each reply's round
trip as the apps' next request reads it; turn 3 by question, the wrong
recalls, turn 4 by reply 2, the replies that came back otherwise, the apps'
16,384-token cap) and Part 1 by memory size, direction and run, 48k a stress
test apart. The summary is the import's own (switch_box.summarise) over
made-up chats and drift runs; no question text anywhere."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import switch_box as sw
from service import config
from test_21d_browser import no_sideways

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase32"
SIZES = [(1280, "light"), (1280, "dark"), (400, "light"), (400, "dark")]


def arm(rng: random.Random, ok_p: float, path: int, cache: int, prompt: int, t: int) -> dict:
    ok = rng.random() < ok_p
    out = path == 6 and rng.random() < 0.02
    return {"score": {"ok": ok and not out, "ran_out": out, "loop": out and rng.random() < 0.7,
                      "no_answer": not ok and not out and rng.random() < 0.1,
                      **({"wrong": [] if ok else ["ship"]} if t == 4 else {})},
            "tokens": rng.randint(9000, 30000) if path == 6 else rng.randint(150, 600),
            "cache_n": cache, "prompt_n": prompt}


def made_up(d: Path, n: int = 200) -> None:
    """an import's files: the inputs, a box's chats, a box's drift runs"""
    rng = random.Random(32)
    lines = [{"part": "chats", "chat": i, "mmlu": f"m{i}", "gsm8k": f"g{i}",
              "hard": {"task": "gpqa_diamond_epoch", "id": f"rec{i % 66}"},
              "facts": sw.facts_of(i)} for i in range(n)]
    d.mkdir(parents=True, exist_ok=True)
    (d / sw.INPUT).write_text("".join(json.dumps(x) + "\n" for x in lines))
    rows = []
    for i in range(n):
        turns = []
        for t, path in enumerate(sw.TURNS, 1):
            switch = t > 1 and path != sw.TURNS[t - 2]
            p = {1: 0.88, 2: 0.95, 3: 0.62, 4: 0.97, 5: 0.9}[t]
            a = arm(rng, p, path, 0 if switch else 400 * (t - 1), 900 + 300 * t, t)
            b = arm(rng, p - (0.03 if t in (3, 4) else 0), path, 380 * (t - 1), 220, t)
            # 32.4: a k3 reply kept to its end, a k6 reply rolled back to its
            # question — but one GSM8K reply in eight, which comes back otherwise
            back = t == 2 and i % 8 == 0
            k6 = path == 6
            turns.append({"turn": t, "path": path, "from": sw.TURNS[t - 2] if t > 1 else None,
                          "arms": {"A": a, "B": b}, "room": 16_384 - 900 - 300 * t,
                          "probe": {"came_back": "to its question" if k6 or back else "to its end",
                                    "as_apps": not back, "rolled_back": 9000 if k6 else 240
                                    if back else 0, "end": 1000, "start": 760,
                                    "cache_n": 760 if k6 or back else 1000},
                          "memory": {"A": "", "B": "" if rng.random() > 0.005 else "no line"}})
        rows.append({"chat": i, "turns": turns, "stopped": ""})
    (d / "chats").mkdir(exist_ok=True)
    (d / "chats" / "box.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    drift = []
    for b, mem in zip(sw.BINS, (1100, 4300, 16800, 49900)):
        for k in range(6):
            for dirn in ("k3→k6", "k6→k3"):
                runs = {}
                for run, (kl, flips) in {"A2": (0.0004, 1), "N": (0.0007, 2), "B": (0.004, 6),
                                         "F": (0.31, 140)}.items():
                    fl = sorted(rng.sample(range(512), flips))
                    runs[run] = {"memory": "", "first_kl": kl * 2, "mean_kl": kl, "flips": fl,
                                 "positions": 512, "first_flip": fl[0] if fl else None}
                drift.append({"chat": k, "bin": b, "direction": dirn, "memory": mem,
                              "ref": {"memory": ""}, "runs": runs})
    (d / "drift").mkdir(exist_ok=True)
    (d / "drift" / "box.jsonl").write_text("".join(json.dumps(r) + "\n" for r in drift))
    (d / "imports.json").write_text(json.dumps({"imports": [
        {"bundle": "switch-served-…-chats-shard-1-of-8.tar.gz", "what": "chats", "part": "1/8",
         "chats": n, "at": "2026-11-06T10-00-00", "gpu": "NVIDIA GeForce RTX 5090"},
        {"bundle": "switch-served-…-drift.tar.gz", "what": "drift", "part": None, "chats": 48,
         "at": "2026-11-06T10-05-00", "gpu": "NVIDIA GeForce RTX 5090"}]}))
    sw.summarise(d)


def open_report(page, live, width: int, scheme: str, wait: str) -> None:
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=switching-mode")
    page.wait_for_selector(wait, state="attached", timeout=30000)


def test_before_plan_s_is_imported_it_says_so(live, page):
    open_report(page, live, 1280, "light", "[data-sw-none]")
    assert "Nothing imported yet: plan S runs after plan C." in page.inner_text("[data-sw-none]")
    assert page.inner_text("[data-sw-tile='chats'] .value") == "0"
    page.goto(live["base"] + "/#tab=reports")
    page.wait_for_selector("[data-report-card='switching-mode']", timeout=30000)
    assert page.inner_text("[data-report-internal='switching-mode']") == "Internal"
    assert "nothing imported yet" in page.inner_text("[data-report-models='switching-mode']")
    assert page.errors == []


@pytest.mark.parametrize("width,scheme", SIZES)
def test_the_report_shows_both_parts_from_the_summary_kept(live, page, width, scheme):
    d = config.BENCH_ROOT / "switch"
    if not (d / sw.SUMMARY).exists() or json.loads((d / sw.SUMMARY).read_text())[
            "part2"]["chats"] == 0:
        made_up(d)
    open_report(page, live, width, scheme, "[data-sw-turns]")
    s = json.loads((d / sw.SUMMARY).read_text())
    assert page.locator("[data-sw-turn]").count() == 5
    assert page.locator("[data-sw-drift-row]").count() == 4 * 2 * 4
    assert page.inner_text("[data-sw-tile='chats'] .value") == "200"
    t4 = next(x for x in s["part2"]["turns"] if x["turn"] == 4)
    assert page.inner_text("[data-sw-a='4']") == f"{t4['arms']['A']['score']:.1f}%"
    assert page.inner_text("[data-sw-b='4']") == f"{t4['arms']['B']['score']:.1f}%"
    more = page.inner_text("[data-sw-more]")
    assert "Loops at the two switches to k6, pooled" in more and "Wrong recalls" in more
    assert "Turn 3 by question" in more
    # 32.2: each reply's round trip, and those that came back otherwise
    assert page.inner_text("[data-sw-trip='1']") == "200 of 200"
    assert page.inner_text("[data-sw-trip='2']") == "175 of 200"
    assert page.inner_text("[data-sw-trip='3']") == "200 of 200"
    assert ("Reply 2 came back otherwise 25 times: to its question 25; rolled back 240 tokens "
            "(median)") in more
    # 32.4: turn 4 by reply 2, the apps' cap, and 48k a stress test apart
    split = page.inner_text("[data-sw-split='4']")
    assert split.startswith("Turn 4 by reply 2: where it came back as written, B kept k3’s "
                            "memory alone — 17")             # of 175, those counted
    assert "where it didn’t, k6 had read turn 2 again — 25 chats" in split
    assert "the apps’ 16,384 tokens" in page.inner_text("[data-sw-cap]")
    assert page.inner_text("[data-sw-stress]") == "Stress test: beyond the apps’ 16,384-token cap"
    assert "stress test" in page.inner_text("[data-sw-drift-row='49152|k3→k6|A2']")
    assert "Not tested: chats saved to disk" in page.inner_text("[data-report-section='sw-notes']")
    assert "rec1" not in page.inner_text("body")                # no question ids or text
    assert no_sideways(page), "the page scrolls sideways"
    assert page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENS / f"switching-mode-{width}-{scheme}.png"), full_page=True)
