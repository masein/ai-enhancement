"""21c: the scorecard's page as it is (docs/prompts/phase-21c-the-scorecard-page-as-it-is.md).
The page masein publishes is served unchanged inside a shell of ours; its
four datasets are live, in the published page's shapes, and for the rows
unchanged since 14 Oct equal to what it showed then (tests/fixtures/
step-checkpoints-page: ours, others and tokens as published; loops with its
benchmark text replaced by invented words, its shape kept). The board stays
fast: the reports' stats are worked out by a process of their own, and a
request only reads what is kept.

The world: our six rows' 14 Oct scores as the board stores them, Epoch AI's
data as it was on 14 Oct (a trimmed copy, CC BY 4.0) with the card numbers
entered through the board's card entry, the 14 Oct token stats kept as the
stats worker keeps them, and invented questions and answers for the loop
examples. No model runs; nothing is fetched or graded."""

from __future__ import annotations

import json
import shutil
import time
from decimal import Decimal
from pathlib import Path

import pytest

import frontier as fb
from conftest import make_service
from service import config, db, ran_out, report_page, report_stats, reported, reports

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
PAGE_FIX = FIX / "step-checkpoints-page"
OURS = json.loads((PAGE_FIX / "ours.json").read_text(encoding="utf-8"))
OTHERS = json.loads((PAGE_FIX / "others.json").read_text(encoding="utf-8"))
TOKENS = json.loads((PAGE_FIX / "tokens.json").read_text(encoding="utf-8"))
LOOPS = json.loads((PAGE_FIX / "loops-shape.json").read_text(encoding="utf-8"))
STATS = [json.loads(x) for x in (FIX / "tokens-2026-10-14.jsonl").read_text().splitlines()]
REPORT = json.loads((reported.REPORTS_DIR / "step-checkpoints.json").read_text(encoding="utf-8"))
SPEC = {o["label"]: o for o in REPORT["ours"]}
TASK = {b["key"]: b["task"] for b in REPORT["benchmarks"]}
MMLU = "mmlupro_tiger"
SHARDS = {"hle_text_cais": 9, "mmlupro_tiger": 3, "math_l5_epoch": 4, "gpqa_diamond_epoch": 4,
          "otis_aime_epoch": 3, "arc_agi2_public": 2}
FINISHED = ["step4000", "Q4 (UD-Q4_K_XL)", "BF16", "Q2 (UD-Q2_K_XL)", "Q4, earlier run"]
OPTS = [f"invented option {c}" for c in "ABCDEFGHIJ"]
# the three loop examples' questions, invented: the file names them by id
ITEMS = [{"id": "486", "question": "Invented: what is the invented fund worth at the end?",
          "options": OPTS, "answer": "E", "category": "business"},
         {"id": "10707", "question": "Invented: which invented routine prints the invented sum?",
          "options": OPTS, "answer": "D", "category": "computer science"},
         {"id": "11648", "question": "Invented: how many bits does the invented code need?",
          "options": OPTS, "answer": "H", "category": "engineering"}]
LOOP = "\n".join(["Wait, let me re-check the invented option against the question once more."]
                 * 30)


def think(body: str, closed: bool = False, after: str = "") -> str:
    return "<think>\n" + body + ("\n</think>\n" + after if closed else "")


def results(row: Path, mid: str, task: str, v: float, se: float, answers: int,
            out: int) -> None:
    d = row / f"{task}_0shot" / "frontier"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results_2026-10-14T07-20-00.000000.json").write_text(json.dumps({
        "results": {task: {"alias": task, "acc,none": v, "acc_stderr,none": se}},
        "config": {"model": "frontier", "model_args": f"pretrained={mid},enable_thinking=True"},
        "date": time.time(),
        "frontier": {"version": "frontier-2", "answers": answers, "ran_out": out,
                     "scored_by": "code", "where": "rented GPU · RTX 5090"}}))


def samples(row: Path, task: str, runs: list[tuple[str, bool, bool | None, str, int]]) -> Path:
    from service import frontier as sf
    d = sf.task_dir(row, task)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"samples_{task}_2026-10-14T07-20-00.000000.jsonl"
    f.write_text("".join(
        json.dumps({"doc_id": k, "doc": {"id": q}, "resps": [[a]],
                    "filtered_resps": ["" if out else fb.visible(a)[-1:]],
                    "frontier": [{"epoch": 0, "tokens": t, "ran_out": out, "ok": ok,
                                  "read": None if out else fb.visible(a)[-1:]}]}) + "\n"
        for k, (q, out, ok, a, t) in enumerate(runs)), encoding="utf-8")
    return f


def registry(row: Path, complete: bool) -> None:
    """the plan L boxes the row's imports came from: every one in, or HLE's
    ninth still to come (20 of 21)"""
    every = {"box": "L1"}
    tasks = {t: {**every, "shards": n} for t, n in SHARDS.items()} | {"simpleqa_epoch": every}
    shards = {}
    if not complete:
        tasks.pop("hle_text_cais")
        shards = {"hle_text_cais": {"n": 9, "have": {str(i): {"box": f"L{i}"}
                                                     for i in range(1, 9)}}}
    (row / "frontier_imports.json").write_text(json.dumps({"imports": [], "tasks": tasks,
                                                           "shards": shards}))


def world(keep_stats: bool = True) -> None:
    """the scorecard as the board held it on 14 Oct"""
    import enter_report_cards as erc
    from test_21_reports import imported
    # Epoch AI's 14 Oct data, and the card numbers entered by masein
    imported()
    erc.main(["step-checkpoints", "--by", "masein"])
    p = fb._cache(config.BENCH_ROOT, MMLU)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"items": ITEMS, "extra": {"shots": {}}}))
    by: dict[str, list[dict]] = {}
    for r in OURS:
        by.setdefault(r["model"], []).append(r)
    for label, rs in by.items():
        row = reports.row_dir(SPEC[label]["row"])
        shutil.rmtree(row, ignore_errors=True)
        row.mkdir(parents=True)
        mid = SPEC[label]["row"].removesuffix(" · thinking")
        db.served_put({"id": mid, "name": mid.split("/")[1], "base_url": "", "key": "",
                       "how": "rented GPU", "based_on": "Qwen/Qwen3.6-35B-A3B",
                       "thinking": "on", "pin": {}, "by": "masein", "at": 0})
        for r in rs:
            if r["score"] is not None:
                # as the board stores them: shares, not yet rounded
                results(row, mid, TASK[r["bench"]], (r["score"] + 0.03) / 100,
                        (r["se"] + 0.02) / 100, r["answers"], r["ran_out_n"])
        if label != "Q4, earlier run":
            registry(row, complete=label != "step1000")
    # the stored answers: the loop examples' among them, the rest a line each
    s4, q4 = (reports.row_dir(SPEC[x]["row"]) for x in ("step4000", "Q4 (UD-Q4_K_XL)"))
    loops = {s4: [(it["id"], True, False, think(f"Counting.\nThe answer is ({it['answer']}).\n"
                                                + LOOP), 32768) for it in ITEMS],
             q4: [(it["id"], False, True, think("Short.", True, f"ANSWER: {it['answer']}"), 2500)
                  for it in ITEMS]}
    for s in STATS:
        d = config.OUT_DIR / s["row"]
        runs = loops.get(d) if s["task"] == MMLU else None
        f = samples(d, s["task"], runs or [("q1", False, True, think("x", True, "ANSWER: A"), 9)])
        if keep_stats:
            # the 14 Oct stats (~/tokens.py, equal to 21's on the same answers),
            # kept as the stats worker keeps them, for the files stored now
            report_stats.keep(reports._tok_slot(d, s["task"]), reports._tok_key(f),
                              {k: v for k, v in s.items() if k not in ("row", "task")})
    # step1000's MATH Level 5: every answer in, not scored yet
    from service import frontier as sf
    m = sf.task_dir(reports.row_dir(SPEC["step1000"]["row"]), "math_l5_epoch")
    (m / sf.ANSWERS).write_text("{}\n")


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    yield client
    client.__exit__(None, None, None)


def kinds(rows: list[dict]) -> dict[str, set[str]]:
    """each column and the JSON types it holds"""
    out: dict[str, set[str]] = {}
    for r in rows:
        for k, v in r.items():
            out.setdefault(k, set()).add("null" if v is None else "bool" if isinstance(v, bool)
                                         else "int" if isinstance(v, int) else
                                         "float" if isinstance(v, float) else type(v).__name__)
    return out


def get(svc, did: str):
    r = svc.get(f"/api/reports/step-checkpoints/datasets/{did}")
    assert r.status_code == 200, (did, r.status_code, r.text[:300])
    return r.json()


# ---------------------------------------------------------------------------
# the datasets: the published shapes, and the 14 Oct values
# ---------------------------------------------------------------------------

def test_each_dataset_has_the_published_columns_and_types(svc):
    world()
    reports.loops(REPORT)                         # the loop examples, worked out
    for did, fix in (("ours", OURS), ("others", OTHERS), ("tokens", TOKENS), ("loops", LOOPS)):
        got = get(svc, did)
        assert got and list(got[0]) == list(fix[0]), (did, list(got[0]), list(fix[0]))
        want, have = kinds(fix), kinds(got)
        for col, types in have.items():
            # a column the fixture never saw empty may be empty here, and an
            # int column may hold a float only where the fixture has floats
            assert types - {"null"} <= want[col] | ({"int"} if "float" in want[col] else set()), \
                (did, col, types, want[col])
        assert set(have) == set(want), did


def test_every_other_model_equals_the_published_row(svc):
    world()
    got = get(svc, "others")
    # 22: Ornith-1.5's card, added after the page was published — its own row
    ornith = [g for g in got if g["model"] == "Ornith-1.5-35B-A3B, its card"]
    assert ornith == [{"model": "Ornith-1.5-35B-A3B, its card", "maker": "Ornith AI",
                       "size": "35B / 3B", "gpqa": 89.2, "gpqa_se": None, "otis": None,
                       "otis_se": None, "math": None, "math_se": None, "simpleqa": None,
                       "simpleqa_se": None, "hle": 25.6, "hle_src": "lab", "mmlupro": None,
                       "source": "Ornith's model card (5-run average)"}]
    got = [g for g in got if g not in ornith]
    assert len(got) == 41 == len(OTHERS)
    for g, w in zip(got, OTHERS):
        if w["model"] == "o4-mini (high)":
            # 21d: its MMLU-Pro is the card's at medium effort, and says so
            w = {**w, "source": "Epoch AI; MMLU-Pro: Qwen card (medium effort)"}
        assert g == w, {k: (g.get(k), v) for k, v in w.items() if g.get(k) != v}


def test_our_rows_equal_the_published_ones_but_step1000s(svc):
    world()
    got = get(svc, "ours")
    # 22: Ornith-1.5's rows, after the page was published, last — not run yet
    assert {r["status"] for r in got if r["model"] == "Ornith-1.5 Q4_K_M"} == {"Not run yet"}
    got = [r for r in got if r["model"] != "Ornith-1.5 Q4_K_M"]
    assert [(r["model"], r["bench"]) for r in got] == [(r["model"], r["bench"]) for r in OURS]
    for g, w in zip(got, OURS):
        if w["model"] == "step1000":
            continue
        assert g == w, {k: (g.get(k), v) for k, v in w.items() if g.get(k) != v}
    s1 = {r["bench"]: r for r in got if r["model"] == "step1000"}
    assert s1["gpqa"]["status"] == "Running · 20 of 21 chunks in"
    assert s1["hle"]["score"] is None and s1["math"]["score"] is None      # not in: a dash


def test_the_finished_models_token_rows_equal_the_published_ones(svc):
    world()
    got = {(r["model"], r["bench"]): r for r in get(svc, "tokens")}
    want = [r for r in TOKENS if r["model"] in FINISHED]
    assert len(want) == 35
    for w in want:
        g = got[(w["model"], w["bench"])]
        assert g == w, {k: (g.get(k), v) for k, v in w.items() if g.get(k) != v}
    # step1000's MATH Level 5: every answer in, some still to be graded
    assert got[("step1000", "math")]["state"] == "all answers in, waiting for grading"


def test_one_token_row_worked_out_by_the_worker_from_stored_answers(svc):
    """the stats the page reads are the worker's: BF16's GPQA Diamond worked
    out by the worker from stored answers, read back in the page's columns as
    ~/tokens.py counts them on the same answers (21d)"""
    import token_answers as ta
    world(keep_stats=False)
    ta.write(config.OUT_DIR)
    want = ta.tokens_py(config.RESULTS_ROOT)[(
        "served__Qwen3.6-35B-A3B-BF16-lda-2610__thinking", "gpqa_diamond_epoch")]
    got = report_stats.work(say=lambda s: None)
    assert got["done"] and not got["failed"]
    row = next(r for r in get(svc, "tokens") if (r["model"], r["bench"]) == ("BF16", "gpqa"))
    assert row["answers"] == want["n"]
    assert {k: row[k] for k in report_page.TOKEN_FIELDS} == {
        k: want[k] for k in report_page.TOKEN_FIELDS}


def test_the_loop_examples_from_the_shown_half_with_the_files_notes(svc):
    world()
    reports.loops(REPORT)
    got = get(svc, "loops")
    assert [r["id"] for r in got] == ["486", "10707", "11648"]
    notes = {str(x["id"]): x["note"] for x in REPORT["loops"]["examples"]}
    for k, r in enumerate(got):
        assert r["what"] == notes[r["id"]] and r["order"] == k + 1
        assert r["benchmark"] == "MMLU-Pro" and r["q4_verdict"].startswith("read as ")
        assert r["s_verdict"] == "ran out of room before its answer"
        assert r["waits"] == 30 and r["top_count"] == 30 and r["excerpt"]
    it = {x["id"]: x for x in ITEMS}
    assert got[0]["question"] == it["486"]["question"] and got[0]["subject"] == "business"
    assert (got[0]["right"], got[0]["right_text"]) == ("E", "invented option E")
    # a question from the hidden half: never on the page
    hidden = {**REPORT, "loops": {**REPORT["loops"], "examples": [
        {"id": "1", "note": "the hidden half"}, *REPORT["loops"]["examples"]]}}
    assert not reports.shown_half(MMLU, "1")
    assert [r["id"] for r in report_page.loops(hidden, reports.loops(hidden))] == \
        ["486", "10707", "11648"]


def test_the_numbers_are_rounded_as_21s_view_rounds_them():
    # JavaScript's toFixed: the exact binary value, a tie away from zero
    assert report_page.fixed(74.25) == 74.3 and round(74.25, 1) == 74.2
    assert report_page.fixed(0.25) == 0.3 and report_page.fixed(1.005, 2) == 1.0
    assert report_page.fixed(100 * 0.742) == 74.2 and report_page.fixed(-2.25) == -2.3
    assert Decimal(1.005) < Decimal("1.005")              # why 1.005 rounds down


def test_where_each_number_comes_from_in_words():
    b = [{"key": "gpqa", "label": "GPQA Diamond"}, {"key": "hle", "label": "HLE"},
         {"key": "mmlupro", "label": "MMLU-Pro"}]
    card = lambda name: {"v": 0.5, "src": "card", "card": f"{name} model card"}   # noqa: E731
    assert report_page.source(b, {"gpqa": {"v": 0.5, "src": "epoch"}}) == "Epoch AI"
    assert report_page.source(b, {"gpqa": {"v": 0.5, "src": "epoch"},
                                  "hle": card("DeepSeek-R1-0528"),
                                  "mmlupro": card("DeepSeek-R1-0528")}) == \
        "Epoch AI; HLE, MMLU-Pro: DeepSeek card"
    assert report_page.source(b, {"gpqa": card("Qwen3.6-35B-A3B"),
                                  "mmlupro": card("Qwen3.6-35B-A3B")}) == "Qwen's model card"
    # HLE from Epoch's collection counts as Epoch AI's
    assert report_page.source(b, {"hle": {"v": 0.1, "src": "epoch_external"},
                                  "mmlupro": card("Qwen3.5-35B-A3B")}) == \
        "Epoch AI; MMLU-Pro: Qwen card"
    # a card number not entered yet isn't named
    assert report_page.source(b, {"gpqa": {"v": 0.5, "src": "epoch"},
                                  "hle": {"v": None, "src": "card", "missing": True}}) == \
        "Epoch AI"


def test_the_status_lines(svc):
    world()
    got = {r["model"]: r["status"] for r in get(svc, "ours")}
    assert got == {"step4000": "Complete · 21 of 21 chunks",
                   "step1000": "Running · 20 of 21 chunks in",
                   "Q4 (UD-Q4_K_XL)": "Complete · 21 of 21 chunks",
                   "BF16": "Complete · 21 of 21 chunks", "Q2 (UD-Q2_K_XL)": "Complete · 21 of 21 chunks",
                   "Q4, earlier run": "7–8 Oct, Qwen's card settings",
                   "Ornith-1.5 Q4_K_M": "Not run yet"}                     # 22
    # every box in, one benchmark waiting for its grader
    row = {"cells": {"a": {"state": "score"}, "b": {"state": "waits"}}, "chunks": [21, 21],
           "status": ""}
    assert report_page.status({}, row) == "Grading · 21 of 21 chunks in"


# ---------------------------------------------------------------------------
# the page itself: unchanged, in our shell, nothing from another host
# ---------------------------------------------------------------------------

def test_the_page_is_served_unchanged_in_our_shell(svc):
    page = (reported.REPORTS_DIR / "pages" / "step-checkpoints.html").read_text(encoding="utf-8")
    r = svc.get("/reports/step-checkpoints/page")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert page in r.text                                       # every byte, as published
    assert r.text.index("dash.js") < r.text.index(page) and "d3.v7.9.0.min.js" in r.text
    assert "<title>Step checkpoints frontier scorecard</title>" in r.text
    csp = r.headers["content-security-policy"]
    assert "default-src 'self'" in csp and "connect-src 'self'" in csp
    assert svc.get("/reports/nothing/page").status_code == 404
    assert svc.get("/reports/ran-out-of-room/page").status_code == 404     # no page of its own
    # the runtime, the theme and d3 from this server
    for name, kind in (("dash.js", "javascript"), ("theme.css", "css"),
                       ("d3.v7.9.0.min.js", "javascript")):
        a = svc.get(f"/reports/static/{name}")
        assert a.status_code == 200 and kind in a.headers["content-type"], name
    assert svc.get("/reports/static/../app.py").status_code == 404
    assert svc.get("/reports/static/report_page.py").status_code == 404


def test_nothing_on_the_page_comes_from_another_host():
    import re
    static = ROOT / "service" / "static" / "reports"
    page = (reported.REPORTS_DIR / "pages" / "step-checkpoints.html").read_text(encoding="utf-8")
    shell = report_page.SHELL
    for name, text in (("page", page), ("shell", shell),
                       ("dash.js", (static / "dash.js").read_text(encoding="utf-8")),
                       ("theme.css", (static / "theme.css").read_text(encoding="utf-8"))):
        assert not re.search(r"(?:src|href)\s*=\s*[\"']?(?:https?:)?//", text), name
        assert not re.search(r"@import|url\(\s*[\"']?(?:https?:)?//", text), name
        assert "fetch(\"http" not in text and "fetch('http" not in text, name
    # d3 7.9.0 as published (the hash cdnjs gives), and its licence beside it
    import base64
    import hashlib
    d3 = (static / "d3.v7.9.0.min.js").read_bytes()
    assert d3.startswith(b"// https://d3js.org v7.9.0")
    assert base64.b64encode(hashlib.sha512(d3).digest()).decode() == \
        "vc58qvvBdrDR4etbxMdlTt4GBQk1qjvyORR2nrsPsFPyrs+/u5c3+1Ct6upOgdZoIl7eq6k3a1UPDSNAQi/32A=="
    assert "Permission to use, copy, modify" in (static / "LICENSE-d3.txt").read_text()


def test_the_header_line_says_when_each_source_changed(svc):
    world()
    got = svc.get("/api/reports/step-checkpoints/datasets").json()
    words = {s["key"]: s["words"] for s in got["sources"]}
    assert words["ours"].startswith("Our scores: as of the last import, ")
    assert words["epoch"].startswith("Epoch AI: imported ")
    assert words["cards"].startswith("Model cards: last entered ")
    assert words["stats"].startswith("Thinking tokens and loop examples: ")
    assert got["credit"]["words"] == "Data: Epoch AI, CC BY 4.0"
    assert svc.get("/api/reports/nothing/datasets").status_code == 404
    assert svc.get("/api/reports/step-checkpoints/datasets/nothing").status_code == 404


def test_the_reports_list_and_the_report_name_the_full_page(svc):
    world()
    got = {r["id"]: r for r in svc.get("/api/reports").json()["reports"]}
    assert got["step-checkpoints"]["page"] == "reports/step-checkpoints/page"
    assert not got["ran-out-of-room"].get("page")
    assert svc.get("/api/reports/step-checkpoints").json()["page"] == \
        "reports/step-checkpoints/page"


# ---------------------------------------------------------------------------
# the board stays fast: a request reads what is kept, never computes
# ---------------------------------------------------------------------------

def no_computing(monkeypatch):
    def no(*a, **k):
        raise AssertionError("a request worked a stat out while the board waited")
    monkeypatch.setattr(reports, "token_stats_of", no)
    monkeypatch.setattr(reports, "loops_of", no)
    monkeypatch.setattr(ran_out, "work_out", no)


def test_a_first_open_says_working_and_never_computes(svc, monkeypatch):
    world(keep_stats=False)
    no_computing(monkeypatch)
    started = []
    monkeypatch.setattr(report_stats, "spawn", lambda: started.append(1))
    monkeypatch.setattr(config, "REPORT_STATS_SPAWN", True)
    r = svc.get("/api/reports/step-checkpoints").json()
    assert r["working"]["first"] and r["working"]["todo"] >= 40 and r["tokens"]["BF16"] == {}
    assert started == [1]                                        # the worker woken, once
    for did in ("tokens", "loops"):
        x = svc.get(f"/api/reports/step-checkpoints/datasets/{did}")
        assert x.status_code == 202 and x.json()["words"] == report_page.WORKING
    assert started == [1]                                        # not again while it starts
    info = svc.get("/api/reports/step-checkpoints/datasets").json()
    assert "working these out, back in a minute" in info["sources"][3]["words"]
    ro = svc.get("/api/reports/ran-out-of-room").json()
    assert ro["working"]["first"] and ro["models"]["step4000"]["pending"]
    assert get(svc, "ours") and get(svc, "others")             # the tables answer at once


def test_a_stat_whose_answers_changed_is_shown_with_its_time_until_the_new_one_is_in(
        svc, monkeypatch):
    world()
    reports.loops(REPORT)
    d = config.OUT_DIR / STATS[0]["row"]
    f = samples(d, STATS[0]["task"], [("q9", False, True, think("y", True, "ANSWER: B"), 7)])
    assert reports.token_kept(d, STATS[0]["task"])["fresh"] is False
    no_computing(monkeypatch)
    r = svc.get("/api/reports/step-checkpoints").json()
    assert r["tokens"]["BF16"]["gpqa_diamond_epoch"]["think_mean"] == 8370      # the kept one
    assert r["working"]["todo"] == 1 and not r["working"]["first"]
    assert r["statsAt"]["words"].startswith("worked out ")
    assert f.exists()


def test_the_board_stays_fast_while_the_stats_are_worked_out(svc, monkeypatch):
    """the worker in a process of its own, at the lowest priority: /healthz
    under a second while it works; then the report and each dataset under a
    second, read from what it kept"""
    world(keep_stats=False)
    # a benchmark's worth of answers, so the work takes a while
    big = think("Let me think about the invented question.\n" * 40 + LOOP, True, "ANSWER: C")
    samples(reports.row_dir(SPEC["step1000"]["row"]), MMLU,
            [(f"b{i}", i % 25 == 0, i % 3 == 0, big, 4000 + i % 900) for i in range(12_032)])
    no_computing(monkeypatch)
    monkeypatch.setattr(config, "REPORT_STATS_SPAWN", True)
    t0 = time.time()
    assert report_stats.wake()
    seen, slow = 0, []
    while time.time() - t0 < 180:
        if report_stats.running():
            t = time.time()
            assert svc.get("/healthz").status_code == 200
            slow.append(time.time() - t)
            seen += 1
        elif seen and not report_stats.stale():
            break
        time.sleep(0.05)
    took = time.time() - t0
    print(f"the worker took {took:.1f} s; /healthz asked {seen} times while it worked, "
          f"the slowest {max(slow or [0]):.3f} s")
    assert seen >= 3, (seen, took)                      # asked while the worker worked
    assert max(slow) < 1.0, max(slow)
    assert not report_stats.stale() and not report_stats.running(), took
    log = (report_stats.folder() / "stats.log").read_text()
    assert "worked out in" in log and "couldn't be worked out" not in log, log[-2000:]
    for path in ["/api/reports/step-checkpoints", "/api/reports/ran-out-of-room",
                 "/api/reports/step-checkpoints/datasets",
                 *[f"/api/reports/step-checkpoints/datasets/{d}" for d in report_page.DATASETS]]:
        t = time.time()
        r = svc.get(path)
        assert r.status_code == 200 and time.time() - t < 1.0, (path, r.status_code,
                                                                 time.time() - t)
        print(f"{path}: {time.time() - t:.3f} s")
    r = svc.get("/api/reports/step-checkpoints").json()
    assert not r["working"] and r["statsAt"]["words"].startswith("worked out ")
    assert r["tokens"]["step1000"]["mmlupro_tiger"]["n"] == 12_032


def test_a_report_without_the_hidden_flag_never_reads_a_list_kept_for_one_with_it(svc):
    world()
    from test_21b_ran_out import REP
    ran_out.rows_of(REP, "step4000", MMLU)                       # kept, for the flagged report
    blind = {k: v for k, v in REP.items() if k != "show_hidden"}
    assert ran_out._slot(REP, "step4000", MMLU) != ran_out._slot(blind, "step4000", MMLU)
    assert ran_out._cached(blind, "step4000", MMLU) is None      # never the flagged one's


def test_the_worker_is_woken_after_scoring_and_import():
    src = (ROOT / "service" / "frontier.py").read_text(encoding="utf-8")
    body = src[src.index("def score_task("):src.index("def outcome(")]
    assert "report_stats.wake_quietly()" in body
    imp = (ROOT / "scripts" / "import_frontier.py").read_text(encoding="utf-8")
    body = imp[imp.index("def import_bundle("):imp.index("def _earlier_row(")]
    assert "report_stats.wake_quietly()" in body


def test_with_no_report_answers_stored_nothing_is_woken(svc, monkeypatch):
    """an import or a scoring on a board with none of a report's answers
    starts no worker: there is nothing to work out"""
    monkeypatch.setattr(config, "REPORT_STATS_SPAWN", True)
    started = []
    monkeypatch.setattr(report_stats, "spawn", lambda: started.append(1))
    assert report_stats.jobs() == [] and report_stats.wake() is False and started == []
