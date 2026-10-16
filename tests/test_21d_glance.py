"""21d: Models at first glance, clearer numbers in the reports, and the date
filter in view (docs/prompts/phase-21d-first-glance-and-clearer-numbers.md) —
the server's half. The published page is the one masein gave; a report names
how many times each question is asked, and the page's own definitions of its
token columns; Models ▸ Latest reads the report alone (its picked rows and the
eight it marks "glance", on four benchmarks); When ▸ added reads each model's
first submission; the Reports list answers without building the board's
numbers; o4-mini's MMLU-Pro says "medium effort"; a loop example's length is
the whole output. The world is 21c's: the 14 Oct scorecard as the board would
hold it, invented questions. Nothing is fetched, run or graded."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import closing

import pytest

from conftest import make_service
from service import config, db, reported, reports
from test_21c_page import ITEMS, REPORT, SPEC, world

PAGE = reported.REPORTS_DIR / "pages" / "step-checkpoints.html"
# docs/prompts/phase-21d/step-checkpoints-page.html, as masein gave it
PAGE_SHA256 = "831d21e60515b535433d33507c35c4c05eda503bbb566b985d81c289990d0433"
GLANCE = ["GPT-6 Astra (max)", "DeepSeek-V4-Pro (max)", "Qwen3.6-35B-A3B, Qwen's report",
          "Qwen3.5-35B-A3B", "GPT-5 mini (high)", "o4-mini (high)", "gpt-oss-120b (high)",
          "DeepSeek-R1-0528"]
PICKED = ["step4000", "step1000", "Q4 (UD-Q4_K_XL)", "BF16", "Q2 (UD-Q2_K_XL)"]


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    yield client
    client.__exit__(None, None, None)


def test_the_page_is_the_one_masein_gave():
    assert hashlib.sha256(PAGE.read_bytes()).hexdigest() == PAGE_SHA256
    text = PAGE.read_text(encoding="utf-8")
    for h in ("Average thinking", "Median thinking", "Output p90", "Average output, right",
              "Average output, wrong", "Ran out", "Output total"):
        assert f">{h}</th>" in text, h


def test_a_report_says_how_many_times_each_question_is_asked(svc):
    world()
    got = svc.get("/api/reports/step-checkpoints").json()
    asked = {b["key"]: (b["epochs"], b["unit"]) for b in got["benchmarks"]}
    assert asked == {"gpqa": (4, "questions"), "hle": (1, "questions"), "otis": (8, "questions"),
                     "math": (1, "questions"), "mmlupro": (1, "questions"),
                     "simpleqa": (1, "questions"), "arc": (2, "tasks")}
    ro = svc.get("/api/reports/ran-out-of-room").json()
    assert {b["key"]: b["epochs"] for b in ro["benchmarks"]}["gpqa"] == 4


def test_the_token_definitions_are_the_pages_own_words(svc):
    world()
    defs = dict(svc.get("/api/reports/step-checkpoints").json()["tokenDefs"])
    assert list(defs) == ["Answers", "Average and median thinking", "Output p90",
                          "Average output, right and wrong", "Ran out", "Output total"]
    text = PAGE.read_text(encoding="utf-8")
    for dt, dd in defs.items():
        assert f"<dt>{dt}</dt><dd>{dd}</dd>" in text.replace("&quot;", '"'), dt
    assert defs["Output p90"].startswith("9 in 10 answers are this long or shorter")


def test_latest_reads_the_report_alone(svc):
    """the picked rows and the eight marked glance, on GPQA Diamond, OTIS Mock
    AIME, MMLU-Pro and HLE — read from the report file; the payload names
    the report, its rows and its benchmarks' tasks"""
    world()
    rep = json.loads((reported.REPORTS_DIR / "step-checkpoints.json").read_text())
    assert [t["label"] for t in rep["theirs"] if t.get("glance")] == [
        t for t in [x["label"] for x in rep["theirs"]] if t in GLANCE]
    assert sorted(t["label"] for t in rep["theirs"] if t.get("glance")) == sorted(GLANCE)
    got = svc.get("/api/reports/step-checkpoints").json()
    assert got["glance"] == {"benchmarks": ["gpqa", "otis", "mmlupro", "hle"]}
    assert sorted(t["label"] for t in got["theirs"] if t["glance"]) == sorted(GLANCE)
    assert [r["label"] for r in got["ours"] if r["pick"]] == PICKED
    from service import app
    meta = app.results_payload()["glance"]
    assert meta == {"report": "step-checkpoints", "title": REPORT["title"],
                    "rows": [SPEC[x]["row"] for x in PICKED],
                    "tasks": ["gpqa_diamond_epoch", "otis_aime_epoch", "mmlupro_tiger",
                              "hle_text_cais"]}


def test_o4_minis_mmlu_pro_says_medium_effort(svc):
    world()
    got = {t["label"]: t for t in svc.get("/api/reports/step-checkpoints").json()["theirs"]}
    o4 = got["o4-mini (high)"]
    assert o4["cells"]["mmlupro"]["note"] == "medium effort"
    assert o4["source"] == "Epoch AI; MMLU-Pro: Qwen card (medium effort)"
    assert got["GPT-5 mini (high)"]["source"] == "Epoch AI; MMLU-Pro: Qwen card"
    assert got["Qwen3.6-35B-A3B, Qwen's report"]["source"] == "Qwen's model card"


def test_the_reports_list_never_builds_the_boards_numbers_and_is_quick(svc, monkeypatch):
    """2.3 s on the server: the list built the board's whole payload for
    names and folders. With the stats kept, under 0.2 s"""
    world()
    reports.loops(REPORT)
    from service import app

    def no(*a, **k):
        raise AssertionError("the Reports list built the board's numbers")
    monkeypatch.setattr(app, "results_payload", no)
    svc.get("/api/reports")                                     # the files read once
    slow = 0.0
    for _ in range(5):
        t = time.time()
        r = svc.get("/api/reports")
        slow = max(slow, time.time() - t)
        assert r.status_code == 200
    got = {x["id"]: x for x in r.json()["reports"]}
    assert got["step-checkpoints"]["others"] == 41 and got["step-checkpoints"]["changed"]
    assert slow < 0.2, slow


def test_added_is_the_first_submission(svc):
    """When ▸ added: a model's first submission to the board, whatever came of
    it; a thinking row's are its model's"""
    world()
    mid = SPEC["step4000"]["row"].removesuffix(" · thinking")
    first = db.add(mid, "instruct", "frontier", "masein", "imported", status="done")
    later = db.add(mid, "instruct", "frontier", "masein", "imported", status="done")
    with closing(db._conn()) as c:
        c.execute("UPDATE submissions SET created_at=? WHERE id=?", (time.time() - 40 * 864e2,
                                                                      first))
        c.execute("UPDATE submissions SET created_at=? WHERE id=?", (time.time() - 864e2, later))
        c.commit()
    added = db.first_added()
    assert abs(added[mid] - (time.time() - 40 * 864e2)) < 5
    from service import app
    app._cache.update(payload=None, key=None)
    m = next(x for x in app.results_payload()["models"] if x["id"] == SPEC["step4000"]["row"])
    assert abs(m["addedAt"] - added[mid]) < 1e-6


def test_a_loop_examples_length_is_its_thinking_alone(svc):
    """21d.1: the page's "thought for N characters" — the thinking alone, not
    the whole output (on the server, Q4 on 486: 6,485 of 7,247), for both
    models; the report's own view says the same"""
    import frontier as fb
    world()
    reports.loops(REPORT)
    thought = {}
    for label in ("Q4 (UD-Q4_K_XL)", "step4000"):
        f = reports.samples_file(reports.row_dir(SPEC[label]["row"]), "mmlupro_tiger")
        for _, row in reports.jsonl(f):
            a = row["resps"][0][0]
            thought[(label, str(row["doc"]["id"]))] = (len(fb.thinking_of(a)), len(a))
    got = {x["id"]: x for x in svc.get("/api/reports/step-checkpoints/datasets/loops").json()}
    for it in ITEMS:
        q4, s4 = thought[("Q4 (UD-Q4_K_XL)", it["id"])], thought[("step4000", it["id"])]
        assert got[it["id"]]["q4_chars"] == q4[0] < q4[1], it["id"]       # less than the whole
        assert got[it["id"]]["s_chars"] == s4[0] < s4[1], it["id"]
    ex = {x["id"]: x for x in svc.get("/api/reports/step-checkpoints").json()["loops"]["examples"]}
    assert ex["486"]["against"]["thought"] == thought[("Q4 (UD-Q4_K_XL)", "486")][0]
    assert ex["486"]["against"]["chars"] == thought[("Q4 (UD-Q4_K_XL)", "486")][1]


def test_stats_kept_under_21cs_counting_are_worked_out_again(svc):
    """kept before the counting changed: the same answers, another key — stale,
    shown with their time until the new ones are in"""
    from service import report_stats
    world()
    d = reports.row_dir(SPEC["BF16"]["row"])
    f = reports.stats_file(d, "gpqa_diamond_epoch")
    report_stats.keep(reports._tok_slot(d, "gpqa_diamond_epoch"), report_stats.identity(f),
                      {"n": 1, "think_mean": 1})
    assert reports.token_kept(d, "gpqa_diamond_epoch")["fresh"] is False
    stale = {j.slot for j in report_stats.stale()}
    assert reports._tok_slot(d, "gpqa_diamond_epoch") in stale
