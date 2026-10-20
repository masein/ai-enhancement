"""23: two builds, question by question (eval_tasks/reports/paired.json,
service/paired.py) — the board's own verdicts lined up by question, the sums
of scripts/paired_stats.py, the questions left out and said, the hidden half
counted and never listed or opened, the worker keeping every pair and a page
never computing. Invented questions and answers; no model runs, nothing is
fetched or graded."""

from __future__ import annotations

import json
import random

import pytest

import frontier as fb
import paired_stats as ps
from conftest import make_service
from service import config, paired, ran_out, report_stats, reported, reports
from service import frontier as sf

REP = json.loads((reported.REPORTS_DIR / "step-checkpoints.json").read_text())
S4, Q4, EARLY, ORN = "step4000", "Q4 (UD-Q4_K_XL)", "Q4, earlier run", "Ornith-1.5 Q4_K_M"
GPQA, OTIS, SQA, MMLU, ARC = ("gpqa_diamond_epoch", "otis_aime_epoch", "simpleqa_epoch",
                              "mmlupro_tiger", "arc_agi2_public")
OPTS = [f"invented option {c}" for c in "ABCDEFGHIJ"]
GPQA_Q = "An invented question about an imagined particle's spin in an invented box?"


def mmlu_ids() -> list[str]:
    return [str(1000 + k) for k in range(50)]


def think(body: str, after: str = "ANSWER: A") -> str:
    return f"<think>\n{body}\n</think>\n\n{after}"


def row_dir(label: str):
    return reports.row_dir(next(o["row"] for o in REP["ours"] if o["label"] == label))


def write(label: str, task: str, rows: list[dict], runtime: str = "lda-2610",
          family: str = "qwen3.6") -> None:
    """a build's scored samples: each row {id, ok: [..], out: [..], group?,
    ungraded?: [..]}, and its setup.json"""
    d = sf.task_dir(row_dir(label), task)
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for k, r in enumerate(rows):
        oks = r["ok"]
        outs = r.get("out") or [False] * len(oks)
        ung = r.get("ungraded") or [False] * len(oks)
        body = GPQA_Q + "\nLet me think." if task == GPQA else f"thinking about {r['id']}"
        lines.append(json.dumps({
            "doc_id": k, "doc": {"id": r["id"], **({"group": r["group"]} if r.get("group")
                                                   else {})},
            "resps": [[think(body)] for _ in oks], "filtered_resps": ["A"] * len(oks),
            "frontier": [{"epoch": e, "ok": ok, "ran_out": out, "tokens": 100,
                          **({"ungraded": True} if u else {})}
                         for e, (ok, out, u) in enumerate(zip(oks, outs, ung))]}))
    (d / f"samples_{task}_2026-10-19T08-00-00.000000.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    (d / sf.SETUP).write_text(json.dumps({
        "task": task, "family": family, "sampling": {"temperature": 1.0, "rt": runtime or family},
        **({"runtime": runtime} if runtime else {})}), encoding="utf-8")


def runs(rng: random.Random, n: int, k: int, p: float) -> list[dict]:
    return [{"id": f"q{i}", "ok": [rng.random() < p for _ in range(k)]} for i in range(n)]


def arc_rows(rng: random.Random, p: float) -> list[dict]:
    return [{"id": f"t{t}#{g}", "group": f"t{t}", "ok": [rng.random() < p, rng.random() < p]}
            for t in range(12) for g in range(1 + t % 3)]


def mmlu_rows(a_better: int, b_better: int, n: int, flip: bool = False) -> list[dict]:
    """n questions: the first a_better right for A only, the next b_better
    right for B only, the rest the same for both (half right)"""
    out = []
    for k, q in enumerate(mmlu_ids()[:n]):
        a = k < a_better or (k >= a_better + b_better and k % 2 == 0)
        b = a_better <= k < a_better + b_better or (k >= a_better + b_better and k % 2 == 0)
        out.append({"id": q, "ok": [b if flip else a]})
    return out


def world() -> None:
    items = {MMLU: [{"id": q, "question": f"Invented MMLU-Pro question {q}?", "options": OPTS,
                     "answer": "A", "category": "math"} for q in mmlu_ids()],
             GPQA: [{"id": f"q{i}", "question": GPQA_Q, "right": "Spin one half",
                     "wrong": ["Spin zero", "Spin one", "Spin three halves"]} for i in range(30)]}
    for task, its in items.items():
        p = fb._cache(config.BENCH_ROOT, task)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"items": its, "extra": {"shots": {}}}))
    rng = random.Random(23)
    for label, p, rt, fam in ((S4, 0.6, "lda-2610", "qwen3.6"), (Q4, 0.5, "lda-2610", "qwen3.6"),
                              (EARLY, 0.55, "", "qwen3.6"), (ORN, 0.7, "card-ornith", "")):
        write(label, GPQA, runs(rng, 30, 4, p), rt, fam)
        write(label, OTIS, runs(rng, 9, 8, p), rt, fam)
        write(label, SQA, runs(rng, 60, 1, p), rt, fam)
        write(label, ARC, arc_rows(rng, p), rt, fam)
    write(S4, MMLU, mmlu_rows(9, 1, 50))
    write(Q4, MMLU, mmlu_rows(9, 1, 50, flip=True))


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    paired._tables.clear()
    world()
    yield client
    client.__exit__(None, None, None)


def got(a: str = S4, b: str = Q4) -> dict:
    """the report for a pair, its stats worked out now"""
    return reports.report("paired", compute=True, a=a, b=b)


def row(d: dict, key: str) -> dict:
    return next(r for r in d["rows"] if r["key"] == key)


def board(label: str, task: str) -> dict:
    """the board's score from the same samples, as summary() makes it"""
    t = paired.table_of(reports.samples_file(row_dir(label), task), task)
    per = {q: [1.0 if x else 0.0 for x in v["ok"]] for q, v in t.items()}
    return fb.summary(per, task, {q: v["group"] for q, v in t.items()})


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------

def test_the_file_reads_its_builds_from_the_scorecard():
    rep = reports.files()["paired"]
    assert rep["kind"] == "paired" and rep["builds"] == {"from": "step-checkpoints"}
    assert [o["label"] for o in rep["ours"]] == [o["label"] for o in REP["ours"]]
    # 23b: and step8804, first — eight builds, 28 pairs, opening on step8804 and step4000
    assert EARLY in paired.labels(rep) and paired.labels(rep)[0] == "step8804"
    assert len(paired.pairs(rep)) == 28
    assert rep["default"] == ["step8804", S4] and not rep.get("show_hidden")
    assert [b["key"] for b in rep["benchmarks"]] == [b["key"] for b in REP["benchmarks"]]
    text = (reported.REPORTS_DIR / "paired.json").read_text()
    assert "epoch_version" not in text and "card" not in json.loads(text)["benchmarks"][0]


# ---------------------------------------------------------------------------
# the sums against the board
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task, key", [(GPQA, "gpqa"), (OTIS, "otis"), (SQA, "simpleqa"),
                                       (ARC, "arc"), (MMLU, "mmlupro")])
def test_a_minus_b_is_the_difference_of_the_two_board_scores(svc, task, key):
    x = row(got(), key)["pair"]
    want = board(S4, task)["score"] - board(Q4, task)["score"]
    assert x["d"] == pytest.approx(want, abs=1e-9)
    assert x["n"] == board(S4, task)["n"] == x["of"]           # every question both answered


def test_one_run_counts_and_the_exact_mcnemar_verdict(svc):
    x = row(got(), "mmlupro")["pair"]
    assert (x["a_better"], x["b_better"], x["n"]) == (9, 1, 50)
    assert x["p"] == pytest.approx(22 / 1024) and x["verdict"] == "A higher"
    assert x["d"] == pytest.approx(8 / 50) and x["differ"] == pytest.approx(10 / 50)
    assert x["se"] == pytest.approx(((10 - 50 * (8 / 50) ** 2) / 49) ** 0.5 / 50 ** 0.5)


def test_several_runs_average_each_question_and_say_how_often_each_disagrees(svc):
    x = row(got(), "gpqa")["pair"]
    ta = paired.table_of(reports.samples_file(row_dir(S4), GPQA), GPQA)
    tb = paired.table_of(reports.samples_file(row_dir(Q4), GPQA), GPQA)
    sa = [sum(ta[q]["ok"]) / 4 for q in sorted(ta)]
    sb = [sum(tb[q]["ok"]) / 4 for q in sorted(ta)]
    want = ps.several_runs(sa, sb)
    assert x["d"] == pytest.approx(want["d"]) and x["se"] == pytest.approx(want["se"])
    assert x["a_better"] == sum(1 for p, q in zip(sa, sb) if p > q)
    selfa = sum(ps.self_disagree(4, sum(ta[q]["ok"])) for q in sorted(ta)) / 30
    assert x["self"]["a"] == pytest.approx(selfa)
    cross = sum(ps.cross_disagree(p, q) for p, q in zip(sa, sb)) / 30
    assert x["self"]["cross"] == pytest.approx(cross)
    assert row(got(), "simpleqa")["pair"]["self"] is None          # one run: none


def test_the_other_order_is_the_same_numbers_flipped(svc):
    ab, ba = got(S4, Q4), got(Q4, S4)
    for k in ("gpqa", "mmlupro", "arc"):
        x, y = row(ab, k)["pair"], row(ba, k)["pair"]
        assert y["d"] == pytest.approx(-x["d"]) and y["lo"] == pytest.approx(-x["hi"])
        assert (y["a_better"], y["b_better"]) == (x["b_better"], x["a_better"])
        assert y["lists"]["a"] == x["lists"]["b"]
    assert row(ba, "mmlupro")["pair"]["verdict"] == "B higher"
    # kept once: no slot for the other order
    assert report_stats.kept(paired._slot(reports.files()["paired"], Q4, S4, MMLU)) is None


# ---------------------------------------------------------------------------
# questions left out, and the settings
# ---------------------------------------------------------------------------

def test_questions_one_build_lacks_or_not_graded_are_left_out_and_counted(svc):
    rows = runs(random.Random(5), 30, 4, 0.5)
    rows[3]["ok"][1] = None                                     # not graded yet
    rows[4]["ungraded"] = [False, False, True, False]           # its grader gave none
    del rows[5]                                                 # never answered
    write(Q4, GPQA, rows)
    x = row(got(), "gpqa")["pair"]
    assert (x["n"], x["of"], x["skip_b"], x["only_a"], x["skip_a"]) == (27, 30, 2, 1, 0)


def test_the_settings_line_says_when_two_ran_another_way(svc):
    assert got()["settings"] == ""
    assert got(ORN, S4)["settings"] == ("Ornith-1.5 Q4_K_M ran on its own card's settings; "
                                        "step4000 on the lda runtime.")
    assert got(EARLY, Q4)["settings"] == ("Q4, earlier run ran on Qwen's card settings; "
                                          "Q4 (UD-Q4_K_XL) on the lda runtime.")


def test_the_sentence_is_the_verdicts(svc):
    d = got()
    verdicts = [r["pair"]["verdict"] for r in d["rows"] if r["pair"] and r["pair"]["n"]]
    up, down = verdicts.count("A higher"), verdicts.count("B higher")
    assert d["sentence"] == (f"step4000 is higher than Q4 (UD-Q4_K_XL) on {up} benchmark"
                             f"{'s' if up != 1 else ''}, lower on {down}, and the two can't be "
                             f"told apart on {len(verdicts) - up - down}.")
    # HLE and MATH: no answers from either here
    assert row(d, "hle")["missing"] and row(d, "hle")["pair"] is None


# ---------------------------------------------------------------------------
# the hidden half
# ---------------------------------------------------------------------------

def test_the_hidden_half_is_counted_never_listed_or_opened(svc):
    x = row(got(), "mmlupro")["pair"]
    ids = mmlu_ids()[:9]
    shown = [q for q in ids if reports.shown_half(MMLU, q)]
    assert 0 < len(shown) < 9                                   # both halves among them
    assert x["a_better"] == 9 and [e["id"] for e in x["lists"]["a"]] == shown
    assert x["held"]["a"] == 9 - len(shown)
    rep = reports.files()["paired"]
    lst = paired.question_list(rep, S4, Q4, MMLU, "a")
    assert (lst["n"], lst["listed"], lst["hidden"]) == (9, len(shown), 9 - len(shown))
    hidden = next(q for q in ids if not reports.shown_half(MMLU, q))
    with pytest.raises(paired.Refused):
        paired.one(rep, S4, MMLU, hidden, 0)


def test_an_unlisted_benchmark_never_shows_its_question(svc):
    got()
    rep = reports.files()["paired"]
    q = next(f"q{i}" for i in range(30) if reports.shown_half(GPQA, f"q{i}"))
    a = paired.one(rep, S4, GPQA, q, 0)
    assert a["question"] is None and a["withheld"]
    assert GPQA_Q not in a["thinking"] and ran_out.WITHHELD_LINE in a["thinking"]
    m = paired.one(rep, S4, MMLU, next(x for x in mmlu_ids() if reports.shown_half(MMLU, x)), 0)
    assert m["question"]["q"].startswith("Invented MMLU-Pro question")


# ---------------------------------------------------------------------------
# the worker, and the page's routes
# ---------------------------------------------------------------------------

def test_the_worker_keeps_every_pair_and_a_request_never_computes(svc, monkeypatch):
    def no(*a, **k):
        raise AssertionError("a request worked a pair out")
    real = paired.work_out
    monkeypatch.setattr(paired, "work_out", no)
    monkeypatch.setattr(report_stats, "spawn", lambda: None)
    r = svc.get("/api/reports/paired").json()
    assert r["a"] == S4 and r["b"] == Q4
    assert all(x["pair"] is None for x in r["rows"]) and r["working"]["first"]
    assert "GPQA Diamond" in r["pending"]
    monkeypatch.setattr(paired, "work_out", real)
    paired._tables.clear()
    done = report_stats.work(say=lambda s: None)
    assert done["done"] and not done["failed"]
    rep = reports.files()["paired"]
    have = [x for x in paired.labels(rep) if x in paired.offered(rep)]
    assert len(have) == 4
    for i, a in enumerate(have):
        for b in have[i + 1:]:
            assert report_stats.kept(paired._slot(rep, a, b, GPQA))
    r = svc.get("/api/reports/paired", params={"a": Q4, "b": S4}).json()
    assert r["a"] == Q4 and row(r, "mmlupro")["pair"]["verdict"] == "B higher"


def test_the_routes(svc):
    got()
    assert svc.get("/api/reports/paired", params={"a": S4, "b": S4}).status_code == 400
    assert svc.get("/api/reports/paired", params={"a": "nobody", "b": S4}).status_code == 404
    lst = svc.get("/api/reports/paired", params={"a": S4, "b": Q4, "bench": "mmlupro",
                                                 "side": "a"}).json()
    assert lst["n"] == 9 and all(set(x) >= {"id", "a", "b"} for x in lst["items"])
    q = lst["items"][0]["id"]
    a = svc.get("/api/reports/paired/answer", params={"model": S4, "task": MMLU, "id": q})
    assert a.status_code == 200 and a.json()["ok"] is True
    hidden = next(x for x in mmlu_ids() if not reports.shown_half(MMLU, x))
    assert svc.get("/api/reports/paired/answer", params={"model": S4, "task": MMLU,
                                                         "id": hidden}).status_code == 403
    cards = {c["id"]: c for c in svc.get("/api/reports").json()["reports"]}
    assert cards["paired"]["title"] == "Two builds, question by question"
    assert cards["paired"]["what"] == ("Which questions one build gets right and the other "
                                       "gets wrong.")
    # the scorecard names it, for its link under Our runs
    from service import app
    assert reports.report("step-checkpoints", app.results_payload())["paired"] == "paired"
    # only builds with scored answers are offered
    assert set(svc.get("/api/reports/paired").json()["offered"]) == {S4, Q4, EARLY, ORN}
