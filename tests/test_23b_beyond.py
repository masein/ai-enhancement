"""23b: beyond the score (service/beyond.py) — reliability, CAIS's
calibration error, SimpleQA's tries and F, and loops by the board's rule,
from stored answers and grades, worked out by the stats worker and read from
what it kept; step8804 into the scorecard. Invented answers; nothing is
fetched, run or graded."""

from __future__ import annotations

import json
import random

import pytest

import frontier as fb
from conftest import make_service
from service import beyond, config, paired, report_stats, reported, reports
from service import frontier as sf

REP = json.loads((reported.REPORTS_DIR / "step-checkpoints.json").read_text())
GPQA, OTIS, HLE, SQA = beyond.GPQA, beyond.OTIS, beyond.HLE, beyond.SQA
S4 = "step4000"
LOOP = "\n".join(["Wait, let me check that step once more before answering it."] * 120)
# CAIS's calib_err, run once with numpy 2.4.3 on the fixtures below (its
# file at a59048963171): the mixed 250, all 250 at 100 and wrong, and 60
CAIS = {"mixed": 4.0, "all100": 63.0, "few": "IndexError"}


def fixtures() -> dict:
    """as the numbers above were worked out with CAIS's code"""
    rng = random.Random(23)
    conf = [rng.randint(0, 49) for _ in range(100)] + [rng.randint(50, 100) for _ in range(150)]
    rng.shuffle(conf)
    ok = [rng.random() < c / 130 for c in conf]
    return {"mixed": (conf, ok), "all100": ([100] * 250, [False] * 250),
            "few": ([rng.randint(0, 100) for _ in range(60)], [True] * 60)}


# ---------------------------------------------------------------------------
# calibration: the port against CAIS's numbers
# ---------------------------------------------------------------------------

def test_the_calibration_error_is_caiss():
    for name, (conf, ok) in fixtures().items():
        got = beyond.calibration_error(conf, ok)
        if CAIS[name] == "IndexError":
            assert got is None                          # CAIS's raises: fewer than 100
            with pytest.raises(IndexError):
                beyond.calib_err([c / 100 for c in conf], ok)
        else:
            assert got == pytest.approx(CAIS[name]), name
    # the quirks kept: the last bin never counted — 100 to 199 answers give 0
    assert beyond.calibration_error([100] * 150, [False] * 150) == 0


def test_against_caiss_numpy_code_where_numpy_is_here():
    np = pytest.importorskip("numpy")

    def calib_err(confidence, correct, p='2', beta=100):     # CAIS's, as it is
        idxs = np.argsort(confidence)
        confidence = confidence[idxs]
        correct = correct[idxs]
        bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
        bins[-1] = [bins[-1][0], len(confidence)]
        cerr = 0
        total_examples = len(confidence)
        for i in range(len(bins) - 1):
            bin_confidence = confidence[bins[i][0]:bins[i][1]]
            bin_correct = correct[bins[i][0]:bins[i][1]]
            num_examples_in_bin = len(bin_confidence)
            if num_examples_in_bin > 0:
                difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))
                cerr += num_examples_in_bin / total_examples * np.square(difference)
        return np.sqrt(cerr)
    for name, (conf, ok) in fixtures().items():
        if CAIS[name] == "IndexError":
            continue
        want = 100 * round(calib_err(np.array(conf) / 100, np.array(ok)), 2)
        assert beyond.calibration_error(conf, ok) == pytest.approx(float(want)), name


def hle_rows(n: int, rng: random.Random) -> list[dict]:
    rows = []
    for i in range(n):
        c = rng.randint(0, 100)
        stated = i % 7 != 0
        text = ("<think>\nhmm\n</think>\n\nExplanation: x\nExact Answer: y"
                + (f"\nConfidence: {c}%" if stated else ""))
        rows.append({"runs": [{"epoch": 0, "ok": rng.random() < 0.2, "ran_out": False,
                               "grade": {"ok": True, "confidence": 100 if not stated else c}}],
                     "texts": [text]})
    rows.append({"runs": [{"epoch": 0, "ok": False, "ran_out": True}], "texts": ["<think>" + LOOP]})
    rows.append({"runs": [{"epoch": 0, "ok": None, "ran_out": False}], "texts": ["x"]})
    return rows


def test_calibration_on_hles_answers():
    rng = random.Random(5)
    rows = hle_rows(260, rng)
    got = beyond.calibration(rows)
    assert got["n"] == 260 and got["ran_out"] == 1 and got["waiting"] == 1
    assert sum(b["n"] for b in got["bins"]) == 260                  # the bins hold every one
    assert got["unstated"] == sum(1 for i in range(260) if i % 7 == 0)
    conf = [r["runs"][0]["grade"]["confidence"] for r in rows[:260]]
    ok = [r["runs"][0]["ok"] for r in rows[:260]]
    assert got["error"] == beyond.calibration_error(conf, ok)
    assert got["mean_confidence"] == pytest.approx(sum(conf) / 260)
    assert got["right"] == pytest.approx(sum(ok) / 260)
    assert [b["lo"] for b in got["bins"]] == list(range(0, 100, 10))
    assert got["bins"][-1]["hi"] == 100


# ---------------------------------------------------------------------------
# reliability, SimpleQA, loops
# ---------------------------------------------------------------------------

def test_reliability_always_sometimes_never():
    rows = [{"runs": [{"ok": x} for x in oks], "texts": [""] * 4}
            for oks in ([True] * 4, [True, False, True, False], [False] * 4,
                        [True, None, True, True])]
    got = beyond.reliability(rows)
    assert (got["always"], got["sometimes"], got["never"]) == (1, 1, 1)
    assert got["n"] == 3 and got["left_out"] == 1 and got["runs"] == 4
    rows[0]["runs"][0]["ungraded"] = True                     # its grader gave none
    assert beyond.reliability(rows)["left_out"] == 2


def test_simpleqa_tries_right_when_it_tries_and_f():
    def run(letter=None, out=False):
        return {"runs": [{"ok": letter == "A" if letter else False, "ran_out": out,
                          **({"grade": {"grade": letter}} if letter else {})}], "texts": [""]}
    rows = ([run("A")] * 70 + [run("B")] * 860 + [run("C")] * 65 + [run(out=True)] * 5)
    got = beyond.simpleqa(rows)
    assert (got["right"], got["wrong"], got["not_attempted"], got["ran_out"]) == (70, 860, 65, 5)
    assert got["tries"] == pytest.approx(0.930)
    assert got["right_when_tries"] == pytest.approx(70 / 930) and round(
        100 * got["right_when_tries"], 1) == 7.5
    assert got["f"] == pytest.approx(0.072539, abs=1e-6)
    assert got["right_share"] == pytest.approx(0.070)          # the board's score, unchanged


def test_loops_by_the_boards_rule():
    texts = ["<think>" + LOOP, "<think>clean\n</think>\nANSWER: A", "x" * 5000 + LOOP,
             "<think>" + "\n".join(f"step {i}: a new idea, {i * 7}" for i in range(400))]
    outs = [True, False, False, True]
    rows = [{"runs": [{"ran_out": o}], "texts": [t]} for t, o in zip(texts, outs)]
    got = beyond.loops(rows)
    assert 0 < got["looped"] < len(texts)                       # looped and clean both
    assert got["looped"] == sum(fb.ends_in_loop(t) for t in texts)
    assert got["ran_out"] == 2
    assert got["ran_out_looped"] == sum(fb.ends_in_loop(t) for t, o in zip(texts, outs) if o)


# ---------------------------------------------------------------------------
# the worker, the route, and step8804
# ---------------------------------------------------------------------------

@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    paired._tables.clear()
    yield client
    client.__exit__(None, None, None)


def write(label: str, task: str, rows: list[dict]) -> None:
    mid = next(o["row"] for o in REP["ours"] if o["label"] == label)
    d = sf.task_dir(reports.row_dir(mid), task)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"samples_{task}_2026-10-20T08-00-00.000000.jsonl").write_text("".join(
        json.dumps({"doc_id": k, "doc": {"id": f"q{k}"}, "resps": [[t] for t in r["texts"]],
                    "filtered_resps": [""] * len(r["texts"]), "frontier": r["runs"]}) + "\n"
        for k, r in enumerate(rows)))


def test_the_worker_keeps_them_and_the_route_never_computes(svc, monkeypatch):
    write(S4, HLE, hle_rows(150, random.Random(1)))
    write(S4, GPQA, [{"runs": [{"ok": k % 3 == 0} for _ in range(4)], "texts": [""] * 4}
                     for k in range(30)])
    mid = next(o["row"] for o in REP["ours"] if o["label"] == S4)

    def no(*a, **k):
        raise AssertionError("a request worked the measures out")
    real = beyond.measures
    monkeypatch.setattr(beyond, "measures", no)
    monkeypatch.setattr(report_stats, "spawn", lambda: None)
    got = svc.get("/api/beyond", params={"model": mid}).json()
    assert got["benches"] == {} and set(got["has"]) == {GPQA, HLE}
    assert got["working"]["first"]
    monkeypatch.setattr(beyond, "measures", real)
    assert report_stats.work(say=lambda s: None)["failed"] == 0
    got = svc.get("/api/beyond", params={"model": mid}).json()
    assert got["benches"][HLE]["calibration"]["n"] == 150
    assert got["benches"][GPQA]["reliability"] == {
        "n": 30, "left_out": 0, "runs": 4, "always": 10, "never": 20, "sometimes": 0,
        "always_share": pytest.approx(1 / 3), "never_share": pytest.approx(2 / 3),
        "sometimes_share": 0.0}
    # the scorecard reads the same
    from service import app
    r = reports.report("step-checkpoints", app.results_payload())
    assert r["beyond"][S4]["hle"]["calibration"]["n"] == 150


def test_step8804_joins_the_scorecard_first_and_the_paired_report_once_scored(svc):
    rep = reports.files()["step-checkpoints"]
    o = rep["ours"][0]
    assert o == {"label": "step8804", "row": "served/step8804-lda-2610 · thinking", "colour": 6,
                 "pick": True, "glance": True}
    from service import app
    r = reports.report("step-checkpoints", app.results_payload())
    row = next(x for x in r["ours"] if x["label"] == "step8804")
    assert row["status"] == "Not scored yet" and all(c["v"] is None for c in row["cells"].values())
    assert r["picks"][0] == "step8804"
    pr = reports.files()["paired"]
    assert pr["default"] == ["step8804", S4] and "step8804" not in paired.offered(pr)
    write("step8804", GPQA, [{"runs": [{"ok": True} for _ in range(4)], "texts": [""] * 4}])
    assert "step8804" in paired.offered(pr)
    write(S4, GPQA, [{"runs": [{"ok": False} for _ in range(4)], "texts": [""] * 4}])
    got = reports.report("paired", compute=True)
    assert (got["a"], got["b"]) == ("step8804", S4)
