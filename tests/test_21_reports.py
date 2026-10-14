"""21: Reports — the place, and the step-checkpoints scorecard in it
(docs/prompts/phase-21-reports-step-checkpoints.md). The Epoch AI import keeps
every version a report names, each value as Epoch's data had it on 14 Oct;
the card cells go through the board's own card entry; the sums (the "level"
band, the higher and lower counts, the summary line, the token line); the
token stats equal masein's ~/tokens.py on stored answers built to match two
of its rows; a loop example from the hidden half is refused; each of our
rows' status line. Epoch's data is a trimmed copy of 14 Oct's (CC BY 4.0);
nothing is fetched, no model runs, nothing is graded."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from conftest import make_service
from service import config, db, reported, reports

FIX = Path(__file__).resolve().parent / "fixtures"
EPOCH = FIX / "epoch-2026-10-14"
TOKENS = [json.loads(x) for x in (FIX / "tokens-2026-10-14.jsonl").read_text().splitlines()]
REPORT = json.loads((reported.REPORTS_DIR / "step-checkpoints.json").read_text())


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    monkeypatch.setattr(config, "REPORTED_MAKERS", ["OpenAI", "Google", "Anthropic"])
    monkeypatch.setattr(config, "REPORTED_PER_MAKER", 10)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    yield client
    client.__exit__(None, None, None)


def epoch_zip(edit=None) -> bytes:
    """the 14 Oct files as Epoch ships them, in a zip"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(EPOCH.glob("*.csv")):
            text = f.read_text(encoding="utf-8")
            z.writestr(zipfile.ZipInfo(f.name, date_time=(2026, 10, 14, 0, 0, 0)),
                       edit(f.name, text) if edit else text)
    return buf.getvalue()


def imported(edit=None) -> dict:
    blob = epoch_zip(edit)
    return reported.import_epoch(fetch=lambda url, headers=None, timeout=60: blob, board=[])


# ---------------------------------------------------------------------------
# the import: every named version kept, every value as on 14 Oct
# ---------------------------------------------------------------------------

def test_every_version_the_report_names_is_kept_at_its_14_oct_value(svc):
    assert imported()["status"] == "imported"
    kept = {r["version"] for r in db.reported_scores() if r.get("source") == "epoch"}
    named = reported.report_versions()
    assert named and named <= kept, sorted(named - kept)       # 518e4a1: most dropped
    rows = {r["label"]: r for r in reports.their_rows(REPORT)}
    checked = 0
    for t in REPORT["theirs"]:
        for k, (v, se) in (t.get("check_14oct") or {}).items():
            c = rows[t["label"]]["cells"][k]
            assert c["src"] == "epoch" and c["setting"] == "Epoch AI's own run"
            assert (round(100 * c["v"], 1), round(100 * (c["se"] or 0), 1)) == (v, se), \
                (t["label"], k)
            checked += 1
        h = t.get("hle") or {}
        if h.get("from") == "epoch_external":
            c = rows[t["label"]]["cells"]["hle"]
            assert round(100 * c["v"], 1) == h["value_14oct"] and c["dagger"], t["label"]
            assert c["setting"].startswith("from ")
            checked += 1
    assert checked >= 120


def test_an_unchanged_file_is_imported_again_when_a_report_names_more(svc):
    # the file as an import before the report kept it: none of the named versions
    blob = epoch_zip()
    fetch = lambda url, headers=None, timeout=60: blob                  # noqa: E731
    assert reported.import_epoch(fetch=fetch, board=[], named=set())["status"] == "imported"
    assert reported.import_epoch(fetch=fetch, board=[], named=set())["status"] == "unchanged"
    # the same file, a report naming versions it left out: imported again
    assert reported.import_epoch(fetch=fetch, board=[])["status"] == "imported"
    assert reported.report_versions() <= {r["version"] for r in db.reported_scores()
                                         if r.get("source") == "epoch"}
    assert reported.import_epoch(fetch=fetch, board=[])["status"] == "unchanged"


def test_the_token_script_sets_the_stats_beside_tokens_py(svc, capsys, monkeypatch):
    import report_token_stats as rts
    row = GPQA
    runs = build(LAYOUT["gpqa"], row)
    from service import frontier as sf
    d = config.OUT_DIR / row["row"]
    td = sf.task_dir(d, row["task"])
    td.mkdir(parents=True, exist_ok=True)
    (td / f"samples_{row['task']}_2026-10-14T07-20-00.000000.jsonl").write_text("".join(
        json.dumps({"doc_id": k, "doc": {"id": f"q{k}"}, "resps": [[a]],
                    "frontier": [{"epoch": 0, "tokens": t, "ran_out": out, "ok": ok}]}) + "\n"
        for k, (t, out, ok, a) in enumerate(runs)))
    f = FIX / "tokens-2026-10-14.jsonl"
    assert rts.main(["step-checkpoints", "--against", str(f)]) == 0
    out = capsys.readouterr().out
    assert f"{row['row']} · gpqa_diamond_epoch: same" in out and out.rstrip().endswith(
        "1 the same, 0 different")


def test_versions_kept_for_a_report_are_never_rows_of_outside_data(svc):
    # gpt-oss-20b at high too, scoring lower: the import's one-effort rule
    # would keep high alone, and the report names medium
    def high(name, text):
        if name not in ("gpqa_diamond.csv", "model_metadata.csv"):
            return text
        import csv
        rows = list(csv.DictReader(io.StringIO(text)))
        col = "Model version" if name == "gpqa_diamond.csv" else "model_version"
        row = dict(next(r for r in rows if r[col] == "gpt-oss-20b_medium"))
        row[col] = "gpt-oss-20b_high"
        if name == "gpqa_diamond.csv":
            row.update({"mean_score": "0.55", "Best score (across scorers)": "0.55"})
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([*rows, row])
        return out.getvalue()
    imported(high)
    every = db.reported_scores()
    only = {r["version"] for r in every if r.get("report_only")}
    assert "gpt-oss-20b_medium" in only                        # below its highest effort
    assert "DeepSeek-R1-0528" in only                          # a maker past the import's list
    assert len(reported.view(board=[])["scores"]) == len(every) - len(
        [r for r in every if r.get("report_only")])
    rows = {r["label"]: r for r in reports.their_rows(REPORT)}
    assert round(100 * rows["gpt-oss-20b (medium)"]["cells"]["gpqa"]["v"], 1) == 60.8


def test_no_artificial_analysis_number_is_ever_read(svc):
    imported()
    db.reported_import_add("aa", "x" * 64, [{"key": "openai/gpt-4.1", "name": "GPT-4.1",
                                             "maker": "OpenAI", "benchmark": "GPQA Diamond",
                                             "unit": "share", "value": 0.999,
                                             "setting": "Artificial Analysis's own run",
                                             "url": "https://artificialanalysis.ai"}])
    rows = {r["label"]: r for r in reports.their_rows(REPORT)}
    assert round(100 * rows["GPT-4.1"]["cells"]["gpqa"]["v"], 1) == 66.9   # Epoch's, not 99.9


# ---------------------------------------------------------------------------
# the cards: through the board's own card entry, with their links
# ---------------------------------------------------------------------------

def test_the_card_cells_go_through_the_card_entry_once(svc, capsys):
    import enter_report_cards as erc
    assert erc.main(["step-checkpoints", "--by", "masein"]) == 0
    cards = [r for r in db.reported_scores() if r.get("source") == "card"]
    assert len(cards) == 19 and all(c["by"] == "masein" for c in cards)
    assert all(c["url"].startswith("https://huggingface.co/") and "/blob/" in c["url"]
               for c in cards)
    assert erc.main(["step-checkpoints", "--by", "masein"]) == 0
    assert "0 card numbers entered by masein, 19 already there" in capsys.readouterr().out
    assert len([r for r in db.reported_scores() if r.get("source") == "card"]) == 19
    rows = {r["label"]: r for r in reports.their_rows(REPORT)}
    for t in REPORT["theirs"]:
        for k, spec in t.items():
            if isinstance(spec, dict) and spec.get("from") == "card":
                c = rows[t["label"]]["cells"][k]
                assert round(100 * c["v"], 1) == spec["value"] and c["url"] == spec["url"]
                assert c["src"] == "card" and c["by"] == "masein"


def test_a_card_cell_not_entered_is_empty_and_says_so(svc):
    rows = {r["label"]: r for r in reports.their_rows(REPORT)}
    c = rows["DeepSeek-V4-Pro (max)"]["cells"]["hle"]
    assert c["v"] is None and c["missing"] and "enter_report_cards" in c["why"]


# ---------------------------------------------------------------------------
# the sums
# ---------------------------------------------------------------------------

def model(label, ours=False, **cells):
    return {"id": ("ours:" if ours else "theirs:") + label, "label": label, "ours": ours,
            "cells": {k: ({"v": v[0], "se": v[1]} if isinstance(v, tuple) else {"v": v})
                      for k, v in cells.items()}}


BENCHES = [{"key": "gpqa", "task": "gpqa_diamond_epoch", "label": "GPQA Diamond",
            "compare": True},
           {"key": "otis", "task": "otis_aime_epoch", "label": "OTIS Mock AIME",
            "compare": True},
           {"key": "arc", "task": "arc_agi2_public", "label": "ARC-AGI-2", "compare": False}]


def test_level_is_two_combined_standard_errors_and_a_card_has_none():
    assert reports.band(0.03, 0.04) == pytest.approx(0.10)
    assert reports.band(0.03, None) == pytest.approx(0.06)     # a card's: no error
    ours = {"v": 0.70, "se": 0.03}
    assert reports.verdict(ours, {"v": 0.61, "se": 0.04}) == "level"       # 9 ≤ 10
    assert reports.verdict(ours, {"v": 0.59, "se": 0.04}) == "higher"      # 11 > 10
    assert reports.verdict(ours, {"v": 0.80, "se": 0.04}) == "level"       # 10 ≤ 10
    assert reports.verdict(ours, {"v": 0.81, "se": 0.04}) == "lower"
    assert reports.verdict(ours, {"v": 0.66, "se": None}) == "level"       # 4 ≤ 6
    assert reports.verdict(ours, {"v": 0.63, "se": None}) == "higher"      # 7 > 6
    assert reports.verdict(ours, {"v": None}) is None


def test_the_counts_and_the_summary_line():
    s4 = model("step4000", True, gpqa=(0.70, 0.03), otis=(0.50, 0.05), arc=(0.10, 0.01))
    q4 = model("Q4", True, gpqa=(0.72, 0.03), otis=(0.30, 0.05))
    a = model("A", gpqa=(0.59, 0.04), otis=(0.50, 0.0))       # higher, level
    b = model("B", gpqa=0.80, otis=None)                       # lower (a card: no error)
    c = model("C", gpqa=None, otis=(0.75, 0.02), arc=0.9)      # lower; ARC never compared
    st = reports.standing(s4, [s4, q4, a, b, c], BENCHES)
    assert (st["higher"], st["lower"], st["level"], st["n"]) == (2, 2, 2, 6)
    assert st["line"] == ("step4000 is higher than 2 and lower than 2 of the 6 scores it can be "
                          "set against, and level with 2.")
    rows = {r["label"]: r for r in st["rows"]}
    assert (rows["A"]["higher"], rows["A"]["lower"]) == (1, 0)          # level counts in neither
    assert (rows["Q4"]["higher"], rows["Q4"]["lower"]) == (1, 0)        # gpqa level, otis higher
    assert rows["B"]["diffs"]["otis"] is None and rows["B"]["lower"] == 1
    assert rows["A"]["diffs"]["gpqa"]["d"] == pytest.approx(0.11)
    assert [t["key"] for t in st["tiles"]] == ["gpqa", "otis"]          # compared only
    assert st["reference"] == {"gpqa": 0.70, "otis": 0.50}


def test_the_token_line():
    rep = {"benchmarks": [{"task": "gpqa_diamond_epoch", "label": "GPQA Diamond"},
                          {"task": "mmlupro_tiger", "label": "MMLU-Pro"},
                          {"task": "hle_text_cais", "label": "HLE"}],
           "reference": "Q4 (UD-Q4_K_XL)"}
    rows = [{"label": "step4000"}, {"label": "Q4 (UD-Q4_K_XL)"}]
    stats = {"step4000": {"gpqa_diamond_epoch": {"think_mean": 12000, "tok_sum": 9_000_000},
                          "mmlupro_tiger": {"think_mean": 6000, "tok_sum": 100_000_000},
                          "hle_text_cais": {"think_mean": 30000, "tok_sum": 50_000_000}},
             "Q4 (UD-Q4_K_XL)": {"gpqa_diamond_epoch": {"think_mean": 8000, "tok_sum": 7_000_000},
                                 "mmlupro_tiger": {"think_mean": 2500,
                                                   "tok_sum": 40_000_000}}}
    assert reports.token_line(rep, rows, stats) == (
        "step4000 thinks 1.5× as long as Q4 on GPQA Diamond and 2.4× on MMLU-Pro; over the 2 "
        "benchmarks both finished, it wrote 109.0 million output tokens to Q4's 47.0 million.")


# ---------------------------------------------------------------------------
# the token stats: equal to ~/tokens.py on answers built to match two rows
# ---------------------------------------------------------------------------

def text(share: float, n: int, closed: bool = True) -> str:
    """an answer of n characters whose thinking is `share` of them"""
    if not closed:
        return "<think>" + "x" * (n - 7)
    k = round(share * n)
    return "<think>" + "x" * k + "</think>" + "y" * (n - k - 15)


def build(groups: list[tuple], target: dict) -> list[tuple]:
    """runs of (tokens, ran out, right, answer) from groups of (count, tokens,
    ran out, right, thinking share, characters); the last group's last run
    is tuned so the thinking tokens' mean lands on the row's"""
    runs = []
    for count, tok, out, ok, share, n in groups:
        runs += [(tok, out, ok, text(share, n, closed=not out)) for _ in range(count)]

    def think(r):
        return r[0] * reports.think_share(r[3])
    want = target["think_mean"] * len(runs)
    t, out, ok, _ = runs[-1]
    n = groups[-1][5]
    rest = sum(think(r) for r in runs[:-1])
    runs[-1] = (t, out, ok, text((want - rest) / t, n))
    return runs


ROW = {(r["row"], r["task"]): r for r in TOKENS}
GPQA = ROW[("served__Qwen3.6-35B-A3B-BF16-lda-2610__thinking", "gpqa_diamond_epoch")]
OTIS = ROW[("served__Qwen3.6-35B-A3B-BF16-lda-2610__thinking", "otis_aime_epoch")]
LAYOUT = {
    # 792 answers, none ran out: 395 short, 2 at the median, 120 wrong, the
    # rest long, the 90th percentile and the longest
    "gpqa": [(394, 3054, False, True, 0.92774, 2000), (1, 3247, False, True, 0.92774, 2000),
             (2, 5594, False, True, 4831 / 5594, 60000), (120, 11648, False, False, 0.92257, 2000),
             (195, 15000, False, True, 0.92257, 2000), (79, 19772, False, True, 0.92257, 2000),
             (1, 76912, False, True, 0.92257, 60000)],
    # 360: 39 ran out at the limit, 306 right, 15 wrong
    "otis": [(39, 81920, True, False, 1.0, 60000), (179, 20000, False, True, 0.98496, 2000),
             (2, 32104, False, True, 31130 / 32104, 60000), (15, 64470, False, False, 0.965666, 2000),
             (124, 50138, False, True, 0.965666, 2000), (1, 50214, False, True, 0.965666, 60000)],
}


@pytest.mark.parametrize("which,row", [("gpqa", GPQA), ("otis", OTIS)])
def test_the_token_stats_equal_tokens_py(svc, which, row):
    runs = build(LAYOUT[which], row)
    d = config.OUT_DIR / row["row"]
    from service import frontier as sf
    td = sf.task_dir(d, row["task"])
    td.mkdir(parents=True, exist_ok=True)
    (td / f"samples_{row['task']}_2026-10-14T07-20-00.000000.jsonl").write_text("".join(
        json.dumps({"doc_id": k, "doc": {"id": f"q{k}"}, "resps": [[a]],
                    "frontier": [{"epoch": 0, "tokens": t, "ran_out": out, "ok": ok}]}) + "\n"
        for k, (t, out, ok, a) in enumerate(runs)))
    got = reports.token_stats(d, row["task"])
    want = {k: v for k, v in row.items() if k not in ("row", "task")}
    assert got == want, {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
    # kept until the file changes: asked again, the same, read from the cache
    assert reports.token_stats(d, row["task"], compute=False) == got


# ---------------------------------------------------------------------------
# the hidden half, and the status lines
# ---------------------------------------------------------------------------

def test_a_loop_example_from_the_hidden_half_is_refused(svc, tmp_path, monkeypatch):
    assert reports.shown_half("mmlupro_tiger", "486") and not reports.shown_half(
        "mmlupro_tiger", "1")
    with pytest.raises(reports.Hidden):
        reports.loop_example("mmlupro_tiger", "1", None, None)
    rep = {**REPORT, "loops": {**REPORT["loops"], "examples": [
        {"id": "1", "note": "from the hidden half"}, *REPORT["loops"]["examples"]]}}
    rows = reports.our_rows(rep, {})
    got = reports.loops(rep, rows)
    assert got[0]["refused"].startswith("question 1 is in the hidden half")
    assert all("refused" not in x for x in got[1:])             # the file's three: shown half


def registry(d: Path, shards: dict, tasks: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "frontier_imports.json").write_text(json.dumps({"imports": [], "tasks": tasks,
                                                         "shards": shards}))


def test_each_rows_status_and_a_dash_for_what_isnt_finished(svc):
    from service import frontier as sf
    rep = {**REPORT, "ours": REPORT["ours"][:2]}
    s4, s1 = (reports.row_dir(o["row"]) for o in rep["ours"])
    # step4000: every plan L box in but HLE's ninth (8 of 9 held), GPQA scored,
    # SimpleQA's answers whole and waiting for its grader
    every = {"box": "L1"}
    registry(s4, {"hle_text_cais": {"n": 9, "have": {str(i): {"box": f"L{i}"}
                                                     for i in range(1, 9)}}},
             {t: {**every, "shards": n} for t, n in (("mmlupro_tiger", 3), ("math_l5_epoch", 4),
                                                    ("gpqa_diamond_epoch", 4),
                                                    ("otis_aime_epoch", 3),
                                                    ("arc_agi2_public", 2))}
             | {"simpleqa_epoch": every})
    (sf.task_dir(s4, "simpleqa_epoch")).mkdir(parents=True)
    (sf.task_dir(s4, "simpleqa_epoch") / sf.ANSWERS).write_text("{}\n")
    cells = {"gpqa_diamond_epoch": {
        "served/step4000-lda-2610 · thinking": {"v": 0.616, "se": 0.031, "answers": 792,
                                                "ranOut": 12, "how": "by code"},
        "served/step1000-lda-2610 · thinking": {"v": 0.5, "se": 0.03, "answers": 792,
                                                "ranOut": 0}},
        "hle_text_cais": {"served/step4000-lda-2610 · thinking": {
            "v": 0.1, "se": 0.01, "shardsHeld": "shards 1–8 of 9 imported and held"}}}
    rows = {r["label"]: r for r in reports.our_rows(rep, cells)}
    a = rows["step4000"]
    assert a["status"] == "20 of 21 chunks in · SimpleQA Verified waits for grading"
    assert a["cells"]["gpqa"] == {"v": 0.616, "se": 0.031, "answers": 792, "ranOut": 12,
                                  "how": "by code", "final": True, "state": "score"}
    assert a["cells"]["hle"]["v"] is None and a["cells"]["hle"]["state"] == "held"  # no old number
    assert a["cells"]["simpleqa"]["state"] == "waits"
    assert rows["step1000"]["status"] == "1 of 7 benchmarks in"
    # every benchmark scored, nothing held: Complete
    full = {b["task"]: {"served/step1000-lda-2610 · thinking": {"v": 0.5, "se": 0.01}}
            for b in rep["benchmarks"]}
    assert {r["label"]: r for r in reports.our_rows(rep, full)}["step1000"]["status"] == \
        "Complete"


def test_the_endpoints_read_and_start_nothing(svc, monkeypatch):
    imported()

    def no(*a, **k):
        raise AssertionError("a report starts nothing")
    monkeypatch.setattr(db, "add", no)
    monkeypatch.setattr(reported, "import_epoch", no)
    got = {r["id"]: r for r in svc.get("/api/reports").json()["reports"]}
    assert sorted(got) == ["ran-out-of-room", "step-checkpoints"]          # 21b: a second
    sc = got["step-checkpoints"]
    assert sc["ours"][0] == "step4000" and sc["others"] == 41
    r = svc.get("/api/reports/step-checkpoints").json()
    assert r["picks"][0] == "step4000" and "Q4, earlier run" not in r["picks"]
    assert r["sources"]["epoch"]["words"].startswith("imported ")
    assert r["sources"]["epoch"]["credit"] == "Data: Epoch AI, CC BY 4.0"
    assert svc.get("/api/reports/nothing").status_code == 404
