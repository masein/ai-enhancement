"""25b: Gemma 4 E4B and E2B in the step-checkpoints scorecard, from Google's
own published numbers only — GPQA Diamond and MMLU-Pro from each model's
card, IFEval and IFBench from the Gemma 4 Technical Report's Table 5 (arXiv
2607.02770 v2) — entered through the board's card entry by masein, each with
its link and its source's date; every benchmark they don't report left
empty. IFEval and IFBench are compared now: their tiles in "Where we stand"
with the two Gemma dots, "level" from our standard error alone (a card's
number has none). Nothing measured here for them, nothing estimated."""

from __future__ import annotations

import json

import pytest

import enter_report_cards as erc
from service import db, reported, reports
from test_25_board import S88, svc  # noqa: F401 — svc is the fixture

REP = json.loads((reported.REPORTS_DIR / "step-checkpoints.json").read_text())
ARXIV = "https://arxiv.org/abs/2607.02770"
CARD = {"Gemma 4 E4B": "https://huggingface.co/google/gemma-4-E4B-it/blob/"
                       "ee0ef6023621cff504d758262d4e04895a5af4a2/README.md",
        "Gemma 4 E2B": "https://huggingface.co/google/gemma-4-E2B-it/blob/"
                       "3e22461f65e89153144f8adb70e3b8c2cc9845a7/README.md"}
# as checked on each source, 21 Oct: the card's table and Table 5 agree on
# GPQA Diamond and MMLU-Pro; IFEval and IFBench are Table 5's alone
WANT = {"Gemma 4 E4B": {"gpqa": 58.6, "mmlupro": 69.4, "ifeval": 96.7, "ifbench": 44.0},
        "Gemma 4 E2B": {"gpqa": 43.4, "mmlupro": 60.0, "ifeval": 94.6, "ifbench": 38.0}}
NOTE = ("Google's own numbers, thinking mode, Google's own harness and prompts. IFEval's metric "
        "(prompt- or instruction-level, strict or loose) isn't stated.")


def test_the_file_names_googles_numbers_and_nothing_else():
    rows = {t["label"]: t for t in REP["theirs"] if t["label"] in WANT}
    assert list(rows) == ["Gemma 4 E4B", "Gemma 4 E2B"]
    for label, t in rows.items():
        assert t["maker"] == "Google (open)" and t["epoch_version"] is None
        assert t["epoch_why"].startswith("not on Epoch AI")
        assert t["note"] == NOTE
        cells = {k: v for k, v in t.items() if isinstance(v, dict) and v.get("from")}
        assert {k: v["value"] for k, v in cells.items()} == WANT[label]
        for k, c in cells.items():
            assert c["from"] == "card" and c["model"] == label and c["maker"] == "Google"
            assert c["url"] == (ARXIV if k in ("ifeval", "ifbench") else CARD[label])
            assert c["date"] == ("2026-07-24" if c["url"] == ARXIV else "2026-07-20")
        # not reported for these two: HLE ("-" on the card), Google's AIME 2026 (not
        # OTIS Mock AIME), MATH Level 5, SimpleQA Verified, ARC-AGI-2
        assert not {"hle", "otis", "math", "simpleqa", "arc"} & set(t)
    b = {x["key"]: x for x in REP["benchmarks"]}
    for k in ("ifeval", "ifbench"):
        assert b[k]["compare"] is True and "in_every" not in b[k]
        assert b[k]["why"] == ("Compared with Google's published Gemma 4 numbers; their harness "
                               "and IFEval metric differ from ours, so treat small gaps as level.")


def test_each_cell_is_entered_once_with_its_link_and_the_rest_stay_empty(svc):  # noqa: F811
    assert erc.main(["step-checkpoints", "--by", "masein"]) == 0
    rows = {r["label"]: r for r in reports.their_rows(REP)}
    for label, want in WANT.items():
        r = rows[label]
        have = {k: c for k, c in r["cells"].items() if c.get("v") is not None}
        assert set(have) == set(want), label                 # exactly its four
        for k, v in want.items():
            c = have[k]
            assert round(100 * c["v"], 1) == v and c["se"] is None
            assert c["src"] == "card" and c["by"] == "masein"
            assert c["url"] == (ARXIV if k in ("ifeval", "ifbench") else CARD[label])
        for k in ("hle", "otis", "math", "simpleqa", "arc"):
            assert r["cells"][k] == {"v": None}               # empty, never estimated
        assert r["note"] == NOTE and r["version"] is None
    # entered again: nothing new
    n = len([x for x in db.reported_scores() if x.get("source") == "card"])
    assert erc.main(["step-checkpoints", "--by", "masein"]) == 0
    assert len([x for x in db.reported_scores() if x.get("source") == "card"]) == n


def test_their_tiles_in_where_we_stand_with_the_two_gemma_dots(svc):  # noqa: F811
    assert erc.main(["step-checkpoints", "--by", "masein"]) == 0
    d = reports.report("step-checkpoints", compute=True)
    tiles = {t["key"]: t for t in d["standing"][S88]["tiles"]}
    assert {"ifeval", "ifbench"} <= set(tiles)
    for k, (e4b, e2b) in (("ifeval", (0.967, 0.946)), ("ifbench", (0.44, 0.38))):
        gemma = [x for x in tiles[k]["dots"] if x["card"]]
        assert [(x["label"], round(x["v"], 3)) for x in gemma] == [("Gemma 4 E4B", e4b),
                                                                   ("Gemma 4 E2B", e2b)]
    # step8804's IFEval, 30.0 ± 10.5, is lower than both; its IFBench, 50.0 ± 11.5, is
    # level with both: a card's number has no error, so the band is ours alone (2 × 11.5)
    states = {k: {x["label"]: x["state"] for x in tiles[k]["dots"] if x["card"]}
              for k in ("ifeval", "ifbench")}
    assert states == {"ifeval": {"Gemma 4 E4B": "lower", "Gemma 4 E2B": "lower"},
                      "ifbench": {"Gemma 4 E4B": "level", "Gemma 4 E2B": "level"}}
    assert reports.band(0.115, None) == pytest.approx(0.23)
    row = next(r for r in d["standing"][S88]["rows"] if r["label"] == "Gemma 4 E2B")
    assert row["diffs"]["ifbench"]["band"] == pytest.approx(2 * tiles["ifbench"]["se"])
    assert row["diffs"]["hle"] is None                         # nothing to set against
    # Every model: both rows, every benchmark a column — sorted with the rest by GPQA
    gpqa = [(m["label"], (m["cells"]["gpqa"] or {}).get("v")) for m in [*d["ours"], *d["theirs"]]]
    assert ("Gemma 4 E4B", pytest.approx(0.586)) in gpqa and ("Gemma 4 E2B",
                                                               pytest.approx(0.434)) in gpqa
    assert all("in_every" not in b for b in d["benchmarks"])
