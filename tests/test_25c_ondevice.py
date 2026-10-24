"""25c: Edge0 and Bonsai on the board from their published numbers, and the
On-device models report — every published number entered through the
board's card entry by masein, each with its link (the sources' tables are
below, as checked on 21 Oct); retention, each row's score over its own base's
from the same source (ours over our BF16); our phone build where it has
numbers; and a reported model never in an average or rank. Invented runs of
ours; nothing measured for theirs, nothing fetched."""

from __future__ import annotations

import json

import pytest

import enter_report_cards as erc
from conftest import make_service
from service import config, db, ondevice, reported, reports
from test_21c_page import results

REP = json.loads((reported.REPORTS_DIR / "on-device.json").read_text())
E0 = "https://github.com/Edge0-AI/Edge0/blob/7a203f255355e2641ec39f8b9df1b5343f22f548/README.md"
B27 = ("https://huggingface.co/prism-ml/Bonsai-27B-mlx-1bit/blob/"
       "ef22f239c670078e1507f9769bcaa66657332b96/README.md")
B2 = "https://prismml.com/news/bonsai-2-family"
GEMMA = "https://arxiv.org/abs/2607.02770"
# 27f: NVIDIA's card, pinned to the commit read on 24 Oct
NV = ("https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16/blob/"
      "49843eed8db1f1c616c47af75cfd36e07aec0667/README.md")
# the sources' own tables, as each was checked: {row: {benchmark key: (value, link)}}
SOURCES = {
    "Edge0-35B-A3B-preview": {"gpqa": (79.8, E0), "mmlupro": (81.0, E0), "ifbench": (57.9, E0)},
    "Qwen3.6-35B-A3B fp16 (Edge0's run)": {"gpqa": (81.8, E0), "mmlupro": (84.6, E0),
                                           "ifbench": (61.7, E0)},
    "Edge0-8B-A1B-preview": {"gpqa": (70.7, E0), "mmlupro": (70.1, E0), "ifbench": (53.9, E0)},
    "Ling 3.0 tiny fp16 (Edge0's run)": {"gpqa": (71.2, E0), "mmlupro": (65.8, E0),
                                         "ifbench": (60.6, E0)},
    "Bonsai 27B (1-bit)": {"ifeval": (79.11, B27), "ifbench": (52.36, B27)},
    "Qwen3.6-27B FP16 (PrismML's run)": {"ifeval": (88.91, B27), "ifbench": (68.03, B27)},
    "Ternary Bonsai 2 2B": {"gpqa": (39.6, B2), "ifeval": (85.77, B2), "ifbench": (48.0, B2)},
    "Ternary Bonsai 2 4B": {"gpqa": (63.84, B2), "ifeval": (88.91, B2), "ifbench": (56.33, B2)},
    "Ternary Bonsai 2 9B": {"gpqa": (69.9, B2), "ifeval": (90.02, B2), "ifbench": (57.0, B2)},
    "Qwen3.5 2B (PrismML's run)": {"gpqa": (58.48, B2), "ifeval": (75.79, B2),
                                   "ifbench": (34.33, B2)},
    "Qwen3.5 4B (PrismML's run)": {"gpqa": (78.08, B2), "ifeval": (86.69, B2),
                                   "ifbench": (50.33, B2)},
    "Qwen3.5 9B (PrismML's run)": {"gpqa": (82.53, B2), "ifeval": (89.65, B2),
                                   "ifbench": (57.33, B2)},
    "Gemma 4 E4B": {"gpqa": (58.6, None), "mmlupro": (69.4, None), "ifeval": (96.7, GEMMA),
                    "ifbench": (44.0, GEMMA)},
    "Gemma 4 E2B": {"gpqa": (43.4, None), "mmlupro": (60.0, None), "ifeval": (94.6, GEMMA),
                    "ifbench": (38.0, GEMMA)},
    # 27f: IFBench alone, Q4_K_M (the file its memory is), NVIDIA's run under NeMo
    # Gym, reasoning on
    "NVIDIA-Nemotron-3.5-Bolt-E4B": {"ifbench": (64.0, NV)},
    "Gemma 4 E4B-it (NVIDIA's run)": {"ifbench": (42.67, NV)},
}
TASK = {b["key"]: b["task"] for b in REP["benchmarks"]}
# ours, invented: {label: {benchmark: (score, se)}}
OURS = {"step8804": {"gpqa": (0.733, 0.02), "mmlupro": (0.780, 0.004)},
        "Q4 (UD-Q4_K_XL)": {"gpqa": (0.80, 0.02), "mmlupro": (0.82, 0.004)},
        "BF16": {"gpqa": (0.818, 0.02), "mmlupro": (0.828, 0.004)}}
PHONE = "served/Qwen3.6-35B-A3B-k4-LDA-phone-build"


def row_of(label: str) -> str:
    return next(o["row"] for o in REP["ours"] if o["label"] == label)


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, cells in OURS.items():
        mid = row_of(label)
        for k, (v, se) in cells.items():
            results(reports.row_dir(mid), mid.removesuffix(" · thinking"), TASK[k], v, se, 100, 0)
    yield client
    client.__exit__(None, None, None)


def test_every_published_number_is_its_sources_with_its_link():
    rows = {t["label"]: t for t in REP["theirs"]}
    assert set(rows) == set(SOURCES)
    for label, want in SOURCES.items():
        got = {k: c for k, c in rows[label].items() if isinstance(c, dict) and c.get("from") == "card"}
        assert {k: c["value"] for k, c in got.items()} == {k: v for k, (v, _) in want.items()}, label
        for k, (_, url) in want.items():
            if url:
                assert got[k]["url"] == url, (label, k)
            assert got[k]["url"].startswith("https://") and got[k]["date"]
        # only the board's benchmarks are cells: AIME 2026, HumanEval and the rest are "also"
        assert not set(rows[label]) & {"aime", "humaneval", "otis", "math"}
    # a base model's numbers are the report's alone, never a row on Models
    assert all(t.get("base_only") for t in REP["theirs"] if "run)" in t["label"])
    e0 = rows["Edge0-35B-A3B-preview"]["facts"]
    assert (e0["disk"]["words"], e0["disk"]["note"]) == ("19.6 GB", "its README says about 23 GB")
    assert e0["memory"]["words"] == "2.9 GiB" and e0["phone"]["none"] == "no phone numbers yet"
    assert [a["benchmark"] for a in e0["also"]] == ["AIME 2026", "HumanEval"]
    b27 = rows["Bonsai 27B (1-bit)"]["facts"]["phone"]
    assert (b27["tok_s"], b27["device"]) == (11.0, "iPhone 17 Pro Max")


def test_entered_by_masein_each_once_and_a_base_never_a_row(svc, capsys):
    assert erc.main(["step-checkpoints", "--by", "masein"]) == 0    # Gemma's, 25b's
    assert erc.main(["on-device", "--by", "masein"]) == 0
    out = capsys.readouterr().out
    # Edge0's 3 and 3 and its bases' 3 and 3, Bonsai 27B's 2 and its base's 2,
    # Ternary Bonsai 2's 9 and its bases' 9; Gemma 4's four each are 25b's
    # entries, the same cells: already there
    # 27f: and NVIDIA's two IFBench numbers
    assert "36 card numbers entered by masein, 8 already there" in out.splitlines()[-1]
    cards = [r for r in db.reported_scores() if r.get("source") == "card"]
    assert all(c["by"] == "masein" for c in cards)
    v = reported.view([])
    names = {m["name"] for m in v["models"]}
    assert {"Edge0-35B-A3B-preview", "Bonsai 27B (1-bit)", "Ternary Bonsai 2 9B"} <= names
    assert not any("run" in n and "fp16" in n.lower() for n in names)   # bases: the report's
    e0 = next(m for m in v["models"] if m["name"] == "Edge0-35B-A3B-preview")
    assert e0["measured"] is None and e0["publishedBy"] == "Edge0"
    assert e0["facts"]["base"].startswith("Qwen3.6-35B-A3B") and e0["link"].startswith(
        "https://huggingface.co/Edge0/")
    sc = [s for s in v["scores"] if s["model"] == e0["id"]]
    assert {s["benchmark"] for s in sc} == {"GPQA-Diamond", "MMLU-Pro", "IFBench"}
    assert all(s["published_by"] == "Edge0" and s["url"] == E0 for s in sc)


def test_retention_and_the_report(svc):
    erc.main(["step-checkpoints", "--by", "masein"])
    erc.main(["on-device", "--by", "masein"])
    d = reports.report("on-device")
    assert d["kind"] == "on_device"
    rows = {r["label"]: r for r in d["rows"]}
    # ours, measured here, BF16 the base and not a row; theirs as published, bases not rows
    # 27e: and Q2, our same-size baseline
    assert [r["label"] for r in d["rows"] if r["ours"]] == ["step8804", "step4000", "step1000",
                                                           "Q4 (UD-Q4_K_XL)", "Q2 (UD-Q2_K_XL)"]
    assert "BF16" not in rows and not any("run)" in x for x in rows)
    e0 = rows["Edge0-35B-A3B-preview"]
    assert e0["retention"]["gpqa"] == pytest.approx(79.8 / 81.8)
    assert round(100 * e0["retention"]["gpqa"], 1) == 97.6
    assert round(100 * e0["retention"]["mmlupro"], 1) == 95.7
    b9 = rows["Ternary Bonsai 2 9B"]
    assert round(100 * b9["retention"]["ifbench"], 1) == 99.4
    assert "mmlupro" not in b9["retention"]                 # neither published it
    assert rows["Gemma 4 E4B"]["retention"] == {}            # no base of its own
    # ours over our BF16, from the board's rows
    s88 = rows["step8804"]
    assert s88["retention"]["gpqa"] == pytest.approx(0.733 / 0.818)
    assert s88["retention"]["mmlupro"] == pytest.approx(0.780 / 0.828)
    assert s88["source"] == "measured here" and s88["cells"]["gpqa"]["src"] == "here"
    assert e0["source"] == "published by Edge0" and e0["cells"]["gpqa"]["url"] == E0
    assert e0["cells"]["ifeval"] == {"v": None, "src": "published", "by_org": "Edge0"}
    assert e0["phone"]["none"] == "no phone numbers yet"
    assert s88["phone"] == {"none": "not measured on a phone yet"}
    assert d["notes"][0].startswith("Published numbers come from each maker's own harness")


def fresh() -> dict:
    """the report, from the board as it is now (its payload kept no longer)"""
    import service.app as appmod
    appmod._cache.update(key=None, payload=None, at=0.0)
    return reports.report("on-device")


def test_our_phone_build_where_it_has_numbers(svc):
    db.served_put({"id": PHONE, "name": "Qwen3.6-35B-A3B k4-LDA phone build", "base_url": "",
                   "key": "", "how": "phone build", "based_on": "Qwen/Qwen3.6-35B-A3B",
                   "thinking": "on", "phone": True, "pin": {}, "by": "masein", "at": 0})
    assert not any(r.get("phoneBuild") for r in fresh()["rows"])       # no numbers yet
    results(reports.row_dir(PHONE + " · thinking"), PHONE, TASK["gpqa"], 0.70, 0.02, 100, 0)
    rows = {r["label"]: r for r in fresh()["rows"]}
    ph = rows["Qwen3.6-35B-A3B k4-LDA phone build (phone build)"]
    assert ph["phoneBuild"] and ph["cells"]["gpqa"]["v"] == pytest.approx(0.70)
    assert ph["phone"] == {"none": "none reported by hand yet"}
    assert ph["testers"] == {"none": "no tests from our testers' phones yet"}
    # one typed in from the phone: said as reported by hand, never as measured
    db.phone_add(PHONE, {"device": "OnePlus 15", "decode_median": 13.5, "by": "kevin",
                         "date": "2026-10-20"})
    ph = {r["label"]: r for r in fresh()["rows"]}[
        "Qwen3.6-35B-A3B k4-LDA phone build (phone build)"]
    assert (ph["phone"]["tok_s"], ph["phone"]["device"], ph["phone"]["by_hand"]) == (
        13.5, "OnePlus 15", "kevin")


def test_a_reported_model_is_never_in_an_average_or_rank(svc):
    import service.app as appmod
    erc.main(["on-device", "--by", "masein"])
    appmod._cache.update(key=None, payload=None, at=0.0)
    p = appmod.results_payload()
    ids = {m["id"] for m in p["models"]}
    assert not any("edge0" in i.lower() or "bonsai" in i.lower() for i in ids)
    got = svc.get("/api/reported").json()
    e0 = next(m for m in got["models"] if m["name"] == "Edge0-35B-A3B-preview")
    assert e0["id"].startswith("reported/") and e0["measured"] is None
    # the Reports list: the fourth, after the others (26: our testers' phones after it)
    ids = [r["id"] for r in svc.get("/api/reports").json()["reports"]]
    assert ids == ["ran-out-of-room", "step-checkpoints", "paired", "on-device", "testers"]
    assert ondevice.listing_counts(REP) == (["step8804", "step4000", "step1000",
                                             "Q4 (UD-Q4_K_XL)", "Q2 (UD-Q2_K_XL)"], 9)  # 27e, 27f
