"""12q.B (the brief's): DeviceMark's board beside ours — their snapshot as
published (their numbers never changed, the credit on each, nothing fetched
at page load), the note on the five rows whose raw outputs don't reproduce
their numbers, their rows read-only, everyone ranked together by their rule,
retention (a phone build over the original), and each of our answers in the
question browser with its pass or fail. No model runs; nothing is fetched."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import devicemark as dm
from conftest import make_service
from service import config, questions

FIVE = {"lfm2.5-1.2b__int8hu__aimodel", "granite-4.0-h-1b__int8hu__aimodel",
        "qwen3.5-0.8b__int8hu__aimodel", "qwen3.5-2b__int8hu__aimodel",
        "gemma-4-e2b__int4__litertlm"}


# ---------------------------------------------------------------------------
# their snapshot
# ---------------------------------------------------------------------------

def test_the_snapshot_is_their_board_as_published():
    snap = dm.snapshot()
    assert snap["source"] == dm.BOARD_URL == "https://devicemark.github.io/data/leaderboard/board.json"
    assert snap["credit"] == "DeviceMark (devicemark.github.io), CC-BY-4.0"
    assert snap["last_modified"] and snap["fetched"] == "2026-09-29" and len(snap["rows"]) == 12
    got = {r["id"]: r for r in dm.external_rows()}
    lfm = got["lfm2.5-1.2b__int8hu__aimodel"]
    assert lfm["composite"] == {"value": 0.6823, "ci": [0.6458, 0.7215]}
    assert (lfm["benches"]["ifeval"]["acc"], lfm["benches"]["mmlu_pro"]["acc"],
            lfm["benches"]["math"]["acc"]) == (0.8836, 0.4133, 0.75)
    assert lfm["device"] == {"tok_s": 45.5, "device": "iPhone 17 Pro", "source": "DeviceMark"}
    assert lfm["time_frontier"]["acc"][-1] == 0.527 and lfm["median_tokens"] == 561
    # every number is theirs, unchanged
    for raw in snap["rows"]:
        r, q = got[raw["artifact_id"]], raw["quality"]
        assert r["composite"] == raw["composite"] and r["external"] is True
        assert r["benches"]["ifeval"]["acc"] == q["ifeval_mean4"]
        assert r["benches"]["mmlu_pro"]["acc"] == q["mmlu_acc"]
        assert r["benches"]["math"]["acc"] == q["math_acc"]
        assert r["time_frontier"] == q["time_frontier"] and r["credit"] == snap["credit"]
        assert (r["device"] or {}).get("tok_s") == raw["iphone_tok_s"]
    # the cloud APIs and the built-in model have no device speed: lines, not points
    assert {r["id"] for r in got.values() if r["device"] is None} == {
        "gemini-flash__api__api", "gemini-pro__api__api", "apple-fm__system__system"}
    assert {got[i]["kind"] for i in ("gemini-flash__api__api", "apple-fm__system__system")} == {
        "cloud", "system"}


def test_the_five_rows_say_their_raw_outputs_dont_reproduce_their_board():
    got = {r["id"]: r for r in dm.external_rows()}
    noted = {i for i, r in got.items() if r["note"]}
    assert noted == FIVE
    for i in FIVE:
        n = got[i]["note"]
        assert "1,024-token cap" in n and "don't reproduce" in n and "board as published" in n
    # the note is ours, beside their rows — never written into them
    ours = set(dm.snapshot()["notes"].values())
    assert all(raw.get("note") not in ours for raw in dm.snapshot()["rows"])


def test_the_snapshot_is_refreshed_by_hand_and_keeps_the_notes(monkeypatch, tmp_path, capsys):
    import io
    import urllib.request
    f = tmp_path / "snap.json"
    f.write_text(dm.SNAPSHOT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(dm, "SNAPSHOT_PATH", f)
    dm.snapshot.cache_clear()
    one = [dm.snapshot()["rows"][0]]

    class Reply(io.BytesIO):
        headers = {"last-modified": "Tue, 29 Sep 2026 10:00:00 GMT"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    seen = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=60: (
        seen.append(url), Reply(json.dumps(one).encode()))[1])
    try:
        assert dm.main(["snapshot", "--fetch"]) == 0
        assert seen == [dm.BOARD_URL]
        got = json.loads(f.read_text(encoding="utf-8"))
        assert got["rows"] == one and got["last_modified"].startswith("Tue, 29 Sep")
        assert set(got["notes"]) == FIVE                             # kept
        assert "1 rows from https://devicemark.github.io" in capsys.readouterr().out
    finally:
        dm.snapshot.cache_clear()


# ---------------------------------------------------------------------------
# ours beside theirs
# ---------------------------------------------------------------------------

def _row(out_dir: Path, mid: str, acc: dict, setup: dict, thinking=False, device=None,
         speed=None) -> None:
    """a row as a run writes it: its answers scored, its numbers beside them"""
    scored = []
    for b, n in (("ifeval", 30), ("mmlu_pro", 20), ("math", 10)):
        for i in range(n):
            ok = i < round(acc[b] * n)
            s = {"bench": b, "key": f"{b}{i}", "text": "\\boxed{A}" if ok else "no",
                 "answer": "\\boxed{A}" if ok else "no", "correct": ok, "answered": ok,
                 "parsed": "A" if ok else None, "gold": "A", "gen_tokens": 300 + i}
            if b == "ifeval":
                s["ifeval"] = {m: {"prompt": ok, "inst": [ok]} for m in ("strict", "loose")}
            scored.append(s)
    d = out_dir / (mid.replace("/", "__") + ("__thinking" if thinking else ""))
    d.mkdir(parents=True, exist_ok=True)
    (d / dm.OUT_NAME).write_text(json.dumps(dm.summarize(scored, {**setup,
                                                                  "thinking": thinking})))
    (d / dm.ITEMS_NAME).write_text("".join(json.dumps(s) + "\n" for s in scored))
    base = out_dir / mid.replace("/", "__")
    if device:
        (base / dm.DEVICE_NAME).write_text(json.dumps(device))
    if speed:
        (base / dm.SPEED_NAME).write_text(json.dumps({"decode_tok_s": speed}))


SERVED = {"runtime": "llama-server", "lookahead": False, "mtp": True}


def test_retention_is_the_phone_build_over_the_original_on_the_same_terms(tmp_path):
    out = tmp_path / "full"
    _row(out, "served/phone-MTP", {"ifeval": 0.6, "mmlu_pro": 0.5, "math": 0.4},
         {**SERVED, "phone": True, "name": "phone MTP"})
    _row(out, "served/phone-LA-MTP", {"ifeval": 0.6, "mmlu_pro": 0.5, "math": 0.4},
         {**SERVED, "phone": True, "lookahead": True, "name": "phone LA MTP"})
    _row(out, "served/orig-MTP", {"ifeval": 0.8, "mmlu_pro": 0.5, "math": 0.5},
         {**SERVED, "phone": False, "name": "original MTP"})
    _row(out, "served/phone-MTP", {"ifeval": 0.3, "mmlu_pro": 0.5, "math": 0.4},
         {**SERVED, "phone": True, "name": "phone MTP"}, thinking=True)
    got = {r["id"]: r for r in dm.rows(out)}
    assert got["served/phone-MTP"]["retention"] == {"of": "served/orig-MTP", "ifeval": 0.75,
                                                    "mmlu_pro": 1.0, "math": 0.8}
    assert got["served/phone-LA-MTP"]["retention"] is None             # lookahead: never
    assert got["served/orig-MTP"]["retention"] is None                 # the original itself
    # thinking on has no original thinking on yet: none
    assert got["served/phone-MTP · thinking"]["retention"] is None


def test_everyone_is_ranked_together_and_theirs_are_read_only(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    try:
        _row(config.OUT_DIR, "served/phone-MTP", {"ifeval": 0.9, "mmlu_pro": 0.9, "math": 0.9},
             {**SERVED, "phone": True, "name": "phone MTP"}, speed=21.5)
        j = client.get("/api/devicemark").json()
        [ours] = j["rows"]
        theirs = j["external"]["rows"]
        assert len(theirs) == 12 and j["external"]["credit"].endswith("CC-BY-4.0")
        assert ours["server_tok_s"] == 21.5 and ours["device"] is None
        allr = [ours] + theirs
        assert [r["rank_all"] for r in allr] == dm.ranks(
            [{"ci": ours["row"]["composite"]["ci"]}] + [{"ci": r["composite"]["ci"]}
                                                        for r in theirs])
        # their rows can't be given a speed here: they are theirs
        r = client.put("/api/devicemark/device", json={
            "model": "lfm2.5-1.2b__int8hu__aimodel", "tok_s": 99, "device": "x", "source": "y"})
        assert r.status_code == 422
        assert dm.external_rows()[2]["device"]["tok_s"] == 45.5
    finally:
        client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# the question browser: each answer with its pass or fail
# ---------------------------------------------------------------------------

def test_the_question_browser_shows_each_answer_and_its_verdict(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    try:
        _row(config.OUT_DIR, "served/phone-MTP", {"ifeval": 0.5, "mmlu_pro": 0.5, "math": 0.5},
             {**SERVED, "phone": True, "name": "phone MTP"})
        config.DM_ITEMS.parent.mkdir(parents=True, exist_ok=True)
        config.DM_ITEMS.write_text(json.dumps({"version": dm.VERSION}) + "\n" + json.dumps(
            {"bench": "math", "key": "math0", "problem": "What is 1/2 + 1/2?", "answer": "1",
             "subject": "Algebra"}) + "\n")
        assert {"dm_ifeval", "dm_mmlu_pro", "dm_math"} <= set(questions.tasks())
        t = questions.table("dm_math")
        assert t["models"] == ["served/phone-MTP"] and len(t["rows"]) == 10
        r = t["rows"]["math0"]
        assert (r["q"], r["subject"], r["reference"]) == ("What is 1/2 + 1/2?", "Algebra", "1")
        res = r["results"]["served/phone-MTP"]
        assert res["ok"] is True and res["verdict"].startswith("read as A · the answer is A")
        wrong = t["rows"]["math9"]["results"]["served/phone-MTP"]
        assert wrong["ok"] is False and wrong["verdict"].startswith("no answer: no box")
        ife = questions.table("dm_ifeval")["rows"]["ifeval0"]["results"]["served/phone-MTP"]
        assert ife["verdict"].startswith("strict: 1 of 1 instructions followed")
        # what may be listed is what the board lists: its diagnose half
        assert {row["half"] for row in t["rows"].values()} <= {"diagnose", "report"}
        m = questions.meta("dm_ifeval")
        assert m["source"].startswith("google/IFEval: DeviceMark's protocol")
        assert m["licence"] == "Apache-2.0" and m["revision"]
    finally:
        client.__exit__(None, None, None)


@pytest.mark.parametrize("task,bench", [("dm_ifeval", "ifeval"), ("dm_mmlu_pro", "mmlu_pro"),
                                        ("dm_math", "math")])
def test_the_three_are_devicemarks_kind(task, bench):
    assert questions.kind_of(task) == "dm" and questions.DM_TASKS[task] == bench


# ---------------------------------------------------------------------------
# our runs of their open models, beside their rows
# ---------------------------------------------------------------------------

HF = {"runtime": "hf transformers (lm_eval)", "dtype": "bfloat16", "phone": False,
      "lookahead": False}


def test_their_nine_open_models_are_named_by_their_repos():
    got = {r["id"]: r for r in dm.external_rows()}
    assert set(dm.THEIR_OPEN) == {i for i, r in got.items() if r["kind"] == "device"}
    assert len(dm.THEIR_OPEN) == 9 and set(dm.CALIBRATION) <= set(dm.THEIR_OPEN.values())
    # what theirs is, beside ours
    assert got["lfm2.5-1.2b__int8hu__aimodel"]["label"] == "int8, iPhone"
    assert got["gemma-4-e2b__int4__litertlm"]["label"] == "int4, iPhone"
    assert got["youtu-2b__int8__aimodel"]["label"] == "int8, iPhone"


def test_our_run_of_their_model_sits_beside_their_row_never_ranked_apart(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    try:
        out = config.OUT_DIR
        nemo, lfm = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "LiquidAI/LFM2.5-1.2B-Instruct"
        _row(out, nemo, {"ifeval": 0.6, "mmlu_pro": 0.6, "math": 0.8}, HF)
        _row(out, nemo, {"ifeval": 0.5, "mmlu_pro": 0.7, "math": 0.9}, HF, thinking=True)
        _row(out, lfm, {"ifeval": 0.9, "mmlu_pro": 0.4, "math": 0.7}, HF)
        _row(out, "Qwen/Qwen3-1.7B", {"ifeval": 0.7, "mmlu_pro": 0.4, "math": 0.6}, HF)
        # a served build of one of their models is ours alone: only hf runs pair
        _row(out, "served/Nanbeige4.1-3B-Q8", {"ifeval": 0.5, "mmlu_pro": 0.3, "math": 0.4},
             {**SERVED, "phone": False, "name": "Nanbeige Q8"})
        j = client.get("/api/devicemark").json()
        ours = {r["id"]: r for r in j["rows"]}
        theirs = {r["id"]: r for r in j["external"]["rows"]}
        assert ours[nemo]["paired"] == ours[f"{nemo} · thinking"]["paired"] == \
            "nemotron-4b__int8hu__aimodel"
        assert ours[lfm]["paired"] == "lfm2.5-1.2b__int8hu__aimodel"
        assert not ours["Qwen/Qwen3-1.7B"].get("paired")
        assert not ours["served/Nanbeige4.1-3B-Q8"].get("paired")
        # beside theirs: thinking off first, how each ran, the calibration pair said
        side = theirs["nemotron-4b__int8hu__aimodel"]["ours"]
        assert [(o["id"], o["label"], o["calibration"]) for o in side] == [
            (nemo, "bf16, our battery", True),
            (f"{nemo} · thinking", "bf16, our battery, thinking", True)]
        assert side[0]["composite"] == ours[nemo]["row"]["composite"]
        assert set(side[0]["benches"]) == {"ifeval", "mmlu_pro", "math"}
        assert theirs["lfm2.5-1.2b__int8hu__aimodel"]["ours"][0]["calibration"] is False
        # their numbers never changed by it
        assert theirs["nemotron-4b__int8hu__aimodel"]["composite"] == {
            "value": 0.6136, "ci": [0.5782, 0.6511]}
        # ranked: our solo rows and theirs; a paired run of ours never apart
        assert ours[nemo]["rank_all"] is None and ours[lfm]["rank_all"] is None
        solo = [ours["Qwen/Qwen3-1.7B"], ours["served/Nanbeige4.1-3B-Q8"]]
        allr = solo + j["external"]["rows"]
        assert [r["rank_all"] for r in allr] == dm.ranks(
            [{"ci": r["row"]["composite"]["ci"]} for r in solo]
            + [{"ci": r["composite"]["ci"]} for r in j["external"]["rows"]])
    finally:
        client.__exit__(None, None, None)
