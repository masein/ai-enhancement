"""C2: the judge is an API call. Pinned to a dated id, on its own provider,
watched by a canary; nothing judged counts unless the judge that graded it
is the judge this server runs, calibrated, and steady."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import judge as jd
import make_fixture
import report_lm_eval as report
from conftest import fresh, make_service
from service import llm, llm_poller

REPO = Path(__file__).resolve().parents[1]
STUB = {"provider": "stub", "model": "overlap-v1", "id": "stub/overlap-v1", "family": "stub"}


# ---------------------------------------------------------------------------
# the canary
# ---------------------------------------------------------------------------

def test_canary_file_spans_the_rubric():
    c = jd.load_canary()
    assert len(c) == 30
    assert {x["human_score"] for x in c} == {0, 1, 2, 3, 4}
    assert len({x["id"] for x in c}) == 30
    assert all(x["prompt"] and x["reference"] and x["task"].startswith("exam_") for x in c)
    assert len({x["task"] for x in c}) >= 10                       # across the topics


def test_canary_arithmetic():
    canary = [{"id": "a", "human_score": 4}, {"id": "b", "human_score": 2},
              {"id": "c", "human_score": 0}, {"id": "d", "human_score": 3}]
    scores = {"a": 4, "b": 3, "c": 0, "d": 1}
    st = jd.canary_stats(scores, canary, previous=None, threshold=0.5)
    assert st["mad_vs_human"] == pytest.approx((0 + 1 + 0 + 2) / 4)
    assert st["mad_vs_previous"] is None and st["drifted"] is False and st["graded"] == 4
    st = jd.canary_stats(scores, canary, previous={"a": 4, "b": 3, "c": 1, "d": 1}, threshold=0.5)
    assert st["mad_vs_previous"] == pytest.approx(0.25) and st["drifted"] is False
    st = jd.canary_stats(scores, canary, previous={"a": 2, "b": 1, "c": 2, "d": 3}, threshold=0.5)
    assert st["mad_vs_previous"] == pytest.approx(2.0) and st["drifted"] is True
    # an ungraded script counts against neither mean, and is reported
    st = jd.canary_stats({"a": 4, "b": None, "c": 0, "d": 3}, canary, previous={"a": 4}, threshold=0.5)
    assert st["graded"] == 3 and st["mad_vs_human"] == 0.0 and st["mad_vs_previous"] == 0.0
    assert st["scores"]["b"] is None and st["human"]["b"] == 2


def test_canary_history_makes_the_second_run_comparable(tree, tmp_path):
    d = tree["models"]["fx/skewed-360m"]["dir"]
    first = jd.run_stub(d, tmp_path, record=True)
    assert first["canary"]["mad_vs_previous"] is None                 # first run for this judge
    second = jd.run_stub(d, tmp_path, record=True)
    assert second["canary"]["mad_vs_previous"] == 0.0 and second["canary"]["drifted"] is False
    assert second["preliminary_reasons"] == []
    hist = [json.loads(x) for x in (tmp_path / jd.CANARY_HISTORY).read_text().splitlines()]
    assert len(hist) == 2 and hist[0]["judge_id"] == "stub/overlap-v1"
    assert jd.previous_canary(tmp_path, "stub/overlap-v1") == hist[1]["scores"]
    assert jd.previous_canary(tmp_path, "someone/else") is None
    # a moved judge: doctor the history and re-assemble
    hist[-1]["scores"] = {k: (v + 2) % 5 for k, v in hist[-1]["scores"].items()}
    (tmp_path / jd.CANARY_HISTORY).write_text("".join(json.dumps(h) + "\n" for h in hist))
    third = jd.run_stub(d, tmp_path, record=False, threshold=0.5)
    assert third["canary"]["drifted"] is True
    assert any("the judge moved" in r for r in third["preliminary_reasons"])


def test_the_stub_is_still_deterministic(tree, tmp_path):
    d = tree["models"]["fx/skewed-360m"]["dir"]
    a = jd.write_judge(d, jd.run_stub(d, tmp_path / "h1"), tmp_path / "a")
    b = jd.write_judge(d, jd.run_stub(d, tmp_path / "h2"), tmp_path / "b")
    assert a.read_bytes() == b.read_bytes()


# ---------------------------------------------------------------------------
# identity, pinning, refusals
# ---------------------------------------------------------------------------

def test_judge_json_records_provider_model_prompt_rubric_batch(tree):
    for mid in make_fixture.JUDGED:
        j = json.loads((tree["models"][mid]["dir"] / "judge.json").read_text())
        jj = j["judge"]
        assert jj["provider"] == "stub" and jj["model"] == "overlap-v1" and jj["id"] == "stub/overlap-v1"
        assert jj["batch_id"] == "stub" and jj["prompt_version"] == 2
        assert jj["prompt_sha256"] == jd.prompt_sha() and jj["single_provider_loop"] is False
        assert all(len(r["sha256"]) == 64 for r in jj["rubrics"].values())
        assert j["canary"]["n"] == 30 and j["canary"]["graded"] == 30
        assert j["preliminary_reasons"] == []
        for t, v in j["tasks"].items():
            assert v["ungraded"] == 0
            assert all(it["graded"] and it["justification"] for it in v["items"])


def test_blocked_reasons_are_stated_not_raised(monkeypatch):
    from service import config
    def cfg(**kw):
        for k, v in {"JUDGE_PROVIDER": "", "JUDGE_MODEL": "", "JUDGE_API_KEY": "",
                     "EXAM_PROVIDER": "openai", "EXAM_MODEL": "gpt-x", "EXAM_API_KEY": "k",
                     "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-x", "LLM_API_KEY": "k",
                     "ALLOW_SINGLE_PROVIDER_LOOP": False, **kw}.items():
            monkeypatch.setattr(config, k, v)
    cfg()
    assert "JUDGE_MODEL is unset" in jd.blocked()
    cfg(JUDGE_MODEL="stub")
    assert jd.blocked() == "" and jd.identity()["id"] == "stub/overlap-v1"
    cfg(JUDGE_MODEL="claude-sonnet-4-5-20250929")
    assert "JUDGE_PROVIDER is unset" in jd.blocked()
    cfg(JUDGE_PROVIDER="mistral", JUDGE_MODEL="x-20250101")
    assert "not one of" in jd.blocked()
    cfg(JUDGE_PROVIDER="anthropic", JUDGE_MODEL="claude-sonnet-4-5", JUDGE_API_KEY="k")
    assert "floating alias" in jd.blocked()
    cfg(JUDGE_PROVIDER="anthropic", JUDGE_MODEL="claude-sonnet-4-5-20250929", JUDGE_API_KEY="")
    assert "JUDGE_API_KEY is unset" in jd.blocked()
    cfg(JUDGE_PROVIDER="anthropic", JUDGE_MODEL="claude-sonnet-4-5-20250929", JUDGE_API_KEY="k")
    assert jd.blocked() == "" and jd.provider_clash() == "" and not jd.single_provider_loop()
    assert jd.identity() == {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929",
                             "id": "anthropic/claude-sonnet-4-5-20250929", "family": "claude"}
    # the three-family rule: same provider as the exam writer → refused, with the reason
    cfg(JUDGE_PROVIDER="openai", JUDGE_MODEL="gpt-4.1-2025-04-14", JUDGE_API_KEY="k")
    assert "same as the exam writer" in jd.blocked() and jd.provider_clash() == "exam writer"
    cfg(JUDGE_PROVIDER="openai", JUDGE_MODEL="gpt-4.1-2025-04-14", JUDGE_API_KEY="k",
        EXAM_PROVIDER="anthropic")
    assert "same as the generator" in jd.blocked() and jd.provider_clash() == "generator"
    # the documented override: allowed, and stamped
    cfg(JUDGE_PROVIDER="openai", JUDGE_MODEL="gpt-4.1-2025-04-14", JUDGE_API_KEY="k",
        ALLOW_SINGLE_PROVIDER_LOOP=True)
    assert jd.blocked() == "" and jd.single_provider_loop() is True
    # a broken judge never crashes startup — it is shown on the page
    cfg(JUDGE_PROVIDER="anthropic", JUDGE_MODEL="claude-sonnet-4-5", JUDGE_API_KEY="")
    llm.startup_check()
    assert "floating alias" in config.judged_blocked()


def test_family_refusal_still_holds(tree):
    d = tree["models"]["fx/good-750m"]["dir"]
    reqs, plan = jd.plan_requests(d, "good")
    assert reqs == [] and "same family as judge" in plan["skipped"]
    out = jd.assemble(plan, {}, {**STUB, "family": "good"}, "", tree["out_dir"], 0.5, False,
                      record=False)
    assert "same family" in out["skipped"] and out["tasks"] == {}
    reqs, plan = jd.plan_requests(d, "claude")
    assert plan["skipped"] is None and len(reqs) == 30 + sum(len(v) for v in plan["tasks"].values())
    assert reqs[0].custom_id.startswith("canary:") and reqs[-1].custom_id.startswith("judge:")
    assert jd.family("anthropic/claude-sonnet-4-5-20250929") == "claude"
    assert jd.family("gpt-4.1-2025-04-14") == "gpt"


def test_parse_grade_takes_json_and_bare_digits():
    assert jd.parse_grade('{"score": 3, "justification": "covers the mechanism"}') == (3, "covers the mechanism")
    assert jd.parse_grade("Score: 2") == (2, "") and jd.parse_grade(" 4") == (4, "")
    assert jd.parse_grade('{"score": 9}')[0] is None
    assert jd.parse_grade("no idea") == (None, "")
    assert jd.parse_grade('[{"question": "case 1"}]')[0] is None    # prose with a digit is not a grade
    assert jd.parse_grade("I would give this a 3")[0] is None


# ---------------------------------------------------------------------------
# the batch path, through the service and the poller
# ---------------------------------------------------------------------------

def test_judged_run_is_submitted_then_finished_by_the_poller(tmp_path, monkeypatch):
    from service import config, db
    client, appmod, tree = make_service(tmp_path, monkeypatch, judged=True, judge_model="")
    try:
        monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
        monkeypatch.setattr(config, "JUDGE_MODEL", "fake-judge-20250101")
        llm.reset()
        st = client.get("/api/judge").json()
        assert st["configured"] and st["judge_id"] == "fake/fake-judge-20250101"
        assert st["provider_clash"] == "" and st["single_provider_loop"] is False
        d = tree["models"]["fx/chance-160m"]["dir"]
        (d / "judge.json").unlink()                                    # the stub's file
        jr = jd.start_run(d, tree["out_dir"])
        assert jr["mode"] == "batch" and jr["batch_id"].startswith("fake_") and jr["n"] > 30
        assert not (d / "judge.json").exists()                         # nothing until the batch completes
        runs = db.judge_runs()
        assert runs[0]["status"] == "submitted" and runs[0]["judge_id"] == "fake/fake-judge-20250101"
        assert [b["kind"] for b in db.batches_pending()] == ["judge"]
        assert llm_poller.tick() == 1
        assert db.judge_runs()[0]["status"] == "done"
        j = json.loads((d / "judge.json").read_text())
        assert j["judge"]["provider"] == "fake" and j["judge"]["batch_id"] == jr["batch_id"]
        assert j["judge"]["stub"] is False and j["canary"]["n"] == 30
        assert set(j["tasks"]) == set(tree["judged"]["manifest"]["tasks"])
        # the fake grades like the stub, so every answer and every canary script has a score
        assert all(v["ungraded"] == 0 for v in j["tasks"].values())
        assert j["canary"]["graded"] == 30 and j["canary"]["mad_vs_previous"] is None
        fresh(appmod)
        p = client.get("/api/results").json()
        row = next(m for m in p["models"] if m["id"] == "fx/chance-160m")
        assert row["judge"]["judge"]["id"] == "fake/fake-judge-20250101"
        assert row["judgeState"]["ok"] is False
        assert any("not the judge this server runs now" not in r for r in row["judgeState"]["reasons"])
        assert any("calibration on file is for stub/overlap-v1" in r for r in row["judgeState"]["reasons"])
        # the other models were graded by the stub — a different judge from the current one
        other = next(m for m in p["models"] if m["id"] == "fx/good-750m")
        assert other["judgeState"]["current"] is False
        assert any("different series" in r for r in other["judgeState"]["reasons"])
        assert any("different judge" in w for w in p["warnings"])
        assert p["judged"]["current"]["id"] == "fake/fake-judge-20250101"
    finally:
        client.__exit__(None, None, None)


def test_stub_run_writes_immediately_and_records_the_canary(tmp_path, monkeypatch):
    from service import config
    client, appmod, tree = make_service(tmp_path, monkeypatch, judged=True, judge_model="stub")
    try:
        d = tree["models"]["fx/chance-160m"]["dir"]
        jr = jd.start_run(d, tree["out_dir"])
        assert jr == {"mode": "stub", "written": True, "skipped": None}
        j = json.loads((d / "judge.json").read_text())
        assert j["canary"]["mad_vs_previous"] == 0.0                   # the fixture ran the stub before
        assert config.judged_blocked() == ""
        assert client.post("/api/submissions", json={"hf_id": "org/m", "suite": "judged"}).status_code == 200
    finally:
        client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# gating on the payload
# ---------------------------------------------------------------------------

def test_judged_state_reasons():
    j = {"judge": {"id": "stub/overlap-v1"}, "preliminaryReasons": [], "tasks": {}}
    cal = {"kappa": 0.8, "n": 60, "calibrated": True, "judge_id": "stub/overlap-v1"}
    assert report.judged_state(j, cal, "stub/overlap-v1") == {"ok": True, "reasons": [], "current": True}
    assert "not been calibrated" in report.judged_state(j, None, "stub/overlap-v1")["reasons"][0]
    low = {**cal, "kappa": 0.41, "calibrated": False}
    assert "below 0.6" in report.judged_state(j, low, "stub/overlap-v1")["reasons"][0]
    other = {**cal, "judge_id": "anthropic/claude-x-20250101"}
    assert "calibration on file is for" in report.judged_state(j, other, "stub/overlap-v1")["reasons"][0]
    st = report.judged_state(j, cal, "anthropic/claude-x-20250101")
    assert st["current"] is False and "different series" in st["reasons"][0]
    moved = {**j, "preliminaryReasons": ["the judge moved: canary grades differ …"]}
    assert "the judge moved" in report.judged_state(moved, cal, "stub/overlap-v1")["reasons"][0]
    assert report.judged_state(None, cal, "x")["reasons"] == ["not judged"]
    skipped = {**j, "skipped": "not judged — same family as judge (stub)"}
    assert report.judged_state(skipped, cal, "stub/overlap-v1")["ok"] is False


def test_fixture_rows_count_and_a_different_current_judge_demotes_them(payload, tree):
    good = next(m for m in payload["models"] if m["id"] == "fx/good-750m")
    assert good["judgeState"] == {"ok": True, "reasons": [], "current": True}
    assert payload["judged"]["current"]["id"] == "stub/overlap-v1"
    assert payload["judged"]["calibration"]["judge_id"] == "stub/overlap-v1"
    assert good["judge"]["canary"]["drifted"] is False and good["judge"]["judge"]["provider"] == "stub"
    runs = report.load_results(tree["out_dir"])
    cal = json.loads((tree["out_dir"] / "judge_calibration.json").read_text())
    p = report.build_payload(report.merge_runs(runs), "t", "", calibration=cal,
                             judge_identity={"provider": "anthropic", "model": "claude-x-20250101",
                                             "id": "anthropic/claude-x-20250101", "family": "claude"})
    good = next(m for m in p["models"] if m["id"] == "fx/good-750m")
    assert good["judgeState"]["ok"] is False and good["judgeState"]["current"] is False
    assert any("different judge" in w for w in p["warnings"])


def test_an_old_local_judge_file_is_labelled_local():
    t = report._trim_judge({"judge": {"id": "meta-llama/Llama-3.1-8B-Instruct", "family": "llama",
                                      "weights_sha256": "ab" * 32, "greedy": True},
                            "tasks": {"exam_law": {"n": 3, "mean": 2.0}}})
    assert t["judge"]["provider"] == "local" and t["judge"]["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert t["canary"] is None and t["preliminaryReasons"] == []
    assert report.published_score(t["tasks"]["exam_law"]) == 2.0          # an older file: the mean


def test_all_three_identities_reach_provenance(tmp_path, monkeypatch):
    # the judge this server runs must be the one that graded the board, or the
    # gate refuses the proposal — so the fixture's stub judge is configured here
    client, appmod, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    try:
        r = client.post("/api/proposals", json={"model": "fx/good-750m", "topic": "economics",
                                                "requested_by": "t"})
        assert r.status_code == 200, r.text
        pid = r.json()["id"]
        llm_poller.tick()
        client.post(f"/api/proposals/{pid}/approve", json={"approver": "Omar"})
        did = client.post(f"/api/proposals/{pid}/generate",
                          json={"requester": "Omar", "count": 10}).json()["dataset_id"]
        llm_poller.tick()
        d = client.get(f"/api/datasets/{did}").json()
        assert d["status"] == "ready", d["error"]
        ids = d["provenance"]["identities"]
        assert ids == {"exam_writer": "fake/fake-exam", "judge": "stub/overlap-v1",
                       "generator": "fake/fake-1", "single_provider_loop": False}
        assert d["provenance"]["judge_run"]["judge_id"] == "stub/overlap-v1"
    finally:
        client.__exit__(None, None, None)


def test_runner_no_longer_runs_a_judge_subprocess():
    from service import runner
    assert not hasattr(runner, "judge_cmd")
    src = (REPO / "service" / "runner.py").read_text()
    assert "start_run(" in src and "never hold the card" in src


def test_cli(tree, tmp_path, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, "argv", ["judge.py", str(tree["out_dir"]), "--stub", "-o", str(tmp_path)])
    assert jd.main() == 0
    out = capsys.readouterr().out
    assert "canary MAD" in out and "STUB" in out
    from service import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    monkeypatch.setattr(sys, "argv", ["judge.py", str(tree["out_dir"])])
    assert jd.main() == 2                                              # stub configured: pass --stub
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-judge-20250101")
    monkeypatch.setattr(sys, "argv", ["judge.py", str(tree["out_dir"])])
    assert jd.main() == 2                                              # API judge: needs --wait
