"""12o.1: the shared suite thinks when asked — GPQA Diamond measured the way
Epoch AI runs it, with reasoning. A thinking run is a row of its own, as a
generative one is: both of its benchmarks' answers are the thinking row's,
and SimpleQA's grades land there. The calibration line says which it was:
"measured here 31.3 (thinking off) · Epoch 38.0".

Fixtures, the stand-in judge and a fake GPU only."""

from __future__ import annotations

import pytest

import report_lm_eval as report  # noqa: F401 — the page's module, as the others import it
import simpleqa as sq
from conftest import make_service
from service import config, db, runner
from test_12n2 import fake_gpu


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client
    client.__exit__(None, None, None)


def submit(client, **kw):
    return client.post("/api/submissions", json={"hf_id": "org/chat-1b", "kind": "instruct",
                                                 **kw})


def test_thinking_is_for_the_generative_and_shared_suites(svc):
    client = svc
    r = submit(client, suite="shared", thinking=True)
    assert r.status_code == 200 and db.get(r.json()["id"])["thinking"]
    for suite in ("full", "safety", "everyday"):
        r = submit(client, suite=suite, thinking=True)
        assert r.status_code == 422
        # 12q: and DeviceMark's battery — 16b: and the Mobile suite — 17: and the Frontier
        assert r.json()["detail"] == ("thinking is for IFEval, MMLU-Pro and MATH-500 (suite "
                                      "generative), GPQA Diamond and SimpleQA Verified "
                                      "(suite shared), the Mobile suite (suite mobile), "
                                      "DeviceMark's battery (suite devicemark) and the "
                                      "Frontier benchmarks (suite frontier) only")
    r = submit(client, suite="shared", subset=100)
    assert r.status_code == 422
    assert r.json()["detail"] == "subset is for MMLU-Pro (suite generative) only"
    # a thinking run and a plain one are two runs, never joined in the queue
    plain = submit(client, suite="shared").json()
    assert plain["status"] == "queued" and "note" not in plain


def test_a_thinking_shared_run_is_the_thinking_row_and_simpleqa_is_graded_there(
        svc, monkeypatch):
    client = svc
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    seen = fake_gpu(monkeypatch)
    # a model with a switch
    pf = runner.preflight
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: {
        **pf(*a, **k), "archinfo": {"thinking": "switch", "think_end": "</think>"}})
    sid = submit(client, suite="shared", thinking=True).json()["id"]
    runner.run_submission(db.get(sid))
    cot, sqa = seen
    for cmd in (cot, sqa):
        margs = cmd[cmd.index("--model_args") + 1]
        assert "enable_thinking=True" in margs and "think_end_token=</think>" in margs
        assert cmd[cmd.index("--gen_kwargs") + 1] == \
            f"max_gen_toks={config.GEN_THINKING_MAX_GEN_TOKS}"
        assert "org__chat-1b__thinking" in cmd[cmd.index("--output_path") + 1]
    # SimpleQA still from its pinned file
    assert sqa[sqa.index("--include_path") + 1] == str(config.SIMPLEQA_TASKS_DIR)
    row = config.OUT_DIR / "org__chat-1b__thinking"
    assert (sq.read(row) or {}).get("items")
    assert sq.read(config.OUT_DIR / "org__chat-1b") is None
    assert db.get(sid)["status"] == "done"
    assert db.get(sid)["progress"] == "SimpleQA Verified 30.0% correct · 30% not attempted"


def test_thinking_off_the_same_model_is_asked_without_it(svc, monkeypatch):
    client = svc
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    seen = fake_gpu(monkeypatch)
    pf = runner.preflight
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: {
        **pf(*a, **k), "archinfo": {"thinking": "switch"}})
    sid = submit(client, suite="shared").json()["id"]
    runner.run_submission(db.get(sid))
    cot, sqa = seen
    assert "enable_thinking=False" in cot[cot.index("--model_args") + 1]
    # SimpleQA as 12n.2 asks it, into the model's own row
    assert "enable_thinking" not in sqa[sqa.index("--model_args") + 1]
    assert sq.read(config.OUT_DIR / "org__chat-1b")["items"]


def test_the_judges_grades_land_in_the_thinking_row(svc, monkeypatch):
    from service import llm, llm_poller
    from test_12n2 import sit_simpleqa, twenty
    client = svc
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-judge")
    llm.reset()
    plain, row = config.OUT_DIR / "org__m", config.OUT_DIR / "org__m__thinking"
    sit_simpleqa(plain, twenty())
    sit_simpleqa(row, twenty())
    sid = submit(client, hf_id="org/m", suite="shared", thinking=True).json()["id"]
    out = sq.start(row, submission=sid)
    assert out["batch_id"] and out["waiting"]
    llm_poller.tick()
    assert sq.read(row)["waiting"] == 0 and sq.read(row)["rates"]["correct"]["rate"] == 0.3
    # the model's own row was not this batch's
    assert sq.read(plain) is None
    assert db.get(sid)["progress"] == "SimpleQA Verified 30.0% correct · 30% not attempted"
