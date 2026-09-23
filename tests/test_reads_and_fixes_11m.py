"""11m: the rest of the 11l brief, off the page.

§7 — dataset #9 asked for question-and-answer items and was handed the
documents' register, "Prose, never question-and-answer pairs." The generator
did as told and kept 0 of 26. Question-and-answer items now have a register
and a system prompt of their own.

§5 — a read-only report of whether the judge skipped the conditional
criteria that did not apply, or scored them on every answer. It changes
nothing; masein runs it on the server, where the judged runs are.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import exam_build as eb
import judge as jd
from conftest import make_service
from service import config, llm, llm_poller
from service import proposals as prop

REPO = Path(__file__).resolve().parents[1]
BANKS = REPO / "eval_tasks" / "fr" / "banks"
NEVER_QA = re.compile(r"never (write )?(a )?question-and-answer pair", re.I)


# ---------------------------------------------------------------------------
# §7: question-and-answer items get their own register
# ---------------------------------------------------------------------------

def test_the_free_format_is_never_told_to_write_prose(tmp_path, monkeypatch):
    """Government & Public Policy, dataset #9's topic: its documents'
    register ends "Prose, never question-and-answer pairs." A request for
    question-and-answer items carries none of that, in the system prompt or
    the request."""
    eb.import_bank(tmp_path / "exam", BANKS / "government_public_policy_v1.json",
                   "Government & Public Policy", "masein")
    monkeypatch.setattr(config, "EXAM_DIR", tmp_path / "exam")
    doc = prop.audience_for("Government & Public Policy")
    free = prop.audience_for("Government & Public Policy", fmt="free")
    assert "Prose, never question-and-answer pairs" in doc          # #9's line, still there for docs
    assert "Register for question-and-answer items:" in free
    assert "Prose" not in free and not NEVER_QA.search(free)
    # the same audience either way: only the register differs
    assert doc.split("\n")[0] == free.split("\n")[0]
    for fmt, audience in (("free", free), ("doc", doc)):
        reqs = prop.generation_requests(9, "State the policy trade-off.", "Government & Public "
                                        "Policy", 26, fmt, seed=1, audience=audience)
        text = [r.system + "\n" + r.user for r in reqs]
        if fmt == "free":
            assert all(r.system == prop.GEN_SYSTEM_QA for r in reqs)
            assert not any(NEVER_QA.search(t) for t in text)
            assert all("question-and-answer pairs" in r.system for r in reqs)
        else:                                                         # documents: unchanged
            assert all(r.system == prop.GEN_SYSTEM for r in reqs)
            assert all(NEVER_QA.search(t) for t in text)


def obedient(req: llm.Request) -> str:
    """A generator that does what it is told: prose when the request forbids
    question-and-answer pairs, the items asked for otherwise. Dataset #9's
    generator, in one line."""
    if req.custom_id.startswith("gen:") and NEVER_QA.search(req.system + "\n" + req.user):
        return llm.default_responder(llm.Request(req.custom_id, req.system, req.user,
                                                 meta={**req.meta, "format": "doc"}))
    return llm.default_responder(req)


def test_a_free_dataset_from_a_generator_that_follows_its_register_is_kept(tmp_path,
                                                                             monkeypatch):
    client, _, _ = make_service(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(llm.FakeBatches, "responder", staticmethod(obedient))
        pid = client.post("/api/proposals", json={"model": "fx/good-750m", "topic": "Economics",
                                                  "requested_by": "masein"}).json()["id"]
        llm_poller.tick()
        client.post(f"/api/proposals/{pid}/approve", json={"approver": "masein"})
        did = client.post(f"/api/proposals/{pid}/generate",
                          json={"requester": "masein", "count": 26, "fmt": "free"}
                          ).json()["dataset_id"]
        llm_poller.tick()
        d = client.get(f"/api/datasets/{did}").json()
        assert d["status"] == "ready", d.get("error")
        assert d["provenance"]["items"]["kept"] == 26
        assert d["provenance"]["items"]["missing"] == []
        items = [json.loads(x) for x in
                 client.get(f"/api/datasets/{did}/items.jsonl").text.splitlines()]
        assert len(items) == 26 and all(it["question"] and it["answer"] for it in items)
        # and a document set from the same generator is still prose
        did = client.post(f"/api/proposals/{pid}/generate",
                          json={"requester": "masein", "count": 4, "fmt": "doc"}
                          ).json()["dataset_id"]
        llm_poller.tick()
        d = client.get(f"/api/datasets/{did}").json()
        assert d["status"] == "ready"
        docs = [json.loads(x) for x in
                client.get(f"/api/datasets/{did}/items.jsonl").text.splitlines()]
        assert all(it["title"] and it["text"] and "question" not in it for it in docs)
    finally:
        client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# §5: the conditional criteria, reported on and left alone
# ---------------------------------------------------------------------------

@pytest.fixture
def judged(tmp_path):
    import make_fixture
    m = make_fixture.build(tmp_path)
    return m["models"]["fx/good-750m"]["dir"], m["out_dir"]


def test_a_judge_that_decides_applicability_leaves_the_conditionals_null(judged):
    """The fixture's stub judge returns null for a conditional unless the
    question is about a medication — the path a real judge should take. The
    report sees the nulls, and leaving the conditionals out moves nothing."""
    mdir, _ = judged
    rows = {r["task"]: r for r in jd.conditional_report(mdir)}
    assert len(rows) >= 30
    for r in rows.values():
        assert 12 <= r["conditional"] <= 15
        assert r["pairs"] == r["answers"] * r["conditional"]
        assert r["applied"] < r["pairs"]
        assert r["never_skipped"] == 0
        if not r["applied"]:
            assert r["mean_as_graded"] == r["mean_without"]


def test_a_judge_that_scores_every_conditional_is_seen_and_its_cost_bounded(judged):
    """What §5 saw — every conditional scored on every answer — planted: each
    conditional set to 0.0 on every scored answer of one topic, and the fold
    recomputed as the judge would have. The report counts every pair as
    scored, every conditional as never skipped, and the mean with them left
    out is the higher one. judge.json is not touched."""
    mdir, _ = judged
    task = "exam_mathematics_statistics"
    p = mdir / "judge.json"
    j = json.loads(p.read_text(encoding="utf-8"))
    spec = jd.rubric_for(task).criteria
    cond = {c["id"] for c in spec["criteria"] if c.get("conditional")}
    assert len(cond) == 12                                  # "all twelve of them"
    for it in j["tasks"][task]["items"]:
        if it.get("graded") and it.get("criteria"):
            it["criteria"] = {k: (0.0 if k in cond else v) for k, v in it["criteria"].items()}
            it["score"] = jd.fold(it["criteria"], it.get("flags") or {}, spec)
    p.write_text(json.dumps(j), encoding="utf-8")
    before = p.read_bytes()
    r = next(x for x in jd.conditional_report(mdir) if x["task"] == task)
    assert r["applied"] == r["pairs"] and r["never_skipped"] == 12
    assert r["at_zero"] == 1.0 and r["mean_when_scored"] == 0.0
    assert r["mean_without"] > r["mean_as_graded"]
    assert p.read_bytes() == before                         # read only


def test_the_command_prints_a_line_a_topic_and_a_total(judged):
    mdir, out = judged
    res = subprocess.run([sys.executable, str(REPO / "scripts" / "judge.py"), str(out),
                          "--conditionals", "-m", "fx/good-750m"],
                         capture_output=True, text=True, cwd=REPO, check=True)
    lines = res.stdout.strip().splitlines()
    econ = next(x for x in lines if " exam_economics:" in x)
    assert econ.startswith("fx__good-750m  exam_economics: 14 conditional criteria, scored on ")
    assert "answer-criterion pairs" in econ and "with every conditional left out" in econ
    assert re.match(r"conditionals: scored on [\d,]+ of [\d,]+ pairs \(\d+%\) across \d+ "
                    r"judged topic\(s\)", lines[-1])
    # no question travels: nothing but ids and numbers
    rows = json.loads((mdir / "judge.json").read_text(encoding="utf-8"))
    for it in rows["tasks"]["exam_economics"]["items"][:5]:
        assert it["qid"] not in res.stdout
