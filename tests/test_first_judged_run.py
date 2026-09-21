"""What the first full judged run (five topics + the control, #46) showed.

Every one of these is a number the page reported about itself and got wrong —
a flag count under the wrong heading, a zero that looked like a gap, one
run's batch on another run's row, a finished run still counting — plus the
provenance the import panel had no way to record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import exam_build as eb
import judge as jd
from conftest import make_service

REPO = Path(__file__).resolve().parents[1]
TOPIC = "Medicine & Clinical Health"
TASK = "exam_medicine_clinical_health"
LAW = "exam_law"
# the provenance tests below are about the banks of that run — the files
# whose source and author came out wrong — so they import those files,
# now retired, into the topic that replaced theirs
RETIRED = REPO / "eval_tasks" / "fr" / "retired"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, appmod, tree
    client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# 1. a flag count belongs to the half it is reported under
# ---------------------------------------------------------------------------

def _item(qid: str, half: str, flagged: bool, spec) -> dict:
    return {"cid": f"c{qid[:6]}", "qid": qid, "half": half, "doc_hash": qid[:8], "id": "i",
            "category": TOPIC, "answer_words": 20, "meta": {"acuity": "mild"},
            "score": 0 if flagged else 3, "graded": True,
            "criteria": {c: 0.5 for c in jd.criteria_ids(spec)},
            "flags": {f: flagged for f in jd.flag_ids(spec)}}


def test_the_flag_count_is_recorded_per_half_and_sums_to_the_whole_bank():
    """The topic page read "55 questions, mean 0.95 / 4 · 45 critical economic
    error" — but 27 of those 45 were on the diagnose half. The published
    score's line now carries the published half's count."""
    spec = jd.rubric_for(TASK).criteria
    fid = jd.flag_ids(spec)[0]
    items = ([_item(f"a{i:063d}", "report", i < 18, spec) for i in range(55)]
             + [_item(f"b{i:063d}", "diagnose", i < 27, spec) for i in range(45)])
    blocks = jd._criteria_blocks(items, spec)
    f = blocks["flags"][fid]
    assert f["n"] == 45                                  # the whole bank, as before
    assert f["n_report"] == 18 and f["n_diagnose"] == 27
    assert f["n_report"] + f["n_diagnose"] == f["n"]     # the two sum to the whole
    assert f["share_report"] == round(18 / 55, 4)
    assert f["share_diagnose"] == round(27 / 45, 4)
    # and the qids named are still the diagnose half's alone
    assert len(f["qids"]) == 27
    assert all(q.startswith("b") for q in f["qids"])


def test_the_answers_endpoint_reports_the_report_halfs_own_count(svc):
    client, _, _ = svc
    from service import config
    model_dir = config.OUT_DIR / "fx__good-750m"
    j = json.loads((model_dir / "judge.json").read_text(encoding="utf-8"))
    t = j["tasks"][TASK]
    fid = next(iter(t["flags"]))
    # plant a flag on one of each half, through the file the page reads
    for it in t["items"]:
        it["flags"] = {fid: False}
    rep = next(it for it in t["items"] if it["half"] == "report")
    dia = next(it for it in t["items"] if it["half"] == "diagnose")
    rep["flags"][fid] = dia["flags"][fid] = True
    t["flags"][fid].update({"n": 2, "n_report": 1, "n_diagnose": 1})
    (model_dir / "judge.json").write_text(json.dumps(j), encoding="utf-8")
    got = client.get("/api/answers", params={"model": "fx/good-750m", "topic": TOPIC}).json()
    assert got["report_half"]["flags"][fid] == 1                 # this half
    assert got["report_half"]["flags_whole_bank"][fid] == 2      # and the bank, said as such
    # the rule the panel exists for is unchanged: no report-half qid anywhere
    body = json.dumps(got)
    assert rep["qid"] not in body


def test_a_judge_json_from_this_repos_own_run_adds_up(tree):
    """Over the fixture's real judged files: every flag's halves sum to its
    whole-bank count, and the report-half count never exceeds the report
    half."""
    checked = 0
    for m in tree["models"].values():
        jf = m["dir"] / "judge.json"
        if not jf.exists():
            continue
        j = json.loads(jf.read_text(encoding="utf-8"))
        for t in j["tasks"].values():
            for f in (t.get("flags") or {}).values():
                assert f["n_report"] + f["n_diagnose"] == f["n"]
                assert f["n_report"] <= (t.get("n_report") or 0)
                assert f["n_diagnose"] <= (t.get("n_diagnose") or 0)
                checked += 1
    assert checked > 3


# ---------------------------------------------------------------------------
# 3 and 4. whose batch is this, and is it still running
# ---------------------------------------------------------------------------

def test_each_judged_row_carries_its_own_batch(svc, monkeypatch):
    """#45 (one topic, 130 answers) showed #46's batch of 680: the lookup was
    "the newest judge run of this model", which is a different thing the
    moment two runs of one model are in flight."""
    client, _, _ = svc
    from service import config, db
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    first = client.post("/api/submissions", json={"hf_id": "fx/good-750m", "suite": "judged",
                                                  "tasks": [LAW]}).json()["id"]
    second = client.post("/api/submissions", json={"hf_id": "fx/good-750m",
                                                   "suite": "judged"}).json()["id"]
    # one topic and the whole exam are two jobs, not one row joined twice
    assert first != second
    r1 = db.judge_run_create("fx/good-750m", "batch_one", 130, "stub/overlap-v1", "{}")
    db.batch_add("batch_one", "judge", r1, 130, "local", "chat")
    db.batch_progress("batch_one", "130/130 done")
    db.update(first, judge_batch="batch_one")
    r2 = db.judge_run_create("fx/good-750m", "batch_two", 680, "stub/overlap-v1", "{}")
    db.batch_add("batch_two", "judge", r2, 680, "local", "chat")
    db.batch_progress("batch_two", "475/680 done")
    db.update(second, judge_batch="batch_two")
    rows = {r["id"]: r for r in client.get("/api/submissions").json()}
    assert rows[first]["judge"]["batch_id"] == "batch_one"
    assert rows[first]["judge"]["n_items"] == 130
    assert rows[first]["judge"]["progress"] == "130/130 done"
    assert rows[second]["judge"]["batch_id"] == "batch_two"
    assert rows[second]["judge"]["n_items"] == 680


def test_a_finished_judge_run_stops_counting_and_says_so(svc, monkeypatch):
    """#46 read as running for hours after judge.json had landed, with the
    count frozen at 475/680."""
    client, _, tree = svc
    from service import config, db, llm, llm_poller
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-1")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    monkeypatch.setattr(llm.FakeBatches, "polls_to_done", 1)
    llm.reset()
    sid = client.post("/api/submissions", json={"hf_id": "fx/good-750m",
                                                "suite": "judged"}).json()["id"]
    model_dir = config.OUT_DIR / "fx__good-750m"
    reqs, plan = jd.plan_requests(model_dir, "stub")
    backend = llm.client("judge")
    bid = backend.submit(reqs)
    rid = db.judge_run_create("fx/good-750m", bid, len(reqs), "stub/overlap-v1",
                              json.dumps(plan))
    db.batch_add(bid, "judge", rid, len(reqs), backend.name, backend.model)
    db.batch_progress(bid, f"{len(reqs) // 2}/{len(reqs)} done")       # a half-way poll
    db.update(sid, judge_batch=bid, status="running",
              progress=f"judge batch {bid} submitted ({len(reqs)} answers); "
                       f"judge.json lands when it completes")
    assert llm_poller.tick() == 1
    row = next(r for r in client.get("/api/submissions").json() if r["id"] == sid)
    # the row says what happened, not what was happening
    assert "judge batch" not in row["progress"]
    assert row["progress"].startswith("judged: ") and "judge.json written" in row["progress"]
    assert "topics" in row["progress"]
    # and the count landed where the batch did
    assert row["judge"]["progress"] == f"{len(reqs)}/{len(reqs)} done"
    assert row["judge"]["status"] == "done"


# ---------------------------------------------------------------------------
# 6. provenance: who wrote them, and which file they came from
# ---------------------------------------------------------------------------

def test_a_source_is_never_the_topics_own_name(tmp_path):
    """Two of the five banks recorded "economics" and "physics & engineering"
    as their source, which says nothing about where the questions came from."""
    assert eb.source_for("economics", "Economics", "economics.json") == "import"
    assert eb.source_for("Economics", "Economics", "economics.json") == "import"
    assert eb.source_for("", "Economics", "economics_v1.json") == "economics_v1"
    assert eb.source_for("physics_astronomy", "Physics & Astronomy", "x.json") == "x"
    assert eb.source_for("law_v2", "Law", "law_v2.json") == "law_v2"
    assert eb.source_for("", "Law", "") == "import"
    root = tmp_path / "exam"
    out = eb.import_bank(root, RETIRED / "economics_v1.json", "Economics",
                         "Dr. Hossein", "economics")          # the topic, as the page sent it
    assert out["source"] == "economics_v1"                    # the file's own name instead
    assert {r["source"] for r in eb.load_bank(root)["Economics"]} == {"economics_v1"}


def test_the_import_records_the_author_and_who_ran_it(svc):
    client, _, _ = svc
    from service import config, db
    items = json.loads((RETIRED / "law_v2.json").read_text("utf-8"))[:5]
    body = {"topic": "Law", "approver": "Dr. Hossein", "imported_by": "masein",
            "source": "", "filename": "law_v2.json", "items": items}
    pre = client.post("/api/exam/import/preview", json=body).json()
    # the preview says what will be written, before a hundred records carry it
    assert pre["approver"] == "Dr. Hossein" and pre["imported_by"] == "masein"
    assert pre["source"] == "law_v2"
    got = client.post("/api/exam/import", json=body).json()
    assert got["source"] == "law_v2"
    mine = [r for r in eb.load_bank(config.EXAM_DIR)["Law"] if r.get("source") == "law_v2"]
    # one of these five is already in the fixture's own law bank, and a
    # matching qid is skipped — that is the qid doing its job
    assert len(mine) == got["imported"] and got["imported"] + got["skipped"] == 5
    assert {r["accepted_by"] for r in mine} == {"Dr. Hossein"}      # who wrote them
    assert {r.get("imported_by") for r in mine} == {"masein"}       # who put them in
    row = db.curation_list(5)[0]
    assert row["approver"] == "masein" and "written by Dr. Hossein" in row["reason"]


def test_a_second_import_does_not_overwrite_a_recorded_source(tmp_path):
    """The file has not changed, so neither has what it is: an import that
    says nothing about the source must not replace one that said something."""
    root = tmp_path / "exam"
    src = RETIRED / "law_v2.json"
    eb.import_bank(root, src, "Law", "Dr. Hossein", "law_v2")
    before = {r["qid"]: (r["source"], r["accepted_by"]) for r in eb.load_bank(root)["Law"]}
    again = eb.import_bank(root, src, "Law", "someone else", "")
    assert (again["imported"], again["updated"], again["skipped"]) == (0, 0, 100)
    after = {r["qid"]: (r["source"], r["accepted_by"]) for r in eb.load_bank(root)["Law"]}
    assert after == before


def test_correcting_a_bank_in_place_never_moves_a_question(tmp_path):
    """The fix for the two banks already written. The qid is sha256 of the
    prompt AND it decides the report/diagnose split: recomputing one here
    would re-roll the split, orphan every judged result and quietly change
    what five published scores are about. Two fields move; nothing else."""
    root = tmp_path / "exam"
    eb.import_bank(root, RETIRED / "economics_v1.json", "Economics",
                   "masein", "economics")
    rows = eb.load_bank(root)["Economics"]
    qids_before = [r["qid"] for r in rows]
    halves_before = {r["qid"]: eb.half_of(r["qid"]) for r in rows}
    bodies_before = {r["qid"]: (r["prompt"], r["reference"], json.dumps(r["meta"], sort_keys=True))
                     for r in rows}
    out = eb.set_provenance(root, "Economics", source="economics_v1", approver="Dr. Hossein")
    assert out["changed"] == 100 and out["rows"] == 100
    after = eb.load_bank(root)["Economics"]
    assert [r["qid"] for r in after] == qids_before          # byte-identical, in order
    assert {r["qid"]: eb.half_of(r["qid"]) for r in after} == halves_before
    assert {r["qid"]: (r["prompt"], r["reference"], json.dumps(r["meta"], sort_keys=True))
            for r in after} == bodies_before
    assert {r["source"] for r in after} == {"economics_v1"}
    assert {r["accepted_by"] for r in after} == {"Dr. Hossein"}
    # and it refuses to write the topic's own name as a source
    with pytest.raises(ValueError, match="says nothing about where"):
        eb.set_provenance(root, "Economics", source="economics")
    with pytest.raises(ValueError, match="says nothing about where"):
        eb.set_provenance(root, "Economics", source="Economics")
    # only the rows that carry a given source, when asked
    eb.set_provenance(root, "Economics", source="mixed", only_source="nothing-matches-this")
    assert {r["source"] for r in eb.load_bank(root)["Economics"]} == {"economics_v1"}


def test_the_cli_corrects_a_bank_and_says_what_it_did(tmp_path):
    import subprocess
    import sys
    root = tmp_path / "exam"
    eb.import_bank(root, RETIRED / "economics_v1.json", "Economics",
                   "masein", "economics")
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "exam_build.py"),
                        "--root", str(root), "set-source", "--topic", "Economics",
                        "--source", "economics_v1", "--approver", "Dr. Hossein"],
                       capture_output=True, text=True, timeout=120, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "100 of 100 rows updated" in r.stdout and "Dr. Hossein" in r.stdout
    assert {x["accepted_by"] for x in eb.load_bank(root)["Economics"]} == {"Dr. Hossein"}
    bad = subprocess.run([sys.executable, str(REPO / "scripts" / "exam_build.py"),
                          "--root", str(root), "set-source", "--topic", "Economics"],
                         capture_output=True, text=True, timeout=120, cwd=REPO)
    assert bad.returncode == 2 and "nothing to set" in bad.stderr
