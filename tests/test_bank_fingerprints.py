"""Phase 10b: results belong to the questions they were given on.

On 2026-09-21 #47 ran SmolLM2-360M, judged, on economics: 0 GPU seconds. The
runner saw exam_economics_0shot/ on disk and skipped the task, and the judge
re-graded #46's answers. Right for an unchanged exam — and after the 37-topic
exam a silent error: exam_law, exam_economics and exam_computer_science keep
their names over different questions. So the build fingerprints each task's
question set, the runner writes the same value beside the answers, and a
result counts only while the two agree."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import exam_build as eb
import judge as jd
from conftest import fresh, make_service

REPO = Path(__file__).resolve().parents[1]
MODEL = "fx/good-750m"
SAFE = MODEL.replace("/", "__")
ECON, LAW = "exam_economics", "exam_law"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    from service import config, llm
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-1")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    llm.reset()
    yield client, appmod, tree
    client.__exit__(None, None, None)


@pytest.fixture
def plain(tmp_path, monkeypatch):
    """The service as the fixture leaves it: the judge that graded the
    fixture is the judge it runs, so a proposal's gate can pass."""
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, appmod, tree
    client.__exit__(None, None, None)


class Harness:
    """lm_eval, as far as the runner can tell: answers every item of the
    built task, writes results and samples where lm_eval would, and records
    that it ran — which is what spending GPU time is."""

    def __init__(self, monkeypatch):
        from service import config, runner
        self.calls: list[str] = []
        self.config = config
        monkeypatch.setattr(runner, "_run_task", self.run)
        monkeypatch.setattr(runner, "preflight", lambda *a, **k: {
            "kind": "instruct", "params": 750_000_000, "vocab": 32000, "batch": 8,
            "need_gb": 2.0, "archinfo": {}, "remote_code": False})
        monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
        monkeypatch.setattr(runner, "release_lock", lambda: None)
        monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
        # the results file the fixture wrote for this model, as the shape a
        # new one takes: the report reads the model's id out of it
        src = next((config.OUT_DIR / SAFE / f"{ECON}_0shot").rglob("results_*.json"))
        self.results = json.loads(src.read_text(encoding="utf-8"))
        self.subdir = src.parent.name

    def run(self, sid, cmd, lf, env, run_as, cwd) -> int:
        task = cmd[cmd.index("--tasks") + 1]
        # 12a.3: lm_eval starts in the task's own output folder, not BENCH_ROOT
        assert cwd == Path(cmd[cmd.index("--output_path") + 1])
        out = Path(cmd[cmd.index("--output_path") + 1]) / self.subdir
        out.mkdir(parents=True, exist_ok=True)
        items = [json.loads(x) for x in (self.config.JUDGED_TASKS_DIR / f"{task}.jsonl")
                 .read_text(encoding="utf-8").splitlines() if x.strip()]
        ts = time.strftime("%Y-%m-%dT%H-%M-%S.000000")
        with open(out / f"samples_{task}_{ts}.jsonl", "w", encoding="utf-8") as fh:
            for i, it in enumerate(items):
                ans = f"An answer about {it['prompt'][:40]}"
                fh.write(json.dumps({"doc_id": i, "doc": it, "doc_hash": it["qid"],
                                     "resps": [[ans]], "filtered_resps": [ans]}) + "\n")
        blob = {**self.results, "results": {task: {"alias": task, "bypass,none": 999}}}
        (out / f"results_{ts}.json").write_text(json.dumps(blob), encoding="utf-8")
        self.calls.append(task)
        time.sleep(0.05)
        return 0


def sit(client, tasks: list[str]) -> dict:
    from service import db, runner
    sid = client.post("/api/submissions", json={"hf_id": MODEL, "suite": "judged",
                                                "tasks": tasks}).json()["id"]
    runner.run_submission(db.get(sid))
    return db.get(sid)


def rebuild(tree):
    from service import config
    return eb.build(tree["out_dir"], config.EXAM_DIR)


def add_question(topic: str = "Economics"):
    from service import config
    eb.import_bank(config.EXAM_DIR, [{"prompt": "Why does a price ceiling below the market "
                                                "price create a shortage rather than a surplus?",
                                      "reference": "Quantity demanded exceeds supplied."}],
                   topic, "masein", "one_more")


# ---------------------------------------------------------------------------
# the build and the runner
# ---------------------------------------------------------------------------

def test_the_build_fingerprints_every_task_it_writes(svc):
    from service import config
    _, _, tree = svc
    m = rebuild(tree)
    rows = {r["qid"] for r in eb.load_bank(config.EXAM_DIR)["Economics"]}
    assert m["tasks"][ECON]["bank_sha256"] == eb.fingerprint(rows)
    assert m["tasks"][eb.CONTROL_TASK]["bank_sha256"]
    assert eb.current_fingerprints(config.JUDGED_TASKS_DIR)[ECON] == m["tasks"][ECON]["bank_sha256"]
    # the same bank, built again: the same fingerprint
    assert rebuild(tree)["tasks"][ECON]["bank_sha256"] == m["tasks"][ECON]["bank_sha256"]
    # one question more: a different question set
    add_question()
    assert rebuild(tree)["tasks"][ECON]["bank_sha256"] != m["tasks"][ECON]["bank_sha256"]


def test_one_question_more_and_the_topic_is_answered_again(svc, monkeypatch):
    from service import config
    client, _, tree = svc
    h = Harness(monkeypatch)
    old = config.OUT_DIR / SAFE / f"{ECON}_0shot"
    before = jd.answered_fingerprint(config.OUT_DIR / SAFE, ECON)
    add_question()
    fp = rebuild(tree)["tasks"][ECON]["bank_sha256"]
    assert before != fp
    row = sit(client, [ECON])
    assert h.calls == [ECON] and row["gpu_seconds"] > 0          # answered, on the GPU
    assert row["status"] == "done" and "reused" not in row["progress"]
    # beside the new answers: which questions they answer, and which row asked
    assert (old / "bank.sha256").read_text().strip() == fp
    assert json.loads((old / "answered_by.json").read_text())["submission"] == row["id"]
    assert jd.answered_fingerprint(config.OUT_DIR / SAFE, ECON) == fp
    # the answers to the old questions were kept, out of the results tree
    kept = list((config.OUT_DIR.with_name("earlier") / SAFE).glob(f"{ECON}_0shot-*"))
    assert len(kept) == 1 and list(kept[0].rglob("samples_*.jsonl"))
    assert not any(p.name.startswith(f"{ECON}_0shot-") for p in (config.OUT_DIR / SAFE).iterdir())


def test_the_same_bank_rebuilt_reuses_the_answers_and_the_row_says_so(svc, monkeypatch):
    client, _, tree = svc
    h = Harness(monkeypatch)
    add_question()
    rebuild(tree)
    first = sit(client, [ECON])
    assert h.calls == [ECON]
    rebuild(tree)                                              # nothing changed
    again = sit(client, [ECON])
    assert h.calls == [ECON]                                   # not answered twice
    assert again["gpu_seconds"] == 0 and again["status"] == "done"
    note = f"answers reused from #{first['id']} (same questions) · re-graded"
    assert again["reuse_note"] == note and note in again["progress"]
    # and the line the row keeps once the judge lands
    from service import llm_poller
    llm_poller.tick()
    rows = {r["id"]: r for r in client.get("/api/submissions").json()}
    assert rows[again["id"]]["progress"].startswith(note + " · judged: Economics")


def test_a_harness_task_resumes_on_its_name_as_always(svc):
    from service import config, runner
    out = config.OUT_DIR / SAFE / "hellaswag_10shot"
    assert runner.current_fingerprint("hellaswag") is None
    assert runner._task_done(out, "hellaswag") is runner._has_results(out)


# ---------------------------------------------------------------------------
# what counts: only a grade on the question set the task holds now
# ---------------------------------------------------------------------------

def _judge_file(tree, model=MODEL) -> Path:
    return tree["models"][model]["dir"] / "judge.json"


def _as_before_10b(tree, task=LAW, model=MODEL, bank=None):
    """The judge.json of a model graded on `task` before fingerprints: no
    `bank_sha256` — or, with `bank`, graded on another question set."""
    p = _judge_file(tree, model)
    j = json.loads(p.read_text(encoding="utf-8"))
    t = j["tasks"][task]
    t.pop("bank_sha256", None)
    t.pop("topic", None)
    for it in t.get("items") or []:
        it["category"] = "law"                                   # the retired topic's name
    if bank:
        t["bank_sha256"] = bank
    p.write_text(json.dumps(j), encoding="utf-8")


def test_every_judged_topic_records_the_question_set_it_was_graded_on(svc):
    from service import config
    _, _, tree = svc
    j = json.loads(_judge_file(tree).read_text(encoding="utf-8"))
    cur = eb.current_fingerprints(config.JUDGED_TASKS_DIR)
    for task, t in j["tasks"].items():
        assert t["bank_sha256"] == cur[task], task
        assert t["topic"] == (eb.TASK_TOPIC.get(task) or jd.CONTROL_LABEL)


# 12i.1: on `plain`, whose judge is the one that graded the fixture — under
# another judge every judged score is in History, whatever its question set
def test_a_pre_10b_entry_is_history_not_the_current_law_result(plain):
    client, appmod, tree = plain
    _as_before_10b(tree)
    fresh(appmod)
    m = next(x for x in client.get("/api/results").json()["models"] if x["id"] == MODEL)
    assert LAW not in m["judge"]["tasks"]                       # not a Law score
    assert ECON in m["judge"]["tasks"]                          # the rest still counts
    hist = [e for e in m["judge"]["history"] if e["task"] == LAW]
    assert len(hist) == 1 and hist[0]["topic"] == "law" and hist[0]["judged_at"]
    assert hist[0]["score_report"] is not None and "items" not in hist[0]
    # the Loop board: this model has not sat the current Law
    j = client.get("/api/loop", params={"model": MODEL}).json()
    row = next(r for r in j["topics"] if r["topic"] == "Law")
    assert row["last_judged"] is None
    # and the Leaderboard's judged columns and average are over what counts
    assert m.get("judgedAvg") is None or LAW not in (m.get("judgedTopics") or [])


def test_a_grade_on_another_question_set_is_history_too(plain):
    client, appmod, tree = plain
    _as_before_10b(tree, bank="0" * 64)
    fresh(appmod)
    m = next(x for x in client.get("/api/results").json()["models"] if x["id"] == MODEL)
    assert LAW not in m["judge"]["tasks"]
    assert [e["bank_sha256"] for e in m["judge"]["history"] if e["task"] == LAW] == ["0" * 64]


def test_the_answers_for_the_new_law_are_nothing_for_a_model_that_sat_only_the_retired(svc):
    client, appmod, tree = svc
    ok = client.get("/api/answers", params={"model": MODEL, "topic": "Law"})
    assert ok.status_code == 200 and ok.json()["items"]
    _as_before_10b(tree)
    fresh(appmod)
    r = client.get("/api/answers", params={"model": MODEL, "topic": "Law"})
    assert r.status_code == 404
    assert "earlier question set" in r.json()["detail"]
    # and no proposal can be built from it either
    from service import config
    from service import proposals as prop
    assert LAW not in {w["task"] for w in prop.weak_topics(config.OUT_DIR / SAFE)}


def test_a_new_run_keeps_the_old_grade_as_history_instead_of_replacing_it(svc):
    """The retired law's grade and the new Law's live side by side in
    judge.json: one counts, the other is what the model scored before."""
    from service import config
    _, _, tree = svc
    _as_before_10b(tree)
    mdir = config.OUT_DIR / SAFE
    out = jd.run_stub(mdir, tree["out_dir"], only=[LAW])
    merged = jd.merge_judged(mdir, out)
    assert merged["tasks"][LAW]["bank_sha256"] == eb.current_fingerprints(
        config.JUDGED_TASKS_DIR)[LAW]
    assert [e["task"] for e in merged["history"]] == [LAW]
    jd.write_judge(mdir, merged)
    # merging again does not duplicate it, and it survives a run of another topic
    again = jd.merge_judged(mdir, jd.run_stub(mdir, tree["out_dir"], only=[ECON]))
    assert [e["task"] for e in again["history"]] == [LAW]
    assert LAW in again["tasks"] and ECON in again["tasks"]


def test_a_topic_the_exam_no_longer_has_moves_to_history_on_the_next_merge(svc):
    from service import config
    _, _, tree = svc
    mdir = config.OUT_DIR / SAFE
    j = json.loads(_judge_file(tree).read_text(encoding="utf-8"))
    j["tasks"]["exam_medicine_health"] = {**j["tasks"][LAW], "topic": "medicine & health"}
    _judge_file(tree).write_text(json.dumps(j), encoding="utf-8")
    merged = jd.merge_judged(mdir, jd.run_stub(mdir, tree["out_dir"], only=[ECON]))
    assert "exam_medicine_health" not in merged["tasks"]
    assert [e["topic"] for e in merged["history"]] == ["medicine & health"]


# ---------------------------------------------------------------------------
# the daily limit is per provider
# ---------------------------------------------------------------------------

def test_a_3870_item_local_judge_batch_leaves_propose_open(plain, monkeypatch):
    """One judged run of the 37-topic exam is ~3,870 items. Against one
    shared cap it refused every Propose that day."""
    client, _, _ = plain
    from service import config, db
    db.batch_add("local_judge_1", "judge", 1, 3870, "local", "chat")
    r = client.post("/api/proposals", json={"model": MODEL, "topic": "Economics",
                                            "requested_by": "tester"})
    assert r.status_code == 200, r.text
    # the same, with the generator on local too: local has no limit unless set
    monkeypatch.setattr(config, "LLM_PROVIDER", "local")
    assert config.daily_cap("local") is None
    from service import app as appmod
    appmod._spend_check(1)
    monkeypatch.setattr(config, "LOCAL_DAILY_ITEM_CAP", 3870)
    with pytest.raises(Exception) as e:
        appmod._spend_check(1)
    assert "LOCAL_DAILY_ITEM_CAP=3870" in str(getattr(e.value, "detail", e.value))


def test_a_paid_provider_keeps_its_cap_and_counts_only_its_own_items(svc, monkeypatch):
    from service import app as appmod
    from service import config, db
    monkeypatch.setattr(config, "LLM_DAILY_ITEM_CAP", 100)
    db.batch_add("other_1", "judge", 1, 5000, "anthropic", "claude")    # the judge's provider
    appmod._spend_check(100)                                             # fake's own: 0 so far
    db.batch_add("fake_1", "generation", 1, 100, "fake", "fake-1")
    with pytest.raises(Exception) as e:
        appmod._spend_check(1)
    assert "LLM_DAILY_ITEM_CAP=100" in str(getattr(e.value, "detail", e.value))


def test_the_review_tab_shows_use_per_provider(svc, monkeypatch):
    client, _, _ = svc
    from service import config, db
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "local")
    db.batch_add("local_judge_2", "judge", 1, 3870, "local", "chat")
    db.batch_add("fake_2", "proposal", 1, 1, "fake", "fake-1")
    u = {x["provider"]: x for x in client.get("/api/llm").json()["usage"]}
    assert u["local"]["items"] == 3870 and u["local"]["cap"] is None
    assert "judge" in u["local"]["roles"]
    assert u["fake"]["items"] == 1 and u["fake"]["cap"] == config.LLM_DAILY_ITEM_CAP
    assert client.get("/api/llm").json()["usage_today"] == 1          # the generator's own
