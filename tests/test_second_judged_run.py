"""What the second judged run (#47, economics only, on 29fcf5d) showed.

Five things #32 left wrong: every judged row of a model still carried the
newest batch, an open page never saw the judge land, a one-topic run said
"5 topics", a one-topic merge re-dated every topic in the file, and the
exam_build commands in SERVICE.md did not parse.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from pathlib import Path

import pytest

import exam_build as eb
import judge as jd
from conftest import choose, fresh, make_service

REPO = Path(__file__).resolve().parents[1]
MODEL = "fx/good-750m"
ECON, LAW, MED = "exam_economics", "exam_law", "exam_medicine_clinical_health"
PHYS = "exam_physics_astronomy"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    from service import config, llm
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-1")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    llm.reset()
    yield client, appmod, tree
    client.__exit__(None, None, None)


def run_judged(monkeypatch, sid: int) -> None:
    """The real runner on a submission whose answers are already on disk (the
    fixture has every task), so it goes straight to submitting the judge."""
    from service import db, runner
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: {
        "kind": "instruct", "params": 750_000_000, "vocab": 32000, "batch": 8,
        "need_gb": 2.0, "archinfo": {}, "remote_code": False})
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    runner.run_submission(db.get(sid))


def submit(client, tasks: list[str] | None = None) -> int:
    body = {"hf_id": MODEL, "suite": "judged", **({"tasks": tasks} if tasks else {})}
    return client.post("/api/submissions", json=body).json()["id"]


def rows(client) -> dict[int, dict]:
    return {r["id"]: r for r in client.get("/api/submissions").json()}


# ---------------------------------------------------------------------------
# 1. a row's judge is the batch that row submitted, and nothing else
# ---------------------------------------------------------------------------

def test_two_judged_runs_of_one_model_each_report_only_their_own_batch(svc, monkeypatch):
    """After #47, #45, #46 and #47 all showed #47's batch (130, 130/130)."""
    client, _, _ = svc
    from service import db
    seen_on_row: list[str] = []
    real_add = db.batch_add

    def batch_add(bid, kind, *a, **k):
        # the batch is on its row BEFORE the poller can see it
        if kind == "judge":
            seen_on_row.append((db.submission_of_batch(bid) or {}).get("judge_batch") or "")
        return real_add(bid, kind, *a, **k)
    monkeypatch.setattr(db, "batch_add", batch_add)

    law = submit(client, [LAW])
    run_judged(monkeypatch, law)
    econ = submit(client, [ECON])
    run_judged(monkeypatch, econ)
    got = rows(client)
    a, b = got[law]["judge"], got[econ]["judge"]
    assert a["batch_id"] != b["batch_id"]
    assert seen_on_row == [a["batch_id"], b["batch_id"]]
    # each is the batch of its own plan: law's answers, economics' answers
    assert a["n_items"] != b["n_items"]
    for sid, task in ((law, LAW), (econ, ECON)):
        run = next(r for r in db.judge_runs() if r["batch_id"] == got[sid]["judge"]["batch_id"])
        plan = json.loads(db.judge_run_get(run["id"])["plan"])
        assert set(plan["tasks"]) == {task}, sid
    # and the older row keeps its own after the newer one lands, too
    from service import llm_poller
    llm_poller.tick()
    got = rows(client)
    assert got[law]["judge"]["batch_id"] == a["batch_id"]
    assert got[econ]["judge"]["batch_id"] == b["batch_id"]


def test_rows_from_before_the_column_get_their_own_batch_back(svc):
    """The live #45 and #46 predate judge_batch; their progress text names
    their batch. They must read that — not "the model's newest run"."""
    client, _, _ = svc
    from service import config, db

    def judge_run(bid, n):
        rid = db.judge_run_create(MODEL, bid, n, "anthropic/claude-x", "{}")
        db.batch_add(bid, "judge", rid, n, "anthropic", "claude-x")
        db.batch_progress(bid, f"{n}/{n} done")
    # in the order it happened: each row, then the batch it submitted
    lost = submit(client, [MED])
    db.update(lost, status="done", progress="economics done")    # nothing to recover from
    s45 = submit(client, [ECON])
    judge_run("msgbatch_45", 130)
    db.update(s45, status="done", progress="economics done · judge batch msgbatch_45 submitted "
                                           "(130 answers); judge.json lands when it completes")
    s46 = submit(client)
    judge_run("msgbatch_46", 680)
    db.update(s46, status="done", progress="re-queued after restart")   # progress lost…
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (config.LOGS_DIR / f"service_{s46}_fx__good-750m.log").write_text(   # …the log has it
        "lm_eval output\n\n===== [%d] judge: %s =====\n" % (s46, json.dumps(
            {"mode": "batch", "written": False, "batch_id": "msgbatch_46", "run_id": 2, "n": 680})),
        encoding="utf-8")
    s47 = submit(client, [LAW])
    judge_run("msgbatch_47", 130)
    db.update(s47, status="done", judge_batch="msgbatch_47")
    db.init()                                                     # the migration, as on restart
    got = rows(client)
    assert got[s45]["judge"]["batch_id"] == "msgbatch_45"
    assert got[s46]["judge"]["batch_id"] == "msgbatch_46"
    assert got[s46]["judge"]["n_items"] == 680
    assert got[s47]["judge"]["batch_id"] == "msgbatch_47"
    # a row whose batch cannot be found shows none — the lookup by model gave
    # it msgbatch_47, the newest run that started after it did
    assert "judge" not in got[lost]


# ---------------------------------------------------------------------------
# 3. the finished line names what THIS run graded
# ---------------------------------------------------------------------------

def test_a_one_topic_run_says_which_topic_it_judged(svc, monkeypatch):
    """#47 read "judged: 5 topics" — the count of the merged file."""
    client, _, _ = svc
    from service import llm_poller
    sid = submit(client, [ECON])
    run_judged(monkeypatch, sid)
    assert llm_poller.tick() == 1
    row = rows(client)[sid]
    # the fixture's answers are on disk already, to the same questions, so
    # the run re-grades them — and says so before it says what it judged
    assert re.fullmatch(r"answers reused from an earlier run \(same questions\) · re-graded · "
                        r"judged: Economics, judge.json written \d\d:\d\d", row["progress"]), \
        row["progress"]
    assert row["judge"]["status"] == "done"


def test_the_finished_line_names_up_to_three_topics_then_counts():
    from service.llm_poller import judged_what

    def run(*tasks):
        return {"plan": json.dumps({"tasks": {t: {} for t in tasks}})}
    assert judged_what(run(ECON)) == "Economics"                  # the topic as stored
    assert judged_what(run(LAW, MED)) == "Law and Medicine & Clinical Health"
    assert judged_what(run(LAW, MED, ECON, jd.CONTROL_TASK)) == \
        "Law, Medicine & Clinical Health and Economics"             # the control is not a topic
    assert judged_what(run(LAW, MED, ECON, "exam_computer_science", PHYS)) == "5 topics"
    assert judged_what({"plan": "not json"}) == ""


# ---------------------------------------------------------------------------
# 4. every topic keeps the time ITS grades landed
# ---------------------------------------------------------------------------

def test_a_one_topic_merge_keeps_the_other_topics_times(svc, monkeypatch):
    """After #47 (economics, 09:25), law, medicine, computer science and
    physics also read 09:25 — their grades were from #46 the night before."""
    client, appmod, _ = svc
    from service import config, llm_poller
    model_dir = config.OUT_DIR / "fx__good-750m"
    last_night = int(time.time()) - 12 * 3600
    for other in config.OUT_DIR.glob("*/judge.json"):          # the board's "last judged"
        if other.parent != model_dir:                          # is then this model's
            other.unlink()
    # #46: the whole exam, by this judge; then pretend it landed last night
    run_judged(monkeypatch, submit(client))
    llm_poller.tick()
    j = json.loads((model_dir / "judge.json").read_text(encoding="utf-8"))
    for t in j["tasks"].values():
        t["judged_at"] = last_night
    jd.write_judge(model_dir, j)
    # #47: economics alone
    sid = submit(client, [ECON])
    run_judged(monkeypatch, sid)
    before = time.time()
    llm_poller.tick()
    after = json.loads((model_dir / "judge.json").read_text(encoding="utf-8"))
    assert after["tasks"][ECON]["judged_at"] >= int(before) - 1       # this run's topic: now
    for task in (LAW, MED, "exam_computer_science", PHYS):
        assert after["tasks"][task]["judged_at"] == last_night, task  # the rest: untouched
    fresh(appmod)
    loop = {r["slug"]: r for r in client.get("/api/loop").json()["topics"]}
    assert loop["law"]["last_judged"]["model"] == MODEL
    assert loop["law"]["last_judged"]["at"] == last_night
    assert loop["economics"]["last_judged"]["at"] >= int(before) - 1


def test_a_file_from_before_per_topic_times_lends_its_mtime_to_what_it_carries(tree, tmp_path):
    d = tree["models"][MODEL]["dir"]
    dest = tmp_path / "m"
    old = jd.run_stub(d, tree["out_dir"], record=False)
    for t in old["tasks"].values():
        t.pop("judged_at", None)
    p = jd.write_judge(d, old, dest)
    import os
    os.utime(p, (1_700_000_000, 1_700_000_000))
    out = jd.run_stub(d, tree["out_dir"], record=False, only=[LAW])
    merged = jd.merge_judged(d, out, dest, when=1_800_000_000)
    assert merged["tasks"][LAW]["judged_at"] == 1_800_000_000
    assert merged["tasks"][MED]["judged_at"] == 1_700_000_000


def test_startup_puts_back_the_times_an_old_merge_overwrote(svc):
    """The live file after #47: five topics, no per-topic times, mtime 09:25.
    The runs that graded them are in the database."""
    client, appmod, _ = svc
    from service import config, db, llm_poller
    model_dir = config.OUT_DIR / "fx__good-750m"
    j = json.loads((model_dir / "judge.json").read_text(encoding="utf-8"))
    five = [ECON, LAW, MED, "exam_computer_science", PHYS]
    j["tasks"] = {t: {k: v for k, v in j["tasks"][t].items() if k != "judged_at"} for t in five}
    j["judge"]["batch_id"] = "b47"
    jd.write_judge(model_dir, j)
    night, morning = 1_790_000_000.0, 1_790_040_000.0
    for bid, tasks, at in (("b45", [ECON], night - 3600), ("b46", five, night),
                           ("b47", [ECON], morning)):
        rid = db.judge_run_create(MODEL, bid, 130, "stub/overlap-v1",
                                  json.dumps({"tasks": {t: {} for t in tasks}}))
        db.judge_run_update(rid, status="done", finished_at=at)
    assert llm_poller.backfill_judged_at() == [MODEL]
    got = json.loads((model_dir / "judge.json").read_text(encoding="utf-8"))
    assert got["tasks"][ECON]["judged_at"] == int(morning)
    for t in five[1:]:
        assert got["tasks"][t]["judged_at"] == int(night), t
    # nothing but the times changed
    assert {t: {k: v for k, v in x.items() if k != "judged_at"} for t, x in got["tasks"].items()} \
        == j["tasks"]
    assert llm_poller.backfill_judged_at() == []                 # once, then settled


def test_a_file_the_service_did_not_write_last_is_left_alone():
    """Written by the CLI after the last service run: its mtime is right for
    all of it, and a run record would be wrong."""
    j = {"judge": {"batch_id": "cli-batch"}, "tasks": {LAW: {}, MED: {}}}
    runs = [{"batch_id": "b46", "finished_at": 1.0e9, "tasks": {LAW, MED}}]
    assert jd.backfill_judged_at(j, runs) == []
    assert "judged_at" not in j["tasks"][LAW]


# ---------------------------------------------------------------------------
# 5. the commands in the docs parse
# ---------------------------------------------------------------------------

def doc_commands() -> list[tuple[str, list[str]]]:
    """Every `exam_build.py` command in a fenced block of the docs this repo
    maintains (the phase briefs are inputs, kept as delivered)."""
    docs = [*REPO.glob("*.md"), *REPO.glob("docs/*.md"), *REPO.glob("eval_tasks/**/*.md")]
    out = []
    for p in docs:
        lines, fence, i = p.read_text(encoding="utf-8").splitlines(), False, 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("```"):
                fence = not fence
            elif fence and re.search(r"python3? \S*exam_build\.py", line):
                while line.rstrip().endswith("\\") and i + 1 < len(lines):
                    i += 1
                    line = line.rstrip()[:-1] + " " + lines[i].strip()
                args = shlex.split(line.split("exam_build.py", 1)[1], comments=True)
                out.append((f"{p.name}: {line.strip()}", args))
            i += 1
    return out


def test_every_exam_build_command_in_the_docs_parses():
    """SERVICE.md put --root after the subcommand; the parser only takes it
    before, so every one of those commands failed as written."""
    cmds = doc_commands()
    assert len(cmds) >= 5
    ap = eb.parser()
    for where, args in cmds:
        try:
            ap.parse_args(args)
        except SystemExit:
            pytest.fail(f"does not parse: {where}")


# ---------------------------------------------------------------------------
# 2. a page left open sees the judge land
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


class Served:
    """The real app behind a browser page, routed in-process: every request
    the page makes goes to the TestClient. `fail_results` makes the next N
    /api/results fetches fail, as a deploy or a blip would."""

    def __init__(self, browser, client, appmod):
        self.client, self.appmod = client, appmod
        self.fail_results, self.failed = 0, []
        self.ctx = browser.new_context(viewport={"width": 1240, "height": 900}, reduced_motion="reduce")
        self.page = self.ctx.new_page()
        self.loads, self.errors = [], []
        self.page.on("load", lambda _: self.loads.append(1))
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.route("https://board.test/**", self.serve)

    def serve(self, route):
        req = route.request
        path = req.url.split("board.test", 1)[1] or "/"
        if path.startswith("/api/results") and self.fail_results:
            self.fail_results -= 1
            self.failed.append(path)
            return route.fulfill(status=503, content_type="application/json",
                                 body='{"detail":"restarting"}')
        fresh(self.appmod)                                 # past the debounce, as minutes would be
        r = self.client.request(req.method, path.split("#")[0], content=req.post_data_buffer,
                                headers={"content-type": req.headers.get("content-type", "")})
        route.fulfill(status=r.status_code, body=r.content,
                      content_type=r.headers.get("content-type", "application/json"))

    def open(self, hash_):
        self.page.goto("https://board.test/" + hash_)
        return self.page


@pytest.mark.dashboard
def test_a_page_open_through_a_whole_judged_run_shows_the_judged_scores(svc, monkeypatch,
                                                                         browser):
    """The row went 'done' when the answers landed, the page refreshed then —
    before the judge — and nothing refreshed it when judge.json landed
    minutes later: the board kept the pre-judge numbers until a hard reload.
    The real app behind the page; one results fetch after the judge lands
    fails, as a deploy or a blip would make it, and the page must still get
    there."""
    client, appmod, _ = svc
    from service import config, llm_poller
    for f in config.OUT_DIR.glob("*/judge.json"):         # nothing judged yet, anywhere
        f.unlink()
    fresh(appmod)
    s = Served(browser, client, appmod)
    try:
        # 12b: the Models tab and its "judged topics" column are gone; the
        # model page's Judged block is where a person watches this land
        pg = s.open("#model=" + MODEL)
        pg.wait_for_selector("[data-sit-progress='0']", timeout=20000)    # 0 of 37 topics judged
        score = pg.locator("[data-topic-score='Economics']")
        assert score.count() == 0

        sid = submit(client, [ECON])
        pg.wait_for_function(f"state.queue.some(r => r.id === {sid})", timeout=20000)
        run_judged(monkeypatch, sid)                       # answers in, judge batch out
        pg.wait_for_function(f"state.queue.some(r => r.id === {sid} && r.status === 'done' "
                             "&& r.judge && r.judge.status !== 'done') && !RESULTS_DUE",
                             timeout=20000)
        assert score.count() == 0                          # the answers alone grade nothing
        assert pg.locator("[data-sit-progress='0']").count() == 1

        s.fail_results = 1                                 # the next results fetch fails once
        assert llm_poller.tick() == 1                      # judge.json lands
        score.wait_for(timeout=40000)
        assert s.failed, "the failed fetch never happened — the retry was not exercised"
        assert pg.locator("[data-sit-progress='1']").count() == 1          # one topic judged
        # two decimals at most, trailing zeros dropped: under its own criteria
        # economics can grade to a round 3
        assert re.fullmatch(r"\d(\.\d\d?)? / 4", score.text_content().strip())
        assert len(s.loads) == 1                           # without a reload
        assert s.errors == []
    finally:
        s.ctx.close()


@pytest.mark.dashboard
def test_an_open_topic_page_lists_a_model_judged_while_it_was_open(svc, monkeypatch, browser):
    """Omar's tab, 2026-09-21: #49 (SmolLM2-135M) landed on medicine and the
    answers panel's model select still listed only the 360M. The page learns
    the judge landed (#33); the select must then list both, say what each
    scored, and mark the new one once."""
    client, appmod, _ = svc
    from service import config, llm_poller
    other = "fx/skewed-360m"
    for f in config.OUT_DIR.glob("*/judge.json"):         # one model judged on economics
        if f.parent.name != "fx__good-750m":
            f.unlink()
    fresh(appmod)
    s = Served(browser, client, appmod)
    try:
        pg = s.open("#topic=economics")
        sel = pg.locator("[data-panel='answers'] [aria-label='model']")
        sel.wait_for(timeout=20000)
        # 11f: the model list is a Combobox: its options are there when it is open
        sel.click()
        opts = pg.locator("#pop-cb-answers-model [role=option]")
        assert [o.get_attribute("data-value") for o in opts.all()] == [MODEL]
        assert re.search(r"good-750m.*\d(\.\d\d?)? / 4", opts.first.text_content())
        pg.keyboard.press("Escape")
        sid = client.post("/api/submissions", json={"hf_id": other, "suite": "judged",
                                                    "tasks": [ECON]}).json()["id"]
        run_judged(monkeypatch, sid)
        assert llm_poller.tick() == 1
        # the runner and the judge both finish between two polls: the page first
        # sees this row already judged, and that counts as a run landing too
        pg.wait_for_function("document.querySelector('[data-new-model]')", timeout=40000)
        sel.click()
        values = sorted(o.get_attribute("data-value") for o in opts.all())
        assert values == sorted([MODEL, other])
        pg.keyboard.press("Escape")
        new = pg.locator("[data-new-model]")
        assert new.get_attribute("data-new-model") == other
        assert "new: " + other in new.text_content()
        # used once, it goes
        choose(sel, other)
        pg.wait_for_selector("[data-new-model]", state="detached")
        assert len(s.loads) == 1 and s.errors == []
    finally:
        s.ctx.close()


# ---------------------------------------------------------------------------
# after #33: rows whose batch finished under older code, and a repair that
# says what it did once
# ---------------------------------------------------------------------------

def _start(appmod):
    """One start of the service: the lifespan, in and out."""
    from fastapi.testclient import TestClient
    c = TestClient(appmod.app)
    c.__enter__()
    return c


def test_rows_whose_batch_finished_under_older_code_read_as_finished(tmp_path, monkeypatch,
                                                                     capsys):
    """Verified live after #33: #45 read "62/130 done" and #46 "475/680 done",
    both "judge.json lands when it completes", with both batches long done.
    62/130 reads as half the answers lost."""
    client, appmod, _ = make_service(tmp_path, monkeypatch)
    client.__exit__(None, None, None)
    from service import config, db, llm_poller
    import sqlite3
    with sqlite3.connect(config.DB_PATH) as c:            # a database from before this deploy
        c.execute("DELETE FROM repairs")

    def row(tasks, bid, n, seen, at, plan_tasks, status="done", batch="done", progress=None):
        sid = db.add(MODEL, "instruct", "judged", "masein", "", tasks=tasks)
        rid = db.judge_run_create(MODEL, bid, n, "stub/overlap-v1",
                                  json.dumps({"tasks": {t: {} for t in plan_tasks}}))
        if batch == "done":
            db.judge_run_update(rid, status="done", finished_at=at)
        db.batch_add(bid, "judge", rid, n, "anthropic", "claude-x")
        db.batch_progress(bid, seen)
        if batch != "pending":
            db.batch_finish(bid, batch, "")
        db.update(sid, status=status, judge_batch=bid, progress=progress or (
            f"done · judge batch {bid} submitted ({n} answers); judge.json lands when it completes"))
        return sid
    five = [ECON, LAW, MED, "exam_computer_science", PHYS]
    t45, t46 = time.mktime((2026, 9, 20, 21, 22, 0, 0, 0, -1)), time.mktime(
        (2026, 9, 20, 23, 5, 0, 0, 0, -1))
    s45 = row([ECON], "b45", 130, "62/130 done", t45, [ECON])
    s46 = row([], "b46", 680, "475/680 done", t46, five + [jd.CONTROL_TASK])
    s47 = row([ECON], "b47", 130, "130/130 done", t46 + 36000, [ECON],
              progress="judged: 5 topics, judge.json written 09:25")    # the old count
    out = row([LAW], "b48", 42, "18/42 done", 0, [LAW], batch="pending")
    bad = row([MED], "b49", 42, "3/42 done", 0, [MED], batch="failed")
    capsys.readouterr()

    c = _start(appmod)                                    # the deploy
    got = rows(c)
    assert got[s45]["judge"]["progress"] == "130/130 done"
    assert got[s45]["progress"] == "judged: Economics, judge.json written 21:22"
    assert got[s46]["judge"]["progress"] == "680/680 done"
    assert got[s46]["progress"] == "judged: 5 topics, judge.json written 23:05"
    assert got[s47]["progress"] == "judged: Economics, judge.json written " + \
        time.strftime("%H:%M", time.localtime(t46 + 36000))
    # a batch still out, or one that failed, says what it said
    assert got[out]["judge"]["progress"] == "18/42 done"
    assert "judge.json lands" in got[out]["progress"]
    assert got[bad]["judge"]["progress"] == "3/42 done"
    log = capsys.readouterr().out
    assert log.count("restored from the run records") == 1, log
    assert f"#{s45}, #{s46}, #{s47}" in log
    c.__exit__(None, None, None)

    # a restart, a second process: the record says it ran, and nothing runs
    called = []
    monkeypatch.setattr(llm_poller, "backfill_judged_at", lambda: called.append(1) or [])
    monkeypatch.setattr(llm_poller, "repair_finished_rows", lambda: called.append(1) or [])
    for _ in range(2):
        _start(appmod).__exit__(None, None, None)
    assert called == []
    assert "restored from the run records" not in capsys.readouterr().out


def test_the_repair_is_claimed_by_one_caller_and_retried_if_it_fails(svc, monkeypatch):
    from service import db, llm_poller
    db.release_repair(llm_poller.REPAIR)
    assert db.claim_repair("x") is True
    assert db.claim_repair("x") is False                   # the second process
    monkeypatch.setattr(llm_poller, "backfill_judged_at",
                        lambda: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        llm_poller.repair_once()
    calls = []
    monkeypatch.setattr(llm_poller, "backfill_judged_at", lambda: calls.append(1) or [])
    assert llm_poller.repair_once() == ""                  # the next start ran it again…
    assert calls == [1]
    assert llm_poller.repair_once() == ""                  # …and that one was the last
    assert calls == [1]
    assert db.claim_repair(llm_poller.REPAIR) is False     # recorded as run
