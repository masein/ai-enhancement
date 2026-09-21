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
from conftest import fresh, make_service

REPO = Path(__file__).resolve().parents[1]
MODEL = "fx/good-750m"
ECON, LAW, MED = "exam_economics", "exam_law", "exam_medicine_health"


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
    assert re.fullmatch(r"judged: economics, judge.json written \d\d:\d\d", row["progress"]), \
        row["progress"]
    assert row["judge"]["status"] == "done"


def test_the_finished_line_names_up_to_three_topics_then_counts():
    from service.llm_poller import judged_what

    def run(*tasks):
        return {"plan": json.dumps({"tasks": {t: {} for t in tasks}})}
    assert judged_what(run(ECON)) == "economics"
    assert judged_what(run(LAW, MED)) == "law and medicine & health"
    assert judged_what(run(LAW, MED, ECON, jd.CONTROL_TASK)) == \
        "law, medicine & health and economics"                      # the control is not a topic
    assert judged_what(run(LAW, MED, ECON, "exam_computer_science",
                           "exam_physics_engineering")) == "5 topics"
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
    for task in (LAW, MED, "exam_computer_science", "exam_physics_engineering"):
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
    five = [ECON, LAW, MED, "exam_computer_science", "exam_physics_engineering"]
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
    fail_results = {"n": 0}
    failed: list[str] = []

    def serve(route):
        req = route.request
        path = req.url.split("board.test", 1)[1] or "/"
        if path.startswith("/api/results") and fail_results["n"]:
            fail_results["n"] -= 1
            failed.append(path)
            return route.fulfill(status=503, content_type="application/json",
                                 body='{"detail":"restarting"}')
        fresh(appmod)                                      # past the debounce, as minutes would be
        r = client.request(req.method, path.split("#")[0], content=req.post_data_buffer,
                           headers={"content-type": req.headers.get("content-type", "")})
        route.fulfill(status=r.status_code, body=r.content,
                      content_type=r.headers.get("content-type", "application/json"))

    ctx = browser.new_context(viewport={"width": 1240, "height": 900})
    pg = ctx.new_page()
    loads, errors = [], []
    pg.on("load", lambda _: loads.append(1))
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.route("https://board.test/**", serve)
    try:
        pg.goto("https://board.test/#tab=models")
        row = pg.locator(f"[data-model-row='{MODEL}']")
        row.wait_for(timeout=20000)
        judged = row.locator("td").nth(5)                  # 'judged topics'
        assert judged.text_content().strip() == "—"

        sid = submit(client, [ECON])
        pg.wait_for_function(f"state.queue.some(r => r.id === {sid})", timeout=20000)
        run_judged(monkeypatch, sid)                       # answers in, judge batch out
        pg.wait_for_function(f"state.queue.some(r => r.id === {sid} && r.status === 'done' "
                             "&& r.judge && r.judge.status !== 'done') && !RESULTS_DUE",
                             timeout=20000)
        assert judged.text_content().strip() == "—"       # the answers alone grade nothing

        fail_results["n"] = 1                              # the next results fetch fails once
        assert llm_poller.tick() == 1                      # judge.json lands
        # the judged-topics cell: its count, and each topic's score in its title
        pg.wait_for_function(
            f"(document.querySelector(\"[data-model-row='{MODEL}']\")"
            ".querySelectorAll('td')[5].querySelector('span[title]') || {}).title", timeout=40000)
        assert failed, "the failed fetch never happened — the retry was not exercised"
        assert judged.text_content().strip().startswith("1")     # one topic judged
        assert re.fullmatch(r"economics \d\.\d\d",
                            judged.locator("span[title]").first.get_attribute("title"))
        assert len(loads) == 1                             # without a reload
        assert errors == []
    finally:
        ctx.close()
