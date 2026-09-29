"""12p.1: the hidden half on the data volume, kept and checked.

Everyday's hidden set moves out of the repo into BENCH_ROOT/everyday/
hidden.jsonl — the same rows, marked hidden, so the bank's version and every
score stay as they are — and is backed up once a day to another disk, with an
encrypted export to copy off. What it should be is committed (a count and a
digest; the exam's report half by its qids). When a set is missing or changed,
the server still starts: every page says so in red with the command that
restores it, and only Everyday's runs and scoring stop. Every OpenRouter
request excludes providers that may store or train on prompts. Fixtures only;
nothing calls OpenRouter."""

from __future__ import annotations

import io
import json
import tarfile
import time

import pytest

import everyday as ev
from conftest import make_service
from fake_openrouter import FakeOpenRouter
from service import config, db, hidden_store, llm, llm_poller
from test_everyday_12a5 import _asked

REPO_MANIFEST = {k: v for k, v in json.loads(config.HIDDEN_MANIFEST.read_text(
    encoding="utf-8")).items() if k != "what"}


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    yield client, appmod
    client.__exit__(None, None, None)


def run(*argv) -> int:
    return hidden_store.main(list(argv))


# ---------------------------------------------------------------------------
# what is committed, and the move
# ---------------------------------------------------------------------------

def test_the_committed_manifest_is_the_repos_hidden_half():
    rows = ev.repo_hidden()
    assert REPO_MANIFEST == ev.manifest_of(rows)
    assert REPO_MANIFEST["count"] == len(rows) == sum(REPO_MANIFEST["groups"].values())
    assert all(q["half"] == ev.HIDDEN for q in rows)
    # counts and a digest only: no question, no id's words
    text = config.HIDDEN_MANIFEST.read_text(encoding="utf-8")
    assert not [q for q in rows if q["prompt"][:30] in text]


def test_the_exams_committed_report_half_is_the_banks():
    got = hidden_store.exam_manifest_of(ev.REPO / "eval_tasks" / "fr" / "banks")
    committed = json.loads(config.EXAM_REPORT_MANIFEST.read_text(encoding="utf-8"))
    assert {k: v for k, v in committed.items() if k != "what"} == got
    assert got["count"] == 1828 and len(got["topics"]) == 37


def test_the_move_keeps_the_bank_and_every_score_as_they_were(svc, capsys):
    before = ev.load_bank()
    version = ev.version()
    assert ev.hidden_status()["state"] == "repo"
    assert run("move") == 0
    out = capsys.readouterr().out
    assert out.startswith(f"moved {REPO_MANIFEST['count']} hidden questions to ")
    assert f"the bank's version {version['hash']} unchanged" in out and "backed up to " in out
    # the store holds the hidden half, each row marked hidden; the bank reads the same
    rows = ev._raw_rows(ev.hidden_path())
    assert len(rows) == REPO_MANIFEST["count"] and {q["half"] for q in rows} == {ev.HIDDEN}
    after = ev.load_bank()
    assert [q["id"] for q in after] == [q["id"] for q in before]
    assert [q["prompt"] for q in after] == [q["prompt"] for q in before]
    assert ev.version() == version and ev.hidden_status()["state"] == "store"
    # nothing printed is a question
    assert not [q for q in rows if q["prompt"][:30] in out]
    # once is enough: a second move leaves it
    assert run("move") == 0 and capsys.readouterr().out.startswith("already in the store:")
    assert len(hidden_store.backups()) == 2


def test_once_the_repo_no_longer_holds_it_the_store_is_read(svc, monkeypatch):
    """12p.2's world: the repo's bank has the practice half only"""
    run("move")
    ids = [q["id"] for q in ev.load_bank()]
    practice = [q for q in ev._raw_rows(ev.BANK_PATH) if ev.half(q) == ev.PRACTICE]
    only = config.BENCH_ROOT / "practice.jsonl"
    only.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in practice),
                    encoding="utf-8")
    monkeypatch.setattr(ev, "BANK_PATH", only)
    got = ev.load_bank()
    assert sorted(q["id"] for q in got) == sorted(ids)
    order = list(ev.groups())
    assert [order.index(q["group"]) for q in got] == sorted(order.index(q["group"]) for q in got)
    assert ev.hidden_status()["state"] == "store"
    # and without the store, the set is missing
    ev.hidden_path().unlink()
    st = ev.hidden_status()
    assert (st["ok"], st["state"]) == (False, "missing")
    assert st["why"] == ("Everyday's hidden set is missing or changed: restore it · "
                         "sudo docker compose exec -T bench python -m service.hidden_store "
                         "restore")


# ---------------------------------------------------------------------------
# missing or changed: the board works, Everyday waits, and every page says so
# ---------------------------------------------------------------------------

def changed() -> None:
    p = ev.hidden_path()
    rows = ev._raw_rows(p)
    rows[0]["prompt"] += " (changed)"
    p.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in rows),
                 encoding="utf-8")


def test_a_changed_set_stops_everyday_and_nothing_else(svc):
    client, appmod = svc
    run("move")
    mdir = config.OUT_DIR / "org__m"
    _asked(mdir, ev.load_bank()[:5])
    ev.write(mdir, ev.mark(mdir))
    changed()
    st = ev.hidden_status()
    assert (st["ok"], st["state"]) == (False, "changed")
    # scoring stops
    with pytest.raises(ev.HiddenMissing, match="missing or changed"):
        ev.mark(mdir)
    # an Everyday run is refused, with the command; another suite queues
    r = client.post("/api/submissions", json={"hf_id": "org/x", "suite": "everyday"})
    assert r.status_code == 409 and r.json()["detail"].endswith(
        "hidden_store restore. Nothing was queued.")
    assert client.post("/api/submissions", json={"hf_id": "org/x", "suite": "quick"}) \
        .status_code == 200
    # every page's data carries the banner
    appmod._cache.update(at=0.0)
    got = client.get("/api/results").json()
    assert got["alarms"] == [{"key": "hidden",
                              "text": "Everyday's hidden set is missing or changed: restore it",
                              "command": ev.RESTORE}]
    assert got["everyday"]["paused"] == st["why"]


def test_while_paused_the_scores_stay_as_they_were_marked(svc):
    client, appmod = svc
    run("move")
    d = next(p for p in config.OUT_DIR.iterdir() if (p / ev.OUT_NAME).exists()
             and not (ev.read(p) or {}).get("earlier"))
    was = ev.read(d)
    appmod._cache.update(key=None, at=0.0)
    before = client.get("/api/results").json()["everyday"]["models"][was["model"]]
    changed()
    appmod._cache.update(key=None, at=0.0)
    now = client.get("/api/results").json()["everyday"]["models"][was["model"]]
    assert (now["passed"], now["total"]) == (before["passed"], before["total"]) \
        == (was["passed"], was["total"])
    assert now["groups"] == was["groups"]


def test_a_queued_everyday_run_stops_before_any_gpu(svc, monkeypatch):
    """queued before the set went missing: it fails with the banner's words,
    and nothing is asked of the model"""
    client, _ = svc
    from service import runner
    from test_everyday_12a import fake_gpu, queue_pilot
    seen = fake_gpu(monkeypatch)
    sid = queue_pilot(client)
    run("move")
    changed()
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "failed" and "hidden set is missing or changed" in row["error"]
    assert seen == []


def test_the_judges_verdicts_wait_until_it_is_restored(svc, monkeypatch):
    run("move")
    called = []

    def finish(*a):
        called.append(a)
        raise ev.HiddenMissing("gone")
    monkeypatch.setattr(ev, "finish_remark", finish)
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-judge")
    llm.reset()
    backend = llm.client("judge")
    bid = backend.submit([llm.Request(custom_id="everyday-remark:x:q", system="", user="u")])
    db.batch_add(bid, "everyday_remark", 0, 1, backend.name, backend.model)
    llm_poller.tick()
    llm_poller.tick()
    assert called and [b["batch_id"] for b in db.batches_pending()] == [bid]      # kept


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------

def test_restore_puts_the_committed_set_back(svc, capsys):
    run("move")
    changed()
    capsys.readouterr()
    assert run("restore") == 0
    out = capsys.readouterr().out
    assert out.startswith("restored from evalboard-store-") and "everyday/hidden.jsonl" in out
    assert ev.hidden_status()["state"] == "store"
    # what was there is kept beside it
    assert [p.name for p in ev.built_dir().glob("hidden.jsonl.before-restore-*")]


def test_restore_skips_a_backup_of_a_changed_set_and_a_tampered_one(svc, capsys, monkeypatch):
    run("move")
    good = hidden_store.backups()[-1]
    changed()
    with monkeypatch.context() as m:
        m.setattr(time, "time", lambda: 4102444800.0)             # a later backup: the changed set
        hidden_store.backup()
    assert len(hidden_store.backups()) == 2
    got = hidden_store.restore()
    assert got["from"] == good.name and got["hidden"]["state"] == "store"
    # a backup whose file doesn't match its sha256 is refused
    bad = config.BACKUP_DIR / "evalboard-store-19990101T000000Z.tar.gz"
    with tarfile.open(good) as t, tarfile.open(bad, "w:gz") as out:
        for m in t.getmembers():
            b = t.extractfile(m).read()
            if m.name == "everyday/hidden.jsonl":
                b += b"\n"
                m.size = len(b)
            out.addfile(m, io.BytesIO(b))
    assert run("restore", str(bad)) == 2
    assert "doesn't match its sha256" in capsys.readouterr().err


def test_the_exams_missing_questions_are_added_back_never_over_newer(svc, monkeypatch):
    import exam_build as eb
    rows = eb.all_bank_rows(config.EXAM_DIR)
    report = [r for r in rows if eb.half_of(r["qid"]) == "report"]
    assert report
    man = config.BENCH_ROOT / "exam-report.json"
    man.write_text(json.dumps({"count": len(report), "topics": {"t": [r["qid"] for r in report]}}))
    monkeypatch.setattr(config, "EXAM_REPORT_MANIFEST", man)
    assert hidden_store.exam_status()["state"] == "store"
    hidden_store.backup()
    # one question lost from its file, another edited since the backup
    f = next(eb.bank_dir(config.EXAM_DIR).glob("*.jsonl"))
    lines = [x for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    gone, kept = json.loads(lines[0]), json.loads(lines[1])
    kept["notes"] = "edited after the backup"
    f.write_text("\n".join([json.dumps(kept)] + lines[2:]) + "\n", encoding="utf-8")
    st = hidden_store.exam_status()
    if eb.half_of(gone["qid"]) == "report":
        assert (st["ok"], st["missing"]) == (False, 1)
    got = hidden_store.restore()
    assert got["exam_rows"] == 1 and hidden_store.exam_status()["ok"]
    back = {json.loads(x)["qid"]: json.loads(x) for x in f.read_text().splitlines() if x.strip()}
    assert gone["qid"] in back and back[kept["qid"]]["notes"] == "edited after the backup"


def test_restore_all_moves_what_is_there_aside(svc):
    run("move")
    changed()
    got = hidden_store.restore(everything=True)
    assert got["aside"] and (config.BENCH_ROOT / got["aside"] / "everyday" / "hidden.jsonl").exists()
    assert ev.hidden_status()["state"] == "store"


# ---------------------------------------------------------------------------
# backups
# ---------------------------------------------------------------------------

def test_a_backup_is_every_file_with_its_sha256_and_keeps_14(svc, monkeypatch):
    run("move")
    b = hidden_store.backups()[-1]
    with tarfile.open(b) as t:
        names = t.getnames()
        man = json.loads(t.extractfile("MANIFEST.json").read())
    assert "everyday/hidden.jsonl" in names and any(n.startswith("exam/bank/") for n in names)
    assert set(man["files"]) == set(names) - {"MANIFEST.json"}
    assert man["hidden"]["state"] == "store"
    t0 = 4102444800.0
    for i in range(20):
        with monkeypatch.context() as m:
            m.setattr(time, "time", lambda i=i: t0 + i * 3600)
            hidden_store.backup()
    assert len(hidden_store.backups()) == config.BACKUP_KEEP == 14


def test_once_a_day_and_a_failure_is_said(svc, monkeypatch):
    assert hidden_store.daily() is not None and len(hidden_store.backups()) == 1
    assert hidden_store.daily() is None                           # not again today
    monkeypatch.setattr(config, "BACKUP_DIR", config.BENCH_ROOT / "nowhere" / "x")
    (config.BENCH_ROOT / "nowhere").write_text("a file, not a folder")
    hidden_store._last_try["at"] = 0.0
    assert hidden_store.daily() is None
    o = hidden_store.overview()
    assert o["backup_error"] and o["backup_stale"] and o["backup"]["kept"] == 0


def test_the_worker_backs_up_when_idle(svc, monkeypatch):
    from service import worker
    got = []
    monkeypatch.setattr(hidden_store, "daily", lambda: got.append(1))
    monkeypatch.setattr(worker.reported, "daily", lambda: None)
    worker._idle()
    assert got == [1]


def test_an_export_needs_a_key_and_is_one_encrypted_file(svc, monkeypatch, capsys):
    assert run("backup", "--export") == 2
    err = capsys.readouterr().err
    assert "EVALBOARD_BACKUP_AGE_RECIPIENT" in err and err.count("\n") == 1       # one line
    # a private key put there by mistake: refused, and never said back
    secret = "AGE-SECRET-KEY-1QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ"
    monkeypatch.setattr(config, "BACKUP_AGE_RECIPIENT", secret)
    assert run("backup", "--export") == 2
    err = capsys.readouterr().err
    assert "holds a private key" in err and secret not in err
    monkeypatch.setattr(config, "BACKUP_AGE_RECIPIENT", "")
    pyrage = pytest.importorskip("pyrage")
    key = pyrage.x25519.Identity.generate()
    monkeypatch.setattr(config, "BACKUP_AGE_RECIPIENT", str(key.to_public()))
    assert run("backup", "--export") == 0
    out = capsys.readouterr().out
    [age] = list(config.BACKUP_DIR.glob("*.tar.gz.age"))
    assert f"exported, encrypted to your age key: {age}" in out
    plain = pyrage.decrypt(age.read_bytes(), [key])
    assert plain == hidden_store.backups()[-1].read_bytes()
    # the server holds the public key only
    assert str(key) not in json.dumps(hidden_store.overview())


# ---------------------------------------------------------------------------
# OpenRouter: never a provider that may store or train on the prompt
# ---------------------------------------------------------------------------

def test_every_openrouter_request_denies_data_collection(svc, monkeypatch):
    from service import ai_models
    fake = FakeOpenRouter.install(monkeypatch)
    pin = ai_models.pin("z-ai/glm-5.3")
    assert pin["provider"] == "inference-net"
    assert fake.probes and fake.probes[0]["provider"] == {
        "order": ["inference-net"], "allow_fallbacks": False, "data_collection": "deny"}
    b = llm.OpenRouterChat("z-ai/glm-5.3", config.OPENROUTER_API_KEY, config.BENCH_ROOT, pin=pin)
    b._complete({"custom_id": "x", "system": "", "user": "hi", "max_tokens": 10, "json": False})
    assert fake.chat[-1]["provider"]["data_collection"] == "deny"
    ai_models.embed(["a", "b"])
    assert fake.embedded[-1]["provider"] == {"data_collection": "deny"}


def test_a_provider_that_may_train_on_prompts_is_passed_over(svc, monkeypatch):
    from service import ai_models
    fake = FakeOpenRouter.install(monkeypatch)
    fake.collects = {"inference-net"}
    assert ai_models.pin("z-ai/glm-5.3")["provider"] == "morph/fp8"
    fake.collects = {"inference-net", "morph/fp8"}
    with pytest.raises(ValueError, match="may store or train on prompts: choose another model"):
        ai_models.pin("z-ai/glm-5.3")


def test_the_handoff_says_where_it_is_set():
    handoff = (ev.REPO / "HANDOFF.md").read_text(encoding="utf-8")
    assert '`provider.data_collection: "deny"`' in handoff and "service/ai_models.py" in handoff
    assert "openrouter.ai/settings/privacy" in handoff


# ---------------------------------------------------------------------------
# the exam's import preview
# ---------------------------------------------------------------------------

def test_the_import_preview_withholds_a_report_questions_reference_and_meta(svc):
    from service.app import _import_preview
    import exam_build as eb
    rep = next(q for q in (f"a question {i} about soil" for i in range(200))
               if eb.half_of(eb.qid_of(q)) == "report")
    dia = next(q for q in (f"a question {i} about soil" for i in range(200))
               if eb.half_of(eb.qid_of(q)) == "diagnose")
    rec = lambda p: {"qid": eb.qid_of(p), "prompt": p, "reference": "ref of " + p,   # noqa: E731
                     "meta": {"intent": "what " + p + " asks"}}
    got = _import_preview({"records": [rec(rep), rec(dia)], "updates": []})
    r, d = got["items"]
    assert (r["prompt"], r["reference"], r["meta"]) == (None, None, None) and r["withheld"]
    assert d["prompt"] == dia and d["reference"] == "ref of " + dia and d["meta"]
    assert rep not in json.dumps(got)


def test_the_cli_says_the_state(svc, capsys):
    assert run("status") == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"Everyday's hidden set: repo, {REPO_MANIFEST['count']} questions"
    assert out[1].startswith("The exam's report half: unchecked")
    assert out[2].startswith("Backups: 0 in ")
