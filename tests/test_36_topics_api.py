"""Phase 10c, the service side: the judge is asked before a judged run spends
GPU, a failed row says what failed in plain words and can be re-graded, and
the Sit and Queue panels get what they need to say what a run costs."""

from __future__ import annotations

import socket
import time

import pytest

from conftest import make_service

MODEL = "fx/good-750m"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    appmod._JUDGE_HEALTH.update(at=0.0, value=None)
    yield client, appmod, tree
    appmod._JUDGE_HEALTH.update(at=0.0, value=None)
    client.__exit__(None, None, None)


def closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def judge_on_local(monkeypatch, appmod, url: str):
    from service import config
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "local")
    monkeypatch.setattr(config, "JUDGE_MODEL", "chat")
    monkeypatch.setattr(config, "LOCAL_BASE_URL", url)
    appmod._JUDGE_HEALTH.update(at=0.0, value=None)


# ---------------------------------------------------------------------------
# 8.1 the judge, before the GPU
# ---------------------------------------------------------------------------

def test_a_judged_run_is_refused_in_plain_words_while_the_judge_is_down(svc, monkeypatch):
    """#50 answered law (37 GPU-seconds) and then failed: vLLM had crashed
    three hours earlier."""
    client, appmod, _ = svc
    from service import db
    url = f"http://127.0.0.1:{closed_port()}/v1"
    judge_on_local(monkeypatch, appmod, url)
    before = len(db.recent(500))
    r = client.post("/api/submissions", json={"hf_id": MODEL, "suite": "judged",
                                              "tasks": ["exam_law"]})
    assert r.status_code == 503
    said = r.json()["detail"]
    assert said.startswith("The grading model isn't answering at " + url)
    assert "nothing was queued" in said and "Traceback" not in said
    assert len(db.recent(500)) == before
    h = client.get("/api/judge/health").json()
    assert h["ok"] is False and h["url"] == url and url in h["why"]
    assert client.get("/api/loop").json()["judge_health"]["ok"] is False
    # a suite that needs no judge is not stopped by it
    assert client.post("/api/submissions", json={"hf_id": MODEL, "suite": "quick"}).status_code == 200


def test_the_judge_is_asked_at_most_every_thirty_seconds(svc, monkeypatch):
    client, appmod, _ = svc
    import urllib.request
    judge_on_local(monkeypatch, appmod, f"http://127.0.0.1:{closed_port()}/v1")
    calls = []
    real = urllib.request.urlopen

    def counting(req, timeout=None):
        calls.append(timeout)
        return real(req, timeout=timeout)
    monkeypatch.setattr(urllib.request, "urlopen", counting)
    for _ in range(5):
        client.get("/api/judge/health")
    assert calls == [appmod.JUDGE_HEALTH_TIMEOUT] == [2.0]
    appmod._JUDGE_HEALTH["at"] = time.time() - appmod.JUDGE_HEALTH_TTL - 1
    client.get("/api/judge/health")
    assert len(calls) == 2


def test_a_judge_that_answers_lets_the_run_through(svc, monkeypatch):
    client, appmod, _ = svc
    import vllm_stub
    stub, srv = vllm_stub.serve()
    try:
        judge_on_local(monkeypatch, appmod, stub.url)
        assert client.get("/api/judge/health").json()["ok"] is True
    finally:
        srv.shutdown()
    # an API judge is not probed: its batches wait for the provider
    from service import config
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "anthropic")
    appmod._JUDGE_HEALTH.update(at=0.0, value=None)
    assert client.get("/api/judge/health").json() == {
        "ok": True, "checked": False, "provider": "anthropic", "url": "", "why": ""}


# ---------------------------------------------------------------------------
# 8.2 a failed row, when only the grading failed
# ---------------------------------------------------------------------------

def test_a_row_whose_grading_failed_says_so_and_is_marked_for_a_retry(svc):
    client, _, _ = svc
    from service import db
    a = db.add(MODEL, "auto", "judged", "omar", "")
    db.update(a, status="failed", progress="failed on: judge",
              error="judge: LocalUnreachable: nothing is answering at "
                    "http://host.docker.internal:8000/v1 — tunnel with ssh -L 8000:…")
    b = db.add(MODEL, "auto", "judged", "omar", "")
    db.update(b, status="failed", error="exam_law: ran out of GPU memory")
    rows = {r["id"]: r for r in client.get("/api/submissions").json()}
    assert rows[a]["judge_failed"] is True
    assert rows[b]["judge_failed"] is False            # the answers are what failed


def test_a_batch_that_failed_after_the_run_marks_the_row_too(svc):
    client, _, _ = svc
    from service import db
    sid = db.add(MODEL, "auto", "judged", "omar", "")
    rid = db.judge_run_create(MODEL, "local_b1", 42, "local/chat", "{}")
    db.batch_add("local_b1", "judge", rid, 42, "local", "chat")
    db.update(sid, status="done", judge_batch="local_b1")
    db.batch_finish("local_b1", "failed", "every request failed: connection refused")
    rows = {r["id"]: r for r in client.get("/api/submissions").json()}
    assert rows[sid]["judge_failed"] is True
    assert "connection refused" in rows[sid]["judge"]["error"]


# ---------------------------------------------------------------------------
# 7. what a run costs: items per built task, and the pace of the last runs
# ---------------------------------------------------------------------------

def test_the_loop_says_how_many_answers_a_topic_costs_and_how_long_they_take(svc):
    client, _, _ = svc
    from service import db
    j = client.get("/api/loop").json()
    items = j["built_items"]
    assert items["exam_economics"] == next(r for r in j["topics"]
                                          if r["topic"] == "Economics")["bank"]["accepted"]
    assert j["pace"] is None                          # no judged run to time it by yet
    now = time.time()
    for k, (gpu, judge_s, n) in enumerate([(600.0, 300.0, 900), (60.0, 30.0, 90)]):
        sid = db.add(MODEL, "auto", "judged", "omar", "")
        rid = db.judge_run_create(MODEL, f"b{k}", n, "stub", "{}")
        db.judge_run_update(rid, status="done", finished_at=now)
        from contextlib import closing

        from service.db import _conn
        with closing(_conn()) as c:
            c.execute("UPDATE judge_runs SET created_at=? WHERE id=?", (now - judge_s, rid))
            c.commit()
        db.update(sid, status="done", gpu_seconds=gpu, judge_batch=f"b{k}")
    pace = client.get("/api/loop").json()["pace"]
    assert pace == {"sec_per_answer": round((660 + 330) / 990, 3), "runs": 2}
