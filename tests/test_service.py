"""The FastAPI service against a temp BENCH_ROOT. No GPU, no network: the
worker thread is never started, so a queued submission stays queued and
nothing here can reach the Hub."""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

import make_fixture


@pytest.fixture
def svc(tmp_path, monkeypatch):
    from service import config, worker
    import service.app as appmod
    root = tmp_path
    for name, val in {"BENCH_ROOT": root, "RESULTS_ROOT": root / "results",
                      "OUT_DIR": root / "results" / "full",
                      "DB_PATH": root / "service.sqlite3",
                      "ARTIFACTS_DIR": root / "artifacts", "LOGS_DIR": root / "logs",
                      "SUBMIT_TOKEN": "", "ALLOW_REMOTE_CODE": False}.items():
        monkeypatch.setattr(config, name, val)
    monkeypatch.setattr(worker, "start", lambda: None)
    appmod._cache.update(key=None, payload=None, at=0.0)
    with TestClient(appmod.app) as c:
        yield c, appmod, root


def _fresh(appmod):
    # the payload is debounced for five seconds; that is a load-shedding
    # choice, not the freshness property under test, so step past it
    appmod._cache["at"] = 0.0


def test_healthz(svc):
    c, _, _ = svc
    assert c.get("/healthz").json() == {"ok": True, "queue": 0}


def test_results_on_an_empty_tree(svc):
    c, _, _ = svc
    p = c.get("/api/results").json()
    assert p["models"] == [] and p["live"] is True


def test_results_notice_a_diagnosis_written_after_the_first_request(svc):
    """The _tree_key fix: diagnose.json is written long after the eval, and a
    cache key that ignored it served a payload with no diagnosis in it until
    the container was restarted."""
    c, appmod, root = svc
    assert "diagnose.json" in appmod._WATCH
    make_fixture.build(root, diagnose=False)
    p1 = c.get("/api/results").json()
    assert len(p1["models"]) == len(make_fixture.MODELS)
    assert p1["meta"]["anyDiag"] is False

    make_fixture.write_diagnoses(root, only=["fx/skewed-360m"])
    _fresh(appmod)
    p2 = c.get("/api/results").json()
    assert p2["meta"]["anyDiag"] is True
    row = next(m for m in p2["models"] if m["id"] == "fx/skewed-360m")
    assert row["diag"] and "mmlu" in row["diag"]["tasks"]
    assert all(m["diag"] is None for m in p2["models"] if m["id"] != "fx/skewed-360m")


def test_results_notice_a_rewritten_diagnosis(svc):
    c, appmod, root = svc
    make_fixture.build(root, diagnose=False)
    make_fixture.write_diagnoses(root, only=["fx/good-750m"])
    p1 = c.get("/api/results").json()
    row = next(m for m in p1["models"] if m["id"] == "fx/good-750m")
    assert "hellaswag" in row["diag"]["tasks"]

    f = root / "results" / "full" / "fx__good-750m" / "diagnose.json"
    d = json.loads(f.read_text())
    del d["tasks"]["hellaswag"]
    f.write_text(json.dumps(d))
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
    _fresh(appmod)
    p2 = c.get("/api/results").json()
    row = next(m for m in p2["models"] if m["id"] == "fx/good-750m")
    assert "hellaswag" not in row["diag"]["tasks"] and "mmlu" in row["diag"]["tasks"]


def test_results_are_debounced_between_polls(svc):
    c, appmod, root = svc
    make_fixture.build(root, diagnose=False)
    c.get("/api/results")
    make_fixture.write_diagnoses(root, only=["fx/skewed-360m"])
    # within the window the cached payload is served — the dashboard polls
    # every five seconds and thirty tabs must not mean thirty tree walks
    assert c.get("/api/results").json()["meta"]["anyDiag"] is False
    _fresh(appmod)
    assert c.get("/api/results").json()["meta"]["anyDiag"] is True


@pytest.mark.parametrize("body, code, needle", [
    ({"hf_id": "not-a-repo-id"}, 422, "org/name"),
    ({"hf_id": "org/model", "kind": "chatty"}, 422, "kind must be"),
    ({"hf_id": "org/model", "suite": "everything"}, 422, "suite must be"),
    ({"hf_id": "org/model", "allow_remote_code": True}, 422, "uploaded artifacts"),
    ({"hf_id": "local/ckpt", "allow_remote_code": True}, 403, "ALLOW_REMOTE_CODE off"),
])
def test_submission_validation(svc, body, code, needle):
    c, _, _ = svc
    r = c.post("/api/submissions", json=body)
    assert r.status_code == code, r.text
    assert needle in r.json()["detail"]
    assert c.get("/api/submissions").json() == []      # nothing was queued


def test_submission_lifecycle_without_a_worker(svc):
    c, _, _ = svc
    r = c.post("/api/submissions", json={"hf_id": "EleutherAI/pythia-31m", "suite": "quick",
                                          "submitter": "tester", "note": "smoke"})
    assert r.status_code == 200 and r.json()["status"] == "queued"
    sid = r.json()["id"]
    # the same model again joins the existing run instead of queueing twice
    again = c.post("/api/submissions", json={"hf_id": "EleutherAI/pythia-31m"}).json()
    assert again["id"] == sid and "joining" in again["note"]
    rows = c.get("/api/submissions").json()
    assert len(rows) == 1 and rows[0]["submitter"] == "tester"
    assert rows[0]["status"] == "queued" and rows[0]["suite"] == "quick"
    assert c.get("/healthz").json()["queue"] == 1
    assert c.get(f"/api/runs/{sid}/log").text.startswith("(no log yet")
    assert c.get("/api/runs/999/log").status_code == 404
    assert c.post(f"/api/submissions/{sid}/cancel").json()["status"] == "canceled"
    assert c.post(f"/api/submissions/{sid}/cancel").status_code == 409
    assert c.get("/healthz").json()["queue"] == 0


def test_control_suite_is_a_queueable_submission(svc):
    c, _, _ = svc
    r = c.post("/api/submissions", json={"hf_id": "HuggingFaceTB/SmolLM2-360M",
                                          "suite": "control", "submitter": "tester"})
    assert r.status_code == 200 and r.json()["status"] == "queued"
    row = c.get("/api/submissions").json()[0]
    assert row["suite"] == "control" and row["status"] == "queued"
    bad = c.post("/api/submissions", json={"hf_id": "org/model", "suite": "perm"})
    assert bad.status_code == 422 and "control" in bad.json()["detail"]


def test_submit_token_gates_every_post_when_set(svc, monkeypatch):
    c, _, _ = svc
    from service import config
    monkeypatch.setattr(config, "SUBMIT_TOKEN", "s3cret")
    body = {"hf_id": "org/model"}
    assert c.post("/api/submissions", json=body).status_code == 401
    assert c.post("/api/submissions", json=body, headers={"X-Token": "wrong"}).status_code == 401
    assert c.post("/api/submissions", json=body, headers={"X-Token": "s3cret"}).status_code == 200
    assert c.post("/api/truns", json={"name": "r"}).status_code == 401
    assert c.get("/api/submissions").status_code == 200        # reads stay open


def test_training_run_round_trip(svc):
    c, _, _ = svc
    rid = c.post("/api/truns", json={"name": "run7", "project": "llm", "submitter": "tester",
                                      "config": {"lr": 3e-4}, "hf_prefix": "local/run7-"}).json()["id"]
    assert c.post("/api/truns", json={"name": "  "}).status_code == 422
    pts = [{"step": s, "name": "loss", "value": 3.0 - s / 100} for s in range(1, 11)]
    pts.append({"step": 5, "name": "loss", "value": "nan"})           # parses, is not finite: dropped
    pts.append({"step": "x", "name": "loss", "value": 1})              # malformed, skipped
    assert c.post(f"/api/truns/{rid}/log", json={"metrics": pts}).json() == {"logged": 10}
    assert c.post("/api/truns/999/log", json={"metrics": pts}).status_code == 404
    assert c.post(f"/api/truns/{rid}/event",
                  json={"step": 10, "detail": "local/run7-step10"}).json() == {"ok": True}
    assert c.post(f"/api/truns/{rid}/finish", json={"status": "done"}).status_code == 422
    assert c.post(f"/api/truns/{rid}/finish", json={"status": "finished"}).json() == {"ok": True}
    d = c.get(f"/api/truns/{rid}").json()
    assert d["run"]["status"] == "finished" and d["run"]["config"] == '{"lr": 0.0003}'
    assert [p[0] for p in d["metrics"]["loss"]] == list(range(1, 11))
    assert d["events"] == [{"step": 10, "kind": "checkpoint", "detail": "local/run7-step10"}]
    lst = c.get("/api/truns").json()
    assert lst[0]["id"] == rid and lst[0]["last_step"] == 10 and lst[0]["n_events"] == 1
    assert c.get("/api/truns/999").status_code == 404


def test_artifacts_index_and_refusals(svc):
    c, _, root = svc
    assert c.get("/api/artifacts").json() == {"artifacts": [], "total_bytes": 0,
                                              "quota_bytes": 150_000_000_000}
    assert c.delete("/api/artifacts/nope").status_code == 404
    assert c.delete("/api/artifacts/bad name").status_code == 422
    r = c.post("/api/artifacts/bad name", content=b"zip")
    assert r.status_code == 422
    (root / "artifacts" / "ckpt").mkdir(parents=True)
    r = c.post("/api/artifacts/ckpt", content=b"zip", headers={"Content-Length": "3"})
    assert r.status_code == 409 and "immutable" in r.json()["detail"]
    assert c.get("/api/artifacts").json()["artifacts"][0]["model_id"] == "local/ckpt"


def test_pages_are_served(svc):
    c, _, _ = svc
    page = c.get("/")
    assert page.status_code == 200
    assert '<script id="data" type="application/json">null</script>' in page.text
    assert "__JS__SLOT__" not in page.text and "__CSS__" not in page.text
    client = c.get("/client")
    assert client.status_code == 200 and client.text.startswith("#!/usr/bin/env python3")
    assert "attachment" in client.headers["content-disposition"]
    guide = c.get("/guide")
    assert guide.status_code == 200 and "<main>" in guide.text
