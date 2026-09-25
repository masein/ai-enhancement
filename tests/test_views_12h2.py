"""12h.2: saved views of the Models table, in the service. A view is the
team's: anyone lists it; it records who saved it; only that name renames or
deletes it (the tailnet is the auth boundary, as for every decision here, so
the owner is the name typed at the top of the page)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

SPEC = {"chip": "all", "cols": ["ifeval", "mmlu_pro", "hendrycks_math500"],
        "models": ["Qwen/Qwen3.5-2B", "LiquidAI/LFM2.5-1.2B-Instruct"]}


@pytest.fixture
def svc(tmp_path, monkeypatch):
    from service import config, worker
    import service.app as appmod
    for name, val in {"BENCH_ROOT": tmp_path, "RESULTS_ROOT": tmp_path / "results",
                      "OUT_DIR": tmp_path / "results" / "full",
                      "DB_PATH": tmp_path / "service.sqlite3",
                      "ARTIFACTS_DIR": tmp_path / "artifacts", "LOGS_DIR": tmp_path / "logs",
                      "SUBMIT_TOKEN": ""}.items():
        monkeypatch.setattr(config, name, val)
    monkeypatch.setattr(worker, "start", lambda: None)
    with TestClient(appmod.app) as c:
        yield c, config


def save(c, name="Phone shortlist", by="masein", spec=SPEC, **kw):
    return c.post("/api/views", json={"name": name, "by": by, "spec": spec}, **kw)


def test_a_view_is_saved_for_the_team_with_who_saved_it(svc):
    c, _ = svc
    assert c.get("/api/views").json() == {"views": []}
    r = save(c)
    assert r.status_code == 200
    v = r.json()
    assert (v["name"], v["saved_by"], v["spec"]) == ("Phone shortlist", "masein", SPEC)
    assert c.get("/api/views").json()["views"] == [v]


@pytest.mark.parametrize("body,status,words", [
    ({"name": "  ", "by": "masein", "spec": SPEC}, 422, "a view needs a name"),
    ({"name": "x", "by": " ", "spec": SPEC}, 422, "saving a view needs a name"),
    ({"name": "x", "by": "masein", "spec": {"chip": "all"}}, 422, "nothing to save"),
    ({"name": "x", "by": "masein", "spec": {"cols": ["ifeval; drop"]}}, 422, "task names"),
])
def test_a_view_needs_a_name_an_owner_and_something_chosen(svc, body, status, words):
    c, _ = svc
    r = c.post("/api/views", json=body)
    assert r.status_code == status and words in r.json()["detail"]


def test_two_views_cannot_share_a_name(svc):
    c, _ = svc
    save(c)
    r = save(c, name="phone SHORTLIST", by="omar")
    assert r.status_code == 409
    assert r.json()["detail"] == ("a view called Phone shortlist already exists, saved by "
                                  "masein — choose another name")


def test_only_the_name_that_saved_it_renames_or_deletes_it(svc):
    c, _ = svc
    vid = save(c).json()["id"]
    r = c.patch(f"/api/views/{vid}", json={"name": "Mine now", "by": "omar"})
    assert r.status_code == 403
    assert r.json()["detail"] == "only masein, who saved this view, can rename it"
    r = c.request("DELETE", f"/api/views/{vid}", json={"by": "omar"})
    assert r.status_code == 403
    assert r.json()["detail"] == "only masein, who saved this view, can delete it"
    # the owner, however the name is capitalised
    r = c.patch(f"/api/views/{vid}", json={"name": "Phone picks", "by": "Masein"})
    assert r.status_code == 200 and r.json()["name"] == "Phone picks"
    r = c.request("DELETE", f"/api/views/{vid}", json={"by": "masein"})
    assert r.json() == {"deleted": vid, "name": "Phone picks"}
    assert c.get("/api/views").json() == {"views": []}
    r = c.request("DELETE", f"/api/views/{vid}", json={"by": "masein"})
    assert r.status_code == 404


def test_a_write_needs_the_token_when_one_is_set(svc, monkeypatch):
    c, config = svc
    monkeypatch.setattr(config, "SUBMIT_TOKEN", "sekrit")
    assert save(c).status_code == 401
    assert save(c, headers={"X-Token": "sekrit"}).status_code == 200
    assert c.get("/api/views").status_code == 200            # reading needs none
