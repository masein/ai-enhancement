"""14.4.5: MOBILE_MMLU_FULL=0 in .env hides the full Mobile-MMLU everywhere and
refuses new runs of it; its files, picks and labels stay on disk, and
switching it back shows them again. On by default. Invented rows, picks
written in; nothing runs, nothing calls OpenRouter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mobile_mmlu as mmp
from conftest import make_service
from service import config, db, gguf, runner
from test_14_3_mobile_mmlu import put_invented
from test_14_4_1_mobile_mmlu_data import put_full
from test_14_4_3_mobile_mmlu_full_run import fake_lm_eval, pool_key, right, sit_full

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch)
    put_invented(monkeypatch, config.MMP_DIR)
    put_full(monkeypatch, config.MMF_DIR)
    yield client, appmod
    client.__exit__(None, None, None)


def off(monkeypatch):
    monkeypatch.setattr(config, "MOBILE_MMLU_FULL", False)


def sat_full(model: str = "org/full-2b") -> Path:
    """a model that sat the full set: every question right"""
    d = config.OUT_DIR / model.replace("/", "__")
    sit_full(d, {q["lid"]: right(q) for q in mmp.pool()})
    mmp.collect(d)
    return d


def results(client, appmod) -> dict:
    appmod._cache.update(key=None, payload=None, at=0.0)
    return client.get("/api/results").json()


def test_on_by_default_and_said_in_env_example_and_compose(monkeypatch):
    monkeypatch.delenv("MOBILE_MMLU_FULL", raising=False)
    assert config._switch("MOBILE_MMLU_FULL") is True and config.MOBILE_MMLU_FULL is True
    assert mmp.full_on() is True
    assert "# MOBILE_MMLU_FULL=1" in (REPO / ".env.example").read_text()
    assert "MOBILE_MMLU_FULL: ${MOBILE_MMLU_FULL:-1}" in (REPO / "docker-compose.yml").read_text()


@pytest.mark.parametrize("value,on", [("0", False), ("off", False), ("No", False),
                                      ("false", False), ("1", True), ("yes", True)])
def test_the_switchs_words(monkeypatch, value, on):
    monkeypatch.setenv("MOBILE_MMLU_FULL", value)
    assert config._switch("MOBILE_MMLU_FULL") is on


def test_switched_off_it_is_nowhere_in_the_pages_data(svc, monkeypatch):
    client, appmod = svc
    pool_key()
    sat_full()
    data = results(client, appmod)
    assert data["mmf"]["name"] == "Mobile-MMLU (full)"
    m = next(x for x in data["models"] if x["id"] == "org/full-2b")
    assert m["mmf"]["acc"] == 1.0
    off(monkeypatch)
    data = results(client, appmod)
    text = json.dumps(data)
    assert data["mmf"] is None and "mobile_mmlu_full" not in text
    assert "Mobile-MMLU (full)" not in text and "Non-commercial" not in text
    m = next(x for x in data["models"] if x["id"] == "org/full-2b")
    assert m.get("mmf") is None
    # Pro's score, from the same picks, is Pro's: it stays
    assert data["cells"]["mobile_mmlu_pro"]["org/full-2b"]["v"] == 1.0
    est = client.get("/api/mobileaibench/estimate", params={"model": "org/x"}).json()
    assert "mmlu_full" not in est["parts"] and "mmlu" in est["parts"]


def test_switched_off_it_isnt_badged_either_its_manifest_names_the_switch(monkeypatch):
    import restrictions
    m = json.loads((restrictions.MANIFESTS / "mobile_mmlu" / "manifest.json").read_text())
    assert m["switch"] == "MOBILE_MMLU_FULL"
    assert "mobile_mmlu_full" in restrictions.sets()
    off(monkeypatch)
    got = restrictions.sets()
    assert "mobile_mmlu_full" not in got and got["mobile_mmlu_pro"]["badge"] == "Internal use"


def test_switched_off_new_runs_are_refused(svc, monkeypatch):
    client, _ = svc
    pool_key()
    off(monkeypatch)
    r = client.post("/api/submissions", json={"hf_id": "org/c", "suite": "mobile",
                                              "part": "mmlu_full", "kind": "instruct"})
    assert r.status_code == 422
    assert r.json()["detail"] == ("Mobile-MMLU (full) is switched off on this server "
                                  "(MOBILE_MMLU_FULL=0 in .env). Nothing was queued.")
    # Pro's part is as it was
    r = client.post("/api/submissions", json={"hf_id": "org/c", "suite": "mobile",
                                              "part": "mmlu", "kind": "instruct"})
    assert r.status_code == 200, r.text


def test_a_run_queued_before_the_switch_asks_nothing_after_it(svc, monkeypatch):
    client, _ = svc
    pool_key()
    seen = fake_lm_eval(monkeypatch)
    sid = client.post("/api/submissions", json={"hf_id": "org/q-2b", "suite": "mobile",
                                                "part": "mmlu_full", "kind": "instruct"}).json()["id"]
    off(monkeypatch)
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "failed" and not seen
    assert row["error"] == ("Mobile-MMLU (full) is switched off on this server "
                            "(MOBILE_MMLU_FULL=0 in .env). Nothing was asked.")


def test_files_picks_and_labels_stay_on_disk_and_come_back(svc, monkeypatch):
    client, appmod = svc
    pool_key()
    d = sat_full()
    before = results(client, appmod)["models"]
    labels = json.loads((mmp.key_dir() / "labels.json").read_text()) if (
        mmp.key_dir() / "labels.json").exists() else None
    off(monkeypatch)
    results(client, appmod)
    # a Pro run's collect while off leaves the full set's picks as they were
    mmp.collect(d)
    assert (d / mmp.FULL_PRED_FILE).exists() and mmp.full_dir().joinpath("test").is_dir()
    assert len(list((mmp.full_dir() / "test").glob("*.csv"))) == 12
    assert mmp.full_key()["counts"]["all"]["kept"] == 18            # the key keeps it whole
    if labels is not None:
        assert json.loads((mmp.key_dir() / "labels.json").read_text()) == labels
    monkeypatch.setattr(config, "MOBILE_MMLU_FULL", True)
    after = results(client, appmod)["models"]
    pick = lambda ms: next(x for x in ms if x["id"] == "org/full-2b")["mmf"]  # noqa: E731
    assert pick(after) == pick(before)


def test_labelling_sends_pro_alone_while_off(svc, monkeypatch):
    from test_14_4_2_mobile_mmlu_key import FULL_ONLY  # noqa: F401
    from service import mmp_key
    client, _ = svc
    off(monkeypatch)
    page = client.get("/api/mobile-mmlu/key").json()
    assert page["full_on"] is False and "full" not in page["sets"]
    assert set(page["estimate"]["sets"]) == {"pro", "all"}
    assert page["estimate"]["sets"]["all"]["questions"] == 12 and page["checks_full"] is None
    # what Start would send: Pro's twelve, none of the full set's own
    sent = mmp_key.due()
    pro = {q["lid"] for q in mmp.pool() if "pro" in q["sets"]}
    assert set(sent["first"]) == pro and set(sent["second"]) == pro
    monkeypatch.setattr(config, "MOBILE_MMLU_FULL", True)
    # 16c: switched on, the full set's own are left once "Pro, then the full set" is chosen
    assert len(mmp_key.left()["first"]) == 12
    from service import db
    db.ai_set("mmp:scope", "all", "masein")
    assert len(mmp_key.left()["first"]) == 19


def test_gguf_hides_and_refuses_it_and_a_run_of_all_never_asks_it(svc, monkeypatch, tmp_path):
    import gguf_data as gd
    from test_gguf_12f3 import docs_of
    client, _ = svc
    pool_key()
    gd.build(config.RESULTS_ROOT / "gguf_data", docs_of=docs_of())
    gd.build(config.RESULTS_ROOT / "gguf_data", only=["mobile_mmlu_full"])
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF" + b"\1" * 500)
    gid = client.post("/api/gguf/models", json={"name": "m", "path": str(f), "how": "q4",
                                                "by": "masein"}).json()["model"]["id"]
    assert "mobile_mmlu_full" in client.get("/api/gguf").json()["order"]
    # a run of all: every benchmark built, but the full set — kept apart, asked only by name
    r = client.post("/api/gguf/runs", json={"model": gid, "by": "masein"})
    assert r.status_code == 200, r.text
    asked = db.get(r.json()["id"])["tasks"]
    assert "mobile_mmlu_full" not in json.dumps(asked) and "mmlu" in json.dumps(asked)
    r = client.post("/api/gguf/runs", json={"model": gid, "benchmarks": ["mobile_mmlu_full"],
                                            "by": "masein"})
    assert r.status_code == 200, r.text
    off(monkeypatch)
    info = client.get("/api/gguf").json()
    assert "mobile_mmlu_full" not in info["order"] and "mobile_mmlu_full" not in info["benchmarks"]
    assert "mobile_mmlu_full" not in info["datasets"]
    r = client.post("/api/gguf/runs", json={"model": gid, "benchmarks": ["mobile_mmlu_full"],
                                            "by": "masein"})
    assert r.status_code == 422 and "switched off" in r.json()["detail"]
    assert gguf.hidden() == {"mobile_mmlu_full"}
