"""12m.2: reported scores — numbers from outside the board, never mixed with
what it measures. Artificial Analysis's free Data API (a key masein makes;
without it the import says so in one line), Epoch AI's Benchmarking Hub,
and model cards typed in; each import with its file's sha256 and date, a
changed file a new import; an alias table that makes two names one model.
Nothing here asks anyone outside: a canned response stands in."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_service
from service import config, db, reported

AA = Path(__file__).resolve().parent / "fixtures" / "artificial_analysis_models.json"
KEY = "aa-test-key-not-real"


def canned(path=AA, edit=None):
    blob = json.loads(path.read_text())
    if edit:
        edit(blob)
    raw = json.dumps(blob).encode()
    asked = []

    def fetch(url, headers=None, timeout=60):
        asked.append((url, dict(headers or {})))
        return raw
    fetch.asked = asked
    return fetch


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "AA_API_KEY", "")
    monkeypatch.setattr(config, "REPORTED_MAKERS", ["OpenAI", "Google", "Anthropic"])
    monkeypatch.setattr(config, "REPORTED_PER_MAKER", 10)
    # an open model on the board that Artificial Analysis reports too
    d = config.OUT_DIR / "Qwen__Qwen3.5-2B"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_meta.json").write_text(json.dumps({"model": "Qwen/Qwen3.5-2B", "kind": "instruct"}))
    yield client, appmod, tree
    client.__exit__(None, None, None)


def test_without_a_key_nothing_is_asked_and_the_line_says_so(svc):
    client, _, _ = svc
    fetch = canned()
    out = reported.import_aa(fetch=fetch)
    assert out == {"source": "aa", "status": "no key", "line": reported.NO_AA_KEY}
    assert fetch.asked == [] and db.reported_import_last("aa") is None
    assert reported.NO_AA_KEY == ("Artificial Analysis: no key yet — add "
                                  "ARTIFICIAL_ANALYSIS_API_KEY to the server's .env, then Import now")
    view = client.get("/api/reported").json()
    assert view["sources"]["aa"]["line"] == reported.NO_AA_KEY
    assert view["sources"]["aa"]["has_key"] is False


def test_the_import_keeps_each_makers_ten_newest_and_the_board_models(svc, monkeypatch):
    client, _, _ = svc
    monkeypatch.setattr(config, "AA_API_KEY", KEY)
    fetch = canned()
    out = reported.import_aa(fetch=fetch)
    assert out["status"] == "imported" and out["line"].startswith("Artificial Analysis: ")
    # asked once, with the key in its header — and the key nowhere else
    [(url, headers)] = fetch.asked
    assert url == config.AA_URL and headers["x-api-key"] == KEY
    view = client.get("/api/reported").json()
    assert KEY not in json.dumps(view)
    names = {m["name"] for m in view["models"]}
    # OpenAI's ten newest of thirteen (by release date), Google's two, Anthropic's one
    assert {"Frontier Test 1", "Frontier Test 2", "Frontier Test 10"}.isdisjoint(names)
    assert sum(1 for m in view["models"] if m["maker"] == "OpenAI") == 10
    assert {"Gemini Test 3", "Gemini Test 2", "Claude Test 5"} <= names
    # Alibaba isn't a maker imported — but the open model on the board is kept, as itself
    assert "Other Test 1" not in names
    qwen = next(m for m in view["models"] if m["name"] == "Qwen3.5 2B")
    assert qwen["id"] == qwen["measured"] == "Qwen/Qwen3.5-2B"
    # the scores: an index in its points, a benchmark a share, each with its setting
    s55 = [s for s in view["scores"] if s["model"] == "reported/openai/frontier-test-5.5"]
    assert {(s["benchmark"], s["value"], s["unit"]) for s in s55} == {
        ("Intelligence Index", 71.5, "points"), ("MMLU-Pro", 0.873, "share"),
        ("GPQA Diamond", 0.702, "share")}
    assert {s["setting"] for s in s55} == {"Artificial Analysis's own run"}
    # its file's sha256 and the date
    last = db.reported_import_last("aa")
    assert len(last["sha256"]) == 64 and last["models"] == 14
    assert view["sources"]["aa"]["line"].endswith(f"· file {last['sha256'][:12]}")


def test_the_same_file_is_only_checked_and_a_changed_one_is_a_new_import(svc, monkeypatch):
    monkeypatch.setattr(config, "AA_API_KEY", KEY)
    reported.import_aa(fetch=canned())
    again = reported.import_aa(fetch=canned())
    assert again["status"] == "unchanged" and len(db.reported_imports("aa")) == 1

    def edit(b):
        next(m for m in b["data"] if m["name"] == "Frontier Test 5.5")["evaluations"]["mmlu_pro"] = 0.9
    new = reported.import_aa(fetch=canned(edit=edit))
    assert new["status"] == "imported" and len(db.reported_imports("aa")) == 2
    # the newest import's numbers only
    got = [s for s in reported.view()["scores"]
           if s["model"] == "reported/openai/frontier-test-5.5" and s["benchmark"] == "MMLU-Pro"]
    assert [s["value"] for s in got] == [0.9]


def test_each_source_is_credited(svc):
    client, _, _ = svc
    src = client.get("/api/reported").json()["sources"]
    assert src["epoch"]["credit"] == "Data: Epoch AI, CC BY 4.0"
    assert src["aa"]["credit"] == "Data: Artificial Analysis"
    assert src["card"]["credit"] == "as the model card or paper reports it"


def test_reported_numbers_never_reach_the_results_or_an_average(svc, monkeypatch):
    client, appmod, _ = svc
    appmod._cache.update(key=None, payload=None, at=0.0)
    before = client.get("/api/results").json()
    monkeypatch.setattr(config, "AA_API_KEY", KEY)
    reported.import_aa(fetch=canned())
    appmod._cache.update(key=None, payload=None, at=0.0)
    after = client.get("/api/results").json()
    # the payload is the same, model for model: no row, no cell, no average moved
    assert [m["id"] for m in after["models"]] == [m["id"] for m in before["models"]]
    assert after["cells"] == before["cells"]
    assert [(m["avg"], m["partialAvg"]) for m in after["models"]] == \
        [(m["avg"], m["partialAvg"]) for m in before["models"]]
    blob = json.dumps(after)
    assert "Frontier Test" not in blob and "Artificial Analysis" not in blob


def test_an_alias_makes_two_names_one_model(svc, monkeypatch):
    client, _, _ = svc
    monkeypatch.setattr(config, "AA_API_KEY", KEY)
    reported.import_aa(fetch=canned())
    # typed in from its card, by another name
    r = client.post("/api/reported/cards", json={
        "model": "frontier-test-5.5", "maker": "openai", "benchmark": "MMLU", "value": "91.2%",
        "setting": "5-shot", "url": "https://example.org/card", "date": "2026-09-20",
        "entered_by": "masein"})
    assert r.status_code == 200, r.text
    ids = {s["model"] for s in client.get("/api/reported").json()["scores"]
           if s["benchmark"] in ("MMLU", "MMLU-Pro") and "5.5" in s["model"]}
    assert ids == {"reported/openai/frontier-test-5.5"}              # one key already
    # measured here too (12m.3: through OpenRouter) — an alias joins them
    r = client.post("/api/reported/aliases", json={"alias": "openai/frontier-test-5.5",
                                                    "target": "fx/good-750m", "by": "masein"})
    assert r.status_code == 200
    view = client.get("/api/reported").json()
    m = next(x for x in view["models"] if x["key"] == "openai/frontier-test-5.5")
    assert m["id"] == m["measured"] == "fx/good-750m"
    assert {s["source"] for s in view["scores"] if s["model"] == "fx/good-750m"} == {"aa", "card"}
    assert view["aliases"] == [{"alias": "openai/frontier-test-5.5", "target": "fx/good-750m"}]
    r = client.post("/api/reported/aliases/delete", json={"alias": "openai/frontier-test-5.5",
                                                           "by": "masein"})
    assert r.status_code == 200 and client.get("/api/reported").json()["aliases"] == []


@pytest.mark.parametrize("field,value,why", [
    ("model", "", "The model, as its card names it"),
    ("value", "lots", "The value, a number: 86.4 or 86.4%"),
    ("setting", "", 'Their setting: the shots, and chain of thought or not ("5-shot, CoT")'),
    ("url", "example.org", "The source: the card's or paper's address, https://…"),
    ("date", "yesterday", "The date the card gives it, as 2026-09-28"),
    ("entered_by", "", "Who entered it")])
def test_a_card_number_is_refused_in_one_line(svc, field, value, why):
    client, _, _ = svc
    body = {"model": "Frontier Test 5.5", "maker": "OpenAI", "benchmark": "MMLU-Pro",
            "value": "86.4", "setting": "5-shot, CoT", "url": "https://example.org/card",
            "date": "2026-09-20", "entered_by": "masein", field: value}
    r = client.post("/api/reported/cards", json=body)
    assert r.status_code == 422 and r.json()["detail"] == why


def test_the_import_runs_once_a_day_off_the_queue(monkeypatch):
    ran = []
    monkeypatch.setattr(reported, "run_all", lambda **k: ran.append(1))
    monkeypatch.setattr(reported, "_daily", {"day": None, "lock": reported._daily["lock"]})
    monkeypatch.setattr(config, "REPORTED_DAILY", False)
    assert reported.daily() is False
    monkeypatch.setattr(config, "REPORTED_DAILY", True)
    assert reported.daily() is True and reported.daily() is False
