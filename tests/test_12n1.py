"""12n.1, the parts the page doesn't compute: one maker by one name ("Google
DeepMind" and "Google" are Google, and one model across sources joins
itself), and a served model and its GGUF file as one model — by the file its
server reports, a setup served with its own routing joining the GGUF's
setup of that name, MTP left its own. Fixtures only; nothing is fetched."""

from __future__ import annotations

import copy

import pytest

import report_lm_eval as report
from conftest import make_service
from service import config, db, reported


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "AA_API_KEY", "")
    yield client
    client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# one Google
# ---------------------------------------------------------------------------

def test_one_maker_by_one_name():
    assert reported.maker("Google DeepMind") == reported.maker("google") == "Google"
    assert reported.maker(" DeepMind ") == "Google"
    assert reported.maker("OpenAI") == "OpenAI" and reported.maker("") == ""
    assert reported.key("Gemini 3.7 Flash", "Google DeepMind") == "google/gemini-3.7-flash"
    assert reported.key("google/gemini-3.7-flash") == "google/gemini-3.7-flash"
    assert reported.canon("google-deepmind/gemini-3-pro") == "google/gemini-3-pro"
    assert reported.canon("openai/gpt-5.5") == "openai/gpt-5.5"


def test_one_model_from_two_sources_is_one_row_under_one_google(svc):
    # Epoch AI files it under "Google DeepMind" — here as an import from before
    # 12n.1, its key still carrying the old maker — and a card under "Google"
    db.reported_import_add("epoch", "f" * 64, [{
        "key": "google-deepmind/gemini-3.7-flash", "name": "Gemini 3.7 Flash (high)",
        "maker": "Google DeepMind", "benchmark": "GPQA diamond", "unit": "share", "value": 0.81,
        "se": 0.02, "setting": "Epoch AI's own run", "url": "https://epoch.ai/benchmarks"}])
    reported.card_add({"model": "Gemini 3.7 Flash", "maker": "Google", "benchmark": "MMLU-Pro",
                       "value": "84%", "setting": "5-shot, CoT", "url": "https://example.org/card",
                       "date": "2026-09-01"}, "masein")
    view = svc.get("/api/reported").json()
    [m] = [m for m in view["models"] if "gemini" in m["key"]]
    assert m["id"] == "reported/google/gemini-3.7-flash" and m["maker"] == "Google"
    # Epoch's own name for the maker stays with it, for the tooltip
    assert m["maker_as"] == "Google DeepMind"
    assert {(s["source"], s["benchmark"]) for s in view["scores"] if s["model"] == m["id"]} == {
        ("epoch", "GPQA diamond"), ("card", "MMLU-Pro")}
    # the day each source's numbers were imported, for the Frontier view's credit
    assert view["sources"]["epoch"]["imported"] and view["sources"]["aa"]["imported"] == ""


def test_an_alias_typed_with_the_old_maker_still_joins(svc):
    db.reported_import_add("epoch", "e" * 64, [{
        "key": "google/gemini-3.8-flash", "name": "Gemini 3.8 Flash", "maker": "Google",
        "benchmark": "GPQA diamond", "unit": "share", "value": 0.7, "setting": "Epoch AI's own run",
        "url": ""}])
    reported.alias_set("google-deepmind/gemini-3.8-flash", "fx/good-750m", "masein")
    view = reported.view(board=["fx/good-750m"])
    assert [s["model"] for s in view["scores"]] == ["fx/good-750m"]


# ---------------------------------------------------------------------------
# a served model and its GGUF file are one model
# ---------------------------------------------------------------------------

AS_BUILT = {"id": "as-built", "name": "as built", "env": {}, "flags": []}
LOOK = {"id": "la1", "name": "lookahead 1", "env": {"LLAMA_MOE_ROUTE_MODE": "lookahead"},
        "flags": []}


def cell(v):
    return {"v": v, "se": 0.01, "n": 1000, "full": True, "subset": 0, "chance": 0.25}


def gguf_data(pin=None, path="/home/masein/k4-LDA-Q4.gguf"):
    gid = "gguf/k4-lda"
    return {"group": "Measured on the GGUF", "order": ["hellaswag", "winogrande"],
            "models": {gid: {"hellaswag": cell(0.8)}},
            "setups": {gid: [{**AS_BUILT, "current": True, "benches": {"hellaswag": cell(0.8)}},
                             {**LOOK, "current": True, "benches": {"hellaswag": cell(0.83)}}]},
            "history": {gid: [{"sid": 7, "status": "done"}]},
            "pairs": [{"a": gid, "b": gid, "setup_a": "lookahead 1", "setup_b": "as built",
                       "by": {}}],
            "registered": {gid: {"name": "k4-LDA (GGUF)", "path": path, "how": "llama.cpp, Q4",
                                 "pin": {"sha256": "ab" * 32, "size": 20_000_000_000,
                                         "name": "k4-LDA-Q4.gguf"} if pin is None else pin,
                                 "setups": [AS_BUILT, LOOK]}}}


def served_of(name, size=19_900_000_000, file="k4-LDA-Q4.gguf", **kw):
    return {"name": name, "how": "llama.cpp fork", "pin": {"model": file, "file": file,
                                                           "size": size}, **kw}


def test_a_served_model_takes_its_files_gguf_results():
    served = {"served/k4-lda": served_of("k4-LDA")}
    g, same = report.join_served_gguf(served, gguf_data())
    sid = "served/k4-lda"
    assert same == {"gguf/k4-lda": sid, "gguf/k4-lda · lookahead 1": f"{sid} · lookahead 1"}
    assert "gguf/k4-lda" not in g["registered"] and "gguf/k4-lda" not in g["models"]
    # its own provenance kept: the GGUF's file, pin and how — and the id it is measured as
    reg = g["registered"][sid]
    assert reg["served"] and reg["gguf_id"] == "gguf/k4-lda" and reg["how"] == "llama.cpp, Q4"
    assert g["models"][sid]["hellaswag"]["v"] == 0.8
    assert [x["id"] for x in g["setups"][sid]] == ["as-built", "la1"]
    assert g["history"][sid][0]["sid"] == 7
    assert g["pairs"][0]["a"] == g["pairs"][0]["b"] == sid


def test_a_served_setup_joins_the_ggufs_setup_of_that_name_and_mtp_stays_apart():
    served = {"served/k4-lda": served_of("k4-LDA"),
              "served/k4-lda-lookahead-1": served_of("k4-LDA · lookahead 1"),
              "served/k4-lda-mtp": served_of("k4-LDA · MTP n_max 3")}
    g, same = report.join_served_gguf(served, gguf_data())
    look = "served/k4-lda-lookahead-1"
    assert same["gguf/k4-lda · lookahead 1"] == look
    assert g["models"][look]["hellaswag"]["v"] == 0.83
    assert g["registered"][look]["setup"] == "lookahead 1"
    # the setup is the served setup's row now, never a row of its own
    assert [x.get("joined") for x in g["setups"]["served/k4-lda"]] == [None, look]
    assert "served/k4-lda-mtp" not in g["registered"] and "served/k4-lda-mtp" not in same.values()


@pytest.mark.parametrize("served, why", [
    ({"a": served_of("k4-LDA"), "b": served_of("k4-LDA again")}, "two plain entries: which?"),
    ({"a": served_of("k4-LDA", size=14_000_000_000)}, "another size: another file"),
    ({"a": served_of("k4-LDA", file="k4-LDA-Q8.gguf")}, "another name"),
    ({"a": served_of("k4-LDA", size=None)}, "no size to check"),
    ({"a": served_of("k4-LDA", gguf_path="/elsewhere.gguf")}, "it has a GGUF of its own"),
])
def test_nothing_joins_that_isnt_surely_the_same_file(served, why):
    g, same = report.join_served_gguf(served, gguf_data())
    assert same == {}, why
    assert "gguf/k4-lda" in g["registered"]


def test_sha256_decides_when_both_have_one():
    g = gguf_data()["registered"]["gguf/k4-lda"]
    pin = {"file": "k4-LDA-Q4.gguf", "size": 20_000_000_000}
    assert report.same_file(pin, g)
    assert report.same_file(pin, g, served_sha="ab" * 32)
    assert not report.same_file(pin, g, served_sha="cd" * 32)
    # the GGUF entry not measured yet has no pin: its name alone is not enough
    assert not report.same_file(pin, {"path": "/x/k4-LDA-Q4.gguf"})


def test_on_the_board_they_are_one_row(tmp_path):
    served = {"served/k4-lda": served_of("k4-LDA"),
              "served/k4-lda-lookahead-1": served_of("k4-LDA · lookahead 1")}
    gguf = gguf_data()
    before = copy.deepcopy(gguf)
    data = report.build_payload({}, "t", "fixture", served=served, gguf=gguf)
    ids = [m["id"] for m in data["models"]]
    assert "gguf/k4-lda" not in ids and not [i for i in ids if i.startswith("gguf/k4-lda ·")]
    assert {"served/k4-lda", "served/k4-lda-lookahead-1"} <= set(ids)
    assert data["sameAs"]["gguf/k4-lda"] == "served/k4-lda"
    assert data["gguf"]["models"]["served/k4-lda"]["hellaswag"]["v"] == 0.8
    assert data["gguf"]["models"]["served/k4-lda-lookahead-1"]["hellaswag"]["v"] == 0.83
    # the caller's data is left as it was
    assert gguf == before
