"""12o.1: a served model and its GGUF file are one model — found live, the
join happened for neither of the server's two. Fixtures shaped as the
server's registry and pins are:

- the original: one served entry pinned to "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
  22,842,671,616 bytes as llama-server counts them, and a GGUF entry of the
  same file (22,853,663,008 bytes) whose pin was never written — the worker
  hashed the file, and only its run says so;
- the phone build: served as built, "· lookahead" (not "lookahead 1") with
  its routing in how it's served, and "· MTP"; its GGUF has one setup,
  "lookahead 1".

A setup is matched by its routing, its name the fallback with trailing
numbers aside; MTP stays its own row. And "Same file as", said outright,
wins over any guess. Nothing is asked of a server: records are written as
registering one writes them."""

from __future__ import annotations

import json
import time

import pytest

import gguf_bench as gb
import report_lm_eval as report
from conftest import make_service
from service import config, db, gguf, served

ORIG_FILE = "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
PHONE_FILE = "Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf"
LOOK_ENV = {"LLAMA_MOE_ROUTE_MODE": "lookahead", "LLAMA_MOE_ROUTE_LOOKAHEAD": "1"}
LOOK_ID = gb.setup_id(LOOK_ENV, [])


def served_rec(name, file, size, how, **kw):
    """a served entry as served.register keeps it — its server's pin as
    probe() reads it"""
    rec = {"id": served.slug(name), "name": name, "base_url": "http://host.docker.internal:8091/v1",
           "key": "", "based_on": "Qwen/Qwen3.6-35B-A3B", "how": how, "thinking": "auto",
           "phone": False, "gguf_path": "", "gguf_flags": "", "gguf_setups": [],
           "pin": {"model": file.rsplit("/", 1)[-1], "file": file, "size": size, "ctx": 32768,
                   "build": "b9090"},
           "answered": [], "by": "masein", "at": time.time(), **kw}
    db.served_put(rec)
    served.write_meta(rec)
    return rec["id"]


def gguf_rec(name, path, setups="", pin=None):
    rec = gguf.register({"name": name, "path": path, "based_on": "Qwen/Qwen3.6-35B-A3B",
                         "how": "unsloth UD-Q4_K_XL · llama.cpp b9090", "flags": "",
                         "setups": setups}, "masein")
    if pin:
        gguf._pin(rec["id"], pin)
    return rec["id"]


def result(gid, rid, setup, acc, file):
    """what the host's worker writes when a run is done"""
    out = config.RESULTS_ROOT / "gguf_results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{rid}.json").write_text(json.dumps({
        "id": str(rid), "sid": rid, "model": gid, "status": "done", "line": "done",
        "subset": 0, "finished_at": time.time() - rid, "setup": setup, "file": file,
        "build": "b9090", "flags": [],
        "benchmarks": {"hellaswag": {"status": "done", "acc": acc, "se": 0.004, "n": 10042,
                                     "done": 10042, "total": 10042}},
        "datasets": {"hellaswag": {"file": "hellaswag_val_full.txt", "sha256": "d" * 64}}}))


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    ids = {}
    # the original: its GGUF entry's pin never written; the worker's run holds the file
    ids["g_orig"] = gguf_rec("Qwen3.6-35B-A3B UD-Q4_K_XL (original)",
                             f"/home/masein/models/{ORIG_FILE}")
    result(ids["g_orig"], 11, gb.AS_BUILT, 0.81,
           {"path": f"/home/masein/models/{ORIG_FILE}", "name": ORIG_FILE,
            "size": 22_853_663_008, "sha256": "a" * 64})
    ids["s_orig"] = served_rec("Qwen3.6-35B-A3B original", ORIG_FILE, 22_842_671_616,
                               "llama.cpp b9090 · -ngl 99 --cpu-moe -c 32768")
    # the phone build: as built, lookahead and MTP served; lookahead 1 a GGUF setup
    ids["g_phone"] = gguf_rec(
        "Qwen3.6 k4-LDA (phone build)", f"/home/masein/models/{PHONE_FILE}",
        setups="lookahead 1: LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=1",
        pin={"sha256": "b" * 64, "size": 21_104_000_000, "name": PHONE_FILE})
    look = {"id": LOOK_ID, "name": "lookahead 1", "env": LOOK_ENV, "flags": []}
    phone_file = {"path": f"/home/masein/models/{PHONE_FILE}", "name": PHONE_FILE,
                  "size": 21_104_000_000, "sha256": "b" * 64}
    result(ids["g_phone"], 21, gb.AS_BUILT, 0.78, phone_file)
    result(ids["g_phone"], 22, look, 0.80, phone_file)
    # its server reports the path it loaded, not only the name
    ids["s_phone"] = served_rec("Qwen3.6 k4-LDA", f"/models/{PHONE_FILE}", 21_093_000_000,
                                "llama.cpp fork b9090 · -ngl 99 --cpu-moe")
    ids["s_look"] = served_rec("Qwen3.6 k4-LDA · lookahead", f"/models/{PHONE_FILE}",
                               21_093_000_000, "llama.cpp fork b9090 · "
                               "LLAMA_MOE_ROUTE_MODE=lookahead · -ngl 99 --cpu-moe")
    ids["s_mtp"] = served_rec("Qwen3.6 k4-LDA · MTP", f"/models/{PHONE_FILE}", 21_093_000_000,
                              "llama.cpp fork b9090 · --spec-type mtp --draft-max 3")
    yield client, ids
    client.__exit__(None, None, None)


def joined():
    sv = report.load_served(config.OUT_DIR)
    g = report.load_gguf(config.RESULTS_ROOT, config.OUT_DIR, sv)
    return sv, g, *report.join_served_gguf(sv, g)


def test_the_original_joins_on_the_file_its_worker_hashed(svc):
    _, ids = svc
    assert (db.gguf_get(ids["g_orig"]) or {}).get("pin") in (None, {})    # as on the server
    sv, g, out, same = joined()
    assert same[ids["g_orig"]] == ids["s_orig"]
    reg = out["registered"][ids["s_orig"]]
    assert reg["served"] and reg["gguf_id"] == ids["g_orig"]
    assert out["models"][ids["s_orig"]]["hellaswag"]["v"] == 0.81


def test_the_phone_builds_lookahead_joins_its_setup_by_routing_and_mtp_stays(svc):
    _, ids = svc
    _, _, out, same = joined()
    assert same[ids["g_phone"]] == ids["s_phone"]
    assert same[f"{ids['g_phone']} · lookahead 1"] == ids["s_look"]
    assert out["models"][ids["s_look"]]["hellaswag"]["v"] == 0.80
    assert out["models"][ids["s_phone"]]["hellaswag"]["v"] == 0.78
    assert out["registered"][ids["s_look"]]["setup"] == "lookahead 1"
    assert ids["s_mtp"] not in same.values() and ids["s_mtp"] not in out["registered"]


def test_on_the_board_each_is_one_row_and_mtp_its_own(svc):
    sv, g, _, _ = joined()
    data = report.build_payload({}, "t", "fixture", served=sv, gguf=g)
    ids_ = {m["id"] for m in data["models"]}
    _, ids = svc
    assert not {ids["g_orig"], ids["g_phone"]} & ids_
    assert {ids["s_orig"], ids["s_phone"], ids["s_look"], ids["s_mtp"]} <= ids_


@pytest.mark.parametrize("how, name, to", [
    # the name alone, a trailing number aside
    ("llama.cpp fork b9090", "Qwen3.6 k4-LDA · lookahead", "look"),
    ("llama.cpp fork b9090", "Qwen3.6 k4-LDA · lookahead 1", "look"),
    # routing that isn't this setup's: not it, whatever the name says
    ("LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=2", "k4-LDA · lookahead", "other"),
    # routing wins over a name that says nothing
    ("LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=1", "k4-LDA · fast", "look"),
    ("-ngl 99", "k4-LDA", None),
])
def test_a_setup_is_matched_by_its_routing_its_name_the_fallback(how, name, to):
    look = {"id": LOOK_ID, "name": "lookahead 1", "env": LOOK_ENV, "flags": []}
    got = report._setup_of({"name": name, "how": how}, [look])
    assert (got["id"] if isinstance(got, dict) else got) == (LOOK_ID if to == "look" else to)


def test_same_file_as_said_outright_wins_over_any_guess(svc, monkeypatch):
    client, ids = svc
    # a second plain entry of the phone file: which? — no guess joins either
    extra = served_rec("Qwen3.6 k4-LDA again", PHONE_FILE, 21_093_000_000, "llama.cpp fork b9090")
    _, _, _, same = joined()
    assert ids["g_phone"] not in same
    r = client.post("/api/served/same-as", json={"served": ids["s_phone"], "gguf": ids["g_phone"],
                                                 "setup": "as-built", "by": "masein"})
    assert r.status_code == 200 and r.json()["same_as"] == {"gguf": ids["g_phone"],
                                                            "setup": "as-built"}
    _, _, _, same = joined()
    assert same[ids["g_phone"]] == ids["s_phone"] and extra not in same.values()
    # "not the same file": out of every join, however alike
    client.post("/api/served/same-as", json={"served": ids["s_orig"], "gguf": "none",
                                             "by": "masein"})
    _, _, _, same = joined()
    assert ids["g_orig"] not in same
    # a link across names and routing: the MTP entry said to be lookahead 1
    client.post("/api/served/same-as", json={"served": ids["s_mtp"], "gguf": ids["g_phone"],
                                             "setup": LOOK_ID, "by": "masein"})
    _, _, _, same = joined()
    assert same[f"{ids['g_phone']} · lookahead 1"] == ids["s_mtp"]
    # from the GGUF entry's side: lookahead 1 is no one's said link now — a guess again
    r = client.post("/api/served/same-as", json={"served": "", "gguf": ids["g_phone"],
                                                 "setup": LOOK_ID, "by": "masein"})
    assert r.json() == {"cleared": [ids["s_mtp"]]}
    _, _, _, same = joined()
    assert same[f"{ids['g_phone']} · lookahead 1"] == ids["s_look"]
    # registered again, an entry keeps what it was said to be
    monkeypatch.setattr(served, "probe", lambda base, key: {
        "model": PHONE_FILE, "file": f"/models/{PHONE_FILE}", "size": 21_093_000_000,
        "ctx": 32768, "build": "b9091", "answered": []})
    served.register({"name": "Qwen3.6 k4-LDA", "base_url": "http://host.docker.internal:8091/v1",
                     "how": "llama.cpp fork b9091 · -ngl 99 --cpu-moe"}, "masein")
    assert db.served_get(ids["s_phone"])["same_as"] == {"gguf": ids["g_phone"],
                                                        "setup": "as-built"}


@pytest.mark.parametrize("body, words", [
    ({"served": "served/nobody", "gguf": "none"}, "no served model served/nobody"),
    ({"gguf": "gguf/nothing", "served": "S"}, "no GGUF entry gguf/nothing"),
    ({"gguf": "G", "served": "S", "setup": "s12345678"}, "has no setup s12345678"),
    ({"served": "", "gguf": ""}, "which served model"),
])
def test_a_link_names_what_is_there(svc, body, words):
    client, ids = svc
    body = {k: ids["s_orig"] if v == "S" else ids["g_orig"] if v == "G" else v
            for k, v in body.items()}
    r = client.post("/api/served/same-as", json={"by": "masein", **body})
    assert r.status_code == 422 and words in r.json()["detail"]
