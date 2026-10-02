"""15.6: the battery's items hash, whatever form the datasets library stored
them in. datasets 4.6.1 and earlier load IFEval's kwargs as one struct — every
key in every dict, null where it's unset (the server's items-v1.jsonl of 29
Sep) — and 4.7.0 and later keep each dict as written (a fresh build on a
rented GPU): the two hashed fd239fd498d8e7a6 and fe5fe34ffa0968a5, and the
rented GPU refused to start. Now the hash reads IFEval's kwargs without their
null-valued keys, the hash both come to is committed
(eval_tasks/devicemark/items-v1.sha256.json), and a run here, a rented GPU's
run and an import each check it. The battery is invented (as on the server);
nothing runs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import devicemark as dm
import import_remote as ir
import remote_bundle as rb
from service import config, db, runner
from service import devicemark as sdm
from test_12q_devicemark_runs import ME, svc, write_items  # noqa: F401 — svc is the fixture
from test_15_1_remote_run import box, lm_eval, run  # noqa: F401 — box is the fixture
from test_15_2_import import GEMMA, board, imported, local, remote  # noqa: F401

REPO = Path(__file__).resolve().parents[1]
# three of IFEval's shapes: no kwargs, one set, two set
KWARGS = [[{}], [{"num_highlights": 3}, {}], [{"num_words": 300, "relation": "at least"}]]


def as_written(items: dict) -> dict:
    """the invented battery with IFEval's kwargs of three shapes, kept as
    written — datasets 4.7.0 and later"""
    out = {k: dict(v) for k, v in items.items()}
    for n, k in enumerate(k for k in dm.keys_for("full") if k[0] == "ifeval"):
        out[k]["kwargs"] = [dict(d) for d in KWARGS[n % 3]]
        out[k]["instruction_id_list"] = ["x"] * len(out[k]["kwargs"])
    return out


def null_filled(items: dict) -> dict:
    """the same, as datasets 4.6.1 and earlier keep them: every key any dict
    has, in every dict, null where it's unset"""
    keys = sorted({k for it in items.values() if it["bench"] == "ifeval"
                   for d in it["kwargs"] for k in d})
    out = {k: dict(v) for k, v in items.items()}
    for it in out.values():
        if it["bench"] == "ifeval":
            it["kwargs"] = [{k: d.get(k) for k in keys} for d in it["kwargs"]]
    return out


def store(path: Path, items: dict) -> None:
    """a cache as load_items writes one"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": dm.VERSION}) + "\n" + "".join(
        json.dumps(items[k]) + "\n" for k in dm.keys_for("full")), encoding="utf-8")


def main_hash(items: dict) -> str:
    """the items hash as 15.1 took it: the items as stored"""
    rows = [items[k] for k in dm.keys_for("full")]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the hash
# ---------------------------------------------------------------------------

def test_both_stored_forms_hash_the_same(tmp_path):
    written = as_written(write_items(tmp_path / "invented.jsonl"))
    filled = null_filled(written)
    assert filled != written                                   # the two forms differ…
    assert main_hash(filled) != main_hash(written)             # …and 15.1's hash told them apart
    # read back from a server's cache of each form
    store(tmp_path / "server" / "items-v1.jsonl", filled)
    store(tmp_path / "fresh" / "items-v1.jsonl", written)
    a = dm.load_items(tmp_path / "server" / "items-v1.jsonl")
    b = dm.load_items(tmp_path / "fresh" / "items-v1.jsonl")
    assert a[next(k for k in dm.keys_for("full") if k[0] == "ifeval")]["kwargs"] == [
        {"num_highlights": None, "num_words": None, "relation": None}]   # as the server keeps it
    assert dm.battery_hashes(a) == dm.battery_hashes(b)
    # the form kept as written is what the hash reads: the server's existing
    # cache comes to a fresh build's hash, and nothing on the server is rewritten
    assert dm.battery_hashes(a)["items"] == main_hash(written)
    assert (tmp_path / "server" / "items-v1.jsonl").read_text().count('"num_words": null') > 0
    # only null-valued keys go: a set key is kept, whatever its value
    one = {("ifeval", "1"): {"bench": "ifeval", "key": "1",
                             "kwargs": [{"relation": "at least", "num_words": 0, "x": None}]}}
    assert dm.normal_item(one[("ifeval", "1")])["kwargs"] == [{"relation": "at least",
                                                                "num_words": 0}]


def test_the_expected_hash_is_committed_and_read_by_the_board():
    f = REPO / "eval_tasks" / "devicemark" / "items-v1.sha256.json"
    got = json.loads(f.read_text())
    assert got["battery"] == dm.VERSION and re.fullmatch(r"[0-9a-f]{64}", got["items"])
    # both machines' builds come to it (README of the file: checked on 2 Oct)
    assert got["items"].startswith("fe5fe34ffa0968a5")
    assert dm.expected_items_hash() == got["items"] == config._items_sha256()
    # in the runner image, which carries eval_tasks/devicemark/ (15.1)
    assert "eval_tasks/devicemark/" in (REPO / "Dockerfile").read_text().split("AS runner")[1] \
        .split("AS board")[0]


# ---------------------------------------------------------------------------
# where it's checked
# ---------------------------------------------------------------------------

def test_a_battery_here_that_isnt_the_repos_asks_nothing(svc, monkeypatch):  # noqa: F811
    asked = []
    monkeypatch.setattr(runner, "_run_task", lambda *a, **k: asked.append(a) or 0)
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: {
        "kind": "instruct", "params": 4.0e9, "vocab": 248320, "batch": 8, "need_gb": 10.0,
        "remote_code": False, "has_template": True, "kind_reason": "chat template",
        "archinfo": {"thinking": "switch", "think_end": "</think>"}})
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    monkeypatch.setattr(runner, "gen_backend", lambda: ("hf", "vLLM is not installed here"))
    monkeypatch.setattr(config, "DM_ITEMS_SHA256", "0" * 64)
    sid = db.add("org/m", "instruct", "devicemark", ME, "", thinking=True, part="full")
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "failed" and not asked
    assert row["error"].startswith(f"the battery's items hash to "
                                   f"{dm.battery_hashes(dm.load_items(config.DM_ITEMS))['items'][:16]}"
                                   ", and the repo expects 0000000000000000 (items-v1.sha256.json)")
    assert row["error"].endswith("— nothing was asked")
    # a served setup's run reads the same battery, and stops the same way
    status, line = sdm.run(sid, {"part": "full"}, {}, config.LOGS_DIR / "x.log")
    assert (status, line) == ("failed", row["error"])


def test_the_repos_battery_runs(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "DM_ITEMS_SHA256",
                        dm.battery_hashes(dm.load_items(config.DM_ITEMS))["items"])
    assert dm.items_differ(dm.load_items(config.DM_ITEMS), config.DM_ITEMS_SHA256) is None


def test_a_rented_gpu_refuses_a_battery_that_isnt_the_repos(box, tmp_path, monkeypatch):  # noqa: F811
    asked = lm_eval(box)
    monkeypatch.setattr(config, "DM_ITEMS_SHA256", "1" * 64)
    with pytest.raises(SystemExit, match=r"the battery's items hash to \w{16}, and the repo expects "
                                         r"1111111111111111 .*the server would refuse the bundle\. "
                                         r"Nothing was run\."):
        run(tmp_path / "run", "--only", "dm_math")
    assert asked == []
    # the repo's: it runs, with no --battery given
    items = write_items(tmp_path / "x.jsonl")
    monkeypatch.setattr(config, "DM_ITEMS_SHA256", dm.battery_hashes(items)["items"])
    assert run(tmp_path / "run2", "--only", "dm_math") == 0 and len(asked) == 100


def test_an_import_onto_a_server_whose_battery_isnt_the_repos_is_refused(board, tmp_path,  # noqa: F811
                                                                          monkeypatch):
    path = remote(tmp_path / "box", GEMMA, "dm_math")
    monkeypatch.setattr(config, "DM_ITEMS_SHA256", "2" * 64)
    code, said = imported(path)
    assert code == ir.REFUSED
    assert any(x.startswith("refused — this server's battery: its items hash to ") and
               x.endswith("and the repo expects 2222222222222222") for x in said)


def test_datasets_and_pyarrow_are_pinned_and_recorded():
    req = (REPO / "requirements.txt").read_text()
    assert re.search(r"^datasets==\d+\.\d+\.\d+$", req, re.M)
    assert re.search(r"^pyarrow==\d+\.\d+\.\d+$", req, re.M)
    libs = rb.library_versions()
    assert {"datasets", "pyarrow"} <= set(libs)
