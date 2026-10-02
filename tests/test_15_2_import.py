"""15.2: a bundle from a rented GPU, into the board — scripts/import_remote.py.
It refuses a bundle whose battery, protocol or pinned libraries aren't the
server's, saying which; a partial bundle merges with the tasks already here
and the row scores as a full local run of the same answers does; the row's
setup says it ran on a rented GPU; the Runs list gets the import with the
bundle's log; and a bundle imported twice changes nothing. The bundles are
made here by remote_run.py with 15.1's stand-in lm_eval; nothing runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import devicemark as dm
import import_remote as ir
import remote_bundle as rb
import remote_run as rr
from service import config, db, runner
from service import devicemark as sdm
from test_12q_devicemark_runs import ME, svc  # noqa: F401 — svc is the fixture
from test_12q_devicemark_runs import write_items
from test_15_1_remote_run import GPU, LIBS, REV, lm_eval

GEMMA, QWEN = "google/gemma-4-E2B-it", "Qwen/Qwen3.5-4B"
GPU_LINE = "run on a rented GPU (NVIDIA GeForce RTX 4090)"


@pytest.fixture
def board(svc, monkeypatch):  # noqa: F811
    """the server, and what a run anywhere needs given: a model that loads,
    the GPU's name, the libraries' versions"""
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: {
        "kind": "instruct", "params": 4.0e9, "vocab": 248320, "batch": 8, "need_gb": 10.0,
        "remote_code": False, "has_template": True, "kind_reason": "chat template",
        "archinfo": {"thinking": "switch", "think_end": "</think>"}})
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    monkeypatch.setattr(runner, "gen_backend", lambda: ("hf", "vLLM is not installed here"))
    monkeypatch.setattr(sdm, "prompt_tokens", lambda *a, **k: (911, True))
    monkeypatch.setattr(rb, "gpu_info", lambda: dict(GPU))
    monkeypatch.setattr(rb, "library_versions", lambda: dict(LIBS))
    monkeypatch.setattr(rr, "model_revision", lambda m: REV)
    monkeypatch.setattr(rr, "board_commit", lambda: "f" * 40)
    lm_eval(monkeypatch)
    return svc


def remote(out: Path, model: str, *only: str, shard: str = "") -> Path:
    """a rented GPU's run and its bundle — the server's config put back after.
    15.5: `shard`, "i/n", one shard of it"""
    keep = {k: getattr(config, k) for k in [*rr.ENV, "OUT_DIR"]}
    lock, env = runner.LOCK, {v: os.environ.get(v) for v in [*rr.ENV.values(), "PATH"]}
    try:
        write_items(out / "bench" / "devicemark" / "items-v1.jsonl")
        args = ["--model", model, "--thinking", "on", "--out", str(out)]
        for t in only:
            args += ["--only", t]
        if shard:
            args += ["--shard", shard]
        assert rr.main(args) == 0
    finally:
        for k, v in keep.items():
            setattr(config, k, v)
        runner.LOCK = lock
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out / rb.bundle_name("devicemark", model, True, dm.parse_shard(shard))


def local(model: str, *only: str) -> dict:
    """a run of the battery here, as the queue runs one"""
    sid = db.add(model, "instruct", "devicemark", ME, "", thinking=True, part="full",
                 tasks=list(only) or None)
    runner.run_submission(db.get(sid))
    return db.get(sid)


def row_of(model: str) -> Path:
    return config.OUT_DIR / (model.replace("/", "__") + "__thinking")


def tamper(path: Path, fix) -> Path:
    """the bundle with its setup changed by `fix`, made whole again"""
    b = rb.read(path)
    setup = json.loads(b["files"]["setup.json"])
    fix(setup)
    files = dict(b["files"])
    files["setup.json"] = json.dumps(setup).encode()
    bundle = json.loads(files["bundle.json"])
    bundle["files"] = {n: hashlib.sha256(d).hexdigest() for n, d in files.items()
                       if n != "bundle.json"}
    files["bundle.json"] = json.dumps(bundle).encode()
    return rb.write(path.with_name("tampered-" + path.name), files)


def imported(path: Path) -> tuple[int, list[str]]:
    said: list[str] = []
    code = ir.import_bundle(path, "masein", said.append)
    return code, said


def tree() -> dict[str, str]:
    return {str(p.relative_to(config.RESULTS_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(config.RESULTS_ROOT.rglob("*")) if p.is_file()}


def test_a_bundle_becomes_a_row_like_any_other(board, tmp_path):
    client = board
    path = remote(tmp_path / "box", GEMMA)
    code, said = imported(path)
    assert code == 0, said
    row = json.loads((row_of(GEMMA) / dm.OUT_NAME).read_text())
    assert row["composite"]["value"] is not None and row["n"] == 596
    assert row["setup"]["where"] == GPU_LINE
    assert row["setup"]["remote"]["dm_ifeval"]["gpu"] == GPU["name"]
    # like any other row: on the board's DeviceMark rows, and the model page's card
    rows = {r["id"]: r for r in client.get("/api/devicemark").json()["rows"]}
    assert rows[f"{GEMMA} · thinking"]["row"]["setup"]["where"] == GPU_LINE
    card = client.get("/api/results").json()["devicemark"][GEMMA]["on"]["row"]
    assert card["where"] == GPU_LINE
    # the Runs list: the import, done, its log the bundle's
    sub = db.recent(5)[0]
    assert (sub["hf_id"], sub["suite"], sub["status"]) == (GEMMA, "devicemark", "done")
    assert sub["note"].startswith(f"imported from a rented GPU ({GPU['name']})")
    assert sub["progress"].startswith("composite ")
    log = client.get(f"/api/runs/{sub['id']}/log", params={"tail": 2000}).text
    assert "===== [1] dm_ifeval (0-shot) =====" in log and "[import] dm_math: 100 answers" in log
    assert said[-1] == f"Runs #{sub['id']}, its log the bundle's"


@pytest.mark.parametrize("what,fix,says", [
    ("battery", lambda s: s["battery"].update(items="0" * 64),
     "refused — the battery's items (each question, key and prompt): the bundle's hash 0000"),
    ("protocol", lambda s: s.update(protocol="devicemark-replica-v2"),
     "refused — the protocol: the bundle's devicemark-replica-v2 (cap 4096, seed 0), this "
     "server's devicemark-replica-v1"),
    ("library", lambda s: s["libraries"].update(transformers="5.6.0"),
     "refused — a pinned library — transformers: the bundle's 5.6.0, this server's 5.5.3"),
    ("kernels", lambda s: s["libraries"].update(fast_kernels=None),
     "refused — a pinned library — the prebuilt fast kernels: the bundle's none, this "
     "server's torch211-cxx11-cu128-x86_64-linux")])
def test_a_bundle_whose_battery_protocol_or_pins_differ_is_refused(board, tmp_path, what, fix,
                                                                    says):
    path = tamper(remote(tmp_path / "box", GEMMA, "dm_math"), fix)
    before, runs = tree(), len(db.recent(500))
    code, said = imported(path)
    assert code == ir.REFUSED
    assert any(x.startswith(says) for x in said), said
    assert said[-1] == "nothing was imported"
    assert tree() == before and len(db.recent(500)) == runs


def test_the_servers_own_libraries_count_too(board, tmp_path, monkeypatch):
    path = remote(tmp_path / "box", GEMMA, "dm_math")
    monkeypatch.setattr(rb, "library_versions", lambda: {**LIBS, "torch": "2.12.0"})
    code, said = imported(path)
    assert code == ir.REFUSED and any("torch: the bundle's 2.11.0, this server's 2.12.0" in x
                                      for x in said)


def test_a_partial_bundle_merges_and_scores_as_a_full_local_run(board, tmp_path):
    # the same answers, all asked here: what the row should come to
    full = local("org/the-same-answers")
    assert full["status"] == "done", full["error"]
    want = json.loads((row_of("org/the-same-answers") / dm.OUT_NAME).read_text())
    # #167's case: MMLU-Pro and MATH here, IFEval from a rented GPU
    local(QWEN, "dm_mmlu_pro", "dm_math")
    code, said = imported(remote(tmp_path / "box", QWEN, "dm_ifeval"))
    assert code == 0, said
    got = json.loads((row_of(QWEN) / dm.OUT_NAME).read_text())
    assert got["composite"] == want["composite"] and got["benches"] == want["benches"]
    assert got["n"] == want["n"] == 596
    assert got["setup"]["where"] == ("IFEval run on a rented GPU (NVIDIA GeForce RTX 4090); "
                                     "MMLU-Pro and MATH on this server")
    assert list(got["setup"]["remote"]) == ["dm_ifeval"]


def test_answers_here_before_are_kept_aside(board, tmp_path):
    local(QWEN, "dm_ifeval")
    before = rb.task_answers(row_of(QWEN) / "dm_ifeval_0shot", "dm_ifeval")
    path = remote(tmp_path / "box", QWEN, "dm_ifeval")
    code, said = imported(path)
    assert code == 0
    aside = list((config.RESULTS_ROOT / "earlier" / row_of(QWEN).name).glob(
        "dm_ifeval_0shot-before-import-*"))
    assert len(aside) == 1 and rb.task_answers(aside[0], "dm_ifeval") == before
    assert any(x.startswith("dm_ifeval: the answers here before are kept at ") for x in said)


def test_the_same_bundle_twice_changes_nothing(board, tmp_path):
    path = remote(tmp_path / "box", GEMMA)
    assert imported(path)[0] == 0
    sid = db.recent(1)[0]["id"]
    before, runs = tree(), len(db.recent(500))
    code, said = imported(path)
    assert code == 0 and said[-1].startswith(f"imported already, as Runs #{sid} on ")
    assert tree() == before and len(db.recent(500)) == runs


def test_a_bundle_missing_answers_is_refused(board, tmp_path):
    path = remote(tmp_path / "box", GEMMA, "dm_math")
    b = rb.read(path)
    files = dict(b["files"])
    name = next(n for n in files if "/samples_dm_math_" in n)
    files[name] = b"\n".join(files[name].splitlines()[:-1]) + b"\n"
    bundle = json.loads(files["bundle.json"])
    bundle["files"] = {n: hashlib.sha256(d).hexdigest() for n, d in files.items()
                       if n != "bundle.json"}
    files["bundle.json"] = json.dumps(bundle).encode()
    code, said = imported(rb.write(tmp_path / "short.tar.gz", files))
    assert code == ir.REFUSED and "refused — dm_math: 99 of 100 items answered" in said


def test_the_server_prints_its_battery_for_the_rented_gpu(board, capsys):
    assert ir.main(["--battery"]) == 0
    want = dm.battery_hashes(dm.load_items(config.DM_ITEMS))
    assert capsys.readouterr().out.splitlines()[0] == f"items {want['items']}"
