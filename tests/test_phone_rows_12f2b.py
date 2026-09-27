"""12f.2b: a GGUF's setup is a row of its own on Models — "Qwen3.6 original ·
lookahead 1" — with only that setup's GGUF scores, its model page its model's,
and no part in any lm_eval cell, average or count."""

from __future__ import annotations

import gguf_worker as gw
from service import config
from test_gguf_12f3 import LOOK, svc  # noqa: F401


def test_a_gguf_setup_is_a_row_with_only_its_gguf_scores(svc, monkeypatch):  # noqa: F811
    client, appmod, box = svc
    before = client.get("/api/results").json()
    gid = client.post("/api/gguf/models", json={
        "name": "Qwen3.6 original", "path": str(box["model"]), "based_on": "Qwen/Qwen3.6-35B-A3B",
        "how": "unsloth", "setups": LOOK, "by": "masein"}).json()["model"]["id"]
    monkeypatch.setenv("FAKE_PPL_ACC", "0.5")
    monkeypatch.setenv("FAKE_PPL_ACC_LOOKAHEAD", "0.8")
    ids = client.post("/api/gguf/runs", json={"model": gid, "benchmarks": ["mmlu"],
                                              "by": "masein"}).json()["ids"]
    for _ in ids:
        gw.Worker(config.RESULTS_ROOT, box["bin"], poll=0.05).once()
    client.get("/api/submissions")
    appmod._cache.update(key=None, payload=None, at=0.0)
    j = client.get("/api/results").json()
    rows = {m["id"]: m for m in j["models"]}
    rid = f"{gid} · lookahead 1"
    row = rows[rid]
    assert row["name"] == "Qwen3.6 original · lookahead 1" and row["rowOf"] == gid
    assert row["avg"] is None and row["judge"] is None and row["official"] is False
    # its column group only, its own setup's numbers
    assert abs(j["gguf"]["models"][rid]["mmlu"]["v"] - 5 / 6) < 1e-6
    assert abs(j["gguf"]["models"][gid]["mmlu"]["v"] - 3 / 6) < 1e-6
    assert not any(rid in c for c in j["cells"].values())
    # the board's counts and warnings are what they were
    assert j["warnings"] == before["warnings"] and j["cells"] == before["cells"]
