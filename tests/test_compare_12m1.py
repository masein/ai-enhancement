"""12m.1, the server's half: a GGUF's own row on Models holds its "as built"
results and nothing else (it fell back to the first setup with results, so
the phone build's plain row showed its lookahead numbers as if built so,
beside the "· lookahead 1" row showing them too), and a comparison is saved
as a view of its models, two to eight."""

from __future__ import annotations

import gguf_worker as gw
from service import config
from test_gguf_12f3 import svc  # noqa: F401 — the service with the GGUF datasets

LOOK = "lookahead 1: LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=1"


def _measure(client, appmod, box, gid, setups):
    ids = client.post("/api/gguf/runs", json={"model": gid, "benchmarks": ["mmlu", "arc_easy"],
                                              "setups": setups, "by": "masein"}).json()["ids"]
    for _ in ids:
        gw.Worker(config.RESULTS_ROOT, box["bin"], poll=0.05).once()
    client.get("/api/submissions")
    appmod._cache.update(key=None, payload=None, at=0.0)
    return client.get("/api/results").json()


def test_a_ggufs_own_row_is_its_as_built_results_alone(svc, monkeypatch):  # noqa: F811
    client, appmod, box = svc
    monkeypatch.setenv("FAKE_PPL_ACC", "0.5")
    monkeypatch.setenv("FAKE_PPL_ACC_LOOKAHEAD", "0.8")
    m = client.post("/api/gguf/models", json={"name": "LDA", "path": str(box["model"]),
                                              "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "k4",
                                              "setups": LOOK, "by": "masein"}).json()["model"]
    gid, look = m["id"], next(x["id"] for x in m["setups"] if x["name"] == "lookahead 1")
    # measured in lookahead only: its own row has nothing, the setup's row has it
    j = _measure(client, appmod, box, gid, [look])
    assert gid not in j["gguf"]["models"]
    rows = {x["id"]: x for x in j["models"]}
    assert abs(j["gguf"]["models"][f"{gid} · lookahead 1"]["mmlu"]["v"] - 5 / 6) < 1e-6
    assert rows[f"{gid} · lookahead 1"]["rowOf"] == gid
    # its page still has every setup
    sets = {x["name"]: x for x in j["gguf"]["setups"][gid]}
    assert sets["lookahead 1"]["benches"] and not sets["as built"]["benches"]
    # measured as built too: its row is that, never the lookahead numbers
    j = _measure(client, appmod, box, gid, ["as-built"])
    own = j["gguf"]["models"][gid]["mmlu"]
    assert own["setup"] == "as built" and abs(own["v"] - 3 / 6) < 1e-6


def test_a_comparison_is_saved_as_its_models(svc):  # noqa: F811
    client, _, _ = svc
    body = lambda spec: {"name": "Phone vs original", "by": "masein", "spec": spec}  # noqa: E731
    r = client.post("/api/views", json=body({"view": "compare",
                                             "models": ["fx/good-750m", "fx/below-135m-it"]}))
    assert r.status_code == 200, r.text
    assert r.json()["spec"] == {"view": "compare",
                                "models": ["fx/good-750m", "fx/below-135m-it"]}
    for models in (["fx/good-750m"], [f"m/{i}" for i in range(9)]):
        r = client.post("/api/views", json={**body({"view": "compare", "models": models}),
                                             "name": "x"})
        assert r.status_code == 422 and r.json()["detail"] == "a comparison is two to eight model ids"
    r = client.post("/api/views", json={**body({"view": "radar", "models": ["a", "b"]}),
                                         "name": "y"})
    assert r.status_code == 422
    # a view of the table is as it was
    r = client.post("/api/views", json={**body({"chip": "trust", "cols": ["mmlu"]}), "name": "z"})
    assert r.status_code == 200 and r.json()["spec"] == {"chip": "trust", "cols": ["mmlu"],
                                                         "models": None}
