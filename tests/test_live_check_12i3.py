"""12i.3: the live check of 12i, 2026-09-26. GET /api/builder returned 500 on
the server — its default prompts were read from docs/, which the image does
not carry — so Build questions never loaded. These load the builder the way
the server is: exam and Everyday banks on file, no draft yet, no checker
chosen, and the judge and the writers on a local vLLM that serves "chat"
backed by gemma weights. And the local model is named by those weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_service
from service import ai_models, builder, config, llm, startup
from test_image_contents import in_image

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = "google/gemma-4-E4B-it"


class FakeVLLM:
    """the box's vLLM, at llm._http: GET /models says what it serves"""
    def __init__(self, real):
        self.real, self.asked = real, 0

    def __call__(self, method, url, headers, body=None, timeout=60.0):
        if url.startswith(config.LOCAL_BASE_URL.rstrip("/")) and url.endswith("/models"):
            self.asked += 1
            return 200, json.dumps({"data": [{"id": "chat", "root": WEIGHTS}]}).encode()
        return self.real(method, url, headers, body, timeout)


@pytest.fixture
def server(tmp_path, monkeypatch):
    """the server's shape: banks on file, no drafts, the judge and the writers
    local, no checker chosen"""
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="chat")
    for k, v in {"JUDGE_PROVIDER": "local", "LLM_PROVIDER": "local", "LLM_MODEL": "chat",
                 "EXAM_PROVIDER": "local", "EXAM_MODEL": "chat",
                 "OPENROUTER_API_KEY": "sk-or-test-not-a-real-key"}.items():
        monkeypatch.setattr(config, k, v)
    monkeypatch.delattr(config, "CHECKER_PROVIDER", raising=False)
    vllm = FakeVLLM(llm._http)
    monkeypatch.setattr(llm, "_http", vllm)
    monkeypatch.setattr(llm, "_SERVED", {})
    monkeypatch.setattr(llm, "_SERVED_ASKED", {})
    llm.reset()
    # the model list is on file, as it is a day after the key went in
    p = config.BENCH_ROOT / "ai" / "openrouter_models.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    import time
    p.write_text(json.dumps({"at": time.time(), "models": []}), encoding="utf-8")
    yield client, vllm
    client.__exit__(None, None, None)


def test_the_builders_prompts_ship_in_the_image():
    """the traceback: /app/docs/prompts/phase-12i/knowledge-question-prompt.md.
    Every file the builder reads is on the list the service checks at start,
    and the image carries it"""
    for kind, path in builder.PROMPTS.items():
        rel = str(path.relative_to(REPO))
        assert rel in startup.REQUIRED_REPO_FILES, rel
        assert in_image(rel), rel
        assert not rel.startswith("docs/")
    assert not in_image("docs/prompts/phase-12i/knowledge-question-prompt.md")


def test_the_brief_s_copies_are_the_prompts_the_builder_uses():
    """docs/prompts/phase-12i/ holds them as the brief names them; the
    builder reads the image's copy — the same text"""
    docs = REPO / "docs" / "prompts" / "phase-12i"
    assert (docs / "knowledge-question-prompt.md").read_text(encoding="utf-8") == \
        builder.PROMPTS["knowledge"].read_text(encoding="utf-8")
    assert (docs / "everyday-question-prompt.md").read_text(encoding="utf-8") == \
        builder.PROMPTS["everyday"].read_text(encoding="utf-8")


def test_the_builder_loads_with_the_servers_data_shape(server):
    client, vllm = server
    r = client.get("/api/builder")
    assert r.status_code == 200, r.text
    page = r.json()
    assert page["drafts"] == []
    assert page["checker_blocked"] and "checking questions" in page["checker_blocked"]
    assert page["writer"]["label"] == "Local (gemma on this server)"
    assert page["writer"]["price_in"] == 0.0 and page["writer_blocked"] == ""
    assert sum(t["bank"] for t in page["topics"]) > 0
    assert sum(g["bank"] for g in page["groups"]) == 388
    for kind in ("knowledge", "everyday"):
        p = page["prompts"][kind]
        assert p["editable"] and p["locked"].startswith("## Output")
        assert p["path"].startswith("eval_tasks/")
    est = client.post("/api/builder/estimate", json={"kind": "knowledge", "count": 60}).json()
    assert est["line"] == "cost not known: no price for the checker here"


def test_the_local_model_is_named_by_its_weights(server):
    client, vllm = server
    import judge
    # a hot path — the judge's identity, its health probe — asks nothing
    judge.identity()
    client.get("/api/judge/health")
    assert vllm.asked == 0
    page = client.get("/api/ai").json()
    now = {j["job"]: j["now"] for j in page["jobs"]}
    assert now == {"judge": "Local (gemma on this server)",
                   "writer": "Local (gemma on this server)",
                   "data": "Local (gemma on this server)", "checker": "none"}
    assert page["local"]["name"] == "gemma"
    # the judge test's "the judge now" row: not "chat"
    assert judge.version()["label"] == "Local (gemma on this server)"
    # the server was asked once, for all of them
    assert vllm.asked == 1


def test_choosing_local_as_the_judge_names_it_without_asking_itself(server):
    """label("judge") and judge.identity() asked each other for ever"""
    client, _ = server
    r = client.post("/api/ai/jobs/judge", json={"model": "local", "by": "masein"})
    assert r.status_code == 200, r.text
    import judge
    assert ai_models.label("judge") == "Local (gemma on this server)"
    assert judge.identity()["name"] == "Local (gemma on this server)"
    assert client.get("/api/ai").status_code == 200


def test_no_local_job_asks_the_local_server_nothing(tmp_path, monkeypatch):
    client, _, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    try:
        vllm = FakeVLLM(llm._http)
        monkeypatch.setattr(llm, "_http", vllm)
        monkeypatch.setattr(llm, "_SERVED", {})
        monkeypatch.setattr(llm, "_SERVED_ASKED", {})
        assert client.get("/api/ai").json()["local"]["name"] == "the model"
        assert vllm.asked == 0
    finally:
        client.__exit__(None, None, None)
