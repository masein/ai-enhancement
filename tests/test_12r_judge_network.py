"""12r (the brief's): the judge's model (gemma-vllm) belongs to another compose
project, on that project's network. The board reached it by name only through
a connection made by hand, which the chat project's recreate dropped — and 30
Summarise answers waited on "Temporary failure in name resolution".

Now docker-compose.judge.yml joins the board to that network as well as its
own (opt in from .env, so every compose command uses it); without it the
board starts on its own network anywhere, and the judge's health says what to
run when the judge's hostname doesn't resolve — and that a 401 wants a key.
Nothing is reached: the probe's answer is made here."""

from __future__ import annotations

import re
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import make_service
from service import config

REPO = Path(__file__).resolve().parents[1]
URL = "http://gemma-vllm:8000/v1"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "local")
    monkeypatch.setattr(config, "JUDGE_MODEL", "chat")
    monkeypatch.setattr(config, "LOCAL_BASE_URL", URL)
    appmod._JUDGE_HEALTH.update(at=0.0, value=None)
    yield client, appmod
    appmod._JUDGE_HEALTH.update(at=0.0, value=None)
    client.__exit__(None, None, None)


def answers(monkeypatch, appmod, error):
    def urlopen(req, timeout=None):
        raise error
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    appmod._JUDGE_HEALTH.update(at=0.0, value=None)


def test_a_name_that_doesnt_resolve_says_what_to_run(svc, monkeypatch):
    client, appmod = svc
    answers(monkeypatch, appmod, urllib.error.URLError(
        socket.gaierror(-3, "Temporary failure in name resolution")))
    h = client.get("/api/judge/health").json()
    assert h["ok"] is False and h["url"] == URL
    assert h["why"] == (
        "the grading model's hostname gemma-vllm doesn't resolve from the board's container "
        "(Temporary failure in name resolution): the board isn't on its Docker network, "
        "teraformer-chat_default. To join it for good, put "
        "COMPOSE_FILE=docker-compose.yml:docker-compose.judge.yml in .env and run `sudo docker "
        "compose up -d`. Until then: `sudo docker network connect teraformer-chat_default "
        "$(sudo docker compose ps -q bench)`. Neither touches the judge's own container")
    # the network's name is the one the server says
    monkeypatch.setattr(config, "JUDGE_NETWORK", "other_default")
    answers(monkeypatch, appmod, urllib.error.URLError(socket.gaierror(-2, "Name or service "
                                                                           "not known")))
    assert "connect other_default $(" in client.get("/api/judge/health").json()["why"]
    # a judged run is refused in the same words, and nothing is queued
    r = client.post("/api/submissions", json={"hf_id": "fx/good-750m", "suite": "judged",
                                              "tasks": ["exam_law"]})
    assert r.status_code == 503 and "doesn't resolve from the board's container" in \
        r.json()["detail"]


def test_a_401_says_it_wants_a_key(svc, monkeypatch):
    client, appmod = svc
    answers(monkeypatch, appmod, urllib.error.HTTPError(URL + "/models", 401, "Unauthorized",
                                                        {}, None))
    why = client.get("/api/judge/health").json()["why"]
    assert why == (f"the grading model at {URL} answered 401: it wants an API key. Set "
                   "JUDGE_API_KEY in .env to the key its server was started with, then `sudo "
                   "docker compose up -d`")
    # the key is never in what the health says
    monkeypatch.setattr(config, "JUDGE_API_KEY", "sk-not-a-real-key")
    answers(monkeypatch, appmod, urllib.error.HTTPError(URL + "/models", 401, "Unauthorized",
                                                        {}, None))
    assert "sk-not-a-real-key" not in str(client.get("/api/judge/health").json())


def test_any_other_failure_keeps_its_words(svc, monkeypatch):
    client, appmod = svc
    answers(monkeypatch, appmod, urllib.error.URLError(ConnectionRefusedError(111, "Connection "
                                                                                  "refused")))
    why = client.get("/api/judge/health").json()["why"]
    assert why.startswith(f"the grading model isn't answering at {URL} (") and "refused" in why
    answers(monkeypatch, appmod, urllib.error.HTTPError(URL + "/models", 500, "boom", {}, None))
    assert client.get("/api/judge/health").json()["why"] == \
        f"the grading model at {URL} answered 500"


def test_the_judge_file_joins_the_chat_projects_network_and_the_boards_own():
    text = (REPO / "docker-compose.judge.yml").read_text(encoding="utf-8")
    body = "\n".join(x for x in text.splitlines() if not x.lstrip().startswith("#"))
    assert re.search(r"services:\n  bench:\n    networks:\n      - default\n      - judge\n", body)
    assert re.search(r"networks:\n  judge:\n    name: \$\{JUDGE_NETWORK:-teraformer-chat_default\}"
                     r"\n    external: true$", body)
    # it changes nothing else of the service, and never names the judge's container
    assert "gemma-vllm" not in body and "image:" not in body and "restart" not in body


def test_without_it_the_board_starts_on_its_own_network_anywhere():
    base = (REPO / "docker-compose.yml").read_text(encoding="utf-8")
    body = "\n".join(x for x in base.splitlines() if not x.lstrip().startswith("#"))
    assert "external:" not in body and not re.search(r"^networks:", body, re.M)
    assert "JUDGE_NETWORK: ${JUDGE_NETWORK:-teraformer-chat_default}" in body
    assert config.JUDGE_NETWORK == "teraformer-chat_default"


def test_how_to_opt_in_is_written_down():
    line = "COMPOSE_FILE=docker-compose.yml:docker-compose.judge.yml"
    assert line in (REPO / "docker-compose.judge.yml").read_text(encoding="utf-8")
    assert "# " + line in (REPO / ".env.example").read_text(encoding="utf-8")
    handoff = (REPO / "HANDOFF.md").read_text(encoding="utf-8")
    assert line in handoff and "Never restart or change gemma-vllm" in handoff
