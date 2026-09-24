"""12b.3, the service's half: the checks say which are problems and which are
known limits, the two chat-template checks are one, and the service's answers
are compressed — the scores payload was 1.4 MB of JSON on the live tree and a
minute's wait over the tailnet."""

from __future__ import annotations

import gzip

import pytest

import report_lm_eval as report
from conftest import make_service

PROBLEMS = {"fewshot", "limit", "kind_unconfirmed", "taught_test", "duplicates", "near_duplicates",
            "judge_stub", "judge_canary", "judge_unfinished", "judge_other"}
LIMITS = {"chat_templates", "required_narrow", "tainted", "preliminary", "harness_builds",
          "judge_uncalibrated", "judge_kappa", "judge_local", "judge_single_provider"}


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, appmod, tree
    client.__exit__(None, None, None)


def test_every_check_is_a_problem_or_a_known_limit(payload):
    """Something happened (a duplicate row, a judge that drifted) or a standing
    condition of the setup (an uncalibrated judge, preliminary models): the
    status dot counts the first kind only."""
    cs = payload["checks"]
    assert cs and all(isinstance(c["limit"], bool) for c in cs)
    for c in cs:
        assert c["key"] in (LIMITS if c["limit"] else PROBLEMS), c["key"]
    # the test board has both: a model that learned the test and the stub
    # grader are problems; the rest are limits
    assert {c["key"] for c in cs if not c["limit"]} == {"taught_test", "judge_stub"}
    assert {c["key"] for c in cs if c["limit"]} == {"chat_templates", "tainted", "preliminary"}


def test_the_chat_template_checks_are_one(payload):
    """'on some models, not others' and 'different templates' said one thing
    twice (the live check: checks 1 and 2 of 7)"""
    keys = [c["key"] for c in payload["checks"]]
    assert "chat_mixed" not in keys and "templates" not in keys
    one = [c for c in payload["checks"] if c["key"] == "chat_templates"]
    assert len(one) == 1 and one[0]["limit"] is True
    assert one[0]["short"] == "Chat templates differ between models"
    assert "some models but not others" in one[0]["text"]


def test_the_limit_list_in_the_code_matches_the_checks_it_can_raise():
    """every key the payload can raise is classified here: a new check has to
    be called a problem or a limit, not neither"""
    src = report.__file__
    text = open(src, encoding="utf-8").read()
    import re
    raised = set(re.findall(r"warn\('([a-z_]+)'", text)) | {"chat_templates"}
    assert raised == PROBLEMS | LIMITS, raised ^ (PROBLEMS | LIMITS)


def test_the_answers_are_compressed(svc):
    client, _, _ = svc
    r = client.get("/api/results", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "gzip"
    assert r.headers.get("x-evalboard-build") is not None        # the stamp still rides along
    raw = r.content                                              # httpx has already unzipped it
    assert len(gzip.compress(raw)) * 3 < len(raw)                 # JSON: several times smaller
    # and the page itself
    page = client.get("/", headers={"Accept-Encoding": "gzip"})
    assert page.headers.get("content-encoding") == "gzip"
    # a client that does not ask gets it plain
    plain = client.get("/api/results", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers
