"""12z D, the service's half: which AI jobs run on the local server (D1, so
Propose can wait while it is down), a served setup's launch beside what its
server reports (D2), and the Playground's suggestions drawn from the public
half alone — never an id the hidden store holds (D3). Fixtures only: nothing
runs a model, nothing calls OpenRouter. Question ids only, never their text."""

from __future__ import annotations

import json

import pytest

import everyday as ev
from fake_openai import FakeServer
from service import ai_models, config, served
from test_12q_devicemark_runs import ME
from test_playground_12d1 import svc  # noqa: F401 — svc is the fixture

# ---------------------------------------------------------------------------
# D1
# ---------------------------------------------------------------------------


def test_the_llm_status_says_which_jobs_run_on_the_local_server(svc, monkeypatch):  # noqa: F811
    client = svc
    got = client.get("/api/llm").json()["ai_local"]
    assert set(got) == set(ai_models.JOBS) and all(isinstance(v, bool) for v in got.values())
    # a job the AI models page put on OpenRouter is not local; one put on Local is
    monkeypatch.setattr(ai_models, "choice", lambda job: {"kind": "openrouter", "id": "x/y"}
                        if job == "data" else {"kind": ai_models.LOCAL})
    got = client.get("/api/llm").json()["ai_local"]
    assert got["data"] is False and got["judge"] is True


# ---------------------------------------------------------------------------
# D2
# ---------------------------------------------------------------------------

def _rec(**kw) -> dict:
    return {"name": "x", "base_url": "http://127.0.0.1:8090/v1", "how": "", "flags": "", "env": "",
            "pin": {"file": "a.gguf", "ctx": 16384, "build": "b6500"}, "speculative": None,
            "thinking": "off", "based_on": "", **kw}


@pytest.mark.parametrize("rec, said", [
    # the walk's: the text says --cpu-moe, the server runs --n-cpu-moe 21
    (_rec(how="llama-server -ngl 99 --cpu-moe", flags="-ngl 99 --n-cpu-moe 21"),
     ["“How it’s served” says --cpu-moe; the launch flags registered don’t"]),
    (_rec(how="llama-server --ctx-size 8192", flags="--n-cpu-moe 21"),
     ["“How it’s served” says --ctx-size; the launch flags registered don’t",
      "“How it’s served” says a context of 8,192; its server reports 16,384"]),
    (_rec(flags="--spec-type draft-mtp -c 32768", speculative=False),
     ["The launch flags say a context of 32,768; its server reports 16,384",
      "The launch registered has MTP, but its server’s slots draft nothing"]),
    (_rec(flags="--n-cpu-moe 21", speculative=True),
     ["Its server’s slots draft tokens, but the launch registered has no --spec-type"]),
    # agreeing: nothing to say; words in the text are never flags
    (_rec(how="llama.cpp fork, routing local (no lookahead), MTP-GGUF", flags="--spec-type draft-mtp",
          env="LLAMA_MOE_ROUTE_MODE=local", speculative=True), []),
    # no flags registered: the free text can't disagree with nothing
    (_rec(how="llama-server --cpu-moe"), []),
])
def test_a_served_setups_launch_is_checked_against_what_its_server_reports(rec, said):
    got = served.launch_check(rec)
    assert got["mismatch"] == said
    assert (got["flags"], got["env"], got["ctx"], got["build"]) == (rec["flags"], rec["env"],
                                                                    16384, "b6500")


def test_a_model_from_openrouter_has_no_launch_to_check():
    assert served.launch_check(_rec(via=served.OPENROUTER)) is None
    assert served.launch_check(None) is None


def test_the_launch_check_reaches_the_page_and_follows_a_new_launch(svc):  # noqa: F811
    fake = FakeServer()
    try:
        sid = served.register({"name": "D2 setup", "base_url": fake.base, "thinking": "off",
                               "how": "llama-server -ngl 99 --cpu-moe",
                               "flags": "-ngl 99 --n-cpu-moe 21"}, ME)["id"]
        meta = json.loads((config.OUT_DIR / sid.replace("/", "__") / "model_meta.json").read_text())
        assert meta["served"]["launch"]["mismatch"] == [
            "“How it’s served” says --cpu-moe; the launch flags registered don’t"]
        # the launch set again — the free text agrees now, and the page says so
        served.set_launch(sid, "-ngl 99 --cpu-moe", "")
        meta = json.loads((config.OUT_DIR / sid.replace("/", "__") / "model_meta.json").read_text())
        assert meta["served"]["launch"]["mismatch"] == []
        assert meta["served"]["launch"]["flags"] == "-ngl 99 --cpu-moe"
    finally:
        fake.close()


# ---------------------------------------------------------------------------
# D3
# ---------------------------------------------------------------------------

def _store_ids() -> set[str]:
    """every id the hidden store holds: Everyday's hidden set (on the volume)
    and the exam's report half — by its committed manifest where there is
    one, and the test banks' report rows"""
    import exam_build as eb
    from service import hidden_store
    ids = {q["id"] for q in ev._raw_rows(ev.hidden_path())}
    ids |= {r["qid"] for rows in eb.load_bank(config.EXAM_DIR).values() for r in rows
            if eb.half_of(r["qid"]) == "report"}
    man = hidden_store.exam_manifest() or {}
    for qids in (man.get("topics") or {}).values():
        ids |= set(qids if isinstance(qids, list) else [])
    return ids


def test_playground_suggestions_never_hold_a_hidden_store_id(svc):  # noqa: F811
    client = svc
    store = _store_ids()
    assert store, "the fixture's hidden store holds questions"
    got = client.get("/api/playground/practice").json()
    offered = [x["id"] for kind in ("everyday", "knowledge") for x in got[kind]]
    assert offered and not set(offered) & store
    # and nowhere in what the Playground's endpoints say, not even as text
    body = json.dumps(got) + json.dumps(client.get("/api/playground").json())
    assert not [i for i in store if f'"{i}"' in body]
    # what they are drawn from: the public half alone
    import exam_build as eb
    assert {x["id"] for x in got["everyday"]} == {q["id"] for q in ev.load_bank()
                                                  if ev.half(q) == ev.PRACTICE}
    assert all(eb.half_of(x["id"]) == "diagnose" for x in got["knowledge"])


# ---------------------------------------------------------------------------
# the page's script: no top-level name declared twice
# ---------------------------------------------------------------------------

def test_the_page_script_declares_no_top_level_name_twice():
    """12z D1 first gave Propose's reason the name of the topics table's
    `proposeWhy(g)`: a later function declaration silently replaces an earlier
    one, and the table's rows lost their reasons. Caught by a browser test far
    from the change; this catches it here"""
    import re
    from collections import Counter

    import report_lm_eval as rl
    src = open(rl.__file__, encoding="utf-8").read()
    names = re.findall(r"^(?:async )?function ([A-Za-z_$][\w$]*)\(|^(?:const|let) ([A-Za-z_$][\w$]*) *=",
                       src, re.M)
    twice = [k for k, n in Counter(a or b for a, b in names).items() if n > 1]
    assert twice == []
