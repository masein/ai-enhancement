"""16 Nov (masein): the raw-run exports' scrub changed public text. With a 100.x address in
SCRUB_HOSTS, "100" became a host name, and every standalone "100" in a release's prompts,
answers, logs and READMEs became [host] (4,591 of them). Its generic rules also rewrote a
URL's host and an example email in IFEval prompts and model answers. Now:
- SCRUB_HOSTS refuses an address or a number, and a number is never a host's name;
- in a dataset's prompt and a model's answer and thinking, only exact matches of our own host
  names, addresses and accounts (and our secrets by value) are replaced. The generic rules
  stay for our own fields: paths, setup, log lines;
- setup.json keeps the row's thinking marks and the rented boxes' part of the run.

The board's runner with 15.1's stand-in lm_eval; nothing runs. No real address is written
here: the 100.x one is built from its parts, as test_15_4's is."""

from __future__ import annotations

import json

import pytest

import export_devicemark_raw as ex
import export_frontier_raw as efr  # noqa: F401 — the Frontier export uses ex.scrub_item
import export_safe as es
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the board's fixture
from test_15_2_import import board, row_of  # noqa: F401 — board is the fixture
from test_15_4_raw_export import QWEN, no_gated_sets, row  # noqa: F401 — fixtures

ADDR = ".".join(["100", "64", "12", "34"])
HOSTS = "evalsrv01,evalsrv01.example-tailnet.net"
# an IFEval-style prompt and answer, with what the generic rules took: a URL's host, an
# example email, a private-looking address, and "100" — all of it must come out as it is
PROMPT = ("Write a resume for a junior engineer in less than 100 words. Include an email "
          "address and a link to https://en.wikipedia.org/wiki/Software_engineering.")
ANSWER = ("**Alex Doe** · alex.doe@example.com · https://www.linkedin.com/in/alexdoe\n"
          "Set up 100 lab routers (192.168.1.1, 10.0.0.1) and read "
          "https://en.wikipedia.org/wiki/Router_(computing). P.S. 100% ready.")
THINKING = "They want under 100 words; mail like john.smith@example.org is fine."


@pytest.fixture
def names(monkeypatch):
    monkeypatch.setenv("SCRUB_HOSTS", HOSTS)
    monkeypatch.setenv("SCRUB_ACCOUNTS", "teamacct")
    es._names.clear()


def test_an_ifeval_answer_with_an_example_email_and_url_comes_out_unchanged(row, names,  # noqa: F811
                                                                             tmp_path, monkeypatch):
    original = ex.item_record
    seen = []

    def record(s, item, reading):
        out = original(s, item, reading)
        if out["test"] == "ifeval" and not seen:
            seen.append(out["key"])
            out.update(prompt=PROMPT, answer=ANSWER, thinking=THINKING)
        elif out["test"] == "ifeval" and len(seen) == 1:
            seen.append(out["key"])
            out.update(answer="ssh evalsrv01.example-tailnet.net, then EVALSRV01; teamacct/x is ours")
        return out
    monkeypatch.setattr(ex, "item_record", record)
    dest = ex.export_row(row_of(QWEN), tmp_path / "out", say=lambda *_: None)
    items = {i["key"]: i for i in map(json.loads, (dest / "items.jsonl").read_text().splitlines())}
    first = items[seen[0]]
    assert (first["prompt"], first["answer"], first["thinking"]) == (PROMPT, ANSWER, THINKING)
    # our own names in an answer: replaced, exactly
    assert items[seen[1]]["answer"] == "ssh [host], then [host]; [account]/x is ours"
    # and no "100" anywhere became a host
    assert "[host] words" not in (dest / "items.jsonl").read_text()
    assert "[host] across" not in (dest / "README.md").read_text()


def test_scrub_hosts_refuses_an_address_or_a_number(monkeypatch):
    monkeypatch.setenv("SCRUB_ACCOUNTS", "teamacct")
    for bad in (ADDR, "12345", "fd7a:115c::1"):
        monkeypatch.setenv("SCRUB_HOSTS", f"evalsrv01,{bad}")
        es._names.clear()
        got = es.scrub_names()
        assert any("is an address or a number" in p and bad in p for p in got["problems"]), got
        assert bad not in got["hosts"] and "100" not in got["hosts"]
        assert es.refused_public().startswith("public/ refused:")
        assert ex.scrub("IFEval 100 of 300; less than 100 words") == "IFEval 100 of 300; less than 100 words"
    # a dotted name whose first label is a number: the name is a host, the number never
    monkeypatch.setenv("SCRUB_HOSTS", "evalsrv01,100.example-tailnet.net")
    es._names.clear()
    got = es.scrub_names()
    assert "100.example-tailnet.net" in got["hosts"] and "100" not in got["hosts"]
    assert ex.scrub("at 100.example-tailnet.net, 100 of 300") == "at [host], 100 of 300"


def test_our_own_fields_keep_the_generic_rules(names):
    line = "[service] see https://en.wikipedia.org/wiki/X and john.doe@example.com from 192.168.1.1"
    assert ex.scrub(line) == "[service] see https://[host]/wiki/X and john.[user]@[host] from [address]"
    item = {"test": "ifeval", "key": "7", "answer": "mail john.doe@example.com",
            "parsed_how": "via http://gemma-vllm:8000/v1"}
    got = ex.scrub_item(item)
    assert got["answer"] == "mail john.doe@example.com"           # the model's: as it is
    assert got["parsed_how"] == "via http://[host]:8000/v1"        # ours: the generic rules


def test_a_frontier_answer_keeps_its_example_email_and_url(names):
    # the Frontier export writes an answer of a shown benchmark through ex.scrub_item, with
    # "answer" as the model's text and every other field ours
    x = {"task": "ifeval_google", "id": "1001", "epoch": 0, "ok": True, "answer": ANSWER,
         "grade": {"ok": True, "by": "the checker at http://gemma-vllm:8000"}}
    got = ex.scrub_item(x, names=ex.own_names(), fields=("answer",))
    assert got["answer"] == ANSWER
    assert got["grade"]["by"] == "the checker at http://[host]:8000"


def test_setup_keeps_its_thinking_marks_and_its_remote_part():
    setup = {"model": "Qwen/Qwen3.5-4B", "reading": {"on": True, "marks": ["<think>", "</think>"],
                                                     "opens": True},
             "remote": {"dm_ifeval": {"gpu": "NVIDIA GeForce RTX 5090",
                                      "bundle": "devicemark-Qwen__Qwen3.5-4B-thinking-on-shard-2-of-2.tar.gz",
                                      "sha256": "0fa354edfe2e4de2", "shards": 2}}}
    assert ex.setup_out(setup) == setup
    gemma = {"reading": {"on": False, "marks": ["<|channel>", "<channel|>"], "opens": False}}
    assert ex.setup_out(gemma) == gemma
    # a mark is a tag and nothing more; a box's address in the remote part never goes out
    got = ex.setup_out({"reading": {"marks": ["<think>", "<script src=x>", "</think>"]},
                        "remote": {"dm_ifeval": {"gpu": "RTX 5090", "host": "203.0.113.7",
                                                 "shards": 2}}})
    assert got == {"reading": {"marks": ["<think>", "</think>"]},
                   "remote": {"dm_ifeval": {"gpu": "RTX 5090", "shards": 2}}}
