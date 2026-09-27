"""12i.4: small fixes found live on 2026-09-27.

1. Build questions: past batches, and each published batch shows every
   question it wrote — the hidden half too, to its author — with its review,
   the checker's blind answer and where it went. Nowhere else shows a hidden
   question. The bank reader can be filtered by who wrote a question.
2. Answer length and running out, from the stored answers: the median length
   in tokens, thinking included, and how many ran out while thinking.
3b. A registered served model is a row before its first result; its runs
   record their seconds an answer."""

from __future__ import annotations

import json

import pytest

import answer_length as al
from service import db, runner, served
from test_builder_12i2 import (BY, _to_publish, create, rest, review_all,  # noqa: F401
                               svc)
from test_served_12f1 import BASE, SID, fake, register  # noqa: F401
from test_served_12f1 import svc as served_svc  # noqa: F401


# ---------------------------------------------------------------------------
# 1. past batches, and a batch's every question
# ---------------------------------------------------------------------------

def published_everyday(client) -> dict:
    d = create(client, kind="everyday", group="quick_maths", count=20)
    for n in (1, 2):
        client.post(f"/api/builder/{d['id']}/review",
                    json={"n": n, "verdict": "reject", "reason": "too easy", "by": BY})
    d = rest(client, review_all(client, client.get(f"/api/builder/{d['id']}").json()))
    _to_publish(client, d)
    return client.get(f"/api/builder/{d['id']}").json()


def test_a_published_batch_shows_every_question_and_where_it_went(svc):  # noqa: F811
    client, _, _ = svc
    d = published_everyday(client)
    went = [it["went"] for it in d["items"]]
    rejected = [it for it in d["items"] if it["verdict"] == "reject"]
    assert [it["went"] for it in rejected] == [{"to": "not published", "why": "rejected"}] * 2
    assert len(d["items"]) == 20
    assert sum(w["to"] in ("practice", "hidden") for w in went) == 18
    # both halves, as the bank splits every question
    assert {w["to"] for w in went} == {"practice", "hidden", "not published"}
    # each published one with its review and whether the checker's blind
    # answer matched; a rejected one was never sent to the checker
    out = [it for it in d["items"] if it["went"]["to"] != "not published"]
    assert all(it["checker_ok"] in (True, False) and it["answer"] for it in out)
    assert all(it["checker_ok"] is None for it in rejected)
    # the author sees the hidden half here, whole
    hidden = [it for it in d["items"] if it["went"]["to"] == "hidden"]
    assert hidden and all(it["q"]["prompt"] and it["q"]["reference"] for it in hidden)
    # the Build page lists it among the past batches, newest first
    page = client.get("/api/builder").json()
    [x] = [x for x in page["drafts"] if x["id"] == d["id"]]
    assert x["published"]["added"] == 18 and x["writer"] == "fake fake-exam"
    assert x["progress"]["line"].endswith(" · 18 published")
    assert x["created_at"] and x["by"] == BY


def test_the_hidden_half_is_still_not_reachable_outside_the_batch(svc):  # noqa: F811
    client, _, _ = svc
    d = create(client, kind="knowledge", topic="Economics", level="general public", count=12)
    d = rest(client, review_all(client, d))
    _to_publish(client, d)
    d = client.get(f"/api/builder/{d['id']}").json()
    hidden = [it["q"]["question"] for it in d["items"] if it["went"]["to"] == "hidden"]
    practice = [it["q"]["question"] for it in d["items"] if it["went"]["to"] == "practice"]
    assert hidden and practice
    # the bank reader, the public bank and the page's data: never a hidden one
    for text in (client.get("/api/exam/bank", params={"topic": "Economics",
                                                      "half": "diagnose"}).text,
                 client.get("/api/exam/bank").text, client.get("/api/results").text):
        assert not [h for h in hidden if h in text]
    bank = client.get("/api/exam/bank", params={"topic": "Economics", "half": "diagnose"}).json()
    mine = [q for q in bank["questions"] if q["batch"] == d["id"]]
    # the practice half is there, with who wrote it — the writer, not the approver
    assert sorted(q["prompt"] for q in mine) == sorted(practice)
    assert {q["written_by"] for q in mine} == {"fake fake-exam (fake/fake-exam)"}
    assert {q["approved_by"] for q in mine} == {BY}


# ---------------------------------------------------------------------------
# 2. answer length, from the stored answers
# ---------------------------------------------------------------------------

def plant(model_dir, rows, kind="everyday"):
    d = model_dir / ("everyday_0shot" if kind == "everyday" else "exam_economics_0shot") / "x"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"samples_{'everyday' if kind == 'everyday' else 'exam_economics'}_2026-09-27T10-00-00"
          f".000000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def exam_rows(texts, tokens=None):
    return [{"doc": {"qid": f"q{i}", "prompt": "p"}, "resps": [[t]],
             **({"tokens": tokens[i]} if tokens else {})} for i, t in enumerate(texts)]


def test_the_median_length_and_the_ran_out_count_come_from_the_stored_answers(tmp_path,
                                                                               monkeypatch):
    texts = ["<think>\n" + "w " * 40 + "\n</think>\n\nThe answer.",   # thought, answered
             "<think>\n" + "w " * 400,                                  # never stopped: ran out
             "<think>\nshort\n</think>\n\n",                            # stopped, said nothing
             "A plain answer with no thinking."]
    # 1. the server's own count wins
    m1 = tmp_path / "a"
    plant(m1, exam_rows(texts, tokens=[50, 4096, 3, 7]), kind="exam")
    s = al.stats(m1, "exam", ["org/a"])
    assert s == {"median": 28, "ran_out": 2, "n": 4, "how": "server", "tokenizer": ""}
    # 2. else the model's tokenizer, when its files are here
    m2 = tmp_path / "b"
    plant(m2, exam_rows(texts), kind="exam")
    monkeypatch.setattr(al, "tokenizer", lambda repos, arts=None: (lambda t: len(t.split()), repos[0]))
    s = al.stats(m2, "exam", ["org/b"])
    assert (s["how"], s["tokenizer"], s["ran_out"]) == ("tokenizer", "org/b", 2)
    assert s["median"] == 25                  # the words: 3, 6, 44 and 401
    # 3. else an estimate, and it says so
    m3 = tmp_path / "c"
    plant(m3, exam_rows(texts), kind="exam")
    monkeypatch.setattr(al, "tokenizer", lambda repos, arts=None: (None, ""))
    s = al.stats(m3, "exam", ["org/c"])
    assert s["how"] == "estimate" and "four characters" in al.how_words(s)
    # cached: the same files are not read again
    monkeypatch.setattr(al, "_records", lambda *a: pytest.fail("read again"))
    assert al.stats(m3, "exam", ["org/c"]) == s


def test_a_served_model_records_the_servers_count_and_the_row_carries_both(served_svc, fake):  # noqa: F811
    client, appmod, _ = served_svc
    assert register(client, fake, thinking="on").status_code == 200
    # most answer after a short thought; every seventh thinks until it runs out
    n = {"i": 0}

    def reply(body):
        n["i"] += 1
        return "" if n["i"] % 7 == 0 else "The answer."
    fake.reply = reply
    fake.reasoning = "Let me think it through."
    fake.tokens = lambda body: 4096 if n["i"] % 7 == 0 else 120
    sid = db.add(SID, "instruct", "everyday", "masein", "")
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    appmod._cache.update(key=None, payload=None, at=0.0)
    rows = {m["id"]: m for m in client.get("/api/results").json()["models"]}
    a = rows[SID]["answerLength"]["everyday"]
    total = len(fake.requests)
    assert a["n"] == total and a["median"] == 120 and a["how"] == "server"
    assert a["ran_out"] == total // 7
    assert a["words"] == "tokens as the server counted them"
    # its base, which it is compared with, has its numbers too
    assert rows[BASE]["answerLength"]["everyday"]["n"] > 0
    # a model that doesn't think, and isn't a base of one, has none
    assert rows["fx/one-option-70m"]["answerLength"] is None
    # and the run measured its seconds an answer, for Test a model's estimate
    assert served.get(SID)["speed"]["n"] == total
    assert client.get("/api/results").json()["served"][SID]["speed"]["secs_each"] > 0


# ---------------------------------------------------------------------------
# 3b. a served model with no result yet
# ---------------------------------------------------------------------------

def test_a_served_model_with_no_result_is_a_row_with_nothing_in_it(served_svc, fake):  # noqa: F811
    client, appmod, _ = served_svc
    assert register(client, fake).status_code == 200
    appmod._cache.update(key=None, payload=None, at=0.0)
    j = client.get("/api/results").json()
    row = next(m for m in j["models"] if m["id"] == SID)
    assert row["served"]["how"] and row["avg"] is None and row["nhave"] == 0
    assert not any(SID in c for c in j["cells"].values())
