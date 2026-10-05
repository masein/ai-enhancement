"""16c, part 1: a model's answers read from its Results. Each question carries
its own score — F1, ROUGE-L, the judge's mark, right or wrong — so a model's
answers can be read lowest first, or only what it got wrong; its Answers tab
lists every benchmark it has readable answers for; a second model sits under
the first. Nothing new becomes readable: the same function and the same
halves as the Benchmarks viewer. Fixtures and answers written in; nothing runs."""

from __future__ import annotations

import json

import pytest

import mobileaibench as mab
from conftest import make_service
from service import config, questions
from test_12n2 import sit_gpqa
from test_12o2 import _measure, every_row
from test_14_1_mab_text import sit

GOOD, SKEWED = "fx/good-750m", "fx/skewed-360m"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    yield client
    client.__exit__(None, None, None)


def page(client, task, **kw):
    r = client.get(f"/api/questions/{task}", params=kw)
    assert r.status_code == 200, r.text
    return r.json()


def mab_answers(mid: str, task: str, answer):
    d = config.OUT_DIR / mid.replace("/", "__")
    sit(d, task, answer=answer)
    mab.write(d, mab.mark(d))
    return d


# ---------------------------------------------------------------------------
# each question's own score, lowest first
# ---------------------------------------------------------------------------

def test_each_question_has_its_own_f1_and_the_lowest_come_first(svc):
    client = svc
    mab_answers(GOOD, mab.HOTPOT, lambda q: q["answer"] if ord(q["id"][-1]) % 3 else "no idea")
    p = page(client, mab.HOTPOT, models=GOOD, sort=f"score:{GOOD}", limit=200)
    scores = [r["results"][GOOD]["score"] for r in p["rows"]]
    assert scores == sorted(scores) and scores[0] < 0.5
    # the last of them, the highest
    r = page(client, mab.HOTPOT, models=GOOD, sort=f"score:{GOOD}", offset=p["total"] - 1,
             limit=1)["rows"][0]["results"][GOOD]
    assert r["score"] > 0.5 and r["score_words"] == f"F1 {r['score']:.2f}"
    assert r["verdict"].startswith("F1 ")
    # the benchmark's own order without the sort
    plain = page(client, mab.HOTPOT, models=GOOD, limit=200)
    assert [x["id"] for x in plain["rows"]] != [x["id"] for x in p["rows"]]
    assert plain["total"] == p["total"]


def test_a_summary_has_its_rouge_l_a_judged_turn_its_mark_and_a_choice_right_or_wrong(svc):
    client = svc
    mab_answers(GOOD, mab.CNNDM, lambda q: "a short summary of the article here")
    r = page(client, mab.CNNDM, models=GOOD, limit=1)["rows"][0]["results"][GOOD]
    assert r["ok"] is None and r["score_words"] == f"ROUGE-L {r['score']:.2f}"
    # MT-Bench: the judge's rating, out of 10
    d = mab_answers(GOOD, mab.MTB1, lambda q: "word " * 40)
    now = {"id": "judge-test-1", "version": mab._judge_now()["version"]}
    mab._record(d, {(it["id"], it["turn"]): 7.0 for it in mab.mt_answers(d)}, now)
    mab.write(d, mab.mark(d))
    r = next(x["results"][GOOD] for x in page(client, mab.MTBENCH, models=GOOD, limit=200)["rows"]
             if x["results"].get(GOOD))
    assert (r["score"], r["score_words"]) == (0.7, "rated 7 of 10")
    # multiple choice: right or wrong, the wrong first
    p = page(client, "mmlu", models=GOOD, sort=f"score:{GOOD}", limit=200)
    oks = [x["results"][GOOD]["ok"] for x in p["rows"]]
    assert oks == sorted(oks) and False in oks and True in oks
    first = p["rows"][0]["results"][GOOD]
    assert (first["score"], first["score_words"]) == (0.0, "wrong")


def test_only_what_it_got_wrong_or_right(svc):
    client = svc
    wrong = page(client, "mmlu", models=GOOD, f=f"wrong:{GOOD}", limit=200)
    right = page(client, "mmlu", models=GOOD, f=f"right:{GOOD}", limit=200)
    every = page(client, "mmlu", models=GOOD, limit=200)
    assert wrong["total"] + right["total"] == every["total"]
    assert all(r["results"][GOOD]["ok"] is False for r in wrong["rows"])
    assert all(r["results"][GOOD]["ok"] is True for r in right["rows"])


# ---------------------------------------------------------------------------
# what can be read, and what can't
# ---------------------------------------------------------------------------

def test_the_answers_tab_lists_what_can_be_read_and_never_gpqa(svc):
    client = svc
    sit_gpqa(config.OUT_DIR / GOOD.replace("/", "__"))
    mab_answers(GOOD, mab.CNNDM, lambda q: "a summary")
    got = client.get("/api/answers/benchmarks", params={"model": GOOD}).json()
    tasks = {x["task"] for x in got["tasks"]}
    assert {"mmlu", "hellaswag", mab.CNNDM} <= tasks
    assert not [t for t in tasks if t.startswith("gpqa")]
    assert not {"everyday", *questions.DM_TASKS} & tasks          # their own views
    assert got["why"]["gpqa"] == questions.NOT_LISTED
    # one with no run of it has none listed; GPQA's are never read
    other = {x["task"] for x in client.get("/api/answers/benchmarks",
                                           params={"model": SKEWED}).json()["tasks"]}
    assert mab.CNNDM not in other
    r = client.get("/api/questions/gpqa_diamond_cot_zeroshot", params={"models": GOOD})
    assert r.status_code == 403 and r.json()["detail"] == questions.NOT_LISTED


def test_the_held_back_are_counted_and_a_hidden_questions_text_is_in_no_response(svc):
    client = svc
    rows, p = every_row(client, "mmlu", models=GOOD, sort=f"score:{GOOD}")
    whole = questions.table("mmlu")["rows"]
    assert p["listed"] + p["other"] == len(whole) and p["other"] > 0
    hidden = [r["q"] for r in whole.values() if r["half"] == "report"]
    listed = json.dumps(rows) + json.dumps(client.get("/api/answers/benchmarks", params={"model": GOOD})
                                           .json())
    for f in (f"wrong:{GOOD}", f"right:{GOOD}"):
        listed += json.dumps(every_row(client, "mmlu", models=GOOD, f=f)[0])
    assert hidden and not [q for q in hidden if q in listed]


def test_two_models_on_one_question(svc):
    client = svc
    p = page(client, "mmlu", models=f"{GOOD},{SKEWED}", sort=f"score:{GOOD}", limit=20)
    assert p["shown"] == sorted([GOOD, SKEWED])
    for r in p["rows"]:
        assert r["results"][GOOD]["score"] is not None and r["results"][SKEWED]["score"] is not None


def test_a_gguf_row_has_the_question_the_options_and_its_pick_or_right_or_wrong(
        svc, tmp_path, monkeypatch):
    client = svc
    _measure(tmp_path, monkeypatch, ["mmlu"])
    p = page(client, "mmlu", models="gguf/k8", sort="score:gguf/k8", limit=200)
    assert p["gguf_models"] == ["gguf/k8"]
    g = [r["gguf"]["gguf/k8"] for r in p["rows"]]
    assert all(x["score_words"] in ("right", "wrong") for x in g)
    assert [x["score"] for x in g] == sorted(x["score"] for x in g)       # wrong first
    assert all(len(r["options"]) == 4 and r["answer_idx"] in range(4) for r in p["rows"])
    # and it can be filtered to what it got wrong
    w = page(client, "mmlu", models="gguf/k8", f="wrong:gguf/k8", limit=200)
    assert w["total"] and all(r["gguf"]["gguf/k8"]["ok"] is False for r in w["rows"])
    # the Answers tab lists it for the GGUF
    got = client.get("/api/answers/benchmarks", params={"model": "gguf/k8"}).json()
    assert {"task": "mmlu", "kind": "lm", "gguf": True} in got["tasks"]
