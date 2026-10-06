"""17b, part 6 of the review of 1388058..62ec4d9 — grading, before anyone
presses Start. Each test fails on 62ec4d9:
- 19: two Start presses at once send each answer once;
- 20: a batch is recorded before a request goes; finish() reads one answer
  at a time;
- 21: an empty reply, or one cut at its cap, is no grade and is asked again;
  each grader's reasoning is set and its cap sized for it, and the dry run
  counts the same;
- 22, 23: SimpleQA's and HLE's readers — and HLE and OTIS graded end to end;
- 24: a grader's prompt filled in one pass;
- 25: a grade kept for the answer it graded;
- 26: a score from two graders says so, and isn't final;
- 27: the card says why a batch failed, and why one waits (Stop, the
  month's limit), and Carry on sends the rest;
- 28: choosing a grader counts its paid probe, and the card says so;
- 29: the Frontier view ranks by the setting, not a model's own numbers.
Invented questions and a stand-in for OpenRouter only: nothing is sent
anywhere."""

from __future__ import annotations

import json
import threading
import time

import pytest

import frontier as fb
import frontier_graders as fg
import report_lm_eval as report
from service import ai_models, config, db, llm, llm_poller
from service import frontier as sf
from service import frontier_grade as fgr
from test_17_grading import ROW, drain, rec, results, svc  # noqa: F401 — svc is the fixture

GPT, GEMINI, O3 = "openai/gpt-4.1", "google/gemini-2.5-flash", "openai/o3-mini"


def row_dir():
    return config.OUT_DIR / ROW


def stub(monkeypatch, answer) -> list:
    """the grader's replies: answer(model, request row) -> (text, finish, error)"""
    asked: list = []
    lock = threading.Lock()

    def complete(self, row):
        with lock:
            asked.append((self.model, row["custom_id"]))
        text, finish, error = answer(self.model, row)
        return {"custom_id": row["custom_id"], "text": text, "error": error, "attempts": 1,
                "finish_reason": finish}
    monkeypatch.setattr(fgr.GraderChat, "_complete", complete)
    return asked


def plain(model, row):
    """SimpleQA's grader says CORRECT, MATH's check not equivalent"""
    return ("No" if "algebra/" in row["custom_id"] else "A"), "stop", ""


def more_data(monkeypatch, extra: dict) -> None:
    before = fb._fetch
    monkeypatch.setattr(fb, "_fetch", lambda task: extra[task] if task in extra else before(task))


def write_runs(task: str, answers: list[tuple[str, str]]) -> None:
    """every run of each question, the same answer"""
    d = sf.task_dir(row_dir(), task)
    d.mkdir(parents=True, exist_ok=True)
    (d / sf.ANSWERS).write_text("".join(json.dumps(
        {"id": q, "epoch": e, "answer": a, "finish": "stop"}) + "\n"
        for q, a in answers for e in range(fb.BENCH[task]["epochs"])))


def grades(task: str) -> dict:
    return sf.read_grades(sf.task_dir(row_dir(), task))


def page(client) -> dict:
    return client.get("/api/frontier/grading").json()


# ---------------------------------------------------------------------------
# 19, 20: Start once; recorded, then sent; one answer at a time
# ---------------------------------------------------------------------------

def test_19_two_starts_at_once_send_each_answer_once(svc, monkeypatch):  # noqa: F811
    """a second press 100 ms after the first paid twice: both listed what
    waited while the first was still pinning its graders"""
    asked = stub(monkeypatch, plain)
    pin = ai_models.pin

    def slow(mid, job="pin"):
        time.sleep(0.3)                     # pinning asks OpenRouter: the window
        return pin(mid, job)
    monkeypatch.setattr(ai_models, "pin", slow)
    got, errors = [], []

    def press():
        try:
            got.append(fgr.start("masein"))
        except Exception as e:              # noqa: BLE001 — said below
            errors.append(e)
    presses = [threading.Thread(target=press) for _ in range(2)]
    for t in presses:
        t.start()
        time.sleep(0.1)
    for t in presses:
        t.join(30)
    assert not errors
    drain()
    assert len(asked) == 7 and len({c for _, c in asked}) == 7
    assert sorted(len(x["sent"]) for x in got) == [0, 2]
    assert sum(1 for b in db.batches_list() if b["kind"] == fgr.KIND) == 2


def test_20_a_batch_is_recorded_before_a_request_goes(svc, monkeypatch):  # noqa: F811
    asked = stub(monkeypatch, plain)
    real = db.batch_add

    def locked(*a, **k):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(db, "batch_add", locked)
    with pytest.raises(RuntimeError):
        fgr.start("masein")
    time.sleep(0.5)
    assert asked == [] and not fgr.pending()            # nothing paid for, nothing lost
    monkeypatch.setattr(db, "batch_add", real)
    fgr.start("masein")
    drain()
    assert len(asked) == 7 and len({c for _, c in asked}) == 7


def test_20_finish_reads_one_answer_at_a_time(svc, monkeypatch):  # noqa: F811
    """a key that isn't an integer failed the whole OTIS batch, its other
    replies with it"""
    more_data(monkeypatch, {"otis_aime_epoch": [
        {"id": "1", "question": "Find x.", "answer": "7"},
        {"id": "2", "question": "Find y.", "answer": "seven"}]})
    write_runs("otis_aime_epoch", [("1", "so x is 5\nANSWER: 5"), ("2", "ANSWER: 5")])
    # OTIS's extractor reads 7 from every reply; MATH's check as ever
    stub(monkeypatch, lambda m, r: plain(m, r) if "algebra/" in r["custom_id"]
         else ("7", "stop", ""))
    fgr.start("masein")
    drain()
    g = grades("otis_aime_epoch")
    runs = fb.BENCH["otis_aime_epoch"]["epochs"]
    assert all(g["items"][f"1#{e}"]["ok"] is True for e in range(runs))
    assert all("couldn't be read" in g["refused"][f"2#{e}"]["words"] for e in range(runs))
    assert not [b for b in db.batches_list() if b["kind"] == fgr.KIND and b["status"] != "done"]


# ---------------------------------------------------------------------------
# 21: a grade is a grade; each grader's reasoning, set
# ---------------------------------------------------------------------------

def test_21_an_empty_or_cut_reply_is_not_a_grade(svc, monkeypatch):  # noqa: F811
    """an empty reply read as NOT_ATTEMPTED, final, nothing waiting"""
    broken = {"on": True}

    def answer(model, row):
        if broken["on"] and row["custom_id"].endswith(":1#0"):
            return "", "stop", ""
        if broken["on"] and row["custom_id"].endswith(":2#0"):
            return "", "length", ""
        return plain(model, row)
    stub(monkeypatch, answer)
    fgr.start("masein")
    drain()
    assert results("simpleqa_epoch") is None                    # two still wait
    ref = grades("simpleqa_epoch")["refused"]
    assert ref["1#0"]["words"] == "the grader's reply isn't a grade: it was empty"
    assert ref["2#0"]["words"] == "the reply was cut at its cap of 16 tokens"
    p = page(svc)
    assert [(x["slot"], x["n"]) for x in p["refused"]] == [("simpleqa", 2)]
    assert {r["slot"]: r["answers"] for r in p["estimate"]["rows"]} == {"simpleqa": 2}
    broken["on"] = False
    fgr.start("masein")
    drain()
    assert results("simpleqa_epoch")["results"]["simpleqa_epoch"]["acc,none"] == \
        pytest.approx(5 / 6)
    assert not page(svc)["refused"]


def test_21_each_graders_reasoning_is_set_and_its_cap_sized_for_it(svc, monkeypatch):  # noqa: F811
    """MATH and OTIS suggested gemini-2.5-flash with 16 and 32 tokens and no
    reasoning setting: thinking first, every reply came back empty, paid for"""
    # OpenRouter lists gemini-2.5-flash as a model that reasons
    p = ai_models._cache_path()
    got = json.loads(p.read_text())
    for m in got["models"]:
        m["reasons"] = m["id"] in (GEMINI, O3)
    p.write_text(json.dumps(got))
    est = page(svc)["estimate"]
    math = est["graders"]["math"]
    assert math["cap"] == 16 + fg.REASONING_ROOM and math["reasoning"] == {"enabled": False}
    assert math["tokens_out"] == 2 * 2 and math["tokens_out_max"] == 2 * math["cap"]
    # 17d: at most, each answer asked its three times too
    assert math["usd_max"] == pytest.approx(sf.GRADE_TRIES * (math["tokens_in"] * 0.3
                                                              + 2 * math["cap"] * 2.5)
                                            / 1e6, abs=1e-4)
    assert est["graders"]["simpleqa"]["cap"] == 16 and est["usd_max"] >= est["usd"]
    assert fg.ask("hle", True) == {"max_tokens": 4096, "reasoning": {"effort": "medium"},
                                   "out_tokens": 900, "schema": None}
    sent = []

    def http(method, url, headers=None, payload=None, timeout=None):
        body = json.loads(payload)
        sent.append(body)
        text = "No" if body["model"] == GEMINI else "A"
        return 200, json.dumps({"choices": [{"message": {"content": text},
                                             "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 100, "completion_tokens": 1,
                                          "cost": 0.0002}, "provider": "Prov"}).encode()
    monkeypatch.setattr(llm, "_http", http)
    fgr.start("masein")
    drain()
    # what was sent is what the dry run counted
    by = {}
    for b in sent:
        by.setdefault(b["model"], set()).add((json.dumps(b["reasoning"]), b["max_tokens"]))
    assert by == {GPT: {('{"enabled": false}', 16)},
                  GEMINI: {('{"enabled": false}', 16 + fg.REASONING_ROOM)}}


# ---------------------------------------------------------------------------
# 22, 23: the readers — and HLE and OTIS graded end to end
# ---------------------------------------------------------------------------

def test_22_simpleqa_reads_a_bare_letter_else_whole_words_else_asks_again():
    for text, grade in {"A": "A", " B\n": "B", "C.": "C", "CORRECT": "A", "incorrect": "B",
                        "NOT_ATTEMPTED": "C", "Not attempted": "C", "B: INCORRECT": "B"}.items():
        assert fg.read("simpleqa", text, {})["grade"] == grade, text
    # Google's reading took "As an AI…" and "NOT_ATTEMPTED" as A, CORRECT as
    # C, and "incorrect" as correct
    for text in ("As an AI I cannot grade this", "The answer is A", "Grade it yourself", ""):
        assert fg.read("simpleqa", text, {})["ok"] is None, text


def test_23_hle_is_graded_end_to_end_by_the_last_correct_and_an_unread_reply_asked_again(
        svc, monkeypatch):  # noqa: F811
    more_data(monkeypatch, {"hle_text_cais": [
        {"id": f"h{k}", "question": f"Hard question {k}?", "answer": f"K{k}",
         "answer_type": "exactMatch", "subject": "x"} for k in range(4)]})
    write_runs("hle_text_cais", [(f"h{k}", f"I believe K{k}.") for k in range(4)])
    said = {"h0": "extracted_final_answer: K0\nreasoning: had it said 'correct: no' it would "
                  "differ\ncorrect: yes\nconfidence: 90",
            "h1": '{"extracted_final_answer": "X", "reasoning": "differs", "correct": "no", '
                  '"confidence": 40}',
            # 17d: the last "correct:" line, its value alone
            "h2": "correct: no\non reflection, it matches\ncorrect: yes",
            "h3": "I can't tell"}

    def answer(model, row):
        if model == O3:
            return said[row["custom_id"].split(":")[1].split("#")[0]], "stop", ""
        return plain(model, row)
    stub(monkeypatch, answer)
    fgr.start("masein")
    drain()
    g = grades("hle_text_cais")
    assert g["items"]["h0#0"]["ok"] is True and g["items"]["h0#0"]["confidence"] == 90
    assert g["items"]["h1#0"]["ok"] is False and "read as X" in g["items"]["h1#0"]["words"]
    assert g["items"]["h2#0"]["ok"] is True
    # 17c: no reply on the card for HLE (the judge may quote its gated question)
    assert "h3#0" not in g["items"]
    assert g["refused"]["h3#0"]["words"] == "the grader's reply isn't a grade"
    assert results("hle_text_cais") is None                     # one waits
    said["h3"] = "correct: yes"
    fgr.start("masein")
    drain()
    r = results("hle_text_cais")
    assert r["results"]["hle_text_cais"]["acc,none"] == pytest.approx(3 / 4)
    assert report.frontier_how(r["frontier"]).startswith(
        "graded by openai/o3-mini-2025-01-31 with CAIS's judge prompt (")


def test_otis_is_graded_end_to_end(svc, monkeypatch):  # noqa: F811
    more_data(monkeypatch, {"otis_aime_epoch": [
        {"id": "1", "question": "Find a.", "answer": "7"},
        {"id": "2", "question": "Find b.", "answer": "12"},
        {"id": "3", "question": "Find c.", "answer": "300"}]})
    # right by code; unread by code (the extractor reads it); wrong by code
    write_runs("otis_aime_epoch", [("1", "ANSWER: 7"), ("2", "so b is twelve, that is 12"),
                                   ("3", "ANSWER: 299")])
    said = {"2": "12", "3": "NONE"}
    asked = stub(monkeypatch, lambda m, r: (said[r["custom_id"].split(":")[1].split("#")[0]],
                                            "stop", "") if m == GEMINI and "algebra/" not in
                 r["custom_id"] else plain(m, r))
    fgr.start("masein")
    drain()
    runs = fb.BENCH["otis_aime_epoch"]["epochs"]
    assert sum(1 for m, c in asked if m == GEMINI and "algebra/" not in c) == 2 * runs
    r = results("otis_aime_epoch")
    assert r["results"]["otis_aime_epoch"]["acc,none"] == pytest.approx(2 / 3)
    assert r["results"]["otis_aime_epoch"]["acc_code,none"] == pytest.approx(1 / 3)
    g = grades("otis_aime_epoch")["items"]
    assert g["3#0"]["words"] == "no final answer" and g["2#0"]["read"] == "12"
    how = report.frontier_how(r["frontier"])
    assert how.startswith("code, then Epoch AI's model check by google/gemini-2.5-flash-20250617 "
                          "with our extraction prompt") and how.endswith("· code alone 33.3")


# ---------------------------------------------------------------------------
# 24, 25, 26: the prompt, the answer, the grader
# ---------------------------------------------------------------------------

def test_24_a_graders_prompt_is_filled_in_one_pass():
    t = fg.render("hle", {"question": "What is {response}?", "answer": "KEY-42"},
                  {"answer": "I'd say {correct_answer}"})
    assert t.count("KEY-42") == 1 and "[response]: I'd say {correct_answer}" in t
    assert "[question]: What is {response}?" in t
    t = fg.render("simpleqa", {"question": "Q {target}", "answer": "GOLD"},
                  {"answer": "{target}"})
    assert t.count("GOLD") == 1


def test_25_a_grade_is_kept_for_the_answer_it_graded(svc, monkeypatch):  # noqa: F811
    """an import replaced answers while their batch was out: the new answers
    took the old grades and were never sent"""
    d = sf.task_dir(row_dir(), "simpleqa_epoch")

    def answer(model, row):
        if row["custom_id"].endswith(":0#0"):
            lines = [json.loads(x) for x in (d / sf.ANSWERS).read_text().splitlines()]
            lines[0]["answer"] = "On reflection, Answer 0."
            (d / sf.ANSWERS).write_text("".join(json.dumps(x) + "\n" for x in lines))
        return plain(model, row)
    asked = stub(monkeypatch, answer)
    fgr.start("masein")
    drain()
    g = grades("simpleqa_epoch")
    assert "0#0" not in g["items"] and results("simpleqa_epoch") is None
    assert {r["slot"]: r["answers"] for r in page(svc)["estimate"]["rows"]} == {"simpleqa": 1}
    n = len(asked)
    fgr.start("masein")
    drain()
    assert [c for _, c in asked[n:]] == ["frgr:0#0"]
    g = grades("simpleqa_epoch")
    assert all(x["answer_sha256"] and x["prompt_sha256"] == fg.prompt_sha("simpleqa")
               for k, x in g["items"].items())
    assert results("simpleqa_epoch")["results"]["simpleqa_epoch"]["acc,none"] == \
        pytest.approx(5 / 6)
    # an answer changed after it was graded waits for its grader again
    lines = [json.loads(x) for x in (d / sf.ANSWERS).read_text().splitlines()]
    lines[1]["answer"] = "Not Answer 1 after all."
    (d / sf.ANSWERS).write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert [x["id"] for x in sf.to_grade(row_dir(), "simpleqa_epoch")] == ["1"]


def test_26_a_score_from_two_graders_says_so_and_isnt_final(svc, monkeypatch):  # noqa: F811
    first = {"on": True}

    def answer(model, row):
        if model == GPT and first["on"] and row["custom_id"].endswith((":3#0", ":4#0")):
            return "", "stop", "HTTP 400: the provider refused this request"
        return plain(model, row)
    stub(monkeypatch, answer)
    fgr.start("masein")
    drain()
    # three graded by gpt-4.1; the other two by Gemini, chosen since
    fgr.save("simpleqa", GEMINI, "masein")
    first["on"] = False
    fgr.start("masein")
    drain()
    f = results("simpleqa_epoch")["frontier"]
    assert f["final"] is False and f["grader"] is None
    assert [(x["version"], x["n"]) for x in f["graders"]] == [
        ("openai/gpt-4.1-2025-04-14", 3), ("google/gemini-2.5-flash-20250617", 2)]
    sha = fg.prompt_sha("simpleqa")[:8]
    assert report.frontier_how(f) == (
        f"graded by 2 graders or prompts — openai/gpt-4.1-2025-04-14 with Google's grader "
        f"prompt ({sha}) on 3, google/gemini-2.5-flash-20250617 with Google's grader prompt "
        f"({sha}) on 2: not final · 17% ran out of room (1 of 6), counted wrong")
    assert report.frontier_setting(f) is None                  # ranked with nothing
    sc = sf.score_task(row_dir(), "simpleqa_epoch", rec())
    assert sf.words("simpleqa_epoch", sc).endswith("graded by 2 graders or prompts: not final")


# ---------------------------------------------------------------------------
# 27, 28: the card
# ---------------------------------------------------------------------------

def test_27_the_card_says_why_a_batch_failed(svc, monkeypatch):  # noqa: F811
    """a data-policy refusal on all six left nothing on the card — 17f: the
    first five all refused, the batch stops and the card says why, nothing
    counted against the answers"""
    stub(monkeypatch, lambda m, r: ("", "", "HTTP 404: No endpoints found matching your data "
                                    "policy") if m == GPT else plain(m, r))
    fgr.start("masein")
    for _ in range(300):
        llm_poller.tick()
        if any("the first 5 requests were all refused" in (w.get("why") or "")
               for w in page(svc).get("waits") or []):
            break
        time.sleep(0.05)
    why = " ".join(w.get("why") or "" for w in page(svc).get("waits") or [])
    assert "the first 5 requests were all refused" in why and "data policy" in why, why
    assert not (grades("simpleqa_epoch").get("refused") or {})


def test_27_after_stop_or_at_the_limit_it_says_why_it_waits_and_carry_on_sends_the_rest(
        svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 1)
    gate = threading.Event()
    seen: list = []

    def answer(model, row):
        seen.append(row["custom_id"])
        if model == GPT and len([c for c in seen if "algebra/" not in c]) == 1:
            gate.wait(10)                   # the first SimpleQA request, still out
        return plain(model, row)
    stub(monkeypatch, answer)
    fgr.start("masein")
    # MATH's two land first (its batch runs beside SimpleQA's: one of them
    # still unsent at Stop counted with SimpleQA's, and a CI run read 5)
    math = next(x["batch_id"] for x in fgr.pending() if x["slot"] == "math")
    for _ in range(200):
        if any("algebra/" not in c for c in seen) and \
                (llm.tally(math) or {}).get("answered") == 2:
            break
        time.sleep(0.05)
    fgr.stop("masein")
    gate.set()
    # the reply that was out when Stop was pressed lands
    out = next(x["batch_id"] for x in fgr.pending() if x["slot"] == "simpleqa")
    for _ in range(200):
        if (llm.tally(out) or {}).get("answered") == 1:
            break
        time.sleep(0.05)
    llm_poller.tick()
    p = page(svc)
    # the one out when Stop was pressed landed; four wait
    assert p["stopped"] and p["waits"] == [{"why": "Stopped by masein: 4 answers wait to be "
                                                   "sent", "carry": "Carry on sends the rest."}]
    n = len(seen)
    time.sleep(0.3)
    assert len(seen) == n                   # nothing more goes
    # the month's limit: said, and Carry on refused with it
    limit = "this month's AI spend has reached its limit"
    monkeypatch.setattr(ai_models, "over_limit", lambda: limit)
    # 17c: at the limit Carry on can't send: said so, not "Carry on sends the rest"
    assert {"why": limit, "carry": "Carry on waits until the limit is raised (AI models ▸ the "
                                   "month's limit) or the month turns."} in page(svc)["waits"]
    with pytest.raises(ValueError, match="its limit"):
        fgr.start("masein")
    monkeypatch.setattr(ai_models, "over_limit", lambda: "")
    fgr.start("masein")                     # Carry on
    drain()
    assert results("simpleqa_epoch")["results"]["simpleqa_epoch"]["acc,none"] == \
        pytest.approx(5 / 6)
    assert not page(svc)["waits"]


def test_28_choosing_a_grader_counts_its_probe_and_the_card_says_so(svc, monkeypatch):  # noqa: F811
    usage = {"prompt_tokens": 9, "completion_tokens": 1, "cost": 0.000026}
    monkeypatch.setattr(llm, "_http", lambda *a, **k: (200, json.dumps(
        {"choices": [{"message": {"content": "p"}}], "usage": usage}).encode()))
    prov = {"name": "Prov", "tag": "prov/x", "price_in": 2.0, "price_out": 8.0}
    assert ai_models.probe(GPT, prov, fgr.JOB)[0] is True
    assert db.spend_this_month_by_job()[fgr.JOB] == pytest.approx(0.000026)
    usage.clear()                           # no figure from OpenRouter: tokens at its price
    assert ai_models.probe(GPT, prov, fgr.JOB)[0] is True
    assert db.spend_this_month_by_job()[fgr.JOB] == pytest.approx(0.000026 + (8 * 2 + 8) / 1e6)
    assert "one token" in page(svc)["probe_words"] and "this month's spend" in \
        page(svc)["probe_words"]


# ---------------------------------------------------------------------------
# 29: the Frontier view's setting
# ---------------------------------------------------------------------------

def test_29_the_frontier_view_ranks_by_setting_not_by_a_models_own_numbers():
    g = {"version": "google/gemini-2.5-flash-20250617",
         "prompt_words": "Epoch AI's equivalence prompt", "prompt_sha256": "ab" * 32}
    a = {"scored_by": "code, then Epoch's model check", "look": {"done": 3, "waiting": 0},
         "grader": g, "code": {"score": 0.613}}
    b = {**a, "code": {"score": 0.42}}
    assert report.frontier_how(a).endswith("· code alone 61.3") and \
        report.frontier_how(b).endswith("· code alone 42.0")
    assert report.frontier_setting(a) == report.frontier_setting(b) == (
        "code, then Epoch AI's model check by google/gemini-2.5-flash-20250617 with Epoch AI's "
        "equivalence prompt (abababab)")
    waiting = {**a, "look": {"done": 0, "waiting": 5}}
    assert report.frontier_setting(waiting) == "by code"
    assert report.frontier_how(waiting) == ("by code: Epoch AI's model check waits for Start on "
                                            "5 answers")
