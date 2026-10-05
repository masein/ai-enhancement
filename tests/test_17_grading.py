"""17, stage 3: the Frontier benchmarks graded on the server. Nothing is
graded before Start (an import, a run, the poller send nothing); the dry run's
answers, tokens and cost; Start sends each grader what waits, the grades land
beside the answers with the grader's pin and prompt, and the benchmark is
scored again — SimpleQA by its grader, MATH Level 5 by code with Epoch's
check as a second look, both numbers kept; a grader that refuses is kept in
its words and asked again by the next Start. Invented questions and a stand-in
for OpenRouter only: nothing is sent anywhere."""

from __future__ import annotations

import json
import time

import pytest

import frontier as fb
import frontier_graders as fg
import report_lm_eval as report
from conftest import make_service
from service import ai_models, config, db, llm, llm_poller
from service import frontier as sf
from service import frontier_grade as fgr

SERVED = "served/lda-box"
ROW = "served__lda-box"
GRADERS = {"openai/gpt-4.1": ("openai/gpt-4.1-2025-04-14", 2.0, 8.0),
           "openai/o3-mini": ("openai/o3-mini-2025-01-31", 1.1, 4.4),
           "google/gemini-2.5-flash": ("google/gemini-2.5-flash-20250617", 0.3, 2.5)}


def sqa_items():
    return [{"id": str(k), "question": f"Invented fact {k}?", "answer": f"Answer {k}",
             "subject": "x"} for k in range(6)]


def math_items():
    return [{"id": f"algebra/{k}", "question": f"Invented problem {k}.", "answer": "\\frac{1}{2}",
             "subject": "Algebra"} for k in range(4)]


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch)
    data = {"simpleqa_epoch": sqa_items(), "math_l5_epoch": math_items()}
    monkeypatch.setattr(fb, "_fetch", lambda task: data[task])
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_http", lambda *a, **k: pytest.fail("nothing calls OpenRouter"))
    monkeypatch.setattr(ai_models, "drifted", lambda pin: "")
    monkeypatch.setattr(ai_models, "pin", lambda mid: {
        "kind": "openrouter", "id": mid, "version": GRADERS[mid][0], "name": mid, "provider": "p",
        "provider_name": "Prov", "precision": "unknown", "price_in": GRADERS[mid][1],
        "price_out": GRADERS[mid][2]})
    # OpenRouter's list as AI models last fetched it: the dry run's prices
    p = ai_models._cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"at": time.time(), "models": [
        {"id": m, "name": m, "version": v, "price_in": pi, "price_out": po}
        for m, (v, pi, po) in GRADERS.items()]}))
    db.served_put({"id": SERVED, "name": "lda box", "base_url": "", "key": "", "how": "x",
                   "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto",
                   "pin": {"file": "m.gguf"}, "by": "masein", "at": 0})
    row = config.OUT_DIR / ROW
    row.mkdir(parents=True)
    (row / "model_meta.json").write_text(json.dumps({"model": SERVED}))
    # SimpleQA: six answers, the last ran out of room
    write(row, "simpleqa_epoch", [(str(k), f"I think Answer {k}.", "stop" if k < 5 else "length")
                                  for k in range(6)])
    # MATH: right by code, wrong by code but equivalent, wrong, and unreadable
    write(row, "math_l5_epoch", [("algebra/0", "ANSWER: \\frac{1}{2}", "stop"),
                                 ("algebra/1", "ANSWER: one half", "stop"),
                                 ("algebra/2", "ANSWER: 3", "stop"),
                                 ("algebra/3", "no final line", "stop")])
    yield client
    client.__exit__(None, None, None)


def write(row, task, answers):
    d = sf.task_dir(row, task)
    d.mkdir(parents=True, exist_ok=True)
    (d / sf.ANSWERS).write_text("".join(json.dumps(
        {"id": q, "epoch": 0, "answer": a, "finish": f}) + "\n" for q, a, f in answers))


def rec():
    from service import served
    return served.get(SERVED)


def results(task):
    fs = list(sf.task_dir(config.OUT_DIR / ROW, task).glob("results_*.json"))
    return json.loads(fs[0].read_text()) if fs else None


def drain(timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        llm_poller.tick()
        if not fgr.pending():
            return
        time.sleep(0.05)
    pytest.fail("the grading batches never finished")


def test_nothing_is_graded_before_start(svc):
    # scored by what code can: SimpleQA waits whole, MATH has the code's number
    assert sf.score_task(config.OUT_DIR / ROW, "simpleqa_epoch", rec()) == {
        "waiting": 5, "of": 6, "label": "SimpleQA Verified"}
    assert results("simpleqa_epoch") is None
    m = sf.score_task(config.OUT_DIR / ROW, "math_l5_epoch", rec())
    assert m["score"] == pytest.approx(0.25) and m["look"] == {"done": 0, "waiting": 2}
    assert "Epoch's model check not run on 2" in sf.words("math_l5_epoch", m)
    res = results("math_l5_epoch")
    assert res["results"]["math_l5_epoch"]["acc_code,none"] == pytest.approx(0.25)
    assert report.frontier_how(res["frontier"]) == (
        "by code: Epoch AI's model check waits for Start on 2 answers")
    # the poller, the page: nothing out, nothing sent
    llm_poller.tick()
    assert not db.batches_pending() and not fgr.pending()
    assert svc.get("/api/frontier/grading").status_code == 200
    assert not db.batches_pending()


def test_the_dry_run_counts_answers_tokens_and_cost(svc):
    est = svc.get("/api/frontier/grading").json()["estimate"]
    # SimpleQA: every answer but the one that ran out; MATH: the two the code
    # marks wrong, never the one it can't read (Epoch scores that wrong unasked)
    by = {r["slot"]: r for r in est["rows"]}
    assert by["simpleqa"]["answers"] == 5 and by["math"]["answers"] == 2
    items = {it["id"]: it for it in sqa_items()}
    m = sf.marks(config.OUT_DIR / ROW, "simpleqa_epoch")
    tin = sum(len(fg.render("simpleqa", items[q], rs[0])) for q, rs in m["runs"].items()
              if not rs[0]["ran_out"]) // fgr.CHARS_A_TOKEN
    g = est["graders"]["simpleqa"]
    assert (g["answers"], g["tokens_in"], g["tokens_out"]) == (5, tin, 10)
    assert g["usd"] == pytest.approx((tin * 2.0 + 10 * 8.0) / 1e6, abs=1e-4)
    assert est["answers"] == 7 and est["usd_known"] is True


def test_start_grades_and_both_numbers_are_kept(svc, monkeypatch):
    asked = []

    def complete(self, row):
        asked.append((self.model, row["custom_id"]))
        if self.model == "openai/gpt-4.1":
            # Google's grader: the fourth answer graded INCORRECT
            text = "B" if row["custom_id"].endswith(":3#0") else "A"
        else:
            text = "Yes" if "one half" in row["user"] else "No"
        return {"custom_id": row["custom_id"], "text": text, "error": "", "attempts": 1,
                "finish_reason": "stop"}
    monkeypatch.setattr(fgr.GraderChat, "_complete", complete)
    r = svc.post("/api/frontier/grading/start", json={"by": "masein"})
    assert r.status_code == 200, r.text
    drain()
    assert sorted(m for m, _ in asked) == ["google/gemini-2.5-flash"] * 2 + ["openai/gpt-4.1"] * 5
    # SimpleQA: the share graded correct over all six (the one that ran out is wrong)
    s = results("simpleqa_epoch")
    assert s["results"]["simpleqa_epoch"]["acc,none"] == pytest.approx(4 / 6)
    g = s["frontier"]["grader"]
    assert (g["version"], g["prompt"]) == ("openai/gpt-4.1-2025-04-14", "simpleqa_google.txt")
    assert g["prompt_sha256"] == fg.prompt_sha("simpleqa") and g["owners"] == "gpt-4.1-2025-04-14"
    assert report.frontier_how(s["frontier"]) == (
        f"graded by openai/gpt-4.1-2025-04-14 with Google's grader prompt "
        f"({fg.prompt_sha('simpleqa')[:8]})")
    # MATH: Epoch's way on the page (one more right), the code's beside it
    m = results("math_l5_epoch")["results"]["math_l5_epoch"]
    assert m["acc,none"] == pytest.approx(0.5) and m["acc_code,none"] == pytest.approx(0.25)
    how = report.frontier_how(results("math_l5_epoch")["frontier"])
    assert how.startswith("code, then Epoch AI's model check by "
                          "google/gemini-2.5-flash-20250617 with Epoch AI's equivalence prompt")
    assert how.endswith("· code alone 25.0")
    # the grades beside the answers
    gr = sf.read_grades(sf.task_dir(config.OUT_DIR / ROW, "math_l5_epoch"))
    assert gr["items"]["algebra/1#0"]["ok"] is True and gr["items"]["algebra/2#0"]["ok"] is False
    # Start again: nothing waits, nothing is sent
    n = len(asked)
    assert svc.post("/api/frontier/grading/start", json={"by": "masein"}).status_code == 200
    drain()
    assert len(asked) == n and svc.get("/api/frontier/grading").json()["estimate"]["answers"] == 0


def test_a_grader_that_refuses_is_kept_in_its_words_and_asked_again(svc, monkeypatch):
    refuse = {"on": True}

    def complete(self, row):
        if refuse["on"] and row["custom_id"].endswith(":2#0"):
            return {"custom_id": row["custom_id"], "text": "", "attempts": 1,
                    "error": "HTTP 400: the provider refused this request", "finish_reason": ""}
        return {"custom_id": row["custom_id"], "text": "A" if self.model == "openai/gpt-4.1"
                else "No", "error": "", "attempts": 1, "finish_reason": "stop"}
    monkeypatch.setattr(fgr.GraderChat, "_complete", complete)
    assert svc.post("/api/frontier/grading/start", json={"by": "masein"}).status_code == 200
    drain()
    page = svc.get("/api/frontier/grading").json()
    ref = {(x["slot"], x["row"]): x for x in page["refused"]}
    assert ref[("simpleqa", ROW)]["n"] == 1 and "refused" in ref[("simpleqa", ROW)]["words"]
    # SimpleQA still waits for the one refused: no score yet; it is asked again
    assert results("simpleqa_epoch") is None
    assert page["estimate"]["rows"] == [
        {"slot": "simpleqa", "task": "simpleqa_epoch", "label": "SimpleQA Verified",
         "model": SERVED, "answers": 1, "usd": page["estimate"]["rows"][0]["usd"]}]
    refuse["on"] = False
    assert svc.post("/api/frontier/grading/start", json={"by": "masein"}).status_code == 200
    drain()
    assert results("simpleqa_epoch")["results"]["simpleqa_epoch"]["acc,none"] == pytest.approx(5 / 6)
    assert not svc.get("/api/frontier/grading").json()["refused"]


def test_start_is_refused_without_a_key_and_a_grader_is_never_local(svc, monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    r = svc.post("/api/frontier/grading/start", json={"by": "masein"})
    assert r.status_code == 409 and "no key" in r.json()["detail"]
    assert not db.batches_pending()
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    r = svc.post("/api/ai/graders/simpleqa", json={"model": "local", "by": "masein"})
    assert r.status_code == 422 and "OpenRouter" in r.json()["detail"]
    r = svc.post("/api/ai/graders/simpleqa", json={"model": "openai/gpt-4.1", "by": "masein"})
    assert r.status_code == 200 and r.json()["grading"]["graders"][0]["chosen"]["version"] == \
        "openai/gpt-4.1-2025-04-14"


def test_each_grader_reads_its_reply_as_its_owners_do():
    item = {"answer": "042"}
    assert fg.read("simpleqa", "A", {})["ok"] is True
    assert fg.read("simpleqa", "C", {})["words"] == "not attempted"
    # Google's own reading: the first capital A, B or C anywhere ("INCORRECT" holds a C),
    # else the words, lower case included
    assert fg.read("simpleqa", "it is INCORRECT", {})["grade"] == "C"
    assert fg.read("simpleqa", "correct", {})["grade"] == "A"
    assert fg.read("simpleqa", "", {})["grade"] == "C"
    h = fg.read("hle", "extracted_final_answer: 12\nreasoning: same\ncorrect: yes\nconfidence: 80",
                {})
    assert h["ok"] is True and h["confidence"] == 80 and "read as 12" in h["words"]
    assert fg.read("hle", "correct: no", {})["ok"] is False
    assert fg.read("hle", "I can't tell", {})["ok"] is False               # unread: wrong
    assert fg.read("math", " Yes\n", {})["ok"] is True and fg.read("math", "No", {})["ok"] is False
    assert fg.read("otis", "42", item)["ok"] is True and fg.read("otis", "NONE", item)["ok"] is False
    # the prompts as their owners wrote them
    assert fg.prompt_text("hle").startswith("Judge whether the following [response] to "
                                            "[question] is correct")
    assert "if there if there" in fg.prompt_text("hle")                    # CAIS's typo, kept
    assert fg.render("math", {"answer": "\\frac{1}{2}"}, {"read": "0.5"}).endswith(
        "  Expression 1: \\frac{1}{2}\n  Expression 2: 0.5")
