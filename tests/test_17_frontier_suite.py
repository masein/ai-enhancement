"""17, stage 2: the Frontier benchmarks as one suite — each scorer on fixed
answers; thinking on or off asked of a served model in every suite Test a
model offers the box for (the server refused what the dialog offered); a
Frontier run that names its benchmarks; the question viewer following each
licence (GPQA and HLE never listed, and their text in no response); a split
GGUF on a rented box; and a model the board doesn't serve registered by its
first bundle. Invented questions; nothing is fetched and no model runs."""

from __future__ import annotations

import json

import pytest

import frontier as fb
from fake_openai import FakeServer
from service import config, db, questions, runner, served
from service import frontier as sf
from test_12q_devicemark_runs import ME, svc  # noqa: F401 — svc is the fixture


def submit(client, hf_id: str, **kw):
    return client.post("/api/submissions", json={"hf_id": hf_id, "kind": "instruct",
                                                 "submitter": ME, **kw})


def a_served(name: str = "lda box") -> tuple[FakeServer, dict]:
    fake = FakeServer()
    rec = served.register({"name": name, "base_url": fake.base, "how": "llama-server",
                           "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "off"}, ME)
    return fake, rec


# ---------------------------------------------------------------------------
# the dialog and the server agree
# ---------------------------------------------------------------------------

def test_a_served_model_is_asked_thinking_on_in_every_suite_the_dialog_offers(svc):  # noqa: F811
    fake, rec = a_served()
    try:
        for suite in ("generative", "shared", "mobile", "frontier"):
            r = submit(svc, rec["id"], suite=suite, thinking=True)
            assert r.status_code == 200, (suite, r.text)
            assert db.get(r.json()["id"])["thinking"] == 1
        # a model from OpenRouter has no switch: refused, saying so
        db.served_put({**rec, "id": "served/or-model", "via": "openrouter"})
        r = submit(svc, "served/or-model", suite="shared", thinking=True)
        assert r.status_code == 422 and "thinks as it does" in r.json()["detail"]
    finally:
        fake.close()


def test_a_frontier_run_names_its_benchmarks(svc):  # noqa: F811
    fake, rec = a_served()
    try:
        r = submit(svc, rec["id"], suite="frontier", tasks=[fb.TASKS[-1], fb.TASKS[0]])
        assert r.status_code == 200, r.text
        # kept in the suite's order
        assert json.loads(db.get(r.json()["id"])["tasks"]) == sorted([fb.TASKS[0], fb.TASKS[-1]])
        r = submit(svc, rec["id"], suite="frontier", tasks=["mmlu"])
        assert r.status_code == 422 and "not Frontier benchmarks: mmlu" in r.json()["detail"]
        r = submit(svc, rec["id"], suite="generative", tasks=["ifeval"])
        assert r.status_code == 422 and "Frontier run" in r.json()["detail"]
    finally:
        fake.close()


def test_the_page_says_each_benchmark_and_how_it_differs(svc):  # noqa: F811
    import report_lm_eval as report
    meta = report.frontier_bench_meta()
    assert list(meta) == fb.TASKS
    for t, b in meta.items():
        assert b["label"] and b["runs"] == fb.BENCH[t]["epochs"]
        assert b["scorer"] in ("code", "grader")
        assert b["unlisted"] == bool(fb.unlisted(t))
    # every cell not run Epoch's way says so
    notes = {t: b["note"] for t, b in meta.items()}
    assert notes["hle_text_cais"] == "text-only questions"
    assert notes["arc_agi2_public"] == "public set"
    assert notes["mmlupro_tiger"] == "TIGER-Lab's 5-shot protocol"
    # Epoch's own four, run its way: no note (their runs are in the cell's tag)
    assert not any(notes[t] for t in ("gpqa_diamond_epoch", "otis_aime_epoch", "math_l5_epoch",
                                      "simpleqa_epoch"))
    # MMLU-Pro is every test question, not a sample
    assert fb.BENCH["mmlupro_tiger"]["n"] == 12032 and "subset" not in fb.BENCH["mmlupro_tiger"]
    # and each is kept out of every Avg and Improve
    assert set(fb.TASKS) <= set(report.FRONTIER_TASKS)


# ---------------------------------------------------------------------------
# each scorer, on fixed answers
# ---------------------------------------------------------------------------

def test_otis_prompt_and_integer_answers():
    text, need = fb.prompt("otis_aime_epoch", {"id": "1", "question": "Find $x$ {sic}.",
                                               "answer": "042"})
    assert text == ("Please solve this AIME problem step by step. The answer is an integer "
                    "ranging from\n000 to 999, inclusive.\n\nFind $x$ {sic}.\n\nRemember to show "
                    "your work clearly and end with 'ANSWER: X' where X is your final numerical "
                    "answer.")
    sc = lambda a, fin="stop": fb.score("otis_aime_epoch", a, fin, need)  # noqa: E731
    assert sc("so it is 41.\nANSWER: 42") == {"ok": True, "read": "42", "ran_out": False}
    assert sc("ANSWER: 042")["ok"] and sc("ANSWER: $\\boxed{42}$")["ok"]
    assert sc("ANSWER: 41\nwait\nANSWER: 42")["read"] == "42"           # the last one
    assert sc("The answer is 42.")["read"] is None                      # code can't read it
    assert sc("ANSWER: 42", "length")["ok"] is False                    # ran out: wrong


def test_math_level5_epoch_extraction():
    text, need = fb.prompt("math_l5_epoch", {"id": "algebra/1", "question": "Compute $1/2$.",
                                             "answer": "\\frac{1}{2}"})
    assert text.startswith('Solve the following math problem step by step. The last line of '
                           'your response should be of the form "ANSWER: $ANSWER"')
    assert text.endswith("and you do not need to use a \\boxed command.")
    read = fb.read_math
    assert read("work\nANSWER: \\frac{1}{2}") == "\\frac{1}{2}"
    assert read("**ANSWER: 7**") == "7"
    # inside \\text{ANSWER…} first, else the rest of its line — as Epoch's code takes it
    assert read("$\\text{ANSWER: 5}$") == "5"
    assert read("$\\text{ANSWER: }5$ and more") == "5$ and more"
    assert read("first ANSWER: 3\nlater ANSWER: 4") == "3"           # Epoch's: the first line
    assert read("no final line") is None
    assert fb.last_boxed_only_string("so \\boxed{\\frac{a}{b}} ok") == "\\boxed{\\frac{a}{b}}"
    assert fb.remove_boxed("\\boxed{\\frac{a}{b}}") == "\\frac{a}{b}"
    sc = fb.score("math_l5_epoch", "ANSWER: 0.5", "stop", need)
    assert sc["read"] == "0.5" and sc["ok"] in (True, False)            # math-verify decides
    assert fb.score("math_l5_epoch", "ANSWER: \\frac{1}{2}", "stop", need)["ok"] is True
    assert fb.score("math_l5_epoch", "ANSWER: 3", "stop", need)["ok"] is False


def test_mmlu_pro_tiger_prompt_and_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "_fetch", lambda task: {
        "items": [{"id": "7", "question": "Which?", "options": ["w", "x", "y"], "answer": "B",
                   "category": "law"}],
        "extra": {"shots": {"law": [{"question": "Q1?", "options": ["a", "b"],
                                     "cot_content": "A: Let's think step by step. So (A)."}]}}})
    fb.set_root(tmp_path)
    item = fb.load("mmlupro_tiger", tmp_path)[0]
    text, need = fb.prompt("mmlupro_tiger", item)
    assert text == ("The following are multiple choice questions (with answers) about law. Think "
                    "step by step and then output the answer in the format of \"The answer is "
                    "(X)\" at the end.\n\nQuestion: Q1?\nOptions: A. a\nB. b\nAnswer: Let's think "
                    "step by step. So (A).\n\nQuestion: Which?\nOptions: A. w\nB. x\nC. y\n"
                    "Answer: Let's think step by step.\n\n")
    read = fb.read_mmlu_pro
    assert read("...the answer is (B).") == "B"
    assert read("**The answer is (C)**") == "C" and read("The answer is **(C)**") == "C"
    assert read("Answer: D") == "D"                                     # extract_again
    assert read("I pick E finally") == "E"                              # extract_final
    assert read("none here") is None                                    # no random guess
    assert fb.score("mmlupro_tiger", "the answer is (B)", "stop", need)["ok"] is True


def test_arc_prompt_and_grids():
    item = {"id": "abc#0", "group": "abc", "train": [{"input": [[1]], "output": [[2]]}],
            "test_input": [[3]], "answer": [[4, 4]]}
    text, need = fb.prompt("arc_agi2_public", item)
    assert "--Example 0-- \n\n INPUT: \n\n[[1]]\n\nOUTPUT: \n\n[[2]]\n\n" in text
    assert text.endswith("--Test Input--\n[[3]]\n--End of Test Input--\n\nYour response:")
    assert fb.read_grid("so \\boxed{[[4, 4]]}") == [[4, 4]]
    assert fb.read_grid("first [[1]] then [[4,4]]") == [[4, 4]]
    assert fb.read_grid("nothing") is None
    assert fb.score("arc_agi2_public", "[[4, 4]]", "stop", need)["ok"] is True
    # pass@2: a grid solved by either attempt; a task the share of its grids
    per = {"t1#0": [0.0, 1.0], "t1#1": [0.0, 0.0], "t2#0": [1.0, 1.0]}
    sm = fb.summary(per, "arc_agi2_public", {"t1#0": "t1", "t1#1": "t1", "t2#0": "t2"})
    assert sm["score"] == pytest.approx((0.5 + 1.0) / 2) and sm["n"] == 2


def test_graded_ones_wait_and_hle_reads_its_answer_line():
    text, need = fb.prompt("simpleqa_epoch", {"id": "5", "question": "Who?", "answer": "Ada"})
    assert text == ("Who?\n\nThis is a guessing game, not a test. There's no penalty for being "
                    "wrong, and an uncertain guess is far more useful to me than \"I don't "
                    "know\". Give your single best guess.")
    assert fb.score("simpleqa_epoch", "Ada Lovelace", "stop", need)["ok"] is None
    assert fb.system_of("hle_text_cais").startswith("Your response should be in the following "
                                                    "format:\nExplanation:")
    sc = fb.score("hle_text_cais", "Explanation: x\nAnswer: 12\nConfidence: 80%", "stop",
                  {"answer": "12"})
    assert sc == {"ok": None, "read": "12", "ran_out": False}


# ---------------------------------------------------------------------------
# the question viewer follows each licence
# ---------------------------------------------------------------------------

def test_unlisted_questions_are_in_no_response(svc, monkeypatch):  # noqa: F811
    """GPQA's and HLE's text never leaves the server: not in the list of
    benchmarks, not in a page of questions, not in a model's answers"""
    hidden = [t for t in fb.TASKS if fb.unlisted(t)]
    assert set(hidden) == {"gpqa_diamond_epoch", "hle_text_cais", "math_l5_epoch"}
    secret = "Invented question that must never be shown"
    for t in hidden:
        monkeypatch.setattr(fb, "_fetch", lambda task, t=t: [
            {"id": f"q{k}", "question": f"{secret} {k}?", "right": "right", "answer": "x",
             "wrong": ["a", "b", "c"], "options": ["right", "a"], "answer_type": "exactMatch",
             "group": f"g{k}"} for k in range(4)])
        fb.load(t, config.BENCH_ROOT)
        row = config.OUT_DIR / "served__lda-box"
        d = sf.task_dir(row, t)
        d.mkdir(parents=True, exist_ok=True)
        (d / sf.ANSWERS).write_text("".join(json.dumps(
            {"id": f"q{k}", "epoch": e, "answer": "ANSWER: A", "finish": "stop"}) + "\n"
            for k in range(4) for e in range(fb.BENCH[t]["epochs"])))
        (row / "model_meta.json").write_text(json.dumps({"model": "served/lda-box"}))
    listed = svc.get("/api/questions").json()
    assert not any(t in json.dumps(listed) for t in hidden)
    for t in hidden:
        r = svc.get(f"/api/questions/{t}")
        assert r.status_code in (403, 404), r.status_code
        assert secret not in r.text
        r = svc.get("/api/answers/benchmarks", params={"model": "served/lda-box"})
        assert secret not in r.text and t not in [x["task"] for x in r.json()["tasks"]]
    with pytest.raises(PermissionError):
        questions.table(hidden[0])


def test_a_listed_frontier_benchmark_shows_its_diagnose_half(svc, monkeypatch):  # noqa: F811
    """MMLU-Pro (MIT) lists its questions from this server's copy, each with
    the model's answer read as TIGER-Lab reads it; the report half stays a count"""
    items = [{"id": str(k), "question": f"Listed question {k}?", "options": ["w", "x", "y"],
              "answer": "B", "category": "law"} for k in range(12)]
    monkeypatch.setattr(fb, "_fetch", lambda task: {"items": items, "extra": {"shots": {}}})
    fb.load("mmlupro_tiger", config.BENCH_ROOT)
    row = config.OUT_DIR / "served__lda-box"
    d = sf.task_dir(row, "mmlupro_tiger")
    d.mkdir(parents=True)
    (d / sf.ANSWERS).write_text("".join(json.dumps(
        {"id": str(k), "epoch": 0, "answer": f"the answer is ({'B' if k % 3 else 'C'})",
         "finish": "stop"}) + "\n" for k in range(12)))
    (row / "model_meta.json").write_text(json.dumps({"model": "served/lda-box"}))
    assert "mmlupro_tiger" in [x["task"] for x in svc.get("/api/questions").json()["tasks"]]
    pg = svc.get("/api/questions/mmlupro_tiger").json()
    assert pg["kind"] == "frontier" and pg["listed"] + pg["other"] == 12 and pg["listed"] > 0
    assert pg["meta"]["licence"] == "MIT" and "TIGER-Lab" in pg["meta"]["source"]
    r0 = pg["rows"][0]
    k = int(r0["key"]) if "key" in r0 else int(r0["q"].split()[-1].rstrip("?"))
    res = next(iter(r0["results"].values()))
    assert r0["q"] == f"Listed question {k}?" and r0["options"] == ["w", "x", "y"]
    assert res["ok"] is bool(k % 3) and res["verdict"].startswith("read as ")
    # the report half: never a question, only its count
    shown = {x["q"] for x in pg["rows"]}
    assert len(shown) == pg["listed"] < 12 or pg["other"] == 0
    # and the unlisted ones answer 403, saying why
    r = svc.get("/api/questions/hle_text_cais")
    assert r.status_code == 403 and "not be shared" in r.json()["detail"]


# ---------------------------------------------------------------------------
# the box: a split GGUF, and a model only rented GPUs run
# ---------------------------------------------------------------------------

def test_a_split_gguf_is_every_part():
    import remote_gguf as rg
    parts = rg.split_parts("hf://me/repo/gemma-4-26b-a4b-it-BF16-00001-of-00003.gguf")
    assert parts == [f"hf://me/repo/gemma-4-26b-a4b-it-BF16-0000{i}-of-00003.gguf"
                     for i in (1, 2, 3)]
    assert rg.split_parts("/x/model.gguf") == ["/x/model.gguf"]
    with pytest.raises(SystemExit, match="first part"):
        rg.split_parts("/x/m-00002-of-00003.gguf")
    a = rg.split_sha([("m-00001-of-00002.gguf", "1" * 64), ("m-00002-of-00002.gguf", "2" * 64)])
    b = rg.split_sha([("m-00001-of-00002.gguf", "1" * 64), ("m-00002-of-00002.gguf", "3" * 64)])
    assert a != b and len(a) == 64


def test_runner_refuses_a_frontier_run_of_a_hub_model(svc):  # noqa: F811
    sid = db.add("Qwen/Qwen3.5-4B", "instruct", "frontier", ME, "")
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "failed" and "running on a server" in row["error"]
