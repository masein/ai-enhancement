"""18b part 4: 17j's follow-ups in grading and the fetch (points 19 to 26).
Invented replies, answers, boxes and listings; a stand-in ssh; nothing is
fetched, no model runs, no paid API is called."""

from __future__ import annotations

import json
import time

import frontier as fb
import frontier_graders as fg
from service import frontier as sf
from test_17_grading import ROW, rec, svc, write  # noqa: F401 — svc is the fixture
from test_17e_review import a_bundle
from test_17f_review import listing_of
from test_17g_review import MMLU, progress_of, with_steps
from test_17j_boxes import main_of, world

READ = {True: "yes", False: "no", None: "no grade"}


def hle(reply: str) -> str:
    return READ[fg.read("hle", reply, {"id": "h1", "answer": "42"})["ok"]]


# ---------------------------------------------------------------------------
# 19. a reply cut at its cap is graded only when it is one whole verdict object
# ---------------------------------------------------------------------------

def test_19_a_cut_reply_is_graded_only_when_it_is_one_whole_object(tmp_path):
    from service import frontier_grade as fgr
    from service import llm
    d = tmp_path / "hle"
    d.mkdir()
    (d / sf.ANSWERS).write_text("".join(json.dumps({"id": f"h{k}", "epoch": 0, "answer": "41"})
                                        + "\n" for k in range(1, 4)))
    meta = {"answers": {f"h{k}#0": sf.answer_sha("41") for k in range(1, 4)},
            "max_tokens": 4096, "by": "masein", "prompt_sha256": fg.prompt_sha("hle")}
    whole = json.dumps({"extracted_final_answer": "41", "reasoning": "differs",
                        "correct": "no", "confidence": 90})
    # the judge, cut mid-reasoning after quoting a line that says yes
    lines = ("extracted_final_answer: 41\nreasoning: the response ends with a line\n"
             "correct: yes\nwhich is its own claim; but 41 is not 42 and so the response")
    g: dict = {}
    items = {f"h{k}": {"id": f"h{k}", "answer": "42"} for k in range(1, 4)}
    fgr._apply(g, d, "hle", {"id": "openai/o3-mini", "version": "o3-mini-2025-01-31"}, meta,
               items, {"frgr:h1#0": llm.Result(text=whole, finish="length"),
                       "frgr:h2#0": llm.Result(text=lines, finish="length"),
                       "frgr:h3#0": llm.Result(text="```json\n" + whole + "\n```",
                                               finish="length")})
    assert g["items"]["h1#0"]["ok"] is False and g["items"]["h3#0"]["ok"] is False
    assert "h2#0" not in g["items"]                                  # 0abb757 … 70df001: yes
    assert g["refused"]["h2#0"]["words"] == "the reply was cut at its cap of 4096 tokens"
    assert fg.whole_verdict("hle", whole) and not fg.whole_verdict("hle", lines)
    assert not fg.whole_verdict("hle", whole + "\nand more")
    assert fg.whole_verdict("simpleqa", "A")                       # a lone grade stands


# ---------------------------------------------------------------------------
# 20. a line ending in { never opens an object
# ---------------------------------------------------------------------------

def test_20_a_line_ending_in_a_brace_leaves_the_line_form_read():
    for reply, want in (
            ("extracted_final_answer: {\nreasoning: the set is empty\ncorrect: no\n"
             "confidence: 80", "no"),
            ("extracted_final_answer: S = {\nreasoning: matches the set\ncorrect: yes\n"
             "confidence: 90", "yes"),
            ("extracted_final_answer: 42\nreasoning: in code, if (x) {\ncorrect: yes\n"
             "confidence: 70", "yes")):
        assert hle(reply) == want, reply                           # 70df001: no grade
    # a quoted name on the next line still opens one, as JSON prints it
    assert hle('{\n  "extracted_final_answer": "41",\n  "reasoning": "r",\n  "correct": "no",'
               '\n  "confidence": 9\n}') == "no"


# ---------------------------------------------------------------------------
# 21. the share on a graded benchmark, waiting for its grader; above 5% said
# ---------------------------------------------------------------------------

def test_21_a_graded_runs_share_is_said_while_it_waits(svc):  # noqa: F811
    from service import config
    row = config.OUT_DIR / ROW
    thought = "<think>\nweighing it\n</think>\n\nI think Answer 0."
    write(row, "simpleqa_epoch", [("0", thought, "stop")] + [
        (str(k), f"I think Answer {k}.", "stop") for k in range(1, 6)])
    (sf.task_dir(row, "simpleqa_epoch") / sf.SETUP).write_text(json.dumps({"thinking": "off"}))
    sc = sf.score_task(row, "simpleqa_epoch", rec())
    assert sc["waiting"] == 6 and sc["thinking_held"] == 1          # 70df001: no share
    words = sf.words("simpleqa_epoch", sc)
    assert words.startswith("SimpleQA Verified: 6 answers wait for its grader (AI models ▸ Start)"
                            " · 1 of 6 thought anyway, 16.7% (thinking off; scored on what "
                            "follows the thinking) — not comparable with thinking-off numbers "
                            "published elsewhere"), words
    # under 5%: the share, not the warning; a quarter stays the refusal line
    assert sf.thought_note(76, 2158) == ("76 of 2,158 thought anyway, 3.5% (thinking off; "
                                         "scored on what follows the thinking)")
    assert sf.THINKING_OFF_SHARE == 0.25


# ---------------------------------------------------------------------------
# 22. a compare that couldn't run is no verdict
# ---------------------------------------------------------------------------

def test_22_a_compare_that_didnt_run_is_asked_again_not_stored(tmp_path, monkeypatch, capsys):
    dest = tmp_path / "b" / "bundles"
    (dest.parent / "parity").mkdir(parents=True)
    (dest.parent / "parity" / "orig-box.jsonl").write_text('{"parity_of": {}}\n')
    server = tmp_path / "orig-server-500.jsonl"
    server.write_text("{}\n")
    ff, calls, key, state = world(tmp_path, monkeypatch, lambda host, rnd: None,
                                  compare=(1, 'service "bench" is not running\n'))
    main_of(ff, key, dest, "--parity", f"served/orig={server}", "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "orig's parity compare didn't run (service \"bench\" is not running) — asked again" \
        in out
    vfile = dest.parent / "parity" / "verdicts.json"
    assert not vfile.exists() or "orig" not in json.loads(vfile.read_text())  # 70df001: "not
    # the same", printed with exit 1 every round
    ff, calls, key, state = world(tmp_path, monkeypatch, lambda host, rnd: None,
                                  compare=(3, "Not the same: 38.0% against 42.4%\n"))
    main_of(ff, key, dest, "--parity", f"served/orig={server}", "1.1.1.1:41")
    assert json.loads(vfile.read_text())["orig"]["same"] is False


# ---------------------------------------------------------------------------
# 23. destroyed only when read safe by this fetch, or unreached 30 minutes
# ---------------------------------------------------------------------------

def test_23_a_safe_box_from_before_a_restart_needs_30_minutes_unreached(tmp_path, monkeypatch,
                                                                        capsys):
    ff, calls, key, state = world(tmp_path, monkeypatch, lambda host, rnd: None)
    dest = tmp_path / "b" / "bundles"
    dest.mkdir(parents=True)
    bid = ff.box_id("1.1.1.1:41", dest)
    (dest.parent / "safe-boxes.json").write_text(json.dumps(
        {bid: {"at": time.time() - 5 * 3600, "missed": 0}}))   # read safe hours ago, before
    assert main_of(ff, key, dest, "1.1.1.1:41") == 1            # a restart
    assert main_of(ff, key, dest, "1.1.1.1:41") == 1            # 70df001: 0, "destroyed: done"
    out = capsys.readouterr().out
    # 18c point 15: each fetch counts its own misses — two restarts are one miss each
    assert "destroyed if it isn't reached on the next round either. NOT safe to destroy " \
           "yet" in out, out
    # 18c point 15: a miss an earlier fetch recorded doesn't count in this one
    # (that was one failed ssh after a restart reading a box "destroyed"):
    # tests/test_18c3_grading_fetch_export.py has the rounds within one fetch
    e = json.loads((dest.parent / "safe-boxes.json").read_text())
    e[bid]["missed_since"] = time.time() - 31 * 60
    (dest.parent / "safe-boxes.json").write_text(json.dumps(e))
    assert main_of(ff, key, dest, "1.1.1.1:41") == 1
    assert "destroyed: done" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 24. a refusal halt waits for its press
# ---------------------------------------------------------------------------

def test_24_a_halt_waits_for_start_and_never_lifts_by_itself(tmp_path):
    from service import frontier_grade as fgr
    from service import llm, mmp_key
    for cls, press in ((fgr.GraderChat, "Start"), (mmp_key.LabellerChat, "Carry on")):
        be = cls.__new__(cls)
        be.dir = tmp_path / cls.__name__
        (be.dir / "or_1").mkdir(parents=True)
        be._halt("or_1", {"error": "HTTP 402: insufficient credits", "status": 402}, 20)
        h = json.loads((be.dir / "or_1" / "halt.json").read_text())
        h["at"] = time.time() - 3600                          # an hour on: still waiting
        (be.dir / "or_1" / "halt.json").write_text(json.dumps(h))
        why = llm._halted(be.dir / "or_1", be.HALT_RETRY_S)
        assert why.endswith(f"It waits for {press}: nothing more is sent until then"), why
    # 18c point 13: the judge's batches wait for Carry on too
    be = llm.OpenRouterChat.__new__(llm.OpenRouterChat)
    be.dir = tmp_path / "judge"
    (be.dir / "or_2").mkdir(parents=True)
    be._halt("or_2", {"error": "HTTP 402", "status": 402}, 20)
    assert "It waits for Carry on" in llm._halted(be.dir / "or_2", 600)


# ---------------------------------------------------------------------------
# 25. a planned step imported from another box is done
# ---------------------------------------------------------------------------

def test_25_a_step_that_ran_on_another_box_is_done(tmp_path, monkeypatch, capsys):
    there = tmp_path / "there"
    there.mkdir()
    b = a_bundle(there / "frontier-served__phone-thinking-off-mmlu-pro.tar.gz", "served/phone")
    # A8's second step on this box; its first ran on a box now gone
    whole = progress_of("phone", "A8-2", "whole", tasks=[MMLU], bundle={"name": b.name})

    def listing(host, rnd):
        got = json.loads(with_steps(listing_of({f"/workspace/phone/A8-2/{b.name}": b}, [whole]),
                                    [("phone", "A8-2")]))
        # 18c point 17: the box's line has ended
        got["lines"] = [{"box": "A8", "state": "ended", "build": "phone"}]
        return json.dumps(got)
    ff, calls, key, state = world(tmp_path, monkeypatch, listing)
    monkeypatch.setattr(ff, "copy", lambda *a, **k: (True, "here already"))
    held = {"on": {"tasks": [], "shards": {}}, "off": {"tasks": [], "shards": {}}}
    plain = ff.run

    def run(cmd, cwd=None, timeout=None, stdin=None):
        if "--imported" in cmd:
            return 0, json.dumps(held)
        return plain(cmd, cwd=cwd, timeout=timeout, stdin=stdin)
    monkeypatch.setattr(ff, "run", run)
    dest = tmp_path / "b" / "bundles"
    main_of(ff, key, dest, "1.1.1.1:41")
    assert "phone A8-1 hasn't started" in capsys.readouterr().out
    held["on"]["tasks"] = ["arc_agi2_public"]                    # A8-1's, imported already
    assert main_of(ff, key, dest, "1.1.1.1:41") == 0             # 70df001: hasn't started
    out = capsys.readouterr().out
    assert "phone/A8-1: imported already from another box — done" in out and \
        "hasn't started" not in out


def test_25_the_board_says_what_it_holds_of_a_build(svc):  # noqa: F811
    import import_frontier as imf
    from service import config
    row = config.OUT_DIR / "served__phone"
    row.mkdir(parents=True)
    (row / imf.REGISTRY).write_text(json.dumps({"imports": [], "tasks": {"mmlu_pro": {}},
                                                "shards": {"hle_text_cais": {
                                                    "n": 4, "have": {"2": {}}}}}))
    got = imf.imported_steps("served/phone")
    assert got["off"] == {"tasks": ["mmlu_pro"], "shards": {"hle_text_cais": ["2-of-4"]}}
    assert got["on"] == {"tasks": [], "shards": {}}


# ---------------------------------------------------------------------------
# 26. the small ones
# ---------------------------------------------------------------------------

def test_26_four_rare_shapes_are_no_grade():
    shapes = {
        "a prose verdict quoting the response's object":
            'The response ends with {"correct": "yes"}, but 41 is not 42: it is incorrect.',
        "an outer object whose first key isn't a field":
            '{"judgement": {"extracted_final_answer": "41", "correct": "yes"}, "correct": "no"}',
        "an unquoted object in prose":
            "The response wrote {correct: yes} at its end, which does not hold here.",
        "the response's line quoted after the judge's":
            "extracted_final_answer: 41\nreasoning: differs from 42\ncorrect: no\n"
            "confidence: 80\nThe response itself ended:\n> correct: yes"}
    for what, reply in shapes.items():
        assert hle(reply) == "no grade", what                      # 70df001: yes
    # the response's line that agrees with the judge's: the judge's grade
    assert hle("extracted_final_answer: 41\nreasoning: r\ncorrect: no\nconfidence: 80\n"
               "> correct: no") == "no"
    # 17d stands: the judge's own later line corrects its earlier one
    assert hle("correct: no\non reflection, it matches\ncorrect: yes") == "yes"


def test_26_the_thinking_refusal_alone_reads_safe():
    import frontier_fetch as ff
    think = ("the row served/x: Humanity's Last Exam: thinking was off, and 900 of its 2,158 "
             "answers hold thinking (41.7%) — more than a quarter: the server ignored the "
             "thinking switch: not scored — not imported")
    assert ff.think_only(think)
    assert not ff.think_only(think + " · GPQA Diamond: not imported — scoring it failed (x) — "
                             "not imported")                       # 70df001: read safe
    assert not ff.think_only("refused — its file isn't the one registered\n" + think)
    assert not ff.think_only("")


def test_26_the_start_refusal_no_longer_says_it_carries_on_by_itself():
    from service import frontier_grade as fgr
    got = fgr._limit_words({"limit": 10.0, "spent": 9.5, "usd": 3.0, "answers": 100}, {})
    assert "carry on by itself" not in got["short"] and "wait there for Start" in got["short"]


def test_26_a_thinking_block_anywhere_is_thinking():
    late = "x" * 250 + " <think>maybe C</think> ANSWER: B"
    after = "<think>a</think> ANSWER: A <think>no, ANSWER: C</think>"
    for text, vis in ((late, "x" * 250 + "  ANSWER: B"), (after, "ANSWER: A")):
        assert fb.visible(text) == vis                              # 70df001: inside thinking
        assert sf.thought(text)                                     # 70df001: not in the share
    assert fb.ran_out("y" * 300 + "<think>never closed", "stop")
    assert fb.visible("y" * 300 + "<think>never closed") == "y" * 300
    assert fb.visible("the template opened it</think>ANSWER: D") == "ANSWER: D"
    assert not sf.thought("<think>\n\n</think>ANSWER: A")          # an empty block is none
    assert fb.thinking_of(after) == "a\n\nno, ANSWER: C"


def test_26_a_long_placeholder_like_value_is_read_at_once():
    t0 = time.time()
    assert fg._template("extracted_final_answer", "<" + "a" * 1200) is False
    assert fg._template("extracted_final_answer", "<the extracted answer>") is True
    assert fg._visible("<think>" + "\n" * 20000) == ""
    assert time.time() - t0 < 0.5                                  # 70df001: 2.7 s
