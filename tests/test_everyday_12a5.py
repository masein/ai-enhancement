"""12a.5a: the Everyday bank and checker.

docs/prompts/phase-12a5/ holds the bank the live runs of 2026-09-25 asked
for: 12a.4's 328 questions with their checks fixed, 45 long texts to
summarise and 10 Honesty questions whose answer is in the message. Its
checks.py replaces 12a.4's reference checker, and everyday_probes_12a5.jsonl
holds 776 answers, each with the verdict it must get — 21 of them real
answers from the live board. scripts/everyday.py decides every check exactly
as checks.py does.

An answer is kept by its question's id and the hash of its words. Answers to
words that haven't changed are marked again by today's checks, with no GPU,
and "Run everyday tasks" asks a model only the questions it has no answer to
— after this bank, the 55 new ones."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

import everyday as ev
import report_lm_eval as report
from conftest import make_service
from service import config, db, runner
from test_everyday_12a import fake_gpu, queue_pilot

REPO = Path(__file__).resolve().parents[1]
BRIEF = REPO / "docs" / "prompts" / "phase-12a5"
MODEL = "fx/one-option-70m"          # a fixture model Everyday tasks have not asked yet


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


PROBES = _jsonl(BRIEF / "everyday_probes_12a5.jsonl")
IMPORTED = _jsonl(BRIEF / "everyday_bank_12a5.jsonl")
BANK = {q["id"]: q for q in ev.load_bank()}
NEW = sorted(q["id"] for q in IMPORTED
             if q["id"].startswith(("everyday-summarising-long-", "everyday-honesty-r4-")))
# what 12a.4's bank asked: every question but the 55 new ones
OLD = [q for q in BANK.values() if q["id"] not in NEW]
# each group's hidden half — said in the PR, and pinned here
HIDDEN = {"understanding": 27, "writing": 24, "shorten": 26, "summarising": 26,
          "transform": 25, "quick_maths": 24, "instructions": 24, "honesty": 24}


def reference_checker():
    spec = importlib.util.spec_from_file_location("checks_12a5", BRIEF / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def script_verdict(item, answer):
    return all(ev.run_check(c, answer, item["prompt"])[0]
               for c in item["checks"] if c["type"] != "judge")


def source(q: dict) -> str:
    """the message a shorten or summarise question quotes"""
    return q["prompt"][q["prompt"].index('"') + 1:q["prompt"].rindex('"')]


def _asked(model_dir: Path, questions, answer=lambda q: q["reference"],
           stamp="2026-09-25T00-00-00"):
    """what the harness logs for a run that asked `questions`: its samples and
    its results file"""
    task = model_dir / "everyday_0shot" / "pretrained__x"
    task.mkdir(parents=True, exist_ok=True)
    with open(task / f"samples_everyday_{stamp}.jsonl", "w", encoding="utf-8") as fh:
        for i, q in enumerate(questions):
            fh.write(json.dumps({"doc_id": i, "doc": q, "resps": [[answer(q)]],
                                 "filtered_resps": [answer(q)]}) + "\n")
    (task / f"results_{stamp}.json").write_text(json.dumps({
        "results": {"everyday": {"alias": "everyday", "bypass,none": 999}},
        "configs": {"everyday": {"generation_kwargs": {"max_gen_toks": 512}}},
        "config": {"model_args": "pretrained=x"}}), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. the bank
# ---------------------------------------------------------------------------

def test_the_bank_is_388_questions_in_eight_groups():
    qs = ev.load_bank()
    assert len(qs) == 388 and len(IMPORTED) == 383 and len(NEW) == 55
    # the brief's 383 by id, written_by and all; the pilot's five as they were
    assert all(BANK[q["id"]] == q for q in IMPORTED)
    assert sorted(set(BANK) - {q["id"] for q in IMPORTED}) == \
        [f"everyday-pilot-0{i}" for i in range(1, 6)]
    # the tldr is a message to shorten now, with the short ones that were
    # "Summarising"; "Summarise" is the long texts
    assert BANK["everyday-pilot-03"]["group"] == "shorten"
    assert list(ev.GROUPS.items()) == [
        ("understanding", "Understanding"), ("writing", "Writing"),
        ("shorten", "Shorten a message"), ("summarising", "Summarise"),
        ("transform", "Transform"), ("quick_maths", "Quick maths"),
        ("instructions", "Instructions"), ("honesty", "Honesty")]
    order = [q["group"] for q in qs]
    assert order == sorted(order, key=list(ev.GROUPS).index)          # the file in their order
    assert {g: order.count(g) for g in ev.GROUPS} == {
        "understanding": 45, "writing": 48, "shorten": 63, "summarising": 45,
        "transform": 46, "quick_maths": 45, "instructions": 45, "honesty": 51}
    long_ = [len(source(q).split()) for q in qs if q["group"] == "summarising"]
    assert (min(long_), max(long_)) == (425, 853)      # the brief's 425–850, split on spaces
    # every group has at least 20 hidden; Honesty had 19
    assert {g: c["hidden"] for g, c in ev.split_counts().items()} == HIDDEN
    assert min(HIDDEN.values()) >= config.EVERYDAY_MIN_HIDDEN == 20


def test_the_bank_has_a_new_version():
    """55 new questions: a new hash, so everything marked before this bank is
    an earlier version's until it is marked again"""
    assert ev.version()["hash"] != "8954b300"                         # 12a.4's
    assert ev.bank_hash(OLD) != ev.version()["hash"]


def test_a_blanket_refusal_fails_every_new_honesty_question():
    r4 = [BANK[i] for i in NEW if i.startswith("everyday-honesty-r4-")]
    assert len(r4) == 10
    for q in r4:
        ok, why = ev.grade(q, "As an AI, I can't know that. I don't have access to your "
                              "information.")
        assert ok is False and why, q["id"]
        assert script_verdict(q, q["reference"]), q["id"]


# ---------------------------------------------------------------------------
# 2. the checker
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("i", range(len(PROBES)),
                         ids=[f"{p['id']}#{n}" for n, p in enumerate(PROBES)])
def test_every_probe_gets_its_verdict(i):
    """a probe on a question with a judge check expects its script checks'"""
    p = PROBES[i]
    item = BANK[p["id"]]
    assert script_verdict(item, p["answer"]) == (p["expected"] == "pass"), (p["answer"], [
        ev.run_check(c, p["answer"], item["prompt"]) for c in item["checks"]])
    ok, why = ev.grade(item, p["answer"])
    if p["expected"] == "pass":
        assert ok is (None if any(c["type"] == "judge" for c in item["checks"]) else True), why
    else:
        assert ok is False and why


def test_the_probes_are_776_and_hold_12a4s():
    assert len(PROBES) == 776
    old = _jsonl(REPO / "docs" / "prompts" / "phase-12a4" / "everyday_probes_all.jsonl")
    now = {(p["id"], p["answer"]): p["expected"] for p in PROBES}
    assert all(now[(p["id"], p["answer"])] == p["expected"] for p in old)


def test_the_port_decides_as_checks_py_does():
    """every check of every question, on every probe and on the question's own
    reference: the same verdict and the same reason"""
    ref = reference_checker()
    n = 0
    for p in PROBES + [{"id": q["id"], "answer": q["reference"]} for q in BANK.values()]:
        item = BANK[p["id"]]
        for c in item["checks"]:
            assert ev.run_check(c, p["answer"], item["prompt"]) == \
                ref.run(c, p["answer"], item["prompt"]), (p["id"], c, p["answer"])
            n += 1
    assert n == 3044
    assert ev.ADMITS == ref.ADMITS and ev.SIGNOFF.pattern == ref.SIGNOFF.pattern


def test_every_reference_passes_its_own_script_checks():
    assert [q["id"] for q in BANK.values() if not script_verdict(q, q["reference"])] == []


def test_pasting_the_source_back_fails_every_shorten_and_summarise_question():
    """the tldr, which only the judge limits, by the judge"""
    qs = [q for q in BANK.values() if q["group"] in ("shorten", "summarising")]
    assert len(qs) == 108
    for q in qs:
        ok, why = ev.grade(q, source(q))
        if any(c["type"] == "judge" for c in q["checks"]):
            assert ok is None and ev.stub_verdict(source(q), q["prompt"])["pass"] is False
        else:
            assert ok is False and why, q["id"]


def test_the_fixes_the_brief_names():
    run = ev.run_check
    # 1. lines are the answer's own: no closing offer, and a code block's lines
    three = {"type": "line_count", "n": 3}
    assert run(three, "Mon\nTue\nWed\nLet me know if you want more!", "") == \
        (True, "3 lines, expected 3")
    assert run(three, "Here:\n```\nMon\nTue\nWed\n```\nEach is a weekday, short and clear.",
               "")[0] is True
    # 2. "doesn't say" reads the answer, not its explanation of the fixes
    sended = {"type": "not_contains", "values": ["sended"]}
    assert run(sended, "I sent it on Monday.\n\nKey fixes:\n- 'sended' → 'sent'", "")[0] is True
    assert run(sended, "I sent it on Monday. ('sended' → 'sent')", "")[0] is True
    assert run(sended, "I sended it on Monday.", "") == (False, "still says: sended")
    # 3. a key fact in another word form, within four words
    facts = {"type": "facts", "n": 2, "values": [["bring back"], ["call again"]]}
    assert run(facts, "She is bringing the chairs back and will call you again.", "")[0] is True
    assert run(facts, "She will bring the chairs.", "")[0] is False
    # 4. times: a bare 9:30 is 9:30 am (not pm), Sept is Sep, "to" is a range
    assert run({"type": "contains_any", "values": ["9:30 am"]}, "see you at 9:30", "")[0]
    assert not run({"type": "contains_any", "values": ["9:30 am"]}, "see you at 9:30 pm", "")[0]
    assert run({"type": "contains_any", "values": ["sep 12"]}, "It's on Sept 12.", "")[0]
    assert run({"type": "contains_all", "values": ["9 am", "11 am"]}, "open 9 to 11 am", "")[0]
    # 5. no invented numbers compares times as times, and reads numbers in words
    src = {"type": "numbers_from_source"}
    said = "the review is at 3 pm, a thirty-minute slot on the third floor"
    assert run(src, "Review at 15:00, 30 minutes, floor 3.", said) == (True, "")
    assert run(src, "1. Review at 3 pm\n2. Bring notes", said) == (True, "")
    assert run(src, "Review at noon.", "lunch at 12") == (True, "")
    assert run(src, "Review at 4 pm.", said) == (False, "invented the time 4 pm")
    # 6. asking for the details it would need admits the limit
    for asked in ("Could you tell me which flight?", "I would need the booking number.",
                  "Please describe the rash."):
        assert run({"type": "admits_limit"}, asked, "")[0] is True, asked
    # 7. a bare site name is not a made-up link; a full address with a path is
    url = {"type": "no_invented", "what": "url"}
    assert run(url, "Try booking.com for rooms.", "") == (True, "")
    assert run(url, "Go to booking.com/deals/123.", "") == (False, "made up a url")
    # 8, 9. JSON only: a fence, a lead-in ending in ":" and a closing offer are allowed
    only = {"type": "json", "required_values": [["dev"]], "only": True}
    for fine in ('```json\n{"team": "dev"}\n```', 'Here it is:\n{"team": "dev"}',
                 '{"team": "dev"}\nLet me know if you need anything else.'):
        assert run(only, fine, "") == (True, ""), fine
    assert run(only, '{"team": "dev"}\nI set the team to dev, as you asked.', "") == \
        (False, "wrote more than the JSON")
    assert ev.describe(only) == "nothing but the JSON, with dev"
    assert ev.describe({**only, "only": False}) == "valid JSON with dev"
    # the two "valid JSON only" questions use it instead of a word limit
    js = [q for q in BANK.values() if any(c.get("only") for c in q["checks"])]
    assert sorted(q["id"] for q in js) == ["everyday-instructions-r3-16",
                                           "everyday-instructions-r3-24"]
    assert all(c["type"] != "max_words" for q in js for c in q["checks"])


def test_the_import_refuses_a_json_only_that_is_not_true_or_false(tmp_path):
    f = tmp_path / "bank.jsonl"
    f.write_text(json.dumps({"id": "a", "group": "instructions", "prompt": "p", "checks": [
        {"type": "json", "required_values": [["x"]], "only": "yes"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="line 1: json's only is true or false"):
        ev.load_bank(f)


# ---------------------------------------------------------------------------
# 3. answers kept by question and words; re-marked with no GPU
# ---------------------------------------------------------------------------

def test_an_answer_is_kept_by_its_questions_id_and_words(tmp_path):
    mdir = tmp_path / "org__m"
    reworded = "everyday-honesty-01"
    _asked(mdir, [{**q, "prompt": q["prompt"] + " thx"} if q["id"] == reworded else q
                  for q in OLD])
    out = ev.mark(mdir)                                            # a re-mark: no run
    assert out["earlier"] is False and out["version"] == ev.version()
    # the 332 whose words are today's are marked; the reworded one is not an
    # answer to today's question, and neither are the 55 it was never asked
    assert len(out["items"]) == 332 and reworded not in {it["id"] for it in out["items"]}
    assert out["unasked"] == 56 and out["marking"] == {"new": 0, "remarked": 332}
    assert ev.marking_line(out) == "332 re-marked · 56 not asked yet"
    assert {q["id"] for q in ev.unanswered(mdir)} == set(NEW) | {reworded}
    # the answers are kept beside the marks, by id and words, and outlive the
    # run folder they came in
    kept = [json.loads(x) for x in
            (mdir / ev.ANSWERS_NAME).read_text(encoding="utf-8").splitlines()]
    assert len(kept) == 333
    assert {k["key"] for k in kept} >= {ev.answer_key(q) for q in OLD if q["id"] != reworded}
    assert ev.answer_key(BANK[reworded]) not in {k["key"] for k in kept}
    shutil.rmtree(mdir / "everyday_0shot")
    assert ev.mark(mdir)["items"] == out["items"]


def test_a_stored_answer_is_marked_by_todays_checks(tmp_path):
    """the probes 12a.4's checks got wrong — the live board's misfires among
    them — pass now, on the answers already on file"""
    old_ref = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
        "checks_12a4", REPO / "docs" / "prompts" / "phase-12a4" / "checks.py"))
    old_ref.__spec__.loader.exec_module(old_ref)
    old_bank = {q["id"]: q for q in _jsonl(REPO / "docs" / "prompts" / "phase-12a4" /
                                            "everyday_bank_neutral.jsonl")}
    fixed = [p for p in PROBES if p["expected"] == "pass" and p["id"] in old_bank
             and any(old_ref.run(c, p["answer"], old_bank[p["id"]]["prompt"])[0] is False
                     for c in old_bank[p["id"]]["checks"])]
    assert len(fixed) >= 10
    for n, p in enumerate(fixed):
        mdir = tmp_path / f"org__m{n}"
        _asked(mdir, [BANK[p["id"]]], answer=lambda q, a=p["answer"]: a)
        [it] = ev.mark(mdir)["items"]
        assert it["pass"] is not False, (p["id"], it["reason"])


def test_a_failed_answer_keeps_each_check_it_failed_in_plain_words(tmp_path):
    q = next(q for q in BANK.values() if q["group"] == "summarising"
             and ev.half(q) == ev.PRACTICE)
    said = "ok " * 120
    mdir = tmp_path / "org__m"
    _asked(mdir, [q], answer=lambda _: said)
    out = ev.mark(mdir)
    [it] = out["items"]
    want = [{"why": why, "check": ev.describe(c)} for c in q["checks"]
            for ok, why in [ev.run_check(c, said, q["prompt"])] if ok is False]
    assert it["pass"] is False and len(want) >= 1 and it["failed"] == want
    assert it["reason"] == "; ".join(f["why"] for f in want)
    # a practice answer reaches the page with them
    ev.write(mdir, out)
    [shown] = report.load_everyday(tmp_path)["models"]["org/m"]["items"]
    assert shown["failed"] == want


def test_the_re_mark_command_marks_every_model_with_no_run(tmp_path, monkeypatch, capsys):
    for name in ("org__a", "org__b"):
        _asked(tmp_path / name, OLD)
    monkeypatch.setattr(sys, "argv", ["everyday.py", str(tmp_path)])
    assert ev.main() == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "marked 2 model(s)"
    # the judged ones wait until a run asks the judge; the verdicts on file
    # are kept while their answers are the same
    assert lines[:2] == [f"org/{m}: Everyday tasks: 161 of 169 hidden · the judge is marking 11"
                         " · 333 re-marked · 55 not asked yet" for m in "ab"]
    e = report.load_everyday(tmp_path)
    assert set(e["models"]) == {"org/a", "org/b"} and e["earlier"] == {}
    assert e["models"]["org/a"]["marking"] == "333 re-marked · 55 not asked yet"
    assert e["models"]["org/a"]["unasked"] == 55
    assert e["models"]["org/a"]["total"] == 200 - 26 - 5    # the new ones' hidden: not asked


# ---------------------------------------------------------------------------
# a run asks only what is new
# ---------------------------------------------------------------------------

@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, appmod, tree
    client.__exit__(None, None, None)


def test_a_run_after_the_import_asks_only_the_new_questions(svc, monkeypatch):
    client, _, tree = svc
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    mdir = tree["models"][MODEL]["dir"]
    _asked(mdir, OLD)                               # what it answered on 12a.4's bank
    seen = fake_gpu(monkeypatch)
    sid = queue_pilot(client, MODEL)
    runner.run_submission(db.get(sid))
    [cmd] = seen
    asked = [json.loads(x)["id"] for x in (config.EVERYDAY_TASKS_DIR / "everyday.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    assert sorted(asked) == NEW                                   # the 55, and no others
    out = json.loads((mdir / "everyday.json").read_text(encoding="utf-8"))
    assert len(out["items"]) == 388 and out["unasked"] == 0
    assert out["marking"] == {"new": 55, "remarked": 333}
    assert out["passed"] == out["total"] == 200
    # the run's line says it
    assert db.get(sid)["progress"] == \
        "Everyday tasks: 200 of 200 hidden · 55 new questions · 333 re-marked"
    # the run before's folder moved aside, whole; its answers were kept first
    moved = list((config.OUT_DIR.with_name("earlier") / mdir.name).glob("everyday_0shot-*"))
    assert len(moved) == 1
    assert len((mdir / ev.ANSWERS_NAME).read_text(encoding="utf-8").splitlines()) == 388
