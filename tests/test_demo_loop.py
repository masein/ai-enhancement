"""Phase 8 P2: the narrated demo, end to end against the `fake` backend.

The script is the thing a person runs to understand what was built, so the
test runs it the way they would — as a subprocess, with its own BENCH_ROOT —
and checks that every step happened, that the safety check it prints is not
vacuous, and that it wrote nothing outside its demo root."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import demo_loop
from conftest import fresh

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "demo_loop.py"
FAKE = {"LLM_PROVIDER": "fake", "LLM_MODEL": "fake-1",
        "EXAM_PROVIDER": "fake", "EXAM_MODEL": "fake-exam",
        "JUDGE_PROVIDER": "fake", "JUDGE_MODEL": "fake-judge-20250101"}


def run(root: Path, *args: str, **env) -> subprocess.CompletedProcess:
    import os
    e = {**os.environ, "BENCH_ROOT": str(root), **FAKE, **env}
    for k in ("LLM_API_KEY", "EXAM_API_KEY", "JUDGE_API_KEY"):
        e.pop(k, None)
    return subprocess.run([sys.executable, str(SCRIPT), "--topics", "Economics",
                           "--sit", "stub", "--per-topic", "8", "--count", "4", *args],
                          capture_output=True, text=True, timeout=300, env=e, cwd=REPO)


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> tuple[str, Path]:
    """One full run, kept on disk, shared by the tests that read it."""
    root = tmp_path_factory.mktemp("bench")
    r = run(root, "--keep")
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout, root / "demo"


def test_every_step_runs_in_order(demo):
    out, _ = demo
    heads = [m.group(1) for m in (re.match(r"^\d\. (.+)$", x) for x in out.splitlines()) if m]
    assert [h.split(" —")[0] for h in heads] == [
        "Preflight", "Draft the exam", "Accept", "Build the harness tasks from the bank",
        "Sit the exam", "Judge", "Pick the weakest topic, and propose a skill spec from the "
        "judge's own words", "Generate the dataset", "Summary"]


def test_it_says_what_the_demo_is_not(demo):
    out, _ = demo
    assert "That is NOT curation" in out and "approver 'demo'" in out
    # the floor, said before the step that would be refused and again at it
    assert "A real run needs 30 report-half questions" in out
    assert "The live service would REFUSE this proposal here" in out
    assert "the Anthropic and OpenAI batch clients" in out
    assert "--sit stub: writing answers WITHOUT running the model" in out


def test_the_loop_produced_a_score_a_spec_and_a_document(demo):
    out, root = demo
    assert "the report half is the score" in out and "/4" in out
    assert "canary: 30 of 30 re-graded" in out
    # nothing to compare against on a first run, and it says that rather than
    # printing "None from the previous run"
    assert "first run for this judge — no previous canary to compare" in out
    assert "None from the previous run" not in out
    assert "The spec that came back" in out
    assert "One document in full — prose, not a question-and-answer pair:" in out
    j = json.loads((root / "results" / "full" / "EleutherAI__pythia-160m" / "judge.json").read_text())
    assert j["tasks"]["exam_economics"]["score_report"] is not None
    items = [json.loads(x) for x in
             (root / "datasets" / "1" / "items.jsonl").read_text().splitlines() if x.strip()]
    assert len(items) == 4 and all(set(it) == {"title", "text"} for it in items)
    prov = json.loads((root / "datasets" / "1" / "provenance.json").read_text())
    assert prov["approver"] == "demo" and prov["items"]["kept"] == 4
    assert prov["split"] == "diagnose"


def test_the_safety_check_passed_and_is_not_vacuous(demo):
    out, root = demo
    assert "SAFETY CHECK — no exam question text in the request" in out
    assert "PASSED: none of them appear" in out
    # the same checker, given a body that DOES quote a bank question, catches it
    import exam_build as eb
    questions, halves = demo_loop.bank_questions(root / "exam")
    assert halves["report"] and halves["diagnose"] and len(questions) == 8
    assert demo_loop.leaked(["nothing of the sort here"], questions) == []
    planted = f"Here is the judge's note. {questions[0]} And that is what went wrong."
    assert demo_loop.leaked([planted], questions) == [questions[0]]
    assert eb.half_of(eb.load_bank(root / "exam")["Economics"][0]["qid"]) in ("report", "diagnose")


def test_nothing_lands_outside_the_demo_root(demo):
    out, root = demo
    bench = root.parent
    assert sorted(p.name for p in bench.iterdir()) == ["demo"]      # not results/, not exam/
    assert (root / "service.sqlite3").exists()                      # its own database
    assert sorted(p.name for p in (root / "exam").iterdir()) == ["bank", "candidates", "tasks"]


def test_the_tree_is_removed_unless_kept_but_the_page_stays(tmp_path):
    root = tmp_path / "bench"
    r = run(root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "removed" in r.stdout
    # the numbers were a demo and go; the page that says they were a demo stays
    assert sorted(p.name for p in (root / "demo").iterdir()) == ["report.html"]
    assert "DEMO RUN" in (root / "demo" / "report.html").read_text(encoding="utf-8")
    assert "the page outlives the run it describes" in r.stdout.replace("\n   ", " ")


def test_dry_run_calls_nothing_and_writes_nothing(tmp_path):
    root = tmp_path / "bench"
    r = run(root, "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Dry run — the plan, in order. Nothing is called." in r.stdout
    assert "the spec becomes 4 training documents" in r.stdout
    assert not (root / "demo").exists()


def test_against_a_local_server_everything_it_made_is_stamped(tmp_path):
    """The configuration the demo exists for: all three roles on `local`,
    against a stub of vLLM. Never the real one — CI has no GPU."""
    import vllm_stub
    stub, srv = vllm_stub.serve()
    try:
        root = tmp_path / "bench"
        r = run(root, "--keep", LLM_PROVIDER="local", LLM_MODEL="chat",
                EXAM_PROVIDER="local", EXAM_MODEL="chat",
                JUDGE_PROVIDER="local", JUDGE_MODEL="chat",
                ALLOW_SINGLE_PROVIDER_LOOP="1", LOCAL_BASE_URL=stub.url)
        assert r.returncode == 0, r.stdout + r.stderr
        out = r.stdout
        assert f"serves ['chat']   PROVISIONAL — chat at {stub.url}" in out
        assert f"(weights {vllm_stub.WEIGHTS})" in out
        assert "CAVEAT single-provider loop" in out and "CAVEAT provisional" in out
        # every artefact says what made it
        assert "stamp: provisional — drafted by a local model — not a pinned benchmark" in out
        assert "stamp: PROVISIONAL — graded by a local model — not a pinned benchmark" in out
        assert "stamp: PROVISIONAL — proposed by a local model — not a pinned benchmark" in out
        assert "stamp: PROVISIONAL — a local model was the generator" in out
        assert "never ranked, never in any average" in out
        # a local model is asked for one at a time, drafting and generating
        # alike, and the batches it ran are inside the demo tree
        assert "1 to a request (a local model answers one at a time)" in out
        assert "4 documents, 1 per request" in out
        assert (root / "demo" / "llm_batches" / "local").is_dir()
        assert "PASSED: none of them appear" in out
        j = json.loads((root / "demo" / "results" / "full" / "EleutherAI__pythia-160m"
                        / "judge.json").read_text())
        assert j["judge"]["provisional"] is True and j["judge"]["weights"] == vllm_stub.WEIGHTS
        prov = json.loads((root / "demo" / "datasets" / "1" / "provenance.json").read_text())
        assert prov["provisional"] is True and set(prov["local_models"]) == {
            "generator", "exam writer", "judge"}
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_model_answering_in_an_uncurable_shape_says_how_many(tmp_path, monkeypatch):
    """The failure a real run hit: three replies, none of them a question,
    and the demo said only "no candidate questions came back"."""
    import vllm_stub
    monkeypatch.setattr(vllm_stub, "DRAFT_SHAPE", "strings")
    stub, srv = vllm_stub.serve()
    try:
        root = tmp_path / "bench"
        r = run(root, "--keep", LLM_PROVIDER="local", LLM_MODEL="chat",
                EXAM_PROVIDER="local", EXAM_MODEL="chat",
                JUDGE_PROVIDER="local", JUDGE_MODEL="chat",
                ALLOW_SINGLE_PROVIDER_LOOP="1", LOCAL_BASE_URL=stub.url)
        assert r.returncode == 2
        assert re.search(r"replies unread\s+8 of 8 — e\.g\. exam:economics:\d", r.stdout)
        assert "none of them a question with both a prompt and a reference answer" in r.stdout
        assert "no candidate questions came back: 8 of 8 replies could not be read" in r.stdout
        assert "read results.jsonl to see what the model actually sent" in r.stdout
        # the replies are where it says they are, and the tree is kept
        assert list((root / "demo" / "llm_batches" / "local").glob("*/results.jsonl"))
    finally:
        srv.shutdown()
        srv.server_close()


# the delivered Medicine & Clinical Health bank of the 37-topic exam, the
# run DEMO.md gives as its example
MEDICINE = REPO / "eval_tasks" / "fr" / "banks" / "medicine_clinical_health_v1.json"
MEDICINE_TOPIC = "Medicine & Clinical Health"


def test_an_imported_bank_replaces_drafting_and_curation(tmp_path):
    """The medicine run: a human-written bank, its author as the approver,
    and no exam writer configured at all — nobody is drafting anything."""
    import judge as jd
    root = tmp_path / "bench"
    # no rubric of the 37-topic exam is marked DRAFT, so the rubric stamp is
    # shown on a DRAFT copy of this topic's own, installed where the page's
    # rubric upload puts one — $BENCH_ROOT/rubrics, which the judge reads
    # first, and which inside the demo is the demo tree's. The criteria file
    # is not copied: the repo's, signed off, is read beside it.
    rubric = jd.RUBRIC_DIR / "medicine_clinical_health.md"
    head, rest = rubric.read_text(encoding="utf-8").split("\n", 1)
    (root / "demo" / "rubrics").mkdir(parents=True)
    (root / "demo" / "rubrics" / "medicine_clinical_health.md").write_text(
        f"{head} (version 1, DRAFT — awaiting sign-off)\n{rest}", encoding="utf-8")
    r = run(root, "--keep", "--import", str(MEDICINE), "--approver", "masein",
            "--topic", MEDICINE_TOPIC, EXAM_PROVIDER="", EXAM_MODEL="")
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert "Import the exam — a human-written bank, not an LLM's drafts" in out
    assert "Draft the exam" not in out and "NOT curation" not in out
    assert re.search(r"imported\s+100 items", out)
    assert re.search(r"split by qid\s+report \d+ / diagnose \d+", out)
    assert re.search(r"acuity\s+emergency 28,", out)
    # 100 questions clears the floor, so the step says so instead of asking
    assert "at or above the 30 this topic needs" in out
    assert "The floor is cleared." in out
    assert "the ask back to the author is at least" not in out.lower()
    # the evidence shown is a sample, and says so rather than looking like all of it
    assert re.search(r"in a person's vocabulary \(3 of \d+ shown\):", out)
    # the next step after a demo of an imported bank is one paste
    assert "This bank is in the demo's exam, not the live one." in out
    assert "exam_build.py --root" in out and "--approver 'masein'" in out
    assert "--topic 'Medicine & Clinical Health'" in out     # quoted: it has spaces and an &
    # the two stamps, one from the judge and one from the rubric
    assert ("CAVEAT draft rubric: Medicine & Clinical Health is graded against "
            "medicine_clinical_health.md") in out
    assert "does not look instruction-tuned" in out          # the default model is a base model
    assert "PASSED: none of them appear" in out              # the split holds for an import too
    j = json.loads((root / "demo" / "results" / "full" / "EleutherAI__pythia-160m"
                    / "judge.json").read_text())
    rub = j["judge"]["rubrics"]["exam_medicine_clinical_health"]
    assert rub["name"] == "medicine_clinical_health" and rub["status"] == "draft"
    assert j["judge"]["rubric_status"] == "draft"
    bank = [json.loads(x) for x in
            (root / "demo" / "exam" / "bank" / "medicine_clinical_health.jsonl")
            .read_text().splitlines() if x.strip()]
    assert len(bank) == 100 and all(b["accepted_by"] == "masein" for b in bank)
    assert all(b["source"] == "medicine_clinical_health_v1" for b in bank)   # the file's stem
    # the judge graded it criterion by criterion, and the demo shows what that is
    assert "is graded criterion by criterion, and the 0-4 above is a fold of them" in out
    # the criteria file is the author's own and carries no draft stamp; the
    # prose rubric beside it still does, and the caveat above says which
    assert re.search(r"medicine_clinical_health\.criteria\.json \([0-9a-f]{12}, signed off\)",
                     out)
    assert re.search(r"Triage and urgency\s+[01]\.\d\d\s+\d+", out)
    # one line per flag, in the author's words, with what it does to a score
    assert re.search(r"Critical medicine clinical health error\s+\d+ of 100 answers", out)
    assert "each sets the whole score to 0" in out
    # a table per metadata field that splits the topic, acuity first
    assert re.search(r"emergency\s+\d\.\d\d\s+28\s+\d", out)
    assert re.search(r"difficulty\s+mean\s+answers", out)
    assert "the author's own level, 1 easiest" in out
    assert "One graded answer in full — the diagnosis half, so it may be shown:" in out
    assert "folded from" in out and "triage_and_urgency" in out
    # and the spec request carried the weakest criteria, as labels and numbers
    assert "the means by acuity, difficulty — labels and numbers only" in out
    t = j["tasks"]["exam_medicine_clinical_health"]
    assert t["criteria_mean"] and t["breakdowns"]["acuity"] and t["unparseable"] == 0
    # intent is a sentence per question here, which is not a table; the bank
    # carries jurisdiction_required on every item and never sets it, which
    # is said once instead of drawn
    assert list(t["breakdowns"]) == ["acuity", "difficulty"]
    assert t["breakdowns_constant"] == {"jurisdiction_required": "False"}
    assert "jurisdiction_required is False on every item — no table for that." in out
    assert set(t["flags"]) == {"critical_medicine_clinical_health_error"}
    assert all(it.get("fold") for it in t["items"])


def test_the_import_flags_are_checked_before_anything_runs(tmp_path):
    root = tmp_path / "bench"
    r = run(root, "--import", str(MEDICINE), "--topic", MEDICINE_TOPIC)
    assert r.returncode == 2 and "--import needs --approver" in r.stdout
    r = run(root, "--import", str(MEDICINE), "--approver", "X", "--topics", "Law,Economics")
    assert r.returncode == 2 and "--import takes one topic, not 2" in r.stdout
    r = run(root, "--import", str(tmp_path / "nope.json"), "--approver", "X",
            "--topic", MEDICINE_TOPIC)
    assert r.returncode == 2 and "no such file" in r.stdout
    assert not (root / "demo").exists()


def test_the_run_ends_with_its_own_page(demo):
    """The demo writes the demo's own dashboard. Without it, "shown greyed on
    the page" is true of a code path nobody can open: the live board reads
    results/full and this ran under demo/."""
    out, root = demo
    page = root / "report.html"
    assert page.is_file() and "THE DEMO'S PAGE" in out and "serves it at /demo" in out
    html = page.read_text(encoding="utf-8")
    assert "DEMO RUN" in html and "nothing on this page is on the leaderboard" in html
    assert 'class="pagebanner"' in html and ">the live dashboard</a>" in html
    assert "<title>DEMO RUN" in html
    # it is the report of the DEMO tree: the model that sat this exam is on it
    assert "EleutherAI__pythia-160m" in html or "pythia-160m" in html
    assert "exam_economics" in html


def test_the_service_serves_the_demo_page_apart_from_the_board(tmp_path, monkeypatch):
    from conftest import make_service
    from service import config
    client, appmod, _ = make_service(tmp_path, monkeypatch)
    try:
        # nothing has run yet: a page that says so, and how to run one
        r = client.get("/demo")
        assert r.status_code == 404
        assert "No demo run yet" in r.text and "demo_loop.py" in r.text
        assert "DEMO RUN" not in client.get("/").text        # the board is the board
        fresh(appmod)
        assert client.get("/api/results").json()["demo"] is None
        # a run leaves a page behind, and the board grows one link to it
        page = config.BENCH_ROOT / "demo" / "report.html"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text("<html><body>DEMO RUN — a page</body></html>", encoding="utf-8")
        assert "DEMO RUN — a page" in client.get("/demo").text
        fresh(appmod)
        demo = client.get("/api/results").json()["demo"]
        assert demo["href"] == "/demo" and demo["at"] > 0
        # ...without a restart: the file is part of the payload's cache key
        assert appmod.demo_report_stamp() > 0
        assert appmod.demo_report_stamp() in appmod._tree_key()
    finally:
        client.__exit__(None, None, None)


def test_the_live_payload_never_opens_the_demo_tree(tmp_path, monkeypatch):
    """Two trees, two pages. The live board must not read a demo number even
    by accident, so watch every path it opens while it builds."""
    import builtins
    from conftest import make_service
    from service import config
    client, appmod, _ = make_service(tmp_path, monkeypatch)
    try:
        demo = config.BENCH_ROOT / "demo"
        (demo / "results" / "full" / "fx__demo-model").mkdir(parents=True)
        (demo / "results" / "full" / "fx__demo-model" / "results_x.json").write_text(
            json.dumps({"results": {"mmlu": {"acc,none": 0.99}}, "config": {
                "model_args": "pretrained=fx/demo-model"}}), encoding="utf-8")
        (demo / "report.html").write_text("DEMO RUN", encoding="utf-8")
        opened: list[str] = []
        real = builtins.open

        def watched(file, *a, **kw):
            opened.append(str(file))
            return real(file, *a, **kw)
        monkeypatch.setattr(builtins, "open", watched)
        monkeypatch.setattr(Path, "read_text", lambda self, *a, **kw: (
            opened.append(str(self)) or real(self, encoding=kw.get("encoding")).read()))
        fresh(appmod)
        payload = client.get("/api/results").json()
        assert not [p for p in opened if "/demo/" in p], \
            [p for p in opened if "/demo/" in p][:3]
        assert not any(m["id"] == "fx/demo-model" for m in payload["models"])
    finally:
        client.__exit__(None, None, None)


def test_demo_md_says_what_a_green_run_does_not_prove():
    doc = (REPO / "DEMO.md").read_text(encoding="utf-8")
    # the sentence the brief asks for, in those words — a green demo must not
    # be mistaken for a green production path
    assert ("It does **not** exercise the\nAnthropic or OpenAI batch clients, which remain the "
            "untested glue until real\nkeys exist." in doc)
    assert "a green demo is not a green production path" in doc
    assert "ssh -L 8000:localhost:8000" in doc and "ALLOW_SINGLE_PROVIDER_LOOP=1" in doc
    assert "python3 scripts/demo_loop.py" in doc
    for other in ("README.md", "SERVICE.md"):
        assert "DEMO.md" in (REPO / other).read_text(encoding="utf-8"), other
    # the medicine run, and the two things a criteria-graded score is not
    assert ("--import eval_tasks/fr/banks/medicine_clinical_health_v1.json --approver masein"
            in doc)
    # and Law, the topic that adds jurisdiction, as its own run
    assert "--import eval_tasks/fr/banks/law_v1.json --approver masein" in doc
    assert "HuggingFaceTB/SmolLM2-360M-Instruct" in doc
    assert "A rubric whose heading says DRAFT is marked so" in doc
    assert "until its author signs it off" in doc
    assert "deterministic fold of that topic's\n  criteria" in doc
    assert "Per-criterion agreement with a human has **not** been measured" in doc
    # the rule applied to this repo's own numbers: the retired topics' scores
    # are about other questions and other criteria, whatever the task is called
    assert "Scores on the five retired topics do not compare to the new ones" in doc
    assert "even where the task name is the same" in doc


def test_the_author_docs_carry_the_import_and_criteria_rules():
    doc = (REPO / "eval_tasks" / "fr" / "AUTHORING.md").read_text(encoding="utf-8")
    assert "## Importing a human-written bank" in doc
    assert "scripts/exam_build.py --root $BENCH_ROOT/exam import" in doc
    assert "Metadata is the reference" in doc and "Acuity: emergency." in doc
    assert "rubrics/<slug>.md" in doc and "rubrics/exam.md" in doc
    assert "round_half_up(4 × Σ w·c / Σ w)" in doc
    assert "conditional" in doc and "zero_score" in doc and "cap_at_N_of_4" in doc
    # the schema is the author's, and the doc shows it as he writes it
    assert '"flags": [{"id": "critical_legal_error"' in doc
    assert "refuses to load" in doc
    hand = (REPO / "HANDOFF.md").read_text(encoding="utf-8")
    # the first delivery, where it lives now that it is retired, and the
    # file that holds his criteria
    assert "eval_tasks/fr/retired/medicine_v2.json" in hand
    assert "retired/rubrics/medicine_health.criteria.json" in hand
    # the asks back to the author: the floor, the sign-off (moot since the
    # draft rubrics were retired), and the one closed by delivery
    assert "The question floor is cleared." in hand
    assert "Rubric sign-off.** *(moot since phase 10)*" in hand
    assert "jurisdiction_required" in hand and "retired/law_v2.json" in hand
    assert "phase-8b-medicine.md" in hand
    # and the 37-topic exam that replaced it, with the steps that load it
    assert "## 10c. Phase 10 — the 37-topic exam" in hand
    assert "import-dir eval_tasks/fr/banks --approver masein" in hand


def test_an_unknown_topic_stops_before_anything_runs(tmp_path):
    r = run(tmp_path / "bench", "--topics", "astrology")
    assert r.returncode == 2 and "not exam topics: astrology" in r.stdout


def test_a_blocked_identity_stops_at_preflight(tmp_path):
    root = tmp_path / "bench"
    r = run(root, JUDGE_MODEL="")
    assert r.returncode == 2
    assert "One identity is not usable" in r.stdout and "JUDGE_MODEL is unset" in r.stdout
    assert "ALLOW_SINGLE_PROVIDER_LOOP=1" in r.stdout      # what to put in .env for an all-local run
    # a run that stopped early keeps its tree, whatever --keep says: its logs
    # and half-written artefacts are the whole reason to look
    assert "the run stopped early" in r.stdout and (root / "demo").exists()


def test_the_demo_doc_carries_every_delivered_bank():
    """Thirty-six banks now; each is one command, and all of them clear the
    floor. The doc shows two and gives the rule for the rest, so the rule
    has to reach every file that was delivered."""
    import exam_build as eb
    import report_lm_eval as report
    from categories import topic_slug
    doc = (REPO / "DEMO.md").read_text(encoding="utf-8")
    for stem in ("medicine_clinical_health_v1", "law_v1"):
        assert f"--import eval_tasks/fr/banks/{stem}.json" in doc, stem
        assert f"--source {stem}" in doc, stem
    assert "`banks/<slug>_v1.json`" in doc
    assert "36 human-written banks of 100 questions" in doc and "Arts arrived empty" in doc
    banks = sorted((REPO / "eval_tasks" / "fr" / "banks").glob("*.json"))
    assert sorted(b.name for b in banks) == sorted(
        f"{topic_slug(t)}_v1.json" for t in eb.TOPICS if t != "Arts")
    # "every bank clears it": the report half of each, split by qid as the
    # import splits it, is at or above the floor
    assert "the 30-question floor.\nEvery bank clears it" in doc
    for b in banks:
        items = json.loads(b.read_text(encoding="utf-8"))
        report_half = sum(1 for it in items if eb.half_of(eb.qid_of(it["prompt"])) == "report")
        assert len(items) == 100 and report_half >= report.PROPOSE_MIN_N, b.name
    # and what differs between them is the author's file, not our code
    assert "routine` on every item" in doc and "difficulty and domain" in doc


def test_the_schema_doc_is_the_table_of_what_arrives():
    doc = (REPO / "docs" / "CRITERIA-SCHEMA.md").read_text(encoding="utf-8")
    assert "Internal" in doc.split("\n")[2] or "*Internal" in doc
    for variant in ("critical_flag", "critical_error_flag", "evaluation_principles",
                    "not_critical", "do_not_classify_as_critical", "score=0", "cap=1"):
        assert variant in doc, variant
    assert "normalise_criteria" in doc
    assert "refuses to load" in doc
    # the promise this document makes about who changes what
    assert "we do not ask the author to change how" in doc
    assert "the loader learns it" in doc
    hand = (REPO / "HANDOFF.md").read_text(encoding="utf-8")
    assert "CRITERIA-SCHEMA.md" in hand
    assert "we extend the\nloader, never ask him to rewrite a file" in hand
