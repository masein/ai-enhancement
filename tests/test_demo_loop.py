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
    return subprocess.run([sys.executable, str(SCRIPT), "--topics", "economics",
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
    assert eb.half_of(eb.load_bank(root / "exam")["economics"][0]["qid"]) in ("report", "diagnose")


def test_nothing_lands_outside_the_demo_root(demo):
    out, root = demo
    bench = root.parent
    assert sorted(p.name for p in bench.iterdir()) == ["demo"]      # not results/, not exam/
    assert (root / "service.sqlite3").exists()                      # its own database
    assert sorted(p.name for p in (root / "exam").iterdir()) == ["bank", "candidates", "tasks"]


def test_the_tree_is_removed_unless_kept(tmp_path):
    root = tmp_path / "bench"
    r = run(root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "removed" in r.stdout and not (root / "demo").exists()


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


MEDICINE = REPO / "eval_tasks" / "fr" / "hossein_medicine_v1.json"


def test_an_imported_bank_replaces_drafting_and_curation(tmp_path):
    """The medicine run: a human-written bank, its author as the approver,
    and no exam writer configured at all — nobody is drafting anything."""
    root = tmp_path / "bench"
    r = run(root, "--keep", "--import", str(MEDICINE), "--approver", "Dr. Hossein",
            "--topic", "medicine & health", EXAM_PROVIDER="", EXAM_MODEL="")
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert "Import the exam — a human-written bank, not an LLM's drafts" in out
    assert "Draft the exam" not in out and "NOT curation" not in out
    assert re.search(r"imported\s+50 items", out)
    assert re.search(r"split by qid\s+report \d+ / diagnose \d+", out)
    assert re.search(r"acuity\s+emergency 3,", out)
    assert "a real run needs 30 before this topic" in out
    assert "the ask back to the author is at least" in out.lower()
    # the two stamps, one from the judge and one from the rubric
    assert "CAVEAT draft rubric: medicine & health is graded against medicine_health.md" in out
    assert "does not look instruction-tuned" in out          # the default model is a base model
    assert "PASSED: none of them appear" in out              # the split holds for an import too
    j = json.loads((root / "demo" / "results" / "full" / "EleutherAI__pythia-160m"
                    / "judge.json").read_text())
    rub = j["judge"]["rubrics"]["exam_medicine_health"]
    assert rub["name"] == "medicine_health" and rub["status"] == "draft"
    assert j["judge"]["rubric_status"] == "draft"
    bank = [json.loads(x) for x in (root / "demo" / "exam" / "bank" / "medicine_health.jsonl")
            .read_text().splitlines() if x.strip()]
    assert len(bank) == 50 and all(b["accepted_by"] == "Dr. Hossein" for b in bank)
    assert all(b["source"] == "hossein_medicine_v1" for b in bank)   # the file's stem
    # the judge graded it criterion by criterion, and the demo shows what that is
    assert "is graded criterion by criterion, and the 0-4 above is a fold of them" in out
    assert re.search(r"medicine_health\.criteria\.json \([0-9a-f]{12}, draft\)", out)
    assert re.search(r"Triage / urgency\s+[01]\.\d\d\s+\d+", out)
    assert re.search(r"critical safety failures: \d+ of 50 answers", out)
    assert re.search(r"emergency\s+\d\.\d\d\s+3\s+\d", out)          # the by-acuity table
    assert "One graded answer in full — the diagnosis half, so it may be shown:" in out
    assert "folded from" in out and "medication_safety" in out
    # and the spec request carried the weakest criteria, as labels and numbers
    assert "three weakest criteria and the per-acuity means — labels and numbers only" in out
    t = j["tasks"]["exam_medicine_health"]
    assert t["criteria_mean"] and t["by_acuity"] and t["unparseable"] == 0
    assert all(it.get("fold") for it in t["items"])


def test_the_import_flags_are_checked_before_anything_runs(tmp_path):
    root = tmp_path / "bench"
    r = run(root, "--import", str(MEDICINE), "--topic", "medicine & health")
    assert r.returncode == 2 and "--import needs --approver" in r.stdout
    r = run(root, "--import", str(MEDICINE), "--approver", "X", "--topics", "law,economics")
    assert r.returncode == 2 and "--import takes one topic, not 2" in r.stdout
    r = run(root, "--import", str(tmp_path / "nope.json"), "--approver", "X",
            "--topic", "medicine & health")
    assert r.returncode == 2 and "no such file" in r.stdout
    assert not (root / "demo").exists()


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
