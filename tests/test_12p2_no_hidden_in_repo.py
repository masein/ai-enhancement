"""12p.2: no question that is the test is in the repo — the mirror is public.

tests/fixtures/protected_fingerprints.txt holds, for Everyday's hidden half
(prompt and reference) and the Knowledge exam's report half (prompt), every
fourth eight-word run of each text, hashed — so any copy of eleven or more
words in a row holds a whole run, and nothing can be read back from them (a
run a question shares with the rest of the repo, a common phrase, was left
out when they were made). Every text file the repo holds is scanned, and a
run found fails the test: it names the file and how many, never the text.
As the GPQA canary test does (12n.2), it walks the tree where there is no git
— deploy step 3's copy — and proves it finds what it should: the tests'
invented hidden set, fingerprinted the same way, in its own file and nowhere
else.

A question under eight words can't be fingerprinted safely: a guess could be
checked against it. Deploy step 3 checks every question of the server's
store, whole, when it is given the store (HIDDEN_STORE_ROOT).

12p.4: the fingerprints left out, as common phrases, every run the repo
held — so a question already whole in the repo had none, and 242 of the exam's
report half (the five retired topics' banks, two docs, a fixture) sat in the
public tree unseen until step 3's whole-question check found them. A question
the repo holds whole is now kept and named when the fingerprints are made
(hidden_store.fingerprints), and a JSON file's strings are read decoded, a
Python file's as written, so no escape hides one."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from service import contamination as c

REPO = Path(__file__).resolve().parents[1]
FINGERPRINTS = REPO / "tests" / "fixtures" / "protected_fingerprints.txt"
INVENTED = REPO / "tests" / "fixtures" / "everyday_hidden_invented.jsonl"
N, STRIDE = 8, 4
# what a walk leaves out: git's own, what a run or a test writes, caches
SKIP = {".git", "results", "__pycache__", "_screens", "node_modules"}
BINARY = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".gz", ".pdf", ".safetensors", ".pyc",
          ".ico", ".woff", ".woff2"}


def fp(run: str) -> str:
    return hashlib.sha256(run.encode("utf-8")).hexdigest()[:16]


def runs(text: str, stride: int = 1) -> list[str]:
    t = c.normalize(text)
    if len(t) < N:
        return []
    every = [" ".join(t[i:i + N]) for i in range(len(t) - N + 1)]
    return every if stride == 1 or len(every) <= 3 else every[::stride] + [every[-1]]


def fingerprints() -> set[str]:
    return {x.strip() for x in FINGERPRINTS.read_text(encoding="utf-8").splitlines()
            if x.strip() and not x.startswith("#")}


def held(root: Path) -> list[Path]:
    """the files the repo holds: git's list where there is git, else a walk"""
    if (root / ".git").exists() and shutil.which("git"):
        out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others",
                              "--exclude-standard"], cwd=root, capture_output=True, check=True)
        names = [n for n in out.stdout.decode().split("\0") if n]
        return [root / n for n in names if not set(Path(n).parts) & SKIP]
    got = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP]
        got += [Path(d) / f for f in files]
    return got


def text_of(f: Path) -> str | None:
    """a file as the guard reads it: its text, and (12p.4) a JSON file's
    strings decoded and a Python file's string literals as written — so
    "don\\u2019t" or a question joined from two literals reads as its words"""
    try:
        text = f.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    extra: list[str] = []
    if f.suffix.lower() in (".json", ".jsonl"):
        docs = []
        try:
            docs = [json.loads(text)]
        except ValueError:
            for line in text.splitlines():
                try:
                    docs.append(json.loads(line))
                except ValueError:
                    continue
        extra = [s for d in docs for s in c.doc_strings(d)]
    elif f.suffix.lower() == ".py":
        import ast
        try:
            extra = [n.value for n in ast.walk(ast.parse(text))
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        except (SyntaxError, ValueError):
            extra = []
    return "\n".join([text, *extra])


def scan(root: Path, want: set[str]) -> dict[str, int]:
    """{file: how many of `want`'s runs it holds} — never the runs"""
    hits = {}
    for f in held(root):
        if f.suffix.lower() in BINARY or f == FINGERPRINTS or not f.is_file():
            continue
        text = text_of(f)
        if text is None:
            continue
        n = len({fp(r) for r in runs(text)} & want)
        if n:
            hits[str(f.relative_to(root))] = n
    return hits


def invented_fingerprints() -> set[str]:
    rows = [json.loads(x) for x in INVENTED.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {fp(r) for q in rows for r in runs(q["prompt"], STRIDE) + runs(q["reference"], STRIDE)}


def test_no_question_that_is_the_test_is_in_the_repo():
    fps = fingerprints()
    assert len(fps) > 20000 and all(len(x) == 16 for x in fps)
    hits = scan(REPO, fps)
    assert hits == {}, ("these files hold words of a question that is the test "
                        f"(runs found, by file): {hits}")


def test_the_scan_finds_a_question_where_one_is(tmp_path):
    fx = invented_fingerprints()
    hits = scan(REPO, fx)
    # every one in the invented set's own file; elsewhere only the tests, and the
    # stand-in judge's notice (scripts/everyday.py's PILOT_NOTICE)
    assert hits.pop("tests/fixtures/everyday_hidden_invented.jsonl") == len(fx)
    assert not [f for f in hits if not f.startswith("tests/") and f != "scripts/everyday.py"], hits
    # eleven words in a row of one, anywhere, are caught
    q = json.loads(INVENTED.read_text(encoding="utf-8").splitlines()[5])
    words = q["prompt"].split()
    assert len(words) >= 11
    (tmp_path / "notes.md").write_text("an aside: " + " ".join(words[:11]) + " — end")
    assert scan(tmp_path, fx).get("notes.md", 0) >= 1


def test_a_copy_with_no_git_is_walked_and_means_the_same(tmp_path):
    """deploy step 3's shape: the commit streamed out with `git archive`"""
    if not ((REPO / ".git").exists() and shutil.which("git")):
        return                                      # this is that copy: the scan above walked it
    tar = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=REPO,
                         capture_output=True, check=True).stdout
    copy = tmp_path / "copy"
    copy.mkdir()
    (tmp_path / "c.tar").write_bytes(tar)
    with tarfile.open(tmp_path / "c.tar") as t:
        t.extractall(copy, filter="data")
    assert not (copy / ".git").exists()
    assert scan(copy, fingerprints()) == {}
    held = scan(copy, invented_fingerprints())
    assert "tests/fixtures/everyday_hidden_invented.jsonl" in held
    assert not [f for f in held if not f.startswith("tests/") and f != "scripts/everyday.py"]


@pytest.mark.skipif(not os.environ.get("HIDDEN_STORE_ROOT"),
                    reason="deploy step 3 runs this, given the server's store (HIDDEN_STORE_ROOT)")
def test_on_the_server_no_question_of_any_length_is_in_the_repo():
    """every hidden and report-half question, whole — the short ones too"""
    import diagnose as dx
    import exam_build as eb
    root = Path(os.environ["HIDDEN_STORE_ROOT"])
    texts = [json.loads(x)["prompt"] for x in (root / "everyday" / "hidden.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    for f in sorted((root / "exam" / "bank").glob("*.jsonl")):
        texts += [r["prompt"] for r in map(json.loads, f.read_text(encoding="utf-8").splitlines())
                  if r.get("prompt") and dx.split_of(eb.qid_of(r["prompt"])) == "report"]
    assert texts
    want = [" ".join(c.normalize(t)) for t in texts if len(c.normalize(t)) >= 4]
    bodies = {}
    for f in held(REPO):
        if f.suffix.lower() in BINARY or f == FINGERPRINTS or not f.is_file():
            continue
        text = text_of(f)
        if text is not None:
            bodies[str(f.relative_to(REPO))] = " " + " ".join(c.normalize(text)) + " "
    hits = {name: n for name, body in bodies.items()
            if (n := sum(1 for w in want if f" {w} " in body))}
    assert hits == {}, f"these files hold a question of the server's store whole: {hits}"


def test_the_server_makes_the_fingerprints_from_its_own_store(tmp_path, monkeypatch, capsys):
    """after a new hidden set (12p.3's rotation), the fingerprints are made
    where the questions are and committed — here, from the tests' invented
    set, over a repo that quotes one of its questions in passing"""
    from service import config, hidden_store
    monkeypatch.setattr(config, "EXAM_DIR", tmp_path / "no-exam")
    repo = tmp_path / "repo"
    repo.mkdir()
    q = json.loads(INVENTED.read_text(encoding="utf-8").splitlines()[5])
    words = q["prompt"].split()
    # a phrase the repo shares with a question: eight words of it, in passing
    (repo / "notes.md").write_text("a phrase: " + " ".join(words[:8]), encoding="utf-8")
    monkeypatch.setattr(hidden_store, "REPO", repo)
    got = hidden_store.fingerprints()
    fx = invented_fingerprints()
    shared = {fp(r) for r in runs(" ".join(words[:8]))} & fx
    # every one of the set's, less the phrase the repo holds: a common phrase
    assert set(got["fingerprints"]) == fx - shared and got["common"] == len(shared) > 0
    assert got["whole"] == []
    out = tmp_path / "fp.txt"
    assert hidden_store.main(["fingerprints", "--out", str(out)]) == 0
    assert out.read_text().splitlines()[1:] == got["fingerprints"]
    assert "commit it as tests/fixtures/protected_fingerprints.txt" in capsys.readouterr().out


def _planted(repo: Path, q: dict) -> None:
    """one question that is the test, whole, three ways the repo held them:
    in a retired bank's JSON (escaped, with a newline), in a doc, and in a
    fixture joined from two literals"""
    words = q["prompt"].split()
    half = len(words) // 2
    (repo / "retired_v1.json").write_text(json.dumps(
        [{"id": 1, "prompt": " ".join(words[:half]) + "\n" + " ".join(words[half:])}]),
        encoding="utf-8")                                   # ensure_ascii: every \u escaped
    (repo / "criteria.md").write_text(f'## Q1\n\n"{q["prompt"]}"\n\n- criterion one\n',
                                      encoding="utf-8")
    (repo / "fixture.py").write_text(
        "ROWS = [{\"prompt\": " + json.dumps(" ".join(words[:half]) + " ") + "\n    "
        + json.dumps(" ".join(words[half:])) + "}]\n", encoding="utf-8")


def test_a_question_the_repo_holds_whole_is_kept_and_named(tmp_path, monkeypatch, capsys):
    """12p.4, the case that went unseen: the question itself in the repo. Its
    fingerprints are kept, not called common, it is named by id, and the guard
    fails on every file that holds it"""
    from service import config, hidden_store
    monkeypatch.setattr(config, "EXAM_DIR", tmp_path / "no-exam")
    repo = tmp_path / "repo"
    repo.mkdir()
    q = json.loads(INVENTED.read_text(encoding="utf-8").splitlines()[5])
    _planted(repo, q)
    monkeypatch.setattr(hidden_store, "REPO", repo)
    got = hidden_store.fingerprints()
    mine = {fp(r) for r in runs(q["prompt"], STRIDE)}
    assert mine <= set(got["fingerprints"])                  # kept: none of it "common"
    [w] = [w for w in got["whole"] if w["id"] == q["id"]]
    assert w["files"] == ["criteria.md", "fixture.py", "retired_v1.json"]
    out = tmp_path / "fp.txt"
    assert hidden_store.main(["fingerprints", "--out", str(out)]) == 1
    said = capsys.readouterr().out
    assert "question(s) that are the test are in the repo whole, in 3 file(s)" in said
    assert q["prompt"] not in said and q["id"][:12] in said        # its id, never its words
    # the guard, given those fingerprints, fails on each file
    hits = scan(repo, set(got["fingerprints"]))
    assert set(hits) == {"criteria.md", "fixture.py", "retired_v1.json"}
    assert all(n >= len(mine) for n in hits.values())


def test_a_planted_question_fails_the_guard_however_it_is_written(tmp_path):
    """escaped in JSON, broken by a newline, joined from two Python literals:
    the guard reads each as its words"""
    q = json.loads(INVENTED.read_text(encoding="utf-8").splitlines()[5])
    _planted(tmp_path, q)
    fx = invented_fingerprints()
    mine = {fp(r) for r in runs(q["prompt"], STRIDE)}
    hits = scan(tmp_path, fx)
    assert {f for f, n in hits.items() if n >= len(mine)} == {
        "criteria.md", "fixture.py", "retired_v1.json"}
