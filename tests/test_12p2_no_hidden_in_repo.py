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
store, whole, when it is given the store (HIDDEN_STORE_ROOT)."""

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


def scan(root: Path, want: set[str]) -> dict[str, int]:
    """{file: how many of `want`'s runs it holds} — never the runs"""
    hits = {}
    for f in held(root):
        if f.suffix.lower() in BINARY or f == FINGERPRINTS or not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
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
    assert list(scan(copy, invented_fingerprints())) == [
        "tests/fixtures/everyday_hidden_invented.jsonl"]


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
        try:
            bodies[str(f.relative_to(REPO))] = " " + " ".join(c.normalize(f.read_text(
                encoding="utf-8"))) + " "
        except (UnicodeDecodeError, OSError):
            continue
    hits = {name: n for name, body in bodies.items()
            if (n := sum(1 for w in want if f" {w} " in body))}
    assert hits == {}, f"these files hold a question of the server's store whole: {hits}"
