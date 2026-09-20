"""What the image carries, checked without building it.

The image that shipped on 2026-09-20 had no `eval_tasks/fr/`, so the first
person to press "Rebuild the harness tasks" got a 500: `.dockerignore`
excluded `eval_tasks/*` and the Dockerfile copied only `mmlu_perm/`. Both
demos had passed, because the demo runs from the bind-mounted checkout.

This walks the Dockerfile's COPY list through .dockerignore's rules and
asserts that every path the service reads at run time survives both.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest

from service import startup

REPO = Path(__file__).resolve().parents[1]


def ignore_rules() -> list[tuple[str, bool]]:
    """(pattern, is_exception) in file order — later rules win, which is how
    docker reads them."""
    out = []
    for line in (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append((s[1:], True) if s.startswith("!") else (s, False))
    return out


def ignored(path: str, rules) -> bool:
    """Docker's rule: the LAST pattern that matches decides; a pattern
    matches a path or any directory prefix of it.

    fnmatch's `*` crosses `/` and docker's does not, so this model excludes
    at least as much as docker would. That is the safe direction: a path this
    says the image HAS is a path docker also keeps."""
    verdict = False
    parts = path.split("/")
    prefixes = ["/".join(parts[:i + 1]) for i in range(len(parts))]
    for pat, is_exception in rules:
        pat = pat.rstrip("/")
        if any(fnmatch.fnmatch(p, pat) for p in prefixes):
            verdict = not is_exception
    return verdict


def copied_paths() -> list[str]:
    """The source side of every COPY in the Dockerfile."""
    out = []
    for line in (REPO / "Dockerfile").read_text(encoding="utf-8").splitlines():
        m = re.match(r"COPY\s+(?!--)(\S+)\s+(\S+)\s*$", line.strip())
        if m:
            out.append(m.group(1).rstrip("/"))
    return out


def in_image(path: str) -> bool:
    """Would this repo path be in the built image?"""
    rules = ignore_rules()
    if ignored(path, rules):
        return False
    return any(path == src or path.startswith(src + "/") for src in copied_paths())


def test_every_file_the_service_reads_is_in_the_image():
    missing = [p for p in startup.REQUIRED_REPO_FILES if not in_image(p)]
    assert missing == [], (
        f"the Dockerfile would ship without {missing}. Add the COPY line and, if "
        f".dockerignore excludes the directory, the '!' exception beside it.")
    for d in startup.REQUIRED_REPO_DIRS:
        assert in_image(d + "/anything.md") or in_image(d + "/anything.json"), d


def test_the_file_that_broke_the_live_build_would_now_ship():
    """The exact path from the traceback."""
    assert in_image("eval_tasks/fr/_fr_template_yaml")
    assert in_image("eval_tasks/fr/rubrics/exam.md")          # *.md is excluded by default
    assert in_image("eval_tasks/fr/rubrics/law.criteria.json")
    assert in_image("eval_tasks/fr/canary.jsonl")
    # and the things that should still NOT be in it
    assert not in_image("tests/test_report.py")
    assert not in_image("results/full/whatever.json")
    assert not in_image("DEMO.md") and not in_image(".env")


def test_the_list_is_true_of_this_checkout():
    """A path on the list that is not in the repo is a typo in the list."""
    assert startup.missing_repo_files() == []


def test_the_service_refuses_to_start_without_one(tmp_path):
    (tmp_path / "scripts").mkdir()
    with pytest.raises(SystemExit) as e:
        startup.check_repo_files(tmp_path)
    said = str(e.value)
    assert "cannot serve requests" in said
    assert "eval_tasks/fr/_fr_template_yaml" in said and "categories.yaml" in said
    assert "Dockerfile" in said or "COPY" in said            # what to do about it
