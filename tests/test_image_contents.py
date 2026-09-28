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


def test_all_111_delivered_files_are_on_the_list_and_in_the_image():
    """The 37-topic exam's 37 banks, criteria files and rubrics — every one,
    and exactly the ones in the checkout, so a 37th file cannot arrive and
    ship without anyone adding it here."""
    files = startup.DELIVERED_TOPIC_FILES
    assert len(files) == 111 == len(set(files))
    assert set(files) <= set(startup.REQUIRED_REPO_FILES)
    on_disk = {f"eval_tasks/fr/banks/{p.name}" for p in (REPO / "eval_tasks/fr/banks").glob("*")}
    on_disk |= {f"eval_tasks/fr/rubrics/{p.name}"
                for s in startup.DELIVERED_TOPIC_SLUGS
                for p in (REPO / "eval_tasks/fr/rubrics").glob(f"{s}.*")}
    assert on_disk == set(files)
    assert [p for p in files if not in_image(p)] == []
    # the retired five stay in the repo as history and out of the image's
    # live paths: nothing the service reads points into retired/
    assert (REPO / "eval_tasks/fr/retired/law_v2.json").is_file()
    assert not any("retired/" in p for p in startup.REQUIRED_REPO_FILES)


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


# ---------------------------------------------------------------------------
# 12k.2's deploy: eval_tasks/trust_safety was read by every full run and was not
# in the image — the list above did not name it, so nothing checked it. The
# second time (after #81). These don't rest on the list: they read the code
# and the repo, so a new eval_tasks folder is in the image or says why not.
# ---------------------------------------------------------------------------

# an eval_tasks folder in the repo that the running service never reads, and why
# (none today: every one is read at run time)
NOT_IN_IMAGE: dict[str, str] = {}
_NAMED = re.compile(r"""eval_tasks["']?\s*/\s*["']?([A-Za-z0-9_\-]+)""")


def eval_folders_the_code_names() -> set[str]:
    """every eval_tasks/<folder> the service or its scripts name — as a path
    ("eval_tasks/fr/…") or joined ("eval_tasks" / "trust_safety")"""
    out = set()
    for f in [*REPO.glob("service/*.py"), *REPO.glob("scripts/*.py")]:
        out |= set(_NAMED.findall(f.read_text(encoding="utf-8")))
    return out


def repo_eval_folders() -> set[str]:
    return {d.name for d in (REPO / "eval_tasks").iterdir() if d.is_dir()}


def test_every_eval_tasks_folder_the_code_reads_is_in_the_image_whole():
    named = eval_folders_the_code_names() & repo_eval_folders()
    # the scan finds them all — 12k.2's among them
    assert {"fr", "everyday", "mmlu_perm", "trust_safety"} <= named
    for name in sorted(named):
        folder = REPO / "eval_tasks" / name
        files = [p.relative_to(REPO).as_posix() for p in folder.rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts and p.suffix != ".pyc"]
        missing = [p for p in files if not in_image(p)]
        assert missing == [], (
            f"eval_tasks/{name} is read at run time and the image would ship without "
            f"{missing[:5]}: add '!eval_tasks/{name}' to .dockerignore and "
            f"'COPY eval_tasks/{name}/ eval_tasks/{name}/' to the Dockerfile")


def test_every_eval_tasks_folder_in_the_repo_ships_or_says_why():
    for name in sorted(repo_eval_folders()):
        assert in_image(f"eval_tasks/{name}/x.jsonl") or name in NOT_IN_IMAGE, (
            f"eval_tasks/{name} is not in the image: copy it, or say in NOT_IN_IMAGE why "
            f"the running service never reads it")
    assert set(NOT_IN_IMAGE) <= repo_eval_folders()


def test_the_folder_that_broke_12k2s_deploy_would_now_ship():
    """the file from the traceback, and the rest of what build_tasks reads"""
    for f in ("_safety_template_yaml", "_bbq_template_yaml", "bbq_utils.py", "manifest.json",
              "do_not_answer.jsonl", "xstest.jsonl", "bbq_ambig_3000.jsonl",
              "bbq_ambig.jsonl.gz"):
        assert f"eval_tasks/trust_safety/{f}" in startup.REQUIRED_REPO_FILES
        assert in_image(f"eval_tasks/trust_safety/{f}"), f
