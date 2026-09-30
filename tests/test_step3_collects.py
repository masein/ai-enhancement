"""Deploy step 3 runs the suite inside the image, where the dashboard's and
the checks' own tooling isn't installed (Playwright, ruff, the pytest
plugins CI runs in parallel with). Its `-m "not dashboard"` deselects a
browser test, but only once its module has been imported: a module that
imports Playwright at the top fails collection, and pytest then runs nothing
at all. On the server, after 12q.C, none of the 711 tests ran.

- Every test module is collected here with those packages made unimportable,
  as they are in the image: step 3's own command, collection only.
- Every distribution the dev and CI requirement files name is sorted into "the
  image has it" or "the image doesn't", so a new one can't go unclassified;
  tests/test_image_deps.py imports the first kind inside the image.

A browser test takes what it needs of Playwright from the `page` fixture, or
imports it inside the function that uses it, as tests/conftest.py does."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# step 3's marker expression (HANDOFF §5b, and every PR's deploy section)
STEP3 = "not gpu and not network and not dashboard"
# what requirements-dev.txt and requirements-ci.txt name, and the image has:
# from requirements.txt, by what it brings (pydantic, with fastapi; IFEval's
# three, with lm_eval[ifeval]), or installed beside it (the Dockerfile's check
# imports pytest and httpx)
IN_IMAGE = {"pytest", "httpx", "fastapi", "uvicorn", "pydantic", "pyrage", "langdetect", "nltk",
            "immutabledict"}
# …and what it hasn't: the distribution, and the module a test would import
NOT_IN_IMAGE = {"playwright": "playwright", "ruff": "ruff", "pytest-xdist": "xdist",
                "pytest-split": "pytest_split"}
_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def _named(file: str) -> set[str]:
    out = set()
    for line in (ROOT / file).read_text(encoding="utf-8").splitlines():
        m = _NAME.match(line.strip())
        if m:
            out.add(m.group(1).lower())
    return out


def collect(blocked: list[str], *args: str) -> subprocess.CompletedProcess:
    """step 3's command, collection only, in a Python where `blocked` can't
    be imported — no plugin is loaded on its own either, as none is there"""
    boot = ("import sys\n"
            f"for m in {blocked!r}:\n"
            "    sys.modules[m] = None\n"
            "import pytest\n"
            "raise SystemExit(pytest.main(sys.argv[1:]))\n")
    return subprocess.run(
        [sys.executable, "-c", boot, "--collect-only", "-q", "-p", "no:cacheprovider",
         "-m", STEP3, *args],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})


def test_every_dev_and_ci_requirement_is_sorted_by_whether_the_image_has_it():
    named = _named("requirements-dev.txt") | _named("requirements-ci.txt")
    assert named == IN_IMAGE | set(NOT_IN_IMAGE), (
        "a requirement of the checks is new or gone: say in tests/test_step3_collects.py "
        "whether the image has it")
    # the ones it has by name are in the image's own requirements
    image = _named("requirements.txt")
    assert {"fastapi", "uvicorn", "pyrage"} <= image
    assert not image & set(NOT_IN_IMAGE)
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"import torch,[^;]*\bpytest, httpx;", docker)
    assert "import math_verify, langdetect, nltk, immutabledict" in docker


def test_every_test_module_is_collected_without_what_the_image_lacks():
    r = collect(sorted(NOT_IN_IMAGE.values()))
    tail = "\n".join((r.stdout + r.stderr).splitlines()[-40:])
    assert r.returncode == 0, f"step 3 would run nothing:\n{tail}"
    assert "error" not in r.stdout.splitlines()[-1].lower(), tail
    # it found the suite, and the browser tests were left out by their marker
    m = re.search(r"(\d+)/(\d+) tests collected \((\d+) deselected\)", r.stdout)
    assert m and int(m.group(1)) > 500 and int(m.group(3)) > 50, tail


def test_the_check_itself_catches_a_module_that_imports_playwright_at_the_top(tmp_path):
    bad = tmp_path / "test_imports_playwright.py"
    bad.write_text("import pytest\nfrom playwright.sync_api import Error\n\n"
                   "pytestmark = pytest.mark.dashboard\n\n\ndef test_x():\n    pass\n")
    r = collect(["playwright"], str(bad), "--rootdir", str(tmp_path), "-c", os.devnull)
    assert r.returncode != 0 and "playwright" in r.stdout
    assert "error" in r.stdout.splitlines()[-1].lower()
    # the same module, the import inside the test that uses it, is collected
    bad.write_text("import pytest\n\npytestmark = pytest.mark.dashboard\n\n\n"
                   "def test_x():\n    from playwright.sync_api import Error  # noqa: F401\n")
    r = collect(["playwright"], str(bad), "--rootdir", str(tmp_path), "-c", os.devnull)
    assert r.returncode in (0, 5), r.stdout + r.stderr       # 5: all deselected, none failed
