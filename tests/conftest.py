"""Shared fixtures.

The synthetic tree is built once per session and shared read-only. Tests that
need to change a tree underneath the service (the freshness tests) build their
own in a function-scoped temp dir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "tests" / "fixtures"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import make_fixture  # noqa: E402


@pytest.fixture(scope="session")
def tree(tmp_path_factory) -> dict:
    """The fixture tree, diagnosed (all but the odd one out) and frozen into a
    single-file report. See make_fixture.build for the manifest's shape."""
    root = tmp_path_factory.mktemp("bench")
    return make_fixture.build(root, report=root / "report.html")


@pytest.fixture(scope="session")
def diag(tree) -> dict[str, dict]:
    """model id -> parsed diagnose.json, for the models that have one."""
    out = {}
    for mid, m in tree["models"].items():
        f = m["dir"] / "diagnose.json"
        if f.exists():
            out[mid] = json.loads(f.read_text(encoding="utf-8"))
    return out


@pytest.fixture(scope="session")
def payload(tree) -> dict:
    """What GET /api/results would serve for the fixture tree."""
    import report_lm_eval as report
    runs = report.load_results(tree["out_dir"])
    return report.build_payload(report.merge_runs(runs), "Fixture board",
                                source=str(tree["out_dir"]))
