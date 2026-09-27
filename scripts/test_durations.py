"""Write .test_durations, the per-test times CI's browser shards are split by
(pytest-split), from JUnit XML of a browser run.

    python scripts/test_durations.py browser.xml [more.xml ...]

Any JUnit XML of `pytest -m dashboard` works: a local run with
`--junitxml`, or the five CI shards' files together. The time is JUnit's,
setup + call + teardown, so a module's server start lands on its first test,
as pytest-split's own `--store-durations` records it. A test missing from the
file is split by the average, so a stale file makes shards uneven; it never
drops a test.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def node_id(classname: str, name: str) -> str:
    """JUnit's "tests.test_x.TestY" + "test_z[1280]" back to pytest's
    "tests/test_x.py::TestY::test_z[1280]": the longest dotted prefix that is
    a file is the module, the rest are classes"""
    parts = classname.split(".")
    for i in range(len(parts), 0, -1):
        f = Path(*parts[:i]).with_suffix(".py")
        if (ROOT / f).exists():
            return "::".join([f.as_posix(), *parts[i:], name])
    raise SystemExit(f"no test file for {classname}")


def durations(xmls: list[Path]) -> dict[str, float]:
    out: dict[str, float] = {}
    for x in xmls:
        for case in ET.parse(x).getroot().iter("testcase"):
            if case.find("skipped") is not None:
                continue
            nid = node_id(case.get("classname", ""), case.get("name", ""))
            out[nid] = round(float(case.get("time") or 0), 3)
    return dict(sorted(out.items()))


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    d = durations([Path(a) for a in argv])
    (ROOT / ".test_durations").write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
    print(f".test_durations: {len(d)} tests, {sum(d.values()) / 60:.1f} min in all")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
