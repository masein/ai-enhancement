#!/usr/bin/env bash
# The check before every merge: what .github/workflows/ci.yml ran, run here.
# The Actions minutes ran out on 2026-09-24 and CI is manual-only since; this
# is the gate now (HANDOFF.md § Checks).
#
#   scripts/check.sh                              # with python3 on PATH
#   PYTHON=.venv/bin/python scripts/check.sh      # or a given interpreter
#
# Needs requirements-dev.txt installed and `playwright install chromium` once.
# Same steps as CI, in the same order: ruff, a compile pass, the unit and API
# tests, then the browser suite — `gpu` and `network` tests deselected. Every
# step runs even after one fails, so the last line says everything at once:
#
#   check of b273738 (clean) · 2026-09-24 14:05 +04
#   lint ok · unit 516/516 · browser 398/398 · 19 min
#
# That last line, the commit and the date go in the PR under "Local check".
# Exit status 0 only when every step passed.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
start=$(date +%s)
sha=$(git rev-parse --short HEAD 2>/dev/null || echo "no-git")
# a check of a tree with edits in it is not a check of the commit
tree=clean
[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] && tree="UNCOMMITTED CHANGES"

lint=ok
echo "== ruff"
"$PY" -m ruff check . || lint=FAILED
echo "== compile scripts/ service/ clients/"
"$PY" -m compileall -q scripts service clients >/dev/null || lint=FAILED
echo "== unit and API tests"
"$PY" -m pytest -q -m "not gpu and not network and not dashboard" --junitxml "$out/unit.xml"
echo "== browser suite"
"$PY" -m pytest -q -m dashboard --junitxml "$out/browser.xml"

mins=$(( ($(date +%s) - start + 30) / 60 ))
echo
echo "check of $sha ($tree) · $(date '+%Y-%m-%d %H:%M %z')"
"$PY" - "$out" "$lint" "$mins" <<'EOF'
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

out, lint, mins = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ok = lint == "ok"
parts = [f"lint {lint}"]
for name in ("unit", "browser"):
    f = out / f"{name}.xml"
    if not f.exists():
        parts.append(f"{name} did not run")
        ok = False
        continue
    root = ET.parse(f).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    n = {k: sum(int(s.get(k, 0)) for s in suites)
         for k in ("tests", "failures", "errors", "skipped")}
    ran = n["tests"] - n["skipped"]
    passed = ran - n["failures"] - n["errors"]
    part = f"{name} {passed}/{ran}"
    if n["skipped"]:
        part += f" ({n['skipped']} skipped)"
    parts.append(part)
    ok = ok and passed == ran and ran > 0
parts.append(f"{mins} min")
print(" · ".join(parts))
sys.exit(0 if ok else 1)
EOF
