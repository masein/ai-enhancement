#!/usr/bin/env bash
# The check before every merge: what .github/workflows/ci.yml ran, run here.
# The Actions minutes ran out on 2026-09-24 and CI is manual-only since; this
# is the gate now (HANDOFF.md § 5b).
#
#   scripts/check.sh
#
# It runs in a container — python:3.12-slim, requirements-dev.txt and
# Playwright's Chromium (scripts/check.Dockerfile) — with the repo mounted, so
# the check never uses the host's Python. The host needs Docker and nothing
# else; the image is built on the first run and when requirements-dev.txt or
# the Dockerfile change, and Docker's cache makes every other build instant.
#
# Same steps as CI, in the same order: ruff, a compile pass, the unit and API
# tests, then the browser suite — `gpu` and `network` tests deselected. Every
# step runs even after one fails, so the last line says everything at once:
#
#   check of b273738 (clean) · 2026-09-24 14:05 +0400
#   lint ok · unit 516/516 · browser 398/398 · 19 min
#
# That last line, the commit and the date go in the PR under "Local check".
# Exit status 0 only when every step passed.
set -u
cd "$(dirname "$0")/.."

if [ -z "${EVALBOARD_CHECK_INSIDE:-}" ]; then
  # ---- on the host: build the image, then run this script inside it ----
  command -v docker >/dev/null || { echo "check.sh needs Docker"; exit 2; }
  ctx=$(mktemp -d)
  trap 'rm -rf "$ctx"' EXIT
  cp requirements-dev.txt scripts/check.Dockerfile "$ctx"/
  docker build -q -f "$ctx/check.Dockerfile" -t evalboard-check \
    --build-arg HOST_UID="$(id -u)" --build-arg HOST_GID="$(id -g)" \
    --build-arg HOST_USER="$(id -un)" "$ctx" >/dev/null \
    || { echo "check.sh: the image did not build"; exit 2; }
  # the commit and whether the tree matches it, read here: a worktree's .git
  # points at the main repository, which the container does not have
  sha=$(git rev-parse --short HEAD 2>/dev/null || echo "no-git")
  tree=clean
  [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] && tree="UNCOMMITTED CHANGES"
  # the repo at the same path as here, and git's own directory beside it, so
  # `git ls-files` inside answers for this checkout
  common=$(cd "$(git rev-parse --git-common-dir)" && pwd)
  # the date on the summary is the host's clock and zone; the tests run in
  # UTC, as they did on the CI runner
  tz=$(readlink /etc/localtime 2>/dev/null | sed 's|.*zoneinfo/||')
  exec docker run --rm --init --shm-size=2g \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD":"$PWD" -v "$common":"$common" -w "$PWD" \
    -e EVALBOARD_CHECK_INSIDE=1 -e CHECK_SHA="$sha" -e CHECK_TREE="$tree" \
    -e CHECK_TZ="${tz:-UTC}" \
    evalboard-check bash scripts/check.sh
fi

# ---- inside the container ----
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
start=$(date +%s)

lint=ok
echo "== ruff"
python -m ruff check . || lint=FAILED
echo "== compile scripts/ service/ clients/"
python -m compileall -q scripts service clients >/dev/null || lint=FAILED
echo "== unit and API tests"
python -m pytest -q -p no:cacheprovider -m "not gpu and not network and not dashboard" \
  --junitxml "$out/unit.xml"
echo "== browser suite"
python -m pytest -q -p no:cacheprovider -m dashboard --junitxml "$out/browser.xml"

mins=$(( ($(date +%s) - start + 30) / 60 ))
echo
when=$(TZ="${CHECK_TZ:-UTC}" date '+%Y-%m-%d %H:%M %z')
pyv=$(python -c 'import platform; print(platform.python_version())')
echo "check of $CHECK_SHA ($CHECK_TREE) · $when · python $pyv"
python - "$out" "$lint" "$mins" <<'EOF'
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
