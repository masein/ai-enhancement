#!/usr/bin/env bash
# Launch the benchmark service on the tailnet.
#
#   cd ~/benchmarks
#   nohup bash aienh/service/run.sh > service.log 2>&1 &
#   tail -f service.log
#
# Must run from the benchmarks directory (the one holding results/, eval_tasks/,
# logs/) so service runs land in the same tree as CLI runs. Binds to the
# Tailscale IP only — friends on the tailnet reach it, the LAN does not.
set -euo pipefail

BENCH_ROOT="${BENCH_ROOT:-$PWD}"
cd "$BENCH_ROOT"
[[ -e .venv/bin/activate ]] && source .venv/bin/activate

BIND="${BIND:-$(tailscale ip -4 2>/dev/null | head -1 || true)}"
[[ -z "$BIND" ]] && { echo "no tailscale IP found — set BIND=... explicitly"; exit 1; }
PORT="${PORT:-8899}"

# repo root (this script lives at <repo>/service/run.sh) goes on PYTHONPATH so
# `service.app` imports regardless of where the repo was cloned
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export BENCH_ROOT

echo "benchmark service on http://$BIND:$PORT  (bench root: $BENCH_ROOT)"
exec python -m uvicorn service.app:app --host "$BIND" --port "$PORT" --log-level info
