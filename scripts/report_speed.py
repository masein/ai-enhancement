"""21c: how fast the reports answer, on the server — and, with --again, how long
the reports' stats take to work out, /healthz timed every second meanwhile.

    sudo docker compose exec -T bench python scripts/report_speed.py --again

Asks the board itself (127.0.0.1 inside the container) and nothing else. The
stats are worked out by `python -m service.report_stats --again`, at the
lowest priority, as the board starts it after an import."""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PATHS = ["healthz", "api/reports", "api/reports/step-checkpoints", "api/reports/ran-out-of-room",
         "api/reports/step-checkpoints/datasets",
         *[f"api/reports/step-checkpoints/datasets/{d}" for d in ("ours", "others", "tokens",
                                                                   "loops")],
         "reports/step-checkpoints/page"]


def timed(base: str, path: str) -> tuple[float, int]:
    t = time.time()
    try:
        with urllib.request.urlopen(f"{base}/{path}", timeout=120) as r:
            r.read()
            code = r.status
    except urllib.error.HTTPError as e:
        code = e.code
    except OSError:
        code = 0
    return time.time() - t, code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8899")
    ap.add_argument("--again", action="store_true",
                    help="work every stat out again first, /healthz timed meanwhile")
    a = ap.parse_args(argv)
    if a.again:
        root = Path(__file__).resolve().parent.parent
        t0 = time.time()
        p = subprocess.Popen([sys.executable, "-m", "service.report_stats", "--again"],
                             cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
        health = []
        while p.poll() is None:
            health.append(timed(a.base, "healthz")[0])
            time.sleep(1)
        out = p.stdout.read() if p.stdout else ""
        print(out.rstrip())
        print(f"the stats: {time.time() - t0:.1f} s to work out (exit {p.returncode})")
        if health:
            print(f"/healthz meanwhile, {len(health)} times: median "
                  f"{statistics.median(health):.3f} s, slowest {max(health):.3f} s")
    for path in PATHS:
        runs = [timed(a.base, path) for _ in range(3)]
        print(f"{max(t for t, _ in runs):7.3f} s  (slowest of 3, HTTP {runs[-1][1]})  /{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
