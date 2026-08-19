"""The queue consumer: one daemon thread, one job at a time.

That single-mindedness is the design, not a limitation — one worker means at most
one lm_eval on the shared GPU, which is the same etiquette the CLI script
enforces with its lockfile (and the runner takes that same lock anyway, so even
adding a second worker later could not cause a race — it would just wait).
"""

from __future__ import annotations

import threading
import time
import traceback

from . import db
from .runner import run_submission

POLL_S = 3
_stop = threading.Event()


def loop() -> None:
    while not _stop.is_set():
        sub = None
        try:
            sub = db.claim_next()
            if sub is None:
                _stop.wait(POLL_S)
                continue
            run_submission(sub)
        except Exception as e:                      # noqa: BLE001 — worker must survive anything
            traceback.print_exc()
            if sub is not None:
                db.update(sub["id"], status="failed", finished_at=time.time(),
                          error=f"internal error: {e!r} — see service log")


def start() -> threading.Thread:
    t = threading.Thread(target=loop, name="benchmark-worker", daemon=True)
    t.start()
    return t


def stop() -> None:
    _stop.set()
