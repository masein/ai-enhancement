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

from . import db, gguf, reported
from .runner import run_submission

POLL_S = 3
_stop = threading.Event()


def _gguf_first() -> bool:
    """12f.5: the host's GGUF worker's jobs, into their rows (their turn
    depends on it), and whether it's there to run them — a board run waits
    for a GGUF job queued before it only then"""
    try:
        gguf.sync()
        if not gguf.worker()["alive"]:
            return False
        ahead = db.gguf_ahead()
        if ahead:
            db.update(ahead[0], progress=gguf.waiting_line(ahead[1]))
        return True
    except Exception:                               # noqa: BLE001 — the queue goes on
        traceback.print_exc()
        return False


def loop() -> None:
    while not _stop.is_set():
        sub = None
        try:
            sub = db.claim_next(gguf_first=_gguf_first())
            if sub is None:
                # 12m.2: the day's reported scores, in a thread of their own
                try:
                    reported.daily()
                except Exception:                   # noqa: BLE001 — the queue goes on
                    traceback.print_exc()
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
