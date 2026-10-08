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

from . import db, gguf, hidden_store, reported
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


def _idle() -> None:
    # 12m.2: the day's reported scores, in a thread of their own
    try:
        reported.daily()
    except Exception:                               # noqa: BLE001 — the queue goes on
        traceback.print_exc()
    _backup()


def _backup() -> None:
    # 12p.1: what lives only on the data volume, backed up once a day
    try:
        hidden_store.daily()
    except Exception:                               # noqa: BLE001 — the queue goes on
        traceback.print_exc()


def _held() -> dict:
    """18c point 11: the served models an agent run is using — the queue
    holds their runs"""
    try:
        from . import agent_runs
        return agent_runs.busy()
    except Exception:                               # noqa: BLE001 — the queue goes on
        traceback.print_exc()
        return {}


def once(remote: bool = False) -> bool:
    """one claim and its run, in the GPU lane or (12m.3) the lane of models
    from OpenRouter; False when nothing was queued for it"""
    sub = None
    try:
        sub = db.claim_next(remote=True) if remote else db.claim_next(
            gguf_first=_gguf_first(), held=_held())
        if sub is None:
            return False
        run_submission(sub)
    except Exception as e:                          # noqa: BLE001 — worker must survive anything
        traceback.print_exc()
        if sub is not None:
            db.update(sub["id"], status="failed", finished_at=time.time(),
                      error=f"internal error: {e!r} — see service log")
    return True


def loop() -> None:
    while not _stop.is_set():
        if not once():
            _idle()
            _stop.wait(POLL_S)


def loop_remote() -> None:
    """12m.3: runs of models from OpenRouter never touch the GPU: they skip the
    run lock and the GPU queue, one at a time among themselves, and a GPU run
    never waits behind one. The judge they call is any run's"""
    while not _stop.is_set():
        if not once(remote=True):
            _backup()                               # a long GPU queue never idles the other lane
            _stop.wait(POLL_S)


def start() -> threading.Thread:
    t = threading.Thread(target=loop, name="benchmark-worker", daemon=True)
    t.start()
    threading.Thread(target=loop_remote, name="remote-worker", daemon=True).start()
    return t


def stop() -> None:
    _stop.set()
