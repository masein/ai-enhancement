"""How a trial stops (12a.5b) — scripts/trial_generative.py and
scripts/trial_standard.py share it.

A trial holds the GPU lock while lm_eval runs, so it must never outlive the
person who started it. On 2026-09-25 one ran for two hours after Ctrl+C,
holding the lock. It stops — kills lm_eval and every process lm_eval started,
then frees the lock and prints "stopped; GPU free" — when:

- it gets Ctrl+C (SIGINT), SIGTERM or SIGHUP;
- the terminal that started it goes away. `docker compose exec -T` passes no
  Ctrl+C on to the process in the container: Ctrl+C there ends the client on
  the host and leaves the trial running. What reaches the trial is its stdin
  closing, so a trial whose stdin is a pipe stops when that pipe reaches its
  end — unless it was at its end from the start (nothing was attached);
- it reaches --max-minutes (default 15). It prints what it finished.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

MAX_MINUTES = 15.0
# stdin already at its end this soon after the start was never a terminal's
QUIET_START_S = 2.0


class Stopped(Exception):
    """Raised out of Stop.run(): why the trial stopped"""


class Stop:
    def __init__(self, max_minutes: float = MAX_MINUTES, watch_stdin: bool = True):
        self.max_s = max_minutes * 60
        self.t0 = time.monotonic()
        self.why = ""
        self._old: dict = {}
        self._watch = watch_stdin

    # -- the signals and the terminal -------------------------------------------
    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                self._old[sig] = signal.signal(sig, self._on_signal)
            except (ValueError, OSError):        # not the main thread
                pass
        if self._watch and _stdin_is_a_pipe():
            threading.Thread(target=self._watch_stdin, daemon=True).start()
        return self

    def __exit__(self, *exc):
        for sig, old in self._old.items():
            signal.signal(sig, old)
        return False

    def _on_signal(self, signum, _frame):
        self.why = self.why or f"stopped by {signal.Signals(signum).name}"

    def _watch_stdin(self):
        try:
            while os.read(0, 4096):
                pass
        except OSError:
            return
        if time.monotonic() - self.t0 > QUIET_START_S:
            self.why = self.why or "stopped: the terminal that started it went away"

    # -- the time ----------------------------------------------------------------
    def over(self) -> str:
        """why the trial must stop now, or ''"""
        if not self.why and time.monotonic() - self.t0 > self.max_s:
            self.why = f"stopped at the {self.max_s / 60:g}-minute limit (--max-minutes)"
        return self.why

    # -- lm_eval -----------------------------------------------------------------
    def run(self, cmd: list[str], cwd: Path, log: Path) -> int:
        """lm_eval as a run starts it, in a process group of its own; its exit
        code, or Stopped once it and every process it started are gone"""
        if self.over():
            raise Stopped(self.why)
        with open(log, "a") as lf:
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            while True:
                try:
                    return proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
                if self.over():
                    kill_group(proc)
                    raise Stopped(self.why)


def kill_group(proc: subprocess.Popen, grace: float = 10.0) -> None:
    """SIGTERM to the child's whole process group, then SIGKILL"""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = None
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
        # the child is gone; its own children may not be
        if pgid is not None and sig == signal.SIGTERM:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return


def _stdin_is_a_pipe() -> bool:
    try:
        mode = os.fstat(0).st_mode
    except OSError:
        return False
    return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)


def say(line: str) -> None:
    """print, even when whoever was reading has gone"""
    try:
        print(line, flush=True)
    except (BrokenPipeError, OSError):
        try:
            sys.stdout = open(os.devnull, "w")
        except OSError:
            pass
