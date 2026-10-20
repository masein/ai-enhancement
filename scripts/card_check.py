#!/usr/bin/env python3
"""23c: the card a rented box runs on, measured — before its first question
and while it answers (docs/prompts/phase-23c-speed-power-and-the-card.md).

The card check (remote_gguf.py, right after llama-server is up and before the
step asks anything):
- the card, as nvidia-smi says: its name, memory, power limit and its
  default, highest SM clock, PCIe link now and at most, driver, and the
  memory in use;
- a speed probe against the step's own llama-server (its slots, its cache
  types): POST /completion with token ids as the prompt — the lengths exact —
  256 out past any end, no prompt cache, temperature 0, read from
  llama-server's own `timings`: a 512-token prompt alone, three times (the
  median); a 16,384-token prompt alone, once; and eight 512-token prompts at
  once (as many as the server has slots, at most eight). The prompt's tokens
  are PROBE_TEXT's, tokenised once by the server and repeated to length, so
  every box asks the same. No benchmark question; nothing it writes is an
  answer. A probe that fails says why; the step goes on.

The sampler, beside the step from the server's start to its end: every five
seconds the card's power, use, memory, clock, heat and throttle reasons, a
line each in the step's gpu.jsonl (it goes in the bundle). From it: the
step's watt-hours, mean power and use while asking, peak memory, and the
share of samples throttled, and why.

And the words the fetch prints, and the slow-card flag it raises: a box
whose card answers 20% or more below the median of its build's other boxes
(at least three), whose PCIe link is below its maximum, whose power limit is
under its default, or whose card has been under 60% busy for a quarter of
an hour — with the hours replacing it would save.

Python's standard library only: the fetch runs it on the server, outside the
image."""

from __future__ import annotations

import json
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# the probe's text: plain prose written for it, no benchmark's, tokenised by
# the server and repeated to each prompt's length
PROBE_TEXT = (
    "A river begins as rain on high ground. The water gathers in small channels, then in "
    "streams that cut through soil and stone, and the streams meet and widen as the land "
    "falls toward the sea. Along the way the river carries sand and silt, drops it where "
    "the current slows, and builds banks, islands and wide flat plains. People have always "
    "settled beside rivers: they bring water for fields and animals, fish for food, a road "
    "for boats and power for mills. A town grows at a ford or a bridge, a market at the "
    "place where two valleys join. In spring the snow melts and the water rises; in late "
    "summer it runs low and slow, and the stones of its bed show through. Over many years a "
    "river moves, leaving old loops behind as quiet lakes, and the map of a valley is the "
    "record of where its water has been. ")
SHORT, LONG, OUT = 512, 16_384, 256          # the prompts' tokens, and each one's output
ALONE_RUNS, TOGETHER = 3, 8
REQUEST_S = 180.0                            # one request, at the most
EVERY_S = 5.0                                # the sampler's interval
GAP_S = 30.0                                 # samples further apart aren't joined (a restart)
WRITE_EVERY_S = 60.0                         # the running means to the progress file
BUSY_WINDOW_S = 15 * 60                      # "under 60% busy for 15 minutes"
BUSY_UNDER = 60.0
SLOW_UNDER = 0.8                             # 20% or more below the others' median
FEW_OTHERS = 3                               # the others a speed is compared with, at least
NEW_BOX_S = 1800.0                           # half an hour to set a new box up
CARD_FIELDS = (("name", "name", str), ("memory.total", "memory_mib", int),
               ("memory.used", "memory_used_mib", int), ("power.limit", "power_limit_w", float),
               ("power.default_limit", "power_default_w", float),
               ("clocks.max.sm", "sm_max_mhz", int),
               ("pcie.link.gen.current", "pcie_gen", int),
               ("pcie.link.gen.max", "pcie_gen_max", int),
               ("pcie.link.width.current", "pcie_width", int),
               ("pcie.link.width.max", "pcie_width_max", int),
               ("driver_version", "driver", str))
SAMPLE_FIELDS = ("power.draw", "utilization.gpu", "memory.used", "clocks.sm", "temperature.gpu")
# nvidia-smi's throttle reasons, as named: renamed clocks_event_reasons in
# newer drivers; a driver with neither samples without them
REASON_FIELDS = ("clocks_throttle_reasons.active", "clocks_event_reasons.active", None)
# its bits (nvml's clocksThrottleReasons): the software power cap and the
# power brake are power; the thermal slowdowns heat; HW slowdown either
REASONS = {"power": 0x4 | 0x80, "heat": 0x20 | 0x40, "hardware": 0x8}


def RUN(cmd: list[str]) -> str:
    """nvidia-smi's output; raises when it can't be run or fails"""
    got = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if got.returncode != 0:
        raise OSError(f"exit {got.returncode}: {(got.stderr or got.stdout).strip()[:200]}")
    return got.stdout


def _num(v: str, kind=float):
    """nvidia-smi's value as a number — None for "[N/A]", "Not Supported" or
    anything else that isn't one"""
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    if x != x or abs(x) == float("inf"):
        return None
    return int(round(x)) if kind is int else x


def _smi(fields: list[str], run=None) -> list[list[str]]:
    out = (run or RUN)(["nvidia-smi", f"--query-gpu={','.join(fields)}",
                        "--format=csv,noheader,nounits"])
    return [[c.strip() for c in ln.split(",")] for ln in out.splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# the card
# ---------------------------------------------------------------------------

def card(run=None) -> dict:
    """the first card as nvidia-smi says — every field it can read; one it
    doesn't know (an older driver) is asked for alone, and left out. {} when
    nvidia-smi can't be run"""
    names = [f for f, _, _ in CARD_FIELDS]
    try:
        rows = _smi(names, run)
        got = dict(zip(names, rows[0])) if rows else {}
    except Exception:                                   # noqa: BLE001 — field by field
        got = {}
        for f in names:
            try:
                rows = _smi([f], run)
                if rows:
                    got[f] = rows[0][0]
            except Exception:                           # noqa: BLE001 — not this one
                continue
    out = {}
    for f, k, kind in CARD_FIELDS:
        if f not in got:
            continue
        v = got[f].strip() if kind is str else _num(got[f], kind)
        if v not in (None, "", "[N/A]"):
            out[k] = v
    return out


def link(run=None) -> dict:
    """the PCIe link now — read under load: a card at rest lowers its
    generation to save power, and would read as slow"""
    try:
        rows = _smi(["pcie.link.gen.current", "pcie.link.width.current"], run)
    except Exception:                                   # noqa: BLE001 — unread
        return {}
    if not rows:
        return {}
    g, w = _num(rows[0][0], int), _num(rows[0][1], int) if len(rows[0]) > 1 else None
    return {k: v for k, v in (("pcie_gen", g), ("pcie_width", w)) if v is not None}


# ---------------------------------------------------------------------------
# the speed probe
# ---------------------------------------------------------------------------

def _post(base: str, path: str, body: dict, timeout: float = REQUEST_S) -> dict:
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            said = json.loads(e.read() or b"{}").get("error") or {}
            said = said.get("message") if isinstance(said, dict) else said
        except (ValueError, AttributeError):
            said = ""
        raise RuntimeError(f"{path} answered {e.code}" + (f": {str(said)[:160]}" if said else ""))
    except urllib.error.URLError as e:
        raise RuntimeError(f"{path}: {e.reason}")
    except TimeoutError:
        raise RuntimeError(f"{path}: no answer in {timeout:.0f} s")


def tokens_of(base: str) -> list[int]:
    """PROBE_TEXT's tokens, as this server makes them"""
    got = _post(base, "/tokenize", {"content": PROBE_TEXT, "add_special": False})
    toks = [t for t in got.get("tokens") or [] if isinstance(t, int)]
    if len(toks) < 32:
        raise RuntimeError(f"/tokenize gave {len(toks)} tokens for the probe's text")
    return toks


def prompt(toks: list[int], n: int, start: int = 0) -> list[int]:
    """n tokens of the text, from `start`, repeated to length"""
    reps = (start + n) // len(toks) + 1
    return (toks * reps)[start:start + n]


def rates(timings: dict) -> dict:
    """a reply's prompt and output tokens a second, from llama-server's own
    timings: tokens over milliseconds"""
    t = timings if isinstance(timings, dict) else {}

    def per_s(n, ms):
        return round(n / ms * 1000, 2) if isinstance(n, (int, float)) and isinstance(
            ms, (int, float)) and n > 0 and ms > 0 else None
    return {"prompt_n": t.get("prompt_n"), "out_n": t.get("predicted_n"),
            "prompt_tok_s": per_s(t.get("prompt_n"), t.get("prompt_ms")),
            "output_tok_s": per_s(t.get("predicted_n"), t.get("predicted_ms"))}


def complete(base: str, ids: list[int], n_out: int = OUT) -> dict:
    got = _post(base, "/completion", {"prompt": ids, "n_predict": n_out, "ignore_eos": True,
                                      "cache_prompt": False, "temperature": 0, "seed": 0,
                                      "stream": False})
    r = rates(got.get("timings"))
    if r["output_tok_s"] is None:
        raise RuntimeError("/completion gave no timings")
    return r


def _median(xs: list) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 2) if xs else None


def probe(base: str, slot_ctx: int | None, slots: int, during=None) -> dict:
    """the three speeds (see the top). `during()` is called while the eight
    are asked (the PCIe link, read under load). Raises with why"""
    t0 = time.time()
    toks = tokens_of(base)
    complete(base, prompt(toks, 64), 16)                # a warm-up, not counted
    alone = [complete(base, prompt(toks, SHORT)) for _ in range(ALONE_RUNS)]
    out = {"alone": {"prompt_n": SHORT, "out_n": OUT, "runs": alone,
                     "prompt_tok_s": _median([r["prompt_tok_s"] for r in alone]),
                     "output_tok_s": _median([r["output_tok_s"] for r in alone])}}
    if slot_ctx and slot_ctx < LONG + OUT:
        out["long"] = {"skipped": f"a slot holds {slot_ctx:,} tokens, under the "
                                  f"{LONG + OUT:,} the long prompt needs"}
    else:
        r = complete(base, prompt(toks, LONG))
        out["long"] = {"prompt_n": LONG, "out_n": OUT, "prompt_tok_s": r["prompt_tok_s"],
                       "output_tok_s": r["output_tok_s"]}
    k = max(1, min(TOGETHER, int(slots or TOGETHER)))
    got: list = [None] * k
    gate = threading.Barrier(k + 1)

    def ask(i: int) -> None:
        try:
            gate.wait(30)
            # each its own slice of the text: eight prompts, not one eight times
            got[i] = complete(base, prompt(toks, SHORT, start=37 * (i + 1)))
        except Exception as e:                          # noqa: BLE001 — said below
            got[i] = e
    th = [threading.Thread(target=ask, args=(i,), daemon=True) for i in range(k)]
    for x in th:
        x.start()
    gate.wait(30)
    w0 = time.time()
    if during is not None:
        time.sleep(0.5)
        during()
    for x in th:
        x.join(REQUEST_S + 30)
    bad = [x for x in got if not isinstance(x, dict)]
    if bad:
        raise RuntimeError(f"{len(bad)} of the {k} asked at once failed: {bad[0]}")
    each = [r["output_tok_s"] for r in got]
    # they decode side by side: the total is the sum of each one's own rate
    out["together"] = {"n": k, "prompt_n": SHORT, "out_n": OUT, "each": each,
                       "output_tok_s": round(sum(each), 2),
                       "wall_s": round(time.time() - w0, 2)}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def check(base: str, slot_ctx: int | None, slots: int, say=lambda s: None, run=None) -> dict:
    """the card check: the card, then the probe. Never raises — a probe that
    fails is said, with the card as read"""
    rec = {"at": round(time.time()), "card": card(run), "slots": slots}
    seen: list[dict] = []
    try:
        rec["probe"] = probe(base, slot_ctx, slots, during=lambda: seen.append(link(run)))
    except Exception as e:                              # noqa: BLE001 — never the step's end
        rec["why"] = f"the speed probe failed: {e}"[:400]
    # the link under load, where it was read: the higher of the two readings
    for k in ("pcie_gen", "pcie_width"):
        vals = [x[k] for x in [rec["card"], *seen] if isinstance(x.get(k), int)]
        if vals:
            rec["card"][k] = max(vals)
    say(words(rec))
    return rec


# ---------------------------------------------------------------------------
# its words
# ---------------------------------------------------------------------------

def _n(x) -> str:
    return f"{x:,.0f}" if isinstance(x, (int, float)) else "?"


def words(rec: dict | None) -> str:
    """'card: 142 tok/s alone, 610 at 8 · 340 W limit · PCIe 5 ×16' — what is
    known of it; why the probe failed, when it did"""
    if not isinstance(rec, dict):
        return ""
    c = rec.get("card") if isinstance(rec.get("card"), dict) else {}
    p = rec.get("probe") if isinstance(rec.get("probe"), dict) else {}
    parts = []
    a, t = (p.get(k) if isinstance(p.get(k), dict) else {} for k in ("alone", "together"))
    if a.get("output_tok_s") is not None:
        parts.append(f"{_n(a['output_tok_s'])} tok/s alone"
                     + (f", {_n(t['output_tok_s'])} at {t.get('n')}"
                        if t.get("output_tok_s") is not None else ""))
    elif rec.get("why"):
        parts.append(str(rec["why"]))
    if c.get("power_limit_w") is not None:
        parts.append(f"{_n(c['power_limit_w'])} W limit"
                     + (f" (default {_n(c['power_default_w'])})"
                        if under_default(c) else ""))
    if c.get("pcie_gen") is not None:
        parts.append(f"PCIe {c['pcie_gen']} ×{c.get('pcie_width', '?')}"
                     + (f" (can do {c.get('pcie_gen_max', '?')} ×{c.get('pcie_width_max', '?')})"
                        if link_below(c) else ""))
    return "card: " + " · ".join(parts) if parts else ""


def under_default(c: dict) -> bool:
    lim, dflt = c.get("power_limit_w"), c.get("power_default_w")
    return isinstance(lim, (int, float)) and isinstance(dflt, (int, float)) and lim < dflt - 1


def link_below(c: dict) -> bool:
    return any(isinstance(c.get(k), int) and isinstance(c.get(k + "_max"), int)
               and c[k] < c[k + "_max"] for k in ("pcie_gen", "pcie_width"))


def alone_tok_s(rec: dict | None) -> float | None:
    """its output tokens a second alone — None for a check without one (or a
    progress file that isn't what a box writes)"""
    p = rec.get("probe") if isinstance(rec, dict) else None
    a = p.get("alone") if isinstance(p, dict) else None
    v = a.get("output_tok_s") if isinstance(a, dict) else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 \
        else None


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------

def parse(rows: list[list[str]], reasons: bool) -> dict:
    """one reading of every card: power summed, use averaged, memory summed,
    the first's clock, the hottest, the throttle reasons of any"""
    w = [_num(r[0]) for r in rows]
    u = [_num(r[1]) for r in rows]
    m = [_num(r[2], int) for r in rows]
    out = {"w": round(sum(x for x in w if x is not None), 1) if any(x is not None for x in w)
           else None,
           "util": round(statistics.fmean([x for x in u if x is not None]), 1)
           if any(x is not None for x in u) else None,
           "mib": sum(x for x in m if x is not None) if any(x is not None for x in m) else None,
           "mhz": _num(rows[0][3], int) if rows and len(rows[0]) > 3 else None,
           "c": max((x for x in (_num(r[4], int) for r in rows if len(r) > 4)
                     if x is not None), default=None)}
    if reasons:
        bits = 0
        for r in rows:
            try:
                bits |= int(r[5], 16) if len(r) > 5 else 0
            except ValueError:
                continue
        out["why"] = bits
    return {k: v for k, v in out.items() if v is not None}


class Sampler(threading.Thread):
    """the card every EVERY_S seconds, a line each in `path` (appended: a
    step's sessions share it), the phase it was in with it — "loading",
    "check", "asking", "after". A reading that fails is a line with the time
    and phase only, the first failure's words kept. `report(use)` is given
    the running means every WRITE_EVERY_S"""

    def __init__(self, path: Path, report=None, every: float | None = None, run=None,
                 clock=time.time):
        super().__init__(daemon=True)
        self.path, self.report, self.smi, self.clock = path, report, run, clock
        self.every = every or EVERY_S
        self.phase = "loading"
        self.samples: list[dict] = []
        self.error = ""
        self._halt = threading.Event()
        self._fields: list[str | None] = list(REASON_FIELDS)
        self._lock = threading.Lock()
        self._said = 0.0

    def read(self) -> dict:
        while True:
            extra = self._fields[0]
            try:
                rows = _smi([*SAMPLE_FIELDS, *([extra] if extra else [])], self.smi)
                if not rows:
                    raise OSError("nvidia-smi said nothing")
                return parse(rows, extra is not None)
            except Exception as e:                      # noqa: BLE001 — the next field, or said
                if extra is not None and len(self._fields) > 1:
                    self._fields.pop(0)                 # a driver that doesn't know that name
                    continue
                if not self.error:
                    self.error = str(e)[:200] or type(e).__name__
                return {}

    def tick(self) -> dict:
        s = {"t": round(self.clock(), 1), "phase": self.phase, **self.read()}
        with self._lock:
            self.samples.append(s)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(s) + "\n")
        if self.report is not None and s["t"] - self._said >= WRITE_EVERY_S:
            self._said = s["t"]
            try:
                self.report(running(self.samples, s["t"]))
            except Exception:                           # noqa: BLE001 — never the step's business
                pass
        return s

    def run(self) -> None:                              # noqa: D401 — Thread's
        self.tick()
        while not self._halt.wait(self.every):
            self.tick()

    def stop(self) -> None:
        self._halt.set()
        if self.is_alive():
            self.join(10)
        try:
            self.tick()                                 # the last reading, at its end
        except OSError:
            pass


def read_samples(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            x = json.loads(ln)
        except ValueError:
            continue
        if isinstance(x, dict) and isinstance(x.get("t"), (int, float)):
            out.append(x)
    return sorted(out, key=lambda x: x["t"])


def summarise(samples: list[dict], phase: str | None = None) -> dict:
    """the samples' sums and means: seconds and watt-hours over the joined
    intervals (two samples at most GAP_S apart; with `phase`, both in it) —
    the power over each interval the mean of its ends; the mean power and use
    of the asking samples; the peak memory; the asking samples throttled, and
    for which reasons"""
    s = sorted(samples, key=lambda x: x["t"])
    secs = wh = 0.0
    secs_w = 0.0
    for a, b in zip(s, s[1:]):
        dt = b["t"] - a["t"]
        if dt <= 0 or dt > GAP_S or (phase and not (a.get("phase") == b.get("phase") == phase)):
            continue
        secs += dt
        if isinstance(a.get("w"), (int, float)) and isinstance(b.get("w"), (int, float)):
            wh += (a["w"] + b["w"]) / 2 * dt / 3600
            secs_w += dt
    asking = [x for x in s if x.get("phase") == "asking"]
    pw = [x["w"] for x in asking if isinstance(x.get("w"), (int, float))]
    ut = [x["util"] for x in asking if isinstance(x.get("util"), (int, float))]
    why = [x["why"] for x in asking if isinstance(x.get("why"), int)]
    mem = [x["mib"] for x in s if isinstance(x.get("mib"), (int, float))]
    out = {"seconds": round(secs, 1), "samples": len(s),
           "wh": round(wh, 3) if secs_w else None,
           "mean_w": round(statistics.fmean(pw), 1) if pw else None,
           "mean_util": round(statistics.fmean(ut), 1) if ut else None,
           "peak_mib": max(mem) if mem else None}
    if why:
        out["throttled"] = round(sum(1 for b in why if b & sum(REASONS.values())) / len(why), 4)
        out["reasons"] = {k: round(sum(1 for b in why if b & m) / len(why), 4)
                          for k, m in REASONS.items() if any(b & m for b in why)}
    return out


def running(samples: list[dict], now: float) -> dict:
    """the progress file's running means: this session's so far, and its use
    over the last BUSY_WINDOW_S once it has been asking that long"""
    ask = [x for x in samples if x.get("phase") == "asking"]
    out = {k: v for k, v in summarise(samples).items()
           if k in ("wh", "mean_w", "mean_util", "peak_mib", "throttled")}
    out["asking_min"] = round((ask[-1]["t"] - ask[0]["t"]) / 60, 1) if len(ask) > 1 else 0.0
    last = [x["util"] for x in ask if x["t"] >= now - BUSY_WINDOW_S
            and isinstance(x.get("util"), (int, float))]
    if ask and now - ask[0]["t"] >= BUSY_WINDOW_S and last:
        out["util_15m"] = round(statistics.fmean(last), 1)
    return out


def use_words(use: dict | None) -> str:
    """'312 W · 96% busy' — while asking"""
    if not isinstance(use, dict) or use.get("mean_w") is None:
        return ""
    return (f"{_n(use['mean_w'])} W"
            + (f" · {_n(use['mean_util'])}% busy" if use.get("mean_util") is not None else ""))


# ---------------------------------------------------------------------------
# a slow card, among its build's boxes
# ---------------------------------------------------------------------------

def hours_saved(left_s: float, step: list[tuple[int, float]], setup_s: float = NEW_BOX_S
                ) -> float:
    """what replacing the box now saves: its time left at its own pace, less
    a new box's — half an hour to set up, then its whole step (a new box
    starts the step from the beginning), each benchmark's answers at the
    others' pace"""
    return (left_s - (setup_s + sum(n * p for n, p in step))) / 3600


def slow_flags(boxes: list[dict]) -> dict[str, dict]:
    """each box of a build that looks slow, by its key: {"words", "why", "hours"}.
    A box: {"key", "model", "label", "check" (its card check), "use" (its
    running means), "asking" (bool), "step": [{"task", "answers", "n", "pace"
    (its own seconds an answer, or None), "state" now|next|done, "plan_pace"}]}.
    Compared with the same model's other boxes: its speed alone with the
    median of theirs (at least FEW_OTHERS); its PCIe link with its maximum;
    its power limit with its default; its use over the last quarter hour.
    The hours its replacement would save, when that is an hour or more"""
    out: dict[str, dict] = {}
    for me in boxes:
        if not me.get("asking"):
            continue
        others = [b for b in boxes if b is not me and b.get("model") == me.get("model")]
        mine = alone_tok_s(me.get("check"))
        theirs = [x for x in (alone_tok_s(b.get("check")) for b in others) if x]
        med = statistics.median(theirs) if len(theirs) >= FEW_OTHERS else None
        c = me["check"].get("card") if isinstance(me.get("check"), dict) else None
        c = c if isinstance(c, dict) else {}
        why, says = [], []
        if mine and med and mine <= SLOW_UNDER * med:
            why.append("speed")
            says.append(f"{_n(mine)} tok/s against the others' {_n(med)}")
        if link_below(c):
            why.append("pcie")
            says.append(f"PCIe {c.get('pcie_gen')} ×{c.get('pcie_width')}, where it can do "
                        f"{c.get('pcie_gen_max')} ×{c.get('pcie_width_max')}")
        if under_default(c):
            why.append("power")
            says.append(f"its power limit {_n(c['power_limit_w'])} W, under its default "
                        f"{_n(c['power_default_w'])} W")
        u = me["use"].get("util_15m") if isinstance(me.get("use"), dict) else None
        if isinstance(u, (int, float)) and u < BUSY_UNDER:
            why.append("busy")
            says.append(f"{_n(u)}% busy over the last 15 min")
        if not why:
            continue
        hours, how = _saved(me, others, mine, theirs)
        text = "slow card — " + "; ".join(says) + (
            f"; replacing it now would save about {round(hours)} hour"
            f"{'s' if round(hours) != 1 else ''}" if hours is not None and hours >= 1 else "")
        out[me["key"]] = {"words": text, "why": why, "hours": hours,
                          "numbers": numbers(me, theirs, med, how)}
    return out


def _saved(me: dict, others: list[dict], mine: float | None, theirs: list[float]
           ) -> tuple[float | None, dict]:
    """(hours saved, how it was worked out) — None when there is no pace to
    work it from"""
    def pace_of(task: str, plan: float | None) -> tuple[float | None, str]:
        got = [s["pace"] for b in others for s in b.get("step") or []
               if s.get("task") == task and s.get("state") == "now"
               and isinstance(s.get("pace"), (int, float)) and s["pace"] > 0]
        if got:
            return statistics.median(got), "the others'"
        return (plan, "the plan's") if isinstance(plan, (int, float)) and plan > 0 else (None, "")
    step = me.get("step") or []
    paces = {s["task"]: pace_of(s["task"], s.get("plan_pace")) for s in step}
    if not step or any(p is None for p, _ in paces.values()):
        return None, {}
    now = next((s for s in step if s.get("state") == "now"), None)
    own = now.get("pace") if now and isinstance(now.get("pace"), (int, float)) \
        and now["pace"] > 0 else None
    if own is not None:
        ratio = own / paces[now["task"]][0]             # its pace against theirs
        basis = "its own pace"
    elif mine and theirs:
        ratio = statistics.median(theirs) / mine        # before it has one: the cards' speeds
        basis = "the cards' speeds"
    else:
        return None, {}
    left = 0.0
    for s in step:
        p = paces[s["task"]][0]
        if s.get("state") == "now":
            left += max(0, (s.get("answers") or 0) - (s.get("n") or 0)) * (
                own if own is not None else p * ratio)
        elif s.get("state") == "next":
            left += (s.get("answers") or 0) * p * ratio
    whole = [(s.get("answers") or 0, paces[s["task"]][0]) for s in step]
    h = hours_saved(left, whole)
    return h, {"left_h": round(left / 3600, 2), "new_h": round(
        (NEW_BOX_S + sum(n * p for n, p in whole)) / 3600, 2), "basis": basis,
        "paces": {t: (round(p, 2), w) for t, (p, w) in paces.items()},
        "labels": {s["task"]: s.get("label") or s["task"] for s in step},
        "own": round(own, 2) if own is not None else None}


def numbers(me: dict, theirs: list[float], med: float | None, how: dict) -> str:
    """the flag's numbers, for the board's hover"""
    parts = [words(me.get("check"))]
    if theirs:
        parts.append("the build's other boxes: " + ", ".join(_n(x) for x in sorted(theirs))
                     + " tok/s alone" + (f" (median {_n(med)})" if med else ""))
    u = me.get("use") if isinstance(me.get("use"), dict) else {}
    if use_words(u):
        parts.append("so far " + use_words(u)
                     + (f", {_n(u['util_15m'])}% over the last 15 min"
                        if u.get("util_15m") is not None else ""))
    if how:
        if how.get("own") is not None:
            parts.append(f"its pace {how['own']:.1f} s an answer")
        parts.append("; ".join(f"{how['labels'].get(t, t)}: {p:.1f} s an answer ({w})"
                               for t, (p, w) in how["paces"].items()))
        parts.append(f"left at its pace about {how['left_h']:.1f} h ({how['basis']}); a new "
                     f"box about {how['new_h']:.1f} h with half an hour to set it up")
    return " · ".join(x for x in parts if x)
