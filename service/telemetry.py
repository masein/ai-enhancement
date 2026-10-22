"""26: our testers' phone checks, from Kevin's telemetry
(docs/prompts/phase-26-phone-telemetry-from-r2.md).

The team's phone app (Etachat) checks itself once on each tester's phone and
uploads one JSON a test to Cloudflare R2: `<prefix><install_id>/<test_id>.json`.
Kevin's format is the source of truth; nothing here asks his app to change.

- **The sync**, every SYNC_S seconds while the R2 keys are in .env: the
  prefix listed, and only the tests not here yet downloaded, by test id; each
  kept raw as it came in BENCH_ROOT/phone/telemetry/<install_id>/<test_id>.json.
  Nothing is ever written to R2. Without the keys it says so and does nothing.
- **Reading** a test (schema 1) the way Kevin's own Testers dashboard does —
  answers are done over run, not_run never counted; prefill cold then warm;
  decode the sustain check's (think-math's when sustain didn't run) — so the
  two never disagree. A field we don't know is kept in the raw file; a higher
  schema is kept raw and said ("schema 2: not read yet").
- **Names** are ours (eval_tasks/phone/names.json): an app id not in it shows
  as itself, "(not named yet)", never guessed.
- **Flags**: a speed a phone can't do (IMPOSSIBLE), a test run while
  charging (its battery columns), one where the app left the foreground (its
  speed columns) — greyed, kept out of the medians it would skew, the reason
  said."""

from __future__ import annotations

import json
import re
import statistics
import threading
import time
from pathlib import Path
from xml.etree.ElementTree import ParseError

from . import config, r2

NAMES = Path(__file__).resolve().parent.parent / "eval_tasks" / "phone" / "names.json"
NOT_SET_UP = "Phone checks: not set up — R2 keys missing in .env"
# 26: what a phone can't do. PocketPal's lesson: its public leaderboard shows
# impossible speeds, and nobody can tell which. A 30B-class model (named as
# one, 25B to 40B) on a phone decodes well under 60 tok/s and prefills well
# under 2,000 — a test above either measured something else
IMPOSSIBLE = {"30B-class": {"decode": 60.0, "prefill": 2000.0}}
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_lock = threading.Lock()
_memo: dict = {"stamp": None, "tests": [], "unread": []}
_names_memo: dict = {"stamp": None, "names": {}}
_stop = threading.Event()


def root() -> Path:
    return Path(config.BENCH_ROOT) / "phone" / "telemetry"


def configured() -> bool:
    return bool(config.R2_ENDPOINT and config.R2_ACCESS_KEY_ID and config.R2_SECRET_ACCESS_KEY)


def bucket() -> r2.Bucket:
    return r2.Bucket(config.R2_ENDPOINT, config.R2_ACCESS_KEY_ID, config.R2_SECRET_ACCESS_KEY,
                     config.R2_BUCKET)


# ---------------------------------------------------------------------------
# the sync
# ---------------------------------------------------------------------------

def _status_file() -> Path:
    return root() / "_sync.json"


def status() -> dict:
    """whether it is set up, and the last sync: when it last worked, what it
    found, and why the last one failed when it did"""
    if not configured():
        return {"set_up": False, "line": NOT_SET_UP}
    try:
        st = json.loads(_status_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}
    ok = st.get("ok_at")
    when = time.strftime("%d %b %H:%M", time.localtime(ok)) if ok else ""
    if st.get("error") and (st.get("tried_at") or 0) >= (ok or 0):
        line = (f"Phone checks: the last sync failed — {st['error']}"
                + (f"; the last good one {when}" if ok else "; none has worked yet"))
    elif ok:
        line = f"Phone checks: synced {when}, {st.get('new', 0)} new"
    else:
        line = "Phone checks: set up, not synced yet"
    return {"set_up": True, "line": line, **st}


def _write_status(**kw) -> None:
    p = _status_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    tmp = p.with_suffix(".part")
    tmp.write_text(json.dumps({**old, **kw}), encoding="utf-8")
    tmp.replace(p)


def have() -> set[str]:
    """the test ids kept here"""
    return {p.stem for p in root().glob("*/*.json")}


def sync(b: r2.Bucket | None = None, say=print) -> dict:
    """list the prefix and download the tests not here yet — never a write
    to R2. {new, total} or {error}"""
    if b is None and not configured():
        return {"set_up": False, "line": NOT_SET_UP}
    prefix = config.R2_TELEMETRY_PREFIX
    now = time.time()
    try:
        b = b or bucket()
        known, new = have(), 0
        for key in b.keys(prefix):
            rel = key[len(prefix):] if key.startswith(prefix) else ""
            parts = rel.split("/")
            if len(parts) != 2 or not parts[1].endswith(".json"):
                continue
            inst, test = parts[0], parts[1][:-5]
            if not SAFE.fullmatch(inst) or not SAFE.fullmatch(test) or test in known:
                continue
            raw = b.get(key)
            out = root() / inst / f"{test}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".part")
            tmp.write_bytes(raw)
            tmp.replace(out)
            known.add(test)
            new += 1
        _write_status(ok_at=now, tried_at=now, new=new, total=len(known), error=None)
        if new:
            say(f"[phone checks] {new} new test{'s' if new != 1 else ''}, {len(known)} in all")
        return {"new": new, "total": len(known)}
    except (r2.R2Error, ParseError, OSError) as e:
        _write_status(tried_at=now, error=str(e)[:300])
        say(f"[phone checks] the sync failed: {e}")
        return {"error": str(e)}


def _loop() -> None:
    _stop.wait(5)
    while not _stop.is_set():
        try:
            sync()
        except Exception as e:                  # noqa: BLE001 — the loop never dies
            print(f"[phone checks] the sync stopped: {e!r}")
        _stop.wait(max(60, config.TELEMETRY_SYNC_S))


def start() -> None:
    """the sync every TELEMETRY_SYNC_S seconds — only while the keys are set"""
    if configured() and config.TELEMETRY_SYNC_S > 0:
        _stop.clear()
        threading.Thread(target=_loop, name="phone-checks", daemon=True).start()


def stop() -> None:
    _stop.set()


# ---------------------------------------------------------------------------
# names, ours: an app id not in the file shows as itself, never guessed
# ---------------------------------------------------------------------------

def names() -> dict:
    try:
        st = NAMES.stat()
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        return {}
    if _names_memo["stamp"] != stamp:
        try:
            _names_memo.update(stamp=stamp, names=json.loads(NAMES.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            _names_memo.update(stamp=stamp, names={})
    return _names_memo["names"]


def named(kind: str, key) -> dict:
    """{id, name, named} — an id the file doesn't name shows as itself"""
    got = (names().get(kind) or {}).get(str(key)) if key not in (None, "") else None
    if isinstance(got, str):
        got = {"name": got}
    if isinstance(got, dict) and got.get("name"):
        return {"id": key, **got, "named": True}
    return {"id": key, "name": f"{key} (not named yet)" if key not in (None, "") else "—",
            "named": False}


def model_class(m: dict) -> str:
    """30B-class when its name (or what it is) says 25B to 40B; '' when it
    doesn't say"""
    sizes = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*B\b",
                                          f"{m.get('name') or ''} {m.get('what') or ''}")]
    return "30B-class" if sizes and 25 <= max(sizes) <= 40 else ""


# ---------------------------------------------------------------------------
# a test, read as Kevin's dashboard reads it
# ---------------------------------------------------------------------------

def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _r1(x):
    return round(x, 1) if isinstance(x, (int, float)) else None


def read(raw: dict, path: Path | None = None) -> dict:
    """one test's row: Kevin's columns and ours, its flags, its checks"""
    dev, cond = raw.get("device") or {}, raw.get("conditions") or {}
    items = [x for x in raw.get("items") or [] if isinstance(x, dict)]
    by = {x.get("id"): x for x in items}

    def done(i):
        x = by.get(i) or {}
        return x if x.get("status") == "done" else {}
    sus, think = done("sustain"), done("think-math")
    dec_from = "sustain" if _num(sus.get("decode_tps")) is not None else (
        "think-math" if _num(think.get("decode_tps")) is not None else "")
    dec = _num((sus if dec_from == "sustain" else think).get("decode_tps")) if dec_from else None
    first, last = _num(sus.get("tps_first_third")), _num(sus.get("tps_last_third"))
    bt0, bt1 = _num(cond.get("battery_temp_start")), _num(cond.get("battery_temp_end"))
    model = named("models", raw.get("model_id"))
    ran = [x for x in items if x.get("status") != "not_run"]
    fl = cond.get("first_load_ms") if _num(cond.get("first_load_ms")) is not None else raw.get("load_ms")
    t = {
        "test_id": raw.get("test_id") or (path.stem if path else ""),
        "install_id": raw.get("install_id") or (path.parent.name if path else ""),
        "received": raw.get("received_at") or "", "taken_at": raw.get("taken_at"),
        "tester": raw.get("label") or "", "channel": raw.get("channel") or "",
        "phone": {**named("devices", dev.get("model")), "code": dev.get("model") or "",
                  "maker": dev.get("manufacturer") or "", "os": dev.get("os") or "",
                  "ram_gb": dev.get("ram_class_gb")},
        "app": raw.get("app_version") or "", "platform": raw.get("platform") or "",
        "engine": named("engines", raw.get("engine_build")),
        "tier": named("tiers", raw.get("tier_id")), "model": model,
        "outcome": raw.get("outcome") or "", "ok": raw.get("outcome") == "ok",
        "first_load_s": _r1(_num(fl) / 1000) if _num(fl) is not None else None,
        "ttft_ms": _num(done("speed-warm").get("ttft_ms")),
        "prefill_cold": _num(done("speed-cold").get("prefill_tps")),
        "prefill_warm": _num(done("speed-warm").get("prefill_tps")),
        "decode": dec, "decode_from": dec_from,
        "sustained": {"first": first, "last": last,
                      "pct": _r1((last - first) / first * 100) if first and last is not None
                      else None},
        "engine_mem_mb": _num(cond.get("engine_mem_mb")),
        "peak_swap_mb": _num(cond.get("peak_swap_mb")),
        "mem_min_mb": _num(cond.get("mem_avail_min_mb")),
        "battery": {"start": _r1(bt0), "end": _r1(bt1),
                    "rise": _r1(bt1 - bt0) if bt0 is not None and bt1 is not None else None},
        "thermal_max": cond.get("thermal_max"),
        "answers": {"done": sum(1 for x in items if x.get("status") == "done"), "run": len(ran)},
        "charging": cond.get("charging") is True,
        "stayed_in_app": cond.get("stayed_in_app") is not False,
        "checks": [{k: x.get(k) for k in ("id", "status", "finish", "wall_ms", "ttft_ms",
                                          "prefill_tps", "decode_tps") if k in x} for x in items]}
    t["flags"] = flags(t)
    return t


def flags(t: dict) -> dict:
    """{speed: [why], battery: [why], impossible: bool} — the reasons a test
    is greyed, and out of the medians it would skew"""
    speed, battery, impossible = [], [], False
    lim = IMPOSSIBLE.get(model_class(t["model"]))
    if lim:
        if (t["decode"] or 0) > lim["decode"]:
            impossible = True
            speed.append(f"decode {t['decode']:g} tok/s: no phone does that with a 30B-class "
                         f"model (above {lim['decode']:g})")
        pre = max(t["prefill_cold"] or 0, t["prefill_warm"] or 0)
        if pre > lim["prefill"]:
            impossible = True
            speed.append(f"prefill {pre:g} tok/s: no phone does that with a 30B-class model "
                         f"(above {lim['prefill']:,.0f})")
    if not t["stayed_in_app"]:
        speed.append("the app didn't stay in the foreground: its speeds aren't the phone's")
    if t["charging"]:
        battery.append("it ran while charging: its battery temperatures aren't the battery's alone")
    return {"speed": speed, "battery": battery, "impossible": impossible}


def tests() -> tuple[list[dict], list[dict]]:
    """(every test read, newest first; the files not read and why) — read
    again only when a file came or changed"""
    fs = sorted(p for p in root().glob("*/*.json")) if root().is_dir() else []
    stamp = []
    for f in fs:
        try:
            st = f.stat()
        except OSError:
            continue
        stamp.append((str(f), st.st_mtime_ns, st.st_size))
    stamp.append(("names", *(_names_memo["stamp"] or (0, 0))))
    names()
    stamp[-1] = ("names", *(_names_memo["stamp"] or (0, 0)))
    with _lock:
        if _memo["stamp"] == stamp:
            return _memo["tests"], _memo["unread"]
    out, unread = [], []
    for f in fs:
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unread.append({"test_id": f.stem, "install_id": f.parent.name, "why": "not JSON"})
            continue
        sch = raw.get("schema") if isinstance(raw, dict) else None
        if sch != 1:
            unread.append({"test_id": f.stem, "install_id": f.parent.name, "schema": sch,
                           "why": f"schema {sch}: not read yet"})
            continue
        out.append(read(raw, f))
    out.sort(key=lambda t: (t["received"] or "", t["taken_at"] or 0), reverse=True)
    with _lock:
        _memo.update(stamp=stamp, tests=out, unread=unread)
    return out, unread


def one(test_id: str) -> dict | None:
    """a test opened: every check with its reply (our app's own output)"""
    if not SAFE.fullmatch(str(test_id)):
        return None
    for f in root().glob(f"*/{test_id}.json"):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if raw.get("schema") != 1:
            return {"test_id": test_id, "unread": f"schema {raw.get('schema')}: not read yet"}
        t = read(raw, f)
        t["checks"] = [x for x in raw.get("items") or [] if isinstance(x, dict)]
        return t
    return None


# ---------------------------------------------------------------------------
# medians, per phone and model; what failed
# ---------------------------------------------------------------------------

def _med(xs: list) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 1) if xs else None


def medians(ts: list[dict]) -> dict:
    """the medians over finished tests: speeds without the speed-flagged,
    temperature rise without the battery-flagged, none of an impossible one"""
    ok = [t for t in ts if t["ok"] and not t["flags"]["impossible"]]
    sp = [t for t in ok if not t["flags"]["speed"]]
    bt = [t for t in ok if not t["flags"]["battery"]]
    days = sorted(t["received"][:10] for t in ok if t["received"])
    return {"n": len(ok), "decode": _med([t["decode"] for t in sp]),
            "prefill_warm": _med([t["prefill_warm"] for t in sp]),
            "ttft_ms": _med([t["ttft_ms"] for t in sp]),
            "engine_mem_mb": _med([t["engine_mem_mb"] for t in ok]),
            "temp_rise": _med([t["battery"]["rise"] for t in bt]),
            "first": days[0] if days else "", "last": days[-1] if days else ""}


def per_phone(ts: list[dict]) -> list[dict]:
    groups: dict[tuple, list] = {}
    for t in ts:
        groups.setdefault((t["phone"]["code"], t["model"]["id"]), []).append(t)
    out = []
    for (code, mid), g in groups.items():
        out.append({"phone": g[0]["phone"], "model": g[0]["model"], **medians(g),
                    "tests": len(g)})
    return sorted(out, key=lambda x: (str(x["phone"]["name"]), str(x["model"]["name"])))


def failed(ts: list[dict]) -> list[dict]:
    groups: dict[tuple, list] = {}
    for t in ts:
        if not t["ok"]:
            groups.setdefault((t["phone"]["code"], t["app"]), []).append(t)
    return [{"phone": g[0]["phone"], "app": app, "n": len(g),
             "outcomes": sorted({t["outcome"] for t in g}),
             "tests": [{"test_id": t["test_id"], "received": t["received"], "tester": t["tester"],
                        "outcome": t["outcome"]} for t in g]}
            for (code, app), g in sorted(groups.items(), key=lambda kv: -len(kv[1]))]


def by_row() -> dict[str, list[dict]]:
    """{a board row the names map a model to: its medians per phone} — what
    the phone build's page and the On-device report show as measured"""
    ts, _ = tests()
    out: dict[str, list[dict]] = {}
    for p in per_phone(ts):
        row = p["model"].get("row")
        if row and p["n"]:
            out.setdefault(row, []).append(p)
    return out


def checkpoint_rows(cp: str, ids: list[str]) -> list[str]:
    """27c: the board rows of an app model's checkpoint (names.json's
    `checkpoint`): a served id as it is, or a checkpoint's name ("step8804")
    — every served row named it, or it and a dash ("served/step8804-lda-2610")"""
    cp = (cp or "").strip()
    if not cp:
        return []
    if "/" in cp:
        return [i for i in ids if i == cp]
    return [i for i in ids if i.startswith("served/")
            and (i.split("/", 1)[1] == cp or i.split("/", 1)[1].startswith(cp + "-"))]


def _row_sha(rec: dict) -> str:
    """a served row's file's sha256, as the board pinned it"""
    fs = rec.get("file_sha256") if isinstance(rec.get("file_sha256"), dict) else {}
    gp = rec.get("gguf_pin") if isinstance(rec.get("gguf_pin"), dict) else {}
    return str(gp.get("sha256") or fs.get("sha256") or "").strip().lower()


def by_checkpoint() -> list[dict]:
    """27c: each named app model with a checkpoint — the board rows of that
    checkpoint (thinking on or off), whether each is the same file (Kevin's
    sha256 in names.json against the row's pin; until it is there, not yet
    checked), its medians per phone over its finished tests, and whether it
    is a variant (another app model of the same checkpoint, named otherwise:
    model-003, step4000-denseq8) — never mixed with its checkpoint's own"""
    from . import db
    ts, _ = tests()
    per = per_phone(ts)
    recs = {r["id"]: r for r in db.served_all() if isinstance(r, dict) and r.get("id")}
    models = {k: v for k, v in (names().get("models") or {}).items()
              if isinstance(v, dict) and str(v.get("checkpoint") or "").strip()}
    out = []
    for mid, m in models.items():
        cp = str(m["checkpoint"]).strip()
        rows = checkpoint_rows(cp, list(recs))
        sha = str(m.get("sha256") or "").strip().lower()
        same = [k for k, v in models.items() if str(v["checkpoint"]).strip() == cp]
        name = m.get("name") or mid
        out.append({"model": {"id": mid, "name": name, "what": m.get("what") or "",
                              "checkpoint": cp},
                    "variant": len(same) > 1 and name != cp.split("/")[-1],
                    "rows": rows, "checked": {r: bool(sha) and sha == _row_sha(recs[r])
                                              for r in rows},
                    "phones": [p for p in per if p["model"]["id"] == mid and p["n"]]})
    return out


def checked_line(entry: dict, row: str) -> str:
    """27c: what a model page and the On-device report say of the app's file"""
    return ("the same file (sha256 checked)" if entry["checked"].get(row)
            else f"the app's {entry['model']['name']} file — not yet checked to be this "
                 "exact file")


def report(rep: dict) -> dict:
    """the place's payload: the sync's line, the tiles, every test, the
    medians per phone and model, what failed, and the files not read"""
    ts, unread = tests()
    return {"id": rep["id"], "kind": "phone_tests", "title": rep.get("title") or rep["id"],
            "what": rep.get("what") or "", "sync": status(),
            "tiles": {"tests": len(ts), "phones": len({t["phone"]["code"] for t in ts}),
                      "testers": len({t["tester"] for t in ts if t["tester"]}),
                      "unfinished": sum(1 for t in ts if not t["ok"])},
            "tests": ts, "perPhone": per_phone(ts), "failed": failed(ts), "unread": unread,
            "limits": IMPOSSIBLE}


def _plural(n: int, one: str) -> str:
    return f"{n} {one}" + ("" if n == 1 else "s")


def listing_line() -> dict:
    """the Reports list's card: what is in, and the sync's own line"""
    ts, _ = tests()
    st = status()
    ok = st.get("ok_at")
    return {"line": (f"{_plural(len(ts), 'test')} on {_plural(len({t['phone']['code'] for t in ts}), 'phone')}"
                     f" by {_plural(len({t['tester'] for t in ts if t['tester']}), 'tester')}"
                     if ts else "No tests here yet"),
            "sync": st["line"], "changed": time.strftime("%d %b %H:%M", time.localtime(ok))
            if ok else "", "changed_at": ok}


def main(argv: list[str] | None = None) -> int:
    """`python -m service.telemetry sync`: one sync now, said"""
    import sys
    args = sys.argv[1:] if argv is None else argv
    if args[:1] != ["sync"]:
        print("usage: python -m service.telemetry sync")
        return 2
    got = sync()
    print(status()["line"] if "error" not in got else f"failed: {got['error']}")
    return 0 if "error" not in got and got.get("set_up") is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
