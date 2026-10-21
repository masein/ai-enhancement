"""26: our testers' phone checks, from Kevin's telemetry on R2 — the sync
(read-only, only the tests not here yet, each kept as it came), reading a
test as his own dashboard reads it, our names for the app's ids, and the
flags that keep an impossible speed, a charging battery or an app sent to
the background out of the medians. The fixture is the attached test's shape
with made-up ids, label and replies; R2 is a fake S3 server on 127.0.0.1
with made-up keys."""

from __future__ import annotations

import copy
import datetime as dt
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from service import config, r2, reports, telemetry

FIX = json.loads((Path(__file__).resolve().parent / "fixtures" / "phone" / "sample-test.json")
                 .read_text(encoding="utf-8"))
KEY_ID, SECRET = "fake-key-id-26", "fake-secret-26"
BUCKET, PREFIX = "teraformer-models", "telemetry/speedcheck/"


def a_test(test_id: str, *, install: str = "", decode=None, charging=None, stayed=None,
           outcome=None, model=None, device=None, received=None, temp_end=None,
           sustain=True) -> dict:
    """the fixture as another test: its ids, and what the case changes"""
    t = copy.deepcopy(FIX)
    t["test_id"] = test_id
    t["install_id"] = install or t["install_id"]
    for x in t["items"]:
        if x["id"] == "sustain" and decode is not None:
            x["decode_tps"] = decode
        if x["id"] == "sustain" and not sustain:
            x.clear()
            x.update({"id": "sustain", "status": "not_run", "finish": "skipped", "wall_ms": 0})
    c = t["conditions"]
    if charging is not None:
        c["charging"] = charging
    if stayed is not None:
        c["stayed_in_app"] = stayed
    if temp_end is not None:
        c["battery_temp_end"] = temp_end
    if outcome is not None:
        t["outcome"] = outcome
    if model is not None:
        t["model_id"] = model
    if device is not None:
        t["device"]["model"] = device
    if received is not None:
        t["received_at"] = received
    return t


def keep(root: Path, t: dict) -> Path:
    p = root / "phone" / "telemetry" / t["install_id"] / f"{t['test_id']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(t), encoding="utf-8")
    return p


@pytest.fixture
def here(tmp_path, monkeypatch):
    """BENCH_ROOT a temporary folder, our names a temporary file, no keys"""
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    for k in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.setattr(config, k, "")
    names = tmp_path / "names.json"
    names.write_text(json.dumps({
        "models": {"model-004": {"name": "step8804 Q4 (Qwen3.6-35B-A3B)",
                                 "row": "served/fx-phone-build", "what": "the 35B-A3B, Q4"}},
        "tiers": {"T0": "the first tier"}, "engines": {},
        "devices": {"CPH2747": "OnePlus 15", "SM-S938B": "Galaxy S25 Ultra"}}),
        encoding="utf-8")
    monkeypatch.setattr(telemetry, "NAMES", names)
    telemetry._memo.update(stamp=None)
    telemetry._names_memo.update(stamp=None)
    yield tmp_path
    telemetry._memo.update(stamp=None)
    telemetry._names_memo.update(stamp=None)


# ---------------------------------------------------------------------------
# a fake R2: ListObjectsV2 two keys a page, and the files; any write refused
# ---------------------------------------------------------------------------

class FakeR2:
    def __init__(self, objects: dict[str, bytes], fail: int | None = None):
        self.objects, self.fail = objects, fail
        self.gets: list[str] = []
        self.writes: list[str] = []
        self.auth: list[tuple[str, dict]] = []
        me = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _write(self):
                me.writes.append(self.path)
                self.send_response(405)
                self.end_headers()
            do_PUT = do_POST = do_DELETE = _write

            def do_GET(self):
                me.auth.append((self.path, dict(self.headers)))
                u = urllib.parse.urlsplit(self.path)
                if me.fail:
                    body = b"<Error><Code>AccessDenied</Code><Message>no</Message></Error>"
                    self.send_response(me.fail)
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if u.path == f"/{BUCKET}":
                    q = dict(urllib.parse.parse_qsl(u.query))
                    keys = sorted(k for k in me.objects if k.startswith(q.get("prefix", "")))
                    at = int(q.get("continuation-token") or 0)
                    page, more = keys[at:at + 2], at + 2 < len(keys)
                    body = ("<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult "
                            "xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
                            + "".join(f"<Contents><Key>{escape(k)}</Key></Contents>" for k in page)
                            + f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>"
                            + (f"<NextContinuationToken>{at + 2}</NextContinuationToken>"
                               if more else "") + "</ListBucketResult>").encode()
                else:
                    key = urllib.parse.unquote(u.path[len(f"/{BUCKET}/"):])
                    me.gets.append(key)
                    body = me.objects.get(key)
                    if body is None:
                        self.send_response(404)
                        self.end_headers()
                        return
                self.send_response(200)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.endpoint = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def keys_set(monkeypatch, endpoint: str) -> None:
    monkeypatch.setattr(config, "R2_ENDPOINT", endpoint)
    monkeypatch.setattr(config, "R2_ACCESS_KEY_ID", KEY_ID)
    monkeypatch.setattr(config, "R2_SECRET_ACCESS_KEY", SECRET)
    monkeypatch.setattr(config, "R2_BUCKET", BUCKET)
    monkeypatch.setattr(config, "R2_TELEMETRY_PREFIX", PREFIX)


# ---------------------------------------------------------------------------
# the sync
# ---------------------------------------------------------------------------

def test_the_sync_downloads_only_the_new_tests_and_keeps_each_as_it_came(here, monkeypatch):
    known = a_test("known1", install="inst-a")
    new1 = a_test("new1", install="inst-a")
    new1["a_field_from_later"] = {"kept": True}          # unknown fields are data, kept
    new2 = a_test("new2", install="inst-b")
    raw1 = json.dumps(new1, indent=2).encode()
    objects = {f"{PREFIX}inst-a/known1.json": json.dumps(known).encode(),
               f"{PREFIX}inst-a/new1.json": raw1,
               f"{PREFIX}inst-b/new2.json": json.dumps(new2).encode(),
               f"{PREFIX}inst-b/notes.txt": b"not a test",          # not one: left alone
               f"{PREFIX}inst-b/deeper/x.json": b"{}",               # nor this
               "telemetry/other/inst-c/t.json": b"{}"}               # outside the prefix
    keep(here, known)
    fake = FakeR2(objects)
    try:
        keys_set(monkeypatch, fake.endpoint)
        got = telemetry.sync(say=lambda *_: None)
        assert got == {"new": 2, "total": 3}
        # only the two new ones asked for — never the known one, never a write
        assert sorted(fake.gets) == [f"{PREFIX}inst-a/new1.json", f"{PREFIX}inst-b/new2.json"]
        assert fake.writes == []
        # three listing pages of two (the prefix's five keys), then the two files
        assert sum(1 for p, _ in fake.auth if "list-type=2" in p) == 3
        # kept byte for byte, by install and test id
        kept = here / "phone" / "telemetry" / "inst-a" / "new1.json"
        assert kept.read_bytes() == raw1
        assert telemetry.read(json.loads(kept.read_text()))["test_id"] == "new1"
        # every request signed, with the key id and never the secret
        for _, h in fake.auth:
            a = {k.lower(): v for k, v in h.items()}["authorization"]
            assert a.startswith(f"AWS4-HMAC-SHA256 Credential={KEY_ID}/")
            assert "/auto/s3/aws4_request" in a and SECRET not in a
        st = telemetry.status()
        assert st["set_up"] and st["new"] == 2 and st["total"] == 3
        assert st["line"].startswith("Phone checks: synced ") and st["line"].endswith(", 2 new")
        # again: nothing new, no file asked for
        fake.gets.clear()
        assert telemetry.sync(say=lambda *_: None) == {"new": 0, "total": 3}
        assert fake.gets == []
    finally:
        fake.close()


def test_without_the_keys_it_says_so_and_asks_nothing(here, monkeypatch):
    called = []
    monkeypatch.setattr(telemetry.r2.Bucket, "keys", lambda *a: called.append(a) or [])
    assert telemetry.configured() is False
    assert telemetry.status() == {"set_up": False, "line": telemetry.NOT_SET_UP}
    assert telemetry.NOT_SET_UP == "Phone checks: not set up — R2 keys missing in .env"
    assert telemetry.sync() == {"set_up": False, "line": telemetry.NOT_SET_UP}
    started = []
    monkeypatch.setattr(telemetry.threading, "Thread", lambda *a, **k: started.append(k))
    telemetry.start()
    assert called == [] and started == []
    rep = reports.files()["testers"]
    got = telemetry.report(rep)
    assert got["sync"]["line"] == telemetry.NOT_SET_UP and got["tests"] == []
    assert telemetry.listing_line()["sync"] == telemetry.NOT_SET_UP


def test_a_failed_sync_says_why_with_the_last_good_one_and_never_the_keys(here, monkeypatch):
    ok = FakeR2({f"{PREFIX}inst-a/t1.json": json.dumps(a_test("t1", install="inst-a")).encode()})
    try:
        keys_set(monkeypatch, ok.endpoint)
        assert telemetry.sync(say=lambda *_: None)["new"] == 1
    finally:
        ok.close()
    bad = FakeR2({}, fail=403)
    said = []
    try:
        keys_set(monkeypatch, bad.endpoint)
        got = telemetry.sync(say=said.append)
    finally:
        bad.close()
    assert got == {"error": "the listing answered 403 (AccessDenied)"}
    line = telemetry.status()["line"]
    assert line.startswith("Phone checks: the last sync failed — the listing answered 403 "
                           "(AccessDenied); the last good one ")
    # unreachable: said, without the endpoint
    keys_set(monkeypatch, "http://127.0.0.1:9")
    got = telemetry.sync(say=said.append)
    assert got["error"].startswith("the listing: the bucket couldn't be reached")
    for text in [line, *said, json.dumps(telemetry.report(reports.files()["testers"])),
                 (here / "phone" / "telemetry" / "_sync.json").read_text()]:
        assert KEY_ID not in text and SECRET not in text
    assert "127.0.0.1:9" not in got["error"]


def test_a_higher_schema_is_kept_raw_and_said(here):
    keep(here, a_test("t1"))
    two = {**a_test("t2"), "schema": 2, "something_new": [1, 2]}
    p = keep(here, two)
    before = p.read_bytes()
    ts, unread = telemetry.tests()
    assert [t["test_id"] for t in ts] == ["t1"]
    assert unread == [{"test_id": "t2", "install_id": FIX["install_id"], "schema": 2,
                       "why": "schema 2: not read yet"}]
    assert telemetry.one("t2") == {"test_id": "t2", "unread": "schema 2: not read yet"}
    assert p.read_bytes() == before                      # kept as it came
    assert telemetry.report(reports.files()["testers"])["unread"][0]["why"] == \
        "schema 2: not read yet"


def test_the_signature_is_pinned():
    """r2.sign matched AWS's two published SigV4 examples when it was written
    (the GET with a Range header, and the listing); this pins it for made-up
    keys, so a change to it shows"""
    h = r2.sign("GET", "https://example.invalid/bucket?list-type=2&prefix=telemetry%2F",
                KEY_ID, SECRET, now=dt.datetime(2026, 10, 21, 12, 0, tzinfo=dt.timezone.utc))
    assert h["x-amz-date"] == "20261021T120000Z" and h["x-amz-content-sha256"] == r2.EMPTY
    assert h["authorization"] == (
        f"AWS4-HMAC-SHA256 Credential={KEY_ID}/20261021/auto/s3/aws4_request, "
        "SignedHeaders=host;x-amz-content-sha256;x-amz-date, "
        f"Signature={PINNED}")


PINNED = "ff80316f968847c11545a440a427016296130a5cc6ddad3d8084e6ebaf134534"


# ---------------------------------------------------------------------------
# reading, as Kevin's dashboard reads it
# ---------------------------------------------------------------------------

def test_reading_the_attached_shape(here):
    t = telemetry.read(FIX)
    assert (t["decode"], t["decode_from"]) == (13.0, "sustain")
    assert (t["prefill_cold"], t["prefill_warm"]) == (32.6, 36.2)
    assert t["answers"] == {"done": 8, "run": 8}           # web-no and web-yes weren't run
    assert t["battery"] == {"start": 32.3, "end": 33.7, "rise": 1.4}
    assert t["first_load_s"] == 10.7 and t["ttft_ms"] == 5030
    assert (t["peak_swap_mb"], t["engine_mem_mb"], t["mem_min_mb"]) == (6452, 4565, 5553)
    assert t["sustained"] == {"first": 13.2, "last": 12.6, "pct": -4.5}
    assert (t["tester"], t["phone"]["name"], t["phone"]["ram_gb"]) == ("tester-a", "OnePlus 15",
                                                                         "16")
    assert (t["app"], t["ok"], t["charging"], t["stayed_in_app"]) == ("1.4", True, False, True)
    assert t["flags"] == {"speed": [], "battery": [], "impossible": False}
    # sustain not run: think-math's decode
    t = telemetry.read(a_test("t2", sustain=False))
    assert (t["decode"], t["decode_from"]) == (12.7, "think-math")
    assert t["answers"] == {"done": 7, "run": 7} and t["sustained"]["pct"] is None


def test_names_ours_and_an_id_not_in_them_shown_as_itself(here):
    t = telemetry.read(FIX)
    assert t["model"] == {"id": "model-004", "name": "step8804 Q4 (Qwen3.6-35B-A3B)",
                          "row": "served/fx-phone-build", "what": "the 35B-A3B, Q4",
                          "named": True}
    assert t["tier"]["name"] == "the first tier"
    assert t["engine"] == {"id": "e-00000001", "name": "e-00000001 (not named yet)",
                           "named": False}
    t = telemetry.read(a_test("t2", model="model-009", device="23078PND5G"))
    assert t["model"]["name"] == "model-009 (not named yet)" and "row" not in t["model"]
    assert t["phone"]["name"] == "23078PND5G (not named yet)"
    # the mapped model lands on its row: the phone build's measured numbers
    keep(here, FIX)
    keep(here, a_test("t2", model="model-009"))
    rows = telemetry.by_row()
    assert list(rows) == ["served/fx-phone-build"]
    (p,) = rows["served/fx-phone-build"]
    assert (p["phone"]["name"], p["n"], p["decode"], p["engine_mem_mb"]) == (
        "OnePlus 15", 1, 13.0, 4565)
    # the repository's own file: plain names, the two phones we're sure of
    real = json.loads((Path(telemetry.__file__).resolve().parent.parent / "eval_tasks" / "phone"
                       / "names.json").read_text())
    assert real["devices"] == {"CPH2747": "OnePlus 15", "SM-S938B": "Galaxy S25 Ultra"}
    assert set(real) >= {"models", "tiers", "engines", "devices"}


# ---------------------------------------------------------------------------
# the flags
# ---------------------------------------------------------------------------

def test_an_impossible_speed_is_flagged_and_out_of_the_medians(here):
    keep(here, a_test("ok1", received="2026-10-20T10:00:00Z"))
    keep(here, a_test("fast", decode=400, received="2026-10-20T11:00:00Z"))
    ts, _ = telemetry.tests()
    fast = next(t for t in ts if t["test_id"] == "fast")
    assert fast["flags"]["impossible"] is True
    assert fast["flags"]["speed"] == ["decode 400 tok/s: no phone does that with a 30B-class "
                                      "model (above 60)"]
    (p,) = telemetry.per_phone(ts)
    assert (p["n"], p["tests"], p["decode"]) == (1, 2, 13.0)
    # a model not named as 30B-class: nothing to hold it to
    t = telemetry.read(a_test("x", decode=400, model="model-009"))
    assert t["flags"]["impossible"] is False
    assert telemetry.IMPOSSIBLE == {"30B-class": {"decode": 60.0, "prefill": 2000.0}}


def test_charging_is_out_of_the_battery_medians_only_and_background_out_of_the_speeds(here):
    keep(here, a_test("ok1", temp_end=33.3))                          # +1.0 °C, 13.0 tok/s
    keep(here, a_test("charging", decode=11.0, charging=True, temp_end=40.3))   # +8.0
    keep(here, a_test("away", decode=5.0, stayed=False, temp_end=35.3))         # +3.0
    ts, _ = telemetry.tests()
    by = {t["test_id"]: t for t in ts}
    assert by["charging"]["flags"] == {"speed": [], "impossible": False, "battery": [
        "it ran while charging: its battery temperatures aren't the battery's alone"]}
    assert by["away"]["flags"] == {"battery": [], "impossible": False, "speed": [
        "the app didn't stay in the foreground: its speeds aren't the phone's"]}
    (p,) = telemetry.per_phone(ts)
    assert p["n"] == 3
    assert p["decode"] == 12.0                     # 13.0 and 11.0; not the 5.0 in the background
    assert p["temp_rise"] == 2.0                   # +1.0 and +3.0; not the +8.0 while charging


def test_what_failed_grouped_by_phone_and_app(here):
    keep(here, a_test("ok1"))
    keep(here, a_test("x1", outcome="server_crash", device="23078PND5G"))
    keep(here, a_test("x2", outcome="server_crash", device="23078PND5G"))
    keep(here, a_test("s1", outcome="server_crash", device="SM-S938B"))
    d = telemetry.report(reports.files()["testers"])
    assert d["tiles"] == {"tests": 4, "phones": 3, "testers": 1, "unfinished": 3}
    assert [(f["phone"]["name"], f["app"], f["n"], f["outcomes"]) for f in d["failed"]] == [
        ("23078PND5G (not named yet)", "1.4", 2, ["server_crash"]),
        ("Galaxy S25 Ultra", "1.4", 1, ["server_crash"])]
    # a test that didn't finish is in no median
    p = next(x for x in d["perPhone"] if x["phone"]["code"] == "SM-S938B")
    assert (p["n"], p["tests"], p["decode"]) == (0, 1, None)


def test_a_test_opened_has_every_check_and_its_reply(here):
    keep(here, FIX)
    t = telemetry.one(FIX["test_id"])
    assert [c["id"] for c in t["checks"]] == [x["id"] for x in FIX["items"]]
    assert t["checks"][0]["text"] == "Ready"
    assert telemetry.one("../etc/passwd") is None and telemetry.one("nope") is None
