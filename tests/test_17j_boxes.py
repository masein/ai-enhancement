"""17j, part 3: the boxes and the fetch (docs/prompts/phase-17j-eighth-review-
hle-cases-runs-page.md, points 14 to 21). A test a point, each failing on
0abb757. The boxes are invented listings and a stand-in ssh; nothing is
fetched, no model runs, no paid API is called."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from fake_openai import FakeServer
from service import db, served
from test_12q_devicemark_runs import ME, svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import ROW, SERVED, TASK, box, bundle_of, register, run_box  # noqa: F401
from test_17b_review import imported
from test_17e_review import a_bundle
from test_17f_review import listing_of
from test_17g_review import MMLU, PHONE, progress_of, with_steps
from test_17i_boxes import PARTS, Killed, step

REPO = Path(__file__).resolve().parents[1]


def main_of(ff, key, dest, *more):
    return ff.main(["--key", str(key), "--dest", str(dest), "--sha", PHONE, "--no-board", *more])


def world(tmp_path, monkeypatch, listing, compare=None):
    """the fetch against boxes whose listing `listing(host, round)` gives (None:
    not reached); each run of main() a round of its own"""
    import frontier_fetch as ff
    calls, state = [], {"round": 0}

    def run(cmd, cwd=None, timeout=None, stdin=None):
        calls.append(cmd)
        if cmd[0] == "ssh":
            host = next(x for x in cmd if "@" in x).split("@")[1]
            got = listing(host, state["round"])
            return (0, got) if got is not None else (255, f"ssh: connect to host {host}: refused")
        if cmd[0] == "scp":
            Path(cmd[-1]).write_bytes(b"x")
            return 0, ""
        if "--served" in cmd:
            return 0, '["served/phone"]'
        if cmd[:9] == ff.COMPARE:
            return compare or (0, "The same: the box answers 42.4% right …\n")
        return 0, "the row served/phone · thinking: imported"
    monkeypatch.setattr(ff, "run", run)
    monkeypatch.setattr(ff, "keep_sudo", lambda: None, raising=False)
    key = tmp_path / "id"
    key.write_text("k")
    return ff, calls, key, state


# ---------------------------------------------------------------------------
# 14: "safe" keeps its time, is forgotten when reached with work, destroyed after two rounds
# ---------------------------------------------------------------------------

def test_14_a_safe_box_given_new_work_then_missed_once_is_not_destroyed(tmp_path, monkeypatch,
                                                                         capsys):
    there = tmp_path / "there"
    there.mkdir()
    b = a_bundle(there / "frontier-served__phone-thinking-on-mmlu-pro.tar.gz", "served/phone")
    whole = progress_of("phone", "A5-1", "whole", tasks=[MMLU], bundle={"name": b.name})

    def listing(host, rnd):
        if rnd == 0:
            return with_steps(listing_of({f"/workspace/phone/A5-1/{b.name}": b}, [whole]),
                              [("phone", "A5-1")])
        return None                                       # the next run: not reached
    ff, calls, key, state = world(tmp_path, monkeypatch, listing)
    dest = tmp_path / "b" / "bundles"
    monkeypatch.setattr(ff, "copy", lambda *a, **k: (True, "here already"))
    assert main_of(ff, key, dest, "1.1.1.1:41") == 0
    e = json.loads((dest.parent / "safe-boxes.json").read_text())
    assert list(e.values())[0]["at"] > time.time() - 60       # 0abb757: a bare list, no time
    state["round"] = 1
    capsys.readouterr()
    # started again, the box not reached once: not destroyed yet, and not done
    assert main_of(ff, key, dest, "1.1.1.1:41") == 1           # 0abb757: 0, "destroyed: done"
    assert "destroyed if it isn't reached on the next round either" in capsys.readouterr().out
    # 18b: read safe before this fetch started: destroyed only once unreached
    # for 30 minutes. 18c point 15: and only by misses this fetch counted —
    # started again, its earlier misses don't count
    assert main_of(ff, key, dest, "1.1.1.1:41") == 1
    assert "destroyed if it isn't reached on the next round either" in capsys.readouterr().out
    # one fetch, a round every 20 minutes: destroyed once unreached 30 of them
    real_time, real_sleep, clock = time.time, time.sleep, [time.time()]
    monkeypatch.setattr(ff.time, "time", lambda: clock[0])
    monkeypatch.setattr(ff.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + 20 * 60))
    try:
        assert main_of(ff, key, dest, "--every", "20m", "1.1.1.1:41") == 0
    finally:
        monkeypatch.setattr(ff.time, "time", real_time)
        monkeypatch.setattr(ff.time, "sleep", real_sleep)
    out = capsys.readouterr().out
    assert "destroyed once it has been unreached for 30 min" in out
    assert "rounds — destroyed: done" in out
    # a box reached with a step not home is forgotten, whatever it read before
    (dest.parent / "safe-boxes.json").write_text(json.dumps(e))
    state["round"] = 0
    monkeypatch.setattr(ff, "one_box", lambda *a, **k: (False, False, ["x: NOT safe"], [], True))
    main_of(ff, key, dest, "1.1.1.1:41")
    assert all(v.get("gone") for v in json.loads((dest.parent / "safe-boxes.json")
                                                 .read_text()).values())


# ---------------------------------------------------------------------------
# 15: --abandoned that matches nothing is named; one still writing keeps the box not safe
# ---------------------------------------------------------------------------

def test_15_abandoned_names_what_matched_nothing_and_a_step_still_writing_isnt_hidden(
        tmp_path, monkeypatch, capsys):
    there = tmp_path / "there"
    there.mkdir()
    still = progress_of("phone", "A3-2", "asking", tasks=[MMLU], line="MMLU-Pro 25 of 539")

    def listing(host, rnd):
        return with_steps(listing_of({}, [still]), [("phone", "A3-2")])
    ff, calls, key, state = world(tmp_path, monkeypatch, listing)
    code = main_of(ff, key, tmp_path / "b" / "bundles", "--abandoned", "phone/A3-2",
                   "--abandoned", "phone/A3-9", "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "--abandoned phone/A3-9 matched no step on the boxes reached" in out   # 0abb757: quiet
    assert "phone/A3-2 is given as --abandoned but is still asking" in out, out
    assert "done, safe to destroy" not in out and code == 0     # 0abb757: safe, hidden
    assert "MMLU-Pro 25 of 539" in out                          # its state, printed


# ---------------------------------------------------------------------------
# 16: --parity compares the copy here once the box is gone
# ---------------------------------------------------------------------------

def test_16_parity_given_after_the_box_is_gone_compares_the_copy_here(tmp_path, monkeypatch,
                                                                        capsys):
    # 18b: "not the same" is compare's own exit code (3); 1 is no verdict
    ff, calls, key, state = world(tmp_path, monkeypatch, lambda host, rnd: None,
                                  compare=(3, "Not the same setup: 38.0% against 42.4%\n"))
    dest = tmp_path / "b" / "bundles"
    (dest.parent / "parity").mkdir(parents=True)
    (dest.parent / "parity" / "orig-box.jsonl").write_text('{"parity_of": {}}\n')
    server = tmp_path / "orig-server-500.jsonl"
    server.write_text("{}\n")
    code = main_of(ff, key, dest, "--parity", f"served/orig={server}", "1.1.1.1:41")
    out = capsys.readouterr().out
    assert any(c[:9] == ff.COMPARE for c in calls)              # 0abb757: never compared
    assert "orig's parity: Not the same setup" in out and code == 1
    v = json.loads((dest.parent / "parity" / "verdicts.json").read_text())["orig"]
    assert v["box"] == "the copy here"
    # the docs' fetch line keeps --parity
    docs = (REPO / "docs" / "REMOTE-RUNS.md").read_text()
    g3 = next(x for x in docs.splitlines() if x.startswith("python3 scripts/frontier_fetch.py"))
    assert "--parity" in g3 and "--every 3m" in g3


# ---------------------------------------------------------------------------
# 17: two fetches at once keep each other's verdicts and safe boxes
# ---------------------------------------------------------------------------

def test_17_two_fetches_at_once_merge_what_they_keep(tmp_path, monkeypatch):
    import frontier_fetch as ff
    f = tmp_path / "parity" / "verdicts.json"
    f.parent.mkdir()
    f.write_text(json.dumps({"a": {"same": True, "first": "The same", "at": 10}}))
    # this fetch read the file before another fetch wrote b; its own c is new
    mine = {"a": {"same": True, "first": "The same", "at": 10},
            "c": {"same": False, "first": "Not the same", "at": 30}}
    f.write_text(json.dumps({**json.loads(f.read_text()),
                             "b": {"same": True, "first": "The same", "at": 20}}))
    got = ff.merge_write(f, mine, ff.read_verdicts)
    assert set(got) == {"a", "b", "c"}                          # 0abb757: b erased
    assert set(json.loads(f.read_text())) == {"a", "b", "c"}
    s = tmp_path / "safe-boxes.json"
    s.write_text(json.dumps({"x": {"at": 5, "missed": 0}}))
    ff.merge_write(s, {"x": {"gone": True, "at": 9}, "y": {"at": 9, "missed": 0}}, ff.read_safe)
    got = ff.read_safe(s)
    assert got["x"].get("gone") and not got["y"].get("gone")


# ---------------------------------------------------------------------------
# 18: --sha for a model the board doesn't serve: the line with --register, every round
# ---------------------------------------------------------------------------

def test_18_sha_for_a_model_the_board_doesnt_serve_prints_the_register_line(tmp_path,
                                                                           monkeypatch, capsys):
    there = tmp_path / "there"
    there.mkdir()
    b = a_bundle(there / "frontier-served__orig-thinking-on-gpqa.tar.gz", "served/orig")
    whole = progress_of("orig", "A5-1", "whole", tasks=["gpqa_diamond_epoch"],
                        bundle={"name": b.name})

    def listing(host, rnd):
        return with_steps(listing_of({f"/workspace/orig/A5-1/{b.name}": b}, [whole]),
                          [("orig", "A5-1")])
    ff, calls, key, state = world(tmp_path, monkeypatch, listing)

    def copy(_box, _key, remote, want, dest, name=""):
        (dest / (name or Path(remote).name)).write_bytes(b.read_bytes())
        return True, "copied"
    monkeypatch.setattr(ff, "copy", copy)
    main_of(ff, key, tmp_path / "b" / "bundles", "--sha", f"served/orig={'cd' * 32}",
            "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "the board doesn't serve served/orig yet: import it once by hand, with --register" \
        in out, out                                            # 0abb757: "refused — add it …"
    assert '--register "<its name>"' in out
    assert not [c for c in calls if c[:8] == ff.IMPORT and str(b.name) in " ".join(c)]
    # the import's own refusal says --register too
    src = (REPO / "scripts" / "import_frontier.py").read_text()
    assert "give this import \"\n                f\"--register" in src or \
        '--register \\"<its name>\\" (a model run on rented GPUs only)' in src


# ---------------------------------------------------------------------------
# 19, 20: the board's list — rows with no box's name, rows aged out on write; quiet by the reading
# ---------------------------------------------------------------------------

def test_19_rows_an_older_fetch_stored_leave_once_a_box_is_reached(svc):  # noqa: F811
    import import_frontier as imf
    now = time.time()
    old = {**step("A9", "", now), "label": "A9", "step": "A3-2"}
    old.pop("box_id")
    p = imf.boxes_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"posted_at": now, "boxes": [old,
                                                         {**step("A1", "a", now,
                                                                 seen=now - 3 * 86400)}]}))
    imf.store_boxes([{"asked": ["b"], "reached": ["b"]}, step("A2", "b", now)])
    kept = json.loads(p.read_text())["boxes"]
    assert [x["label"] for x in kept] == ["A2"]                 # 0abb757: A9 and A1 kept


def test_20_quiet_is_a_steps_own_write_against_when_it_was_read(svc):  # noqa: F811
    import import_frontier as imf
    now = time.time()
    read = now - 70 * 60                                       # the fetch stopped 70 min ago
    imf.store_boxes([{"asked": ["a"], "reached": ["a"], "every": 180},
                     {**step("A1", "a", read, line="MMLU-Pro 300 of 12,032 · 13.3 h left"),
                      "at": read - 60}])
    p = imf.boxes_path()
    blob = json.loads(p.read_text())
    blob["posted_at"] = read
    blob["boxes"][0]["seen_at"] = read
    p.write_text(json.dumps(blob))
    got = imf.read_boxes(now)
    assert got["boxes"][0]["quiet_min"] is None                 # 0abb757: 71, "Stopped?"
    assert got["runs"][0]["status"] == "running"
    assert got["stale_min"] == 70


# ---------------------------------------------------------------------------
# 21: small ones
# ---------------------------------------------------------------------------

def test_21_a_kill_after_the_runs_row_leaves_one_runs_row(box, monkeypatch):  # noqa: F811
    import import_frontier as imf
    assert run_box(box, "s1", "--shard", "1/2") == 0
    assert run_box(box, "s2", "--shard", "2/2") == 0
    register(box["sha"])
    assert imported(bundle_of(box, "s1", (1, 2)))[0] == 0
    real = imf._write_registry

    def killed(row, reg):
        raise Killed()                          # after the Runs row, before the registry
    monkeypatch.setattr(imf, "_write_registry", killed)
    with pytest.raises(Killed):
        imported(bundle_of(box, "s2", (2, 2)))
    monkeypatch.setattr(imf, "_write_registry", real)
    assert imported(bundle_of(box, "s2", (2, 2)))[0] == 0
    name = bundle_of(box, "s2", (2, 2)).name
    rows = [r for r in db.recent(50) if str(r.get("note") or "").endswith(f"· {name}")]
    assert len(rows) == 1 and rows[0]["status"] == "done", rows   # 0abb757: two done rows


def test_21_another_file_under_the_same_name_doesnt_keep_the_sha256(svc):  # noqa: F811
    fake = FakeServer()
    try:
        fake.model_path = f"/models/{PARTS[0][0]}"
        rec = served.register({"name": "board box", "base_url": fake.base, "how": "x",
                               "thinking": "off"}, ME)
        # imported from rented GPUs: the pin's size the files' bytes — another
        # file of the same name, twice the size, is served now
        db.served_put({**rec, "pin": {**rec["pin"], "size": fake.size * 2},
                       "file_sha256": {"sha256": "ab" * 32, "by": "masein",
                                       "name": PARTS[0][0], "size": fake.size * 2}})
        served.register({"name": "board box", "base_url": fake.base, "how": "y",
                         "thinking": "off"}, ME)
        assert "file_sha256" not in served.get(rec["id"])      # 0abb757: kept by name alone
    finally:
        fake.close()


def test_21_the_by_hand_line_pastes_and_the_docs_say_every_3m(tmp_path):
    import frontier_fetch as ff
    b = a_bundle(tmp_path / "frontier-served__orig-thinking-on-gpqa.tar.gz", "served/orig")
    line = ff.by_hand(b, "masein", None, "served/orig")
    first = line.splitlines()[0]
    assert first.endswith("--file-sha256 <its sha256>"), first    # 0abb757: "(and --register …)"
    assert "(" not in first
    docs = (REPO / "docs" / "REMOTE-RUNS.md").read_text()
    assert "--every 15m" not in docs                              # 0abb757: line 1014
