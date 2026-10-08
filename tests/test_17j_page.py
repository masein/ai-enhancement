"""17j, part 5, on the board's side (points 28, 30, 31 and 33): where_check
relabels a run "this server" only on its own log; a rented run's row from its
boxes — a parity step done and a step abandoned said on their box's line, a
step's two benchmarks both counted; the fetch posts an abandoned step as
abandoned. Rows and steps invented; nothing is fetched."""

from __future__ import annotations

import json
import sqlite3
import time

import import_frontier as imf
from service import config, db
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17g_review import MMLU, progress_of, with_steps
from test_17f_review import listing_of
from test_17j_boxes import world

HLE, GPQA, OTIS = "hle_text_cais", "gpqa_diamond_epoch", "otis_aime_epoch"


def a_log(sid: int, model: str, first: str) -> None:
    p = config.LOGS_DIR / f"service_{sid}_{model.replace('/', '__')}.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(first + "\n[import] the row …\n")


# ---------------------------------------------------------------------------
# 28: "this server" only on the run's own log
# ---------------------------------------------------------------------------

def test_28_this_server_only_when_the_runs_own_log_is_here(svc):  # noqa: F811
    import where_check
    rows = {}
    for name, where in (("no-log", "rented GPU · RTX 5090 · box A3"),
                        ("spaced", "rented GPU · RTX 5090 · box A4"),
                        ("own-log", "rented GPU · my own words")):
        sid = db.add(f"served/{name}", "instruct", "frontier", "masein", "a note", status="done")
        db.update(sid, where_ran=where)
        rows[name] = sid
    # an import's log, its bundle's name with a space: the header can't be read whole
    a_log(rows["spaced"], "served/spaced",
          f"===== [{rows['spaced']}] imported a b.tar.gz (sha256 0123456789abcdef) by masein: "
          "hle_text_cais, run on a rented GPU (RTX 5090) =====")
    a_log(rows["own-log"], "served/own-log", "[frontier] served/own-log · thinking off · "
          "gpqa_diamond_epoch · on this server")
    got = {x["id"]: x for x in where_check.wrong(sqlite3.connect(config.DB_PATH), config.OUT_DIR,
                                                 config.LOGS_DIR)}
    assert rows["no-log"] not in got, got                       # 0abb757: "this server"
    assert rows["spaced"] not in got or got[rows["spaced"]]["should"].startswith("rented GPU")
    assert got[rows["own-log"]]["should"] == ""
    db.init()                                                   # the start-up fix
    assert db.get(rows["no-log"])["where_ran"] == "rented GPU · RTX 5090 · box A3"
    assert db.get(rows["spaced"])["where_ran"].startswith("rented GPU · RTX 5090")
    assert db.get(rows["own-log"])["where_ran"] == ""


# ---------------------------------------------------------------------------
# 30, 31, 33: a rented run's row from its boxes
# ---------------------------------------------------------------------------

def box(label, step, state="asking", tasks=(HLE,), line="", **more):
    now = time.time()
    return {"label": label, "model": "served/m", "step": step, "thinking": "on",
            "tasks": list(tasks), "state": state, "at": now - 60, "seen_at": now,
            "line": line, "reachable": True, "box_id": label.lower(), **more}


def runs_of(*steps):
    import frontier_box as fbx
    now = time.time()
    read = [imf._box_read(imf.box_row(s), now, fbx) for s in steps]
    return imf.rented_runs([r for r in read if r], now, now)


def test_30_31_a_parity_step_done_and_an_abandoned_step_are_no_rows(svc):  # noqa: F811
    got = runs_of(box("A3", "A3-1", state="whole", tasks=(), parity=True, line="parity done"),
                  box("A3", "A3-2", line="Humanity's Last Exam 25 of 539 · 2 h left"),
                  box("A3", "A3-3", state="abandoned", tasks=(GPQA,)))
    assert [r["tasks"] for r in got] == [HLE], got              # 0abb757: three rows
    assert got[0]["where"] == "rented GPU · box A3 · parity done · a step abandoned"
    assert got[0]["boxes"][0]["notes"] == ["parity done", "a step abandoned"]
    # a parity step still asking is a row: it is running
    got = runs_of(box("A3", "A3-1", tasks=(), parity=True, line="parity 412 of 1,000"))
    assert [r["parity"] for r in got] == [True]


def test_33_a_step_asking_two_benchmarks_counts_both(svc):  # noqa: F811
    import frontier_box as fbx
    first = box("A1", "A1-2", tasks=(GPQA, OTIS), thinking="off",
                line="GPQA Diamond 120 of 198 · 18 s an answer · 2 h left")
    r = runs_of(first)[0]
    otis = fbx.ANSWERS[OTIS]
    assert [(b["label"], b["n"], b["of"], b["state"]) for b in r["benchmarks"]] == [
        ("GPQA Diamond", 120, 198, "now"),
        (imf.fb.BENCH[OTIS]["label"], 0, otis, "next")]         # 0abb757: GPQA's alone
    assert (r["n"], r["of"]) == (120, 198 + otis)
    # the step's finish counts the benchmark after it
    assert r["finish"] > time.time() + 2 * 3600 + 60
    # on to the second: the first counted done
    r = runs_of({**first, "line": "OTIS Mock AIME 2024–2025 10 of 45 · 1 h left"})[0]
    assert [b["state"] for b in r["benchmarks"]] == ["done", "now"]
    assert r["n"] == fbx.ANSWERS[GPQA] + 10


# ---------------------------------------------------------------------------
# 31: the fetch posts an abandoned step as abandoned
# ---------------------------------------------------------------------------

def test_31_the_fetch_posts_an_abandoned_step_as_abandoned(tmp_path, monkeypatch):
    left = progress_of("phone", "A3-2", "stopped", tasks=[MMLU], line="MMLU-Pro 25 of 539")
    left["at"] = time.time() - 6 * 3600

    def listing(host, rnd):
        return with_steps(listing_of({}, [left]), [("phone", "A3-2")])
    ff, calls, key, state = world(tmp_path, monkeypatch, listing)
    dest = tmp_path / "b" / "bundles"
    ff.main(["--key", str(key), "--dest", str(dest), "--abandoned", "phone/A3-2", "1.1.1.1:41"])
    posted = json.loads((dest.parent / "boxes.json").read_text())
    steps = [x for x in posted if isinstance(x, dict) and x.get("step") == "A3-2"]
    assert [x["state"] for x in steps] == ["abandoned"], posted   # 0abb757: not posted
