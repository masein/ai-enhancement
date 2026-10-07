"""17g, what checking 17f found (docs/prompts/phase-17g-fifth-review-fetch-
grading-export.md). Part 1: the boxes and the fetch. Each test fails on
308fcf3. The box is tests/fixtures/fake_llama_server.py; the boxes the fetch
reads are invented listings; nothing is fetched, no model runs, and no paid
API is called."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import pytest

import frontier as fb
import remote_bundle as rb
import remote_gguf as rg
from service import config
from service import frontier as sf
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import (GPU, N, ROW, RUNS, SERVED, TASK, box,  # noqa: F401
                              bundle_of, register, run_box)
from test_17b_review import answers_name, imported, rewrite
from test_17e_review import PILOT_FILE, PILOT_SHAPE, Q8, a_bundle, questions
from test_17f_review import fetch_world, listing_of

HLE, MMLU, ARC = "hle_text_cais", "mmlupro_tiger", "arc_agi2_public"
PHONE = f"served/phone={'ab' * 32}"


def progress_of(build: str, step: str, state: str, **more) -> dict:
    return {"dir": f"/workspace/{build}/{step}", "label": step.rsplit("-", 1)[0],
            "model": f"served/{build}", "state": state, "sessions": 1, "at": time.time(),
            "started_at": "2026-10-06T20:00:00Z", **more}


def with_steps(listing: str, steps: list[tuple[str, str]]) -> str:
    """the 17g listing: the step folders a box holds, beside what they wrote"""
    got = json.loads(listing)
    got["steps"] = [{"build": b, "step": s} for b, s in steps]
    return json.dumps(got)


def fetch_one(tmp_path, monkeypatch, files: dict, progress: list, parity: dict | None = None,
              steps: list | None = None, compare: tuple[int, str] | None = None):
    """frontier_fetch against one invented box, its folders listed too"""
    ff, calls, key = fetch_world(tmp_path, monkeypatch, {"1.1.1.1": (files, progress, parity)})
    run = ff.run

    def run2(cmd, cwd=None, timeout=None, stdin=None):
        if cmd[0] == "ssh" and steps is not None:
            return 0, with_steps(listing_of(files, progress, parity), steps)
        if compare is not None and cmd[:9] == ff.COMPARE:
            calls.append(cmd)
            return compare
        return run(cmd, cwd=cwd, timeout=timeout, stdin=stdin)
    monkeypatch.setattr(ff, "run", run2)

    def sleep(s):                                 # --every: a second round, never a third
        sleep.n = getattr(sleep, "n", 0) + 1
        if sleep.n > 1:
            raise AssertionError("--every never stopped")
    monkeypatch.setattr(ff.time, "sleep", sleep)
    return ff, calls, key


# ---------------------------------------------------------------------------
# 1: "safe to destroy" by the plan, for every build started on the box
# ---------------------------------------------------------------------------

def test_1_a3_with_only_its_parity_whole_is_not_safe(tmp_path, monkeypatch, capsys):
    there = tmp_path / "there"
    there.mkdir()
    par = there / "parity.jsonl"
    par.write_text('{"parity_of": {}}\n')
    ff, calls, key = fetch_one(
        tmp_path, monkeypatch, {},
        [progress_of("phone", "A3-parity", "whole", parity=True)],
        {"/workspace/phone/A3-parity/parity.jsonl": par})
    code = ff.main(["--key", str(key), "--dest", str(tmp_path / "b" / "bundles"), "--sha", PHONE,
                    "1.1.1.1:41"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "1.1.1.1:41 (A3): NOT safe to destroy" in out, out
    assert "phone A3-2 hasn't started" in out
    assert "done, safe to destroy" not in out


def test_1_a_box_between_two_steps_and_a_second_build_starting_are_not_safe(tmp_path,
                                                                              monkeypatch, capsys):
    there = tmp_path / "there"
    there.mkdir()
    b = a_bundle(there / "frontier-served__phone-thinking-on-hle-shard-1-of-4.tar.gz",
                 "served/phone")
    files = {f"/workspace/phone/A1-1/{b.name}": b}
    whole = progress_of("phone", "A1-1", "whole", tasks=[HLE], thinking="on", shard="1/4",
                        bundle={"name": b.name})
    # A1's second step hasn't made its folder yet
    ff, calls, key = fetch_one(tmp_path, monkeypatch, files, [whole], steps=[("phone", "A1-1")])
    dest = tmp_path / "b" / "bundles"
    assert ff.main(["--key", str(key), "--dest", str(dest), "--sha", PHONE, "1.1.1.1:41"]) == 0
    out = capsys.readouterr().out
    assert "NOT safe to destroy" in out and "phone A1-2 hasn't started" in out, out
    # a second build on the box: its folder made, its GGUF still coming, nothing written
    ff, calls, key = fetch_one(tmp_path, monkeypatch, files, [whole], steps=[
        ("phone", "A1-1"), ("phone", "A1-2"), ("orig", "A1-1")])
    assert ff.main(["--key", str(key), "--dest", str(dest), "--sha", PHONE, "1.1.1.1:41"]) == 0
    out = capsys.readouterr().out
    assert "orig A1-1 is starting" in out and "orig A1-2 hasn't started" in out, out
    assert "done, safe to destroy" not in out


def test_1_every_planned_step_whole_and_home_is_safe_and_every_stops(tmp_path, monkeypatch,
                                                                     capsys):
    there = tmp_path / "there"
    there.mkdir()
    b = a_bundle(there / "frontier-served__phone-thinking-on-mmlu-pro.tar.gz", "served/phone")
    ff, calls, key = fetch_one(
        tmp_path, monkeypatch, {f"/workspace/phone/A5-1/{b.name}": b},
        [progress_of("phone", "A5-1", "whole", tasks=[MMLU], thinking="on",
                     bundle={"name": b.name})], steps=[("phone", "A5-1")])
    assert ff.main(["--key", str(key), "--dest", str(tmp_path / "b" / "bundles"), "--sha", PHONE,
                    "--every", "1s", "1.1.1.1:41"]) == 0
    out = capsys.readouterr().out
    assert "1.1.1.1:41 (A5): done, safe to destroy" in out, out


# ---------------------------------------------------------------------------
# 2: a step that fails at start-up says "stopped", and why
# ---------------------------------------------------------------------------

def test_2_a_server_that_never_comes_up_writes_stopped_and_why(box, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FAKE_LLAMA_LOAD_S", "30")
    with pytest.raises(SystemExit, match="llama-server wasn't healthy after 2 s"):
        run_box(box, "run", "--load-timeout", "2")
    got = json.loads((box["root"] / "run" / rg.PROGRESS).read_text())
    assert got["state"] == "stopped", got
    assert got["why"].startswith("llama-server wasn't healthy after 2 s"), got


def test_2_a_refusal_before_the_fetch_writes_stopped_and_why(tmp_path):
    with pytest.raises(SystemExit):
        rg.main(["--as", "served/phone", "--gguf", "hf://me/private/m.gguf", "--server",
                 "hf://me/private/s.tar.gz", "--thinking", "on", "--only", ARC, "--label", "A8",
                 "--out", str(tmp_path / "A8-1")])
    got = json.loads((tmp_path / "A8-1" / rg.PROGRESS).read_text())
    assert (got["state"], got["label"], got["model"], got["tasks"]) == \
        ("stopped", "A8", "served/phone", [ARC])
    assert "917,504 tokens of context" in got["why"], got


def test_2_a_parity_question_the_server_fails_is_said_and_stopped(box, monkeypatch):  # noqa: F811
    from test_17c_review import PARITY_ITEMS
    monkeypatch.setattr(fb, "_fetch", lambda task: {"items": PARITY_ITEMS,
                                                    "extra": {"shots": {}}})
    monkeypatch.setenv("FAKE_LLAMA_STATUS", "400")
    with pytest.raises(SystemExit, match=r"parity: .* nothing kept; paste the box's line again "
                                         r"to ask it again"):
        run_box(box, "par", "--parity", "--n", "50")
    got = json.loads((box["root"] / "par" / rg.PROGRESS).read_text())
    assert got["state"] == "stopped" and got["why"].startswith("parity: "), got


def test_2_the_fetch_and_the_list_say_why_a_step_stopped(tmp_path, monkeypatch, capsys, svc):  # noqa: F811
    import import_frontier as imf
    why = "llama-server didn't come up"
    ff, calls, key = fetch_one(tmp_path, monkeypatch, {},
                               [progress_of("phone", "A5-1", "stopped", tasks=[MMLU], why=why)],
                               steps=[("phone", "A5-1")])
    ff.main(["--key", str(key), "--dest", str(tmp_path / "b" / "bundles"), "--sha", PHONE,
             "--no-board", "1.1.1.1:41"])
    out = capsys.readouterr().out
    assert f"A5 A5-1 · {MMLU} · thinking None · stopped · {why}" in out, out
    assert f"phone A5-1 is stopped: {why}" in out
    # the board keeps why, and the list says it
    # 17i: with when the fetch saw it, as the fetch always sends — a row never
    # seen has no age to leave the list by
    imf.store_boxes([{"label": "A5", "model": "served/phone", "step": "A5-1",
                      "state": "stopped", "why": why, "seen_at": time.time()}])
    assert imf.read_boxes()["boxes"][0]["why"] == why
    # 17i: and the run's row on Runs says it, in plain words
    assert imf.read_boxes()["runs"][0]["status_words"] == f"Stopped: {why}"


def test_2_a_failed_parity_step_says_so_last_and_a_paste_asks_it_again(tmp_path, monkeypatch,
                                                                      capsys):
    import frontier_box as fbx
    ran = []
    root = tmp_path / "ws"

    def run(cmd, *k, **kw):
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        ran.append(out.name)
        if "--parity" in cmd:
            rg.progress(out, state="stopped", why="parity: the server failed on a question — "
                        "nothing kept; paste the box's line again to ask it again")
            return subprocess.CompletedProcess(cmd, 1)
        rg.progress(out, state="whole", bundle={"name": "b.tar.gz"})
        (out / "b.tar.gz").write_bytes(b"x")
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(fbx.subprocess, "run", run)
    line = ["A3", "--as", "served/phone", "--gguf", "g", "--server", "s",
            "--based-on", "b", "--flags", "f", "--root", str(root)]
    assert fbx.main(line) == 1
    out = capsys.readouterr().out
    assert ran == ["A3-parity", "A3-2"]
    assert re.search(r"A3, step 1: not whole \(exit 1\): parity: the server failed on a question"
                     r".* — paste the same line again to ask the parity questions again", out), out
    # the same line again: HLE is whole, not run again; parity is asked again
    ran.clear()
    assert fbx.main(line) == 1
    assert ran == ["A3-parity"]
    assert "A3, step 2 of 2: whole already — not run again" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 3: --parity checked at the start; each build's verdict every round
# ---------------------------------------------------------------------------

def test_3_a_parity_file_that_isnt_there_is_refused_at_the_start(tmp_path, monkeypatch, capsys):
    ff, calls, key = fetch_one(tmp_path, monkeypatch, {}, [])
    with pytest.raises(SystemExit) as e:
        ff.main(["--key", str(key), "--parity",
                 f"served/phone={tmp_path / 'phone-server.jsonl'}", "1.1.1.1:41"])
    assert e.value.code == 2 and not calls
    assert "phone-server.jsonl: no such file" in capsys.readouterr().err
    # the help names the file the server holds
    with pytest.raises(SystemExit):
        ff.main(["--help"])
    assert "phone-server-500.jsonl" in capsys.readouterr().out.replace("\n", "")


@pytest.mark.parametrize("compare,code", [
    ((1, "Not the same setup: the box answers 38.0% right, the server 42.4%\n"), 1),
    ((2, "refused — the server's file isn't the server's parity file\n"), 1),
    ((0, "The same: the box answers 42.4% right (the mean of its two runs) …\n"), 0)])
def test_3_each_builds_verdict_on_its_own_line_and_the_exit_says_it(tmp_path, monkeypatch,
                                                                    capsys, compare, code):
    there = tmp_path / "there"
    there.mkdir()
    par = there / "parity.jsonl"
    par.write_text('{"parity_of": {}}\n')
    b = a_bundle(there / "frontier-served__phone-thinking-on-hle-shard-3-of-4.tar.gz",
                 "served/phone")
    ref = tmp_path / "phone-server-500.jsonl"
    ref.write_text("{}\n")
    ff, calls, key = fetch_one(
        tmp_path, monkeypatch, {f"/workspace/phone/A3-2/{b.name}": b},
        [progress_of("phone", "A3-parity", "whole", parity=True),
         progress_of("phone", "A3-2", "whole", tasks=[HLE], bundle={"name": b.name})],
        {"/workspace/phone/A3-parity/parity.jsonl": par},
        steps=[("phone", "A3-parity"), ("phone", "A3-2")], compare=compare)
    got = ff.main(["--key", str(key), "--dest", str(tmp_path / "b" / "bundles"), "--sha", PHONE,
                   "--parity", f"served/phone={ref}", "--no-board", "1.1.1.1:41"])
    out = capsys.readouterr().out
    assert got == code, out
    first = compare[1].splitlines()[0]
    assert re.search(rf"^phone's parity: {re.escape(first)}$", out, re.M), out


# ---------------------------------------------------------------------------
# 4: the box's line runs a short step again by itself, up to three runs
# ---------------------------------------------------------------------------

def test_4_a_step_left_short_is_run_again_by_the_line_up_to_three_times(tmp_path, monkeypatch,
                                                                        capsys):
    import frontier_box as fbx
    runs = []

    def run(cmd, *k, **kw):
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        runs.append(out.name)
        if len(runs) < 3:                         # one question failing with a 500 twice
            rg.progress(out, state="stopped",
                        incomplete={MMLU: {"answers": 11_999 + len(runs), "of": 12_032}})
            return subprocess.CompletedProcess(cmd, 1)
        rg.progress(out, state="whole", incomplete={}, bundle={"name": "b.tar.gz"})
        (out / "b.tar.gz").write_bytes(b"x")
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(fbx.subprocess, "run", run)
    line = ["A5", "--as", "served/phone", "--gguf", "g", "--server", "s",
            "--based-on", "b", "--flags", "f", "--root", str(tmp_path)]
    assert fbx.main(line) == 0
    out = capsys.readouterr().out
    assert runs == ["A5-1"] * 3
    assert "A5, step 1: not whole (MMLU-Pro 12,000 of 12,032) — asking what is left again, " \
           "run 2 of 3" in out
    assert "run 3 of 3" in out and "A5, step 1: whole" in out
    # never more than three
    runs.clear()
    monkeypatch.setattr(fbx.subprocess, "run", lambda cmd, *k, **kw: (
        runs.append(1), rg.progress(Path(cmd[cmd.index("--out") + 1]), state="stopped",
                                    incomplete={MMLU: {"answers": 1, "of": 2}}),
        subprocess.CompletedProcess(cmd, 1))[-1])
    (tmp_path / "phone" / "A5-1" / "progress.json").unlink()
    assert fbx.main(line) == 1 and len(runs) == 3


# ---------------------------------------------------------------------------
# 5: HLE's slots from the measured slope, said; --slots and --slots-fit pass through
# ---------------------------------------------------------------------------

def test_5_hle_runs_8_slots_from_the_measured_slope_and_says_so(box, monkeypatch):  # noqa: F811
    import frontier_box as fbx
    monkeypatch.setattr(rg, "header_of", lambda src: (PILOT_SHAPE, PILOT_FILE))
    monkeypatch.setattr(fb, "_fetch", questions)
    monkeypatch.setattr(rb, "gpu_info", lambda: {**GPU, "memory_mib": 32_607})
    assert run_box(box, "h", "--only", HLE, "--slots", "8", "--min-slots", "7",
                   "--flags", " ".join(Q8)) == 0
    own = (box["root"] / "h" / rg.OWN_LOG).read_text()
    m = re.search(r"memory, from the pilot's measured slope \(12\.76 KiB a token of context, 666 "
                  r"MiB above the file at 8 slots\): about [\d.]+ GB for 8 slots of 86,016", own)
    assert m, own
    spare = re.search(r"([\d,]+) MiB spare; up to (\d+) slots fit", own)
    assert spare and 1_500 < int(spare[1].replace(",", "")) < 1_650 and int(spare[2]) >= 8, own
    path = next((box["root"] / "h").glob("frontier-*.tar.gz"))
    assert rb.read(path)["setup"]["server"]["slots"] == 8
    assert fbx.fits(86_016) == 8 and fbx.slots_run(("on", (HLE,), "1/4", 8)) == 8


def test_5_the_refusals_advice_is_what_frontier_box_takes():
    import frontier_box as fbx
    a = fbx.parser().parse_args(["A8", "--as", "served/phone",
                                 "--gguf", "g", "--server", "s", "--based-on", "b", "--flags",
                                 "f", "--slots", "4", "--slots-fit"])
    args = fbx.argv_of("A8", 1, fbx.PLANS["A"]["A8"][0], a)
    assert args[args.index("--slots") + 1] == "4" and "--slots-fit" in args
    # a cap: a step planned at fewer keeps its own
    a = argparse.Namespace(**{**vars(a), "slots": 6, "slots_fit": False})
    args = fbx.argv_of("A8", 1, fbx.PLANS["A"]["A8"][0], a)
    assert args[args.index("--slots") + 1] == "5" and "--slots-fit" not in args


# ---------------------------------------------------------------------------
# 6: sudo asked once and kept alive, or one line says it waits
# ---------------------------------------------------------------------------

def test_6_sudo_is_asked_once_and_kept_alive(monkeypatch, capsys):
    import frontier_fetch as ff
    calls, started = [], []
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/sudo")
    monkeypatch.setattr(ff.subprocess, "run", lambda cmd, **kw: (
        calls.append(cmd), subprocess.CompletedProcess(cmd, 0))[-1])

    class T:
        def __init__(self, target, daemon):
            self.target, self.daemon = target, daemon

        def start(self):
            started.append(self)
    monkeypatch.setattr("threading.Thread", T)
    ff.keep_sudo()
    assert calls[0][:1] == ["sudo"] and "-v" in calls[0] and started and started[0].daemon
    # it lapses: one line says the import waits for it
    calls.clear()
    monkeypatch.setattr(ff.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1))
    sleeps = iter([None, None, StopIteration])

    def sleep(s):
        x = next(sleeps)
        if x is StopIteration:
            raise RuntimeError("stop")
    monkeypatch.setattr(ff.time, "sleep", sleep)
    with pytest.raises(RuntimeError):
        started[0].target()
    out = capsys.readouterr().out
    assert out.count("sudo's password has lapsed: the next import waits for it here") == 1
    # not given at the start: said once, and nothing kept alive
    started.clear()
    ff.keep_sudo()
    assert "sudo's password wasn't given: each import will ask for it" in \
        capsys.readouterr().out and not started


def test_6_the_fetch_asks_for_sudo_before_its_first_round(tmp_path, monkeypatch):
    ff, calls, key = fetch_one(tmp_path, monkeypatch, {}, [])
    order = []
    monkeypatch.setattr(ff, "keep_sudo", lambda: order.append("sudo"))
    run = ff.run
    monkeypatch.setattr(ff, "run", lambda cmd, **kw: (order.append(cmd[0]), run(cmd, **kw))[-1])
    ff.main(["--key", str(key), "--dest", str(tmp_path / "b"), "--no-board", "1.1.1.1:41"])
    assert order[0] == "sudo" and order.count("sudo") == 1


# ---------------------------------------------------------------------------
# 7: a model with no --sha: home, its import to type, and the box done
# ---------------------------------------------------------------------------

def test_7_a_bundle_with_no_sha_is_home_with_its_import_and_the_box_done(tmp_path, monkeypatch,
                                                                       capsys):
    there = tmp_path / "there"
    there.mkdir()
    g = a_bundle(there / "frontier-served__gemma-thinking-on-gpqa.tar.gz", "served/gemma")
    ff, calls, key = fetch_one(
        tmp_path, monkeypatch, {f"/workspace/gemma/G6-1/{g.name}": g},
        [progress_of("gemma", "G6-1", "whole", tasks=["gpqa_diamond_epoch"],
                     bundle={"name": g.name})], steps=[("gemma", "G6-1")])
    dest = tmp_path / "b" / "bundles"
    # --every ends: the box is done
    assert ff.main(["--key", str(key), "--dest", str(dest), "--sha", PHONE, "--every", "1s",
                    "--no-board", "1.1.1.1:41"]) == 0
    out = capsys.readouterr().out
    assert "1.1.1.1:41 (G6): done, safe to destroy" in out, out
    assert (f"{g.name}: copied — home; no --sha for served/gemma: import it by hand: sudo docker "
            f"compose exec -T bench python scripts/import_remote.py {dest / g.name} --by masein "
            # 17h: --register only for a model the board doesn't serve (here it can't say)
            '--file-sha256 <its sha256> (and --register "<its name>" if the board doesn\'t '
            "serve it yet)") in out
    # 17h: nothing imported (the board asked only which models it serves)
    assert not [c for c in calls if c[:8] == ff.IMPORT and "--served" not in c]


# ---------------------------------------------------------------------------
# 8: the timeout's comment
# ---------------------------------------------------------------------------

def test_8_the_timeouts_comment_says_17fs_limit():
    src = Path(rg.__file__).read_text()
    i = src.index('"SERVED_TIMEOUT_S": int(')
    said = src[src.rindex("#", 0, i):i]
    assert "81,920" in said and "65,536" not in said


# ---------------------------------------------------------------------------
# part 2, 9: shards asked under two limits never merge
# ---------------------------------------------------------------------------

def test_9_shards_asked_under_another_limit_are_refused_not_merged(box, monkeypatch):  # noqa: F811
    limit = fb.BENCH[TASK]["budget"]["on"]
    monkeypatch.setitem(fb.BENCH[TASK]["budget"], "on", 4096)  # the limits before a deploy
    assert run_box(box, "s1", "--shard", "1/2") == 0
    assert run_box(box, "s2", "--shard", "2/2") == 0
    register(box["sha"])
    assert imported(bundle_of(box, "s1", (1, 2)))[0] == 0
    code, said = imported(bundle_of(box, "s2", (2, 2)))
    assert code == 0 and any("every shard is in (2 of 2)" in x for x in said), said
    # the new limit: shard 1 asked again, at 81,920
    monkeypatch.setitem(fb.BENCH[TASK]["budget"], "on", limit)
    assert run_box(box, "s1b", "--shard", "1/2") == 0
    code, said = imported(bundle_of(box, "s1b", (1, 2)))
    assert code == 2, said
    assert any(f"{TASK}: shard 2 here was asked another way (budget: 4,096 there, "
               f"{limit:,} here) — every shard of a task is run the same way: import with "
               "--set-aside-shards" in x for x in said), said
    rows = json.loads((sf.task_dir(config.OUT_DIR / ROW, TASK) / sf.SETUP).read_text())
    assert rows["budget"] == 4096                       # the row is as it was
    # set aside, it starts the task's shards again — not scored with shard 2's
    code, said = imported(bundle_of(box, "s1b", (1, 2)), aside=True)
    assert code == 0 and any("shard 2 of 2 missing" in x for x in said), said


# ---------------------------------------------------------------------------
# part 2, 10: a few changed answers keep the other answers' grades
# ---------------------------------------------------------------------------

def graded_all(d: Path) -> dict:
    """a grader's grade for every answer the row holds, each naming its answer"""
    got = sf.read_answers(d / sf.ANSWERS)
    g = {"grader": {"version": "g1"},
         "items": {sf.gkey(q, e): {"ok": True, "by": "g1", "prompt_sha256": "p",
                                   "answer_sha256": sf.answer_sha(r.get("answer") or "")}
                   for (q, e), r in got.items()},
         "refused": {}}
    (d / sf.GRADES).write_text(json.dumps(g))
    return g


def changed(n: int, which=None):
    def fix(files):
        lines = [json.loads(x) for x in files[answers_name()].decode().splitlines()]
        for r in lines[:n] if which is None else [x for x in lines if which(x)]:
            r["answer"] = (r.get("answer") or "") + " — and something else"
        files[answers_name()] = "".join(json.dumps(x) + "\n" for x in lines).encode()
    return fix


def test_10_two_changed_answers_keep_the_others_grades(box):  # noqa: F811
    assert run_box(box, "run") == 0
    register(box["sha"])
    assert imported(bundle_of(box, "run"))[0] == 0
    d = sf.task_dir(config.OUT_DIR / ROW, TASK)
    g = graded_all(d)
    code, said = imported(rewrite(bundle_of(box, "run"), changed(2)))
    assert code == 0, said
    assert any(f"{TASK}: 2 of {N * RUNS} answers differ from the row — those are graded again, "
               "the unchanged keep their grades" in x for x in said), said
    now = sf.read_grades(d)
    assert len(now["items"]) == N * RUNS - 2 and now["grader"] == g["grader"]
    assert all(g["items"][k] == x for k, x in now["items"].items())


def test_10_one_changed_answer_in_a_remade_shard_keeps_both_shards_grades(box):  # noqa: F811
    assert run_box(box, "s1", "--shard", "1/2") == 0
    assert run_box(box, "s2", "--shard", "2/2") == 0
    register(box["sha"])
    assert imported(bundle_of(box, "s1", (1, 2)))[0] == 0
    assert imported(bundle_of(box, "s2", (2, 2)))[0] == 0
    d = sf.task_dir(config.OUT_DIR / ROW, TASK)
    g = graded_all(d)
    code, said = imported(rewrite(bundle_of(box, "s1", (1, 2)), changed(1)))
    assert code == 0, said
    now = sf.read_grades(d)
    assert len(now["items"]) == N * RUNS - 1, said
    # shard 2's every grade, and shard 1's unchanged ones
    s2 = {sf.gkey(q, e) for (q, e) in sf.read_answers(
        config.OUT_DIR.with_name("shards") / ROW / TASK / "2-of-2" / sf.ANSWERS)}
    assert s2 and s2 <= set(now["items"])
    assert all(g["items"][k] == x for k, x in now["items"].items())


# ---------------------------------------------------------------------------
# part 3: grading (the stand-in grader of tests/test_17_grading.py; nothing
# calls OpenRouter)
# ---------------------------------------------------------------------------

from service import ai_models, db, llm  # noqa: E402
from service import frontier_grade as fgr  # noqa: E402
from test_17_grading import ROW as GROW  # noqa: E402
from test_17_grading import drain, write  # noqa: E402
from test_17_grading import svc as gsvc  # noqa: E402,F401 — the grading fixture
from test_17b_grading import GEMINI, GPT, O3, plain, stub  # noqa: E402

GPT_V, GEMINI_V, O3_V = ("openai/gpt-4.1-2025-04-14", "google/gemini-2.5-flash-20250617",
                         "openai/o3-mini-2025-01-31")
SQA = "simpleqa_epoch"


def gd(row: str = GROW) -> Path:
    return sf.task_dir(config.OUT_DIR / row, SQA)


def replies(monkeypatch, answer) -> list:
    """the grader's replies as OpenRouter's requests keep them: answer(model,
    custom_id) -> (text, error, status) — a refusal with its status and kind"""
    asked = []

    def complete(self, row):
        asked.append((self.model, row["custom_id"]))
        text, error, status = answer(self.model, row["custom_id"])
        rec = {"custom_id": row["custom_id"], "text": text, "error": error, "attempts": 1,
               "finish_reason": "stop" if text else ""}
        if error:
            rec.update(status=status, kind=ai_models.refusal(status, error)[0])
        return rec
    monkeypatch.setattr(fgr.GraderChat, "_complete", complete)
    return asked


def test_11_a_run_of_refusals_after_a_timeout_stops_and_counts_nothing(gsvc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 1)         # in order
    monkeypatch.setattr(fgr.GraderChat, "FIRST_REFUSALS", 3)
    # graded yesterday: one answer, by this grader and prompt
    fgr.save("simpleqa", GPT, "masein")
    a0 = sf.read_answers(gd() / sf.ANSWERS)[("0", 0)]["answer"]
    (gd() / sf.GRADES).write_text(json.dumps({"items": {"0#0": {
        "ok": True, "by": GPT_V, "prompt_sha256": fgr.fg.prompt_sha("simpleqa"),
        "answer_sha256": sf.answer_sha(a0)}}, "refused": {}}))
    # today: the first reply a timeout, the rest "not a valid model ID"
    replies(monkeypatch, lambda m, c: (("", "POST u: no response within 120 s", None)
                                       if c.endswith(":1#0") else
                                       ("", 'POST u: HTTP 400: {"error": {"message": "openai/'
                                        'gpt-4.1 is not a valid model ID"}}', 400))
            if m == GPT else ("No", "", None))
    fgr.start("masein")
    from test_17f_review import wait_halted
    why = wait_halted()
    assert why.startswith("waiting: 3 requests in a row were all refused"), why
    # 308fcf3: the timeout ended "the first five", and each 400 took a try
    ref = sf.read_grades(gd()).get("refused") or {}
    assert not any(int(x.get("tries") or 0) for x in ref.values()), ref


def test_11_a_spend_cap_or_a_providers_failure_is_never_a_try():
    cap = 'POST u: HTTP 403: {"error": {"message": "Key spend cap reached"}}'
    prov = 'POST u: HTTP 400: {"error": {"message": "Provider returned error"}}'
    assert ai_models.refusal(403, cap)[0] == "limit"
    assert ai_models.refusal(400, prov)[0] == "down"
    for e, st in ((cap, 403), (prov, 400)):
        assert not fgr._permanent(llm.Result(error=e, status=st))
        assert not fgr._permanent(e)
    # a refusal of the answer itself still is
    assert fgr._permanent(llm.Result(error="HTTP 400: prompt too long", status=400))


def ungraded_by(monkeypatch, model: str, key: str = ":2#0"):
    """`model` gives no grade to one answer (three Starts), every other a grade"""
    stub(monkeypatch, lambda m, r: ("I would rather not say", "stop", "")
         if m == model and r["custom_id"].endswith(key) else plain(m, r))
    for _ in range(3):
        fgr.start("masein")
        drain()


def sqa_row(est: dict) -> dict | None:
    return next((r for r in est["rows"] if r["slot"] == "simpleqa"), None)


def test_12_choosing_a_grader_moves_nothing_and_choosing_the_first_again_uses_its_grades(
        gsvc, monkeypatch):  # noqa: F811
    fgr.save("simpleqa", GPT, "masein")
    ungraded_by(monkeypatch, GPT)                    # 1 of 5 ungraded: more than 5%
    before = (gd() / sf.GRADES).read_bytes()
    fgr.save("simpleqa", GEMINI, "masein")
    assert (gd() / sf.GRADES).read_bytes() == before            # 308fcf3: moved aside
    row = sqa_row(fgr.estimate())
    assert row["answers"] == 5 and "the first grader left 1 of 5" in row["regrade"]
    # the first again: what it graded is used, nothing priced again
    fgr.save("simpleqa", GPT, "masein")
    assert (gd() / sf.GRADES).read_bytes() == before
    row = sqa_row(fgr.estimate())
    assert row is None or row["answers"] == 0                   # 308fcf3: 5, priced again


def test_12_each_graders_grades_kept_by_name_and_a_third_grades_once(gsvc, monkeypatch):  # noqa: F811
    fgr.save("simpleqa", GPT, "masein")
    ungraded_by(monkeypatch, GPT)
    fgr.save("simpleqa", GEMINI, "masein")
    ungraded_by(monkeypatch, GEMINI, ":3#0")       # the second, whole — and one ungraded
    g = sf.read_grades(gd())
    assert {x["by"] for x in g["items"].values()} == {GEMINI_V}
    assert {x["by"] for v in g["kept"].values() for x in v["items"].values()} == {GPT_V}
    # the first chosen again: its own 4 grades back, nothing sent
    fgr.save("simpleqa", GPT, "masein")
    row = sqa_row(fgr.estimate())
    assert row["answers"] == 0 and row["reused"] == 4
    asked = stub(monkeypatch, plain)
    fgr.start("masein")
    drain()
    assert not [c for m, c in asked if "algebra/" not in c]
    g = sf.read_grades(gd())
    assert {x["by"] for x in g["items"].values()} == {GPT_V}
    assert {x["by"] for v in g["kept"].values() for x in v["items"].values()} == {GEMINI_V}
    # a third: the five answers once each (17f's 98 requests for 30 answers)
    fgr.save("simpleqa", O3, "masein")
    assert sqa_row(fgr.estimate())["answers"] == 5
    asked = stub(monkeypatch, plain)
    fgr.start("masein")
    drain()
    assert len([c for m, c in asked if m == O3 and "algebra/" not in c]) == 5


ROW2, SERVED2 = "served__orig-box", "served/orig-box"


def two_rows():
    db.served_put({"id": SERVED2, "name": "orig box", "base_url": "", "key": "", "how": "x",
                   "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto",
                   "pin": {"file": "o.gguf"}, "by": "masein", "at": 0})
    row = config.OUT_DIR / ROW2
    row.mkdir(parents=True)
    (row / "model_meta.json").write_text(json.dumps({"model": SERVED2}))
    write(row, SQA, [(str(k), f"I think Answer {k}.", "stop" if k < 5 else "length")
                     for k in range(6)])


def graded_by(row: str) -> str:
    """the grader the row's written score names"""
    fs = list(gd(row).glob("results_*.json"))
    return ((json.loads(fs[0].read_text())["frontier"].get("grader") or {}).get("version")
            if fs else "")


def test_13_two_rows_scored_by_two_graders_say_so_and_offer_the_regrade(gsvc, monkeypatch):  # noqa: F811
    import report_lm_eval as report
    two_rows()
    fgr.save("simpleqa", GPT, "masein")
    # the first grader gives no grade to one answer of the first row only
    stub(monkeypatch, lambda m, r: ("I would rather not say", "stop", "")
         if m == GPT and r["custom_id"].endswith(":2#0")
         and fgr._meta(r["batch_id"])["row"] == GROW else plain(m, r))
    for _ in range(3):
        fgr.start("masein")
        drain()
    fgr.save("simpleqa", GEMINI, "masein")
    stub(monkeypatch, plain)
    fgr.start("masein")
    drain()
    graders = {r: graded_by(r) for r in (GROW, ROW2)}
    assert graders == {GROW: GEMINI_V, ROW2: GPT_V}, graders      # each final on its row
    # said on both rows' cells (308fcf3: nothing), and on the card with the offer
    payload = report.build_payload(report.merge_runs(report.load_results(config.OUT_DIR)), "t",
                                   source="")
    cells = payload["cells"][SQA]
    said = {m: c.get("graderDiffers") or "" for m, c in cells.items()}
    assert len(said) == 2 and all("one benchmark, two graders" in w for w in said.values()), said
    mm = fgr.mismatches()
    assert [m["task"] for m in mm] == [SQA]
    offer = {r["row"]: r.get("offer") for r in mm[0]["rows"]}
    assert offer[GROW] is None and offer[ROW2]["answers"] == 5 and offer[ROW2]["usd"] > 0
    # the offer taken: graded again by the grader chosen now at Start, priced first
    fgr.regrade_row(ROW2, SQA, "masein")
    row = next(r for r in fgr.estimate()["rows"] if r["model"] == SERVED2)
    assert row["answers"] == 5 and "so that the rows compared share" in row["regrade"]
    fgr.start("masein")
    drain()
    assert graded_by(ROW2) == GEMINI_V
    assert not fgr.mismatches()


# ---------------------------------------------------------------------------
# 14: HLE's three reply shapes
# ---------------------------------------------------------------------------

def test_14_hles_three_shapes_are_read():
    import frontier_graders as fgs
    o = {"extracted_final_answer": "42", "reasoning": "matches", "correct": "yes",
         "confidence": 90}
    ex = {"extracted_final_answer": "<the final answer>", "reasoning": "<why>",
          "correct": "yes or no", "confidence": "0-100"}
    item = {"id": "h1", "answer": "42"}
    two = json.dumps(o) + "\n" + json.dumps(o)
    example = "Use this format:\n" + json.dumps(ex) + "\nMine:\n" + json.dumps({**o, "correct": "no"})
    line = json.dumps(o) + "\nCorrect: yes — the extracted answer is the key's"
    assert fgs.read("hle", two, item)["ok"] is True                    # 308fcf3: no grade
    assert fgs.read("hle", example, item)["ok"] is False
    assert fgs.read("hle", line, item)["ok"] is True
    # objects or lines that say otherwise stay no grade
    assert fgs.read("hle", json.dumps(o) + "\nCorrect: no", item)["ok"] is None
    assert fgs.read("hle", json.dumps(o) + "\n" + json.dumps({**o, "correct": "no"}),
                    item)["ok"] is None


# ---------------------------------------------------------------------------
# part 4: the dashboard and the export
# ---------------------------------------------------------------------------

from service import served  # noqa: E402
from test_17b_review import results_of  # noqa: E402

LONG = [{**x, "question": f"Invented question number {k} about the boiling point of a "
                         "liquid at altitude?"} for k, x in enumerate(__import__(
                             "test_17_gguf_box").invented())]


def a3_imported(box, monkeypatch=None, label: str = "A3"):  # noqa: F811
    if monkeypatch is not None:
        monkeypatch.setattr(fb, "_fetch", lambda task: [dict(x) for x in LONG])
    assert run_box(box, "a3", "--label", label) == 0
    register(box["sha"])
    assert imported(bundle_of(box, "a3"))[0] == 0
    return db.recent(5)[0]["id"]


def test_15_the_exported_log_withholds_each_line_quoting_a_hidden_question(
        box, tmp_path, monkeypatch):  # noqa: F811
    import export_frontier_raw as efr
    sid = a3_imported(box, monkeypatch)
    log = config.LOGS_DIR / f"service_{sid}_{SERVED.replace('/', '__')}.log"
    q = LONG[3]["question"]
    log.write_text(log.read_text() + f"\n[frontier] the server failed on: {q}\n"
                   "What is the correct answer to this question: which one?\n(A) the first\n")
    dest = efr.export_run(sid, tmp_path / "raw")
    text = (dest / "log.txt").read_text()
    assert q not in text and "the correct answer to this question" not in text  # 308fcf3: both
    # 17h: left out, not marked — and said in the README
    assert "quotes a question" in (dest / "README.md").read_text()


def test_15_the_scrub_takes_a_boxs_address_flags_tokens_hosts_and_the_account():
    import export_devicemark_raw as dmx
    for raw, gone in (("ssh -p 41234 root@203.0.113.77 -L 8080:localhost:8080",
                       ("41234", "203.0.113.77", "root@")),
                      ("scp -P 41234 root@ssh4.vast.ai:/workspace/x .",
                       ("41234", "ssh4.vast.ai", "root@")),
                      ('{"flags": "--api-key abcdefghij --flash-attn on"}', ("abcdefghij",)),
                      ("x-token: correcthorse", ("correcthorse",)),
                      ("SUBMIT_TOKEN=correcthorsebattery", ("correcthorsebattery",)),
                      ("http://bench.internal:8080/api, http://vllm.lan/v1",
                       ("bench.internal", "vllm.lan")),
                      ("hf://masein/evalboard-private/llama-server.tar.gz", ("masein",))):
        got = dmx.scrub(raw, env={}, hosts=[])
        assert not any(g in got for g in gone), got
    assert "masein" not in dmx.scrub("hf://masein/phone-builds/q.gguf",
                                     env={"SCRUB_ACCOUNTS": "masein"}, hosts=[])
    # what may stay: a public repo's path, a version, words
    assert dmx.scrub("hf://ggml-org/gemma-GGUF@bb45/g.gguf", env={}, hosts=[]) == \
        "hf://ggml-org/gemma-GGUF@bb45/g.gguf"
    assert dmx.scrub("llama.cpp 6543 · token: limit reached", env={}, hosts=[]) == \
        "llama.cpp 6543 · token: limit reached"


def test_15_never_public_for_a_model_the_board_doesnt_know_as_public(box, tmp_path, capsys):  # noqa: F811
    import export_frontier_raw as efr
    sid = a3_imported(box)
    assert efr.export_run(sid, tmp_path / "raw", public=True).parent.name == "private"
    assert efr.main(["--all", "--public", "--out", str(tmp_path / "all")]) == 0
    assert "--public: no effect" in capsys.readouterr().out
    assert not (tmp_path / "all" / "public").exists()            # 308fcf3: the build, public
    # 17h: public only when the board's mark says so (its page, or --public-weights)
    assert not efr.known_public({"hf_id": "served/gemma-cal"})
    assert not efr.known_public({"hf_id": "google/gemma-3-1b-it"})  # never by its name
    db.public_set("served/gemma-cal", True, "masein")
    assert efr.known_public({"hf_id": "served/gemma-cal"})


def test_15_each_run_exports_what_it_brought(box, tmp_path):  # noqa: F811
    import export_frontier_raw as efr
    assert run_box(box, "s1", "--shard", "1/2", "--label", "A1") == 0
    assert run_box(box, "s2", "--shard", "2/2", "--label", "A2") == 0
    register(box["sha"])
    assert imported(bundle_of(box, "s1", (1, 2)))[0] == 0
    assert imported(bundle_of(box, "s2", (2, 2)))[0] == 0
    got = []
    for sid in [r["id"] for r in db.recent(5)][:2]:
        dest = efr.export_run(sid, tmp_path / "raw")
        got.append({(x["id"], x["epoch"]) for x in map(
            json.loads, (dest / "items.jsonl").read_text().splitlines())})
    # 308fcf3: each shard's run, every answer of the row
    assert got[0] and got[1] and not got[0] & got[1] and len(got[0] | got[1]) == N * RUNS


def test_16_a_score_made_again_keeps_where_it_ran_and_its_runs(box):  # noqa: F811
    a3_imported(box)
    f = results_of()["frontier"]
    assert f["where"] == "rented GPU · RTX 5090 · box A3" and f["runs"]
    # grading scores it again
    sf.score_task(config.OUT_DIR / ROW, TASK, served.get(SERVED))
    f2 = results_of()["frontier"]
    assert (f2["where"], f2["runs"]) == (f["where"], f["runs"])   # 308fcf3: "a rented GPU", none


def test_17_earlier_imports_say_where_they_ran(svc):  # noqa: F811
    import import_frontier as imf
    from service import devicemark as sdm
    fr = db.add("served/x", "instruct", "frontier", "masein", "imported from a rented GPU "
                "(NVIDIA GeForce RTX 4090) · shard 1 of 2 · box A2 · x.tar.gz", status="done")
    dm = db.add("google/gemma-3-1b-it", "instruct", "devicemark", "masein",
                "imported from a rented GPU (NVIDIA GeForce RTX 4090) · y.tar.gz", status="done")
    here = db.add("served/x", "instruct", "frontier", "masein", "a board run", status="done")
    # 17h: from each row's own record of its imports, never the note
    for row, name, rec in (("served__x", imf.REGISTRY,
                            {"sid": fr, "gpu": "NVIDIA GeForce RTX 4090", "box": "A2"}),
                           ("google__gemma-3-1b-it", sdm.REMOTE_NAME,
                            {"sid": dm, "gpu": "NVIDIA GeForce RTX 4090"})):
        (config.OUT_DIR / row).mkdir(parents=True, exist_ok=True)
        (config.OUT_DIR / row / name).write_text(json.dumps({"imports": [rec]}))
    db.init()
    assert db.get(fr)["where_ran"] == "rented GPU · RTX 4090 · box A2"
    assert db.get(dm)["where_ran"] == "rented GPU · RTX 4090"      # 308fcf3: "this server"
    assert not db.get(here)["where_ran"]


def test_18_the_boxes_list_quiet_only_while_asking_done_leaves_destroyed_unreached(svc):  # noqa: F811
    import import_frontier as imf
    now = time.time()

    def step(label, k, state, at, seen, safe):
        return {"label": label, "model": "served/phone", "step": f"{label}-{k}",
                "state": state, "at": at, "seen_at": seen, "safe": safe, "reachable": True,
                "thinking": "on", "tasks": [MMLU], "line": ""}
    first = [step("A1", 1, "whole", now - 600 * 60, now, False),     # done; its box on step 2
             step("A1", 2, "asking", now - 60, now, False),
             step("A4", 1, "asking", now - 60, now, False),
             step("A5", 1, "whole", now - 60, now, True),
             step("A9", 1, "whole", now - 31 * 86400, now - 30 * 86400, True)]
    imf.store_boxes(first)
    got = {b["step"]: b for b in imf.read_boxes(now)["boxes"]}
    assert got["A1-1"]["quiet_min"] is None                         # 308fcf3: 600
    assert "A9-1" not in got                                        # 308fcf3: 30 days on
    # the next fetch reads A1 alone: A4, destroyed before done, isn't reached; A5 was done
    imf.store_boxes(first[:2])
    got = {b["step"]: b for b in imf.read_boxes(now)["boxes"]}
    assert got["A4-1"]["reachable"] is False and "A5-1" not in got  # 308fcf3: A4 vanished


def test_19_a_malformed_progress_file_breaks_nothing(tmp_path, monkeypatch, capsys, svc):  # noqa: F811
    import import_frontier as imf
    bad = [progress_of("phone", "A5-1", "asking", sessions="two", tasks=[1], at=float("nan")),
           progress_of("phone", "A6-1", "asking", tasks=[MMLU])]
    ff, calls, key = fetch_one(tmp_path, monkeypatch, {}, bad,
                               steps=[("phone", "A5-1"), ("phone", "A6-1")])
    ff.main(["--key", str(key), "--dest", str(tmp_path / "b" / "bundles"), "--sha", PHONE,
             "--no-board", "1.1.1.1:41"])
    out = capsys.readouterr().out
    assert "A5 A5-1" in out and "A6 A6-1" in out                    # 308fcf3: a traceback
    # the board: NaN never kept, the list's endpoint answers JSON
    imf.store_boxes([{"label": "A5", "model": "served/phone", "step": "A5-1", "at": float("nan"),
                      "sessions": "two", "tasks": [1], "state": "asking"}])
    json.dumps(imf.read_boxes(), allow_nan=False)
    assert svc.get("/api/frontier/boxes").status_code == 200


def test_19_a_box_given_as_a_number_is_refused_before_anything_moves(box):  # noqa: F811
    assert run_box(box, "a3", "--label", "A3") == 0
    register(box["sha"])

    def numbers(files):
        st = json.loads(files["setup.json"])
        st.update(box=5, sessions="two")
        files["setup.json"] = json.dumps(st).encode()
    code, said = imported(rewrite(bundle_of(box, "a3"), numbers))
    assert code == 2 and any("setup.json's box" in x for x in said), said  # 308fcf3: TypeError
    assert any("setup.json's sessions" in x for x in said)
    assert not sf.task_dir(config.OUT_DIR / ROW, TASK).exists()


def test_20_the_whole_run_is_checked_before_the_first_question(svc, monkeypatch, tmp_path):  # noqa: F811
    asked = []
    monkeypatch.setattr(sf, "ask_task", lambda rec, task, row, on, progress, canceled, log,
                        sid=None:
                        (asked.append(task), (1, 1))[1])
    monkeypatch.setattr(fb, "load", lambda task, root=None: [])
    monkeypatch.setattr(config, "FRONTIER_SCORE_AFTER_RUN", False)
    sid = db.add("served/x", "instruct", "frontier", "masein", "n")
    ctx = fb.slot_context(["gpqa_diamond_epoch"], True)               # 83,968
    status, line = sf.run(sid, {"tasks": "[]"}, {"id": "served/x", "pin": {"ctx": ctx}},
                          {"on": True}, tmp_path / "row", tmp_path / "log")
    fit = [t for t in fb.TASKS if fb.slot_context([t], True) <= ctx]
    # 308fcf3: stopped at HLE, SimpleQA and MMLU-Pro never asked
    assert asked == fit and HLE not in asked and {"simpleqa_epoch", MMLU} <= set(asked)
    assert status == "failed" and "Humanity's Last Exam" in line and "Not asked; the rest are" \
        in line


def test_21_where_in_one_wording_an_unknown_gpu_and_a_box_of_two_cards(monkeypatch):
    import import_frontier as imf
    import report_lm_eval as rle
    assert imf.where_words(["a GPU"], []) == "rented GPU"              # 308fcf3: "· a GPU"
    assert rle.frontier_where({"where": "run on a rented GPU (NVIDIA GeForce RTX 4090)"}) == \
        "rented GPU · RTX 4090" == imf.where_words(["NVIDIA GeForce RTX 4090"], [])
    two = "NVIDIA GeForce RTX 5090, 575.51, 32607\nNVIDIA GeForce RTX 5090, 575.51, 32607\n"
    monkeypatch.setattr(rb.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, stdout=two))
    g = rb.gpu_info()
    assert g["count"] == 2 and g["names"] == ["NVIDIA GeForce RTX 5090"] * 2
    assert imf.where_words(imf.gpu_names({"gpu": g}), ["A3"]) == \
        "rented GPU · 2 × RTX 5090 · box A3"                            # 308fcf3: its first


def test_21_the_loop_count_reads_letter_counters_and_dots_never_tables_or_colour_grids():
    import itertools
    import string
    two = ["".join(p) for p in itertools.product(string.ascii_lowercase, repeat=2)]
    case = "".join(f"Case {x}: we try the next arrangement of the pieces and check whether it "
                   "holds the rule.\n" for x in two[:300])
    dots = "The answer is " + "." * 3000
    table = "".join(f"| step {k} | value {k * 3} | ratio {k / 7:.3f} |\n" for k in range(400))
    grid = "".join("black black red blue green green black yellow\n" for _ in range(400))
    assert fb.ends_in_loop(case) and fb.ends_in_loop(dots)
    assert not fb.ends_in_loop(table) and not fb.ends_in_loop(grid)
