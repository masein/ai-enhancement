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
    imf.store_boxes([{"label": "A5", "model": "served/phone", "step": "A5-1",
                      "state": "stopped", "why": why}])
    assert imf.read_boxes()["boxes"][0]["why"] == why
    src = Path(__file__).resolve().parents[1].joinpath("scripts", "report_lm_eval.py").read_text()
    assert "b.state === 'stopped' ? `stopped${b.why ? ': ' + b.why : ''}`" in src


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
            '--register "<its name>" --file-sha256 <its sha256>') in out
    assert not [c for c in calls if c[:8] == ff.IMPORT]


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
