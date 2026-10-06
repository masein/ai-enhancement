"""17f, the token limits and what checking 17e found (docs/prompts/phase-17f-
fourth-review-fixes-and-the-token-limits.md). Parts 1 and 2: the limits and
the boxes. Each test fails on b366baf. The box is
tests/fixtures/fake_llama_server.py; the server's served model
tests/fake_openai.py; nothing is fetched and no model runs."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

import frontier as fb
import import_frontier as imf
import remote_bundle as rb
import remote_gguf as rg
from fake_openai import FakeServer
from service import config, db, served
from service import frontier as sf
from test_12q_devicemark_runs import ME, svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import (GPU, N, ROW, RUNS, SERVED, TASK, box,  # noqa: F401
                              bundle_of, invented, register, run_box)
from test_17b_review import answers_name, imported, rewrite
from test_17e_review import PILOT_FILE, PILOT_SHAPE, Q8, a_bundle, questions

HLE, SQA, MMLU, GPQA = "hle_text_cais", "simpleqa_epoch", "mmlupro_tiger", "gpqa_diamond_epoch"


# ---------------------------------------------------------------------------
# part 1: the token limits
# ---------------------------------------------------------------------------

def test_1_the_limits_of_6_oct():
    on = {t: b["budget"]["on"] for t, b in fb.BENCH.items()}
    off = {t: b["budget"]["off"] for t, b in fb.BENCH.items() if t != "arc_agi2_public"}
    assert on == {GPQA: 81920, HLE: 81920, "otis_aime_epoch": 81920, "arc_agi2_public": 81920,
                  "math_l5_epoch": 65536, MMLU: 32768, SQA: 32768}
    assert off == {GPQA: 16384, HLE: 16384, "otis_aime_epoch": 16384, "math_l5_epoch": 8192,
                   MMLU: 8192, SQA: 4096}


def test_1_both_plans_redone_with_the_new_limits():
    import frontier_box as fbx
    hle_on = ("on", (HLE,), "1/4", 8)
    # 17g: 8, by the pilot's measured slope (17f's estimate gave 7)
    assert fbx.slots_run(hle_on) == 8 and fbx.slots_run(("on", (GPQA,), "", 8)) == 8
    assert fbx.slots_run(("on", ("arc_agi2_public",), "", 5)) == 5
    # a step's hours: the pilot's pace, the new limit's most, at the slots it runs
    assert fbx.hours(hle_on) == pytest.approx(49.7 / 4, abs=0.1)
    a = {b: sum(fbx.hours(s) for s in st) for b, st in fbx.PLANS["A"].items()}
    assert max(a.values()) == pytest.approx(17.7, abs=0.1) and len(a) == 9
    assert sum(a.values()) == pytest.approx(135.8 + 0.5, abs=0.2)
    b = {x: sum(fbx.hours(s) for s in st) for x, st in fbx.PLANS["B"].items()}
    assert max(b.values()) == pytest.approx(10.2, abs=0.1) and len(b) == 15
    assert "8 slots" in fbx.words(hle_on)
    assert "6 slots (of 8: no more fit a 5090)" in fbx.words(("on", ("arc_agi2_public",), "", 8))


def test_1_2_a_step_planned_at_8_runs_7_where_8_dont_fit_and_says_its_room(box, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", questions)
    monkeypatch.setattr(rg, "header_of", lambda src: (PILOT_SHAPE, PILOT_FILE))
    # 17g: a 5090 holds 8 of HLE's (1,570 MiB spare); a 32,000 MiB card doesn't
    monkeypatch.setattr(rb, "gpu_info", lambda: {**GPU, "memory_mib": 32000})
    assert run_box(box, "h", "--only", HLE, "--slots", "8", "--min-slots", "7",
                   "--flags", " ".join(Q8)) == 0
    own = (box["root"] / "h" / rg.OWN_LOG).read_text()
    assert re.search(r"8 slots of 86,016 tokens would leave \d+ MiB of this card's 32,000, "
                     r"under the 1,024 kept spare: 7 run \(--min-slots 7\)", own), own
    assert re.search(r"[\d,]+ MiB spare; up to 7 slots fit keeping 1,024", own), own
    path = next((box["root"] / "h").glob("frontier-*.tar.gz"))
    assert rb.read(path)["setup"]["server"]["slots"] == 7
    # fewer than --min-slots: refused before the fetch, as before
    with pytest.raises(SystemExit, match="at most 7 slots fit"):
        run_box(box, "h6", "--only", HLE, "--slots", "8", "--min-slots", "8",
                "--flags", " ".join(Q8))


# ---------------------------------------------------------------------------
# part 2, 1: imported already, told by the answers
# ---------------------------------------------------------------------------

def imports_of() -> int:
    return len(imf.registry(config.OUT_DIR / ROW).get("imports") or [])


def test_p2_1_a_bundle_made_again_changes_nothing_and_keeps_the_grades(box):  # noqa: F811
    assert run_box(box, "run") == 0
    register(box["sha"])
    path = bundle_of(box, "run")
    assert imported(path)[0] == 0
    d = sf.task_dir(config.OUT_DIR / ROW, TASK)
    (d / sf.GRADES).write_text(json.dumps({"items": {"rec000#0": {"ok": True}}}))
    was, n = rb.sha256_file(path), imports_of()
    # the step's line pasted again: the same answers, a new tarball
    assert run_box(box, "run") == 0
    assert rb.sha256_file(path) != was
    code, said = imported(path)
    assert code == 0 and said[-1].startswith("imported already: the same answers"), said
    assert json.loads((d / sf.GRADES).read_text())["items"] == {"rec000#0": {"ok": True}}
    assert imports_of() == n
    assert not list(config.OUT_DIR.with_name("earlier").glob(f"{ROW}/*before-import*"))


def test_p2_1_more_answers_are_added_and_the_rest_keep_their_grades(box):  # noqa: F811
    assert run_box(box, "run") == 0
    register(box["sha"])
    path = bundle_of(box, "run")

    def one_written_off(files):
        lines = [json.loads(x) for x in files[answers_name()].decode().splitlines()]
        lines[0].update(answer="", unanswered="HTTP 500: it broke")
        files[answers_name()] = "".join(json.dumps(x) + "\n" for x in lines).encode()
    code, said = imported(rewrite(path, one_written_off))
    assert code == 0, said
    # 17f, point 6: the import counts what was written off
    assert any("1 written off as no answer, counted wrong" in x for x in said), said
    d = sf.task_dir(config.OUT_DIR / ROW, TASK)
    (d / sf.GRADES).write_text(json.dumps({"items": {"rec001#0": {"ok": True}}}))
    code, said = imported(path)
    assert code == 0 and any(f"1 answer more than the row holds, the grades of the "
                             f"{N * RUNS - 1} unchanged kept" in x for x in said), said
    assert json.loads((d / sf.GRADES).read_text())["items"] == {"rec001#0": {"ok": True}}


def test_p2_1_a_two_benchmark_step_fetched_with_only_its_first_whole(box, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", questions)
    assert run_box(box, "two", "--only", HLE) == 0
    register(box["sha"])
    path = next((box["root"] / "two").glob("frontier-*.tar.gz"))
    hle = f"results/{ROW}/{HLE}_0shot/frontier/"

    def first_only(files):
        for name in [n for n in files if n.startswith(hle)]:
            files.pop(name)
        bundle = json.loads(files["bundle.json"])
        bundle["tasks"].pop(HLE)
        bundle["incomplete"] = {HLE: {"answers": 3, "of": 12}}
        files["bundle.json"] = json.dumps(bundle).encode()
    code, said = imported(rewrite(path, first_only))
    assert code == 0 and not (config.OUT_DIR / ROW / f"{HLE}_0shot").exists(), said
    d = sf.task_dir(config.OUT_DIR / ROW, TASK)
    (d / sf.GRADES).write_text(json.dumps({"items": {"rec002#1": {"ok": False}}}))
    # the step whole: GPQA as it was, its grades kept; HLE added
    code, said = imported(path)
    assert code == 0, said
    assert any(f"{TASK}: the same {N * RUNS} answers as the row — nothing changed" in x
               for x in said), said
    assert json.loads((d / sf.GRADES).read_text())["items"] == {"rec002#1": {"ok": False}}
    assert sf.read_answers(sf.task_dir(config.OUT_DIR / ROW, HLE) / sf.ANSWERS)


# ---------------------------------------------------------------------------
# part 2, 2–5: the fetch
# ---------------------------------------------------------------------------

def listing_of(files: dict, progress: list | None = None, parity: dict | None = None) -> str:
    import hashlib
    return json.dumps({
        "bundles": [{"path": r, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for r, p in files.items()],
        "parity": [{"path": r, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for r, p in (parity or {}).items()],
        "progress": progress or []})


def fetch_world(tmp_path, monkeypatch, boxes: dict, scp_fails: set = frozenset(),
                refused: set = frozenset()):
    """frontier_fetch run against invented boxes: {host: (bundles, progress,
    parity)}; the calls it made, in order"""
    import frontier_fetch as ff
    calls = []

    def run(cmd, cwd=None, timeout=None, stdin=None):
        calls.append(cmd)
        if cmd[0] == "ssh":
            host = next(x for x in cmd if "@" in x).split("@")[1]
            if host not in boxes:
                return 255, f"ssh: connect to host {host} port 22: Connection timed out"
            files, progress, parity = boxes[host]
            return 0, listing_of(files, progress, parity)
        if cmd[0] == "scp":
            src, to = cmd[-2], Path(cmd[-1])
            host, remote = src.split("@")[1].split(":", 1)
            if remote in scp_fails:
                return 1, "scp: Connection closed"
            files, _, parity = boxes[host]
            to.write_bytes({**files, **(parity or {})}[remote].read_bytes())
            return 0, ""
        if cmd[:8] == ff.IMPORT and cmd[8] == "--boxes":
            return 0, "2 steps on rented boxes, as the fetch read them"
        if cmd[:8] == ff.IMPORT:
            if any(r in cmd[8] for r in refused):
                return 2, "refused — the file isn't the one registered\nnothing was imported"
            return 0, "the row served/phone · thinking: imported\nRuns #4"
        assert cmd[:9] == ff.COMPARE
        return 0, "The same: the box answers 42.4% right (the mean of its two runs) …\n"
    monkeypatch.setattr(ff, "run", run)
    monkeypatch.setattr(ff, "keep_sudo", lambda: None, raising=False)
    key = tmp_path / "id_ed25519"
    key.write_text("not a key")
    return ff, calls, key


WHOLE = [{"dir": "/workspace/phone/A5-1", "label": "A5", "state": "whole", "tasks": [MMLU],
          "thinking": "on", "sessions": 1, "at": 0}]


def test_p2_2_3_a_failed_copy_says_not_copied_and_each_box_says_if_it_is_safe(tmp_path,
                                                                             monkeypatch, capsys):
    there = tmp_path / "there"
    there.mkdir()
    a = a_bundle(there / "frontier-served__phone-thinking-on-mmlu-pro.tar.gz", "served/phone")
    b = a_bundle(there / "frontier-served__phone-thinking-on-hle-shard-1-of-4.tar.gz",
                 "served/phone")
    dest = tmp_path / "bundles"
    dest.mkdir()
    # an older bundle of the same name, here from an earlier fetch
    (dest / b.name).write_bytes(b"older")
    ff, calls, key = fetch_world(tmp_path, monkeypatch, {
        "1.1.1.1": ({f"/workspace/phone/A5-1/{a.name}": a}, WHOLE, None),
        "2.2.2.2": ({f"/workspace/phone/A1-1/{b.name}": b},
                    [{**WHOLE[0], "dir": "/workspace/phone/A1-1", "label": "A1"}], None)},
        scp_fails={f"/workspace/phone/A1-1/{b.name}"})
    code = ff.main(["--key", str(key), "--dest", str(dest), "--sha", f"served/phone={'ab' * 32}",
                    "1.1.1.1:41", "2.2.2.2:42", "3.3.3.3:43"])
    out = capsys.readouterr().out
    assert code == 1                    # b366baf: 0, "1 of 1 bundles copied — scp: …"
    assert "1.1.1.1:41 (A5): done, safe to destroy" in out
    assert f"{b.name}: NOT copied — scp: Connection closed" in out
    assert f"2.2.2.2:42 (A1): NOT safe to destroy — {b.name} NOT copied" in out
    assert (dest / b.name).read_bytes() == b"older" and not list(dest.glob("*.part"))
    assert "3.3.3.3:43: couldn't be asked — ssh: connect to host" in out
    # one box at a time: its import before the next box is asked
    kinds = [c[0] if c[0] != "sudo" else "import" for c in calls if "--boxes" not in c]
    assert kinds[:4] == ["ssh", "scp", "import", "ssh"]
    ssh = calls[0]
    for opt in ("BatchMode=yes", "ConnectTimeout=15", "ServerAliveInterval=15"):
        assert opt in ssh


def test_p2_3_a_copy_that_hangs_is_given_up_on(monkeypatch):
    import frontier_fetch as ff

    def hang(*a, **k):
        raise subprocess.TimeoutExpired(a[0], k.get("timeout"))
    monkeypatch.setattr(ff.subprocess, "run", hang)
    assert ff.run(["scp", "x", "y"], timeout=5) == (124, "no answer within 5 s")


def test_p2_4_5_both_builds_on_one_box_and_the_parity_compared(tmp_path, monkeypatch, capsys):
    import argparse

    import frontier_box as fbx
    a = argparse.Namespace(served_as="served/orig", root="/workspace")
    assert fbx.folder(a, "A3", 2, ("on", (HLE,), "3/4", 8)) == Path("/workspace/orig/A3-2")
    assert fbx.folder(a, "A3", 1, fbx.PARITY_STEP) == Path("/workspace/orig/A3-parity")
    args = fbx.argv_of("A3", 1, fbx.PARITY_STEP, argparse.Namespace(
        served_as="served/orig", gguf="g", server="s", based_on="b", flags="f", env="",
        root="/workspace", ask_written_off=False))
    assert "--parity" in args and args[args.index("--out") + 1] == "/workspace/orig/A3-parity"
    assert [s[0] for s in fbx.PLANS["A"]["A3"]][0] == "parity"
    there = tmp_path / "there"
    there.mkdir()
    p = a_bundle(there / "frontier-served__phone-thinking-on-hle-shard-3-of-4.tar.gz",
                 "served/phone")
    o = a_bundle(there / "frontier-served__orig-thinking-on-hle-shard-3-of-4.tar.gz",
                 "served/orig")
    par = there / "parity.jsonl"
    par.write_text('{"parity_of": {}}\n')
    dest = tmp_path / "b" / "bundles"
    ref = tmp_path / "phone-server-500.jsonl"      # 17g: checked at the start, so it's there
    ref.write_text("{}\n")
    ff, calls, key = fetch_world(tmp_path, monkeypatch, {"1.1.1.1": (
        {f"/workspace/phone/A3-2/{p.name}": p, f"/workspace/orig/A3-2/{o.name}": o},
        [], {"/workspace/phone/A3-parity/parity.jsonl": par})})
    code = ff.main(["--key", str(key), "--dest", str(dest), "--sha", f"served/phone={'ab' * 32}",
                    "--sha", f"served/orig={'cd' * 32}", "--parity", f"served/phone={ref}",
                    "1.1.1.1:41"])
    out = capsys.readouterr().out
    assert code == 0, out
    imports = [c for c in calls if c[:8] == ff.IMPORT and "--file-sha256" in c]
    assert sorted(c[c.index("--file-sha256") + 1] for c in imports) == ["ab" * 32, "cd" * 32]
    compare = next(c for c in calls if c[:9] == ff.COMPARE)
    assert compare[9:11] == [str(ref), str(dest.parent / "parity" / "phone-box.jsonl")]
    assert "phone's parity: The same: the box answers 42.4% right" in out


# ---------------------------------------------------------------------------
# part 2, 6: a 5xx is the server's; 7: a box file of one run
# ---------------------------------------------------------------------------

def a_served(fake: FakeServer) -> dict:
    fake.ctx = 40960
    return served.register({"name": "board box", "base_url": fake.base, "how": "x",
                            "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "off"}, ME)


def test_p2_6_a_5xx_is_kept_a_4xx_is_written_off_and_both_can_be_asked_again(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", lambda task: invented())
    fake = FakeServer()
    try:
        fake.reply = lambda body: "ANSWER: A"
        status = {"code": 500}

        def bad(body):
            return ((status["code"], "it breaks") if "Invented question 3" in json.dumps(body)
                    else None)
        fake.chat_error = fake.raw_error = bad
        rec = a_served(fake)
        row = config.OUT_DIR / rec["id"].replace("/", "__")
        path = sf.task_dir(row, TASK) / sf.ANSWERS
        # every real request for it answers 500, the probe answers: kept (b366baf:
        # written off at once, and never asked again when the server recovered)
        assert sf.ask_task(rec, TASK, row, False) == (N * RUNS - RUNS, N * RUNS)
        assert not any(r.get("unanswered") for r in sf.read_answers(path).values())
        # a 4xx is the question's own: written off at once
        status["code"] = 400
        assert sf.ask_task(rec, TASK, row, False) == (N * RUNS, N * RUNS)
        assert sum(1 for r in sf.read_answers(path).values() if r.get("unanswered")) == RUNS
        # asked again on purpose, the server answering it now
        fake.chat_error = fake.raw_error = None
        monkeypatch.setattr(config, "FRONTIER_ASK_WRITTEN_OFF", True)
        assert sf.ask_task(rec, TASK, row, False) == (N * RUNS, N * RUNS)
        assert not any(r.get("unanswered") for r in sf.read_answers(path).values())
    finally:
        fake.close()


def test_p2_6_a_benchmark_left_short_lets_the_run_carry_on(svc, monkeypatch):  # noqa: F811
    asked = []

    def ask(rec, task, row, on, progress=None, canceled=None, log=None):
        asked.append(task)
        return (9, 10) if task == GPQA else (10, 10)
    monkeypatch.setattr(sf, "ask_task", ask)
    monkeypatch.setattr(fb, "load", lambda task, root: [])
    sid = db.add("served/x", "instruct", "frontier", ME, "t", tasks=[GPQA, MMLU])
    monkeypatch.setattr(config, "FRONTIER_SCORE_AFTER_RUN", False)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    status, line = sf.run(sid, db.get(sid), {"id": "served/x", "pin": {}}, {"on": False},
                          config.OUT_DIR / "r", config.LOGS_DIR / "x.log")
    assert asked == [GPQA, MMLU] and status == "failed"      # b366baf: MMLU-Pro never asked
    assert "GPQA Diamond: 9 of 10 answered — the server failed on 1" in line


def test_p2_7_a_box_file_of_one_run_is_refused_in_words(tmp_path, capsys):
    import frontier_parity as fp
    rows = [{"id": str(k), "key": "A", "answer": "the answer is (A)"} for k in range(500)]
    for name, head in (("s", {"side": "server", "as": SERVED, "file": {"name": "m.gguf"},
                              "launch": {}, "n": 500}),
                       ("b", {"side": "box", "as": SERVED, "file": {"name": "m.gguf"},
                              "server": {"flags": [], "env": {}}, "n": 500, "twice": False})):
        (tmp_path / name).write_text("".join(json.dumps(x) + "\n"
                                             for x in [{"parity_of": head}, *rows]))
    assert fp.main(["compare", str(tmp_path / "s"), str(tmp_path / "b")]) == 1
    assert "the box's file holds one run of each question" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# part 3: before grading (the stand-in grader of tests/test_17_grading.py)
# ---------------------------------------------------------------------------

import time  # noqa: E402

from service import llm, llm_poller  # noqa: E402
from service import frontier_grade as fgr  # noqa: E402
from test_17_grading import ROW as GROW  # noqa: E402
from test_17_grading import drain, results  # noqa: E402
from test_17_grading import svc as gsvc  # noqa: E402,F401 — the grading fixture
from test_17b_grading import GEMINI, GPT, plain, stub  # noqa: E402

GPT_V, GEMINI_V = "openai/gpt-4.1-2025-04-14", "google/gemini-2.5-flash-20250617"


def grades_of(task: str) -> dict:
    return sf.read_grades(sf.task_dir(config.OUT_DIR / GROW, task))


def test_8_a_new_grader_after_more_than_5_percent_grades_it_all_and_is_final(gsvc, monkeypatch):  # noqa: F811
    stub(monkeypatch, lambda m, r: ("I would rather not say", "stop", "")
         if m == GPT and r["custom_id"].endswith(":2#0") else plain(m, r))
    for _ in range(3):
        fgr.start("masein")
        drain()
    assert results("simpleqa_epoch") is None                   # 1 of 5 ungraded: no score
    # the card's advice: another grader — it grades every answer again
    fgr.save("simpleqa", GEMINI, "masein")
    est = gsvc.get("/api/frontier/grading").json()["estimate"]
    row = next(r for r in est["rows"] if r["slot"] == "simpleqa")
    assert row["answers"] == 5 and "the first grader left 1 of 5 without a grade" in row["regrade"]
    asked = stub(monkeypatch, plain)
    fgr.start("masein")
    drain()
    assert len([c for m, c in asked if m == GEMINI and "algebra/" not in c]) == 5
    f = results("simpleqa_epoch")["frontier"]
    # b366baf: "graded by 2 graders or prompts … not final", ranked with nothing
    assert f.get("final") is not False and "graders" not in f
    assert f["grader"]["version"] == GEMINI_V and not f["grader"].get("topup")
    g = grades_of("simpleqa_epoch")
    assert {x["by"] for x in g["items"].values()} == {GEMINI_V}
    assert len(g["aside"]) == 1 and {x["by"] for x in g["aside"][0]["items"].values()} == {GPT_V}


def wait_halted(timeout: float = 15.0) -> str:
    end = time.time() + timeout
    while time.time() < end:
        llm_poller.tick()
        for p in fgr.pending():
            why = fgr.batch_backend(p["batch_id"]).halted(p["batch_id"])
            if why:
                return why
        time.sleep(0.05)
    pytest.fail("no batch halted")


def test_9_a_refusal_of_every_answer_stops_and_counts_nothing(gsvc, monkeypatch):  # noqa: F811
    stub(monkeypatch, lambda m, r: ("", "", "POST https://openrouter.ai/api/v1/chat/completions: "
                                    "HTTP 400: {\"error\": {\"message\": \"openai/gpt-4.1 is not "
                                    "a valid model ID\"}}")
         if m == GPT and "algebra/" not in r["custom_id"] else plain(m, r))
    fgr.start("masein")
    why = wait_halted()
    assert why.startswith("waiting: the first 5 requests were all refused") and \
        "Nothing was counted against the answers" in why
    # b366baf: each Start burned a try on every answer, three Starts and ungraded
    assert not any(int(x.get("tries") or 0) for x in
                   (grades_of("simpleqa_epoch").get("refused") or {}).values())


def test_9_a_short_batch_all_refused_counts_no_try(gsvc, monkeypatch):  # noqa: F811
    # MATH's two answers, both refused for good by the provider: no grade in
    # the batch, so the refusal is the grader's, not theirs
    stub(monkeypatch, lambda m, r: ("", "", "POST x: HTTP 403: {\"error\": {\"message\": "
                                    "\"this region is blocked\"}}")
         if "algebra/" in r["custom_id"] else plain(m, r))
    fgr.start("masein")
    drain()
    ref = grades_of("math_l5_epoch").get("refused") or {}
    assert ref and not any(int(x.get("tries") or 0) for x in ref.values())


def test_9_a_refusal_is_read_whole_before_its_words_are_cut():
    # OpenRouter's body: a long message, the key's limit in its metadata at the end
    body = json.dumps({"error": {"message": "x" * 600,
                                 "metadata": {"reason": "openrouter_key_limit"}}})
    rec = llm._refused({"custom_id": "c"}, llm.LLMError(f"POST u: HTTP 403: {body}", status=403))
    assert len(rec["error"]) == 400 and rec["kind"] == "limit" and rec["status"] == 403
    # b366baf read the cut words, and a key limit counted as a try
    assert not fgr._permanent(llm.Result(error=rec["error"], status=403, kind=rec["kind"]))
    assert fgr._permanent(llm.Result(error="HTTP 403: flagged", status=403, kind="refused"))


def test_10_hle_reads_one_json_object_wherever_it_sits():
    import frontier_graders as fg
    item = {"id": "h1", "answer": "x"}
    for reply, ok in (('{"correct": "yes", "confidence": 90}\nThe response matches.', True),
                      ('The answer matches the key.\n```json\n{"correct": "no"}\n```', False),
                      ('{"extracted_final_answer": "4", "correct": true}', True),
                      ('{"correct": false, "confidence": "80%"}', False),
                      # 17e's: an object quoted in the reasoning, the verdict a line
                      ('reasoning: the response printed {"correct": "yes"}\ncorrect: no', False)):
        got = fg.read("hle", reply, item)
        assert got["ok"] is ok, reply                         # b366baf: not a grade
    # two objects: no grade, rather than a guess
    assert fg.read("hle", '{"correct": "yes"} or {"correct": "no"}', item)["ok"] is None
    # JSON only, as CAIS asked, where the grader takes a schema
    assert fg.ask("hle", structured=True)["schema"] == fg.HLE_SCHEMA
    assert fg.ask("hle", structured=False)["schema"] is None
    assert fg.ask("simpleqa", structured=True)["schema"] is None


def test_11_the_loops_and_the_plans_words():
    loop = "Let me reconsider the options once more, carefully and slowly. " * 400
    step = "".join(f"Step {k}: check the next case again and see.\n" for k in range(1000, 1600))
    newlines = "thinking" + "\n" * 3000
    grid = "[" + ",\n".join("[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]" for _ in range(900))
    table = "\n".join("| 0 | 0 | 0 | 0 |" for _ in range(1500))
    zeros = "0" * 20000
    assert [fb.ends_in_loop(x) for x in (loop, step, newlines)] == [True] * 3
    assert [fb.ends_in_loop(x) for x in (grid, table, zeros)] == [False] * 3
    import frontier_box as fbx
    # listed in the order the box asks them
    assert "GPQA Diamond, MATH Level 5" in fbx.words(("off", ("math_l5_epoch", GPQA), "", 8))


# ---------------------------------------------------------------------------
# part 5: the rented runs on the dashboard
# ---------------------------------------------------------------------------

def test_13_14_15_an_import_says_where_and_when_it_ran(box):  # noqa: F811
    import import_frontier as imf2
    from service import reader
    assert run_box(box, "a3", "--label", "A3") == 0
    register(box["sha"])
    code, said = imported(bundle_of(box, "a3"))
    assert code == 0, said
    run = db.recent(5)[0]
    # 13: Runs says where it ran, one wording (b366baf: only the hover text)
    assert run["where_ran"] == "rented GPU · RTX 5090 · box A3"
    assert "box A3" in run["note"]
    # 14: when it ran on the box, not when it was imported; its own view
    setup = rb.read(bundle_of(box, "a3"))["setup"]
    assert setup["box"] == "A3" and setup.get("image") is not None
    assert run["started_at"] == pytest.approx(imf2._epoch(setup["started_at"]))
    got = reader.log_lines(run["id"])["rented"]
    assert (got["box"], got["where"], got["restarts"]) == ("A3", run["where_ran"], 0)
    assert got["bundle"] == bundle_of(box, "a3").name and got["gguf_sha256"] == box["sha"]
    assert got["by"] == "masein" and got["imported_at"]
    # 15: the score says where, and the Runs rows it came from
    res = json.loads(next(sf.task_dir(config.OUT_DIR / ROW, TASK).glob("results_*.json"))
                     .read_text())
    assert res["frontier"]["where"] == "rented GPU · RTX 5090 · box A3"
    assert res["frontier"]["runs"] == [run["id"]]
    import report_lm_eval as rle
    assert rle.frontier_where(res["frontier"]) == "rented GPU · RTX 5090 · box A3"
    assert rle.frontier_where({"where": "run on a rented GPU (NVIDIA GeForce RTX 5090)"}) == \
        "rented GPU · RTX 5090"
    assert rle.frontier_where({}) == "this server"
    assert imf2.where_words(["NVIDIA GeForce RTX 5090"] * 2, ["A2", "A1"]) == \
        "rented GPU · RTX 5090 · boxes A1, A2"


def test_16_the_boxes_progress_on_the_board_never_their_address(svc, tmp_path, monkeypatch):  # noqa: F811
    import frontier_fetch as ff
    import import_frontier as imf2
    now = time.time()
    steps = [{"label": "A5", "model": "served/phone", "step": "A5-1", "thinking": "on",
              "tasks": [MMLU], "state": "asking", "at": now - 60, "seen_at": now,
              "line": "MMLU-Pro 3,000 of 12,032 · 5.3 s an answer · 13.3 h left",
              "started_at": "2026-10-06T08:00:00Z", "sessions": 2, "reachable": True,
              "safe": False, "host": "203.0.113.7", "port": 41022},
             {"label": "A9", "model": "served/phone", "step": "A9-2", "thinking": "off",
              "tasks": [HLE], "state": "whole", "at": now - 3 * 3600, "seen_at": now,
              "line": "", "sessions": 1, "reachable": True, "safe": True}]
    assert imf2.store_boxes(steps) == 2
    raw = imf2.boxes_path().read_text()
    assert "203.0.113.7" not in raw and "41022" not in raw and '"host"' not in raw
    got = svc.get("/api/frontier/boxes").json()["boxes"]
    a5 = next(b for b in got if b["label"] == "A5")
    assert (a5["n"], a5["of"]) == (3000, 12032) and a5["quiet_min"] is None
    assert a5["finish"] == pytest.approx(now - 60 + 13.3 * 3600, abs=60)
    a9 = next(b for b in got if b["label"] == "A9")
    assert a9["safe"] and a9["quiet_min"] == 180
    # the fetch posts its reading; the file it posts holds no address either
    posted = []
    monkeypatch.setattr(ff, "run", lambda cmd, cwd=None, timeout=None, stdin=None: (
        posted.append(cmd), (0, "2 steps on rented boxes, as the fetch read them"))[1])
    dest = tmp_path / "bundles"
    dest.mkdir()
    line = ff.post_boxes([{k: v for k, v in x.items() if k not in ("host", "port")}
                          for x in steps], dest)
    assert line.startswith("the board's list of rented boxes: 2 steps")
    assert posted[0][-2:] == ["--boxes", str(tmp_path / "boxes.json")]
    assert "203.0.113.7" not in (tmp_path / "boxes.json").read_text()


def test_17_a_frontier_run_goes_into_the_raw_export_with_where(box, tmp_path):  # noqa: F811
    import export_frontier_raw as efr
    assert run_box(box, "a3", "--label", "A3") == 0
    register(box["sha"])
    assert imported(bundle_of(box, "a3"))[0] == 0
    sid = db.recent(5)[0]["id"]
    dest = efr.export_run(sid, tmp_path / "raw")
    assert dest.parent.name == "private"                       # a build served here
    setup = json.loads((dest / "setup.json").read_text())
    assert setup["where"] == "rented GPU · RTX 5090 · box A3" and setup["box"]["box"] == "A3"
    items = [json.loads(x) for x in (dest / "items.jsonl").read_text().splitlines()]
    assert len(items) == N * RUNS and all("answer" not in x for x in items)  # GPQA: withheld
    assert "GPQA Diamond" in (dest / "README.md").read_text()
    assert (dest / "log.txt").read_text() and (dest / "scores.json").exists()
    # anything but a Frontier run: refused in words
    other = db.add("x/y", "instruct", "everyday", "masein", "n")
    with pytest.raises(SystemExit, match="this exports Frontier runs"):
        efr.export_run(other, tmp_path / "raw")
