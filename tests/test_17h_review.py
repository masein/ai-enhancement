"""17h, what checking 17g found (docs/prompts/phase-17h-sixth-review-grading-
export-fetch.md). Parts 2 to 5, a test a point, each failing on 0adb522 (part
1 is tests/test_17h_grading_rules.py). The boxes are invented listings and a
stand-in ssh; nothing is fetched, no model runs, no paid API is called."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

import remote_bundle as rb
import remote_gguf as rg
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import GPU, box, run_box  # noqa: F401
from test_17e_review import PILOT_FILE, PILOT_SHAPE, Q8, a_bundle, questions
from test_17f_review import listing_of
from test_17g_review import MMLU, PHONE, fetch_one, progress_of

HLE = "hle_text_cais"
REPO = Path(__file__).resolve().parents[1]


def main_of(ff, key, dest, *more):
    return ff.main(["--key", str(key), "--dest", str(dest), "--sha", PHONE, "--no-board", *more])


# ---------------------------------------------------------------------------
# part 2, 25: the box's banner, on stderr, is never read as its listing
# ---------------------------------------------------------------------------

STAND_IN = r'''#!{python}
import json, sys
open({argv!r}, "a").write(json.dumps(sys.argv) + "\n")
sys.stderr.write("Welcome to vast.ai. If authentication fails, try again after a few seconds,"
                 " and double check your ssh key.\nHave fun!\n")
sys.stderr.write("Warning: Permanently added '[ssh4.vast.ai]:41234' (ED25519) to the list of "
                 "known hosts.\n")
if sys.argv[0].endswith("scp"):
    open(sys.argv[-1], "w").write("x")
else:
    sys.stdin.read()
    print(json.dumps({{"bundles": [], "parity": [], "progress": [], "steps": []}}))
'''


def test_25_a_banner_on_stderr_is_never_the_listing(tmp_path, monkeypatch):
    import frontier_fetch as ff
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    argv = tmp_path / "argv.jsonl"
    for name in ("ssh", "scp"):
        f = bin_ / name
        f.write_text(STAND_IN.format(python=sys.executable, argv=str(argv)))
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    one = ff._BOX.fullmatch("ssh4.vast.ai:41234")
    got, why = ff.listing(one, str(tmp_path / "key"))
    assert why == "" and got == {"bundles": [], "parity": [], "progress": [], "steps": []}, why
    ok, words = ff.copy(one, str(tmp_path / "key"), "/workspace/x", "nope", tmp_path)
    for line in argv.read_text().splitlines():
        cmd = json.loads(line)
        assert cmd[cmd.index("LogLevel=ERROR") - 1] == "-o", cmd      # 0adb522: none
    # a failure still says why, from stderr
    code, said = ff.run([sys.executable, "-c", "import sys; sys.stderr.write('no route');"
                                              "sys.exit(255)"])
    assert code == 255 and "no route" in said


# ---------------------------------------------------------------------------
# 6: --ask-written-off asks a whole step again
# ---------------------------------------------------------------------------

def test_6_ask_written_off_runs_a_whole_step_that_wrote_some_off(tmp_path, monkeypatch):
    import frontier_box as fbx
    ran = []
    monkeypatch.setattr(fbx.subprocess, "run", lambda cmd, *k, **kw: (
        ran.append(Path(cmd[cmd.index("--out") + 1]).name),
        subprocess.CompletedProcess(cmd, 0))[-1])
    for k, off in ((1, 2), (2, 0)):
        out = tmp_path / "phone" / f"A1-{k}"
        rg.progress(out, state="whole", bundle={"name": "b.tar.gz"}, written_off=off)
        (out / "b.tar.gz").write_bytes(b"x")
    line = ["A1", "--as", "served/phone", "--gguf", "g", "--server", "s", "--based-on", "b",
            "--flags", "f", "--root", str(tmp_path)]
    assert fbx.main(line) == 0 and ran == []                # whole: not run again
    assert fbx.main([*line, "--ask-written-off"]) == 0
    assert ran == ["A1-1"]                                  # 0adb522: nothing asked


# ---------------------------------------------------------------------------
# 7, 8: every step on the box counts, wherever its folder
# ---------------------------------------------------------------------------

def test_7_a_step_outside_the_plans_folders_keeps_the_box_not_safe(tmp_path, monkeypatch,
                                                                   capsys):
    there = tmp_path / "there"
    there.mkdir()
    b = a_bundle(there / "frontier-served__phone-thinking-on-mmlu-pro.tar.gz", "served/phone")
    # --out /workspace/run, stopped at 12 of 720 with a partial bundle home
    ff, calls, key = fetch_one(tmp_path, monkeypatch, {f"/workspace/run/{b.name}": b}, [
        {**progress_of("phone", "x", "stopped", tasks=[MMLU], why="the server stopped"),
         "dir": "/workspace/run", "label": ""}], steps=[])
    main_of(ff, key, tmp_path / "b" / "bundles", "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "NOT safe to destroy — run is stopped: the server stopped" in out, out
    # A4 whole, and a step started by hand still asking
    c = a_bundle(there / "frontier-served__phone-thinking-on-hle-shard-4-of-4.tar.gz",
                 "served/phone")
    ff, calls, key = fetch_one(tmp_path, monkeypatch, {f"/workspace/phone/A4-1/{c.name}": c}, [
        progress_of("phone", "A4-1", "whole", tasks=[HLE], bundle={"name": c.name}),
        progress_of("phone", "A4-extra", "asking", tasks=[MMLU])],
        steps=[("phone", "A4-1"), ("phone", "A4-extra")])
    main_of(ff, key, tmp_path / "b" / "bundles", "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "(A4): NOT safe to destroy — phone A4-extra is asking" in out, out  # 0adb522: safe


def test_8_g6s_own_folder_is_a_step_with_its_progress(tmp_path, monkeypatch, capsys):
    ff, calls, key = fetch_one(tmp_path, monkeypatch, {}, [
        {**progress_of("gemma", "x", "asking", tasks=["gpqa_diamond_epoch"],
                       line="GPQA Diamond 120 of 792"),
         "dir": "/workspace/gemma-cal", "label": "", "model": "served/gemma-4-26b-a4b-bf16"}],
        steps=[])
    posted = []
    monkeypatch.setattr(ff, "post_boxes", lambda steps, dest, asked=None, reached=None, every=0: (
        posted.extend(steps), "posted")[1])
    ff.main(["--key", str(key), "--dest", str(tmp_path / "b" / "bundles"), "--sha", PHONE,
             "1.1.1.1:41"])
    out = capsys.readouterr().out
    assert "NOT safe to destroy — gemma-cal is asking" in out, out     # 0adb522: no step
    assert [x["label"] for x in posted] == ["gemma-cal"]                # and on the board


# ---------------------------------------------------------------------------
# 9: a box that read safe and is gone was destroyed: done
# ---------------------------------------------------------------------------

def test_9_a_box_destroyed_after_it_read_safe_ends_every(tmp_path, monkeypatch, capsys):
    import frontier_fetch as ff
    there = tmp_path / "there"
    there.mkdir()
    b = a_bundle(there / "frontier-served__phone-thinking-on-mmlu-pro.tar.gz", "served/phone")
    whole = progress_of("phone", "A5-1", "whole", tasks=[MMLU], bundle={"name": b.name})
    asking = progress_of("phone", "A6-1", "asking", tasks=[MMLU])
    rounds = {"n": 0}

    def run(cmd, cwd=None, timeout=None, stdin=None):
        if cmd[0] == "ssh":
            host = next(x for x in cmd if "@" in x).split("@")[1]
            if host == "1.1.1.1":
                if rounds["n"] > 0:
                    return 255, "ssh: connect to host 1.1.1.1 port 41: Connection refused"
                listing = listing_of({f"/workspace/phone/A5-1/{b.name}": b}, [whole])
                return 0, json.dumps({**json.loads(listing),
                                      "steps": [{"build": "phone", "step": "A5-1"}]})
            mine = {**asking, "dir": "/workspace/phone/A6-1", "label": "A6"} if not rounds["n"] \
                else {**whole, "dir": "/workspace/phone/A6-1", "label": "A6"}
            files = {} if not rounds["n"] else {f"/workspace/phone/A6-1/{b.name}": b}
            return 0, json.dumps({**json.loads(listing_of(files, [mine])),
                                  "steps": [{"build": "phone", "step": "A6-1"}]})
        if cmd[0] == "scp":
            Path(cmd[-1]).write_bytes(b.read_bytes())
            return 0, ""
        return 0, "the row served/phone · thinking: imported"
    monkeypatch.setattr(ff, "run", run)
    monkeypatch.setattr(ff, "keep_sudo", lambda: None, raising=False)

    def sleep(s):
        rounds["n"] += 1
        if rounds["n"] > 2:
            raise AssertionError("--every never stopped")
    monkeypatch.setattr(ff.time, "sleep", sleep)
    key = tmp_path / "id"
    key.write_text("k")
    code = ff.main(["--key", str(key), "--dest", str(tmp_path / "b" / "bundles"), "--sha", PHONE,
                    "--no-board", "--every", "1s", "1.1.1.1:41", "2.2.2.2:42"])
    out = capsys.readouterr().out
    assert code == 0, out                                   # 0adb522: --every never stopped
    assert "1.1.1.1:41: read safe to destroy earlier, not reached now — destroyed: done" in out


# ---------------------------------------------------------------------------
# 10, 12: the parity verdict kept on disk; ~ in --parity
# ---------------------------------------------------------------------------

def test_10_12_the_parity_verdict_outlives_the_fetch_and_tilde_reaches_the_compare(
        tmp_path, monkeypatch, capsys):
    there = tmp_path / "there"
    there.mkdir()
    par = there / "parity.jsonl"
    par.write_text('{"parity_of": {}}\n')
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "phone-server-500.jsonl").write_text("{}\n")
    dest = tmp_path / "b" / "bundles"
    ff, calls, key = fetch_one(
        tmp_path, monkeypatch, {}, [progress_of("phone", "A3-parity", "whole", parity=True)],
        {"/workspace/phone/A3-parity/parity.jsonl": par}, steps=[("phone", "A3-parity")],
        compare=(1, "Not the same setup: the box answers 38.0% right, the server 42.4%\n"))
    assert main_of(ff, key, dest, "--parity", "served/phone=~/phone-server-500.jsonl",
                   "1.1.1.1:41") == 1
    compare = next(c for c in calls if c[:9] == ff.COMPARE)
    assert compare[9] == str(tmp_path / "home" / "phone-server-500.jsonl")    # 0adb522: "~/…"
    capsys.readouterr()
    # A3 destroyed, the fetch started again for another box alone
    b = a_bundle(there / "frontier-served__phone-thinking-on-mmlu-pro.tar.gz", "served/phone")
    ff, calls, key = fetch_one(
        tmp_path, monkeypatch, {f"/workspace/phone/A5-1/{b.name}": b},
        [progress_of("phone", "A5-1", "whole", tasks=[MMLU], bundle={"name": b.name})],
        steps=[("phone", "A5-1")])
    code = main_of(ff, key, dest, "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "phone's parity: Not the same setup" in out and code == 1, out   # 0adb522: 0


# ---------------------------------------------------------------------------
# 11: a build with no --sha: home, to import by hand — never "and imported"
# ---------------------------------------------------------------------------

def test_11_no_sha_reads_home_to_import_by_hand_and_register_only_when_new(tmp_path,
                                                                           monkeypatch, capsys):
    there = tmp_path / "there"
    there.mkdir()
    g = a_bundle(there / "frontier-served__orig-thinking-on-gpqa.tar.gz", "served/orig")
    ff, calls, key = fetch_one(
        tmp_path, monkeypatch, {f"/workspace/orig/A5-1/{g.name}": g},
        [progress_of("orig", "A5-1", "whole", tasks=["gpqa_diamond_epoch"],
                     bundle={"name": g.name})], steps=[("orig", "A5-1")])
    run = ff.run
    monkeypatch.setattr(ff, "run", lambda cmd, cwd=None, timeout=None, stdin=None: (
        0, '["served/orig", "served/phone"]') if "--served" in cmd else run(cmd, cwd, timeout,
                                                                            stdin))
    main_of(ff, key, tmp_path / "b" / "bundles", "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "and imported" not in out                         # 0adb522: "… and imported"
    assert "(0 imported, 1 to import by hand: the lines below)" in out, out
    by_hand = next(x for x in out.splitlines() if "import it by hand" in x)
    assert "--register" not in by_hand and "--file-sha256 <its sha256>" in by_hand


# ---------------------------------------------------------------------------
# 13, 14: the board's list — the boxes a fetch asked; a line that can't break it
# ---------------------------------------------------------------------------

def test_13_a_fetch_of_one_box_leaves_the_others_as_they_were(svc):  # noqa: F811
    import import_frontier as imf
    now = time.time()

    def step(label, box_id, state="asking", seen=now):
        return {"label": label, "model": "served/phone", "step": f"{label}-1", "state": state,
                "at": now - 60, "seen_at": seen, "safe": False, "reachable": True,
                "box_id": box_id, "line": ""}
    imf.store_boxes({"steps": [step("A1", "a"), step("A2", "b")], "asked": ["a", "b"]})
    imf.store_boxes({"steps": [step("A1", "a")], "asked": ["a"]})
    got = {b["label"]: b for b in imf.read_boxes(now)["boxes"]}
    assert got["A2"]["reachable"] is True                    # 0adb522: "not reached"
    # A1's box gone: not reached; a day on, off the list
    imf.store_boxes({"steps": [], "asked": ["a"]})
    got = {b["label"]: b for b in imf.read_boxes(now)["boxes"]}
    assert got["A1"]["reachable"] is False
    assert "A1" not in {b["label"] for b in imf.read_boxes(now + 25 * 3600)["boxes"]}


def test_14_a_progress_line_can_never_break_the_list(svc):  # noqa: F811
    import import_frontier as imf
    now = time.time()
    imf.store_boxes([{"label": f"A{k}", "model": "served/phone", "step": f"A{k}-1",
                      "state": "asking", "at": now, "seen_at": now, "line": line}
                     for k, line in enumerate(("v 1.2.3 h left", "pages , of , done",
                                               "MMLU-Pro 3,000 of 12,032 · 13.3 h left"))])
    got = imf.read_boxes(now)["boxes"]                       # 0adb522: ValueError
    assert [(b["n"], b["of"]) for b in got][2] == (3000, 12032)
    assert svc.get("/api/frontier/boxes").status_code == 200


# ---------------------------------------------------------------------------
# 15: the memory check reads what is free
# ---------------------------------------------------------------------------

def test_15_memory_in_use_by_something_else_is_taken_off(box, monkeypatch):  # noqa: F811
    monkeypatch.setattr(rg, "header_of", lambda src: (PILOT_SHAPE, PILOT_FILE))
    import frontier as fb
    monkeypatch.setattr(fb, "_fetch", questions)
    monkeypatch.setattr(rb, "gpu_info", lambda: {**GPU, "memory_mib": 32_607,
                                                 "memory_used_mib": 300})
    assert run_box(box, "h", "--only", HLE, "--slots", "8", "--min-slots", "7",
                   "--flags", " ".join(Q8)) == 0
    own = (box["root"] / "h" / rg.OWN_LOG).read_text()
    assert re.search(r"1,27\d MiB spare", own) and "300 MiB in use before this step" in own, \
        own                                                    # 0adb522: 1,571 spare
    two = "NVIDIA GeForce RTX 5090, 575.51, 32607, 300\n"
    monkeypatch.undo()
    monkeypatch.setattr(rb.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, stdout=two))
    assert rb.gpu_info()["memory_used_mib"] == 300


# ---------------------------------------------------------------------------
# 16: small ones
# ---------------------------------------------------------------------------

def test_16_a_refusal_never_reruns_on_a_stale_incomplete(tmp_path):
    out = tmp_path / "o"
    rg.progress(out, state="stopped", incomplete={MMLU: {"answers": 3, "of": 9}})
    with pytest.raises(SystemExit):
        rg.main(["--as", "served/phone", "--gguf", "hf://me/private/m.gguf", "--server",
                 "hf://me/private/s.tar.gz", "--thinking", "on", "--only", "arc_agi2_public",
                 "--out", str(out)])
    got = json.loads((out / rg.PROGRESS).read_text())
    assert got["state"] == "stopped" and not got.get("incomplete"), got  # 0adb522: kept


def test_16_a_killed_step_says_how_long_it_hasnt_written(tmp_path, monkeypatch, capsys):
    ff, calls, key = fetch_one(tmp_path, monkeypatch, {}, [
        {**progress_of("phone", "A5-1", "asking", tasks=[MMLU]), "at": time.time() - 47 * 60}],
        steps=[("phone", "A5-1")])
    main_of(ff, key, tmp_path / "b" / "bundles", "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "phone A5-1 is asking, but hasn't written for 47 min — stopped? paste its line " \
           "again" in out, out                                   # 0adb522: "is asking"


def test_16_the_docs_and_the_help_say_todays_numbers():
    doc = (REPO / "docs" / "REMOTE-RUNS.md").read_text()
    assert "8 fit, 1,775 MiB spare" in doc and "about 29,412 MiB" in doc
    assert "| about 297 | $131–187 |" in doc
    fetch = subprocess.run([sys.executable, str(REPO / "scripts" / "frontier_fetch.py"),
                            "--help"], capture_output=True, text=True).stdout
    assert not re.search(r"phone-server\.jsonl", fetch) and "phone-server-500.jsonl" in fetch
    box_help = subprocess.run([sys.executable, str(REPO / "scripts" / "frontier_box.py"),
                               "--help"], capture_output=True, text=True).stdout
    assert "8 on a 5090" in " ".join(box_help.split())


# ---------------------------------------------------------------------------
# part 3, 17: the same answers with another finish or token count
# ---------------------------------------------------------------------------

def test_17_the_same_answers_with_new_flags_are_taken_and_keep_their_grades(box):  # noqa: F811
    from service import config
    from service import frontier as sf
    from test_17_gguf_box import N, ROW, RUNS, TASK, bundle_of
    from test_17b_review import answers_name, imported, rewrite
    from test_17g_review import a3_imported, graded_all
    a3_imported(box)
    d = sf.task_dir(config.OUT_DIR / ROW, TASK)
    g = graded_all(d)

    def ran_out(files):
        lines = [json.loads(x) for x in files[answers_name()].decode().splitlines()]
        lines[0].update(finish="length", tokens=81920)
        files[answers_name()] = "".join(json.dumps(x) + "\n" for x in lines).encode()
    code, said = imported(rewrite(bundle_of(box, "a3"), ran_out))
    assert code == 0, said
    assert any(f"{TASK}: the same {N * RUNS} answers as the row, 1 with another finish or "
               "token count — taken from this bundle, every grade kept" in x for x in said), said
    now = sf.read_answers(d / sf.ANSWERS)
    assert sorted(r.get("finish") for r in now.values()).count("length") == 1  # 0adb522: none
    assert sf.read_grades(d)["items"] == g["items"]


# ---------------------------------------------------------------------------
# part 4: the export — by a list of what may go out
# ---------------------------------------------------------------------------

def _export_world(box, monkeypatch, questions=None):  # noqa: F811
    """a box's run imported onto the row (the board's own served build)"""
    import frontier as fb
    from test_17g_review import LONG, a3_imported
    items = questions or [dict(x) for x in LONG]
    monkeypatch.setattr(fb, "_fetch", lambda task: [dict(x) for x in items])
    return a3_imported(box)


def test_18_no_key_survives_in_the_exported_setup(box, tmp_path, monkeypatch):  # noqa: F811
    import export_frontier_raw as efr
    from service import config
    from service import frontier as sf
    from test_17_gguf_box import ROW, TASK
    sid = _export_world(box, monkeypatch)
    f = sf.task_dir(config.OUT_DIR / ROW, TASK) / sf.SETUP
    st = json.loads(f.read_text())
    st["launch"] = {"flags": "--api-key 'quiet words' -ctk q8_0 --flash-attn on",
                    "env": "LLAMA_API_KEY=plainword LLAMA_MOE_ROUTE_MODE=lookahead"}
    st["launch_setup"] = {**(st.get("launch_setup") or {}),
                          "env": {"LLAMA_MOE_ROUTE_MODE": "lookahead", "API_KEY": "nodigits"},
                          "answers": ["--api-key", "elementkey", "-ctv", "q8_0"]}
    st["server_flags"] = "Authorization: Bearer nodigitsatall"
    f.write_text(json.dumps(st))
    out = (efr.export_run(sid, tmp_path / "raw") / "setup.json").read_text()
    for secret in ("quiet words", "plainword", "nodigits", "elementkey", "nodigitsatall",
                   "api-key", "API_KEY"):
        assert secret not in out, secret                    # 0adb522: through, whole
    got = json.loads(out)["tasks"][TASK]
    assert got["launch"] == {"flags": ["-ctk", "q8_0", "--flash-attn", "on"],
                             "env": {"LLAMA_MOE_ROUTE_MODE": "lookahead"}}
    assert got["launch_setup"]["answers"] == ["-ctv", "q8_0"]


def test_19_publishing_waits_for_a_typed_yes_to_the_list(box, tmp_path, monkeypatch, capsys):  # noqa: F811
    import builtins

    import export_devicemark_raw as dmx
    import export_frontier_raw as efr
    from service import db
    from test_17_gguf_box import SERVED
    sid = _export_world(box, monkeypatch)
    # 17i: public/ only with the scrub's hosts and accounts given
    monkeypatch.setenv("SCRUB_HOSTS", "board-host")
    monkeypatch.setenv("SCRUB_ACCOUNTS", "teamacct")
    db.public_set(SERVED, True, "masein")
    asked = []
    monkeypatch.setattr(builtins, "input", lambda prompt="": (asked.append(prompt), "no")[1])
    assert efr.main(["--run", str(sid), "--out", str(tmp_path / "a")]) == 0
    out = capsys.readouterr().out
    assert asked and f"  {SERVED}" in out and "not published" in out
    assert not (tmp_path / "a" / "public").exists() and (tmp_path / "a" / "private").exists()
    monkeypatch.setattr(builtins, "input", lambda prompt="": "yes")
    assert efr.main(["--run", str(sid), "--out", str(tmp_path / "b")]) == 0
    assert (tmp_path / "b" / "public").exists()             # 0adb522: never, by the mark
    # 17i: the mark decides, never the name — unsloth's Qwen3.6 files are public
    assert efr.known_public({"hf_id": SERVED}) and dmx is not None


def test_20_a_quote_of_a_gated_question_never_reaches_the_log(box, tmp_path, monkeypatch):  # noqa: F811
    import export_frontier_raw as efr
    import frontier as fb
    from service import config
    from test_17_gguf_box import SERVED, invented
    q = ("First line of the problem about boiling water at altitude.\nSecond line asks for "
         "the height in metres of the summit where it boils.")
    items = [{**x, "question": q if k == 0 else x["question"]} for k, x in enumerate(invented())]
    sid = _export_world(box, monkeypatch, items)
    log = config.LOGS_DIR / f"service_{sid}_{SERVED.replace('/', '__')}.log"
    log.write_text(log.read_text()
                   + "[frontier] later line: Second line asks for the height in metres\n"
                   + "[frontier] escaped: " + json.dumps(q)[1:60] + "\n"
                   + "[frontier] middle: the problem about boiling water at altitude Second\n"
                   + "[frontier] GPQA Diamond: 48 of 48 answered\n")
    dest = efr.export_run(sid, tmp_path / "raw")
    text = (dest / "log.txt").read_text()
    for bit in ("Second line asks", "boiling water", "First line of the"):
        assert bit not in text, bit                          # 0adb522: through
    assert "GPQA Diamond: 48 of 48 answered" in text
    # fail closed: no list of the questions, no log
    real = fb.load
    monkeypatch.setattr(fb, "load", lambda task, root=None: (_ for _ in ()).throw(
        RuntimeError("gated")))
    assert efr.es.private_questions() is None               # 17i: every gated set, always
    monkeypatch.setattr(fb, "load", real)
    monkeypatch.setattr(efr.es, "private_questions", lambda say=None: None)
    dest = efr.export_run(sid, tmp_path / "raw2")
    assert (dest / "log.txt").read_text() == ""
    assert "no question list" in (dest / "README.md").read_text()


def test_21_addresses_ports_hosts_and_the_account_are_scrubbed():
    import export_devicemark_raw as dmx
    for raw, gone in (("fd00:1234::5 and 2001:db8:85a3::8a2e:370:7334", ("fd00", "2001:db8")),
                      ("[address]:41234", ("41234",)), ("ssh -o Port=2222 x", ("2222",)),
                      ("-p 2222 user@host.example.com", ("2222", "user@", "example.com")),
                      ("http://llm.corp.example.com/v1", ("corp.example.com",)),
                      ("https://u:pw@internal.example.com/x", ("u:pw", "internal")),
                      ("-hf acct/evalboard-q-gguf", ("acct/",)),
                      ("datasets/acct/evalboard-raw-private", ("acct/",)),
                      ("hf://acct/my-evalboard-builds/x", ("acct/",))):
        got = dmx.scrub(raw, env={}, hosts=[])
        assert not any(g in got for g in gone), (raw, got)   # 0adb522: through
    assert dmx.scrub("2026-10-06 17:53:58 · v 1.2.3", env={}, hosts=[]) == \
        "2026-10-06 17:53:58 · v 1.2.3"


def test_22_two_runs_on_this_server_export_what_each_asked(box, tmp_path, monkeypatch):  # noqa: F811
    import export_frontier_raw as efr
    from service import config, db, served
    from service import frontier as sf
    from test_17_gguf_box import N, ROW, RUNS, SERVED, TASK, register
    register(box["sha"])
    rec = served.get(SERVED)
    d = sf.task_dir(config.OUT_DIR / ROW, TASK)
    d.mkdir(parents=True, exist_ok=True)
    sids = [db.add(SERVED, "instruct", "frontier", "masein", "a board run", thinking=True,
                   tasks=[TASK], status="done") for _ in range(2)]
    (d / sf.ANSWERS).write_text("".join(json.dumps(
        {"id": f"rec{k:03d}", "epoch": e, "answer": "ANSWER: A", "finish": "stop",
         "tokens": 5, "run": sids[k % 2]}) + "\n" for k in range(N) for e in range(RUNS)))
    (d / sf.SETUP).write_text(json.dumps({"thinking": "on", "where": "this server"}))
    sf.score_task(config.OUT_DIR / ROW, TASK, rec)
    got = [{(x["id"], x["epoch"]) for x in map(json.loads, (efr.export_run(
        s, tmp_path / "raw") / "items.jsonl").read_text().splitlines())} for s in sids]
    assert got[0] and got[1] and not got[0] & got[1]         # 0adb522: the same items
    assert len(got[0] | got[1]) == N * RUNS


# ---------------------------------------------------------------------------
# part 5, 23: the start-up fill — from each row's record, guarded
# ---------------------------------------------------------------------------

def test_23_the_start_up_fill_reads_the_records_and_never_stops_the_board(svc, monkeypatch):  # noqa: F811
    import import_frontier as imf
    from service import config, db
    typed = db.add("served/x", "instruct", "frontier", "masein",
                   "imported from a rented GPU (my own words) · nothing imported", status="done")
    broken = db.add("served/x", "instruct", "frontier", "masein",
                    "imported from a rented GPU\n(Tesla V100 (16 GB)) · z.tar.gz", status="done")
    (config.OUT_DIR / "served__x").mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "served__x" / imf.REGISTRY).write_text(json.dumps({"imports": [
        {"sid": broken, "gpu": "Tesla V100 (16 GB)", "box": "B3"}]}))
    db.init()
    assert not db.get(typed)["where_ran"]                     # 0adb522: "rented GPU · my own…"
    assert db.get(broken)["where_ran"] == "rented GPU · Tesla V100 (16 GB) · box B3"
    # an error inside it: said, and the board starts
    other = db.add("served/x", "instruct", "frontier", "masein", "n", status="done")
    (config.OUT_DIR / "served__x" / imf.REGISTRY).write_text(json.dumps({"imports": [
        {"sid": other, "gpu": "x"}]}))
    monkeypatch.setattr(imf, "where_words", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    db.init()                                                 # 0adb522: raised
