"""17c, the second review round (docs/prompts/phase-17c-second-review-fixes.md),
re-run on 0a81fc2. Part 1, before the pilot: the parity check counts a
question only when both sides read a letter, refuses the same file twice and
a question twice, and refuses two sides that aren't the same setup; the
parity run and the full run fetch the GGUF once. Each test fails on 0a81fc2.
The box is tests/fixtures/fake_llama_server.py; the server's served model
tests/fake_openai.py. Questions invented; nothing is fetched and no model
runs."""

from __future__ import annotations

import json
import re
import shutil
import sys
import types
from pathlib import Path

import pytest

import frontier as fb
import frontier_parity as fp
import remote_bundle as rb
import remote_gguf as rg
from fake_openai import FakeServer
from service import config, db, served
from service import frontier as sf
from test_12q_devicemark_runs import ME, svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import (GGUF_NAME, N, ROW, RUNS, SERVED, TASK, box,  # noqa: F401
                              bundle_of, invented, register, run_box)
from test_17b_review import imported, results_of, rewrite

PARITY_ITEMS = [{"id": str(k), "question": f"Q{k}?", "options": ["w", "x", "y"],
                 "answer": "A", "category": "law"} for k in range(60)]


def lines(answer, n: int = 50, key: str = "A") -> list[dict]:
    return [{"id": str(k), "key": key, "answer": answer(k) if callable(answer) else answer}
            for k in range(n)]


def write(path: Path, head: dict, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(x) + "\n" for x in [{"parity_of": head}, *rows]))
    return path


# ---------------------------------------------------------------------------
# 1: a question counts only when both sides read a letter
# ---------------------------------------------------------------------------

def test_1_two_sides_that_read_no_letter_are_not_the_same():
    """50 empty replies on both sides printed "The same: 50 of 50" """
    got = fb.parity_compare(lines(""), lines(""))
    assert got["ok"] is False and got["both"] == 0 and got["same"] == 0
    assert got["words"].startswith("Not the same: a letter was read from 0 of the server's "
                                   "answers and 0 of the box's: each side needs 46 of 50")
    # one side unreadable on five: the 45 read on both don't make 46
    five_unread = lines(lambda k: "" if k < 5 else "the answer is (A)")
    got = fb.parity_compare(lines("the answer is (A)"), five_unread)
    assert got["ok"] is False and (got["both"], got["same"], got["read_box"]) == (45, 45, 45)
    got = fb.parity_compare(lines("the answer is (A)"), lines("The answer is (A)."))
    assert got["ok"] is True and got["same"] == 50 and got["identical"] == 0


def test_1_the_same_file_twice_or_a_question_twice_is_refused(tmp_path):
    head = {"side": "server", "as": SERVED, "file": {"name": "m.gguf"}, "launch": {}}
    a = write(tmp_path / "server.jsonl", head, lines("the answer is (A)"))
    with pytest.raises(SystemExit, match="the same file was given twice"):
        fp.main(["compare", str(a), str(a)])
    copy = tmp_path / "copy.jsonl"
    shutil.copy(a, copy)
    with pytest.raises(SystemExit, match="the same file was given twice"):
        fp.main(["compare", str(a), str(copy)])
    # a question twice collapsed into one
    twice = lines("the answer is (A)")
    twice[1] = {**twice[0]}
    got = fb.parity_compare(lines("the answer is (A)"), twice)
    assert got["ok"] is False and "the box's file holds question 0 more than once" in got["words"]


# ---------------------------------------------------------------------------
# 2, 3: what answered, compared; the GGUF fetched once
# ---------------------------------------------------------------------------

def test_2_each_side_says_what_answered_and_compare_refuses_another_setup(  # noqa: F811
        box, svc, monkeypatch, capsys):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", lambda task: {"items": PARITY_ITEMS,
                                                    "extra": {"shots": {}}})
    # the box: lookahead routing, as run_box launches it
    assert run_box(box, "parity", "--parity") == 0
    bpath = box["root"] / "parity" / "parity.jsonl"
    bh, brows = fp.read(bpath)
    assert bh["side"] == "box" and bh["as"] == SERVED and len(brows) == 50
    assert {k: bh["file"][k] for k in ("name", "size", "sha256")} == {
        "name": GGUF_NAME, "size": box["gguf"].stat().st_size, "sha256": box["sha"]}
    assert bh["server"]["env"] == {"LLAMA_MOE_ROUTE_MODE": "lookahead"}
    # the server: the same file, registered with the same routing
    fake = FakeServer()
    try:
        fake.model_path = f"/models/{GGUF_NAME}"
        # 17d: llama-server's count of the same file's weights, as the box's says
        fake.size = 21_000_000_000
        fake.reply = lambda body: "the answer is (A)"
        rec = served.register({"name": "lda box", "base_url": fake.base, "how": "llama-server",
                               "thinking": "off"}, ME)
        assert rec["id"] == SERVED
        db.served_put({**rec, "env": "LLAMA_MOE_ROUTE_MODE=lookahead"})
        spath = box["root"] / "server.jsonl"
        assert fp.main(["ask", "--as", SERVED, "--out", str(spath)]) == 0
        sh, _ = fp.read(spath)
        assert sh["side"] == "server" and sh["file"]["name"] == GGUF_NAME
        assert sh["launch"]["env"] == {"LLAMA_MOE_ROUTE_MODE": "lookahead"}
        capsys.readouterr()
        assert fp.main(["compare", str(spath), str(bpath), "--file-sha256", box["sha"]]) == 0
        assert capsys.readouterr().out.startswith("The same: 50 of the 50 read on both sides")
        # without the file's sha256 the board has none: compared by name and size, said
        assert fp.main(["compare", str(spath), str(bpath)]) == 0
        assert "compared by name" in capsys.readouterr().out
        # another file's sha256: refused
        assert fp.main(["compare", str(spath), str(bpath), "--file-sha256", "0" * 64]) == 1
        assert "the box's file has sha256" in capsys.readouterr().out
        # the server registered with no routing: the box isn't the registered setup
        db.served_put({**rec, "env": ""})
        assert fp.main(["ask", "--as", SERVED, "--out", str(spath)]) == 0
        capsys.readouterr()
        assert fp.main(["compare", str(spath), str(bpath), "--file-sha256", box["sha"]]) == 1
        out = capsys.readouterr().out
        assert out.startswith("Not the same setup: routing: the box ran with "
                              "LLAMA_MOE_ROUTE_MODE=lookahead; served/lda-box is registered "
                              "with none")
        # the server serving another file than the one registered: never asked
        fake.model_path = "/models/another.gguf"
        n = fake.answered
        with pytest.raises(SystemExit, match="serves a different file"):
            fp.main(["ask", "--as", SERVED, "--out", str(spath)])
        assert fake.answered == n
    finally:
        fake.close()
    # the two files given the wrong way round
    assert fp.main(["compare", str(bpath), str(spath)]) == 1
    assert "give the server's, then the box's" in capsys.readouterr().out


def test_3_the_parity_run_and_the_full_run_fetch_the_gguf_once(box, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", lambda task: {"items": PARITY_ITEMS,
                                                    "extra": {"shots": {}}}
                        if task == fb.PARITY["task"] else __import__("test_17_gguf_box")
                        .invented())
    calls = []
    sources = {GGUF_NAME: box["gguf"], "llama-server.tar.gz": box["tarball"]}

    def download(repo, path, repo_type=None, revision=None, token=None, local_dir=None):
        calls.append((path, local_dir))
        dest = Path(local_dir) / path
        if not dest.exists():                  # as Hugging Face's: a file already there stays
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(sources[path], dest)
        return str(dest)
    # Hugging Face's client, stood in for (the check image has none)
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(hf_hub_download=download))
    hf = ["--gguf", f"hf://me/private/{GGUF_NAME}", "--server", "hf://me/private/llama-server.tar.gz"]
    assert run_box(box, "parity", "--parity", *hf) == 0
    assert run_box(box, "run", *hf) == 0
    assert {d for _, d in calls} == {str(box["root"] / "files")}
    # …and hashed once: the full run's log says no sha256 was read
    import remote_gguf as rg
    assert "sha256 of" in (box["root"] / "parity" / rg.OWN_LOG).read_text()
    assert "sha256 of" not in (box["root"] / "run" / rg.OWN_LOG).read_text()


# ---------------------------------------------------------------------------
# part 2: before the full run
# ---------------------------------------------------------------------------

def a_served(fake: FakeServer, **more) -> dict:
    fake.ctx = 40960
    return served.register({"name": "board box", "base_url": fake.base, "how": "x",
                            "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "off", **more}, ME)


def test_4_a_question_that_always_fails_is_no_answer_and_the_run_carries_on(svc, monkeypatch):  # noqa: F811
    """a prompt too long for its slot answers 400 every time: the task was never
    scored, the benchmarks after it never asked, and the question never named"""
    monkeypatch.setattr(fb, "_fetch", lambda task: invented())
    fake = FakeServer()
    try:
        fake.reply = lambda body: "ANSWER: A"
        too_long = (lambda body: (400, "the request exceeds the available context size")
                    if re.search(r"Invented question 3\b", json.dumps(body)) else None)
        fake.chat_error = fake.raw_error = too_long
        rec = a_served(fake)
        row = config.OUT_DIR / rec["id"].replace("/", "__")
        logged: list[str] = []
        assert sf.ask_task(rec, TASK, row, False, log=logged.append) == (N * RUNS, N * RUNS)
        lines = [json.loads(x) for x in (sf.task_dir(row, TASK) / sf.ANSWERS).read_text()
                 .splitlines()]
        never = [x for x in lines if x.get("unanswered")]
        assert [(x["id"], x["answer"]) for x in never] == [("rec003", "")] * RUNS
        assert "exceeds the available context size" in never[0]["unanswered"]
        assert any("question rec003, run 0: the server failed on it twice — written as no "
                   "answer, counted wrong" in x for x in logged), logged
        # scored, counted wrong, and named on the row
        sc = sf.score_task(row, TASK, rec)
        assert sc["unanswered"] == RUNS and sc["unanswered_ids"] == ["rec003"]
        assert "4 the server never answered, counted wrong (rec003)" in sf.words(TASK, sc)
        # a server that fails every question it is asked stops, keeping none
        fake.chat_error = fake.raw_error = lambda body: (500, "it failed")
        row2 = config.OUT_DIR / "served__another"
        with pytest.raises(served.ServerStopped, match="the server failed on"):
            sf.ask_task({**rec, "id": "served/another"}, TASK, row2, False)
        assert not sf.read_answers(sf.task_dir(row2, TASK) / sf.ANSWERS)
    finally:
        fake.close()


def test_5_6_thinking_as_scoring_reads_it_and_a_few_in_an_off_row_are_counted():
    closing_only = "weighing the options\n</think>\n\nANSWER: A"
    assert sf.thought(closing_only) and fb.visible(closing_only) == "ANSWER: A"
    assert not sf.thought("<think>\n\n</think>\n\nANSWER: A")
    assert sf.thinking_refused(TASK, "on", [closing_only] * 10) == ""
    # one in 12,032 with thinking: scored, and said; more than 1%: refused
    plain = ["ANSWER: A"] * 199
    assert sf.thinking_refused(TASK, "off", [closing_only, *plain]) == ""
    assert sf.thinking_kept("off", [closing_only, *plain]) == 1
    assert "more than 1%" in sf.thinking_refused(TASK, "off", [closing_only] * 3 + plain)
    assert "1 thought though thinking was off (scored on what follows the thinking)" in \
        sf.words(TASK, {"score": 0.5, "se": 0.1, "epochs": 1, "questions": 200,
                        "thinking_held": 1})


def test_7_a_box_names_its_benchmarks_in_its_bundle(box):  # noqa: F811
    """G5's boxes 6, 7 and 8 all wrote frontier-served__<build>-thinking-on.tar.gz"""
    assert run_box(box, "run") == 0
    assert bundle_of(box, "run").name == "frontier-served__lda-box-thinking-on-gpqa.tar.gz"
    assert bundle_of(box, "run").exists()
    names = {rb.bundle_name("frontier", SERVED, True, parts=p)
             for p in (["gpqa", "otis"], ["math-l5", "simpleqa"], ["arc-agi-2"])}
    assert len(names) == 3


def test_8_a_context_the_card_cant_hold_is_refused_before_the_download(tmp_path):
    with pytest.raises(SystemExit, match="8 slots of 98,304 tokens \\(ARC-AGI-2's, thinking "
                                         "on\\) is 786,432 tokens of context"):
        rg.main(["--as", SERVED, "--gguf", "hf://me/private/m.gguf", "--server",
                 "hf://me/private/s.tar.gz", "--thinking", "on", "--only", "arc_agi2_public",
                 "--out", str(tmp_path / "o")])
    assert not (tmp_path / "files").exists() and not (tmp_path / "o").exists()


def test_9_a_variable_in_the_shell_never_reaches_llama_server(box, monkeypatch, tmp_path):  # noqa: F811
    seen = tmp_path / "argv.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_ARGV", str(seen))
    monkeypatch.setenv("LLAMA_MOE_ROUTE_LOOKAHEAD", "4")
    monkeypatch.setenv("GGML_CUDA_FORCE_MMQ", "1")
    assert run_box(box, "run") == 0
    got = json.loads(seen.read_text().splitlines()[-1])["env"]
    assert got == {"LLAMA_MOE_ROUTE_MODE": "lookahead"}       # --env's, and only it
    srv = rb.read(bundle_of(box, "run"))["setup"]["server"]
    assert srv["env"] == got
    own = (box["root"] / "run" / rg.OWN_LOG).read_text()
    assert "not given to llama-server: GGML_CUDA_FORCE_MMQ, LLAMA_MOE_ROUTE_LOOKAHEAD" in own


def test_10_a_launch_that_answered_nothing_pins_nothing(box, monkeypatch):  # noqa: F811
    """a typo in --flags: the corrected command was refused"""
    monkeypatch.setenv("FAKE_LLAMA_MODEL_PATH", "/models/another.gguf")
    with pytest.raises(SystemExit):
        run_box(box, "run", "--flags", "-ctk q8_0 --typo")
    monkeypatch.delenv("FAKE_LLAMA_MODEL_PATH")
    assert run_box(box, "run", "--flags", "-ctk q8_0") == 0
    # once it holds answers, another setup is refused as before
    with pytest.raises(SystemExit, match="another setup"):
        run_box(box, "run", "--flags", "-ctk f16")


def test_11_12_otis_reads_the_last_answer_only_and_never_hangs():
    """"ANSWER: 42 … ANSWER: 43 (mod 1000)" read 42, right against a key of 42;
    an integer, 200 spaces and text took over 100 s; 4,301 digits raised"""
    import time
    r = fb.read_integer
    assert r("ANSWER: 42\nwait, that's off\nANSWER: 43 (mod 1000)") is None
    assert r("ANSWER: 42\nANSWER: 43") == "43"
    sc = fb.score("otis_aime_epoch", "ANSWER: 42\nANSWER: 43 (mod 1000)", "stop",
                  {"answer": "42"})
    assert sc["read"] is None and sc["ok"] is False            # the model check reads it
    for slow in ("ANSWER: 5" + " " * 200 + "x" * 50, "ANSWER: " + "9" * 4301,
                 ("ANSWER: 5" + " \t" * 5000 + "x\n") * 50):
        t0 = time.time()
        assert r(slow) is None
        assert time.time() - t0 < 1.0
    assert r("**ANSWER: $\\boxed{7}$**") == "7" and r("ANSWER: 1,024.") == "1024"


def test_13_the_boards_resume_compares_the_launch_as_the_import_does(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", lambda task: invented())
    fake = FakeServer()
    try:
        fake.reply = lambda body: "ANSWER: A"
        rec = a_served(fake)
        db.served_put({**rec, "flags": "--flash-attn on -np 4 -c 163840"})
        rec = served.get(rec["id"])
        row = config.OUT_DIR / rec["id"].replace("/", "__")
        logged: list[str] = []
        assert sf.ask_task(rec, TASK, row, False, log=logged.append)[0] == N * RUNS
        n = len(fake.requests)
        # reordered, more slots, another context: the same launch
        db.served_put({**rec, "flags": "-np 8 -c 327680 --flash-attn on"})
        assert sf.ask_task(served.get(rec["id"]), TASK, row, False, log=logged.append)[0] \
            == N * RUNS
        assert len(fake.requests) == n and not any("another setup" in x for x in logged)
        # a setup.json from before 17b: what it doesn't say is unknown — kept, said
        p = sf.task_dir(row, TASK) / sf.SETUP
        old = json.loads(p.read_text())
        p.write_text(json.dumps({k: v for k, v in old.items()
                                 if k not in ("server", "launch", "launch_setup")}))
        assert sf.ask_task(served.get(rec["id"]), TASK, row, False, log=logged.append)[0] \
            == N * RUNS
        assert len(fake.requests) == n
        assert any("doesn't say their launch_setup, server: they are kept, with those unknown"
                   in x for x in logged), logged
        assert json.loads(p.read_text())["unknown_earlier"] == ["launch_setup", "server"]
        # lookahead now registered: another setup, set aside and asked again
        db.served_put({**served.get(rec["id"]), "env": "LLAMA_MOE_ROUTE_MODE=lookahead"})
        sf.ask_task(served.get(rec["id"]), TASK, row, False, log=logged.append)
        assert len(fake.requests) == n + N * RUNS
    finally:
        fake.close()


def test_14_15_a_copy_on_disk_is_counted_and_a_count_isnt_a_gated_sets_terms(monkeypatch,
                                                                            tmp_path):
    monkeypatch.setattr(fb, "expected", fb._expected_n)          # the real count (conftest)
    p = fb._cache(tmp_path, "mmlupro_tiger")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"items": [{"id": str(k), "category": "law"} for k in range(11000)],
                             "extra": {}}))
    with pytest.raises(ValueError, match="11,000 questions are in this machine's copy") as e:
        fb.load("mmlupro_tiger", tmp_path)
    assert "12,032" in str(e.value)
    hle = fb._cache(tmp_path, "hle_text_cais")
    hle.write_text(json.dumps([{"id": str(k)} for k in range(2500)]))
    with pytest.raises(ValueError) as e:
        fb.load("hle_text_cais", tmp_path)
    said = fb.load_failed("hle_text_cais", e.value)
    assert said.startswith("Humanity's Last Exam: 2,500 questions are in this machine's copy")
    assert "accept its terms" not in said
    refused = fb.load_failed("hle_text_cais", OSError("401 Client Error: Unauthorized: gated"))
    assert refused.endswith("accept its terms on Hugging Face with the account whose token "
                            "this machine has")


# ---------------------------------------------------------------------------
# part 3: before the imports
# ---------------------------------------------------------------------------

OTIS = [{"id": f"o{k}", "question": f"Problem {k}.", "answer": "42"} for k in range(3)]


def two_tasks(monkeypatch):
    monkeypatch.setattr(fb, "_fetch", lambda task: {TASK: invented(),
                                                    "otis_aime_epoch": OTIS}[task])


def test_16_a_bundle_with_a_task_that_cant_be_scored_imports_nothing(box, monkeypatch):  # noqa: F811
    """tasks swapped in one at a time, and the bundle recorded as imported: the
    second import said "imported already" and exited 0"""
    two_tasks(monkeypatch)
    assert run_box(box, "run", "--only", "otis_aime_epoch") == 0
    register(box["sha"])
    path = box["root"] / "run" / rb.bundle_name("frontier", SERVED, True,
                                                parts=["gpqa", "otis"])
    real = sf.score_task

    def otis_breaks(row, task, rec):
        if task == "otis_aime_epoch":
            raise RuntimeError("the scorer broke")
        return real(row, task, rec)
    monkeypatch.setattr(sf, "score_task", otis_breaks)
    code, said = imported(path)
    assert code == 1 and any("nothing was imported; the row is as it was" in x for x in said)
    assert results_of() is None                       # GPQA, which scored, isn't in either
    assert not sf.task_dir(config.OUT_DIR / ROW, TASK).exists()
    # not recorded as imported: once it can be scored, the same command imports it
    monkeypatch.setattr(sf, "score_task", real)
    code, said = imported(path)
    assert code == 0 and not any("imported already" in x for x in said), said
    assert results_of() is not None and results_of(task="otis_aime_epoch") is not None


def test_17_a_rerun_with_a_rebuilt_tarball_sets_the_old_shards_aside(box):  # noqa: F811
    assert run_box(box, "s1", "--shard", "1/2") == 0
    assert run_box(box, "s2", "--shard", "2/2", "--flags", "-ctk q8_0") == 0
    register(box["sha"])
    assert imported(bundle_of(box, "s1", (1, 2)))[0] == 0
    code, said = imported(bundle_of(box, "s2", (2, 2)))
    assert code == 2 and any("import with --set-aside-shards" in x for x in said), said
    # the flag: shard 1 set aside, shard 2 starts the task's shards again
    code, said = imported(bundle_of(box, "s2", (2, 2)), aside=True)
    assert code == 0, said
    assert any("the shards waiting here were set aside" in x for x in said), said
    assert any("shard 1 of 2 missing" in x for x in said), said
    assert list(config.OUT_DIR.with_name("earlier").glob(f"{ROW}/{TASK}-shards*"))
    # shard 1 again, with the second's setup: the task is whole
    assert run_box(box, "s1b", "--shard", "1/2", "--flags", "-ctk q8_0") == 0
    assert imported(bundle_of(box, "s1b", (1, 2)))[0] == 0
    assert results_of() is not None


def test_18_a_setup_json_of_the_wrong_types_is_refused_before_anything(box):  # noqa: F811
    """a list as gpu.name swapped the row in, then failed with a traceback and
    no Runs entry"""
    assert run_box(box, "run") == 0
    register(box["sha"])

    def odd(files):
        setup = json.loads(files["setup.json"])
        setup["gpu"]["name"] = ["RTX 5090", "twice"]
        setup["server"]["slots"] = "eight"
        files["setup.json"] = json.dumps(setup).encode()
    code, said = imported(rewrite(bundle_of(box, "run"), odd))
    assert code == 2, said
    assert any("setup.json's server.slots is str, not int" in x for x in said), said
    assert any("setup.json's gpu.name is list, not str" in x for x in said), said
    assert results_of() is None and not sf.task_dir(config.OUT_DIR / ROW, TASK).exists()


def test_19_a_long_name_header_is_never_read_whole(tmp_path, monkeypatch):
    """a 1.5 MB bundle held 1.57 GB in memory: tarfile reads a GNU long name
    whole while it lists the members"""
    import io
    import tarfile
    path = tmp_path / "b.tar.gz"
    with tarfile.open(path, "w:gz", format=tarfile.GNU_FORMAT) as tar:
        for name, data in (("bundle.json", b"{}"), ("setup.json", b"{}"),
                           ("results/" + "x" * 5000 + "/answers.jsonl", b"")):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    monkeypatch.setattr(rb, "MAX_MEMBER", 1024)
    with pytest.raises(ValueError, match="holds a part larger than a bundle's file can be"):
        rb.read(path)
