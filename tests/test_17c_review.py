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
import shutil
import sys
import types
from pathlib import Path

import pytest

import frontier as fb
import frontier_parity as fp
from fake_openai import FakeServer
from service import db, served
from test_12q_devicemark_runs import ME, svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import GGUF_NAME, SERVED, box, run_box  # noqa: F401 — box is a fixture

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
    assert bh["file"] == {"name": GGUF_NAME, "size": box["gguf"].stat().st_size,
                          "sha256": box["sha"]}
    assert bh["server"]["env"] == {"LLAMA_MOE_ROUTE_MODE": "lookahead"}
    # the server: the same file, registered with the same routing
    fake = FakeServer()
    try:
        fake.model_path = f"/models/{GGUF_NAME}"
        fake.size = box["gguf"].stat().st_size
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
        assert "compared by name and size" in capsys.readouterr().out
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
