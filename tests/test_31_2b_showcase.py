"""31.2b: the whitepaper's showcase on step2388 — its own questions asked on
U39's box, on the k6 path at the file's own settings, one slot, each once,
recorded as step8804's were, and brought home into the showcase's own folder.
The server is the stand-in (tests/fixtures/fake_llama_server.py), the questions
invented: the real file is the private repository's, pinned by its sha256."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import frontier as fb
import frontier_box as fx
import import_remote as ir
import remote_bundle as rb
import remote_gguf as rg
import showcase_box as scb
from service import config
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import box, requests  # noqa: F401
from test_20_step4000 import card  # noqa: F401
from test_31_1_k3k6 import NAME, RT, STEP, k3k6, run_k, step_kept  # noqa: F401

REV = "fe12fa27825b3a0cc6cde3dedec3e838bd3e6de4"
PINNED = "978829b74ff537be8f3650c2c73799011a5b46c189964516cac2505c827047ff"
INVENTED = [{"n": 1, "kind": "Everyday help", "question": "Invented question one?"},
            {"n": 2, "kind": "Maths word problem", "question": "Invented question two?"},
            {"n": 3, "kind": "Text to JSON", "question": "Invented question three?"}]


@pytest.fixture
def sc(k3k6, tmp_path, monkeypatch):  # noqa: F811
    """a k6 box given invented questions, the import holding them to their file"""
    q = tmp_path / "qs" / scb.NAME
    q.parent.mkdir(parents=True)
    q.write_text(json.dumps(INVENTED, indent=1) + "\n", encoding="utf-8")
    monkeypatch.setattr(scb, "pinned_sha", lambda: rb.sha256_file(q))
    return {**k3k6, "questions": q}


def asked_showcase(b: dict) -> list[dict]:
    texts = {x["question"] for x in INVENTED}
    return [r for r in requests(b) if (r.get("messages") or [{}])[-1].get("content") in texts]


def bundle_of(b: dict, out: str) -> Path:
    return b["root"] / out / rb.bundle_name(scb.SUITE, STEP, True)


def test_the_showcase_is_asked_on_k6_once_each_at_the_files_own_settings(sc):
    assert run_k(sc, "s", "--showcase", str(sc["questions"]), only=False, slots=1) == 0
    asked = asked_showcase(sc)
    assert len(asked) == 3
    # the question alone, its limit, step8804's seed, the path: nothing that samples
    assert all(sorted(r) == ["expert_path", "max_tokens", "messages", "seed", "stream"]
               for r in asked)
    assert [(r["expert_path"], r["max_tokens"], r["seed"]) for r in asked] == [
        (6, 81920, fb.seed_of("showcase", f"own-{n}", 0)) for n in (1, 2, 3)]
    assert fb.seed_of("showcase", "own-1", 0) == 1683674005           # step8804's own-1
    assert all(r["messages"] == [{"role": "user", "content": x["question"]}]
               for r, x in zip(asked, INVENTED))
    out = sc["root"] / "s"
    assert json.loads((out / "settings-check-k6.json").read_text())["pass"]
    rows = [json.loads(x) for x in (out / scb.ANSWERS).read_text().splitlines()]
    assert [r["q"] for r in rows] == ["own-1", "own-2", "own-3"]
    r = rows[0]
    assert (r["model"], r["context"], r["max_tokens"], r["expert_path"]) == (
        "the Masein model", 83968, 81920, 6)
    assert r["sampling"].startswith("the file's own for the path: temperature 0.8")
    assert {"thinking", "reply", "output_tokens", "thinking_tokens", "finish", "ran_out",
            "looped", "seconds", "prompt_tokens", "seed", "date", "file"} <= set(r)
    prog = json.loads((out / "progress.json").read_text())
    assert prog["state"] == "whole" and prog["tasks"] == ["showcase"]
    b = rb.read(bundle_of(sc, "s"))
    assert b["bundle"]["suite"] == "showcase-served" and b["bundle"]["tasks"] == {"showcase": 3}
    assert b["setup"]["showcase"]["questions_sha256"] == rb.sha256_file(sc["questions"])
    # pasted again: nothing answered is asked again
    assert run_k(sc, "s", "--showcase", str(sc["questions"]), only=False, slots=1) == 0
    assert len(asked_showcase(sc)) == 3
    # home: into the showcase's own folder, no row of the board touched
    assert ir.main([str(bundle_of(sc, "s")), "--by", "masein", "--register", "step2388 (k3/k6)",
                    "--file-sha256", sc["sha"]]) == 0
    own = config.BENCH_ROOT / "showcase" / (STEP.replace("/", "__") + "__thinking") / "own.jsonl"
    assert [json.loads(x)["q"] for x in own.read_text().splitlines()] == [
        "own-1", "own-2", "own-3"]
    assert not (config.OUT_DIR / (STEP.replace("/", "__") + "__thinking") / scb.ANSWERS).exists()
    assert ir.main([str(bundle_of(sc, "s")), "--by", "masein"]) == 0       # nothing changed


def test_the_import_holds_the_showcase_to_its_questions_and_requests(sc):
    assert run_k(sc, "s2", "--showcase", str(sc["questions"]), only=False, slots=1) == 0
    b = rb.read(bundle_of(sc, "s2"))
    import import_frontier as imf
    rec = imf._would_be(b, sc["sha"])
    assert scb.checks(b, rec, sc["sha"]) == []

    def refused(change) -> list[str]:
        c = json.loads(json.dumps({k: v for k, v in b.items() if k != "files"}))
        c["files"] = dict(b["files"])
        change(c)
        return scb.checks(c, rec, sc["sha"])
    name = f"results/{b['bundle']['row']}/{scb.ANSWERS}"

    def resend(c, **change):
        rows = [json.loads(x) for x in c["files"][name].decode().splitlines()]
        rows[0]["request"] = {**rows[0]["request"], **change}
        c["files"][name] = "".join(json.dumps(r) + "\n" for r in rows).encode()
    got = refused(lambda c: resend(c, temperature=0.0))
    assert any(x.startswith("own-1 was asked with") for x in got)
    got = refused(lambda c: resend(c, seed=7))
    assert any(x.startswith("own-1 was asked with") for x in got)
    got = refused(lambda c: c["setup"]["showcase"].update(questions_sha256="0" * 64))
    assert any("its questions aren't the pinned file" in x for x in got)

    def short(c):
        c["files"][name] = "\n".join(c["files"][name].decode().splitlines()[:2]).encode()
    assert any("2 answers to its 3 questions" in x for x in refused(short))


def test_the_real_questions_file_is_plan_us_pin():
    assert scb.pinned_sha() == PINNED == fx.PLAN_PINS["U"][scb.NAME]


def test_the_showcase_runs_on_a_path_one_slot_and_alone(tmp_path, capsys):
    base = ["--as", STEP, "--gguf", str(tmp_path / NAME), "--server", str(tmp_path / "t.tgz"),
            "--thinking", "on", "--out", str(tmp_path / "o"), "--showcase", str(tmp_path / "q")]
    for more in (["--runtime", "lda-2610", "--as", "served/step8804-lda-2610", "--slots", "1"],
                 ["--runtime", RT, "--slots", "8"],
                 ["--runtime", RT, "--slots", "1", "--only", "gpqa_diamond_epoch"]):
        with pytest.raises(SystemExit):
            rg.main([*base, *more])
    err = capsys.readouterr().err
    assert "--showcase: on a runtime of expert paths (--runtime k3k6-2611)" in err
    assert "--showcase: one slot (--slots 1)" in err
    assert "--showcase asks its questions alone" in err


def test_the_questions_file_is_read_whole_or_refused(tmp_path):
    f = tmp_path / scb.NAME
    for bad, words in (("[]", "a list of up to 50"), ('[{"n": 1}]', "{n, kind, question}"),
                       ('[{"n": 1, "question": "a"}, {"n": 1, "question": "b"}]', "given twice"),
                       ("{", "can't be read as JSON")):
        f.write_text(bad)
        with pytest.raises(ValueError, match=words):
            scb.questions(f)


def test_u39_asks_the_showcase_after_devicemark_with_its_pinned_questions():
    assert fx.PLANS["U"]["U39"] == [("on", (fx.DM,), "", 8), ("on", (fx.SHOWCASE,), "", 1)]
    a = fx.parser().parse_args([
        "U39", "--as", STEP, "--gguf", f"hf://me/p@{REV}/{NAME}",
        "--server", f"hf://me/p@{REV}/llama-server-k3k6-3ad38b7.tar.gz",
        "--showcase", "hf://me/p@abc/showcase-own12.json", "--price", "0.45"])
    dm, show = (fx.argv_of("U39", k, s, a) for k, s in enumerate(fx.PLANS["U"]["U39"], 1))
    assert "--showcase" not in dm and "--devicemark" in dm
    assert show[show.index("--showcase") + 1] == "hf://me/p@abc/showcase-own12.json"
    assert show[show.index("--slots") + 1] == "1" and show[show.index("--thinking") + 1] == "on"
    assert f"showcase-own12.json={PINNED}" in fx.pins_of(a, "U39")
    lines = fx.listing("U")
    assert "  2. thinking on · the showcase's 12 own questions (one slot, the file's own " \
           "settings) · 1 slot · about 0.1 h" in "\n".join(lines)
    assert lines[-1] == "40 boxes a build · 183.8 box-hours a build · the longest 5.2 h"


@pytest.mark.parametrize("box_, given, words", [
    ("U39", "", "its last step asks the showcase's questions: give --showcase"),
    ("U38", "hf://me/p@abc/showcase-own12.json", "--showcase is U39's alone"),
])
def test_the_showcase_file_goes_with_u39_alone(box_, given, words):
    argv = [box_, "--as", STEP, "--gguf", f"hf://me/p@{REV}/{NAME}",
            "--server", f"hf://me/p@{REV}/llama-server-k3k6-3ad38b7.tar.gz"]
    with pytest.raises(SystemExit, match=words):
        fx.main(argv + (["--showcase", given] if given else []))


def test_the_runbook_gives_u39_its_questions_file():
    text = (Path(__file__).resolve().parent.parent / "docs" / "REMOTE-RUNS.md").read_text()
    part = text[text.index("# step2388 on plan U (31.1, 31.2)"):]
    part = part[:part.index("# Speed tests")]
    assert PINNED in part
    assert "python scripts/frontier_box.py U39 --as served/step2388-k3k6-2611" in part
    assert "--showcase hf://<you>/evalboard-private@<its commit>/showcase-own12.json" in part
