"""18c part 3: grading, the fetch and the export (points 12 to 17). A score
that changes says so, and the deploy's own read-only check says which will;
every OpenRouter batch's halt waits for its press, an old halt file too, and
the AI models page has Carry on; an earlier fetch's misses don't count; the
server's words never go out after an unclosed bracket; and the small ones.
Invented answers, batches, boxes and listings; nothing is fetched, no model
runs, no paid API is called."""

from __future__ import annotations

import json
import time
from pathlib import Path


import frontier as fb
import frontier_graders as fg
from service import config
from service import frontier as sf
from test_17_grading import ROW, math_items, rec, svc, write  # noqa: F401 — svc is the fixture
from test_17e_review import a_bundle
from test_17f_review import listing_of
from test_17g_review import MMLU, progress_of, with_steps
from test_17j_boxes import main_of, world

ROOT = Path(__file__).resolve().parent.parent
MATH = "math_l5_epoch"
RIGHT = "Working it out.\nANSWER: \\frac{1}{2}"
# a thinking block opened late and never closed: the new split reads it as
# ran out (18b); the old one read the answer after it
LATE = ("Working it out at some length. " * 10 + "<think>\nlet me check again "
        + "ANSWER: \\frac{1}{2}")


def answers(late: int = -1) -> list:
    return [(it["id"], LATE if k == late else RIGHT, "stop")
            for k, it in enumerate(math_items())]


# ---------------------------------------------------------------------------
# 12. a score that changes says so; frontier-2; the deploy's own check
# ---------------------------------------------------------------------------

def test_12_frontier_2_keeps_frontier_1s_answers_and_bundles():
    assert fb.VERSION == "frontier-2" and "frontier-1" in fb.ASKED_AS   # d94f7f6: frontier-1
    old = {"version": "frontier-1", "protocol_version": "x", "epochs": 1}
    assert sf.setup_differs(old, {**old, "version": "frontier-2"}) == []      # not set aside
    assert sf.setup_differs(old, {**old, "version": "frontier-0"}) == ["version"]


def test_12_a_replaced_score_is_said(svc):  # noqa: F811
    row = config.OUT_DIR / ROW
    said: list[str] = []
    write(row, MATH, answers())
    assert sf.score_task(row, MATH, rec(), log=said.append)["score"] == 1.0
    assert said == []                                          # the first score: nothing replaced
    write(row, MATH, answers(late=1))
    got = sf.score_task(row, MATH, rec(), log=said.append)
    assert got["score"] == 0.75 and said == [
        f"[frontier] {ROW} · MATH Level 5: score 100.0% → 75.0%"]   # d94f7f6: silent
    assert got["before"] == {"score": 1.0, "version": "frontier-2"}
    # scored the frontier-1 way before: said too
    d = sf.task_dir(row, MATH)
    p = next(d.glob("results_*.json"))
    r = json.loads(p.read_text())
    r["frontier"]["version"] = "frontier-1"
    r["results"][MATH]["acc,none"] = 1.0
    p.write_text(json.dumps(r))
    said.clear()
    sf.score_task(row, MATH, rec(), log=said.append)
    assert said == [f"[frontier] {ROW} · MATH Level 5: score 100.0% → 75.0% (scored by "
                    "frontier-1 before, frontier-2 now)"]
    said.clear()
    sf.score_task(row, MATH, rec(), log=said.append)           # the same again: nothing said
    assert said == []


def test_12_the_deploys_check_counts_without_text_and_writes_nothing(svc, capsys):  # noqa: F811
    import think_shift_check as tsc
    row = config.OUT_DIR / ROW
    write(row, MATH, answers(late=2))
    sf.score_task(row, MATH, rec())
    # as 70df001 left it: the late block's answer read right, the score whole
    d = sf.task_dir(row, MATH)
    sp = next(d.glob(f"samples_{MATH}_*.jsonl"))
    rows = [json.loads(x) for x in sp.read_text().splitlines()]
    for r in rows:
        for run in r["frontier"]:
            run.update(ok=True, ran_out=False)
    sp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rp = next(d.glob("results_*.json"))
    r = json.loads(rp.read_text())
    r["results"][MATH]["acc,none"] = 1.0
    r["frontier"]["version"] = "frontier-1"
    rp.write_text(json.dumps(r))
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in row.rglob("*") if p.is_file()}
    assert tsc.main([]) == 0
    out = capsys.readouterr().out
    assert f"{ROW} · MATH Level 5: 1 of 4 answers read differently (1 now ran out, 1 right → " \
           "wrong) · score 100.0% → 75.0%" in out, out
    assert out.rstrip().endswith("1 of 1 stored Frontier scores change the next time they are "
                                 "scored (frontier-2); nothing was written")
    for text in ("frac", "Working it out", "let me check"):
        assert text not in out                                 # never an answer's text
    after = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in row.rglob("*") if p.is_file()}
    assert after == before


def test_12_the_check_runs_in_step_1_before_the_deploy():
    doc = (ROOT / "docs" / "AGENT-RUNS.md").read_text()
    a1 = doc[doc.index("## A."):][:4000]
    a1 = a1[a1.index("\n1. "):a1.index("\n2. ")]
    assert "nothing running" in a1 and "grading card" in a1
    assert "docker compose build" in a1 and a1.index("docker compose build") < a1.index(
        "think_shift_check")
    assert "sudo docker compose run --rm --no-deps bench python scripts/think_shift_check.py" in a1
    assert a1.index("think_shift_check") < a1.index("docker compose up -d")


# ---------------------------------------------------------------------------
# 13 and 14. every halt waits for its press; an old halt file too; Carry on
# ---------------------------------------------------------------------------

def halted_batch(job_kind: str = "judge", hold: bool | None = True) -> str:
    from service import db
    bid = "or_" + "ab12cd34ef56"
    d = Path(config.BENCH_ROOT) / "llm_batches" / "openrouter" / bid
    d.mkdir(parents=True, exist_ok=True)
    (d / "requests.jsonl").write_text(json.dumps({"custom_id": "q1", "body": {}}) + "\n")
    h = {"why": "waiting: OpenRouter refused 20 requests in a row — the key's limit. It tries "
                "again in 10 minutes", "status": 402, "at": time.time() - 3600, "n": 20}
    if hold is not None:
        h["hold"] = hold
    (d / "halt.json").write_text(json.dumps(h))
    db.batch_add(bid, job_kind, 0, 1, "openrouter", "openai/gpt-4.1")
    return bid


def test_13_the_judges_batches_wait_for_carry_on(tmp_path):
    from service import llm
    assert llm.OpenRouterChat.HALT_PRESS == "Carry on"         # d94f7f6: ""
    be = llm.OpenRouterChat.__new__(llm.OpenRouterChat)
    be.dir = tmp_path
    (tmp_path / "or_1").mkdir()
    be._halt("or_1", {"error": "HTTP 402: insufficient credits", "status": 402}, 20)
    h = json.loads((tmp_path / "or_1" / "halt.json").read_text())
    assert h["hold"] is True
    h["at"] = time.time() - 3600                                 # an hour on: still waiting
    (tmp_path / "or_1" / "halt.json").write_text(json.dumps(h))
    assert llm._halted(tmp_path / "or_1", 600).endswith(
        "It waits for Carry on: nothing more is sent until then")


def test_14_a_halt_file_written_before_holds_is_held(tmp_path):
    from service import llm
    (tmp_path / "or_2").mkdir()
    (tmp_path / "or_2" / "halt.json").write_text(json.dumps({
        "why": "waiting: OpenRouter refused 20 requests in a row — the limit. It tries again in "
               "10 minutes", "status": 402, "at": time.time() - 3600, "n": 20}))   # 70df001's
    why = llm._halted(tmp_path / "or_2", 600)
    assert why == ("waiting: OpenRouter refused 20 requests in a row — the limit. It waits for "
                   "Carry on: nothing more is sent until then")         # d94f7f6: "" (lifted)


def test_13_carry_on_takes_a_halted_batch_up_again(svc, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient

    import service.app as appmod
    from service import llm
    bid = halted_batch(hold=None)
    c = TestClient(appmod.app)
    page = c.get("/api/ai").json()
    judge = next(j for j in page["jobs"] if j["job"] == "judge")
    assert judge["halted_batch"] == bid and "It waits for Carry on" in judge["last_batch"]
    assert c.post("/api/ai/carry-on", json={"batch_id": bid, "by": ""}).status_code == 422
    r = c.post("/api/ai/carry-on", json={"batch_id": bid, "by": "masein"})
    assert r.status_code == 200, r.text
    assert not (llm.batch_dir(bid) / "halt.json").exists()
    assert next(j for j in r.json()["jobs"] if j["job"] == "judge")["halted_batch"] is None
    assert c.post("/api/ai/carry-on", json={"batch_id": bid, "by": "masein"}).status_code == 409


# ---------------------------------------------------------------------------
# 15. an earlier fetch's misses don't count in this one
# ---------------------------------------------------------------------------

def test_15_one_failed_ssh_after_a_restart_never_reads_a_box_destroyed(tmp_path, monkeypatch,
                                                                        capsys):
    import frontier_fetch as ff
    ff_, calls, key, state = world(tmp_path, monkeypatch, lambda host, rnd: None)
    dest = tmp_path / "b" / "bundles"
    dest.mkdir(parents=True)
    bid = ff.box_id("1.1.1.1:41", dest)
    # read safe hours ago; missed once by the fetch before the restart, an hour ago
    (dest.parent / "safe-boxes.json").write_text(json.dumps(
        {bid: {"at": time.time() - 5 * 3600, "missed": 1,
               "missed_since": time.time() - 3600}}))
    assert main_of(ff, key, dest, "1.1.1.1:41") == 1           # d94f7f6: 0, "destroyed: done"
    out = capsys.readouterr().out
    assert "destroyed: done" not in out and "NOT safe to destroy yet" in out
    # within one fetch, a round every 20 minutes: destroyed once unreached 30
    real_time, real_sleep, clock = time.time, time.sleep, [time.time()]
    monkeypatch.setattr(ff.time, "time", lambda: clock[0])
    monkeypatch.setattr(ff.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + 20 * 60))
    try:
        assert main_of(ff, key, dest, "--every", "20m", "1.1.1.1:41") == 0
    finally:
        monkeypatch.setattr(ff.time, "time", real_time)
        monkeypatch.setattr(ff.time, "sleep", real_sleep)
    out = capsys.readouterr().out
    assert out.count("NOT safe to destroy yet") == 2 and "destroyed: done" in out


# ---------------------------------------------------------------------------
# 16. the server's words never go out after an unclosed bracket
# ---------------------------------------------------------------------------

def test_16_the_written_off_line_holds_the_status_alone():
    src = (ROOT / "service" / "frontier.py").read_text()
    assert "counted wrong ({why[:200]})" not in src                       # d94f7f6: there
    assert "counted wrong ({status_of(why)})" in src


def test_16_an_unclosed_reason_is_cut_where_it_opens():
    import export_safe as es
    lines = ["[frontier] gpqa_diamond_epoch: question rec1, run 0: the server failed on it "
             "twice — written as no answer, counted wrong (HTTP 400: model gemma-judge rejected "
             "secret-words", "",
             "[frontier] gpqa_diamond_epoch: 120 of 198 answered"]
    q = es.Questions(["An invented question about something else entirely here?"])
    out, _ = es.log_lines("\n".join(lines), q)
    text = "\n".join(out)
    for word in ("secret-words", "gemma-judge", "rejected"):
        assert word not in text, text                                   # d94f7f6: out
    assert es._cut_unclosed("a (b) c (d e") == "a (b) c"
    assert es._cut_unclosed("a (b) c") == "a (b) c"


# ---------------------------------------------------------------------------
# 17. the small ones
# ---------------------------------------------------------------------------

def test_17_a_shard_is_its_number_of_its_split(tmp_path):
    import argparse

    import frontier_fetch as ff
    a = argparse.Namespace(imported={"served/phone": {"on": {"tasks": [], "shards": {
        "hle_text_cais": ["2-of-3"]}}}})
    assert ff.imported_here(a, "served/phone", ("on", ["hle_text_cais"], "2/3"))
    assert not ff.imported_here(a, "served/phone", ("on", ["hle_text_cais"], "2/2"))  # d94f7f6


def test_17_a_step_imported_elsewhere_is_done_only_once_the_line_ended(tmp_path, monkeypatch,
                                                                       capsys):
    there = tmp_path / "there"
    there.mkdir()
    b = a_bundle(there / "frontier-served__phone-thinking-off-mmlu-pro.tar.gz", "served/phone")
    whole = progress_of("phone", "A8-2", "whole", tasks=[MMLU], bundle={"name": b.name})
    ended = {"v": False}

    def listing(host, rnd):
        got = json.loads(with_steps(listing_of({f"/workspace/phone/A8-2/{b.name}": b}, [whole]),
                                    [("phone", "A8-2")]))
        got["lines"] = [{"box": "A8", "build": "phone",
                         "state": "ended" if ended["v"] else "running"}]
        return json.dumps(got)
    ff, calls, key, state = world(tmp_path, monkeypatch, listing)
    monkeypatch.setattr(ff, "copy", lambda *a, **k: (True, "here already"))
    held = {"on": {"tasks": ["arc_agi2_public"], "shards": {}}, "off": {"tasks": [],
                                                                       "shards": {}}}
    plain = ff.run

    def run(cmd, cwd=None, timeout=None, stdin=None):
        if "--imported" in cmd:
            return 0, json.dumps(held)
        return plain(cmd, cwd=cwd, timeout=timeout, stdin=stdin)
    monkeypatch.setattr(ff, "run", run)
    dest = tmp_path / "b" / "bundles"
    main_of(ff, key, dest, "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "NOT safe to destroy — phone A8-1 hasn't started" in out, out   # d94f7f6: done
    ended["v"] = True
    assert main_of(ff, key, dest, "1.1.1.1:41") == 0
    assert "phone/A8-1: imported already from another box — done" in capsys.readouterr().out


def test_17_the_box_line_says_when_it_ran_and_ended(tmp_path, monkeypatch):
    import subprocess

    import frontier_box as fbx
    states = []
    real = Path.write_text

    def spy(self, data, *a, **k):
        if self.name.endswith(".line.json"):
            states.append(json.loads(data)["state"])
        return real(self, data, *a, **k)
    monkeypatch.setattr(Path, "write_text", spy)
    monkeypatch.setattr(subprocess, "run", lambda cmd: type("R", (), {"returncode": 0})())
    fbx.main(["A5", "--as", "served/phone", "--gguf", "x.gguf", "--server", "s.tar.gz",
              "--root", str(tmp_path)])
    assert states == ["running", "ended"]
    got = json.loads((tmp_path / "phone" / "A5.line.json").read_text())
    assert got["box"] == "A5" and got["state"] == "ended"


def test_17_a_scoring_failure_is_never_hidden_behind_a_thinking_refusal():
    import frontier_fetch as ff
    # the import's own words (import_frontier: "not imported — scoring it failed (…)")
    said = ("served/phone · thinking: GPQA Diamond: not imported — scoring it failed "
            "(KeyError: 'x') · HLE: thinking was off, and 900 of its 2,158 answers hold "
            "thinking (41.7%) — more than a quarter: the server ignored the thinking switch: "
            "not scored — not imported")
    assert not ff.think_only(said)                                      # d94f7f6: True
    assert ff.think_only("refused — HLE: thinking was off, and 900 of its 2,158 answers hold "
                         "thinking (41.7%)")


def test_17_a_verdict_alone_is_the_judges_only_when_it_ends_the_reply():
    def hle(reply):
        return fg.read("hle", reply, {"id": "h1", "answer": "42"})["ok"]
    assert hle('{"correct": "yes"}\nThe response says 41, but the answer is 42, so it is '
               'wrong.') is None                                          # d94f7f6: True
    assert hle('The response says 42, which is the answer.\n{"correct": "yes"}') is True
    assert hle('```json\n{"correct": "no"}\n```') is False


def test_17_the_export_names_what_it_wont_remove(svc, monkeypatch):  # noqa: F811
    import export_safe as es
    monkeypatch.setenv("SCRUB_HOSTS", "evalsrv01")
    monkeypatch.setenv("SCRUB_ACCOUNTS", "teamacct")
    monkeypatch.setattr(config, "LOCAL_BASE_URL", "http://gemma-judge:8000/v1")
    es._names.clear()
    line = es.will_remove()
    removes, _, left = line.partition("; NOT removed")
    assert "gemma-judge" not in removes and "gemma-judge" in left       # d94f7f6: unsaid
    assert left.startswith(", the board's undotted names — add those you want removed to "
                           "SCRUB_HOSTS: ")
    monkeypatch.setenv("SCRUB_HOSTS", "evalsrv01,gemma-judge")
    es._names.clear()
    assert "NOT removed" not in es.will_remove()
