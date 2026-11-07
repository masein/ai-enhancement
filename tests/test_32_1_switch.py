"""32.1, 32.2 and 32.4 (brief 32, Kevin and Roohi's review): switching mode
mid-chat with the memory kept — the team's keep_cache_across_paths measured
on step2388, on runtime k3k6r-0411 (the team's research build run as the
apps run it: the history each earlier reply's content alone, never its
thinking, --reasoning-preserve, 4 checkpoints a slot, no prompt cache in
host memory, a conversation capped at 16,384 tokens). Part 2: five-turn
chats in two arms, the switch off (A) and on (B), the same text and seed a
turn, A's reply replayed in B's slot as A's server generated it (its token
ids), each reply's round trip measured — a k3 reply kept to its end, a k6
reply rolled back to its question; Part 1: five runs of a chat's last turn
forced along the reference, 48k a stress test. The server is the stand-in
(tests/fixtures/fake_llama_server.py, FAKE_LLAMA_MEMORY=1: each slot's
memory kept as the team's server keeps it, the path's line in its log, an
earlier reply without its thinking written with an empty think block, a
GSM8K reply that doesn't come back as it was written, and a memory another
path computed answering a little differently); the questions invented;
nothing is fetched, no model runs."""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
from pathlib import Path

import pytest

import frontier as fb
import frontier_box as fx
import frontier_fetch as ff
import import_remote as ir
import remote_bundle as rb
import remote_gguf as rg
import switch_box as sw
from service import config, runner
from service import frontier as sf
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import box, free_port, requests  # noqa: F401
from test_20_step4000 import card  # noqa: F401
from test_31_1_k3k6 import NAME, k3k6_tarball, step_kept, write_gguf  # noqa: F401

RT = "k3k6r-0411"
SERVED = "served/step2388-k3k6r-0411"
VERSION_LINE = "version: 7100 (917f3d911)\nbuilt with GNU 11.4.0 for Linux x86_64"
N = 6
ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "REMOTE-RUNS.md"


def invented(task: str) -> list[dict]:
    """each task's questions, invented: the stand-in answers the option that
    begins "right", and a number the question gives as "(answer N)\""""
    if task == "mmlu_chat":
        return [{"id": f"m{k}", "question": f"Invented MMLU question {k}?",
                 "options": [f"right m{k}", f"wrong m{k}a", f"wrong m{k}b", f"wrong m{k}c"],
                 "answer": "A"} for k in range(N)]
    if task == "gsm8k_chat":
        return [{"id": f"g{k}", "question": f"Invented sum {k} (answer {k + 10})?",
                 "answer": str(k + 10)} for k in range(N)]
    if task == "math_l5_epoch":
        return [{"id": f"x{k}", "question": f"Find x {k} (answer {k})", "answer": str(k)}
                for k in range(N)]
    return [{"id": f"rec{k}", "question": f"Invented hard question {k}?",
             "right": f"right answer {k}", "wrong": [f"wrong {k}a", f"wrong {k}b", f"wrong {k}c"]}
            for k in range(2)]


def stored_replies() -> None:
    """step2388's k3 answers to MMLU and GSM8K, as plan U's import keeps them
    — 32.4: and its k6 answers to GPQA Diamond (4 runs) and MATH Level 5:
    rec0's all within 10,240 tokens, one of rec1's not; x5 ran out"""
    row = config.OUT_DIR / "served__step2388-k3k6-2611"
    for task, pre in (("mmlu_chat", "m"), ("gsm8k_chat", "g")):
        d = sf.task_dir(row, task)
        d.mkdir(parents=True, exist_ok=True)
        (d / sf.ANSWERS).write_text("".join(json.dumps(
            {"id": f"{pre}{k}", "epoch": 0, "finish": "stop",
             "answer": f"Stored reply {pre}{k}, worked through.\nANSWER: A"}) + "\n"
            for k in range(N)), encoding="utf-8")
    k6 = config.OUT_DIR / "served__step2388-k3k6-2611__thinking"
    runs = {"gpqa_diamond_epoch": [("rec0", e, 4000 + 1000 * e, "stop") for e in range(4)]
            + [("rec1", e, 12_000 if e == 2 else 3000, "stop") for e in range(4)],
            "math_l5_epoch": [(f"x{k}", 0, 2000 + 1000 * k, "stop" if k < 5 else "length")
                              for k in range(N)]}
    for task, rows in runs.items():
        d = sf.task_dir(k6, task)
        d.mkdir(parents=True, exist_ok=True)
        (d / sf.ANSWERS).write_text("".join(json.dumps(
            {"id": q, "epoch": e, "finish": f, "tokens": n,
             "answer": "<think>\nworked\n</think>\n\nANSWER: A" if f == "stop" else "<think>\n"})
            + "\n" for q, e, n, f in rows), encoding="utf-8")


@pytest.fixture
def sb(box, card, svc, monkeypatch, tmp_path):  # noqa: F811
    """the server with the experiment's inputs, and a box given step2388's
    file and the team's research build, its server keeping each slot's memory"""
    monkeypatch.setattr(fb, "_fetch", invented)
    monkeypatch.setenv("FAKE_LLAMA_PATHS", "1")
    monkeypatch.setenv("FAKE_LLAMA_MEMORY", "1")
    monkeypatch.setenv("FAKE_LLAMA_VERSION", VERSION_LINE)
    gguf = write_gguf(tmp_path / "models" / NAME)
    rs = tmp_path / "research"
    rs.mkdir()
    tb = k3k6_tarball(rs, commit="917f3d911", changes="no", changed="")
    stored_replies()
    meta = sw.inputs(sw.where(), say=lambda x: None, n_chats=4, bins=(30, 90), per_bin=1)
    monkeypatch.setitem(fx.PLAN_PINS, "S", {"step2388.k3k6.gguf": rb.sha256_file(gguf),
                                            fx.TARBALL_S: rb.sha256_file(tb)})
    return {**box, "gguf": gguf, "tarball": tb, "inputs": Path(meta["file"]),
            "inputs_sha": meta["sha256"]}


def run_s(b: dict, out: str, *more: str, slots: int = 4) -> int:
    """remote_gguf.py as a plan S box runs it — the server's config put back after"""
    keep = {k: getattr(config, k) for k in [*rg.ENV, "OUT_DIR", "DM_ITEMS"]}
    lock, env = runner.LOCK, {v: os.environ.get(v) for v in rg.ENV.values()}
    try:
        return rg.main(["--as", SERVED, "--gguf", str(b["gguf"]), "--server", str(b["tarball"]),
                        "--based-on", "Qwen/Qwen3.6-35B-A3B", "--thinking", "on",
                        "--slots", str(slots), "--port", str(free_port()), "--load-timeout", "60",
                        "--runtime", RT, "--max-context", "100000000",
                        "--switch", str(b["inputs"]), "--pin",
                        f"{b['inputs'].name}={b['inputs_sha']}", "--out", str(b["root"] / out),
                        *more])
    finally:
        for k, v in keep.items():
            setattr(config, k, v)
        runner.LOCK = lock
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def bundle_at(b: dict, out: str, what: str, part=None) -> Path:
    return b["root"] / out / rb.bundle_name(sw.SUITE, SERVED, True, part, parts=[what])


def imported(path: Path) -> tuple[int, list[str]]:
    said: list[str] = []
    return ir.import_bundle(path, "masein", said.append), said


# ---------------------------------------------------------------------------
# the runtime
# ---------------------------------------------------------------------------

def test_the_runtime_is_k3k6_2611s_with_the_prompt_cache_in_host_memory_off():
    rt, was = fb.RUNTIMES[RT], fb.RUNTIMES["k3k6-2611"]
    # 32.2: as the apps run it — the thinking kept, 4 checkpoints a slot
    assert rt["flags"] == [*was["flags"], "--reasoning-preserve", "--ctx-checkpoints", "4",
                           "--cache-ram", "0"]
    assert rt["env"] == was["env"] == {"LOOKAHEAD_MOE_ROUTER_NORM": "1",
                                       "GGML_CPU_DISABLE_FUSION": "1"}
    assert (rt["sampling"], rt["paths"], rt["refuses_flags"]) == (
        {}, {"off": 3, "on": 6}, was["refuses_flags"])
    assert rt["experiment"] == "switch" and list(rt["models"]) == [SERVED]
    assert rt["build"]["commits"] == ("917f3d911",)
    # the research build at its commit, and nothing else
    assert fb.runtime_build_refused(RT, None, "917f3d9115", {"commit": "917f3d911521"}) == ""
    # 32.3: the tarball as built — "build 20", the shallow clone's commit count,
    # is never held to anything: its commit is what identifies it
    assert fb.runtime_build_refused(RT, 20, "917f3d9", {
        "commit": "917f3d91152142e952b7ea128980faa75ba7934b", "uncommitted_changes": "no",
        "build": "20", "cuda_archs": "120"}) == ""
    assert "runs the team's llama.cpp-teraformer at 917f3d911" in fb.runtime_build_refused(
        RT, None, "3ad38b7cc")
    # the board's model runs on its own runtime alone, the experiment's on this one
    assert "runs on runtime k3k6-2611 alone" in fb.runtime_model_refused(
        RT, "served/step2388-k3k6-2611")
    assert fb.runtime_model_refused(RT, SERVED) == ""


@pytest.mark.parametrize("more,said", [
    (["--only", "gpqa_diamond_epoch"], "runtime k3k6r-0411 is the switch experiment's alone"),
    # the board's settings check keeps its words: the runtimes of its rows
    (["--check"], "--check: the settings check of a runtime of expert paths (--runtime "
                  "k3k6-2611 or k3k6-2611-app)"),
    (["--switch", "x.jsonl", "--switch-part", "chats", "--runtime", "k3k6-2611",
      "--as", "served/step2388-k3k6-2611"], "--switch: on runtime k3k6r-0411"),
    (["--switch", "x.jsonl"], "--switch: give --switch-part chats"),
    (["--switch", "x.jsonl", "--switch-part", "drift", "--part", "1/2"],
     "--part shares out the chats"),
    (["--switch", "x.jsonl", "--switch-part", "chats", "--only", "gpqa_diamond_epoch"],
     "--switch runs the experiment alone"),
])
def test_the_research_runtime_asks_the_experiment_alone(tmp_path, capsys, more, said):
    base = ["--as", SERVED, "--gguf", "x.gguf", "--server", "x.tar.gz", "--thinking", "on",
            "--runtime", RT, "--out", str(tmp_path / "o")]
    argv = base[:]
    for k in ("--runtime", "--as"):
        if k in more:
            i = argv.index(k)
            del argv[i:i + 2]
    with pytest.raises(SystemExit):
        rg.main(argv + more)
    assert said in capsys.readouterr().err


# ---------------------------------------------------------------------------
# what is asked, and how each turn is scored
# ---------------------------------------------------------------------------

def test_the_facts_are_scored_exactly_and_their_sum_by_its_value():
    f = {"code": 482, "crates": 36, "courier": "Odile", "gate": 27}
    assert sw.facts_of(7) == sw.facts_of(7) != sw.facts_of(8)
    assert set(sw.facts_of(7)) == {"code", "crates", "courier", "gate"}
    assert sw.compute_answer(f) == sw.fractions.Fraction(482 * 36, 518)
    assert sw.recall_score("courier: Odile\ngate: 27", "stop", f)["ok"]
    # its own markdown, "the" before the name, "gate number" read as the same
    assert sw.recall_score("**courier:** the odile\n- gate number: **27**.", "stop", f)["ok"]
    got = sw.recall_score("courier: Odile\ngate: 72", "stop", f)
    assert (got["ok"], got["wrong"], got["no_answer"]) == (False, ["gate"], False)
    assert sw.recall_score("I don't remember.", "stop", f)["no_answer"]
    cut = sw.recall_score("<think>\nlet me see", "length", f)
    assert (cut["ok"], cut["ran_out"], cut["wrong"]) == (False, True, [])
    want = sw.compute_answer(f)
    good = sw.compute_score(f"So.\nANSWER: {want.numerator}/{want.denominator}", "stop", f)
    assert good["ok"] and good["lowest_terms"]
    # the same value not in lowest terms: right, and said
    loose = sw.compute_score(f"ANSWER: {want.numerator * 2}/{want.denominator * 2}", "stop", f)
    assert loose["ok"] and not loose["lowest_terms"]
    assert not sw.compute_score("ANSWER: 33", "stop", f)["ok"]
    assert sw.compute_score("no answer line", "stop", f)["no_answer"]
    # 32.4: the facts all read by k3 — nothing k6 reads outlives its turn in
    # the apps' memory: a in turn 1, b, the courier and the gate in turn 2;
    # a and b used on k6 (turn 5), the courier and gate recalled on k3 (turn 4)
    line = {"facts": f, "mmlu": "m0", "gsm8k": "g0", "hard": {"task": "math_l5_epoch", "id": "x0"}}
    items = {t: {x["id"]: x for x in invented(t)} for t in ("mmlu_chat", "gsm8k_chat",
                                                            "math_l5_epoch")}
    t1, t2, t3 = (sw.turn_text(t, line, items) for t in (1, 2, 3))
    assert "locker code is 482" in t1 and "36 crates" in t2
    assert "courier's name is Odile" in t2 and "gate number is 27" in t2
    assert not any(w in t1 for w in ("Odile", "gate", "crates"))
    assert t1.endswith(fb.prompt("mmlu_chat", items["mmlu_chat"]["m0"])[0])
    assert t3 == fb.prompt("math_l5_epoch", items["math_l5_epoch"]["x0"])[0]   # no fact
    assert (sw.turn_text(4, line, items), sw.turn_text(5, line, items)) == (sw.RECALL, sw.COMPUTE)
    assert [sw.TURNS[t - 1] for t in (1, 2, 3, 4, 5)] == [3, 3, 6, 3, 6]


def test_mcnemar_the_paired_difference_and_the_clustered_test():
    assert sw.mcnemar_p(0, 0) == 1.0
    assert sw.mcnemar_p(0, 6) == pytest.approx(2 / 64)
    assert sw.mcnemar_p(3, 3) == 1.0
    p = sw.paired([True] * 90 + [False] * 10, [True] * 80 + [False] * 20)
    assert (p["only_a"], p["only_b"], p["diff"]) == (10, 0, -10.0)
    assert p["ci"][0] < -10 < p["ci"][1] < 0 and p["p"] == pytest.approx(2 / 1024)
    rows = [({"arms": {"A": {"score": {"ok": a}}, "B": {"score": {"ok": b}}}},
             {"hard": {"task": "gpqa_diamond_epoch", "id": q}})
            for q, a, b in (("r1", True, False), ("r1", True, False), ("r2", False, True),
                            ("r3", True, True))]
    assert sw.clustered(rows) == {"questions": 3, "b_more": 1, "a_more": 1, "p": 1.0}


def test_the_kl_is_over_the_references_twenty_and_the_rest():
    p = [(1, 0.6), (2, 0.3), (3, 0.05)]
    assert sw.kl(p, p) == pytest.approx(0.0, abs=1e-9)
    assert sw.kl(p, [(2, 0.6), (1, 0.3), (3, 0.05)]) > 0.1
    # a token of the reference's outside the run's 20 shares what they leave
    # with the rest: here the same distribution, so nothing apart
    assert sw.kl(p, [(1, 0.6), (2, 0.3)]) == pytest.approx(0.0, abs=1e-9)
    assert sw.kl(p, [(1, 0.6), (2, 0.38)]) == pytest.approx(
        0.3 * sw.math.log(0.3 / 0.38) + 0.05 * sw.math.log(0.05 / 0.01) * 2, rel=1e-6)
    assert sw.kl([], p) is None


def test_each_arms_memory_is_held_to_what_the_source_says():
    keep = {"kind": "keeping", "from": 3, "to": 6, "n": 900, "text": "…"}
    clear = {"kind": "clearing", "from": 3, "to": 6, "n": 900, "text": "…"}
    assert sw.memory_why(1, 3, None, "B", {"cache_n": 0, "line": None}, 0) == ""
    assert "turn 1 began with 12" in sw.memory_why(1, 3, None, "A", {"cache_n": 12}, 0)
    # on the same path A's memory to the turn's question (its probe read it)
    assert sw.memory_why(2, 3, 3, "A", {"cache_n": 79, "line": None}, 79) == ""
    assert "where the apps' memory came back to 79" in sw.memory_why(2, 3, 3, "A",
                                                                     {"cache_n": 77}, 79)
    # a switch: A cleared and read again; B kept what A's probe found
    assert sw.memory_why(3, 6, 3, "A", {"cache_n": 0, "line": clear}, 0) == ""
    assert "no \"clearing\" line" in sw.memory_why(3, 6, 3, "A", {"cache_n": 0, "line": None},
                                                   0)
    assert sw.memory_why(3, 6, 3, "B", {"cache_n": 141, "line": keep}, 141) == ""
    assert "came back to 141" in sw.memory_why(3, 6, 3, "B", {"cache_n": 85, "line": keep}, 141)
    assert "no \"keeping\" line" in sw.memory_why(3, 6, 3, "B", {"cache_n": 141, "line": clear},
                                                  141)
    back = {**keep, "from": 6, "to": 3}
    assert "A's probe gave no cache_n" in sw.memory_why(4, 3, 6, "B",
                                                        {"cache_n": 85, "line": back}, None)
    # 32.4: a reply's round trip, as the apps' next request reads it — a k3
    # reply kept to its end, a k6 reply rolled back to its question
    assert sw.round_trip({"cache_n": 141}, 141, 85, 3) == {
        "cache_n": 141, "end": 141, "start": 85, "path": 3, "came_back": "to its end",
        "as_apps": True, "rolled_back": 0}
    k6 = sw.round_trip({"cache_n": 85}, 141, 85, 6)
    assert (k6["came_back"], k6["as_apps"], k6["rolled_back"]) == ("to its question", True, 56)
    k3 = sw.round_trip({"cache_n": 85}, 141, 85, 3)
    assert (k3["came_back"], k3["as_apps"]) == ("to its question", False)
    assert sw.round_trip({"cache_n": 100}, 141, 85, 6)["came_back"] == "elsewhere"
    # an earlier reply's thinking found in a prompt
    assert sw.thinking_in("… <think>\nLet me weigh the options first.\n</think> …",
                          "Let me weigh the options first.")
    assert not sw.thinking_in("<think>\n\n</think>\n\nA reply.", "Let me weigh it.")
    assert not sw.thinking_in("anything", "")


# ---------------------------------------------------------------------------
# the inputs, on the server
# ---------------------------------------------------------------------------

def test_the_inputs_hold_ids_facts_and_stored_replies_and_no_question_text(sb):
    lines = sw.read_input(sb["inputs"])
    chats = [x for x in lines if x["part"] == "chats"]
    drift = [x for x in lines if x["part"] == "drift"]
    assert [x["chat"] for x in chats] == [0, 1, 2, 3] and len(drift) == 2
    assert all(x["facts"] == sw.facts_of(x["chat"]) for x in chats)
    # turn 3: 32.4 — the questions whose k6 replies in plan U all finished
    # within 10,240 tokens (rec0, not rec1; x0–x4, not x5, which ran out),
    # each GPQA Diamond one 3 times, MATH Level 5 after; no question text
    hard = [(x["hard"]["task"], x["hard"]["id"]) for x in chats]
    assert sorted(hard).count(("gpqa_diamond_epoch", "rec0")) == 3
    assert all(q not in ("rec1", "x5") for _, q in hard)
    assert sum(1 for t, _ in hard if t == "math_l5_epoch") == 1
    raw = sb["inputs"].read_text()
    assert "Invented" not in raw and "Stored reply" in raw
    # Part 1: each history step2388's stored exchanges, each once, stacked past
    # its memory size, the previous turn after; the last question none of them asks
    used = [h["id"] for x in drift for h in x["history"]]
    assert len(used) == len(set(used)) and [x["bin"] for x in drift] == [30, 90]
    assert all(x["last"]["id"] not in used for x in drift)
    assert len(drift[1]["history"]) > len(drift[0]["history"]) >= 2
    meta = json.loads((sw.where() / sw.INPUTS_META).read_text())
    assert meta["sha256"] == rb.sha256_file(sb["inputs"]) == sb["inputs_sha"]
    pool = meta["hard_pool"]
    assert (pool["gpqa_diamond_epoch"], pool["math_l5_epoch"]) == (
        {"questions": 1, "of": 2}, {"questions": 5, "of": N})
    assert (pool["max"], pool["replies"], pool["median"]) == (10_240, 4 + 5, 5000)
    # the same inputs on every run
    again = sw.inputs(sb["root"] / "again", say=lambda x: None, n_chats=4, bins=(30, 90),
                      per_bin=1)
    assert again["sha256"] == meta["sha256"]


# ---------------------------------------------------------------------------
# the box: the check, Part 2 and Part 1
# ---------------------------------------------------------------------------

def test_the_check_shows_every_turns_memory_and_passes(sb, capsys):
    assert run_s(sb, "chk", "--switch-check", "2", slots=6) == 0
    out = capsys.readouterr().out
    # 32.5: the check runs Part 1 too, in Part 1's slots (its 48k memory needs
    # them; on S1's 16,384 its 16k memory didn't fit)
    assert f"6 slots of {sw.DRIFT_CONTEXT:,} tokens" in out
    turns = [x for x in out.splitlines() if x.startswith("chat ") and " · turn " in x]
    assert len(turns) == 2 * 5 * 2
    # 32.4: the history as the apps send it, said before a question; then each
    # arm's memory to the token: B's every later turn what A's probe found
    assert ("the history as the apps send it: an earlier reply without its thinking is written "
            "with an empty think block on k6 and k3") in out
    assert "the earlier thinking out of every prompt: no reply's thinking in the next turn's ✓" \
        in out
    # at each switch, A re-read (its line, nothing reused) and B kept the chat
    a3 = next(x for x in turns if x.startswith("chat 0 · turn 3") and " · A:" in x)
    b3 = next(x for x in turns if x.startswith("chat 0 · turn 3") and " · B:" in x)
    assert "cache_n 0 (expected 0) ·" in a3 and "expert path k3 -> k6: clearing" in a3 \
        and "re-read ✓" in a3
    assert "expert path k3 -> k6: keeping" in b3 and "kept ✓" in b3
    for x in turns:
        exp = int(re.search(r"\(expected ([\d,]+)", x)[1].replace(",", ""))
        assert x.split(": cache_n ")[1].startswith(f"{exp} ")
    assert all("NOT as said" not in x for x in turns)
    # the k6→k3 switch: B kept only what k3 built — to the start of question 3
    b4 = next(x for x in turns if x.startswith("chat 0 · turn 4") and " · B:" in x)
    assert "expert path k6 -> k3: keeping" in b4 and "as A's probe found" in b4
    # each reply as the apps' next request reads it: a k3 reply kept to its
    # end (but a GSM8K reply whose end the template trims — the stand-in's
    # odd answers), a k6 reply rolled back to the start of its question
    lines = {x["chat"]: x for x in sw.read_input(sb["inputs"]) if x["part"] == "chats"}
    trims = {c: (int(lines[c]["gsm8k"][1:]) + 10) % 2 == 1 for c in (0, 1)}
    trips = [x for x in out.splitlines() if "as the apps' next request reads it" in x]
    assert len(trips) == 2 * 5
    for x in trips:
        k6, c = "(k6)" in x, int(x.split()[1])
        assert ("rolled back to the start of its question" in x) == k6, x
        assert x.endswith("✓") == (" reply 2 " not in x or not trims[c]), x
    drift = [x for x in out.splitlines() if x.startswith("drift chat")]
    assert len(drift) == 4 and all(x.endswith("✓") for x in drift)
    assert "the switch check: every memory as said" in out
    assert "Invented" not in out                         # no question text
    assert (sb["root"] / "chk" / sw.CHECK).exists()
    assert not list((sb["root"] / "chk").glob("switch-served-*.tar.gz"))      # no bundle
    # each path's settings check passed, k6 and k3
    assert (sb["root"] / "chk" / rg.CHECK_FILE.format(path=6)).exists()
    assert (sb["root"] / "chk" / rg.CHECK_FILE.format(path=3)).exists()


def test_part_2_asks_both_arms_the_same_text_and_seed_and_imports_marked_again(sb, svc):  # noqa: F811
    assert run_s(sb, "s1", "--switch-part", "chats", "--part", "1/2") == 0
    chats = [r for r in requests(sb) if "id_slot" in r and "messages" in r]
    # two chats, five turns, two arms
    assert len(chats) == 2 * 5 * 2
    for r in chats:
        assert set(r) - {"messages", "keep_cache_across_paths"} == sw.CHAT_KEYS
        assert (r["logprobs"], r["top_logprobs"]) == (True, 1)
        # 32.4: the history as the apps send it — no reply's thinking, ever
        assert not any("reasoning_content" in m for m in r["messages"])
    by = {}
    for r in chats:
        by.setdefault((r["seed"], r["expert_path"]), []).append(r)
    assert len(by) == 10
    for pair in by.values():
        a, b = sorted(pair, key=lambda x: "keep_cache_across_paths" in x)
        assert a["messages"] == b["messages"] and "keep_cache_across_paths" not in a
        assert b["keep_cache_across_paths"] is True and a["id_slot"] != b["id_slot"]
    t5 = next(r for r in chats if r["expert_path"] == 6 and len(r["messages"]) == 9)
    assert "Thinking it through." not in json.dumps(t5["messages"])
    # B's replays and A's probes: the chat as token ids, one token, nothing else
    reads = [r for r in requests(sb) if r.get("n_predict") == 1 and "message_delimiters" in r
             and "n_probs" not in r]
    assert len(reads) == 2 * (4 * 2 + 1)
    assert all(set(r) == {"prompt", "n_predict", "expert_path", "id_slot", "cache_prompt",
                          "message_delimiters"} for r in reads)
    path = bundle_at(sb, "s1", "chats", (1, 2))
    assert path.exists()
    code, said = imported(path)
    assert code == 0, said
    s = json.loads((sw.where() / sw.SUMMARY).read_text())
    t = {x["turn"]: x for x in s["part2"]["turns"]}
    assert s["part2"]["chats"] == 2 and all(x["not_counted"] == 0 for x in t.values())
    # the placebo turns agree. Turn 4, the recall on k3 after the k6 turn:
    # chat 2's reply 2 came back as written, so B kept memory k3 built alone
    # — right, the control; chat 0's didn't (its odd answer, trimmed), so turn
    # 3 rolled back to before question 2, k6 read the courier and the gate
    # again, and the stand-in's mixed memory recalls them wrong. Turn 5, a and
    # b read by k3 and used on k6: the stand-in's sum wrong in B
    assert t[1]["arms"]["A"]["score"] == t[1]["arms"]["B"]["score"] == 100.0
    assert t[2]["paired"]["only_a"] == t[2]["paired"]["only_b"] == 0
    assert (t[4]["arms"]["A"]["score"], t[4]["arms"]["B"]["score"]) == (100.0, 50.0)
    sp = t[4]["by_reply_2"]
    assert (sp["k3_alone"]["n"], sp["k3_alone"]["only_a"]) == (1, 0)
    assert (sp["k6_read_turn_2"]["n"], sp["k6_read_turn_2"]["only_a"]) == (1, 1)
    assert (t[5]["arms"]["A"]["score"], t[5]["arms"]["B"]["score"]) == (100.0, 0.0)
    assert t[5]["paired"]["only_a"] == 2
    assert t[3]["switch"] and t[3]["arms"]["B"]["cache_n"] > 0 == t[3]["arms"]["A"]["cache_n"]
    assert "clustered" in t[3]
    # each reply's round trip, as the apps' next request reads it: all as the
    # apps' history makes it, but chat 0's GSM8K reply (turn 2)
    assert [(x["turn"], x["round_trip"]["as_apps"], x["round_trip"]["otherwise"])
            for x in s["part2"]["turns"]] == [(1, 2, 0), (2, 1, 1), (3, 2, 0), (4, 2, 0), (5, 2, 0)]
    assert t[3]["round_trip"]["came_back"]["to its question"] == 2
    # each turn's limit what is left of 16,384
    rows = [json.loads(x) for x in
            rb.read(path)["files"]["results/switch/chats.jsonl"].decode().splitlines()]
    for r in rows:
        for tr in r["turns"]:
            assert tr["room"] == sw.APP_CONTEXT - tr["prompt"] == \
                tr["arms"]["A"]["request"]["max_tokens"]
    # B's memory each later turn what A's probe found: after a k3 reply its
    # end, after a k6 reply the start of its question (turn 4: only what k3
    # built is kept)
    for r in rows:
        for tr in r["turns"][1:]:
            before = r["turns"][tr["turn"] - 2]
            assert tr["arms"]["B"]["cache_n"] == tr["expected"]["B"] == before["probe"]["cache_n"]
        assert r["turns"][3]["arms"]["B"]["cache_n"] == r["turns"][2]["start"]
        assert r["turns"][2]["arms"]["B"]["cache_n"] == r["turns"][1]["probe"]["cache_n"]
        # B's replay held A's reply as generated: one token short of it
        rep = r["turns"][0]["replay"]
        assert rep["tokens"] == r["turns"][0]["prompt"] + r["turns"][0]["arms"]["A"]["tokens"]
    # a second import changes nothing
    code, said = imported(path)
    assert code == 0 and any("imported already" in x for x in said)
    # the report reads the summary kept: internal, no question text
    rep = svc.get("/api/reports/switching-mode").json()
    assert rep["kind"] == "switch" and rep["internal"] and rep["part2"]["chats"] == 2
    assert [x["turn"] for x in rep["part2"]["turns"]] == [1, 2, 3, 4, 5]
    assert "Invented" not in json.dumps(rep)
    listed = {c["id"]: c for c in svc.get("/api/reports").json()["reports"]}["switching-mode"]
    assert listed["internal"] and "2 chats and 0 drift runs imported" in listed["line"]


def test_part_1_forces_each_run_along_the_reference(sb, svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(sw, "REF_TOKENS", 24)        # the reference's 512, shrunk
    assert run_s(sb, "s11", "--switch-part", "drift", slots=5) == 0
    path = bundle_at(sb, "s11", "drift")
    rows = [json.loads(x) for x in
            rb.read(path)["files"]["results/switch/drift.jsonl"].decode().splitlines()]
    assert sorted((r["chat"], r["direction"]) for r in rows) == [
        (0, "k3→k6"), (0, "k6→k3"), (1, "k3→k6"), (1, "k6→k3")]
    for r in rows:
        runs = r["runs"]
        assert all(v["memory"] == "" for v in runs.values()) and r["ref"]["memory"] == ""
        # B kept the chat to the end of its previous reply, read by the other path
        assert r["memory"] == r["e_prev"] > 0 and r["e_prev"] < r["s_last"]
        assert runs["B"]["requests"][0]["line"]["kind"] == "keeping"
        # A′ and N are the reference again; B parts from it now and then; F
        # (the other path, fresh) almost everywhere
        assert runs["A2"]["flips"] == runs["N"]["flips"] == []
        assert 0 < len(runs["B"]["flips"]) < len(runs["F"]["flips"])
        assert runs["F"]["mean_kl"] > runs["B"]["mean_kl"] > runs["N"]["mean_kl"] == 0
        assert runs["B"]["sent"]["temperature"] == 0.0 and runs["B"]["sent"]["n_probs"] == 20
    code, said = imported(path)
    assert code == 0, said
    s = json.loads((sw.where() / sw.SUMMARY).read_text())
    assert s["part1"]["jobs"] == 4 and len(s["part1"]["rows"]) == 4
    assert "greedy, no penalties" in s["part1"]["protocol"]
    # 32.4: 48k is the stress test, apart
    assert not any(x["stress"] for x in s["part1"]["rows"])
    monkeypatch.setattr(sw, "STRESS_BINS", (90,))
    s = sw.summarise(sw.where())
    assert [(x["bin"], x["stress"]) for x in s["part1"]["rows"]][::2] == [(30, False), (90, True)]


def test_the_size_fitted_to_the_apps_cap_drops_its_oldest_exchanges_until_it_fits(
        sb, monkeypatch, capsys):
    # 32.5 (masein, 6 Nov): the largest memory that fits the apps' cap with the
    # last question and the 512-token reference — stacked past it on the
    # server, cut on the box, the same in both directions (here the cap and
    # the reference shrunk, words for tokens)
    monkeypatch.setattr(sw, "APP_FIT_BINS", (90,))
    monkeypatch.setattr(sw, "APP_CONTEXT", 160)
    monkeypatch.setattr(sw, "REF_TOKENS", 24)
    meta = sw.inputs(sw.where(), say=lambda x: None, n_chats=4, bins=(30, 90), per_bin=1)
    fitted = {**sb, "inputs": Path(meta["file"]), "inputs_sha": meta["sha256"]}
    assert run_s(fitted, "fit", "--switch-part", "drift", slots=5) == 0
    rows = [json.loads(x) for x in rb.read(bundle_at(fitted, "fit", "drift"))["files"][
        "results/switch/drift.jsonl"].decode().splitlines()]
    fit = [r for r in rows if r["bin"] == 90]
    assert len(fit) == 2 and all(r["fits_app"] for r in fit)
    assert all(r["prompt"] + 24 <= 160 for r in fit)
    assert fit[0]["dropped"] == fit[1]["dropped"] >= 1        # both directions alike
    assert fit[0]["e_prev"] == fit[1]["e_prev"]
    assert all("fits_app" not in r for r in rows if r["bin"] == 30)
    code, said = imported(bundle_at(fitted, "fit", "drift"))
    assert code == 0, said
    s = json.loads((sw.where() / sw.SUMMARY).read_text())
    assert [x["fit"]["fits"] for x in s["part1"]["rows"] if x["bin"] == 90] == [1, 1]
    # the check says so, and holds a chat that doesn't fit as not as said
    assert run_s(fitted, "fitchk", "--switch-check", "1", slots=5) == 0
    out = capsys.readouterr().out
    assert "fits the apps' 160 with the last question and the 24-token reference" in out
    monkeypatch.setattr(sw, "APP_CONTEXT", 40)               # nothing fits now
    assert run_s(fitted, "fitchk2", "--switch-check", "1", slots=5) == 1
    assert "past the apps' 40" in capsys.readouterr().out


def test_the_import_refuses_what_wasnt_asked_as_the_experiment_asks(sb, svc, tmp_path,  # noqa: F811
                                                                    monkeypatch):
    assert run_s(sb, "s2", "--switch-part", "chats", "--part", "2/2") == 0
    path = bundle_at(sb, "s2", "chats", (2, 2))

    def tampered(change) -> Path:
        b = rb.read(path)
        files = dict(b["files"])
        change(files)
        bundle = json.loads(files["bundle.json"])
        bundle["files"] = {k: rb.hashlib.sha256(v).hexdigest() for k, v in files.items()
                           if k != "bundle.json"}
        files["bundle.json"] = json.dumps(bundle).encode()
        d = tmp_path / f"t{len(list(tmp_path.glob('t*')))}"
        return rb.write(d / path.name, files)

    def temp(files):
        rows = [json.loads(x) for x in files["results/switch/chats.jsonl"].decode().splitlines()]
        rows[0]["turns"][2]["arms"]["B"]["request"]["temperature"] = 0.7
        files["results/switch/chats.jsonl"] = "".join(json.dumps(r) + "\n" for r in rows).encode()
    code, said = imported(tampered(temp))
    assert code == sw.REFUSED and any("turn 3 arm B was asked with" in x for x in said)

    def other_inputs(files):
        st = json.loads(files["setup.json"])
        st["switch"]["inputs_sha256"] = "0" * 64
        files["setup.json"] = json.dumps(st).encode()
    code, said = imported(tampered(other_inputs))
    assert code == sw.REFUSED and any("its inputs file is 0000" in x for x in said)
    # plan S's tarball not pinned yet: nothing is imported
    monkeypatch.setitem(fx.PLAN_PINS, "S", {**fx.PLAN_PINS["S"], fx.TARBALL_S: None})
    code, said = imported(path)
    assert code == sw.REFUSED and any("tarball isn't pinned yet" in x for x in said)
    assert not (sw.where() / "chats").exists()


# ---------------------------------------------------------------------------
# plan S, the fetch and the runbook
# ---------------------------------------------------------------------------

TARBALL_SHA = "be1ba7e967f19afd5f2a835a83cc9d04e151d9a3bb733733a4a69483838b33bf"
TARBALL_REV = "884b970e978ef821d5006bc7bff6bedfd3f84a41"


def test_plan_s_runs_the_pinned_tarball_at_its_commit_and_each_box_its_share(monkeypatch,
                                                                              capsys):
    # 32.3: the research tarball as masein uploaded it (5 Nov) — its sha256
    # and the private repository's commit; step2388's file at plan U's
    assert (fx.PLAN_PINS["S"][fx.TARBALL_S], fx.PLAN_REVISIONS["S"][fx.TARBALL_S]) == (
        TARBALL_SHA, TARBALL_REV)
    assert fx.PLAN_PINS["S"]["step2388.k3k6.gguf"] == fx.PLAN_PINS["U"]["step2388.k3k6.gguf"]
    line = ["S1", "--as", SERVED, "--gguf",
            "hf://o/evalboard-private@fe12fa27825b3a0cc6cde3dedec3e838bd3e6de4/step2388.k3k6.gguf",
            "--server", f"hf://o/evalboard-private@{TARBALL_REV}/{fx.TARBALL_S}",
            "--switch", "/workspace/switch-inputs.jsonl", "--switch-sha256", "b" * 64]
    # the tarball at another commit: refused before anything is fetched
    with pytest.raises(SystemExit) as e:
        fx.main([*line[:5], "--server", f"hf://o/evalboard-private@{'a' * 40}/{fx.TARBALL_S}",
                 *line[7:]])
    assert f"{fx.TARBALL_S} at {'a' * 40}, where plan S fetches it at {TARBALL_REV}" in str(e.value)
    # and a pin taken away refuses the whole plan
    monkeypatch.setitem(fx.PLAN_PINS, "S", {**fx.PLAN_PINS["S"], fx.TARBALL_S: None})
    with pytest.raises(SystemExit) as e:
        fx.main(line)
    assert f"plan S's {fx.TARBALL_S} isn't pinned yet" in str(e.value)
    monkeypatch.setitem(fx.PLAN_PINS, "S", {**fx.PLAN_PINS["S"], fx.TARBALL_S: TARBALL_SHA})
    monkeypatch.setattr(fx, "gpu_refused", lambda label: "")
    got = {}
    monkeypatch.setattr(fx, "run_box", lambda a, label: got.setdefault(label, [
        fx.argv_of(label, k, st, a) for k, (_, st) in enumerate(fx.steps_of(label), 1)]) and 0)
    monkeypatch.setattr(fx, "line_state", lambda *a, **k: None)
    assert fx.main(line + ["--price", "0.45"]) == 0
    argv = got["S1"][0]
    for want in (["--runtime", RT], ["--switch", "/workspace/switch-inputs.jsonl"],
                 ["--pin", f"switch-inputs.jsonl={'b' * 64}"], ["--switch-part", "chats"],
                 ["--part", "1/8"], ["--slots", "16"], ["--thinking", "on"],
                 ["--pin", f"{fx.TARBALL_S}={TARBALL_SHA}"]):
        assert " ".join(want) in " ".join(argv)
    assert fx.main(["S9", *line[1:]]) == 0
    assert "--switch-part drift" in " ".join(got["S9"][0]) and "--part" not in got["S9"][0]
    assert "--slots 18" in " ".join(got["S9"][0])
    # the check on S1, into a folder the fetch never reads
    monkeypatch.setattr(fx.subprocess, "run", lambda cmd: got.setdefault("check", cmd) and
                        type("R", (), {"returncode": 0}))
    assert fx.main(line + ["--switch-check", "5"]) == 0
    assert "--switch-check 5" in " ".join(got["check"]) and got["check"][-1].endswith(
        "switchcheck")
    # --switch is plan S's alone, and plan S takes it
    with pytest.raises(SystemExit):
        fx.main(["U35", *line[1:]])
    assert "--switch is for plan S's boxes alone" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        fx.main(line[:5 + 2])
    assert "give --switch with plan S's inputs file" in capsys.readouterr().err
    # its size: 32.4, the chats at the apps' 16,384 tokens — eight boxes of
    # 250 (16 slots, about 4.2 h each), the drift on S9
    assert [x.split(" · ")[0] for x in fx.listing("S") if x.startswith("S")][:2] == ["S1", "S2"]
    assert "16 slots of 16,384 tokens" in "\n".join(fx.listing("S"))
    assert fx.listing("S")[-1] == "9 boxes a build · 38.3 box-hours a build · the longest 5.0 h"


def test_the_fetch_imports_the_experiments_bundles_with_no_register():
    assert ff.experiment(SERVED) and not ff.experiment("served/step2388-k3k6-2611")
    assert "switch-served-*.tar.gz" in ff.LISTING
    name = rb.bundle_name(sw.SUITE, SERVED, True, (3, 10), parts=["chats"])
    assert ir.model_of_name(name) == SERVED


def test_the_runbook_gives_plan_s_and_its_check():
    doc = DOC.read_text(encoding="utf-8")
    sec = doc[doc.index("# S. Switching mode mid-chat (32)"):]
    for want in ("switch_box.py inputs", "--switch-check 5", "S9", "--cache-ram 0",
                 "--reasoning-preserve", "--ctx-checkpoints 4", "Not tested: chats saved to disk",
                 "16,384", "10,240", "stress test", "never its thinking", "Roohi",
                 "keep_cache_across_paths", "llama-server-k3k6r-917f3d9.tar.gz",
                 "Part 1's protocol: greedy, no penalties", "Switching mode mid-chat"):
        assert want in sec, want


def test_the_bundle_never_holds_a_secret_or_a_binary(sb):
    assert run_s(sb, "s3", "--switch-part", "chats", "--part", "1/2") == 0
    path = bundle_at(sb, "s3", "chats", (1, 2))
    with tarfile.open(path) as tar:
        names = tar.getnames()
    assert sorted(names) == ["bundle.json", "results/switch/chats.jsonl", "run.log",
                             "setup.json"]
    b = rb.read(path)
    raw = b"".join(b["files"].values())
    assert os.environ.get("HF_TOKEN", "x" * 8).encode() not in raw
    st = b["setup"]
    assert st["runtime"] == RT and st["server"]["flags"][-2:] == ["--cache-ram", "0"]
    assert set(st["server"]["paths_checks"]) == {"6", "3"}
    assert st["server"]["history_check"] == {"pass": True, "empty_block": {"6": True, "3": True},
                                             "warned": False}
    assert (st["switch"]["app_context"], st["switch"]["hard_max"]) == (16_384, 10_240)
    assert st["switch"]["inputs_sha256"] == sb["inputs_sha"]
    shutil.rmtree(sb["root"] / "s3")


def test_a_server_that_writes_no_empty_block_asks_nothing(sb, monkeypatch, capsys):
    # a template that ignores --reasoning-preserve (and says so at start-up)
    # writes an earlier reply without its block: every reply, k3's too, would
    # roll the memory back — not the apps' history
    monkeypatch.setenv("FAKE_LLAMA_PRESERVE", "off")
    with pytest.raises(SystemExit) as e:
        run_s(sb, "nopreserve", "--switch-part", "chats", "--part", "1/2")
    assert "this server doesn't write the history as the apps send it" in str(e.value)
    out = capsys.readouterr().out
    assert "the history NOT as the apps': the server says --reasoning-preserve has no effect" in out
    assert not [r for r in requests(sb) if "id_slot" in r and "messages" in r]


def test_an_earlier_replys_thinking_in_the_history_stops_the_box(sb, monkeypatch, capsys):
    # 32.4: a server that leaves a reply's thinking in its content — sent
    # back, the history would carry it: the box stops at the first, no chat
    # started after it, no bundle
    monkeypatch.setenv("FAKE_LLAMA_THINK_IN_CONTENT", "1")
    assert run_s(sb, "thinkin", "--switch-part", "chats", "--part", "1/2", slots=2) != 0
    out = capsys.readouterr().out
    assert "stopped: chat 0: reply 3's content holds its thinking — the history isn't the apps'" \
        in out
    turn3 = [r for r in requests(sb) if "messages" in r and "id_slot" in r
             and r["expert_path"] == 6]
    assert len(turn3) == 2                          # chat 0's turn 3, both arms; nothing after
    assert not bundle_at(sb, "thinkin", "chats", (1, 2)).exists()
