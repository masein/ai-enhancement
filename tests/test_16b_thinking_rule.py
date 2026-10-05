"""16b: one thinking rule for the Mobile suite on every path, and for
Instruction & maths and Frontier on a served model too (masein, 5 Oct).

A model that can turn its thinking off is asked with it off — a Hugging Face
model's chat template told so, a served model's server told so in each
request (through the board's relay where lm_eval asks) — unless a thinking
run is asked for, which is a row of its own ("· thinking"). Answers made
under the other setting are never reused: thinking-on answers in a model's
own row go to its thinking row (run #180's), else beside the tree, with the
judge's verdicts on them; a served model's lm_eval answers from before, "as
its server decided", aren't known to be off and go beside the tree too. The
estimate before Start reads the pace of the setting the run asks with.

Fake GPUs, a fake server and the stand-in judge; nothing runs."""

from __future__ import annotations

import json
import urllib.request

import pytest

import mobileaibench as mab
from conftest import make_service
from fake_openai import FakeServer
from service import config, db, runner, served, thinking_sort
from test_14_1_mab_text import fake_gpu, sit

ME = "masein"
THINKS = "<think>\nLet me weigh it, step by step.\n</think>\n\nan answer of six words here"
OPENS = "Let me weigh it.\n</think>\n\nan answer"       # Qwen3.5: its template opened <think>
EMPTY = "<think>\n\n</think>\n\nan answer"               # the switch off: an empty block
SID = "served/lda-auto"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "SERVED_RETRY_S", 0.4)
    yield client, appmod
    client.__exit__(None, None, None)


@pytest.fixture
def fake():
    s = FakeServer()
    yield s
    s.close()


def register(client, fake, thinking="auto"):
    r = client.post("/api/served", json={"name": "LDA auto", "base_url": fake.base, "key": "",
                                         "how": "k4", "thinking": thinking, "by": ME})
    assert r.status_code == 200, r.text
    return r.json()["model"]


def samples(d, replies, prompt="Q"):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "samples_t_2026-10-05T10-00-00.jsonl", "w", encoding="utf-8") as fh:
        for i, r in enumerate(replies):
            fh.write(json.dumps({"doc_id": i, "arguments": [[prompt, {}]], "resps": [[r]],
                                 "filtered_resps": [r]}) + "\n")


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------

def test_the_rule_is_one_on_every_path():
    lda = {"id": SID, "thinking": "auto", "pin": {}}
    th = runner.suite_thinking({}, {}, lda)
    assert (th["mode"], th["on"], th["separate"], th["budget"]) == (
        "switch", False, False, config.GEN_MAX_GEN_TOKS)
    th = runner.suite_thinking({"thinking": True}, {}, lda)
    assert (th["on"], th["separate"], th["budget"]) == (
        True, True, config.GEN_THINKING_MAX_GEN_TOKS)
    # whatever the server was registered with: said out loud, either way
    assert runner.suite_thinking({}, {}, {**lda, "thinking": "on"})["on"] is False
    # a Hugging Face model with a switch: 12h.1's rule, off unless asked
    sw = {"archinfo": {"thinking": "switch", "think_end": "</think>"}}
    assert runner.suite_thinking({}, sw, None) == runner.gen_thinking({}, sw)
    assert runner.suite_thinking({}, sw, None)["on"] is False
    # one from OpenRouter has no template to switch: it thinks as it does
    oro = {"id": "served/openrouter-x", "via": "openrouter", "thinking": "off"}
    assert runner.suite_thinking({}, {}, oro)["mode"] != "switch"
    assert "unless a thinking run is asked for" in runner.THINKING_RULE


@pytest.mark.parametrize("reply,unmarked,want", [
    (THINKS, "off", "on"), (OPENS, "off", "on"), (EMPTY, "off", "off"),
    ("an answer", "off", "off"),
    # a served model's lm_eval answers: the server kept the thinking apart
    ("an answer", "default", "default"), ("an answer", "on", "on")])
def test_what_a_tasks_answers_were_made_with(tmp_path, reply, unmarked, want):
    d = tmp_path / "mab_hotpotqa_0shot"
    # the prompt's empty block (the switch off) is never read as thinking
    samples(d / "m", [reply], prompt="<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert runner.answered_thinking(d, unmarked) == want


def test_a_task_marked_before_it_asks_is_what_its_mark_says(tmp_path):
    d = tmp_path / "mab_sql_0shot"
    runner.mark_thinking(d, {"on": False})
    assert runner.answered_thinking(d) is None              # a mark with no answers is none
    samples(d / "m", [THINKS])
    assert runner.answered_thinking(d) == "off"
    assert json.loads((d / runner.THINKING_NAME).read_text())["rule"] == runner.THINKING_RULE
    assert runner.unmarked_of(None, True) == "off"
    assert runner.unmarked_of({"thinking": "auto"}, True) == "default"
    assert runner.unmarked_of({"thinking": "on"}, True) == "on"
    assert runner.unmarked_of({"thinking": "auto"}, False) == "off"   # served.ask keeps it


# ---------------------------------------------------------------------------
# answers under the other setting, never reused
# ---------------------------------------------------------------------------

def test_thinking_answers_go_to_the_thinking_row_with_their_verdicts_by_turn(svc):
    row = config.OUT_DIR / "org__chat-1b"
    sit(row, mab.MTB1, answer=lambda q: THINKS)
    sit(row, mab.MTB2, answer=lambda q: THINKS)
    qid = mab.load(mab.MTB1)[0]["id"]

    def v(s):
        return {"score": s, "judge": "j", "version": ""}
    mab.write_judged(row, {"verdicts": {f"mab_mtbench:{qid}:1": v(9),
                                        f"mab_mtbench:{qid}:2": v(7), "mab_adv:x:0": v(1)}})
    think = row.with_name(row.name + "__thinking")
    got = runner.sort_thinking(row / "mab_mtbench_t1_0shot", mab.MTB1, False, think,
                               {"model": "org/chat-1b · thinking", "base_model": "org/chat-1b"})
    assert got == ("thinking row", think / "mab_mtbench_t1_0shot")
    assert (think / "mab_mtbench_t1_0shot").is_dir() and not (row / "mab_mtbench_t1_0shot").exists()
    assert json.loads((think / "model_meta.json").read_text())["base_model"] == "org/chat-1b"
    # its results say it thought, so the report files them under the thinking row
    [res] = (think / "mab_mtbench_t1_0shot").rglob("results*.json")
    args = json.loads(res.read_text())["config"]["model_args"]
    assert args == "pretrained=org/chat-1b,dtype=bfloat16,enable_thinking=True"
    # turn 1's verdict goes with its answers; turn 2's and the others' stay
    assert set(mab.read_judged(think)["verdicts"]) == {f"mab_mtbench:{qid}:1"}
    assert set(mab.read_judged(row)["verdicts"]) == {f"mab_mtbench:{qid}:2", "mab_adv:x:0"}
    runner.sort_thinking(row / "mab_mtbench_t2_0shot", mab.MTB2, False, think)
    assert set(mab.read_judged(think)["verdicts"]) == {f"mab_mtbench:{qid}:1",
                                                       f"mab_mtbench:{qid}:2"}
    # each row's scores from what it holds now
    assert mab.read(think)["tasks"][mab.MTBENCH]["n"] == 160
    assert mab.MTBENCH not in ((mab.read(row) or {}).get("tasks") or {})


def test_answers_the_thinking_row_has_already_go_beside_the_tree(svc):
    row = config.OUT_DIR / "org__chat-1b"
    think = row.with_name(row.name + "__thinking")
    sit(row, mab.HOTPOT, answer=lambda q: THINKS)
    sit(think, mab.HOTPOT, answer=lambda q: THINKS)
    kind, where = runner.sort_thinking(row / "mab_hotpotqa_0shot", mab.HOTPOT, False, think)
    assert kind == "earlier" and where.parent == config.OUT_DIR.with_name("earlier") / row.name
    assert where.name.startswith("mab_hotpotqa_0shot-thinking-on-") and where.is_dir()
    # a thinking run meets thinking-off answers in its row: beside the tree too
    sit(think, mab.SQL, answer=lambda q: "plain")
    kind, where = runner.sort_thinking(think / "mab_sql_0shot", mab.SQL, True, None)
    assert kind == "earlier" and where.name.startswith("mab_sql_0shot-thinking-off-")
    # answers of the setting asked stay where they are
    sit(row, mab.HOTPOT, answer=lambda q: "plain")
    assert runner.sort_thinking(row / "mab_hotpotqa_0shot", mab.HOTPOT, False, think) is None


# ---------------------------------------------------------------------------
# a Hugging Face model's Mobile run
# ---------------------------------------------------------------------------

def test_a_hugging_face_mobile_run_asks_with_thinking_off_unless_asked(svc, monkeypatch):
    client, appmod = svc
    seen = fake_gpu(monkeypatch)
    pf = runner.preflight
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: {
        **pf(*a, **k), "archinfo": {"thinking": "switch", "think_end": "</think>"}})
    row = config.OUT_DIR / "org__chat-2b"
    sit(row, mab.HOTPOT, answer=lambda q: THINKS)           # before 16b: its template's default
    sid = client.post("/api/submissions", json={"hf_id": "org/chat-2b", "suite": "mobile",
                                                "kind": "instruct"}).json()["id"]
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    # every part asked again, with the switch off and said
    assert [c[c.index("--tasks") + 1] for c in seen] == list(mab.TASKS)
    for cmd in seen:
        margs = cmd[cmd.index("--model_args") + 1]
        assert "enable_thinking=False" in margs and "think_end_token" not in margs
        assert cmd[cmd.index("--output_path") + 1].endswith("/org__chat-2b/" + cmd[
            cmd.index("--tasks") + 1] + "_0shot")
    think = row.with_name("org__chat-2b__thinking")
    assert (think / "mab_hotpotqa_0shot").is_dir() and mab.HOTPOT in mab.read(think)["tasks"]
    assert json.loads((row / "mab_hotpotqa_0shot" / runner.THINKING_NAME).read_text()
                      )["enable_thinking"] is False
    log = next(config.LOGS_DIR.glob(f"service_{sid}_*.log")).read_text()
    assert ("mab_hotpotqa: its answers on disk were made with thinking on, and this run asks "
            "with it off — moved to its thinking row") in log
    # each row's cell is its own: the thinking answers the thinking row's
    appmod._cache.update(key=None, payload=None, at=0.0)
    data = client.get("/api/results").json()
    assert set(data["cells"][mab.HOTPOT]) == {"org/chat-2b", "org/chat-2b · thinking"}
    assert not [m["id"] for m in data["models"] if "__thinking" in m["id"]]
    # a thinking run is the thinking row's, with the switch on and its end said
    seen.clear()
    r = client.post("/api/submissions", json={"hf_id": "org/chat-2b", "suite": "mobile",
                                              "kind": "instruct", "thinking": True})
    assert r.status_code == 200, r.text
    runner.run_submission(db.get(r.json()["id"]))
    assert db.get(r.json()["id"])["status"] == "done"
    assert seen and mab.HOTPOT not in [c[c.index("--tasks") + 1] for c in seen]   # reused
    for cmd in seen:
        margs = cmd[cmd.index("--model_args") + 1]
        assert "enable_thinking=True" in margs and "think_end_token=</think>" in margs
        assert "/org__chat-2b__thinking/" in cmd[cmd.index("--output_path") + 1]


# ---------------------------------------------------------------------------
# a served model
# ---------------------------------------------------------------------------

def test_a_served_mobile_run_says_the_switch_and_run_180s_answers_are_the_thinking_rows(
        svc, fake, monkeypatch):
    client, appmod = svc
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)
    rec = register(client, fake)
    row = config.OUT_DIR / rec["id"].replace("/", "__")
    sit(row, mab.MTB1, answer=lambda q: THINKS)             # run #180: thinking on, unlabelled
    sid = db.add(rec["id"], "instruct", "mobile", ME, "", part="judged")
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    assert len(fake.requests) == 160
    assert all(b["chat_template_kwargs"] == {"enable_thinking": False} for b in fake.requests)
    think = row.with_name(row.name + "__thinking")
    assert (think / "mab_mtbench_t1_0shot").is_dir()
    assert json.loads((think / "model_meta.json").read_text())["base_model"] == rec["id"]
    # its pace is kept by the setting it was asked with
    by = served.get(rec["id"])["speed_by"]
    assert set(by) == {"off"} and by["off"]["n"] >= 80
    # a thinking run: the switch on, into the thinking row
    fake.requests.clear()
    fake.reasoning = "Let me think."
    sid = db.add(rec["id"], "instruct", "mobile", ME, "", part="judged", thinking=True)
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    assert fake.requests and all(b["chat_template_kwargs"] == {"enable_thinking": True}
                                 for b in fake.requests)
    assert (think / "mab_mtbench_t2_0shot").is_dir()
    assert set(served.get(rec["id"])["speed_by"]) == {"off", "on"}
    # a thinking run's results say so; run #180's, moved, were made to say so
    for t in ("mab_mtbench_t1_0shot", "mab_mtbench_t2_0shot"):
        for res in (think / t).rglob("results*.json"):
            assert json.loads(res.read_text())["config"]["model_args"].endswith(
                ",enable_thinking=True"), res
    for res in (row / "mab_mtbench_t2_0shot").rglob("results*.json"):
        assert "enable_thinking" not in json.loads(res.read_text())["config"]["model_args"]
    appmod._cache.update(key=None, payload=None, at=0.0)
    ids = {m["id"] for m in client.get("/api/results").json()["models"]}
    assert {rec["id"], rec["id"] + " · thinking"} <= ids
    assert not [i for i in ids if "__thinking" in i]


def test_the_estimate_before_start_reads_the_pace_of_the_setting_asked(svc, fake):
    client, _ = svc
    rec = register(client, fake)
    served.record_speed(rec["id"], 4.0, 80, "off")
    got = client.get("/api/mobileaibench/estimate", params={"model": rec["id"]}).json()
    assert got["thinking"] is False and got["rule"] == runner.THINKING_RULE
    assert got["parts"]["none"]["line"] == ("5,000 answers, about 5.6 h, at its measured 4.0 s "
                                            "an answer with thinking off")
    # thinking on, never measured: said, never the other setting's pace
    got = client.get("/api/mobileaibench/estimate", params={"model": rec["id"],
                                                            "thinking": True}).json()
    line = got["parts"]["none"]["line"]
    assert line.endswith("— with thinking on not measured yet; with thinking off it took 4.0 s "
                         "an answer") and "5.6 h" not in line
    served.record_speed(rec["id"], 40.0, 80, "on")
    got = client.get("/api/mobileaibench/estimate", params={"model": rec["id"],
                                                            "thinking": True}).json()
    assert got["parts"]["none"]["line"].startswith("5,000 answers, about 55.6 h, at its measured 40.0 s")


def test_the_relay_says_the_switch_lm_eval_cant(fake):
    rec = {"id": SID, "name": "LDA", "base_url": fake.base, "key": "", "thinking": "auto",
           "pin": {"model": "m"}}
    with served.Relay(rec, extra={"chat_template_kwargs": {"enable_thinking": False}}) as r:
        req = urllib.request.Request(
            r.base + "/chat/completions", headers={"Content-Type": "application/json"},
            data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
            .encode())
        got = json.loads(urllib.request.urlopen(req, timeout=10).read())
    assert got["choices"][0]["message"]["content"]
    assert fake.requests[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    # lm_eval's cache is kept by setting: an answer of the other is never read back
    a, b = (served.cache_path({**rec, "id": SID}, "ifeval", x) for x in (False, True))
    assert a != b and a.name == "ifeval-thinking-off" and b.name == "ifeval-thinking-on"


def test_thinking_is_for_the_mobile_suite_too(svc):
    client, _ = svc
    r = client.post("/api/submissions", json={"hf_id": "org/chat-1b", "suite": "mobile",
                                              "kind": "instruct", "thinking": True})
    assert r.status_code == 200, r.text
    r = client.post("/api/submissions", json={"hf_id": "org/chat-1b", "suite": "everyday",
                                              "kind": "instruct", "thinking": True})
    assert r.status_code == 422 and "the Mobile suite" in r.json()["detail"]


# ---------------------------------------------------------------------------
# the deploy step: what is on disk already
# ---------------------------------------------------------------------------

def test_the_deploy_step_sorts_what_is_on_disk_after_a_dry_run(svc, fake, capsys,
                                                                 monkeypatch):
    client, _ = svc
    rec = register(client, fake)
    srow = config.OUT_DIR / rec["id"].replace("/", "__")
    sit(srow, mab.MTB1, answer=lambda q: THINKS)            # thinking on: the thinking row's
    sit(srow, mab.HOTPOT, answer=lambda q: "plain")         # off already: kept
    sit(srow, "ifeval", docs=[{"id": "1", "prompt": "p"}])  # lm_eval's, as the server decided
    hrow = config.OUT_DIR / "org__switch-2b"                # a Hugging Face model with a switch
    sit(hrow, mab.SQL, answer=lambda q: THINKS)
    (hrow / "model_meta.json").write_text(json.dumps({"model": "org/switch-2b",
                                                      "thinking": "switch"}))
    nrow = config.OUT_DIR / "org__always-2b"                # one that can't turn it off
    sit(nrow, mab.SQL, answer=lambda q: THINKS)
    (nrow / "model_meta.json").write_text(json.dumps({"model": "org/always-2b",
                                                      "thinking": "always"}))
    assert thinking_sort.main([]) == 0
    out = capsys.readouterr().out
    assert f"{rec['id']} · mab_mtbench_t1: thinking on → its thinking row" in out
    assert (f"{rec['id']} · ifeval: as its server decided → earlier (not known to be thinking "
            "off)") in out
    assert "org/switch-2b · mab_sql: thinking on → its thinking row" in out
    assert "always-2b" not in out and "mab_hotpotqa" not in out
    assert "3 task(s) to sort. A dry run: nothing moved." in out
    assert (srow / "mab_mtbench_t1_0shot").is_dir()         # nothing moved
    # not while a run is going
    recent = db.recent
    monkeypatch.setattr(db, "recent", lambda n=100: [{"id": 181, "status": "running"}])
    assert thinking_sort.main(["--apply"]) == 2
    assert "Not now: run #181 is going." in capsys.readouterr().err
    assert (srow / "mab_mtbench_t1_0shot").is_dir()
    monkeypatch.setattr(db, "recent", recent)
    assert thinking_sort.main(["--apply"]) == 0
    assert (srow.with_name(srow.name + "__thinking") / "mab_mtbench_t1_0shot").is_dir()
    assert (hrow.with_name(hrow.name + "__thinking") / "mab_sql_0shot").is_dir()
    assert not (srow / "ifeval_0shot").exists() and (srow / "mab_hotpotqa_0shot").is_dir()
    earlier = config.OUT_DIR.with_name("earlier") / srow.name
    assert [p.name.split("-thinking-")[1].split("-")[0] for p in earlier.iterdir()] == ["default"]
    assert (nrow / "mab_sql_0shot").is_dir()
    capsys.readouterr()
    assert thinking_sort.main([]) == 0
    assert capsys.readouterr().out.startswith("Nothing to sort")
