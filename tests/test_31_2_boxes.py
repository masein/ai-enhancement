"""31.2: step2388's boxes beyond the whitepaper's set — DeviceMark on both
expert paths asked on a box's own server (greedy, no penalties; and the k3
path at the file's own settings), the easy set (MMLU and GSM8K, zero-shot
chat) beside the original Qwen3.6 Q2 file, the settings check's default path
and neutral penalties, plan U's U38–U40 with its files pinned, and HLE's judge
through OpenRouter's Batch API. The server is the stand-in
(tests/fixtures/fake_llama_server.py), the battery's items and the questions
invented, OpenRouter a stand-in: nothing is fetched, sent or run."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import devicemark as dm
import devicemark_box as dmb
import frontier as fb
import frontier_box as fx
import frontier_graders as fg
import import_frontier as imf
import import_remote as ir
import path_check
import remote_bundle as rb
import report_lm_eval as report
from service import ai_models, config, db, llm
from service import devicemark as sdm
from service import frontier_grade as fgr
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import box, free_port, requests  # noqa: F401
from test_20_step4000 import card  # noqa: F401
from test_31_1_k3k6 import NAME, RT, SHA, STEP, k3k6, run_k, step_kept  # noqa: F401

FILE_ID = STEP + "-file"
Q2 = "served/Qwen3.6-35B-A3B-Q2-k3k6-2611"
TARBALL = "llama-server-k3k6-3ad38b7.tar.gz"
TB_SHA = "65dd1b90f19be00e918e558b3dbff9fc73e8ff325716ae8482a8211d9f05eb41"
Q2_SHA = "ed7cda7e38985b4fcff76475865135039641d2bfbac3c169df15ca770f37fb0c"
REV = "fe12fa27825b3a0cc6cde3dedec3e838bd3e6de4"
UNSLOTH = "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d"
BATCH = "openai/o3-mini:batch"


# ---------------------------------------------------------------------------
# the easy set
# ---------------------------------------------------------------------------

def test_the_easy_set_is_beside_the_frontier_suite_never_in_it():
    assert fb.EASY == ["mmlu_chat", "gsm8k_chat"] and fb.ALL == fb.TASKS + fb.EASY
    assert not set(fb.EASY) & set(fb.TASKS)
    assert not set(fb.EASY) & set(report.frontier_bench_meta())         # no Frontier column
    m, g = fb.BENCH["mmlu_chat"], fb.BENCH["gsm8k_chat"]
    assert (m["n"], m["epochs"], m["source"]["hf"], m["source"]["licence"]) == (
        14042, 1, "cais/mmlu", "MIT")
    assert (g["n"], g["epochs"], g["source"]["hf"], g["source"]["licence"]) == (
        1319, 1, "openai/gsm8k", "MIT")
    # asked of step2388's k3 path and of the original model, thinking off alone
    assert set(fb.EASY) <= set(fb.RUNTIMES[RT]["off_for"])
    assert fb.RUNTIMES["card-qwen-k3k6"]["off_for"] == tuple(fb.EASY)


def test_mmlu_is_asked_as_a_lettered_chat_question_and_its_letter_read():
    it = {"id": "0", "question": "Which is a prime?", "options": ["4", "6", "7", "9"],
          "answer": "C", "subject": "x"}
    text, need = fb.prompt("mmlu_chat", it)
    assert text.startswith("Answer the following multiple choice question.")
    assert "A) 4\nB) 6\nC) 7\nD) 9" in text and need == {"key": "C"}     # the dataset's order
    assert fb.score("mmlu_chat", "Seven is prime.\nANSWER: C", "stop", need)["ok"] is True
    assert fb.score("mmlu_chat", "ANSWER: A", "stop", need)["ok"] is False


@pytest.mark.parametrize("reply, read, ok", [
    ("She pays 3 + 4 = 7 dollars.\nThe answer is 1,234", "1234", True),
    ("So the total is $1,234.00", "1234", True),
    ("<think>maybe 99</think>It is 1234 apples.", "1234", True),
    ("It is 1235.", "1235", False),
    ("No number at all.", None, False),
])
def test_gsm8k_reads_the_last_number(reply, read, ok):
    it = {"id": "0", "question": "How much?", "answer": "1234"}
    text, need = fb.prompt("gsm8k_chat", it)
    assert text.endswith("number on the last line.\n\nHow much?") and need == {"answer": "1234"}
    got = fb.score("gsm8k_chat", reply, "stop", need)
    assert (got["read"], got["ok"]) == (read, ok)


def test_the_q2_runtime_says_its_card_settings_and_never_thinks():
    rt = fb.RUNTIMES["card-qwen-k3k6"]
    assert rt["env"] == fb.RUNTIMES[RT]["env"]                   # the router norm set: no effect
    assert rt["build"] == fb.RUNTIMES[RT]["build"] and rt["flags"] == fb.RUNTIMES[RT]["flags"]
    assert rt["sampling"] == {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0,
                              "presence_penalty": 1.5, "repeat_penalty": 1.0}
    assert fb.runtime_of_model(Q2) == "card-qwen-k3k6"
    assert fb.runtime_thinking_refused("card-qwen-k3k6", "off", fb.EASY) == ""
    assert fb.runtime_thinking_refused("card-qwen-k3k6", "on", fb.EASY)
    assert fb.runtime_thinking_refused("card-qwen-k3k6", "off", ["gpqa_diamond_epoch"])
    assert fb.runtime_model_refused(RT, Q2) and fb.runtime_model_refused("card-qwen-k3k6", STEP)


# ---------------------------------------------------------------------------
# DeviceMark's requests
# ---------------------------------------------------------------------------

def test_devicemark_on_a_path_is_plain_argmax_or_the_files_own():
    on_path = {"id": STEP, "runtime": RT}
    assert sdm.settings(False, rec=on_path) == {"max_tokens": 4096, "temperature": 0.0,
                                                "presence_penalty": 0.0, "seed": 0,
                                                "expert_path": 3}
    assert sdm.settings(True, rec=on_path) == {"max_tokens": 4096, "temperature": 0.0,
                                               "presence_penalty": 0.0, "seed": 0,
                                               "expert_path": 6}
    assert sdm.settings(False, rec={"id": FILE_ID, "runtime": RT}) == {
        "max_tokens": 4096, "seed": 0, "expert_path": 3}
    # a model on no runtime of paths: as before, the thinking switch said
    assert sdm.settings(True, rec={"id": "served/x", "runtime": "lda-2610"}) == {
        "max_tokens": 4096, "temperature": 0.0, "seed": 0,
        "chat_template_kwargs": {"enable_thinking": True}}
    assert sdm.PROTOCOL_WORDS[sdm.sampling_of(sdm.settings(True, rec=on_path))] == \
        "DeviceMark's protocol: greedy, no penalties"
    assert sdm.sampling_of(sdm.settings(False, rec={"id": FILE_ID, "runtime": RT})) == "file"


# ---------------------------------------------------------------------------
# DeviceMark on a box, and its import
# ---------------------------------------------------------------------------

def dm_requests(b: dict) -> list[dict]:
    """the battery's requests the server answered (the settings check's and
    the card check's left out)"""
    texts = {it["text"] for it in dm.load_items(config.DM_ITEMS).values()}
    return [r for r in requests(b) if (r.get("messages") or [{}])[-1].get("content") in texts]


def dm_bundle(b: dict, out: str, model: str = STEP, thinking: bool = False) -> Path:
    return b["root"] / out / rb.bundle_name(dmb.SUITE, model, thinking)


@pytest.fixture
def dmbox(k3k6, monkeypatch):  # noqa: F811
    """a k3k6 box whose DeviceMark items are the server's (invented)"""
    monkeypatch.setenv("DM_ITEMS", str(config.DM_ITEMS))
    return k3k6


def test_a_k3_box_asks_devicemark_greedy_and_the_import_marks_it(dmbox):
    assert run_k(dmbox, "dm3", "--devicemark", thinking="off", only=False) == 0
    asked = dm_requests(dmbox)
    assert len(asked) == len(dm.keys_for("full")) == 596
    # plain argmax on the path: temperature 0, presence 0, seed 0, and nothing more
    # (beside its messages, its cap and the reply not streamed)
    assert all(sorted(r) == ["expert_path", "max_tokens", "messages", "presence_penalty",
                             "seed", "stream", "temperature"] for r in asked)
    assert {(r["expert_path"], r["temperature"], r["presence_penalty"], r["seed"],
             r["max_tokens"]) for r in asked} == {(3, 0.0, 0.0, 0, 4096)}
    out = dmbox["root"] / "dm3"
    assert json.loads((out / "settings-check-k3.json").read_text())["pass"]
    prog = json.loads((out / "progress.json").read_text())
    assert prog["state"] == "whole" and prog["tasks"] == ["devicemark"]
    b = rb.read(dm_bundle(dmbox, "dm3"))
    assert b["bundle"]["suite"] == "devicemark-served" and b["bundle"]["tasks"] == {
        "devicemark": 596}
    d = b["setup"]["devicemark"]
    assert d["request"]["expert_path"] == 3 and d["sampling"] == "greedy"
    assert d["protocol_words"] == "DeviceMark's protocol: greedy, no penalties"
    assert b["setup"]["server"]["paths_check"]["pass"]
    # the import: registered, marked here, a Runs entry, its row where the board keeps it
    assert ir.main([str(dm_bundle(dmbox, "dm3")), "--by", "masein", "--register",
                    "step2388 (k3/k6)", "--file-sha256", dmbox["sha"]]) == 0
    row = config.OUT_DIR / STEP.replace("/", "__")
    got = json.loads((row / dm.OUT_NAME).read_text())
    assert got["setup"]["protocol_words"] == "DeviceMark's protocol: greedy, no penalties"
    assert got["setup"]["request"] == {"temperature": 0.0, "presence_penalty": 0.0, "seed": 0,
                                       "expert_path": 3}
    assert got["setup"]["where"].startswith("run on a rented GPU")
    assert len((row / dm.ITEMS_NAME).read_text().splitlines()) == 596
    runs = [r for r in db.recent() if r["suite"] == "devicemark" and r["hf_id"] == STEP
            and "imported" in (r.get("note") or "")]
    assert runs and runs[0]["status"] == "done"
    # the same bundle again changes nothing
    assert ir.main([str(dm_bundle(dmbox, "dm3")), "--by", "masein"]) == 0


def test_the_k3_path_at_the_files_own_settings_is_a_row_of_its_own(dmbox):
    assert run_k(dmbox, "dmf", "--devicemark", thinking="off", only=False,
                 served_as=FILE_ID) == 0
    asked = dm_requests(dmbox)
    assert len(asked) == 596
    assert all(sorted(r) == ["expert_path", "max_tokens", "messages", "seed", "stream"]
               and r["expert_path"] == 3 for r in asked)
    path = dm_bundle(dmbox, "dmf", FILE_ID)
    assert rb.read(path)["setup"]["devicemark"]["sampling"] == "file"
    assert ir.main([str(path), "--by", "masein", "--register", "step2388 (k3, the file's own)",
                    "--file-sha256", dmbox["sha"]]) == 0
    got = json.loads((config.OUT_DIR / FILE_ID.replace("/", "__") / dm.OUT_NAME).read_text())
    assert got["setup"]["protocol_words"] == ("the file's own settings for the path: no "
                                              "sampling field sent")


def test_the_import_refuses_a_devicemark_bundle_asked_another_way(dmbox):
    assert run_k(dmbox, "dmx", "--devicemark", thinking="off", only=False) == 0
    b = rb.read(dm_bundle(dmbox, "dmx"))
    rec = imf._would_be(b, dmbox["sha"])
    battery = dm.battery_hashes(dm.load_items(config.DM_ITEMS))
    assert dmb.checks(b, rec, dmbox["sha"], battery) == []

    def refused(change) -> list[str]:
        c = json.loads(json.dumps({k: v for k, v in b.items() if k != "files"}))
        c["files"] = dict(b["files"])
        change(c)
        return dmb.checks(c, rec, dmbox["sha"], battery)
    # a request that sent something else: the path's own penalty left on
    got = refused(lambda c: c["setup"]["devicemark"]["request"].pop("presence_penalty"))
    assert any(x.startswith("each request sent") for x in got)
    got = refused(lambda c: c["setup"]["devicemark"]["battery"].update(items="0" * 64))
    assert any("its battery's items (each question, key and prompt) aren't the board's" in x
               for x in got)
    row = b["bundle"]["row"]
    name = f"results/{row}/{sdm.ANSWERS_NAME}"

    def short(c):
        lines = c["files"][name].decode().splitlines()
        c["files"][name] = ("\n".join(lines[:-3]) + "\n").encode()
    assert any("593 of the battery's 596 items answered" in x for x in refused(short))
    got = refused(lambda c: c["setup"]["server"]["paths_check"].update(
        **{"pass": False, "mismatches": ["x"]}))
    assert any("a settings check that passed" in x for x in got)
    got = refused(lambda c: c["setup"].update(gguf={**c["setup"]["gguf"], "sha256": "1" * 64}))
    assert any("its GGUF isn't the file registered" in x for x in got)


# ---------------------------------------------------------------------------
# the settings check: the default path, and no penalty beside a path's own
# ---------------------------------------------------------------------------

def test_the_check_holds_the_default_path_and_the_neutral_penalties(dmbox, monkeypatch):
    assert run_k(dmbox, "c", "--check", thinking="off", only=False) == 0
    chk = json.loads((dmbox["root"] / "c" / "settings-check-k3.json").read_text())
    assert chk["server"]["default"] == 3
    assert chk["server"]["neutral"]["3"] == chk["server"]["neutral"]["6"] == {
        "repeat_penalty": 1.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0}
    monkeypatch.setenv("FAKE_LLAMA_PATHS_BAD", "0.repeat_penalty=1.1")
    with pytest.raises(SystemExit, match=re.escape(
            "the server's k3 (/props): repeat penalty 1.1, not 1.0")):
        run_k(dmbox, "c2", "--check", thinking="off", only=False)


def test_the_default_path_is_read_as_the_server_names_it():
    assert path_check.default_of({"default_generation_settings": {"params": {
        "expert_path": 3}}}) == 3
    assert path_check.default_of({"default_generation_settings": {"params": {
        "expert_path": "k3"}}}) == 3
    assert path_check.default_of({"default_generation_settings": {"n_ctx": 1}}) is None
    assert path_check.neutral({"repeat_penalty": 1.0, "frequency_penalty": 0.0}, "x") == [
        "x gives no dry multiplier"]


# ---------------------------------------------------------------------------
# plan U: U38–U40, and its files pinned
# ---------------------------------------------------------------------------

def line(box_: str, model: str = STEP, gguf: str = "", server: str = ""):
    return fx.parser().parse_args([
        box_, "--as", model, "--gguf", gguf or f"hf://me/p@{REV}/{NAME}",
        "--server", server or f"hf://me/p@{REV}/{TARBALL}", "--price", "0.45"])


def test_u38_to_u40_ask_devicemark_on_both_paths_and_the_easy_set():
    plan = fx.PLANS["U"]
    assert plan["U38"] == [("off", (fx.DM,), "", 8), ("off", (fx.DM_FILE,), "", 8),
                           ("off", tuple(fb.EASY), "", 8)]
    assert plan["U39"][0] == ("on", (fx.DM,), "", 8)     # then the showcase (31.2b)
    assert plan["U40"] == [("off", tuple(fb.EASY), "", 8)]
    a = line("U38")
    greedy, own, easy = (fx.argv_of("U38", k, s, a) for k, s in enumerate(plan["U38"], 1))
    assert "--devicemark" in greedy and greedy[greedy.index("--as") + 1] == STEP
    assert "--only" not in greedy and greedy[greedy.index("--thinking") + 1] == "off"
    assert "--devicemark" in own and own[own.index("--as") + 1] == FILE_ID
    assert [easy[i + 1] for i, x in enumerate(easy) if x == "--only"] == fb.EASY
    k6 = fx.argv_of("U39", 1, plan["U39"][0], a)
    assert "--devicemark" in k6 and k6[k6.index("--thinking") + 1] == "on"
    assert fx.pins_of(a, "U38") == [f"{TARBALL}={TB_SHA}", f"{NAME}={SHA}"]
    lines = fx.listing("U")
    assert any(ln.startswith("U38 · about 5.1 h") for ln in lines)
    assert "DeviceMark (greedy, no penalties)" in "\n".join(lines)


def test_u40_runs_the_q2_file_alone_and_the_others_never_do():
    assert fx.model_refused("U40", Q2) == "" and fx.model_refused("U38", STEP) == ""
    assert "U40 runs served/Qwen3.6-35B-A3B-Q2-k3k6-2611" in fx.model_refused("U40", STEP)
    assert "runs on U40 alone, not on U1" in fx.model_refused("U1", Q2)
    a = line("U40", Q2, gguf=f"hf://unsloth/Qwen3.6-35B-A3B-MTP-GGUF@{UNSLOTH}/"
                             "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf")
    assert fx.runtime_of(a, "U40") == "card-qwen-k3k6"
    assert fx.pins_of(a, "U40") == [f"Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf={Q2_SHA}",
                                    f"{TARBALL}={TB_SHA}"]
    assert fx.revision_refused(a, "U40") == ""


def test_plan_us_files_are_fetched_at_their_commits(capsys):
    assert fx.revision_refused(line("U1"), "U1") == ""
    a = line("U1", gguf=f"hf://me/p/{NAME}")
    assert fx.revision_refused(a, "U1") == (
        f"U1: {NAME} at no commit, where plan U fetches it at {REV} — give the line @<commit> "
        "as the runbook does. Nothing was fetched")
    a = line("U1", server=f"hf://me/p@main/{TARBALL}")
    assert f"{TARBALL} at main, where plan U fetches it at {REV}" in fx.revision_refused(a, "U1")
    # a file on the box itself is held to its sha256 alone
    assert fx.revision_refused(line("U1", gguf=f"/root/{NAME}"), "U1") == ""
    with pytest.raises(SystemExit, match="at no commit"):
        fx.main(["U1", "--as", STEP, "--gguf", f"hf://me/p/{NAME}", "--server",
                 f"hf://me/p@{REV}/{TARBALL}"])


def test_the_runbook_gives_u0_to_u2_with_the_real_files():
    text = (Path(__file__).resolve().parent.parent / "docs" / "REMOTE-RUNS.md").read_text()
    part = text[text.index("# step2388 on plan U (31.1, 31.2)"):]
    part = part[:part.index("# Speed tests")]
    assert TB_SHA in part and SHA in part and Q2_SHA in part and REV in part
    assert part.index("## U0.") < part.index("## U1.") < part.index("## U2.") < part.index(
        "## U3.")
    assert "--unified ~/step2388.k3k6.gguf --ngl 99" in part
    assert '("general.sampling.", "teraformer.qad.")' in part
    assert f"--pin {TARBALL}={TB_SHA}" in part
    assert f"python scripts/frontier_box.py U40 --as {Q2}" in part
    assert "openai/o3-mini:batch" in part


# ---------------------------------------------------------------------------
# HLE's judge through OpenRouter's Batch API
# ---------------------------------------------------------------------------

class FakeOpenRouter:
    """OpenRouter's batches, as its quickstart gives them: POST answers 202
    with an id; GET says validating, in_progress, then the end it is set to"""

    def __init__(self):
        self.posts, self.gets, self.state, self.results = [], 0, "in_progress", None
        self.fail_post: tuple[int | None, str] | None = None
        self.err = None

    def http(self, method, url, headers, body=None, timeout=60.0):
        if method == "POST":
            if self.fail_post:
                st, words = self.fail_post
                raise llm.LLMError(f"POST {url}: HTTP {st}: {words}" if st else
                                   f"POST {url}: no response within 30s", status=st)
            self.posts.append(json.loads(body))
            return 202, json.dumps({"id": f"batch_{len(self.posts)}", "status": "validating",
                                    "request_counts": {"total": len(self.posts[-1]["requests"]),
                                                       "completed": 0, "failed": 0}}).encode()
        self.gets += 1
        return 200, json.dumps({"id": url.rsplit("/", 1)[-1], "status": self.state,
                                "request_counts": {"total": 2, "completed": 1, "failed": 0},
                                "results": self.results, "error": self.err,
                                "usage": {"cost": 0.01}}).encode()


def reply(cid: str, text: str = '{"correct": "yes"}', cost: float = 0.002) -> dict:
    return {"id": "r", "custom_id": cid, "error": None, "response": {
        "status_code": 200, "request_id": "q", "body": {
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 800, "cost": cost},
            "provider": "OpenAI"}}}


PIN = {"kind": "openrouter", "id": BATCH, "version": "openai/o3-mini-2025-01-31",
       "name": "o3-mini", "provider": "openai", "provider_name": "OpenAI",
       "precision": "unknown", "price_in": 0.55, "price_out": 2.2, "batch": True}


@pytest.fixture
def orb(tmp_path, monkeypatch):
    fake = FakeOpenRouter()
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    monkeypatch.setattr(llm, "_http", fake.http)
    monkeypatch.setattr(ai_models, "over_limit", lambda: "")
    monkeypatch.setattr(ai_models, "drifted", lambda pin: "")
    spent = []
    monkeypatch.setattr(db, "spend_add", lambda *a: spent.append(a))
    # the worker runs where the test says (settle), never in a thread of its own
    monkeypatch.setattr(llm.OpenRouterBatchAPI, "_ensure_worker", lambda self, bid: False)
    be = llm.OpenRouterBatchAPI(BATCH, "test-key", tmp_path, pin=dict(PIN), role="grader")
    be.POLL_S = 0.0
    fake.spent = spent
    return fake, be


def settle(be, bid: str) -> None:
    """the poller's tick: its worker, run here to its end"""
    import os
    fd = os.open(be.dir / bid / "worker.lock", os.O_CREAT | os.O_RDWR, 0o644)
    be._work(bid, fd)


def reqs() -> list[llm.Request]:
    return [llm.Request(f"frgr:{k}", "", f"Judge answer {k}.", max_tokens=4096,
                        reasoning={"effort": "medium"}) for k in ("a", "b")]


def test_a_batch_goes_to_openrouter_whole_once_and_its_replies_come_back(orb):
    fake, be = orb
    bid = be.submit(reqs())
    assert bid.startswith("orb_") and llm.batch_dir(bid) == be.dir / bid
    settle(be, bid)
    assert len(fake.posts) == 1
    post = fake.posts[0]
    # the batch's fields before its requests, the model without its suffix, the
    # pinned provider alone with no fallback; each body no model and no provider
    assert list(post) == ["endpoint", "model", "provider", "completion_window", "requests"]
    assert (post["endpoint"], post["model"], post["completion_window"]) == (
        "/v1/chat/completions", "openai/o3-mini", "24h")
    assert post["provider"] == {"only": ["openai"], "allow_fallbacks": False}
    assert [r["custom_id"] for r in post["requests"]] == ["frgr:a", "frgr:b"]
    body = post["requests"][0]["body"]
    assert "model" not in body and "provider" not in body
    assert body["reasoning"] == {"effort": "medium"} and body["max_tokens"] == 4096
    # out at OpenRouter: pending, said, never sent again
    state, detail = be.status(bid)
    settle(be, bid)
    assert state == "pending" and "OpenRouter's batch" in detail and len(fake.posts) == 1
    assert be.busy(bid) and be.cancel(bid, ["frgr:a"], "moved") == 0
    # done: each reply recorded with its dollars, counted against the month
    fake.state, fake.results = "completed", [reply("frgr:b"), reply("frgr:a")]
    settle(be, bid)
    state, _ = be.status(bid)
    assert state == "done" and not be.out(bid)
    got = be.fetch(bid)
    assert got["frgr:a"].text == '{"correct": "yes"}' and got["frgr:a"].finish == "stop"
    assert [(s[0], s[1], s[3], s[4], s[5]) for s in fake.spent] == [
        ("grader", BATCH, 1000, 800, 0.002)] * 2
    assert (be.dir / bid / "remote-batch_1.json").exists()


def test_a_batch_that_ends_short_holds_and_sends_the_rest_again(orb):
    fake, be = orb
    bid = be.submit(reqs())
    settle(be, bid)
    fake.state, fake.results = "expired", [reply("frgr:a")]
    settle(be, bid)
    state, detail = be.status(bid)
    assert state == "pending" and be.halted(bid).startswith(
        "waiting: OpenRouter's batch expired with 1 of its 2 requests unanswered")
    settle(be, bid)
    assert len(fake.posts) == 1                     # held: nothing sent until its press
    be.resume(bid)
    settle(be, bid)
    assert len(fake.posts) == 2 and [r["custom_id"] for r in fake.posts[1]["requests"]] == [
        "frgr:b"]


@pytest.mark.parametrize("fail, holds", [
    ((None, ""), "it may have reached OpenRouter: look at its Batches tab"),
    ((502, "bad gateway"), "it may have reached OpenRouter: look at its Batches tab"),
    ((402, '{"error": {"message": "Insufficient credits"}}'), "Nothing was sent"),
    ((429, "too many batches"), ""),
])
def test_a_send_that_failed_is_never_sent_twice_blind(orb, fail, holds):
    fake, be = orb
    fake.fail_post = fail
    bid = be.submit(reqs())
    settle(be, bid)
    assert holds in be.halted(bid) if holds else be.halted(bid) == ""
    fake.fail_post = None
    settle(be, bid)
    assert len(fake.posts) == (0 if holds else 1)       # 429: asked again at the next tick


def test_nothing_is_sent_while_the_grading_is_stopped_but_a_batch_out_lands(orb, monkeypatch,
                                                                              tmp_path):
    fake, _ = orb
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    be = fgr._backend(dict(PIN))
    assert isinstance(be, fgr.GraderBatch) and isinstance(be, llm.OpenRouterBatchAPI)
    assert isinstance(fgr._backend({**PIN, "id": "openai/gpt-4.1", "batch": False}),
                      fgr.GraderChat)
    be.POLL_S = 0.0

    def stop() -> None:                             # masein's Stop, as fgr.stop writes it
        fgr.gdir().mkdir(parents=True, exist_ok=True)
        (fgr.gdir() / "stopped.json").write_text(json.dumps({"by": "masein"}))
    stop()
    bid = be.submit(reqs())
    settle(be, bid)
    assert fake.posts == []
    (fgr.gdir() / "stopped.json").unlink()
    settle(be, bid)
    assert len(fake.posts) == 1
    stop()
    fake.state, fake.results = "completed", [reply("frgr:a"), reply("frgr:b")]
    settle(be, bid)
    assert be.status(bid)[0] == "done"


def test_hles_judge_is_suggested_through_the_batch_api(hle_graded_in_sync):
    # the session's grading tests keep the sync id (conftest); this is the real one
    assert hle_graded_in_sync == BATCH
    assert fg.GRADERS["hle"]["owners"] == "o3-mini-2025-01-31"


def test_a_batch_model_is_pinned_without_a_probe(monkeypatch):
    monkeypatch.setattr(ai_models, "has_key", lambda: True)
    monkeypatch.setattr(ai_models, "model", lambda mid, wait=True: {
        "id": mid, "version": "openai/o3-mini-2025-01-31", "name": "OpenAI: o3 Mini",
        "price_in": 0.55, "price_out": 2.2})
    monkeypatch.setattr(ai_models, "providers", lambda mid: [
        {"name": "OpenAI", "tag": "openai", "precision": "unknown", "price_in": 0.55,
         "price_out": 2.2, "up": True}])
    monkeypatch.setattr(ai_models, "probe", lambda *a, **k: pytest.fail("no probe of a batch id"))
    got = ai_models.pin(BATCH, "grader")
    assert got["batch"] is True and got["provider"] == "openai"
    assert got["version"] == "openai/o3-mini-2025-01-31"          # step8804's grades stand
    assert llm.is_batch(got) and not llm.is_batch({"id": "openai/o3-mini"})
