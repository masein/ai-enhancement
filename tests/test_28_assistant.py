"""28: the assistant, fast (docs/prompts/phase-28-chat-assistant.md, Part 5).

A fake model returns each bank question's expected action; the board must
run it right: the page it opens (its link), the table's values, the chart,
the phones and boxes, the help text, the clarify and the refusal. Also: an
id the board doesn't know is refused for a clarify, a reply that isn't the
JSON is handled, a person not in ASSISTANT_USERS gets nothing, and nothing in
the assistant reaches a write. The world is tests/assistant_world.py's;
invented numbers, made-up testers, nothing asks a model."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import assistant_world as aw
from conftest import make_service
from service import assistant, config, telemetry

BANK = [json.loads(x) for x in assistant.BANK_FILE.read_text(encoding="utf-8").splitlines()
        if x.strip()]
SCENARIOS = ("tab", "model", "compare", "best", "number", "chart", "phones", "boxes", "explain",
             "ambiguous", "refused", "messy")
KIND = {"open_page": "open", "compare": "table", "best_on": "table", "chart": "chart",
        "phone": "table", "boxes": "list", "explain": "text", "clarify": "clarify",
        "cant": "text"}
S88, S40, Q4, Q2 = (aw.S88, "served/step4000-lda-2610", "served/Qwen3.6-35B-A3B-Q4-lda-2610",
                    "served/Qwen3.6-35B-A3B-Q2-lda-2610")


@pytest.fixture(scope="module")
def board(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("a28")
    client, appmod, _ = make_service(root, mp)
    saved = telemetry.NAMES
    aw.world()
    mp.setattr(config, "ASSISTANT_USERS", "masein, Tester-B")
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield {"client": client, "cat": assistant.catalog(), "root": root}
    aw.unwind(saved)
    client.__exit__(None, None, None)
    mp.undo()


def fake(want: dict):
    """a model that answers with the action the bank expects"""
    return lambda system, user, schema: json.dumps(want)


def ask(board, b: dict) -> dict:
    return assistant.ask(b["q"], "masein", b.get("prev"), cat=board["cat"], model=fake(b["want"]))


def item(q: str) -> dict:
    return next(b for b in BANK if b["q"] == q)


# ---------------------------------------------------------------------------
# the bank
# ---------------------------------------------------------------------------

def test_the_bank_has_twelve_scenarios_of_fifteen_or_more_in_two_halves():
    assert len(BANK) >= 250
    for sc in SCENARIOS:
        assert sum(b["scenario"] == sc for b in BANK) >= 15, sc
        # 28b: a working half to tune the prompt on, a held-out half it never is
        w = sum(b["scenario"] == sc and b["half"] == "work" for b in BANK)
        h = sum(b["scenario"] == sc and b["half"] == "held" for b in BANK)
        assert w >= 7 and h >= 7 and abs(w - h) <= 1, (sc, w, h)
    assert len({b["q"] for b in BANK}) == len(BANK)                 # no question twice
    assert all(b["want"]["action"] in assistant.ACTIONS for b in BANK)


def test_every_bank_question_runs_as_its_action(board):
    """the fake model's action is the board's: resolved to itself, run, its
    reply the action's kind — nothing refused, nothing raised"""
    bad = []
    for b in BANK:
        got = ask(board, b)
        want = assistant.resolve(b["want"], board["cat"])
        if not assistant.same(got["action"], b["want"], board["cat"]) or not assistant.same(
                want, b["want"], board["cat"]):
            bad.append((b["id"], b["q"], got["action"]))
            continue
        kind = got["reply"]["kind"]
        if kind != KIND[b["want"]["action"]]:
            bad.append((b["id"], b["q"], kind))
    assert not bad, bad[:10]


# ---------------------------------------------------------------------------
# open_page: the board's own links — 27c–27e's parts among them
# ---------------------------------------------------------------------------

LINKS = {
    "frontier tab, ours only, sorted by HLE": "#tab=models&view=frontier&only=ours&sort=hle",
    "show the Latest table with just IFEval and IFBench":
        "#tab=models&view=latest&cols=ifeval,ifbench",
    "standard models tested in the last 7 days": "#tab=models&view=standard&tested=7d",
    "frontier, models tested between 14 and 22 October":
        "#tab=models&view=frontier&tested=2026-10-14..2026-10-22",
    "search the models for ornith": "#tab=models&q=ornith",
    "open models only, thinking off": "#tab=models&only=open&thinking=off",
    "mobile devicemark chart": "#tab=models&view=mobile&group=devicemark&show=chart",
    "hide the thinking-off rows": "#tab=models&thinking=on",
    "frontier sorted by IFBench, lowest first": "#tab=models&view=frontier&sort=ifbench:asc",
    "frontier with GPQA, HLE and MMLU-Pro columns":
        "#tab=models&view=frontier&cols=gpqa,hle,mmlu-pro",
    "open the two-builds reader for GPQA, step8804 against Q4":
        "#tab=reports&report=paired&a=step8804&b=Q4%20(UD-Q4_K_XL)&bench=gpqa&side=a&q=1",
    "show me what step8804 got wrong on GPQA that Q4 got right":
        "#tab=reports&report=paired&a=step8804&b=Q4%20(UD-Q4_K_XL)&bench=gpqa&side=b&q=1",
    "step checkpoints report with the q4 baseline":
        "#tab=reports&report=step-checkpoints&base=Q4%20(UD-Q4_K_XL)",
    "step checkpoints, q2 as the baseline":
        "#tab=reports&report=step-checkpoints&base=Q2%20(UD-Q2_K_XL)",
    "open the on-device report": "#tab=reports&report=on-device",
    "the ran-out report for step4000 on HLE":
        "#tab=reports&report=ran-out-of-room&model=step4000&bench=hle",
    "show me the GPQA questions": "#tab=benchmarks&q=gpqa_diamond_epoch",
    "open runs": "#tab=runs",
    "step4000's answers on GPQA":
        "#model=served%2Fstep4000-lda-2610%20%C2%B7%20thinking&answers=gpqa_diamond_epoch",
}


@pytest.mark.parametrize("q", list(LINKS))
def test_open_page_builds_the_boards_own_link(board, q):
    got = ask(board, item(q))
    assert got["reply"]["kind"] == "open"
    assert got["reply"]["link"] == LINKS[q]
    assert got["reply"]["say"].startswith("Opened ")


def test_the_models_with_a_choice_and_the_phone_builds(board):
    got = ask(board, item("just step8804 and Q2 on the latest tab"))["reply"]["link"]
    ms = re.search(r"models=([^&]+)", got).group(1).split(",")
    assert {m.replace("%20%C2%B7%20", " · ").replace("%2F", "/") for m in ms} == {
        S88, S88 + " · thinking", Q2 + " · thinking"}
    got = ask(board, item("only phone builds"))["reply"]
    assert "the phone builds" in got["say"]
    assert "k4-LDA-phone-build" in got["link"] and "step8804" not in got["link"]


def test_when_since_a_model_was_added_is_a_range_to_today(board):
    cat = board["cat"]
    got = ask(board, item("models tested since step8804 was added"))["reply"]["link"]
    m = re.search(r"tested=(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})", got)
    assert m and m.group(2) == cat["today"]
    s88 = next(x for x in cat["models"] if x["id"] == S88)
    assert m.group(1) == assistant._date(s88["added"])


def test_the_newest_test_on_a_phone_opens_its_panel(board):
    got = ask(board, item("open the latest test on the OnePlus 15"))["reply"]
    tid = re.search(r"&test=([^&]+)$", got["link"]).group(1)
    ts, _ = telemetry.tests()
    one = [t for t in ts if t["phone"]["name"] == "OnePlus 15"]
    assert tid == max(one, key=lambda t: t["received"])["test_id"]
    assert "the newest test of" in got["say"]


# ---------------------------------------------------------------------------
# the numbers: the board's own, never the model's
# ---------------------------------------------------------------------------

def test_a_number_is_the_boards_cell(board):
    from service import app
    cells = app.results_payload()["cells"]
    got = ask(board, item("what's Q4's GPQA?"))["reply"]
    c = cells["gpqa_diamond_epoch"][Q4 + " · thinking"]
    assert got["rows"][0]["cells"]["gpqa"]["text"] == (
        f"{100 * c['v']:.1f} ± {100 * c['se']:.1f}")
    assert got["link"].startswith("#tab=models&view=frontier&cols=gpqa&models=")


def test_compare_tells_published_and_measured_apart(board):
    got = ask(board, item("edge0 35b vs step8804 on gpqa"))["reply"]
    e0, s88 = got["rows"]
    assert e0["cells"]["gpqa"]["pub"] == "Edge0" and e0["cells"]["gpqa"]["text"] == "79.8"
    assert "pub" not in s88["cells"]["gpqa"] and "±" in s88["cells"]["gpqa"]["text"]
    assert "never ranked with ours" in got["say"]


def test_keeps_runs_out_and_loops(board):
    got = ask(board, item("how much of BF16 does Q2 keep on MMLU-Pro?"))["reply"]
    assert got["rows"][0]["cells"]["mmlupro"]["text"].endswith("% of our BF16, measured here")
    got = ask(board, item("what percent of its base does bolt keep on ifbench"))["reply"]
    assert got["rows"][0]["cells"]["ifbench"]["text"].startswith("150.0% of Gemma 4 E4B-it")
    got = ask(board, item("how often does step4000 run out of room on mmlu-pro"))["reply"]
    assert re.match(r"^[\d,]+ of [\d,]+$", got["rows"][0]["cells"]["mmlupro"]["text"])


def test_best_on_ranks_within_its_group_and_lists_published_apart(board):
    cat = board["cat"]
    got = ask(board, item("which of ours is best on GPQA?"))["reply"]
    vs = [r["cells"]["gpqa"]["v"] for r in got["rows"]]
    assert vs == sorted(vs, reverse=True) and len(vs) == 3
    ours = {m["id"] for m in cat["models"] if m["group"] == "ours"}
    assert {r["id"] for r in got["rows"]} <= ours and not got["published"]
    got = ask(board, item("who's best on IFEval among phone-sized models?"))["reply"]
    phone = {m["id"] for m in cat["models"] if m["phone"]}
    assert {r["id"] for r in got["rows"] + got["published"]} <= phone
    # phone-sized: our builds that ran on a phone, never Q4, Q2 or BF16
    assert {S88, S40, aw.PHONE} <= phone and not {Q4, Q2} & phone
    assert "reported/nvidia/nvidia-nemotron-3.5-bolt-e4b" in phone


def test_phones_the_oneplus_is_both_oneplus_phones(board):
    got = ask(board, item("how fast is step8804 on the oneplus?"))["reply"]
    assert [r["name"] for r in got["rows"]] == ["step8804 on OnePlus 15",
                                                "step8804 on CPH2719 (not named yet)"]
    assert got["rows"][0]["cells"]["decode"]["text"] == "13.0 tok/s"


def test_boxes_the_rented_step4000_box_and_the_server(board):
    got = ask(board, item("when does the step4000 box finish?"))["reply"]
    assert [(x["where"], x["name"], x["what"]) for x in got["items"]] == [
        ("rented", "step4000", "HLE")]
    got = ask(board, item("is anything running on the server?"))["reply"]
    assert [x["name"] for x in got["items"]] == ["Qwen/Qwen3-0.6B"]


def test_explain_is_the_boards_own_words(board):
    from service import reports
    got = ask(board, item("what does \"near the ceiling\" mean?"))["reply"]
    assert got["text"] in reports.NOTES and got["text"].startswith("Near the ceiling marks")
    got = ask(board, item("what does ± mean?"))["reply"]
    assert got["text"] == "" and got["page"] == "score-legend"      # the page's own legend
    got = ask(board, item("what is IFBench?"))["reply"]
    assert got["title"] == "IFBench" and got["link"] == "#tab=benchmarks&q=ifbench_test"


def test_clarify_offers_the_nearest_and_cant_says_where(board):
    got = ask(board, item("show me the qwen one"))
    assert got["reply"]["kind"] == "clarify"
    ids = [c["value"] for c in got["reply"]["choices"]]
    assert 2 <= len(ids) <= 6 and all("qwen" in x.lower() for x in ids)   # its name or ours
    got = ask(board, item("start a run on step8804"))["reply"]
    assert got["say"] == "I can't start or queue runs." and got["link"] == "#tab=runs"
    got = ask(board, item("what's the weather"))["reply"]
    assert got["say"] == "That's outside what I can help with."


# ---------------------------------------------------------------------------
# what the model gets wrong is the board's to catch
# ---------------------------------------------------------------------------

def test_an_unknown_id_is_refused_for_a_clarify(board):
    got = assistant.ask("show gpt-9", "masein", cat=board["cat"], model=fake(
        {"action": "open_page", "page": "model_page", "model": "openai/gpt-9-ultra"}))
    assert got["action"]["action"] == "clarify" and got["reply"]["kind"] == "clarify"
    got = assistant.ask("step 8804 vs q4", "masein", cat=board["cat"], model=fake(
        {"action": "compare", "models": ["step 8804", "q4"], "benchmarks": ["GPQA Diamond"]}))
    assert got["action"]["models"] == [S88, Q4] and got["action"]["benchmarks"] == ["gpqa"]


def test_a_reply_that_isnt_the_json_is_handled(board):
    for raw in ("Sure! Q4 scores 80.", "{\"action\": \"launch\"}", "{not json"):
        got = assistant.ask("anything", "masein", cat=board["cat"], model=lambda *a, r=raw: r)
        assert got["reply"]["kind"] == "text"
        assert got["reply"]["say"].startswith("I didn't get that")


def test_a_model_down_says_so(board, monkeypatch):
    def down(*a, **k):
        raise assistant.Down("connection refused")
    monkeypatch.setattr(assistant, "call", down)
    got = assistant.ask("what's running?", "masein", cat=board["cat"])
    assert got["down"] and got["reply"]["kind"] == "down"


def test_the_request_asks_for_the_schema_thinking_off(board, monkeypatch):
    from service import db, served
    sent = {}

    def http(method, url, key="", body=None, timeout=10.0):
        sent.update(url=url, body=body)
        return 200, json.dumps({"choices": [{"message": {"content": '{"action":"boxes"}'}}],
                                "usage": {"prompt_tokens": 9}}).encode()
    db.served_put({"id": "served/Qwen3.6-35B-A3B-Q4-original-k-8", "name": "q4 k8",
                   "base_url": "http://127.0.0.1:8091/v1", "key": "", "how": "x",
                   "based_on": "", "thinking": "on", "pin": {"model": "q4.gguf"}, "by": "masein",
                   "at": 0})
    monkeypatch.setattr(served, "_http", http)
    got = assistant.ask("what's running?", "masein", cat=board["cat"])
    b = sent["body"]
    assert sent["url"] == "http://127.0.0.1:8091/v1/chat/completions"
    assert (b["temperature"], b["max_tokens"], b["chat_template_kwargs"]) == (
        0, 600, {"enable_thinking": False})
    sch = b["response_format"]["json_schema"]["schema"]
    assert {br["properties"]["action"]["const"] for br in sch["anyOf"]} == set(assistant.ACTIONS)
    assert got["reply"]["kind"] == "list"
    # the prompt: every model by its name and benchmark the board has; today's date
    # with the message, so the system prompt is the same message to message
    sysm, user = b["messages"][0]["content"], b["messages"][1]["content"]
    assert all(m["name"] in sysm for m in board["cat"]["models"])
    assert board["cat"]["today"] not in sysm and board["cat"]["today"] in user


def test_a_follow_up_carries_the_previous_action_by_name(board):
    """28b: the previous action, its models by name; keep it, change only what
    the new message changes"""
    b = item("and on HLE?")
    u = assistant.user_message(b["q"], b["prev"], "2026-10-24", board["cat"])
    head, act, *_ = u.split("\n")
    assert head == "Previous message: step8804 vs step4000 on GPQA and MMLU-Pro"
    assert act.startswith("Its action (keep it; change only what the new message changes): ")
    got = json.loads(act.split(": ", 1)[1])
    assert got == {"action": "compare", "models": ["step8804", "step4000"],
                   "benchmarks": ["gpqa", "mmlupro"]}
    assert u.endswith("Message: and on HLE?")


# ---------------------------------------------------------------------------
# 28b: the schema strict, names the board's to match, a second request a button
# ---------------------------------------------------------------------------

def test_each_action_is_strict_and_names_are_strings(board):
    sch = assistant.schema(board["cat"])
    for br in sch["anyOf"]:
        assert br["additionalProperties"] is False
        assert "question" not in br["properties"]
        for k in ("view", "only", "sort", "thinking", "measure", "page", "among", "chart"):
            p = br["properties"].get(k)
            if p:
                assert "enum" in p or "const" in p, (br["properties"]["action"], k)
        for k in ("model", "beside", "a", "b", "base", "since", "phone", "app"):
            if k in br["properties"]:
                assert br["properties"][k] == {"type": "string"}
    pages = [br["properties"]["page"] for br in sch["anyOf"] if "page" in br["properties"]]
    assert {"const": "models_table", "type": "string"} in pages
    assert {"const": "model_page", "type": "string"} in pages


def test_the_board_matches_names_and_asks_back_when_several_do(board):
    cat = board["cat"]
    got = assistant.resolve({"action": "compare", "models": ["the q4", "8804"],
                             "benchmarks": ["gpqa"]}, cat)
    assert got["models"] == [Q4, S88]
    got = assistant.resolve({"action": "open_page", "page": "model_page", "model": "step one"}, cat)
    assert got["action"] == "clarify" and set(got["candidates"]) == {
        S88, S40, "served/step1000-lda-2610"}
    got = assistant.resolve({"action": "phone", "app": "the phone build", "phone": "the galaxy"},
                            cat)
    assert got["app"] == "Qwen3.6-35B-A3B k4-LDA, UD-Q4_K_XL" and got["phone"] == "the galaxy"
    assert assistant.resolve({"action": "phone", "app": "4000"}, cat)["app"] == "step4000"
    got = assistant.resolve({"action": "phone", "app": "step8804", "phone": "the pixel"}, cat)
    assert got["action"] == "clarify" and got["about"] == "phone"


def test_a_second_request_is_a_button_and_a_search_is_as_good(board):
    got = ask(board, item("q4's gpqa and then open the on-device report"))
    assert got["reply"]["kind"] == "table" and got["reply"]["then"] == "open the on-device report"
    want = item("show the qwen3.6 model")["want"]
    for alt in ({"action": "clarify", "about": "model"},
                {"action": "open_page", "page": "models_table", "search": "Qwen3.6"}):
        assert assistant.same(alt, want, board["cat"])
    assert not assistant.same({"action": "open_page", "page": "models_table", "search": "llama"},
                              want, board["cat"])


# ---------------------------------------------------------------------------
# only us; read-only; the log
# ---------------------------------------------------------------------------

def test_only_the_named_people(board, monkeypatch):
    c = board["client"]
    assert c.get("/api/assistant", params={"by": "masein"}).json()["on"] is True
    assert c.get("/api/assistant", params={"by": "tester-b"}).json()["on"] is True   # any case
    assert c.get("/api/assistant", params={"by": "someone"}).json() == {"on": False}
    r = c.post("/api/assistant/ask", json={"by": "someone", "q": "what's running?"})
    assert r.status_code == 403
    assert c.post("/api/assistant/thumbs", json={"by": "", "id": "x"}).status_code == 403
    # nobody named: it doesn't exist
    monkeypatch.setattr(config, "ASSISTANT_USERS", "")
    assert c.get("/api/assistant", params={"by": "masein"}).status_code == 404
    assert c.post("/api/assistant/ask", json={"by": "masein", "q": "x"}).status_code == 404
    import service.app as appmod
    appmod._cache.update(key=None, payload=None, at=0.0)
    assert appmod.results_payload()["assistant"] is False
    monkeypatch.setattr(config, "ASSISTANT_USERS", "masein")
    appmod._cache.update(key=None, payload=None, at=0.0)
    assert appmod.results_payload()["assistant"] is True


def test_nothing_in_the_assistant_reaches_a_write():
    """read-only by construction: its module calls no write of the board's,
    and its one outgoing POST is the model's chat completion"""
    src = Path(assistant.__file__).read_text(encoding="utf-8")
    writes = re.findall(r"\bdb\.(\w+)\(", src)
    assert set(writes) <= {"served_get", "recent"}, set(writes)
    assert re.findall(r'_http\("(\w+)"', src) == ["POST", "GET"]
    assert src.count('"/chat/completions"') == 1
    for bad in ("served_put", "store_boxes", "card_add", "submit(", "unlink(", "rmtree",
                "subprocess", "requests."):
        assert bad not in src, bad
    # the page's side: the assistant's calls are its own three, and reports' reads
    import report_lm_eval as rpt
    js = rpt.JS[rpt.JS.index("const ASK = {"):rpt.JS.index("const ptN = ")]
    assert set(re.findall(r"post\('([^']+)'", js)) == {"api/assistant/ask", "api/assistant/thumbs"}
    assert set(re.findall(r"api\('([^'?]+)", js)) == {"api/reports/", "api/assistant",
                                                     "api/assistant/state"}


def test_every_question_and_its_thumbs_are_logged(board):
    got = assistant.ask("what's running?", "masein", cat=board["cat"],
                        model=fake({"action": "boxes"}))
    assistant.thumbs(got["id"], "masein", False, "the server's runs first")
    lines = [json.loads(x) for x in assistant.log_path().read_text().splitlines()]
    q = next(x for x in lines if x.get("id") == got["id"] and "q" in x)
    t = next(x for x in lines if x.get("id") == got["id"] and "thumbs" in x)
    assert (q["q"], q["action"], q["who"]) == ("what's running?", {"action": "boxes"}, "masein")
    assert (t["thumbs"], t["note"]) == ("down", "the server's runs first")
    assert assistant.log_path() == Path(config.BENCH_ROOT) / "assistant" / "log.jsonl"


def test_three_examples_from_the_bank_each_its_own_scenario():
    got = assistant.examples()
    assert len(got) == 3
    sc = {next(b["scenario"] for b in BANK if b["q"] == q) for q in got}
    assert len(sc) == 3 and not sc & {"refused", "messy", "ambiguous"}
