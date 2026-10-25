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
        "cant": "text", "not_found": "text"}
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


def fake(want: dict, cat: dict | None = None):
    """a model that answers with the action the bank expects, as {"do", "then"}
    — 28d: with the board's names for its ids and keys, as a model writes them"""
    named = (lambda a: assistant.as_named(a, cat)) if cat else (lambda a: a)
    first = named({k: v for k, v in want.items() if k not in ("also", "then")})
    return lambda system, user, schema: json.dumps(
        {"do": first, **({"then": named(want["then"])} if want.get("then") else {})})


def ask(board, b: dict) -> dict:
    return assistant.ask(b["q"], "masein", b.get("prev"), cat=board["cat"],
                         model=fake(b["want"], board["cat"]))


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
        if not assistant.same(got["action"], b["want"], board["cat"], b.get("prev")):
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
    text = f"{100 * c['v']:.1f} ± {100 * c['se']:.1f}"
    assert got["rows"][0]["cells"]["gpqa"]["text"] == text
    assert got["link"].startswith("#tab=models&view=frontier&cols=gpqa&models=")
    # 28d: one model on one benchmark — the reply says the number, its ± and
    # the thinking setting it was measured with
    assert got["say"] == f"{got['rows'][0]['name']} on GPQA Diamond: {text}, thinking on."


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

def test_a_name_nothing_matches_is_not_found_and_offers_a_search(board):
    """28d: never a clarify without choices — "I couldn't find it", and a
    search for it where that kind of name is"""
    got = assistant.ask("show gpt-9", "masein", cat=board["cat"], model=fake(
        {"action": "open_page", "page": "model_page", "model": "gpt-9"}))
    assert got["action"] == {"action": "not_found", "about": "model", "term": "gpt-9"}
    assert got["reply"] == {"kind": "text", "say": "I couldn't find “gpt-9”.", "text": "",
                            "link": "#tab=models&q=gpt-9", "linkText": "Search Models for it ▸"}
    # an id the board doesn't have: not found either, never the nearest few
    got = assistant.ask("show gpt-9", "masein", cat=board["cat"], model=fake(
        {"action": "open_page", "page": "model_page", "model": "openai/gpt-9-ultra"}))
    assert got["action"]["action"] == "not_found" and got["reply"]["kind"] == "text"
    got = assistant.ask("step 8804 vs q4", "masein", cat=board["cat"], model=fake(
        {"action": "compare", "models": ["step 8804", "the q4"], "benchmarks": ["GPQA Diamond"]}))
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
    assert sch["type"] == "object" and set(sch["properties"]) == {"do", "then"}
    assert {br["properties"]["action"]["const"] for br in sch["properties"]["do"]["anyOf"]} == set(
        assistant.ACTIONS)
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
                   "benchmarks": ["GPQA Diamond", "MMLU-Pro"]}            # 28d: by name too
    assert u.endswith("Message: and on HLE?")


# ---------------------------------------------------------------------------
# 28b: the schema strict, names the board's to match, a second request a button
# ---------------------------------------------------------------------------

def test_each_action_is_strict_and_names_are_strings(board):
    """28b, 28c: a branch an action, each strict; the closed lists enums; every
    name — model, benchmark, report, phone, app, and (28d) a Models group or a
    place on the top bar — a string; a second request a whole action of the
    same branches"""
    sch = assistant.schema(board["cat"])
    assert sch["additionalProperties"] is False and sch["required"] == ["do"]
    assert sch["properties"]["then"] == sch["properties"]["do"]
    places = 0
    for br in sch["properties"]["do"]["anyOf"]:
        assert br["additionalProperties"] is False
        assert "question" not in br["properties"] and "then" not in br["properties"]
        for k in ("view", "only", "thinking", "measure", "among", "chart", "kind"):
            p = br["properties"].get(k)
            if p:
                assert "enum" in p or "const" in p, (br["properties"]["action"], k)
        for k in ("model", "beside", "a", "b", "base", "since", "phone", "app", "benchmark",
                  "report", "sort", "group"):
            if k in br["properties"]:
                assert br["properties"][k] == {"type": "string"}, k
        for k in ("models", "benchmarks", "columns"):
            if k in br["properties"]:
                assert br["properties"][k]["items"] == {"type": "string"}, k
        # open_page's page: a page's own const, or a place as the person names it
        if "page" in br["properties"] and "const" not in br["properties"]["page"]:
            assert br["properties"]["page"] == {"type": "string"}
            places += 1
    assert places == 1
    pages = [br["properties"]["page"] for br in sch["properties"]["do"]["anyOf"]
             if "page" in br["properties"]]
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
    assert got == {"action": "not_found", "about": "phone", "term": "the pixel"}


def test_a_second_request_is_a_whole_action_behind_a_button(board):
    got = ask(board, item("q4's gpqa and then open the on-device report"))
    assert got["reply"]["kind"] == "table"
    t = got["reply"]["then"]
    assert t["label"] == "Open On-device models"
    assert t["action"] == {"action": "open_page", "page": "report", "report": "on-device"}
    assert t["reply"]["link"] == "#tab=reports&report=on-device"
    # a free-text "then" is no second action
    a = assistant.parse('{"do": {"action": "boxes"}, "then": "open runs"}')
    assert a == {"action": "boxes"}


def test_the_board_asks_back_for_benchmarks_and_reports_too(board):
    cat = board["cat"]
    got = assistant.resolve({"action": "chart", "chart": "bars", "benchmark": "the arc one"}, cat)
    assert got["action"] == "clarify" and got["about"] == "benchmark"
    assert set(got["candidates"]) == {"arc", "arc_challenge", "arc_easy"}
    got = assistant.resolve({"action": "open_page", "page": "report",
                             "report": "the report about phones"}, cat)
    assert got["action"] == "clarify" and set(got["candidates"]) == {"testers", "on-device"}
    got = assistant.resolve({"action": "compare", "models": ["q4"], "benchmarks": ["GPQA Diamond",
                                                                                "math l5"]}, cat)
    assert got["benchmarks"] == ["gpqa", "math"]
    assert assistant.resolve({"action": "open_page", "page": "report",
                              "report": "the two-builds report"}, cat)["report"] == "paired"


def test_a_tab_or_thinking_no_message_names_is_left_out(board):
    cat = board["cat"]
    a = {"action": "open_page", "page": "models_table", "view": "latest", "only": "ours"}
    assert assistant.resolve(a, cat, "only ours") == {
        "action": "open_page", "page": "models_table", "only": "ours"}
    assert assistant.resolve(a, cat, "latest tab, only ours")["view"] == "latest"
    # a follow-up: the tab the message before named is kept
    assert assistant.resolve(a, cat, "only ours", "the latest tab")["view"] == "latest"
    c = {"action": "compare", "models": [S88], "benchmarks": ["gpqa"], "thinking": "on"}
    assert "thinking" not in assistant.resolve(c, cat, "step8804 gpqa")
    assert assistant.resolve(c, cat, "step8804 gpqa, thinking on")["thinking"] == "on"
    link = ask(board, item("search the models for ornith"))["reply"]["link"]
    assert link == "#tab=models&q=ornith"
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


@pytest.mark.parametrize("name", ["the qwen one", "the gemma page", "the bonsai", "the edge0 model",
                                  "the claude one", "the gpt model", "the step one"])
def test_an_ambiguous_name_as_written_is_asked_back(board, name):
    got = assistant.resolve({"action": "open_page", "page": "model_page", "model": name},
                            board["cat"])
    assert got["action"] == "clarify" and len(got["candidates"]) >= 2, (name, got)


# ---------------------------------------------------------------------------
# 28d: graded by what the person sees; the board's fixes from the second run
# ---------------------------------------------------------------------------

def test_the_grade_is_what_the_person_sees(board):
    """a field at its default, a refusal's kind, a base the board shows anyway,
    a follow-up's new models alone or beside the old, the tab a follow-up is
    on, one model's number from its page or from compare, "not found" for a
    clarify: each the same answer"""
    cat = board["cat"]
    same = assistant.same
    # boxes: every run shows the rented ones
    assert same({"action": "boxes"}, {"action": "boxes", "which": "rented"}, cat)
    assert not same({"action": "boxes", "which": "server"},
                    {"action": "boxes", "which": "rented"}, cat)
    # a refusal is a refusal
    assert same({"action": "cant", "kind": "start_run"}, {"action": "cant", "kind": "spend"}, cat)
    # keeps: BF16 is the base the On-device report shows beside Q2
    bf16 = "served/Qwen3.6-35B-A3B-BF16-lda-2610"
    k = {"action": "compare", "models": [Q2], "benchmarks": ["mmlupro"], "measure": "keeps"}
    assert same({**k, "models": [bf16, Q2]}, k, cat)
    assert not same({**k, "models": [S40, Q2]}, k, cat)
    # "and q4?": q4 alone, or beside q2, never something else as well
    b = item("and q4?")
    assert same({"action": "compare", "models": [Q2, Q4], "benchmarks": ["gpqa"]}, b["want"], cat,
                b["prev"])
    assert not same({"action": "compare", "models": [Q2, Q4, S40], "benchmarks": ["gpqa"]},
                    b["want"], cat, b["prev"])
    # "only ours" after "frontier tab": on the Frontier tab, named or not
    b = item("only ours")
    assert same({"action": "open_page", "page": "models_table", "only": "ours"}, b["want"], cat,
                b["prev"])
    # one model on one benchmark: its page opened there says the number too
    w = {"action": "compare", "models": [Q4], "benchmarks": ["gpqa"]}
    assert same({"action": "open_page", "page": "model_page", "model": Q4, "benchmark": "gpqa"},
                w, cat)
    assert not same({"action": "open_page", "page": "model_page", "model": Q4}, w, cat)
    assert not same({"action": "open_page", "page": "model_page", "model": Q4, "benchmark": "gpqa"},
                    {**w, "thinking": "off"}, cat)
    # "I couldn't find it" is a clarify without choices
    assert same({"action": "not_found", "about": "model", "term": "x"},
                {"action": "clarify", "about": "model"}, cat)


def test_settings_rows_are_one_model_and_never_asked_between(board):
    """thinking on and off and "lookahead 1" are settings of one model: one
    entity, its row the one with the most scores (a tie to thinking on, then
    to the fewest settings)"""
    from service import app
    payload = json.loads(json.dumps(app.results_payload()))
    look = aw.PHONE + " · thinking · lookahead 1"
    payload["models"].append({**next(m for m in payload["models"]
                                     if m["id"] == aw.PHONE + " · thinking"),
                              "id": look, "name": "Qwen3.6-35B-A3B k4-LDA (phone build) · thinking"
                              " · lookahead 1"})
    cat = assistant.catalog(payload)
    pb = [m for m in cat["models"] if m["id"].startswith(aw.PHONE)]
    assert len(pb) == 1 and look in pb[0]["rows"] and "lookahead" not in pb[0]["name"]
    assert assistant.resolve({"action": "open_page", "page": "model_page",
                              "model": "the phone build"}, cat)["model"] == aw.PHONE
    assert assistant.pick_row(cat, aw.PHONE, ["gpqa_diamond_epoch"], "default") == (
        aw.PHONE + " · thinking")
    # step8804: IFEval has both settings, a tie — thinking on; asked off, off
    assert assistant.pick_row(cat, S88, ["ifeval_google"], "default") == S88 + " · thinking"
    assert assistant.pick_row(cat, S88, ["ifeval_google"], "off") == S88


def test_only_real_matches_are_asked_about(board):
    """the nearest few by spelling are gone: a candidate matches the words;
    a date, "none" or a page's name is never a name"""
    cat = board["cat"]
    assert assistant.nearest("zzqx model", cat) == []
    # "none" is no phone: every phone
    assert assistant.resolve({"action": "phone", "app": "step8804", "phone": "none"}, cat) == {
        "action": "phone", "app": "step8804"}
    # a date is no model: since none, so no since_model
    got = assistant.resolve({"action": "open_page", "page": "models_table", "view": "frontier",
                             "when": "since_model", "since": "2026-10-25", "only": "closed",
                             "sort": "hle"}, cat, "frontier sorted by hle and only closed ones")
    assert got == {"action": "open_page", "page": "models_table", "view": "frontier",
                   "only": "closed", "sort": "hle"}
    # a page's name is no benchmark
    got = assistant.resolve({"action": "open_page", "page": "playground", "benchmark": "playground"},
                            cat, "go to the playground")
    assert got == {"action": "open_page", "page": "playground"}
    # an app written as the phone is the app
    got = assistant.resolve({"action": "open_page", "page": "report", "report": "testers",
                             "phone": "step8804"}, cat, "open the latest step8804 test")
    assert got == {"action": "open_page", "page": "report", "report": "testers",
                   "app": "step8804"}


def test_a_name_is_the_persons_words(board):
    """a name the model chose that the message doesn't hold: kept when the
    person's words mean it alone, their words when they mean several (asked
    back), left out when nothing they wrote means it"""
    cat = board["cat"]
    r = assistant.resolve
    # "the q4" means one model: the model's longer name for it is kept
    got = r({"action": "compare", "models": ["Q4 (UD-Q4_K_XL)"],
             "benchmarks": ["GPQA Diamond"]}, cat, "the q4 on gpqa")
    assert got["models"] == [Q4] and got["benchmarks"] == ["gpqa"]
    # "the report about phones" is two: the testers' title the model chose is asked back
    got = r({"action": "open_page", "page": "report", "report": "testers"}, cat,
            "open that report about phones")
    assert got["action"] == "clarify" and set(got["candidates"]) == {"testers", "on-device"}
    # "the report" alone: every report, asked
    got = r({"action": "open_page", "page": "report", "report": "on-device"}, cat,
            "open the report")
    assert got["action"] == "clarify" and got["about"] == "report"
    # a model nothing in the message means: left out, and asked for
    got = r({"action": "compare", "models": [S88, S40], "benchmarks": ["gpqa"]}, cat,
            "compare the two on GPQA")
    assert got["action"] == "clarify" and got.get("about", "model") == "model"
    # several the same words mean, spelt out by the model: kept
    steps = ["step8804", "step4000", "step1000"]
    got = r({"action": "compare", "models": steps, "benchmarks": ["hle"]}, cat,
            "the step checkpoints on hle")
    assert len(got["models"]) == 3
    # a report the message asks for without its name: kept
    got = r({"action": "open_page", "page": "report", "report": "two builds", "a": "step8804",
             "b": "q4", "benchmark": "gpqa", "side": "b_better"}, cat,
            "show me what step8804 got wrong on GPQA that Q4 got right")
    assert got["report"] == "paired"
    # a follow-up's names are in the action before it
    prev = {"q": "step8804 vs step4000 on GPQA", "action": {"action": "compare",
                                                            "models": [S88, S40],
                                                            "benchmarks": ["gpqa"]}}
    got = r({"action": "compare", "models": ["step8804", "step4000"], "benchmarks": ["HLE"]},
            cat, "and on HLE?", prev["q"], prev["action"])
    assert got["models"] == [S88, S40]


def test_places_groups_and_the_fields_a_report_takes(board):
    cat = board["cat"]
    r = assistant.resolve
    assert r({"action": "open_page", "page": "the benchmarks catalogue"}, cat,
             "open the benchmarks catalogue") == {"action": "open_page", "page": "benchmarks"}
    assert r({"action": "open_page", "page": "GPQA questions"}, cat,
             "show me the GPQA questions") == {"action": "open_page", "page": "benchmarks",
                                               "benchmark": "gpqa"}
    assert r({"action": "open_page", "page": "the models"}, cat, "the models") == {
        "action": "open_page", "page": "models_table"}
    assert r({"action": "open_page", "page": "the moon"}, cat, "take me to the moon")[
        "action"] == "not_found"
    # a group as written, on its tab
    assert r({"action": "open_page", "page": "models_table", "view": "mobile",
              "group": "mobile-mmlu"}, cat, "mobile tab, mobile-mmlu group")["group"] == "mmlu"
    got = r({"action": "open_page", "page": "models_table", "group": "Mobile-MMLU"}, cat,
            "the mobile-mmlu group")
    assert (got["view"], got["group"]) == ("mobile", "mmlu")
    # the ran-out report takes a model and a benchmark, no app
    got = r({"action": "open_page", "page": "report", "report": "ran-out report",
             "model": "step4000", "app": "step4000"}, cat, "the ran-out report for step4000")
    assert "app" not in got and got["model"] == S40
    # On-device's memory chart is its own models: no "only"
    got = r({"action": "chart", "chart": "memory", "benchmark": "IFBench", "only": "ours"}, cat,
            "IFBench against memory for the phone models")
    assert got == {"action": "chart", "chart": "memory", "benchmark": "ifbench"}


def test_one_model_one_benchmark_says_its_number(board):
    """compare or its page opened on that benchmark: the reply says the
    number, its ± and the thinking setting — or who published it"""
    cat = board["cat"]
    got = assistant.run({"action": "compare", "models": [S88], "benchmarks": ["ifeval"],
                         "thinking": "off"}, cat)
    assert got["say"] == "step8804 on IFEval: 88.0 ± 2.0, thinking off."
    got = assistant.run({"action": "open_page", "page": "model_page", "model": S88,
                         "benchmark": "gpqa"}, cat)
    assert got["say"] == "Opened step8804’s page, its answers on GPQA Diamond: 70.0 ± 2.0, " \
        "thinking on."
    assert got["link"] == "#model=served%2Fstep8804-lda-2610%20%C2%B7%20thinking" \
        "&answers=gpqa_diamond_epoch"
    e0, _ = assistant.model_id("edge0 35b", cat)
    got = assistant.run({"action": "compare", "models": [e0], "benchmarks": ["gpqa"]}, cat)
    assert got["say"].endswith(": 79.8, as published by Edge0.")


def test_the_eval_keeps_every_answer_and_grades_them_again(board, monkeypatch, capsys, tmp_path):
    """the raw answers go to a file, and --rescore grades them again on the
    board as it is, asking no model"""
    import assistant_eval as ev
    b = item("what's Q4's GPQA?")
    kept = tmp_path / "run.jsonl"
    kept.write_text(json.dumps({"id": b["id"], "raw": json.dumps(
        {"do": {"action": "open_page", "page": "model_page", "model": "q4",
                "benchmark": "gpqa"}}), "ms": 500, "provider": "served:x"}) + "\n")

    def never(*a, **k):
        raise AssertionError("asked a model")
    monkeypatch.setattr(assistant, "call", never)
    assert ev.main(["--rescore", str(kept)]) == 0
    out = capsys.readouterr().out
    assert "1 questions, served:x's answers kept in" in out
    assert "| number | 1/1 (100%) | – | 1/1 (100%) |" in out


# ---------------------------------------------------------------------------
# 28e: masein's notes on the third run
# ---------------------------------------------------------------------------

def test_a_group_where_a_benchmark_goes_is_the_group(board):
    """groups are recognised before benchmarks: "mobile devicemark chart"
    written as a chart of the benchmark "DeviceMark", or with it as a column,
    is Mobile's DeviceMark as a chart — never "not found". A name a benchmark
    has itself stays the benchmark ("math" is MATH Level 5)"""
    cat = board["cat"]
    q = "mobile devicemark chart"
    want = {"action": "open_page", "page": "models_table", "view": "mobile",
            "group": "devicemark", "show": "chart"}
    for wrote in ({"action": "chart", "chart": "bars", "benchmark": "DeviceMark"},
                  {"action": "open_page", "page": "models_table", "view": "mobile",
                   "columns": ["DeviceMark"], "show": "chart"},
                  {"action": "open_page", "page": "models_table", "view": "mobile",
                   "group": "DeviceMark", "show": "chart"}):
        assert assistant.resolve(wrote, cat, q) == want, wrote
    got = assistant.resolve({"action": "compare", "models": ["step8804"],
                             "benchmarks": ["Mobile-MMLU"]}, cat, "step8804 on mobile-mmlu")
    assert got == {"action": "open_page", "page": "models_table", "view": "mobile",
                   "group": "mmlu", "models": [S88]}
    got = assistant.resolve({"action": "chart", "chart": "bars", "benchmark": "math"}, cat,
                            "chart math")
    assert got == {"action": "chart", "chart": "bars", "benchmark": "math"}


def test_a_model_page_takes_thinking_and_states_that_rows_number(board):
    """b111: "step8804's gpqa with thinking off" opened as its page lost the
    setting and stated the thinking-on number. Its page takes thinking: the
    row, the link and the number are that setting's. Beside itself is its page"""
    cat = board["cat"]
    a = assistant.resolve({"action": "open_page", "page": "model_page", "model": "step8804",
                           "benchmark": "IFEval", "beside": "step8804", "thinking": "off"}, cat,
                          "step8804's ifeval with thinking off")
    assert a == {"action": "open_page", "page": "model_page", "model": S88, "benchmark": "ifeval",
                 "thinking": "off"}
    got = assistant.run(a, cat)
    assert got["link"] == "#model=served%2Fstep8804-lda-2610&answers=ifeval_google"
    assert got["say"] == "Opened step8804’s page, its answers on IFEval: 88.0 ± 2.0, thinking off."
    # graded as compare with the same setting, never another
    w = {"action": "compare", "models": [S88], "benchmarks": ["ifeval"], "thinking": "off"}
    assert assistant.same(a, w, cat)
    assert not assistant.same({**a, "thinking": "on"}, w, cat)
    assert "thinking" in assistant.schema(cat)["properties"]["do"]["anyOf"][1]["properties"]


def test_two_actions_in_the_order_the_person_wants(board):
    """"what's X in a report": the explanation first, the report second; a
    follow-up whose first action is the turn before's again shows the new one"""
    cat = board["cat"]
    got = assistant.resolve({"action": "open_page", "page": "report", "report": "on-device",
                             "then": {"action": "explain", "topic": "stand_in"}}, cat,
                            "what's the stand-in base on the on-device report?")
    assert got == {"action": "explain", "topic": "stand_in",
                   "then": {"action": "open_page", "page": "report", "report": "on-device"}}
    b = item("and as a chart")
    got = assistant.resolve({"action": "best_on", "benchmark": "GPQA Diamond",
                             "then": {"action": "chart", "chart": "bars",
                                      "benchmark": "GPQA Diamond"}}, cat, b["q"],
                            b["prev"]["q"], b["prev"]["action"])
    assert got == {"action": "chart", "chart": "bars", "benchmark": "gpqa"}
    assert assistant.same(got, b["want"], cat, b["prev"])
    # a first action that isn't the one before stays first
    got = assistant.resolve({"action": "best_on", "benchmark": "HLE",
                             "then": {"action": "chart", "chart": "bars", "benchmark": "HLE"}},
                            cat, "best on hle, and as a chart")
    assert got["action"] == "best_on" and got["then"]["action"] == "chart"


@pytest.mark.parametrize("q,wrote,want", [
    # the models and the benchmark of the turn before, and the new model
    ("and q4?", {"models": ["Q2 (UD-Q2_K_XL)", "q4"], "benchmarks": ["GPQA Diamond"]},
     {"models": [Q2, Q4], "benchmarks": ["gpqa"]}),
    ("same for step1000", {"models": ["step4000", "step1000"], "benchmarks": ["HLE"]},
     {"models": [S40, "served/step1000-lda-2610"], "benchmarks": ["hle"]}),
    # the models and benchmarks before, and the new benchmark
    ("and on HLE?", {"models": ["step8804", "step4000"],
                     "benchmarks": ["GPQA Diamond", "MMLU-Pro", "HLE"]},
     {"models": [S88, S40], "benchmarks": ["gpqa", "mmlupro", "hle"]}),
])
def test_a_follow_up_keeps_what_the_turn_before_named(board, q, wrote, want):
    """masein's condition on the name check: words from earlier turns are the
    person's words, so a follow-up keeps the models and benchmarks named
    before — none is left out as a name the model chose"""
    b = item(q)
    got = assistant.resolve({"action": "compare", **wrote}, board["cat"], b["q"],
                            b["prev"]["q"], b["prev"]["action"])
    assert got == {"action": "compare", **want}, got
    assert assistant.same(got, b["want"], board["cat"], b["prev"])


def test_the_eval_prints_what_the_model_wrote_for_a_miss(board, monkeypatch, capsys, tmp_path):
    import assistant_eval as ev
    b = item("open the frontier chart")
    raw = json.dumps({"do": {"action": "open_page", "page": "models_table", "view": "frontier"}})
    kept = tmp_path / "run.jsonl"
    kept.write_text(json.dumps({"id": b["id"], "raw": raw, "ms": 400,
                                "provider": "served:x"}) + "\n")
    monkeypatch.setattr(assistant, "call", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("asked a model")))
    assert ev.main(["--rescore", str(kept)]) == 0
    out = capsys.readouterr().out
    assert f"    wrote  {raw}" in out
    assert '    got    {"action": "open_page", "page": "models_table", "view": "frontier"}' in out


@pytest.mark.parametrize("q,wrote", [
    ("the ran-out report for step4000 on HLE", {"action": "open_page", "page": "report",
                                                "report": "ran-out report", "model": "step4000",
                                                "benchmark": "HLE"}),
    ("open the frontier chart", {"action": "open_page", "page": "models_table",
                                 "view": "frontier", "show": "chart"}),
    ("models tested in the last 30 days, ours only", {"action": "open_page",
                                                      "page": "models_table", "when": "30d",
                                                      "only": "ours"}),
])
def test_the_board_keeps_each_field_the_model_wrote(board, q, wrote):
    """b027, b030, b032 in the third run: the ran-out report takes its
    benchmark, a Models tab its chart and its Show — none is dropped as a
    field the page doesn't use, so a missing one was never written"""
    assert assistant.same(assistant.resolve(wrote, board["cat"], q), item(q)["want"],
                          board["cat"])


# ---------------------------------------------------------------------------
# 28e: the paid pass on OpenRouter, and two name-matching bugs
# ---------------------------------------------------------------------------

SONNET = "anthropic/claude-sonnet-5.5"


def openrouter(monkeypatch, params):
    """OpenRouter's models API with one model taking `params`, and its chat
    completion: what was sent, kept"""
    from service import ai_models, served
    sent, asked = [], []
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(ai_models, "_get", lambda path: asked.append(path) or {"data": [
        {"id": SONNET, "supported_parameters": params}]})

    def http(method, url, key="", body=None, timeout=10.0):
        sent.append(body)
        return 200, json.dumps({"choices": [{"message": {"content": '{"do": {"action": "boxes"}}'}}],
                                "usage": {"prompt_tokens": 9}}).encode()
    monkeypatch.setattr(served, "_http", http)
    assistant._or_params.clear()
    return sent, asked


def test_openrouter_is_sent_only_the_parameters_the_model_takes(board, monkeypatch):
    """the paid pass: OpenRouter's page for the model lists response_format and
    reasoning, not temperature — and with require_parameters one it doesn't
    take found no endpoint (404). It is read once, and left out"""
    sent, asked = openrouter(monkeypatch, ["max_tokens", "response_format", "structured_outputs",
                                           "reasoning"])
    for _ in range(2):
        assistant.call("system", "user", {"type": "object"}, "openrouter:" + SONNET)
    b = sent[0]
    assert "temperature" not in b and b["max_tokens"] == 600 and b["reasoning"] == {
        "enabled": False}
    assert b["response_format"]["type"] == "json_schema"
    assert b["provider"] == {"require_parameters": True}
    assert asked == ["/models"]                             # once, not a request each
    # a model that can't take response_format: a clear stop, not 266 refusals
    sent, _ = openrouter(monkeypatch, ["max_tokens", "temperature"])
    with pytest.raises(assistant.Unfit, match="doesn't take response_format"):
        assistant.call("system", "user", {"type": "object"}, "openrouter:" + SONNET)
    assert sent == []


def test_the_eval_says_what_it_left_out_and_stops_on_no_schema(board, monkeypatch, capsys,
                                                               tmp_path):
    import assistant_eval as ev
    openrouter(monkeypatch, ["max_tokens", "response_format", "reasoning"])
    assert ev.main(["--provider", "openrouter:" + SONNET, "--limit", "1",
                    "--json", str(tmp_path / "a.jsonl")]) == 0
    assert (f"OpenRouter: {SONNET} is sent only the parameters it takes — left out: "
            "temperature") in capsys.readouterr().out
    openrouter(monkeypatch, ["max_tokens"])
    assert ev.main(["--provider", "openrouter:" + SONNET, "--json",
                    str(tmp_path / "b.jsonl")]) == 2
    assert "Stopped before asking: " in capsys.readouterr().out


def test_three_downs_in_a_row_stop_the_run(board, monkeypatch, capsys, tmp_path):
    import assistant_eval as ev
    calls = []

    def down(*a, **k):
        calls.append(1)
        raise assistant.Down('HTTP 404: {"error": "No endpoints found"}')
    monkeypatch.setattr(assistant, "call", down)
    assert ev.main(["--provider", "served:x", "--json", str(tmp_path / "c.jsonl")]) == 1
    out = capsys.readouterr().out
    assert len(calls) == 3
    assert "Stopped after three answers in a row didn't come (b003 the third): down: HTTP 404" \
        in out


@pytest.mark.parametrize("q,wrote,key,want", [
    ("q4s gpqa", {"action": "compare", "models": ["Q4 (UD-Q4_K_XL)"],
                  "benchmarks": ["GPQA Diamond"]}, "models", [Q4]),
    ("step4000s answers on gpqa", {"action": "open_page", "page": "model_page",
                                   "model": "step4000 (lda)", "benchmark": "GPQA Diamond"},
     "model", S40),
])
def test_a_name_with_an_s_stuck_on_is_the_name(board, q, wrote, key, want):
    """"q4s" matched nothing — not exact, not inside a name, too far for a
    close one — so the model's name for it was left out and the board asked
    for one. A run that matches nothing and ends in "s" is tried without it"""
    assert assistant._match("q4s", assistant._model_items(board["cat"])) == [Q4]
    got = assistant.resolve(wrote, board["cat"], q)
    assert got[key] == want, got


def test_ordinary_words_are_never_ask_back_candidates(board):
    """"then" was inside "the newest checkpoint", so any message with "then"
    could offer step8804: a name is inside another only from a word's start,
    and an ordinary word never feeds the candidates"""
    cat = board["cat"]
    items = assistant._model_items(cat)
    assert assistant._match("then", items) == []
    assert assistant._match("newest checkpoint", items) == [S88]       # from a word's start
    assert assistant._match("8804", items) == [S88]
    got = assistant.resolve({"action": "clarify", "about": "model"}, cat,
                            "compare them and then chart it")
    assert got == {"action": "clarify", "candidates": []}
