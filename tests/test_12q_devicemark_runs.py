"""12q (the brief's): the devicemark suite through the board's one queue — a
served setup asked over its server (the fake llama-server here: every reply
written by the test), the pilot and its check that the cap counts the
thinking, a stopped battery asked only the rest next time, a thinking-on row
apart, the MTP parity check, the speed test and a speed from a device; and a
Hugging Face model's battery through lm_eval (not run: its answers written
here). The questions are invented, in DM_ITEMS as the server keeps them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import devicemark as dm
from conftest import make_service
from fake_openai import FakeServer
from service import config, db, runner, served

ME = "tests"


def write_items(path: Path) -> dict:
    """the battery's items as load_items keeps them on the server — invented:
    IFEval asks for no commas; MMLU-Pro's key is A for an even id, else B;
    MATH's is 1/2"""
    items = {}
    for b, k in dm.keys_for("full"):
        if b == "ifeval":
            it = {"bench": b, "key": k, "prompt": f"Write a fixture note {k} with no commas.",
                  "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
        elif b == "mmlu_pro":
            it = {"bench": b, "key": k, "question": f"Fixture question {k}?",
                  "options": [f"option {i}" for i in range(10)],
                  "answer": "A" if int(k) % 2 == 0 else "B", "category": "fixture"}
        else:
            it = {"bench": b, "key": k, "problem": f"Fixture problem {k}.",
                  "answer": "\\frac{1}{2}", "subject": "fixture"}
        it["text"] = dm.prompt_for(it)
        items[(b, k)] = it
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": dm.VERSION}) + "\n" + "".join(
        json.dumps(items[k]) + "\n" for k in dm.keys_for("full")), encoding="utf-8")
    return items


def replies(server: FakeServer, wrong: set[str] = frozenset()) -> None:
    """the fake answers as a small model might: A in a box for MMLU-Pro, 0.5
    for MATH, a comma-free note for IFEval — thinking when asked, and the cap
    check's 64 tokens cut off inside the thinking"""
    def text(body):
        p = body["messages"][-1]["content"]
        if body.get("max_tokens") == 64:
            return ""
        if "correct option" in p:
            return "Step by step. So " + ("\\boxed{C}" if p in wrong else "\\boxed{A}")
        if "\\boxed{}" in p:
            return "It is one half: \\boxed{0.5}"
        return "A plain note without any" + (", comma" if p in wrong else " commas")
    server.reply = text
    server.reasoning = lambda b: ("Let me think this through."
                                  if (b.get("chat_template_kwargs") or {}).get("enable_thinking")
                                  else "")
    server.tokens = lambda b: 64 if b.get("max_tokens") == 64 else 120
    server.finish = lambda b: "length" if b.get("max_tokens") == 64 else "stop"
    server.timings = lambda b: {"prompt_n": 40, "predicted_n": 120,
                                "predicted_per_second": 31.5}


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    monkeypatch.setattr(config, "SERVED_RETRY_S", 0.3)
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)

    def no_vram():
        raise AssertionError("a served run waits for no VRAM here")
    monkeypatch.setattr(runner, "gpu_free_mib", no_vram)
    # 12q.G: an hf run counts the battery's longest prompt with the model's
    # tokenizer; no test fetches one (deploy step 3 runs these beside transformers)
    from service import devicemark as svc_dm
    monkeypatch.setattr(svc_dm, "prompt_tokens", lambda *a, **k: (900, True))
    # …and counts each answer with it once lm_eval has answered (mark_hf). Where
    # transformers is installed — the image, in step 3 — that went to the Hub
    # for every model id a test names. No test does: the lengths stay uncounted,
    # as they are where it isn't installed
    if importlib.util.find_spec("transformers"):
        import transformers

        def no_tokenizer(*a, **k):
            raise OSError("no tokenizer is fetched in a test")
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", no_tokenizer)
    write_items(config.DM_ITEMS)
    yield client
    client.__exit__(None, None, None)


@pytest.fixture
def fake():
    s = FakeServer()
    replies(s)
    yield s
    s.close()


def register(server: FakeServer, name: str, how: str = "llama-server -ngl 99 --n-cpu-moe 30",
             thinking: str = "off") -> str:
    return served.register({"name": name, "base_url": server.base, "how": how,
                            "thinking": thinking, "phone": True}, ME)["id"]


def queue(mid: str, part: str = "full", thinking: bool = False, pair: str = "") -> dict:
    sid = db.add(mid, "instruct", "devicemark", ME, "", thinking=thinking, part=part, pair=pair)
    runner.run_submission(db.get(sid))
    return db.get(sid)


def row_dir(mid: str, thinking: bool = False) -> Path:
    return config.OUT_DIR / (mid.replace("/", "__") + ("__thinking" if thinking else ""))


# ---------------------------------------------------------------------------
# the queue takes it
# ---------------------------------------------------------------------------

def test_what_the_queue_takes_and_refuses(svc, fake, monkeypatch):
    client = svc
    mtp, plain = register(fake, "Phone MTP", "--spec-type mtp"), register(fake, "Phone plain")

    def post(**kw):
        return client.post("/api/submissions", json={"suite": "devicemark", **kw})
    r = post(hf_id=mtp, part="pilot")
    assert r.status_code == 200 and r.json()["part"] == "pilot"
    # a served setup's thinking is asked for this suite, whatever it was registered with
    assert post(hf_id=mtp, thinking=True).status_code == 200
    assert post(hf_id=mtp, part="parity", pair=plain).json()["pair"] == plain
    assert post(hf_id=mtp, part="speed").status_code == 200
    # the same work again joins it; another part is another run
    assert post(hf_id=mtp, part="pilot").json()["note"].startswith("already in the queue")
    for kw, why in [({"hf_id": mtp, "part": "parity"}, "needs pair"),
                    ({"hf_id": mtp, "part": "parity", "pair": "served/nobody"},
                     "not one of our registered served setups"),
                    ({"hf_id": mtp, "part": "parity", "pair": mtp}, "the same setup"),
                    ({"hf_id": mtp, "part": "speed", "thinking": True}, "doesn't apply"),
                    ({"hf_id": mtp, "part": "all"}, "part must be one of"),
                    ({"hf_id": mtp, "pair": plain}, "parity check only")]:
        r = post(**kw)
        assert r.status_code == 422 and why in r.json()["detail"], (kw, r.text)
    monkeypatch.setattr("service.hfmeta.remote_code_check", lambda h: {"own_code": False})
    # 12q.D: a model this server can run (Nemotron-3-Nano-4B can't, and says so first)
    r = post(hf_id="Qwen/Qwen3.5-4B", part="pilot")
    assert r.status_code == 422 and "a Hugging Face model sits the whole battery" in \
        r.json()["detail"]
    assert post(hf_id="Qwen/Qwen3.5-4B").status_code == 200
    r = client.post("/api/submissions", json={"hf_id": mtp, "suite": "everyday",
                                              "part": "pilot"})
    assert r.status_code == 422 and "devicemark suite only" in r.json()["detail"]


# ---------------------------------------------------------------------------
# a served setup
# ---------------------------------------------------------------------------

def test_the_pilot_asks_30_by_the_protocol_and_checks_the_cap(svc, fake):
    mid = register(fake, "Qwen3.6 phone MTP", "llama-server --spec-type mtp --port 8094")
    row = queue(mid, "pilot")
    assert row["status"] == "done", row["error"]
    asked = [b for b in fake.requests if b["max_tokens"] != 64]
    assert len(asked) == 30
    for b in asked:
        assert (b["max_tokens"], b["temperature"], b["seed"]) == (4096, 0.0, 0)
        assert b["chat_template_kwargs"] == {"enable_thinking": False}
        assert [m["role"] for m in b["messages"]] == ["user"]
    probe = [b for b in fake.requests if b["max_tokens"] == 64]
    assert len(probe) == 1 and probe[0]["chat_template_kwargs"] == {"enable_thinking": True}
    p = json.loads((row_dir(mid) / dm.PILOT_NAME).read_text())
    assert p["part"] == "pilot" and p["n"] == 30 and len(p["items"]) == 30
    assert p["cap_check"]["ok"] is True and "the cap counts the thinking" in p["cap_check"]["line"]
    assert p["setup"]["mtp"] and not p["setup"]["lookahead"] and p["setup"]["battery"] == dm.VERSION
    assert p["benches"]["math"]["acc"] == 1.0 and p["benches"]["ifeval"]["acc"] == 1.0
    even = sum(int(k) % 2 == 0 for k in dm.battery()["pilot"]["mmlu_pro"])
    assert p["benches"]["mmlu_pro"]["acc"] == round(even / 10, 4)
    it = p["items"][0]
    assert (it["gen_tokens"], it["decode_tok_s"], it["finish"], it["capped"]) == (
        120, 31.5, "stop", False)
    assert "cap check: thinking on, max_tokens 64" in row["progress"]
    # the pilot is not the row
    assert not (row_dir(mid) / dm.OUT_NAME).exists()


def test_a_stopped_battery_asks_only_the_rest_and_reuses_the_pilot(svc, fake):
    mid = register(fake, "Qwen3.6 phone")
    assert queue(mid, "pilot")["status"] == "done"
    n0 = len(fake.requests)
    fake.stop_after = fake.answered + 100
    row = queue(mid)
    # 12q.F: where it stopped, counted — and how many the next run asks
    assert row["status"] == "failed" and row["error"].endswith(
        "the server stopped answering after 130 of 596 (HTTP 503) · the answers it gave are "
        "kept: the next run asks only the other 466")
    assert len(fake.requests) - n0 == 100                     # the pilot's 30 were not asked
    fake.stop_after = None
    row = queue(mid)
    assert row["status"] == "done", row["error"]
    assert len(fake.requests) - n0 == 566
    out = json.loads((row_dir(mid) / dm.OUT_NAME).read_text())
    assert out["n"] == 596 and out["part"] == "full" and out["version"] == dm.VERSION
    assert out["composite"]["value"] is not None and len(out["composite"]["ci"]) == 2
    assert out["time_frontier"]["b"] == list(dm.BUDGETS)
    scored = dm.read_items(row_dir(mid))
    assert len(scored) == 596 and {s["bench"] for s in scored} == set(dm.BENCHES)
    assert all("text" in s and "correct" in s for s in scored)   # the question browser's


def test_thinking_on_is_a_row_of_its_own(svc, fake):
    mid = register(fake, "Qwen3.6 phone")
    row = queue(mid, "pilot", thinking=True)
    assert row["status"] == "done"
    assert all(b["chat_template_kwargs"] == {"enable_thinking": True} for b in fake.requests)
    p = json.loads((row_dir(mid, True) / dm.PILOT_NAME).read_text())
    assert p["setup"]["thinking"] is True and not (row_dir(mid) / dm.PILOT_NAME).exists()
    assert p["items"][0]["text"].startswith("<think>\nLet me think this through.\n</think>")


def test_parity_and_the_setup_without_mtp_takes_the_mtp_rows_quality(svc):
    a, b = FakeServer(), FakeServer()
    try:
        par = dm.battery()["parity"]
        items = {(bb, k): v for (bb, k), v in dm.load_items(config.DM_ITEMS).items()}
        two = {items[("mmlu_pro", k)]["text"] for k in par["mmlu_pro"][:2]}
        replies(a)
        replies(b, wrong=two)                         # two answers differ without MTP
        mtp = register(a, "Qwen3.6 phone MTP", "--spec-type mtp")
        plain = register(b, "Qwen3.6 phone")
        assert queue(mtp)["status"] == "done"
        row = queue(mtp, "parity", pair=plain)
        assert row["status"] == "done", row["error"]
        assert "parity 48/50 answers the same, 48/50 outputs identical" in row["progress"]
        rep = json.loads((row_dir(plain) / dm.PARITY_NAME).read_text())
        assert (rep["mtp"], rep["plain"], rep["passes"]) == (mtp, plain, True)
        assert json.loads((row_dir(mtp) / dm.PARITY_NAME).read_text()) == rep
        rows = {r["id"]: r for r in dm.rows(config.OUT_DIR)}
        assert rows[plain]["row"]["inherited"]["line"] == "quality from MTP run, parity 48/50"
        assert rows[plain]["row"]["composite"] == rows[mtp]["row"]["composite"]
        # three differ: its own run is needed
        replies(b, wrong=two | {items[("math", par["math"][0])]["text"]})
        b.reply = (lambda f: (lambda body: "\\boxed{7}" if "Fixture problem " + par["math"][0]
                              in body["messages"][-1]["content"] else f(body)))(b.reply)
        (row_dir(plain) / "devicemark_parity_plain_answers.jsonl").unlink()
        row = queue(mtp, "parity", pair=plain)
        assert "parity 47/50" in row["progress"] and "needs its own run" in row["progress"]
        assert plain not in {r["id"] for r in dm.rows(config.OUT_DIR)}
    finally:
        a.close()
        b.close()


def test_the_speed_test_is_the_servers_never_a_phones(svc, fake):
    mid = register(fake, "Qwen3.6 phone")
    fake.decode_tok_s = 18.25
    row = queue(mid, "speed")
    assert row["status"] == "done", row["error"]
    assert len(fake.completions) == 3                         # a warm-up, then two trials
    for c in fake.completions:
        assert len(c["prompt"]) == 128 and c["n_predict"] == 256
        assert (c["temperature"], c["ignore_eos"], c["cache_prompt"]) == (0.0, True, False)
    sp = json.loads((row_dir(mid) / dm.SPEED_NAME).read_text())
    assert sp["decode_tok_s"] == 18.25 and sp["label"] == "server: RTX 5090 + CPU experts"
    assert [t["warmup"] for t in sp["trials"]] == [True, False, False]
    assert "not a phone's speed" in row["progress"]
    assert fake.requests == []                                # no chat message asked


def test_a_speed_from_a_device_is_entered_by_a_person(svc, fake, monkeypatch):
    client = svc
    mid = register(fake, "Qwen3.6 phone")
    assert queue(mid)["status"] == "done"
    put = lambda **kw: client.put("/api/devicemark/device", json=kw)  # noqa: E731
    r = put(model=mid, tok_s=12.4, device="iPhone 17 Pro")
    assert r.status_code == 422 and "where it came from" in r.json()["detail"]
    r = put(model=mid, tok_s=12.4, device="iPhone 17 Pro",
            source="measured by a colleague, 29 Sep 2026")
    assert r.status_code == 200 and r.json()["device"]["tok_s"] == 12.4
    got = client.get("/api/devicemark").json()
    assert got["version"] == dm.VERSION and got["whose"] == dm.WHOSE
    [row] = got["rows"]
    assert row["id"] == mid and row["device"]["device"] == "iPhone 17 Pro"
    assert row["rank"] == "1" and row["server_tok_s"] is None
    assert put(model=mid, tok_s=None).json()["device"] is None
    assert client.get("/api/devicemark").json()["rows"][0]["device"] is None
    monkeypatch.setattr(config, "SUBMIT_TOKEN", "secret")
    assert put(model=mid, tok_s=1, device="d", source="s").status_code == 401


# ---------------------------------------------------------------------------
# a Hugging Face model: three lm_eval tasks, on hf
# ---------------------------------------------------------------------------

def test_a_hugging_face_models_battery_through_lm_eval(svc, monkeypatch):
    hf = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    seen = []
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: {
        "kind": "instruct", "params": 4.0e9, "vocab": 131072, "batch": 8, "need_gb": 10.0,
        "remote_code": False, "has_template": True, "kind_reason": "chat template",
        "archinfo": {"thinking": "switch", "think_end": "</think>"}})
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    monkeypatch.setattr(runner, "gen_backend", lambda: ("hf", "vLLM is not installed in this image"))
    items = dm.load_items(config.DM_ITEMS)

    def run_task(sid, cmd, lf, env, run_as, cwd, on_poll=None):
        task = cmd[cmd.index("--tasks") + 1]
        seen.append(cmd)
        out = Path(cmd[cmd.index("--output_path") + 1]) / hf.replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        b = dm.BENCH_OF[task]
        with open(out / f"samples_{task}_2026-09-29T10-00-00.jsonl", "w") as fh:
            for bb, k in dm.keys_for("full"):
                if bb == b:
                    ans = {"ifeval": "a note without any commas", "mmlu_pro": "\\boxed{A}",
                           "math": "\\boxed{1/2}"}[b]
                    fh.write(json.dumps({"doc": {"bench": b, "key": k,
                                                 "text": items[(b, k)]["text"]},
                                         "resps": [[ans]]}) + "\n")
        return 0
    monkeypatch.setattr(runner, "_run_task", run_task)
    sid = db.add(hf, "instruct", "devicemark", ME, "", part="full")
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "done", row["error"]
    assert [c[c.index("--tasks") + 1] for c in seen] == config.DM_TASKS
    for c in seen:
        assert c[c.index("--model")] and c[c.index("--model") + 1] == "hf"
        assert "--apply_chat_template" in c and "max_gen_toks=4096" in c
        assert "enable_thinking=False" in c[c.index("--model_args") + 1]
        assert str(config.DM_TASKS_DIR) in c
    out = json.loads((row_dir(hf) / dm.OUT_NAME).read_text())
    assert out["n"] == 596 and out["setup"]["runtime"].startswith("hf transformers")
    assert out["benches"]["math"]["acc"] == 1.0 and out["benches"]["ifeval"]["acc"] == 1.0
    assert "composite" in row["progress"]
    # the tasks lm_eval was handed: each item's prompt, greedy, capped at 4,096
    y = (config.DM_TASKS_DIR / "dm_math.yaml").read_text()
    assert "max_gen_toks: 4096" in y and "do_sample: false" in y and "until: []" in y


def test_a_logged_answers_length_is_counted_again_and_the_cap_found():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "dm_math_0shot" / "org__m"
        d.mkdir(parents=True)
        (d / "samples_dm_math_2026-09-29T10-00-00.jsonl").write_text("\n".join(json.dumps(
            {"doc": {"bench": "math", "key": k}, "resps": [[t]]}) for k, t in
            [("a", "short \\boxed{1}"), ("b", "long " * 5000)]) + "\n")
        recs = dm.records_from_samples(Path(tmp), lambda t: len(t.split()))
        assert [(r["gen_tokens"], r["capped"]) for r in recs] == [(2, False), (5000, True)]


# ---------------------------------------------------------------------------
# the cap counts the thinking: llama-server's reply, as recorded on the server
# ---------------------------------------------------------------------------

CAPPED = Path(__file__).resolve().parent / "fixtures" / "devicemark" / \
    "llama_server_capped_thinking.json"


def test_a_capped_thinking_reply_as_the_server_sent_it(svc, fake, monkeypatch):
    from service import devicemark as svc_dm
    fx = json.loads(CAPPED.read_text(encoding="utf-8"))
    thought = fx["choices"][0]["message"]["reasoning_content"]
    assert "completion_tokens_details" not in fx["usage"]      # no split in the usage
    fake.raw_reply = fx
    mid = register(fake, "Qwen3.6 phone MTP", "--spec-type mtp --port 8094")
    rec = served.get(mid)
    r = svc_dm.ask(rec, "a MATH item", svc_dm.settings(True, 64))
    assert (r["finish"], r["gen_tokens"], r["capped"], r["answer"], r["thinking_chars"]) == (
        "length", 64, True, "", 215)
    # the thinking counted with the server's own /tokenize; the answer is the rest
    assert fake.tokenized == [thought]
    words = len(thought.split())                               # the fake's tokenizer
    assert (r["think_tokens"], r["think_tokens_from"], r["answer_tokens"]) == (
        words, "/tokenize", 64 - words)
    # a capped reply with no box is no answer: wrong, and in the denominator
    items = dm.load_items(config.DM_ITEMS)
    k = dm.keys_for("pilot")[-1]
    [s] = dm.score_items([{"bench": k[0], "key": k[1], **r}], items)
    assert (s["answered"], s["correct"], s["parsed"]) == (False, False, None)
    # the pilot's check reads it as the cap counting the thinking
    cc = svc_dm.cap_check(rec, items, dm.keys_for("pilot"))
    assert cc["ok"] is True and cc["think_tokens_from"] == "/tokenize"
    # a server that can't tokenize: the characters, labelled as characters
    monkeypatch.setattr(svc_dm, "count_tokens", lambda rec, text: None)
    r = svc_dm.ask(rec, "a MATH item", svc_dm.settings(True, 64))
    assert (r["think_tokens"], r["think_tokens_from"], r["answer_tokens"],
            r["thinking_chars"]) == (None, None, None, 215)
