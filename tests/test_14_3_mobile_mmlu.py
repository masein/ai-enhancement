"""14.3: Mobile-MMLU-Pro with our own answer key. The file is never in the
repo (CC BY-ND 4.0, and our key is a derivative): every test here uses
invented rows with its columns (tests/fixtures/mmp_invented.csv), pinned as
the real file is. The key's rules; its labellers — never local, in-house, a
model scored on this set, or another labeller's maker; the run, sent only
from Start and carried on after a stop; the scoring on the key as it stands;
the paper's three checks; the portal's file for public HF models only. No
model runs, and nothing calls OpenRouter or Hugging Face: a fake GPU, a fake
labeller, picks written in."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import time
from pathlib import Path

import pytest

import fetch_data
import mobile_mmlu as mmp
from conftest import make_service
from service import ai_models, config, contamination, db, llm, llm_poller, mmp_key, runner, served

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mmp_invented.csv"
REPO = Path(__file__).resolve().parent.parent
STAMP = "2026-10-01T10-00-00.000000"
# the invented rows' right answers
RIGHT = {"inv00001": "A", "inv00002": "B", "inv00003": "B", "inv00004": "C", "inv00005": "C",
         "inv00006": "A", "inv00007": "B", "inv00008": "B", "inv00009": "D", "inv00010": "B",
         "inv00011": "C", "inv00012": "A"}


def pin_fixture(monkeypatch) -> None:
    """the manifest's pin, moved to the invented file"""
    real, raw = mmp.manifest, FIXTURE.read_bytes()

    def pinned():
        m = real()
        m["file"].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), n=12)
        return m
    monkeypatch.setattr(mmp, "manifest", pinned)
    mmp._rows.clear()
    mmp._keyc.clear()


def put_invented(monkeypatch, d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "mobile-mmlu-pro.csv").write_bytes(FIXTURE.read_bytes())
    monkeypatch.setenv("MMP_DIR", str(d))
    monkeypatch.setattr(config, "MMP_DIR", d)
    pin_fixture(monkeypatch)
    return d


@pytest.fixture
def invented(tmp_path, monkeypatch):
    """Mobile-MMLU-Pro on this "server": the invented rows"""
    return put_invented(monkeypatch, tmp_path / "data" / "mobile_mmlu_pro")


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    put_invented(monkeypatch, config.MMP_DIR)
    yield client, appmod
    client.__exit__(None, None, None)


def fake_pin(model_id: str) -> dict:
    return {"kind": "openrouter", "id": model_id, "version": model_id + "-20260901",
            "name": model_id.split("/")[-1], "provider": "prov/x", "provider_name": "Prov",
            "precision": "unknown", "price_in": 1.0, "price_out": 5.0}


def label(slot: str, got: dict[str, tuple[str | None, bool]], model_id: str | None = None):
    cur = mmp_key.current()[slot]
    mmp.add_labels(slot, {q: {"letter": L, "now": now, "model": model_id or cur["id"],
                              "version": cur["version"] if model_id is None else model_id,
                              "at": 0} for q, (L, now) in got.items()})


# ---------------------------------------------------------------------------
# the data: pinned, never committed
# ---------------------------------------------------------------------------

def test_the_80_fields_fall_in_the_papers_9_categories_once_each():
    m = mmp.manifest()
    fields = [f for fs in mmp.CATEGORIES.values() for f in fs]
    assert len(mmp.CATEGORIES) == 9 and len(fields) == 80 == len(set(fields))
    assert set(fields) == set(m["fields"]) and sum(m["fields"].values()) == m["file"]["n"] == 9497
    assert mmp.FIELD_CATEGORY["first_aid"] == "Health & Safety"
    assert mmp.FIELD_CATEGORY["marketing__and_sales_strarigies"] == "Business & Career"


def test_the_file_is_pinned_by_revision_and_hash_and_never_committed():
    f, m = mmp.manifest()["file"], mmp.manifest()
    assert f["committed"] is False and len(f["sha256"]) == 64 and f["bytes"] == 16_899_380
    assert m["revision"] in f["url"] and f["url"].startswith(
        "https://huggingface.co/datasets/MBZUAI-LLM/Mobile-MMLU-Pro/resolve/")
    assert m["licence"] == "CC BY-ND 4.0" and "CC BY-NC-ND 4.0" in m["not_used"]


def test_no_mobile_mmlu_pro_text_is_in_the_repo_outside_the_invented_rows():
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True,
                             check=True).stdout.decode().split("\0")
    # the file's header, built here so that this test's own text never holds it
    head = ",".join(["question_id", "Question", "A", "B", "C", "D", "Field"])
    pinned = mmp.manifest()["file"]["sha256"]
    holders = []
    for name in filter(None, tracked):
        p = REPO / name
        if not p.is_file() or p.stat().st_size > 30_000_000:
            continue
        raw = p.read_bytes()
        assert not name.endswith("mobile-mmlu-pro.csv"), name
        assert raw[:4096].find(head.encode()) == -1 or name.endswith(
            "tests/fixtures/mmp_invented.csv"), name
        if p.stat().st_size == mmp.manifest()["file"]["bytes"]:
            assert hashlib.sha256(raw).hexdigest() != pinned, name
        if head.encode() in raw:
            holders.append(name)
    assert holders == ["tests/fixtures/mmp_invented.csv"]
    rows = list(csv.DictReader(io.StringIO(FIXTURE.read_text(encoding="utf-8"))))
    assert list(rows[0]) == head.split(",") and all(r["question_id"].startswith("inv") for r in rows)


def test_without_the_file_the_part_says_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv("MMP_DIR", str(tmp_path / "empty"))
    assert mmp.available().startswith("Mobile-MMLU-Pro isn't on this server: fetch it with "
                                      "the data step (scripts/fetch_data.py)")
    assert mmp.load() == []


def test_a_changed_file_is_refused(invented):
    (invented / "mobile-mmlu-pro.csv").write_bytes(FIXTURE.read_bytes().replace(b"butter",
                                                                                 b"bUtter"))
    with pytest.raises(ValueError, match="isn't the pinned file"):
        mmp.load()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_the_data_step_fetches_it_from_its_revision_and_checks_it(tmp_path, monkeypatch, invented):
    raw = FIXTURE.read_bytes()
    item = next(i for i in fetch_data.wanted() if i["name"] == "Mobile-MMLU-Pro")
    assert item["dest"] == invented / "mobile-mmlu-pro.csv"
    item = {**item, "dest": tmp_path / "out" / "mobile-mmlu-pro.csv"}
    asked = []
    line = fetch_data.fetch(item, opener=lambda req, timeout: asked.append(req.full_url)
                            or _Resp(raw))
    assert line.startswith("Mobile-MMLU-Pro: fetched to ") and asked == [item["url"]]
    with pytest.raises(RuntimeError, match="not fetched — got"):
        fetch_data.fetch({**item, "dest": tmp_path / "x" / "m.csv"},
                         opener=lambda req, timeout: _Resp(raw[:-1]))


def test_it_is_in_the_contamination_index_once_fetched(tmp_path, monkeypatch):
    before = contamination.BenchmarkIndex(tmp_path / "nothing").refresh().n_pinned
    put_invented(monkeypatch, tmp_path / "mmp")
    ix = contamination.BenchmarkIndex(tmp_path / "nothing2").refresh()
    assert ix.n_pinned == before + 12


# ---------------------------------------------------------------------------
# how it is asked
# ---------------------------------------------------------------------------

def test_it_is_asked_as_the_board_asks_mmlu_and_the_authors_ask_a_letter(invented):
    import gguf_data
    q = mmp.by_id()["inv00006"]
    p = mmp.mc_prompt(q)
    assert p.startswith("The following are multiple choice questions (with answers) about mobile "
                        "customization.\n\nWhich setting changes")
    assert p.endswith("\nA. Display font size\nB. Ringtone volume\nC. Location services\n"
                      "D. Keyboard vibration\nAnswer:")
    assert mmp.MMLU_HEAD == gguf_data.MMLU_HEAD
    # the GGUF's file asks the very same text
    doc = {"question": q["question"], "choices": [q[L] for L in "ABCD"],
           "subject": mmp.subject(q["field"]), "answer": 0}
    assert gguf_data.mmlu_task(doc)["question"] == p
    # the authors' own prompt, its full-width colon too
    assert mmp.ask_prompt(q) == ("Answer the following multiple choice question and provide only "
                                 "letter of the correct answer.\n" + q["question"] + "\nA. Display "
                                 "font size\nB. Ringtone volume\nC. Location services\nD. Keyboard "
                                 "vibration\nAnswer：\n")


def test_the_run_task_never_holds_the_key(invented, tmp_path):
    mmp.write_key({"items": {"inv00001": {"key": "A", "decision": "agreed"}}})
    d = mmp.build_tasks(tmp_path / "tasks")
    yaml = (d / "mobile_mmlu_pro.yaml").read_text()
    assert "output_type: multiple_choice" in yaml and "doc_to_target: 0" in yaml
    assert 'doc_to_choice: ["A", "B", "C", "D"]' in yaml
    docs = [json.loads(x) for x in (d / "mobile_mmlu_pro.jsonl").read_text().splitlines()]
    assert len(docs) == 12 and set(docs[0]) == {"id", "field", "prompt"}
    ask = [json.loads(x) for x in (d / "mobile_mmlu_pro_ask.jsonl").read_text().splitlines()]
    assert ask[0]["prompt"].startswith(mmp.ASK_HEAD)


@pytest.mark.parametrize("text,want", [
    ("B", "B"), ("B.", "B"), ("(C) because it is", "C"), ("**D**", "D"),
    ("The answer is A.", "A"), ("<think>A or B?</think>\nB", "B"), ("Option C", "C"),
    ("A balanced diet helps", None), ("A or B", None), ("", None)])
def test_a_served_models_letter_is_read(text, want):
    assert mmp.letter_of(text) == want


@pytest.mark.parametrize("text,want", [
    ('{"answer": "c", "depends_on_now": true}', {"letter": "C", "now": True}),
    ('thinking… {"answer": "B", "depends_on_now": "no"}', {"letter": "B", "now": False}),
    ('```json\n{"answer": "(D)", "depends_on_now": false}\n```', {"letter": "D", "now": False}),
    ('{"answer": "E", "depends_on_now": false}', {"letter": None, "now": False}),
    ("A", {"letter": "A", "now": False})])
def test_a_labellers_answer_is_read(text, want):
    assert mmp.parse_label(text) == want


# ---------------------------------------------------------------------------
# the key's rules
# ---------------------------------------------------------------------------

def L(letter, now=False):
    return {"letter": letter, "now": now}


@pytest.mark.parametrize("a,b,c,key,decision", [
    (L("B"), L("B"), None, "B", "agreed"),                     # both agree: their letter
    (L("B", True), L("B", True), None, None, "time"),          # both flag now: dropped
    (L("A", True), L("C", True), None, None, "time"),
    (L("B", True), L("B"), None, "B", "agreed"),               # one flag is not both
    (L("A"), L("C"), None, None, "waiting"),                   # a split waits for the third
    (L("A"), L("C"), L("C"), "C", "settled"),                  # the third settles it, two of three
    (L("A"), L("C"), L("A"), "A", "settled"),
    (L("A"), L("C"), L("D"), None, "split"),                   # all three differ: dropped
    (L(None), L("C"), L("C"), "C", "settled"),                 # no letter never agrees
    (L(None), L(None), L("C"), None, "split"),
    (None, L("C"), None, None, "waiting")])
def test_the_key_rules(a, b, c, key, decision):
    got = mmp.decide(a, b, c)
    assert (got["key"], got["decision"]) == (key, decision) and got["reason"]


def test_the_key_keeps_each_questions_labels_and_counts_them_per_category(invented):
    rows = mmp.load()
    cur = {s: {"id": s, "version": f"{s}-v1"} for s in mmp.SLOTS}
    lab = {"first": {q: {"letter": RIGHT[q], "now": False, "version": "first-v1"} for q in RIGHT},
           "second": {q: {"letter": RIGHT[q], "now": False, "version": "second-v1"} for q in RIGHT},
           "third": {}}
    lab["first"]["inv00003"]["letter"] = "C"                   # settled by the third
    lab["third"]["inv00003"] = {"letter": "B", "now": False, "version": "third-v1"}
    lab["first"]["inv00007"]["letter"] = "C"                   # three-way split
    lab["second"]["inv00007"]["letter"] = "D"
    lab["third"]["inv00007"] = {"letter": "A", "now": False, "version": "third-v1"}
    for s in ("first", "second"):                              # time-sensitive
        lab[s]["inv00008"]["now"] = True
    lab["second"]["inv00009"]["version"] = "second-v0"         # another labeller's: waits
    k = mmp.build_key(rows, lab, cur)
    c = k["counts"]["all"]
    assert (c["agreed"], c["settled"], c["split"], c["time"], c["waiting"]) == (8, 1, 1, 1, 1)
    assert c["kept"] == 9 and c["questions"] == 12
    it = k["items"]["inv00003"]
    assert it["key"] == "B" and it["decision"] == "settled" and it["labels"]["third"] == L("B")
    assert "two of three" in it["reason"] and it["category"] == "Home & Family"
    assert k["counts"]["by_category"]["Health & Safety"]["questions"] == 1
    assert k["counts"]["by_category"]["Home & Family"]["kept"] == 1     # 03 kept, 07 split
    assert len(k["version"]) == 10 and mmp.kept(k["items"])["inv00001"] == "A"


# ---------------------------------------------------------------------------
# the labellers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id,why", [
    ("local", "can't be local"),
    ("qwen/qwen3.6-35b-a3b", "in-house"),
    ("Qwen/Qwen3-6-27B", "in-house"),
    ("served/openrouter-z-ai-glm-5.3", "in-house"),
    ("google/gemini-3.1-flash", "the second labeller is Google too")])
def test_an_in_house_or_local_model_cant_label_the_key(svc, model_id, why):
    client, _ = svc
    assert why in mmp_key.refused("first", model_id)
    r = client.post("/api/ai/labellers/first", json={"model": model_id, "by": "masein"})
    assert r.status_code == 422 and why in r.json()["detail"]


def test_a_model_scored_on_this_set_cant_label_it(svc):
    d = config.OUT_DIR / "served__openrouter-x-ai-grok-5"
    d.mkdir(parents=True)
    (d / "model_meta.json").write_text(json.dumps({"model": "served/openrouter-x-ai-grok-5",
                                                   "served": {"pin": {"model": "x-ai/grok-5"}}}))
    (d / mmp.PRED_FILE).write_text(json.dumps({"predictions": {}}))
    assert "has a Mobile-MMLU-Pro score on the board" in mmp_key.refused("third", "x-ai/grok-5")
    assert not mmp_key.refused("third", "x-ai/grok-6")


def test_the_defaults_are_the_reasoning_labs_two_and_a_third_maker():
    d = mmp.DEFAULT_LABELLERS
    assert (d["first"]["version"], d["first"]["provider"]) == ("openai/gpt-6-sol-20260922",
                                                               "openai/flex")
    assert d["second"]["version"] == "google/gemini-3.1-pro-preview-20260219"
    assert {ai_models.maker(x["id"]) for x in d.values()} == {"OpenAI", "Google", "Anthropic"}


# ---------------------------------------------------------------------------
# the run: the dry run first, Start, Stop, carry on
# ---------------------------------------------------------------------------

def test_the_dry_run_prints_each_labellers_tokens_and_cost(capsys):
    assert mmp.main(["--dry-run", "--stats"]) == 0
    out = capsys.readouterr().out
    assert "a dry run, nothing is sent (the published mean lengths)" in out
    assert "GPT-6 Sol: 9,497 questions · 4,708,138 tokens in (about 496 each)" in out
    assert "$15.63 at $1 in / $5 out per million" in out
    assert "Gemini 3.1 Pro: 9,497 questions" in out and "$40.04 at $1 in / $6 out" in out
    assert "Claude Sonnet 5.5: 950 questions" in out and "10% is a guess" in out


def drain(timeout: float = 15.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        llm_poller.tick()
        if not mmp_key.pending():
            return
        time.sleep(0.05)
    pytest.fail("the labellers' batches never finished")


def test_labelling_sends_nothing_before_start_and_carries_on_after_a_stop(svc, monkeypatch):
    client, appmod = svc
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(ai_models, "pin", fake_pin)
    monkeypatch.setattr(llm, "_http", lambda *a, **k: pytest.fail("nothing calls OpenRouter"))
    monkeypatch.setattr(ai_models, "drifted", lambda pin: "")
    asked = []
    wrong = {("openai", "inv00003"): "C", ("openai", "inv00007"): "C", ("google", "inv00007"): "D",
             ("anthropic", "inv00007"): "A"}

    def complete(self, row):
        qid = row["custom_id"].split(":", 2)[2]
        org = self.model.split("/")[0]
        asked.append((org, qid))
        reply = {"answer": wrong.get((org, qid), RIGHT[qid]), "depends_on_now": qid == "inv00008"}
        return {"custom_id": row["custom_id"], "text": json.dumps(reply), "error": "",
                "attempts": 1, "finish_reason": "stop"}
    monkeypatch.setattr(mmp_key.LabellerChat, "_complete", complete)
    # the dry run: what it would take, and nothing sent
    page = client.get("/api/mobile-mmlu/key").json()
    est = page["estimate"]["slots"]
    assert est["first"]["questions"] == est["second"]["questions"] == 12 and not asked
    assert [x["slot"] for x in page["labellers"]] == ["first", "second", "third"]
    # held: a batch out while the run is stopped sends nothing
    mmp_key.stop("masein")
    db.ai_set("labeller:first", fake_pin("openai/gpt-6-sol"), "masein")
    held = mmp_key._submit("first", mmp_key.chosen("first"), ["inv00001", "inv00002"], "masein")
    time.sleep(0.3)
    assert not asked and mmp_key.pending()[0]["batch_id"] == held
    # Start: the stop lifted, the held batch carries on, and the rest is sent once
    r = client.post("/api/mobile-mmlu/key/start", json={"by": "masein"})
    assert r.status_code == 200, r.text
    drain()
    per = {o: sorted(q for x, q in asked if x == o) for o in ("openai", "google", "anthropic")}
    assert per["openai"] == per["google"] == sorted(RIGHT)
    assert per["anthropic"] == ["inv00003", "inv00007"]           # the splits alone
    c = mmp.current_key()["counts"]["all"]
    assert (c["agreed"], c["settled"], c["split"], c["time"], c["waiting"]) == (9, 1, 1, 1, 0)
    # Start again: nothing left, nothing sent
    n = len(asked)
    assert client.post("/api/mobile-mmlu/key/start", json={"by": "masein"}).status_code == 200
    drain()
    assert len(asked) == n


def test_start_is_refused_until_the_labellers_can_label(svc, monkeypatch):
    client, _ = svc
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_http", lambda *a, **k: pytest.fail("nothing calls OpenRouter"))
    db.ai_set("labeller:second", fake_pin("openai/gpt-6-luna"), "masein")
    r = client.post("/api/mobile-mmlu/key/start", json={"by": "masein"})
    assert r.status_code == 409 and "another maker" in r.json()["detail"]


# ---------------------------------------------------------------------------
# scoring: the key as it stands, and the paper's checks
# ---------------------------------------------------------------------------

def sit_picks(model_dir: Path, picks: dict[str, str], served_text: bool = False) -> None:
    """what lm_eval (four log-likelihoods) or a server (a letter) logs"""
    d = model_dir / "mobile_mmlu_pro_0shot" / "x"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"samples_mobile_mmlu_pro_{STAMP}.jsonl", "w") as fh:
        for i, (q, letter) in enumerate(picks.items()):
            fr = ([f"The answer is {letter}."] if served_text else
                  [[-1.0 if L == letter else -3.0, False] for L in "ABCD"])
            fh.write(json.dumps({"doc_id": i, "doc": {"id": q, "prompt": "…"},
                                 "filtered_resps": fr, "doc_hash": q}) + "\n")
    mid = model_dir.name.replace("__", "/", 1)
    (d / f"results_{STAMP}.json").write_text(json.dumps({
        "results": {"mobile_mmlu_pro": {"alias": "mobile_mmlu_pro", "acc,none": 0.25}},
        "group_subtasks": {"mobile_mmlu_pro": []}, "n-shot": {"mobile_mmlu_pro": 0},
        "n-samples": {"mobile_mmlu_pro": {"original": 12, "effective": 12}},
        "higher_is_better": {"mobile_mmlu_pro": {"acc": True}},
        "config": {"model": "hf", "model_args": f"pretrained={mid}"}, "date": 1790200000.0}))
    if not (model_dir / "model_meta.json").exists():
        (model_dir / "model_meta.json").write_text(json.dumps({"model": mid, "kind": "instruct"}))


SPLIT = "all three differ (A and C, then D)"


def a_key(drop=("inv00007", "inv00008")) -> dict:
    items = {q: {"key": None if q in drop else L_, "decision": "split" if q in drop else "agreed",
                 "reason": SPLIT if q in drop else f"both chose {L_}",
                 "category": mmp.by_id()[q]["category"], "labels": {}}
             for q, L_ in RIGHT.items()}
    k = {"items": items, "counts": mmp.counts(items), "version": mmp.key_version(items)}
    mmp.write_key(k)
    return k


def test_picks_are_scored_on_the_kept_questions_with_n_and_categories(invented, tmp_path):
    a_key()
    picks = dict(RIGHT, inv00001="B", inv00002="A")             # two wrong, of ten kept
    sit_picks(tmp_path / "org__m", picks)
    got = mmp.collect(tmp_path / "org__m")
    assert got["how"] == "log-likelihood" and got["predictions"] == picks
    s = mmp.score(got)
    assert (s["n"], s["of"], s["correct"]) == (10, 10, 8) and s["acc"] == pytest.approx(0.8)
    assert s["by_category"]["Health & Safety"] == {"acc": 0.0, "n": 1, "correct": 0, "se": 0.0}
    # a served model's letters are read from what it wrote
    sit_picks(tmp_path / "served__s", RIGHT, served_text=True)
    assert mmp.score(mmp.collect(tmp_path / "served__s"))["acc"] == 1.0
    # the key moves: the same picks, scored again
    a_key(drop=("inv00001", "inv00002"))
    assert mmp.score(got)["acc"] == pytest.approx(1.0)


@pytest.mark.parametrize("ours,provisional", [
    ({"Qwen/Qwen2.5-3B-Instruct": 0.62, "meta-llama/Llama-3.2-3B-Instruct": 0.40,
      "google/gemma-2-2b-it": 0.33}, False),
    ({"Qwen/Qwen2.5-3B-Instruct": 0.62, "meta-llama/Llama-3.2-3B-Instruct": 0.40,
      "google/gemma-2-2b-it": 0.35}, True),                     # 3.8 points off
    ({"Qwen/Qwen2.5-3B-Instruct": 0.62, "meta-llama/Llama-3.2-3B-Instruct": 0.40}, True)])
def test_the_key_is_provisional_until_all_three_paper_checks_land_within_3_points(ours,
                                                                                 provisional):
    got = mmp.paper_checks({m: {"acc": v, "n": 9000} for m, v in ours.items()})
    assert got["provisional"] is provisional and got["within"] == 3.0
    assert [r["paper"] for r in got["rows"]] == [60.6, 42.0, 31.2]


def test_a_run_of_the_part_asks_every_question_and_its_cell_is_on_our_key(svc, monkeypatch):
    client, appmod = svc
    a_key()
    from test_14_1_mab_text import fake_gpu
    fake_gpu(monkeypatch)
    seen = []

    def run(sid, cmd, *a, **k):
        seen.append(cmd)
        model_dir = Path(cmd[cmd.index("--output_path") + 1]).parent
        inc = Path(cmd[cmd.index("--include_path") + 1])
        docs = [json.loads(x) for x in (inc / "mobile_mmlu_pro.jsonl").read_text().splitlines()]
        sit_picks(model_dir, {d["id"]: RIGHT[d["id"]] if d["id"] != "inv00001" else "D"
                              for d in docs})
        return 0
    monkeypatch.setattr(runner, "_run_task", run)
    sid = client.post("/api/submissions", json={"hf_id": "org/chat-2b", "suite": "mobile",
                                                "part": "mmlu", "kind": "instruct"}).json()["id"]
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done", db.get(sid)
    cmd = seen[0]
    assert cmd[cmd.index("--tasks") + 1] == "mobile_mmlu_pro" and cmd[cmd.index("--num_fewshot")
                                                                       + 1] == "0"
    assert "--apply_chat_template" not in cmd           # as the paper ran lm-evaluation-harness
    appmod._cache.update(key=None, payload=None, at=0.0)
    data = client.get("/api/results").json()
    cell = data["cells"]["mobile_mmlu_pro"]["org/chat-2b"]
    assert cell["v"] == pytest.approx(0.9) and cell["n"] == 10      # lm_eval's own acc unseen
    m = next(x for x in data["models"] if x["id"] == "org/chat-2b")
    assert m["mmp"]["by_category"] and m["avg"] is None
    assert "mobile_mmlu_pro" not in data["required"] and data["mmp"]["provisional"] is True
    # a base model may sit it: the letters are scored, not written
    r = client.post("/api/submissions", json={"hf_id": "org/base-1b", "suite": "mobile",
                                              "part": "mmlu", "kind": "base"})
    assert r.status_code == 200, r.text


def test_the_part_is_refused_without_the_file(svc, monkeypatch, tmp_path):
    client, _ = svc
    monkeypatch.setattr(config, "MMP_DIR", tmp_path / "none")
    monkeypatch.delenv("MMP_DIR", raising=False)
    r = client.post("/api/submissions", json={"hf_id": "org/c", "suite": "mobile",
                                              "part": "mmlu", "kind": "instruct"})
    assert r.status_code == 422 and "fetch it with the data step" in r.json()["detail"]


def test_each_part_says_what_it_takes_and_a_served_model_its_cost(svc, monkeypatch):
    client, _ = svc
    got = client.get("/api/mobileaibench/estimate", params={"model": "org/chat-2b"}).json()
    assert got["parts"]["mmlu"]["answers"] == 12 and got["parts"]["mmlu"]["judgements"] == 0
    rec = {"id": "served/x", "name": "x", "thinking": "off", "base_url": "http://x/v1",
           "how": "", "based_on": "", "via": "openrouter",
           "pin": {"model": "org/x", "price_in": 1.0, "price_out": 2.0}}
    monkeypatch.setattr(served, "model_dir", lambda r: config.OUT_DIR / "served__x")
    est = served.estimate(rec, "mobile", part="mmlu")
    assert est["n"] == 12 and est["tokens_in"] > 12 * 30


def test_a_labeller_scored_later_says_labelled_the_key_and_is_never_ranked(svc):
    client, appmod = svc
    a_key()
    sit_picks(config.OUT_DIR / "org__labeller-9b", RIGHT)
    mmp.collect(config.OUT_DIR / "org__labeller-9b")
    sit_picks(config.OUT_DIR / "org__other-1b", RIGHT)
    mmp.collect(config.OUT_DIR / "org__other-1b")
    db.ai_set("labeller:third", fake_pin("org/labeller-9b"), "masein")
    appmod._cache.update(key=None, payload=None, at=0.0)
    data = client.get("/api/results").json()
    assert "org/labeller-9b" in data["mmp"]["labelled"]
    assert "org/labeller-9b" not in data["cells"]["mobile_mmlu_pro"]
    assert "org/other-1b" in data["cells"]["mobile_mmlu_pro"]


def test_the_portals_file_is_offered_for_public_hf_models_only(svc, monkeypatch):
    client, appmod = svc
    a_key()
    for mid in ("org/public-1b", "org/private-1b", "served/openrouter-org-x",
                "teraformer/qwen3.6-35b-a3b-k4-lda", "org/ckpt-1b"):
        d = config.OUT_DIR / mid.replace("/", "__")
        sit_picks(d, RIGHT)
        mmp.collect(d)
    (config.OUT_DIR / "org__ckpt-1b" / "model_meta.json").write_text(json.dumps(
        {"model": "org/ckpt-1b", "source": "artifact"}))
    hub = []
    monkeypatch.setattr(appmod, "_hub_public", lambda m: hub.append(m) or m != "org/private-1b")
    r = client.get("/api/mobile-mmlu/predictions", params={"model": "org/public-1b"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    lines = r.text.splitlines()
    assert lines[0] == "question_id,predicted_answer" and len(lines) == 13
    assert "attachment" in r.headers["content-disposition"]
    for mid, why in (("org/private-1b", "public on Hugging Face"),
                     ("served/openrouter-org-x", "this one is served"),
                     ("teraformer/qwen3.6-35b-a3b-k4-lda", "in-house"),
                     ("org/ckpt-1b", "in-house")):
        r = client.get("/api/mobile-mmlu/predictions", params={"model": mid})
        assert r.status_code == 403 and why in r.json()["detail"], mid
    assert hub == ["org/public-1b", "org/private-1b"]          # asked only of the rest
    # no endpoint hands out the key
    paths = [getattr(x, "path", "") for x in appmod.app.routes]
    assert not [p for p in paths if "mobile-mmlu" in p and "key" in p and "download" in p]
    assert "inv00001,A" not in client.get("/api/mobile-mmlu/key").text


def test_the_question_browser_credits_the_authors_and_shows_our_key(svc):
    client, appmod = svc
    a_key()
    sit_picks(config.OUT_DIR / "org__m", dict(RIGHT, inv00001="C"))
    mmp.collect(config.OUT_DIR / "org__m")
    got = client.get("/api/questions/mobile_mmlu_pro", params={"limit": 50}).json()
    assert got["kind"] == "mmp" and "CC BY-ND 4.0" in got["meta"]["licence"]
    assert "MBZUAI" in got["meta"]["source"] and "our own answer key" in got["meta"]["source"]
    rows = {r["id"]: r for r in got["rows"]}
    assert rows and got["listed"] + got["other"] == 12       # the listed half, as everywhere
    for qid, row in rows.items():
        res, pick = row["results"]["org/m"], "C" if qid == "inv00001" else RIGHT[qid]
        if qid in ("inv00007", "inv00008"):                  # dropped: never right or wrong
            assert row["answer_idx"] is None and res["ok"] is None
            assert res["verdict"] == f"picked {pick} · not scored: dropped as split ({SPLIT})"
        else:
            assert row["answer_idx"] == "ABCD".index(RIGHT[qid])
            assert res["verdict"] == f"picked {pick} · our key {RIGHT[qid]}"
            assert res["ok"] is (pick == RIGHT[qid])
            assert row["agreement"] == f"kept by agreement (both chose {RIGHT[qid]})"


def test_never_a_training_target_and_never_in_the_avg(invented, tmp_path):
    import diagnose as dx
    a_key()
    sit_picks(tmp_path / "org__m", RIGHT)
    assert "mobile_mmlu_pro" not in dx.diagnose_model(tmp_path / "org__m")["tasks"]
    import report_lm_eval as rep
    req, _ = rep.required_tasks(["mmlu", "mobile_mmlu_pro"])
    assert "mobile_mmlu_pro" not in req


def test_a_gguf_dataset_is_built_from_the_kept_questions_only_when_asked(invented, tmp_path):
    import gguf_bench as gb
    import gguf_data
    with pytest.raises(ValueError, match="no answer key yet"):
        mmp.gguf_docs()
    a_key()
    docs = mmp.gguf_docs()
    assert len(docs) == 10 and {d["answer"] for d in docs} <= {0, 1, 2, 3}
    assert gb.BENCHMARKS["mobile_mmlu_pro"]["apart"]
    out = gguf_data.build(tmp_path / "gg", only=["mobile_mmlu_pro"])
    entry = out["benchmarks"]["mobile_mmlu_pro"] if "benchmarks" in out else json.loads(
        (tmp_path / "gg" / "manifest.json").read_text())["benchmarks"]["mobile_mmlu_pro"]
    assert entry["n"] == 10 and entry["key"] == mmp.current_key()["version"]
