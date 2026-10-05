"""12o.2 in the service: every benchmark's questions, browsed — 50 a page,
each with every chosen model's result on it, from the samples lm_eval logged
and the marks beside them. What is listed is what may be listed today: an
lm_eval benchmark's diagnose half, the exam's, Everyday's practice half; the
other half is a count and a line, and the owner's audit, logged; GPQA never.
Nothing browsed reaches anything that writes. Fixtures only."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import diagnose as dx
import everyday as ev
import exam_build as eb
from conftest import make_service
from service import config, db, questions
from test_12n2 import sit_gpqa

OWNER = "masein"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "BOARD_OWNER", OWNER)
    yield client
    client.__exit__(None, None, None)


def page(client, task, **kw):
    r = client.get(f"/api/questions/{task}", params=kw)
    assert r.status_code == 200, r.text
    return r.json()


def every_row(client, task, **kw):
    out, off = [], 0
    while True:
        p = page(client, task, offset=off, limit=200, **kw)
        out += p["rows"]
        off += 200
        if off >= p["total"]:
            return out, p


# ---------------------------------------------------------------------------
# what is listed
# ---------------------------------------------------------------------------

def test_every_benchmark_on_file_is_listed_and_gpqa_never(svc):
    client = svc
    sit_gpqa(config.OUT_DIR / "fx__good-750m")
    tasks = [t["task"] for t in client.get("/api/questions").json()["tasks"]]
    assert {"mmlu", "hellaswag", "arc_easy", "winogrande", "everyday", "exam_law"} <= set(tasks)
    assert not [t for t in tasks if t.startswith("gpqa") or t.startswith("everyday_")]
    r = client.get("/api/questions/gpqa_diamond_cot_zeroshot")
    assert r.status_code == 403 and r.json()["detail"] == questions.NOT_LISTED
    r = client.post("/api/questions/gpqa_diamond_cot_zeroshot/audit",
                    json={"by": OWNER, "confirm": True})
    assert r.status_code == 403
    assert client.get("/api/questions/nonesuch").status_code == 404


def half_rule(task: str, row: dict) -> str:
    kind = questions.kind_of(task)
    if kind == "everyday":
        return "diagnose" if ev.half(next(q for q in ev.load_bank() if q["id"] == row["id"])) \
            == ev.PRACTICE else "report"
    if kind == "exam":
        return "diagnose" if task == "fr_control_mmlu" else eb.half_of(questions.table(task)[
            "rows"][row["id"]]["qid"])
    return dx.split_of(row["id"])


def test_every_listed_task_lists_only_the_half_that_may_be_listed(svc):
    client = svc
    for t in client.get("/api/questions").json()["tasks"]:
        rows, p = every_row(client, t["task"])
        assert len(rows) == p["listed"] == p["total"], t
        assert all(half_rule(t["task"], r) == "diagnose" for r in rows), t["task"]
        # the rest are counted, never listed, and said why
        whole = questions.table(t["task"])["rows"]
        assert p["other"] == len(whole) - p["listed"]
        assert p["hidden_why"]


# ---------------------------------------------------------------------------
# a page, and each model's result
# ---------------------------------------------------------------------------

def samples(model: str, task: str) -> dict[str, dict]:
    d = config.OUT_DIR / model.replace("/", "__")
    return {r["doc_hash"]: r for f in d.glob(f"{task}_*shot/**/samples_*.jsonl")
            for r in map(json.loads, f.read_text().splitlines())}


def test_fifty_a_page_each_with_the_pick_right_or_wrong_and_the_margin(svc):
    client = svc
    p = page(client, "mmlu")
    assert len(p["rows"]) == 50 and p["total"] > 50 and p["limit"] == 50
    assert p["kind"] == "lm" and "abstract algebra" in p["subjects"]
    got = samples("fx/good-750m", "mmlu")
    for row in p["rows"]:
        assert len(row["options"]) == 4 and row["answer_idx"] in range(4)
        r = row["results"]["fx/good-750m"]
        rec = got[row["id"]]
        assert r["ok"] == (rec["acc"] >= 0.5)
        lps = [float(x[0][0]) for x in rec["resps"]]
        assert r["pick"] == max(range(4), key=lambda i: lps[i])
        assert 0 <= r["margin"] <= 1
    # the next page is the next fifty
    nxt = page(client, "mmlu", offset=50)
    assert not {r["id"] for r in p["rows"]} & {r["id"] for r in nxt["rows"]}
    assert nxt["offset"] == 50 and nxt["total"] == p["total"]


def test_a_chosen_model_with_no_run_is_not_run(svc):
    client = svc
    p = page(client, "mmlu", models="fx/good-750m,local/nodiag-step400", limit=5)
    assert p["shown"] == ["fx/good-750m", "local/nodiag-step400"]
    assert all(r["results"]["local/nodiag-step400"] is None for r in p["rows"])
    assert all(r["results"]["fx/good-750m"] for r in p["rows"])


def test_the_filters(svc):
    client = svc
    two = "fx/good-750m,fx/chance-160m"
    table = questions.table("mmlu")["rows"]
    ok = lambda k, m: table[k]["results"][m]["ok"]  # noqa: E731
    rows, _ = every_row(client, "mmlu", subject="anatomy")
    assert rows and {r["subject"] for r in rows} == {"anatomy"}
    rows, _ = every_row(client, "mmlu", models=two, f="disagree")
    assert rows and all(ok(r["id"], "fx/good-750m") != ok(r["id"], "fx/chance-160m") for r in rows)
    rows, _ = every_row(client, "mmlu", models=two, f="allwrong")
    assert rows and all(not ok(r["id"], "fx/good-750m") and not ok(r["id"], "fx/chance-160m")
                        for r in rows)
    rows, _ = every_row(client, "mmlu", models=two, f="onlyright:fx/good-750m")
    assert rows and all(ok(r["id"], "fx/good-750m") and not ok(r["id"], "fx/chance-160m")
                        for r in rows)
    rows, _ = every_row(client, "mmlu", models=two, f="onlywrong:fx/good-750m")
    assert all(not ok(r["id"], "fx/good-750m") and ok(r["id"], "fx/chance-160m") for r in rows)
    # search: the question's words, or an option's
    one = page(client, "mmlu", limit=1)["rows"][0]
    word = max(one["q"].split(), key=len)
    assert one["id"] in [r["id"] for r in every_row(client, "mmlu", q=word)[0]]


def test_truthfulqa_mc2_marks_every_true_answer_and_the_mass_on_them(svc):
    client = svc
    row = page(client, "truthfulqa_mc2", limit=1)["rows"][0]
    assert isinstance(row["answer_idx"], list) and len(row["answer_idx"]) > 1
    r = row["results"]["fx/good-750m"]
    assert "pick" not in r and 0 <= r["mass"] <= 1 and r["ok"] == (r["mass"] >= 0.5)


def write_gen(model: str) -> None:
    """MMLU-Pro answered in writing, one answer with its thinking"""
    d = config.OUT_DIR / model.replace("/", "__") / "mmlu_pro_5shot" / "run"
    d.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, (q, key, ans) in enumerate([
            ("Which organelle makes proteins?", "C", "<think>Ribosomes do.</think>The answer is (C)."),
            ("Which gas do plants take in?", "A", "The answer is (B)."),
            ("Which planet is largest?", "D", "The answer is (D).")]):
        doc = {"question": q, "options": ["A1", "B1", "C1", "D1"], "answer": key,
               "answer_index": "ABCD".index(key), "category": "biology"}
        recs.append({"doc_id": i, "doc": doc, "doc_hash": hashlib.sha256(q.encode()).hexdigest(),
                     "resps": [[ans]], "filtered_resps": ["[invalid]"]})
    (d / "samples_mmlu_pro_2026-09-25T00-00-00.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs))


def test_a_written_answer_with_its_thinking_folded_and_its_verdict(svc):
    client = svc
    write_gen("fx/below-135m-it")
    rows, p = every_row(client, "mmlu_pro")
    assert p["kind"] == "gen"
    by_q = {r["q"]: r["results"]["fx/below-135m-it"] for r in rows}
    shown = set(by_q)
    wanted = {"Which organelle makes proteins?", "Which gas do plants take in?",
              "Which planet is largest?"}
    # the half rule applies to written answers too
    assert shown == {q for q in wanted
                     if dx.split_of(hashlib.sha256(q.encode()).hexdigest()) == "diagnose"}
    for q, r in by_q.items():
        assert r["verdict"].startswith("read as ") and "· the answer is " in r["verdict"]
        if q.startswith("Which organelle"):
            assert r["ok"] and r["thinking"] == "Ribosomes do." and r["answer"] == \
                "The answer is (C)."
        if q.startswith("Which gas"):
            assert r["ok"] is False


def test_the_exam_lists_its_diagnose_half_with_the_judges_score(svc):
    client = svc
    rows, p = every_row(client, "exam_law")
    assert p["kind"] == "exam" and p["other"] > 0
    assert rows and all(r["results"].get("fx/good-750m") for r in rows[:3])
    r = next(x["results"]["fx/good-750m"] for x in rows if x["results"].get("fx/good-750m"))
    assert r["verdict"].endswith("of 4 from the judge") and r["answer"]


def test_everyday_lists_its_practice_half(svc):
    client = svc
    rows, p = every_row(client, "everyday")
    practice = {q["id"] for q in ev.load_bank() if ev.half(q) == ev.PRACTICE}
    assert {r["id"] for r in rows} == practice
    assert p["other"] == len(ev.load_bank()) - len(practice)
    assert p["hidden_why"] == "hidden: they score it and are not shown"


def test_the_ggufs_line(svc):
    client = svc
    assert page(client, "hellaswag", limit=1)["gguf"] == {
        "benchmark": "hellaswag", "line": "llama.cpp records only the total", "models": []}
    assert page(client, "truthfulqa_mc2", limit=1)["gguf"] is None


# ---------------------------------------------------------------------------
# the owner's audit
# ---------------------------------------------------------------------------

def test_the_audit_is_the_owners_after_the_warning_and_logged(svc):
    client = svc
    r = client.post("/api/questions/mmlu/audit", json={"by": "sam", "confirm": True})
    assert r.status_code == 403
    r = client.post("/api/questions/mmlu/audit", json={"by": OWNER})
    assert r.status_code == 428 and "This opening is logged." in r.json()["detail"]
    assert db.hidden_audits() == []
    got = client.post("/api/questions/mmlu/audit", json={"by": OWNER, "confirm": True,
                                                         "limit": 200}).json()
    assert got["half"] == "report" and got["rows"]
    assert all(dx.split_of(x["id"]) == "report" for x in got["rows"])
    assert got["audit"] == "report half" and "Don’t train on them" in got["warning"]
    log = client.get("/api/everyday/audits").json()["audits"]
    assert (log[0]["by"], log[0]["group"]) == (OWNER, "mmlu · report half")     # newest first
    # Everyday's, from the browser: its hidden half
    got = client.post("/api/questions/everyday/audit", json={"by": OWNER, "confirm": True,
                                                             "limit": 200}).json()
    hidden = {q["id"] for q in ev.load_bank() if ev.half(q) == ev.HIDDEN}
    assert {x["id"] for x in got["rows"]} <= hidden and got["audit"] == "hidden half"


# ---------------------------------------------------------------------------
# nothing browsed feeds anything
# ---------------------------------------------------------------------------

def _files(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and not p.name.startswith("service.sqlite3")}


def test_nothing_browsed_reaches_a_writer_improve_the_playground_or_chat(svc):
    client = svc
    before = _files(config.BENCH_ROOT)
    for t in [x["task"] for x in client.get("/api/questions").json()["tasks"]][:8]:
        every_row(client, t)
    client.post("/api/questions/mmlu/audit", json={"by": OWNER, "confirm": True, "limit": 200})
    # browsing writes nothing anything reads: every file as it was
    assert _files(config.BENCH_ROOT) == before
    # …and no module that writes or asks a model reads the browser
    service = Path(questions.__file__).resolve().parent
    for mod in ("builder", "proposals", "playground", "chat", "judge_test", "llm_poller",
                "runner", "worker"):
        f = service / f"{mod}.py"
        if f.exists():
            text = f.read_text()
            assert not re.search(r"from \. import[^\n]*\bquestions\b|from \.questions |"
                                 r"import questions\b|\bquestions\.(?:page|table|tasks)\(",
                                 text), mod
    # the report half's questions reach no practice list
    report = {r["q"] for r in questions.table("mmlu")["rows"].values() if r["half"] == "report"}
    practice = json.dumps(client.get("/api/playground/practice").json())
    assert not [q for q in report if q in practice]


# ---------------------------------------------------------------------------
# the GGUF's results, question by question, where llama.cpp's log says them
# ---------------------------------------------------------------------------

def _board_docs(task: str) -> list[dict]:
    """the questions the board's lm_eval runs answered, in lm_eval's order —
    the file gguf_data.py builds from the same documents"""
    d = config.OUT_DIR / "fx__good-750m"
    recs = []
    for f in sorted(d.glob(f"{task}_*shot/**/samples_*.jsonl")):
        recs += [json.loads(x) for x in f.read_text().splitlines()]
    return [r["doc"] for r in sorted(recs, key=lambda r: r["doc_id"])] if task != "mmlu" else \
        [r["doc"] for f in sorted(d.glob("mmlu_*shot/**/samples_*.jsonl"))
         for r in sorted(map(json.loads, f.read_text().splitlines()), key=lambda r: r["doc_id"])]


def _measure(tmp_path, monkeypatch, benchmarks, subset=0, acc="0.6"):
    import time as _t
    import gguf_bench as gb
    import gguf_data as gd
    import gguf_worker as gw
    from test_gguf_12f3 import fake_binary
    res = config.RESULTS_ROOT
    gd.build(res / "gguf_data", only=list(benchmarks), docs_of=lambda t: _board_docs(
        "mmlu" if t == "mmlu" else t))
    man = json.loads((res / "gguf_data" / "manifest.json").read_text())["benchmarks"]
    model = tmp_path / "k8.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 64)
    monkeypatch.setattr(gw, "_stop", {"why": ""})
    monkeypatch.setenv("FAKE_PPL_ACC", acc)
    rid = str(int(_t.time() * 1000) % 100000)
    req = {"id": rid, "sid": int(rid), "model": "gguf/k8", "name": "k8", "path": str(model),
           "flags": [], "benchmarks": list(benchmarks), "subset": subset, "pin": {},
           "datasets": {b: {"sha256": man[b]["sha256"], "n": man[b]["n"]} for b in benchmarks},
           "at": _t.time()}
    (res / "gguf_requests").mkdir(parents=True, exist_ok=True)
    (res / "gguf_requests" / f"{rid}.json").write_text(json.dumps(req))
    gw.Worker(res, fake_binary(tmp_path), poll=0.05).once()
    assert json.loads((res / "gguf_results" / f"{rid}.json").read_text())["status"] == "done"
    return [t["question"] for t in gb.read_mc((res / "gguf_data" / man[benchmarks[0]]["file"])
                                              .read_bytes())] if benchmarks[0] != "winogrande" \
        else None


def _fake_right(n: int, acc: float) -> list[bool]:
    """what the fake llama-perplexity gets right, task by task (its rule)"""
    out, right = [], 0
    for i in range(1, n + 1):
        ok = round(acc * i) > right
        right += ok
        out.append(ok)
    return out


def test_a_full_runs_log_gives_the_ggufs_result_on_each_question(svc, tmp_path, monkeypatch):
    client = svc
    import gguf_data as gd
    texts = _measure(tmp_path, monkeypatch, ["mmlu"])
    want = dict(zip(texts, _fake_right(len(texts), 0.6)))
    rows, p = every_row(client, "mmlu")
    assert p["gguf"] == {"benchmark": "mmlu", "line": None, "models": ["gguf/k8"]}
    docs = {dx_key: d for d in _board_docs("mmlu")
            for dx_key in [gd.mmlu_task(d)["question"]]}
    for r in rows:
        g = r["gguf"]["gguf/k8"]
        key = next(k for k in docs if docs[k]["question"] in k and r["q"] in k)
        # 16c: and its own score
        assert g == {"ok": want[key], "score": float(want[key]),
                     "score_words": "right" if want[key] else "wrong"}, r["id"]


def test_winogrande_gives_its_pick_and_margin(svc, tmp_path, monkeypatch):
    client = svc
    _measure(tmp_path, monkeypatch, ["winogrande"])
    rows, p = every_row(client, "winogrande")
    assert p["gguf"]["models"] == ["gguf/k8"] and p["gguf"]["line"] is None
    g = [r["gguf"]["gguf/k8"] for r in rows if r["gguf"]["gguf/k8"]]
    # the fake prints scores -1.5 and -2.5, picks 1, answer 1
    assert g and all(x == {"ok": True, "pick": 0, "margin": 0.4621, "score": 1.0,
                           "score_words": "right"} for x in g)


def test_a_subset_or_hellaswag_has_only_the_total(svc, tmp_path, monkeypatch):
    client = svc
    _measure(tmp_path, monkeypatch, ["mmlu"], subset=5)
    p = page(client, "mmlu", limit=1)
    assert p["gguf"] == {"benchmark": "mmlu", "line": "llama.cpp records only the total",
                         "models": []}
    assert "gguf" not in p["rows"][0]
    assert page(client, "hellaswag", limit=1)["gguf"]["line"] == "llama.cpp records only the total"
