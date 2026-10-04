"""14.4.3: the full Mobile-MMLU as a benchmark — the Mobile suite's second
multiple-choice part, "mmlu_full", shown "Mobile-MMLU (full)"; Pro keeps its
name and stays the default. Scored as Pro is (the same prompt and method per
runtime) on its own key's kept questions: overall, in the paper's 9
categories and in the 80 fields. One run, two scores: a full run asks Pro's
wording of the question the two word apart (9933ec55; inv00004 in the
invented rows) as well as its own, so each set is scored on its own wording,
and Pro's score comes from the same picks; a Pro run counts towards a later
full run, which asks only what's missing. An estimate before queuing. Kept
apart: never a column, in no average, rank or "overall"; its own table.
Invented rows, picks written in; nothing runs, nothing calls OpenRouter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mobile_mmlu as mmp
from conftest import make_service
from service import config, db, runner, served
from test_14_3_mobile_mmlu import RIGHT, STAMP, put_invented, sit_picks
from test_14_4_1_mobile_mmlu_data import put_full
from test_14_4_2_mobile_mmlu_key import FULL_ONLY

# the full set's wording of inv00004 is another question: its answer is D, Pro's C
FULL_04 = "D"


@pytest.fixture
def both(tmp_path, monkeypatch):
    put_invented(monkeypatch, tmp_path / "data" / "mobile_mmlu_pro")
    put_full(monkeypatch, tmp_path / "data" / "mobile_mmlu")


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch)
    put_invented(monkeypatch, config.MMP_DIR)
    put_full(monkeypatch, config.BENCH_ROOT / "data" / "mobile_mmlu")
    yield client, appmod
    client.__exit__(None, None, None)


def right(q: dict) -> str:
    """a pool row's right answer, in its own wording"""
    if q["id"] == "inv00004" and q["sets"] == ["full"]:
        return FULL_04
    return {**RIGHT, **FULL_ONLY}[q["id"]]


def pool_key(drop: tuple[str, ...] = ()) -> dict:
    """the one key for both sets, both labellers agreeing on every question"""
    rows = mmp.pool()
    cur = {s: {"id": s, "version": f"{s}-v1"} for s in mmp.SLOTS}
    lab = {s: {q["lid"]: {"letter": right(q), "now": q["id"] in drop, "version": f"{s}-v1"}
               for q in rows} for s in ("first", "second")}
    k = mmp.build_key(rows, lab, cur)
    mmp.write_key(k)
    mmp._keyc.clear()
    return k


def sit_full(model_dir: Path, picks: dict[str, str], served_text: bool = False) -> None:
    """what a full run logs: every doc it asked (each with its label id)"""
    d = model_dir / "mobile_mmlu_full_0shot" / "x"
    d.mkdir(parents=True, exist_ok=True)
    by_lid = {q["lid"]: q for q in mmp.pool()}
    n = len(list(d.glob("samples_*.jsonl")))
    with open(d / f"samples_mobile_mmlu_full_{STAMP}{n}.jsonl", "w") as fh:
        for i, (k, letter) in enumerate(picks.items()):
            q = by_lid[k]
            fr = ([f"The answer is {letter}."] if served_text else
                  [[-1.0 if L == letter else -3.0, False] for L in "ABCD"])
            fh.write(json.dumps({"doc_id": i, "doc": {"id": q["id"], "lid": k,
                                                      "field": q["field"], "prompt": "…"},
                                 "filtered_resps": fr}) + "\n")
    mid = model_dir.name.replace("__", "/", 1)
    (d / f"results_{STAMP}{n}.json").write_text(json.dumps({
        "results": {"mobile_mmlu_full": {"alias": "mobile_mmlu_full", "acc,none": 0.25}},
        "group_subtasks": {"mobile_mmlu_full": []}, "n-shot": {"mobile_mmlu_full": 0},
        "n-samples": {"mobile_mmlu_full": {"original": 19, "effective": len(picks)}},
        "higher_is_better": {"mobile_mmlu_full": {"acc": True}},
        "config": {"model": "hf", "model_args": f"pretrained={mid}"}, "date": 1790200000.0}))
    if not (model_dir / "model_meta.json").exists():
        (model_dir / "model_meta.json").write_text(json.dumps({"model": mid, "kind": "instruct"}))


def fake_lm_eval(monkeypatch, wrong: dict[str, str] | None = None) -> list[list[str]]:
    """lm_eval, faked: it answers the docs the command asks — its task's
    file, narrowed by --samples — each right unless `wrong` says otherwise"""
    from test_14_1_mab_text import fake_gpu
    fake_gpu(monkeypatch)
    seen: list[list[str]] = []

    def run(sid, cmd, *a, **k):
        seen.append(cmd)
        task = cmd[cmd.index("--tasks") + 1]
        model_dir = Path(cmd[cmd.index("--output_path") + 1]).parent
        inc = Path(cmd[cmd.index("--include_path") + 1])
        docs = [json.loads(x) for x in (inc / f"{task}.jsonl").read_text().splitlines()]
        if "--samples" in cmd:
            only = json.loads(Path(cmd[cmd.index("--samples") + 1]).read_text())[task]
            docs = [docs[i] for i in only]
        if task == "mobile_mmlu_pro":
            sit_picks(model_dir, {d["id"]: (wrong or {}).get(d["id"], RIGHT[d["id"]])
                                  for d in docs})
        else:
            rows = {q["lid"]: q for q in mmp.pool()}
            sit_full(model_dir, {d["lid"]: (wrong or {}).get(d["lid"], right(rows[d["lid"]]))
                                 for d in docs})
        return 0
    monkeypatch.setattr(runner, "_run_task", run)
    return seen


def submit(client, model: str, part: str, kind: str = "instruct") -> int:
    r = client.post("/api/submissions", json={"hf_id": model, "suite": "mobile", "part": part,
                                              "kind": kind})
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done", db.get(sid)
    return sid


def lids(which: str) -> dict[str, str]:
    return {q["id"]: q["lid"] for q in mmp.pool() if which in q["sets"]}


# ---------------------------------------------------------------------------
# the part
# ---------------------------------------------------------------------------

def test_the_part_is_the_full_set_and_pro_keeps_its_name_and_stays_the_default():
    assert config.tasks_for_suite("mobile", part="mmlu_full") == ["mobile_mmlu_full"]
    assert config.tasks_for_suite("mobile", part="mmlu") == ["mobile_mmlu_pro"]
    assert "mmlu_full" in config.MAB_PARTS and config.MAB_PARTS[0] == ""
    assert (mmp.FULL_NAME, mmp.NAME) == ("Mobile-MMLU (full)", "Mobile-MMLU-Pro")
    import gguf_bench as gb
    assert gb.BENCHMARKS["mobile_mmlu_full"]["label"] == "Mobile-MMLU (full)"
    assert "mobile_mmlu_full" not in gb.DEFAULT and "mobile_mmlu_pro" not in gb.DEFAULT


def test_the_run_task_is_the_pool_both_wordings_of_9933ec55_and_never_a_key(both, tmp_path):
    pool_key()
    d = mmp.build_full_tasks(tmp_path / "t")
    docs = [json.loads(x) for x in (d / "mobile_mmlu_full.jsonl").read_text().splitlines()]
    assert len(docs) == 19 and [x["lid"] for x in docs] == [q["lid"] for q in mmp.pool()]
    # inv00004 twice, once in each wording, each with its own prompt
    two = [x for x in docs if x["id"] == "inv00004"]
    assert len(two) == 2 and two[0]["prompt"] != two[1]["prompt"]
    assert {x["lid"] for x in two} == {lids("pro")["inv00004"], lids("full")["inv00004"]}
    # asked as Pro is: lm_eval's prompt, and the authors' a served model gets
    pro = {json.loads(x)["id"]: json.loads(x)["prompt"] for x in
           (mmp.build_tasks(tmp_path / "p") / "mobile_mmlu_pro.jsonl").read_text().splitlines()}
    assert two[0]["prompt"] == pro["inv00004"]
    ask = [json.loads(x) for x in (d / "mobile_mmlu_full_ask.jsonl").read_text().splitlines()]
    assert ask[0]["prompt"] == mmp.ask_prompt(mmp.pool()[0])
    text = "".join(p.read_text() for p in d.iterdir())
    assert '"key"' not in text and '"answer"' not in text
    assert (d / "mobile_mmlu_full.yaml").read_text().startswith("task: mobile_mmlu_full\n")


# ---------------------------------------------------------------------------
# one run, two scores
# ---------------------------------------------------------------------------

def test_a_full_run_asks_both_wordings_of_9933ec55_and_scores_each_set_on_its_own(svc,
                                                                                  monkeypatch):
    client, _ = svc
    pool_key()
    seen = fake_lm_eval(monkeypatch)
    submit(client, "org/full-2b", "mmlu_full")
    cmd = seen[0]
    assert cmd[cmd.index("--tasks") + 1] == "mobile_mmlu_full" and "--samples" not in cmd
    assert "--apply_chat_template" not in cmd           # as Pro: as the paper ran it
    d = config.OUT_DIR / "org__full-2b"
    full, pro = mmp.full_predictions(d), mmp.predictions(d)
    # Pro's inv00004 was asked in Pro's wording, the full set's in its own
    assert pro["predictions"]["inv00004"] == "C" and full["predictions"]["inv00004"] == FULL_04
    assert (pro["n"], full["n"]) == (12, 18)
    # each scored on its own wording and its own key: right everywhere
    assert mmp.score(pro)["acc"] == 1.0
    assert mmp.score(full, mmp.full_key())["acc"] == 1.0
    # scored on the other's wording, inv00004 would be wrong: the two are apart
    crossed = {**full, "predictions": {**full["predictions"], "inv00004": "C"}}
    assert mmp.score(crossed, mmp.full_key())["correct"] == 17


def test_a_full_runs_picks_give_the_pro_score_and_a_pro_run_after_asks_nothing(svc,
                                                                               monkeypatch):
    client, appmod = svc
    pool_key()
    seen = fake_lm_eval(monkeypatch, wrong={lids("pro")["inv00001"]: "D"})
    submit(client, "org/full-2b", "mmlu_full")
    appmod._cache.update(key=None, payload=None, at=0.0)
    data = client.get("/api/results").json()
    # the Pro cell, from the full run's picks: 11 of 12
    assert data["cells"]["mobile_mmlu_pro"]["org/full-2b"]["v"] == pytest.approx(11 / 12)
    m = next(x for x in data["models"] if x["id"] == "org/full-2b")
    assert m["mmp"]["n"] == 12 and m["mmf"]["n"] == 18
    # a Pro run now asks none: every question is answered already
    n = len(seen)
    sid = submit(client, "org/full-2b", "mmlu")
    assert len(seen) == n
    log = (config.LOGS_DIR / f"service_{sid}_org__full-2b.log").read_text()
    assert "mobile_mmlu_pro: every question answered already" in log


def test_a_pro_run_counts_towards_a_later_full_run_which_asks_only_whats_missing(svc,
                                                                                 monkeypatch):
    client, _ = svc
    pool_key()
    seen = fake_lm_eval(monkeypatch)
    submit(client, "org/pro-first-2b", "mmlu")
    d = config.OUT_DIR / "org__pro-first-2b"
    left = mmp.full_left(d)
    rows = mmp.pool()
    # the full set's own six, and its wording of inv00004
    assert sorted(rows[i]["lid"] for i in left) == sorted(
        [lids("full")[q] for q in FULL_ONLY] + [lids("full")["inv00004"]])
    est = client.get("/api/mobileaibench/estimate", params={"model": "org/pro-first-2b"}).json()
    e = est["parts"]["mmlu_full"]
    assert (e["questions"], e["of"], e["reused"]) == (7, 19, 12)
    assert "12 answered already, by its Mobile-MMLU-Pro run" in e["line"]
    submit(client, "org/pro-first-2b", "mmlu_full")
    cmd = seen[-1]
    asked = json.loads(Path(cmd[cmd.index("--samples") + 1]).read_text())
    assert asked == {"mobile_mmlu_full": left}
    full = mmp.full_predictions(d)
    assert full["n"] == 18 and full["predictions"]["inv00004"] == FULL_04
    assert mmp.score(full, mmp.full_key())["acc"] == 1.0
    # and once both are whole, a full run asks nothing more
    assert mmp.full_left(d) == []


def test_a_served_full_run_asks_only_whats_missing_in_the_authors_prompt(svc, monkeypatch):
    pool_key()
    d = config.OUT_DIR / "served__x"
    sit_picks(d, RIGHT, served_text=True)
    mmp.build_tasks(config.MMP_TASKS_DIR)
    mmp.build_full_tasks(config.MMF_TASKS_DIR)
    got = []
    monkeypatch.setattr(served, "answer_task", lambda rec, task, docs, *a, **k:
                        got.append(docs) or 0)
    rec = {"id": "served/x", "name": "x", "base_url": "http://x/v1", "thinking": "off",
           "how": "", "based_on": "", "pin": {"model": "org/x"}}
    status, stopped = runner._ask_served(1, rec, {}, "mobile_mmlu_full",
                                         d / "mobile_mmlu_full_0shot", "1/1",
                                         config.BENCH_ROOT / "x.log", False, None)
    assert (status, stopped) == (0, None)
    pool = {q["lid"]: q for q in mmp.pool()}
    assert len(got[0]) == 7 and all(x["lid"] in pool and "pro" not in pool[x["lid"]]["sets"]
                                    for x in got[0])
    assert got[0][0]["prompt"] == mmp.ask_prompt(pool[got[0][0]["lid"]])
    # its cost before queuing counts the same seven
    monkeypatch.setattr(served, "model_dir", lambda r: d)
    rec_or = {**rec, "via": "openrouter", "pin": {"model": "org/x", "price_in": 1.0,
                                                   "price_out": 2.0}}
    est = served.estimate(rec_or, "mobile", part="mmlu_full")
    assert est["n"] == 7 and est["usd"] > 0


# ---------------------------------------------------------------------------
# scored as Pro: overall, the 9 categories and the 80 fields
# ---------------------------------------------------------------------------

def test_scored_on_kept_questions_overall_in_the_9_categories_and_the_80_fields(both,
                                                                                tmp_path):
    pool_key(drop=("inv00013",))                     # time-sensitive: dropped
    picks = {k: right(q) for k, q in ((q["lid"], q) for q in mmp.pool())}
    picks[lids("full")["inv00014"]] = "B"            # one wrong
    sit_full(tmp_path / "org__m", picks)
    mmp.collect(tmp_path / "org__m")
    s = mmp.score(mmp.full_predictions(tmp_path / "org__m"), mmp.full_key())
    assert (s["n"], s["of"], s["correct"]) == (17, 17, 16)
    q14 = mmp.full_by_id()["inv00014"]
    assert s["by_field"][q14["field"]]["correct"] == s["by_field"][q14["field"]]["n"] - 1
    assert set(s["by_category"]) <= set(mmp.CATEGORIES)
    assert sum(v["n"] for v in s["by_field"].values()) == 17
    assert set(s["by_field"]) <= set(mmp.FIELD_CATEGORY)


# ---------------------------------------------------------------------------
# before queuing, and refused
# ---------------------------------------------------------------------------

def test_the_estimate_before_queuing_time_for_a_local_model_cost_for_openrouter(svc,
                                                                                monkeypatch):
    client, _ = svc
    est = client.get("/api/mobileaibench/estimate", params={"model": "org/new-2b"}).json()
    e = est["parts"]["mmlu_full"]
    assert (e["questions"], e["reused"], e["judgements"]) == (19, 0, 0)
    assert e["seconds"] > 0 and "19 answers" in e["line"] and "answered already" not in e["line"]
    assert est["parts"]["mmlu"]["questions"] == 12          # Pro's, as before
    rec = {"id": "served/x", "name": "x", "thinking": "off", "base_url": "http://x/v1",
           "how": "", "based_on": "", "via": "openrouter",
           "pin": {"model": "org/x", "price_in": 1.0, "price_out": 2.0}}
    monkeypatch.setattr(served, "model_dir", lambda r: config.OUT_DIR / "served__x")
    full, pro = (served.estimate(rec, "mobile", part=p) for p in ("mmlu_full", "mmlu"))
    assert (full["n"], pro["n"]) == (19, 12) and full["usd"] > pro["usd"]


def test_the_part_is_refused_without_the_full_set_and_a_base_model_may_sit_it(svc, monkeypatch,
                                                                              tmp_path):
    client, _ = svc
    monkeypatch.setenv("MMF_DIR", str(tmp_path / "none"))
    mmp._full_rows.clear()
    r = client.post("/api/submissions", json={"hf_id": "org/c", "suite": "mobile",
                                              "part": "mmlu_full", "kind": "instruct"})
    assert r.status_code == 422 and "Mobile-MMLU (full) isn't on this server" in r.json()["detail"]
    assert "Nothing was queued" in r.json()["detail"]
    put_full(monkeypatch, config.BENCH_ROOT / "data" / "mobile_mmlu")
    r = client.post("/api/submissions", json={"hf_id": "org/base-1b", "suite": "mobile",
                                              "part": "mmlu_full", "kind": "base"})
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# kept apart
# ---------------------------------------------------------------------------

def test_kept_apart_no_column_no_average_no_rank_its_own_table(svc, monkeypatch):
    client, appmod = svc
    pool_key()
    fake_lm_eval(monkeypatch)
    submit(client, "org/full-2b", "mmlu_full")
    appmod._cache.update(key=None, payload=None, at=0.0)
    data = client.get("/api/results").json()
    assert "mobile_mmlu_full" not in data["cells"] and "mobile_mmlu_full" not in data["accTasks"]
    assert "mobile_mmlu_full" not in data["required"]
    assert "mobile_mmlu_full" not in json.dumps(data.get("tasks") or {})
    m = next(x for x in data["models"] if x["id"] == "org/full-2b")
    assert m["avg"] is None and m["mmf"]["acc"] == 1.0
    assert set(m["mmf"]["by_category"]) <= set(mmp.CATEGORIES) and m["mmf"]["by_field"]
    f = data["mmf"]
    assert (f["name"], f["licence"], f["restriction"]) == (
        "Mobile-MMLU (full)", "CC BY-NC-ND 4.0", "non-commercial")
    assert f["key"]["counts"]["all"]["kept"] == 18 and len(f["fields"]) == 80
    assert [r["paper"] for r in f["checks"]["rows"]] == [68.1, 50.2, 38.9]
    # the GGUF table never makes it a column either
    import report_lm_eval as rep
    assert "mobile_mmlu_full" not in rep.required_tasks(["mmlu", "mobile_mmlu_full"])[0]


def test_never_a_training_target(both, tmp_path):
    import diagnose as dx
    pool_key()
    sit_full(tmp_path / "org__m", {q["lid"]: right(q) for q in mmp.pool()})
    assert "mobile_mmlu_full" not in dx.diagnose_model(tmp_path / "org__m")["tasks"]


def test_a_gguf_dataset_is_the_full_sets_wording_on_its_own_key_only_when_asked(both, tmp_path):
    import gguf_bench as gb
    import gguf_data
    with pytest.raises(ValueError, match="Mobile-MMLU \\(full\\) has no answer key yet"):
        mmp.gguf_docs("full")
    pool_key()
    docs = mmp.gguf_docs("full")
    assert len(docs) == 18
    four = next(x for x in docs if x["id"] == "inv00004")
    assert four["question"] == mmp.full_by_id()["inv00004"]["question"]
    assert four["answer"] == "ABCD".index(FULL_04)
    assert gb.BENCHMARKS["mobile_mmlu_full"]["kept_apart"]
    gguf_data.build(tmp_path / "gg", only=["mobile_mmlu_full"])
    entry = json.loads((tmp_path / "gg" / "manifest.json").read_text())["benchmarks"][
        "mobile_mmlu_full"]
    assert entry["n"] == 18 and entry["key"] == mmp.full_key()["version"]
    assert (entry["licence"], entry["restriction"]) == ("CC BY-NC-ND 4.0", "non-commercial")
