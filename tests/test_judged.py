"""Judged free response: the control set is diagnose-only, the judge is
pinned and deterministic, refuses its own family, and nothing judged counts
until a person agrees with it."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


import diagnose as dx
import exam_build as fr_build
import judge as jd
import judge_calibrate as jc
import make_fixture
import report_lm_eval as report
from conftest import fresh, make_service

REPO = Path(__file__).resolve().parents[1]


def _items(tree, task):
    p = tree["judged"]["tasks_dir"] / f"{task}.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# the tasks
# ---------------------------------------------------------------------------

def test_control_set_is_diagnose_only_and_stratified(tree):
    """THE RULE for this phase: every control item's source hash splits to
    diagnose. The report half is never asked, never judged, never shown."""
    items = _items(tree, fr_build.CONTROL_TASK)
    assert len(items) >= 20
    assert all(dx.split_of(it["mmlu_doc_hash"]) == "diagnose" for it in items)
    docs = {d["doc_hash"]: d for d in tree["docs"]["mmlu"]}
    for it in items:
        d = docs[it["mmlu_doc_hash"]]
        assert it["prompt"].startswith(d["q"]) and it["prompt"].endswith(fr_build.CONTROL_SUFFIX)
        assert "A." not in it["prompt"]                          # the question only, no options
        assert it["category"] and it["subject"] == d["group"]
    per_cat = {}
    for it in items:
        per_cat[it["category"]] = per_cat.get(it["category"], 0) + 1
    assert all(1 <= n <= fr_build.CONTROL_PER_CATEGORY for n in per_cat.values())
    assert len(per_cat) >= 4
    assert len({it["id"] for it in items}) == len(items)
    # the gold option's text is the reference
    raw = {}
    for f in tree["models"]["fx/good-750m"]["dir"].glob("mmlu_5shot/*/samples_mmlu_*.jsonl"):
        for ln in f.read_text().splitlines():
            r = json.loads(ln)
            raw[r["doc_hash"]] = r["doc"]
    for it in items[:10]:
        doc = raw[it["mmlu_doc_hash"]]
        assert it["reference"] == doc["choices"][doc["answer"]]
    # mmlu_perm's rotated documents were not mistaken for MMLU
    perm_hashes = {d["doc_hash"] for d in tree["docs"]["mmlu_perm"]}
    assert not any(it["mmlu_doc_hash"] in perm_hashes for it in items)


def test_control_build_is_deterministic(tree, tmp_path):
    a = fr_build.build(tree["out_dir"], tmp_path / "a")
    b = fr_build.build(tree["out_dir"], tmp_path / "b")
    assert (tmp_path / "a" / "tasks" / "fr_control_mmlu.jsonl").read_bytes() == \
        (tmp_path / "b" / "tasks" / "fr_control_mmlu.jsonl").read_bytes()
    assert a["tasks"]["fr_control_mmlu"]["sha256"] == b["tasks"]["fr_control_mmlu"]["sha256"]
    assert a["tasks"]["fr_control_mmlu"]["built_from"] == "mmlu diagnose half"
    assert a["tasks"]["fr_control_mmlu"]["report"] == 0


def test_skill_suites_are_migrated_not_thrown_away(tree):
    """The four skill suites' 40 items live on in the bank under `other`, as
    they were, with their skill on the record; AUTHORING.md points at the exam."""
    bank = fr_build.load_bank(tree["judged"]["exam_root"])
    migrated = [r for r in bank["other"] if r["source"] == "migrated"]
    assert len(migrated) == 40
    assert {r["skill"] for r in migrated} == set(fr_build.SKILL_SUITES)
    seeds = {}
    for skill in fr_build.SKILL_SUITES:
        for ln in (REPO / "eval_tasks" / "fr" / f"fr_{skill}.jsonl").read_text().splitlines():
            it = json.loads(ln)
            seeds[it["prompt"]] = it
    for r in migrated:
        assert r["prompt"] in seeds and r["reference"] == seeds[r["prompt"]]["reference"]
        assert r["accepted_by"] == "migration" and r["accepted_at"] and r["edited"] is False
    assert fr_build.migrate_seeds(tree["judged"]["exam_root"]) == 0          # idempotent
    guide = (REPO / "eval_tasks" / "fr" / "AUTHORING.md").read_text()
    assert "Exam tab" in guide and "report half" in guide and "Do not" in guide


def test_task_yamls_point_at_absolute_items_files(tree):
    for t in tree["judged"]["manifest"]["tasks"]:
        y = (tree["judged"]["tasks_dir"] / f"{t}.yaml").read_text()
        assert y.startswith(f"task: {t}\n")
        assert "output_type: generate_until" in y and "metric: bypass" in y
        assert "do_sample: false" in y and "temperature: 0.0" in y
        m = re.search(r"test: (\S+)", y)
        assert m and Path(m.group(1)).is_absolute() and Path(m.group(1)).exists()


def test_rubrics_have_anchors_and_a_length_clause():
    for task in ("exam_economics", "exam_law", "exam_medicine_health", fr_build.CONTROL_TASK):
        r = jd.rubric_for(task)
        # a rubric carries a version; the author's topics are on their second
        assert re.fullmatch(r"[0-9]+", r.version) and re.fullmatch(r"[0-9a-f]{64}", r.sha256)
        for s in range(5):
            assert re.search(rf"^- \*\*{s}\*\*", r.text, re.M), (task, s)
        assert "Length" in r.text
    # a topic with no rubric of its own shares the exam one; the control keeps
    # the factual one; a topic with its own is graded by its own
    assert jd.rubric_for("exam_economics")[1] == jd.rubric_for("exam_history")[1]
    assert jd.rubric_for(fr_build.CONTROL_TASK)[1] != jd.rubric_for("exam_economics")[1]
    assert jd.rubric_for("exam_medicine_health")[1] != jd.rubric_for("exam_economics")[1]
    assert (REPO / "eval_tasks" / "fr" / "rubrics" / "exam.md").exists()


# ---------------------------------------------------------------------------
# the judge
# ---------------------------------------------------------------------------

def test_family_rule_and_refusal(tree):
    assert jd.family("meta-llama/Llama-3.1-8B-Instruct") == "llama"
    assert jd.family("HuggingFaceTB/SmolLM2-360M") == "smollm2"
    assert jd.family("EleutherAI/pythia-160m") == "pythia"
    assert jd.family("mistralai/Mistral-7B-Instruct-v0.3") == "mistral"
    assert jd.family("local/run7-step400") == "run7"
    _, plan = jd.plan_requests(tree["models"]["fx/good-750m"]["dir"], "good")
    assert "same family as judge" in plan["skipped"]
    out = jd.run_stub(tree["models"]["fx/good-750m"]["dir"], tree["root"] / "h")
    assert "skipped" not in out and len(out["tasks"]) == len(tree["judged"]["manifest"]["tasks"])
    assert jd.run_stub(tree["models"][tree["nodiag"]]["dir"], tree["root"] / "h2") is None


def test_judge_json_shape_and_hashes(tree):
    for mid in make_fixture.JUDGED:
        j = json.loads((tree["models"][mid]["dir"] / "judge.json").read_text())
        jj = j["judge"]
        assert re.fullmatch(r"[0-9a-f]{64}", jj["prompt_sha256"]) and jj["prompt_sha256"] == jd.prompt_sha()
        assert jj["stub"] is True and jj["id"] == "stub/overlap-v1"
        assert set(jj["rubrics"]) == set(tree["judged"]["manifest"]["tasks"])
        assert all(r["version"].isdigit() and len(r["sha256"]) == 64
                   for r in jj["rubrics"].values())
        assert j["split_salt"] == dx.SPLIT_SALT and j["correct_at"] == 3
        for t, v in j["tasks"].items():
            assert sum(v["dist"].values()) == v["n"] == len(v["items"])
            assert 0 <= v["mean"] <= 4 and v["max"] == 4
            assert v["n_report"] + v["n_diagnose"] == v["n"]
            assert all(it["half"] in ("report", "diagnose") for it in v["items"])
            if t.startswith("exam_"):
                assert all(it["qid"] and it["half"] == dx.split_of(it["qid"]) for it in v["items"])
            else:
                assert all(it["half"] == "diagnose" for it in v["items"])
            assert sum(b["n"] for b in v["score_vs_length"]) == v["n"]
            assert all(0 <= it["score"] <= 4 and it["doc_hash"] for it in v["items"])
        ctl = j["tasks"][fr_build.CONTROL_TASK]
        assert all(it["mc_right"] is not None for it in ctl["items"])          # joined to mmlu
        for c in ctl["control"].values():
            assert c["mc_wrong"] == c["knew"] + c["didnt"] and c["unjoined"] == 0


def test_control_sentence_direction(tree):
    def share(mid):
        j = json.loads((tree["models"][mid]["dir"] / "judge.json").read_text())
        c = j["tasks"][fr_build.CONTROL_TASK]["control"]
        wrong = sum(v["mc_wrong"] for v in c.values())
        knew = sum(v["knew"] for v in c.values())
        return knew, wrong
    knew, wrong = share("fx/skewed-360m")
    assert wrong >= 10 and knew / wrong >= 0.5            # knew it, couldn't pick it
    knew, wrong = share("fx/chance-160m")
    assert wrong >= 10 and knew / wrong < 0.2             # didn't know it either way


def test_stub_grader_and_the_length_clause():
    g = jd.StubGrader()
    ref = "The femur (thigh bone)."
    p = lambda ans: jd.build_prompt("rubric", "q", ref, ans)  # noqa: E731
    assert g.grade(p("The femur, the thigh bone."))[0] == 4
    assert g.grade(p("The femur, the thigh bone. " + make_fixture.FILLER))[0] == 3
    assert g.grade(p(""))[0] == 0
    assert g.grade(p("Paris is the capital of France, and it is lovely in spring."))[0] <= 1
    assert g.grade(p("The largest bone is in the leg."))[0] in (1, 2)
    assert "length clause" in g.grade(p("The femur, the thigh bone. " + make_fixture.FILLER))[1]


def test_judge_cli(tree, tmp_path, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, "argv", ["judge.py", str(tree["out_dir"]), "--stub", "-o", str(tmp_path)])
    assert jd.main() == 0
    out = capsys.readouterr().out
    assert "STUB" in out and "fx__skewed-360m" in out
    assert (tmp_path / "fx__skewed-360m" / "judge.json").exists()
    assert not (tmp_path / "fx__below-135m-it").exists()             # no fr answers → no file
    from service import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(sys, "argv", ["judge.py", str(tree["out_dir"])])
    assert jd.main() == 2                                             # no judge configured


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def test_kappa_arithmetic_against_a_known_table():
    # po = 4/6, pe = 3 * (2/6)^2 = 1/3  →  κ = (2/3 - 1/3) / (2/3) = 0.5
    assert jc.cohen_kappa([0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 2, 1]) == 0.5
    assert jc.cohen_kappa([1, 2, 3], [1, 2, 3]) == 1.0
    assert jc.cohen_kappa([4, 4, 4], [4, 4, 4]) == 1.0                 # pe = 1, po = 1
    assert jc.cohen_kappa([4, 4, 4], [3, 3, 3]) == 0.0                 # pe = 1, po = 0
    assert jc.cohen_kappa([0, 1, 2, 3], [3, 2, 1, 0]) < 0
    assert jc.cohen_kappa([], []) is None and jc.cohen_kappa([1], [1, 2]) is None
    assert jc.KAPPA_MIN == report.KAPPA_MIN == 0.6


def test_calibration_round_trip(tree, tmp_path):
    out_dir = tree["out_dir"]
    csv_path = tmp_path / "cal.csv"
    n = jc.export(out_dir, csv_path, [], 30, seed=7)
    assert n == 30
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    # a topic graded criterion by criterion adds a column per criterion after
    # the shared ones; the judge's own numbers are never written either way
    assert list(rows[0])[:len(jc.FIELDS)] == jc.FIELDS
    assert not any(k.startswith("judge") for k in rows[0])
    assert all(r["human_score"] == "" and r["rubric"].startswith("# Rubric") for r in rows)
    assert len({r["id"] for r in rows}) == 30
    assert len({r["category"] for r in rows}) >= 4                              # stratified
    judged = {r["id"]: r["judge_score"] for r in jc._judged_rows(out_dir, set())}
    assert len({judged[r["id"]] for r in rows}) >= 3                            # across scores
    # a grader who agrees with the judge on every row
    for r in rows:
        r["human_score"] = str(judged[r["id"]])
    rows[3]["human_score"] = ""                                                # left blank: skipped
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))   # whatever the export wrote
        w.writeheader()
        w.writerows(rows)
    cal = jc.import_csv(out_dir, csv_path)
    assert cal["kappa"] == 1.0 and cal["calibrated"] is True and cal["n"] == 29
    assert cal["rows_skipped"] == 1 and cal["judge"]["id"] == "stub/overlap-v1"
    assert (out_dir / jc.CALIBRATION_FILE).exists()
    # and one who never agrees
    for r in rows:
        r["human_score"] = str((int(judged[r["id"]]) + 2) % 5)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))   # whatever the export wrote
        w.writeheader()
        w.writerows(rows)
    cal = jc.import_csv(out_dir, csv_path)
    assert cal["kappa"] < 0.6 and cal["calibrated"] is False
    # restore the fixture's calibration for the other tests
    jc.import_csv(out_dir, tree["judged"]["calibration_csv"])
    assert json.loads((out_dir / jc.CALIBRATION_FILE).read_text())["calibrated"] is True


def test_fixture_calibration_clears_the_line(tree):
    cal = tree["judged"]["calibration"]
    assert cal["calibrated"] is True and cal["kappa"] >= 0.6 and cal["n"] == 60
    assert len(cal["per_category"]) >= 5 and set(cal["per_category"]) <= set(fr_build.TOPICS)


# ---------------------------------------------------------------------------
# the payload and its gating
# ---------------------------------------------------------------------------

def test_payload_judged_block(payload, tree):
    J = payload["judged"]
    assert set(J["tasks"]) == set(tree["judged"]["manifest"]["tasks"])
    assert J["exam"] == fr_build.exam_tasks() and J["topics"]["exam_economics"] == "economics"
    assert J["calibration"]["calibrated"] is True and J["kappaMin"] == 0.6
    assert J["judge"]["stub"] is True
    for t in fr_build.ALL_TASKS:                        # never a leaderboard column
        assert t not in payload["accTasks"] and t not in payload["required"]
        assert t not in payload["cells"] and t not in payload["tasks"]
    assert not any(row[2] == "bypass" for row in payload["extra"])
    good = next(m for m in payload["models"] if m["id"] == "fx/good-750m")
    assert set(good["judge"]["tasks"]) == set(tree["judged"]["manifest"]["tasks"])
    assert good["judgedAvg"] is not None and 0 <= good["judgedAvg"] <= 4
    econ = good["judge"]["tasks"]["exam_economics"]
    assert "items" not in econ and econ["score_report"] is not None      # trimmed; report half published
    assert report.published_score(econ) == econ["score_report"]
    assert report.published_score({"mean": 2.0}) == 2.0                  # an older judge.json
    assert good["official"] is True                                    # unchanged by judging
    nodiag = next(m for m in payload["models"] if m["id"] == tree["nodiag"])
    assert nodiag["judge"] is None and nodiag["judgedAvg"] is None
    assert any("STUB grader" in w for w in payload["warnings"])


def test_preliminary_gating(tree):
    runs = report.load_results(tree["out_dir"])
    p0 = report.build_payload(report.merge_runs(runs), "t", "", calibration=None)
    assert p0["judged"]["calibration"] is None
    assert any("not been calibrated" in w for w in p0["warnings"])
    p1 = report.build_payload(report.merge_runs(runs), "t", "",
                              calibration={"kappa": 0.41, "n": 100, "calibrated": False})
    assert p1["judged"]["calibration"]["calibrated"] is False
    assert any("below the 0.6 line" in w for w in p1["warnings"])
    # judgedAvg exists in the data either way; the page decides what to rank
    assert next(m for m in p1["models"] if m["id"] == "fx/good-750m")["judgedAvg"] is not None
    assert report.judged_avg({"skipped": "x", "tasks": {}}) is None
    assert report.judged_avg({"tasks": {"exam_law": {"mean": 2}}}) is None   # one topic: no avg


def test_judge_json_is_picked_up_without_a_restart(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judged=False)
    try:
        p = client.get("/api/results").json()
        assert p["judged"]["tasks"] == [] and p["judged"]["calibration"] is None
        j = make_fixture.write_judged(tmp_path, tree["out_dir"])
        fresh(appmod)
        p = client.get("/api/results").json()
        assert set(p["judged"]["tasks"]) == set(j["manifest"]["tasks"])
        assert p["judged"]["calibration"]["calibrated"] is True
        good = next(m for m in p["models"] if m["id"] == "fx/good-750m")
        assert good["judge"]["tasks"]["fr_control_mmlu"]["control"]
    finally:
        client.__exit__(None, None, None)


def test_judged_suite_through_the_service(tmp_path, monkeypatch):
    from service import config, runner
    client, appmod, tree = make_service(tmp_path, monkeypatch, judged=True, judge_model="")
    try:
        st = client.get("/api/judge").json()
        assert st["configured"] is False and "JUDGE_MODEL is unset" in st["reason"]
        assert set(st["tasks"]) == set(tree["judged"]["manifest"]["tasks"])
        r = client.post("/api/submissions", json={"hf_id": "org/model", "suite": "judged"})
        assert r.status_code == 503
        monkeypatch.setattr(config, "JUDGE_PROVIDER", "anthropic")
        monkeypatch.setattr(config, "JUDGE_MODEL", "claude-sonnet-4-5-20250929")
        monkeypatch.setattr(config, "JUDGE_API_KEY", "k")
        st = client.get("/api/judge").json()
        assert st["configured"] and st["judge_family"] == "claude", st["reason"]
        assert st["calibration"]["calibrated"] is True
        r = client.post("/api/submissions", json={"hf_id": "org/model", "suite": "judged"})
        assert r.status_code == 200 and r.json()["status"] == "queued"
        assert set(config.tasks_for_suite("judged")) == set(tree["judged"]["manifest"]["tasks"])
        assert runner.include_args_for("exam_economics") == ["--include_path",
                                                             str(config.JUDGED_TASKS_DIR)]
        assert runner.include_args_for("fr_control_mmlu") == ["--include_path",
                                                              str(config.JUDGED_TASKS_DIR)]
        monkeypatch.setattr(config, "JUDGED_TASKS_DIR", tmp_path / "nowhere")
        assert "exam_build.py" in config.judged_blocked()
    finally:
        client.__exit__(None, None, None)
