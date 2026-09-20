"""build_payload on the fixture: what the dashboard is handed."""

from __future__ import annotations

import json
import os

import pytest

import make_fixture
import report_lm_eval as report


def _model(payload, mid):
    return next(m for m in payload["models"] if m["id"] == mid)


def test_every_fixture_model_is_on_the_board(payload, tree):
    assert {m["id"] for m in payload["models"]} == set(tree["models"])
    assert payload["accTasks"] == ["mmlu", "mmlu_perm", "hellaswag", "arc_challenge",
                                   "arc_easy", "winogrande", "piqa", "truthfulqa_mc2"]
    # MMLU's subjects and categories are children of the group, never headline
    assert not any(t.startswith("mmlu_") and t != "mmlu_perm" for t in payload["tasks"])
    assert any(row[1] == "mmlu_econometrics" for row in payload["extra"])


def test_official_requires_the_whole_protocol(payload, tree):
    for mid, m in tree["models"].items():
        row = _model(payload, mid)
        if mid in tree["taint"]:
            assert not row["official"] and row["avg"] is None and row["tainted"] == ["mmlu"]
            continue
        if set(make_fixture.FULL) <= set(m["tasks"]):
            assert row["official"] and row["avg"] is not None and row["missing"] == []
        else:
            assert not row["official"] and row["avg"] is None
            assert set(row["missing"]) == set(payload["required"]) - set(m["tasks"])
            assert row["partialAvg"] is not None      # a diagnostic, never a rank


def test_local_artifact_id_is_normalised_and_marked(payload, tree):
    row = _model(payload, tree["nodiag"])
    assert row["source"] == "artifact" and row["name"] == "nodiag-step400"
    assert all(_model(payload, mid)["source"] == "hub"
               for mid in tree["models"] if mid.startswith("fx/"))


def test_kind_follows_the_chat_template(payload):
    assert _model(payload, "fx/below-135m-it")["kind"] == "instruct"
    assert _model(payload, "fx/chance-160m")["kind"] == "base"
    assert any("Chat template applied to some models" in w for w in payload["warnings"])


def test_any_diag_and_salt(payload, tree, diag):
    assert payload["meta"]["anyDiag"] is True
    import diagnose as dx
    assert payload["meta"]["diagSalt"] == dx.SPLIT_SALT
    for mid in tree["models"]:
        row = _model(payload, mid)
        assert (row["diag"] is not None) == (mid in diag), mid
    assert _model(payload, tree["nodiag"])["diag"] is None


def test_any_diag_is_false_before_anyone_has_diagnosed(tmp_path):
    m = make_fixture.build(tmp_path, diagnose=False)
    runs = report.load_results(m["out_dir"])
    p = report.build_payload(report.merge_runs(runs), "t", source="")
    assert p["meta"]["anyDiag"] is False and p["meta"]["diagSalt"] is None
    assert all(row["diag"] is None for row in p["models"])


def test_trim_diag_caps_the_archive_for_the_page(diag):
    raw = diag["fx/skewed-360m"]
    t = report._trim_diag(raw)
    assert t["split_salt"] == raw["split_salt"]
    assert set(t["tasks"]) == set(raw["tasks"])
    for task, v in t["tasks"].items():
        for bucket, items in (v.get("examples") or {}).items():
            assert len(items) <= report._DIAG_EXAMPLES
            assert len(items) == min(len(raw["tasks"][task]["examples"][bucket]),
                                     report._DIAG_EXAMPLES)
            for e in items:
                assert len(e["q"]) <= report._DIAG_Q
                assert e["chose"] is None or len(e["chose"]) <= 70
                assert e["answer"] is None or len(e["answer"]) <= 70
                assert set(e) == {"group", "q", "chose", "answer", "p"}
        # the archive keeps the counts the page reads; nothing else rides along
        assert set(v) <= {"metric", "n", "n_report", "n_diagnose", "score_all",
                          "score_report", "score_diagnose", "buckets", "approx_buckets",
                          "groups", "answers", "examples", "categories", "unmapped"}


def test_trim_diag_clips_long_text_and_drops_empty():
    long_q = "q" * 500
    raw = {"split_salt": "s", "tasks": {"t": {
        "n": 1, "buckets": {"wrong": 1},
        "examples": {"wrong": [{"q": long_q, "chose": "c" * 200, "answer": None, "p": 0.5,
                                "group": "g"}] * 10, "right": []}}}}
    t = report._trim_diag(raw)
    ex = t["tasks"]["t"]["examples"]
    assert list(ex) == ["wrong"] and len(ex["wrong"]) == report._DIAG_EXAMPLES
    e = ex["wrong"][0]
    assert len(e["q"]) == report._DIAG_Q and e["q"].endswith("…")
    assert len(e["chose"]) == 70 and e["answer"] is None
    assert report._trim_diag(None) is None
    assert report._trim_diag({"tasks": {}}) is None
    assert report._trim_diag({"split_salt": "s", "tasks": {"t": "not a dict"}}) == {
        "split_salt": "s", "tasks": {}}


def test_beside_rereads_a_rewritten_file_and_never_caches_a_miss(tmp_path):
    """The service imports this module once and lives for weeks. A cache that
    remembered 'no diagnose.json here' by path meant a diagnosis written after
    startup never appeared until a restart. Real bug; keep the regression."""
    src = tmp_path / "m" / "task_5shot" / "run" / "results_x.json"
    src.parent.mkdir(parents=True)
    src.write_text("{}")
    report._META_CACHE.clear()
    assert report._beside(src, "diagnose.json") is None
    f = tmp_path / "m" / "diagnose.json"
    f.write_text(json.dumps({"tasks": {"a": 1}}))
    assert report._beside(src, "diagnose.json") == {"tasks": {"a": 1}}
    # rewritten with different content: re-read, not served from cache
    f.write_text(json.dumps({"tasks": {"a": 1, "b": 2}}))
    assert report._beside(src, "diagnose.json") == {"tasks": {"a": 1, "b": 2}}
    # same size, different content, newer mtime: still re-read
    f.write_text(json.dumps({"tasks": {"a": 1, "b": 3}}))
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
    assert report._beside(src, "diagnose.json") == {"tasks": {"a": 1, "b": 3}}
    # one level up works too (results files sit one or two levels below the model)
    assert report._beside(tmp_path / "m" / "task_5shot" / "results_y.json",
                          "diagnose.json") == {"tasks": {"a": 1, "b": 3}}
    # a file that is not JSON is a miss, not a crash
    f.write_text("{not json")
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 20_000_000_000))
    assert report._beside(src, "diagnose.json") is None


def test_item_count_disagreement_is_visible_in_the_payload(payload, diag, tree):
    mc = tree["miscount"]
    cell = payload["cells"][mc["task"]][mc["model"]]
    assert cell["n"] == mc["declared"]
    assert diag[mc["model"]]["tasks"][mc["task"]]["n"] == mc["logged"]
    assert cell["n"] != diag[mc["model"]]["tasks"][mc["task"]]["n"]
    # and every other model agrees with its own log on that task
    for mid, d in diag.items():
        if mid != mc["model"]:
            assert payload["cells"][mc["task"]][mid]["n"] == d["tasks"][mc["task"]]["n"]


def test_group_task_counts_come_from_its_leaves(payload, tree, diag):
    """The harness records n-samples and n-shot per leaf only; the group must
    still show an item count and a shot count, and they must agree with the
    per-item log so the Diagnose item-count check covers MMLU too."""
    for mid in tree["models"]:
        if "mmlu" not in tree["models"][mid]["tasks"]:
            continue
        c = payload["cells"]["mmlu"][mid]
        assert c["n"] == len(make_fixture.MMLU_SUBJECTS) * make_fixture.MMLU_PER_SUBJECT
        assert c["shots"] == 5
        if mid in diag:
            assert c["n"] == diag[mid]["tasks"]["mmlu"]["n"]
    c = payload["cells"]["mmlu_perm"]["fx/good-750m"]
    assert c["n"] == len(make_fixture.PERM_SUBJECTS) * make_fixture.PERM_PER_SUBJECT
    assert c["shots"] == 5
    # leaves keep their own counts; a lone task is untouched
    assert payload["cells"]["arc_easy"]["fx/good-750m"]["n"] == make_fixture.TASKS["arc_easy"]["n"]
    assert report._leaves("mmlu", {"mmlu": ["a", "b"], "a": ["a1", "a2"], "b": []}) == [
        "a1", "a2", "b"]
    assert report._leaves("x", {}) == ["x"]
    assert report._leaves("x", {"x": ["x"]}) == ["x"]                 # self-reference is not a child
    # disagreeing leaf shot counts produce no group shot count
    run = report.parse_run({"results": {}, "group_subtasks": {"g": ["a", "b"]},
                            "n-shot": {"a": 5, "b": 0},
                            "n-samples": {"a": {"effective": 10}, "b": {"effective": 5}}},
                           tree["out_dir"] / "x" / "y" / "z.json")
    assert run["n_samples"]["g"] == 15 and "g" not in run["n_shot"]


def test_model_meta_reaches_archinfo(payload):
    a = _model(payload, "fx/good-750m")["archinfo"]
    assert a["arch"] == "FixtureForCausalLM" and a["kind"] == "base"
    assert _model(payload, "fx/good-750m")["params"] == 750_000_000
    assert _model(payload, "fx/good-750m")["paramsSrc"] == "config"


def test_significance_is_pairwise_over_present_models(payload):
    rows = payload["sig"]["mmlu"]
    present = [m["id"] for m in payload["models"] if m["id"] in payload["cells"]["mmlu"]]
    assert len(rows) == len(present) * (len(present) - 1) // 2
    good_vs_chance = next(r for r in rows
                          if {r[0], r[1]} == {"fx/good-750m", "fx/chance-160m"})
    assert good_vs_chance[4] is True


def test_build_report_embeds_the_payload(tree):
    html = tree["report"].read_text(encoding="utf-8")
    assert '<script id="data" type="application/json">' in html
    blob = html.split('<script id="data" type="application/json">', 1)[1].split("</script>")[0]
    data = json.loads(blob.replace("<\\/", "</"))
    assert {m["id"] for m in data["models"]} == set(tree["models"])
    assert "</script>" not in blob            # the one sequence that could end the tag early


# ---------------------------------------------------------------------------
# phase 8c P5b: the same run submitted twice
# ---------------------------------------------------------------------------

def test_an_identical_row_is_shown_named_and_not_ranked(tree, tmp_path):
    """HANDOFF §11: two byte-identical rows sat at #1 and #2 for weeks. Both
    stay on the board — deleting a submission is not the page's call — but
    only one is ranked, and the other says which it duplicates."""
    import shutil
    out = tmp_path / "results" / "full"
    shutil.copytree(tree["out_dir"], out)
    src = out / "fx__good-750m"
    twin = out / "fx__good-750m-v2"
    shutil.copytree(src, twin)
    for f in twin.rglob("results*.json"):
        blob = json.loads(f.read_text())
        blob["config"]["model_args"] = blob["config"]["model_args"].replace(
            "fx/good-750m", "fx/good-750m-v2")
        f.write_text(json.dumps(blob))
    p = report.build_payload(report.merge_runs(report.load_results(out)), "t", source="")
    rows = {m["id"]: m for m in p["models"]}
    a, b = rows["fx/good-750m"], rows["fx/good-750m-v2"]
    # the earlier one keeps the rank; the other names it and carries no claim
    dup, keep = (b, a) if b.get("duplicateOf") else (a, b)
    assert dup["duplicateOf"] == keep["id"] and dup["duplicateOfName"] == keep["name"]
    assert not keep.get("duplicateOf")
    assert dup["avg"] == keep["avg"]                      # shown, not deleted
    assert any("score identically to another row" in w for w in p["warnings"])
    assert any("Nothing has been deleted" in w for w in p["warnings"])
    # a model that merely shares ONE task score is not a duplicate
    assert not any(m.get("duplicateOf") for m in p["models"]
                   if m["id"] not in (a["id"], b["id"]))


def test_a_single_shared_task_score_is_not_a_duplicate(tree):
    p = report.build_payload(report.merge_runs(report.load_results(tree["out_dir"])),
                             "t", source="")
    assert not any(m.get("duplicateOf") for m in p["models"])


def test_the_pages_javascript_parses(tmp_path):
    """The page is ~4k lines of JS inside a Python string: one unbalanced
    paren renders a blank page, and only a browser test would catch it. node
    is on the CI runner; where it is not, the browser tests still do."""
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available; the dashboard tests cover this in a browser")
    js = tmp_path / "page.js"
    js.write_text(report.JS, encoding="utf-8")
    r = subprocess.run([node, "--check", str(js)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def test_a_resubmission_that_finished_more_tasks_is_the_one_that_ranks(tree, tmp_path):
    """The pair on the live board: a run that failed on a perplexity task and
    its resubmission, byte-identical on all seven ranked tasks. Comparing
    'every task this row happens to have' made them look different, so both
    sat at #1 and #2. The ranked tasks are the test, and the more complete
    run is the one that keeps the rank."""
    import shutil
    out = tmp_path / "results" / "full"
    shutil.copytree(tree["out_dir"], out)
    src, twin = out / "fx__good-750m", out / "fx__good-750m-v2"
    shutil.copytree(src, twin)
    for f in twin.rglob("results*.json"):
        blob = json.loads(f.read_text())
        blob["config"]["model_args"] = blob["config"]["model_args"].replace(
            "fx/good-750m", "fx/good-750m-v2")
        f.write_text(json.dumps(blob))
    # only the resubmission finished the perplexity task; the first attempt
    # died on it, which is the whole difference between the two rows
    ppl = twin / "wikitext_0shot"
    ppl.mkdir(parents=True)
    sample = next(iter((twin / "mmlu_5shot").rglob("results*.json")))
    blob = json.loads(sample.read_text())
    blob["results"] = {"wikitext": {"alias": "wikitext", "word_perplexity,none": 12.5,
                                    "bits_per_byte,none": 0.9}}
    blob["n-samples"] = {"wikitext": {"original": 100, "effective": 100}}
    blob["configs"] = {"wikitext": {"num_fewshot": 0, "metric_list": [
        {"metric": "word_perplexity", "higher_is_better": False}]}}
    blob["higher_is_better"] = {"wikitext": {"word_perplexity": False}}
    (ppl / "results_2026-09-20T00-00-00.json").write_text(json.dumps(blob))
    p = report.build_payload(report.merge_runs(report.load_results(out)), "t", source="")
    rows = {m["id"]: m for m in p["models"]}
    a, b = rows["fx/good-750m"], rows["fx/good-750m-v2"]
    assert "wikitext" in p["pplTasks"] and "wikitext" not in p["required"]
    # identical on every ranked task, so: one run, one rank
    assert a["duplicateOf"] == b["id"] and a["duplicateOfName"] == b["name"]
    assert not b.get("duplicateOf")                 # the complete run keeps it
    assert a["avg"] == b["avg"]                     # shown, not deleted
    assert any("score identically to another row" in w for w in p["warnings"])


def test_two_rows_that_differ_on_a_ranked_task_are_not_a_duplicate(tree, tmp_path):
    import shutil
    out = tmp_path / "results" / "full"
    shutil.copytree(tree["out_dir"], out)
    src, twin = out / "fx__good-750m", out / "fx__good-750m-v3"
    shutil.copytree(src, twin)
    for f in twin.rglob("results*.json"):
        blob = json.loads(f.read_text())
        blob["config"]["model_args"] = blob["config"]["model_args"].replace(
            "fx/good-750m", "fx/good-750m-v3")
        if "piqa" in blob.get("results", {}):       # one ranked task moves
            for k in ("acc,none", "acc_norm,none"):
                if k in blob["results"]["piqa"]:
                    blob["results"]["piqa"][k] = blob["results"]["piqa"][k] - 0.05
        f.write_text(json.dumps(blob))
    p = report.build_payload(report.merge_runs(report.load_results(out)), "t", source="")
    rows = {m["id"]: m for m in p["models"]}
    assert not rows["fx/good-750m"].get("duplicateOf")
    assert not rows["fx/good-750m-v3"].get("duplicateOf")
