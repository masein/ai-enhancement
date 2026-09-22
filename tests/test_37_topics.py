"""Phase 10a: the 37-topic exam — the files, the loader, retiring the old five.

The retired five share three slugs with new topics (law, economics, computer
science), and the old rows on the box carry their topic in lower case in the
same files the new topics will write to. Everything here is about keeping the
two apart: by the topic string stored on each row, never by the slug."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import categories
import exam_build as eb
import judge as jd

REPO = Path(__file__).resolve().parents[1]
BANKS = REPO / "eval_tasks" / "fr" / "banks"
RUBRICS = REPO / "eval_tasks" / "fr" / "rubrics"
RETIRED = REPO / "eval_tasks" / "fr" / "retired"

# the five topics and their delivered banks, as the box holds them today
OLD = {"medicine & health": "medicine_v2.json", "law": "law_v2.json",
       "economics": "economics_v1.json", "computer science": "computer_science_v1.json",
       "physics & engineering": "physics_engineering_v1.json"}
REASON = "replaced by the 37-topic exam (2026-09-21)"


def seed_old_bank(root: Path, topics=OLD) -> dict[str, int]:
    """The rows an import under the old topic list wrote: same fields, the
    old topic string, in the file named by the old slug."""
    n = {}
    for topic, name in topics.items():
        items, _ = eb.unwrap_items(json.loads((RETIRED / name).read_text(encoding="utf-8")))
        for it in items:
            meta = {k: v for k, v in it.items() if k not in ("prompt", "reference", "notes")}
            eb.append_bank(root, {"qid": eb.qid_of(it["prompt"]), "topic": topic,
                                  "prompt": it["prompt"], "reference": eb.metadata_reference(it),
                                  "meta": meta, "source": Path(name).stem,
                                  "accepted_by": "masein", "accepted_at": 1.0, "edited": False})
        n[topic] = len(items)
    return n


def rows_on_disk(root: Path, slug: str) -> list[dict]:
    return eb._read(eb.bank_dir(root) / f"{slug}.jsonl")


# ---------------------------------------------------------------------------
# the files
# ---------------------------------------------------------------------------

def test_37_topics_arrived_with_three_files_each():
    """36 on 2026-09-21; Arts, delivered empty then, on 2026-09-22."""
    slugs = {categories.topic_slug(t) for t in eb.TOPICS}
    banks = sorted(p.name for p in BANKS.glob("*"))
    assert len(banks) == 37 and all(b.endswith("_v1.json") for b in banks)
    assert {b[:-len("_v1.json")] for b in banks} == slugs
    for b in banks:
        s = b[:-len("_v1.json")]
        assert (RUBRICS / f"{s}.criteria.json").is_file() and (RUBRICS / f"{s}.md").is_file()
        items = json.loads((BANKS / b).read_text(encoding="utf-8"))
        assert isinstance(items, list) and len(items) == 100, b      # a bare array of 100
    assert not (REPO / "docs" / "Knowledge Classification").exists()
    # the retired five, whole, outside every path the service reads
    assert sorted(p.name for p in RETIRED.glob("*.json")) == sorted(OLD.values())
    assert len(list((RETIRED / "rubrics").glob("*"))) == 10
    assert not (RUBRICS / "medicine_health.md").exists()


def test_no_new_question_repeats_an_old_one():
    old = {eb.qid_of(it["prompt"]) for name in OLD.values()
           for it in eb.unwrap_items(json.loads((RETIRED / name).read_text("utf-8")))[0]}
    new = [eb.qid_of(it["prompt"]) for b in BANKS.glob("*.json")
           for it in json.loads(b.read_text("utf-8"))]
    assert len(new) == len(set(new)) == 3700
    assert not old & set(new)


# ---------------------------------------------------------------------------
# the loader: two more criteria layouts, one internal shape
# ---------------------------------------------------------------------------

CRITERIA = sorted(RUBRICS.glob("*.criteria.json")) + sorted((RETIRED / "rubrics").glob(
    "*.criteria.json"))


def test_all_42_criteria_files_normalise_to_one_shape():
    """The 37 of the exam and the 5 retired ones."""
    assert len(CRITERIA) == 42
    for p in CRITERIA:
        raw = json.loads(p.read_text(encoding="utf-8"))
        assert jd.validate_criteria(raw) == [], p.name
        spec = jd.normalise_criteria(raw, p)
        assert spec["criteria"] and spec["flags"], p.name
        for c in spec["criteria"]:
            assert {"id", "name", "definition"} <= set(c), p.name
            assert c["id"] == c["id"].strip() and c["id"].replace("_", "a").isalnum()
        for f in spec["flags"]:
            assert {"id", "condition", "effect", "examples", "not_critical"} <= set(f), p.name
            assert jd.normalise_effect(f["effect"]) == f["effect"], p.name
            assert isinstance(f["examples"], list) and isinstance(f["not_critical"], list)
        assert all(isinstance(x, str) and x for x in spec["evaluation_principles"]), p.name
        # no flag key survives beside the list: a second copy would be read twice
        assert not set(spec) & (set(jd._FLAG_KEYS) - {"flags"}), p.name


@pytest.mark.parametrize("key,slugs", [
    ("critical_error", ["architecture_built_environment", "language_literature",
                        "mathematics_statistics", "political_science_international_relations"]),
    ("critical_flags", ["biology_life_sciences", "public_health_wellness"]),
])
def test_the_two_new_flag_layouts_become_the_list(key, slugs):
    have = sorted(p.name[:-len(".criteria.json")] for p in RUBRICS.glob("*.criteria.json")
                  if key in json.loads(p.read_text(encoding="utf-8")))
    assert have == slugs
    for s in slugs:
        raw = json.loads((RUBRICS / f"{s}.criteria.json").read_text(encoding="utf-8"))
        want = raw[key] if isinstance(raw[key], list) else [raw[key]]
        spec = jd.normalise_criteria(raw)
        assert [f["id"] for f in spec["flags"]] == [f["id"] for f in want]
        assert [f["condition"] for f in spec["flags"]] == [f["condition"] for f in want]
        # the sha is the file's, as delivered: nothing was rewritten to load it
        assert jd.rubric_for(f"exam_{s}").criteria_sha256 == eb.sha256_file(
            RUBRICS / f"{s}.criteria.json")


def test_informational_keys_ride_along_and_principles_are_read_under_every_name():
    raw = json.loads((RUBRICS / "law.criteria.json").read_text(encoding="utf-8"))
    assert jd.normalise_criteria(raw)["scale"] == raw["scale"]      # kept, never used
    for slug, key in (("engineering", "principles"),
                      ("psychology_cognitive_sciences", "important_evaluation_principles"),
                      ("computer_science", "evaluation_principles")):
        raw = json.loads((RUBRICS / f"{slug}.criteria.json").read_text(encoding="utf-8"))
        said = jd.normalise_criteria(raw)["evaluation_principles"]
        assert len(said) == len(raw[key]) and all("{" not in s for s in said), slug
        if isinstance(raw[key][0], dict):             # a heading and a sentence, in words
            head = raw[key][0].get("name") or raw[key][0].get("title")
            assert said[0].startswith(f"{head}: ")


def test_acuity_reads_from_emergency_down_and_a_constant_one_is_still_a_sentence():
    order = ["emergency", "critical", "urgent", "high", "moderate", "mild", "routine"]
    assert jd._value_order("acuity", set(order)) == order
    assert jd._value_order("acuity", {"routine", "high", "critical"}) == [
        "critical", "high", "routine"]
    items = [{"meta": {"acuity": "routine", "difficulty": d, "domain": f"d{d % 3}"}}
             for d in range(1, 6)] * 4
    fields, constant = jd.breakdown_fields(items)
    assert "acuity" not in fields and "acuity" in constant
    # and the delivered banks carry the two new levels where the brief says
    seen = {}
    for b in BANKS.glob("*.json"):
        seen[b.name[:-len("_v1.json")]] = {it.get("acuity") for it in
                                           json.loads(b.read_text("utf-8"))}
    assert {s for s, v in seen.items() if "critical" in v} == {
        "engineering", "government_public_policy", "it", "manufacturing_applied_sciences"}
    assert {s for s, v in seen.items() if "high" in v} == {
        "architecture_built_environment", "law", "systems_cybersecurity"}


# ---------------------------------------------------------------------------
# import-dir
# ---------------------------------------------------------------------------

def test_import_dir_on_the_real_folder_is_37_topics_of_100(tmp_path):
    root = tmp_path / "exam"
    done = eb.import_dir(root, BANKS, "masein")
    assert len(done) == 37 and all(r["imported"] == 100 for r in done)
    bank = eb.load_bank(root)
    assert {t: len(r) for t, r in bank.items()} == {t: 100 for t in eb.TOPICS}
    for t, rows in bank.items():
        for r in rows:
            assert r["topic"] == t and r["accepted_by"] == "masein"
            assert r["source"] == f"{categories.topic_slug(t)}_v1"         # the file's stem
            assert "imported_by" not in r
    # the halves come from the qid, as for every question: about 50/50
    s = eb.summary(root)
    assert all(35 <= s[t]["report"] <= 65 for t in eb.TOPICS)
    # idempotent, like import
    again = eb.import_dir(root, BANKS, "masein")
    assert sum(r["imported"] for r in again) == 0 and sum(r["skipped"] for r in again) == 3700


def test_import_dir_refuses_a_file_no_topic_owns_by_name_and_writes_nothing(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    for name in ("law_v1.json", "sociology_v1.json"):
        (folder / name).write_bytes((BANKS / name).read_bytes())
    (folder / "underwater_basketweaving_v1.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="underwater_basketweaving_v1.json") as e:
        eb.import_dir(tmp_path / "exam", folder, "masein")
    assert "nothing was imported" in str(e.value)
    assert not eb.bank_dir(tmp_path / "exam").exists()
    # two versions of one topic's bank: which is the bank is a person's call
    (folder / "underwater_basketweaving_v1.json").unlink()
    (folder / "law_v2.json").write_bytes((BANKS / "law_v1.json").read_bytes())
    with pytest.raises(ValueError, match="law_v1.json, law_v2.json"):
        eb.import_dir(tmp_path / "exam", folder, "masein")


def test_the_cli_prints_a_line_per_topic_then_a_total(tmp_path):
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "exam_build.py"), "--root",
                          str(tmp_path / "exam"), "import-dir", str(BANKS), "--approver",
                          "masein"], capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert sum(1 for ln in lines if " imported 100 " in ln) == 37
    assert any(ln.startswith("Medicine & Clinical Health ") and "source medicine_clinical_"
               "health_v1" in ln for ln in lines)
    total = next(ln for ln in lines if ln.startswith("total:"))
    assert "37 topics, imported 3700" in total and "written by masein" in total


# ---------------------------------------------------------------------------
# retire
# ---------------------------------------------------------------------------

def test_the_old_rows_never_leak_into_the_topic_that_shares_their_slug(tmp_path):
    root = tmp_path / "exam"
    seed_old_bank(root)
    # before anything is retired: `law` is not `Law`, even in the same file
    assert eb.load_bank(root)["Law"] == []
    eb.import_bank(root, BANKS / "law_v1.json", "Law", "masein", "law_v1")
    assert len(eb.load_bank(root)["Law"]) == 100
    assert len(rows_on_disk(root, "law")) == 200                     # one file, two topics
    assert eb.summary(root)["Law"]["accepted"] == 100


def test_retiring_law_leaves_Law_untouched(tmp_path):
    root = tmp_path / "exam"
    seed_old_bank(root, {"law": "law_v2.json"})
    eb.import_bank(root, BANKS / "law_v1.json", "Law", "masein", "law_v1")
    before = {r["qid"]: r for r in rows_on_disk(root, "law") if r["topic"] == "Law"}
    r = eb.retire(root, "law", REASON)
    assert (r["rows"], r["changed"]) == (100, 100)
    rows = rows_on_disk(root, "law")
    assert len(rows) == 200                                          # nothing deleted
    for row in rows:
        if row["topic"] == "law":
            assert row["retired_reason"] == REASON and row["retired_at"]
        else:
            assert row == before[row["qid"]]                         # not one byte moved
    assert len(eb.load_bank(root)["Law"]) == 100
    # retiring again changes nothing and says so
    assert eb.retire(root, "law", REASON)["changed"] == 0


def test_retiring_Law_leaves_law_untouched(tmp_path):
    root = tmp_path / "exam"
    seed_old_bank(root, {"law": "law_v2.json"})
    eb.import_bank(root, BANKS / "law_v1.json", "Law", "masein", "law_v1")
    old = {r["qid"]: r for r in rows_on_disk(root, "law") if r["topic"] == "law"}
    assert eb.retire(root, "Law", "a test of the reverse")["changed"] == 100
    for row in rows_on_disk(root, "law"):
        if row["topic"] == "law":
            assert row == old[row["qid"]]
        else:
            assert row["retired_reason"] == "a test of the reverse"
    assert eb.load_bank(root)["Law"] == []


def test_retire_refuses_a_name_no_row_carries_and_needs_a_reason(tmp_path):
    root = tmp_path / "exam"
    seed_old_bank(root, {"law": "law_v2.json"})
    with pytest.raises(ValueError, match="'Medicine & Health'.*The bank holds: 'law'"):
        eb.retire(root, "Medicine & Health", REASON)
    with pytest.raises(ValueError, match="reason"):
        eb.retire(root, "law", "  ")
    assert not any(r.get("retired_at") for r in rows_on_disk(root, "law"))


def test_retired_rows_are_out_of_the_counts_the_imports_and_the_provenance(tmp_path):
    root = tmp_path / "exam"
    seed_old_bank(root, {"law": "law_v2.json"})
    eb.import_bank(root, BANKS / "law_v1.json", "Law", "masein", "law_v1")
    eb.retire(root, "law", REASON)
    # a retired question may be asked again under the new topic: it is new there
    old = json.loads((RETIRED / "law_v2.json").read_text(encoding="utf-8"))[:3]
    r = eb.import_bank(root, old, "Law", "masein", "law_again")
    assert (r["imported"], r["skipped"]) == (3, 0)
    assert len(eb.load_bank(root)["Law"]) == 103
    assert len([x for x in rows_on_disk(root, "law") if x["qid"] == eb.qid_of(
        old[0]["prompt"])]) == 2                                     # history kept beside it
    # correcting the new topic's provenance never touches the retired rows
    p = eb.set_provenance(root, "Law", approver="someone else")
    assert (p["rows"], p["changed"]) == (103, 103)
    assert all(x["accepted_by"] == "masein" for x in rows_on_disk(root, "law")
               if x["topic"] == "law")


def test_the_deploy_steps_build_exactly_the_37_new_banks_and_the_control(tree, tmp_path):
    """The brief's deploy sequence, on a bank shaped like the box's: the five
    old topics (and the tasks an earlier build wrote for them), retired by
    name, then the folder imported, then a build."""
    root = tmp_path / "exam"
    seed_old_bank(root)
    tasks = eb.tasks_dir(root)
    tasks.mkdir(parents=True)
    for stale in ("exam_medicine_health", "exam_physics_engineering", "exam_other"):
        (tasks / f"{stale}.jsonl").write_text("{}\n", encoding="utf-8")
        (tasks / f"{stale}.yaml").write_text(f"task: {stale}\n", encoding="utf-8")
    for topic in OLD:
        assert eb.retire(root, topic, REASON)["changed"] == 100
    eb.import_dir(root, BANKS, "masein")
    m = eb.build(tree["out_dir"], root)
    exam = sorted(t for t in m["tasks"] if t != eb.CONTROL_TASK)
    assert exam == sorted(eb.topic_task(t) for t in eb.TOPICS)
    assert len(exam) == 37
    assert all(m["tasks"][t]["items"] == 100 for t in exam)
    assert sorted(p.stem for p in tasks.glob("*.yaml")) == sorted(exam + [eb.CONTROL_TASK])
    assert m["removed"] == ["exam_medicine_health", "exam_other", "exam_physics_engineering"]
    # no retired question is in any built task
    old = {eb.qid_of(it["prompt"]) for name in OLD.values()
           for it in eb.unwrap_items(json.loads((RETIRED / name).read_text("utf-8")))[0]}
    built = {json.loads(ln)["qid"] for t in exam
             for ln in (tasks / f"{t}.jsonl").read_text("utf-8").splitlines()}
    assert len(built) == 3700 and not built & old
    # the control set: only topics MMLU has subjects for, ten at most each
    ctl = m["tasks"][eb.CONTROL_TASK]
    assert set(ctl["per_category"]) <= set(categories.with_subjects())
    assert all(n <= eb.CONTROL_PER_CATEGORY for n in ctl["per_category"].values())
    # and the summary is the 37 topics, the old five nowhere
    s = eb.summary(root)
    assert list(s) == eb.TOPICS and all(s[t]["accepted"] == 100 for t in eb.TOPICS)
    assert not set(OLD) & set(s)


def test_the_cli_retires_by_name_and_says_what_it_changed(tmp_path):
    root = tmp_path / "exam"
    seed_old_bank(root, {"law": "law_v2.json"})
    cmd = [sys.executable, str(REPO / "scripts" / "exam_build.py"), "--root", str(root)]
    out = subprocess.run(cmd + ["retire", "--topic", "law", "--reason", REASON],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == f"law: retired 100 of 100 rows in law.jsonl · reason: {REASON}"
    out = subprocess.run(cmd + ["retire", "--topic", "Law", "--reason", REASON],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 2 and "no question in the bank has the topic 'Law'" in out.stderr


def test_a_calibration_export_reaches_every_topic_and_every_score():
    """The export's promise — every category and every score level — held
    for 15 categories. With 37 × 5 strata a 100-row sample taken as one flat
    round-robin stopped at the twentieth topic of the alphabet."""
    import collections

    import judge_calibrate as jc
    rows = [{"id": f"{t}|{i}", "category": t, "judge_score": i % 5}
            for t in eb.TOPICS if t != "Arts" for i in range(40)]
    picked = jc.sample(rows, 100, seed=7)
    per_topic = collections.Counter(r["category"] for r in picked)
    assert len(picked) == 100 and len(per_topic) == 36
    assert max(per_topic.values()) - min(per_topic.values()) <= 1
    assert set(collections.Counter(r["judge_score"] for r in picked)) == {0, 1, 2, 3, 4}
    assert jc.sample(rows, 100, seed=7) == picked                    # deterministic
