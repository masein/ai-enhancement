"""11e item 1: spread the documents on every topic.

Mathematics & Statistics has 18 labels and 11a's cap of 15 refused them, so 13
of its 20 documents were about conditional probability. Now a small label set
is used as written, a large one by the prefix its labels share, and anything
larger one failed concept at a time. The plan is shown before Approve, Approve
freezes it, and Generate takes its first N labels. Only diagnose-half items
are ever read to build a request.
"""

from __future__ import annotations

import collections
import json
import re
import sqlite3
from pathlib import Path

import pytest

import exam_build as eb
from conftest import label_domains, make_service
from service import config, llm, llm_poller
from service import proposals as prop

REPO = Path(__file__).resolve().parents[1]
TOPIC = "Economics"
TASK = "exam_economics"
MODEL = "fx/good-750m"

# the brief's tables, as the banks were delivered
BY_AREA_AS_WRITTEN = {"ai_machine_learning": 6, "finance_accounting": 9, "physics_astronomy": 12,
                      "food_veterinary_sciences": 16, "mathematics_statistics": 18,
                      "earth_environmental_sciences": 21, "general_multidisciplinary": 22,
                      "agriculture": 25}
BY_AREA_GROUPED = {"architecture_built_environment": (86, 7), "design": (97, 11),
                   "education": (100, 11), "ethics_religion": (95, 8),
                   "history_archaeology": (97, 11), "law": (52, 8),
                   "manufacturing_applied_sciences": (88, 10), "media_communication": (100, 10),
                   "philosophy": (81, 9), "political_science_international_relations": (100, 9)}


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, tree
    client.__exit__(None, None, None)


def exam_root(tree):
    return tree["judged"]["exam_root"]


def propose(client):
    pid = client.post("/api/proposals", json={"model": MODEL, "topic": TOPIC,
                                              "requested_by": "tester"}).json()["id"]
    assert llm_poller.tick() == 1
    return pid


def approve(client, pid, spread=True):
    r = client.post(f"/api/proposals/{pid}/approve", json={"approver": "Omar", "spread": spread})
    assert r.status_code == 200, r.text
    return r.json()


def generate(client, pid, count):
    r = client.post(f"/api/proposals/{pid}/generate", json={"requester": "Omar", "count": count})
    assert r.status_code == 200, r.text
    did = r.json()["dataset_id"]
    reqs = [q for q in llm.client().recorded() if q["custom_id"].startswith(f"gen:{did}:")]
    return did, reqs


def focus_lines(reqs):
    return [line for q in reqs for line in q["user"].splitlines() if line.startswith("Focus:")]


def failing_diagnose(tree, labels_by_qid):
    """The fixture's failed diagnose-half items, weakest first, ties by qid."""
    md = Path(tree["out_dir"]) / MODEL.replace("/", "__")
    items = json.loads((md / "judge.json").read_text())["tasks"][TASK]["items"]
    return sorted((it for it in items if it["half"] == "diagnose" and it.get("graded")
                   and it.get("score") is not None and it["score"] < prop.WEAK_SCORE),
                  key=lambda it: (it["score"], str(it["qid"])))


# ---------------------------------------------------------------------------
# the rule, on its own
# ---------------------------------------------------------------------------

def test_a_labels_group_is_its_prefix_and_a_word_hyphen_never_splits():
    assert prop.label_group("Legal Method – Precedent") == "Legal Method"
    assert prop.label_group("Manufacturing Processes - Casting") == "Manufacturing Processes"
    assert prop.label_group("Ethics — Virtue") == "Ethics"
    assert prop.label_group("Design: Colour Theory") == "Design"
    assert prop.label_group("Media / Broadcast") == "Media"
    assert prop.label_group("Evidence-Based Practice") == "Evidence-Based Practice"
    assert prop.label_group("Algebra") == "Algebra"


def test_a_small_set_is_used_as_written_and_a_prefixed_one_by_its_groups():
    maths = [f"Topic {i}" for i in range(18)]
    mode, key = prop.focus_scheme(maths * 5)
    assert mode == "area" and key("Topic 3") == "Topic 3"
    law = [f"Legal Method – Case {i}" for i in range(30)] + [f"Contract – Term {i}" for i in range(22)]
    mode, key = prop.focus_scheme(law)
    assert mode == "area" and key("Legal Method – Case 4") == "Legal Method"
    mfg = [f"Manufacturing Processes - Step {i}" for i in range(60)] + \
          [f"Quality Control - Check {i}" for i in range(28)]
    mode, key = prop.focus_scheme(mfg)
    assert mode == "area" and key("Quality Control - Check 1") == "Quality Control"
    # one label per question, and no shared prefix: one concept at a time
    assert prop.focus_scheme([f"Concept {i}" for i in range(100)])[0] == "concept"
    assert prop.focus_scheme([]) == (None, None)


def test_the_37_real_banks_split_18_by_area_and_19_by_concept():
    modes = collections.Counter()
    for p in sorted((REPO / "eval_tasks" / "fr" / "banks").glob("*_v1.json")):
        slug = p.name.removesuffix("_v1.json")
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("items") or data.get("questions")
        labels = [str(it.get("domain") or "").strip() for it in items]
        distinct = {x for x in labels if x}
        mode, key = prop.focus_scheme(labels)
        modes[mode] += 1
        if slug in BY_AREA_AS_WRITTEN:
            assert mode == "area" and len(distinct) == BY_AREA_AS_WRITTEN[slug], slug
            assert {key(x) for x in distinct} == distinct, slug             # as written
        elif slug in BY_AREA_GROUPED:
            n, g = BY_AREA_GROUPED[slug]
            assert mode == "area" and len(distinct) == n, slug
            assert len({key(x) for x in distinct}) == g, slug               # by prefix
        else:
            assert mode == "concept", slug
    assert modes == {"area": 18, "concept": 19}


def test_a_long_label_sends_its_group_and_a_hopeless_one_is_skipped():
    long = "Legal Method – " + "a very long description of precedent " * 3
    assert len(long) > prop.FOCUS_MAX_CHARS
    assert prop.to_send(long) == "Legal Method"
    assert prop.to_send("Algebra") == "Algebra"
    assert prop.to_send("Why do markets fail? A question.") is None
    assert prop.to_send("x" * 70) is None


# ---------------------------------------------------------------------------
# the plan, on a served board
# ---------------------------------------------------------------------------

MATHS = ("Algebra", "Probability", "Linear Algebra", "Calculus", "Statistics", "Geometry",
         "Number Theory", "Combinatorics", "Topology", "Logic", "Set Theory", "Analysis",
         "Optimisation", "Graph Theory", "Differential Equations", "Numerical Methods",
         "Game Theory", "Measure Theory")


def test_a_topic_shaped_like_mathematics_is_spread_by_area(svc):
    client, tree = svc
    label_domains(exam_root(tree), TOPIC, labels=MATHS, both_halves=False)
    pid = propose(client)
    before = client.get(f"/api/proposals/{pid}/focus?count=20").json()
    assert before["frozen"] is False and before["mode"] == "area"
    assert len(before["labels"]) == prop.FOCUS_LIST_LEN
    # more than 15 labels, which 11a refused: every failing area gets a share
    assert len(set(before["labels"][:20])) >= min(len(before["failing"]), 20) > 3
    approve(client, pid)
    did, reqs = generate(client, pid, 12)
    lines = focus_lines(reqs)
    assert lines and all(re.fullmatch(r"Focus: [^\d.?!]+", x) for x in lines)
    assert {x[len("Focus: "):] for x in lines} <= set(MATHS)


def test_a_topic_shaped_like_sociology_is_spread_by_concept(svc):
    client, tree = svc
    rows = eb.load_bank(exam_root(tree))[TOPIC]
    unique = tuple(f"Concept {i:02d}" for i in range(len(rows)))       # one label per question
    label_domains(exam_root(tree), TOPIC, labels=unique, both_halves=False)
    rows = eb.load_bank(exam_root(tree))[TOPIC]
    label_of = {r["qid"]: r["meta"]["domain"] for r in rows}
    pid = propose(client)
    f = client.get(f"/api/proposals/{pid}/focus?count=20").json()
    assert f["mode"] == "concept"
    want = list(dict.fromkeys(label_of[it["qid"]] for it in failing_diagnose(tree, label_of)))
    assert want, "the fixture has failed diagnose-half answers"
    # each document a distinct failed label, weakest first; then round again
    assert f["labels"][:len(want)] == want
    assert f["labels"][len(want):2 * len(want)] == want[:prop.FOCUS_LIST_LEN - len(want)]
    approve(client, pid)
    did, reqs = generate(client, pid, len(want))
    assert sorted(x[len("Focus: "):] for x in focus_lines(reqs)) == sorted(want)
    assert all(q["meta"]["count"] == 1 for q in reqs)                    # one label, one document


def test_a_failed_report_half_label_never_reaches_a_request(svc):
    client, tree = svc
    rows = eb.load_bank(exam_root(tree))[TOPIC]
    unique = tuple(f"Concept {i:02d}" for i in range(len(rows)))
    label_domains(exam_root(tree), TOPIC, labels=unique, both_halves=False)
    rows = eb.load_bank(exam_root(tree))[TOPIC]
    label_of = {r["qid"]: r["meta"]["domain"] for r in rows}
    # every report-half answer fails, so each report-half label is a failed one
    md = Path(tree["out_dir"]) / MODEL.replace("/", "__")
    j = json.loads((md / "judge.json").read_text())
    report = [it for it in j["tasks"][TASK]["items"] if it["half"] == "report"]
    for it in report:
        it["score"], it["graded"] = 0, True
    (md / "judge.json").write_text(json.dumps(j), encoding="utf-8")
    secret = {label_of[it["qid"]] for it in report if it["qid"] in label_of}
    assert secret
    pid = propose(client)
    f = client.get(f"/api/proposals/{pid}/focus?count=100").json()
    assert not secret & set(f["labels"])
    approve(client, pid)
    did, reqs = generate(client, pid, 100)
    bodies = "\n".join(q["system"] + "\n" + q["user"] for q in reqs)
    assert not [s for s in secret if f"Focus: {s}\n" in bodies]


def test_the_dataset_uses_the_first_n_labels_of_the_frozen_plan(svc):
    client, tree = svc
    label_domains(exam_root(tree), TOPIC, labels=MATHS, both_halves=False)
    pid = propose(client)
    shown = client.get(f"/api/proposals/{pid}/focus?count=12").json()["labels"]
    approve(client, pid)
    frozen = json.loads(sqlite3.connect(config.DB_PATH).execute(
        "SELECT approved_focus FROM proposals WHERE id=?", (pid,)).fetchone()[0])
    assert frozen["mode"] == "area" and frozen["labels"] == shown   # what was shown is what froze
    did, reqs = generate(client, pid, 12)
    sent = collections.Counter(x[len("Focus: "):] for x in focus_lines(reqs)
                               for _ in range(1))
    per_req = collections.Counter()
    for q in reqs:
        per_req[q["meta"]["focus"]] += q["meta"]["count"]
    assert per_req == collections.Counter(frozen["labels"][:12])
    assert set(sent) == set(per_req)
    assert llm_poller.tick() == 1
    pv = client.get(f"/api/datasets/{did}").json()["provenance"]
    assert pv["focus_mode"] == "area" and pv["focus_labels"] == frozen["labels"][:12]


def test_unticking_the_box_sends_no_focus_and_says_why(svc):
    client, tree = svc
    label_domains(exam_root(tree), TOPIC, labels=MATHS, both_halves=False)
    pid = propose(client)
    assert approve(client, pid, spread=False)["focus_mode"] == "off"
    f = client.get(f"/api/proposals/{pid}/focus?count=20").json()
    assert f["mode"] == "off" and f["labels"] == []
    assert f["reason"] == "the approver turned spreading off"
    did, reqs = generate(client, pid, 6)
    assert focus_lines(reqs) == []
    assert llm_poller.tick() == 1
    pv = client.get(f"/api/datasets/{did}").json()["provenance"]
    assert pv["focus_mode"] == "off" and pv["focus_labels"] == []


def test_the_true_reason_when_there_is_no_plan(svc):
    client, tree = svc
    pid = propose(client)                          # the fixture's questions carry no labels
    f = client.get(f"/api/proposals/{pid}/focus").json()
    assert f["reason"] == "no sub-area labels on this topic's questions"
    assert "carry no domain labels" not in json.dumps(f)
    # labels, every one of them a sentence: too long to send, even by group
    label_domains(exam_root(tree), TOPIC, both_halves=False, labels=(
        "The person asks how prices adjust when demand rises and supply is fixed.",
        "The person asks why central banks raise rates when inflation climbs."))
    f = client.get(f"/api/proposals/{pid}/focus").json()
    assert f["labels"] == [] and f["reason"] == "every label is too long to send"


def test_a_proposal_approved_before_11e_keeps_its_old_behaviour(svc):
    client, tree = svc
    label_domains(exam_root(tree), TOPIC)          # three labels: 11a's rule accepts them
    pid = propose(client)
    approve(client, pid)
    with sqlite3.connect(config.DB_PATH) as c:     # as a proposal approved before 11e is
        c.execute("UPDATE proposals SET approved_focus=NULL WHERE id=?", (pid,))
    f = client.get(f"/api/proposals/{pid}/focus?count=12").json()
    assert f["legacy"] is True and f["frozen"] is False and f["plan"]
    did, reqs = generate(client, pid, 12)
    assert focus_lines(reqs)                       # 11a's plan, computed at Generate
