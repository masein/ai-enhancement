"""The 37-topic live check of 2026-09-22 (11a), at the API and below it.

Three of the five findings live here: documents that clustered on one corner
of a topic, missing documents nobody could explain, and a `local/` model the
search offered although its weights are not on this server. The other two —
the clipped More menu and "rubrics v?" — are in the browser file beside this
one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import exam_build as eb
from conftest import DOMAIN_LABELS, label_domains, make_service
from service import config, llm, llm_poller, suggest
from service import proposals as prop

TOPIC = "Economics"                 # the fixture's one topic above the 30-question floor
TASK = "exam_economics"
MODEL = "fx/good-750m"


@pytest.fixture
def gap(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, appmod, tree
    client.__exit__(None, None, None)


def _approved(client, tree, labels=True):
    """A proposal through to `approved`, on a topic whose questions carry
    domain labels unless a test asks for the fixture as built."""
    if labels:
        label_domains(tree["judged"]["exam_root"], TOPIC)
    pid = client.post("/api/proposals", json={"model": MODEL, "topic": TOPIC,
                                              "requested_by": "tester"}).json()["id"]
    assert llm_poller.tick() == 1
    r = client.post(f"/api/proposals/{pid}/approve", json={"approver": "Omar"})
    assert r.status_code == 200, r.text
    return pid


def model_dir(tree, mid=MODEL):
    return Path(tree["out_dir"]) / mid.replace("/", "__")


# ---------------------------------------------------------------------------
# 2. the documents are spread over the areas whose answers failed
# ---------------------------------------------------------------------------

def test_the_split_is_by_largest_remainder_and_one_each_while_it_can():
    # 9 / 5 / 4 failures over 7 documents: the worst gets 3, the others 2
    assert prop.allocate({"A": 9, "B": 5, "C": 4}, 7) == [("A", 3), ("B", 2), ("C", 2)]
    # fewer documents than areas: one each, worst first, and no area twice
    assert prop.allocate({"A": 9, "B": 5, "C": 4}, 2) == [("A", 1), ("B", 1)]
    # a long tail still gets a document each before the leader gets a second
    assert prop.allocate({"A": 40, "B": 1, "C": 1}, 4) == [("A", 2), ("B", 1), ("C", 1)]
    assert prop.allocate({}, 7) == [] and prop.allocate({"A": 3}, 0) == []


def test_the_requests_go_round_robin_so_a_batch_cut_short_still_spreads():
    plan = [{"domain": d, "documents": n} for d, n in (("A", 3), ("B", 2), ("C", 2))]
    # one document per request: A B C A B C A
    assert prop.focus_chunks(plan, 1, 7) == [("A", 1), ("B", 1), ("C", 1), ("A", 1),
                                             ("B", 1), ("C", 1), ("A", 1)]
    # two per request, which is what a paid provider gets for documents
    assert prop.focus_chunks(plan, 2, 7) == [("A", 2), ("B", 2), ("C", 2), ("A", 1)]
    # no plan: chunked as it was before 11a, with no domain
    assert prop.focus_chunks([], 2, 5) == [(None, 2), (None, 2), (None, 1)]


def test_only_a_closed_set_of_labels_counts_as_a_domain():
    labels = {"Classical Mechanics": 14, "Relativity": 8, "Quantum Mechanics": 10}
    assert prop.is_domain_label_set(labels)
    assert not prop.is_domain_label_set({})
    # a sentence per question — computer science's `intent` is one — is not a
    # label set, however many questions share the field
    assert not prop.is_domain_label_set(
        {"The user asks how to make their code run faster.": 9, "Another sentence here.": 7})
    assert not prop.is_domain_label_set({f"d{i}": 4 for i in range(16)})     # too many
    assert not prop.is_domain_label_set({"Relativity": 8, "Seen once": 1})   # a singleton
    assert not prop.is_domain_label_set({"a b c d e f g h i": 4})            # too many words
    assert not prop.is_domain_label_set({"x" * 65: 4})                       # too long
    # the audience line's rule is stricter and stays that way: no whitespace
    assert not prop.is_label_set({"Classical Mechanics": 14, "Relativity": 8})


def test_the_plan_counts_the_diagnose_half_and_only_that_half(gap):
    client, _, tree = gap
    label_domains(tree["judged"]["exam_root"], TOPIC)
    md = model_dir(tree)
    plan = prop.focus_plan(md, TASK, TOPIC, 7)
    assert plan and sum(p["documents"] for p in plan) == 7
    assert {p["domain"] for p in plan} <= set(DOMAIN_LABELS)
    assert all(p["failing_diagnose"] >= 1 for p in plan)
    # worst first, so a batch cut short spends its documents where it hurts
    assert plan == sorted(plan, key=lambda p: (-p["failing_diagnose"], p["domain"]))
    # and only the diagnose half is counted: the bank says which half each is
    bank = {r["qid"]: r for r in eb.load_bank(tree["judged"]["exam_root"])[TOPIC]}
    j = json.loads((md / "judge.json").read_text())
    weak = [it for it in j["tasks"][TASK]["items"]
            if it.get("graded") and (it.get("score") is not None) and it["score"] < 3]
    by_domain = {}
    for it in weak:
        if it["half"] == "diagnose":
            d = bank[it["qid"]]["meta"]["domain"]
            by_domain[d] = by_domain.get(d, 0) + 1
    assert {p["domain"]: p["failing_diagnose"] for p in plan} == {
        d: n for d, n in by_domain.items() if d in {p["domain"] for p in plan}}

    # a report-half answer that failed changes nothing: the plan never reads it
    jp = md / "judge.json"
    j = json.loads(jp.read_text())
    items = j["tasks"][TASK]["items"]
    rep = [it for it in items if it["half"] == "report" and (it.get("score") or 0) >= 3]
    assert rep, "the fixture has report-half answers that landed"
    for it in rep:
        it["score"], it["graded"] = 0, True
    jp.write_text(json.dumps(j), encoding="utf-8")
    assert prop.focus_plan(md, TASK, TOPIC, 7) == plan


def test_a_sentence_valued_domain_gets_no_focus_line(gap):
    client, _, tree = gap
    label_domains(tree["judged"]["exam_root"], TOPIC, labels=(
        "The person wants to know how prices adjust when demand rises.",
        "The person is asking what a central bank does about inflation.",
        "The person asks why two countries trade at all."))
    assert prop.focus_plan(model_dir(tree), TASK, TOPIC, 7) == []
    reqs = prop.generation_requests(1, "a spec", TOPIC, 4, "doc", 7, plan=[])
    assert reqs and not any("Focus:" in q.user for q in reqs)


def test_the_focus_line_carries_the_label_and_never_a_count(gap):
    client, _, tree = gap
    pid = _approved(client, tree)
    r = client.post(f"/api/proposals/{pid}/generate",
                    json={"requester": "Omar", "count": 12})
    assert r.status_code == 200, r.text
    did = r.json()["dataset_id"]
    fake = llm.client()
    gen = [q for q in fake.recorded() if q["custom_id"].startswith(f"gen:{did}:")]
    assert len(gen) == 6                                   # two documents a request
    focus = [q["meta"]["focus"] for q in gen]
    # 11e: generation takes the first twelve labels of the plan Approve froze
    f = client.get(f"/api/proposals/{pid}/focus?count=12").json()
    assert f["frozen"] is True and f["mode"] == "area"
    first = f["labels"][:12]
    order = list(dict.fromkeys(first))
    # round-robin: the first lap covers every area, so a batch cut short still
    # spreads — and over the whole batch each area gets what the plan says
    assert focus[:len(order)] == order
    got = {}
    for q in gen:
        got[q["meta"]["focus"]] = got.get(q["meta"]["focus"], 0) + q["meta"]["count"]
    assert got == {lab: first.count(lab) for lab in order}
    plan = [{"domain": lab, "documents": first.count(lab)} for lab in order]
    assert llm_poller.tick() == 1
    pv = client.get(f"/api/datasets/{did}").json()["provenance"]
    assert pv["focus_plan"] == plan
    assert sum(p["documents"] for p in pv["focus_plan"]) == 12
    assert pv["focus_mode"] == "area" and pv["focus_labels"] == first


def test_the_focus_endpoint_says_when_a_topic_has_no_labels(gap):
    client, _, tree = gap
    pid = _approved(client, tree, labels=False)
    j = client.get(f"/api/proposals/{pid}/focus?count=20").json()
    # 11e: the true reason, in the brief's words — never "carry no domain
    # labels" when labels exist; here there are none
    assert j["labels"] == [] and j["mode"] == "off"
    assert j["reason"] == "no sub-area labels on this topic's questions"
    assert client.get(f"/api/proposals/{pid}/focus?count=0").status_code == 422
    assert client.get("/api/proposals/9999/focus").status_code == 404


# ---------------------------------------------------------------------------
# 3. every document asked for is accounted for
# ---------------------------------------------------------------------------

SHORT = "A short note. " * 20               # 60 words: under DOC_MIN_WORDS


def _outcome_responder():
    """One planted failure per request, so a batch covers every outcome the
    accounting can record."""
    def responder(req):
        if not req.custom_id.startswith("gen:"):
            return llm.default_responder(req)
        k = int(req.custom_id.rsplit(":", 1)[1])
        if k == 1:
            return "I am sorry, I cannot write that."               # reply not JSON
        if k == 2:
            return json.dumps([{"text": "x " * 300}, {"title": "t", "text": "y " * 300}])
        if k == 3:
            return json.dumps([{"title": "Too brief", "text": SHORT},
                               {"title": "Also brief", "text": SHORT}])
        if k == 4:
            return ""                                               # empty reply
        if k == 5:
            raise RuntimeError("the provider fell over")
        return llm.default_responder(req)
    return responder


def test_every_document_asked_for_is_accounted_for(gap, monkeypatch):
    client, _, tree = gap
    pid = _approved(client, tree)
    monkeypatch.setattr(llm.FakeBatches, "responder", staticmethod(_outcome_responder()))
    did = client.post(f"/api/proposals/{pid}/generate",
                      json={"requester": "Omar", "count": 12}).json()["dataset_id"]
    assert llm_poller.tick() == 1
    d = client.get(f"/api/datasets/{did}").json()
    assert d["status"] == "ready", d["error"]
    it = d["provenance"]["items"]
    # the invariant: nothing asked for is unaccounted for
    assert it["requested"] == 12
    assert it["kept"] + len(it["missing"]) == it["requested"]
    whys = sorted(m["why"] for m in it["missing"])
    assert whys.count("reply not JSON") == 2                 # a whole request, two documents
    assert whys.count("empty reply") == 2
    assert whys.count("no title") == 1
    assert sum(1 for w in whys if w.startswith("too short (")) == 2
    assert sum(1 for w in whys if w.startswith("error: ")) == 2
    assert "the provider fell over" in " ".join(whys)
    # each missing document says which request it was, and which area it was for
    assert all(isinstance(m["request"], int) for m in it["missing"])
    plan_domains = {p["domain"] for p in d["provenance"]["focus_plan"]}
    assert {m["focus"] for m in it["missing"]} <= plan_domains
    # and the gap in the plan is visible: one area is short of its documents
    got = {}
    for m in it["missing"]:
        got[m["focus"]] = got.get(m["focus"], 0) + 1
    assert got and max(got.values()) >= 1


def test_a_reply_short_of_what_was_asked_says_so():
    """No outcome is invented: the reply that arrives with one of two
    documents accounts for the other as missing."""
    items, why = prop.read_reply(json.dumps([{"title": "One", "text": "w " * 300}]), "doc", 2)
    assert len(items) == 1 and why == ["not in the reply"]
    assert prop.read_reply("", "doc", 2)[1] == ["empty reply"] * 2
    assert prop.read_reply("nope", "doc", 2)[1] == ["reply not JSON"] * 2
    short = json.dumps([{"title": "t", "text": SHORT}])
    assert prop.read_reply(short, "doc", 1)[1] == ["too short (60 words)"]
    assert prop.read_reply(json.dumps([{"text": "w " * 300}]), "doc", 1)[1] == ["no title"]


# ---------------------------------------------------------------------------
# 4. a local/ model with no weights on this server
# ---------------------------------------------------------------------------

def test_a_local_model_with_results_but_no_weights_is_marked_and_refused(gap):
    client, _, tree = gap
    name = "qwen35-delta-moe-7d560104-step945"
    mid = f"local/{name}"
    # the search marks it: on the board, because its results came from
    # elsewhere, and no weights here to run it with
    board = {"models": [{"id": mid, "params": 7e9, "kind": "base"},
                        {"id": "local/uploaded-step100", "params": 1e9, "kind": "base"},
                        {"id": MODEL, "params": 7.5e8, "kind": "base"}]}
    cands = {c["id"]: c for c in suggest.local_candidates(board, [], ["uploaded-step100"])}
    assert cands[mid]["on_board"] is True and cands[mid]["weights"] is False
    assert cands["local/uploaded-step100"]["weights"] is True
    assert "weights" not in cands[MODEL]                  # not a local/ id at all
    assert suggest.suggest(name, list(cands.values()))["items"][0]["weights"] is False

    # and the API refuses it before anything is queued, in the runner's words
    (Path(tree["out_dir"]) / mid.replace("/", "__")).mkdir(parents=True, exist_ok=True)
    r = client.post("/api/submissions", json={"hf_id": mid, "kind": "base", "suite": "quick",
                                         "submitter": "Omar"})
    assert r.status_code == 422
    assert r.json()["detail"] == (
        "local/qwen35-delta-moe-7d560104-step945 has results on this board but no weights on "
        "this server — upload it first (POST /api/artifacts/"
        "qwen35-delta-moe-7d560104-step945) to run it here.")
    assert client.get("/api/submissions").json() == []
    # an id that is not on the board either is refused in preflight's words
    r = client.post("/api/submissions", json={"hf_id": "local/never-seen", "kind": "base",
                                         "suite": "quick", "submitter": "Omar"})
    assert r.status_code == 422 and "no uploaded artifact named" in r.json()["detail"]


def test_a_local_model_whose_weights_are_here_is_offered_and_accepted(gap):
    client, _, tree = gap
    name = "uploaded-step100"
    d = Path(config.ARTIFACTS_DIR) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    hits = client.get(f"/api/models/suggest?q={name}").json()["items"]
    me = [h for h in hits if h["id"] == f"local/{name}"]
    assert me and me[0]["weights"] is True
    r = client.post("/api/submissions", json={"hf_id": f"local/{name}", "kind": "base",
                                         "suite": "quick", "submitter": "Omar"})
    assert r.status_code == 200, r.text
