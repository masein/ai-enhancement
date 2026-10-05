"""14.4.2: one answer key for Mobile-MMLU-Pro and the full Mobile-MMLU. One
pool of labels, by question — its id and a hash of its wording — so a question
the two sets word alike is labelled once, and the one worded apart (9933ec55;
inv00004 in the invented rows) twice, once in each wording. The same rule and
the same labellers as 14.3. Pro's questions go first: the full set's own are
sent once Pro's key is whole. The dry run shows Pro alone, the full set, and
what Start sends; the full set's key is checked against the paper's Table 2,
its own column. Invented rows, a fake labeller; nothing calls OpenRouter."""

from __future__ import annotations

import json
import time

import pytest

import mobile_mmlu as mmp
from conftest import make_service
from service import ai_models, config, llm, llm_poller, mmp_key
from test_14_3_mobile_mmlu import RIGHT, fake_pin, put_invented
from test_14_4_1_mobile_mmlu_data import put_full

# the invented full set's own questions, and their right answers
FULL_ONLY = {"inv00013": "A", "inv00014": "A", "inv00015": "A", "inv00016": "A",
             "inv00017": "A", "inv00018": "A"}


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


def test_a_question_both_sets_word_alike_is_one_label_and_the_one_worded_apart_two(both):
    pro, full = mmp.load(), mmp.load_full()
    rows = mmp.pool()
    assert len(rows) == 19                       # 12 Pro, 6 the full set's, inv00004 twice
    by = {}
    for q in rows:
        by.setdefault(q["id"], []).append(q)
    assert [sorted(q["sets"]) for q in by["inv00001"]] == [["full", "pro"]]
    assert [q["sets"] for q in by["inv00004"]] == [["pro"], ["full"]]
    assert len({q["lid"] for q in by["inv00004"]}) == 2
    assert by["inv00013"][0]["sets"] == ["full"]
    # Pro's first, in its order
    assert [q["id"] for q in rows[:12]] == [q["id"] for q in pro]
    # a label id is the id and its wording's hash: the manifest's for the real 9933ec55
    assert mmp.lid(next(q for q in full if q["id"] == "inv00004")).startswith("inv00004#")
    wa = mmp.full_manifest()["overlap"]["worded_apart"]["9933ec55"]
    assert mmp.lid({"id": "9933ec55", "question": "x"}) != f"9933ec55#{wa['pro']}"


def test_each_sets_key_is_its_view_of_the_pool(both):
    rows = mmp.pool()
    cur = {s: {"id": s, "version": f"{s}-v1"} for s in mmp.SLOTS}
    right = {**RIGHT, **FULL_ONLY}
    lab = {s: {q["lid"]: {"letter": right[q["id"]], "now": False, "version": f"{s}-v1"}
               for q in rows} for s in ("first", "second")}
    # the full set's wording of inv00004 has another answer than Pro's
    full_04 = next(q["lid"] for q in rows if q["id"] == "inv00004" and q["sets"] == ["full"])
    for s in ("first", "second"):
        lab[s][full_04]["letter"] = "D"
    k = mmp.build_key(rows, lab, cur)
    pro, full = mmp.set_view(k, "pro"), mmp.set_view(k, "full")
    assert len(pro["items"]) == 12 and len(full["items"]) == 18
    assert pro["items"]["inv00004"]["key"] == "C" and full["items"]["inv00004"]["key"] == "D"
    assert pro["items"]["inv00001"] is full["items"]["inv00001"]        # one label, both sets
    assert (pro["counts"]["all"]["kept"], full["counts"]["all"]["kept"]) == (12, 18)
    assert pro["version"] != full["version"] and len(full["version"]) == 10
    assert k["sets"]["full"]["version"] == full["version"]
    # Pro's key is as 14.3 scored it: by id, the same version for the same letters
    assert pro["version"] == mmp.key_version({q: {"key": right[q], "decision": "agreed"}
                                              for q in RIGHT})


def test_labels_kept_by_id_before_the_pool_are_moved_to_pros_wording(both):
    mmp._write(mmp.key_dir() / "labels.json", {
        "first": {"inv00004": {"letter": "C", "now": False, "version": "v1"}}})
    lab = mmp.read_labels()
    pro_04 = mmp.lid(mmp.by_id()["inv00004"])
    assert list(lab["first"]) == [pro_04]
    assert json.loads((mmp.key_dir() / "labels.json").read_text())["first"][pro_04]["letter"] == "C"


def test_pros_questions_go_first_and_the_rest_once_pros_key_is_whole(svc, monkeypatch):
    client, _ = svc
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(ai_models, "pin", fake_pin)
    monkeypatch.setattr(llm, "_http", lambda *a, **k: pytest.fail("nothing calls OpenRouter"))
    monkeypatch.setattr(ai_models, "drifted", lambda pin: "")
    asked: list[tuple[str, str, str]] = []
    right = {**RIGHT, **FULL_ONLY}
    split = {("openai", "inv00003"): "C"}                  # a Pro split, for the third

    def complete(self, row):
        k = row["custom_id"].split(":", 2)[2]
        qid, org = k.split("#")[0], self.model.split("/")[0]
        asked.append((org, qid, k))
        reply = {"answer": split.get((org, qid), right[qid]), "depends_on_now": False}
        return {"custom_id": row["custom_id"], "text": json.dumps(reply), "error": "",
                "attempts": 1, "finish_reason": "stop"}
    monkeypatch.setattr(mmp_key.LabellerChat, "_complete", complete)
    # 16c: "Pro, then the full set" is a choice on the card now (Pro only by default)
    assert client.post("/api/mobile-mmlu/key/scope", json={"scope": "all", "by": "masein"}) \
        .status_code == 200
    page = client.get("/api/mobile-mmlu/key").json()
    sets = page["estimate"]["sets"]
    assert (sets["pro"]["questions"], sets["full"]["questions"], sets["all"]["questions"]) == (
        12, 18, 19)
    assert sets["full"]["slots"]["first"]["questions"] == 18 and not asked
    r = client.post("/api/mobile-mmlu/key/start", json={"by": "masein"})
    assert r.status_code == 200, r.text
    # Start sends Pro's alone
    assert {k for x in r.json()["sent"].values() for k in [x["n"]]} == {12}
    end = time.time() + 20
    while time.time() < end and (mmp_key.pending() or mmp_key.left()["first"]):
        llm_poller.tick()
        time.sleep(0.05)
    pro = {q["lid"] for q in mmp.pool() if "pro" in q["sets"]}
    openai = [k for org, _, k in asked if org == "openai"]
    # every one of Pro's before any of the full set's own, and each label id once
    first_rest = next(i for i, k in enumerate(openai) if k not in pro)
    assert set(openai[:first_rest]) == pro and len(openai) == len(set(openai)) == 19
    assert [qid for org, qid, _ in asked if org == "anthropic"] == ["inv00003"]
    # inv00004 asked twice — once in each wording — and inv00001 once
    assert sum(1 for org, q, _ in asked if org == "openai" and q == "inv00004") == 2
    assert sum(1 for org, q, _ in asked if org == "openai" and q == "inv00001") == 1
    page = client.get("/api/mobile-mmlu/key").json()
    assert page["sets"]["pro"]["counts"]["all"]["kept"] == 12
    assert page["sets"]["full"]["counts"]["all"]["kept"] == 18


def test_the_dry_run_shows_pro_alone_the_full_set_and_what_start_sends(capsys):
    assert mmp.main(["--dry-run", "--stats"]) == 0
    out = capsys.readouterr().out
    pro = out[out.index("Mobile-MMLU-Pro alone:"):out.index("the full Mobile-MMLU")]
    full = out[out.index("the full Mobile-MMLU"):out.index("both, once each")]
    both = out[out.index("both, once each"):]
    assert "GPT-6 Sol: 9,497 questions" in pro and "Gemini 3.1 Pro: 9,497 questions" in pro
    assert "(Non-commercial: internal research only)" in full
    assert "GPT-6 Sol: 16,186 questions" in full and "Claude Sonnet 5.5: 1,619 questions" in full
    assert "GPT-6 Sol: 16,187 questions" in both and "Pro's questions first" in both


def test_the_full_sets_key_is_checked_against_its_own_column_of_table_2():
    full = mmp.paper_checks({"Qwen/Qwen2.5-3B-Instruct": {"acc": 0.672, "n": 15000},
                             "meta-llama/Llama-3.2-3B-Instruct": {"acc": 0.512, "n": 15000},
                             "google/gemma-2-2b-it": {"acc": 0.40, "n": 15000}}, "full")
    assert [r["paper"] for r in full["rows"]] == [68.1, 50.2, 38.9]
    assert [r["ok"] for r in full["rows"]] == [True, True, True] and not full["provisional"]
    assert "Mobile-MMLU (0-shot)" in full["setting"] and "Table 2" in full["table"]
    # Pro's numbers would be outside on the full set's column
    pro_like = mmp.paper_checks({"Qwen/Qwen2.5-3B-Instruct": {"acc": 0.606, "n": 9000}}, "full")
    assert pro_like["provisional"] and pro_like["rows"][0]["diff"] == -7.5


def test_a_model_with_a_full_set_score_cant_label(svc):
    d = config.OUT_DIR / "x-ai__grok-5"
    d.mkdir(parents=True, exist_ok=True)
    (d / mmp.FULL_PRED_FILE).write_text(json.dumps({"predictions": {}}))
    (d / "model_meta.json").write_text(json.dumps({"model": "x-ai/grok-5"}))
    assert "has a Mobile-MMLU score on the board" in mmp_key.refused("third", "x-ai/grok-5")
