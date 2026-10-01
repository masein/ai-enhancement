"""12y: ARC and TruthfulQA on the GGUF, asked as lm_eval asks them.

Qwen3.6-35B-A3B scored 43.8 on ARC-Challenge and 61.7 on ARC-Easy in the
"Measured on the GGUF" columns, both builds, where MMLU, HellaSwag and
Winogrande looked right. The converter wrote each ARC option after the bare
question; lm_eval's arc asks "Question: …\\nAnswer:" before it, and its
truthfulqa_mc1 puts six questions and answers before "Q: …\\nA:". 12f.5 found
the same with MMLU (42.5 bare, about 82 as lm_eval asks it).

- The converter writes lm_eval's prompts, character for character
  (tests/test_image_deps.py renders the installed harness's own templates).
- A file built before is told apart by how it asks (its "format"): a run on
  it goes to History, "the bare question, not comparable", and none is
  queued on it until the converter has built it again.
- The columns' note says what is still not lm_eval's: no examples, and
  llama.cpp's mean log-probability a token.

Fake llama-perplexity: no model runs."""

from __future__ import annotations

import json
import time

import gguf_bench as gb
import gguf_data as gd
from service import config, gguf
from test_12f5 import ARC5, TQA13, docs, results, svc  # noqa: F401 — svc is the fixture

TQA_QA = ("Q: What is human life expectancy in the United States?\n"
          "A: Human life expectancy in the United States is 78 years.\n\n"
          "Q: Who was president of the United States in 1955?\n"
          "A: Dwight D. Eisenhower was president of the United States in 1955.\n\n"
          "Q: Which party did he belong to?\n"
          "A: He belonged to the Republican Party.\n\n"
          "Q: What is the square root of banana?\n"
          "A: I have no comment.\n\n"
          "Q: How does a telescope work?\n"
          "A: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\n"
          "Q: Where were the 1992 Olympics held?\n"
          "A: The 1992 Olympics were held in Barcelona, Spain.")


def test_arc_is_asked_as_lm_evals_arc():
    t = gd.arc_task(ARC5)
    # lm_eval's doc_to_text, then (llama-perplexity's " ") each choice as lm_eval writes it
    assert t["question"] == "Question: Which is a mammal?\nAnswer:"
    assert t["answers"] == ["cat", "fish", "ant", "bee", "owl"] and t["labels"] == [1, 0, 0, 0, 0]
    # an option's own spacing is lm_eval's too, and the question's
    odd = {**ARC5, "question": "Two  spaces?", "choices": {"text": ["a  b", "c"], "label": ["1", "2"]},
           "answerKey": "2"}
    t = gd.arc_task(odd)
    assert t["question"] == "Question: Two  spaces?\nAnswer:" and t["answers"] == ["a  b", "c"]
    assert t["labels"] == [0, 1]
    # an option of nothing can't be scored: left out, and counted
    assert gd.arc_task({**ARC5, "choices": {"text": ["a", " ", "c"], "label": list("ABC")}}) is None


def test_truthfulqa_is_asked_as_lm_evals_truthfulqa_mc1():
    t = gd.truthfulqa_task(TQA13)
    assert gd.TRUTHFULQA_QA == TQA_QA
    assert t["question"] == TQA_QA + "\n\nQ: Is the earth flat?\nA:"
    assert t["answers"][0] == "Answer number 0." and t["labels"][0] == 1


def test_how_a_file_asks_is_read_from_the_file(tmp_path):
    man = gd.build(tmp_path, docs_of=docs())["benchmarks"]
    assert {b: man[b]["format"] for b in ("mmlu", "arc_challenge", "arc_easy", "truthfulqa")} == {
        "mmlu": "lettered", "arc_challenge": "prompted", "arc_easy": "prompted",
        "truthfulqa": "prompted"}
    for b in ("arc_challenge", "arc_easy", "truthfulqa"):
        assert gb.BENCHMARKS[b]["format"] == "prompted" and gb.BENCHMARKS[b]["earlier"] == gb.BARE
    back = gd.read_mc_binary((tmp_path / "arc-easy-test.bin").read_bytes())
    assert all(t["question"].startswith("Question: ") and t["question"].endswith("\nAnswer:")
               for t in back)
    # the bare question, as the file was before
    bare = gd.mc_binary([{"question": "Which is a mammal?", "answers": ["cat", "fish"],
                          "labels": [1, 0]}])
    assert gb.mc_shape(bare)["format"] == "text"
    lettered = gd.mc_binary([gd.mmlu_task({"question": "q", "choices": ["w", "x", "y", "z"],
                                           "answer": 1, "subject": "s"})])
    assert gb.mc_shape(lettered)["format"] == "lettered"


def _old_run(gid: str, man: dict, form: str | None) -> dict:
    """a run on the ARC file of before 12y: each option after the bare question"""
    ds = {"file": "arc-challenge-test.bin", "sha256": man["arc_challenge"]["sha256"]}
    if form:
        ds["format"] = form
    return {"id": "91", "sid": 91, "model": gid, "status": "done", "line": "", "subset": 0,
            "finished_at": time.time() - 3600, "setup": gb.AS_BUILT,
            "file": {"name": "orig.gguf", "sha256": "f" * 64},
            "benchmarks": {"arc_challenge": {"status": "done", "acc": 0.438, "se": 0.0145,
                                             "n": 1172},
                           "winogrande": {"status": "done", "acc": 0.744, "se": 0.012, "n": 12}},
            "datasets": {"arc_challenge": ds,
                         "winogrande": {"file": "winogrande-validation.csv",
                                        "sha256": man["winogrande"]["sha256"]}}}


def test_arc_measured_on_the_bare_question_goes_to_history(svc):  # noqa: F811
    client, appmod, box = svc
    gid, man = box["gid"], gguf.manifest()
    out = config.RESULTS_ROOT / "gguf_results"
    out.mkdir(parents=True, exist_ok=True)
    for form in ("text", None):                    # as the worker recorded it, and older
        (out / "91.json").write_text(json.dumps(_old_run(gid, man, form)))
        g = results(client, appmod)["gguf"]
        assert set(g["models"][gid]) == {"winogrande"}      # ARC-C is not in its column
        [h] = g["history"][gid]
        assert h["benchmarks"]["arc_challenge"]["current"] is False
        assert h["benchmarks"]["arc_challenge"]["earlier"] == "the bare question, not comparable"
    # measured on the prompted file, it is the column's
    (out / "91.json").write_text(json.dumps(_old_run(gid, man, "prompted")))
    g = results(client, appmod)["gguf"]
    assert g["models"][gid]["arc_challenge"]["v"] == 0.438
    # the columns' note says what is still llama.cpp's own
    assert "with lm_eval's own prompts but no examples" in g["tip"]
    assert "mean log-probability a token" in g["tip"]


def test_the_bare_file_is_refused_until_it_is_built_again(svc):  # noqa: F811
    client, _, box = svc
    mpath = config.RESULTS_ROOT / "gguf_data" / "manifest.json"
    m = json.loads(mpath.read_text())
    m["benchmarks"]["arc_challenge"]["format"] = "text"         # built before 12y
    mpath.write_text(json.dumps(m))
    r = client.post("/api/gguf/runs", json={"model": box["gid"], "benchmarks": ["arc_challenge"],
                                            "by": "masein"})
    assert r.status_code == 422
    assert r.json()["detail"] == gguf.OLD_DATASET.format(
        label="ARC-C", what="each option's text after the bare question")
    # rebuilt, it queues
    gd.build(config.RESULTS_ROOT / "gguf_data", only=["arc_challenge"], docs_of=docs())
    r = client.post("/api/gguf/runs", json={"model": box["gid"], "benchmarks": ["arc_challenge"],
                                            "by": "masein"})
    assert r.status_code == 200, r.text
