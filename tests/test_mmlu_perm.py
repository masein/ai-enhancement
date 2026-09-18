"""The permutation control: the rotation, the task files, and the rule that a
control never enters an average."""

from __future__ import annotations

import collections
import importlib.util
import os
import random
import re
from pathlib import Path

import pytest

import report_lm_eval as report
from test_categories import HARNESS_SUBJECTS_0_4_12

REPO = Path(__file__).resolve().parents[1]
PERM = REPO / "eval_tasks" / "mmlu_perm"


def _utils():
    spec = importlib.util.spec_from_file_location("mmlu_perm_utils", PERM / "utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _doc(i: int, answer: int) -> dict:
    return {"question": f"q{i}", "subject": "anatomy",
            "choices": [f"opt{i}-{j}" for j in range(4)], "answer": answer}


def test_rotation_lands_the_answer_on_every_slot_equally():
    u = _utils()
    # the worst key imaginable — every answer is A — still comes out uniform
    slots = collections.Counter(u.rotate(_doc(i, 0), i)["answer"] for i in range(400))
    assert slots == {0: 100, 1: 100, 2: 100, 3: 100}
    # a realistic key: near-uniform with a lean, over a run the size of the subset
    rng = random.Random(7)
    key = [rng.choices(range(4), weights=[22, 24, 26, 28])[0] for _ in range(2300)]
    slots = collections.Counter(u.rotate(_doc(i, a), i)["answer"] for i, a in enumerate(key))
    for s in range(4):
        assert abs(slots[s] / len(key) - 0.25) < 0.03, slots


def test_rotation_keeps_the_correct_text_and_only_moves_it():
    u = _utils()
    for i in range(16):
        d = _doc(i, (i * 5) % 4)
        r = u.rotate(d, i)
        assert r["choices"][r["answer"]] == d["choices"][d["answer"]]
        assert sorted(r["choices"]) == sorted(d["choices"])
        assert r["perm_shift"] == i % 4 and r["orig_answer"] == d["answer"]
        assert r["question"] == d["question"] and r["subject"] == d["subject"]
        assert d["answer"] == (i * 5) % 4                      # the input is not mutated
    assert u.rotate(_doc(0, 1), 8) == u.rotate(_doc(0, 1), 8)  # deterministic
    assert u.rotate(_doc(0, 1), 0)["choices"] == _doc(0, 1)["choices"]
    assert u.rotate(_doc(0, 1), 1)["choices"] == ["opt0-1", "opt0-2", "opt0-3", "opt0-0"]
    assert u.rotate(_doc(0, 1), 1)["answer"] == 0


def test_process_docs_maps_with_indices():
    u = _utils()

    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

        def map(self, fn, with_indices=False):
            assert with_indices, "the shift is by doc_id — the index is not optional"
            return FakeDataset([fn(r, i) for i, r in enumerate(self.rows)])

    out = u.process_docs(FakeDataset([_doc(i, 2) for i in range(8)]))
    assert [r["answer"] for r in out.rows] == [2, 1, 0, 3, 2, 1, 0, 3]


def test_task_files_are_consistent():
    tmpl = (PERM / "_default_template_yaml").read_text()
    assert "process_docs: !function utils.process_docs" in tmpl
    assert "dataset_path: cais/mmlu" in tmpl and "fewshot_split: dev" in tmpl
    assert "doc_to_target: answer" in tmpl
    body = "\n".join(ln for ln in tmpl.splitlines() if not ln.startswith("#"))
    assert "num_fewshot" not in body               # shots come from the run, same as mmlu
    group = (PERM / "_mmlu_perm.yaml").read_text()
    assert re.search(r"^group: mmlu_perm$", group, re.M)
    assert "- mmlu_perm_tasks" in group and "weight_by_size: True" in group
    subjects = []
    for f in sorted(PERM.glob("mmlu_perm_*.yaml")):
        y = f.read_text()
        s = f.stem[len("mmlu_perm_"):]
        subjects.append(s)
        assert f'"dataset_name": "{s}"' in y
        assert f'"task": "mmlu_perm_{s}"' in y
        assert '"tag": "mmlu_perm_tasks"' in y
        assert '"include": "_default_template_yaml"' in y
        assert s.replace("_", " ") in y            # the description names the subject
    assert 8 <= len(subjects) <= 20 and len(set(subjects)) == len(subjects)
    assert set(subjects) <= set(HARNESS_SUBJECTS_0_4_12)
    assert (PERM / "README.md").exists()


def test_control_never_enters_the_official_average(payload):
    assert "mmlu_perm" in payload["accTasks"]                  # shown…
    assert "mmlu_perm" not in payload["required"]              # …never averaged
    assert payload["tasks"]["mmlu_perm"]["control"] is True
    assert payload["tasks"]["mmlu"]["control"] is False
    assert payload["tasks"]["mmlu_perm"]["chance"] == 0.25
    good = next(m for m in payload["models"] if m["id"] == "fx/good-750m")
    assert "fx/good-750m" in payload["cells"]["mmlu_perm"]
    cells = payload["cells"]
    expect = sum(report.above_chance(t, cells[t]["fx/good-750m"]["v"])
                 for t in payload["required"]) / len(payload["required"])
    assert good["avg"] == pytest.approx(expect)
    assert good["navg"] == 7                                    # 8 acc tasks ran; 7 count
    # and a sibling without the control has the same required list and rank basis
    below = next(m for m in payload["models"] if m["id"] == "fx/below-135m-it")
    assert below["nreq"] == good["nreq"] == 7


def test_required_tasks_drops_the_control_even_when_asked_for_it(monkeypatch):
    monkeypatch.setenv("REQUIRED_TASKS", "mmlu,mmlu_perm,hellaswag")
    req, absent = report.required_tasks(["mmlu", "mmlu_perm", "hellaswag", "piqa"])
    assert req == ["mmlu", "hellaswag"] and absent == []
    monkeypatch.delenv("REQUIRED_TASKS")
    req, absent = report.required_tasks(["mmlu_perm"])
    assert req == [] and "mmlu_perm" not in absent


def test_control_suite_goes_through_the_service_queue(tmp_path, monkeypatch):
    from service import config, runner
    assert config.tasks_for_suite("control") == ["mmlu_perm"]
    assert "mmlu_perm" not in config.tasks_for_suite("full")
    assert "mmlu_perm" not in config.tasks_for_suite("quick")
    assert config.NFEWSHOT["mmlu_perm"] == config.NFEWSHOT["mmlu"] == 5
    assert config.CONTROL_TASKS_DIR == PERM and (PERM / "_mmlu_perm.yaml").exists()
    monkeypatch.setattr(config, "EVAL_TASKS_DIR", tmp_path / "none")
    assert runner.include_args_for("mmlu_perm") == ["--include_path", str(PERM)]
    assert runner.include_args_for("mmlu") == []
    (tmp_path / "ppl").mkdir()
    (tmp_path / "ppl" / "ppl_code.yaml").write_text("task: ppl_code\n")
    monkeypatch.setattr(config, "EVAL_TASKS_DIR", tmp_path / "ppl")
    assert runner.include_args_for("ppl_code") == ["--include_path", str(tmp_path / "ppl")]
    assert runner.include_args_for("mmlu_perm") == ["--include_path", str(PERM)]


def test_docker_context_carries_the_control():
    assert "COPY eval_tasks/mmlu_perm/ eval_tasks/mmlu_perm/" in (REPO / "Dockerfile").read_text()
    di = (REPO / ".dockerignore").read_text().splitlines()
    assert "!eval_tasks/mmlu_perm" in di and "eval_tasks/" not in di


@pytest.mark.network
def test_real_mmlu_subset_is_uniform_after_rotation():
    """On the server: the real subset, the real transform, every slot 25% ± 3."""
    datasets = pytest.importorskip("datasets")
    u = _utils()
    subjects = [f.stem[len("mmlu_perm_"):] for f in PERM.glob("mmlu_perm_*.yaml")]
    slots = collections.Counter()
    for s in subjects:
        ds = datasets.load_dataset("cais/mmlu", s, split="test")
        slots.update(u.process_docs(ds)["answer"])
    n = sum(slots.values())
    assert n > 1500
    for k in range(4):
        assert abs(slots[k] / n - 0.25) < 0.03, dict(slots)


def test_env_example_documents_the_knob():
    assert "CONTROL_TASKS_DIR" in (REPO / ".env.example").read_text()
    assert os.environ.get("CONTROL_TASKS_DIR") is None or True
