"""12f.4: the image's Python has what its runs import. A served model sits the
generative suite through lm_eval's local-chat-completions, which needs lm_eval's
`api` extra; the image lacked it, and the first two served runs failed at once
("missing package"). The fake-server tests never import lm_eval, so they missed
it. These do — in CI's image-deps job (the image's requirements over a CPU
torch, EVALBOARD_IMAGE_DEPS=1, where a missing lm_eval fails) and in deploy
step 3, inside the image. Elsewhere, with no lm_eval at all, they skip."""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^lm_eval\[([^\]]+)\]==([\d.]+)", re.M)
# a distribution whose module is named otherwise
MODULE = {"beautifulsoup4": "bs4", "pyyaml": "yaml", "scikit-learn": "sklearn"}


def lm_eval():
    try:
        return importlib.import_module("lm_eval")
    except ImportError:
        if os.environ.get("EVALBOARD_IMAGE_DEPS") == "1":
            raise
        pytest.skip("lm_eval isn't installed here: CI's image-deps job and deploy step 3 run this")


def test_the_pin_carries_the_extras_the_runs_use():
    m = PIN.search((ROOT / "requirements.txt").read_text(encoding="utf-8"))
    assert m, "requirements.txt pins lm_eval with its extras"
    assert {"ifeval", "math", "api"} <= {x.strip() for x in m.group(1).split(",")}


def test_what_is_installed_is_the_pin():
    lm_eval()
    assert md.version("lm_eval") == PIN.search(
        (ROOT / "requirements.txt").read_text(encoding="utf-8")).group(2)


def test_local_chat_completions_and_every_package_of_the_api_extra_import():
    lm_eval()
    from lm_eval.api.registry import get_model
    cls = get_model("local-chat-completions")
    assert cls.__module__ == "lm_eval.models.openai_completions"
    importlib.import_module("lm_eval.models.openai_completions")
    # what lm_eval itself says the api extra needs
    extra = [re.split(r"[\s;<>=!~\[]", r, maxsplit=1)[0] for r in md.requires("lm_eval") or []
             if re.search(r"extra\s*==\s*['\"]api['\"]", r)]
    assert extra, "lm_eval declares an api extra"
    for dist in extra:
        name = MODULE.get(dist.lower(), dist.replace("-", "_").lower())
        importlib.import_module(name)


def test_trust_and_safety_tasks_load_in_the_installed_harness(tmp_path):
    """12k.2: BBQ, Do-Not-Answer and XSTest as a run builds them, found by the
    installed lm_eval under our own names (it ships a `bbq` and a `bbq_ambig`
    of its own, on an unpinned copy), and posing what we mean them to — no
    model is loaded"""
    lm_eval()
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import trust_safety as ts
    from lm_eval.tasks import TaskManager, get_task_dict
    d = ts.build_tasks(tmp_path / "tasks")
    tm = TaskManager(include_path=str(d))
    assert {"bbq", "bbq_ambig"} <= set(tm.all_tasks)          # lm_eval's own, left alone
    tasks = get_task_dict(["bbq_3000", "do_not_answer", "xstest"], task_manager=tm)
    bbq = tasks["bbq_3000"]
    docs = list(bbq.test_docs())
    assert len(docs) == 3000 and bbq.OUTPUT_TYPE == "multiple_choice"
    q = docs[0]
    assert bbq.doc_to_text(q) == f"{q['context']}\n\nQ: {q['question']}\nA:"
    assert bbq.doc_to_choice(q) == q["choices"] and bbq.doc_to_target(q) == q["label"]
    # its scoring: the option with the highest log-likelihood, and the bias score beside acc
    got = bbq.process_results(q, [(-1.0 if i == q["label"] else -9.0, False) for i in range(3)])
    assert got["acc"] == 1 and tuple(got["bias_score"]) == (1, 0, 0)
    assert bbq.aggregation()["bias_score"]([(0, 1, 1), (0, 0, 1)]) == 0.0
    for name, n in (("do_not_answer", 939), ("xstest", 450)):
        t = tasks[name]
        docs = list(t.test_docs())
        assert len(docs) == n and t.OUTPUT_TYPE == "generate_until"
        assert t.doc_to_text(docs[0]) == docs[0]["prompt"]
        assert t.config.generation_kwargs["max_gen_toks"] == 512
    assert len(list(get_task_dict(["bbq_all"], task_manager=tm)["bbq_all"].test_docs())) == 29_246


def test_the_encrypted_export_imports():
    """12p.1b: `hidden_store backup --export` encrypts in-process with pyrage"""
    try:
        import pyrage
    except ImportError:
        if os.environ.get("EVALBOARD_IMAGE_DEPS") == "1":
            raise
        pytest.skip("pyrage isn't installed here: CI's image-deps job and deploy step 3 run this")
    assert re.search(r"^pyrage==1\.4\.0", (ROOT / "requirements.txt").read_text(encoding="utf-8"),
                     re.M)
    assert md.version("pyrage") == "1.4.0" and pyrage.x25519.Identity.generate()
