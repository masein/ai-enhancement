"""15.7a: a thinking-on DeviceMark row on hf whose replies were saved before
15.7 — cut by lm_eval at the end of the thinking, or with their markers dropped
(Gemma 4, marked "needs a new run") — is asked again when it is resubmitted:
its answers are set aside under results/earlier/, and every task is asked with
each reply whole. They were found complete and skipped ("answered already; not
asked again"), so Gemma 4 thinking on had to be moved aside by hand. A row
saved whole, or with the thinking off, is skipped as before. lm_eval is a
stand-in; nothing runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import devicemark as dm
from service import config, db, runner
from test_12q_devicemark_runs import ME, row_dir, svc  # noqa: F401 — svc is the fixture
from test_devicemark_resume import hf_model

QWEN, GEMMA = "Qwen/Qwen3.5-4B", "google/gemma-4-E2B-it"
ANSWER = {"ifeval": "a note without any commas", "mmlu_pro": "\\boxed{A}",
          "math": "\\boxed{\\frac{1}{2}}"}


def lm_eval(monkeypatch, model: str) -> list[str]:
    """lm_eval as the runner calls it: each task's replies, whole when the
    runner asks for them whole — the thinking in the family's own markers"""
    asked = []

    def run_task(sid, cmd, lf, env, run_as, cwd, on_poll=None):
        task = cmd[cmd.index("--tasks") + 1]
        asked.append(task)
        whole = cmd[cmd.index("--model") + 1] == "hf-whole"
        out = Path(cmd[cmd.index("--output_path") + 1]) / model.replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        b = dm.BENCH_OF[task]
        think = ("<|channel>thought\nThinking it over.<channel|>" if model == GEMMA
                 else "Thinking it over.\n</think>\n\n")
        with open(out / f"samples_{task}_2026-10-03T10-00-00.jsonl", "w") as fh:
            for k in sorted(dm.task_keys(config.DM_TASKS_DIR, task)):
                text = (think if whole else "") + ANSWER[b]
                fh.write(json.dumps({"doc": {"bench": b, "key": k}, "resps": [[text]]}) + "\n")
        return 0
    monkeypatch.setattr(runner, "_run_task", run_task)
    return asked


def saved_cut(model: str, thinking: bool = True, unreadable: bool = False) -> None:
    """a row as a run before 15.7 left it: every task answered, in lm_eval's
    cut form (no reply_form.json) — for Gemma 4, "thought\\n…" and the answer
    run together — and, for an unreadable one, marked as rescore marks it"""
    d = row_dir(model, thinking=thinking)
    for b, task in dm.TASK.items():
        out = d / f"{task}_0shot" / model.replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        text = ("thought\nLet me see." if unreadable else "") + ANSWER[b]
        (out / f"samples_{task}_2026-10-01T10-00-00.jsonl").write_text("".join(
            json.dumps({"doc": {"bench": b, "key": k}, "resps": [[text]]}) + "\n"
            for bb, k in dm.keys_for("full") if bb == b))
    setup = {"model": model, "runtime": "hf transformers (lm_eval)", "thinking": thinking,
             "scoring": "v1"}
    if unreadable:
        setup["provisional"] = "needs a new run: 596 of its answers can't be told from its thinking"
    (d / dm.OUT_NAME).write_text(json.dumps({"composite": {"value": 0.57}, "setup": setup}))


@pytest.mark.usefixtures("svc")
@pytest.mark.parametrize("model,unreadable", [(QWEN, False), (GEMMA, True)],
                         ids=["qwen, saved cut", "gemma, marked needs a new run"])
def test_a_row_saved_cut_is_asked_again_whole_when_resubmitted(monkeypatch, model, unreadable):
    hf_model(monkeypatch)
    saved_cut(model, unreadable=unreadable)
    asked = lm_eval(monkeypatch, model)
    sid = db.add(model, "instruct", "devicemark", ME, "", thinking=True, part="full")
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "done", row["error"]
    assert asked == ["dm_ifeval", "dm_mmlu_pro", "dm_math"]          # every task, again
    d = row_dir(model, thinking=True)
    log = (config.LOGS_DIR / f"service_{sid}_{model.replace('/', '__')}.log").read_text()
    for task in dm.TASK.values():
        assert dm.reply_form(d / f"{task}_0shot") == "whole"
        kept = list((config.OUT_DIR.with_name("earlier") / d.name).glob(f"{task}_0shot-cut-*"))
        assert len(kept) == 1 and list(kept[0].rglob(f"samples_{task}_*.jsonl"))
        assert (f"[service] {task}: its replies were saved before 15.7, cut at the end of the "
                f"thinking — kept at {kept[0]}, asking again with each reply whole") in log
    out = json.loads((d / dm.OUT_NAME).read_text())
    assert out["setup"]["scoring"] == dm.SCORING and dm.provisional_why(out["setup"]) is None
    assert out["n"] == 596 and out["unreadable"] == 0 and out["benches"]["ifeval"]["acc"] == 1.0
    # saved whole now: a resubmission asks nothing again
    again = lm_eval(monkeypatch, model)
    sid2 = db.add(model, "instruct", "devicemark", ME, "", thinking=True, part="full")
    runner.run_submission(db.get(sid2))
    assert db.get(sid2)["status"] == "done" and again == []


@pytest.mark.usefixtures("svc")
def test_a_thinking_off_row_is_skipped_as_before(monkeypatch):
    # thinking off, nothing was cut: its answers are whole as they are
    hf_model(monkeypatch)
    saved_cut(QWEN, thinking=False)
    asked = lm_eval(monkeypatch, QWEN)
    sid = db.add(QWEN, "instruct", "devicemark", ME, "", thinking=False, part="full")
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done" and asked == []
    assert not (config.OUT_DIR.with_name("earlier") / row_dir(QWEN).name).exists()
