"""A DeviceMark run on Hugging Face resumes at the task it was on (#167): the
12z deploy restarted the container while Qwen3.5-4B (thinking on, full) was
on dm_math, and the re-queued run asked dm_ifeval again from its first item.

A task whose saved answers cover every item the built task holds is done, as
a task with lm_eval's results file is in every suite. No model runs: lm_eval
is a stand-in that writes what it writes, and the restart is the service
starting again (db.init(startup=True) re-queues the run that was running)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import devicemark as dm
from service import config, db, runner
from test_12q_devicemark_runs import ME, row_dir, svc  # noqa: F401 — svc is the fixture

QWEN = "Qwen/Qwen3.5-4B"


class Restart(BaseException):
    """the container restarting under the run: nothing after it runs"""


def hf_model(monkeypatch) -> None:
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: {
        "kind": "instruct", "params": 4.0e9, "vocab": 248320, "batch": 8, "need_gb": 10.0,
        "remote_code": False, "has_template": True, "kind_reason": "chat template",
        "archinfo": {"thinking": "switch", "think_end": "</think>"}})
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    monkeypatch.setattr(runner, "gen_backend", lambda: ("hf", "vLLM is not installed in this image"))


def lm_eval(monkeypatch, restart_on: str = "", results: bool = False,
            only: int | None = None) -> list[str]:
    """lm_eval, as the runner calls it: each task's answers, the samples file
    written whole at the end (and, with `results`, its results file first, as
    lm_eval 0.4.12 writes them); `only` answers that many items and stops"""
    asked = []

    def run_task(sid, cmd, lf, env, run_as, cwd, on_poll=None):
        task = cmd[cmd.index("--tasks") + 1]
        asked.append(task)
        if task == restart_on:
            raise Restart
        out = Path(cmd[cmd.index("--output_path") + 1]) / QWEN.replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        stamp = f"2026-10-01T15-{len(list(out.parent.parent.rglob('samples_*'))):02d}-00"
        if results:
            (out / f"results_{stamp}.json").write_text(json.dumps(
                {"results": {task: {"bypass,none": 999}}}))
        keys = sorted(dm.task_keys(config.DM_TASKS_DIR, task))[:only]
        b = dm.BENCH_OF[task]
        ans = {"ifeval": "a note without any commas", "mmlu_pro": "\\boxed{A}",
               "math": "\\boxed{1/2}"}[b]
        # 15.7: a thinking-on reply is saved whole, its thinking with it (hf-whole)
        if cmd[cmd.index("--model") + 1] == "hf-whole":
            ans = "Thinking it over.\n</think>\n\n" + ans
        with open(out / f"samples_{task}_{stamp}.jsonl", "w") as fh:
            for k in keys:
                fh.write(json.dumps({"doc": {"bench": b, "key": k}, "resps": [[ans]]}) + "\n")
        return 0
    monkeypatch.setattr(runner, "_run_task", run_task)
    return asked


@pytest.mark.usefixtures("svc")
@pytest.mark.parametrize("results", [False, True], ids=["samples only", "with results"])
def test_a_restart_during_the_second_task_resumes_at_the_second_task(monkeypatch, results):
    hf_model(monkeypatch)
    sid = db.add(QWEN, "instruct", "devicemark", ME, "", thinking=True, part="full")
    first = lm_eval(monkeypatch, restart_on="dm_mmlu_pro", results=results)
    with pytest.raises(Restart):
        runner.run_submission(db.get(sid))
    assert first == ["dm_ifeval", "dm_mmlu_pro"]
    ifeval = row_dir(QWEN, thinking=True) / "dm_ifeval_0shot"
    assert len(dm.answered_keys(ifeval, "dm_ifeval")) == 300
    # the service starts again: the run that was running is queued again
    db.init(startup=True)
    assert (db.get(sid)["status"], db.get(sid)["progress"]) == ("queued",
                                                                "re-queued after restart")
    second = lm_eval(monkeypatch, results=results)
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "done", row["error"]
    assert second == ["dm_mmlu_pro", "dm_math"]             # dm_ifeval is not asked again
    log = (config.LOGS_DIR / f"service_{sid}_{QWEN.replace('/', '__')}.log").read_text()
    assert "[service] dm_ifeval: answered already (dm_ifeval_0shot); not asked again" in log
    # the row is scored on every item: the first run's dm_ifeval with the rest
    out = json.loads((row_dir(QWEN, thinking=True) / dm.OUT_NAME).read_text())
    assert out["n"] == 596 and out["benches"]["ifeval"]["acc"] == 1.0


@pytest.mark.usefixtures("svc")
def test_a_task_answered_only_in_part_is_asked_again(monkeypatch):
    hf_model(monkeypatch)
    sid = db.add(QWEN, "instruct", "devicemark", ME, "", thinking=True, part="full")
    first = lm_eval(monkeypatch, restart_on="dm_mmlu_pro", only=120)
    with pytest.raises(Restart):
        runner.run_submission(db.get(sid))
    assert first == ["dm_ifeval", "dm_mmlu_pro"]
    db.init(startup=True)
    second = lm_eval(monkeypatch)
    runner.run_submission(db.get(sid))
    assert second == ["dm_ifeval", "dm_mmlu_pro", "dm_math"]   # 120 of 300 is not done
    assert db.get(sid)["status"] == "done"


def test_a_line_cut_short_is_no_answer(tmp_path):
    d = tmp_path / "dm_math_0shot" / "x"
    d.mkdir(parents=True)
    (d / "samples_dm_math_2026-10-01T15-00-00.jsonl").write_text(
        json.dumps({"doc": {"bench": "math", "key": "a"}}) + "\n" + '{"doc": {"bench": "ma')
    assert dm.answered_keys(tmp_path / "dm_math_0shot", "dm_math") == {"a"}
    assert dm.answered_keys(tmp_path / "nothing", "dm_math") == set()
