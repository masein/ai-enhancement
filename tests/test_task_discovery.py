"""12a.3: every everyday run, #62 to #65, failed before asking a question:

    Including path: /home/masein/benchmarks/everyday/tasks
    Selected Tasks: []
    ValueError: No tasks specified, or no tasks found.

The runner started lm_eval in BENCH_ROOT, and lm_eval 0.4.12 reads a --tasks
value that names a folder in its working directory as a folder of task yaml
files (lm_eval/config/evaluate_config.py, process_tasks). BENCH_ROOT/everyday
is such a folder, with no yaml directly in it. lm_eval now runs in the task's
own output folder, and scripts/check_tasks.py (deploy step 4) asks the
installed harness to find every task the board runs, the way a run asks.

The real lm_eval is not installed where this suite runs, so the check is
tested here against tests/fixtures/fake_lm_eval, which follows 0.4.12's rules
for --tasks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from service import config, runner

FAKE = str(Path(__file__).resolve().parent / "fixtures" / "fake_lm_eval")


@pytest.fixture
def harness():
    """the stand-in lm_eval, in place of any installed one, for this test only"""
    def ours(k):
        return k == "lm_eval" or k.startswith("lm_eval.")
    saved = {k: v for k, v in sys.modules.items() if ours(k)}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, FAKE)
    try:
        import check_tasks
        yield check_tasks
    finally:
        sys.path.remove(FAKE)
        for k in [k for k in sys.modules if ours(k)]:
            del sys.modules[k]
        sys.modules.update(saved)


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """a BENCH_ROOT as the server has it: everyday/tasks/ from earlier runs,
    three built exam topics, the control from the repo"""
    root = tmp_path / "benchmarks"
    (root / "everyday" / "tasks").mkdir(parents=True)
    exam = root / "exam" / "tasks"
    exam.mkdir(parents=True)
    for t in ("exam_economics", "exam_law", "fr_control_mmlu"):
        (exam / f"{t}.yaml").write_text(f"task: {t}\ndataset_path: json\n", encoding="utf-8")
    monkeypatch.setattr(config, "BENCH_ROOT", root)
    monkeypatch.setattr(config, "JUDGED_TASKS_DIR", exam)
    monkeypatch.setattr(config, "EVAL_TASKS_DIR", root / "eval_tasks")
    monkeypatch.setattr(config, "EVERYDAY_TASKS_DIR", root / "everyday" / "tasks")
    monkeypatch.chdir(root)            # where the service itself happens to run
    return root


def test_lm_eval_runs_in_the_tasks_own_output_folder(tmp_path, monkeypatch):
    """not BENCH_ROOT, where a folder can share a task's name"""
    seen = {}

    class Proc:
        def wait(self, timeout=None):
            return 0

    def popen(cmd, **kw):
        seen.update(kw)
        return Proc()
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    out = tmp_path / "results" / "full" / "org__m" / "everyday_0shot"
    out.mkdir(parents=True)
    with open(tmp_path / "log.txt", "a") as lf:
        assert runner._run_task(1, ["lm_eval"], lf, {}, None, cwd=runner.lm_eval_cwd(out)) == 0
    assert seen["cwd"] == out != config.BENCH_ROOT
    # and the command a run builds names the task and the folder it lives in
    cmd = runner.lm_eval_cmd("pretrained=org/m", "everyday", 0, 8, out, chat=True)
    assert cmd[cmd.index("--tasks") + 1] == "everyday"
    assert cmd[cmd.index("--include_path") + 1] == str(config.EVERYDAY_TASKS_DIR)
    assert cmd[cmd.index("--output_path") + 1] == str(out)
    assert "--apply_chat_template" in cmd and "--gen_kwargs" not in cmd
    assert runner.lm_eval_cmd("pretrained=org/m", "everyday", 0, 8, out, chat=True,
                              max_gen_toks=2048)[-2:] == ["--gen_kwargs", "max_gen_toks=2048"]


def test_the_check_finds_every_task_of_every_suite(harness, bench, capsys):
    assert harness.main() == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"lm_eval 0.4.12-fake · BENCH_ROOT {bench}"
    assert out[1:] == ["quick     2 of 2 found", "full      8 of 8 found",
                       "control   1 of 1 found", "judged    3 of 3 found",
                       "everyday  1 of 1 found",
                       "tasks OK: 13 of 13 found by lm_eval 0.4.12-fake"]
    # it wrote nothing outside its temporary folder
    assert sorted(p.name for p in bench.iterdir()) == ["everyday", "exam"]
    assert list((bench / "everyday" / "tasks").iterdir()) == []


def test_the_check_catches_what_failed_62_to_65(harness, bench, capsys, monkeypatch):
    """the runner as it was: lm_eval started in BENCH_ROOT"""
    monkeypatch.setattr(runner, "lm_eval_cwd", lambda task_out: config.BENCH_ROOT)
    assert harness.main() == 1
    out = capsys.readouterr().out
    assert (f"  everyday: lm_eval selected nothing: everyday/ is a folder in {bench}, where "
            f"lm_eval runs, and lm_eval reads a --tasks value that names a folder as a folder "
            f"of task files") in out
    assert "everyday  0 of 1 found" in out and "judged    3 of 3 found" in out
    assert out.rstrip().endswith("tasks FAILED: 12 of 13 found by lm_eval 0.4.12-fake — "
                                 "not found: everyday")


def test_a_task_the_harness_cannot_find_by_name_is_named_with_why(harness, bench, capsys):
    """the runner runs an exam topic by its file's name; a file whose task is
    called something else is not found"""
    (bench / "exam" / "tasks" / "exam_law.yaml").write_text("task: exam_laws\n",
                                                           encoding="utf-8")
    assert harness.main() == 1
    out = capsys.readouterr().out
    assert "  exam_law: ValueError: Tasks not found: exam_law" in out
    assert "judged    2 of 3 found" in out
    assert out.rstrip().endswith("not found: exam_law")


def test_without_lm_eval_it_says_where_to_run_it(monkeypatch, capsys):
    import check_tasks
    monkeypatch.setitem(sys.modules, "lm_eval", None)          # import fails
    assert check_tasks.main() == 2
    assert "sudo docker compose exec -T bench python scripts/check_tasks.py" in \
        capsys.readouterr().out
