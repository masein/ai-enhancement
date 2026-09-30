"""12q.E (the brief's): the hf DeviceMark runs of DeviceMark's own models, as
the server ran them.

- A row is its newest scored run: a resubmit's answers replace the earlier
  run's, and a later run that fails before scoring leaves the row as it was.
- A pair whose median answers differ by more than 2× ran in different modes
  (Youtu-LLM-2B: ours answered 97% at a median of 370 tokens, theirs 65% at
  2,893 — theirs reasoned): said on its row, and not a calibration point. Any
  other pair says whether the two intervals overlap.
- A generative task that runs out of GPU memory on hf is tried again at half
  the batch, down to 1 (Granite-4.0-H-1B on MMLU-Pro, #132 and #144), and the
  run's later tasks start at the batch that worked.

No model runs: lm_eval is a stand-in, as in test_12q_devicemark_runs.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import devicemark as dm
from service import config, db, runner
from test_12q_devicemark_board import HF, _row
from test_12q_devicemark_runs import ME, row_dir, svc  # noqa: F401 — svc is the fixture

YOUTU, THEIR_YOUTU = "tencent/Youtu-LLM-2B", "youtu-2b__int8__aimodel"
QWEN, THEIR_QWEN = "Qwen/Qwen3.5-4B", "qwen3.5-4b__int8hu__aimodel"
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12qe"


# ---------------------------------------------------------------------------
# a row is its newest scored run
# ---------------------------------------------------------------------------

def _samples(d: Path, stamp: str, answers: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"samples_dm_math_{stamp}.jsonl").write_text("".join(json.dumps(
        {"doc": {"bench": "math", "key": k}, "resps": [[t]]}) + "\n" for k, t in answers.items()))


def test_a_resubmits_answers_replace_the_earlier_runs(tmp_path):
    d = tmp_path / "dm_math_0shot" / "org__m"
    _samples(d, "2026-09-29T22-10-00.000000", {"a": "\\boxed{1}", "b": "\\boxed{2}"})
    _samples(d, "2026-09-30T08-45-00.000000", {"a": "\\boxed{7}", "b": "\\boxed{8}"})
    recs = dm.records_from_samples(tmp_path)
    assert [(r["key"], r["text"]) for r in recs] == [("a", "\\boxed{7}"), ("b", "\\boxed{8}")]


def test_a_later_run_that_fails_before_scoring_leaves_the_row(tmp_path):
    out = tmp_path / "full"
    _row(out, YOUTU, {"ifeval": 0.6, "mmlu_pro": 0.6, "math": 0.8}, HF)
    was = json.loads((out / YOUTU.replace("/", "__") / dm.OUT_NAME).read_text())
    # a newer run's answers to one task, and no score: the run stopped there
    _samples(out / YOUTU.replace("/", "__") / "dm_math_0shot" / "x", "2026-09-30T09-00-00.000000",
             {"a": "no"})
    [row] = [r for r in dm.rows(out) if r["model"] == YOUTU]
    assert row["row"]["composite"] == was["composite"] and row["row"]["at"] == was["at"]


# ---------------------------------------------------------------------------
# a pair in different modes; a pair that agrees
# ---------------------------------------------------------------------------

def _medians(out: Path, model: str, tokens: int) -> None:
    """a row of ours whose answers have this median length"""
    f = out / model.replace("/", "__") / dm.OUT_NAME
    row = json.loads(f.read_text())
    row["median_tokens"] = tokens
    f.write_text(json.dumps(row))


def test_a_pair_in_different_modes_says_so_and_isnt_calibration(tmp_path):
    out = tmp_path / "full"
    _row(out, YOUTU, {"ifeval": 0.7, "mmlu_pro": 0.65, "math": 0.6}, HF)
    _medians(out, YOUTU, 370)
    _row(out, QWEN, {"ifeval": 0.3, "mmlu_pro": 0.2, "math": 0.3}, HF)
    _medians(out, QWEN, 3000)
    theirs = {r["id"]: r for r in dm.board(out)["external"]["rows"]}
    assert theirs[THEIR_YOUTU]["median_tokens"] == 2893                 # theirs reasoned
    [o] = theirs[THEIR_YOUTU]["ours"]
    ours_pct = round(100 * o["answered_pct"])
    assert o["mode_differs"] == (
        f"the modes differ: ours answered {ours_pct}% at a median of 370 tokens, theirs 65% at "
        "2,893 — not a calibration point")
    assert o["calibration"] is False and o["within"] is None            # though it is on the list
    assert YOUTU in dm.CALIBRATION
    # within 2×: the same mode, a calibration point, and whether the intervals overlap
    [q] = theirs[THEIR_QWEN]["ours"]
    assert theirs[THEIR_QWEN]["median_tokens"] == 3917
    assert (q["mode_differs"], q["calibration"]) == ("", True)
    assert q["within"] is dm.intervals_overlap(q["composite"], theirs[THEIR_QWEN]["composite"])


@pytest.mark.parametrize("ours,theirs,differ", [
    (370, 2893, True), (2893, 370, True), (400, 800, False), (801, 400, True),
    (None, 800, False), (370, None, False)])
def test_more_than_twice_apart_is_another_mode(ours, theirs, differ):
    got = dm.modes_differ({"median_tokens": ours, "answered_pct": 0.97},
                          {"median_tokens": theirs, "answered_pct": 0.645})
    assert bool(got) is differ


def test_intervals_overlap():
    a = {"value": 0.43, "ci": [0.39, 0.47]}
    assert dm.intervals_overlap(a, {"value": 0.419, "ci": [0.378, 0.462]}) is True
    assert dm.intervals_overlap(a, {"value": 0.6, "ci": [0.55, 0.65]}) is False
    assert dm.intervals_overlap(a, {"value": 0.6}) is None


@pytest.mark.dashboard
def test_on_the_chart_the_pairs_row_and_hover_say_the_modes_differ(live, page):
    out = Path(live["root"]) / "results" / "full"
    _row(out, YOUTU, {"ifeval": 0.7, "mmlu_pro": 0.65, "math": 0.6}, HF)
    _medians(out, YOUTU, 370)
    _row(out, QWEN, {"ifeval": 0.3, "mmlu_pro": 0.2, "math": 0.3}, HF)
    _medians(out, QWEN, 3000)
    try:
        page.set_viewport_size({"width": 1400, "height": 1000})
        page.goto("about:blank")
        page.goto(live["base"] + "/#tab=models&chip=ondevice")
        page.wait_for_selector("[data-dm-chart]")
        cell = page.locator(f"[data-dm-composite='{THEIR_YOUTU}']")
        flag = cell.locator(f"[data-dm-mode-differs='{YOUTU}']")
        assert flag.inner_text() == "modes differ · not a calibration point"
        [why] = json.loads(flag.get_attribute("data-tip"))
        assert why.startswith("the modes differ: ours answered ") and "theirs 65% at 2,893" in why
        tip = json.loads(page.locator(f"[data-dm-point='{THEIR_YOUTU}']").get_attribute(
            "data-tip"))
        assert why in tip and not any("· calibration" in x for x in tip)
        # a pair in the same mode: calibration, and whether the intervals overlap
        assert page.locator(f"[data-dm-composite='{THEIR_QWEN}'] [data-dm-mode-differs]").count() \
            == 0
        qtip = json.loads(page.locator(f"[data-dm-point='{THEIR_QWEN}']").get_attribute(
            "data-tip"))
        assert any("· calibration · the intervals" in x for x in qtip)
        SCREENS.mkdir(parents=True, exist_ok=True)
        page.locator(f"[data-dm-row='{THEIR_YOUTU}']").screenshot(
            path=SCREENS / "youtu-modes-differ.png")
        assert page.errors == []
    finally:
        import shutil
        for m in (YOUTU, QWEN):
            shutil.rmtree(out / m.replace("/", "__"), ignore_errors=True)


# ---------------------------------------------------------------------------
# out of GPU memory: the task again at half the batch, down to 1
# ---------------------------------------------------------------------------

GRANITE = "ibm-granite/granite-4.0-h-1b"
OOM = ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.40 GiB. GPU 0 has a "
       "total capacity of 31.36 GiB\n")


def _hf_run(monkeypatch, fits: dict):
    """a run of the battery on hf, lm_eval a stand-in: a task runs out of
    memory above the batch `fits` gives it. Returns the (task, batch) asked"""
    seen = []
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: {
        "kind": "instruct", "params": 1.5e9, "vocab": 100352, "batch": 16, "need_gb": 6.0,
        "remote_code": False, "has_template": True, "kind_reason": "chat template",
        "archinfo": {"thinking": "never", "think_end": "</think>"}})
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    monkeypatch.setattr(runner, "gen_backend", lambda: ("hf", "vLLM is not installed here"))
    items = dm.load_items(config.DM_ITEMS)

    def run_task(sid, cmd, lf, env, run_as, cwd, on_poll=None):
        task = cmd[cmd.index("--tasks") + 1]
        batch = int(cmd[cmd.index("--batch_size") + 1])
        seen.append((task, batch))
        if batch > fits.get(task, 10 ** 6):
            lf.write(OOM)
            lf.flush()
            return 1
        out = Path(cmd[cmd.index("--output_path") + 1]) / GRANITE.replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        b = dm.BENCH_OF[task]
        with open(out / f"samples_{task}_2026-09-30T10-00-00.jsonl", "w") as fh:
            for bb, k in dm.keys_for("full"):
                if bb == b:
                    fh.write(json.dumps({"doc": {"bench": b, "key": k,
                                                 "text": items[(b, k)]["text"]},
                                         "resps": [["\\boxed{A}"]]}) + "\n")
        return 0
    monkeypatch.setattr(runner, "_run_task", run_task)
    sid = db.add(GRANITE, "instruct", "devicemark", ME, "", part="full")
    runner.run_submission(db.get(sid))
    return sid, seen


def test_a_task_out_of_memory_runs_again_at_half_the_batch(svc, monkeypatch):  # noqa: F811
    sid, seen = _hf_run(monkeypatch, {"dm_mmlu_pro": 1})
    row = db.get(sid)
    assert row["status"] == "done", row["error"]
    # 12q.G: it starts at 4, the most a DeviceMark run writes at a time, whatever
    # preflight sized for scoring (16). MMLU-Pro at 4, 2, then 1 — and MATH
    # starts at the batch that worked
    assert seen == [("dm_ifeval", 4), ("dm_mmlu_pro", 4), ("dm_mmlu_pro", 2),
                    ("dm_mmlu_pro", 1), ("dm_math", 1)]
    [log_file] = config.LOGS_DIR.glob(f"service_{sid}_*.log")
    log = log_file.read_text()
    assert "[service] dm_mmlu_pro ran out of GPU memory: again at batch 2" in log
    assert "[service] dm_mmlu_pro ran out of GPU memory: again at batch 1" in log
    assert json.loads((row_dir(GRANITE) / dm.OUT_NAME).read_text())["n"] == 596


def test_out_of_memory_at_a_batch_of_one_fails_in_plain_words(svc, monkeypatch):  # noqa: F811
    sid, seen = _hf_run(monkeypatch, {"dm_mmlu_pro": 0})
    row = db.get(sid)
    assert [b for t, b in seen if t == "dm_mmlu_pro"] == [4, 2, 1]
    assert ("dm_math", 1) not in seen                       # it would run out again: stopped
    assert row["status"] == "failed" and "ran out of GPU memory" in row["error"]
    assert not (row_dir(GRANITE) / dm.OUT_NAME).exists()
