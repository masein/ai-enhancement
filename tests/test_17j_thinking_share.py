"""17j, part 3, point 38: a thinking-off run is scored when a few of its
answers think anyway — on the hardest questions the model opens a thinking
block of its own, the server doing as it was told — and says the share ("76
of 2,158 thought anyway, 3.5%"); refused only when more than a quarter say
the switch was ignored; the same for a shard; and a box whose bundle was
refused for this reads safe, said. Answers invented; no model runs."""

from __future__ import annotations

import json
from pathlib import Path

import report_lm_eval as report
from service import frontier as sf
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import N, RUNS, TASK, box, bundle_of, register, run_box  # noqa: F401
from test_17b_review import imported, rewrite
from test_17f_review import listing_of
from test_17g_review import progress_of, with_steps
from test_17j_boxes import main_of, world

THOUGHT = "<think>\nweighing the options\n</think>\n\n"


def test_38_a_few_that_thought_are_scored_and_the_share_said():
    thinks, plain = [THOUGHT + "ANSWER: A"] * 76, ["ANSWER: A"] * (2158 - 76)
    assert sf.thinking_refused(TASK, "off", thinks + plain) == ""     # d4abe7d: refused (1%)
    assert sf.thinking_kept("off", thinks + plain) == 76
    sc = {"score": 0.5, "se": 0.01, "epochs": 1, "questions": 2158, "answers": 2158,
          "thinking_held": 76}
    assert "76 of 2,158 thought anyway, 3.5%" in sf.words(TASK, sc)
    assert "76 of 2,158 thought anyway, 3.5%" in report.frontier_how(sc)   # the score's cell
    # more than a quarter: the switch was ignored
    why = sf.thinking_refused(TASK, "off", [THOUGHT + "ANSWER: A"] * 600 + plain[:1400])
    assert "more than a quarter" in why and "ignored the thinking switch" in why


def answers_with(path: Path, k: int):
    """the bundle with its first `k` answers opened by a thinking block"""
    row = path.name.split("-thinking")[0].replace("frontier-", "")
    name = f"results/{row}/{TASK}_0shot/frontier/answers.jsonl"

    def fix(files):
        lines = files[name].decode().splitlines()
        out = []
        for i, x in enumerate(lines):
            r = json.loads(x)
            if i < k:
                r["answer"] = THOUGHT + r["answer"]
            out.append(json.dumps(r))
        files[name] = ("\n".join(out) + "\n").encode()
    return rewrite(path, fix)


def test_38_the_bundle_home_imports_as_it_is_and_its_run_says_the_share(box):  # noqa: F811
    from service import config, db
    assert run_box(box, "run", thinking="off") == 0
    register(box["sha"])
    k = 3                                                    # 3 of 48: 6%, above 1%
    code, said = imported(answers_with(bundle_of(box, "run", thinking=False), k))
    assert code == 0, said                                   # d4abe7d: refused
    run = db.recent(1)[0]
    words = f"{k} of {N * RUNS} thought anyway, {k / (N * RUNS):.1%}"
    assert words in (run.get("progress") or ""), run      # on the run
    row = config.OUT_DIR / ("served__lda-box")
    blob = json.loads(sorted(sf.task_dir(row, TASK).glob("results_*.json"))[-1].read_text())
    assert blob["frontier"]["thinking_held"] == k                     # on the score


def test_38_a_shard_more_than_a_quarter_of_which_thought_is_refused(box):  # noqa: F811
    assert run_box(box, "s1", "--shard", "1/2", thinking="off") == 0
    register(box["sha"])
    half = N * RUNS // 2
    code, said = imported(answers_with(bundle_of(box, "s1", (1, 2), thinking=False), half))
    assert code != 0 and any("more than a quarter" in x for x in said), said   # d4abe7d: in
    code, said = imported(answers_with(bundle_of(box, "s1", (1, 2), thinking=False), 1))
    assert code == 0, said
    assert any(f"this shard's 1 of {half} thought anyway, {1 / half:.1%}" in x for x in said), \
        said


def test_38_a_box_whose_bundle_was_refused_for_this_reads_safe(tmp_path, monkeypatch, capsys):
    from test_17e_review import a_bundle
    there = tmp_path / "there"
    there.mkdir()
    b = a_bundle(there / "frontier-served__phone-thinking-off-hle.tar.gz", "served/phone")
    whole = progress_of("phone", "A5-1", "whole", tasks=["hle_text_cais"],
                        bundle={"name": b.name})

    def listing(host, rnd):
        return with_steps(listing_of({f"/workspace/phone/A5-1/{b.name}": b}, [whole]),
                          [("phone", "A5-1")])
    ff, calls, key, state = world(tmp_path, monkeypatch, listing)
    plain = ff.run

    def run(cmd, cwd=None, timeout=None, stdin=None):
        if cmd[:8] == ff.IMPORT and "--served" not in cmd:
            # the board before this fix: refused at 1%
            return 1, ("refused — Humanity's Last Exam: thinking was off, and 76 of its 2,158 "
                       "answers hold thinking — more than 1%: the server thought anyway: not "
                       "scored\nnothing was imported")
        return plain(cmd, cwd, timeout, stdin)
    monkeypatch.setattr(ff, "run", run)

    def copy(_box, _key, remote, want, dest, name=""):
        (dest / (name or Path(remote).name)).write_bytes(b.read_bytes())
        return True, "copied"
    monkeypatch.setattr(ff, "copy", copy)
    main_of(ff, key, tmp_path / "b" / "bundles", "1.1.1.1:41")
    out = capsys.readouterr().out
    first = next(x for x in out.splitlines() if "safe to destroy" in x)
    assert "done, safe to destroy" in first, out                  # d4abe7d: NOT safe
    assert "1 refused for its thinking share and kept here" in first, first
    assert "kept here, refused for its thinking share" in out
    assert "scripts/import_remote.py" in out and "--file-sha256" in out   # the line to type
