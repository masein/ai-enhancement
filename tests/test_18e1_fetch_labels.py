"""18e, point 1 (docs/prompts/phase-18e-twelfth-review-fetch-scores-agent-
page.md): a box that runs two labels, one after the other, never reads "safe
to destroy" between them. The fetch takes each host's labels (host:port=L1,L2)
— working with the boxes running 9310719 — and a line file not "ended" is
never safe; the next image's box line takes several labels and writes each
later one "queued" at its start. Invented boxes and a stand-in ssh: nothing is
fetched, no model runs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import frontier_box as fx
from test_17e_review import a_bundle
from test_17f_review import listing_of
from test_17g_review import progress_of, with_steps
from test_17j_boxes import main_of, world

HLE = "hle_text_cais"


def box_listing(there: Path, steps: dict[str, str], lines: dict[str, str]) -> str:
    """a box of the phone build: each step's state ({"L1-1": "whole"}), its
    bundle when whole, and each label's line file"""
    files, progress = {}, []
    for step, st in steps.items():
        b = a_bundle(there / f"frontier-served__phone-thinking-on-hle-{step}.tar.gz",
                     "served/phone")
        if st == "whole":
            files[f"/workspace/phone/{step}/{b.name}"] = b
        progress.append(progress_of("phone", step, st, tasks=[HLE],
                                    **({"bundle": {"name": b.name}} if st == "whole" else {})))
    got = json.loads(with_steps(listing_of(files, progress), [("phone", s) for s in steps]))
    got["lines"] = [{"box": lab, "build": "phone", "state": st} for lab, st in lines.items()]
    return json.dumps(got)


@pytest.fixture
def box(tmp_path, monkeypatch):
    there = tmp_path / "there"
    there.mkdir()
    now: dict = {"steps": {"L1-1": "whole"}, "lines": {"L1": "ended"}}
    ff, calls, key, state = world(tmp_path, monkeypatch,
                                  lambda host, rnd: box_listing(there, now["steps"], now["lines"]))
    monkeypatch.setattr(ff, "copy", lambda *a, **k: (True, "here already"))
    return ff, key, tmp_path / "b" / "bundles", now


def test_1_between_two_labels_the_box_is_not_safe_with_its_labels_given(box, capsys):
    ff, key, dest, now = box
    # L1 whole and imported, L2 not started yet: the few seconds between them
    main_of(ff, key, dest, "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "1.1.1.1:41 (L1): done, safe to destroy" in out      # the reviewers' reproduction
    main_of(ff, key, dest, "1.1.1.1:41=L1,L2")
    out = capsys.readouterr().out
    assert "1.1.1.1:41 (L1; L2 not started): NOT safe to destroy — phone L2-1 hasn't " \
           "started (its label given for this box)" in out, out  # 518e4a1: no labels taken
    # L2 running, then whole and imported, its line ended: safe
    now["steps"]["L2-1"], now["lines"]["L2"] = "asking", "running"
    main_of(ff, key, dest, "1.1.1.1:41=L1,L2")
    out = capsys.readouterr().out
    assert "NOT safe to destroy" in out and "phone L2's line is running" in out
    now["steps"]["L2-1"], now["lines"]["L2"] = "whole", "ended"
    main_of(ff, key, dest, "1.1.1.1:41=L1,L2")
    assert "1.1.1.1:41 (L1, L2): done, safe to destroy" in capsys.readouterr().out


def test_1_a_queued_line_is_never_safe(box, capsys):
    ff, key, dest, now = box
    now["lines"]["L2"] = "queued"                               # the next image's, at its start
    main_of(ff, key, dest, "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "NOT safe to destroy — phone L2's line is queued" in out, out


def test_1_a_label_of_no_plan_is_refused_before_anything(box, capsys):
    ff, key, dest, _ = box
    with pytest.raises(SystemExit):
        main_of(ff, key, dest, "1.1.1.1:41=L1,L99")
    assert "L99 isn't a box of any plan" in capsys.readouterr().err   # 518e4a1: no = at all


def test_1_the_box_line_runs_several_labels_the_later_ones_queued_first(tmp_path, monkeypatch):
    import remote_bundle as rb
    monkeypatch.setattr(rb, "gpu_info", lambda: {"name": "NVIDIA GeForce RTX 5090",
                                                 "names": ["NVIDIA GeForce RTX 5090"]})
    seen = []

    def run(cmd, *a, **k):
        out = Path(cmd[cmd.index("--out") + 1])
        line = {lab: json.loads((out.parent / f"{lab}.line.json").read_text())["state"]
                for lab in ("L1", "L2") if (out.parent / f"{lab}.line.json").exists()}
        seen.append((out.name, line))
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(fx.subprocess, "run", run)
    assert fx.main(["L1", "L2", "--as", "served/step4000-lda-2610", "--gguf", "g.gguf",
                    "--server", "s.tar.gz", "--root", str(tmp_path)]) == 0
    assert seen == [("L1-1", {"L1": "running", "L2": "queued"}),   # 518e4a1: L2 unknown
                    ("L2-1", {"L1": "ended", "L2": "running"})]
    build = tmp_path / "step4000-lda-2610"
    for lab in ("L1", "L2"):
        assert json.loads((build / f"{lab}.line.json").read_text())["state"] == "ended"
    # a label of no plan: refused before any step
    seen.clear()
    with pytest.raises(SystemExit):
        fx.main(["L1", "L99", "--as", "served/x", "--gguf", "g", "--server", "s",
                 "--root", str(tmp_path)])
    assert seen == []


def test_1_the_docs_give_the_labels():
    doc = (Path(__file__).resolve().parents[1] / "docs" / "REMOTE-RUNS.md").read_text()
    sec = doc[doc.index("## L3."):doc.index("## L4.")]
    assert "<host 1>:<port 1>=L1,L2" in sec
    l2 = doc[doc.index("## L2."):doc.index("## L3.")]
    assert "for b in L1 L2; do" in l2 and "frontier_box.py L1 L2" in l2
