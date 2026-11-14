"""r25 (masein, 14 Nov): the release-day re-export put all 27 DeviceMark rows in private/
without a word. The 13 rows published in the public repository went public in early
October by 15.4's name rule; their models were never marked public after 17h made the mark
the rule, and the scrub's names weren't given. Now:
- with the 8 public models marked as their pages mark them, the 13 come out in public/ and
  every served/ row stays in private/;
- a published row that would go to private/ (no mark, or public/ refused) stops the export
  before it writes anything, naming the rows and why;
- --private still writes every row private, and an export with no published row still
  writes private with a word.

The board's 27 DeviceMark rows as listed on 14 Nov, cloned from one real row made with
15.1's stand-in lm_eval; nothing runs."""

from __future__ import annotations

import builtins
import json
import shutil

import pytest

import devicemark as dm
import export_devicemark_raw as ex
from service import config, db
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the board's fixture
from test_15_2_import import board, row_of  # noqa: F401 — board is the fixture
from test_15_4_raw_export import QWEN, no_gated_sets, row  # noqa: F401 — fixtures

PUBLIC = f"https://huggingface.co/datasets/{ex.PUBLIC_REPO}/tree/main/"
PRIVATE = f"https://huggingface.co/datasets/{ex.PUBLIC_REPO.split('/')[0]}/evalboard-devicemark-raw-private/tree/main/"
# (model, thinking, where its raw link points): the board's DeviceMark rows on 14 Nov
ROWS = [(m, t, PUBLIC) for m, t in [
    ("LiquidAI/LFM2.5-1.2B-Instruct", False), ("Nanbeige/Nanbeige4.1-3B", False),
    ("Qwen/Qwen3.5-0.8B", False), ("Qwen/Qwen3.5-0.8B", True), ("Qwen/Qwen3.5-2B", False),
    ("Qwen/Qwen3.5-2B", True), ("Qwen/Qwen3.5-4B", False), ("Qwen/Qwen3.5-4B", True),
    ("google/gemma-4-E2B-it", False), ("google/gemma-4-E2B-it", True),
    ("ibm-granite/granite-4.0-h-1b", False), ("tencent/Youtu-LLM-2B", False),
    ("tencent/Youtu-LLM-2B", True)]] + [(m, t, PRIVATE) for m, t in [
        ("served/Qwen3.6-35B-A3B-Q4-original-k-8", False), ("served/Qwen3.6-35B-A3B-Q4-original-k-8", True),
        ("served/Qwen3.6-35B-A3B-k4-LDA-phone-build", False), ("served/Qwen3.6-k4-LDA-MTP", False),
        ("served/Qwen3.6-k4-LDA-MTP", True), ("served/Qwen3.6-k4-LDA-lookahead", False),
        ("served/Qwen3.6-k4-LDA-lookahead-MTP", False)]] + [(m, t, None) for m, t in [
            ("served/step2388-k3k6-2611", False), ("served/step2388-k3k6-2611", True),
            ("served/step2388-k3k6-2611-app", True), ("served/step2388-k3k6-2611-file", False),
            ("served/step2388-k3k6-2611-medium", False), ("served/step8804-lda-2610", False),
            ("served/step8804-lda-2610", True)]]
MARKED = sorted({m for m, _, link in ROWS if link == PUBLIC})          # the 8 public models


def name(model: str, thinking: bool) -> str:
    return model.replace("/", "__") + ("__thinking" if thinking else "")


PUBLISHED = {name(m, t) for m, t, link in ROWS if link == PUBLIC}
SERVED = {name(m, t) for m, t, _ in ROWS if m.startswith("served/")}


@pytest.fixture
def rows27(row, monkeypatch):  # noqa: F811
    """the 27 rows, each with its raw link as the board has it"""
    base = row_of(QWEN)
    meta = json.loads((base / "model_meta.json").read_text())
    for m, t, link in ROWS:
        d = config.OUT_DIR / name(m, t)
        if d != base:
            shutil.copytree(base, d, dirs_exist_ok=True)   # the run's own model folder may be there
        (d / "model_meta.json").write_text(json.dumps({**meta, "model": m + (" · thinking" if t else "")}))
        (d / dm.RAW_NAME).unlink(missing_ok=True)
        if link:
            ex.set_link(d, link + name(m, t), "masein")
    assert len(ex.all_rows()) == 27
    monkeypatch.setenv("SCRUB_HOSTS", "board-host")
    monkeypatch.setenv("SCRUB_ACCOUNTS", "teamacct")
    asked = []
    monkeypatch.setattr(builtins, "input", lambda prompt="": (asked.append(prompt), "yes")[1])
    return asked


def written(out) -> dict:
    return {kind: {p.name for p in (out / kind).iterdir()} if (out / kind).exists() else set()
            for kind in ("public", "private")}


def test_the_13_public_models_come_out_public_and_every_served_row_stays_private(
        rows27, tmp_path, capsys):
    for m in MARKED:
        db.public_set(m, True, "masein")              # as each model's page marks it
    out = tmp_path / "release-dm"
    assert ex.main(["--all", "--out", str(out)]) == 0
    said = capsys.readouterr().out
    got = written(out)
    assert got["public"] == PUBLISHED and len(got["public"]) == 13
    assert got["private"] == SERVED and len(got["private"]) == 14
    assert not any(n.startswith("served__") for n in got["public"])
    # the typed yes was asked once, to the 8 models' list, and no served/ model is on it
    assert len(rows27) == 1
    listed = [x.strip().split(" — ")[0] for x in said.splitlines()
              if x.startswith("  ") and not x.startswith("   ") and " — " in x]
    assert sorted(listed) == MARKED, said


def test_published_rows_with_no_mark_stop_the_export_before_it_writes(rows27, tmp_path, capsys):
    out = tmp_path / "release-dm"
    assert ex.main(["--all", "--out", str(out)]) == 2
    said = capsys.readouterr().out
    assert not out.exists() and not rows27              # nothing written, nothing asked
    assert "stopped, nothing written: 13 row(s) published in" in said
    for n in PUBLISHED:
        assert f"  {n}\n" in said, n
    assert "mark each one's weights public on its page (" + ", ".join(MARKED) + ")" in said
    assert "--private" in said
    # --private, on purpose: every row private, as before
    assert ex.main(["--all", "--private", "--out", str(out)]) == 0
    got = written(out)
    assert got["public"] == set() and len(got["private"]) == 27


def test_published_rows_and_public_refused_stop_too(rows27, tmp_path, capsys, monkeypatch):
    for m in MARKED:
        db.public_set(m, True, "masein")
    monkeypatch.delenv("SCRUB_HOSTS")
    monkeypatch.delenv("SCRUB_ACCOUNTS")
    out = tmp_path / "release-dm"
    assert ex.main(["--all", "--out", str(out)]) == 2
    said = capsys.readouterr().out
    assert not out.exists() and not rows27
    assert "stopped, nothing written: 13 row(s)" in said and "SCRUB_HOSTS isn't given" in said
    assert "mark each one" not in said                  # they are marked: only the refusal


def test_with_no_published_row_an_export_writes_private_and_says_so(rows27, tmp_path, capsys):
    for m, t, _ in ROWS:
        (config.OUT_DIR / name(m, t) / dm.RAW_NAME).unlink(missing_ok=True)
    out = tmp_path / "internal"
    assert ex.main(["--all", "--out", str(out)]) == 0
    said = capsys.readouterr().out
    assert "no row's model is marked public on the board: every row goes to private/" in said
    assert len(written(out)["private"]) == 27 and not written(out)["public"]


def test_which_links_are_the_public_repository(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    assert ex.on_public_repo(d) is False                                 # no link
    for url, want in ((PUBLIC + "x", True), (PRIVATE + "x", False),
                      # the repository's name until 31 Oct still counts
                      ("https://huggingface.co/datasets/Teraformer-limited/evalboard-devicemark-raw/tree/main/x", True),
                      ("https://huggingface.co/datasets/someone/devicemark-replica-runs/tree/main/x", False)):
        (d / dm.RAW_NAME).write_text(json.dumps({"url": url}))
        assert ex.on_public_repo(d) is want, url
    (d / dm.RAW_NAME).write_text("not json")
    assert ex.on_public_repo(d) is False
