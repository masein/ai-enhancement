"""31: the DeviceMark raw runs' public repository under its new name. A stored
link moves to it, a pasted old link lands on it, and what an export writes names
neither the board nor a box, nor who imported a bundle."""

import json

import pytest

import export_devicemark_raw as ex

OLD = "https://huggingface.co/datasets/Teraformer-limited/evalboard-devicemark-raw/tree/main/x"
NEW = "https://huggingface.co/datasets/Teraformer-limited/devicemark-replica-runs/tree/main/x"
PRIVATE = ("https://huggingface.co/datasets/Teraformer-limited/evalboard-devicemark-raw-private"
           "/tree/main/x")


def stored(out, name, url, by="masein", at=5.0):
    d = out / name
    d.mkdir(parents=True)
    p = d / ex.RAW_NAME
    p.write_text(json.dumps({"url": url, "by": by, "at": at}), encoding="utf-8")
    return p


def test_relink_moves_the_public_links_and_keeps_who_set_them(tmp_path):
    public = stored(tmp_path, "a", OLD)
    personal = stored(tmp_path, "b", OLD.replace("Teraformer-limited", "masein"))
    said = []
    assert ex.relink(tmp_path, said.append) == 2
    assert json.loads(public.read_text()) == {"url": NEW, "by": "masein", "at": 5.0}
    assert json.loads(personal.read_text())["url"] == NEW
    assert len(said) == 2 and NEW in said[0]


def test_relink_leaves_the_private_repository_and_a_link_already_moved(tmp_path):
    private, moved = stored(tmp_path, "a", PRIVATE), stored(tmp_path, "b", NEW)
    before = private.read_text(), moved.read_text()
    assert ex.relink(tmp_path, lambda _: None) == 0
    assert (private.read_text(), moved.read_text()) == before


def test_a_pasted_old_link_lands_on_the_new_name(tmp_path):
    assert ex.set_link(tmp_path, OLD, "masein")["url"] == NEW


@pytest.mark.parametrize("line, out", [
    ("Including path: /workspace/qwen-1/bench/devicemark/tasks",
     "Including path: $BENCH_ROOT/devicemark/tasks"),
    ("cd /workspace/gemma-3 && ls", "cd $BENCH_ROOT && ls"),
    ("rows-shard-3-of-3.tar.gz (sha256 6ac9e5f60ff0eda8) by masein: dm_ifeval · shard 3 of 3",
     "rows-shard-3-of-3.tar.gz (sha256 6ac9e5f60ff0eda8): dm_ifeval · shard 3 of 3"),
])
def test_the_scrub_takes_the_box_and_the_person_off_a_log_line(line, out):
    assert ex.scrub(line, env={}, hosts=[]) == out


def test_nothing_written_for_every_row_says_board_or_evalboard():
    acc = {"acc": 0.5}
    scores = {"composite": {"value": 0.5, "ci": [0.4, 0.6]},
              "tests": {"ifeval": acc, "mmlu_pro": acc, "math": acc}}
    readme = ex.readme("a/b", False, {}, scores, {"revisions": {}})
    assert "board" not in readme.lower() and "board" not in ex.RECOMPUTE.lower()


def test_a_setup_goes_out_without_the_serving_switches():
    out, _ = ex.setup_out({"model": "a/b", "lookahead": False, "mtp": False, "cap": 4096})
    assert out["model"] == "a/b" and out["cap"] == 4096
    assert "lookahead" not in out and "mtp" not in out
