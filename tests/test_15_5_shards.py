"""15.5: a battery in shards — remote_run.py --shard i/n answers every n-th
item of each task from the i-th (the same split on every machine), resumes per
answer as a whole run does, and writes one bundle a shard; import_remote.py
takes the shards in any order, says which are missing, scores a task only once
every item of it is in, and changes nothing for a shard imported twice. The
bundles are made by remote_run.py with 15.1's stand-in lm_eval; nothing runs."""

from __future__ import annotations

import hashlib
import json

import pytest

import devicemark as dm
import import_remote as ir
import remote_bundle as rb
from service import config, db
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is board's fixture
from test_15_1_remote_run import Killed, answers_in, box, lm_eval, run  # noqa: F401 — box
from test_15_2_import import GEMMA, QWEN, board, imported, local, remote, row_of, tree  # noqa: F401

TASKS = ("dm_ifeval", "dm_mmlu_pro", "dm_math")
SIZE = {"dm_ifeval": 300, "dm_mmlu_pro": 196, "dm_math": 100}


def every(task: str) -> list[str]:
    return [k for b, k in dm.keys_for("full") if dm.TASK[b] == task]


def relabel(path, shard: dict):
    """the bundle with bundle.json saying another shard, its answers as they were"""
    b = rb.read(path)
    files = dict(b["files"])
    bundle = json.loads(files["bundle.json"])
    bundle["shard"] = shard
    files["bundle.json"] = json.dumps(bundle).encode()
    return rb.write(path.with_name("relabelled-" + path.name), files)


# ---------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3])
def test_shards_split_each_task_the_same_way_everywhere(n):
    for t in TASKS:
        keys = every(t)
        parts = [dm.shard_of(keys, (i, n)) for i in range(1, n + 1)]
        assert sorted(k for p in parts for k in p) == sorted(keys)        # every item once
        assert max(map(len, parts)) - min(map(len, parts)) <= 1
        assert parts == [dm.shard_of(list(keys), (i, n)) for i in range(1, n + 1)]
        assert parts[0][:2] == [keys[0], keys[n]]                          # every n-th, from the i-th
    assert dm.parse_shard("2/3") == (2, 3) and dm.parse_shard(" 1 / 2 ") == (1, 2)
    assert dm.parse_shard("") is None and dm.parse_shard("1/1") is None
    for bad in ("0/2", "3/2", "2", "a/b", "1/65"):
        with pytest.raises(ValueError):
            dm.parse_shard(bad)


# ---------------------------------------------------------------------------
# on the rented GPU
# ---------------------------------------------------------------------------

def test_a_shard_asks_its_share_resumes_per_answer_and_bundles_alone(box, tmp_path,  # noqa: F811
                                                                    capsys):
    whole = lm_eval(box)
    assert run(tmp_path / "whole", "--shard", "2/3") == 0
    want = {t: dm.shard_of(every(t), (2, 3)) for t in TASKS}
    assert {t: [k for tt, k in whole if tt == t] for t in TASKS} == want
    assert len(whole) == 100 + 65 + 33
    out = capsys.readouterr().out
    assert ("shard 2 of 3: items 2, 5, 8, … of each task · dm_ifeval 100 of 300, dm_mmlu_pro 65 "
            "of 196, dm_math 33 of 100") in out
    assert "dm_ifeval  100/100" in out                    # a line an answer, of the shard's
    # the box stopped half-way through the shard: the same command asks only the rest
    made = lm_eval(box, die_after=120)
    with pytest.raises(Killed):
        run(tmp_path / "halves", "--shard", "2/3")
    again = lm_eval(box)
    assert run(tmp_path / "halves", "--shard", "2/3") == 0
    assert len(made) == 120 and len(again) == 78 and not set(made) & set(again)
    name = rb.bundle_name("devicemark", "Qwen/Qwen3.5-4B", True, (2, 3))
    assert name == "devicemark-Qwen__Qwen3.5-4B-thinking-on-shard-2-of-3.tar.gz"
    a, b = rb.read(tmp_path / "whole" / name), rb.read(tmp_path / "halves" / name)
    assert answers_in(a) == answers_in(b) and set(answers_in(a)["dm_math"]) == set(want["dm_math"])
    assert a["bundle"]["shard"] == {"i": 2, "n": 3} and a["setup"]["shard"] == {"i": 2, "n": 3}
    assert a["bundle"]["tasks"] == {"dm_ifeval": 100, "dm_mmlu_pro": 65, "dm_math": 33}


def test_another_shard_in_the_same_folder_is_refused(box, tmp_path):  # noqa: F811
    lm_eval(box)
    assert run(tmp_path / "run", "--only", "dm_math", "--shard", "1/2") == 0
    with pytest.raises(SystemExit, match="holds shard 1/2 of Qwen/Qwen3.5-4B: give shard 2 of 2 "
                                         "another --out"):
        run(tmp_path / "run", "--only", "dm_math", "--shard", "2/2")
    with pytest.raises(SystemExit):
        run(tmp_path / "bad", "--shard", "3/2")


# ---------------------------------------------------------------------------
# on the server
# ---------------------------------------------------------------------------

def test_the_qwen_case_in_two_shards(board, tmp_path):  # noqa: F811
    # what the row should come to: the same answers, all asked here
    full = local("org/the-same-answers")
    assert full["status"] == "done", full["error"]
    want = json.loads((row_of("org/the-same-answers") / dm.OUT_NAME).read_text())
    # MMLU-Pro and MATH here (#167); IFEval on two rented GPUs
    local(QWEN, "dm_mmlu_pro", "dm_math")
    one = remote(tmp_path / "one", QWEN, "dm_ifeval", shard="1/2")
    two = remote(tmp_path / "two", QWEN, "dm_ifeval", shard="2/2")
    row = row_of(QWEN)
    before = {k: v for k, v in tree().items() if k.startswith(f"full/{row.name}/")}
    code, said = imported(one)
    assert code == 0, said
    assert "dm_ifeval: shard 2 of 2 missing — dm_ifeval is scored once it is imported" in said
    assert said[-2].endswith(": shard 1 of 2 imported; dm_ifeval waits for shard 2 of 2")
    assert not (row / "dm_ifeval_0shot").exists()                        # not scored on half
    now = {k: v for k, v in tree().items() if k.startswith(f"full/{row.name}/")}
    assert {k for k in now if k not in before} == {f"full/{row.name}/remote_imports.json"}
    assert {k: v for k, v in now.items() if k in before} == before
    # the shard waits beside the results, not in them
    assert rb.task_answers(ir.shards_dir(row.name) / "dm_ifeval" / "1-of-2", "dm_ifeval")
    # the same shard again changes nothing
    snap, runs = tree(), len(db.recent(500))
    code, said = imported(one)
    assert code == 0 and said[-1].startswith("imported already, as Runs #")
    assert tree() == snap and len(db.recent(500)) == runs
    # the other: every item in, merged and scored
    code, said = imported(two)
    assert code == 0, said
    assert "dm_ifeval: every shard is in (2 of 2) · 300 answers, merged in the battery's order" in said
    got = json.loads((row / dm.OUT_NAME).read_text())
    assert got["composite"] == want["composite"] and got["benches"] == want["benches"]
    assert got["n"] == 596
    assert got["setup"]["where"] == ("IFEval run on a rented GPU (NVIDIA GeForce RTX 4090), in 2 "
                                     "shards; MMLU-Pro and MATH on this server")
    # one samples file, in the battery's order, numbered as one run's
    files = rb.samples_files(row / "dm_ifeval_0shot", "dm_ifeval")
    lines = [json.loads(x) for x in files[-1].read_text().splitlines()]
    assert len(files) == 1 and [x["doc"]["key"] for x in lines] == every("dm_ifeval")
    assert [x["doc_id"] for x in lines] == list(range(300))


def test_three_shards_in_any_order(board, tmp_path):  # noqa: F811
    paths = {i: remote(tmp_path / f"s{i}", GEMMA, shard=f"{i}/3") for i in (1, 2, 3)}
    code, said = imported(paths[3])
    assert code == 0 and said[-2].endswith(
        "shard 3 of 3 imported; dm_ifeval, dm_mmlu_pro and dm_math wait for shards 1 and 2 of 3")
    code, said = imported(paths[1])
    assert "dm_math: shard 2 of 3 missing — dm_math is scored once it is imported" in said
    assert not (row_of(GEMMA) / dm.OUT_NAME).exists()
    code, said = imported(paths[2])
    assert code == 0, said
    row = json.loads((row_of(GEMMA) / dm.OUT_NAME).read_text())
    assert row["n"] == 596 and row["composite"]["value"] is not None
    assert row["setup"]["where"] == "run on a rented GPU (NVIDIA GeForce RTX 4090), in 3 shards"
    assert {t: x["shards"] for t, x in row["setup"]["remote"].items()} == dict.fromkeys(TASKS, 3)
    notes = [r["note"] for r in db.recent(3)]
    assert [n.split(" · ")[1] for n in notes] == ["shard 2 of 3", "shard 1 of 3", "shard 3 of 3"]
    for t in TASKS:
        assert set(rb.task_answers(row_of(GEMMA) / f"{t}_0shot", t)) == set(every(t))


def test_a_shard_of_another_n_is_refused_while_the_others_wait(board, tmp_path):  # noqa: F811
    assert imported(remote(tmp_path / "a", GEMMA, "dm_math", shard="1/2"))[0] == 0
    snap, runs = tree(), len(db.recent(500))
    code, said = imported(remote(tmp_path / "b", GEMMA, "dm_math", shard="2/3"))
    assert code == ir.REFUSED and said[-1] == "nothing was imported"
    assert ("refused — dm_math: shard 1 of 2 is here, waiting for the rest, and this bundle is "
            "shard 2 of 3 — every shard of a task is one of the same n") in said
    assert tree() == snap and len(db.recent(500)) == runs


def test_a_shard_is_checked_for_its_own_items(board, tmp_path):  # noqa: F811
    path = remote(tmp_path / "a", GEMMA, "dm_math", shard="1/2")
    code, said = imported(relabel(path, {"i": 2, "n": 2}))       # shard 1's answers, called 2
    assert code == ir.REFUSED
    assert "refused — dm_math: 0 of 50 items answered (shard 2 of 2)" in said
    code, said = imported(relabel(path, {"i": 1, "n": 4}))       # more than shard 1 of 4 holds
    assert code == ir.REFUSED and "refused — dm_math: 25 answers to items outside shard 1 of 4" in said
    code, said = imported(relabel(path, {"i": 3, "n": 2}))
    assert code == ir.REFUSED and "refused — its shard 3 of 2 isn't one" in said


def test_a_newer_bundle_of_a_shard_replaces_it_and_keeps_the_earlier(board, tmp_path):  # noqa: F811
    path = remote(tmp_path / "a", GEMMA, "dm_math", shard="1/2")
    assert imported(path)[0] == 0
    b = rb.read(path)                                    # the same answers, another session
    files = dict(b["files"])
    setup = json.loads(files["setup.json"])
    setup["sessions"] = 2
    files["setup.json"] = json.dumps(setup).encode()
    bundle = json.loads(files["bundle.json"])
    bundle["files"] = {n: hashlib.sha256(d).hexdigest() for n, d in files.items()
                       if n != "bundle.json"}
    files["bundle.json"] = json.dumps(bundle).encode()
    code, said = imported(rb.write(tmp_path / "newer.tar.gz", files))
    assert code == 0
    assert any(x.startswith("dm_math: shard 1 of 2 as it was here before is kept at ") for x in said)
    reg = ir.registry(row_of(GEMMA))
    assert reg["shards"]["dm_math"]["have"]["1"]["bundle"] == "newer.tar.gz"
    assert list((config.OUT_DIR.with_name("earlier") / row_of(GEMMA).name).glob(
        "dm_math_0shot-shard-1-of-2-before-import-*"))


def test_where_names_each_card_the_shards_ran_on(tmp_path):
    from service import devicemark as sdm
    row = tmp_path / "row"
    for t in TASKS:
        (row / f"{t}_0shot").mkdir(parents=True)
        (row / f"{t}_0shot" / f"samples_{t}_1.jsonl").write_text("")
    entry = {"gpu": "NVIDIA GeForce RTX 4090 and NVIDIA GeForce RTX 5090", "shards": 2,
             "gpus": ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090"],
             "samples": ["dm_ifeval_0shot/samples_dm_ifeval_1.jsonl"]}
    (row / sdm.REMOTE_NAME).write_text(json.dumps({"tasks": {"dm_ifeval": entry}}))
    assert sdm.where_of(row)["where"] == (
        "IFEval run on rented GPUs (NVIDIA GeForce RTX 4090 and NVIDIA GeForce RTX 5090), in 2 "
        "shards; MMLU-Pro and MATH on this server")
    assert sdm.where_of(row)["remote"]["dm_ifeval"]["shards"] == 2


def test_the_docs_commands_for_the_two_waiting_runs():
    """docs/REMOTE-RUNS.md: five instances — Qwen3.5-4B's IFEval in 2 shards,
    Gemma 4 E2B's battery in 3 — each fetched by the name its run writes"""
    import shlex
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1] / "docs" / "REMOTE-RUNS.md").read_text()
    runs = [shlex.split(x) for x in doc.splitlines() if x.startswith("python scripts/remote_run.py")]
    got = {}
    for argv in runs:
        a = {argv[i]: argv[i + 1] for i in range(2, len(argv) - 1) if argv[i].startswith("--")}
        got[(a["--model"], a["--shard"])] = (a.get("--only"), a["--out"], a["--thinking"])
    assert got == {
        ("Qwen/Qwen3.5-4B", "1/2"): ("dm_ifeval", "/workspace/qwen-1", "on"),
        ("Qwen/Qwen3.5-4B", "2/2"): ("dm_ifeval", "/workspace/qwen-2", "on"),
        ("google/gemma-4-E2B-it", "1/3"): (None, "/workspace/gemma-1", "on"),
        ("google/gemma-4-E2B-it", "2/3"): (None, "/workspace/gemma-2", "on"),
        ("google/gemma-4-E2B-it", "3/3"): (None, "/workspace/gemma-3", "on")}
    fetched = [x.split(":", 1)[1].split()[0] for x in doc.splitlines() if x.startswith("scp ")]
    assert fetched == [f"{out}/{rb.bundle_name('devicemark', m, True, dm.parse_shard(s))}"
                       for (m, s), (_, out, _) in got.items()]


def test_a_new_run_in_shards_replaces_shards_saved_in_the_cut_form(board, tmp_path):  # noqa: F811
    """15.7: the Gemma 4 case — its three shards imported before 15.7, saved
    cut (no reply_form.json beside them); its new run's first shard sets the
    other two aside, to wait for their new runs, rather than merge whole
    replies with cut ones"""
    for i in (1, 2, 3):
        assert imported(remote(tmp_path / f"old{i}", GEMMA, "dm_math", shard=f"{i}/3"))[0] == 0
    row = row_of(GEMMA)
    for i in (1, 2, 3):
        (ir.shards_dir(row.name) / "dm_math" / f"{i}-of-3" / dm.FORM_NAME).unlink()
    before = rb.task_answers(row / "dm_math_0shot", "dm_math")
    new = remote(tmp_path / "new1", GEMMA, "dm_math", shard="1/3")
    assert any(n.endswith("dm_math_0shot/" + dm.FORM_NAME) for n in rb.read(new)["files"])
    code, said = imported(new)
    assert code == 0
    assert sum("here was saved in another form" in x for x in said) == 2
    assert "dm_math: shards 2 and 3 of 3 missing — dm_math is scored once they are imported" \
        in said
    assert rb.task_answers(row / "dm_math_0shot", "dm_math") == before   # the old merge stands
