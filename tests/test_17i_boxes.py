"""17i, part 2: the boxes and the fetch (docs/prompts/phase-17i-seventh-review-
hle-reader-export-fetch.md, points 7 to 15). A test a point, each failing on
511854e. The boxes are invented listings and a stand-in ssh; nothing is
fetched, no model runs, no paid API is called."""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tarfile
import time
from pathlib import Path

import pytest

import remote_bundle as rb
import remote_gguf as rg
from fake_openai import FakeServer
from service import config, db, served
from service import frontier as sf
from test_12q_devicemark_runs import ME, svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import ROW, SERVED, TASK, box, bundle_of, register, run_box  # noqa: F401
from test_17b_review import imported, rewrite
from test_17e_review import a_bundle
from test_17g_review import MMLU, PHONE, fetch_one, progress_of

BF16 = "Qwen3.6-35B-A3B-BF16"
PARTS = [(f"{BF16}-00001-of-00002.gguf", "1a" * 32), (f"{BF16}-00002-of-00002.gguf", "2b" * 32)]
IDENT = rb.split_sha(PARTS)


def main_of(ff, key, dest, *more):
    return ff.main(["--key", str(key), "--dest", str(dest), "--sha", PHONE, "--no-board", *more])


def split_bundle(path: Path, model: str, parts=PARTS, ident: str = IDENT) -> Path:
    """a bundle whose setup.json names a GGUF in parts"""
    with tarfile.open(path, "w:gz") as tar:
        for name, blob in (("bundle.json", {"model": model}),
                           ("setup.json", {"gguf": {
                               "name": parts[0][0], "sha256": ident,
                               "parts": [{"name": n, "sha256": h, "size": 1} for n, h in parts]}})):
            data = json.dumps(blob).encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    return path


# ---------------------------------------------------------------------------
# 7: --abandoned <build>/<step>
# ---------------------------------------------------------------------------

def bf16_box(there: Path, a3_2: bool = True):
    """the BF16 box today: A3-parity whole, A3-2 stopped by hand, A9's two
    steps whole with their bundles"""
    par = there / "parity.jsonl"
    par.write_text('{"parity_of": {}}\n')
    b1 = a_bundle(there / "frontier-served__phone-thinking-on-mmlu-pro.tar.gz", "served/phone")
    b2 = a_bundle(there / "frontier-served__phone-thinking-off-mmlu-pro.tar.gz", "served/phone")
    progress = [progress_of("phone", "A3-parity", "whole", parity=True),
                progress_of("phone", "A9-1", "whole", tasks=[MMLU], bundle={"name": b1.name}),
                progress_of("phone", "A9-2", "whole", tasks=[MMLU], bundle={"name": b2.name})]
    steps = [("phone", "A3-parity"), ("phone", "A9-1"), ("phone", "A9-2")]
    if a3_2:
        progress.append(progress_of("phone", "A3-2", "stopped", tasks=[MMLU],
                                    why="stopped by hand"))
        steps.append(("phone", "A3-2"))
    files = {f"/workspace/phone/A9-1/{b1.name}": b1, f"/workspace/phone/A9-2/{b2.name}": b2}
    return files, progress, {"/workspace/phone/A3-parity/parity.jsonl": par}, steps


def test_7_an_abandoned_step_is_left_out_and_said(tmp_path, monkeypatch, capsys):
    there = tmp_path / "there"
    there.mkdir()
    files, progress, parity, steps = bf16_box(there)
    ff, calls, key = fetch_one(tmp_path, monkeypatch, files, progress, parity, steps=steps)
    dest = tmp_path / "b" / "bundles"
    main_of(ff, key, dest, "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "NOT safe to destroy — phone A3-2 is stopped" in out, out
    code = main_of(ff, key, dest, "--abandoned", "phone/A3-2", "1.1.1.1:41")
    out = capsys.readouterr().out
    first = next(x for x in out.splitlines() if x.startswith("1.1.1.1:41"))
    assert "done, safe to destroy" in first and "abandoned: phone/A3-2" in first, out  # 511854e
    assert "phone/A3-2: abandoned (--abandoned)" in out and code == 0
    # deleted, it reads "hasn't started" no more (A3-parity's folder implies A3's plan)
    files, progress, parity, steps = bf16_box(there, a3_2=False)
    ff, calls, key = fetch_one(tmp_path, monkeypatch, files, progress, parity, steps=steps)
    main_of(ff, key, dest, "--abandoned", "phone/A3-2", "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "hasn't started" not in out and "done, safe to destroy" in out, out
    with pytest.raises(SystemExit):
        main_of(ff, key, dest, "--abandoned", "A3-2", "1.1.1.1:41")


# ---------------------------------------------------------------------------
# 8: "safe" forgotten when a box is reached and isn't, and kept across a restart
# ---------------------------------------------------------------------------

def test_8_safe_is_forgotten_when_reached_and_not_and_kept_across_a_restart(
        tmp_path, monkeypatch, capsys):
    import frontier_fetch as ff
    from test_17f_review import listing_of
    from test_17g_review import with_steps
    there = tmp_path / "there"
    there.mkdir()
    b = a_bundle(there / "frontier-served__phone-thinking-on-mmlu-pro.tar.gz", "served/phone")
    whole = progress_of("phone", "A5-1", "whole", tasks=[MMLU], bundle={"name": b.name})
    asking = progress_of("orig", "A6-1", "asking", tasks=[MMLU])
    state = {"round": 0, "down": set()}

    def run(cmd, cwd=None, timeout=None, stdin=None):
        if cmd[0] == "ssh":
            host = next(x for x in cmd if "@" in x).split("@")[1]
            if host in state["down"]:
                return 255, f"ssh: connect to host {host} port 41: Connection refused"
            if host == "2.2.2.2":                       # another box, still working
                return 0, with_steps(listing_of({}, [asking]), [("orig", "A6-1")])
            # 1.1.1.1: safe; then another build asking on it; then not reached
            prog = [whole] if state["round"] == 0 else [whole, asking]
            return 0, with_steps(listing_of({f"/workspace/phone/A5-1/{b.name}": b}, prog),
                                 [("phone", "A5-1")]
                                 + ([("orig", "A6-1")] if state["round"] else []))
        if cmd[0] == "scp":
            Path(cmd[-1]).write_bytes(b.read_bytes())
            return 0, ""
        return 0, "the row served/phone · thinking: imported"
    monkeypatch.setattr(ff, "run", run)
    monkeypatch.setattr(ff, "keep_sudo", lambda: None, raising=False)

    def sleep(s):
        state["round"] += 1
        if state["round"] == 2:
            state["down"] = {"1.1.1.1"}
        if state["round"] > 3:
            raise AssertionError("--every never stopped")
    monkeypatch.setattr(ff.time, "sleep", sleep)
    key = tmp_path / "id"
    key.write_text("k")
    dest = tmp_path / "b" / "bundles"
    with pytest.raises(AssertionError, match="never stopped"):
        ff.main(["--key", str(key), "--dest", str(dest), "--sha", PHONE, "--no-board",
                 "--every", "1s", "1.1.1.1:41", "2.2.2.2:42"])
    out = capsys.readouterr().out
    assert "destroyed: done" not in out, out          # 511854e: "destroyed: done"
    # read safe, then the fetch started again with the box gone: done, at once
    state.update(round=0, down=set())
    assert ff.main(["--key", str(key), "--dest", str(dest), "--sha", PHONE, "--no-board",
                    "1.1.1.1:41"]) == 0
    state["down"] = {"1.1.1.1"}
    capsys.readouterr()
    # 18b: read safe before this fetch started: destroyed once unreached for
    # 30 minutes, never on two rounds one --every apart — a clock that moves
    # half of that a round
    clock = {"t": time.time()}
    monkeypatch.setattr(ff.time, "sleep", lambda s: clock.update(t=clock["t"] + 15.5 * 60))
    monkeypatch.setattr(ff.time, "time", lambda: clock["t"])
    assert ff.main(["--key", str(key), "--dest", str(dest), "--sha", PHONE, "--no-board",
                    "--every", "1s", "1.1.1.1:41"]) == 0    # 511854e: loops for ever
    out = capsys.readouterr().out
    assert "destroyed once it has been unreached for 30 min" in out, out
    assert "destroyed: done" in out
    assert "1.1.1.1" not in (dest.parent / "safe-boxes.json").read_text()


# ---------------------------------------------------------------------------
# 9: a GGUF in parts, imported by hand
# ---------------------------------------------------------------------------

def test_9_the_by_hand_line_gives_the_split_identity_and_its_parts(tmp_path, monkeypatch,
                                                                     capsys):
    there = tmp_path / "there"
    there.mkdir()
    par = there / "parity.jsonl"
    par.write_text('{"parity_of": {}}\n')
    g = split_bundle(there / f"frontier-served__{BF16}-thinking-on-gpqa.tar.gz",
                     f"served/{BF16}")
    ff, calls, key = fetch_one(
        tmp_path, monkeypatch, {f"/workspace/{BF16}/A5-1/{g.name}": g},
        [progress_of(BF16, "A5-1", "whole", tasks=["gpqa_diamond_epoch"],
                     bundle={"name": g.name}),
         progress_of(BF16, "A5-parity", "whole", parity=True)],
        {f"/workspace/{BF16}/A5-parity/parity.jsonl": par},
        steps=[(BF16, "A5-1"), (BF16, "A5-parity")])
    run = ff.run
    served_now = {"ids": []}
    monkeypatch.setattr(ff, "run", lambda cmd, cwd=None, timeout=None, stdin=None: (
        0, json.dumps(served_now["ids"])) if "--served" in cmd else run(cmd, cwd, timeout,
                                                                        stdin))
    dest = tmp_path / "b" / "bundles"
    main_of(ff, key, dest, "1.1.1.1:41")
    out = capsys.readouterr().out
    line = out[out.index("import it by hand"):]
    assert f"--file-sha256 {IDENT}" in line, out                # 511854e: <its sha256>
    assert f"{IDENT} is the split identity of its 2 parts" in line
    for n, h in PARTS:
        assert f"{n}  {h}" in line
    assert f"give the fetch --sha served/{BF16}={IDENT}" in line
    assert '--register "<its name>"' in line
    # "1 imported" counted the parity file
    assert "(0 imported, 1 to import by hand" in out, out
    # registered since: the next round asks again, and says no --register
    served_now["ids"] = [f"served/{BF16}"]
    main_of(ff, key, dest, "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "--register" not in out[out.index("import it by hand"):].splitlines()[0], out


def split_of(b: Path, parts, ident: str, name: str) -> Path:
    """the bundle as a box writes one for a GGUF in parts"""
    def fix(files):
        setup = json.loads(files["setup.json"])
        setup["gguf"] = {**setup["gguf"], "sha256": ident,
                         "parts": [{"name": n, "sha256": h, "size": 1} for n, h in parts]}
        files["setup.json"] = json.dumps(setup).encode()
        bundle = json.loads(files["bundle.json"])
        bundle["gguf_sha256"] = ident
        files["bundle.json"] = json.dumps(bundle).encode()
    out = rewrite(b, fix)
    return out.rename(out.with_name(f"{name}-{b.name}"))


def test_9_the_import_checks_the_parts_make_the_identity(box):  # noqa: F811
    assert run_box(box, "run") == 0
    good = split_of(bundle_of(box, "run"), PARTS, IDENT, "good")
    register(IDENT)
    # a part's own sha256: refused, saying so
    code, said = imported(good, file_sha=PARTS[0][1])
    assert code == 2 and any(f"is the sha256 of one part, {PARTS[0][0]}" in x and IDENT in x
                             for x in said), said            # 511854e: isn't the registered file
    # parts that don't make the identity: refused, never recorded
    bad = split_of(bundle_of(box, "run"), [PARTS[0], (PARTS[1][0], "3c" * 32)], IDENT, "bad")
    code, said = imported(bad, file_sha=IDENT)
    assert code == 2 and any("don't make the split identity" in x for x in said), said
    code, said = imported(good, file_sha=IDENT)
    assert code == 0, said


# ---------------------------------------------------------------------------
# 10: a shard import killed part-way keeps its shard
# ---------------------------------------------------------------------------

class Killed(BaseException):
    pass


def test_10_a_shard_import_killed_part_way_is_recorded_on_the_next(box, monkeypatch):  # noqa: F811
    import import_frontier as imf
    assert run_box(box, "s1", "--shard", "1/2") == 0
    assert run_box(box, "s2", "--shard", "2/2") == 0
    register(box["sha"])
    assert imported(bundle_of(box, "s1", (1, 2)))[0] == 0
    real = imf._write_registry

    def killed(row, reg):
        raise Killed()                          # a deploy during the round
    monkeypatch.setattr(imf, "_write_registry", killed)
    with pytest.raises(Killed):
        imported(bundle_of(box, "s2", (2, 2)))
    monkeypatch.setattr(imf, "_write_registry", real)
    code, said = imported(bundle_of(box, "s2", (2, 2)))
    assert code == 0 and any("shard 2 of 2 was here but not recorded" in x for x in said), said
    reg = imf.registry(config.OUT_DIR / ROW)
    assert set(reg["shards"][TASK]["have"]) == {"1", "2"}       # 511854e: "nothing changed"
    assert TASK in reg["tasks"] and reg["tasks"][TASK]["shards"] == 2
    code, said = imported(bundle_of(box, "s2", (2, 2)))
    assert code == 0 and any("imported already" in x for x in said), said


# ---------------------------------------------------------------------------
# 11, 14, 15: the board's list of boxes
# ---------------------------------------------------------------------------

def step(label, box_id, now, state="asking", seen=None, **more):
    return {"label": label, "model": "served/phone", "step": f"{label}-1", "state": state,
            "at": now - 60, "seen_at": now if seen is None else seen, "safe": False,
            "reachable": True, "box_id": box_id, "line": "", **more}


def test_11_rows_leave_by_when_they_were_seen_and_gone_steps_at_once(svc):  # noqa: F811
    import import_frontier as imf
    now = time.time()
    # a box taken off the fetch's command line: never "not reached", seen two days ago
    imf.store_boxes([{"asked": ["a", "b"], "reached": ["a", "b"]},
                     step("A1", "a", now, seen=now - 2 * 86400), step("A2", "b", now)])
    assert {b["label"] for b in imf.read_boxes(now)["boxes"]} == {"A2"}   # 511854e: A1 too
    # a build's steps gone from a box the fetch reached: gone at once
    imf.store_boxes([{"asked": ["b"], "reached": ["b"]}, step("A3", "b", now)])
    assert {b["label"] for b in imf.read_boxes(now)["boxes"]} == {"A3"}, \
        imf.read_boxes(now)                                  # 511854e: A2 "not reached"
    # not reached: kept as not reached, as before
    imf.store_boxes([{"asked": ["b"], "reached": []}])
    got = imf.read_boxes(now)["boxes"]
    assert [(b["label"], b["reachable"]) for b in got] == [("A3", False)]


def test_14_a_stored_row_can_never_take_the_list_down(svc):  # noqa: F811
    import import_frontier as imf
    now = time.time()
    digits = "1" * 5000
    imf.store_boxes([step("A1", "a", now, line=f"{digits} of {digits} · {digits} min left"),
                     step("A2", "a", now, line="x" * 2_000_000), step("A3", "a", now)])
    got = imf.read_boxes(now)["boxes"]                       # 511854e: ValueError
    assert len(got) == 3 and max(len(b["line"]) for b in got) <= imf.CAP
    # rows stored by hand, of the wrong kinds: left out, the list served
    blob = json.loads(imf.boxes_path().read_text())
    blob["boxes"] += [{**step("B1", "a", now), "step": 7},
                      {**step("B2", "a", now), "label": ["A", "B"]},
                      {**step("B3", "a", now), "at": "yesterday", "seen_at": "today"}]
    imf.boxes_path().write_text(json.dumps(blob))
    assert {b["label"] for b in imf.read_boxes(now)["boxes"]} == {"A1", "A2", "A3"}
    assert svc.get("/api/frontier/boxes").status_code == 200          # 511854e: 500
    imf.store_boxes([{"asked": ["a"], "reached": ["a"]}, step("A1", "a", now)])  # 511854e: TypeError


def test_15_the_new_fetch_reads_on_a_board_from_before(tmp_path, monkeypatch, svc):  # noqa: F811
    import frontier_fetch as ff
    import import_frontier as imf
    got = {}

    def run(cmd, cwd=None, timeout=None, stdin=None):
        got["file"] = json.loads(Path(cmd[-1]).read_text())
        return 0, "1 step on rented boxes"
    monkeypatch.setattr(ff, "run", run)
    now = time.time()
    ff.post_boxes([step("A1", "a", now)], tmp_path / "bundles", ["a"], ["a"])
    # a board before 17i read a list of steps: every element it can't read passed by
    assert isinstance(got["file"], list)                      # 511854e: {"steps", "asked"}
    old_read = [r for r in got["file"] if isinstance(r, dict) and r.get("label")]
    assert [r["label"] for r in old_read] == ["A1"]
    assert imf.store_boxes(got["file"]) == 1


def test_15_a_card_whose_memory_says_n_a_keeps_its_name(monkeypatch):
    monkeypatch.setattr(rb.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, "NVIDIA RTX PRO 6000 Blackwell, 580.65, 97887, [N/A]\n", ""))
    g = rb.gpu_info()
    assert g["name"] == "NVIDIA RTX PRO 6000 Blackwell" and g["memory_mib"] == 97887, g
    assert "memory_used_mib" not in g                         # 511854e: name None, no memory


def test_15_registering_the_same_name_keeps_the_files_sha256_and_parts(svc):  # noqa: F811
    fake = FakeServer()
    try:
        fake.model_path = f"/models/{PARTS[0][0]}"
        rec = served.register({"name": "board box", "base_url": fake.base, "how": "x",
                               "thinking": "off"}, ME)
        # as an import from rented GPUs keeps it: the files' bytes as its size
        # (17j: the same file's — kept by its name only when the sizes agree)
        db.served_put({**rec, "pin": {**rec["pin"], "size": fake.size},
                       "file_sha256": {"sha256": IDENT, "by": "masein", "name": PARTS[0][0],
                                       "size": fake.size + 4_000_000,
                                       "parts": [{"name": n, "sha256": h} for n, h in PARTS]}})
        served.register({"name": "board box", "base_url": fake.base, "how": "y",
                         "thinking": "off"}, ME)
        fs = served.get(rec["id"]).get("file_sha256") or {}
        assert fs.get("sha256") == IDENT and len(fs.get("parts") or []) == 2   # 511854e: dropped
    finally:
        fake.close()


def test_15_a_shard_held_after_a_whole_run_is_said_on_its_score(box):  # noqa: F811
    assert run_box(box, "run") == 0
    assert run_box(box, "s1", "--shard", "1/2") == 0
    register(box["sha"])
    assert imported(bundle_of(box, "run"))[0] == 0
    code, said = imported(bundle_of(box, "s1", (1, 2)))
    assert code == 0, said
    d = sf.task_dir(config.OUT_DIR / ROW, TASK)
    fr = json.loads(next(d.glob("results_*.json")).read_text())["frontier"]
    assert fr.get("shards_held", "").startswith("shard 1 of 2 imported and held"), fr
    # and scored again (grading), still said
    sf.score_task(config.OUT_DIR / ROW, TASK, served.get(SERVED))
    fr = json.loads(next(d.glob("results_*.json")).read_text())["frontier"]
    assert "held until the other 1 is in" in fr.get("shards_held", ""), fr


# ---------------------------------------------------------------------------
# 12: parity verdicts carry their box, file and time
# ---------------------------------------------------------------------------

def test_12_a_parity_verdict_is_its_files_and_shown_only_for_a_build_asked(
        tmp_path, monkeypatch, capsys):
    there = tmp_path / "there"
    there.mkdir()
    par = there / "parity.jsonl"
    par.write_text('{"parity_of": {}}\n')
    server = tmp_path / "orig-server-500.jsonl"
    server.write_text("{}\n")
    dest = tmp_path / "b" / "bundles"
    listing = ([progress_of("orig", "A3-parity", "whole", parity=True)],
               {"/workspace/orig/A3-parity/parity.jsonl": par})
    ff, calls, key = fetch_one(tmp_path, monkeypatch, {}, *listing,
                               steps=[("orig", "A3-parity")],
                               compare=(3, "Not the same setup: 38.0% against 42.4%\n"))
    assert main_of(ff, key, dest, "--parity", f"served/orig={server}", "1.1.1.1:41") == 1
    v = json.loads((dest.parent / "parity" / "verdicts.json").read_text())["orig"]
    assert v["file_sha256"] == hashlib.sha256(par.read_bytes()).hexdigest()
    assert v["box"] and v["at"] and v["same"] is False        # 511854e: [false, "Not …"]
    capsys.readouterr()
    # a fetch for another build: the verdict isn't its, and exits 0
    ff, calls, key = fetch_one(tmp_path, monkeypatch, {}, [], steps=[])
    code = main_of(ff, key, dest, "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "orig's parity" not in out and code == 0, out      # 511854e: printed, exit 1
    # a new parity file from the box replaces the verdict
    par.write_text('{"parity_of": {"again": true}}\n')
    ff, calls, key = fetch_one(tmp_path, monkeypatch, {}, *listing,
                               steps=[("orig", "A3-parity")])
    code = main_of(ff, key, dest, "--sha", f"served/orig={'cd' * 32}", "1.1.1.1:41")
    out = capsys.readouterr().out
    assert "Not the same" not in out and code == 0, out
    # 17j: replaced — kept as a tombstone, so a second fetch's merge can't bring it back
    assert json.loads((dest.parent / "parity" / "verdicts.json").read_text())["orig"]["gone"]


# ---------------------------------------------------------------------------
# 13: a parity step writes its progress as it asks
# ---------------------------------------------------------------------------

def test_13_a_parity_step_writes_its_progress_as_it_goes(box, monkeypatch):  # noqa: F811
    import frontier as fb
    monkeypatch.setitem(fb.PARITY, "floor", 50)
    items = [{"id": str(k), "question": f"Q{k}?", "options": ["w", "x", "y"],
              "answer": "A", "category": "law"} for k in range(60)]
    monkeypatch.setattr(fb, "_fetch", lambda task: {"items": items, "extra": {"shots": {}}})
    wrote = []
    real = rg.progress

    def progress(out, **fields):
        wrote.append(fields)
        return real(out, **fields)
    monkeypatch.setattr(rg, "progress", progress)
    assert run_box(box, "parity", "--parity", "--n", "50") == 0
    lines = [f.get("line") or "" for f in wrote if f.get("state") == "asking"]
    assert any(re.fullmatch(r"parity [1-9][\d,]* of [\d,]+", x) for x in lines), \
        wrote                                                 # 511854e: "starting" throughout
