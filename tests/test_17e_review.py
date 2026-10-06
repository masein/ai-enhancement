"""17e, the pilot's numbers and the run plan (docs/prompts/phase-17e-pilot-
numbers-and-the-run-plan.md). Part 2: the plan, and SimpleQA's missing
answers; part 3's point 8, the memory check. Each test fails on 5083cc7. The
box is tests/fixtures/fake_llama_server.py; nothing is fetched and no model
runs."""

from __future__ import annotations

import io
import json
import struct
import tarfile
from pathlib import Path

import pytest

import frontier as fb
import remote_bundle as rb
import remote_gguf as rg
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import (GPU, ROW, box, bundle_of, invented,  # noqa: F401
                              register, run_box)
from test_17b_review import imported, rewrite

HLE, SQA, MMLU = "hle_text_cais", "simpleqa_epoch", "mmlupro_tiger"


def questions(task):
    """invented questions in each benchmark's own form"""
    if task == SQA:
        return [{"id": str(k), "question": f"Who made thing {k}?", "answer": f"Person {k}"}
                for k in range(12)]
    if task == HLE:
        return [{"id": f"h{k}", "question": f"Hard {k}?", "answer": "x",
                 "answer_type": "exactMatch"} for k in range(12)]
    if task == MMLU:
        return {"items": [{"id": str(k), "question": f"Q{k}?", "options": ["w", "x", "y"],
                           "answer": "A", "category": "law"} for k in range(12)],
                "extra": {"shots": {}}}
    return invented()


# ---------------------------------------------------------------------------
# the SimpleQA finding: asked in the order the box lists
# ---------------------------------------------------------------------------

def test_a_box_asks_its_benchmarks_in_the_order_it_lists_them(box, monkeypatch):  # noqa: F811
    """the pilot's box listed SimpleQA before MMLU-Pro and asked MMLU-Pro first
    (the run's record keeps its tasks sorted): MMLU-Pro, stopped by hand, left
    SimpleQA never asked"""
    monkeypatch.setattr(fb, "_fetch", questions)
    assert run_box(box, "pace", "--only", HLE, "--only", SQA, "--only", MMLU) == 0
    own = (box["root"] / "pace" / rg.OWN_LOG).read_text()
    assert own.index("SimpleQA Verified: 12 of its") < own.index("MMLU-Pro: 12 of its")
    said = "".join(p.read_text() for p in (box["root"] / "pace").rglob("service_*.log"))
    # asked as listed: SimpleQA, then MMLU-Pro (5083cc7: MMLU-Pro first)
    order = sorted((said.index(f"[frontier] {t}: "), t) for t in fb.TASKS
                   if f"[frontier] {t}: " in said)
    assert [t for _, t in order] == [t for t in fb.TASKS if t in ("gpqa_diamond_epoch", HLE,
                                                                  SQA, MMLU)], said


# ---------------------------------------------------------------------------
# 8: the memory check, set from the pilot
# ---------------------------------------------------------------------------

# the phone build's header as the pilot read it: qwen35moe, 41 layers, one in
# four with attention (10), 2 KV heads of 256, a recurrent state on the rest
PILOT_SHAPE = {"arch": "qwen35moe", "layers": 41,
               "kv_heads": [2 if (i + 1) % 4 == 0 else 0 for i in range(41)],
               "key_length": 256, "value_length": 256,
               "ssm": {"conv_kernel": 4, "inner_size": 4096, "state_size": 128,
                       "group_count": 16}}
PILOT_FILE, PILOT_CARD = 22_854_339_808, 32_607
Q8 = ["-ctk", "q8_0", "-ctv", "q8_0", "--flash-attn", "on"]


def test_8_the_memory_check_is_set_from_the_pilot(monkeypatch):
    monkeypatch.setattr(rg, "header_of", lambda src: (PILOT_SHAPE, PILOT_FILE))
    assert rg.kv_per_token(PILOT_SHAPE, "q8_0", "q8_0") == 10_880
    mib = 1024 ** 2
    for ctx, measured in ((36_864, 26_136), (67_584, 29_198)):
        fit, why = rg.kv_fit("hf://x/y/m.gguf", Q8, 8, ctx, PILOT_CARD)
        assert not why
        # within 2.5% of what the pilot measured, and never below it (17f: as said)
        assert 0 <= fit["used"] / mib - measured < 0.025 * measured, (ctx, fit["used"] / mib)
    # the plan's slots fit: 8, 8 and ARC-AGI-2's 5 (0f7c943's check allowed
    # them only with nothing to spare on the largest)
    allowed = [rg.kv_fit("hf://x/y/m.gguf", Q8, 8, c, PILOT_CARD)[0]["fit"]
               for c in (36_864, 67_584, 98_304)]
    assert allowed[0] >= 8 and allowed[1] >= 8 and allowed[2] >= 5, allowed
    # and say why when nothing can be worked out
    monkeypatch.setattr(rg, "header_of", lambda src: (None, None))
    assert rg.kv_fit("hf://x/y/m.gguf", Q8, 8, 36_864, PILOT_CARD) == (
        None, "the GGUF's header couldn't be read before the fetch")


def gguf(path: Path) -> Path:
    """a GGUF header, 48 layers of 4 KV heads of 128, and nothing else"""
    arch = "qwen3moe"
    meta = {"general.architecture": arch, f"{arch}.block_count": 48,
            f"{arch}.attention.head_count": 32, f"{arch}.attention.head_count_kv": 4,
            f"{arch}.attention.key_length": 128, f"{arch}.attention.value_length": 128,
            f"{arch}.embedding_length": 2048}

    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(meta))
    for k, v in meta.items():
        out += s(k) + (struct.pack("<I", 8) + s(v) if isinstance(v, str)
                       else struct.pack("<II", 4, v))
    path.write_bytes(out + b"\0" * 4096)
    return path


def test_8_slots_fit_overrides_the_check_and_an_unread_header_is_said(box, monkeypatch,  # noqa: F811
                                                                      tmp_path):
    # an 8 GB card, 8 slots it says can't fit: --slots-fit runs them, the
    # estimate printed beside
    monkeypatch.setattr(rb, "gpu_info", lambda: {**GPU, "memory_mib": 8192})
    path = gguf(tmp_path / "m.gguf")
    with pytest.raises(SystemExit, match="or --slots-fit if you know these do"):
        run_box(box, "no", "--gguf", str(path), "--slots", "8", "--flags", " ".join(Q8))
    assert run_box(box, "yes", "--gguf", str(path), "--slots", "8", "--flags", " ".join(Q8),
                   "--slots-fit") == 0
    own = (box["root"] / "yes" / rg.OWN_LOG).read_text()
    # 17f: 8 slots of GPQA's 83,968; the room said in MiB
    assert "MiB short; up to 1 slot fit keeping 1,024 — 8 run, as --slots-fit says" in own
    # the fixture's file has no header to read: said, where it was silent
    assert run_box(box, "unread") == 0
    own = (box["root"] / "unread" / rg.OWN_LOG).read_text()
    assert ("memory not worked out: the GGUF's header couldn't be read before the fetch. The "
            "slots were checked against --max-context") in own


# ---------------------------------------------------------------------------
# 2, 3: the plans, one line a box
# ---------------------------------------------------------------------------

def test_2_3_each_plan_runs_every_benchmark_once_one_line_a_box():
    import frontier_box as fbx
    for name, boxes in fbx.PLANS.items():
        have: dict[tuple[str, str], set[int]] = {}
        of: dict[tuple[str, str], int] = {}
        parity = 0
        for steps in boxes.values():
            for th, tasks, shard, slots in steps:
                if th == "parity":
                    parity += 1
                    continue
                i, n = fb.parse_shard(shard) if shard else (1, 1)
                for t in tasks:
                    key = (th, t)
                    assert of.setdefault(key, n) == n, (name, key)
                    assert i not in have.setdefault(key, set()), (name, key)
                    have[key].add(i)
                # ARC-AGI-2 alone, at 5 slots; never thinking off
                assert (slots == 5) == (fbx.ARC in tasks) and (tasks == (fbx.ARC,)
                                                               or fbx.ARC not in tasks)
                # every step fits the card as the box checks it
                assert fb.slot_context(list(tasks), th == "on") * slots <= rg.MAX_CONTEXT
        want = {("on", t) for t in fb.TASKS} | {("off", t) for t in fb.TASKS if t != fbx.ARC}
        assert set(have) == want, name
        assert all(have[k] == set(range(1, of[k] + 1)) for k in have), name
        assert parity == 1, name                       # 17f: one box a build asks it first
    longest = {p: max(sum(fbx.hours(s) for s in st) for st in b.values())
               for p, b in fbx.PLANS.items()}
    assert longest["A"] <= 18 and longest["B"] <= 10.5
    assert len(fbx.PLANS["A"]) == 9 and len(fbx.PLANS["B"]) == 15
    # ARC-AGI-2's answers are its test grids', not its tasks'
    assert fbx.ANSWERS[fbx.ARC] == 334


def test_3_a_box_line_runs_its_steps_and_carries_on_past_one_that_stops(monkeypatch, capsys):
    import frontier_box as fbx
    ran = []

    class Done:
        def __init__(self, code):
            self.returncode = code

    def fake(cmd):
        ran.append(cmd)
        return Done(1 if len(ran) == 1 else 0)
    monkeypatch.setattr(fbx.subprocess, "run", fake)
    code = fbx.main(["A1", "--as", "served/phone", "--gguf", "hf://me/p/m.gguf",
                     "--server", "hf://me/p/s.tar.gz"])
    assert code == 1 and len(ran) == 2
    one, two = ran
    assert one[1].endswith("remote_gguf.py")
    assert one[one.index("--thinking") + 1] == "on" and one[one.index("--shard") + 1] == "1/4"
    # 17f: the build's own folder, the box's label, 7 slots where 8 don't fit
    assert one[one.index("--only") + 1] == HLE
    assert one[one.index("--out") + 1] == "/workspace/phone/A1-1"
    assert one[one.index("--label") + 1] == "A1" and one[one.index("--min-slots") + 1] == "7"
    assert one[one.index("--files") + 1] == "/workspace/files"
    assert one[one.index("--flags") + 1] == "-ctk q8_0 -ctv q8_0 --flash-attn on"
    assert two[two.index("--thinking") + 1] == "off" and "--shard" not in two
    out = capsys.readouterr().out
    assert "A1, step 1: not whole (exit 1)" in out and "paste the same line again" in out
    assert "A1, step 2: whole — its bundle is in /workspace/phone/A1-2" in out


# ---------------------------------------------------------------------------
# 4: one command to fetch and import
# ---------------------------------------------------------------------------

def a_bundle(path: Path, model: str) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        data = json.dumps({"model": model}).encode()
        ti = tarfile.TarInfo("bundle.json")
        ti.size = len(data)
        tar.addfile(ti, io.BytesIO(data))
    return path


def a_listing(dirs: dict[str, Path]) -> str:
    """what a box's listing prints: its bundles and their sha256, no progress"""
    import hashlib
    return json.dumps({"bundles": [{"path": remote, "sha256": hashlib.sha256(
        local.read_bytes()).hexdigest()} for remote, local in dirs.items()], "parity": [],
        "progress": []})


def test_4_one_command_fetches_and_imports_every_box(tmp_path, monkeypatch, capsys):
    import frontier_fetch as ff
    there = tmp_path / "boxes"
    there.mkdir()
    a_bundle(there / "frontier-served__phone-thinking-on-hle-shard-1-of-2.tar.gz",
             "served/phone")
    a_bundle(there / "frontier-served__orig-thinking-off-hle.tar.gz", "served/orig")
    a_bundle(there / "frontier-served__gemma-thinking-on-gpqa.tar.gz", "served/gemma")
    key = tmp_path / "id_ed25519"
    key.write_text("not a key")
    dest = tmp_path / "bundles"
    on_box = {"1.2.3.4": {f"/workspace/phone/A1-1/{n}": there / n for n in
                          ["frontier-served__phone-thinking-on-hle-shard-1-of-2.tar.gz"]}
              | {f"/workspace/orig/A1-2/{n}": there / n for n in
                 ["frontier-served__orig-thinking-off-hle.tar.gz"]},
              "5.6.7.8": {"/workspace/x/y/frontier-served__gemma-thinking-on-gpqa.tar.gz":
                          there / "frontier-served__gemma-thinking-on-gpqa.tar.gz"}}
    calls = []

    def run(cmd, cwd=None, timeout=None, stdin=None):
        calls.append(cmd)
        if cmd[0] == "ssh":
            host = next(x for x in cmd if "@" in x).split("@")[1]
            if host == "9.9.9.9":
                return 255, "ssh: connect to host 9.9.9.9 port 4300: Connection timed out"
            return 0, a_listing(on_box[host])
        if cmd[0] == "scp":
            src, to = cmd[-2], Path(cmd[-1])
            host, remote = src.split("@")[1].split(":", 1)
            to.write_bytes(on_box[host][remote].read_bytes())
            return 0, ""
        assert cmd[:8] == ff.IMPORT and cwd == ff.REPO
        if "orig" in cmd[8]:
            return 2, "frontier-…\nrefused — the file isn't the one registered\nnothing was imported"
        return 0, "x.tar.gz · sha256 …\nthe row served/phone · thinking: HLE: shard 1 of 2 in\nRuns #4"
    monkeypatch.setattr(ff, "run", run)
    sha = "ab" * 32
    code = ff.main(["--key", str(key), "--dest", str(dest), "--sha", f"served/phone={sha}",
                    "--sha", f"served/orig={'cd' * 32}", "1.2.3.4:4100", "root@5.6.7.8:4200",
                    "9.9.9.9:4300"])
    out = capsys.readouterr().out
    assert code == 1
    assert ("frontier-served__gemma-thinking-on-gpqa.tar.gz: copied; no --sha for served/gemma "
            "— import it by hand (G4, G6)") in out
    assert ("frontier-served__orig-thinking-off-hle.tar.gz: copied · refused — the file isn't "
            "the one registered") in out
    assert ("frontier-served__phone-thinking-on-hle-shard-1-of-2.tar.gz: copied · the row "
            "served/phone · thinking: HLE: shard 1 of 2 in") in out
    assert "9.9.9.9:4300: couldn't be asked — ssh: connect to host" in out
    imports = [c for c in calls if c[0] == "sudo"]
    assert [c[c.index("--file-sha256") + 1] for c in imports] == [sha, "cd" * 32]
    ssh = next(c for c in calls if c[0] == "ssh")
    assert ssh[:5] == ["ssh", "-i", str(key), "-p", "4100"]
    assert "BatchMode=yes" in ssh and "ConnectTimeout=15" in ssh


# ---------------------------------------------------------------------------
# 5: the share that ran out, on the cell; the loops, on the import's line
# ---------------------------------------------------------------------------

def test_5_the_cell_shows_the_share_that_ran_out():
    import report_lm_eval as rle
    how = rle.frontier_how({"scored_by": "grader", "answers": 54, "ran_out": 17,
                            "grader": {"model": "openai/o3-mini"}})
    assert "31% ran out of room (17 of 54), counted wrong" in how
    assert "ran out" not in rle.frontier_how({"scored_by": "code", "answers": 54, "ran_out": 0})


def test_5_the_import_says_how_many_that_ran_out_end_in_a_loop(box):  # noqa: F811
    loop = "Let me reconsider the options once more, carefully and slowly. " * 400
    assert fb.ends_in_loop(loop) and not fb.ends_in_loop("Step " + " ".join(
        str(k) for k in range(5000)))
    assert run_box(box, "s", thinking="on") == 0
    register(box["sha"])
    path = bundle_of(box, "s")
    name = f"results/{ROW}/gpqa_diamond_epoch_0shot/frontier/answers.jsonl"

    def two_cut(files):
        lines = [json.loads(x) for x in files[name].decode().splitlines()]
        lines[0].update(finish="length", answer="<think>\n" + loop)
        lines[1].update(finish="length", answer="<think>\nweighing it " + " ".join(
            str(k) for k in range(1500)))
        files[name] = "".join(json.dumps(x) + "\n" for x in lines).encode()
    code, said = imported(rewrite(path, two_cut))
    assert code == 0, said
    assert any(f"2 of these {len(invented()) * fb.BENCH['gpqa_diamond_epoch']['epochs']} "
               "answers ran out of room, 1 of them ending in a loop (the last passage "
               "repeating) — for information" in x for x in said), said


# ---------------------------------------------------------------------------
# part 3: leftovers from checking 17d on 5083cc7
# ---------------------------------------------------------------------------

from fake_openai import FakeServer  # noqa: E402
from service import config, llm, served  # noqa: E402
from service import frontier as sf  # noqa: E402
from service import frontier_grade as fgr  # noqa: E402
from test_12q_devicemark_runs import ME  # noqa: E402
from test_17_gguf_box import N, RUNS, TASK  # noqa: E402
from test_17_grading import ROW as GROW  # noqa: E402
from test_17_grading import drain, results  # noqa: E402
from test_17_grading import svc as gsvc  # noqa: E402,F401 — the grading fixture
from test_17b_grading import GEMINI, GPT, plain, stub  # noqa: E402

GPT_V, GEMINI_V = "openai/gpt-4.1-2025-04-14", "google/gemini-2.5-flash-20250617"


def grades_of(task: str) -> dict:
    return sf.read_grades(sf.task_dir(config.OUT_DIR / GROW, task))


def test_9_the_box_side_is_the_mean_of_its_two_runs_and_its_false_alarms_are_said():
    # 14% of answers flipping between two runs of one setup: 500 questions
    # against one run called them apart in about 19% of checks; against the
    # mean of two, about 7%; at 10%, 6.5% and 1.5%
    assert 0.17 < fb.false_alarm(0.14, 500, 1) < 0.19
    assert 0.06 < fb.false_alarm(0.14, 500, 2) < 0.08
    assert 0.05 < fb.false_alarm(0.10, 500, 1) < 0.07
    assert fb.false_alarm(0.10, 500, 2) < 0.02

    def rows(n, twice=False):
        out = []
        for k in range(n):
            right = k % 5 < 2
            r = {"id": str(k), "key": "A", "answer": f"the answer is ({'A' if right else 'B'})"}
            if twice:       # one question in ten flips, right to wrong as often as back
                again = (not right) if k % 20 in (0, 3) else right
                r["answer2"] = f"the answer is ({'A' if again else 'B'})"
            out.append(r)
        return out
    got = fb.parity_compare(rows(500), rows(500, twice=True))
    assert got["ok"] and got["flip"] == pytest.approx(0.10)
    assert got["right_box"] == pytest.approx(200)               # the mean of its two runs
    assert ("at the box's own flip rate (10% of questions right on one run and wrong on the "
            "other), two identical setups would be called not the same in 1.5% of checks of 500 "
            "questions") in got["words"]
    # fewer than the floor: refused, said
    got = fb.parity_compare(rows(400), rows(400, twice=True))
    assert not got["ok"] and "400 questions on both sides, fewer than 500" in got["words"]


def test_10_a_resume_left_with_one_question_that_truly_fails_finishes(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", lambda task: invented())
    fake = FakeServer()
    try:
        fake.reply = lambda body: "ANSWER: A"

        def bad(body):
            return ((500, "this question breaks the template")
                    if "Invented question 3" in json.dumps(body) else None)
        fake.chat_error = fake.raw_error = bad
        fake.ctx = 40960
        rec = served.register({"name": "board box", "base_url": fake.base, "how": "x",
                               "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "off"}, ME)
        row = config.OUT_DIR / rec["id"].replace("/", "__")
        path = sf.task_dir(row, TASK) / sf.ANSWERS
        # 17f: a 5xx is the server's: kept, and asked again by each run, until
        # it has failed on three separate runs (5083cc7 stopped every resume
        # with "failed on every question it was asked")
        for k in range(1, sf.WRITE_OFF_RUNS):
            assert sf.ask_task(rec, TASK, row, False) == (N * RUNS - RUNS, N * RUNS), k
        assert sf.ask_task(rec, TASK, row, False) == (N * RUNS, N * RUNS)
        never = [r for r in sf.read_answers(path).values() if r.get("unanswered")]
        assert sorted(r["id"] for r in never) == ["rec003"] * RUNS
    finally:
        fake.close()


def test_11_a_second_graders_top_up_is_final_and_named(gsvc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(sf, "UNGRADED_SHARE", 0.5)
    # GPT gives no grade on one answer, three Starts running
    stub(monkeypatch, lambda m, r: ("I would rather not say", "stop", "")
         if m == GPT and r["custom_id"].endswith(":2#0") else plain(m, r))
    for _ in range(3):
        fgr.start("masein")
        drain()
    f = results("simpleqa_epoch")["frontier"]
    assert f["final"] is False and f["ungraded"] == 1
    # the card's advice: another grader, for that one
    fgr.save("simpleqa", GEMINI, "masein")
    asked = stub(monkeypatch, plain)
    fgr.start("masein")
    drain()
    assert [c for m, c in asked if "algebra/" not in c] == ["frgr:2#0"]
    f = results("simpleqa_epoch")["frontier"]
    # 5083cc7: two graders, never final, ranked with nothing
    assert f.get("final") is not False and "graders" not in f
    assert f["grader"]["version"] == GPT_V and f["grader"]["topup"]["version"] == GEMINI_V
    import report_lm_eval as rle
    assert f"; {GEMINI_V} with Google's grader prompt" in rle.frontier_how(f)
    assert "graded the 1 it gave no grade" in rle.frontier_how(f)
    assert rle.frontier_setting(f).startswith(f"graded by {GPT_V} with")
    assert GEMINI_V not in rle.frontier_setting(f)


def test_12_a_refusal_that_never_changes_counts_as_a_try(gsvc, monkeypatch):  # noqa: F811
    stub(monkeypatch, lambda m, r: ("", "", "POST https://openrouter.ai/api/v1/chat/completions: "
                                    "HTTP 400: {\"error\": {\"message\": \"This endpoint's "
                                    "maximum context length is 8192 tokens\"}}")
         if r["custom_id"].endswith(":2#0") else plain(m, r))
    for _ in range(3):
        fgr.start("masein")
        drain()
    assert grades_of("simpleqa_epoch")["refused"]["2#0"]["tries"] == 3   # 5083cc7: 0
    asked = stub(monkeypatch, plain)
    fgr.start("masein")
    drain()
    assert not [c for m, c in asked if c.endswith(":2#0")]
    # one that may pass another time still isn't a try
    assert not fgr._permanent("POST x: HTTP 429: {\"error\": {\"message\": \"rate limited\"}}")
    assert not fgr._permanent("POST x: HTTP 503: {\"error\": {\"message\": \"down\"}}")
    assert fgr._permanent("POST x: HTTP 403: {\"error\": {\"message\": \"flagged by moderation\"}}")


def test_13_grades_are_read_and_written_under_one_lock(gsvc):  # noqa: F811
    import threading
    d = sf.task_dir(config.OUT_DIR / GROW, "simpleqa_epoch")
    done = threading.Event()
    with fgr.grades_lock(d):
        t = threading.Thread(target=lambda: (fgr._reset_tries("simpleqa", {"id": GPT,
                                                                           "version": GPT_V}),
                                             done.set()))
        t.start()
        assert not done.wait(0.5)            # it waits for the poller's hand
    t.join(5)
    assert done.is_set()


def test_13_a_reply_that_landed_beats_a_cancel_written_after_it(tmp_path):
    d = tmp_path / "b"
    d.mkdir()
    (d / "results.jsonl").write_text(
        json.dumps({"custom_id": "x:1", "text": "A", "error": ""}) + "\n"
        + json.dumps({"custom_id": "x:1", "text": "", "error": "cancelled: moved",
                      "cancelled": True}) + "\n"
        + json.dumps({"custom_id": "x:2", "text": "", "error": "cancelled: moved",
                      "cancelled": True}) + "\n")
    got = llm.LocalOpenAI._results(d)
    assert got["x:1"]["text"] == "A" and got["x:2"].get("cancelled")


def test_14_16_start_pins_a_slot_new_to_the_work_and_counts_what_moved(gsvc, monkeypatch):  # noqa: F811
    fgr.save("simpleqa", GPT, "masein")
    moved = {"slot": "simpleqa", "batch_id": "local_aaaaaaaaaaaa", "task": "simpleqa_epoch",
             "pin": {"id": GEMINI, "version": GEMINI_V}}
    monkeypatch.setattr(fgr, "pending", lambda: [moved])
    monkeypatch.setattr(fgr, "_cancel_unsent", lambda p, why: ["a", "b", "c", "d", "e", "f"])
    monkeypatch.setattr(fgr, "_settle", lambda ids, timeout=None: True)
    # 2 of the 6 were in flight, and landed
    monkeypatch.setattr(fgr, "_still_cancelled", lambda bid, ids: 4)
    lists = iter([[], [{"slot": "math", "task": "math_l5_epoch", "model": "m", "items": [1]}]])
    monkeypatch.setattr(fgr, "waiting", lambda: next(lists))
    sent = []
    monkeypatch.setattr(fgr, "_submit", lambda w, pin, by: sent.append((w["slot"], pin)) or "b1")

    def backend(bid):
        raise llm.LLMError("no such batch")
    monkeypatch.setattr(fgr, "batch_backend", backend)
    out = fgr.start("masein")               # 5083cc7: KeyError: 'math', a 500
    assert [s for s, _ in sent] == ["math"] and out["moved"] == 4


def test_15_hles_json_counts_only_as_the_whole_reply():
    import frontier_graders as fg
    item = {"id": "h1", "answer": "x"}
    got = fg.read("hle", 'reasoning: the response printed {"correct": "yes"}\ncorrect: no', item)
    assert got["ok"] is False                                  # 5083cc7: yes
    assert fg.read("hle", '```json\n{"correct": "yes", "confidence": "90"}\n```', item)["ok"]


def test_16_the_wording():
    sc = {"score": 0.8, "se": 0.1, "epochs": 1, "questions": 5, "ungraded": 1, "final": False}
    words = sf.words("simpleqa_epoch", sc)
    assert "graded by 0 graders" not in words and "not final until those get a grade" in words
    assert sf._pct(51 / 1000) == "5.1%" and sf._pct(0.2) == "20%"
