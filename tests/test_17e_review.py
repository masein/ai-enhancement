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
        # within 5% of what the pilot measured, and never below it
        assert 0 <= fit["used"] / mib - measured < 0.05 * measured, (ctx, fit["used"] / mib)
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
    assert "up to 3 slots fit with 1 GB spare — 8 run, as --slots-fit says" in own
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
        for steps in boxes.values():
            for th, tasks, shard, slots in steps:
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
    longest = {p: max(sum(fbx.hours(s) for s in st) for st in b.values())
               for p, b in fbx.PLANS.items()}
    assert longest["A"] <= 18 and longest["B"] <= 10.5
    assert len(fbx.PLANS["A"]) == 6 and len(fbx.PLANS["B"]) == 10
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
    assert one[one.index("--thinking") + 1] == "on" and one[one.index("--shard") + 1] == "1/2"
    assert one[one.index("--only") + 1] == HLE and one[one.index("--out") + 1] == "/workspace/A1-1"
    assert one[one.index("--flags") + 1] == "-ctk q8_0 -ctv q8_0 --flash-attn on"
    assert two[two.index("--thinking") + 1] == "off" and "--shard" not in two
    out = capsys.readouterr().out
    assert "A1, step 1: not whole (exit 1)" in out and "paste the same line again" in out
    assert "A1, step 2: whole — its bundle is in /workspace/A1-2" in out


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
    on_box = {"1.2.3.4": ["/workspace/A1-1/frontier-served__phone-thinking-on-hle-shard-1-of-2"
                          ".tar.gz", "/workspace/A1-2/frontier-served__orig-thinking-off-hle"
                          ".tar.gz"],
              "5.6.7.8": ["/workspace/x/frontier-served__gemma-thinking-on-gpqa.tar.gz"]}
    calls = []

    def run(cmd, cwd=None):
        calls.append(cmd)
        if cmd[0] == "ssh":
            host = cmd[-2].split("@")[1]
            if host == "9.9.9.9":
                return 2, "ls: cannot access '/workspace/*/frontier-*.tar.gz': No such file"
            return 0, "\n".join(on_box[host]) + "\n"
        if cmd[0] == "scp":
            for src in cmd[cmd.index("StrictHostKeyChecking=accept-new") + 1:-1]:
                name = Path(src.split(":", 1)[1]).name
                (Path(cmd[-1]) / name).write_bytes((there / name).read_bytes())
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
    out = capsys.readouterr().out.splitlines()
    assert code == 1
    assert out[0].startswith("1.2.3.4:4100: 2 bundles · ")
    assert out[1] == "5.6.7.8:4200: 1 bundle · frontier-served__gemma-thinking-on-gpqa.tar.gz"
    assert out[2] == "9.9.9.9:4300: no bundle there yet"
    assert ("frontier-served__gemma-thinking-on-gpqa.tar.gz: no --sha for served/gemma — import "
            "it by hand (G4, G6)") in out
    assert ("frontier-served__orig-thinking-off-hle.tar.gz: refused — the file isn't the one "
            "registered") in out
    assert ("frontier-served__phone-thinking-on-hle-shard-1-of-2.tar.gz: the row served/phone · "
            "thinking: HLE: shard 1 of 2 in") in out
    imports = [c for c in calls if c[0] == "sudo"]
    assert len(imports) == 2
    assert [c[c.index("--file-sha256") + 1] for c in imports] == ["cd" * 32, sha]
    ssh = next(c for c in calls if c[0] == "ssh")
    assert ssh[:5] == ["ssh", "-i", str(key), "-p", "4100"]


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
