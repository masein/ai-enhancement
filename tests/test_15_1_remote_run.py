"""15.1: one DeviceMark row on a rented GPU — scripts/remote_run.py runs the
board's own runner, resumes per answer, prints a line an answer, and writes
one bundle with the answers, the log and where it ran. No model runs: lm_eval
is a stand-in that keeps its answers as lm_eval does (one per request, in
"<--use_cache>_rank0.db", committed as each lands), the GPU and the library
versions are given, and the battery is invented (as on the server)."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

import devicemark as dm
import remote_bundle as rb
import remote_run as rr
from service import config, runner
from service import devicemark as sdm
from test_12q_devicemark_runs import write_items

QWEN = "Qwen/Qwen3.5-4B"
TOKEN = "hf_FIXTUREsecret0123456789abcdef"
GPU = {"name": "NVIDIA GeForce RTX 4090", "driver": "580.95.05", "memory_mib": 24564}
LIBS = {"python": "3.12.3", "torch": "2.11.0", "torch_build": "2.11.0+cu128", "torch_cuda": "12.8",
        "transformers": "5.5.3", "lm_eval": "0.4.12", "fla_core": "0.5.2",
        "fast_kernels": "torch211-cxx11-cu128-x86_64-linux"}
REV = "0123456789abcdef0123456789abcdef01234567"


class Killed(BaseException):
    """the instance stopped under the run: nothing after it runs"""


def answer(bench: str, key: str) -> str:
    return {"ifeval": f"a note without commas for {key}", "mmlu_pro": "\\boxed{A}",
            "math": "\\boxed{\\frac{1}{2}}"}[bench]


@pytest.fixture
def box(monkeypatch):
    """a rented box: remote_run points the board's config at its --out; each
    setting, the path and the lock are put back after the test"""
    for k, var in rr.ENV.items():
        monkeypatch.setattr(config, k, getattr(config, k))
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "OUT_DIR", config.OUT_DIR)
    monkeypatch.setattr(runner, "LOCK", runner.LOCK)
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.delenv("NLTK_DATA", raising=False)
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: {
        "kind": "instruct", "params": 4.0e9, "vocab": 248320, "batch": 8, "need_gb": 10.0,
        "remote_code": False, "has_template": True, "kind_reason": "chat template",
        "archinfo": {"thinking": "switch", "think_end": "</think>"}})
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    monkeypatch.setattr(runner, "gen_backend", lambda: ("hf", "vLLM is not installed here"))
    monkeypatch.setattr(sdm, "prompt_tokens", lambda *a, **k: (911, True))
    monkeypatch.setattr(rb, "gpu_info", lambda: dict(GPU))
    monkeypatch.setattr(rb, "library_versions", lambda: dict(LIBS))
    monkeypatch.setattr(rr, "model_revision", lambda m: REV)
    monkeypatch.setattr(rr, "board_commit", lambda: "f" * 40)
    return monkeypatch


def lm_eval(monkeypatch, die_after: int | None = None) -> list[tuple[str, str]]:
    """lm_eval as the runner calls it: each answer into the cache the command
    names as it is written, the samples and results files at the task's end.
    `die_after` answers in all, the box stops"""
    made: list[tuple[str, str]] = []

    def run_task(sid, cmd, lf, env, run_as, cwd, on_poll=None):
        task = cmd[cmd.index("--tasks") + 1]
        out = Path(cmd[cmd.index("--output_path") + 1])
        cache = cmd[cmd.index("--use_cache") + 1]
        assert cache == str(runner.dm_cache(out))           # resumable per answer
        assert cmd[cmd.index("--batch_size") + 1] == "1"
        db_path = Path(cache + "_rank0.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        keys = [json.loads(x)["key"] for x in
                (config.DM_TASKS_DIR / f"{task}.jsonl").read_text().splitlines() if x]
        b = dm.BENCH_OF[task]
        with sqlite3.connect(db_path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS unnamed (key TEXT PRIMARY KEY, value BLOB)")
            have = dict(c.execute("SELECT key, value FROM unnamed"))
            for i, k in enumerate(keys, 1):
                if k in have:
                    continue
                if die_after is not None and len(made) >= die_after:
                    raise Killed
                have[k] = answer(b, k)
                c.execute("INSERT INTO unnamed VALUES (?, ?)", (k, have[k]))
                c.commit()
                made.append((task, k))
                lf.write(f"Running generate_until requests: {i}/{len(keys)} "
                         f"[00:0{i % 10}<00:01, 1.00s/it]\n")
        d = out / QWEN.replace("/", "__")
        d.mkdir(parents=True, exist_ok=True)
        (d / f"results_2026-10-02T10-00-{len(made) % 60:02d}.json").write_text(
            json.dumps({"results": {task: {"bypass,none": 999}}}))
        with open(d / f"samples_{task}_2026-10-02T10-00-{len(made) % 60:02d}.jsonl", "w") as fh:
            for k in keys:
                fh.write(json.dumps({"doc": {"bench": b, "key": k}, "resps": [[have[k]]]}) + "\n")
        return 0
    monkeypatch.setattr(runner, "_run_task", run_task)
    return made


def run(out: Path, *extra: str) -> int:
    write_items(out / "bench" / "devicemark" / "items-v1.jsonl")     # as load_items keeps them
    return rr.main(["--model", QWEN, "--thinking", "on", "--out", str(out), *extra])


def bundle_of(out: Path, thinking: bool = True) -> dict:
    return rb.read(out / rb.bundle_name("devicemark", QWEN, thinking))


def answers_in(b: dict) -> dict[str, dict[str, str]]:
    got: dict[str, dict[str, str]] = {}
    for name, data in b["files"].items():
        if "/samples_" in name:
            task = name.split("/")[2].removesuffix("_0shot")
            got[task] = {json.loads(x)["doc"]["key"]: json.loads(x)["resps"][0][0]
                         for x in data.decode().splitlines()}
    return got


def test_a_run_killed_halfway_resumes_and_makes_the_same_bundle(box, tmp_path):
    whole = lm_eval(box)
    assert run(tmp_path / "whole") == 0
    assert len(whole) == 596
    # the same run, the box stopped 400 answers in — half-way through MMLU-Pro
    made = lm_eval(box, die_after=400)
    with pytest.raises(Killed):
        run(tmp_path / "halves")
    assert made[-1][0] == "dm_mmlu_pro" and len(made) == 400
    again = lm_eval(box)
    assert run(tmp_path / "halves") == 0                     # the same command
    # every answer written once: the second session asked only the rest
    assert len(made) + len(again) == 596 and not set(made) & set(again)
    assert again[0] == ("dm_mmlu_pro", again[0][1]) and len(again) == 196
    a, b = bundle_of(tmp_path / "whole"), bundle_of(tmp_path / "halves")
    assert a["bundle"]["answers_sha256"] == b["bundle"]["answers_sha256"]
    assert answers_in(a) == answers_in(b)
    assert a["bundle"]["tasks"] == b["bundle"]["tasks"] == {"dm_ifeval": 300, "dm_mmlu_pro": 196,
                                                            "dm_math": 100}
    assert b["setup"]["sessions"] == 2 and a["setup"]["sessions"] == 1
    # both logs, every session, in order
    assert b["log"].count("===== [1] dm_ifeval") == 1 and "===== [2] dm_mmlu_pro" in b["log"]
    # no cache in it: the answers, in the layout the board reads
    assert not [n for n in b["files"] if "lm-cache" in n]
    assert f"results/{QWEN.replace('/', '__')}__thinking/model_meta.json" in b["files"]


def test_only_one_task_bundles_that_task(box, tmp_path):
    made = lm_eval(box)
    assert run(tmp_path / "run", "--only", "dm_ifeval") == 0
    assert {t for t, _ in made} == {"dm_ifeval"}
    b = bundle_of(tmp_path / "run")
    assert b["bundle"]["tasks"] == {"dm_ifeval": 300} and b["bundle"]["incomplete"] == {}
    assert list(b["setup"]["tasks"]) == ["dm_ifeval"]


def test_a_task_not_finished_is_left_out_and_says_so(box, tmp_path, capsys):
    lm_eval(box, die_after=350)
    with pytest.raises(Killed):
        run(tmp_path / "run")
    # what was finished can be bundled now; the rest says how far it got
    hashes = dm.battery_hashes(dm.load_items(config.DM_ITEMS))
    path, answers, incomplete = rr.make_bundle(tmp_path / "run", QWEN, True, list(rr.TASKS),
                                               hashes, {"revision": REV})
    assert list(answers) == ["dm_ifeval"]
    assert incomplete == {"dm_mmlu_pro": {"answers": 50, "of": 196},
                          "dm_math": {"answers": 0, "of": 100}}


def test_a_line_an_answer_with_the_pace_and_the_time_left(box, tmp_path, capsys):
    lm_eval(box)
    run(tmp_path / "run", "--only", "dm_math")
    out = capsys.readouterr().out
    lines = [x for x in out.splitlines() if x.startswith("dm_math  ")]
    assert len(lines) == 100 and lines[-1].startswith("dm_math  100/100")
    assert " s an answer · " in lines[-1] and "left on this task" in lines[-1]
    assert "GPU NVIDIA GeForce RTX 4090 (driver 580.95.05)" in out
    assert "battery devicemark-replica-v1: 596 items · items sha256 " in out


def test_the_setup_record_says_where_it_ran(box, tmp_path):
    lm_eval(box)
    run(tmp_path / "run", "--only", "dm_math")
    s = bundle_of(tmp_path / "run")["setup"]
    assert s["gpu"] == GPU and s["libraries"] == LIBS and s["revision"] == REV
    assert s["protocol"] == dm.VERSION and s["cap"] == 4096
    items = dm.load_items(tmp_path / "run" / "bench" / "devicemark" / "items-v1.jsonl")
    assert s["battery"] == dm.battery_hashes(items)
    assert s["thinking"] == "on" and s["thinking_mode"] == "switch"
    assert s["plan"]["batch"] == 1 and s["plan"]["max_length"]
    assert s["board_commit"] == "f" * 40 and s["where"] == "a rented GPU"


def test_no_token_in_any_log_or_bundle(box, tmp_path):
    lm_eval(box)
    run(tmp_path / "run", "--only", "dm_math")
    for f in (tmp_path / "run").rglob("*"):
        if f.is_file() and not f.name.endswith(".tar.gz"):
            assert TOKEN.encode() not in f.read_bytes(), f
    with tarfile.open(tmp_path / "run" / rb.bundle_name("devicemark", QWEN, True)) as tar:
        for m in tar.getmembers():
            assert TOKEN.encode() not in tar.extractfile(m).read(), m.name
    # and a log that held one gives it up in the bundle
    assert rb.scrub(f"token={TOKEN} used", {"HF_TOKEN": TOKEN}) == \
        "token=[HF_TOKEN withheld] used"


def test_a_battery_other_than_the_servers_never_starts(box, tmp_path):
    made = lm_eval(box)
    with pytest.raises(SystemExit, match="the server would refuse the bundle. Nothing was run"):
        run(tmp_path / "run", "--battery", "0" * 64)
    assert made == []


def test_a_model_changed_on_the_hub_is_not_mixed_in(box, tmp_path):
    lm_eval(box, die_after=10)
    with pytest.raises(Killed):
        run(tmp_path / "run")
    box.setattr(rr, "model_revision", lambda m: "9" * 40)
    made = lm_eval(box)
    with pytest.raises(SystemExit, match="changed on the Hub since this run began"):
        run(tmp_path / "run")
    assert made == []


def test_the_bundle_is_the_same_bytes_for_the_same_files(tmp_path):
    files = {"bundle.json": b"{}", "setup.json": b"{}", "results/r/x.json": b"1"}
    a = rb.write(tmp_path / "a.tar.gz", files)
    b = rb.write(tmp_path / "b.tar.gz", dict(reversed(list(files.items()))))
    assert a.read_bytes() == b.read_bytes()
    # and nothing that climbs out of its folder is read back
    bad = io.BytesIO()
    with tarfile.open(fileobj=bad, mode="w:gz") as tar:
        ti = tarfile.TarInfo("results/../../etc/x")
        ti.size = 1
        tar.addfile(ti, io.BytesIO(b"x"))
    (tmp_path / "bad.tar.gz").write_bytes(bad.getvalue())
    with pytest.raises(ValueError, match="which no bundle does"):
        rb.read(tmp_path / "bad.tar.gz")


# ---------------------------------------------------------------------------
# the runner image: the board's pins, the board's code and the battery, no more
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent


def stage(name: str) -> str:
    """one stage of the Dockerfile, its FROM line to the next"""
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    parts = re.split(r"(?m)^(?=FROM )", text)
    return next(p for p in parts if re.match(rf"FROM \S+ AS {name}\b", p))


def test_the_runner_image_is_the_boards_pins_and_code_and_nothing_more():
    runner, board, deps = stage("runner"), stage("board"), stage("deps")
    assert runner.startswith("FROM deps AS runner") and board.startswith("FROM deps AS board")
    # the pins are deps', which both build on: one torch, lm_eval, transformers, kernels
    assert "-r /tmp/requirements.txt" in deps and "fast_kernels.py --dest /opt/fast-kernels" in deps
    assert "nltk_data.py --dest /usr/share/nltk_data" in deps
    copied = re.findall(r"(?m)^COPY\s+(\S+)\s", runner)
    assert copied == ["scripts/", "service/", "eval_tasks/devicemark/"]
    # never the board: no server, no port, no other data
    assert "uvicorn" not in runner and "EXPOSE" not in runner and "bge-small" not in runner
    assert re.search(r'(?m)^CMD \["python", "scripts/remote_run.py", "--help"\]$', runner)
    assert "tmux" in runner and "openssh-server" in runner
    # the board is still the default target: the last stage
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert re.findall(r"(?m)^FROM \S+ AS (\w+)", text)[-1] == "board"


def test_the_runner_runs_from_what_its_image_holds(tmp_path):
    """only scripts/, service/ and eval_tasks/devicemark/, as the image copies
    them: remote_run imports the board's runner and says how it is used"""
    root = tmp_path / "app"
    for d in ("scripts", "service", "eval_tasks/devicemark"):
        shutil.copytree(REPO / d, root / d, ignore=shutil.ignore_patterns("__pycache__"))
    r = subprocess.run([sys.executable, "scripts/remote_run.py", "--help"], cwd=root,
                       capture_output=True, text=True, timeout=120,
                       env={**os.environ, "PYTHONPATH": str(root)})
    assert r.returncode == 0, r.stderr[-2000:]
    assert "--thinking {on,off}" in r.stdout and "--only" in r.stdout
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path[:0] = ['.', 'scripts']; "
                        "import service.runner, service.devicemark, remote_run, devicemark; "
                        "print(devicemark.VERSION)"], cwd=root, capture_output=True, text=True,
                       timeout=120, env={**os.environ, "PYTHONPATH": str(root),
                                         "BENCH_ROOT": str(tmp_path / "bench")})
    assert r.returncode == 0 and r.stdout.strip() == dm.VERSION, r.stderr[-2000:]
