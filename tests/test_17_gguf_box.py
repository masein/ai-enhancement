"""17, stage 1: a GGUF on a rented box — scripts/remote_gguf.py starts
llama-server from a tarball (here a stand-in, tests/fixtures/
fake_llama_server.py), asks the Frontier benchmarks through the board's own
runner, resumes answer by answer, takes --shard, and leaves one bundle; the
import (import_remote.py → import_frontier.py) puts its answers on the served
model's row, or its "· thinking" row, scored by code and marked as run on a
rented GPU, and refuses a bundle whose file isn't the registered one. And
the same suite from Test this model on the board. GPQA's questions are
invented here; nothing is fetched and no model runs."""

from __future__ import annotations

import hashlib
import io
import json
import os
import socket
import tarfile
from pathlib import Path

import pytest

import frontier as fb
import import_remote as ir
import remote_bundle as rb
import remote_gguf as rg
import report_lm_eval as report
from fake_openai import FakeServer
from service import config, db, runner, served
from service import frontier as sf
from test_12q_devicemark_runs import ME, svc  # noqa: F401 — svc is the fixture

FIX = Path(__file__).resolve().parent / "fixtures"
TASK = "gpqa_diamond_epoch"
SERVED = "served/lda-box"            # its name says nothing of Qwen: --based-on and the file do
ROW = "served__lda-box__thinking"
GGUF_NAME = "Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf"
GPU = {"name": "NVIDIA GeForce RTX 5090", "driver": "580.65", "memory_mib": 32607}
SECRET = "hf_FIXTUREsecret17box0123456789"   # invented
N, HARD = 12, (3, 7, 11)                     # the fake answers the "hard" ones wrong
RUNS = fb.BENCH[TASK]["epochs"]


def invented() -> list[dict]:
    return [{"id": f"rec{k:03d}",
             "question": f"Invented question {k}{' (hard)' if k in HARD else ''}?",
             "right": f"right answer {k}",
             "wrong": [f"wrong answer {k}a", f"wrong answer {k}b", f"wrong answer {k}c"]}
            for k in range(N)]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def box(svc, tmp_path, monkeypatch):  # noqa: F811
    """the server, and what a rented box is given: a GGUF, a llama-server
    tarball, the GPU's name — and the invented questions on both"""
    monkeypatch.setattr(fb, "_fetch", lambda task: invented())
    monkeypatch.setattr(rb, "gpu_info", lambda: dict(GPU))
    monkeypatch.setattr(rg, "board_commit", lambda: "f" * 40)
    gguf = tmp_path / "models" / GGUF_NAME
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF" + bytes(range(256)) * 64)
    tb = tmp_path / "llama-server-cuda12.8.tar.gz"
    with tarfile.open(tb, "w:gz") as tar:
        def add(name: str, data: bytes, mode: int = 0o644) -> None:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode = len(data), mode
            tar.addfile(ti, io.BytesIO(data))
        add("llama/bin/llama-server", (FIX / "fake_llama_server.py").read_bytes(), 0o755)
        add("llama/lib/libggml-cuda.so", b"\x7fELF not really")
        add("llama/VERSION", b"commit abc1234\ncuda_archs 80;86;89;90;120\n")
    log = tmp_path / "requests.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_LOG", str(log))
    monkeypatch.setenv("HF_TOKEN", SECRET)
    monkeypatch.setenv("SERVED_RETRY_S", "1")
    return {"gguf": gguf, "tarball": tb, "log": log, "root": tmp_path,
            "sha": hashlib.sha256(gguf.read_bytes()).hexdigest()}


def run_box(b: dict, out: str, *more: str, thinking: str = "on") -> int:
    """remote_gguf.py on the "box" — the server's config put back after"""
    keep = {k: getattr(config, k) for k in [*rg.ENV, "OUT_DIR"]}
    lock = runner.LOCK
    env = {v: os.environ.get(v) for v in rg.ENV.values()}
    try:
        return rg.main(["--as", SERVED, "--gguf", str(b["gguf"]), "--server", str(b["tarball"]),
                        "--based-on", "Qwen/Qwen3.6-35B-A3B", "--thinking", thinking,
                        "--slots", "3", "--port", str(free_port()), "--load-timeout", "60",
                        "--flags", "--flash-attn on", "--env", "LLAMA_MOE_ROUTE_MODE=lookahead",
                        "--out", str(b["root"] / out), *more])
    finally:
        for k, v in keep.items():
            setattr(config, k, v)
        runner.LOCK = lock
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def bundle_of(b: dict, out: str, shard: tuple[int, int] | None = None,
              thinking: bool = True) -> Path:
    return b["root"] / out / rb.bundle_name("frontier", SERVED, thinking, shard)


def record(sha: str | None) -> dict:
    """the served model as the board keeps it, its file hashed when it was registered"""
    rec = {"id": SERVED, "name": "lda box", "base_url": "http://127.0.0.1:9/v1", "key": "",
           "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "build-lda, --cpu-moe", "thinking": "auto",
           "phone": False, "gguf_path": "", "gguf_flags": "", "gguf_setups": [],
           "pin": {"model": GGUF_NAME, "file": GGUF_NAME, "size": 21_000_000_000, "ctx": 16384,
                   "build": "b6500"}, "answered": ["models"], "by": ME, "at": 0}
    if sha:
        rec.update(gguf_path=f"/models/{GGUF_NAME}",
                   gguf_pin={"sha256": sha, "size": 16388, "name": GGUF_NAME})
    return rec


def register(sha: str | None) -> dict:
    rec = record(sha)
    db.served_put(rec)
    return rec


def requests(b: dict) -> list[dict]:
    return [json.loads(x) for x in b["log"].read_text().splitlines()] if b["log"].exists() else []


def expected() -> float:
    return (N - len(HARD)) / N


# ---------------------------------------------------------------------------
# the scorer, on fixed answers
# ---------------------------------------------------------------------------

def test_gpqa_prompt_and_choice_parse():
    it = invented()[0]
    text, need = fb.prompt(TASK, it)
    # Epoch's template word for word, the four choices shuffled once, seeded
    assert text.startswith("Answer the following multiple choice question. The last line of "
                           "your response should be of the following format: 'ANSWER: LETTER' "
                           "(without quotes) where LETTER is one of ABCD. Think step by step "
                           "before answering.\n\nInvented question 0?\n\nA) ")
    assert fb.prompt(TASK, it) == (text, need)
    assert f"{need['key']}) right answer 0" in text
    read = fb.read_choice
    assert read("Reasoning.\nANSWER: B") == "B"
    assert read("ANSWER: (c)") == "C"
    assert read("the answer is B.\nANSWER: A, B") is None            # two letters: none
    assert read("no letter given") is None
    assert read("ANSWER: E") is None                                # not a choice
    assert read("ANSWER: **B**") == "B" and read("ANSWER: $D$") == "D" and read("ANSWER: B.") == "B"
    # Inspect's order: the last line that is "ANSWER: X", before one inside a line;
    # and the thinking's never counts
    assert read("ANSWER: A\nANSWER: C\nchecked: ANSWER: D") == "C"
    assert read("I think ANSWER: A. Then again ANSWER: D") == "D"
    assert read("I think ANSWER: A, no, ANSWER: D") is None         # as Inspect reads it
    assert read("<think>ANSWER: A</think>\nANSWER: C") == "C"
    assert read("<think>so ANSWER: A</think>\nI won't say.") is None
    assert fb.score(TASK, "ANSWER: " + need["key"], "stop", need) == {
        "ok": True, "read": need["key"], "ran_out": False}
    # an answer cut at its budget, or whose thinking never closed, ran out: wrong
    assert fb.score(TASK, "ANSWER: " + need["key"], "length", need)["ok"] is False
    assert fb.score(TASK, "<think>\nstill thinking ANSWER: " + need["key"], "stop",
                    need)["ran_out"] is True
    # the error is over questions: a question's runs move together
    sm = fb.summary({"a": [1, 1], "b": [0, 0], "c": [1, 0]})
    assert sm["score"] == pytest.approx(0.5) and sm["se"] == pytest.approx(0.2887, abs=1e-4)


def test_shards_split_and_settings():
    assert fb.parse_shard("2/3") == (2, 3) and fb.parse_shard("") is None
    for bad in ("0/2", "3/2", "1/1", "x"):
        with pytest.raises(ValueError):
            fb.parse_shard(bad)
    xs = list(range(10))
    assert sorted(fb.shard_of(xs, (1, 3)) + fb.shard_of(xs, (2, 3)) + fb.shard_of(xs, (3, 3))) == xs
    rec = record(None)
    on, off = sf.settings(rec, TASK, True), sf.settings(rec, TASK, False)
    # the model card's sampling for each setting, the thinking switch said out loud
    assert on == {"max_tokens": 32768, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                  "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": True}}
    assert off["max_tokens"] == 4096 and off["temperature"] == 0.7 and \
        off["chat_template_kwargs"] == {"enable_thinking": False}
    assert fb.slot_context([TASK], True) == 32768 + 2048
    # the Frontier suite isn't lm_eval's: deploy step 4 doesn't look for it
    assert "frontier" in config.SUITES and "frontier" in config.NOT_LM_EVAL
    assert "frontier" in served.SUITES


# ---------------------------------------------------------------------------
# a GGUF on a rented box, and back
# ---------------------------------------------------------------------------

def test_gguf_run_bundle_and_import(box):
    assert run_box(box, "run") == 0
    path = bundle_of(box, "run")
    b = rb.read(path)
    setup, bundle = b["setup"], b["bundle"]
    # what it records: the file's sha256, the build, the launch, the GPU
    assert setup["gguf"] == {"name": GGUF_NAME, "sha256": box["sha"],
                             "size": box["gguf"].stat().st_size, "source": "a path on the box"}
    srv = setup["server"]
    assert (srv["build"], srv["commit"]) == (6543, "abc1234")
    assert srv["flags"] == ["--flash-attn", "on"]
    assert srv["env"] == {"LLAMA_MOE_ROUTE_MODE": "lookahead"}
    assert srv["argv"][:2] == ["llama-server", "-m"] and "--jinja" in srv["argv"]
    assert srv["argv"][srv["argv"].index("-c") + 1] == str(3 * (32768 + 2048))
    assert srv["binary_sha256"] and srv["tarball_sha256"] == rb.sha256_file(box["tarball"])
    assert srv["chat_template_sha256"] == hashlib.sha256(b"{# a fake template #}").hexdigest()
    assert setup["gpu"] == GPU and setup["where"] == "a rented GPU"
    assert setup["tasks"][TASK]["budget"] == 32768 and setup["tasks"][TASK]["family"] == "qwen3.6"
    assert bundle["format"] == 2 and bundle["row"] == ROW and bundle["tasks"] == {TASK: N * RUNS}
    # asked as the board asks: the card's sampling, the switch, a seed a question and run
    reqs = requests(box)
    assert len(reqs) == N * RUNS
    assert all(r["temperature"] == 1.0 and r["top_k"] == 20 and r["presence_penalty"] == 1.5
               and r["max_tokens"] == 32768 and r["chat_template_kwargs"] == {
                   "enable_thinking": True} for r in reqs)
    assert len({r["seed"] for r in reqs}) == N * RUNS
    # no key and no model file in it, anywhere
    assert not any(n.endswith(".gguf") or n.endswith(".so") for n in b["files"])
    assert not any(SECRET.encode() in data for data in b["files"].values())
    assert not any(b"GGUF" + bytes(range(16)) in data for data in b["files"].values())

    register(box["sha"])
    said: list[str] = []
    assert ir.import_bundle(path, "masein", said.append) == 0
    row = config.OUT_DIR / ROW
    res = json.loads(next(sf.task_dir(row, TASK).glob("results_*.json")).read_text())
    assert res["results"][TASK]["acc,none"] == pytest.approx(expected())
    assert res["frontier"]["where"] == "run on a rented GPU (NVIDIA GeForce RTX 5090)"
    assert res["frontier"]["epochs"] == RUNS and res["frontier"]["ran_out"] == 0
    # the board reads it as the model's "· thinking" row — a Frontier benchmark,
    # never in an average
    runs = [r for r in report.load_results(config.OUT_DIR) if r["model"] == SERVED + " · thinking"]
    assert runs and runs[0]["tasks"][TASK]["acc"]["value"] == pytest.approx(expected())
    assert TASK in report.FRONTIER_TASKS
    # the Runs list has the import, with the bundle's log
    imp = [r for r in db.recent(10) if r["suite"] == "frontier"]
    assert imp and imp[0]["status"] == "done" and "rented GPU" in imp[0]["note"]
    assert any("GPQA Diamond 75.0%" in x for x in said)
    # imported twice: nothing changes
    said.clear()
    assert ir.import_bundle(path, "masein", said.append) == 0
    assert any("imported already" in x for x in said)
    # a whole run's answers sent as a shard: answers outside it are refused
    b["bundle"]["shard"] = {"i": 1, "n": 2}
    import import_frontier as imf
    assert any("outside its shard" in x for x in imf.checks(b, served.get(SERVED)))


def test_resume_after_a_stop(box, monkeypatch):
    # the box stops after 7 answers: the run fails, what it answered is kept, no bundle
    monkeypatch.setenv("FAKE_LLAMA_DIE_AFTER", "7")
    assert run_box(box, "run") == 1
    first = len(requests(box))
    answers = box["root"] / "run" / "bench" / "results" / "full" / ROW / f"{TASK}_0shot" / \
        "frontier" / "answers.jsonl"
    kept = len(answers.read_text().splitlines())
    assert 7 <= kept < N * RUNS
    assert not bundle_of(box, "run").exists()
    # the same command again: only the rest is asked
    monkeypatch.delenv("FAKE_LLAMA_DIE_AFTER")
    assert run_box(box, "run") == 0
    asked_again = requests(box)[first:]
    assert len(asked_again) == N * RUNS - kept
    b = rb.read(bundle_of(box, "run"))
    assert b["setup"]["sessions"] == 2 and b["bundle"]["tasks"] == {TASK: N * RUNS}
    # another file in the same folder can't be mixed in
    box["gguf"].write_bytes(b"GGUF another file")
    with pytest.raises(SystemExit, match="another file"):
        run_box(box, "run")


def test_two_shards_add_up(box):
    assert run_box(box, "s1", "--shard", "1/2") == 0
    assert run_box(box, "s2", "--shard", "2/2") == 0
    register(box["sha"])
    said: list[str] = []
    assert ir.import_bundle(bundle_of(box, "s2", (2, 2)), "masein", said.append) == 0
    assert any("shard 1 of 2 missing" in x for x in said)
    assert not list(sf.task_dir(config.OUT_DIR / ROW, TASK).glob("results_*.json"))
    said.clear()
    assert ir.import_bundle(bundle_of(box, "s1", (1, 2)), "masein", said.append) == 0
    assert any("every shard is in (2 of 2)" in x for x in said)
    got = sf.read_answers(sf.task_dir(config.OUT_DIR / ROW, TASK) / sf.ANSWERS)
    assert set(got) == {(it["id"], e) for it in invented() for e in range(RUNS)}
    res = json.loads(next(sf.task_dir(config.OUT_DIR / ROW, TASK).glob("results_*.json"))
                     .read_text())
    assert res["results"][TASK]["acc,none"] == pytest.approx(expected())
    assert res["frontier"]["where"] == "run on a rented GPU (NVIDIA GeForce RTX 5090)"


def test_another_file_is_refused(box):
    assert run_box(box, "run") == 0
    path = bundle_of(box, "run")
    said: list[str] = []
    # the board's file is another one: refused, nothing on the row
    register("0" * 64)
    assert ir.import_bundle(path, "masein", said.append) == ir.REFUSED
    assert any(f"its GGUF isn't the file registered for {SERVED}" in x for x in said)
    assert not (config.OUT_DIR / ROW / f"{TASK}_0shot").exists()
    # no file registered: refused, unless the sha256 of the file its server serves is given
    register(None)
    said.clear()
    assert ir.import_bundle(path, "masein", said.append) == ir.REFUSED
    assert any("no file registered with its sha256" in x for x in said)
    said.clear()
    assert ir.import_bundle(path, "masein", said.append, file_sha="1" * 64) == ir.REFUSED
    said.clear()
    assert ir.import_bundle(path, "masein", said.append, file_sha=box["sha"]) == 0
    assert served.get(SERVED)["file_sha256"]["sha256"] == box["sha"]
    assert served.get(SERVED)["file_sha256"]["by"] == "masein"
    # a model that isn't registered at all
    b = rb.read(path)
    b["bundle"]["model"] = "served/nobody"
    import import_frontier as imf
    assert "isn't registered" in imf.checks(b, None)[0]


def test_asked_another_way_is_refused(box):
    assert run_box(box, "run") == 0
    rec = register(box["sha"])
    rec["based_on"], rec["pin"]["file"], rec["pin"]["model"] = "", "model.gguf", "model.gguf"
    db.served_put(rec)
    said: list[str] = []
    # the board would ask this model with the server's defaults, not qwen3.6's card
    assert ir.import_bundle(bundle_of(box, "run"), "masein", said.append) == ir.REFUSED
    assert any("sampling" in x and "--based-on" in x for x in said)


def test_no_key_and_no_model_file_in_any_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    with pytest.raises(ValueError, match="model file or a binary"):
        rb.write(tmp_path / "x.tar.gz", {"results/r/model.gguf": b"GGUF", "bundle.json": b"{}"})
    with pytest.raises(ValueError, match="holds HF_TOKEN"):
        rb.write(tmp_path / "x.tar.gz", {"run.log": f"token {SECRET}".encode(),
                                         "bundle.json": b"{}"})
    # and a bundle made elsewhere that holds one is refused as it is read
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as tar:
        for name, data in (("bundle.json", b"{}"), ("setup.json", b"{}"),
                           ("results/r/t_0shot/model.gguf", b"GGUF")):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    with pytest.raises(ValueError, match="which no bundle does"):
        rb.read(bad)
    # the box never passes a key to llama-server either
    env = rg.server_env({"libs": ["/x/lib"]}, {"A": "1"})
    assert "HF_TOKEN" not in env and env["LD_LIBRARY_PATH"].startswith("/x/lib")
    assert rg.own_flags(["--port", "1", "-ngl", "99", "--api-key=x"]) == ["--port", "--api-key=x"]


# ---------------------------------------------------------------------------
# the same suite from Test this model, on the board
# ---------------------------------------------------------------------------

def test_served_thinking_run_from_the_board(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", lambda task: invented())
    fake = FakeServer()
    try:
        def answer(body):
            text = body["messages"][-1]["content"]
            letter = next(ln[0] for ln in text.splitlines() if ln[1:3] == ") "
                          and ln[3:].startswith("right"))
            return f"Checked.\nANSWER: {letter}"
        fake.reply = answer
        fake.reasoning = "thinking it through"
        rec = served.register({"name": "lda box", "base_url": fake.base, "how": "llama-server",
                               "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "off"}, ME)
        # registered thinking off, asked with thinking on: the run says so itself
        r = svc.post("/api/submissions", json={"hf_id": rec["id"], "suite": "frontier",
                                          "thinking": True, "submitter": ME})
        assert r.status_code == 200, r.text
        sid = r.json()["id"]
        # a Hugging Face model can't sit it yet; a model from OpenRouter neither
        r2 = svc.post("/api/submissions", json={"hf_id": "Qwen/Qwen3.5-4B", "suite": "frontier",
                                           "submitter": ME})
        assert r2.status_code == 422 and "running on a server" in r2.json()["detail"]
        runner.run_submission(db.get(sid))
        row = db.get(sid)
        assert row["status"] == "done", row
        assert "GPQA Diamond 100.0%" in row["progress"]
        assert all(b["chat_template_kwargs"] == {"enable_thinking": True}
                   and b["max_tokens"] == 32768 for b in fake.requests)
        out = config.OUT_DIR / (rec["id"].replace("/", "__") + "__thinking")
        res = next(sf.task_dir(out, TASK).glob("results_*.json"))
        blob = json.loads(res.read_text())
        assert blob["config"]["model_args"].endswith("enable_thinking=True")
        assert blob["frontier"]["where"] == "this server"
        # the row of its own, and nothing on the thinking-off row
        assert not (config.OUT_DIR / rec["id"].replace("/", "__") / f"{TASK}_0shot").exists()
        # GPQA's questions are never written beside its scores: ids only
        for f in sf.task_dir(out, TASK).glob("*"):
            if f.name != sf.ANSWERS:
                assert "Invented question" not in f.read_text()
        # asked again: every answer is kept, nothing is asked twice
        n = len(fake.requests)
        sid2 = db.add(rec["id"], "instruct", "frontier", ME, "", thinking=True)
        runner.run_submission(db.get(sid2))
        assert len(fake.requests) == n and db.get(sid2)["status"] == "done"
    finally:
        fake.close()


def test_the_docs_command_is_the_scripts_and_fetches_its_bundle():
    """docs/REMOTE-RUNS.md § G2–G4: the box's command parses as remote_gguf.py
    reads it, and the server fetches and imports the bundle it writes"""
    import shlex
    doc = (Path(__file__).resolve().parents[1] / "docs" / "REMOTE-RUNS.md").read_text()
    lines = doc.split("## G2.", 1)[1].split("```bash", 1)[1].split("```", 1)[0]
    cmd = shlex.split(" ".join(x.rstrip("\\").strip() for x in lines.splitlines()
                               if x.strip() and not x.startswith(("tmux", "cd ", "read "))))
    assert cmd[:2] == ["python", "scripts/remote_gguf.py"]
    seen = {}
    real = rg.argparse.ArgumentParser.parse_args

    def keep(self, argv=None, namespace=None):
        a = real(self, argv, namespace)
        seen["a"] = a
        raise SystemExit(0)
    import unittest.mock as um
    with um.patch.object(rg.argparse.ArgumentParser, "parse_args", keep), \
            pytest.raises(SystemExit):
        rg.main(cmd[2:])
    a = seen["a"]
    assert a.thinking == "on" and a.gguf.startswith("hf://") and a.server.startswith("hf://")
    name = rb.bundle_name("frontier", a.served_as, True)
    assert f"{a.out}/{name}".replace(a.served_as.replace("/", "__"), "served__<name>") in doc
