"""23c: the card a rented box runs on, measured (scripts/card_check.py) — the
speed probe against the fake llama-server, its timings fixed there; the
sampler's sums over an invented nvidia-smi; the slow-card flag among a
build's boxes and its hours, worked by hand; a step that runs on when the
probe fails; the box line's --price into the bundle and the import; the
board's "Speed and energy" from what the imports kept, nothing for a row
without; and CAIS's last group said. No GPU, no model runs; nvidia-smi is
stood in for."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

import card_check as cc
import frontier as fb
import frontier_box as fbx
import frontier_fetch as ff
import import_frontier as imf
import import_remote as ir
import remote_bundle as rb
from service import beyond, config, report_stats, reports
from service import frontier as sf
from test_17_gguf_box import (ROW, SERVED, TASK, box, bundle_of, register,  # noqa: F401
                              run_box)
from test_12q_devicemark_runs import svc  # noqa: F401 — box's fixture

FIX = Path(__file__).resolve().parent / "fixtures"
HLE, GPQA = "hle_text_cais", "gpqa_diamond_epoch"


class Smi:
    """nvidia-smi, stood in for: one RTX 5090 at rest (its PCIe link down to
    1, as a card at rest lowers it) and at work when its link is read alone
    — the card check reads it while the eight are asked — and each of
    `readings` in turn for the sampler"""

    CARD = {"name": "NVIDIA GeForce RTX 5090", "memory.total": "32607", "memory.used": "2",
            "power.limit": "575.00", "power.default_limit": "575.00", "clocks.max.sm": "3090",
            "pcie.link.gen.current": "1", "pcie.link.gen.max": "5",
            "pcie.link.width.current": "16", "pcie.link.width.max": "16",
            "driver_version": "580.65.06"}

    def __init__(self, readings: list[str] | None = None, card: dict | None = None,
                 reasons: str | None = "clocks_throttle_reasons.active"):
        self.readings, self.card, self.reasons = list(readings or []), {**self.CARD, **(card or {})}, reasons
        self.asked: list[list[str]] = []

    def __call__(self, cmd: list[str]) -> str:
        fields = cmd[1].split("=", 1)[1].split(",")
        self.asked.append(fields)
        if fields == ["pcie.link.gen.current", "pcie.link.width.current"]:
            return f"{self.card['pcie.link.gen.max']}, {self.card['pcie.link.width.max']}\n"
        if fields[0] == "power.draw":
            if fields[-1].startswith("clocks_") and fields[-1] != self.reasons:
                raise OSError(f'Field "{fields[-1]}" is not a valid field to query.')
            line = self.readings.pop(0)
            return line if fields[-1].startswith("clocks_") else \
                ", ".join(line.split(", ")[:5]) + "\n"
        return ", ".join(self.card[f] for f in fields) + "\n"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path, monkeypatch):
    """the fake llama-server, eight slots of 20,480 tokens"""
    procs = []

    def start(**env) -> str:
        port = free_port()
        p = subprocess.Popen([sys.executable, str(FIX / "fake_llama_server.py"), "-m",
                              str(tmp_path / "x.gguf"), "--port", str(port), "-c", str(8 * 20480),
                              "-np", "8"], env={**__import__("os").environ, **env},
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
        base = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                with urllib.request.urlopen(base + "/health", timeout=1):
                    return base
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("the fake server didn't start")
    yield start
    for p in procs:
        p.terminate()
        p.wait(10)


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------

def test_the_probe_as_worked_by_hand(server):
    base = server()
    smi = Smi()
    rec = cc.check(base, 20480, 8, run=smi)
    p = rec["probe"]
    # alone, three times: outputs 140, 150, 120 and prompts 9,000, 8,000, 10,000 — the medians
    assert [r["output_tok_s"] for r in p["alone"]["runs"]] == [140.0, 150.0, 120.0]
    assert (p["alone"]["output_tok_s"], p["alone"]["prompt_tok_s"]) == (140.0, 9000.0)
    assert all(r["prompt_n"] == 512 and r["out_n"] == 256 for r in p["alone"]["runs"])
    # 16,384 tokens alone, once
    assert p["long"] == {"prompt_n": 16384, "out_n": 256, "prompt_tok_s": 6000.0,
                         "output_tok_s": 110.0}
    # eight at once: 80 + 81 + … + 87, side by side
    assert p["together"]["n"] == 8 and sorted(p["together"]["each"]) == [80.0 + k for k in range(8)]
    assert p["together"]["output_tok_s"] == 668.0
    # the card: its link read under load (1 at rest, 5 at work), never flagged for resting
    assert rec["card"]["pcie_gen"] == 5 and rec["card"]["power_limit_w"] == 575.0
    assert ["pcie.link.gen.current", "pcie.link.width.current"] in smi.asked
    assert "why" not in rec
    assert cc.words(rec) == "card: 140 tok/s alone, 668 at 8 · 575 W limit · PCIe 5 ×16"
    # a slot too small for the long prompt: said, the rest asked
    short = cc.check(base, 8192, 8, run=Smi())
    assert short["probe"]["long"] == {"skipped": "a slot holds 8,192 tokens, under the 16,640 "
                                                 "the long prompt needs"}
    assert short["probe"]["together"]["output_tok_s"] == 668.0
    # fewer slots than eight: as many as it has, at once
    assert cc.check(base, 20480, 3, run=Smi())["probe"]["together"]["output_tok_s"] == 243.0


def test_a_probe_that_fails_says_why_and_never_raises(server):
    base = server(FAKE_LLAMA_PROBE_STATUS="500")
    rec = cc.check(base, 20480, 8, run=Smi())
    assert rec["why"] == "the speed probe failed: /completion answered 500: probe refused"
    assert "probe" not in rec and rec["card"]["name"] == "NVIDIA GeForce RTX 5090"
    assert cc.words(rec).startswith("card: the speed probe failed: /completion answered 500")
    # no server at all, and no nvidia-smi: the same, in its words
    gone = cc.check(f"http://127.0.0.1:{free_port()}", 20480, 8, run=lambda cmd: (_ for _ in ()).throw(
        FileNotFoundError("nvidia-smi")))
    assert gone["why"].startswith("the speed probe failed: /tokenize") and gone["card"] == {}


def test_a_failing_probe_leaves_the_step_running(box, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FAKE_LLAMA_PROBE_STATUS", "500")
    assert run_box(box, "run") == 0                       # the step ran, and made its bundle
    prog = json.loads((box["root"] / "run" / "progress.json").read_text())
    assert prog["state"] == "whole"
    assert prog["card"]["why"] == "the speed probe failed: /completion answered 500: probe refused"
    setup = rb.read(bundle_of(box, "run"))["setup"]
    assert setup["card_checks"][0]["why"] == prog["card"]["why"]
    assert "price" not in setup                          # none given: none kept
    assert "cost_usd" not in setup["tasks"][TASK]


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------

def readings() -> tuple[list[float], list[str], list[str]]:
    """a step's few minutes, every five seconds: loading at 100 W, the card
    check at 300, two minutes asking at 320 W (96% busy, then 50%), and —
    after a restart a minute later — 80 W"""
    t = [0.0, 5.0, *[10.0 + 5 * k for k in range(25)], 200.0, 205.0]
    phase = ["loading", "check", *["asking"] * 25, "after", "after"]
    rows = ["100, 5, 1200, 1500, 40, 0x0000000000000001",
            "300, 99, 27000, 2900, 60, 0x0000000000000000"]
    for k in range(25):
        why = "0x0000000000000004" if k < 5 else "0x0000000000000020" if k < 7 else \
            "0x0000000000000001" if k == 7 else "0x0000000000000000"
        rows.append(f"320, {96 if k < 13 else 50}, {29500 if k == 20 else 28000}, 2700, 70, {why}")
    rows += ["80, 0, 1200, 600, 45, 0x0000000000000001"] * 2
    return t, phase, rows


def test_the_sampler_as_worked_by_hand(tmp_path):
    t, phase, rows = readings()
    clock = iter(t)
    s = cc.Sampler(tmp_path / "gpu.jsonl", run=Smi(rows, reasons="clocks_event_reasons.active"),
                   clock=lambda: next(clock))
    for ph in phase:
        s.phase = ph
        s.tick()
    use = cc.summarise(cc.read_samples(tmp_path / "gpu.jsonl"))
    # a driver that names them clocks_event_reasons: asked that way, once refused the other
    assert s.error == "" and len(cc.read_samples(tmp_path / "gpu.jsonl")) == 29
    # watt-hours: each interval's mean power × its seconds, the gap over 30 s not joined —
    # (100+300)/2·5 + (300+320)/2·5 + 24·320·5 + 80·5 = 41,350 W·s
    assert use["wh"] == round(41350 / 3600, 3) == 11.486
    assert use["seconds"] == 135.0
    # while asking: 320 W, (13·96 + 12·50)/25 = 73.92% busy; the peak memory
    assert (use["mean_w"], use["mean_util"], use["peak_mib"]) == (320.0, 73.9, 29500)
    # throttled: 5 at the power cap, 2 for heat (idle isn't throttling) of 25
    assert use["throttled"] == 0.28 and use["reasons"] == {"power": 0.2, "heat": 0.08}
    assert cc.use_words(use) == "320 W · 74% busy"
    # nvidia-smi missing: a line with the time and phase, the reason kept once
    s2 = cc.Sampler(tmp_path / "none.jsonl", run=lambda cmd: (_ for _ in ()).throw(
        FileNotFoundError("nvidia-smi")))
    s2.tick()
    s2.tick()
    assert "nvidia-smi" in s2.error and all(set(x) == {"t", "phase"} for x in s2.samples)
    assert cc.summarise(s2.samples)["wh"] is None


def test_the_running_means_and_a_quarter_hour_under_60_percent_busy():
    samples = [{"t": 5.0 * k, "phase": "asking", "w": 191.0, "util": 44.0} for k in range(241)]
    run = cc.running(samples, 1200.0)                     # 20 minutes asking
    assert run["asking_min"] == 20.0 and run["util_15m"] == 44.0 and run["mean_w"] == 191.0
    assert "util_15m" not in cc.running(samples[:100], 495.0)     # not a quarter hour yet


# ---------------------------------------------------------------------------
# the slow-card flag
# ---------------------------------------------------------------------------

LABEL = fb.BENCH[HLE]["label"]


def progress(label: str, tok_s: float | None, n: int = 0, pace: float | None = None,
             of: int = 2158, state: str = "asking", card: dict | None = None,
             use: dict | None = None, model: str = "served/x") -> dict:
    """a step as the box's progress file says it, its card check's speed alone"""
    p = {"label": label, "model": model, "thinking": "on", "tasks": [HLE], "shard": "",
         "state": state, "at": round(time.time()), "dir": f"/workspace/x/{label}-1",
         "line": f"{LABEL} {n:,} of {of:,} · {pace:.1f} s an answer · 9.9 h left"
         if pace else ""}
    if tok_s is not None:
        p["card"] = {"card": {**{"power_limit_w": 575.0, "power_default_w": 575.0,
                                 "pcie_gen": 5, "pcie_gen_max": 5, "pcie_width": 16,
                                 "pcie_width_max": 16}, **(card or {})},
                     "probe": {"alone": {"output_tok_s": tok_s, "prompt_tok_s": 9000.0},
                               "together": {"n": 8, "output_tok_s": 4 * tok_s}}}
    if use:
        p["use"] = use
    return p


def steps(*ps: dict) -> list[dict]:
    """as one_box hands them to the round: a step each, its progress kept"""
    return [{"label": p["label"], "step": f"{p['label']}-1", "box_id": p["label"].lower(),
             "state": p["state"], "reachable": True, "_p": p} for p in ps]


def test_only_the_slow_card_is_flagged_with_its_hours_worked_by_hand():
    # four boxes of a build at 140, 145, 150 and 100 tok/s, each asking HLE's 2,158 answers
    got = steps(progress("L1", 140, 600, 12.0), progress("L2", 145, 600, 11.0),
                progress("L3", 150, 600, 10.0), progress("L4", 100, 200, 20.0))
    lines = ff.slow_cards(got)
    flagged = [x["label"] for x in got if x.get("slow")]
    assert flagged == ["L4"]
    # 100 ≤ 0.8 × 145 (the others' median). Left at its own pace: 1,958 × 20 s = 39,160 s;
    # a new box: 1,800 s to set up + 2,158 × 11 s (the others' median) = 25,538 s.
    # (39,160 − 25,538) / 3,600 = 3.78 h
    assert got[3]["slow"] == ("slow card — 100 tok/s against the others' 145; replacing it "
                              "now would save about 4 hours")
    assert lines == [f"L4 L4-1 (served/x): {got[3]['slow']}"]
    assert "the build's other boxes: 140, 145, 150 tok/s alone (median 145)" in got[3]["slow_why"]
    assert "left at its pace about 10.9 h (its own pace)" in got[3]["slow_why"]
    assert "a new box about 7.1 h" in got[3]["slow_why"]
    # nearly done: replacing it saves under an hour — no hours said
    late = steps(progress("L1", 140, 600, 12.0), progress("L2", 145, 600, 11.0),
                 progress("L3", 150, 600, 10.0), progress("L4", 100, 1500, 20.0))
    ff.slow_cards(late)
    assert late[3]["slow"] == "slow card — 100 tok/s against the others' 145"
    # before a pace of its own: the others' pace scaled by the cards' speeds — 2,158 × 11 s ×
    # 145/100 = 34,420 s; (34,420 − 25,538) / 3,600 = 2.47 h
    early = steps(progress("L1", 140, 600, 12.0), progress("L2", 145, 600, 11.0),
                  progress("L3", 150, 600, 10.0), progress("L4", 100, state="starting"))
    ff.slow_cards(early)
    assert early[3]["slow"].endswith("would save about 2 hours")
    # fewer than three others: no speed to compare with, and nothing flagged
    few = steps(progress("L1", 140, 600, 12.0), progress("L2", 145, 600, 11.0),
                progress("L4", 100, 200, 20.0))
    assert ff.slow_cards(few) == [] and not any(x.get("slow") for x in few)
    # another build's boxes aren't its others
    other = steps(*[progress(f"B{k}", v, 600, 11.0, model="served/y")
                    for k, v in enumerate((140, 145, 150), 1)], progress("L4", 100, 200, 20.0))
    assert ff.slow_cards(other) == []


def test_a_link_below_its_maximum_a_lowered_power_limit_and_a_quiet_card_are_flagged():
    got = steps(progress("L1", 140, 600, 12.0, card={"pcie_gen": 3}),
                progress("L2", 145, 600, 11.0, card={"power_limit_w": 400.0}),
                progress("L3", 150, 600, 10.0, use={"mean_w": 191.0, "mean_util": 44.0,
                                                    "util_15m": 44.0}),
                progress("L5", 147, 600, 11.0))
    ff.slow_cards(got)
    assert got[0]["slow"].startswith("slow card — PCIe 3 ×16, where it can do 5 ×16")
    assert got[1]["slow"].startswith("slow card — its power limit 400 W, under its default 575 W")
    assert got[2]["slow"].startswith("slow card — 44% busy over the last 15 min")
    assert "slow" not in got[3]
    # a box done isn't flagged, whatever its card
    done = steps(progress("L1", 140, 600, 12.0, card={"pcie_gen": 3}, state="whole"))
    assert ff.slow_cards(done) == []


def test_a_progress_file_not_as_a_box_writes_it_never_breaks_the_round():
    odd = progress("L1", 140, 600, 12.0)
    odd["card"] = {"card": [1], "probe": {"alone": "fast", "together": 3}}
    odd2 = progress("L2", 145, 600, 11.0)
    odd2["use"] = "busy"
    got = steps(odd, odd2, progress("L3", 150, 600, 10.0), progress("L4", 100, 200, 20.0))
    assert ff.slow_cards(got) == []              # L1 has no speed: L4 has two others, not three
    assert ff.progress_words(odd).startswith("L1 L1-1 · ")


def test_the_fetchs_line_says_the_card_and_its_use():
    p = progress("L4", 142, 200, 20.0, use={"mean_w": 312.0, "mean_util": 96.0})
    p["card"]["probe"]["together"] = {"n": 8, "output_tok_s": 610.0}
    p["card"]["card"]["power_limit_w"] = 340.0
    p["card"]["card"]["power_default_w"] = 340.0
    words = ff.progress_words(p)
    assert (f"· {LABEL} 200 of 2,158 · 20.0 s an answer · 9.9 h left · 312 W · 96% busy · card: "
            "142 tok/s alone, 610 at 8 · 340 W limit · PCIe 5 ×16 · started") in words
    # a step loading, its card checked: the card, and no use yet
    p2 = {**p, "state": "starting", "use": {}}
    assert "card: 142 tok/s" in ff.progress_words(p2) and " W · " not in ff.progress_words(p2)


def test_the_boards_list_keeps_the_flag_and_its_numbers(svc):  # noqa: F811
    got = steps(progress("L1", 140, 600, 12.0), progress("L2", 145, 600, 11.0),
                progress("L3", 150, 600, 10.0), progress("L4", 100, 200, 20.0))
    ff.slow_cards(got)
    now = time.time()
    rows = [{**{k: v for k, v in x.items() if not k.startswith("_")},
             **{k: x["_p"][k] for k in ("model", "thinking", "tasks", "shard", "line", "at")},
             "seen_at": now, "safe": False} for x in got]
    imf.store_boxes([{"asked": [x["box_id"] for x in got], "reached": [x["box_id"] for x in got]},
                     *rows])
    run = imf.read_boxes()["runs"][0]
    assert [f["label"] for f in run["slow"]] == ["L4"]
    assert run["slow"][0]["words"] == got[3]["slow"]
    assert run["slow"][0]["numbers"] == got[3]["slow_why"]
    assert next(b for b in run["boxes"] if b["label"] == "L4")["slow"] == got[3]["slow"]
    # what only the fetch reads never reaches the board
    assert "_p" not in json.dumps(imf.read_boxes())


# ---------------------------------------------------------------------------
# the box, end to end: the check, the sampler, the price
# ---------------------------------------------------------------------------

def test_the_box_keeps_its_card_its_energy_and_its_price(box, monkeypatch):  # noqa: F811
    def smi(cmd):
        fields = cmd[1].split("=", 1)[1].split(",")
        if fields[0] == "power.draw":
            return "300, 90, 28000, 2700, 70, 0x0000000000000000\n"
        return Smi()(cmd)
    monkeypatch.setattr(cc, "RUN", smi)
    monkeypatch.setattr(cc, "EVERY_S", 0.05)
    assert run_box(box, "run", "--price", "0.5") == 0
    prog = json.loads((box["root"] / "run" / "progress.json").read_text())
    # the card check, as the fetch sees it at once (the run's three slots: 80 + 81 + 82)
    assert cc.words(prog["card"]) == "card: 140 tok/s alone, 243 at 3 · 575 W limit · PCIe 5 ×16"
    assert prog["use"]["mean_w"] == 300.0
    b = rb.read(bundle_of(box, "run"))
    setup = b["setup"]
    assert "gpu.jsonl" in b["files"] and setup["price"] == {"per_hour": 0.5,
                                                            "by": "the box line (--price)"}
    use = setup["gpu_use"]
    assert use["wh"] > 0 and use["mean_w"] == 300.0 and use["seconds"] > 0
    # one benchmark in the step: all of its energy, time and dollars
    task = setup["tasks"][TASK]
    assert task["energy_wh"] == pytest.approx(use["wh"], abs=1e-3)
    assert task["server_s"] == pytest.approx(use["seconds"], abs=0.1)
    assert task["cost_usd"] == round(0.5 * task["server_s"] / 3600, 4)
    assert setup["card_checks"][0]["session"] == 1
    # the import keeps them on the row's record
    register(box["sha"])
    assert ir.import_bundle(bundle_of(box, "run"), "masein", lambda s: None) == 0
    reg = imf.registry(config.OUT_DIR / ROW)["tasks"][TASK]
    assert reg["card"]["alone_out"] == 140.0 and reg["card"]["together_out"] == 243.0
    assert reg["card"]["together_n"] == 3 and reg["card"]["name"] == "NVIDIA GeForce RTX 5090"
    assert (reg["energy_wh"], reg["cost_usd"], reg["price_per_hour"]) == (
        task["energy_wh"], task["cost_usd"], 0.5)


def test_the_box_line_passes_its_price_on():
    a = fbx.parser().parse_args(["L4", "--as", "served/x", "--gguf", "g.gguf", "--server", "s.tgz",
                                 "--price", "0.42"])
    argv = fbx.argv_of("L4", 1, fbx.box_of("L4")[0], a)
    assert argv[argv.index("--price") + 1] == "0.42"
    none = fbx.parser().parse_args(["L4", "--as", "served/x", "--gguf", "g.gguf", "--server",
                                    "s.tgz"])
    assert "--price" not in fbx.argv_of("L4", 1, fbx.box_of("L4")[0], none)


# ---------------------------------------------------------------------------
# the board: speed and energy from what the imports kept
# ---------------------------------------------------------------------------

def samples(d: Path, task: str, walls: list[float], rates: list[float] | None = None) -> None:
    """a task's scored samples, each run with its seconds (and its timings)"""
    t = sf.task_dir(d, task)
    t.mkdir(parents=True, exist_ok=True)
    rates = rates or []
    (t / f"samples_{task}_2026-10-20T08-00-00.000000.jsonl").write_text("".join(
        json.dumps({"doc_id": k, "doc": {}, "resps": [["ANSWER: A"]], "filtered_resps": [""],
                    "frontier": [{"epoch": 0, "ok": True, "ran_out": False, "wall_s": w,
                                  **({"predicted_n": 256, "predicted_ms": 256 / rates[k] * 1000}
                                     if k < len(rates) else {})}]}) + "\n"
        for k, w in enumerate(walls)))
    (t / "setup.json").write_text(json.dumps({"concurrency": {"in_flight": 8,
                                                              "server_slots": 8}}))


def card(alone: float, who: str) -> dict:
    return {"name": "NVIDIA GeForce RTX 5090", "alone_out": alone, "alone_prompt": 64 * alone,
            "long_prompt": 42 * alone, "long_out": alone - 30, "together_out": 4 * alone,
            "together_n": 8, "box": who}


def registry(d: Path, priced: bool = True) -> None:
    d.mkdir(parents=True, exist_ok=True)
    gpqa = {"answers": 200, "shard_bundles": [
        {"shard": 1, "box": "L13", "answers": 100, "energy_wh": 50.0, "server_s": 3600.0,
         **({"cost_usd": 0.2} if priced else {}), "card": card(140.0, "L13")},
        {"shard": 2, "box": "L14", "answers": 100, "energy_wh": 70.0, "server_s": 5400.0,
         **({"cost_usd": 0.3} if priced else {}), "card": card(100.0, "L14")}]}
    hle = {"answers": 50, "box": "L1", "energy_wh": 455.0, "server_s": 9000.0,
           **({"cost_usd": 1.25} if priced else {}), "card": card(150.0, "L1")}
    (d / "frontier_imports.json").write_text(json.dumps({"tasks": {GPQA: gpqa, HLE: hle}}))


def test_a_row_with_card_checks_shows_its_speed_and_energy(svc):  # noqa: F811
    mid = "served/step4000-lda-2610 · thinking"
    d = reports.row_dir(mid)
    samples(d, GPQA, [10.0 * k for k in range(1, 11)], [100.0, 110.0, 120.0])
    samples(d, HLE, [144.0] * 5)
    registry(d)
    assert report_stats.work(say=lambda s: None)["failed"] == 0
    sp = svc.get("/api/beyond", params={"model": mid}).json()["speed"]
    assert sp["source"] == "frontier" and sp["cards"] == ["RTX 5090"] and sp["boxes"] == 3
    # the card checks, one a box: the median and the range
    assert sp["probe"]["alone_out"] == {"median": 140.0, "lo": 100.0, "hi": 150.0, "n": 3}
    assert sp["together_n"] == 8 and sp["label"] == "step4000"
    g = sp["benches"][GPQA]
    # seconds 10 … 100: the median 55, the 90th percentile (nearest rank) 90
    assert (g["wall_median"], g["wall_p90"], g["n"]) == (55.0, 90.0, 10)
    assert g["out_tok_s"] == pytest.approx(110.0) and g["at_once"] == 8
    # energy: (50 + 70) Wh over 200 answers; dollars: (0.2 + 0.3) over 200
    assert g["wh_per_answer"] == pytest.approx(0.6) and g["usd_per_answer"] == pytest.approx(0.0025)
    h = sp["benches"][HLE]
    assert h["wall_median"] == 144.0 and h["wh_per_answer"] == pytest.approx(9.1)
    assert "out_tok_s" in h and h["out_tok_s"] is None              # no timings kept: none
    # the report's two columns
    from service import app
    r = reports.report("step-checkpoints", app.results_payload())
    assert r["speed"]["step4000"]["alone_out"]["median"] == 140.0
    assert r["speed"]["step4000"]["gpqa_wh"] == pytest.approx(0.6)
    assert "step8804" not in r["speed"]                              # measured from the next run


def test_no_price_no_dollars_and_an_old_row_shows_nothing_new(svc):  # noqa: F811
    mid = "served/step4000-lda-2610 · thinking"
    d = reports.row_dir(mid)
    samples(d, GPQA, [10.0, 20.0])
    registry(d, priced=False)
    report_stats.work(say=lambda s: None)
    g = svc.get("/api/beyond", params={"model": mid}).json()["speed"]["benches"][GPQA]
    assert "usd_per_answer" not in g and g["wh_per_answer"] == pytest.approx(0.6)
    # a row from before 23: no timings, no card checks — nothing, not zeros
    old = "served/step2000-lda-2610 · thinking"
    od = reports.row_dir(old)
    t = sf.task_dir(od, GPQA)
    t.mkdir(parents=True)
    (t / f"samples_{GPQA}_2026-10-01T08-00-00.000000.jsonl").write_text(json.dumps(
        {"doc_id": 0, "doc": {}, "resps": [["ANSWER: A"]], "filtered_resps": [""],
         "frontier": [{"epoch": 0, "ok": True, "ran_out": False}]}) + "\n")
    report_stats.work(say=lambda s: None)
    got = svc.get("/api/beyond", params={"model": old}).json()
    assert got["speed"] is None and "speed" not in got["benches"][GPQA]


def test_a_devicemark_served_row_shows_its_items_speed(svc):  # noqa: F811
    mid = "served/dm-box"
    d = reports.row_dir(mid)
    d.mkdir(parents=True)
    items = [{"bench": "mmlu_pro", "key": str(k), "wall_s": 2.0 + k, "decode_tok_s": 40.0 + k}
             for k in range(5)] + [{"bench": "math", "key": "m", "wall_s": None,
                                    "decode_tok_s": None}]
    (d / beyond.DM_ITEMS).write_text("".join(json.dumps(x) + "\n" for x in items))
    report_stats.work(say=lambda s: None)
    sp = svc.get("/api/beyond", params={"model": mid}).json()["speed"]
    assert sp["source"] == "devicemark" and sp["protocol"] == beyond.DM_PROTOCOL
    assert sp["benches"] == {"mmlu_pro": {"n": 5, "wall_median": 4.0, "wall_p90": 6.0,
                                          "out_n": 5, "out_tok_s": 42.0}}


# ---------------------------------------------------------------------------
# 23b's calibration, its last group said
# ---------------------------------------------------------------------------

def test_the_calibration_note_says_what_cais_leaves_out():
    note = next(n for n in reports.NOTES if n.startswith("Calibration error is CAIS's measure"))
    assert note.endswith("CAIS's measure leaves out its last group, the answers stated with the "
                         "highest confidence; the chart above shows every answer.")
    # 260 answers: bins of 100 and the last of 160 — only the first counted, the 160 most
    # confident left out, as the page says under its chart (byCalLeft)
    conf = [c / 260 for c in range(260)]
    diff = abs(sum(conf[:100]) / 100 - 1)
    assert beyond.calib_err(conf, [True] * 260) == pytest.approx((100 / 260 * diff ** 2) ** 0.5)
