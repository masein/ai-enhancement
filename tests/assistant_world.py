"""28: the assistant's test world — every model, phone, report and box the
test bank names: 21c's scorecard (our step checkpoints, the Q4, Q2 and BF16
baselines, Epoch AI's models and the cards), 24b's file sizes, the On-device
report's cards entered by masein, step8804 measured here thinking on and off,
the phone build, made-up tests from made-up testers on three phones, a rented
box running step4000 and a run on the server. Invented numbers throughout;
nothing runs, nothing is fetched."""

from __future__ import annotations

import json
import time
from pathlib import Path

from service import config, db, reports, telemetry
from test_21c_page import SPEC, results
from test_21c_page import world as scorecard
from test_24b_browser import FILES
from test_26_telemetry import a_test, keep

S88 = "served/step8804-lda-2610"
PHONE = "served/Qwen3.6-35B-A3B-k4-LDA-phone-build"
NAMES = {
    "models": {
        "model-001": {"name": "Qwen3.6-35B-A3B k4-LDA, UD-Q4_K_XL", "row": "",
                      "what": "the phone build of the original model", "checkpoint": PHONE,
                      "sha256": ""},
        "model-002": {"name": "step4000", "row": "", "what": "", "checkpoint": "step4000",
                      "sha256": ""},
        "model-003": {"name": "step4000-denseq8", "row": "",
                      "what": "step4000 with its non-expert weights in q8_0 instead of q5_K",
                      "checkpoint": "step4000", "sha256": ""},
        "model-004": {"name": "step8804", "row": "", "what": "", "checkpoint": "step8804",
                      "sha256": ""}},
    "tiers": {"T0": "RAM 14 GB or more"}, "engines": {},
    "devices": {"CPH2747": "OnePlus 15", "SM-S938B": "Galaxy S25 Ultra"}}
# (test id, app model, phone code, its maker, decode)
TESTS = [("a1", "model-004", "CPH2747", "OnePlus", 13.0), ("a2", "model-004", "CPH2719", "OnePlus",
                                                         3.2),
         ("a3", "model-004", "CPH2719", "OnePlus", 3.4), ("b1", "model-003", "CPH2747", "OnePlus",
                                                         13.1),
         ("b2", "model-003", "SM-S938B", "samsung", 9.4), ("c1", "model-001", "CPH2747",
                                                          "OnePlus", 14.2)]


def world(tmp_path_factory=None) -> Path:
    """the world, under the live service's BENCH_ROOT; returns the names file"""
    import enter_report_cards as erc
    import import_frontier as imf

    import service.app as appmod
    from service import report_stats
    scorecard()
    for label, more in FILES.items():
        mid = SPEC[label]["row"].removesuffix(" · thinking")
        db.served_put({**(db.served_get(mid) or {}), **more})
    assert erc.main(["on-device", "--by", "masein"]) == 0
    # step8804, measured here thinking on and off
    db.served_put({"id": S88, "name": "step8804-lda-2610", "base_url": "", "key": "",
                   "how": "rented GPU", "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "on",
                   "pin": {}, "by": "masein", "at": 0})
    for row in (S88 + " · thinking", S88):
        d = reports.row_dir(row)
        for task, v in (("gpqa_diamond_epoch", 0.70), ("hle_text_cais", 0.11),
                        ("mmlupro_tiger", 0.81), ("ifeval_google", 0.88)):
            if not row.endswith(" · thinking") and task != "ifeval_google":
                continue
            results(d, S88, task, v, 0.02, 500, 3)
        if not row.endswith(" · thinking"):
            for f in d.glob("*/frontier/results_*.json"):
                f.write_text(f.read_text().replace(",enable_thinking=True", ""))
    db.add(S88, "instruct", "frontier", "masein", "imported", status="done")   # its "added"
    # Ornith-1.5, served here on its card's settings
    orn = "served/Ornith-1.5-35B-A3B-Q4_K_M"
    db.served_put({"id": orn, "name": "Ornith-1.5-35B-A3B Q4_K_M (card settings)",
                   "base_url": "", "key": "", "how": "rented GPU", "thinking": "on",
                   "based_on": "", "pin": {}, "by": "masein", "at": 0})
    for task, v in (("gpqa_diamond_epoch", 0.74), ("hle_text_cais", 0.12),
                    ("math_l5_epoch", 0.93)):
        results(reports.row_dir(orn + " · thinking"), orn, task, v, 0.02, 500, 0)
    # three Standard models the bank names, as the fixture tree writes its own
    import make_fixture as mf
    for mid, policy, kind, params in (("Qwen/Qwen3-1.7B", "good", "instruct", 1_700_000_000),
                                      ("google/gemma-3-270m-it", "chance", "instruct",
                                       270_000_000),
                                      ("EleutherAI/pythia-410m", "short_pick", "base",
                                       410_000_000)):
        mf.write_model_meta(config.OUT_DIR, mid, kind, params)
        for task in mf.FULL:
            mf.write_task(config.OUT_DIR, mid, kind, params, policy, task, mf.make_docs(task),
                          mf.SEED)
    # the phone build, served here
    db.served_put({"id": PHONE, "name": "Qwen3.6-35B-A3B k4-LDA (phone build)", "base_url": "",
                   "key": "", "how": "phone build", "based_on": "Qwen/Qwen3.6-35B-A3B",
                   "thinking": "on", "phone": True, "pin": {}, "by": "masein", "at": 0})
    results(reports.row_dir(PHONE + " · thinking"), PHONE, "gpqa_diamond_epoch", 0.66, 0.02,
            792, 0)
    # the testers' phones, made-up testers
    root = Path(config.BENCH_ROOT)
    names = root / "assistant-names.json"
    names.write_text(json.dumps(NAMES))
    telemetry.NAMES = names
    telemetry._memo.update(stamp=None)
    telemetry._names_memo.update(stamp=None)
    for i, (tid, model, device, maker, decode) in enumerate(TESTS):
        t = a_test(tid, install=f"00000000-0000-4000-8000-{i:012d}", model=model, device=device,
                   decode=decode, received=f"2026-10-2{i % 4}T10:00:00Z")
        t["device"]["manufacturer"] = maker
        t["label"] = "tester-" + tid[0]
        keep(root, t)
    # a rented box running step4000 on HLE, and a run on the server
    now = time.time()
    imf.store_boxes([{"asked": ["a"], "reached": ["a"]},
                     {"label": "A1", "model": "served/step4000-lda-2610 · thinking",
                      "step": "A1-1", "state": "asking", "at": now - 60, "seen_at": now,
                      "safe": False, "reachable": True, "box_id": "a", "line": "",
                      "tasks": ["hle_text_cais"]}])
    sid = db.add("Qwen/Qwen3-0.6B", "instruct", "standard", "masein", "a test run")
    db.update(sid, status="running")
    report_stats.work(say=lambda s: None)
    appmod._cache.update(key=None, payload=None, at=0.0)
    return names


def unwind(saved_names: Path) -> None:
    telemetry.NAMES = saved_names
    telemetry._memo.update(stamp=None)
    telemetry._names_memo.update(stamp=None)
