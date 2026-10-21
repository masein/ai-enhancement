"""25 on the board: IFEval and IFBench answers scored as an import scores
them (service/frontier.score_task), on a build's · thinking row and its
plain row — the four numbers on the score's line, the step-checkpoints
report's "Instruction following" section with thinking on and off side by
side and its sentence from the paired verdicts, both benchmarks in the
two-builds report question by question (McNemar's exact test, one run a
prompt) with each instruction ✓ or ✗ under each reply, never in "Where we
stand" or "Every model", and never DeviceMark's IFEval or the board's other
IFEval in these columns. Invented prompts, the instructions real; nothing
runs, nothing is fetched."""

from __future__ import annotations

import json

import pytest

import frontier as fb
import report_lm_eval as rpt
from conftest import make_service
from service import config, db, follow, paired, reported, reports
from service import frontier as sf

REP = json.loads((reported.REPORTS_DIR / "step-checkpoints.json").read_text())
S88, S40, Q4 = "step8804", "step4000", "Q4 (UD-Q4_K_XL)"
IFE, IFB = "ifeval_google", "ifbench_test"
N = 20
END = "Is there anything else I can help with?"
# each build's prompts, in order: y followed (strict), l followed loose only, n broken
PLAN = {
    (S88, IFE, "on"): "y" * 6 + "l" * 2 + "n" * 12, (Q4, IFE, "on"): "y" * 16 + "l" * 2 + "n" * 2,
    (S40, IFE, "on"): "y" * 10 + "n" * 10,
    (S88, IFE, "off"): "y" * 12 + "n" * 8, (Q4, IFE, "off"): "y" * 12 + "n" * 8,
    (S88, IFB, "on"): "y" * 5 + "l" * 5 + "n" * 10, (Q4, IFB, "on"): "y" * 10 + "n" * 10,
    (S40, IFB, "on"): "y" * 4 + "n" * 16,
    (S88, IFB, "off"): "y" * 2 + "n" * 18, (Q4, IFB, "off"): "y" * 12 + "n" * 8,
}


def items(task: str) -> list[dict]:
    if task == IFE:                               # no commas, or an ending phrase
        return [{"id": str(1000 + k), "question": f"Invented prompt {k}.",
                 "instructions": ["startend:end_checker" if k % 2 else "punctuation:no_comma"],
                 "kwargs": [{"end_phrase": END} if k % 2 else {}]} for k in range(N)]
    return [{"id": str(k), "question": f"Invented prompt {k}.",
             "instructions": ["format:no_whitespace"], "kwargs": [{}]} for k in range(N)]


def reply(task: str, k: int, how: str) -> str:
    if task == IFE and k % 2:
        return {"y": f"Done. {END}", "l": f"Done. {END}\nThanks!", "n": "Done."}[how]
    if task == IFE:
        return {"y": "No commas here.", "l": "Sure, here:\nNo commas here.",
                "n": "Yes, a comma."}[how]
    return {"y": "Hello,world!", "l": "Sure:\nHello,world!", "n": "Hello world"}[how]


def row_of(label: str, on: bool) -> tuple[str, object]:
    mid = next(o["row"] for o in REP["ours"] if o["label"] == label)
    mid = mid if on else mid.removesuffix(" · thinking")
    return mid, reports.row_dir(mid)


def build(label: str, task: str, th: str) -> None:
    """one build's answers on one benchmark, scored as its import scores them"""
    mid, row = row_of(label, th == "on")
    base = mid.removesuffix(" · thinking")
    rec = db.served_get(base)
    if rec is None:                                   # kept as registered where it is
        rec = {"id": base, "name": base.split("/")[1], "base_url": "", "key": "", "how": "x",
               "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "on", "pin": {}, "by": "masein",
               "at": 0, "runtime": "lda-2610"}
        db.served_put(rec)
    d = sf.task_dir(row, task)
    d.mkdir(parents=True, exist_ok=True)
    (d / sf.SETUP).write_text(json.dumps({"thinking": th, "where": "rented GPU · RTX 5090",
                                          "runtime": "lda-2610"}))
    lead = "<think>\nLet me follow every instruction.\n</think>\n\n" if th == "on" else ""
    (d / sf.ANSWERS).write_text("".join(json.dumps(
        {"id": it["id"], "epoch": 0, "answer": lead + reply(task, k, how), "finish": "stop",
         "tokens": 50}) + "\n" for k, (it, how) in enumerate(zip(items(task),
                                                                  PLAN[(label, task, th)]))))
    (row / "model_meta.json").write_text(json.dumps({"model": mid}))
    got = sf.score_task(row, task, rec)
    assert got and got.get("score") is not None, got


def world() -> None:
    for task in (IFE, IFB):
        p = fb._cache(config.BENCH_ROOT, task)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"items": items(task), "extra": {}}))
    for label, task, th in PLAN:
        build(label, task, th)


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    paired._tables.clear()
    world()
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield client
    client.__exit__(None, None, None)


def test_the_score_is_the_authors_measure_and_its_line_says_the_other_three(svc):
    import service.app as appmod
    cells = appmod.results_payload()["cells"]
    on88 = row_of(S88, True)[0]
    ife, ifb = cells[IFE][on88], cells[IFB][on88]
    assert ife["v"] == pytest.approx(6 / 20)                     # IFEval: prompt-level strict
    assert ifb["v"] == pytest.approx(10 / 20)                    # IFBench: prompt-level loose
    assert ife["follow"]["prompt_loose"]["v"] == pytest.approx(8 / 20)
    assert ifb["follow"]["prompt_strict"]["v"] == pytest.approx(5 / 20)
    assert ife["how"] == ("by its own checkers, on the reply alone: prompt-level strict · "
                          "prompt-level loose 40.0 · instruction-level strict 30.0, loose 40.0")
    assert ifb["how"].startswith("by its own checkers, on the reply alone: prompt-level loose · "
                                 "prompt-level strict 25.0 · instruction-level strict 25.0")
    # the thinking-off row beside it, its own score
    assert cells[IFE][row_of(S88, False)[0]]["v"] == pytest.approx(12 / 20)
    # the samples keep each instruction's verdict; the results the four numbers
    d = sf.task_dir(row_of(S88, True)[1], IFE)
    s = [json.loads(x) for x in next(d.glob(f"samples_{IFE}_*.jsonl")).read_text().splitlines()]
    assert s[1]["frontier"][0]["follow"] == {"ids": ["startend:end_checker"], "strict": [True],
                                             "loose": [True]}
    res = json.loads(next(d.glob("results_*.json")).read_text())
    assert res["frontier"]["follow"]["metric"] == "strict"
    assert (res["frontier"]["follow"]["prompts"], res["frontier"]["follow"]["instructions"]) == (
        20, 20)


def test_the_section_puts_thinking_on_and_off_side_by_side_with_its_sentence(svc):
    d = reports.report("step-checkpoints", compute=True)
    F = d["follow"]
    assert [c["label"] for c in F["columns"]] == ["IFEval strict", "IFEval loose",
                                                  "IFBench loose"]
    rows = {r["label"]: r for r in F["rows"]}
    assert [r["label"] for r in F["rows"]] == [o["label"] for o in REP["ours"]]   # every build
    c = rows[S88]["cells"]
    assert (c["ifeval_google:strict"]["on"]["v"], c["ifeval_google:strict"]["off"]["v"]) == (
        pytest.approx(0.30), pytest.approx(0.60))
    assert c["ifeval_google:loose"]["on"]["v"] == pytest.approx(0.40)
    assert c["ifbench_test:loose"]["on"]["v"] == pytest.approx(0.50)
    assert c["ifbench_test:loose"]["off"]["v"] == pytest.approx(0.10)
    se = c["ifeval_google:strict"]["on"]["se"]
    assert se == pytest.approx((0.30 * 0.70 * 20 / 19 / 20) ** 0.5)              # over prompts
    # step4000 has no thinking-off answers: nothing said for them
    assert rows[S40]["cells"]["ifeval_google:strict"]["off"] is None
    assert rows[S40]["rowOff"] == "served/step4000-lda-2610"
    assert F["sentence"] == (
        "step8804 against Q4 (UD-Q4_K_XL) on IFEval: lower, by 50.0 points; with thinking "
        "off, can't tell. step8804 against Q4 (UD-Q4_K_XL) on IFBench: can't tell; with "
        "thinking off, lower, by 50.0 points.")
    # Our runs: the thinking-on score, the rest of the report as before
    ours = {r["label"]: r for r in d["ours"]}
    assert ours[S88]["cells"]["ifeval"]["v"] == pytest.approx(0.30)
    assert ours[S88]["status"].endswith("benchmarks in")                       # 2 of 9
    # 25b: compared with Google's own Gemma 4 numbers — in Where we stand and
    # Every model (tests/test_25b_gemma.py)
    b = {x["key"]: x for x in d["benchmarks"]}
    assert b["ifeval"]["compare"] is True and b["ifbench"]["compare"] is True
    assert "in_every" not in b["ifeval"] and "in_every" not in b["ifbench"]
    # Across training: a segment between the two checkpoints, with its verdict —
    # four prompts apart, too few to tell (McNemar's p 0.125)
    seg = d["training"]["segments"]["ifeval"]
    assert [(x["a"], x["b"]) for x in seg] == [(S40, S88)]
    assert seg[0]["verdict"] == "can't tell" and seg[0]["d"] == pytest.approx(-20.0)


def test_the_board_s_other_ifevals_are_never_in_these_columns(svc):
    """lm_eval's IFEval (the board's own column) and DeviceMark's are other
    protocols: a row that has them keeps them where they are"""
    import service.app as appmod
    payload = appmod.results_payload()
    on88 = row_of(S88, True)[0]
    for t in ("ifeval", "dm_ifeval"):
        payload["cells"].setdefault(t, {})[on88] = {"v": 0.99, "se": 0.01, "how": t}
    d = reports.report("step-checkpoints", payload=payload, compute=True)
    assert {c["task"] for c in d["follow"]["columns"]} == {IFE, IFB}
    assert "0.99" not in json.dumps(d["follow"])
    assert {x["task"] for x in d["benchmarks"]}.isdisjoint({"ifeval", "dm_ifeval"})
    ours = {r["label"]: r for r in d["ours"]}
    assert ours[S88]["cells"]["ifeval"]["v"] == pytest.approx(0.30)
    assert follow.COLUMNS == ((IFE, "strict"), (IFE, "loose"), (IFB, "loose"))
    # and the Models table never names two columns alike
    names = rpt.JS.split("const BENCH_NAMES", 1)[1].split("};", 1)[0]
    assert "ifeval_google: 'IFEval (the reply alone)'" in names and "ifeval: 'IFEval'" in names


def test_the_two_builds_report_compares_them_and_marks_each_instruction(svc):
    d = reports.report("paired", compute=True, a=S88, b=Q4)
    rows = {r["key"]: r for r in d["rows"]}
    ife, ifb = rows["ifeval"]["pair"], rows["ifbench"]["pair"]
    # one run a prompt: McNemar's exact test, on IFEval's strict and IFBench's loose
    assert ife["runs"] == 1 and ife["p"] is not None and ife["verdict"] == "B higher"
    assert (ife["a_better"], ife["b_better"]) == (0, 10)
    assert ifb["verdict"] == "can't tell" and (ifb["a_better"], ifb["b_better"]) == (0, 0)
    # a prompt of the shown half, opened: its prompt, the reply, each instruction
    rep = reports.files()["paired"]
    shown = next(it["id"] for it in items(IFB) if reports.shown_half(IFB, it["id"]))
    k = int(shown)
    a = paired.one(rep, S88, IFB, shown, 0)
    assert a["question"]["q"] == f"Invented prompt {k}."
    assert a["right"] is None and a["marks"]["final"] is None
    want = PLAN[(S88, IFB, "on")][k]
    assert a["instructions"] == [{"id": "format:no_whitespace", "strict": want == "y",
                                  "loose": want in "yl", "ok": want in "yl"}]
    assert a["metric"] == "loose" and a["reply"] == reply(IFB, k, want)
    assert a["verdict"].startswith("strict: ")
    # a hidden-half prompt is never opened
    hidden = next(it["id"] for it in items(IFB) if not reports.shown_half(IFB, it["id"]))
    with pytest.raises(paired.Refused):
        paired.one(rep, S88, IFB, hidden, 0)


def test_the_worker_keeps_the_thinking_off_pairs_too(svc):
    rep = reports.files()["paired"]
    slots = {j.slot for j in paired.jobs(rep)}
    assert f"paired|paired|{S88}|{Q4}|{IFE}|off" in slots or \
        f"paired|paired|{Q4}|{S88}|{IFE}|off" in slots
    assert not any(s.endswith("|off") and (IFE not in s and IFB not in s) for s in slots)
    # step4000 has no plain-row answers: no off pair of it
    assert not any(s.endswith("|off") and S40 in s for s in slots)
