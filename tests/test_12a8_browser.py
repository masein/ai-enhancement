"""12a.8 on the page: after a rubric change and a re-mark, the Summarise
group reader shows each practice answer's new verdict — with a result file
dated a day ahead among the watched ones, as on the server, where the
readers kept 12a.6's marks. The re-mark is the command line's, as the
server's was; the stand-in judge marks. Fixtures only."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import everyday as ev
from service import config
from test_12n1_everyday_browser import GOOD, everyday, open_group

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12a8"
OLD = "4 of 4 (an old verdict, from before the rubric changed)"


def decided(page, qid):
    return page.locator(f"[data-grp-answer='{GOOD}|{qid}'] [data-grp-decided]").inner_text()


def test_after_a_rubric_change_and_a_remark_the_group_reader_shows_the_new_verdict(live, page):
    import service.app as appmod
    d = Path(live["tree"]["models"][GOOD]["dir"])
    f = d / ev.OUT_NAME
    was = f.read_bytes()
    qs = {q["id"]: q for q in ev.load_bank()}
    it = next(it for it in ev.read(d)["items"]
              if it["group"] == "summarising" and it["half"] == ev.PRACTICE
              and it.get("answer_text") and ev.grade(qs[it["id"]], it["answer_text"])[0] is None)
    qid = it["id"]
    # the marks from before the change, as they sat on the server
    out = json.loads(was)
    for x in out["items"]:
        if x["id"] == qid:
            x.update({"pass": True, "reason": OLD, "score": 4, "rubric": "000000000000"})
    f.write_text(json.dumps(out), encoding="utf-8")
    # a result file dated a day ahead
    meta = next(p for p in d.iterdir() if p.name == "model_meta.json" or p.name.startswith("results"))
    st = meta.stat()
    t = time.time() + 86400
    os.utime(meta, (t, t))
    real_bank, real_judge = ev.load_bank, config.JUDGE_MODEL

    def bank(path=None):
        rows = real_bank(path)
        for q in rows:
            if path is None and q["id"] == qid:
                c = ev.judge_check(q)
                c["rubric"] = c["rubric"] + "\n7. (a changed rubric)"
        return rows
    try:
        # the stand-in judge from the start: the judge's version is in the
        # page's key too, and only the re-mark may change what it sees
        config.JUDGE_MODEL = "stub"
        appmod._cache.update(at=0.0)
        everyday(page, live)
        open_group(page, "summarising")
        assert decided(page, qid) == f"the judge: {OLD}"          # what the file said then
        # the rubric changes; the command line re-marks, with the stand-in judge
        ev.load_bank = bank
        ev.remark(config.OUT_DIR, want={d.name}, judge=True, snapshot=False)
        now = next(x for x in ev.read(d)["items"] if x["id"] == qid)
        assert now["reason"] != OLD and now["rubric"] == ev.rubric_key(
            next(q for q in ev.load_bank() if q["id"] == qid))
        appmod._cache.update(at=0.0)                  # past its five seconds: the key decides
        page.reload()
        page.wait_for_selector(f"[data-grp-answer='{GOOD}|{qid}'] [data-grp-decided]")
        assert decided(page, qid) == f"the judge: {now['reason']}"
        SCREENS.mkdir(parents=True, exist_ok=True)
        page.locator(f"[data-grp-q='{qid}']").screenshot(path=SCREENS / "summarise-after-remark.png")
        assert page.errors == []
    finally:
        ev.load_bank, config.JUDGE_MODEL = real_bank, real_judge
        f.write_bytes(was)
        os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns))
        appmod._cache.update(key=None, payload=None, at=0.0)
