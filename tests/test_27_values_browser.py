"""27: no number changes — every (view, model, column) the Models tabs showed
before one table drew them all, recorded from the old renderers on this
world (tests/fixtures/models27/before.json, `RECORD27=1` wrote it), and
checked against what modelTable draws now: each number a cell showed is in
the same model's cell of the same column, on the same view.

The world: the fixture board; 21c's scorecard (Epoch AI's 14 Oct data, the
card numbers, our served rows with their Frontier scores); 25c's On-device
cards (Edge0, Bonsai, Gemma); MobileAIBench for two fixture models (12o3);
and DeviceMark rows for a served phone build and a Hugging Face model, both
thinking modes (12x). Every model, question and number is invented."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.dashboard
BEFORE = Path(__file__).resolve().parent / "fixtures" / "models27" / "before.json"
# every Models view a table is drawn in, and DeviceMark's chart page (its table)
VIEWS = ["view=latest", "view=standard&group=all", "view=standard&group=knowledge",
         "view=standard&group=commonsense", "view=standard&group=reasoning",
         "view=standard&group=math", "view=standard&group=trust",
         "view=standard&group=instruction", "view=mobile&group=all",
         "view=mobile&group=devicemark", "view=mobile&group=mobileaibench",
         "view=mobile&group=mmlu", "view=everyday", "view=frontier", "chip=ondevice"]
SETUP = "served/DM-k4-LDA-phone-build-MTP"
QWEN = "Qwen/Qwen3.5-2B"


def _api(live, path, body=None):
    req = urllib.request.Request(live["base"] + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def world27(live, mp) -> None:
    """every Models tab with rows: the scorecard, the On-device cards,
    MobileAIBench, Mobile-MMLU-Pro, an IFEval score of a served model and
    DeviceMark, on the fixture board"""
    import enter_report_cards as erc
    import mobile_mmlu as mmp
    import mobileaibench as mab
    import service.app as appmod
    from fake_openai import FakeServer
    from service import config, db, served
    from test_12o3 import sit_mab
    from test_14_3_mobile_mmlu import FIXTURE, RIGHT, a_key, fake_pin, pin_fixture, sit_picks
    from test_12q_devicemark_board import HF, SERVED, _row
    from test_12x_models_devicemark_browser import OFF, ON, Q_OFF, Q_ON, _dm_results
    from test_21c_page import SPEC, world
    world()
    # our rows, named as on the board (the scorecard names them by their ids)
    real = {"step8804": "step8804 (lda runtime)", "step4000": "step4000 (lda runtime)",
            "step1000": "step1000 (lda runtime)", "Q4 (UD-Q4_K_XL)": "Qwen3.6-35B-A3B Q4 (lda runtime)",
            "BF16": "Qwen3.6-35B-A3B BF16 (lda runtime)", "Q2 (UD-Q2_K_XL)":
            "Qwen3.6-35B-A3B Q2 (lda runtime)", "Q4, earlier run": "Qwen3.6-35B-A3B Q4 (original, k=8)",
            "Ornith-1.5 Q4_K_M": "Ornith-1.5-35B-A3B Q4_K_M (card settings)"}
    for label, name in real.items():
        row = SPEC[label]["row"]
        mid = row.removesuffix(" · thinking")
        rec = db.served_get(mid)
        if rec:
            db.served_put({**rec, "name": name})
            # as a run through it keeps how it was served
            from service import reports
            (reports.row_dir(row) / "model_meta.json").write_text(json.dumps(
                {"model": row, "served": served.view(db.served_get(mid))}))
    erc.main(["on-device", "--by", "masein"])
    tree = live["tree"]
    good, skewed = tree["models"]["fx/good-750m"]["dir"], tree["models"]["fx/skewed-360m"]["dir"]
    sit_mab(good)
    sit_mab(skewed, answer=lambda q: "```sql\nSELECT 1 FROM t\n```" if q["id"].startswith(
        "sql-") else "Answer: I don't know")
    for d in (good, skewed):
        mab.write(d, mab.mark(d))
    # Mobile-MMLU-Pro on our key (14.3): good-750m misses one kept question,
    # skewed-360m labelled the key
    k = Path(config.MMP_DIR)
    k.mkdir(parents=True, exist_ok=True)
    (k / "mobile-mmlu-pro.csv").write_bytes(FIXTURE.read_bytes())
    pin_fixture(mp)
    a_key()
    for md, picks in ((good, dict(RIGHT, inv00001="C")), (skewed, RIGHT)):
        sit_picks(md, picks)
        mmp.collect(md)
    db.ai_set("labeller:third", fake_pin("fx/skewed-360m"), "masein")
    fake = FakeServer()
    _api(live, "/api/served", {"name": "DM k4-LDA phone build, MTP", "base_url": fake.base,
                               "thinking": "off", "based_on": "Qwen/Qwen3.6-35B-A3B",
                               "how": "llama.cpp fork, k=4 + LDA, MTP 3", "phone": True,
                               "by": "masein"})
    fake.close()
    named = {**SERVED, "phone": True, "name": "DM k4-LDA phone build, MTP"}
    _row(config.OUT_DIR, SETUP, OFF, named)
    _row(config.OUT_DIR, SETUP, ON, named, thinking=True)
    for th, acc in ((False, Q_OFF), (True, Q_ON)):
        _dm_results(QWEN, th)
        _row(config.OUT_DIR, QWEN, acc, HF, thinking=th)
    # the phone build's IFEval, as lm_eval's local-chat-completions leaves it
    d = config.OUT_DIR / SETUP.replace("/", "__") / "ifeval_0shot" / "served"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results_2026-09-27T10-00-00.000000.json").write_text(json.dumps({
        "results": {"ifeval": {"alias": "ifeval", "prompt_level_strict_acc,none": 0.62}},
        "config": {"model": "local-chat-completions", "model_args": f"pretrained={SETUP},x=1"},
        "served": served.view(served.get(SETUP)), "date": 2.0}))
    appmod._cache.update(key=None, payload=None, at=0.0)


# the old renderers' tables: each row by its id (Latest: its report label),
# each cell by its column's key, as the cell read
OLD = """() => {
  const out = {};
  const card = document.querySelector('[data-lb-card]');
  for (const t of card.querySelectorAll('table')) {
    const head = t.querySelector('thead tr.names') || t.querySelector('thead tr:last-child');
    if (!head) continue;
    const keys = [...head.children].map(th => th.dataset.col || th.dataset.frCol
      || th.dataset.latestCol || th.textContent.replace(/\\s+/g, ' ').trim());
    for (const tr of t.querySelectorAll('tbody tr')) {
      const id = tr.dataset.lbRow || tr.dataset.frRow || tr.dataset.latestRow || tr.dataset.dmRow;
      if (!id) continue;
      const row = out[id] = out[id] || {};
      [...tr.children].forEach((td, i) => {
        if (keys[i]) row[keys[i]] = td.innerText.replace(/\\s+/g, ' ').trim(); });
    }
  }
  return out;
}"""


def capture(page, live, js: str) -> dict:
    page.set_viewport_size({"width": 1400, "height": 1000})
    out = {}
    for v in VIEWS:
        page.goto("about:blank")
        page.goto(live["base"] + "/#tab=models&" + v)
        page.wait_for_selector("[data-lb-card] table tbody tr", timeout=30000)
        page.wait_for_timeout(300)
        out[v] = page.evaluate(js)
    return out


@pytest.fixture(scope="module")
def board(live):
    from service import db
    mp = pytest.MonkeyPatch()
    world27(live, mp)
    yield
    db.ai_set("labeller:third", None, "masein")
    mp.undo()


@pytest.mark.skipif(not os.environ.get("RECORD27"), reason="records the old renderers' values")
def test_record_the_values_before(live, page, board):
    got = capture(page, live, OLD)
    BEFORE.parent.mkdir(parents=True, exist_ok=True)
    BEFORE.write_text(json.dumps(got, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    assert all(got[v] for v in VIEWS), [v for v in VIEWS if not got[v]]


NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def numbers(text: str) -> list[str]:
    """the numbers a cell shows, its ± error's apart"""
    main = re.split(r"\s*±", text or "")[0]
    return NUM.findall(main.replace("−", "-"))


# modelTable's tables: each row by its model's id (Latest: its report labels; a
# published twin by its model; our run of DeviceMark's model by theirs too),
# each cell by its column, its words and its hover together
NEW = """() => {
  const rows = {}, heads = {};
  for (const t of document.querySelectorAll('[data-lb-card] table[data-model-table]')) {
    for (const th of t.querySelectorAll('thead tr.mt-hrow th[data-mt-col]'))
      heads[th.dataset.mtCol] = (th.querySelector('.mt-h1') || th).textContent
        .replace(/[▲▼]/g, '').trim();
    for (const tr of t.querySelectorAll('tbody tr.mt-row')) {
      const ids = [tr.dataset.lbRow, (tr.dataset.frRow || '').replace(/\\|published$/, ''),
        ...(tr.dataset.latestRows || tr.dataset.latestRow || '').split('|'), tr.dataset.dmRow,
        tr.dataset.dmPair].filter(Boolean);
      for (const td of tr.querySelectorAll('td[data-mt-col]')) {
        const vis = td.innerText, all = [vis, td.dataset.tip || '', td.title || ''].join(' ');
        for (const id of new Set(ids)) {
          const row = rows[id] = rows[id] || {};
          const was = row[td.dataset.mtCol] || { vis: '', all: '' };
          row[td.dataset.mtCol] = { vis: was.vis + ' ' + vis, all: was.all + ' ' + all };
        }
      }
    }
  }
  return { rows, heads, keys: Object.fromEntries([...new Set(Object.values(rows)
    .flatMap(r => Object.keys(r)))].map(k => [k, k])) };
}"""
SKIP = {"Model", "name", "Model / setup", "#", "Budget", "rank"}


def nums(text: str) -> list[str]:
    """the numbers a cell says, in order — its labels' parentheses and its ±
    aside, commas out and a whole number's ".0" too"""
    t = re.sub(r"\([^)]*\)", " ", text or "").replace(",", "")
    t = re.sub(r"±\s*[\d.]+", " ", t)
    return [re.sub(r"\.0$", "", n) for n in NUM.findall(t)]


def canon(text: str) -> set[str]:
    return set(nums(text))


def test_no_number_changes(live, page, board):
    before = json.loads(BEFORE.read_text())
    page.goto(live["base"] + "/")
    page.evaluate("() => { localStorage.setItem('bench-mt-thinking', 'both'); }")
    lost = []
    for v in VIEWS:
        page.set_viewport_size({"width": 1400, "height": 1000})
        page.goto("about:blank")
        page.goto(live["base"] + "/#tab=models&" + v)
        page.wait_for_selector("[data-lb-card] table[data-model-table] tbody tr", timeout=30000)
        for b in page.locator("[data-mt-empty-toggle]").all():
            if b.inner_text() == "show them":
                b.click()
        page.wait_for_timeout(300)
        got = page.evaluate(NEW)
        heads = {h.lower(): k for k, h in got["heads"].items()}
        keys = page.evaluate("ks => Object.fromEntries(ks.map(k => [k, mtBenchKey(k)]))",
                             sorted({k for r in before[v].values() for k in r}))
        for rid, cells in before[v].items():
            for k, text in cells.items():
                if k in SKIP or not canon(text):
                    continue
                nk = k if k in got["keys"] else "src" if k == "Source" else keys.get(k) \
                    if keys.get(k) in got["keys"] else heads.get(k.lower())
                cell = (got["rows"].get(rid) or {}).get(nk) or {"vis": "", "all": ""}
                # the cell's own number in sight; every other it said, in it or its hover
                if nums(text)[0] not in canon(cell["vis"]) or not canon(text) <= canon(cell["all"]):
                    lost.append((v, rid, k, text, cell["vis"]))
    assert lost == [], "\n".join(map(str, lost[:40]))

