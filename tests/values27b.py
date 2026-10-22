"""27b: no number changes on the Reports pages — every (report, table, model,
column) value the old renderers drew, recorded on a test module's world
(tests/fixtures/reports27b/before-<world>.json, `RECORD27B=1` writes it), and
checked against what modelTable and scoreCell draw now: each number a cell
showed is in the same model's cell of the same column (or the column it
moved into), its own number in sight and every other in the cell or its
hover. Used by test_27b_values_*_browser.py, one world each."""

from __future__ import annotations

import json
import re
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "reports27b"

# the old tables: [name, the table, how a row says its model]
OLD = """(specs) => {
  const out = {};
  const txt = e => (e.innerText || '').replace(/\\s+/g, ' ').trim();
  const tips = e => [e, ...e.querySelectorAll('[title]')].map(x => x.getAttribute('title') || '')
    .join(' ');
  for (const [name, sel] of specs) {
    const t = document.querySelector(sel);
    if (!t) continue;
    const rows = out[name] = {};
    if (t.matches('[data-pd-table]')) {
      for (const r of t.querySelectorAll('.pdx-row[data-pd-row]')) {
        const row = rows[r.dataset.pdRow] = {};
        for (const c of r.children) {
          const k = [...c.classList].find(x => /^pdx-(name|a|b|diff|verdict|split|wait)$/.test(x));
          if (k) row[k] = { text: txt(c), tip: tips(c) };
        }
      }
      continue;
    }
    const head = t.querySelector('thead tr:last-child');
    const keys = [...head.children].map(th => {
      const f = th.childNodes[0];
      return ((f && f.nodeType === 3 ? f.textContent : th.textContent) || '').replace(/\\s+/g, ' ')
        .trim(); });
    for (const tr of t.querySelectorAll('tbody tr')) {
      const id = tr.dataset.odRow || tr.dataset.rpRow || tr.dataset.ptPerRow || tr.dataset.ptRow
        || (tr.children[0] && tr.children[0].getAttribute('title'));
      if (!id || tr.classList.contains('rp-group')) continue;
      const row = rows[id] = {};
      [...tr.children].forEach((td, i) => { if (keys[i]) row[keys[i]] = { text: txt(td), tip: tips(td) }; });
    }
  }
  return out;
}"""

# the new tables: modelTable's, each row by the same hook, each cell by its
# column's key (the name's cell "name", its line in it), its words and hover
NEW = """(specs) => {
  const out = {};
  const txt = e => (e.innerText || '').replace(/\\s+/g, ' ').trim();
  const tips = e => [e, ...e.querySelectorAll('[title], [data-tip]')].map(x =>
    (x.getAttribute('title') || '') + ' ' + (x.dataset.tip || '')).join(' ');
  for (const [name, sel] of specs) {
    const t = document.querySelector(sel);
    if (!t) continue;
    const rows = out[name] = {}, heads = out[name + ':heads'] = {};
    if (t.matches('[data-pd-table]')) {
      for (const r of t.querySelectorAll('.pdx-row[data-pd-row]')) {
        const row = rows[r.dataset.pdRow] = {};
        for (const c of r.children) {
          const k = [...c.classList].find(x => /^pdx-(name|a|b|diff|verdict|split|wait)$/.test(x));
          if (k) row[k] = { vis: txt(c), all: txt(c) + ' ' + tips(c) };
        }
      }
      continue;
    }
    for (const th of t.querySelectorAll('thead tr.mt-hrow th[data-mt-col]'))
      heads[th.getAttribute('title') || th.dataset.mtCol] = th.dataset.mtCol;
    for (const tr of t.querySelectorAll('tbody tr.mt-row')) {
      // a row two sources make one answers to each one's label
      const ids = [tr.dataset.odRow, tr.dataset.rpRow, tr.dataset.rpRanoutRow, tr.dataset.ptPerRow,
        tr.dataset.ptRow,
        ...(tr.dataset.rpEveryRows || tr.dataset.rpEveryRow || '').split('|')].filter(Boolean);
      const nm = tr.querySelector('td.mt-name');
      for (const id of ids) {
        const row = rows[id] = {};
        row.name = { vis: txt(nm), all: txt(nm) + ' ' + tips(nm) };
        for (const td of tr.querySelectorAll('td[data-mt-col]'))
          row[td.dataset.mtCol] = { vis: txt(td), all: txt(td) + ' ' + tips(td) };
      }
    }
  }
  return out;
}"""

# 27b: a report's modelTable rows as the old tests read the old tables: each
# row by each label it answers to (a row two sources make, by both), then each
# cell's own number ("—" for none), a published one without its letter
MT_FIRST = """rows => rows.flatMap(r => {
  if (r.classList.contains('mt-group')) return [];
  const own = c => { const n = c.querySelector('.mt-n'), f = n ? n.firstChild : c.firstChild;
    return (f ? f.textContent : c.textContent).trim().replace(/^–$/, '—'); };
  if (r.classList.contains('mt-row')) {
    const ids = (r.dataset.rpEveryRows || r.dataset.rpEveryRow || r.dataset.rpRow
      || r.dataset.rpRanoutRow || '').split('|').filter(Boolean);
    const vs = [...r.querySelectorAll('td[data-mt-col]')].map(own);
    return ids.map(id => [id, ...vs]);
  }
  return [[r.children[0].title || r.children[0].textContent.trim(), ...[...r.children].slice(1)
    .map(c => c.firstChild && c.firstChild.nodeType === 3 ? c.firstChild.textContent.trim()
      : c.textContent.trim())]];
})"""

NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def nums(text: str) -> list[str]:
    """the numbers a cell says, in order — its labels' parentheses and its ±
    aside, commas out and a whole number's ".0" too (27's rule)"""
    t = re.sub(r"\([^)]*\)", " ", text or "").replace(",", "")
    t = re.sub(r"±\s*[\d.]+", " ", t)
    return [re.sub(r"\.0$", "", n) for n in NUM.findall(t)]


def canon(text: str) -> set[str]:
    """every number in a text, parentheses and ± included"""
    t = (text or "").replace(",", "").replace("±", " ")
    return {re.sub(r"\.0$", "", n) for n in NUM.findall(t)}


def capture(page, live, views: dict, js: str) -> dict:
    """each report's tables, as `js` reads them"""
    page.set_viewport_size({"width": 1400, "height": 1000})
    out = {}
    for rid, (wait, specs) in views.items():
        page.goto("about:blank")
        page.goto(live["base"] + "/#tab=reports&report=" + rid)
        page.wait_for_selector(wait, timeout=30000)
        page.wait_for_timeout(400)
        # modelTable hides a row with no score in its columns: shown, to be read
        for _ in range(4):
            more = page.locator("[data-mt-empty-toggle]").filter(has_text="show them")
            if not more.count():
                break
            more.first.click()
            page.wait_for_timeout(300)
        out[rid] = page.evaluate(js, specs)
    return out


def record(page, live, world: str, views: dict) -> dict:
    got = capture(page, live, views, OLD)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / f"before-{world}.json").write_text(
        json.dumps(got, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    return got


# where each old column went: On-device's base and kind into the name's line,
# its two sizes into Memory, its two phone columns into On a phone; the
# testers' table's phone, model and dates into the row's name and line
OD_COLS = {"Model": ["name"], "Base model": ["name"], "Size on disk": ["memory"],
           "Memory it needs": ["memory"], "Phone speed": ["phone"],
           "Measured on our testers’ phones": ["phone"]}


OD_TESTERS = "Measured on our testers’ phones"


# 27c: Every test's old columns — the received date, tester and what moved to
# the test's panel (first load, thermal max, charging) in the row's name and
# its hover; the battery's start → end and rise in two columns now
TESTS_COLS = {"Received": ["name"], "Tester": ["name"], "First load": ["name"],
              "Thermal max": ["name"], "Charging": ["name"], "Phone": ["phone"],
              "App · engine": ["app"], "Model": ["model"], "Result": ["result"],
              "Prefill, cold → warm": ["prefill"], "Battery": ["battery", "rise"]}


def where(rid: str, tname: str, col: str, heads: dict) -> list[str]:
    if tname == "pd":
        return [col]
    if tname == "tests":
        return TESTS_COLS.get(col) or [heads.get(col, col)]
    if tname == "od" and col in OD_COLS:
        return OD_COLS[col]
    if col in ("Model", "Phone", "Dates"):
        return ["name"]
    return [heads.get(col, col)]


def seen_only(rid: str, tname: str, row: str, col: str, cells: dict) -> bool:
    """numbers a new cell may keep on hover alone, as the brief asks: a phone
    speed reported by hand where our testers' phones measured the build (the
    median wins the cell); a phone note with no speed in it ("no phone number
    for the 4B"); and the phones after the first of our testers' (one phone's
    median shows, every phone's on hover)"""
    # 27c: Every test's first load, thermal max and charging moved to the test's
    # panel (and the row's hover); its prefill shows warm, cold on hover
    if tname == "tests":
        return col in ("First load", "Thermal max", "Charging", "Prefill, cold → warm")
    if tname != "od":
        return False
    if col == "Phone speed":
        return "tok/s" not in cells[col]["text"] or bool(
            nums((cells.get("Measured on our testers’ phones") or {}).get("text", "")))
    return col == "Measured on our testers’ phones"


def lost(page, live, world: str, views: dict) -> list:
    """what the new tables lost: (report, table, row, column, old text, new
    text) — each old number must be in the new cell or its hover, the cell's
    own in sight, and every ± error too"""
    before = json.loads((FIXTURES / f"before-{world}.json").read_text())
    after = capture(page, live, views, NEW)
    out = []
    for rid, tables in before.items():
        for tname, rows in tables.items():
            heads = after.get(rid, {}).get(tname + ":heads", {})
            for row, cells in rows.items():
                # 27c: a phone with no finished test of a model ("0 of 2") leaves the
                # medians for What failed
                if tname == "per" and (cells.get("Tests") or {}).get("text", "").startswith("0 of"):
                    continue
                for col, cell in cells.items():
                    want = nums(cell["text"])
                    # 27c: a test that crashed has no times: "–", not "0.0 s"
                    if tname == "tests" and col == "Time to first word" and (
                            cells.get("Result") or {}).get("text") != "ok":
                        continue
                    if not want:
                        continue
                    keys = where(rid, tname, col, heads)
                    got = [(after.get(rid, {}).get(tname, {}).get(row) or {}).get(k) for k in keys]
                    vis = " ".join(g["vis"] for g in got if g)
                    every = " ".join(g["all"] for g in got if g)
                    # a testers' cell: one phone's median in sight, any phone's; a
                    # model two sources give one number (27): the newer in sight,
                    # the other on hover as "also …"
                    shown = want[0] in canon(vis) or (seen_only(rid, tname, row, col, cells)
                                                      and (col != OD_TESTERS
                                                           or bool(set(want) & canon(vis)))) \
                        or f"also {want[0]}" in every
                    errs = set(re.findall(r"±\s*([\d.]+)", cell["text"]))
                    if not got or not any(got) or not shown or not set(want) <= canon(every) \
                            or not {re.sub(r"\.0$", "", e) for e in errs} <= canon(every):
                        out.append((rid, tname, row, col, cell["text"], vis))
    return out
