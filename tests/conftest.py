"""Shared fixtures.

The synthetic tree is built once per session and shared read-only. Tests that
need to change a tree underneath the service (the freshness tests) build their
own in a function-scoped temp dir.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "clients", ROOT / "tests" / "fixtures"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import make_fixture  # noqa: E402


@pytest.fixture(scope="session")
def tree(tmp_path_factory) -> dict:
    """The fixture tree, diagnosed (all but the odd one out) and frozen into a
    single-file report. See make_fixture.build for the manifest's shape."""
    root = tmp_path_factory.mktemp("bench")
    return make_fixture.build(root, report=root / "report.html")


@pytest.fixture(scope="session")
def diag(tree) -> dict[str, dict]:
    """model id -> parsed diagnose.json, for the models that have one."""
    out = {}
    for mid, m in tree["models"].items():
        f = m["dir"] / "diagnose.json"
        if f.exists():
            out[mid] = json.loads(f.read_text(encoding="utf-8"))
    return out


@pytest.fixture(scope="session")
def payload(tree) -> dict:
    """What GET /api/results would serve for the fixture tree."""
    import report_lm_eval as report
    runs = report.load_results(tree["out_dir"])
    cal_path = tree["out_dir"] / "judge_calibration.json"
    cal = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else None
    return report.build_payload(report.merge_runs(runs), "Fixture board",
                                source=str(tree["out_dir"]), calibration=cal,
                                taint=tree["taint"], parents=tree["parents"],
                                judge_identity={"provider": "stub", "model": "overlap-v1",
                                                "id": "stub/overlap-v1", "family": "stub"})


def make_service(root: Path, monkeypatch, *, llm_provider: str = "fake", tree: bool = True,
                 diagnose: bool = True, judged: bool = True, judge_model: str = ""):
    """The app against `root` as BENCH_ROOT: fixture tree built (optionally
    diagnosed), the GPU worker never started, the LLM poller not threaded
    (tests drive llm_poller.tick() by hand), LLM backend as asked. Returns
    (TestClient, app module, manifest)."""
    from fastapi.testclient import TestClient

    from service import config, llm, llm_poller, worker
    import service.app as appmod
    manifest = make_fixture.build(root, diagnose=diagnose, judged=judged) if tree else None
    for name, val in {"BENCH_ROOT": root, "RESULTS_ROOT": root / "results",
                      "OUT_DIR": root / "results" / "full", "DB_PATH": root / "service.sqlite3",
                      "ARTIFACTS_DIR": root / "artifacts", "LOGS_DIR": root / "logs",
                      "DATASETS_DIR": root / "datasets", "SUBMIT_TOKEN": "",
                      "ALLOW_REMOTE_CODE": False, "LLM_PROVIDER": llm_provider,
                      "LLM_MODEL": "fake-1" if llm_provider == "fake" else "",
                      "LLM_API_KEY": "", "LLM_MAX_ITEMS_PER_BATCH": 200,
                      "LLM_DAILY_ITEM_CAP": 2000, "DATASET_QUOTA_GB": 20.0,
                      "JUDGE_MODEL": judge_model,
                      "EXAM_PROVIDER": "fake", "EXAM_MODEL": "fake-exam", "EXAM_API_KEY": "",
                      "EXAM_DIR": root / "exam",
                      "JUDGED_TASKS_DIR": root / "exam" / "tasks",
                      "EVERYDAY_TASKS_DIR": root / "everyday" / "tasks"}.items():
        monkeypatch.setattr(config, name, val)
    monkeypatch.setattr(worker, "start", lambda: None)
    monkeypatch.setattr(llm_poller, "start", lambda: None)
    monkeypatch.setattr(llm.FakeBatches, "polls_to_done", 1)
    monkeypatch.setattr(llm.FakeBatches, "responder", staticmethod(llm.default_responder))
    llm.reset()
    appmod._cache.update(key=None, payload=None, at=0.0)
    client = TestClient(appmod.app)
    client.__enter__()
    return client, appmod, manifest


def fresh(appmod) -> None:
    """Step past the payload's five-second debounce (load shedding, not the
    property under test)."""
    appmod._cache["at"] = 0.0


# ---------------------------------------------------------------------------
# A live service in a thread, per module, and a browser page that records its
# own errors — the LIVE dashboard fetches, so these need a real HTTP server.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live(tmp_path_factory):
    import uvicorn

    from service import config, llm, llm_poller, worker
    import service.app as appmod
    root = tmp_path_factory.mktemp("live")
    tree = make_fixture.build(root)
    saved = {k: getattr(config, k) for k in (
        "BENCH_ROOT", "RESULTS_ROOT", "OUT_DIR", "DB_PATH", "ARTIFACTS_DIR", "LOGS_DIR",
        "DATASETS_DIR", "SUBMIT_TOKEN", "LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_POLL_S",
        "EXAM_DIR", "EXAM_PROVIDER", "EXAM_MODEL", "EXAM_API_KEY", "JUDGED_TASKS_DIR",
        "EVERYDAY_TASKS_DIR")}
    for k, v in {"BENCH_ROOT": root, "RESULTS_ROOT": root / "results",
                 "OUT_DIR": root / "results" / "full", "DB_PATH": root / "service.sqlite3",
                 "ARTIFACTS_DIR": root / "artifacts", "LOGS_DIR": root / "logs",
                 "DATASETS_DIR": root / "datasets", "SUBMIT_TOKEN": "",
                 "LLM_PROVIDER": "fake", "LLM_MODEL": "fake-1", "LLM_API_KEY": "",
                 "LLM_POLL_S": 0.3, "EXAM_DIR": root / "exam", "EXAM_PROVIDER": "fake",
                 "EXAM_MODEL": "fake-exam", "EXAM_API_KEY": "",
                 "JUDGED_TASKS_DIR": root / "exam" / "tasks",
                 "EVERYDAY_TASKS_DIR": root / "everyday" / "tasks"}.items():
        setattr(config, k, v)
    worker_start = worker.start
    worker.start = lambda: None
    llm.reset()
    appmod._cache.update(key=None, payload=None, at=0.0)
    # an upload from the page must not land in the developer's checkout: the
    # service prefers the repo's rubrics directory when it can write there,
    # and on this machine it can
    rubric_store = appmod._rubric_store
    appmod._rubric_store = lambda: (root / "rubrics", False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(appmod.app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "/healthz", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("live server did not come up")
    yield {"base": base, "tree": tree, "root": root}
    server.should_exit = True
    th.join(5)
    llm_poller.stop()
    worker.start = worker_start
    appmod._rubric_store = rubric_store
    for k, v in saved.items():
        setattr(config, k, v)
    llm.reset()


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser):
    # 11f gave the board motion; a test about what a page says or does runs
    # with reduced motion, so a panel that is still sliding in is never "not
    # stable" to a click. The motion itself is tested with it switched on
    # (test_masein_seven_11f.py), in contexts of its own.
    ctx = browser.new_context(viewport={"width": 1240, "height": 900}, reduced_motion="reduce")
    pg = ctx.new_page()
    # a shorter wait for a quick triage run: PW_TIMEOUT_MS=5000
    if os.environ.get("PW_TIMEOUT_MS"):
        pg.set_default_timeout(int(os.environ["PW_TIMEOUT_MS"]))
    errors = []
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.on("console", lambda m: errors.append(f"console.error: {m.text}")
          if m.type == "error" else None)
    pg.errors = errors
    yield pg
    ctx.close()


# 12b: the old tab names, and where each one lives now (the brief's §1)
PLACE_OF = {
    "Overview": ("home", None), "Home": ("home", None),
    "Leaderboard": ("models", None), "Models": ("models", None),
    "Loop": ("improve", "topics"), "Review": ("improve", "review"),
    "Training": ("improve", "training"),
    "Exam": ("benchmarks", "exam"), "Tasks": ("benchmarks", "standard"),
}


def go_tab(page, label: str) -> None:
    """A tab by its old name, reached the way a person reaches it now: a
    place in the header (Menu ▾ on a phone) and its switch; the run
    counter's All runs; the name menu's Data & sources (12b)."""
    page.wait_for_selector("#tabs [role=tab]", state="attached")   # built on first render
    if label in PLACE_OF:
        place, sub = PLACE_OF[label]
        if page.locator("#menuBtn").is_visible():
            page.locator("#menuBtn").click()
            page.locator(f"#pop-places [data-place='{place}']").click()
        else:
            page.locator(f"#tabs [role=tab][data-tab='{place}']").click()
        if sub:
            page.locator(f"[data-subswitch] [data-sub='{sub}']").click()
        return
    if label == "Perplexity & Loss":
        go_tab(page, "Models")
        page.locator("[data-chip='lm']").click()
        return
    if label == "Queue":
        page.locator("#runs [data-runs]").click()
        page.locator("[data-all-runs-link]").click()
        return
    if label == "Provenance":
        page.locator("#who button.who").click()
        page.locator("#pop-who [data-menu='data']").click()
        return
    raise AssertionError(f"no place for the old tab {label!r}")


def open_filters(page) -> None:
    """Models' Kind, Size, Status, Columns, Models and Scale live in
    Filters ▾ at every width (12b): open it if it is not open."""
    if not page.locator("[data-filter-sheet]").count():
        page.locator("[data-filters]").click()
    page.locator("[data-filter-sheet]").wait_for()


def open_submit(page, base: str, name: str | None = None) -> None:
    """Queue ▸ Submit a model is the Test a model dialog now (12b): the old
    address #tab=submit opens it, over All runs. A name to record is typed
    in the header first — the dialog covers the header while it is open."""
    if name:
        page.goto(base + "/")
        set_name(page, name)
    page.goto(base + "/#tab=submit")
    page.wait_for_selector("[data-dialog='test'] [data-submit-form]")


def bar_reveal(page, sel: str) -> bool:
    """11e hid the checks, the name and the theme behind ⋯ at 400 px; 12b
    keeps the right side on the bar at every width. Nothing to reveal."""
    return False


def set_name(page, name: str) -> None:
    """The one name, in the header, that every action records (phase 9b).
    With a name already set the box is a popover on the body (11a), not an
    input inside the header."""
    who = page.locator("#who")
    who.wait_for(state="attached")
    opened = bar_reveal(page, "#who")
    if not who.locator("input").count():
        who.locator("button[data-who]").click()
        page.wait_for_selector("#pop-who input")
    box = page.locator("#pop-who") if page.locator("#pop-who input").count() else who
    box.locator("input").fill(name)
    box.locator("input").press("Enter")
    page.wait_for_selector(f"#who button[data-who='{name}']", state="attached")
    if opened and page.locator("#bar[data-more='open']").count():
        page.locator("#barMore").click()              # and put the ⋯ panel away again


def show_all_columns(page) -> None:
    """The Leaderboard shows six task columns by default (phase 9c); a test
    about a column that may be hidden asks for all of them first."""
    if page.locator("[data-filters]").count():            # 12b: the pills are in Filters ▾
        open_filters(page)
    menu = page.locator("[data-columns-menu]")
    if menu.count() == 0 or not menu.get_attribute("data-hidden-tasks"):
        return
    # the Columns pill opens 11a's popover (11c); Show all applies at once
    if page.locator("#pop-columns").count() == 0:
        menu.click()
    page.locator("#pop-columns [data-show-all]").click()
    page.wait_for_function("!document.querySelector('[data-hidden-tasks]')")
    page.keyboard.press("Escape")


def choose(ctl, value) -> None:
    """Pick a value in the board's Select or Combobox (11f: no native <select>
    is left in the view) the way a person does: open it, click the option."""
    page = ctl.page
    ctl.click()
    opt = page.locator(f"[role=listbox] [role=option][data-value={json.dumps(str(value))}]").first
    opt.wait_for()
    opt.click()
    page.wait_for_function("([el, v]) => (el.dataset.value || '') === v || !el.isConnected",
                           arg=[ctl.element_handle(), str(value)])


def choice(ctl) -> str:
    """What a Select or Combobox holds now."""
    return ctl.get_attribute("data-value") or ""


def all_rows(page, key: str, n: int = 100) -> None:
    """A table behind the shared pager (25 a page since 36 topics, 10c), all
    on one page: for a test about what the whole table holds."""
    sel = page.locator(f"[data-pager='{key}'] [aria-label='rows per page']")
    if sel.count() and sel.is_visible():
        choose(sel, str(n))
        page.wait_for_function(f"document.querySelector(\"[data-pager='{key}'] "
                               f"[data-page-range]\").dataset.pageRange.startsWith('1-')")


# ---------------------------------------------------------------------------
# The split's rule, over the bodies actually sent. A report-half question may
# reach the judge and nothing else — and a question is more than its prompt:
# the 37-topic banks' `intent` is a sentence describing the question, and in
# 10b four of them were going into every proposal and generation request
# while the leak tests looked only for prompts. So these look for every free
# text a report-half row carries, under whatever name its author gave it.
# ---------------------------------------------------------------------------

REPORT_TEXT_MIN = 12          # "yes no" is not a question's; a phrase of this length is


def report_half_text(rows: list[dict]) -> list[tuple[str, str, str]]:
    """(qid, field, text) for every free-text value a report-half row
    carries: its prompt, its reference line, its notes, and each metadata
    value that is a phrase rather than a label — `intent`, `domain`, or a
    field no bank has used yet. A phrase that a diagnose-half row carries too
    is not the report half's own, and is left out."""
    import exam_build as eb

    def texts(r):
        vals = {"prompt": r.get("prompt"), "reference": r.get("reference"),
                "notes": r.get("notes")}
        vals.update({f"meta.{k}": v for k, v in (r.get("meta") or {}).items()})
        return {k: v.strip() for k, v in vals.items()
                if isinstance(v, str) and len(v.strip()) >= REPORT_TEXT_MIN
                and any(c.isspace() for c in v.strip())}
    shared = {t for r in rows if eb.half_of(r["qid"]) == "diagnose" for t in texts(r).values()}
    return [(r["qid"], k, t) for r in rows if eb.half_of(r["qid"]) == "report"
            for k, t in texts(r).items() if t not in shared]


def assert_no_report_half_text(body: str, rows: list[dict]) -> int:
    """Nothing of any report-half row in `body`: no qid, and none of its free
    text, whole or its first sixty characters. Returns how many texts were
    checked, so a test can say it checked something."""
    import exam_build as eb
    fields = report_half_text(rows)
    leaked = [(q[:12], f, t[:70]) for q, f, t in fields
              if t in body or (len(t) > 60 and t[:60] in body)]
    assert not leaked, f"report-half text reached a request: {leaked[:5]}"
    qids = [r["qid"] for r in rows if eb.half_of(r["qid"]) == "report" and r["qid"] in body]
    assert not qids, f"report-half qids reached a request: {qids[:5]}"
    return len(fields)


def without_its_own_rubric(monkeypatch, tmp_path, slug: str = "arts"):
    """Every topic now has its own rubric — Arts, the last, arrived on
    2026-09-22 — but a topic without one is still graded by the shared
    exam.md, and the page still says so. A test about that path takes one
    topic's two files out of the repo copy the judge reads (a copy, never the
    checkout). Arts by default: the fixture leaves its bank empty too."""
    import shutil

    import exam_build
    import judge
    d = tmp_path / f"repo-rubrics-without-{slug}"
    if not d.exists():
        shutil.copytree(judge.RUBRIC_DIR, d,
                        ignore=shutil.ignore_patterns(f"{slug}.md", f"{slug}.criteria.json"))
    monkeypatch.setattr(judge, "RUBRIC_DIR", d)
    monkeypatch.setattr(exam_build, "RUBRIC_DIR", d)
    return d


@pytest.fixture
def arts_without_rubric(monkeypatch, tmp_path):
    """without_its_own_rubric for Arts, as a fixture: the repo copy the judge
    reads, less arts.md and arts.criteria.json."""
    return without_its_own_rubric(monkeypatch, tmp_path, "arts")


# ---------------------------------------------------------------------------
# A topic whose questions carry domain labels. The 37-topic banks label every
# question with one — Physics & Astronomy has twelve — and that is what the
# focus plan spreads documents over (11a). The fixture's drafted questions
# carry no metadata at all, so a test that needs labels writes them on.
# ---------------------------------------------------------------------------

DOMAIN_LABELS = ("Market Structure", "Trade and Money", "Growth and Development")


def label_domains(exam_root, topic: str, labels=DOMAIN_LABELS, both_halves: bool = True):
    """Put a `domain` from a small closed set on every question of `topic`,
    round-robin. Returns (path, original bytes) so a test running against the
    shared tree can put the bank back. Every label lands in both halves, as a
    real bank's does — a label only the report half carries would be report-
    half text, and no request may hold it."""
    import json as _json

    import categories as _categories
    import exam_build as eb
    p = eb.bank_dir(Path(exam_root)) / f"{_categories.topic_slug(topic)}.jsonl"
    was = p.read_bytes()
    rows = [_json.loads(x) for x in was.decode("utf-8").splitlines() if x.strip()]
    for i, r in enumerate(rows):
        r.setdefault("meta", {})["domain"] = labels[i % len(labels)]
    p.write_text("".join(_json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                 encoding="utf-8")
    if both_halves:
        seen = {}
        for r in rows:
            seen.setdefault(r["meta"]["domain"], set()).add(eb.half_of(r["qid"]))
        assert all(h == {"report", "diagnose"} for h in seen.values()), seen
    return p, was
