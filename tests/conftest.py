"""Shared fixtures.

The synthetic tree is built once per session and shared read-only. Tests that
need to change a tree underneath the service (the freshness tests) build their
own in a function-scoped temp dir.
"""

from __future__ import annotations

import json
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
                      "JUDGED_TASKS_DIR": root / "exam" / "tasks"}.items():
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
        "EXAM_DIR", "EXAM_PROVIDER", "EXAM_MODEL", "EXAM_API_KEY", "JUDGED_TASKS_DIR")}
    for k, v in {"BENCH_ROOT": root, "RESULTS_ROOT": root / "results",
                 "OUT_DIR": root / "results" / "full", "DB_PATH": root / "service.sqlite3",
                 "ARTIFACTS_DIR": root / "artifacts", "LOGS_DIR": root / "logs",
                 "DATASETS_DIR": root / "datasets", "SUBMIT_TOKEN": "",
                 "LLM_PROVIDER": "fake", "LLM_MODEL": "fake-1", "LLM_API_KEY": "",
                 "LLM_POLL_S": 0.3, "EXAM_DIR": root / "exam", "EXAM_PROVIDER": "fake",
                 "EXAM_MODEL": "fake-exam", "EXAM_API_KEY": "",
                 "JUDGED_TASKS_DIR": root / "exam" / "tasks"}.items():
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
    ctx = browser.new_context(viewport={"width": 1240, "height": 900})
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.on("console", lambda m: errors.append(f"console.error: {m.text}")
          if m.type == "error" else None)
    pg.errors = errors
    yield pg
    ctx.close()


def go_tab(page, label: str) -> None:
    """A tab by its name: one of the six, or one under More ▾ (phase 9b)."""
    page.wait_for_selector("#tabs #moreBtn")              # the bar is built on first render
    t = page.get_by_role("tab", name=label, exact=True)
    if not t.count():
        page.locator("#moreBtn").click()
        t = page.get_by_role("menuitem", name=label, exact=True)
    t.click()


def set_name(page, name: str) -> None:
    """The one name, in the header, that every action records (phase 9b)."""
    who = page.locator("#who")
    who.wait_for()
    if not who.locator("input").count():
        who.locator("button[data-who]").click()
    who.locator("input").fill(name)
    who.locator("input").press("Enter")
    page.wait_for_selector(f"#who button[data-who='{name}']")


def show_all_columns(page) -> None:
    """The Leaderboard shows six task columns by default (phase 9c); a test
    about a column that may be hidden asks for all of them first."""
    menu = page.locator("[data-columns-menu]")
    if menu.count() == 0:
        return
    if menu.get_attribute("open") is None:
        menu.locator("summary").click()
    menu.get_by_role("button", name="show all").click()
    page.wait_for_function("!document.querySelector('[data-hidden-tasks]')")


def pick_topic(page, task: str) -> None:
    """The model page shows one judged topic's tables at a time (phase 9c),
    chosen in a select since there are thirty-six of them (10c)."""
    sel = page.locator("select[data-topic-switch]")
    if sel.count():
        sel.select_option(task)
        page.wait_for_function("document.querySelector('select[data-topic-switch]').value === "
                               f"'{task}'")


def all_rows(page, key: str, n: int = 100) -> None:
    """A table behind the shared pager (25 a page since 36 topics, 10c), all
    on one page: for a test about what the whole table holds."""
    sel = page.locator(f"[data-pager='{key}'] select[aria-label='rows per page']")
    if sel.count():
        sel.select_option(str(n))
        page.wait_for_function(f"document.querySelector(\"[data-pager='{key}'] "
                               f"[data-page-range]\").dataset.pageRange.startsWith('1-')")
