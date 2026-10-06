"""17f on the page: Save asks the name in the dialog and says its word beside
it (12); Runs says where each run ran, filters by it (13), and lists what the
rented boxes are doing (16). Nothing runs; no address is shown."""

from __future__ import annotations

import time

import pytest

from conftest import open_add

pytestmark = pytest.mark.dashboard


def test_12_save_with_no_name_asks_it_in_the_dialog_and_says_why_beside_save(live, page):
    page.set_viewport_size({"width": 1200, "height": 800})
    page.goto(live["base"] + "/#tab=home")
    page.evaluate("localStorage.removeItem('bench-name')")
    page.reload()
    open_add(page, live["base"], "server")
    page.wait_for_selector("[data-srv='name']")
    page.fill("[data-srv='name']", "a served build")
    page.fill("[data-srv='base_url']", "http://127.0.0.1:9/v1")
    page.locator("[data-srv-save]").click()
    # b366baf: the name box opened in the header, behind the dialog, and
    # nothing seemed to happen
    name = page.locator("[data-dialog='test'] [data-srv-name]")
    name.wait_for()
    msg = page.locator("[data-dialog='test'] [data-srv-msg]")
    assert "Your name is recorded with what you save" in msg.inner_text()
    assert not page.locator("#pop-who").is_visible()
    # the word beside Save — in Save's own row — and in sight
    assert page.locator(".frm:has([data-srv-save]) [data-srv-msg]").count() == 1
    box = msg.bounding_box()
    assert box and 0 <= box["y"] and box["y"] + box["height"] <= 800
    name.fill("masein")
    name.press("Enter")
    page.wait_for_function("(document.querySelector('[data-srv-msg]') || {}).textContent"
                           " && !document.querySelector('[data-srv-msg]').textContent"
                           ".startsWith('Your name')", timeout=30000)
    assert page.locator("[data-srv-msg]").is_visible()
    # the refusal of a server that isn't there is the only error
    assert all("status of 422" in e for e in page.errors), page.errors


def test_13_16_runs_says_where_each_ran_and_what_the_boxes_are_doing(live, page):
    import import_frontier as imf
    from service import db
    here = db.add("served/board-box", "instruct", "frontier", "masein", "a board run",
                  status="done")
    away = db.add("served/board-box", "instruct", "frontier", "masein",
                  "imported from a rented GPU (NVIDIA GeForce RTX 5090) · box A3 · x.tar.gz",
                  status="done")
    db.update(away, where_ran="rented GPU · RTX 5090 · box A3", finished_at=time.time())
    now = time.time()
    imf.store_boxes([
        {"label": "A5", "model": "served/board-box", "step": "A5-1", "thinking": "on",
         "tasks": ["mmlupro_tiger"], "state": "asking", "at": now - 60, "seen_at": now,
         "line": "MMLU-Pro 3,000 of 12,032 · 5.3 s an answer · 13.3 h left", "sessions": 1,
         "reachable": True, "safe": False},
        {"label": "A9", "model": "served/board-box", "step": "A9-2", "thinking": "off",
         "tasks": ["hle_text_cais"], "state": "whole", "at": now - 300, "seen_at": now,
         "line": "", "sessions": 1, "reachable": True, "safe": True}])
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=runs")
    page.wait_for_selector(f"[data-run-where='{away}']")
    assert page.locator(f"[data-run-where='{away}']").inner_text() == \
        "rented GPU · RTX 5090 · box A3"
    assert page.locator(f"[data-run-where='{here}']").inner_text() == "this server"
    # the filter: rented GPUs only
    page.get_by_label("where filter").click()
    page.locator("[role=listbox][aria-label='where filter'] [data-value='rented']").click()
    page.wait_for_function(f"!document.querySelector(\"[data-queue-row='{here}']\")")
    assert page.locator(f"[data-queue-row='{away}']").count() == 1
    # what the boxes are doing
    boxes = page.locator("[data-rented-boxes]")
    boxes.wait_for()
    a5 = page.locator("[data-rented-box='A5|A5-1']").inner_text()
    assert "3,000 of 12,032" in a5 and "MMLU-Pro" in a5
    assert "done, safe to destroy" in page.locator("[data-rented-box='A9|A9-2']").inner_text()
    assert "203.0.113" not in boxes.inner_text()
    assert page.errors == []
