"""17i on the page, part 3, point 20: a model marked public says what the
mark does — the export writes its raw runs to public/ once the list is
answered yes, and the upload sends public/ for anyone to download."""

from __future__ import annotations

import pytest

from conftest import set_name

pytestmark = pytest.mark.dashboard
SID = "served/Qwen3.6-35B-A3B-BF16"


def test_20_a_marked_models_page_says_what_will_happen(live, page):
    import service.app as appmod
    from service import db, served
    db.served_put({"id": SID, "name": "Qwen3.6 35B A3B BF16", "base_url": "", "key": "",
                   "how": "unsloth's BF16 file, unmodified", "based_on": "Qwen/Qwen3.6-35B-A3B",
                   "thinking": "auto", "pin": {"file": "Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf"},
                   "rented_only": True, "by": "masein", "at": 0})
    served.write_meta(db.served_get(SID))
    db.public_set(SID, True, "masein")
    appmod._cache.update(key=None, payload=None, at=0.0)
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + f"/#model={SID}")
    set_name(page, "masein")
    line = page.locator("[data-public-weights='1']")
    line.wait_for()                                   # 511854e: "may be published", no more
    words = line.inner_text()
    assert "writes its runs to public/ once you type yes to the list it prints" in words
    assert "for anyone to download" in words and "Clear the mark before the export" in words
    assert page.errors == []
