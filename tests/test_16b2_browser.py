"""16b.2 on the page: a model's actions — Test this model · Chat · Download —
and Download's panel: its file, size and sha256, a link made for this one
download, a command that resumes and never holds the token, and the switch
for the person who added it. A Hugging Face model's link to Hugging Face;
a served model's one line. At 1400 and 375 px; nothing runs."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import set_name
from service import config, uploads
from test_14_3_browser import no_sideways, steady_shot
from test_16_1_sizes import gguf_bytes

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase16b2"


@pytest.fixture(scope="module")
def phone(live, tmp_path_factory):
    """a GGUF uploaded by masein, downloads on"""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "UPLOADS_DIR", tmp_path_factory.mktemp("dl") / "uploads")
        mp.setattr(config, "UPLOAD_FREE_GB", 0.0)
        yield from _phone(live)


def _phone(live):
    import json
    import time
    import urllib.request
    data = gguf_bytes() + bytes(range(256)) * 40

    def call(method, path, body=None, raw=None):
        req = urllib.request.Request(live["base"] + path, method=method,
                                     data=raw if raw is not None else
                                     json.dumps(body).encode() if body is not None else None,
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    rec = call("POST", "/api/uploads", {"filename": "dl-phone.gguf", "size": len(data),
                                        "mtime": 2.0, "by": "masein"})
    call("PUT", f"/api/uploads/{rec['id']}?offset=0", raw=data)
    call("POST", f"/api/uploads/{rec['id']}/finish", {})
    for _ in range(200):
        if call("GET", f"/api/uploads/{rec['id']}")["state"] == "ready":
            break
        time.sleep(0.02)
    got = call("POST", f"/api/uploads/{rec['id']}/add", {"name": "DL phone", "how": "Q4",
                                                        "by": "masein"})
    import service.app as appmod
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield got["id"], data
    uploads.delete(got["id"])
    appmod._cache.update(key=None, payload=None, at=0.0)


def open_model(page, live, mid, width=1400, who="masein"):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, who)
    page.goto(live["base"] + "/#model=" + quote(mid, safe=""))
    page.wait_for_selector(f"[data-model-actions='{mid}']")


@pytest.mark.parametrize("width", [1400, 375])
def test_a_models_actions_and_its_download(live, page, phone, width):
    mid, data = phone
    open_model(page, live, mid, width)
    acts = page.locator(f"[data-model-actions='{mid}'] button")
    assert acts.all_inner_texts()[:3] == ["Measure this model", "Chat", "Download"]
    page.locator(f"[data-act-download='{mid}']").click()
    panel = page.locator(f"[data-dl-panel='{mid}']")
    panel.locator("[data-dl-facts]").wait_for()
    import hashlib
    assert panel.locator("[data-dl-facts]").inner_text().startswith("DL-phone.gguf · 0.0 GB")
    assert panel.locator("[data-dl-sha]").inner_text() == "sha256 " + hashlib.sha256(data).hexdigest()
    cmd = panel.locator("[data-dl-cmd]").inner_text()
    assert cmd.startswith('curl -C - -fL -H "X-Token: $BOARD_TOKEN" -H "X-Who: masein" '
                          "-o 'DL-phone.gguf' '") and "api/download?model=gguf%2FDL-phone'" in cmd
    assert "token=" not in cmd
    # its adder sees the switch
    assert panel.locator(f"[data-dl-allow='{mid}']").is_checked()
    no_sideways(page)
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(panel, SCREENS / f"download-{width}.png")
    # Download: a link made for this one download, and the browser saves it
    with page.expect_download() as dl:
        panel.locator(f"[data-dl-go='{mid}']").click()
    assert dl.value.suggested_filename == "DL-phone.gguf"
    assert Path(dl.value.path()).read_bytes() == data
    page.wait_for_function("+(document.querySelector('[data-dl-count]') || {dataset: {}})"
                           ".dataset.dlCount >= 1")
    assert page.errors == []


def test_someone_else_sees_no_switch_and_off_is_said(live, page, phone):
    mid, _ = phone
    open_model(page, live, mid, who="omar")
    page.locator(f"[data-act-download='{mid}']").click()
    panel = page.locator(f"[data-dl-panel='{mid}']")
    panel.locator("[data-dl-facts]").wait_for()
    assert panel.locator("[data-dl-allow]").count() == 0
    # switched off by masein, from the API: said, and no Download (on again after)
    import json
    import urllib.request
    req = urllib.request.Request(live["base"] + "/api/models/file/allow", method="POST",
                                 data=json.dumps({"model": mid, "allowed": False,
                                                  "by": "masein"}).encode(),
                                 headers={"content-type": "application/json"})
    urllib.request.urlopen(req).read()
    page.evaluate(f"dlLoad({json.dumps(mid)})")
    panel.locator("[data-dl-off]").wait_for()
    assert panel.locator("[data-dl-off]").inner_text() == "Downloads are switched off by masein."
    assert panel.locator("[data-dl-go]").count() == 0
    req = urllib.request.Request(live["base"] + "/api/models/file/allow", method="POST",
                                 data=json.dumps({"model": mid, "allowed": True,
                                                  "by": "masein"}).encode(),
                                 headers={"content-type": "application/json"})
    urllib.request.urlopen(req).read()
    assert page.errors == []


def test_a_hub_model_links_to_hugging_face(live, page):
    open_model(page, live, "fx/good-750m")
    page.locator("[data-act-download='fx/good-750m']").click()
    a = page.locator("[data-dl-hub='fx/good-750m']")
    a.wait_for()
    assert a.inner_text() == "Get it on Hugging Face ↗"
    assert a.get_attribute("href").startswith("https://huggingface.co/fx/good-750m")
    assert page.locator("[data-dl-hub-line]").inner_text().endswith(
        "The board doesn’t copy Hugging Face’s files.")
    assert page.errors == []
