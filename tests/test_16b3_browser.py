"""16b.3 on the page: a model's Use as an API — its state, the address, its
name, a curl and a Python example that hold no key, and the viewer's keys:
Create my key (shown once), each one's use, Revoke. A base model says why it
isn't offered. At 1400 and 375 px; nothing runs."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import set_name
from test_14_3_browser import no_sideways, steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase16b3"
SMALL, BASE = "fx/below-135m-it", "fx/good-750m"


def open_api(page, live, mid, width=1400, who="masein"):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, who)
    page.goto(live["base"] + "/#model=" + quote(mid, safe=""))
    page.wait_for_selector(f"[data-act-api='{mid}']")
    page.locator(f"[data-act-api='{mid}']").click()
    panel = page.locator(f"[data-api-panel='{mid}']")
    panel.locator("[data-api-keys]").wait_for()
    return panel


@pytest.mark.parametrize("width", [1400, 375])
def test_use_as_an_api_shows_how_and_makes_a_key_shown_once(live, page, width):
    panel = open_api(page, live, SMALL, width, who=f"api{width}")
    assert page.locator(f"[data-model-actions='{SMALL}'] button").all_inner_texts()[:4] == \
        ["Test this model", "Chat", "Download", "Use as an API"]
    assert panel.locator("[data-api-state]").inner_text().startswith("Starts on the first request")
    base = panel.locator("[data-api-base]").inner_text()
    assert base == live["base"] + "/v1"
    assert panel.locator("[data-api-model]").inner_text() == SMALL
    curl = panel.locator("[data-api-curl]").inner_text()
    assert curl.startswith(f"curl {base}/chat/completions") and "$BOARD_API_KEY" in curl
    py = panel.locator("[data-api-python]").inner_text()
    assert 'api_key=os.environ["BOARD_API_KEY"]' in py and f'model="{SMALL}"' in py
    # Create my key: shown once, then its first characters alone
    panel.locator("[data-api-make]").click()
    shown = panel.locator("[data-api-key]")
    shown.wait_for()
    key = shown.inner_text()
    assert key.startswith("ebk_")
    row = panel.locator("[data-api-key-row]").first
    row.wait_for()
    assert row.inner_text().startswith(key[:10] + "…") and key not in row.inner_text()
    no_sideways(page)
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(panel, SCREENS / f"use-as-an-api-{width}.png")
    # and it works: the address answers with it
    import json
    import urllib.request
    req = urllib.request.Request(base + "/models", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req) as r:
        assert SMALL in [m["id"] for m in json.loads(r.read())["data"]]
    # Revoke
    row.locator("[data-api-revoke]").click()
    page.wait_for_selector("[data-api-key-row] >> text=revoked")
    assert panel.locator("[data-api-key]").count() == 0
    assert page.errors == []


def test_a_base_model_says_why_it_isnt_offered(live, page):
    panel = open_api(page, live, BASE)
    assert panel.locator("[data-api-not]").inner_text() == (
        "Not offered through the API: a base model, with no chat template: it isn't asked as a "
        "chat.")
    assert page.errors == []
