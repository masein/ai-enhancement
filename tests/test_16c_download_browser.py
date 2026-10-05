"""16c, part 8, on the page: Download and Use as an API open directly under
their buttons, inside the header's card, and in view — in a 790 px window, on
a served model whose page has its setups and score tiles; a second click
closes them. Download says what it will do before the click: quiet, with its
reason, when there is no file. Its file is registered from the panel, for
every setup of that file. At 1400 and 375 px. Records and a small file
written in; nothing runs."""

from __future__ import annotations

import shlex
from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import set_name
from test_14_3_browser import steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase16c_download"
FILE = "Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf"
SIZE = 4096
ME = "masein"
SETUPS = (("served/qwen36-lda-mtp", "Qwen3.6 k4-LDA · MTP"),
          ("served/qwen36-lda-lookahead", "Qwen3.6 k4-LDA · lookahead"))
LINE = "Its file isn’t registered here, so there is nothing to download yet."


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import service.app as appmod
    from service import db, served
    for sid, name in SETUPS:
        rec = {"id": sid, "name": name, "base_url": "http://127.0.0.1:9/v1",
               "how": "llama.cpp k4-LDA, --n-cpu-moe 21", "based_on": "Qwen/Qwen3.6-35B-A3B",
               "thinking": "auto", "by": ME, "gguf_path": "",
               "pin": {"model": FILE, "file": FILE, "size": SIZE, "ctx": 16384,
                       "build": "b6500"}}
        db.served_put(rec)
        served.write_meta(rec)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    appmod._cache.update(key=None, payload=None, at=0.0)


def go(page, live, mid, width, height=790):
    page.set_viewport_size({"width": width, "height": height})
    page.goto("about:blank")
    page.goto(live["base"] + "/#model=" + quote(mid, safe=""))
    page.wait_for_selector("[data-model-hero]")


def top_in_view(page, sel):
    return page.evaluate(f"""() => {{ const r = document.querySelector("{sel}")
        .getBoundingClientRect(); return [r.top, innerHeight]; }}""")


@pytest.mark.parametrize("width", [1400, 375])
def test_each_panel_opens_under_its_button_in_view_and_a_second_click_closes_it(
        live, page, width):
    mid = SETUPS[0][0]
    go(page, live, mid, width)
    btn = page.locator(f"[data-act-download='{mid}']")
    # it says before the click that there is nothing to give: quiet, its reason on hover
    page.wait_for_function(f"document.querySelector(\"[data-act-download='{mid}']\")"
                           ".dataset.dlKind === 'served'")
    assert "quiet" in btn.get_attribute("class").split()
    assert btn.get_attribute("title") == LINE
    # down the page first, as someone reading it would be
    page.evaluate("window.scrollTo(0, 400)")
    btn.click()
    panel = page.locator("[data-dl-panel]")
    panel.wait_for()
    page.wait_for_timeout(100)
    top, h = top_in_view(page, "[data-dl-panel]")
    assert 0 <= top < h, (top, h)
    assert page.locator("[data-model-hero] [data-dl-panel]").count() == 1
    assert btn.get_attribute("aria-expanded") == "true" and "on" in btn.get_attribute("class")
    assert panel.locator("[data-dl-none]").inner_text() == LINE
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(panel, SCREENS / f"download-{width}.png")
    btn.click()
    page.wait_for_selector("[data-dl-panel]", state="detached")
    assert btn.get_attribute("aria-expanded") == "false"
    # Use as an API: the same
    api = page.locator(f"[data-act-api='{mid}']")
    api.click()
    page.locator("[data-api-panel]").wait_for()
    page.wait_for_timeout(100)
    top, h = top_in_view(page, "[data-api-panel]")
    assert 0 <= top < h, (top, h)
    assert page.locator("[data-model-hero] [data-api-panel]").count() == 1
    api.click()
    page.wait_for_selector("[data-api-panel]", state="detached")
    assert page.errors == []


def test_its_file_is_registered_from_the_panel_for_every_setup_of_it(live, page):
    from service import config
    mid, sib = SETUPS[0][0], SETUPS[1][0]
    page.set_viewport_size({"width": 1400, "height": 790})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, ME)
    go(page, live, mid, 1400)
    page.locator(f"[data-act-download='{mid}']").click()
    form = page.locator(f"[data-dl-register='{mid}']")
    form.wait_for()
    assert form.locator(f"[data-dl-hint='{mid}']").inner_text() == (
        f"Its server reports {FILE}. One registration serves every setup of that file.")
    # 16c review: which setups it will change, before Register
    assert form.locator("[data-dl-setups]").inner_text() == (
        "It will be registered for 2 setups of this file: Qwen3.6 k4-LDA · MTP, "
        "Qwen3.6 k4-LDA · lookahead.")
    # the terminal line quotes every value: a file's name runs nothing
    cmd = page.evaluate("dlCmd({name: \"a'b; touch PWNED; echo .gguf\"}, 'served/x')")
    words = shlex.split(cmd)
    assert words[:4] == ["curl", "-C", "-", "-fL"] and words[8:10] == [
        "-o", "a'b; touch PWNED; echo .gguf"] and len(words) == 11
    assert words[5] == "X-Token: $BOARD_TOKEN" and words[7] == f"X-Who: {ME}"
    # a path the board can't read: the next step, in words
    away = f"/srv/llama-models/{FILE}"
    form.locator("[data-dl-path]").fill(away)
    form.locator("[data-dl-register-save]").click()
    msg = page.locator(f"[data-dl-msg='{mid}']")
    msg.wait_for()
    assert msg.inner_text().startswith(f"The board can’t read {away}: the folders it sees are "
                                       "under ") and f"&& ln {shlex.quote(away)} " in \
        msg.inner_text()
    # the file, linked in: registered for both setups, downloads still off
    f = Path(config.BENCH_ROOT) / "models" / FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"GGUF" + b"\0" * (SIZE - 4))
    page.locator("[data-dl-path]").fill(str(f))
    page.locator("[data-dl-register-save]").click()
    page.wait_for_function(f"document.querySelector(\"[data-dl-msg='{mid}']\")"
                           ".textContent.startsWith('Registered')")
    assert msg.inner_text() == ("Registered for 2 setups of this file. Downloads stay off "
                                "until you switch them on.")
    # 16c review: who added it may take it while the switch is off — one line
    # says others can't yet, and the button and the curl line are there
    assert page.locator(f"[data-dl-others-off='{mid}']").inner_text() == (
        "Others can’t take it yet: “Others can download it” is off. You can, as who added it.")
    assert page.locator(f"[data-dl-go='{mid}']").count() == 1
    assert page.locator(f"[data-dl-cmd='{mid}']").count() == 1
    assert page.locator(f"[data-dl-off='{mid}']").count() == 0
    assert page.locator(f"[data-dl-allow='{mid}']").count() == 1      # the switch, as before
    steady_shot(page.locator("[data-dl-panel]"), SCREENS / "registered-1400.png")
    # the other setup's button now says what it will give
    go(page, live, sib, 1400)
    page.wait_for_function(f"document.querySelector(\"[data-act-download='{sib}']\")"
                           ".dataset.dlKind === 'gguf'")
    b = page.locator(f"[data-act-download='{sib}']")
    assert "quiet" not in b.get_attribute("class").split()
    assert b.get_attribute("title").startswith(f"{FILE} · ") and b.get_attribute(
        "title").endswith("downloads are switched off")
    # the one refusal it asked for: the unreadable path's 422
    assert [e for e in page.errors if "status of 422" not in e] == []
