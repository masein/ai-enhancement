"""16b.1 on the page: Add a model.

- "Test a model" is Add a model. Its first step asks where the model is: On
  Hugging Face · On my computer · Running on a server · On OpenRouter, and
  under More, Already on this server. A model's page tests its own model.
- On my computer: a .gguf picked, sent in pieces with its progress; a dropped
  piece sent again by itself; a reload resumed by picking the file again; then
  checked (its sha256, what its header says), its details, and its tests.
- What is kept, in view, and a delete that asks first.

At 1400 and 375 px. A GGUF header with no weights; pieces of 64 kB. Nothing
runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import choose_where, set_name
from service import config, uploads
from test_14_3_browser import no_sideways, steady_shot
from test_16_1_sizes import gguf_bytes

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase16b1"
PIECE = 64 * 1024


@pytest.fixture
def room(live, tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "PIECE", PIECE)
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "UPLOAD_FREE_GB", 0.0)       # the test machine's disk isn't the server's
    yield tmp_path


def phone_file(tmp_path: Path, name="Qwen3.6-k4-phone.gguf", pieces=5) -> Path:
    f = tmp_path / name
    f.write_bytes(gguf_bytes() + bytes(range(256)) * (pieces * PIECE // 256))
    return f


def open_dialog(page, live, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.locator("[data-test-model]").click()
    page.wait_for_selector("[data-dialog='test'] [data-add-chooser]")


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(part, SCREENS / name)


@pytest.mark.parametrize("width", [1400, 375])
def test_add_a_model_asks_where_the_model_is(live, page, room, width):
    open_dialog(page, live, width)
    assert page.locator("[data-test-model]").inner_text() in ("Add a model", "Add")
    dlg = page.locator("[data-dialog='test']")
    assert dlg.locator("h2#test-title").inner_text() == "Add a model"
    assert dlg.locator(".addq").first.inner_text() == "Where is the model?"
    assert dlg.locator(".addchoices > [data-add-where]").all_inner_texts() == \
        ["On Hugging Face", "On my computer", "Running on a server", "On OpenRouter"]
    # under More, for people with SSH
    assert dlg.locator(".addmore [data-add-where]").all_text_contents() == ["Already on this server"]
    # On Hugging Face: today's form
    assert dlg.locator("[data-submit-form] [data-ms='submit']").count() == 1
    shot(dlg.locator(".dlg"), f"add-a-model-{width}.png")
    for where, part in (("computer", "[data-upload-card]"), ("server", "[data-served-card]"),
                        ("openrouter", "[data-openrouter-card]"), ("here", "[data-gguf-card]")):
        choose_where(page, where)
        assert dlg.locator(part).count() == 1, where
        assert dlg.locator("[data-submit-form]").count() == 0
        no_sideways(page)
    # a model's page: that model, tested — no question where it is
    page.keyboard.press("Escape")
    page.goto(live["base"] + "/#model=fx%2Fgood-750m")
    page.wait_for_selector("[data-test-model][data-test-this='fx/good-750m']")
    page.locator("[data-test-model]").click()
    assert dlg.locator("h2#test-title").inner_text() == "Test this model"
    assert dlg.locator("[data-add-chooser]").count() == 0
    assert dlg.locator("[data-ms='submit'] input").input_value() == "fx/good-750m"
    assert page.errors == []


def test_a_phone_build_from_my_computer_cut_resumed_checked_and_added(live, page, room):
    f = phone_file(room)
    open_dialog(page, live)
    choose_where(page, "computer")
    card = page.locator("[data-upload-card]")
    card.locator("[data-up-storage]").wait_for()
    # the second piece is cut on its way: the page sends it again by itself
    seen = {"n": 0}

    def cut_once(route):
        seen["n"] += 1
        if seen["n"] == 2:
            route.abort()
        else:
            route.continue_()
    page.route("**/api/uploads/*?offset=*", cut_once)
    card.locator("[data-up-file]").set_input_files(str(f))
    assert card.locator("[data-up-chosen]").inner_text().startswith(
        f"{f.name} · 0.0 GB · a GGUF file")
    assert card.locator("[data-up-name]").input_value() == "Qwen3.6-k4-phone"
    card.locator("[data-up-name]").fill("Qwen3.6 k4 phone")
    card.locator("[data-up-start]").click()
    details = card.locator("[data-up-details]")
    details.wait_for(timeout=30000)
    page.unroute("**/api/uploads/*?offset=*")
    assert seen["n"] > 6                                     # five pieces, one sent twice
    import hashlib
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    assert card.locator("[data-up-checked]").inner_text() == (
        f"Checked: {f.name} · 0.0 GB · sha256 {sha}")
    assert card.locator("[data-up-facts]").inner_text().startswith("qwen3moe · ")
    assert card.locator("[data-up-facts]").inner_text().endswith("has a chat template")
    # its size from the header, there to confirm
    assert card.locator("[data-up='size']").input_value().endswith("K")
    card.locator("[data-up='based_on']").fill("Qwen/Qwen3.6-35B-A3B")
    card.locator("[data-up='how']").fill("llama.cpp fork k4 + LDA, Q4_K_XL")
    card.locator("[data-up='size']").fill("35B")
    card.locator("[data-up='active']").fill("3B")
    card.locator("[data-up-download]").uncheck()
    shot(card, "details-1400.png")
    card.locator("[data-up-add]").click()
    tests = card.locator("[data-up-tests]")
    tests.wait_for()
    assert tests.get_attribute("data-up-tests") == "gguf/Qwen3.6-k4-phone"
    assert card.locator("[data-up-test]").inner_text() == "Choose what to measure ▸"
    shot(card, "tests-1400.png")
    # what is kept: in view, with who added it; delete asks first
    row = card.locator("[data-up-file-row='gguf/Qwen3.6-k4-phone']")
    row.wait_for()
    assert "masein" in row.inner_text() and "Qwen3.6-k4-phone.gguf" in row.inner_text()
    assert card.locator("[data-up-line]").inner_text() == "Uploads: 0 of 150 GB"
    shot(card.locator("[data-up-storage]"), "kept-1400.png")
    card.locator("[data-up-skip]").click()                   # Add without testing
    page.wait_for_selector("[data-model-hero]")
    assert page.evaluate("state.model") == "gguf/Qwen3.6-k4-phone"
    open_dialog(page, live)
    choose_where(page, "computer")
    card = page.locator("[data-upload-card]")
    card.locator("[data-up-delete='gguf/Qwen3.6-k4-phone']").click()
    ask = card.locator("[data-up-confirm='gguf/Qwen3.6-k4-phone']")
    assert ask.inner_text().startswith("Delete Qwen3.6 k4 phone? Its results stay.")
    ask.locator("[data-up-delete-yes]").click()
    msg = card.locator("[data-up-msg]")
    msg.wait_for()
    assert msg.inner_text().startswith("Deleted Qwen3.6 k4 phone:")
    assert not (config.UPLOADS_DIR / "gguf" / "Qwen3.6-k4-phone.gguf").exists()
    # the one error: the piece this test cut
    assert [e for e in page.errors if "net::ERR_FAILED" not in e] == []


def test_a_reload_is_resumed_by_picking_the_file_again(live, page, room):
    f = phone_file(room, "resume-me.gguf", pieces=6)
    open_dialog(page, live)
    choose_where(page, "computer")
    card = page.locator("[data-upload-card]")
    card.locator("[data-up-storage]").wait_for()
    # two pieces arrive, then the connection goes and the page is reloaded
    seen = {"n": 0}

    def two_then_gone(route):
        seen["n"] += 1
        route.continue_() if seen["n"] <= 2 else route.abort()
    page.route("**/api/uploads/*?offset=*", two_then_gone)
    card.locator("[data-up-file]").set_input_files(str(f))
    card.locator("[data-up-start]").click()
    for _ in range(100):
        if seen["n"] >= 3:
            break
        page.wait_for_timeout(100)
    page.unroute("**/api/uploads/*?offset=*")
    page.reload()
    page.wait_for_selector("[data-test-model]")
    page.locator("[data-test-model]").click()
    choose_where(page, "computer")
    card = page.locator("[data-upload-card]")
    pending = card.locator("[data-up-pending]")
    pending.wait_for()
    assert pending.inner_text().startswith("resume-me.gguf · 0.0 GB of 0.0 GB · masein")
    pending.get_by_role("button", name="Resume").click()
    assert card.locator("[data-up-msg]").inner_text().startswith("Pick resume-me.gguf again")
    shot(card, "resume-1400.png")
    card.locator("[data-up-file]").set_input_files(str(f))
    card.locator("[data-up-details]").wait_for(timeout=30000)
    assert card.locator("[data-up='name']").input_value() == "resume-me"
    card.locator("[data-up-details] .ghost", has_text="Cancel").click()       # cancelled: removed
    page.wait_for_selector("[data-up-pending]", state="detached")
    assert [e for e in page.errors if "net::ERR_FAILED" not in e] == []      # the cut ones
