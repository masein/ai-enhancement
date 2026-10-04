"""14.4.2 on the page: AI models' card for the one Mobile-MMLU answer key — the
dry run for Pro alone and for the full set (marked Non-commercial), what
Start sends (both, once each, Pro's first), each set's counts from the one
key, and both paper checks. At 1400 and 375 px. Invented rows, labels written
in; nothing calls OpenRouter."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import mobile_mmlu as mmp
from conftest import set_name
from test_14_3_browser import go, no_sideways, shot
from test_14_3_mobile_mmlu import FIXTURE, RIGHT, pin_fixture
from test_14_4_1_mobile_mmlu_data import FIXTURE as FULL, pin_full

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase14_4_2"


@pytest.fixture(scope="module", autouse=True)
def both(live):
    """both sets on this server, and every question of each labelled by the
    first two — the key's files as they were put back after"""
    import service.app as appmod
    from service import config, mmp_key
    kd = mmp.key_dir()
    saved = {f: (kd / f).read_bytes() for f in ("labels.json", "key.json") if (kd / f).exists()}
    full_dir = Path(config.MMF_DIR)
    with pytest.MonkeyPatch.context() as mp:
        d = Path(config.MMP_DIR)
        d.mkdir(parents=True, exist_ok=True)
        (d / "mobile-mmlu-pro.csv").write_bytes(FIXTURE.read_bytes())
        pin_fixture(mp)
        (full_dir / "test").mkdir(parents=True, exist_ok=True)
        for p in (FULL / "test").glob("*.csv"):
            (full_dir / "test" / p.name).write_bytes(p.read_bytes())
        pin_full(mp)
        right = {**RIGHT, **{f"inv000{n}": "A" for n in range(13, 19)}}
        cur = mmp_key.current()
        for slot in ("first", "second"):
            mmp.add_labels(slot, {q["lid"]: {"letter": right[q["id"]], "now": False,
                                             "model": cur[slot]["id"],
                                             "version": cur[slot]["version"], "at": 0}
                                  for q in mmp.pool()})
        mmp_key.rebuild()
        appmod._cache.update(key=None, payload=None, at=0.0)
        yield
        for f in ("labels.json", "key.json"):
            if f in saved:
                (kd / f).write_bytes(saved[f])
            else:
                (kd / f).unlink(missing_ok=True)
        shutil.rmtree(full_dir / "test", ignore_errors=True)
        mmp._keyc.clear()
    appmod._cache.update(key=None, payload=None, at=0.0)


@pytest.mark.parametrize("width", [1400, 375])
def test_the_key_card_shows_both_sets(live, page, width):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    go(page, live, "tab=ai", "[data-mmp-key]:not([data-mmp-key='loading'])", width)
    card = page.locator("[data-mmp-key]")
    assert card.locator("h2").first.inner_text() == "Mobile-MMLU answer key"
    # the dry run: Pro alone, then the full set, marked
    pro = card.locator("[data-mmp-est='first']").inner_text().split("\t")
    full = card.locator("[data-mmp-full-est='first']").inner_text().split("\t")
    assert (pro[1].strip(), full[1].strip()) == ("0", "0")        # every one labelled
    assert card.locator("[data-mmp-nc='estimate']").inner_text() == "Non-commercial"
    assert "Pro’s questions first" in card.locator("[data-mmp-start-sends]").inner_text()
    assert "19 questions in all" in card.locator("[data-mmp-start-sends]").inner_text()
    # each set's counts, from the one key
    assert card.locator("[data-mmp-counts]").inner_text().startswith("12 of 12 kept · ")
    assert card.locator("[data-mmp-full-counts]").get_attribute("data-mmp-full-counts") == "18|18"
    assert "18 of 18 kept" in card.locator("[data-mmp-full-counts]").inner_text()
    # both paper checks: the full set's against its own column
    assert card.locator("[data-mmp-full-check]").count() == 3
    paper = card.locator("[data-mmp-full-check='Qwen/Qwen2.5-3B-Instruct'] td").nth(1)
    assert paper.inner_text() == "68.1"
    no_sideways(page)
    shot(card, f"key-card-{width}.png", SCREENS)
    assert page.errors == []
