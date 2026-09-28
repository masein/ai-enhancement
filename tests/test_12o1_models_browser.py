"""12o.1 on the page, found live after #103 and #104: a custom set that mixes
methods lists every model with any value in it, with one Avg a method and
never one across them; Models ▾ groups a served model and its GGUF by model
and setup, a tick a setup, badged with what each has; and "Same file as",
set from the served entry or from the GGUF entry, joins or parts them.

12n.1's board: a GGUF file measured as built and in lookahead 1, served
plain, "· lookahead 1" and "· MTP" by a fake server; the MTP one answered
SimpleQA through it. Fixtures only."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

import simpleqa as sq
from conftest import set_name
from test_12n1_browser import (GOOD, NAME, SERVED, SERVED_LA, SERVED_MTP, SKEWED,  # noqa: F401
                               board)
from test_12n2 import sit_simpleqa, twenty

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12o1"
LM = "mmlu,hellaswag,winogrande,arc_easy"
GG = "gguf:hellaswag,gguf:winogrande"


@pytest.fixture(scope="module", autouse=True)
def chat(live, board):  # noqa: F811 — 12n.1's board, then this
    """the MTP entry answered SimpleQA through its server"""
    import service.app as appmod
    from service import config
    d = config.OUT_DIR / SERVED_MTP.replace("/", "__")
    sit_simpleqa(d, twenty())
    qs = {q["id"]: q for q in sq.load()}
    out = sq.mark(d)
    sq.write(d, sq.mark(d, {it["id"]: sq.stub_grade(qs[it["id"]], it["answer_text"])
                            for it in out["items"] if it["grade"] is None}))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


def ids(*ms):
    return ",".join(quote(m, safe="") for m in ms)


def go(page, live, hash_, sel, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def rows(page):
    return page.locator("[data-lb-table] tbody tr[data-lb-row]").evaluate_all(
        "xs => xs.map(x => x.dataset.lbRow)")


def test_a_mixed_set_lists_every_model_with_any_value_and_one_avg_a_method(live, page):
    go(page, live, f"tab=models&cols={LM},{GG}&models={ids(GOOD, SKEWED, SERVED, SERVED_LA)}",
       "[data-lb-table]")
    # the GGUFs have values in two of six: rows, never "can't be measured"
    assert set(rows(page)) == {GOOD, SKEWED, SERVED, SERVED_LA}
    assert page.locator("[data-none-has]").count() == 0
    assert "no Avg across methods" in page.locator("[data-gguf-mix]").inner_text()
    heads = page.locator("[data-lb-table] thead th[data-col^='cavg'] .hname").all_text_contents()
    assert heads == ["Avg · lm_eval", "Avg · llama.cpp"]
    # each has its method's Avg only: a board model none of llama.cpp's
    avg = lambda m, meth: page.locator(  # noqa: E731
        f"tr[data-lb-row='{m}'] [data-cavg='{m}'][data-cavg-method='{meth}']").inner_text()
    assert avg(GOOD, "lm_eval") != "—" and avg(GOOD, "llama.cpp") == "—"
    assert avg(SERVED, "llama.cpp") != "—" and avg(SERVED, "lm_eval") == "—"
    # the missing cells read "—" — a server can't be asked MMLU's log-likelihoods: blank, why
    # on hover
    for m, col in ((GOOD, "gguf:hellaswag"), (SERVED, "mmlu")):
        assert page.evaluate("""([m, c]) => {
          const t = document.querySelector('[data-lb-table]');
          const i = [...t.querySelector('thead tr:last-child').children]
            .findIndex(th => th.dataset.col === c);
          return t.querySelector(`tbody tr[data-lb-row="${m}"]`).children[i].textContent.trim();
        }""", [m, col]) in ("—", ""), (m, col)
    # sorted by one method's Avg: its models first
    page.locator("th[data-col='cavg:llama.cpp']").click()
    page.wait_for_selector("th[data-col='cavg:llama.cpp'][aria-sort='descending']")
    assert set(rows(page)[:2]) == {SERVED, SERVED_LA}
    shot(page.locator("[data-lb-card]"), "models-mixed-set-1400.png")
    assert page.errors == []


def test_one_method_is_as_before_and_the_empty_line_only_when_none_has_any(live, page):
    go(page, live, f"tab=models&cols={GG}&models={ids(SERVED, SERVED_LA, GOOD)}",
       "[data-lb-table]")
    assert page.locator("[data-gguf-mix]").count() == 0
    assert page.locator("thead th[data-col^='cavg'] .hname").all_text_contents() == [
        "Avg above chance"]
    # a model with none of them: under the line, not a row
    assert GOOD not in rows(page)
    go(page, live, f"tab=models&cols={LM}&models={ids(SERVED_MTP)}", "[data-lb-table]")
    assert page.locator("[data-none-has] p").inner_text().startswith(
        "None of the 1 has any of these 4.")
    assert page.errors == []


def open_models(page):
    page.locator("#pill-models").click()
    page.wait_for_selector("#pop-models")


def test_models_picker_is_by_model_and_setup_with_what_each_has(live, page):
    go(page, live, "tab=models", "[data-lb-table]")
    open_models(page)
    grp = page.locator("#pop-models [data-model-group^='model:']")
    assert grp.count() == 1
    key = grp.get_attribute("data-model-group")[len("model:"):]
    assert grp.inner_text().split(" all")[0].strip() == NAME
    setups = page.locator(f"#pop-models [data-setup-row^='{key}|']")
    labels = [setups.nth(i).inner_text().replace("\n", " ").strip() for i in range(setups.count())]
    assert labels == ["as built llama.cpp ✓", "lookahead 1 llama.cpp ✓", "MTP chat ✓"]
    # the served entries are no longer rows of their own anywhere else in it
    assert page.locator(f"#pop-models [data-model-pick='{SERVED}']").count() == 0
    assert page.locator("#pop-models [data-model-group='served']").count() == 0
    # one tick a setup: "only these" for the model, then its MTP off
    page.locator(f"#pop-models [data-model-group-only='model:{key}']").click()
    page.wait_for_function("location.hash.includes('models=')")
    page.locator(f"#pop-models [data-setup-pick='{key}|mtp']").uncheck()
    page.wait_for_function(f"!location.hash.includes({json.dumps(quote(SERVED_MTP, safe=''))})")
    assert set(rows(page)) == {SERVED, SERVED_LA}
    shot(page.locator("#pop-models"), "models-picker-setups.png")
    # the other groups as they were
    assert page.locator("#pop-models [data-model-group='base']").count() == 1
    assert page.errors == []


def test_same_file_as_from_either_side_joins_or_parts_them(live, page):
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]")
    set_name(page, "masein")
    page.locator("[data-test-model]").click()
    page.locator("[data-served-card] > summary").click()
    page.wait_for_selector(f"[data-srv-row='{SERVED_LA}'] [data-same-as]")
    # the GGUF entries it can be, loaded
    page.wait_for_function("state.gg.page && !state.gg.loading")
    btn = page.locator(f"[data-srv-row='{SERVED_LA}'] [data-same-as]")
    assert btn.inner_text() == "Same file as: guessed from the file ▾"
    btn.click()
    page.locator(f"[id='pop-same-{SERVED_LA}'] [data-same-as-choice='none']").click()
    page.wait_for_function(f"(document.querySelector('[data-srv-row=\"{SERVED_LA}\"] "
                           "[data-same-as]') || {}).textContent === "
                           "'Same file as: not a GGUF entry ▾'")
    # parted: its setup is two rows now — and one tick in Models ▾ still
    page.keyboard.press("Escape")
    go(page, live, "tab=models", "[data-lb-table]")
    open_models(page)
    box = page.locator("#pop-models [data-setup-pick$='|lookahead']")
    assert len(json.loads(box.get_attribute("data-ids"))) == 2
    # from the GGUF entry's side: its lookahead 1 is that served entry again
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]")
    page.locator("[data-test-model]").click()
    page.locator("[data-gguf-card] > summary").click()
    page.wait_for_selector("[data-same-as-gguf] [data-same-as-setup]")
    page.wait_for_function("state.srv.list && !state.srv.loading")
    look = page.locator("[data-same-as-gguf] [data-same-as-setup]", has_text="lookahead 1")
    look.click()
    page.locator(f"[role='menuitemradio'][data-same-as-served='{SERVED_LA}']").click()
    page.wait_for_function("[...document.querySelectorAll('[data-same-as-setup]')].some(b => "
                           f"b.textContent.startsWith('lookahead 1: {NAME} · lookahead 1'))")
    go(page, live, "tab=models", "[data-lb-table]")
    open_models(page)
    box = page.locator("#pop-models [data-setup-pick$='|lookahead']")
    assert json.loads(box.get_attribute("data-ids")) == [SERVED_LA]
    assert page.errors == []
