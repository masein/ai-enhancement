"""17c, six fixes found on the deployed board (0a81fc2), on the page:
2. Mobile-MMLU's card shows the full set — its questions, of which Pro's are a
   part, its run time and its models — where it showed Pro's;
3. both its parts count their questions one way, "9,497 (9,462 on our key)",
   and the full set's card has its run time;
4. "A run here" says "about" once;
5. a suite's "Its 2 parts" carries the browser's marker only;
6. Test this model's two Frontier entries are told apart at a glance, and
   Benchmarks ▸ Frontier has a card for each of the suite's benchmarks.
The cards are built from DATA as the page has it, with Mobile-MMLU's keys and
times put in (the fixture board has neither). Nothing runs."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.dashboard
MMP, MMF = "mobile_mmlu_pro", "mobile_mmlu_full"


def go(page, live, hash_="tab=benchmarks", sel="[data-catalog]"):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


# Mobile-MMLU as the deployed board has it: Pro's key over its 9,497, the
# full set's over its 16,186, and a run of each timed here
MOBILE = """() => {
  DATA.mmp = { ...(DATA.mmp || {}), credit: { by: 'Mobile-MMLU', url: 'https://x', licence: 'L' },
    key: { version: 'v', counts: { all: { questions: 9497, kept: 9462 } } } };
  DATA.mmf = { name: 'Mobile-MMLU (full)', n: 16186, licence: 'CC BY-NC 4.0', source: 'https://y',
    key: { version: 'v', counts: { all: { questions: 16186, kept: 16127 } } } };
  DATA.taskTime = { ...(DATA.taskTime || {}), [MMP]: { secs: 1500, n: 3 },
    [MMF]: { secs: 2700, n: 2 } };
  const mob = catCards().mobile.find(c => c.key === 'mobile_mmlu');
  const parts = Object.fromEntries(mob.parts.map(p => [p.key, p]));
  const t = ts => (catTime(ts) || {}).text || null;
  return { suite: [mob.questions, mob.qnote, mob.tasks, t(mob.tasks)],
    pro: [parts[MMP].questions, parts[MMP].qnote, t(parts[MMP].tasks)],
    full: [parts[MMF].questions, parts[MMF].qnote, t(parts[MMF].tasks)] };
}"""


def test_2_3_mobile_mmlu_shows_the_full_set_and_counts_both_parts_one_way(live, page):
    go(page, live)
    got = page.evaluate(MOBILE)
    assert got["suite"] == [16186, "(9,497 of them Mobile-MMLU-Pro)", [MMF],
                            "about 45 min a model on this server"]
    assert got["pro"] == [9497, "(9,462 on our key)", "about 25 min a model on this server"]
    assert got["full"] == [16186, "(16,127 on our key)", "about 45 min a model on this server"]
    assert page.errors == []


def test_4_5_a_run_here_says_about_once_and_a_suites_parts_one_marker(live, page):
    go(page, live)
    times = page.locator("[data-cat-fact='time']").all_inner_texts()
    assert times and all(x.startswith("about ") and "about about" not in x for x in times)
    summaries = page.locator("[data-cat-parts] > summary").all_inner_texts()
    assert summaries and all(s.startswith("Its ") and s.endswith(" parts") for s in summaries)
    assert page.errors == []


def test_6_the_frontier_entries_are_told_apart_and_each_benchmark_has_its_card(live, page):
    go(page, live)
    sec = page.locator("[data-cat-section='frontier']")
    for key, name, licence in (("otis_aime_epoch", "OTIS Mock AIME 2024–2025", "Apache-2.0 (gated)"),
                               ("math_l5_epoch", "MATH Level 5", "MIT"),
                               ("hle_text_cais", "Humanity's Last Exam", "MIT (gated)"),
                               ("arc_agi2_public", "ARC-AGI-2", "Apache-2.0"),
                               ("mmlupro_tiger", "MMLU-Pro, all of it", "MIT")):
        card = sec.locator(f"[data-cat-card='{key}']")
        assert card.locator("h3").inner_text().startswith(name), key
        assert card.locator(".catline").inner_text(), key
        assert card.locator("[data-cat-fact='licence']").inner_text() == licence, key
        assert card.locator("[data-cat-fact='by'] a").get_attribute("href").startswith(
            ("https://huggingface.co/datasets/", "https://github.com/")), key
        assert card.locator("[data-cat-fact='marked']").inner_text(), key
        assert card.locator("[data-cat-protocol]").inner_text().startswith(
            "How it differs from Epoch AI’s way: "), key
    assert sec.locator("[data-cat-card='mmlupro_tiger'] [data-cat-fact='questions']") \
        .inner_text() == "12,032"
    assert sec.locator("[data-cat-card='arc_agi2_public'] [data-cat-fact='questions']") \
        .inner_text() == "120 tasks"
    # Test this model's two entries: tests/test_12n2_browser.py
    # (test_test_a_model_offers_the_suite)
    assert page.errors == []
