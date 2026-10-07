"""17i on the page, part 4, point 26: Compare — two names of one word, no
separator in them, still read the same at 12 to 16 characters."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.dashboard


@pytest.mark.parametrize("width", [12, 13, 14, 15, 16])
def test_26_one_word_names_cut_to_a_column_read_apart(live, page, width):
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-lb-card]")
    got = page.evaluate("""(n) => {
      const fits = x => x.length <= n;
      return [[...headNames(['phonebuildalphaq4', 'phonebuildbetaq4'], fits).values()],
              [...headNames(['Qwen36A3Bk4LDAphone', 'Qwen36A3Bk8LDAphone',
                             'Qwen36A3Bk16LDAphone'], fits).values()],
              [...headNames(['averyveryverylongsinglename', 'averyveryverylongsinglenamf'],
                            fits).values()]];
    }""", width)
    for labels in got:
        assert len(set(labels)) == len(labels), labels        # 511854e: two read the same
        assert all(len(x) <= width for x in labels), labels
    assert page.errors == []
