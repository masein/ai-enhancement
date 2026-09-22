"""scripts/areas.yaml: the 37 topics in 8 areas (phase 11c). Every topic in
exactly one area, spelled exactly as categories.yaml spells it, and every area
with something MMLU can say about it."""

from __future__ import annotations

import re

import pytest

import categories

AREAS = ["Mathematics & Science", "Computing & Technology", "Engineering & Applied",
         "Health & Mind", "Society", "Humanities & Arts", "Business & Law", "General"]


def test_every_topic_is_in_exactly_one_area():
    areas = categories.areas()
    assert list(areas) == AREAS
    listed = [t for ts in areas.values() for t in ts]
    assert len(listed) == len(set(listed)) == 37
    # the same names, character for character, as the topics the exam has
    assert set(listed) == set(categories.category_order())
    assert [len(ts) for ts in areas.values()] == [5, 7, 6, 3, 6, 6, 3, 1]


def test_every_area_has_mmlu_subjects():
    """"MMLU by area" is a mean over the MMLU subjects mapped to an area's
    topics: an area with none would be a column of dashes."""
    has = set(categories.with_subjects())
    for area, topics in categories.areas().items():
        assert has & set(topics), f"{area} has no topic with MMLU subjects"


def test_the_file_keeps_its_shape():
    text = categories.AREAS_PATH.read_text(encoding="utf-8")
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        assert re.match(r"^[A-Z][A-Za-z &]*:$", line) or re.match(r"^  - \S.*$", line), \
            f"areas.yaml:{no}: {raw!r}"


@pytest.mark.parametrize("text,why", [
    ("Science:\n  - Physics & Astronomy\nScience:\n  - Law\n", "listed twice"),
    ("A:\n  - Law\nB:\n  - Law\n", "already under"),
    ("A:\nLaw\n", "expected"),
])
def test_a_malformed_file_names_the_line(text, why):
    with pytest.raises(ValueError, match=why):
        categories.parse_areas(text)
