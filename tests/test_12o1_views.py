"""12o.1: a saved view keeps its table's widths and column order, so a view
someone shares opens the way it was saved. A layout is a table's name, its
widths in pixels and its columns' names in order — nothing else is kept."""

from __future__ import annotations

import pytest

from test_views_12h2 import SPEC, save, svc  # noqa: F401

LAYOUT = {"models:standard:custom": {"w": {"name": 320, "mmlu_pro": 96},
                                     "order": ["mmlu_pro", "ifeval", "hendrycks_math500"],
                                     "groups": []}}


def test_a_view_keeps_its_layout_and_one_saved_as_it_comes(svc):  # noqa: F811
    c, _ = svc
    v = save(c, spec={**SPEC, "layout": LAYOUT}).json()
    assert v["spec"]["layout"] == LAYOUT
    # the table as it comes is kept too: the view opens it so
    plain = {"compare": {"w": {}, "order": [], "groups": []}}
    v = save(c, name="Two", spec={"view": "compare", "models": ["a/b", "c/d"],
                                  "layout": plain}).json()
    assert v["spec"] == {"view": "compare", "models": ["a/b", "c/d"], "layout": plain}
    # a view from before layouts: none
    v = save(c, name="Three").json()
    assert "layout" not in v["spec"]


@pytest.mark.parametrize("layout,words", [
    ({"t": {"w": {"name": 20}}}, "44 to 640"),
    ({"t": {"w": {"name": 9000}}}, "44 to 640"),
    ({"t": {"w": {"name": True}}}, "44 to 640"),
    ({"t": {"w": {"name": "wide"}}}, "44 to 640"),
    ({"t": {"order": [{"x": 1}]}}, "column names"),
    ({"t": {"order": ["x" * 201]}}, "column names"),
    ({"": {"w": {}}}, "a table's name"),
    ({f"t{i}": {} for i in range(9)}, "at most 8 tables"),
])
def test_a_layout_is_widths_and_names_only(svc, layout, words):  # noqa: F811
    c, _ = svc
    r = save(c, spec={**SPEC, "layout": layout})
    assert r.status_code == 422 and words in r.json()["detail"]
