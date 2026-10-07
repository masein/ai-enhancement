"""17j on the page, part 1, point 8: the grading card says whether the HLE
judge chosen answers in CAIS's JSON schema, and warns when it doesn't — that
is when a reply in prose can be asked again, and paid. OpenRouter's list is
invented; nothing is sent anywhere."""

from __future__ import annotations

import json
import time

import pytest

pytestmark = pytest.mark.dashboard
O3, O3_V = "openai/o3-mini", "o3-mini-2025-01-31"


def models(structured):
    from service import ai_models
    p = ai_models._cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"at": time.time(), "models": [
        {"id": O3, "name": "OpenAI: o3 Mini", "version": O3_V, "price_in": 1.1,
         "price_out": 4.4, "reasons": True, "structured": structured}]}))


@pytest.mark.parametrize("structured", [False, True])
def test_8_the_card_says_whether_hles_judge_takes_the_json_schema(live, page, structured):
    import service.app as appmod
    models(structured)
    appmod._cache.update(key=None, payload=None, at=0.0)
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=ai")
    line = page.locator("[data-frontier-grader='hle'] [data-frontier-json-only]")
    line.wait_for()                                      # 0abb757: nothing said
    if structured:
        assert line.get_attribute("data-frontier-json-only") == "1"
        assert line.inner_text().startswith("answers in CAIS’s JSON schema")
    else:
        assert line.get_attribute("data-frontier-json-only") == "0"
        words = line.inner_text()
        assert words.startswith("doesn’t take a JSON schema") and "asked again, and paid" in words
    assert page.errors == []
