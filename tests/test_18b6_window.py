"""18b part 6, point 35: the new window is checked — a whole number, at least
the board's smallest slot — before the pin takes it. A stand-in llama-server;
no model runs."""

from __future__ import annotations

import pytest

from fake_openai import FakeServer
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture


@pytest.mark.parametrize("ctx", ["65536", "", 512, 0, True, 4095])
def test_35_a_window_that_isnt_a_whole_number_of_a_slot_is_refused(svc, ctx):  # noqa: F811
    from service import db, served
    srv = FakeServer()
    try:
        mid = "served/window-check-18b"
        db.served_put({"id": mid, "name": "w", "base_url": srv.base, "key": "",
                       "based_on": "x/y", "how": "-c 16384", "thinking": "auto", "phone": False,
                       "pin": served.pin_of(served.probe(srv.base, "")), "by": "masein", "at": 0})
        srv.ctx = ctx
        was = served.get(mid)["pin"].get("ctx")
        with pytest.raises(ValueError):                         # 70df001: taken as it came
            served.use_new_window(mid, "masein")
        assert served.get(mid)["pin"].get("ctx") == was
        srv.ctx = 65536
        assert served.use_new_window(mid, "masein")["pin"]["ctx"] == 65536
    finally:
        srv.close()
