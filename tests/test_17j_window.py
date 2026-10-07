"""17j, part 5, point 37: a served model's context window isn't its file.
Started again with a larger -c, the Playground's limit is the window its
server runs it with now; a run is refused in the window's own words, not "a
different file"; the model's page takes the new window without the key. A
stand-in llama-server; no model runs."""

from __future__ import annotations

import pytest

from fake_openai import FakeServer
from service import chat, db, served
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture

MID = "served/window-17j"


@pytest.fixture
def server(svc):  # noqa: F811
    fake = FakeServer()
    fake.key = "kept-key-17j"
    p = served.pin_of(served.probe(fake.base, fake.key))
    db.served_put({"id": MID, "name": "window 17j", "base_url": fake.base, "key": fake.key,
                   "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "llama-server -c 16384",
                   "thinking": "auto", "phone": False, "pin": p, "by": "masein", "at": 0})
    yield fake
    fake.close()


def test_37_only_the_window_differs_said_in_its_words_and_taken_without_the_key(server):
    assert served.check_pin(served.get(MID)) == ""
    server.ctx = 65536                                     # started again with -c 65536
    why = served.check_pin(served.get(MID))
    assert why == served.WINDOW_LINE.format(now=65536, was=16384), why   # 0abb757: a file
    assert served.changed(why)
    rec = served.use_new_window(MID, "masein")
    assert rec["pin"]["ctx"] == 65536 and rec["key"] == "kept-key-17j"
    assert served.get(MID)["pin"]["ctx"] == 65536
    assert server.auth[-1] == "Bearer kept-key-17j"         # its own key, never asked for
    assert served.check_pin(served.get(MID)) == ""
    # another file is still another file, and isn't taken here
    server.model_path = "/models/another.gguf"
    assert served.check_pin(served.get(MID)) == served.CHANGED_LINE
    with pytest.raises(ValueError, match="different file"):
        served.use_new_window(MID, "masein")


def test_37_the_playgrounds_limit_is_the_window_its_server_runs_now(server):
    from service import playground
    server.ctx = 131072
    chat._check(MID)                                       # the background check, now
    row = next(m for m in chat.board_models() if m["id"] == MID)
    assert row["archinfo"]["ctx"] == 131072                # 0abb757: the pinned 16,384
    offered = next(m for m in playground.models()["models"] if m["id"] == MID)
    assert offered["ctx"] == 131072
    assert served.get(MID)["pin"]["ctx"] == 16384           # the pin as registered


def test_37_a_run_is_refused_in_the_windows_words(server):
    from service.hfmeta import PreflightError
    server.ctx = 32768
    with pytest.raises(PreflightError, match="context window of 32,768 tokens, not the 16,384"):
        served.preflight({"hf_id": MID, "suite": "frontier", "tasks": "[]"})
