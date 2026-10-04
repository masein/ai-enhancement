"""16.6 on the page: dates and times, one way through the whole board — the
viewer's own time zone, 24-hour.

- **The chat list:** each chat says when it was last used ("13:05" today,
  "Yesterday", a weekday within the week, then "28 Sep", with the year in
  another year), grouped Today · Yesterday · Earlier this week · then by month.
- **Inside a chat:** a date line between days ("Mon 5 Oct"), and the time on
  each message — a reply's beside its tokens and seconds — in full on hover.
  A message with no time stored shows none.
- **The model page's Chat tab** has the same; Runs and History say a time the
  same way.

The viewer is in Dubai, and it is Wednesday 7 Oct 2026, 14:32, by the page's
clock. Chats written in; nothing runs."""

from __future__ import annotations

import datetime as dt
import json
import re
from contextlib import closing
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from conftest import set_name
from service import chat, db
from test_14_3_browser import steady_shot
from test_playground_12d1_browser import fake  # noqa: F401 — the fake chat backend

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase16_6"
SMALL = "fx/below-135m-it"
TZ = "Asia/Dubai"
NOW = dt.datetime(2026, 10, 7, 14, 32, tzinfo=ZoneInfo(TZ))           # a Wednesday


def at(*a) -> float:
    return dt.datetime(*a, tzinfo=ZoneInfo(TZ)).timestamp()


def reply(t, text="It arrives at 10:05.") -> dict:
    return {"role": "assistant", "shown": 0, "replies": [
        {"text": text, "words": len(text.split()), "secs": 1.2, "wps": 3.3, "device": "cpu",
         "at": t}]}


def plant(title, updated, messages=()):
    c = chat.new_chat(SMALL, "masein")
    c.update(title=title, updated_at=updated, messages=list(messages))
    with closing(db._conn()) as cn:
        cn.execute("UPDATE chats SET data=?, updated_at=? WHERE id=?",
                   (json.dumps(c), updated, c["id"]))
        cn.commit()
    return c["id"]


@pytest.fixture(scope="module")
def chats(live):
    with closing(db._conn()) as cn:
        cn.execute("DELETE FROM chats")
        cn.commit()
    two_days = plant("Trains from the Marina", at(2026, 10, 7, 13, 5), [
        {"role": "user", "text": "when is the next train", "at": at(2026, 10, 5, 10, 0)},
        reply(at(2026, 10, 5, 10, 1)),
        {"role": "user", "text": "and on Wednesday?", "at": at(2026, 10, 7, 13, 4)},
        reply(at(2026, 10, 7, 13, 5)),
        {"role": "user", "text": "an old one, no time kept"},
        {"role": "assistant", "shown": 0, "replies": [
            {"text": "no time", "words": 2, "secs": 0.5, "wps": 4.0, "device": "cpu"}]}])
    plant("Yesterday's", at(2026, 10, 6, 18, 0))
    plant("Monday's", at(2026, 10, 5, 9, 0))
    plant("September's", at(2026, 9, 28, 21, 15))
    plant("Last December's", at(2025, 12, 15, 8, 0))
    yield two_days


@pytest.fixture
def viewer(browser):
    ctx = browser.new_context(viewport={"width": 1400, "height": 900}, reduced_motion="reduce",
                              timezone_id=TZ)
    page = ctx.new_page()
    page.clock.install(time=NOW.timestamp())
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.errors = errors
    yield page
    ctx.close()


def playground(page, live, width=1400, chat_id=""):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=playground" + (f"&chat={chat_id}" if chat_id else ""))
    page.wait_for_selector("[data-pg-main]")


def test_one_way_to_say_when(live, viewer):
    page = viewer
    page.goto(live["base"] + "/#tab=home")
    page.wait_for_function("typeof whenShort === 'function'")
    said = page.evaluate(f"""() => [{at(2026, 10, 7, 9, 5)}, {at(2026, 10, 6, 21, 15)},
      {at(2026, 10, 3, 12, 0)}, {at(2026, 9, 28, 21, 15)}, {at(2025, 12, 15, 8, 0)}]
      .map(t => [whenShort(t), whenFull(t), whenGroup(t), dayLine(t)])""")
    assert said == [
        ["09:05", "Wed 7 Oct 2026, 09:05", "Today", "Wed 7 Oct"],
        ["Yesterday", "Tue 6 Oct 2026, 21:15", "Yesterday", "Tue 6 Oct"],
        # Saturday is in the week, and before Monday: by month
        ["Sat", "Sat 3 Oct 2026, 12:00", "October", "Sat 3 Oct"],
        ["28 Sep", "Mon 28 Sep 2026, 21:15", "September", "Mon 28 Sep"],
        ["15 Dec 2025", "Mon 15 Dec 2025, 08:00", "December 2025", "Mon 15 Dec 2025"]]
    # Monday of this week is "Earlier this week"
    assert page.evaluate(f"whenGroup({at(2026, 10, 5, 9, 0)})") == "Earlier this week"
    assert page.errors == []


@pytest.mark.parametrize("width", [1400, 375])
def test_the_chat_list_says_when_each_was_last_used(live, viewer, chats, width):
    page = viewer
    playground(page, live, width)
    if width < 720:
        page.locator("[data-pg-chats]").click()
        root = page.locator("#pop-pg-chats")
    else:
        root = page.locator("[data-pg-list]")
    groups = root.locator("[data-pg-group]").evaluate_all("xs => xs.map(x => x.dataset.pgGroup)")
    assert groups == ["Today", "Yesterday", "Earlier this week", "September", "December 2025"]
    when = {root.locator("[data-pg-chat] .pgtitle", has_text=t).first.evaluate(
        "e => e.closest('[data-pg-chat]').querySelector('[data-pg-when]').textContent"): t
        for t in ("Trains from the Marina", "Yesterday's", "Monday's", "September's",
                  "Last December's")}
    assert set(when) == {"13:05", "Yesterday", "Mon", "28 Sep", "15 Dec 2025"}
    title = root.locator("[data-pg-when]").first.get_attribute("title")
    assert title == "Wed 7 Oct 2026, 13:05"
    shot(root, f"chat-list-{width}.png")
    assert page.errors == []


@pytest.mark.parametrize("width", [1400, 375])
def test_inside_a_chat_a_line_between_days_and_each_messages_time(live, viewer, chats, width):
    page = viewer
    playground(page, live, width, chats)
    page.wait_for_selector("[data-pg-day]")
    assert page.locator("[data-pg-day]").all_inner_texts() == ["Mon 5 Oct", "Wed 7 Oct"]
    you = page.locator("[data-pg-you] [data-pg-at]")
    assert you.all_inner_texts() == ["10:00", "13:04"]                  # the third has none
    assert page.locator("[data-pg-you]").count() == 3
    assert you.nth(1).locator("time").get_attribute("title") == "Wed 7 Oct 2026, 13:04"
    stats = page.locator("[data-pg-stats]")
    assert stats.nth(0).inner_text().startswith("10:01 · 4 words · 1.2s")
    assert stats.nth(1).inner_text().startswith("13:05 · ")
    assert not re.match(r"\d\d:\d\d", stats.nth(2).inner_text())       # no time stored, none said
    shot(page.locator("[data-pg-main]"), f"chat-days-{width}.png")
    assert page.errors == []


def test_runs_and_history_say_a_time_the_same_way(live, viewer):
    page = viewer
    sid = db.add(SMALL, "instruct", "quick", "masein", "")
    db.update(sid, created_at=at(2026, 10, 7, 14, 30), started_at=at(2026, 10, 7, 14, 31))
    try:
        page.goto(live["base"] + "/#tab=home")
        set_name(page, "masein")
        page.goto(live["base"] + "/#tab=runs")
        cell = page.locator(f"[data-queue-row='{sid}'] td").nth(1)
        cell.wait_for()
        assert cell.get_attribute("title") == ("submitted Wed 7 Oct 2026, 14:30\n"
                                               "started Wed 7 Oct 2026, 14:31")
    finally:
        db.update(sid, status="canceled")
    assert page.errors == []


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(part, SCREENS / name)
