"""16c's review: a printed shell line quotes every path; a run of refusals
pauses a batch only when they are about the key; prune cancels nothing when
Pro's file can't be read; registering a served file changes only the setups
the caller may change, and a new path switches downloads off again. Records,
small files and an OpenRouter double; nothing runs."""

from __future__ import annotations

import json
import shlex
import time

import pytest

import mobile_mmlu as mmp
from conftest import make_service
from fake_openrouter import KEY
from service import ai_models, config, db, downloads, llm, mmp_key, served
from test_14_3_mobile_mmlu import fake_pin, put_invented
from test_16c_ai_models import Chat

ME, OTHER, OWNER = "masein", "someone", "the-owner"
FILE, SIZE = "Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf", 4096


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "BOARD_OWNER", OWNER)
    monkeypatch.setattr(llm.LocalOpenAI, "BACKOFF", (0, 0, 0))
    yield client
    client.__exit__(None, None, None)


def setup(sid, name, by=ME, path=""):
    rec = {"id": sid, "name": name, "base_url": "http://127.0.0.1:9/v1", "how": "k4",
           "based_on": "", "thinking": "auto", "by": by, "gguf_path": path,
           "pin": {"model": FILE, "file": FILE, "size": SIZE}}
    db.served_put(rec)
    served.write_meta(rec)
    return rec


def gguf(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\0" * (SIZE - 4))
    return path


def command(line: str) -> list[str]:
    """the shell line inside the words, as a shell splits it"""
    return shlex.split(line.split("takes no space: ", 1)[1].split(" — then register", 1)[0])


# ---------------------------------------------------------------------------
# 1. every path in a printed line quoted
# ---------------------------------------------------------------------------

def test_the_printed_line_quotes_every_path_and_makes_the_folder(svc):
    root = str(config.BENCH_ROOT).rstrip("/")
    away = f"/srv/llama models/{FILE}"
    words = downloads.unreadable_words(away)
    assert words == (
        f"The board can’t read {away}: the folders it sees are under {root}. Put the file, or a "
        f"hard link to it, there — on the same disk a hard link takes no space: mkdir -p "
        f"{shlex.quote(root + '/models')} && ln '{away}' {shlex.quote(root + '/models/' + FILE)} "
        "— then register that path.")
    assert command(words) == ["mkdir", "-p", f"{root}/models", "&&", "ln", away,
                              f"{root}/models/{FILE}"]
    # a leading ~/ stays outside the quotes, so it still expands
    assert f"ln ~/'llama models/{FILE}' " in downloads.unreadable_words(f"~/llama models/{FILE}")


def test_a_path_with_a_quote_or_a_control_character_is_refused_and_an_old_one_is_quoted(svc):
    client = svc
    setup("served/lda", "LDA")
    for bad in ("/srv/x'; touch PWNED; echo '.gguf", '/srv/a"b.gguf', "/srv/a`id`.gguf",
                "/srv/a\nb.gguf"):
        r = client.put("/api/served/served/lda/file", json={"path": bad, "by": ME})
        assert r.status_code == 422, bad
        assert r.json()["detail"] == ("The GGUF file's path has no quotes, backticks or control "
                                      "characters in it")
    # one stored before the rule: the line still runs nothing but mkdir and ln
    rec = served.get("served/lda")
    rec["gguf_path"] = "/srv/x'; touch PWNED; echo '.gguf"
    db.served_put(rec)
    line = client.get("/api/models/file", params={"id": "served/lda"}).json()["line"]
    cmd = command(line)
    assert cmd[:4] == ["mkdir", "-p", cmd[2], "&&"] and cmd[4] == "ln" and len(cmd) == 7
    assert cmd[5] == "/srv/x'; touch PWNED; echo '.gguf"


# ---------------------------------------------------------------------------
# 2. a pause only for refusals about the key
# ---------------------------------------------------------------------------

FLAGGED = json.dumps({"error": {"code": 403, "message": "Your input was flagged for moderation"}})
RATE = json.dumps({"error": {"code": 429, "message": "Rate limit exceeded"}})


def _batch(n=30):
    be = llm.OpenRouterChat("x/y", KEY, config.BENCH_ROOT, pin=fake_pin("x/y"), role="judge")
    return be, be.submit([llm.Request(f"r:{i}", "", f"q{i}", max_tokens=10) for i in range(n)])


def _settle(be, bid, until=lambda: False):
    end = time.time() + 10
    while time.time() < end and be.status(bid)[0] == "pending" and not until():
        time.sleep(0.02)
    time.sleep(0.2)


def test_requests_refused_for_their_own_content_fail_and_the_batch_goes_on(svc, monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", KEY)
    monkeypatch.setattr(ai_models, "drifted", lambda pin: "")
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 1)
    # the first 25 refused for what they ask, the rest answered
    monkeypatch.setattr(llm, "_http", Chat(lambda model, n: (403, FLAGGED) if n <= 25 else None))
    be, bid = _batch(30)
    _settle(be, bid)
    tl = llm.tally(bid)
    assert be.status(bid)[0] == "done" and not tl["halted"]
    assert (tl["failed"], tl["answered"]) == (25, 5)
    assert llm.plain_error(tl["first_error"], 403) == (
        "OpenRouter refused it (HTTP 403): Your input was flagged for moderation")


def test_a_lasting_rate_limit_pauses_like_the_keys_limit(svc, monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", KEY)
    monkeypatch.setattr(ai_models, "drifted", lambda pin: "")
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 1)
    chat = Chat(lambda model, n: (429, RATE))
    monkeypatch.setattr(llm, "_http", chat)
    be, bid = _batch(30)
    _settle(be, bid, until=lambda: bool(be.halted(bid)))
    tl = llm.tally(bid)
    # each request tried with its back-off, then twenty in a row: paused, none failed
    assert len(chat.sent) == 20 * 4 and (tl["failed"], tl["answered"]) == (0, 0)
    assert tl["halted"] == ("waiting: OpenRouter refused 20 requests in a row — OpenRouter is "
                            "limiting this key’s requests (HTTP 429). It tries again in 10 "
                            "minutes")


# ---------------------------------------------------------------------------
# 3. prune: nothing cancelled when Pro's file can't be read
# ---------------------------------------------------------------------------

def test_prune_cancels_nothing_when_pros_file_cant_be_read(svc, monkeypatch, tmp_path):
    put_invented(monkeypatch, config.MMP_DIR)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", KEY)
    monkeypatch.setattr(ai_models, "over_limit", lambda: "waiting: this month's AI spend")
    db.ai_set("labeller:first", fake_pin(mmp.DEFAULT_LABELLERS["first"]["id"]), ME)
    pro = [q["lid"] for q in mmp.pool() if "pro" in q["sets"]]
    bid = mmp_key._submit("first", mmp_key.chosen("first"), pro, ME)
    # Pro's file gone from the server (its key's folder still there): the pool
    # reads empty, without raising
    mmp.csv_path().unlink()
    mmp._rows.clear()
    mmp._keyc.clear()
    assert mmp.available() and mmp_key._rows() == []          # empty, and no error
    assert mmp_key.prune() == 0
    assert llm.tally(bid)["cancelled"] == 0 and mmp_key.pending()


# ---------------------------------------------------------------------------
# 4. a served file: only the setups the caller may change, and downloads off again
# ---------------------------------------------------------------------------

def test_a_setup_someone_else_added_is_skipped_unless_the_owner_registers(svc):
    client = svc
    setup("served/lda", "LDA")
    setup("served/lda-mtp", "LDA · MTP")
    setup("served/theirs", "LDA · theirs", by=OTHER)
    # the panel says which setups it reaches, before Register
    got = client.get("/api/models/file", params={"id": "served/lda"}).json()
    assert {(x["id"], x["by"]) for x in got["setups"]} == {
        ("served/lda", ME), ("served/lda-mtp", ME), ("served/theirs", OTHER)}
    f = gguf(config.BENCH_ROOT / "models" / FILE)
    r = client.put("/api/served/served/lda/file", json={"path": str(f), "by": ME}).json()
    assert sorted(r["models"]) == ["served/lda", "served/lda-mtp"]
    assert r["skipped"] == [{"id": "served/theirs", "name": "LDA · theirs", "by": OTHER}]
    assert served.get("served/theirs")["gguf_path"] == ""
    # the owner may register every one
    r = client.put("/api/served/served/lda/file", json={"path": str(f), "by": OWNER}).json()
    assert sorted(r["models"]) == ["served/lda", "served/lda-mtp", "served/theirs"]
    assert r["skipped"] == []


def test_a_new_path_switches_downloads_off_again(svc):
    client = svc
    old = gguf(config.BENCH_ROOT / "models" / "old" / FILE)
    setup("served/lda", "LDA", path=str(old))
    db.download_set("served/lda", True, ME)
    assert downloads.info("served/lda")["allowed"] is True
    new = gguf(config.BENCH_ROOT / "models" / FILE)
    r = client.put("/api/served/served/lda/file", json={"path": str(new), "by": ME}).json()
    assert r["switched_off"] == ["served/lda"] and r["file"]["allowed"] is False
    # the same path again leaves the switch as it is
    db.download_set("served/lda", True, ME)
    r = client.put("/api/served/served/lda/file", json={"path": str(new), "by": ME}).json()
    assert r["switched_off"] == [] and r["file"]["allowed"] is True


# ---------------------------------------------------------------------------
# 7. who added a file, and the owner, may take it while the switch is off
# ---------------------------------------------------------------------------

def test_the_adder_and_the_owner_may_download_while_others_cant(svc):
    from test_16b2_download import model_bytes
    client = svc
    f = config.BENCH_ROOT / "models" / FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(model_bytes())                                    # a GGUF the board reads
    setup("served/lda", "LDA", path=str(f))
    assert downloads.info("served/lda")["allowed"] is False          # off, as it starts
    for who in (ME, OWNER):
        got = client.post("/api/models/file/link", json={"model": "served/lda", "by": who})
        assert got.status_code == 200, (who, got.text)
        assert client.get("/" + got.json()["url"]).content == f.read_bytes()
        r = client.get("/api/download", params={"model": "served/lda"}, headers={"X-Who": who})
        assert r.status_code == 200 and r.content == f.read_bytes()
    # anyone else: refused, the switch being off
    r = client.post("/api/models/file/link", json={"model": "served/lda", "by": OTHER})
    assert r.status_code == 403 and "switched off" in r.json()["detail"]
    assert client.get("/api/download", params={"model": "served/lda"},
                      headers={"X-Who": OTHER}).status_code == 403
    assert client.get("/api/download", params={"model": "served/lda"}).status_code == 403
