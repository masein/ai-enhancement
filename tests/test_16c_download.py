"""16c, part 8: a served model's file registered from its Download panel, by
whoever added it or the board's owner, for every setup of that file; a file
the board can't read says what to do; downloads still start off. Served
records and small files written in; nothing runs."""

from __future__ import annotations

import pytest

from conftest import make_service
from service import config, db, downloads, served

ME, OTHER = "masein", "someone"
FILE = "Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf"
SIZE = 4096


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "BOARD_OWNER", "the-owner")
    yield client
    client.__exit__(None, None, None)


def setup(sid: str, name: str, file: str = FILE, size: int = SIZE, by: str = ME) -> dict:
    rec = {"id": sid, "name": name, "base_url": "http://127.0.0.1:9/v1", "how": "k4",
           "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto", "by": by,
           "pin": {"model": file, "file": file, "size": size}, "gguf_path": ""}
    db.served_put(rec)
    served.write_meta(rec)
    return rec


def gguf(path, size=SIZE):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\0" * (size - 4))
    return path


def put(client, sid, path, by=ME):
    return client.put(f"/api/served/{sid}/file", json={"path": str(path), "by": by})


def test_a_served_models_unregistered_file_says_so_and_offers_its_servers_name(svc):
    client = svc
    setup("served/lda-mtp", "Qwen3.6 k4-LDA · MTP")
    i = client.get("/api/models/file", params={"id": "served/lda-mtp"}).json()
    assert (i["kind"], i["can_register"], i["hint"]) == ("served", True, FILE)
    assert i["line"] == "Its file isn’t registered here, so there is nothing to download yet."


def test_one_registration_serves_every_setup_of_the_file_and_downloads_start_off(svc):
    client = svc
    for sid, name in (("served/lda", "LDA"), ("served/lda-mtp", "LDA · MTP"),
                      ("served/lda-look", "LDA · lookahead")):
        setup(sid, name)
    setup("served/orig", "original", file="Qwen3.6-original-k-8.gguf")
    f = gguf(config.BENCH_ROOT / "models" / FILE)
    r = put(client, "served/lda-mtp", f)
    assert r.status_code == 200, r.text
    assert sorted(r.json()["models"]) == ["served/lda", "served/lda-look", "served/lda-mtp"]
    for sid in ("served/lda", "served/lda-look"):
        i = downloads.info(sid)
        assert (i["kind"], i["name"], i["allowed"]) == ("gguf", FILE, False)
        assert i["off_why"] == downloads.OFF_BY_DEFAULT
        assert served.get(sid)["gguf_path_by"] == ME
    # another file's setup is untouched
    assert served.get("served/orig")["gguf_path"] == ""
    # the owner may register it too; anyone else is refused
    assert put(client, "served/lda", f, by="the-owner").status_code == 200
    r = put(client, "served/lda", f, by=OTHER)
    assert r.status_code == 403 and r.json()["detail"] == (
        f"Only {ME}, who added it, or the board's owner registers its file.")


def test_a_file_the_board_cant_read_gives_the_next_step(svc, tmp_path):
    client = svc
    setup("served/lda", "LDA")
    away = f"/srv/llama-models/{FILE}"
    r = put(client, "served/lda", away)
    assert r.status_code == 422
    root = str(config.BENCH_ROOT).rstrip("/")
    assert r.json()["detail"] == (
        f"The board can’t read {away}: the folders it sees are under {root}. Put the file, or a "
        f"hard link to it, there — on the same disk a hard link takes no space: ln '{away}' "
        f"'{root}/models/{FILE}' — then register that path.")
    # one registered before it moved says the same, on its panel
    rec = served.get("served/lda")
    rec["gguf_path"] = away
    db.served_put(rec)
    i = downloads.info("served/lda")
    assert i["kind"] == "none" and i["unreadable"] and i["line"].startswith(
        f"The board can’t read {away}") and i["can_register"]


def test_a_file_of_another_size_isnt_the_one_it_serves(svc):
    client = svc
    setup("served/lda", "LDA", size=SIZE)
    f = gguf(config.BENCH_ROOT / "models" / "other.gguf", size=SIZE * 2)
    r = put(client, "served/lda", f)
    assert r.status_code == 422 and r.json()["detail"].endswith(
        "this isn't the file it serves")
    # and the path rule is the GGUF form's
    assert put(client, "served/lda", config.BENCH_ROOT / "models" / "x.bin").status_code == 422
