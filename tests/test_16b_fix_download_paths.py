"""16b.2's review (HIGH 1, 2): Download built a folder from the id with no
check, so "local/.." was the whole BENCH_ROOT and "local//etc/ssl" an
absolute path, served with the board's token; and its archive could zip its
own part file and fill the disk.

An id becomes a model folder in one place (downloads.folder_of): a name of
letters, digits, dot, dash and underscore, not beginning with a dot, a
folder directly inside ARTIFACTS_DIR once resolved, with config.json at its
root. Anything else is "not a model on this board" — the same words whether
or not something is there, with no size or count. The archive never follows
a symlink out of the folder, never reads its own folder, and stops at the
free-disk floor as it writes.

Throwaway trees; nothing runs."""

from __future__ import annotations

import io
import os
import time
import zipfile

import pytest

from conftest import make_service
from service import config, downloads, uploads

ME = "masein"
BAD = ["local/..", "local/../private", "local//etc", "local//etc/ssl", "local/a/b",
       "local/", "local/.hidden", "local/.", "local/no-config", "local/outside-link"]


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "UPLOAD_FREE_GB", 0.0)
    downloads._links.clear()
    downloads._building.clear()
    arts = config.ARTIFACTS_DIR
    arts.mkdir(parents=True, exist_ok=True)
    # what must never be served: a folder beside the model folders…
    private = arts.parent / "private"
    private.mkdir(parents=True, exist_ok=True)
    (private / "answers.json").write_text('{"q1": "A"}')
    (private / "config.json").write_text("{}")             # even one that looks like a model
    # …a folder with a slash in its name, one with no config.json, a hidden one,
    # and a folder that is a link to one outside
    (arts / "a" / "b").mkdir(parents=True, exist_ok=True)
    (arts / "a" / "b" / "config.json").write_text("{}")
    (arts / "no-config").mkdir(exist_ok=True)
    (arts / "no-config" / "model.safetensors").write_bytes(b"\0" * 10)
    (arts / ".hidden").mkdir(exist_ok=True)
    (arts / ".hidden" / "config.json").write_text("{}")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.json").write_text("{}")
    (outside / "secret.txt").write_text("secret")
    os.symlink(outside, arts / "outside-link")
    yield client, tmp_path
    client.__exit__(None, None, None)


@pytest.mark.parametrize("mid", BAD)
def test_an_id_that_isnt_a_model_folder_is_never_one(svc, mid):
    client, _ = svc
    assert downloads.folder_of(mid) is None
    i = downloads.info(mid)
    assert i["kind"] == "none" and i["allowed"] is False
    assert "bytes" not in i and "files" not in i
    with pytest.raises(downloads.Refused) as e:
        downloads.file_for(mid)
    assert e.value.code == 404
    assert client.get("/api/download", params={"model": mid}).status_code == 404
    r = client.post("/api/models/file/link", json={"model": mid, "by": ME})
    assert r.status_code == 404
    # switching it on changes nothing: there is nothing to switch
    r = client.post("/api/models/file/allow", json={"model": mid, "allowed": True, "by": ME})
    assert r.status_code == 409
    assert client.get("/api/download", params={"model": mid}).status_code == 404


def test_the_info_route_without_a_token_says_nothing_of_what_is_there(svc, monkeypatch):
    client, _ = svc
    monkeypatch.setattr(config, "SUBMIT_TOKEN", "s3cret")
    there = client.get("/api/models/file", params={"id": "local/../private"}).json()
    nothing = client.get("/api/models/file", params={"id": "local/../nothing-here"}).json()
    # the same words whether or not a folder is there, and no size or count
    for got in (there, nothing):
        assert got["kind"] == "none" and "bytes" not in got and "files" not in got
    assert there["line"] == nothing["line"] == downloads.NOT_A_MODEL


def test_a_model_folders_archive_never_follows_a_link_out_of_it(svc, tmp_path):
    client, base = svc
    d = config.ARTIFACTS_DIR / "ck"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\0" * 3000)
    os.symlink(base / "outside" / "secret.txt", d / "leak.txt")      # a file out of it
    os.symlink(base / "outside", d / "leakdir")                       # a folder out of it
    (d / "sub").mkdir()
    (d / "sub" / "tokenizer.json").write_text("{}")
    assert downloads.folder_of("local/ck") == d.resolve()
    i = downloads.info("local/ck")
    assert i["kind"] == "folder" and i["allowed"] is True and i["files"] == 3
    for _ in range(200):
        r = client.post("/api/models/file/link", json={"model": "local/ck", "by": ME})
        if r.status_code == 200:
            break
        time.sleep(0.02)
    got = client.get("/" + r.json()["url"]).content
    with zipfile.ZipFile(io.BytesIO(got)) as z:
        assert sorted(z.namelist()) == ["ck/config.json", "ck/model.safetensors",
                                        "ck/sub/tokenizer.json"]
        assert b"secret" not in got


def test_the_archive_stops_at_the_disk_floor_as_it_writes_and_removes_its_part(svc,
                                                                                monkeypatch):
    client, _ = svc
    d = config.ARTIFACTS_DIR / "big"
    d.mkdir()
    (d / "config.json").write_text("{}")
    for k in range(4):
        (d / f"shard-{k}.safetensors").write_bytes(b"\0" * 5000)
    monkeypatch.setattr(config, "UPLOAD_FREE_GB", 1.0)
    # room before it starts; the disk fills as it writes
    calls = {"n": 0}

    def free():
        calls["n"] += 1
        return int(50e9) if calls["n"] <= 2 else int(1.0e9) + 4000
    monkeypatch.setattr(uploads, "free", free)
    downloads.build_archive("local/big")
    for _ in range(200):
        if downloads._building.get("local/big", {}).get("state") != "building":
            break
        time.sleep(0.02)
    got = downloads._building["local/big"]
    assert got["state"] == "refused" and "UPLOAD_FREE_GB" in got["words"], got
    assert not list(downloads.archives().glob("*.part")) and not list(
        downloads.archives().glob("*.zip"))


def test_an_archive_never_reads_the_archives_own_folder(svc, monkeypatch):
    """even a model folder that holds the archives' folder (a misconfigured
    UPLOADS_DIR inside it) never zips its own part"""
    client, _ = svc
    d = config.ARTIFACTS_DIR / "nest"
    d.mkdir()
    (d / "config.json").write_text("{}")
    monkeypatch.setattr(config, "UPLOADS_DIR", d / "uploads")
    downloads.archives().mkdir(parents=True)
    (downloads.archives() / "old.zip").write_bytes(b"PK")
    downloads.build_archive("local/nest")
    for _ in range(200):
        if downloads._building.get("local/nest", {}).get("state") == "ready":
            break
        time.sleep(0.02)
    a = downloads._building["local/nest"]
    with zipfile.ZipFile(a["path"]) as z:
        assert z.namelist() == ["nest/config.json"]


# ---------------------------------------------------------------------------
# every other place a request's name became a path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["..", "%2E%2E", ".", ".hidden", "outside-link"])
def test_deleting_an_artifact_never_reaches_past_its_folder(svc, name):
    """DELETE /api/artifacts/.. removed BENCH_ROOT: ".." matched the old name
    rule, and the folder was built from it"""
    client, base = svc
    r = client.delete(f"/api/artifacts/{name}")
    # refused — "." is tidied to the list's address, which deletes nothing (405)
    assert r.status_code in (404, 405, 422), (name, r.status_code)
    assert (config.ARTIFACTS_DIR.parent / "private" / "answers.json").exists()
    assert (base / "outside" / "secret.txt").exists()
    assert config.ARTIFACTS_DIR.exists() and config.DB_PATH.exists()


def test_a_submission_of_local_dotdot_is_refused_before_it_is_queued(svc):
    client, _ = svc
    for mid in ("local/..", "local/.hidden", "local/outside-link"):
        r = client.post("/api/submissions", json={"hf_id": mid, "suite": "quick"})
        assert r.status_code in (404, 422), (mid, r.status_code, r.text)


def test_an_upload_id_that_isnt_one_touches_nothing(svc):
    client, _ = svc
    for uid in ("..%2F..%2Fservice", "../x", "0" * 15, "zzzzzzzzzzzzzzzz"):
        assert client.put(f"/api/uploads/{uid}?offset=0", content=b"x").status_code == 404
        assert client.get(f"/api/uploads/{uid}").status_code == 404
        assert client.delete(f"/api/uploads/{uid}").status_code == 404
    with pytest.raises(uploads.Refused):
        uploads.data_path("../../x")


def test_a_zips_name_is_never_a_hidden_or_climbing_folder(svc):
    client, _ = svc
    for name in ("..", ".hidden", "a..b"):
        r = client.post("/api/uploads", json={"filename": "m.zip", "size": 10, "name": name,
                                              "by": ME})
        assert r.status_code in (409, 422), (name, r.status_code)
