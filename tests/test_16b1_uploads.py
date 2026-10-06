"""16b.1: Add a model ▸ On my computer — a .gguf file, or a model folder as a
.zip, sent in pieces that resume.

- An upload cut in the middle goes on from the server's offset, and the same
  file picked again after a reload resumes it.
- Refused before it starts, with the numbers: a file over the limit, uploads
  over the quota, a disk that would keep less than UPLOAD_FREE_GB.
- Checked after: the sha256; a GGUF's header (a file that isn't one is
  refused, nothing kept); a zip with today's checks (a .bin, no config.json,
  a member outside its folder).
- Added under a name nothing else has, as the GGUF form registers a file (its
  sha256 pinned) or as an uploaded folder (local/<name>); its size from the
  header, confirmed. Kept, in view; deleted only when no run uses it, its
  results left.
- The path form ("Already on this server") and a served model's GGUF: a path
  that ends in .gguf, and no flag that names a file or is a path.

Fake files: a GGUF header with no weights. Nothing runs."""

from __future__ import annotations

import io
import json
import time
import zipfile

import pytest

from conftest import make_service
from service import config, db, gguf, uploads
from test_16_1_sizes import gguf_bytes

ME = "masein"
PIECE = 4096


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(uploads, "PIECE", PIECE)
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    # the machine's own disk isn't the server's: its floor is the test's to set
    monkeypatch.setattr(config, "UPLOAD_FREE_GB", 0.0)
    yield client
    client.__exit__(None, None, None)


def model_file(pad: int = 3 * PIECE + 100) -> bytes:
    """a GGUF header, and bytes after it standing in for its weights"""
    return gguf_bytes() + bytes(range(256)) * (pad // 256 + 1)


def start(client, data: bytes, filename="phone-k4.gguf", name="", mtime=1790000000.0):
    return client.post("/api/uploads", json={"filename": filename, "size": len(data),
                                             "mtime": mtime, "name": name, "by": ME})


def put(client, uid, at, chunk):
    return client.put(f"/api/uploads/{uid}?offset={at}", content=chunk)


def send_all(client, uid, data, at=0):
    while at < len(data):
        r = put(client, uid, at, data[at:at + PIECE])
        assert r.status_code == 200, r.text
        at = r.json()["offset"]
    return at


def finished(client, uid) -> dict:
    assert client.post(f"/api/uploads/{uid}/finish").status_code == 200
    for _ in range(200):
        rec = client.get(f"/api/uploads/{uid}").json()
        if rec["state"] != "checking":
            return rec
        time.sleep(0.02)
    raise AssertionError("still checking")


# ---------------------------------------------------------------------------
# a GGUF, cut and resumed, checked, added, kept and deleted
# ---------------------------------------------------------------------------

def test_a_gguf_cut_in_the_middle_resumes_and_is_added_with_its_sha256_and_header(svc):
    client = svc
    data = model_file()
    rec = start(client, data).json()
    assert (rec["offset"], rec["state"], rec["resumed"]) == (0, "receiving", False)
    # the connection drops halfway through a piece: what arrived is kept
    assert put(client, rec["id"], 0, data[:PIECE]).json()["offset"] == PIECE
    uploads.data_path(rec["id"]).open("ab").write(data[PIECE:PIECE + 1000])
    # a piece sent from where the page thought it was: refused, with the server's offset
    r = put(client, rec["id"], PIECE, data[PIECE:2 * PIECE])
    assert r.status_code == 409 and f"offset={PIECE + 1000}" in r.json()["detail"]
    # the page reloads, the same file is picked again: the same upload, from there
    again = start(client, data).json()
    assert (again["id"], again["offset"], again["resumed"]) == (rec["id"], PIECE + 1000, True)
    send_all(client, rec["id"], data, PIECE + 1000)
    got = finished(client, rec["id"])
    import hashlib
    assert got["state"] == "ready" and got["sha256"] == hashlib.sha256(data).hexdigest()
    assert got["header"]["arch"] == "qwen3moe" and got["header"]["chat_template"] is True
    # added under its details: a GGUF registered as the path form registers one
    r = client.post(f"/api/uploads/{rec['id']}/add", json={
        "name": "Qwen3.6 k4 phone", "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "k4 + LDA, Q4",
        "total": "35B", "active": "3B", "download": False, "by": ME})
    assert r.status_code == 200, r.text
    mid = r.json()["id"]
    assert mid == "gguf/Qwen3.6-k4-phone"
    g = db.gguf_get(mid)
    assert g["path"] == str(config.UPLOADS_DIR / "gguf" / "Qwen3.6-k4-phone.gguf")
    assert g["pin"]["sha256"] == got["sha256"] and g["flags"] == ["-ngl", "99", "--cpu-moe"]
    assert (config.UPLOADS_DIR / "gguf" / "Qwen3.6-k4-phone.gguf").read_bytes() == data
    assert db.sizes_all()[mid]["total"] == 35e9
    # kept, in view — and the part under way is gone
    st = client.get("/api/uploads").json()
    assert st["line"] == "Uploads: 0 of 150 GB" and st["pending"] == []
    [row] = st["files"]
    assert (row["model"], row["by"], row["download"], row["bytes"]) == (mid, ME, False, len(data))
    # refused while a run uses it; deleted after, its results left
    sid = db.add(mid, "instruct", "gguf", ME, "")
    r = client.delete(f"/api/uploads/file/{mid}")
    assert r.status_code == 409 and f"run #{sid} uses it" in r.json()["detail"]
    db.update(sid, status="done")
    r = client.delete(f"/api/uploads/file/{mid}")
    assert r.status_code == 200 and r.json()["note"] == "Its results stay on the board."
    assert not (config.UPLOADS_DIR / "gguf" / "Qwen3.6-k4-phone.gguf").exists()
    assert db.gguf_get(mid)                        # the model stays, with what it measured


def test_a_file_that_isnt_a_gguf_is_refused_and_nothing_is_kept(svc):
    client = svc
    data = b"PK not a model at all " * 300
    rec = start(client, data, "notes.gguf").json()
    send_all(client, rec["id"], data)
    got = finished(client, rec["id"])
    assert got["state"] == "failed"
    assert got["error"] == "notes.gguf isn't a GGUF file: its header can't be read as one. " \
                           "Nothing was kept."
    # said first, then removed (uploads._failed): the bytes go a moment after
    # the record says "failed" — a CI run once looked in between
    for _ in range(100):
        if not uploads.data_path(rec["id"]).exists():
            break
        time.sleep(0.02)
    assert not uploads.data_path(rec["id"]).exists()


def test_neither_a_gguf_nor_a_zip_is_refused_at_once(svc):
    r = start(svc, b"x" * 10, "model.safetensors")
    assert r.status_code == 422 and r.json()["detail"].startswith("A .gguf file, or a model folder")


# ---------------------------------------------------------------------------
# refused before it starts, with the numbers
# ---------------------------------------------------------------------------

def test_a_file_over_the_limit_is_refused_with_its_numbers(svc, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_MAX_GB", 0.00001)          # 10 kB
    r = start(svc, b"x" * 20_000)
    assert r.status_code == 413
    assert r.json()["detail"] == ("This file is 0.0 GB: one upload is at most 0.0 GB "
                                  "(UPLOAD_MAX_GB).")


def test_uploads_over_the_quota_are_refused_with_what_is_in_use(svc, monkeypatch):
    monkeypatch.setattr(uploads, "used", lambda: int(140e9))
    r = svc.post("/api/uploads", json={"filename": "big.gguf", "size": int(20e9), "by": ME})
    assert r.status_code == 507
    assert r.json()["detail"] == ("Uploads would hold 160.0 GB, over the 150.0 GB they may "
                                  "(ARTIFACT_QUOTA_GB): 140.0 GB are in use. Delete one you no "
                                  "longer need.")


def test_a_disk_that_would_keep_under_50_gb_refuses_it(svc, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_FREE_GB", 50.0)
    monkeypatch.setattr(uploads, "free", lambda: int(60e9))
    r = svc.post("/api/uploads", json={"filename": "big.gguf", "size": int(20e9), "by": ME})
    assert r.status_code == 507
    assert r.json()["detail"] == ("The disk would keep 40.0 GB free after this file, and an "
                                  "upload leaves at least 50.0 GB (UPLOAD_FREE_GB). It has "
                                  "60.0 GB free now.")
    # and a disk that fills while it comes stops the next piece
    monkeypatch.setattr(uploads, "free", lambda: int(500e9))
    data = model_file()
    rec = start(svc, data).json()
    monkeypatch.setattr(uploads, "free", lambda: int(50e9))
    r = put(svc, rec["id"], 0, data[:PIECE])
    assert r.status_code == 507 and "UPLOAD_FREE_GB" in r.json()["detail"]


def test_two_uploads_with_one_name(svc):
    client = svc
    a = start(client, model_file(), "a.gguf", name="Phone build").json()
    r = start(client, model_file(200), "b.gguf", name="Phone build")
    assert r.status_code == 409 and "another upload of that name is under way (a.gguf)" in \
        r.json()["detail"]
    data = model_file()
    send_all(client, a["id"], data)
    finished(client, a["id"])
    client.post(f"/api/uploads/{a['id']}/add", json={"how": "Q4", "by": ME})
    r = start(client, model_file(200), "c.gguf", name="Phone build")
    assert r.status_code == 409 and "a model of that name is on the board already" in \
        r.json()["detail"]


# ---------------------------------------------------------------------------
# a zip: today's checks
# ---------------------------------------------------------------------------

def zipped(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


@pytest.mark.parametrize("files,why", [
    ({"m/config.json": b"{}", "m/pytorch_model.bin": b"x"}, "Pickle-format weights (*.bin)"),
    ({"m/model.safetensors": b"x"}, "No config.json at the folder's root"),
    ({"config.json": b"{}", "../escape.txt": b"x"}, "would land outside its folder")])
def test_a_zip_with_today_s_checks(svc, files, why):
    client = svc
    data = zipped(files)
    rec = start(client, data, "ckpt.zip").json()
    send_all(client, rec["id"], data)
    got = finished(client, rec["id"])
    assert got["state"] == "failed" and why in got["error"], got["error"]
    assert not (config.ARTIFACTS_DIR / "ckpt").exists()


def test_a_model_folder_zip_is_added_as_an_uploaded_model(svc):
    client = svc
    data = zipped({"step900/config.json": json.dumps({"model_type": "llama"}).encode(),
                   "step900/model.safetensors": b"\0" * 5000})
    rec = start(client, data, "step900.zip").json()
    send_all(client, rec["id"], data)
    got = finished(client, rec["id"])
    assert got["state"] == "ready" and {f["file"] for f in got["files"]} == {
        "config.json", "model.safetensors"}
    r = client.post(f"/api/uploads/{rec['id']}/add", json={"by": ME})
    assert r.status_code == 200 and r.json()["id"] == "local/step900"
    assert (config.ARTIFACTS_DIR / "step900" / "config.json").exists()
    assert db.upload_get("local/step900")["download"] is True


def test_the_api_clients_zip_has_the_same_checks(svc):
    r = svc.post("/api/artifacts/ckpt", content=zipped({"config.json": b"{}", "w.bin": b"x"}))
    assert r.status_code == 422 and "Pickle-format weights" in r.json()["detail"]
    r = svc.post("/api/artifacts/ok", content=zipped({"config.json": b"{}",
                                                      "model.safetensors": b"x"}))
    assert r.status_code == 200 and r.json()["model_id"] == "local/ok"


# ---------------------------------------------------------------------------
# cancelled and abandoned uploads are cleaned up
# ---------------------------------------------------------------------------

def test_a_cancelled_or_abandoned_upload_is_removed(svc):
    client = svc
    data = model_file()
    a = start(client, data, "a.gguf").json()
    put(client, a["id"], 0, data[:PIECE])
    assert client.delete(f"/api/uploads/{a['id']}").status_code == 200
    assert not uploads.data_path(a["id"]).exists()
    b = start(client, data, "b.gguf").json()
    put(client, b["id"], 0, data[:PIECE])
    gone = uploads.sweep(time.time() + uploads.STALE_S + 1)
    assert gone == [b["id"]] and not uploads.data_path(b["id"]).exists()
    assert client.get(f"/api/uploads/{b['id']}").status_code == 404


# ---------------------------------------------------------------------------
# the path form, and a served model's GGUF: nothing that reads or writes a file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("f,why", [
    ({"path": "/etc/shadow"}, "ends in .gguf"),
    ({"path": "/home/masein/../root/x.gguf"}, "ends in .gguf"),
    ({"path": "/m/x.gguf", "flags": "-ngl 99 -o /tmp/out"}, "-o names a file"),
    ({"path": "/m/x.gguf", "flags": "--kl-divergence-base base.kld"}, "names a file"),
    ({"path": "/m/x.gguf", "flags": "-ngl 99 --chunks ../x"}, "is a path"),
    ({"path": "/m/x.gguf", "setups": "look: LLAMA_X=1 --log-file /tmp/l"}, "look: --log-file")])
def test_the_path_form_refuses_what_reads_or_writes_a_file(svc, f, why):
    r = svc.post("/api/gguf/models", json={"name": "x", "how": "Q4", "by": ME, **f})
    assert r.status_code == 422 and why in r.json()["detail"], r.json()
    # a kept one is run as it was: only a registration is checked
    assert gguf.check_flags(["-ngl", "99", "--cpu-moe", "-c", "4096"]) == \
        ["-ngl", "99", "--cpu-moe", "-c", "4096"]


def test_a_served_models_gguf_path_and_flags_are_checked_too(svc):
    from fake_openai import FakeServer
    fake = FakeServer()
    try:
        r = svc.post("/api/served", json={"name": "lda", "base_url": fake.base, "how": "k4",
                                          "thinking": "off", "by": ME,
                                          "gguf_path": "/m/x.gguf", "gguf_flags": "-o /tmp/x"})
        assert r.status_code == 422 and "-o names a file" in r.json()["detail"]
        r = svc.post("/api/served", json={"name": "lda", "base_url": fake.base, "how": "k4",
                                          "thinking": "off", "by": ME, "gguf_path": "/etc/passwd"})
        assert r.status_code == 422 and "ends in .gguf" in r.json()["detail"]
    finally:
        fake.close()
