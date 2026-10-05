"""16b.2: Download — a model file on this server, streamed with HTTP Range
so a download resumes, behind the board's token, each download logged.

- A GGUF uploaded here: its name, size and sha256; a link for one browser
  download (no token in it), and the token route curl uses; a range resumes;
  never gzipped; logged with who and when.
- Who may: the upload's word, switched by the person who added it (or the
  owner) and no one else; a GGUF registered by its path starts off.
- Only model files: a "GGUF" whose header isn't one is never served; nothing
  else is reachable.
- A model folder comes as one archive, made on its first download.
- A Hugging Face model: its commit, from the run; a served model: no file.

Fake files; nothing runs."""

from __future__ import annotations

import io
import json
import time
import zipfile

import pytest

from conftest import make_service
from service import config, db, downloads, uploads
from test_16_1_sizes import gguf_bytes

ME = "masein"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(uploads, "PIECE", 1 << 20)
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "UPLOAD_FREE_GB", 0.0)
    downloads._links.clear()
    yield client, tree
    client.__exit__(None, None, None)


def upload(client, data: bytes, filename="phone.gguf", **f) -> str:
    rec = client.post("/api/uploads", json={"filename": filename, "size": len(data),
                                            "mtime": 1.0, "by": ME}).json()
    assert client.put(f"/api/uploads/{rec['id']}?offset=0", content=data).status_code == 200
    client.post(f"/api/uploads/{rec['id']}/finish")
    for _ in range(200):
        if client.get(f"/api/uploads/{rec['id']}").json()["state"] != "checking":
            break
        time.sleep(0.02)
    r = client.post(f"/api/uploads/{rec['id']}/add", json={"how": "Q4", "by": ME, **f})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def model_bytes() -> bytes:
    return gguf_bytes() + bytes(range(256)) * 40


def test_an_uploaded_gguf_is_downloaded_whole_or_from_where_it_stopped(svc):
    client, _ = svc
    data = model_bytes()
    mid = upload(client, data, name="Phone k4")
    import hashlib
    i = client.get("/api/models/file", params={"id": mid}).json()
    assert (i["kind"], i["name"], i["bytes"], i["sha256"], i["allowed"]) == (
        "gguf", "Phone-k4.gguf", len(data), hashlib.sha256(data).hexdigest(), True)
    # a browser's: a link made for this one download, no token in it
    link = client.post("/api/models/file/link", json={"model": mid, "by": "omar"}).json()
    assert link["url"].startswith("api/dl/") and link["url"].endswith("/Phone-k4.gguf")
    r = client.get("/" + link["url"], headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200 and r.content == data
    assert r.headers.get("content-encoding") != "gzip"           # a range must stay exact
    assert "attachment" in r.headers["content-disposition"]
    # cut at 1,000 bytes: the rest, from there
    r = client.get("/" + link["url"], headers={"Range": "bytes=1000-"})
    assert r.status_code == 206 and r.content == data[1000:]
    # curl's: the token in a header, its name in another
    r = client.get("/api/download", params={"model": mid}, headers={"X-Who": "lina"})
    assert r.status_code == 200 and r.content == data
    # each logged: who, how, and where a resume went on
    rows = client.get("/api/models/file", params={"id": mid}).json()
    assert rows["downloads"] == 2
    assert [(x["who"], x["how"], x["start"]) for x in rows["log"]] == [
        ("lina", "token", 0), ("omar", "link", 1000), ("omar", "link", 0)]


def test_the_token_is_needed_and_never_in_a_link(svc, monkeypatch):
    client, _ = svc
    mid = upload(client, model_bytes())
    monkeypatch.setattr(config, "SUBMIT_TOKEN", "s3cret")
    assert client.post("/api/models/file/link", json={"model": mid, "by": ME}).status_code == 401
    assert client.get("/api/download", params={"model": mid}).status_code == 401
    r = client.get("/api/download", params={"model": mid}, headers={"X-Token": "s3cret"})
    assert r.status_code == 200
    link = client.post("/api/models/file/link", json={"model": mid, "by": ME},
                       headers={"X-Token": "s3cret"}).json()
    assert "s3cret" not in json.dumps(link)
    assert client.get("/" + link["url"]).status_code == 200      # the link is the permission
    # an ended link
    downloads._links[link["url"].split("/")[2]]["until"] = 0
    r = client.get("/" + link["url"])
    assert r.status_code == 404 and "This download link has ended" in r.json()["detail"]


def test_who_may_download_it_is_the_word_of_the_one_who_added_it(svc):
    client, _ = svc
    mid = upload(client, model_bytes(), download=False)
    # 16c review: who added it may take it while the switch is off; anyone else can't
    assert client.post("/api/models/file/link", json={"model": mid, "by": ME}).status_code == 200
    r = client.post("/api/models/file/link", json={"model": mid, "by": "omar"})
    assert r.status_code == 403 and r.json()["detail"] == f"Downloads of {mid} are switched off by masein."
    r = client.post("/api/models/file/allow", json={"model": mid, "allowed": True, "by": "omar"})
    assert r.status_code == 403 and r.json()["detail"] == \
        "Only masein, who added it, switches its downloads."
    r = client.post("/api/models/file/allow", json={"model": mid, "allowed": True, "by": ME})
    assert r.status_code == 200 and r.json()["allowed"] is True
    assert client.post("/api/models/file/link", json={"model": mid, "by": "omar"}).status_code == 200


def test_a_gguf_registered_by_its_path_starts_off_and_its_adder_or_the_owner_switches_it(
        svc, tmp_path):
    client, _ = svc
    f = tmp_path / "Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf"
    f.write_bytes(model_bytes())
    r = client.post("/api/gguf/models", json={"name": "Qwen3.6 k4-LDA", "path": str(f),
                                               "how": "Q4", "by": "omar"})
    mid = r.json()["model"]["id"]
    i = client.get("/api/models/file", params={"id": mid}).json()
    assert i["kind"] == "gguf" and i["allowed"] is False and i["off_why"] == downloads.OFF_BY_DEFAULT
    assert client.get("/api/download", params={"model": mid}).status_code == 403
    r = client.post("/api/models/file/allow", json={"model": mid, "allowed": True, "by": ME})
    assert r.status_code == 200                                 # the board's owner
    assert client.get("/api/download", params={"model": mid}).status_code == 200


def test_only_model_files_never_a_bank_named_as_one(svc, tmp_path):
    client, tree = svc
    bank = tmp_path / "hidden-bank.gguf"
    bank.write_text(json.dumps({"q": "a hidden question", "answer": "A"}))
    r = client.post("/api/gguf/models", json={"name": "not a model", "path": str(bank),
                                               "how": "Q4", "by": ME})
    mid = r.json()["model"]["id"]
    client.post("/api/models/file/allow", json={"model": mid, "allowed": True, "by": ME})
    r = client.get("/api/download", params={"model": mid})
    assert r.status_code == 404 and "isn't a GGUF the board can read" in r.json()["detail"]
    # and nothing that isn't a model on the board
    for model in ("../results", "exam/law", "everyday"):
        assert client.get("/api/download", params={"model": model}).status_code == 404


def test_a_model_folder_comes_as_one_archive_made_on_its_first_download(svc):
    client, _ = svc
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("ck/config.json", "{}")
        z.writestr("ck/model.safetensors", b"\0" * 3000)
    mid = upload(client, buf.getvalue(), "ck.zip")
    i = client.get("/api/models/file", params={"id": mid}).json()
    assert (i["kind"], i["name"], i["files"]) == ("folder", "ck.zip", 2)
    r = client.post("/api/models/file/link", json={"model": mid, "by": ME})
    assert r.status_code == 202 and r.json()["detail"].startswith("Its archive is being made")
    for _ in range(200):
        r = client.post("/api/models/file/link", json={"model": mid, "by": ME})
        if r.status_code == 200:
            break
        time.sleep(0.02)
    got = client.get("/" + r.json()["url"])
    with zipfile.ZipFile(io.BytesIO(got.content)) as z:
        assert sorted(z.namelist()) == ["ck/config.json", "ck/model.safetensors"]
        assert all(x.compress_type == zipfile.ZIP_STORED for x in z.infolist())


def test_a_hub_model_is_fetched_from_hugging_face_and_a_served_one_has_no_file(svc):
    client, tree = svc
    mid = next(iter(tree["models"]))
    assert client.get("/api/models/file", params={"id": mid}).json()["kind"] == "hub"
    from service import served
    rec = {"id": "served/x", "name": "x", "base_url": "http://h:1/v1", "how": "k4",
           "based_on": "", "thinking": "off", "pin": {}, "gguf_path": ""}
    db.served_put(rec)
    served.write_meta(rec)
    i = client.get("/api/models/file", params={"id": "served/x"}).json()
    # 16c: said plainly, and its file can be registered from its panel
    assert i["kind"] == "served" and i["can_register"] and i["line"] == (
        "Its file isn’t registered here, so there is nothing to download yet.")


def test_a_runs_hub_commit_is_on_its_row():
    import report_lm_eval as rep
    run = rep.parse_run({"results": {"arc_easy": {"acc,none": 0.5}},
                         "config": {"model": "hf", "model_args": "pretrained=org/m",
                                    "model_sha": "a" * 40}}, __import__("pathlib").Path("x.json"))
    assert run["model_sha"] == "a" * 40
