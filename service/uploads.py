"""16b.1: a model added from a browser — a .gguf file, or a model folder as a
.zip — sent in pieces, so a dropped connection or a reloaded page resumes.

- **Before it starts:** the file's size against the limit for one file
  (UPLOAD_MAX_GB), every upload and model folder together (ARTIFACT_QUOTA_GB),
  and the disk: none may leave less than UPLOAD_FREE_GB free. A refusal gives
  the numbers.
- **While it comes:** pieces of at most PIECE bytes, each at the offset the
  server has; the raw file body, never a form upload, so nothing passes
  through /tmp on the container's root disk (the disk that filled on 26 Sep).
  Written off the main loop (app.py).
- **After:** the sha256, worked out here; a GGUF's header read to confirm it
  is one and to fill in what it is (scripts/gguf_header.py: the header only);
  a zip with today's checks — config.json at its root, safetensors only (a
  `.bin` is refused: pickle runs code on load), no member escaping its folder.
  Nothing in an upload is ever run.
- **Then** it is added under the name and details a person gives: a GGUF
  registered as the "A GGUF file" form registers one (gguf.register), a folder
  as an uploaded model (local/<name>, as POST /api/artifacts makes one).

Under $BENCH_ROOT/uploads, so the host's GGUF worker sees the same path. An
upload not finished, or finished and never added, is removed after STALE_S
untouched."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

from . import config, db

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gguf_header  # noqa: E402

GB = 1e9
PIECE = 64 * 1024 * 1024
STALE_S = 24 * 3600
KINDS = {".gguf": "gguf", ".zip": "zip"}
_ID = re.compile(r"^[0-9a-f]{16}$")
# 16b review: a model folder's name, everywhere one is taken from a request —
# letters, digits, dot, dash, underscore; no slash, no "..", never a leading
# dot. "." and ".." matched the old rule, and became paths
FOLDER_NAME = re.compile(r"^[A-Za-z0-9_\-][A-Za-z0-9._\-]{0,79}$")


def folder_name_ok(name) -> bool:
    return isinstance(name, str) and bool(FOLDER_NAME.match(name)) and ".." not in name


def artifact_dir(name) -> Path | None:
    """the model folder a name names, or None: a good name, and — links and
    all, resolved — a folder directly inside ARTIFACTS_DIR"""
    if not folder_name_ok(name):
        return None
    try:
        root = config.ARTIFACTS_DIR.resolve(strict=True)
        d = (root / name).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return d if d.parent == root and d.is_dir() else None
# a run that hasn't ended (app.ACTIVE): its model's file stays
RUN_ACTIVE = ("queued", "preflight", "waiting_gpu", "waiting_lock", "running")


class Refused(ValueError):
    """an upload refused, with the HTTP status that says why"""
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def gb_words(n: float) -> str:
    return f"{n / GB:.1f} GB"


# ---------------------------------------------------------------------------
# where, and how much
# ---------------------------------------------------------------------------

def root() -> Path:
    return config.UPLOADS_DIR


def parts() -> Path:
    return root() / ".parts"


def gguf_dir() -> Path:
    return root() / "gguf"


def dir_bytes(d: Path) -> int:
    if not d.is_dir():
        return 0
    total = 0
    for f in d.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def used() -> int:
    """every upload and model folder, parts included: what the quota counts"""
    return dir_bytes(config.ARTIFACTS_DIR) + dir_bytes(root())


def free() -> int | None:
    """free bytes on the disk uploads are written to"""
    p = root()
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


def limits() -> dict:
    return {"max": int(config.UPLOAD_MAX_GB * GB), "quota": int(config.ARTIFACT_QUOTA_GB * GB),
            "floor": int(config.UPLOAD_FREE_GB * GB)}


def room_for(size: int, *, have: int = 0) -> str:
    """'' when `size` more bytes may be written (`have` of them already are),
    else why not — with the numbers"""
    lim, more = limits(), max(0, size - have)
    if size > lim["max"]:
        return (f"This file is {gb_words(size)}: one upload is at most "
                f"{gb_words(lim['max'])} (UPLOAD_MAX_GB).")
    u = used()
    if u + more > lim["quota"]:
        return (f"Uploads would hold {gb_words(u + more)}, over the {gb_words(lim['quota'])} "
                f"they may (ARTIFACT_QUOTA_GB): {gb_words(u)} are in use. Delete one you no "
                "longer need.")
    f = free()
    if f is not None and f - more < lim["floor"]:
        return (f"The disk would keep {gb_words(max(0, f - more))} free after this file, and an "
                f"upload leaves at least {gb_words(lim['floor'])} (UPLOAD_FREE_GB). It has "
                f"{gb_words(f)} free now.")
    return ""


# ---------------------------------------------------------------------------
# an upload in progress: a record and its bytes, under .parts/
# ---------------------------------------------------------------------------

def _uid(uid: str) -> None:
    """an upload's id from a request: 16 hex characters, or nothing is touched"""
    if not isinstance(uid, str) or not _ID.match(uid):
        raise Refused(404, "No such upload.")


def _rec_path(uid: str) -> Path:
    _uid(uid)
    return parts() / f"{uid}.json"


def data_path(uid: str) -> Path:
    _uid(uid)
    return parts() / f"{uid}.part"


def _dir_path(uid: str) -> Path:
    _uid(uid)
    return parts() / f"{uid}.dir"


def _save(rec: dict) -> None:
    parts().mkdir(parents=True, exist_ok=True)
    tmp = _rec_path(rec["id"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    tmp.replace(_rec_path(rec["id"]))


def get(uid: str) -> dict:
    if not _ID.match(uid or ""):
        raise Refused(404, "No such upload.")
    try:
        rec = json.loads(_rec_path(uid).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise Refused(404, "No such upload: it was finished, cancelled, or removed after a "
                           "day untouched.") from None
    return {**rec, "offset": offset(uid)}


def offset(uid: str) -> int:
    try:
        return data_path(uid).stat().st_size
    except OSError:
        return 0


def _all() -> list[dict]:
    out = []
    for f in sorted(parts().glob("*.json")) if parts().is_dir() else []:
        try:
            out.append(get(f.stem))
        except Refused:
            continue
    return out


def kind_of(filename: str) -> str:
    k = KINDS.get(Path(filename or "").suffix.lower())
    if not k:
        raise Refused(422, "A .gguf file, or a model folder as a .zip: "
                           f"{filename or 'this file'} is neither.")
    return k


def model_id(kind: str, name: str) -> str:
    from . import gguf
    return gguf.slug(name) if kind == "gguf" else f"local/{name}"


def check_name(kind: str, name: str, *, skip: str = "") -> str:
    """the model id it would be, or Refused: a name nothing else has"""
    from . import gguf
    name = (name or "").strip()
    if kind == "zip" and not folder_name_ok(name):
        raise Refused(422, "A name of letters, digits, dot, dash and underscore, not starting "
                           "with a dot: it is the folder's name on the server")
    if kind == "gguf" and not gguf.slug(name)[len(gguf.PREFIX):]:
        raise Refused(422, "A name: it is shown everywhere")
    mid = model_id(kind, name)
    taken = (db.gguf_get(mid) if kind == "gguf" else (config.ARTIFACTS_DIR / name).exists()) \
        or db.upload_get(mid) or (kind == "gguf" and (gguf_dir() / f"{mid[len(gguf.PREFIX):]}.gguf")
                                  .exists())
    if taken:
        raise Refused(409, f"{name}: a model of that name is on the board already. Give this "
                           "one a name of its own.")
    for other in _all():
        if other["id"] != skip and model_id(other["kind"], other["name"]) == mid:
            raise Refused(409, f"{name}: another upload of that name is under way "
                               f"({other['filename']}). Give this one a name of its own.")
    return mid


def start(f: dict, by: str) -> dict:
    """a new upload — or the same file's unfinished one, which resumes"""
    sweep()
    by = (by or "").strip()[:80]
    if not by:
        raise Refused(422, "Who is adding it: a name")
    filename = Path(str(f.get("filename") or "")).name[:200]
    kind = kind_of(filename)
    try:
        size = int(f.get("size") or 0)
        mtime = float(f.get("mtime") or 0)
    except (TypeError, ValueError):
        raise Refused(422, "The file's size and date, as numbers") from None
    if size <= 0:
        raise Refused(422, f"{filename} is empty.")
    # the same file picked again — after a dropped connection or a reload
    for rec in _all():
        if (rec["filename"], rec["size"], rec["mtime"]) == (filename, size, mtime) \
                and rec["state"] == "receiving":
            rec["touched"] = time.time()
            _save({k: v for k, v in rec.items() if k != "offset"})
            return {**rec, "resumed": True}
    name = (f.get("name") or Path(filename).stem).strip()[:120]
    check_name(kind, name)
    why = room_for(size)
    if why:
        raise Refused(507 if "disk" in why or "hold" in why else 413, why)
    now = time.time()
    rec = {"id": secrets.token_hex(8), "kind": kind, "name": name, "filename": filename,
           "size": size, "mtime": mtime, "by": by, "at": now, "touched": now,
           "state": "receiving"}
    _save(rec)
    data_path(rec["id"]).touch()
    return {**rec, "offset": 0, "resumed": False}


def at_offset(uid: str, at: int) -> dict:
    """the upload a piece may be written to at `at`, or Refused"""
    rec = get(uid)
    if rec["state"] != "receiving":
        raise Refused(409, f"{rec['filename']} has all its bytes: it is {rec['state']}.")
    if at != rec["offset"]:
        raise Refused(409, f"The server has {rec['offset']} bytes of {rec['filename']}: send "
                           f"from there (offset={rec['offset']}).")
    why = room_for(rec["size"], have=rec["offset"])
    if why and "disk" in why:                     # the disk filled meanwhile: stop here
        raise Refused(507, why)
    return rec


def touch(uid: str) -> None:
    try:
        _uid(uid)
        rec = json.loads(_rec_path(uid).read_text(encoding="utf-8"))
    except (OSError, ValueError, Refused):
        return
    rec["touched"] = time.time()
    _save(rec)


def cancel(uid: str) -> dict:
    rec = get(uid)
    _remove(uid)
    return {"cancelled": rec["filename"]}


def _remove(uid: str) -> None:
    data_path(uid).unlink(missing_ok=True)
    shutil.rmtree(_dir_path(uid), ignore_errors=True)
    _rec_path(uid).unlink(missing_ok=True)


def sweep(now: float | None = None) -> list[str]:
    """an upload untouched for a day, finished or not, is removed — its parts
    and its unpacked folder too. And a part no record names"""
    now = now or time.time()
    gone = []
    for rec in _all():
        if now - float(rec.get("touched") or rec.get("at") or 0) > STALE_S \
                and rec["state"] != "checking":
            _remove(rec["id"])
            gone.append(rec["id"])
    for p in parts().iterdir() if parts().is_dir() else []:
        uid = p.name.split(".", 1)[0]
        if p.suffix in (".part", ".dir") and not _rec_path(uid).exists():
            try:
                if now - p.stat().st_mtime > STALE_S:
                    shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
                    gone.append(uid)
            except OSError:
                continue
    return gone


# ---------------------------------------------------------------------------
# after its last byte: the sha256, and what it is
# ---------------------------------------------------------------------------

def finish(uid: str) -> dict:
    """every byte is in: check it in the background — the page asks how it went"""
    rec = get(uid)
    if rec["state"] != "receiving":
        return rec
    if rec["offset"] != rec["size"]:
        raise Refused(409, f"The server has {rec['offset']} of {rec['size']} bytes of "
                           f"{rec['filename']}: send the rest first.")
    rec = {k: v for k, v in rec.items() if k != "offset"}
    rec.update(state="checking", touched=time.time())
    _save(rec)
    threading.Thread(target=_check, args=(uid,), daemon=True).start()
    return {**rec, "offset": rec["size"]}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(1 << 22)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _check(uid: str) -> None:
    rec = json.loads(_rec_path(uid).read_text(encoding="utf-8"))
    path = data_path(uid)
    try:
        rec["sha256"] = sha256_of(path)
        if rec["kind"] == "gguf":
            head = gguf_header.read(path)
            if not head:
                raise Refused(422, f"{rec['filename']} isn't a GGUF file: its header can't be "
                                   "read as one. Nothing was kept.")
            rec["header"] = head
        else:
            rec["files"] = unpack_zip(path, _dir_path(uid), limits()["max"])
            path.unlink(missing_ok=True)           # the folder is what is kept
        rec.update(state="ready", error="")
    except Refused as e:
        _failed(uid, rec, str(e))
        return
    except Exception as e:                         # noqa: BLE001 — a disk error, said
        _failed(uid, rec, f"Couldn't check {rec['filename']}: {e}. Nothing was kept.")
        return
    rec["touched"] = time.time()
    _save(rec)


def _failed(uid: str, rec: dict, why: str) -> None:
    """said first — the record replaced in one step, so the page never asks
    between — then its bytes and its folder removed; the record is swept"""
    rec.update(state="failed", error=why, touched=time.time())
    _save(rec)
    data_path(uid).unlink(missing_ok=True)
    shutil.rmtree(_dir_path(uid), ignore_errors=True)


def unpack_zip(src: Path, dest: Path, cap: int) -> list[dict]:
    """a model folder from a zip, with the checks POST /api/artifacts makes:
    at most three times the cap unpacked, no member outside the folder,
    config.json at its root, and safetensors only (a .bin is pickle, which runs
    code when it loads). The disk keeps UPLOAD_FREE_GB. [{file, bytes}]"""
    try:
        z = zipfile.ZipFile(src)
    except (zipfile.BadZipFile, OSError) as e:
        raise Refused(422, f"Not a zip that can be read: {e}. Nothing was kept.") from e
    with z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        total = sum(i.file_size for i in infos)
        if total > cap * 3:
            raise Refused(413, f"The zip unpacks to {gb_words(total)}, more than three times "
                               f"the {gb_words(cap)} an upload may be.")
        f = free()
        if f is not None and f - total < limits()["floor"]:
            raise Refused(507, f"Unpacked, the zip would leave {gb_words(max(0, f - total))} "
                               f"free, and uploads leave at least "
                               f"{gb_words(limits()['floor'])} (UPLOAD_FREE_GB).")
        bins = [i.filename for i in infos if i.filename.endswith(".bin")]
        if bins:
            raise Refused(422, "Pickle-format weights (*.bin) run code when they load, and are "
                               f"refused ({bins[0]}): save the model with safetensors.")
        # a single shared top-level folder is the model's
        roots = {i.filename.split("/", 1)[0] for i in infos}
        strip = (roots.pop() + "/") if len(roots) == 1 and all(
            "/" in i.filename for i in infos) else ""
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        base = str(dest.resolve()) + "/"
        out = []
        try:
            for i in infos:
                rel = i.filename[len(strip):] if i.filename.startswith(strip) else i.filename
                target = (dest / rel).resolve()
                if not str(target).startswith(base):
                    raise Refused(422, f"A file in the zip would land outside its folder: "
                                       f"{i.filename}. Nothing was kept.")
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(i) as s, open(target, "wb") as o:
                    shutil.copyfileobj(s, o, 1 << 20)
                out.append({"file": rel, "bytes": i.file_size})
            if not (dest / "config.json").exists():
                raise Refused(422, "No config.json at the folder's root: zip the folder "
                                   "save_pretrained() made.")
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)
            raise
    return out


# ---------------------------------------------------------------------------
# added to the board
# ---------------------------------------------------------------------------

def register(uid: str, f: dict, by: str) -> dict:
    """a checked upload, added under its details: the model it is"""
    from . import gguf, sizes
    rec = get(uid)
    if rec["state"] != "ready":
        raise Refused(409, f"{rec['filename']} is {rec['state']}: it can be added once it is "
                           "checked.")
    by = (by or "").strip()[:80]
    if not by:
        raise Refused(422, "Who is adding it: a name")
    name = (f.get("name") or rec["name"]).strip()[:120]
    mid = check_name(rec["kind"], name, skip=uid)
    total, active = f.get("total"), f.get("active")
    if total not in (None, ""):
        try:
            sizes.check(total, active)
        except ValueError as e:
            raise Refused(422, str(e)) from e
    if rec["kind"] == "gguf":
        gguf_dir().mkdir(parents=True, exist_ok=True)
        path = gguf_dir() / f"{mid[len(gguf.PREFIX):]}.gguf"
        try:
            model = gguf.register({"name": name, "path": str(path),
                                   "based_on": f.get("based_on") or "", "how": f.get("how") or "",
                                   "flags": "", "setups": ""}, by, check_only=True)
        except ValueError as e:
            raise Refused(422, str(e)) from e
        os.replace(data_path(uid), path)
        os.chmod(path, 0o644)
        model = gguf.register({"name": name, "path": str(path),
                               "based_on": f.get("based_on") or "", "how": f.get("how") or "",
                               "flags": "", "setups": ""}, by,
                              pin={"sha256": rec["sha256"], "size": rec["size"],
                                   "name": path.name})
        nbytes = rec["size"]
    else:
        dest = config.ARTIFACTS_DIR / name
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        os.replace(_dir_path(uid), dest)
        path, nbytes = dest, dir_bytes(dest)
        model = {"id": mid, "name": name}
    if total not in (None, ""):
        sizes.set_size(mid, total, active, by)
    db.upload_put({"model": mid, "kind": rec["kind"], "name": name, "path": str(path),
                   "filename": rec["filename"], "bytes": nbytes, "sha256": rec["sha256"],
                   "by": by, "at": time.time(), "download": bool(f.get("download", True)),
                   "header": rec.get("header") or {}})
    _remove(uid)
    return {"model": model, "id": mid, "kind": rec["kind"], "sha256": rec["sha256"],
            "bytes": nbytes}


# ---------------------------------------------------------------------------
# what is kept, in view
# ---------------------------------------------------------------------------

def in_use(row: dict) -> str:
    """'' when a kept upload can go, else what uses it"""
    for r in db.recent(500):
        if r["hf_id"] == row["model"] and r["status"] in RUN_ACTIVE:
            return f"run #{r['id']} uses it"
    for s in db.served_all():
        if s.get("gguf_path") == row["path"]:
            return f"the served model {s.get('name') or s['id']} is its file"
    return ""


def storage() -> dict:
    """"Uploads: 62 of 150 GB", every file kept (size, date, who added it), and
    what is under way"""
    sweep()
    lim, u, f = limits(), used(), free()
    files = []
    for row in db.uploads_all():
        p = Path(row["path"])
        files.append({**row, "here": p.exists(), "busy": in_use(row)})
    pending = [{k: rec[k] for k in ("id", "kind", "name", "filename", "size", "offset", "by",
                                     "at", "state") if k in rec} | {"error": rec.get("error", "")}
               for rec in _all()]
    return {"used": u, "quota": lim["quota"], "max": lim["max"], "floor": lim["floor"],
            "free": f, "piece": PIECE,
            "line": f"Uploads: {u / GB:.0f} of {lim['quota'] / GB:.0f} GB",
            "files": files, "pending": pending}


def delete(mid: str) -> dict:
    """a kept upload's file removed — refused while a run uses it; its results
    stay on the board, as DELETE /api/artifacts leaves them"""
    row = db.upload_get(mid)
    if not row:
        raise Refused(404, f"{mid} wasn't uploaded here.")
    busy = in_use(row)
    if busy:
        raise Refused(409, f"{row['name']} can't be deleted now: {busy}.")
    p = Path(row["path"])
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink(missing_ok=True)
    db.upload_del(mid)
    return {"deleted": row["name"], "bytes": row["bytes"],
            "note": "Its results stay on the board."}
