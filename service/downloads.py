"""16b.2: a model's file, downloaded — to try it on one's own machine.

- **What can be downloaded:** a model file on this server — an uploaded or
  registered GGUF (its header must read as one), or a model folder under
  ARTIFACTS_DIR, as one archive — and only when the person who added it
  allows it. A Hugging Face model is fetched from Hugging Face at the commit
  our runs loaded; a model served elsewhere has no file here.
- **Who:** the board's token. From a browser, which can't send a header on a
  link, the page asks (with the token) for a link made for that one download,
  kept a day; with curl, the token comes from an environment variable and is
  never in the command. Each download is logged with who and when.
- **How:** streamed from disk with HTTP Range, so `curl -C -` and a browser
  resume; never a whole file in memory, and never gzipped (that breaks a
  range). A folder's archive is an uncompressed zip made on its first
  download, kept a day, and refused if it would leave the disk under
  UPLOAD_FREE_GB.
- **Only model files:** nothing here reaches a question bank, an answer key or
  a hidden set — a GGUF path is served only when its header is a GGUF's, a
  folder only from under ARTIFACTS_DIR.

The Qwen3.6 phone builds (GGUFs registered by their path) start with
downloads off (16b decision 2); an upload starts as its form said."""

from __future__ import annotations

import secrets
import sys
import threading
import time
import zipfile
from pathlib import Path

from . import config, db, served, uploads

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gguf_header  # noqa: E402

LINK_S = 24 * 3600
ARCHIVE_S = 24 * 3600
OFF_BY_DEFAULT = "Downloads start off for a GGUF registered by its path (16b decision 2)"
_links: dict[str, dict] = {}
_lock = threading.Lock()
_building: dict[str, dict] = {}


class Refused(ValueError):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def archives() -> Path:
    return config.UPLOADS_DIR / ".downloads"


# ---------------------------------------------------------------------------
# a model's file
# ---------------------------------------------------------------------------

def _gguf_path(mid: str) -> tuple[str, dict]:
    """(path, record) of a model's GGUF on this server, or ('', {})"""
    up = db.upload_get(mid) or {}
    if up.get("kind") == "gguf":
        return up["path"], up
    g = db.gguf_get(mid) if mid.startswith("gguf/") else None
    if g:
        return g.get("path") or "", g
    s = served.get(mid)
    if s and s.get("gguf_path") and not served.is_openrouter(s):
        return s["gguf_path"], s
    return "", {}


def _readable(p: str) -> Path | None:
    try:
        f = Path(p).expanduser()
        return f if f.is_file() and f.stat().st_size > 0 else None
    except OSError:
        return None


def allowed(mid: str) -> tuple[bool, str, str]:
    """(allowed, who decided, the default's reason when no one has)"""
    row = db.download_get(mid)
    if row:
        return bool(row["allowed"]), row["by"], ""
    up = db.upload_get(mid)
    if up:
        return bool(up.get("download", True)), up.get("by") or "", ""
    if mid.startswith("local/"):
        return True, "", ""                       # a folder uploaded with the API client
    return False, "", OFF_BY_DEFAULT


def adder(mid: str) -> str:
    """who added it: who may switch its downloads"""
    for rec in (db.upload_get(mid), db.gguf_get(mid) if mid.startswith("gguf/") else None,
                served.get(mid)):
        if rec and rec.get("by"):
            return rec["by"]
    return ""


def info(mid: str) -> dict:
    """what the page says: {kind, name, bytes, sha256, allowed, by, line, …}.
    kind: gguf · folder · hub · served · none"""
    ok, by, why_off = allowed(mid)
    base = {"model": mid, "allowed": ok, "decided_by": by, "off_why": why_off,
            "adder": adder(mid), "downloads": db.download_count(mid)}
    if mid.startswith("local/"):
        d = config.ARTIFACTS_DIR / mid[len("local/"):]
        if not d.is_dir():
            return {**base, "kind": "none", "line": "Its folder isn't on this server any more: "
                    "its results stay on the board."}
        up = db.upload_get(mid) or {}
        return {**base, "kind": "folder", "name": d.name + ".zip",
                "bytes": uploads.dir_bytes(d), "sha256": None,
                "files": sum(1 for f in d.rglob("*") if f.is_file()),
                "archive": _archive_state(mid), "added": up.get("at")}
    path, rec = _gguf_path(mid)
    if path:
        f = _readable(path)
        if not f:
            return {**base, "kind": "none", "line": "Its file isn't readable by the board: it "
                    "is on the host outside the folders the board sees."}
        sha = rec.get("sha256") or (rec.get("pin") or {}).get("sha256") or \
            (rec.get("gguf_pin") or {}).get("sha256")
        return {**base, "kind": "gguf", "name": f.name, "bytes": f.stat().st_size,
                "sha256": sha, "path_known": True}
    s = served.get(mid)
    if s:
        return {**base, "kind": "served", "line": (
            "Served by OpenRouter: there is no file to download." if served.is_openrouter(s)
            else "Served elsewhere, and its file isn't registered here: there is nothing to "
                 "download from this board.")}
    return {**base, "kind": "hub"}


def set_allowed(mid: str, on: bool, by: str) -> dict:
    """the person who added it, or the board's owner, switches its downloads"""
    by = (by or "").strip()[:80]
    who = adder(mid)
    if not by:
        raise Refused(422, "Who is switching it: a name")
    if who and by.lower() != who.lower() and not config.is_owner(by):
        raise Refused(403, f"Only {who}, who added it, switches its downloads.")
    if info(mid)["kind"] not in ("gguf", "folder"):
        raise Refused(409, "It has no file on this board to download.")
    db.download_set(mid, on, by)
    return info(mid)


# ---------------------------------------------------------------------------
# the file, for one download
# ---------------------------------------------------------------------------

def file_for(mid: str) -> tuple[Path, str]:
    """(path, file name) to stream — or Refused: not allowed, not a model file,
    or its archive still being made (202)"""
    i = info(mid)
    if i["kind"] not in ("gguf", "folder"):
        raise Refused(404, i.get("line") or f"{mid} has no file on this board: it is on "
                                             "Hugging Face.")
    if not i["allowed"]:
        raise Refused(403, f"Downloads of {mid} are switched off"
                           + (f" by {i['decided_by']}" if i["decided_by"] else "") + ".")
    if i["kind"] == "gguf":
        f = _readable(_gguf_path(mid)[0])
        # only a model file: its header is a GGUF's — never a bank or a key
        if not f or not gguf_header.read(f):
            raise Refused(404, f"{mid}'s file isn't a GGUF the board can read.")
        return f, f.name
    a = _archive_state(mid)
    if a["state"] != "ready":
        build_archive(mid)
        raise Refused(202, f"Its archive is being made: {a.get('words') or 'starting'}. "
                           "Ask again in a moment.")
    return Path(a["path"]), Path(mid).name + ".zip"


def link(mid: str, who: str) -> dict:
    """a link for one download from a browser — kept a day, no token in it"""
    who = (who or "").strip()[:80]
    if not who:
        raise Refused(422, "Who is downloading it: a name")
    _, name = file_for(mid)
    tid = secrets.token_urlsafe(24)
    with _lock:
        now = time.time()
        for k in [k for k, v in _links.items() if v["until"] < now]:
            del _links[k]
        _links[tid] = {"model": mid, "who": who, "until": now + LINK_S}
    return {"url": f"api/dl/{tid}/{name}", "until": now + LINK_S, "name": name}


def by_link(tid: str) -> dict:
    with _lock:
        got = _links.get(tid)
    if not got or got["until"] < time.time():
        raise Refused(404, "This download link has ended: ask the model's page for another.")
    return got


def logged(mid: str, who: str, how: str, start: int) -> None:
    """each download, with who and when; a resumed one says where it went on"""
    db.download_log(mid, (who or "").strip()[:80] or "someone", how, start)


# ---------------------------------------------------------------------------
# a folder's archive: an uncompressed zip, made once, kept a day
# ---------------------------------------------------------------------------

def _archive_path(mid: str) -> Path:
    d = config.ARTIFACTS_DIR / mid[len("local/"):]
    stamp = int(max((f.stat().st_mtime for f in d.rglob("*") if f.is_file()), default=0))
    return archives() / f"{d.name}-{stamp}.zip"


def _archive_state(mid: str) -> dict:
    sweep()
    with _lock:
        b = _building.get(mid)
    if b and b["state"] != "ready":
        return b
    try:
        p = _archive_path(mid)
    except OSError:
        return {"state": "none"}
    if p.exists():
        return {"state": "ready", "path": str(p), "bytes": p.stat().st_size}
    return b or {"state": "none"}


def build_archive(mid: str) -> None:
    with _lock:
        if (_building.get(mid) or {}).get("state") == "building":
            return
        _building[mid] = {"state": "building", "words": "starting"}
    threading.Thread(target=_build, args=(mid,), daemon=True).start()


def _build(mid: str) -> None:
    d = config.ARTIFACTS_DIR / mid[len("local/"):]
    out = _archive_path(mid)
    total = uploads.dir_bytes(d)
    floor = uploads.limits()["floor"]
    f = uploads.free()
    if f is not None and f - total < floor:
        with _lock:
            _building[mid] = {"state": "refused", "words": (
                f"making it would leave {uploads.gb_words(max(0, f - total))} free, and the disk "
                f"keeps {uploads.gb_words(floor)} (UPLOAD_FREE_GB)")}
        return
    archives().mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    done = 0
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for p in sorted(x for x in d.rglob("*") if x.is_file()):
                z.write(p, f"{d.name}/{p.relative_to(d)}")
                done += p.stat().st_size
                with _lock:
                    _building[mid] = {"state": "building", "words":
                                      f"{uploads.gb_words(done)} of {uploads.gb_words(total)}"}
        tmp.replace(out)
        with _lock:
            _building[mid] = {"state": "ready", "path": str(out), "bytes": out.stat().st_size}
    except OSError as e:
        tmp.unlink(missing_ok=True)
        with _lock:
            _building[mid] = {"state": "refused", "words": f"it couldn't be made: {e}"}


def sweep(now: float | None = None) -> None:
    """archives older than a day are removed"""
    now = now or time.time()
    for p in archives().glob("*.zip") if archives().is_dir() else []:
        try:
            if now - p.stat().st_mtime > ARCHIVE_S:
                p.unlink()
        except OSError:
            continue
    for p in archives().glob("*.part") if archives().is_dir() else []:
        try:
            if now - p.stat().st_mtime > ARCHIVE_S:
                p.unlink()
        except OSError:
            continue


def log_rows(mid: str, limit: int = 20) -> list[dict]:
    return db.download_rows(mid, limit)

