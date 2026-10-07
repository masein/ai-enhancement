"""17j: the public file a model is — a Hugging Face repository and a path,
given on its page and checked here, so the raw-run export's typed-yes list can
tell an unmodified public file (unsloth's Qwen3.6 BF16) from a team build
marked public by mistake: both read "from hf://<account>/…" before.

The check asks Hugging Face as anyone would, with no token — a private or
gated file answers "sign in" — for the sha256 it publishes for the file (the
LFS object's, `X-Linked-Etag`), and compares it with the sha256 the board
registered for the model's file; for a GGUF in parts, each part's. A model the
board runs by its Hugging Face id (no file of its own) is checked as that
repository: public when Hugging Face shows it to anyone. The result is kept
with when and by whom; the export compares it again with the file registered
then, so a file registered since is not "checked"."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from . import db

HF = "https://huggingface.co"
TIMEOUT_S = 20
REPO = re.compile(r"[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}")
PATH = re.compile(r"(?:[\w.+-]{1,160}/){0,6}[\w.+-]{1,200}")
_SHA = re.compile(r"[0-9a-f]{64}")


class NotPublic(Exception):
    """why Hugging Face didn't show the file to anyone, in words"""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):            # the 302 carries the sha256
        return None


def _open(req: urllib.request.Request):
    return urllib.request.build_opener(_NoRedirect).open(req, timeout=TIMEOUT_S)


def _ask(url: str, method: str = "HEAD") -> tuple[dict, bytes]:
    """(the answer's headers, its body) — as anyone asks: no token"""
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "evalboard"})
    try:
        with _open(req) as r:
            return dict(r.headers), (r.read() if method == "GET" else b"")
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return dict(e.headers), b""
        if e.code in (401, 403):
            raise NotPublic("Hugging Face asks to sign in for it: it is private or gated") \
                from None
        if e.code == 404:
            raise NotPublic("Hugging Face has no such file there") from None
        raise NotPublic(f"Hugging Face answered {e.code}") from None
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise NotPublic(f"Hugging Face couldn't be asked ({getattr(e, 'reason', e)})") from None


def file_sha(repo: str, path: str) -> tuple[str, str]:
    """(the sha256 Hugging Face publishes for a file, the commit it read) —
    NotPublic when it doesn't show the file to anyone, or has no sha256 for it
    (a file kept outside LFS)"""
    h, _ = _ask(f"{HF}/{repo}/resolve/main/{urllib.parse.quote(path)}")
    low = {k.lower(): v for k, v in h.items()}
    tag = (low.get("x-linked-etag") or low.get("etag") or "").removeprefix("W/").strip('"')
    if not _SHA.fullmatch(tag):
        raise NotPublic("Hugging Face gives no sha256 for it (not a file kept in LFS)")
    return tag, str(low.get("x-repo-commit") or "")[:40]


def repo_public(repo: str) -> str:
    """'' when Hugging Face shows the repository to anyone; else why not"""
    _, body = _ask(f"{HF}/api/models/{repo}", "GET")
    try:
        got = json.loads(body or b"{}")
    except ValueError:
        return "Hugging Face's answer couldn't be read"
    if got.get("private"):
        return "the repository is private"
    return ""


def ours(model: str) -> dict:
    """the file the board registered for a model: {sha256, name, parts:
    [{name, sha256}]}; {} for a model with no file of its own"""
    from . import served
    try:
        rec = (served.get(model) if served.is_served(model) else None) or {}
    except Exception:                                   # noqa: BLE001 — no file, said
        rec = {}
    fs, gp = rec.get("file_sha256") or {}, rec.get("gguf_pin") or {}
    sha = gp.get("sha256") or fs.get("sha256") or ""
    parts = [{"name": str(x.get("name") or ""), "sha256": str(x.get("sha256") or "")}
             for x in fs.get("parts") or [] if isinstance(x, dict)]
    name = gp.get("name") or fs.get("name") or (rec.get("pin") or {}).get("file") or ""
    return {k: v for k, v in (("sha256", sha), ("name", name), ("parts", parts)) if v}


def check(model: str, repo: str, path: str) -> dict:
    """the record kept for a model's public file: {repo, path, sha256 (Hugging
    Face's), commit, same, why, at} — `same` only when Hugging Face showed it to
    anyone and its sha256 is the registered file's (each part's, for a GGUF in
    parts); `why` says what didn't hold"""
    repo, path = repo.strip().strip("/"), path.strip().strip("/")
    if not REPO.fullmatch(repo):
        raise ValueError("a Hugging Face repository is <account>/<name>, as in "
                         "unsloth/Qwen3.6-35B-A3B-GGUF")
    if path and (not PATH.fullmatch(path) or ".." in path.split("/")):
        raise ValueError("the file's path in that repository, as in BF16/x-00001-of-00002.gguf")
    mine = ours(model)
    out: dict = {"repo": repo, "path": path, "at": time.time(), "same": False, "why": ""}
    try:
        if not path:
            # a model run by its Hugging Face id: that repository, shown to anyone
            if not mine and repo.lower() == model.removesuffix(" · thinking").lower():
                out["why"] = repo_public(repo)
                out["same"] = not out["why"]
            else:
                out["why"] = ("give the file's path: the board runs a file of this model's, "
                              "not its repository" if mine else
                              "the repository isn't this model's own id: give the file's path")
            return out
        if not mine.get("sha256"):
            out["why"] = "the board has no sha256 registered for this model's file"
            return out
        parts = mine.get("parts") or []
        if parts:
            folder = path.rsplit("/", 1)[0] + "/" if "/" in path else ""
            theirs = []
            for p in parts:
                sha, commit = file_sha(repo, folder + p["name"])
                theirs.append({"name": p["name"], "sha256": sha})
            out.update(sha256=mine["sha256"], parts=theirs, commit=commit)
            bad = [p["name"] for p, t in zip(parts, theirs) if p["sha256"] != t["sha256"]]
            out["why"] = (f"{len(bad)} of its {len(parts)} parts differ from the files there "
                          f"({', '.join(bad[:3])})" if bad else "")
        else:
            sha, commit = file_sha(repo, path)
            out.update(sha256=sha, commit=commit)
            out["why"] = ("" if sha == mine["sha256"] else
                          f"its sha256 there is {sha[:16]}…, the registered file's "
                          f"{mine['sha256'][:16]}…")
        out["same"] = not out["why"]
    except NotPublic as e:
        out["why"] = str(e)
    return out


def still_same(model: str, rec: dict | None) -> bool:
    """the check still holds for the file registered now: the record's sha256
    is the registered one (a file registered since the check isn't checked)"""
    if not rec or not rec.get("same"):
        return False
    if not rec.get("path"):
        return not ours(model)
    return bool(rec.get("sha256")) and rec.get("sha256") == ours(model).get("sha256")


def words(model: str, rec: dict | None) -> str:
    """the typed-yes list's words: "the same file as unsloth/…, checked" when
    it holds; else '' (the list then says CHECK)"""
    if not still_same(model, rec):
        return ""
    where = rec["repo"] + (f"/{rec['path']}" if rec.get("path") else "")
    what = "the same file as" if rec.get("path") else "the public repository"
    return f"{what} {where}, checked"


def get(model: str) -> dict | None:
    return db.public_files_all().get(model.removesuffix(" · thinking"))


# ---------------------------------------------------------------------------
# the names this server is opened by — for the export's scrub (export_safe)
# ---------------------------------------------------------------------------

SEEN_MAX = 20
_seen: set | None = None
_IP = re.compile(r"[\d.]+|\[?[0-9a-fA-F:]+\]?")
_NOT_NAMES = {"localhost", "testserver"}


def seen_path():
    from . import config
    return config.BENCH_ROOT / "seen-hosts.json"


def seen_hosts() -> list[str]:
    """the host names a browser opened the board by (the tailnet's name,
    this server's) — an address or localhost never"""
    try:
        got = json.loads(seen_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [str(x) for x in got if isinstance(x, str)][:SEEN_MAX] if isinstance(got, list) else []


def seen(host_header: str) -> None:
    """17j: a request's Host, kept when it is a name not seen before"""
    global _seen
    name = (host_header or "").rsplit(":", 1)[0].strip().lower() \
        if not (host_header or "").startswith("[") else ""
    if (not name or _IP.fullmatch(name) or name in _NOT_NAMES or name.endswith(".localhost")
            or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,252}", name)):
        return
    if _seen is None:
        _seen = set(seen_hosts())
    if name in _seen or len(_seen) >= SEEN_MAX:
        return
    _seen.add(name)
    try:
        p = seen_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        tmp.write_text(json.dumps(sorted(_seen)), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        pass
