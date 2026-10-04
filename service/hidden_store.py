"""12p.1: what lives only on the data volume, kept and checked.

Everyday's hidden half moves out of the repo — which is mirrored in public —
into BENCH_ROOT/everyday/hidden.jsonl, beside what the question builder
published and the edits; the Knowledge exam's questions are there already
(EXAM_DIR/bank). The repo keeps what each should be: Everyday's hidden set as
a count and a digest, the exam's report half as its qids (public already).
When either is missing or doesn't match, every page says so in red; without
Everyday's, Everyday runs and scoring stop until it is restored. The rest of
the board works.

    python -m service.hidden_store status
    python -m service.hidden_store move               once: the repo's hidden half into the store
    python -m service.hidden_store backup [--export [--out FILE]]
    python -m service.hidden_store restore [FILE] [--all]
    python -m service.hidden_store manifest           what to commit for the set on this server
    python -m service.hidden_store fingerprints --out FILE    12p.2: the CI guard's fingerprints

Backups go to BACKUP_DIR (the server's second disk, /data-03/evalboard-backups
by default), one .tar.gz a day, BACKUP_KEEP of them, each with the sha256 of
every file in it. `--export` also writes one file encrypted to an age public
key (EVALBOARD_BACKUP_AGE_RECIPIENT) to copy off the server: the server can
encrypt with it and never decrypt. Restoring puts back Everyday's hidden set
and any exam question missing from the store — never over a newer one; `--all`
puts back every file, the ones there now moved aside first. It prints counts,
never a question."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import threading
import time
from pathlib import Path

from . import config

PREFIX = "evalboard-store-"
EVERYDAY_FILES = ("hidden.jsonl", "built.jsonl", "edits.jsonl", "retired.jsonl", "groups.json")
DAY_S = 24 * 3600
_lock = threading.Lock()
_last_try = {"at": 0.0}


def _scripts():
    here = Path(__file__).resolve().parent.parent / "scripts"
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def _ev():
    _scripts()
    import everyday as ev
    return ev


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# what should be there
# ---------------------------------------------------------------------------

def exam_manifest() -> dict | None:
    try:
        return json.loads(Path(config.EXAM_REPORT_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def exam_status() -> dict:
    """{ok, state, count, missing, why}: every report-half question the repo's
    banks held is in the exam's store (retired or not)"""
    m = exam_manifest()
    if not m:
        return {"ok": True, "state": "unchecked", "count": 0, "missing": 0, "why": ""}
    want = {q for qs in m.get("topics", {}).values() for q in qs}
    _scripts()
    import exam_build as eb
    bank = eb.bank_dir(config.EXAM_DIR)
    have = {r.get("qid") for r in eb.all_bank_rows(config.EXAM_DIR)} if bank.is_dir() else set()
    gone = want - have
    if not gone:
        return {"ok": True, "state": "store", "count": len(want), "missing": 0, "why": ""}
    return {"ok": False, "state": "missing" if len(gone) == len(want) else "changed",
            "count": len(want) - len(gone), "missing": len(gone),
            "why": f"The Knowledge exam's report half is missing or changed ({len(gone):,} of "
                   f"{len(want):,} not in the store): restore it · {_ev().RESTORE}"}


def exam_manifest_of(root: Path) -> dict:
    """the report half's qids, per topic, of a folder of exam banks (the repo's)"""
    _scripts()
    import diagnose as dx
    import exam_build as eb
    topics: dict[str, list[str]] = {}
    for f in sorted(Path(root).glob("*_v1.json")):
        for it in json.loads(f.read_text(encoding="utf-8")):
            qid = eb.qid_of(it["prompt"])
            if dx.split_of(qid) == "report":
                topics.setdefault(f.name[:-len("_v1.json")], []).append(qid)
    return {"count": sum(len(v) for v in topics.values()),
            "topics": {k: sorted(v) for k, v in topics.items()}}


# ---------------------------------------------------------------------------
# backups
# ---------------------------------------------------------------------------

def _members() -> list[tuple[str, Path]]:
    """(name in the backup, file) for everything that lives only on the volume"""
    ev = _ev()
    out = [(f"everyday/{n}", ev.built_dir() / n) for n in EVERYDAY_FILES]
    _scripts()
    import exam_build as eb
    bank = eb.bank_dir(config.EXAM_DIR)
    out += [(f"exam/bank/{p.name}", p) for p in sorted(bank.glob("*.jsonl"))] if bank.is_dir() \
        else []
    rub = config.BENCH_ROOT / "rubrics"
    out += [(f"rubrics/{p.name}", p) for p in sorted(rub.glob("*")) if p.is_file()] \
        if rub.is_dir() else []
    return [(n, p) for n, p in out if p.is_file()]


def backups() -> list[Path]:
    d = Path(config.BACKUP_DIR)
    return sorted(d.glob(PREFIX + "*.tar.gz")) if d.is_dir() else []


def last_backup() -> dict:
    """what Data & sources says: the newest backup, when, how many kept"""
    got = backups()
    if not got:
        return {"at": None, "file": "", "kept": 0, "dir": str(config.BACKUP_DIR)}
    return {"at": got[-1].stat().st_mtime, "file": got[-1].name, "kept": len(got),
            "dir": str(config.BACKUP_DIR)}


def backup(now: float | None = None) -> dict:
    """one .tar.gz in BACKUP_DIR, with MANIFEST.json (each file's sha256 and
    size); the oldest beyond BACKUP_KEEP removed"""
    d = Path(config.BACKUP_DIR)
    d.mkdir(parents=True, exist_ok=True)
    now = time.time() if now is None else now
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now)) + f"{int(now * 1000) % 1000:03d}Z"
    files, blobs = {}, []
    for name, p in _members():
        b = p.read_bytes()
        files[name] = {"sha256": _sha(b), "size": len(b)}
        blobs.append((name, b))
    manifest = {"created_at": now, "files": files, "hidden": _ev().hidden_status(),
                "exam": exam_status()}
    out = d / f"{PREFIX}{stamp}.tar.gz"
    tmp = out.with_suffix(".part")
    with tarfile.open(tmp, "w:gz") as t:
        for name, b in [("MANIFEST.json", json.dumps(manifest, indent=1).encode())] + blobs:
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(b), int(now)
            t.addfile(info, io.BytesIO(b))
    tmp.replace(out)
    for old in backups()[:-max(1, config.BACKUP_KEEP)]:
        old.unlink()
    return {"file": str(out), "files": len(files), "bytes": out.stat().st_size}


def export(src: Path, out: Path | None = None) -> Path:
    """the backup encrypted to the age public key: one file to copy off"""
    if not config.BACKUP_AGE_RECIPIENT:
        raise ValueError("no key to encrypt to: set EVALBOARD_BACKUP_AGE_RECIPIENT to an age public "
                         "key (age-keygen on your own machine; HANDOFF says how)")
    if config.BACKUP_AGE_RECIPIENT.upper().startswith("AGE-SECRET-KEY"):
        # never echoed: it is the key that opens every export
        raise ValueError("EVALBOARD_BACKUP_AGE_RECIPIENT holds a private key: take it off this "
                         "server and put only the public key (age1…) there")
    try:
        import pyrage
    except ImportError:
        raise ValueError("--export needs pyrage, which isn't in this image") from None
    rcpt = pyrage.x25519.Recipient.from_str(config.BACKUP_AGE_RECIPIENT)
    out = Path(out) if out else src.with_name(src.name + ".age")
    out.write_bytes(pyrage.encrypt(src.read_bytes(), [rcpt]))
    return out


def daily() -> dict | None:
    """the worker's, when idle: a backup once a day; a failure is said on Data
    & sources and tried again an hour later"""
    if not _lock.acquire(blocking=False):
        return None
    try:
        last = last_backup()["at"]
        if last and time.time() - last < DAY_S:
            return None
        if time.time() - _last_try["at"] < 3600:
            return None
        _last_try["at"] = time.time()
        try:
            got = backup()
            _last_try.update(error="")
            return got
        except OSError as e:
            _last_try.update(error=f"{config.BACKUP_DIR}: {e.strerror or e}")
            return None
    finally:
        _lock.release()


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------

def _read_backup(path: Path) -> tuple[dict, dict[str, bytes]]:
    """its manifest and files — each checked against its sha256"""
    with tarfile.open(path, "r:gz") as t:
        blobs = {m.name: t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
    manifest = json.loads(blobs.pop("MANIFEST.json"))
    for name, meta in manifest["files"].items():
        if name not in blobs or _sha(blobs[name]) != meta["sha256"]:
            raise ValueError(f"{path.name}: {name} doesn't match its sha256 — not restored")
    return manifest, blobs


def _target(name: str) -> Path:
    _scripts()
    import exam_build as eb
    if name.startswith("everyday/"):
        return _ev().built_dir() / name.split("/", 1)[1]
    if name.startswith("exam/bank/"):
        return eb.bank_dir(config.EXAM_DIR) / name.rsplit("/", 1)[1]
    return config.BENCH_ROOT / name


def restore(path: Path | None = None, everything: bool = False) -> dict:
    """Everyday's hidden set from the newest backup whose set is the one
    committed (or `path`), and each exam question the store is missing —
    added, never over one there now. `everything`: every file, the ones there
    now moved aside to BENCH_ROOT/restore-before-<time>"""
    ev = _ev()
    m = ev.hidden_manifest() or {}
    cands = [Path(path)] if path else list(reversed(backups()))
    if not cands:
        raise ValueError(f"no backup in {config.BACKUP_DIR}: copy an export back (HANDOFF)")
    chosen = None
    for c in cands:
        man, blobs = _read_backup(c)
        hid = blobs.get("everyday/" + ev.HIDDEN_NAME)
        rows = [json.loads(x) for x in (hid or b"").decode().splitlines() if x.strip()]
        if path or everything or (hid is not None and (not m or ev.hidden_digest(rows)
                                                       == m.get("digest"))):
            chosen = (c, blobs)
            break
    if chosen is None and ev.hidden_status()["ok"]:
        # the hidden set is fine where it is: the newest backup, for the exam's questions
        chosen = (cands[0], _read_backup(cands[0])[1])
        chosen[1].pop("everyday/" + ev.HIDDEN_NAME, None)
    if chosen is None:
        raise ValueError("no backup holds the hidden set that was committed: copy an export "
                         "back (HANDOFF)")
    src, blobs = chosen
    done = {"from": src.name, "files": [], "exam_rows": 0, "aside": ""}
    if everything:
        aside = config.BENCH_ROOT / time.strftime("restore-before-%Y%m%dT%H%M%SZ", time.gmtime())
        for name, b in sorted(blobs.items()):
            t = _target(name)
            if t.exists():
                (aside / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(t, aside / name)
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(b)
            done["files"].append(name)
        done["aside"] = str(aside) if aside.exists() else ""
        return {**done, "hidden": ev.hidden_status(), "exam": exam_status()}
    name = "everyday/" + ev.HIDDEN_NAME
    if name in blobs:
        t = _target(name)
        if t.exists() and t.read_bytes() != blobs[name]:
            t.replace(t.with_name(t.name + time.strftime(".before-restore-%Y%m%dT%H%M%SZ",
                                                         time.gmtime())))
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(blobs[name])
        done["files"].append(name)
    for name, b in sorted(blobs.items()):
        if not name.startswith("exam/bank/"):
            continue
        t = _target(name)
        have = {json.loads(x).get("qid") for x in t.read_text(encoding="utf-8").splitlines()
                if x.strip()} if t.exists() else set()
        add = [x for x in b.decode().splitlines() if x.strip() and json.loads(x).get("qid")
               not in have]
        if add:
            t.parent.mkdir(parents=True, exist_ok=True)
            with open(t, "a", encoding="utf-8") as fh:
                fh.write("".join(x + "\n" for x in add))
            done["exam_rows"] += len(add)
    return {**done, "hidden": ev.hidden_status(), "exam": exam_status()}


# ---------------------------------------------------------------------------
# the move, once
# ---------------------------------------------------------------------------

def move() -> dict:
    """the repo's hidden half into the store — the same rows, each marked
    hidden, so the bank's version (and every score) is unchanged — then a
    backup. Idempotent: a store already holding the committed set is left"""
    ev = _ev()
    p = ev.hidden_path()
    m = ev.hidden_manifest()
    if not m:
        raise ValueError(f"nothing committed to check the set against ({config.HIDDEN_MANIFEST})")
    before = ev.version()["hash"]
    if p.exists():
        st = ev.hidden_status()
        if not st["ok"]:
            raise ValueError("the store holds a hidden set that isn't the one committed: "
                             "restore it first (`restore`)")
        moved = False
    else:
        rows = ev.repo_hidden()
        if ev.hidden_digest(rows) != m["digest"]:
            raise ValueError("the repo's hidden half isn't the one committed — nothing moved "
                             "(is this checkout already past 12p.2?)")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        tmp.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in rows),
                       encoding="utf-8")
        tmp.replace(p)
        moved = True
    after = ev.version()["hash"]
    if after != before:
        p.unlink()
        raise ValueError(f"the bank's version would change ({before} → {after}): moved back")
    b = backup()
    return {"moved": moved, "count": m["count"], "version": after, "backup": b["file"],
            "exam": exam_status()}


# ---------------------------------------------------------------------------
# 12p.2: the CI guard's fingerprints — made here, where the questions are
# ---------------------------------------------------------------------------

FP_N, FP_STRIDE = 8, 4
REPO = Path(__file__).resolve().parent.parent


def _fp_runs(text: str, stride: int = FP_STRIDE) -> list[str]:
    from .contamination import normalize
    t = normalize(text)
    if len(t) < FP_N:
        return []
    every = [" ".join(t[i:i + FP_N]) for i in range(len(t) - FP_N + 1)]
    return every if stride == 1 or len(every) <= 3 else every[::stride] + [every[-1]]


def _fp(run: str) -> str:
    return hashlib.sha256(run.encode("utf-8")).hexdigest()[:16]


# 12p.4: a question at least this share of whose runs the repo holds is the
# question itself, copied — a leak, never a common phrase
WHOLE = 0.5


def fingerprints(repo: Path | None = None) -> dict:
    """every fourth eight-word run of each question that is the test —
    Everyday's hidden set (prompt and reference) and the exam's report half
    (prompt) — hashed. What tests/fixtures/protected_fingerprints.txt holds:
    after a new hidden set, write it here and commit it.

    A run the repo (`repo`) holds is a common phrase and left out — unless the
    repo holds the question itself (12p.4): WHOLE or more of its runs. Then
    every run of it is kept, so the guard fails on it, and it is named in
    `whole` (by id, and the files that hold it). Before 12p.4 such a question's
    runs were all "common": 242 of the exam's report half, whole in the repo,
    left the guard nothing to find."""
    import diagnose as dx
    import exam_build as eb
    ev = _ev()
    questions: list[tuple[str, list[str]]] = []
    for q in (ev._raw_rows(ev.hidden_path()) if ev.hidden_path().exists() else []):
        questions.append((q["id"], _fp_runs(q["prompt"]) + _fp_runs(q.get("reference") or "")))
    for r in eb.all_bank_rows(config.EXAM_DIR) if eb.bank_dir(config.EXAM_DIR).is_dir() else []:
        if r.get("prompt") and dx.split_of(eb.qid_of(r["prompt"])) == "report":
            questions.append((eb.qid_of(r["prompt"]), _fp_runs(r["prompt"])))
    keep = {_fp(x): x for _, runs in questions for x in runs}
    held: dict[str, set[str]] = {}
    root = Path(repo or REPO)
    skip = {".git", "results", "__pycache__", "_screens", "node_modules"}
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in skip]
        for name in files:
            f = Path(d) / name
            if f.suffix.lower() in (".png", ".jpg", ".gz", ".pdf", ".pyc") \
                    or f.name == "protected_fingerprints.txt":
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for h in {_fp(x) for x in _fp_runs(text, 1)} & keep.keys():
                held.setdefault(h, set()).add(str(f.relative_to(root)))
    common: set[str] = set()
    kept: set[str] = set()
    whole = []
    for qid, runs in questions:
        mine = {_fp(x) for x in runs}
        here = mine & held.keys()
        if mine and len(here) >= WHOLE * len(mine):
            whole.append({"id": qid, "files": sorted(set().union(*(held[h] for h in here)))})
            kept |= mine                   # every run of it, whatever else shares one
        else:
            common |= here
    common -= kept
    return {"fingerprints": sorted(set(keep) - common), "common": len(common), "whole": whole}


# ---------------------------------------------------------------------------
# what the page says
# ---------------------------------------------------------------------------

def alarms() -> list[dict]:
    """the red banner on every page: a set that is missing or changed"""
    out = []
    # 16.5: the exam's report half only while the Knowledge exam is switched on
    sets = [("hidden", _ev().hidden_status())]
    if config.KNOWLEDGE_EXAM:
        sets.append(("exam-report", exam_status()))
    for key, st in sets:
        if not st["ok"]:
            text, cmd = st["why"].rsplit(" · ", 1)
            out.append({"key": key, "text": text, "command": cmd})
    return out


def overview() -> dict:
    """Data & sources: where each set is, and the backups"""
    st = _ev().hidden_status()
    b = last_backup()
    stale = not b["at"] or time.time() - b["at"] > DAY_S + 2 * 3600
    return {"hidden": st, "exam": exam_status() if config.KNOWLEDGE_EXAM else None, "backup": b,
            "backup_error": _last_try.get("error", ""), "backup_stale": stale,
            "export_key": bool(config.BACKUP_AGE_RECIPIENT)}


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m service.hidden_store",
                                 description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("move")
    sub.add_parser("manifest")
    fpp = sub.add_parser("fingerprints")
    fpp.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("backup")
    b.add_argument("--export", action="store_true")
    b.add_argument("--out", type=Path)
    r = sub.add_parser("restore")
    r.add_argument("file", nargs="?", type=Path)
    r.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    ev = _ev()
    try:
        if a.cmd == "status":
            o = overview()
            h, e = o["hidden"], o["exam"]
            print(f"Everyday's hidden set: {h['state']}, {h['count']} questions"
                  + (f" — {h['why']}" if h["why"] else ""))
            print(f"The exam's report half: {e['state']}, {e['count']:,} questions"
                  + (f" — {e['why']}" if e["why"] else ""))
            bk = o["backup"]
            print(f"Backups: {bk['kept']} in {bk['dir']}" + (
                f", the newest {time.strftime('%Y-%m-%d %H:%M', time.gmtime(bk['at']))} UTC"
                if bk["at"] else ", none yet"))
            return 0 if h["ok"] and e["ok"] else 1
        if a.cmd == "manifest":
            print(json.dumps(ev.manifest_of(
                ev._raw_rows(ev.hidden_path()) if ev.hidden_path().exists() else ev.repo_hidden()),
                indent=1))
            return 0
        if a.cmd == "fingerprints":
            got = fingerprints()
            a.out.write_text("# 12p.2: eight-word runs of the questions that are the test, hashed "
                             "(tests/test_12p2_no_hidden_in_repo.py)\n"
                             + "\n".join(got["fingerprints"]) + "\n", encoding="utf-8")
            print(f"{len(got['fingerprints'])} fingerprints to {a.out} ({got['common']} common "
                  "phrases left out); commit it as tests/fixtures/protected_fingerprints.txt")
            if got["whole"]:
                # 12p.4: the repo holds questions that are the test — named by id and
                # file, never by their words; the guard fails on them until they go
                files = sorted({f for w in got["whole"] for f in w["files"]})
                print(f"{len(got['whole'])} question(s) that are the test are in the repo whole, "
                      f"in {len(files)} file(s): {', '.join(files)}. Their ids: "
                      + ", ".join(w["id"][:12] for w in got["whole"]))
                return 1
            return 0
        if a.cmd == "move":
            got = move()
            print(("moved" if got["moved"] else "already in the store:")
                  + f" {got['count']} hidden questions to {ev.hidden_path()}, the bank's version "
                  f"{got['version']} unchanged; backed up to {got['backup']}. The exam's report "
                  f"half: {got['exam']['state']}, {got['exam']['count']:,} questions"
                  + (f", {got['exam']['missing']:,} missing" if got["exam"]["missing"] else ""))
            return 0
        if a.cmd == "backup":
            got = backup()
            print(f"backed up {got['files']} files to {got['file']} ({got['bytes']:,} bytes)")
            if a.export:
                out = export(Path(got["file"]), a.out)
                print(f"exported, encrypted to your age key: {out} — copy it off the server")
            return 0
        got = restore(a.file, everything=a.all)
        print(f"restored from {got['from']}: " + (", ".join(got["files"]) or "no file")
              + (f", {got['exam_rows']:,} exam questions added" if got["exam_rows"] else "")
              + (f"; what was there is in {got['aside']}" if got["aside"] else "")
              + f". Everyday's hidden set: {got['hidden']['state']}; the exam's report half: "
              f"{got['exam']['state']}.")
        return 0 if got["hidden"]["ok"] and got["exam"]["ok"] else 1
    except (ValueError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
