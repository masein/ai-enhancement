#!/usr/bin/env python3
"""20: a box's finished bundles and its progress files, uploaded to our
private Hugging Face repository — the way home while the board is out of
reach (the fetch, scripts/frontier_fetch.py, is the board's).

    read -rs HF_TOKEN && export HF_TOKEN
    python scripts/box_upload.py --to hf://<you>/evalboard-private/runs/step4000-2610

Every build folder under --root (/workspace/<build>/, as frontier_box.py
makes them), each step's folder in it (<box>-<step>): its progress file, and
the bundle the progress file names when the file there has the sha256 it
records, to <path>/<build>/<step>/; the box's line file (<box>.line.json)
to <path>/<build>/. One commit a run of this line, then each file checked on
Hugging Face against its sha256 here. Run it again when more is whole: the
progress files are replaced, a bundle already there is the same file.

It refuses a repository that isn't private: bundles go nowhere else. The
token is the one typed (HF_TOKEN, needing write on that repository), never
printed or stored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import remote_bundle as rb  # noqa: E402

_HF = re.compile(r"^hf://(?P<type>datasets/|models/)?(?P<repo>[^/@\s]+/[^/@\s]+)"
                 r"/(?P<path>[A-Za-z0-9._/-]+)$")
_STEP = re.compile(r"^[A-Z]{1,2}\d{1,3}-(\d{1,2}|parity)$")
_LINE = re.compile(r"^[A-Z]{1,2}\d{1,3}\.line\.json$")


def blob_sha1(data: bytes) -> str:
    """the git blob id Hugging Face gives a file kept in git, not LFS"""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def collect(root: Path, builds: list[str], say) -> list[tuple[Path, str, str]]:
    """(the file here, its path in the repository under --to, its sha256) for
    every file to go: each step's progress file and its finished bundle, and
    each box's line file"""
    out = []
    folders = [root / b for b in builds] if builds else sorted(
        d for d in root.iterdir() if d.is_dir() and d.name not in ("files",))
    for build in folders:
        if not build.is_dir():
            raise SystemExit(f"{build}: no such build folder")
        for f in sorted(build.iterdir()):
            if f.is_file() and _LINE.match(f.name):
                out.append((f, f"{build.name}/{f.name}", rb.sha256_file(f)))
        for step in sorted(d for d in build.iterdir() if d.is_dir() and _STEP.match(d.name)):
            prog = step / "progress.json"
            if not prog.is_file():
                say(f"{build.name}/{step.name}: no progress file yet — nothing of it goes")
                continue
            try:
                st = json.loads(prog.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                st = {}
            out.append((prog, f"{build.name}/{step.name}/progress.json", rb.sha256_file(prog)))
            b = (st.get("bundle") or {}) if isinstance(st, dict) else {}
            name, want = str(b.get("name") or ""), str(b.get("sha256") or "")
            if not name:
                say(f"{build.name}/{step.name}: {st.get('state') or 'no state'} — no bundle yet, "
                    "its progress file goes")
                continue
            p = step / name
            if "/" in name or not p.is_file():
                say(f"{build.name}/{step.name}: its bundle {name} isn't there — not uploaded")
                continue
            got = rb.sha256_file(p)
            if got != want:
                say(f"{build.name}/{step.name}: {name} isn't the bundle its progress file "
                    f"records ({got[:16]}, not {want[:16]}) — not uploaded; paste the box's line "
                    "again to make it")
                continue
            out.append((p, f"{build.name}/{step.name}/{name}", got))
            if st.get("state") != "whole":
                say(f"{build.name}/{step.name}: not whole ({st.get('why') or st.get('state')}) — "
                    f"its bundle of what is whole goes")
    return out


def main(argv: list[str] | None = None, api=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--to", required=True,
                    help="hf://<org>/<repo>/<path>: the private repository, and the folder in it")
    ap.add_argument("--root", default="/workspace", help="where the build folders are")
    ap.add_argument("--build", action="append", default=[],
                    help="one build's folder (repeatable); every one under --root when none")
    a = ap.parse_args(argv)
    m = _HF.match(a.to.strip())
    if not m or ".." in m["path"].split("/"):
        ap.error("--to: hf://<org>/<repo>/<path>, the folder in the repository")
    repo, kind = m["repo"], "dataset" if m["type"] == "datasets/" else "model"
    base = m["path"].strip("/")
    files = collect(Path(a.root), a.build, print)
    if not files:
        print("nothing to upload")
        return 1
    token = os.environ.get("HF_TOKEN") or None
    if not token:
        raise SystemExit("HF_TOKEN isn't set: read -rs HF_TOKEN && export HF_TOKEN (a token "
                         "that can write to the repository), then this line again")
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    try:
        info = api.repo_info(repo, repo_type=kind)
    except Exception as e:                                  # noqa: BLE001 — said in one line
        raise SystemExit(rb.scrub(f"{repo}: can't be read with this token ({e})")) from None
    if not getattr(info, "private", False):
        raise SystemExit(f"{repo} isn't private: bundles go only to a private repository. "
                         "Nothing was uploaded")
    from huggingface_hub import CommitOperationAdd
    ops = [CommitOperationAdd(path_in_repo=f"{base}/{rel}", path_or_fileobj=str(p))
           for p, rel, _ in files]
    boxes = sorted({rel.split("/")[1].split("-")[0].split(".")[0] for _, rel, _ in files
                    if "/" in rel})
    try:
        api.create_commit(repo, operations=ops, repo_type=kind,
                          commit_message=f"{', '.join(boxes) or 'a box'}: bundles and progress")
    except Exception as e:                                  # noqa: BLE001 — said in one line
        raise SystemExit(rb.scrub(f"the upload to {repo} failed: {e} — a token that can write "
                                  "to it? Nothing is lost here: run the line again")) from None
    # each file checked where it landed
    there = {}
    for f in api.list_repo_tree(repo, path_in_repo=base, repo_type=kind, recursive=True,
                                expand=True):
        there[getattr(f, "path", "")] = f
    bad = 0
    for p, rel, sha in files:
        f = there.get(f"{base}/{rel}")
        lfs = getattr(f, "lfs", None)
        lfs_sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                   ) if lfs else None
        ok = f is not None and (lfs_sha == sha if lfs_sha else
                                getattr(f, "blob_id", None) == blob_sha1(p.read_bytes()))
        bad += not ok
        print(f"{'uploaded' if ok else 'NOT as here'}: {base}/{rel} · sha256 {sha[:16]}"
              + ("" if ok else " — run the line again"))
    print(f"{len(files) - bad} of {len(files)} files in {repo} as they are here")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
