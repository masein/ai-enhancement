#!/usr/bin/env python3
"""Upload the report's data folder to its Hugging Face dataset, privately.

    python3 upload_report_data.py --status      # what the repository holds now (read-only)
    python3 upload_report_data.py --dry-run     # what an upload would send (no token, no network)
    python3 upload_report_data.py               # the upload, after the same checks and a yes

The token comes from HF_TOKEN (read -rs HF_TOKEN && export HF_TOKEN), never from the
command line. The upload creates the repository as private if it doesn't exist, and stops
if it exists and is public: making it public is a step taken on huggingface.co, by hand.
It replaces the repository's contents with the folder in one commit (files the folder no
longer has are deleted; .gitattributes stays) and prints the final file list.

Before anything is sent: no file may hold a TBD (--trial lets them through, for a private
trial only), and the scan of internal terms (scan_terms.py, with --private and --allow)
must come back clean.

    --folder   the folder (default: docs/whitepaper-data/publish/masein-model-report-data)
    --repo     the dataset (default: Teraformer-limited/masein-model-report-data)
    --private  the scan's private terms (default: publish-src/scan-private.json beside it)
    --allow    the scan's allowed hits (default: publish-src/scan-allow.json beside it)"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scan_terms  # noqa: E402

REPO = "Teraformer-limited/masein-model-report-data"
# the folder beside this script's copy in publish-src/, or the repository's docs/whitepaper-data/
FOLDER = next((b / "publish" / "masein-model-report-data"
               for b in (HERE.parent, HERE.parent.parent / "docs" / "whitepaper-data")
               if (b / "publish" / "masein-model-report-data").is_dir()),
              HERE.parent.parent / "docs" / "whitepaper-data" / "publish" / "masein-model-report-data")
KEEP = {".gitattributes"}


def opt(args: list[str], name: str, default):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return default


def local_files(folder: Path) -> dict[str, int]:
    return {str(p.relative_to(folder)): p.stat().st_size for p in sorted(folder.rglob("*"))
            if p.is_file() and not any(x.startswith(".") for x in p.relative_to(folder).parts)}


def size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.1f} kB"


def checks(folder: Path, private: Path, allow: Path, trial: bool) -> bool:
    files = local_files(folder)
    tbd = [f for f in files if "TBD" in (folder / f).read_text(encoding="utf-8", errors="replace")]
    for f in tbd:
        print(f"  TBD in {f}")
    ok = True
    if tbd and not trial:
        print(f"stopped: {len(tbd)} files hold a TBD (fill them; --trial is for a private trial only)")
        ok = False
    args = [str(folder)] + (["--private", str(private)] if private.exists() else []) + \
           (["--allow", str(allow)] if allow.exists() else [])
    if not private.exists():
        print(f"stopped: the private terms ({private}) aren't there, so the scan would miss them")
        return False
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        bad = scan_terms.main(args)
    print("  scan: " + out.getvalue().strip().splitlines()[-1])
    if bad:
        print(out.getvalue())
        print("stopped: the scan found internal terms in our own text")
        ok = False
    logs = [f for f in files if f.startswith("logs/")]
    if not logs:
        print("  note: logs/ is empty")
    return ok


def plan(files: dict[str, int], remote: dict[str, int] | None) -> list[str]:
    print(f"to upload: {len(files)} files, {size(sum(files.values()))}")
    for f, n in files.items():
        was = "" if remote is None else ("new" if f not in remote else
                                         "same size" if remote[f] == n else "replaces")
        print(f"  {size(n):>9}  {f}" + (f"  ({was})" if was else ""))
    gone = sorted(set(remote or {}) - set(files) - KEEP)
    if gone:
        print(f"to delete from the repository: {len(gone)}")
        for f in gone:
            print(f"  {f}")
    return gone


def main(argv: list[str]) -> int:
    args = list(argv)
    folder = Path(opt(args, "--folder", FOLDER)).resolve()
    repo = opt(args, "--repo", REPO)
    src = folder.parent.parent / "publish-src"
    private = Path(opt(args, "--private", src / "scan-private.json"))
    allow = Path(opt(args, "--allow", src / "scan-allow.json"))
    trial = "--trial" in args
    files = local_files(folder)
    if "--dry-run" in args:
        print(f"{folder} → {repo} (dry run: nothing sent, nothing read from the Hub)")
        ok = checks(folder, private, allow, trial)
        plan(files, None)
        return 0 if ok else 1
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN isn't set: read -rs HF_TOKEN && export HF_TOKEN, then run this again")
        return 2
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        from huggingface_hub.utils import RepositoryNotFoundError
    api = HfApi(token=token)

    def remote_state():
        try:
            info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
        except RepositoryNotFoundError:
            return None, None
        return info, {s.rfilename: s.size or 0 for s in info.siblings}
    info, remote = remote_state()
    if "--status" in args:
        if info is None:
            print(f"{repo}: doesn't exist, or this token can't see it")
            return 0
        print(f"{repo}: {'private' if info.private else 'PUBLIC'}, last commit {info.sha}, "
              f"{len(remote)} files")
        for f, n in sorted(remote.items()):
            print(f"  {size(n):>9}  {f}")
        return 0
    print(f"{folder} → {repo}")
    if not checks(folder, private, allow, trial):
        return 1
    if info is not None and not info.private:
        print(f"stopped: {repo} is public. This script uploads to a private repository only")
        return 1
    gone = plan(files, remote)
    if input(f"{'create the private repository and ' if info is None else ''}"
             f"upload these files (type upload): ").strip() != "upload":
        print("nothing sent")
        return 1
    if info is None:
        api.create_repo(repo, repo_type="dataset", private=True)
        print(f"created {repo}, private")
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(folder / f)) for f in files]
    ops += [CommitOperationDelete(path_in_repo=f) for f in gone]
    c = api.create_commit(repo, operations=ops, repo_type="dataset",
                          commit_message="Report data: replace with the current folder")
    print(f"committed {c.oid}")
    info, remote = remote_state()
    print(f"{repo}: {'private' if info.private else 'PUBLIC'}, {len(remote)} files")
    for f, n in sorted(remote.items()):
        mark = "" if f in KEEP or files.get(f) == n else "   ← NOT THE LOCAL SIZE"
        print(f"  {size(n):>9}  {f}{mark}")
    missing = sorted(set(files) - set(remote))
    if missing:
        print("NOT ON THE HUB: " + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
