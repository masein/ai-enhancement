#!/usr/bin/env python3
"""12q.D: NLTK's punkt_tab, which IFEval's checkers split sentences with —
fetched once, when the image is built, never by a run. Five DeviceMark runs
on hf failed at "Resource 'punkt_tab' not found": the checker asks NLTK to
download it the first time a sentence is counted, and a run has no business
reaching the network for it.

Pinned as the image's other downloads are: the nltk_data commit that last
changed the file, its size and its git blob hash (what that commit holds,
byte for byte). Unpacked into /usr/share/nltk_data, on NLTK's own search
path, so nothing needs NLTK_DATA set.

    python scripts/nltk_data.py                       into /usr/share/nltk_data
    python scripts/nltk_data.py --dest DIR            elsewhere (CI: NLTK_DATA=DIR)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

# nltk/nltk_data, gh-pages: "Add Malayalam to punkt_tab" (2025-02-17), the
# last commit to change packages/tokenizers/punkt_tab.zip
COMMIT = "4f15a3d89eefe9748ec1c05be495d91289197155"
BLOB = "5e5ff6137d5ee6025e400d1c3a7b21914c48b635"
SIZE = 4_319_076
URL = f"https://raw.githubusercontent.com/nltk/nltk_data/{COMMIT}/packages/tokenizers/punkt_tab.zip"
DEST = Path("/usr/share/nltk_data")


def blob_hash(data: bytes) -> str:
    """git's hash of a file's bytes: what a commit's tree names it by"""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check(data: bytes) -> None:
    if len(data) != SIZE or blob_hash(data) != BLOB:
        raise SystemExit(f"punkt_tab.zip is not the pinned one: {len(data):,} bytes, git blob "
                         f"{blob_hash(data)} (want {SIZE:,} bytes, {BLOB})")


def fetch(dest: Path = DEST, url: str = URL) -> Path:
    """the pinned punkt_tab, unpacked into dest/tokenizers/punkt_tab"""
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    check(data)
    out = dest / "tokenizers"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(out)
    if not (out / "punkt_tab" / "english").is_dir():
        raise SystemExit(f"{url} held no punkt_tab/english")
    return out / "punkt_tab"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEST)
    a = ap.parse_args(argv)
    got = fetch(a.dest)
    print(f"punkt_tab at {got} (nltk_data {COMMIT[:12]}, {SIZE:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
