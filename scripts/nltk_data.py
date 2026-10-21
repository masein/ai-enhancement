#!/usr/bin/env python3
"""12q.D: NLTK's punkt_tab, which IFEval's checkers split sentences with —
fetched once, when the image is built, never by a run. Five DeviceMark runs
on hf failed at "Resource 'punkt_tab' not found": the checker asks NLTK to
download it the first time a sentence is counted, and a run has no business
reaching the network for it.

Pinned as the image's other downloads are: the nltk_data commit that last
changed the file, its size and its git blob hash (what that commit holds,
byte for byte). Unpacked into /usr/share/nltk_data, on NLTK's own search
path, so nothing needs NLTK_DATA set. 25: with English stopwords and the
English part-of-speech tagger beside it, pinned the same way — IFBench's
checkers count the one and tag with the other, and nothing is fetched when
they check a reply.

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
# 25: what IFBench's checkers also need: each zip at the last nltk_data commit
# to change it, its size and git blob hash, the folder it unpacks into and
# what it must hold there
MORE = {
    # "Add Uzbek stopwords, resolves #255" (2026-01-09)
    "stopwords": {"commit": "98e14261eaec2244af1db4ec5b282112444495aa",
                  "blob": "56d35b5d204d64480f71edc0f5f82aebce6730b6", "size": 37_733,
                  "folder": "corpora", "holds": "stopwords/english"},
    # "fixed positions of classes" (2024-07-05)
    "averaged_perceptron_tagger_eng": {
        "commit": "304a3c5432dd2fe08ae8b436dc6eec54c3593445",
        "blob": "b792e19546a5711218a05f4029ef17c8d62cd839", "size": 1_539_115,
        "folder": "taggers", "holds": "averaged_perceptron_tagger_eng"},
}


def url_of(name: str) -> str:
    p = MORE[name]
    return (f"https://raw.githubusercontent.com/nltk/nltk_data/{p['commit']}/packages/"
            f"{p['folder']}/{name}.zip")


def blob_hash(data: bytes) -> str:
    """git's hash of a file's bytes: what a commit's tree names it by"""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check(data: bytes, name: str = "punkt_tab", size: int | None = None,
          blob: str | None = None) -> None:
    size, blob = SIZE if size is None else size, BLOB if blob is None else blob
    if len(data) != size or blob_hash(data) != blob:
        raise SystemExit(f"{name}.zip is not the pinned one: {len(data):,} bytes, git blob "
                         f"{blob_hash(data)} (want {size:,} bytes, {blob})")


def _unpack(data: bytes, url: str, out: Path, holds: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(out)
    if not (out / holds).exists():
        raise SystemExit(f"{url} held no {holds}")


def fetch(dest: Path = DEST, url: str = URL) -> Path:
    """the pinned punkt_tab, unpacked into dest/tokenizers/punkt_tab"""
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    check(data)
    _unpack(data, url, dest / "tokenizers", "punkt_tab/english")
    return dest / "tokenizers" / "punkt_tab"


def fetch_more(name: str, dest: Path = DEST) -> Path:
    """25: one of MORE, pinned as punkt_tab is, unpacked into dest/<its folder>"""
    p, url = MORE[name], url_of(name)
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    check(data, name, p["size"], p["blob"])
    _unpack(data, url, dest / p["folder"], p["holds"])
    return dest / p["folder"] / name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEST)
    a = ap.parse_args(argv)
    got = fetch(a.dest)
    print(f"punkt_tab at {got} (nltk_data {COMMIT[:12]}, {SIZE:,} bytes)")
    for name, p in MORE.items():
        got = fetch_more(name, a.dest)
        print(f"{name} at {got} (nltk_data {p['commit'][:12]}, {p['size']:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
