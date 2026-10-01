"""14.2: the one data step at deploy — what the board asks from but must not
commit to its public mirror, fetched from its pinned source into the server's
data folder and checked against its pinned sha256 and size:

- MobileAIBench's Privacy Leakage (real people's names, from the Enron
  corpus), from the commit eval_tasks/mobileaibench/manifest.json pins, into
  MAB_PRIVATE_DIR;
- 14.3: Mobile-MMLU-Pro (CC BY-ND 4.0: used here, never published, and our
  key is a derivative), from the Hugging Face revision
  eval_tasks/mobile_mmlu_pro/manifest.json pins, into MMP_DIR.

A file already here with the pinned hash is left alone; one that isn't, or a
download that doesn't match, is never kept (a .part is removed) and the step
says so and fails. Nothing is fetched anywhere else, ever: runs read only
what this put here.

    python scripts/fetch_data.py            fetch what is missing or changed
    python scripts/fetch_data.py --check    say what is here; fetch nothing
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CHUNK = 1 << 20


def wanted() -> list[dict]:
    """every pinned file that lives only on the server: {name, url, sha256,
    bytes, dest}"""
    sys.path.insert(0, str(HERE))
    import mobileaibench as mab
    import mobile_mmlu as mmp
    m = mab.manifest()
    f = mmp.manifest()["file"]
    return [{"name": f"MobileAIBench {k}", "url": f["url"], "sha256": f["sha256"],
             "bytes": f["bytes"], "dest": mab.private_dir() / f["file"]}
            for k, f in m["files"].items() if not f.get("committed", True)] + [
        {"name": "Mobile-MMLU-Pro", "url": f["url"], "sha256": f["sha256"], "bytes": f["bytes"],
         "dest": mmp.data_dir() / f["file"]}]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def here(item: dict) -> bool:
    p = item["dest"]
    return p.exists() and p.stat().st_size == item["bytes"] and _sha(p) == item["sha256"]


def fetch(item: dict, opener=urllib.request.urlopen) -> str:
    """one file: fetched into a .part, checked, then put in place — or why not"""
    if here(item):
        return f"{item['name']}: here, as pinned ({item['sha256'][:12]})"
    dest = item["dest"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        req = urllib.request.Request(item["url"], headers={"User-Agent": "evalboard"})
        with opener(req, timeout=120) as r, open(part, "wb") as out:
            for b in iter(lambda: r.read(CHUNK), b""):
                out.write(b)
        size, sha = part.stat().st_size, _sha(part)
        if size != item["bytes"] or sha != item["sha256"]:
            raise ValueError(f"got {size:,} bytes, sha256 {sha[:12]}; pinned {item['bytes']:,} "
                             f"bytes, {item['sha256'][:12]}")
        part.replace(dest)
    except Exception as e:                          # noqa: BLE001 — said, and nothing kept
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{item['name']}: not fetched — {e}") from None
    return f"{item['name']}: fetched to {dest} ({item['bytes']:,} bytes, {item['sha256'][:12]})"


def main(argv: list[str]) -> int:
    check = "--check" in argv
    bad = 0
    for item in wanted():
        if check:
            ok = here(item)
            bad += not ok
            print(f"{item['name']}: " + ("here, as pinned" if ok else "missing — run "
                                                                        "scripts/fetch_data.py"))
            continue
        try:
            print(fetch(item))
        except RuntimeError as e:
            bad += 1
            print(e)
    print("data OK" if not bad else f"data: {bad} file(s) not as pinned")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    raise SystemExit(main(sys.argv[1:]))
