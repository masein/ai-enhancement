#!/usr/bin/env python3
"""17b: the parity check, the pilot's first step — does a rented box answer as
the server does? The same 50 MMLU-Pro questions (scripts/frontier.py PARITY),
thinking off and greedy, asked of the served model on the server (`ask`, in
the board's container) and of the same GGUF on the box, with the box's own
flags (`remote_gguf.py --parity`); then `compare` reads both.

What is compared: the letter each side reads from each reply (TIGER-Lab's
extraction), and whether the replies are identical. What counts as the same:
the same letter on at least 46 of the 50, a question counting only when both
sides read a letter from it. Greedy decoding on two machines with other batch
sizes, KV cache types and kernels isn't bit-for-bit the same, so whole
replies may part, and are counted, not required.

17c: each side's file opens with what answered — the side, the served model,
the file (name, size, sha256) and the launch (its routing environment and
speculative decoding). `ask` asks the server only while it serves the file
registered (served.check_pin). `compare` refuses:
- the same file twice, a file holding a question twice, or a file without
  that first line (made before 17c: ask again);
- two sides that answered as other models, from other files, or with
  another routing or speculative setup;
- a box whose launch isn't the one registered for the model.
The board may have no sha256 of the server's file: the two are then compared
by name and size, and saying so; --file-sha256 (sha256sum on the server)
compares them whole.

    sudo docker compose exec -T bench python scripts/frontier_parity.py ask \\
        --as served/<name> --out /home/masein/benchmarks/parity/<name>-server.jsonl
    sudo docker compose exec -T bench python scripts/frontier_parity.py compare \\
        /home/masein/benchmarks/parity/<name>-server.jsonl /home/masein/benchmarks/parity/<name>-box.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import frontier as fb  # noqa: E402
import import_frontier as imf  # noqa: E402


def read(path: Path) -> tuple[dict, list[dict]]:
    """(what answered, its answers) — SystemExit for a file without the first
    line that says what answered"""
    lines = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()
             if x.strip()]
    if not lines or not isinstance(lines[0], dict) or not isinstance(lines[0].get("parity_of"),
                                                                     dict):
        raise SystemExit(f"{path}: no first line saying what answered — a parity file from "
                         "before 17c: ask again")
    return lines[0]["parity_of"], lines[1:]


def _launch_words(env: dict, spec: list[str]) -> str:
    return " ".join([*(f"{k}={v}" for k, v in sorted(env.items())), *spec]) or "none"


def identity_problems(server: dict, box: dict, file_sha: str = "") -> tuple[list[str], list[str]]:
    """(each way the two sides aren't the same setup, notes) — the box's launch
    against the one registered, which the server's file carries"""
    out, notes = [], []
    if server.get("side") != "server" or box.get("side") != "box":
        out.append(f"the first file is {server.get('side') or 'no'} side's and the second "
                   f"{box.get('side') or 'no'} side's: give the server's, then the box's")
    if server.get("as") != box.get("as"):
        out.append(f"the server answered as {server.get('as')}, the box as {box.get('as')}")
    fs, fx = server.get("file") or {}, box.get("file") or {}
    if fs.get("name") != fx.get("name"):
        out.append(f"the server serves {fs.get('name')}, the box ran {fx.get('name')}")
    if fs.get("size") and fx.get("size") and int(fs["size"]) != int(fx["size"]):
        out.append(f"the server's file is {int(fs['size']):,} bytes, the box's "
                   f"{int(fx['size']):,}")
    if file_sha and fs.get("sha256") and file_sha != fs["sha256"]:
        out.append(f"--file-sha256 {file_sha[:16]}… isn't the sha256 the board has for "
                   f"{server.get('as')}, {fs['sha256'][:16]}…")
    want = fs.get("sha256") or file_sha
    if want and fx.get("sha256") != want:
        out.append(f"the box's file has sha256 {str(fx.get('sha256'))[:16]}…, the server's "
                   f"{want[:16]}…")
    if not want:
        notes.append("The board has no sha256 of the server's file: the two were compared by "
                     "name and size. Give --file-sha256 (sha256sum on the server) to compare "
                     "them whole.")
    reg = server.get("launch") or {}
    got = imf.box_launch(box.get("server") or {})
    if dict(reg.get("env") or {}) != got["env"]:
        out.append(f"routing: the box ran with {_launch_words(got['env'], [])}; "
                   f"{server.get('as')} is registered with "
                   f"{_launch_words(dict(reg.get('env') or {}), [])}")
    if sorted(reg.get("spec") or []) != got["spec"] or (reg.get("drafts") and not got["spec"]):
        out.append(f"speculative decoding: the box ran with {_launch_words({}, got['spec'])}; "
                   f"{server.get('as')} is registered with "
                   + (_launch_words({}, sorted(reg.get('spec') or [])) if reg.get("spec")
                      else "drafting tokens" if reg.get("drafts") else "none"))
    sp, bp = server.get("speculative"), box.get("speculative")
    if sp is not None and bp is not None and bool(sp) != bool(bp):
        out.append(f"the server's slots draft tokens: {'yes' if sp else 'no'}; the box's: "
                   f"{'yes' if bp else 'no'}")
    return out, notes


def server_identity(rec: dict) -> dict:
    """what answers on the server: its file as registered — SystemExit while
    its server serves another (served.check_pin) — and the launch registered"""
    from service import served
    why = served.check_pin(rec)
    if why:
        raise SystemExit(f"{rec['id']}: {why} The parity check asks the server only as "
                         "registered.")
    p = served.probe(rec["base_url"], rec.get("key", ""))
    pin = rec.get("pin") or {}
    return {"side": "server", "as": rec["id"],
            "file": {"name": pin.get("file") or p.get("file"), "size": pin.get("size")
                     or p.get("size"), "sha256": imf.registered_sha(rec)},
            "launch": imf.record_launch(rec), "speculative": p.get("speculative"),
            "build": p.get("build") or pin.get("build")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("ask", help="ask the served model on this server")
    a1.add_argument("--as", dest="served_as", required=True, help="served/<name>")
    a1.add_argument("--out", required=True, type=Path)
    a2 = sub.add_parser("compare", help="the server's answers beside the box's")
    a2.add_argument("server", type=Path)
    a2.add_argument("box", type=Path)
    a2.add_argument("--file-sha256", default="", help="the server's file's sha256, when the "
                                                      "board has none (sha256sum on the server)")
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        if a.server.resolve() == a.box.resolve() or a.server.read_bytes() == a.box.read_bytes():
            raise SystemExit("Not the same: the same file was given twice — the server's, "
                             "then the box's")
        (sh, srows), (bh, brows) = read(a.server), read(a.box)
        problems, notes = identity_problems(sh, bh, a.file_sha256.strip().lower())
        if problems:
            print("Not the same setup: " + "; ".join(problems))
            return 1
        got = fb.parity_compare(srows, brows)
        print(got["words"])
        for n in notes:
            print(n)
        return 0 if got["ok"] else 1
    from service import db, served
    from service import frontier as sf
    db.init()
    rec = served.get(a.served_as)
    if not rec or served.is_openrouter(rec):
        raise SystemExit(f"{a.served_as} isn't a model served by a llama-server here")
    ident = server_identity(rec)
    n = sf.parity_ask(rec, a.out, lambda k, of: print(f"{k} of {of}", end="\r", flush=True),
                      identity=ident)
    print(f"\n{n} answers · {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
