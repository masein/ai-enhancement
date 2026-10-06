#!/usr/bin/env python3
"""17b: the parity check, the pilot's first step — does a rented box answer as
the server does? The same 50 MMLU-Pro questions (scripts/frontier.py PARITY),
thinking off and greedy, asked of the served model on the server (`ask`, in
the board's container) and of the same GGUF on the box, with the box's own
flags (`remote_gguf.py --parity`); then `compare` reads both.

17d: the same 500 questions by default (`--n` on both sides). "The same" is
decided on accuracy: the difference in right answers on the same questions
(the box's share right minus the server's, question by question), its 90%
paired interval inside ±5 points (PARITY's margin, stated before the run: the
two one-sided tests of equivalence). Letter agreement and identical replies
are information only, beside the box's agreement with a second run of itself
(the box asks each question twice): on the pilot, two runs on one box agreed
on 43 letters of 50, so letters measured run-to-run noise, not the setup.
`ask` says how long its questions take at this server's measured pace with
thinking off, and how long they took.

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
compares them whole. 17d: a size only with its like — llama-server's count of
the weights on both sides, never the server's count against the box's file
on disk.

    sudo docker compose exec -T bench python scripts/frontier_parity.py ask \\
        --as served/<name> --out /home/masein/benchmarks/parity/<name>-server.jsonl
    sudo docker compose exec -T bench python scripts/frontier_parity.py compare \\
        /home/masein/benchmarks/parity/<name>-server.jsonl /home/masein/benchmarks/parity/<name>-box.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    # 17d: sizes only like with like — llama-server's count of the weights on
    # both sides; never the server's count against the box's file on disk
    # (about 11 MB apart for the header: every real pair was refused)
    if isinstance(fs.get("weights"), int) and isinstance(fx.get("weights"), int) \
            and fs["weights"] != fx["weights"]:
        out.append(f"llama-server counts the server's weights as {fs['weights']:,} bytes and "
                   f"the box's as {fx['weights']:,}")
    if file_sha and fs.get("sha256") and file_sha != fs["sha256"]:
        out.append(f"--file-sha256 {file_sha[:16]}… isn't the sha256 the board has for "
                   f"{server.get('as')}, {fs['sha256'][:16]}…")
    want = fs.get("sha256") or file_sha
    if want and fx.get("sha256") != want:
        out.append(f"the box's file has sha256 {str(fx.get('sha256'))[:16]}…, the server's "
                   f"{want[:16]}…")
    if not want:
        notes.append("The board has no sha256 of the server's file: the two were compared by "
                     "name, and by llama-server's count of their weights where both said it. "
                     "Give --file-sha256 (sha256sum on the server) to compare them whole.")
    reg = server.get("launch") or {}
    if reg.get("words"):
        out.append(f"{server.get('as')}'s record says lookahead in words, with no routing "
                   "variable to compare the box with: give its launch's environment on its page")
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
            # 17d: llama-server's count of the weights, said as such — never the
            # file's size on disk, which this side doesn't know
            "file": {"name": pin.get("file") or p.get("file"), "weights": pin.get("size")
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
    a1.add_argument("--n", type=int, default=fb.PARITY["n"],
                    help="17d: the number of questions (the box's --parity --n the same)")
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
        if sh.get("n") != bh.get("n"):
            problems.append(f"the server was asked {sh.get('n')} questions and the box "
                            f"{bh.get('n')}: ask both with the same --n")
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
    # 17d: how long this takes, at the pace the board measured with thinking off
    pace = ((rec.get("speed_by") or {}).get("off") or {}).get("secs_each")
    print(f"{a.n} questions, thinking off: "
          + (f"about {round(a.n * pace / 60)} min at {pace:.1f} s an answer (this server's "
             "pace with thinking off)" if pace else
             "this server has no pace with thinking off on record yet"), flush=True)
    t0 = time.time()
    n = sf.parity_ask(rec, a.out, lambda k, of: print(f"{k} of {of}", end="\r", flush=True),
                      identity=ident, n=a.n)
    took = time.time() - t0
    print(f"\n{n} answers in {took / 60:.0f} min ({took / max(1, n):.1f} s an answer) · {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
