#!/usr/bin/env python3
"""17b: the parity check, the pilot's first step — does a rented box answer as
the server does? The same 50 MMLU-Pro questions (scripts/frontier.py PARITY),
thinking off and greedy, asked of the served model on the server (`ask`, in
the board's container) and of the same GGUF on the box, with the box's own
flags (`remote_gguf.py --parity`); then `compare` reads both.

What is compared: the letter each side reads from each reply (TIGER-Lab's
extraction), and whether the replies are identical. What counts as the same:
the same letter on at least 46 of the 50. Greedy decoding on two machines with
other batch sizes, KV cache types and kernels isn't bit-for-bit the same, so
whole replies may part, and are counted, not required.

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


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()
            if x.strip()]


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
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        got = fb.parity_compare(read(a.server), read(a.box))
        print(got["words"])
        return 0 if got["ok"] else 1
    from service import db, served
    from service import frontier as sf
    db.init()
    rec = served.get(a.served_as)
    if not rec or served.is_openrouter(rec):
        raise SystemExit(f"{a.served_as} isn't a model served by a llama-server here")
    n = sf.parity_ask(rec, a.out, lambda k, of: print(f"{k} of {of}", end="\r", flush=True))
    print(f"\n{n} answers · {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
