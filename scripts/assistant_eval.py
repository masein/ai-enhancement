#!/usr/bin/env python3
"""28: the assistant's real-model evaluation (docs/ASSISTANT.md) — the test
bank (tests/fixtures/assistant/queries.jsonl) asked of one model, each
question as the board asks it (the same prompt, schema and follow-up), its
action resolved against the board and compared with the bank's. Prints the
pass rate per scenario, the median and 90th-percentile reply time, the prompt's
tokens, and every miss with what the model returned.

On the server, in the board's container, never in CI (it asks a model):

    sudo docker compose exec -T bench python scripts/assistant_eval.py \\
        --provider served:served/Qwen3.6-35B-A3B-Q4-original-k-8

--provider: served:<a served model's id>, llama:<a llama-server's /v1 URL>,
openrouter:<model> (paid: ask first). Default: ASSISTANT_PROVIDER, else the
Playground's Q4. It reads the board and writes nothing but its output."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TARGET, FLOOR = 0.95, 0.85


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="")
    ap.add_argument("--bank", default="")
    ap.add_argument("--only", default="", help="scenarios, comma-separated")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default="", help="also write each answer here, a line each")
    a = ap.parse_args(argv)
    from service import assistant, db
    db.init()
    prov = a.provider or assistant.provider()
    bank_path = Path(a.bank) if a.bank else assistant.BANK_FILE
    bank = [json.loads(x) for x in bank_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.only:
        bank = [b for b in bank if b["scenario"] in a.only.split(",")]
    if a.limit:
        bank = bank[:a.limit]
    cat = assistant.catalog()
    system, sch = assistant.prompt(cat), assistant.schema(cat)
    per: dict[str, list[bool]] = defaultdict(list)
    times, misses, tokens, rows = [], [], [], []
    print(f"{len(bank)} questions, asked of {prov}; the prompt is {len(system):,} characters",
          flush=True)
    for i, b in enumerate(bank, 1):
        user = assistant.user_message(b["q"], b.get("prev"), cat["today"])
        t0 = time.time()
        try:
            got = assistant.call(system, user, sch, prov)
            raw, err = got["raw"], ""
            if (got.get("usage") or {}).get("prompt_tokens"):
                tokens.append(got["usage"]["prompt_tokens"])
        except assistant.Down as e:
            raw, err = "", f"down: {e}"
        ms = round(1000 * (time.time() - t0))
        times.append(ms)
        act = assistant.parse(raw)
        res = assistant.resolve(act, cat) if act else None
        ok = bool(res) and assistant.same(res, b["want"], cat)
        per[b["scenario"]].append(ok)
        rows.append({"id": b["id"], "scenario": b["scenario"], "q": b["q"], "ok": ok, "ms": ms,
                     "raw": raw, "got": res, "want": b["want"], "error": err})
        if not ok:
            misses.append(rows[-1])
        if i % 25 == 0:
            print(f"  {i} of {len(bank)} asked", flush=True)
    if a.json:
        Path(a.json).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                encoding="utf-8")
    n = sum(len(v) for v in per.values())
    passed = sum(sum(v) for v in per.values())
    print(f"\n{prov}\n")
    print("| Scenario | Questions | Passed | Rate |")
    print("|---|---:|---:|---:|")
    for sc in sorted(per, key=lambda s: [b["scenario"] for b in bank].index(s)):
        v = per[sc]
        print(f"| {sc} | {len(v)} | {sum(v)} | {100 * sum(v) / len(v):.0f}% |"
              + ("  ← under 85%" if sum(v) / len(v) < FLOOR else ""))
    print(f"| **all** | {n} | {passed} | **{100 * passed / max(n, 1):.1f}%** |")
    if times:
        q = sorted(times)
        print(f"\nReply time: median {statistics.median(q) / 1000:.2f} s, 90th percentile "
              f"{q[int(0.9 * (len(q) - 1))] / 1000:.2f} s"
              + (f"; the prompt {statistics.median(tokens):,.0f} tokens" if tokens else ""))
    worst = min((sum(v) / len(v) for v in per.values()), default=0)
    print("Target: 95% overall and no scenario under 85% — "
          + ("met." if passed / max(n, 1) >= TARGET and worst >= FLOOR else "not met."))
    if misses:
        print(f"\nMisses ({len(misses)}):")
        for m in misses:
            print(f"- {m['id']} [{m['scenario']}] {m['q']!r}\n    wanted {json.dumps(m['want'])}"
                  f"\n    got    {json.dumps(m['got']) if m['got'] else (m['error'] or repr(m['raw'][:300]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
