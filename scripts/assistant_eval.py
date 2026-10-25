#!/usr/bin/env python3
"""28: the assistant's real-model evaluation (docs/ASSISTANT.md) — the test
bank (tests/fixtures/assistant/queries.jsonl) asked of one model, each
question as the board asks it (the same prompt, schema and follow-up), its
action resolved against the board and compared with the bank's. Prints the
pass rate per scenario for each half of the bank — the working half, which
the prompt is tuned on, and the held-out half, which it never is — the median
and 90th-percentile reply time, the prompt's tokens, and every miss with what
the model wrote (before the board matched it) and what the board made of it.
Tune with --half work; report both.

On the server, in the board's container, never in CI (it asks a model):

    sudo docker compose exec -T bench python scripts/assistant_eval.py \\
        --provider served:served/Qwen3.6-35B-A3B-Q4-original-k-8

--provider: served:<a served model's id>, llama:<a llama-server's /v1 URL>,
openrouter:<model> (paid: ask first; sent only the parameters the model takes,
read once for the run). Three answers in a row that don't come stop it.
Default: ASSISTANT_PROVIDER, else the Playground's Q4. It reads the board and
writes nothing but its output and every answer it got, raw, to
BENCH_ROOT/assistant/evals/ (internal, as the log is; the run prints the
file) — so a change to the grading or the board can be scored again on the
same answers, without asking the model:

    sudo docker compose exec -T bench python scripts/assistant_eval.py \
        --rescore <the file the run printed>

--thumbs-down prints the questions the board's log has a thumbs down on,
with the turn before, the board's action, the reason chosen and what the
person wanted, and under their own heading every change made in the panel
(a chip, a suggestion) — the next bank's questions; it asks nothing."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TARGET, FLOOR = 0.95, 0.85


def wrote(raw: str):
    """what the model wrote, before the board matched anything: its JSON as
    sent ({"do", "then"}), or None"""
    try:
        got = json.loads(raw or "")
    except ValueError:
        return None
    return got if isinstance(got, dict) else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="")
    ap.add_argument("--bank", default="")
    ap.add_argument("--only", default="", help="scenarios, comma-separated")
    ap.add_argument("--half", default="", choices=("", "work", "held"),
                    help="one half of the bank alone (tune on work)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default="", help="write each answer here, a line each (default: "
                    "BENCH_ROOT/assistant/evals/<time>-<provider>.jsonl)")
    ap.add_argument("--rescore", default="", help="a run's answers file: resolve and grade them "
                    "again, asking nothing")
    ap.add_argument("--thumbs-down", action="store_true", help="print the questions the board's "
                    "log has a thumbs down on, a JSON line each — the next bank's; asks nothing")
    a = ap.parse_args(argv)
    from service import assistant, config, db
    if a.thumbs_down:
        down, changed = assistant.thumbs_down(), assistant.edits()
        print(f"Thumbs down ({len(down)}):")
        for x in down:
            print(json.dumps(x, ensure_ascii=False))
        # 28g: a chip changed, a suggestion or "All three" taken — a sign of a miss
        print(f"\nEdits in the panel ({len(changed)}):")
        for x in changed:
            print(json.dumps(x, ensure_ascii=False))
        return 0
    db.init()
    prov = a.provider or assistant.provider()
    bank_path = Path(a.bank) if a.bank else assistant.BANK_FILE
    bank = [json.loads(x) for x in bank_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    kept = {}
    if a.rescore:
        kept = {r["id"]: r for r in (json.loads(x) for x in Path(a.rescore).read_text(
            encoding="utf-8").splitlines() if x.strip())}
        bank = [b for b in bank if b["id"] in kept]
        prov = next(iter(kept.values()), {}).get("provider") or prov
    if a.only:
        bank = [b for b in bank if b["scenario"] in a.only.split(",")]
    if a.half:
        bank = [b for b in bank if b.get("half") == a.half]
    if a.limit:
        bank = bank[:a.limit]
    cat = assistant.catalog()
    system, sch = assistant.prompt(cat), assistant.schema(cat)
    per: dict[tuple[str, str], list[bool]] = defaultdict(list)
    times, misses, tokens, rows = [], [], [], []
    print(f"{len(bank)} questions, " + (f"{prov}'s answers kept in {a.rescore} graded again"
                                        if a.rescore else f"asked of {prov}")
          + f"; the prompt is {len(system):,} characters", flush=True)
    if prov.startswith("openrouter:") and not a.rescore:
        # 28e: only the parameters the model takes, read once for the run
        model = prov.split(":", 1)[1]
        try:
            left = assistant.openrouter_fit(model, fresh=True)
        except assistant.Unfit as e:
            print(f"Stopped before asking: {e}.")
            return 2
        print(f"OpenRouter: {model} is sent only the parameters it takes — "
              + (f"left out: {', '.join(left)}" if left else "nothing left out"), flush=True)
    downs = 0
    for i, b in enumerate(bank, 1):
        if a.rescore:
            k = kept[b["id"]]
            raw, err, ms, tok = (k.get("raw") or "", k.get("error") or "", k.get("ms") or 0,
                                 k.get("tokens"))
        else:
            user = assistant.user_message(b["q"], b.get("prev"), cat["today"], cat)
            t0, tok = time.time(), None
            try:
                got = assistant.call(system, user, sch, prov)
                raw, err = got["raw"], ""
                tok = (got.get("usage") or {}).get("prompt_tokens")
            except assistant.Down as e:
                raw, err = "", f"down: {e}"
            ms = round(1000 * (time.time() - t0))
            # 28e: three in a row that didn't come — the model isn't there; stop
            downs = downs + 1 if err else 0
            if downs == 3:
                print(f"Stopped after three answers in a row didn't come ({b['id']} the third): "
                      f"{err}")
                return 1
        times.append(ms)
        if tok:
            tokens.append(tok)
        act = assistant.parse(raw)
        pv = b.get("prev") or {}
        res = assistant.resolve(act, cat, b["q"], pv.get("q") or "", pv.get("action")) if act \
            else None
        ok = bool(res) and assistant.same(res, b["want"], cat, b.get("prev"))
        per[(b["scenario"], b.get("half") or "work")].append(ok)
        rows.append({"id": b["id"], "scenario": b["scenario"], "half": b.get("half") or "work",
                     "q": b["q"], "ok": ok, "ms": ms, "raw": raw, "wrote": wrote(raw),
                     "got": res, "want": b["want"], "error": err, "provider": prov,
                     "tokens": tok})
        if not ok:
            misses.append(rows[-1])
        if i % 25 == 0 and not a.rescore:
            print(f"  {i} of {len(bank)} asked", flush=True)
    if not a.rescore:
        out = Path(a.json) if a.json else (
            Path(config.BENCH_ROOT) / "assistant" / "evals"
            / f"{time.strftime('%Y%m%d-%H%M')}-{re.sub(r'[^A-Za-z0-9.]+', '-', prov)[-60:]}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                       encoding="utf-8")
        print(f"Every answer is kept in {out}", flush=True)
    def rate(v):
        return f"{sum(v)}/{len(v)} ({100 * sum(v) / len(v):.0f}%)" if v else "–"
    print(f"\n{prov}\n")
    print("| Scenario | Working half | Held-out half | Both |")
    print("|---|---:|---:|---:|")
    scs = list(dict.fromkeys(b["scenario"] for b in bank))
    for sc in scs:
        w, h = per.get((sc, "work"), []), per.get((sc, "held"), [])
        print(f"| {sc} | {rate(w)} | {rate(h)} | {rate(w + h)} |"
              + ("  ← under 85%" if (w + h) and sum(w + h) / len(w + h) < FLOOR else ""))
    allw = [x for (s, hf), v in per.items() if hf == "work" for x in v]
    allh = [x for (s, hf), v in per.items() if hf == "held" for x in v]
    print(f"| **all** | **{rate(allw)}** | **{rate(allh)}** | **{rate(allw + allh)}** |")
    n, passed = len(allw + allh), sum(allw + allh)
    if times:
        q = sorted(times)
        print(f"\nReply time: median {statistics.median(q) / 1000:.2f} s, 90th percentile "
              f"{q[int(0.9 * (len(q) - 1))] / 1000:.2f} s"
              + (f"; the prompt {statistics.median(tokens):,.0f} tokens" if tokens else ""))
    by_sc = defaultdict(list)
    for (s, _), v in per.items():
        by_sc[s] += v
    worst = min((sum(v) / len(v) for v in by_sc.values()), default=0)
    print("Target: 95% overall and no scenario under 85% — "
          + ("met." if passed / max(n, 1) >= TARGET and worst >= FLOOR else "not met."))
    if misses:
        print(f"\nMisses ({len(misses)}):")
        for m in misses:
            print(f"- {m['id']} [{m['scenario']}, {m['half']}] {m['q']!r}\n"
                  f"    wanted {json.dumps(m['want'])}\n"
                  f"    wrote  {json.dumps(m['wrote']) if m['wrote'] else repr(m['raw'][:300])}\n"
                  f"    got    {json.dumps(m['got']) if m['got'] else (m['error'] or '–')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
