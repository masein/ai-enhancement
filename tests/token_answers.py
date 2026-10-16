"""21d: stored answers for the token stats, and masein's ~/tokens.py run on
them (tests/fixtures/tokens_py/tokens.py, as he gave it). The answers are
invented and varied where the two ways of counting could part: thinking of
every size, tags left out of an answer's share, a reply that starts inside
its thinking, two thinking blocks, thinking that never closes, a cut after the
thinking closed, an answer with no tokens counted, right, wrong and not graded
yet, counts where 0.9·n and n/2 fall on a whole number, and a benchmark whose
answers are in but not scored (read from answers.jsonl, as tokens.py reads it)."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKENS_PY = Path(__file__).resolve().parent / "fixtures" / "tokens_py" / "tokens.py"
ROWS = ["served__Qwen3.6-35B-A3B-BF16-lda-2610__thinking",
        "served__step4000-lda-2610__thinking",
        "served__Qwen3.6-35B-A3B-Q4-original-k-8__thinking"]
# task, questions, runs a question
TASKS = [("gpqa_diamond_epoch", 30, 4), ("otis_aime_epoch", 5, 8), ("simpleqa_epoch", 100, 1),
         ("arc_agi2_public", 15, 2)]
UNSCORED = "math_l5_epoch"                  # its answers in, no samples yet


def answer(rng: random.Random) -> tuple[str, int | None, str | None, bool | None]:
    """one run: (its text, its output tokens, how it finished, right)"""
    think = "x" * rng.randint(0, 3000)
    after = "The answer is (B). " * rng.randint(0, 6)
    kind = rng.randint(0, 9)
    if kind == 0:                            # its thinking never closed: ran out
        return "<think>" + think, rng.choice([32768, 81920]), rng.choice(["length", None]), False
    if kind == 1:                            # cut after its thinking closed
        return f"<think>{think}</think>{after}", 81920, "length", False
    if kind == 2:                            # began inside its thinking
        text = f"{think}</think>\n{after}"
    elif kind == 3:                          # two blocks
        text = f"<think>{think}</think>{after}<think>{'y' * rng.randint(1, 400)}</think>ok"
    elif kind == 4:                          # no thinking at all
        text = after or "B"
    else:
        text = f"<think>\n{think}\n</think>\n\n{after}"
    tokens = None if rng.random() < 0.04 else rng.randint(40, 30000)
    ok = rng.choice([True, True, False, None]) if rng.random() < 0.1 else rng.choice([True, False])
    return text, tokens, "stop", ok


def write(out_dir: Path, seed: int = 21) -> list[tuple[str, str]]:
    """the rows' stored answers under out_dir (the board's OUT_DIR); each
    (row, task) written"""
    from service import frontier as sf
    rng, done = random.Random(seed), []
    for row in ROWS:
        for task, n, k in TASKS:
            d = sf.task_dir(out_dir / row, task)
            d.mkdir(parents=True, exist_ok=True)
            lines = []
            for q in range(n):
                runs = [answer(rng) for _ in range(k)]
                lines.append(json.dumps({
                    "doc_id": q, "doc": {"id": f"q{q}"}, "resps": [[t] for t, _, _, _ in runs],
                    "filtered_resps": ["B"] * k,
                    "frontier": [{"epoch": e, "tokens": tok, "finish": fin, "ok": ok,
                                  "ran_out": fin == "length" or "</think>" not in t
                                  and t.startswith("<think>")}
                                 for e, (t, tok, fin, ok) in enumerate(runs)]}))
            (d / f"samples_{task}_2026-10-16T09-00-00.000000.jsonl").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            done.append((row, task))
        d = sf.task_dir(out_dir / row, UNSCORED)
        d.mkdir(parents=True, exist_ok=True)
        recs = []
        for q in range(24):
            t, tok, fin, _ = answer(rng)
            recs.append({"id": f"m{q}", "epoch": 0, "answer": t, "tokens": tok, "finish": fin})
        recs.append({"id": "m-err", "epoch": 0, "error": "the server timed out"})
        (d / sf.ANSWERS).write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        done.append((row, UNSCORED))
    return done


def tokens_py(results_root: Path) -> dict[tuple[str, str], dict]:
    """masein's script on the stored answers: {(row, task): its line}"""
    got = subprocess.run([sys.executable, str(TOKENS_PY)], cwd=str(ROOT), capture_output=True,
                         text=True, timeout=300,
                         env={**os.environ, "OUT_ROOT": str(results_root)})
    assert got.returncode == 0, got.stderr[-2000:]
    out = {}
    for line in got.stdout.splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            out[(r.pop("row"), r.pop("task"))] = r
    return out
