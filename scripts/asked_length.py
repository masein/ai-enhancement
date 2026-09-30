"""12t: which results on disk were asked under lm_eval's 2,048-token fallback.

lm_eval looks for a model's limit at the top of its config and takes 2,048
when it finds none. A model whose config keeps the limit under text_config
(Gemma 4) was therefore asked with every prompt longer than 2,048 tokens cut
from the left, without a word in its log. Since 12t such a model's runs are
told its real limit (runner.nested_limit), so a task it sits again is asked
whole — and may score differently from the same task's result on disk.

lm_eval writes the length it used into each results file (`max_length`), and
the board keeps what the model reads beside its results (model_meta.json's
`ctx`). This lists every task whose newest result was asked at 2,048 by a
model that reads more. It reads files only: no model, no network.

    python scripts/asked_length.py            # on the server, in the container
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FALLBACK = 2048          # lm_eval's HFLM._DEFAULT_MAX_LENGTH (tests/test_image_deps.py pins it)


def _json(p: Path) -> dict:
    try:
        out = json.loads(p.read_text(encoding="utf-8"))
        return out if isinstance(out, dict) else {}
    except (OSError, ValueError):
        return {}


def under_fallback(out_dir: Path) -> list[dict]:
    """[{model, task, asked, reads, file}]: each task folder whose newest
    results file says max_length 2,048, of a model that reads more"""
    rows = []
    for meta in sorted(out_dir.glob("*/model_meta.json")):
        m = _json(meta)
        reads = m.get("ctx")
        if not isinstance(reads, int) or reads <= FALLBACK:
            continue
        for d in sorted(x for x in meta.parent.iterdir() if x.is_dir()):
            files = sorted(d.rglob("results_*.json"), key=lambda f: f.name)
            if files and _json(files[-1]).get("max_length") == FALLBACK:
                rows.append({"model": m.get("model") or meta.parent.name, "task": d.name,
                             "asked": FALLBACK, "reads": reads, "file": str(files[-1])})
    return rows


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args:
        out_dir = Path(args[0])
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from service import config
        out_dir = config.OUT_DIR
    rows = under_fallback(out_dir)
    if not rows:
        print(f"none: no result under {out_dir} was asked at lm_eval's {FALLBACK:,} by a model "
              f"that reads more")
        return 0
    for r in rows:
        print(f"{r['model']} · {r['task']} · asked at {r['asked']:,}, the model reads "
              f"{r['reads']:,}")
    models = sorted({r["model"] for r in rows})
    print(f"{len(rows)} task(s) of {len(models)} model(s): any prompt of theirs longer than "
          f"{FALLBACK:,} tokens was cut from the left. A run that sits them again is asked whole.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
