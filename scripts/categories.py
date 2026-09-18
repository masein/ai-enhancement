"""MMLU subjects -> the human categories in scripts/categories.yaml.

Read here without pyyaml on purpose: diagnose.py and report_lm_eval.py run
with whatever python the box has (`sudo python3 …` on the server), and a
dependency for a fifteen-line flat file is a dependency too many. The reader
accepts exactly the shape the file documents and refuses anything else, so a
stray indent or a duplicate subject fails loudly instead of silently
reshuffling a category.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

YAML_PATH = Path(__file__).with_name("categories.yaml")
OTHER = "other"        # where an unmapped subject lands — visible, never dropped

_CAT = re.compile(r"^([A-Za-z][A-Za-z0-9 &,'\-]*?):\s*$")
_SUB = re.compile(r"^\s+-\s+([a-z0-9_]+)\s*$")


def parse(text: str) -> dict[str, list[str]]:
    """category -> subjects, in file order. ValueError names the bad line."""
    out: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    cur: str | None = None
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        m = _CAT.match(line)
        if m:
            cur = m.group(1).strip()
            if cur in out:
                raise ValueError(f"categories.yaml:{no}: category {cur!r} listed twice")
            out[cur] = []
            continue
        m = _SUB.match(line)
        if m and cur is not None:
            s = m.group(1)
            if s in seen:
                raise ValueError(f"categories.yaml:{no}: subject {s!r} is already under "
                                 f"{seen[s]!r}")
            seen[s] = cur
            out[cur].append(s)
            continue
        raise ValueError(f"categories.yaml:{no}: expected 'category:' or '  - subject', "
                         f"got {raw!r}")
    return out


@functools.lru_cache(maxsize=4)
def _table(path: Path) -> dict[str, list[str]]:
    return parse(path.read_text(encoding="utf-8"))


def load(path: Path = YAML_PATH) -> dict[str, str]:
    """subject -> category."""
    return {s: c for c, subs in _table(path).items() for s in subs}


def category_order(path: Path = YAML_PATH) -> list[str]:
    """Categories in file order, `other` last whatever the file says — it is
    the bucket for what the file does not know, and reads that way."""
    cats = list(_table(path))
    return [c for c in cats if c != OTHER] + [OTHER]


def topic_slug(name: str) -> str:
    """'medicine & health' -> 'medicine_health': the topic as a task name."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def categorize(subject: str, path: Path = YAML_PATH) -> str | None:
    """The category, or None when the subject is not in the file. Callers
    that roll up decide what None means (diagnose.py: `other`, and listed)."""
    return load(path).get(subject)
