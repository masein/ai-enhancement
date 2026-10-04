"""14.4.4: what a data set's questions and scores may be used for, said the same
way wherever they appear — a badge of a word or two, and one sentence.

Each set's manifest under eval_tasks/ says it: its `licence`, its `restriction`
and the board's keys it `covers` (task names, which are its GGUF benchmarks'
too). A source of reported scores says it in its own record
(service/reported.SOURCES). This file turns that into the badge and the
sentence, and says what is refused:

- "non-commercial" (the full Mobile-MMLU): internal research only;
- "internal-use" (Mobile-MMLU-Pro): its scores may be used, its questions and
  our answer key never published;
- "internal-only" (Artificial Analysis's free data): its own words.

Every restricted set is refused as a training target (the improve pipeline,
the question builder, few-shot examples and the training-data builders), and
a non-commercial one is refused by every export that leaves the server."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "eval_tasks"

BADGE = {"non-commercial": "Non-commercial", "internal-use": "Internal use",
         "internal-only": "Internal only"}
WORDS = {
    "non-commercial": ("For internal research only. Don't use these questions or scores in "
                       "anything commercial: product claims, marketing, sales material or "
                       "customer reports. Don't publish the questions or our answer key."),
    "internal-use": ("Its scores may be used. The questions and our answer key must not be "
                     "published."),
}
# what never leaves the server, whatever the export
STAYS = ("non-commercial",)


def entry(key: str, name: str, licence: str, restriction: str, sentence: str = "") -> dict:
    """one set's restriction, as every surface shows it: {key, name, licence,
    restriction, badge, sentence}"""
    if restriction not in BADGE:
        raise ValueError(f"{key}: unknown restriction {restriction!r} — one of "
                         + ", ".join(BADGE))
    return {"key": key, "name": name, "licence": licence, "restriction": restriction,
            "badge": BADGE[restriction],
            "sentence": sentence or (f"{licence}. " if licence else "") + WORDS[restriction]}


def _manifests() -> list[dict]:
    out = []
    for p in sorted(MANIFESTS.glob("*/manifest.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(m, dict) and m.get("restriction"):
            out.append(m)
    return out


def sets() -> dict[str, dict]:
    """every restricted key the manifests cover — {key: entry} — but a set
    that is switched off on this server (14.4.5)"""
    out = {}
    for m in _manifests():
        if switched_off(m):
            continue
        for key in m.get("covers") or []:
            out[key] = entry(key, m.get("board_name") or m.get("name") or key,
                             m.get("licence") or "", m["restriction"])
    return out


def switched_off(m: dict) -> bool:
    """14.4.5: a set whose manifest names a switch (`switch`: an .env setting,
    MOBILE_MMLU_FULL) that is off on this server is nowhere, not even badged"""
    name = m.get("switch")
    if not name:
        return False
    try:
        from service import config
        return not getattr(config, name, True)
    except ImportError:
        import os
        return os.environ.get(name, "1").strip().lower() in ("0", "no", "off", "false")


def of(key: str) -> dict | None:
    return sets().get(key)


def never_trained(keys) -> str:
    """'' when none of `keys` is restricted; else why it can't be a training
    target, in one line"""
    for k in keys or []:
        e = of(str(k))
        if e:
            return (f"{e['name']} is {e['badge'].lower()} ({e['licence']}): never a "
                    "training target, a few-shot example or a source of questions")
    return ""


def stays_here(keys) -> str:
    """'' when every one of `keys` may leave the server in an export; else why
    not, in one line"""
    for k in keys or []:
        e = of(str(k))
        if e and e["restriction"] in STAYS:
            return f"{e['name']} is {e['badge'].lower()}: it never leaves this server. {e['sentence']}"
    return ""


def csv_columns(keys) -> tuple[list[str], list[str]]:
    """the two columns an export's row carries for its benchmark: (licence,
    restriction), '' for an unrestricted one"""
    e = next((of(str(k)) for k in keys or [] if of(str(k))), None)
    return (["licence", "restriction"], [e["licence"], e["badge"]] if e else ["", ""])
