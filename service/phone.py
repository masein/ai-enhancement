"""12f.2: On phone — what someone measured on the phone, typed in.

A phone build is a model served elsewhere (12f.1) whose registration says
so: its "phone build" box, or "phone" in how it's served. Its card holds the
numbers someone measured on the phone — the device, chip and RAM, decode
tokens per second (median and best), the settings, the date, who measured
it, and any quality they reported with its source.

- **Every number is as reported**, with who measured it and when. The board
  never computes or estimates a phone's speed, and doesn't show its server's
  own speed here: the fork's README says CUDA throughput doesn't represent
  the phone.
- **Reported and measured are never mixed.** Reports live in their own table
  and reach the page on their own endpoint: never in a results file, a board
  column or an average. A reported MMLU is shown as reported, beside its
  source — not in the Standard column the board measured itself.
"""

from __future__ import annotations

import datetime as _dt
import time

from . import db, served

# The first card, from the fork's README. masein confirms it with the
# colleague who measured it: the form offers it, and saves nothing until a
# person enters who measured it
README = {
    "device": "OnePlus 15", "chip": "Snapdragon 8 Elite Gen 5", "ram_gb": 16,
    "settings": "experts streamed from flash, lookahead 1, MTP n_max 3, fusion off",
    "decode_median": 13.5, "decode_best": 16.0, "repeats": "3 cold repeats, ≤65 °C",
    "quality": [{"name": "MMLU", "value": "81.98%", "note": "all 14,042, measured on Metal"}],
    "source": "the fork's README (teraformer/lda-2026-09-22)"}
TEXT = ("device", "chip", "settings", "repeats", "source")


def builds() -> list[dict]:
    """the phone builds, each with what was reported from the phone, newest first"""
    out = []
    for r in db.served_all():
        if served.is_phone(r):
            out.append({"id": r["id"], "name": r["name"], "based_on": r["based_on"],
                        "reports": reports(r["id"])})
    return out


def reports(model_id: str) -> list[dict]:
    return db.phone_list(model_id)


def _num(v, what: str, required: bool = False) -> float | None:
    if v in (None, ""):
        if required:
            raise ValueError(f"{what}: the number measured on the phone")
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what} is a number") from None
    if not x > 0:
        raise ValueError(f"{what} is a number above 0")
    return x


def add(f: dict, entered_by: str) -> dict:
    """keep one report — as typed, with who measured it and when — or say why not"""
    rec = served.get(f.get("model") or "")
    if not rec or not served.is_phone(rec):
        raise ValueError("Numbers from the phone go with a phone build: a model served "
                         "elsewhere, registered as a phone build")
    by = (f.get("by") or "").strip()
    if not by:
        raise ValueError("Who measured it: every number here is shown as reported by them")
    date = (f.get("date") or "").strip()
    try:
        _dt.date.fromisoformat(date)
    except ValueError:
        raise ValueError("The date it was measured, as 2026-09-26") from None
    out = {k: (f.get(k) or "").strip()[:300] for k in TEXT}
    if not out["device"]:
        raise ValueError("The device it was measured on")
    out["ram_gb"] = _num(f.get("ram_gb"), "RAM in GB")
    out["decode_median"] = _num(f.get("decode_median"), "Decode tokens per second, the median",
                                required=True)
    out["decode_best"] = _num(f.get("decode_best"), "Decode tokens per second, the best")
    if out["decode_best"] is not None and out["decode_best"] < out["decode_median"]:
        raise ValueError("The best decode speed is under the median: check the two")
    quality = []
    for q in f.get("quality") or []:
        name, value = str(q.get("name") or "").strip(), str(q.get("value") or "").strip()
        if not name and not value:
            continue
        if not (name and value):
            raise ValueError("Each reported score has a benchmark and its value")
        quality.append({"name": name[:60], "value": value[:40],
                        "note": str(q.get("note") or "").strip()[:200]})
    out.update({"model": rec["id"], "by": by[:80], "date": date, "quality": quality,
                "entered_by": (entered_by or "").strip()[:80], "at": time.time()})
    out["id"] = db.phone_add(rec["id"], out)
    return out
