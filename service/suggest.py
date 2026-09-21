"""Model search behind every model-id box on the page.

Local first — the models this board already knows (board rows, every hf_id in
the queue, uploaded artifacts) — then Hugging Face Hub matches. The Hub is
asked from here, never from the browser: one place to cache (ten minutes per
query), one timeout (two seconds), and a page that never waits on it. When the
Hub does not answer, the local matches come back alone and the list says so.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import config

HUB_TTL_S = 600
HUB_TIMEOUT_S = 2.0
HUB_LIMIT = 10
LOCAL_LIMIT = 12

_hub_cache: dict[str, tuple[float, list[dict]]] = {}
_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hub-search")


def looks_instruct(hf_id: str) -> bool:
    """The kind a name claims, for a model the board has never run: '-Instruct',
    '-it' (a whole token), '-chat'. Preflight decides for real."""
    name = hf_id.split("/")[-1].lower()
    return "it" in set(re.split(r"[^a-z0-9]+", name)) or any(
        w in name for w in ("instruct", "chat"))


def hub_search(q: str) -> list[dict]:
    """The Hub's text-generation models matching `q`, most downloaded first.
    Raises on any failure; the caller turns that into a footer."""
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        found = api.list_models(search=q, sort="downloads", limit=HUB_LIMIT,
                                pipeline_tag="text-generation",
                                expand=["safetensors", "downloads", "pipeline_tag"])
    except TypeError:                     # an older huggingface_hub
        found = api.list_models(search=q, sort="downloads", limit=HUB_LIMIT,
                                filter="text-generation")
    out = []
    for m in found:
        st = getattr(m, "safetensors", None)
        params = getattr(st, "total", None) if st is not None else None
        if params is None and isinstance(st, dict):
            params = st.get("total")
        out.append({"id": m.id, "params": params, "downloads": getattr(m, "downloads", None)})
    return out


def hub(q: str) -> tuple[list[dict], bool]:
    """(matches, answered). Cached per query; a query the Hub is slow on is
    still cached when it does answer, so the next keystroke may find it."""
    key = q.strip().lower()
    now = time.time()
    with _lock:
        hit = _hub_cache.get(key)
    if hit and now - hit[0] < HUB_TTL_S:
        return hit[1], True
    fut = _pool.submit(hub_search, key)

    def keep(f):
        if not f.exception():
            with _lock:
                if len(_hub_cache) > 500:
                    _hub_cache.clear()
                _hub_cache[key] = (time.time(), f.result())
    fut.add_done_callback(keep)
    try:
        return fut.result(timeout=HUB_TIMEOUT_S), True
    except Exception:                     # noqa: BLE001 — timeout, offline, rate limit: all one answer
        return [], False


def local_candidates(payload: dict, queue: list[dict], artifacts: list[str]) -> list[dict]:
    """Every model this board knows, with what it knows about each."""
    seen: dict[str, dict] = {}
    for m in payload.get("models") or []:
        judged = [t for t in ((m.get("judge") or {}).get("tasks") or {})
                  if t.startswith("exam_")]
        seen[m["id"]] = {"id": m["id"], "params": m.get("params"), "kind": m.get("kind"),
                         "on_board": True, "judged": len(judged)}
    for r in queue:
        hid = r.get("hf_id") or ""
        if hid and hid not in seen:
            kind = r.get("kind") if r.get("kind") in ("base", "instruct") else None
            seen[hid] = {"id": hid, "params": r.get("params"), "kind": kind,
                         "on_board": False, "judged": 0, "queued": True}
    for name in artifacts:
        mid = f"local/{name}"
        if mid not in seen:
            seen[mid] = {"id": mid, "params": None, "kind": None, "on_board": False,
                         "judged": 0, "artifact": True}
    return list(seen.values())


def suggest(q: str, local: list[dict]) -> dict:
    """{items, hub_ok, footer}: local matches (case-insensitive substring on
    the whole id), then Hub matches not already listed. A Hub model over the
    size cap is marked, not dropped — picking it lets preflight say why."""
    q = (q or "").strip()
    if len(q) < 2:
        return {"items": [], "hub_ok": True, "footer": ""}
    ql = q.lower()
    mine = [dict(c, source="board" if c.get("on_board") else "known")
            for c in local if ql in c["id"].lower()]
    # the board's own first, most-judged first, then by name
    mine.sort(key=lambda c: (not c.get("on_board"), -(c.get("judged") or 0), c["id"].lower()))
    mine = mine[:LOCAL_LIMIT]
    listed = {c["id"] for c in mine}
    found, ok = hub(q)
    cap = config.MAX_PARAMS_B * 1e9
    theirs = []
    for h in found:
        if h["id"] in listed:
            continue
        theirs.append({"id": h["id"], "params": h.get("params"),
                       "kind": "instruct" if looks_instruct(h["id"]) else None,
                       "kind_guessed": True, "on_board": False, "judged": 0,
                       "over_cap": bool(h.get("params") and h["params"] > cap),
                       "source": "hub"})
    for c in mine:
        if not c.get("kind"):
            c["kind"] = "instruct" if looks_instruct(c["id"]) else None
            c["kind_guessed"] = bool(c["kind"])
    return {"items": mine + theirs, "hub_ok": ok,
            "footer": "" if ok else "Hub search unavailable — models this board knows only",
            "cap_b": config.MAX_PARAMS_B}
