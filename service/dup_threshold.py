"""12o.1: the duplicate check's cosine for the embedding model on this server,
checked on this bank — run once on the server after deploying:

    sudo docker compose exec -T bench python -m service.dup_threshold

Every text the builder compared before (the banks, and each batch's questions)
that OpenRouter's model embedded is in the builder's cache. Their pairs, as the
builder compares them (Everyday with Everyday; a Knowledge exam topic with its
own topic), are flagged where OpenRouter's cosine reaches QB_DUP_COSINE. The
local model embeds the same texts on this server, and the cosine chosen is the
one whose flags agree with those best — every pair OpenRouter's model caught,
and as few others as that allows. Written to BENCH_ROOT/builder/
dup_threshold.json, which the builder reads; it prints counts and ids, never a
question."""

from __future__ import annotations

import json
import math
import time

from . import builder, config, db, embed_local


def _cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a))
                                                * math.sqrt(sum(y * y for y in b)) or 1.0)


def sets() -> list[list[str]]:
    """the texts the builder compares with each other: each kind's bank — a
    Knowledge topic's own — with every batch of that kind (and topic)"""
    builder._scripts()
    import everyday as ev
    import exam_build as eb
    out: dict[str, set[str]] = {"everyday": {q["prompt"] for q in ev.load_bank()}}
    if config.EXAM_DIR.is_dir():
        for topic, rows in eb.load_bank(config.EXAM_DIR).items():
            out["knowledge:" + topic] = {r["prompt"] for r in rows}
    for d in db.qb_list():
        k = d["kind"] if d["kind"] != "knowledge" else "knowledge:" + d["spec"].get("topic", "")
        out.setdefault(k, set()).update(builder._text(d, it["q"]) for it in d["items"])
    return [sorted(x) for x in out.values()]


def choose(pairs: list[tuple[float, float]], remote_cut: float) -> dict:
    """pairs: (remote cosine, local cosine). The local cosine whose flags
    disagree least with the remote model's — no pair it caught missed, where
    one exists that misses none and adds fewest; rounded down, so it catches"""
    flagged = sorted(lc for rc, lc in pairs if rc >= remote_cut)
    others = sorted(lc for rc, lc in pairs if rc < remote_cut)
    if not flagged:
        return {"cosine": None, "flagged": 0, "caught": 0, "extra": 0}
    best = None
    for t in sorted({round(x, 4) for x in flagged}):
        miss = sum(1 for x in flagged if x < t)
        extra = sum(1 for x in others if x >= t)
        k = (miss + extra, miss, -t)
        if best is None or k < best[0]:
            best = (k, t, miss, extra)
    _, t, miss, extra = best
    t = math.floor(t * 100) / 100
    return {"cosine": t, "flagged": len(flagged), "caught": len(flagged) - sum(
        1 for x in flagged if x < t), "extra": sum(1 for x in others if x >= t)}


def check(embed=None, write: bool = True) -> dict:
    embed = embed or embed_local.embed
    p = config.BENCH_ROOT / "builder" / "embeddings.json"
    cache = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    remote = lambda t: cache.get(builder._sha(config.OPENROUTER_EMBED_MODEL + "\0" + t))  # noqa: E731
    pairs, n_texts = [], 0
    for texts in sets():
        have = [t for t in texts if remote(t) is not None]
        if len(have) < 2:
            continue
        local = embed(have)
        if local is None:
            raise SystemExit("the embedding model isn't in this image: build it again")
        n_texts += len(have)
        rv = [remote(t) for t in have]
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                pairs.append((_cos(rv[i], rv[j]), _cos(local[i], local[j])))
    got = choose(pairs, config.QB_DUP_COSINE)
    out = {"model": embed_local.IDENT, "remote_model": config.OPENROUTER_EMBED_MODEL,
           "remote_cosine": config.QB_DUP_COSINE, "texts": n_texts, "pairs": len(pairs),
           **got, "at": time.time()}
    if write and got["cosine"] is not None:
        f = config.BENCH_ROOT / "builder" / "dup_threshold.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out), encoding="utf-8")
    return out


def main() -> None:
    out = check()
    if out["cosine"] is None:
        print(f"{out['texts']} texts, {out['pairs']} pairs: none flagged by "
              f"{out['remote_model']} at {out['remote_cosine']}, so nothing to check the local "
              f"model against. It keeps {builder.dup_cosine()}.")
        return
    print(f"{out['texts']} texts, {out['pairs']} pairs. {out['remote_model']} flagged "
          f"{out['flagged']} at {out['remote_cosine']}; {embed_local.MODEL} at {out['cosine']} "
          f"catches {out['caught']} of them and flags {out['extra']} it didn't. The builder "
          f"uses {out['cosine']} from now on (BENCH_ROOT/builder/dup_threshold.json).")


if __name__ == "__main__":
    main()
