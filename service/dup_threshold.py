"""12o.5: the duplicate check's cosine for the embedding model in this image,
chosen on a labelled set — run once on the server after deploying:

    sudo docker compose exec -T bench python -m service.dup_threshold

eval_tasks/everyday/dup_pairs.jsonl holds pairs from the Everyday bank's
PRACTICE half only — the half that may be listed: sixty questions each with
the same question reworded by hand (no 13 words in a row shared, so only the
embeddings can catch them), and sixty hard negatives, two different
questions of one group, the most alike by the words they share. The model
embeds them here; the cosine chosen is the highest that still flags at least
95% of the rewordings — so the fewest different questions flagged — rounded
down to two places. Both rates are printed and written, with the set's sha256,
to BENCH_ROOT/builder/dup_threshold.json, which the builder reads. It prints
counts and rates, never a question.

Before 12o.5 it agreed the local model with the pairs OpenRouter's model had
flagged; with nothing flagged there, it had nothing to go on."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from . import builder, config, embed_local

PAIRS = Path(__file__).resolve().parent.parent / "eval_tasks" / "everyday" / "dup_pairs.jsonl"
RECALL = 0.95


def _cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a))
                                                * math.sqrt(sum(y * y for y in b)) or 1.0)


def labelled(path: Path = PAIRS) -> dict:
    """the set's pairs as texts: a practice question and its rewording, or two
    practice questions of one group. A pair whose question is no longer in
    the practice half — retired, or moved by an edit — is left out, counted"""
    builder._scripts()
    import everyday as ev
    bank = {q["id"]: q for q in ev.load_bank() if ev.half(q) == ev.PRACTICE}
    same, different, gone = [], [], 0
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        ids = [r["id"]] if r["kind"] == "reworded" else r["ids"]
        if any(i not in bank for i in ids):
            gone += 1
            continue
        if r["kind"] == "reworded":
            same.append((bank[r["id"]]["prompt"], r["text"]))
        else:
            different.append((bank[ids[0]]["prompt"], bank[ids[1]]["prompt"]))
    return {"same": same, "different": different, "gone": gone,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def choose(same: list[float], different: list[float], recall: float = RECALL) -> dict:
    """same/different: each pair's cosine. The highest cosine that flags at
    least `recall` of the rewordings — a higher one would miss more, and a
    lower one flags more different questions — rounded down to two places,
    so it still does"""
    if not same:
        return {"cosine": None}
    need = math.ceil(recall * len(same) - 1e-9)
    t = math.floor(sorted(same, reverse=True)[need - 1] * 100 + 1e-9) / 100
    caught = sum(1 for x in same if x >= t)
    false = sum(1 for x in different if x >= t)
    return {"cosine": t, "reworded": len(same), "caught": caught,
            "recall": round(caught / len(same), 4), "different": len(different),
            "false_flags": false,
            "false_rate": round(false / len(different), 4) if different else None}


def check(embed=None, write: bool = True, path: Path = PAIRS) -> dict:
    embed = embed or embed_local.embed
    got = labelled(path)
    texts = sorted({t for p in got["same"] + got["different"] for t in p})
    vecs = embed(texts)
    if vecs is None:
        raise SystemExit("the embedding model isn't in this image: build it again")
    v = dict(zip(texts, vecs))
    out = {"model": embed_local.IDENT, "pairs_sha256": got["sha256"], "gone": got["gone"],
           "want_recall": RECALL,
           **choose([_cos(v[a], v[b]) for a, b in got["same"]],
                    [_cos(v[a], v[b]) for a, b in got["different"]]),
           "at": time.time()}
    if write and out["cosine"] is not None:
        f = config.BENCH_ROOT / "builder" / "dup_threshold.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out), encoding="utf-8")
    return out


def main() -> None:
    out = check()
    if out["cosine"] is None:
        print(f"no pair of the labelled set is in the practice half now ({out['gone']} left "
              f"out), so nothing to choose on. It keeps {builder.dup_cosine()}.")
        return
    print(f"{embed_local.MODEL} at cosine {out['cosine']} flags {out['caught']} of "
          f"{out['reworded']} reworded questions ({out['recall']:.0%}) and {out['false_flags']} "
          f"of {out['different']} different ones ({out['false_rate']:.0%})"
          + (f"; {out['gone']} pairs left out, no longer in the practice half" if out["gone"]
             else "")
          + f". The builder uses {out['cosine']} from now on "
          f"(BENCH_ROOT/builder/dup_threshold.json).")


if __name__ == "__main__":
    main()
