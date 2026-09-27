"""12i.4: how long a model's answers are, thinking included, and how many ran
out while thinking — from the answers on disk.

On 2026-09-27 the phone build (k=4 + LDA) answered a simple question right every
time but thought about 2.5 times as long as the original, and the original once
thought for 6,000 tokens and never answered. For a phone, both matter as much
as the score.

- **Tokens** are counted in this order of preference, and the number says which:
  1. the server's own count of what it generated (a served model's answers
     record it, 12i.4);
  2. the model's own tokenizer, when its tokenizer.json is on this server (an
     upload's directory, the Hub cache, or the base model's for a served one);
  3. an estimate: a token for every four characters.
- **Ran out while thinking**: an answer whose reasoning never ended, or ended
  with nothing after it (judge.answer_parts' no_answer) — it reached its budget
  before it wrote an answer.

Cached per model in answer_length.json, by the answer files' names, sizes and
mtimes.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

CACHE = "answer_length.json"
CHARS_PER_TOKEN = 4
KINDS = {"everyday": re.compile(r"everyday_\d+shot"),
         "exam": re.compile(r"exam_[a-z0-9_]+_\d+shot")}
_TOKENIZERS: dict[str, object] = {}


def _files(model_dir: Path, kind: str) -> list[Path]:
    pat = KINDS[kind]
    dirs = [d for d in model_dir.iterdir() if d.is_dir() and pat.fullmatch(d.name)] \
        if model_dir.is_dir() else []
    return sorted(f for d in dirs for f in d.rglob("samples_*.jsonl"))


def _records(model_dir: Path, kind: str) -> list[dict]:
    """the newest answer to each question: a later run's file wins"""
    import diagnose as dx
    out: dict[str, dict] = {}
    for f in dx.newest_per_subtask(_files(model_dir, kind)) if kind == "exam" \
            else _files(model_dir, kind):
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                doc = rec.get("doc") or {}
                key = str(doc.get("id") or doc.get("qid") or rec.get("doc_hash") or len(out))
                out[f"{f.parent.parent.name}|{key}"] = rec
    if kind == "everyday":
        # only this wording's questions: an earlier wording's answers are in
        # no score (12a.4), and so in no length either
        import everyday as ev
        now = {q["id"]: q["prompt"] for q in ev.load_bank()}
        out = {k: r for k, r in out.items()
               if now.get((r.get("doc") or {}).get("id")) == (r.get("doc") or {}).get("prompt")}
    return list(out.values())


def _tokenizer_file(repo: str, artifacts: Path | None) -> Path | None:
    if not repo:
        return None
    if repo.startswith("local/"):
        p = (artifacts / repo.split("/", 1)[1] / "tokenizer.json") if artifacts else None
        return p if p and p.is_file() else None
    try:
        from huggingface_hub import try_to_load_from_cache
        got = try_to_load_from_cache(repo, "tokenizer.json")
    except Exception:                       # noqa: BLE001 — no Hub library: no tokenizer
        return None
    return Path(got) if isinstance(got, str) and Path(got).is_file() else None


def tokenizer(repos: list[str], artifacts: Path | None = None):
    """(count function, the repo whose tokenizer it is) — or (None, "")"""
    for repo in repos:
        if repo in _TOKENIZERS:
            if _TOKENIZERS[repo]:
                return _TOKENIZERS[repo], repo
            continue
        f = _tokenizer_file(repo, artifacts)
        fn = None
        if f:
            try:
                from tokenizers import Tokenizer
                tk = Tokenizer.from_file(str(f))
                fn = lambda s, tk=tk: len(tk.encode(s, add_special_tokens=False).ids)  # noqa: E731
            except Exception:               # noqa: BLE001 — unreadable: the next one
                fn = None
        _TOKENIZERS[repo] = fn
        if fn:
            return fn, repo
    return None, ""


def stats(model_dir: Path, kind: str, repos: list[str], artifacts: Path | None = None) -> dict | None:
    """{median, ran_out, n, how, tokenizer} over the model's answers of one
    kind ("everyday" or "exam"); None when it has none"""
    files = _files(model_dir, kind)
    if not files:
        return None
    key = [[f.name, f.stat().st_size, int(f.stat().st_mtime)] for f in files] + [repos]
    cache_path = model_dir / CACHE
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    hit = cache.get(kind) or {}
    if hit.get("key") == key:
        return hit.get("stats")
    import judge
    recs = _records(model_dir, kind)
    if not recs:
        return None
    count, repo = tokenizer(repos, artifacts)
    lengths, ran_out, sources = [], 0, {"server": 0, "tokenizer": 0, "estimate": 0}
    drafted = accepted = 0                    # 12f.3 addendum: MTP's, when the server says
    for rec in recs:
        d = rec.get("draft") or {}
        if isinstance(d.get("n"), int) and d["n"] > 0:
            drafted += d["n"]
            accepted += int(d.get("accepted") or 0)
        t = rec.get("tokens")
        if isinstance(t, int) and t >= 0:
            lengths.append(t)
            sources["server"] += 1
        else:
            raw = judge._answer(rec)
            if count:
                lengths.append(count(raw))
                sources["tokenizer"] += 1
            else:
                lengths.append(round(len(raw) / CHARS_PER_TOKEN))
                sources["estimate"] += 1
        if judge.answer_parts(rec)["no_answer"]:
            ran_out += 1
    how = max(sources, key=sources.get)
    out = {"median": int(statistics.median(lengths)), "ran_out": ran_out, "n": len(lengths),
           "how": how, "tokenizer": repo if how == "tokenizer" else "",
           "draft": {"n": drafted, "accepted": accepted, "rate": accepted / drafted}
           if drafted else None}
    cache[kind] = {"key": key, "stats": out}
    try:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    except OSError:
        pass
    return out


def how_words(s: dict) -> str:
    """what a length was counted with, in a line"""
    return {"server": "tokens as the server counted them",
            "tokenizer": f"tokens counted with {s.get('tokenizer') or 'its'} tokenizer",
            "estimate": "estimated: a token for every four characters"}.get(s.get("how"), "")
