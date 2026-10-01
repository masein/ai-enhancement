"""The contamination gate: no generated document may share a 13-gram with any
benchmark item OR any exam question, in either half.

The Pythia/GPT-3 convention. It is the backstop for the day someone bypasses
the airlock — hands a generator the failed questions instead of the skill
spec — and it is cheap: normalise, hash every 13-token window of every
benchmark document and exam question on disk, set-membership per generated
window. Above MAX_DROP_SHARE dropped the whole dataset is refused: a
generator echoing at that rate is not a coincidence, and keeping the 98% that
slipped through would be keeping paraphrases.

The exam matters more than the benchmarks here, because the exam is our own
generated questions and nothing outside protects it: a document that repeats
an exam question teaches the test directly, and the per-topic score it moves
is the number the whole loop steers by.

12g.2: and every Everyday question, both halves — a chat example's request
and reply are checked against the bank's requests and good answers, and a
request that IS one of the bank's, however short, is a copy.

14.1: and every pinned benchmark file the board asks from — MobileAIBench's
samples, their questions, contexts and reference answers — indexed before any
model has answered them (a run's own files hold the prompts, never the
references).

Both halves of both are indexed on purpose. The report half is never SHOWN to
anyone or anything; it is still the thing the published score comes from, so
a generated document that collides with it is exactly the leak this exists to
stop.
"""

from __future__ import annotations

import re
from pathlib import Path

NGRAM = 13
EVERYDAY_BANK = Path(__file__).resolve().parent.parent / "eval_tasks" / "everyday" / "bank.jsonl"
# 14.1: MobileAIBench's pinned samples; not its judge's prompt templates
MAB_DIR = Path(__file__).resolve().parent.parent / "eval_tasks" / "mobileaibench"
NOT_ITEMS = {"mt_bench_judge_prompts"}
# 14.3: Mobile-MMLU-Pro's manifest; its file lives only in the server's data folder
MMP_DIR = Path(__file__).resolve().parent.parent / "eval_tasks" / "mobile_mmlu_pro"
MAX_DROP_SHARE = 0.02
NEAR_DUP_SHINGLE = 5
NEAR_DUP_JACCARD = 0.8

_WORD = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> list[str]:
    """lower-case, strip punctuation, collapse whitespace → tokens."""
    return _WS.sub(" ", _WORD.sub(" ", str(text).lower())).strip().split()


def windows(tokens: list[str], n: int = NGRAM):
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i:i + n])


def doc_strings(doc):
    """Every string in a harness doc, recursively: question, choices, passages."""
    if isinstance(doc, str):
        yield doc
    elif isinstance(doc, dict):
        for v in doc.values():
            yield from doc_strings(v)
    elif isinstance(doc, (list, tuple)):
        for v in doc:
            yield from doc_strings(v)


def item_text(item: dict) -> str:
    """Everything a generated item says, whatever its format — a document's
    title and body, or a question-shaped item's parts."""
    parts = [str(item.get("title") or ""), str(item.get("text") or ""),
             str(item.get("question") or ""),
             # 12g.2: a chat example's request and reply
             str(item.get("user") or ""), str(item.get("assistant") or "")]
    for c in item.get("choices") or []:
        parts.append(str(c))
    parts.append(str(item.get("answer") or ""))
    parts.append(str(item.get("rationale") or ""))
    return "\n".join(p for p in parts if p)


class BenchmarkIndex:
    """Every 13-gram of every benchmark document under results/full and every
    exam question in the bank, keyed by both trees' shape so a new task, a
    re-run or an accepted question rebuilds it."""

    def __init__(self, root: Path, exam_root: Path | None = None):
        self.root = Path(root)
        self.exam_root = Path(exam_root) if exam_root else None
        self.key: tuple = ()
        self.grams: set[int] = set()
        self.exam_grams: set[int] = set()
        self.everyday_grams: set[int] = set()
        self.everyday_exact: set[str] = set()
        self.n_docs = 0
        self.n_files = 0
        self.n_exam = 0
        self.n_everyday = 0
        self.n_pinned = 0

    def _bank_files(self) -> list[Path]:
        d = (self.exam_root / "bank") if self.exam_root else None
        return sorted(d.glob("*.jsonl")) if d and d.is_dir() else []

    @staticmethod
    def _everyday_files() -> list[Path]:
        """the repo's Everyday bank, (12i.2) what the question builder
        published beside the data, and (12p.2) the hidden half, which lives
        only on the data volume now — a training example copying it is still
        dropped"""
        from . import config
        d = config.BENCH_ROOT / "everyday"
        return [p for p in (EVERYDAY_BANK, d / "built.jsonl", d / "hidden.jsonl") if p.exists()]

    @staticmethod
    def pinned_files() -> list[Path]:
        """14.1: the pinned benchmark files the board asks from — every row's
        every string is a benchmark item (MobileAIBench's manifest)"""
        import json

        from . import config
        try:
            m = json.loads((MAB_DIR / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        # 14.2: and what lives only on the server (Privacy Leakage), once fetched
        where = lambda f: (MAB_DIR if f.get("committed", True)  # noqa: E731
                           else Path(config.MAB_PRIVATE_DIR)) / f["file"]
        out = [where(f) for k, f in sorted((m.get("files") or {}).items())
               if k not in NOT_ITEMS and where(f).exists()]
        # 14.3: Mobile-MMLU-Pro's questions and options, once the data step fetched it
        try:
            mm = json.loads((MMP_DIR / "manifest.json").read_text(encoding="utf-8"))
            p = Path(config.MMP_DIR) / mm["file"]["file"]
            if p.exists():
                out.append(p)
        except (OSError, ValueError, KeyError):
            pass
        return out

    @staticmethod
    def pinned_rows(path: Path):
        """a pinned file's rows: a CSV's, or a JSON-lines file's"""
        import csv
        import io
        import json
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".csv":
            csv.field_size_limit(10 ** 9)
            yield from csv.DictReader(io.StringIO(text))
            return
        for line in text.splitlines():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    def _key(self) -> tuple:
        files = list(self.root.rglob("samples_*.jsonl")) if self.root.is_dir() else []
        bank = self._bank_files()
        evd = tuple(p.stat().st_mtime for p in self._everyday_files())
        pinned = tuple((str(p), p.stat().st_mtime, p.stat().st_size) for p in self.pinned_files())
        return (len(files), max((f.stat().st_mtime for f in files), default=0.0),
                len(bank), max((f.stat().st_mtime for f in bank), default=0.0), evd, pinned)

    def refresh(self) -> "BenchmarkIndex":
        import json
        key = self._key()
        if key == self.key and (self.grams or self.exam_grams):
            return self
        grams: set[int] = set()
        n_docs = n_files = 0
        seen_hash: set[str] = set()
        for f in sorted(self.root.rglob("samples_*.jsonl")):
            n_files += 1
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    dh = rec.get("doc_hash")
                    if dh and dh in seen_hash:          # thirty models, one benchmark
                        continue
                    if dh:
                        seen_hash.add(dh)
                    n_docs += 1
                    for s in doc_strings(rec.get("doc") or {}):
                        for w in windows(normalize(s)):
                            grams.add(hash(w))
        exam: set[int] = set()
        n_exam = 0
        for f in self._bank_files():
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_exam += 1
                for w in windows(normalize(rec.get("prompt") or "")):
                    exam.add(hash(w))
        # 12g.2: the Everyday bank, both halves: its requests and good answers
        evd: set[int] = set()
        exact: set[str] = set()
        n_evd = 0
        for path in self._everyday_files():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_evd += 1
                exact.add(" ".join(normalize(rec.get("prompt") or "")))
                for s in (rec.get("prompt") or "", rec.get("reference") or ""):
                    for w in windows(normalize(s)):
                        evd.add(hash(w))
        # 14.1: the pinned benchmark files, every string of every row
        n_pinned = 0
        for path in self.pinned_files():
            for row in self.pinned_rows(path):
                n_pinned += 1
                for s in doc_strings(row):
                    for w in windows(normalize(s)):
                        grams.add(hash(w))
        self.n_pinned = n_pinned
        self.key, self.grams, self.exam_grams = key, grams, exam
        self.everyday_grams, self.everyday_exact = evd, exact - {""}
        self.n_docs, self.n_files, self.n_exam, self.n_everyday = n_docs, n_files, n_exam, n_evd
        return self

    def hits(self, text: str) -> list[str]:
        both = self.grams | self.exam_grams | self.everyday_grams
        return [w for w in windows(normalize(text)) if hash(w) in both]

    def copies_request(self, item: dict) -> str:
        """12g.2: a chat example whose request is one of the Everyday bank's,
        however short — below 13 words no window can see it"""
        u = " ".join(normalize(item.get("user") or ""))
        return u if u and u in self.everyday_exact else ""

    def source_of(self, gram: str) -> str:
        """Which corpus a matched n-gram came from — the exam is named
        separately because a document echoing it is the worse failure, and
        the Everyday bank because it is ours too (12g.2)."""
        h = hash(gram)
        return "exam" if h in self.exam_grams else "everyday" if h in self.everyday_grams \
            else "benchmark"


_INDEX: dict[str, BenchmarkIndex] = {}


def index(root: Path, exam_root: Path | None = None) -> BenchmarkIndex:
    root = Path(root)
    k = f"{root}|{exam_root or ''}"
    ix = _INDEX.get(k)
    if ix is None:
        ix = _INDEX[k] = BenchmarkIndex(root, exam_root)
    return ix.refresh()


def _shingles(text: str) -> set[int]:
    toks = normalize(text)
    if len(toks) <= NEAR_DUP_SHINGLE:
        return {hash(" ".join(toks))}
    return {hash(w) for w in windows(toks, NEAR_DUP_SHINGLE)}


def check(items: list[dict], ix: BenchmarkIndex) -> dict:
    """Drop every item sharing a 13-gram with the benchmark, then every
    near-duplicate of an item already kept. Returns kept items and a report
    that provenance.json records verbatim."""
    kept: list[dict] = []
    dropped: list[dict] = []
    offending: list[str] = []
    shingles: list[set[int]] = []
    n_exam = n_everyday = 0
    for i, item in enumerate(items):
        text = item_text(item)
        hits = ix.hits(text)
        copy = ix.copies_request(item)
        if copy and not hits:
            n_everyday += 1
            dropped.append({"index": i, "reason": "benchmark", "source": "everyday",
                            "ngram": copy})
            if len(offending) < 5:
                offending.append(copy)
            continue
        if hits:
            src = ix.source_of(hits[0])
            n_exam += src == "exam"
            n_everyday += src == "everyday"
            dropped.append({"index": i, "reason": "benchmark", "source": src, "ngram": hits[0]})
            if len(offending) < 5:
                offending.append(hits[0])
            continue
        sh = _shingles(text)
        dup = None
        for j, other in enumerate(shingles):
            inter = len(sh & other)
            if inter and inter / len(sh | other) >= NEAR_DUP_JACCARD:
                dup = j
                break
        if dup is not None:
            dropped.append({"index": i, "reason": "duplicate", "of": dup})
            continue
        shingles.append(sh)
        kept.append(item)
    n_in = len(items)
    n_bench = sum(1 for d in dropped if d["reason"] == "benchmark")
    n_dup = len(dropped) - n_bench
    share = (n_bench / n_in) if n_in else 0.0
    return {
        "kept": kept,
        "dropped": dropped,
        "report": {
            "ngram": NGRAM, "items_in": n_in, "items_kept": len(kept),
            "dropped_benchmark": n_bench, "dropped_exam": n_exam,
            "dropped_everyday": n_everyday,
            "dropped_duplicate": n_dup,
            "share_dropped_benchmark": round(share, 4), "max_share": MAX_DROP_SHARE,
            "rejected": bool(n_in) and share > MAX_DROP_SHARE,
            "offending_ngrams": offending,
            "benchmark_docs": ix.n_docs, "benchmark_files": ix.n_files,
            "exam_questions": ix.n_exam, "everyday_questions": ix.n_everyday,
        },
    }
