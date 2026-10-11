#!/usr/bin/env python3
"""20: a box's Frontier bundles scored where they are — on the box, with no
board and no paid API — by the board's own scorers and reading
(service/frontier.marks, scripts/frontier.score and summary): the same
numbers the board writes, before anything only the board does.

    python scripts/score_bundle.py /workspace/<build>/L13-1/frontier-….tar.gz

For each benchmark scored by code — GPQA Diamond, OTIS Mock AIME, MATH
Level 5, MMLU-Pro, ARC-AGI-2 — its % right ± its standard error, its answers,
how many ran out of room and how many of those end in a loop (17e's rule:
the last passage repeating). MATH and OTIS are the code's score: the board's
number before Epoch's model check, which runs there later and can only turn
a wrong answer right. Humanity's Last Exam and SimpleQA Verified are graded
on the board later, with masein's yes: their answers, ran out and loops only.

Several bundles at once: the shards of one model's benchmark are scored
together (with the shards present said: "shards 1, 3 of 9"), and the same
benchmark of two or three models side by side, with the difference in right
answers question by question where both answered the same questions.
--from hf://<org>/<repo>/<path> fetches every bundle under that path of a
private repository first (HF_TOKEN typed on the box), into a folder of its
own.

It reads; it writes nothing but a temporary folder of its own, removed when
it ends — never into the board's folders. The questions come from --questions
(the box's step folder's own copy by default), fetched there once if they
aren't (a gated set needs HF_TOKEN).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import frontier as fb  # noqa: E402
import remote_bundle as rb  # noqa: E402

CODE = ("gpqa_diamond_epoch", "otis_aime_epoch", "math_l5_epoch", "mmlupro_tiger",
        "arc_agi2_public")
GRADED = ("hle_text_cais", "simpleqa_epoch")
_HF = re.compile(r"^hf://(?P<type>datasets/|models/)?(?P<repo>[^/@\s]+/[^/@\s]+)"
                 r"(?:@(?P<rev>[^/\s]+))?(?:/(?P<path>\S*))?$")


ENV_KEYS = ("BENCH_ROOT", "OUT_ROOT", "LOGS", "SERVICE_DB")
CONFIG_KEYS = ("BENCH_ROOT", "RESULTS_ROOT", "OUT_DIR", "LOGS_DIR", "DB_PATH")


def configure(tmp: Path, questions: Path) -> dict:
    """the board's config pointed at this run's own folders, set before the
    service is imported: nothing it might write lands in the board's. What
    was there, to put back (restore)"""
    was = {"env": {k: os.environ.get(k) for k in ENV_KEYS}}
    for k, v in zip(ENV_KEYS, (questions, tmp / "results", tmp / "logs",
                               tmp / "service.sqlite3")):
        os.environ[k] = str(v)
    from service import config
    was["config"] = {k: getattr(config, k, None) for k in CONFIG_KEYS}
    config.BENCH_ROOT = questions
    config.RESULTS_ROOT = tmp / "results"
    config.OUT_DIR = tmp / "results" / "full"
    config.LOGS_DIR = tmp / "logs"
    config.DB_PATH = tmp / "service.sqlite3"
    return was


def restore(was: dict) -> None:
    from service import config
    for k, v in was["config"].items():
        setattr(config, k, v)
    for k, v in was["env"].items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def fetch_from(src: str, into: Path, say) -> list[Path]:
    """every bundle under an hf:// path of a private repository, fetched with
    HF_TOKEN (never printed or kept)"""
    m = _HF.match(src.strip())
    if not m:
        raise SystemExit(f"--from {src}: not hf://<org>/<repo>/<path>")
    from huggingface_hub import snapshot_download
    path = (m["path"] or "").strip("/")
    say(f"fetching the bundles under {path or 'the repository'} of {m['repo']}…")
    try:
        got = snapshot_download(m["repo"], repo_type="dataset" if m["type"] == "datasets/"
                                else "model", revision=m["rev"],
                                allow_patterns=[f"{path}/**/*.tar.gz" if path else "**/*.tar.gz"],
                                token=os.environ.get("HF_TOKEN") or None, local_dir=str(into))
    except Exception as e:                                  # noqa: BLE001 — said in one line
        raise SystemExit(rb.scrub(f"--from {src} could not be fetched: {e}. A private "
                                  "repository needs HF_TOKEN set on this box")) from None
    return sorted(Path(got).rglob("*.tar.gz"))


def read_all(paths: list[Path], say) -> dict:
    """{(model, thinking): {task: {answers: {(id, epoch): line}, shards: {(i, n)}
    | None, whole: bool, setup: bytes, bundles: [name]}}} — a bundle that
    isn't one is said and left out"""
    from service import frontier as sf
    out: dict = {}
    for p in paths:
        try:
            b = rb.read(p)
        except (OSError, ValueError) as e:
            say(f"{p.name}: left out — {e}")
            continue
        bundle = b["bundle"]
        if bundle.get("suite") != "frontier":
            say(f"{p.name}: left out — not a Frontier bundle")
            continue
        try:
            shard = rb.shard_of_bundle(bundle)
        except ValueError as e:
            say(f"{p.name}: left out — {e}")
            continue
        key = (str(bundle.get("model") or ""), bool(bundle.get("thinking")))
        row = str(bundle.get("row") or "")
        # the file its box hashed: what the board's import is given (--file-sha256)
        gg = b["setup"].get("gguf") if isinstance(b["setup"].get("gguf"), dict) else {}
        files = out.setdefault(key, {}).setdefault("_files", {})
        files.setdefault(str(gg.get("sha256") or "?"), gg)
        for t in bundle.get("tasks") or {}:
            if t not in fb.BENCH:
                continue
            name = f"results/{row}/{t}_0shot/{sf.SUB}/{sf.ANSWERS}"
            lines = {}
            for ln in b["files"].get(name, b"").decode("utf-8", "replace").splitlines():
                try:
                    r = json.loads(ln)
                    lines[(str(r["id"]), int(r["epoch"]))] = r
                except (ValueError, KeyError, TypeError):
                    continue
            got = out.setdefault(key, {}).setdefault(t, {
                "answers": {}, "shards": set(), "whole": False, "bundles": [],
                "setup": b["files"].get(f"results/{row}/{t}_0shot/{sf.SUB}/{sf.SETUP}", b"{}"),
                # the questions its step fetched, beside its bundle on the box
                "bench": p.parent / "bench" if (p.parent / "bench").is_dir() else None})
            clash = [k for k in lines if k in got["answers"]
                     and got["answers"][k].get("answer") != lines[k].get("answer")]
            if clash:
                say(f"{p.name}: {fb.BENCH[t]['label']} answers {len(clash):,} questions another "
                    "bundle given here answers differently — the first kept")
            for k, r in lines.items():
                got["answers"].setdefault(k, r)
            if shard:
                got["shards"].add(shard)
            else:
                got["whole"] = True
            got["bundles"].append(p.name)
    return out


def items_of(task: str, got: dict, questions: Path) -> tuple[list[dict], str]:
    """the questions these answers are of — every question for a whole
    benchmark, else the shards present — and the words for them"""
    items = fb.load(task, questions)
    if got["whole"] or not got["shards"]:
        return items, ""
    ns = {n for _, n in got["shards"]}
    if len(ns) > 1:
        raise ValueError("shards of different splits (" + ", ".join(
            f"{i} of {n}" for i, n in sorted(got["shards"])) + ")")
    n = ns.pop()
    have = sorted(i for i, _ in got["shards"])
    mine = [x for i in have for x in fb.shard_of(items, (i, n))]
    order = {it["id"]: k for k, it in enumerate(items)}
    mine.sort(key=lambda x: order[x["id"]])
    if len(have) == n:
        return items, ""
    return mine, (f"shards {', '.join(map(str, have))} of {n}: {len(mine):,} of its "
                  f"{len(items):,} questions")


def score(task: str, got: dict, questions: Path, tmp: Path, model: str) -> dict:
    """the benchmark's numbers, by the board's marks() on a copy of its
    answers in this run's own folder"""
    from service import config
    from service import frontier as sf
    config.BENCH_ROOT = questions
    items, part = items_of(task, got, questions)
    row = tmp / "rows" / re.sub(r"[^A-Za-z0-9._-]", "_", model)
    d = sf.task_dir(row, task)
    d.mkdir(parents=True, exist_ok=True)
    (d / sf.ANSWERS).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                        for r in got["answers"].values()), encoding="utf-8")
    (d / sf.SETUP).write_bytes(got["setup"])
    m = sf.marks(row, task, items=items, grades={})
    flat = [r for it in items for r in m["runs"].get(it["id"], [])]
    cut = [r for r in flat if r["ran_out"]]
    loops = sum(1 for r in cut if fb.ends_in_loop(r["answer"]))
    out = {"answers": len(flat), "missing": len(m["missing"]), "ran_out": len(cut),
           "loops": loops, "part": part, "unanswered": sum(1 for r in flat if r.get("unanswered")),
           "questions": len(items), "per": {}}
    if task in CODE:
        per = {q: [1.0 if r["code_ok"] else 0.0 for r in rs] for q, rs in m["runs"].items()}
        groups = {it["id"]: fb.group_of(task, it) for it in items}
        s = fb.summary(per, task, groups)
        out.update({"score": s["score"], "se": s["se"], "per": per, "groups": groups})
    return out


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def line_of(task: str, model: str, sc: dict) -> str:
    """"GPQA Diamond · served/x: 61.6% ± 3.1 · 792 answers · 12 ran out of room, 3 of them
    ending in a loop" """
    spec = fb.BENCH[task]
    if sc["missing"]:
        what = (f"{sc['missing']:,} answers missing of {sc['answers'] + sc['missing']:,} — no "
                "score until the rest are in")
    elif task in CODE:
        what = f"{pct(sc['score'])} ± {100 * (sc['se'] or 0):.1f}"
    else:
        what = "graded on the board later"
    bits = [f"{spec['label']} · {model}: {what}",
            f"{sc['answers']:,} answers" + (f" ({sc['part']})" if sc["part"] else "")]
    bits.append(f"{sc['ran_out']:,} ran out of room ({pct(sc['ran_out'] / max(1, sc['answers']))})"
                + (f", {sc['loops']:,} of them ending in a loop" if sc["ran_out"] else ""))
    if sc["unanswered"]:
        bits.append(f"{sc['unanswered']:,} written off as no answer, counted wrong")
    if task in ("math_l5_epoch", "otis_aime_epoch") and not sc["missing"]:
        bits.append("the code's score: the board's before Epoch's model check, which can only "
                    "raise it")
    return " · ".join(bits)


def paired(a: dict, b: dict) -> tuple[float, float, int] | None:
    """(b − a, its standard error, questions): the difference in right
    answers question by question, over the questions both answered"""
    qs = [q for q in a["per"] if q in b["per"]]
    if len(qs) < 2:
        return None
    diffs = [sum(b["per"][q]) / len(b["per"][q]) - sum(a["per"][q]) / len(a["per"][q])
             for q in qs]
    n = len(diffs)
    mu = sum(diffs) / n
    var = sum((x - mu) ** 2 for x in diffs) / (n - 1)
    return mu, (var / n) ** 0.5, n


def table(task: str, scores: list[tuple[str, dict]]) -> list[str]:
    """the same benchmark of two or three models, side by side"""
    spec = fb.BENCH[task]
    w = max(len(m) for m, _ in scores)
    out = [f"{spec['label']}, side by side:"]
    for m, sc in scores:
        cell = (f"{pct(sc.get('score'))} ± {100 * (sc.get('se') or 0):4.1f}"
                if task in CODE and not sc["missing"] else "graded later".rjust(13))
        out.append(f"  {m.ljust(w)}  {cell}  {sc['answers']:>6,} answers  "
                   f"ran out {sc['ran_out']:>5,} ({pct(sc['ran_out'] / max(1, sc['answers'])):>6})"
                   f"  in a loop {sc['loops']:>5,} ({pct(sc['loops'] / max(1, sc['answers'])):>6})")
    if task in CODE:
        base_m, base = scores[0]
        for m, sc in scores[1:]:
            d = paired(base, sc) if not base["missing"] and not sc["missing"] else None
            if d:
                out.append(f"  {m} − {base_m}: {100 * d[0]:+.1f} points ± {100 * d[1]:.1f}, "
                           f"question by question over the {d[2]:,} both answered")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundles", nargs="*", type=Path, help="bundles (.tar.gz), or folders of them")
    ap.add_argument("--from", dest="src", default="",
                    help="hf://<org>/<repo>/<path>: every bundle under it, fetched first")
    ap.add_argument("--only", action="append", default=[],
                    help="a benchmark (its short name: gpqa, hle, …), repeatable")
    ap.add_argument("--questions", type=Path, default=None,
                    help="where the questions are (default: each bundle's step folder's "
                         "bench/ on the box, else /workspace/files/questions)")
    a = ap.parse_args(argv)
    shorts = {fb.BENCH[t]["short"]: t for t in fb.TASKS}
    bad = [x for x in a.only if x not in shorts and x not in fb.BENCH]
    if bad:
        ap.error(f"--only: {', '.join(bad)} — one of {', '.join(shorts)}")
    only = {shorts.get(x, x) for x in a.only}
    with tempfile.TemporaryDirectory(prefix="score-bundle-") as td:
        tmp = Path(td)
        paths: list[Path] = []
        for p in a.bundles:
            # a folder's bundles only — a step's bench/ holds other tarballs
            paths += sorted(p.rglob("frontier-*.tar.gz")) if p.is_dir() else [p]
        if a.src:
            paths += fetch_from(a.src, tmp / "fetched", print)
        if not paths:
            ap.error("give bundles, a folder of them, or --from hf://…")
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(f"{missing[0]}: no such bundle")
        q = (a.questions or Path("/workspace/files/questions")).resolve()
        was = configure(tmp, q)
        try:
            by_task = score_all(read_all(paths, print), only, q, tmp, bool(a.questions))
        finally:
            restore(was)
        for t in fb.TASKS:
            if len(by_task.get(t) or []) > 1:
                print()
                print("\n".join(table(t, by_task[t])))
    return 0


def score_all(got: dict, only: set, q: Path, tmp: Path,
              given: bool = False) -> dict[str, list[tuple[str, dict]]]:
    """each model's each benchmark scored and its line printed: {task: [(model,
    numbers)]} — its questions from --questions when `given`, else from its
    step's own copy beside its bundle, else from `q`"""
    by_task: dict[str, list[tuple[str, dict]]] = {}
    for (model, thinking), tasks in sorted(got.items()):
        name = model.split("/", 1)[-1] + ("" if thinking else " (thinking off)")
        for sha, gg in (tasks.get("_files") or {}).items():
            print(f"{name}: its GGUF {gg.get('name') or '?'} · sha256 {sha}, as its box hashed "
                  "it" + (" — the split identity of its parts:" if gg.get("parts") else ""))
            for p in gg.get("parts") or []:
                if isinstance(p, dict):
                    print(f"    {p.get('name')}  {p.get('sha256')}")
        if len(tasks.get("_files") or {}) > 1:
            print(f"{name}: its bundles hold {len(tasks['_files'])} different files — not one "
                  "model's")
        for t in fb.TASKS:
            if t not in tasks or (only and t not in only):
                continue
            try:
                sc = score(t, tasks[t], q if given else (tasks[t]["bench"] or q), tmp, name)
            except Exception as e:                      # noqa: BLE001 — said, the rest scored
                print(rb.scrub(f"{fb.BENCH[t]['label']} · {name}: not scored — {e}"))
                continue
            print(line_of(t, name, sc))
            by_task.setdefault(t, []).append((name, sc))
    return by_task


if __name__ == "__main__":
    raise SystemExit(main())
