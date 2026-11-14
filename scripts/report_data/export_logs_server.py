"""logs/ for the public report repository: every IFEval and MMLU-Pro answer of the
Masein model, in both modes where run. Run in the bench container, this file on
stdin, from ~/benchmarks/aienh:

    sudo docker compose exec -T bench python - < ~/export-logs-server.py

(In the repository this file is scripts/report_data/export_logs_server.py; copy it to the
server as ~/export-logs-server.py.)

Reads the stored answers; writes $BENCH_ROOT/report-data/logs/ and samples/. Each line
is built field by field from the list below: the thinking is never written, and an
answer that still holds a thinking tag stops the export. MMLU-Pro's final answers keep
their last 400 characters only; IFEval's stay whole. It rewrites every file, from the
same answers.

Prints, and keeps in samples/report.txt: each file's counts beside the report's, one
sample line with its prompt shortened, and a scan of every field for the internal terms.
Our own fields must come back empty. Prompts and answers are ordinary text, so their
hits are counted by term, with examples of the rare ones. MMLU-Pro's files also say how
many of those last 400 characters read the letter the whole answer read. samples/ also
holds each file's first line whole."""
import json
import re
import sys
from collections import Counter

sys.path[:0] = ["scripts", "."]
from service import config, reports  # noqa: E402
import frontier as fb  # noqa: E402

OUT = config.BENCH_ROOT / "report-data" / "logs"
# brief 33 (masein, 7 Nov): the paper's model is step2388 — thinking on its k6
# path (limit 98,304), off its k3 path (32,768), every answer at its final limit
ME = "served/step2388-k3k6-2611"
ROWS = {"on": ME + " · thinking", "off": ME}
FILES = {"mmlupro_tiger": "mmlu-pro", "ifeval_google": "ifeval"}
LABEL = {"mmlupro_tiger": "MMLU-Pro", "ifeval_google": "IFEval"}
TAG = re.compile(r"(?i)</?think")
SAMPLES = OUT.parent / "samples"
TAIL = 400                                 # MMLU-Pro: the end of each final answer
# what the board says (7 Nov), to tie each file out: answers, right, ran out, looped —
# the paper's scores (round 17's Table 5) and ran-out counts (Tables 5 and 8) print the same
WANT = {"ifeval-thinking-on": (541, 490, 7, 8), "ifeval-thinking-off": (541, 437, 2, 3),
        "mmlu-pro-thinking-on": (12032, 9983, 0, 0), "mmlu-pro-thinking-off": (12032, 9638, 1, 1)}
TERMS = {"checkpoint name": r"(?i)\bstep[-_]?\d{3,}\b", "step": r"(?i)\bstep\b",
         "Q2": r"(?<![A-Za-z0-9])Q2(?![A-Za-z0-9])", "Q4": r"(?<![A-Za-z0-9])Q4(?![A-Za-z0-9])",
         "BF16": r"(?i)\bbf16\b", "phone build": r"(?i)phone[ _-]build", "k4": r"(?i)\bk4\b",
         "lda": r"(?i)\blda\b", "LOOKAHEAD": r"(?i)lookahead", "evalboard": r"(?i)evalboard",
         "k3k6": r"(?i)\bk3k6\b|\bk[36]\b",
         "board": r"(?i)\bboard\b",
         "box names": r"(?i)\bdgx-?0?\d\b|\bteraformer-5090|\bvast\.ai\b",
         # r25 (14 Nov): runtime details stay in internal docs, as scan_terms.py's
         "runtime switches": r"LLAMA_MOE_ROUTE_\w+|GGML_CPU_DISABLE_FUSION|LOOKAHEAD_MOE_ROUTER_NORM",
         "environment variables": r"\b(?!(?:I?Q\d|PQ\d|TQ\d|PTQ\d)\w*)[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b",
         "server argv": r"\bllama-server\b|(?<![\w-])--(?:jinja|metrics|no-mmap|n-cpu-moe|flash-attn|"
                        r"ctx-size|cache-type-[kv]|parallel|n-gpu-layers|host|port|threads|spec-type|"
                        r"reasoning-budget|reasoning-format|chat-template(?:-file|-kwargs)?)\b|"
                        r"(?<![\w-])-(?:ngl|ctk|ctv|np|fa)\b",
         "served model ids": r"\bserved(?:/|__)[A-Za-z0-9][\w.-]*",
         "IP addresses": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
         "paths": r"/home/[a-z]+|/Users/[a-z]+|/workspace/"}
# people's names are scanned on the Mac, from a list kept out of the repository
# (scan_terms.py --private), once the files are copied there
TERMS = {k: re.compile(v) for k, v in TERMS.items()}
TEXT = ("prompt", "final_answer")          # text from the benchmark and the model
TIER = 25                                  # a term this rare gets its examples shown
LINES: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    LINES.append(text)


def grade(task: str, run: dict, need: dict) -> dict:
    if task == "mmlupro_tiger":
        return {"answer_read": run.get("read"), "right_answer": need["key"],
                "right": bool(run.get("ok"))}
    f = run.get("follow") or {}
    return {"passed_strict": bool(run.get("ok")),
            "instructions": [{"id": i, "strict": bool(s), "loose": bool(lo)}
                             for i, s, lo in zip(f.get("ids") or [], f.get("strict") or [],
                                                 f.get("loose") or [])]}


def line(task: str, mode: str, qid: str, item: dict, row: dict) -> dict:
    run = row["frontier"][0]
    text = row["resps"][0][0] or ""
    out = bool(run.get("ran_out"))
    final = fb.visible(text)
    if TAG.search(final):
        raise SystemExit(f"{task} {qid}: thinking tag in the final answer, nothing more written")
    prompt, need = fb.prompt(task, item)
    return {"benchmark": LABEL[task], "question_id": qid, "thinking": mode, "prompt": prompt,
            "final_answer": final[-TAIL:] if task == "mmlupro_tiger" else final,
            "grade": grade(task, run, need), "output_tokens": run.get("tokens"),
            "ran_out_of_room": out, "looped": fb.ended_in_loop(text, out)}


def tail_check(rows: list[dict]) -> None:
    """the last 400 characters against the whole answer: TIGER-Lab's reader takes the first
    "answer is (X)", so an answer that says it twice can read another letter from its end"""
    done = [r for r in rows if not r["ran_out_of_room"]]
    said = sum(1 for r in done if re.search(r"answer is \(?[A-J]\)?", r["final_answer"].replace("**", "")))
    same = [r for r in done if fb.read_mmlu_pro(r["final_answer"]) == r["grade"]["answer_read"]]
    say(f"  last {TAIL} characters: \"answer is (X)\" in {said:,} of {len(done):,} finished answers · the "
        f"same letter as the whole answer in {len(same):,} · another in {len(done) - len(same):,}")


def scan(name: str, rows: list[dict]) -> bool:
    """every field for every term; False when one of our own fields has a hit"""
    ours, text, seen = Counter(), Counter(), {}
    for r in rows:
        for field, value in r.items():
            words = value if field in TEXT else json.dumps(value, ensure_ascii=False)
            # a benchmark's name and an IFEval instruction's id are the dataset's own words
            if field not in TEXT and field in ("benchmark", "prompt"):
                continue
            for term, pat in TERMS.items():
                hits = list(pat.finditer(str(words)))
                if not hits:
                    continue
                if field in TEXT:
                    text[(field, term)] += len(hits)
                    h = hits[0]
                    seen.setdefault((field, term), []).append(
                        (r["question_id"], str(words)[max(0, h.start() - 50):h.end() + 50]))
                else:
                    ours[(field, term)] += len(hits)
    say(f"  scan, our own fields: {dict(ours) or 'empty'}")
    for field in TEXT:
        say(f"  scan, {field}: " + (", ".join(f"{t} {n:,}" for (f, t), n in sorted(text.items())
                                              if f == field) or "empty"))
    for (field, term), n in sorted(text.items()):
        if n <= TIER:
            for qid, bit in seen[(field, term)][:3]:
                say(f"    {term} in {field}, question {qid}: ...{bit!r}...".replace("\\n", " "))
    return not ours


def main() -> int:
    fb.set_root(config.BENCH_ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    clean = True
    for task, name in FILES.items():
        items = {str(x["id"]): x for x in fb.load(task, config.BENCH_ROOT)}
        for mode, row_id in ROWS.items():
            f = reports.samples_file(reports.row_dir(row_id), task)
            stem = f"{name}-thinking-{mode}"
            if f is None:
                say(f"{stem}: not imported yet, no file")
                continue
            dest, rows = OUT / f"{stem}.jsonl", []
            with open(dest, "w", encoding="utf-8") as fh:
                for _, row in reports.jsonl(f):
                    qid = str((row.get("doc") or {}).get("id"))
                    rec = line(task, mode, qid, items[qid], row)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    rows.append(rec)
            right = sum(1 for r in rows if r["grade"].get("right", r["grade"].get("passed_strict")))
            out, looped = (sum(1 for r in rows if r[k]) for k in ("ran_out_of_room", "looped"))
            blank = sum(1 for r in rows if not r["final_answer"].strip())
            got = (len(rows), right, out, looped)
            want = WANT.get(stem)
            say(f"{dest.name}: {dest.stat().st_size / 1e6:.1f} MB · {got[0]:,} answers, {right:,} "
                f"right, {out} ran out of room, {looped} looped, {blank} with an empty answer · "
                + ("agrees with the report" if got == want else
                   f"THE REPORT SAYS {want}" if want else "no report figure yet"))
            (SAMPLES / f"{stem}.json").write_text(json.dumps(rows[0], ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
            short = {**rows[0], "prompt": rows[0]["prompt"][:160] + "..."
                     + f" ({len(rows[0]['prompt']):,} characters)"}
            short["final_answer"] = rows[0]["final_answer"][:400]
            say("  sample: " + json.dumps(short, ensure_ascii=False))
            if task == "mmlupro_tiger":
                tail_check(rows)
            clean = scan(stem, rows) and clean
    say("our own fields are clean" if clean else "A HIT IN OUR OWN FIELDS: don't upload")
    (SAMPLES / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
