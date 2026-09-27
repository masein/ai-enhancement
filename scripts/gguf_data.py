"""12f.3: the GGUF benchmarks' datasets — lm_eval's own questions, written in
llama-perplexity's formats, so the "Measured on the GGUF" columns ask exactly
the questions the lm_eval columns do.

Run it once, inside the board's container (it has lm_eval and the Hub's
dataset cache):

    sudo docker compose exec -T bench python scripts/gguf_data.py \\
      --out /home/masein/benchmarks/results/gguf_data

Each benchmark is lm_eval's task, its eval documents after its own
process_docs (HellaSwag's cleaning is lm_eval's, as llama-perplexity's source
says its file must be), written as:
- **HellaSwag**: six lines a task — the context, the right ending's index, the
  four endings;
- **Winogrande**: a CSV row a task, "index,sentence,choice 1,choice 2,answer";
  only the sentence may be quoted, so a question the reader can't take back
  exactly (a double quote in it, a comma in a choice) is left out and counted.
  The file ends with a blank line: `-f` drops the last newline, and the reader
  then drops a last line with none;
- **MMLU, ARC-Challenge, ARC-Easy, TruthfulQA (MC1)**: llama-perplexity's
  multiple-choice binary (below).

The files, their sha256 and counts go in manifest.json beside them. A changed
file is a new dataset version, as for the Everyday bank: results record the
sha256 they were measured on.

The multiple-choice binary, as perplexity.cpp's multiple_choice_score reads
it (little-endian):
    uint32 n_tasks
    uint32 offset[n_tasks]            each task's byte offset from the start
    then per task:
      string question                 uint32 length, then the UTF-8 bytes
      block mc1: uint32 n; n strings; int32 labels[n]   (1 marks the right one)
      block mc2: the same; empty here (llama-perplexity doesn't score it)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_bench as gb  # noqa: E402


# ---------------------------------------------------------------------------
# the multiple-choice binary
# ---------------------------------------------------------------------------

def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _block(answers: list[str], labels: list[int]) -> bytes:
    return (struct.pack("<I", len(answers)) + b"".join(_string(a) for a in answers)
            + struct.pack(f"<{len(labels)}i", *labels))


def mc_binary(tasks: list[dict]) -> bytes:
    """tasks: [{question, answers, labels}] -> the file's bytes"""
    bodies = [_string(t["question"]) + _block(t["answers"], t["labels"]) + _block([], [])
              for t in tasks]
    head = 4 + 4 * len(bodies)
    offsets, at = [], head
    for b in bodies:
        offsets.append(at)
        at += len(b)
    return struct.pack("<I", len(bodies)) + struct.pack(f"<{len(bodies)}I", *offsets) \
        + b"".join(bodies)


def read_mc_binary(data: bytes) -> list[dict]:
    """the file back, as multiple_choice_score deserialises it — and each
    task read again from its own offset, as a subset run seeks to it"""
    def string(at):
        (n,) = struct.unpack_from("<I", data, at)
        return data[at + 4:at + 4 + n].decode("utf-8"), at + 4 + n

    def block(at):
        (n,) = struct.unpack_from("<I", data, at)
        if n > 100:
            raise ValueError("more than 100 answers: llama-perplexity refuses it")
        at += 4
        answers = []
        for _ in range(n):
            a, at = string(at)
            answers.append(a)
        labels = list(struct.unpack_from(f"<{n}i", data, at))
        return answers, labels, at + 4 * n

    (n_task,) = struct.unpack_from("<I", data, 0)
    offsets = struct.unpack_from(f"<{n_task}I", data, 4)
    out, at = [], 4 + 4 * n_task
    for i in range(n_task):
        if offsets[i] != at:
            raise ValueError(f"task {i} starts at {at}, its offset says {offsets[i]}")
        q, at = string(at)
        a1, l1, at = block(at)
        a2, l2, at = block(at)
        out.append({"question": q, "answers": a1, "labels": l1, "mc2": a2})
    if at != len(data):
        raise ValueError(f"{len(data) - at} bytes after the last task")
    return out


# ---------------------------------------------------------------------------
# lm_eval's documents, as llama-perplexity's tasks
# ---------------------------------------------------------------------------

def _one_hot(n: int, i: int) -> list[int]:
    return [1 if k == i else 0 for k in range(n)]


def _flat(text) -> str:
    return " ".join(str(text).split())


def mmlu_task(doc: dict) -> dict | None:
    answers = [_flat(c) for c in doc["choices"]]
    if not all(answers):
        return None                         # llama-perplexity refuses an empty answer
    return {"question": _flat(doc["question"]), "answers": answers,
            "labels": _one_hot(len(answers), int(doc["answer"]))}


def arc_task(doc: dict) -> dict | None:
    answers = [_flat(c) for c in doc["choices"]["text"]]
    if not all(answers):
        return None
    return {"question": _flat(doc["question"]), "answers": answers,
            "labels": _one_hot(len(answers), doc["choices"]["label"].index(doc["answerKey"]))}


def truthfulqa_task(doc: dict) -> dict | None:
    t = doc["mc1_targets"]
    answers = [_flat(c) for c in t["choices"]]
    if not all(answers):
        return None
    return {"question": _flat(doc["question"]), "answers": answers,
            "labels": [int(x) for x in t["labels"]]}


MC = {"mmlu": mmlu_task, "arc_challenge": arc_task, "arc_easy": arc_task,
      "truthfulqa": truthfulqa_task}


def hellaswag_text(docs: list[dict]) -> tuple[str, int, int]:
    """lm_eval's processed docs ({query, choices, gold}) as six lines a task:
    (text, written, left out)"""
    lines, skipped = [], 0
    for d in docs:
        ends = [_flat(e) for e in d["choices"]]
        if len(ends) != 4 or not _flat(d["query"]):
            skipped += 1
            continue
        lines += [_flat(d["query"]), str(int(d["gold"])), *ends]
    return "".join(x + "\n" for x in lines), len(lines) // 6, skipped


def winogrande_text(docs: list[dict]) -> tuple[str, int, int]:
    """sentence, option1, option2, answer (1|2) as the reader parses them:
    (text, written, left out)"""
    rows, skipped = [], 0
    for i, d in enumerate(docs):
        s, o1, o2 = _flat(d["sentence"]), _flat(d["option1"]), _flat(d["option2"])
        ans = str(d["answer"]).strip()
        if ('"' in s or "_" not in s or "," in o1 or "," in o2 or '"' in o1 + o2
                or ans not in ("1", "2") or not o1 or not o2):
            skipped += 1
            continue
        rows.append(f"{i},{chr(34) + s + chr(34) if ',' in s else s},{o1},{o2},{ans}")
    # the reader drops a last line with no newline after it, and -f takes one off
    return "".join(r + "\n" for r in rows) + "\n", len(rows), skipped


def read_winogrande(text: str) -> list[dict]:
    """the file back, as load_winogrande_from_csv parses it (after -f's trim)"""
    if text.endswith("\n"):
        text = text[:-1]                          # what -f does
    # its getline loop stops at eof before using what it read: a line counts
    # only when a newline ends it
    lines = text.split("\n")[:-1]
    out = []
    for line in lines:
        commas, quote = [], False
        for i, ch in enumerate(line):
            if not quote:
                if ch == ",":
                    commas.append(i)
                    if len(commas) == 4:
                        break
                elif ch == '"':
                    quote = True
            elif ch == '"':
                quote = False
        if len(commas) != 4:
            continue
        c0, c1, c2, c3 = commas
        sent = line[c0 + 2:c1 - 1] if line[c0 + 1] == '"' else line[c0 + 1:c1]
        out.append({"sentence": sent, "option1": line[c1 + 1:c2], "option2": line[c2 + 1:c3],
                    "answer": line[c3 + 1:]})
    return out


# ---------------------------------------------------------------------------
# lm_eval's own documents
# ---------------------------------------------------------------------------

def lm_eval_docs(task_name: str) -> list[dict]:
    """the documents lm_eval evaluates for a task (a group's subtasks in
    order), after its process_docs"""
    from lm_eval.tasks import TaskManager, get_task_dict

    def flat(d):
        for v in d.values():
            if isinstance(v, dict):
                yield from flat(v)
            else:
                yield v
    tasks = sorted(flat(get_task_dict([task_name], TaskManager())),
                   key=lambda t: t.config.task)
    docs = []
    for t in tasks:
        docs += list(t.eval_docs)
    return docs


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(out: Path, only: list[str] | None = None, docs_of=lm_eval_docs) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {"benchmarks": {}}
    try:
        import lm_eval
        version = getattr(lm_eval, "__version__", "")
    except ImportError:
        version = ""
    for key in only or gb.ORDER:
        info = gb.BENCHMARKS[key]
        docs = docs_of(info["lm_eval"])
        mode = info["mode"]
        if mode == "hellaswag":
            text, n, skipped = hellaswag_text(docs)
            (out / info["data"]).write_text(text, encoding="utf-8")
        elif mode == "winogrande":
            text, n, skipped = winogrande_text(docs)
            (out / info["data"]).write_text(text, encoding="utf-8")
        else:
            tasks = [MC[key](d) for d in docs]
            kept = [t for t in tasks if t]
            n, skipped = len(kept), len(tasks) - len(kept)
            (out / info["data"]).write_bytes(mc_binary(kept))
        manifest["benchmarks"][key] = {
            "file": info["data"], "sha256": sha256(out / info["data"]), "n": n,
            "skipped": skipped, "of": len(docs),
            "source": f"lm_eval {version} {info['lm_eval']} ({info['split']})".strip(),
            "made_at": time.time()}
        print(f"{info['label']}: {n} of {len(docs)} questions"
              + (f" ({skipped} left out: the format can't hold them exactly)" if skipped else "")
              + f" -> {out / info['data']}")
    manifest["made_at"] = time.time()
    mpath.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="lm_eval's questions in llama-perplexity's formats")
    ap.add_argument("--out", help="the results folder's gguf_data (default: the board's)")
    ap.add_argument("--only", default="", help="comma-separated benchmarks: " + ",".join(gb.ORDER))
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else None
    if out is None:
        from service import config
        out = config.RESULTS_ROOT / "gguf_data"
    only = [x.strip() for x in a.only.split(",") if x.strip()] or None
    unknown = [x for x in only or [] if x not in gb.BENCHMARKS]
    if unknown:
        ap.error(f"unknown: {', '.join(unknown)}")
    build(out, only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
