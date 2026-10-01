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
  multiple-choice binary (below). 12f.5: MMLU as lm_eval's mmlu asks it —
  "The following are multiple choice questions (with answers) about
  {subject}.", the question, the options lettered A. to D., "Answer:" — with
  the letters as its answers (mmlu_task). The others are each option's text,
  as lm_eval scores them — 12y: after lm_eval's own prompt, not the bare
  question: ARC's "Question: …\nAnswer:", TruthfulQA's six questions and
  answers before "Q: …\nA:". Asked bare, Qwen3.6-35B-A3B scored 43.8 on
  ARC-Challenge and 61.7 on ARC-Easy.

The files, their sha256 and counts go in manifest.json beside them, and for a
multiple-choice file the most answers a task has, its biggest task's tokens
and whether its answers are letters (gguf_bench.mc_shape): the worker's -np
and -c. A build fills those in for the files it didn't rebuild too, so
`--only mmlu` leaves every entry with them. A changed file is a new dataset
version, as for the Everyday bank: results record the sha256 they were
measured on.

14.3: Mobile-MMLU-Pro is built apart, once our answer key is (and again
when the key moves): `--only mobile_mmlu_pro`.

12f.5, once after deploying: MMLU again, lettered (the others' files don't
change, so their results stay current):

    sudo docker compose exec -T bench python scripts/gguf_data.py \\
      --out /home/masein/benchmarks/results/gguf_data --only mmlu

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
import random
import re
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


# the file back, as multiple_choice_score reads it: the worker reads it too
read_mc_binary = gb.read_mc


# ---------------------------------------------------------------------------
# lm_eval's documents, as llama-perplexity's tasks
# ---------------------------------------------------------------------------

def _one_hot(n: int, i: int) -> list[int]:
    return [1 if k == i else 0 for k in range(n)]


def _flat(text) -> str:
    return " ".join(str(text).split())


MMLU_HEAD = "The following are multiple choice questions (with answers) about {subject}.\n\n"


def mmlu_task(doc: dict) -> dict | None:
    """12f.5: as lm_eval's mmlu asks it, 0-shot — its description (the subject,
    underscores as spaces), then doc_to_text: the question stripped, each
    option as lm_eval writes it after its letter, and "Answer:". The answers
    are the letters: llama-perplexity scores question + " " + answer, so the
    letter it scores is " A", lm_eval's target delimiter and choice. An empty
    option is still a letter, so no question is left out"""
    choices = [str(c) for c in doc["choices"]]
    if not 2 <= len(choices) <= len(gb.LETTERS):
        return None
    letters = list(gb.LETTERS[:len(choices)])
    subject = str(doc.get("subject") or "").replace("_", " ").strip()
    text = (MMLU_HEAD.format(subject=subject) + str(doc["question"]).strip() + "\n"
            + "".join(f"{k}. {c}\n" for k, c in zip(letters, choices)) + "Answer:")
    return {"question": text, "answers": letters,
            "labels": _one_hot(len(choices), int(doc["answer"]))}


# 12y: lm_eval 0.4.12's prompts, character for character (tests/test_image_deps.py
# renders the installed harness's own templates and compares). llama-perplexity
# scores question + " " + answer: lm_eval's target delimiter, then the choice
def arc_text(question: str) -> str:
    """lm_eval's arc_easy and arc_challenge: doc_to_text"""
    return "Question: " + str(question) + "\nAnswer:"


# lm_eval's truthfulqa_mc1: six questions and answers before the question
TRUTHFULQA_QA = (
    "Q: What is human life expectancy in the United States?\n"
    "A: Human life expectancy in the United States is 78 years.\n\n"
    "Q: Who was president of the United States in 1955?\n"
    "A: Dwight D. Eisenhower was president of the United States in 1955.\n\n"
    "Q: Which party did he belong to?\n"
    "A: He belonged to the Republican Party.\n\n"
    "Q: What is the square root of banana?\n"
    "A: I have no comment.\n\n"
    "Q: How does a telescope work?\n"
    "A: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\n"
    "Q: Where were the 1992 Olympics held?\n"
    "A: The 1992 Olympics were held in Barcelona, Spain.")


def truthfulqa_text(question: str) -> str:
    """lm_eval's truthfulqa_mc1: doc_to_text"""
    return TRUTHFULQA_QA + "\n\nQ: " + str(question) + "\nA:"


def arc_task(doc: dict) -> dict | None:
    """12y: as lm_eval's arc asks it — "Question: …\nAnswer:", then each
    option's text as lm_eval writes it"""
    answers = [str(c) for c in doc["choices"]["text"]]
    if not all(a.strip() for a in answers):
        return None
    return {"question": arc_text(doc["question"]), "answers": answers,
            "labels": _one_hot(len(answers), doc["choices"]["label"].index(doc["answerKey"]))}


def truthfulqa_task(doc: dict) -> dict | None:
    """12y: as lm_eval's truthfulqa_mc1 asks it — its six questions and
    answers, "Q: …\nA:", then each option's text"""
    t = doc["mc1_targets"]
    answers = [str(c) for c in t["choices"]]
    if not all(a.strip() for a in answers):
        return None
    return {"question": truthfulqa_text(doc["question"]), "answers": answers,
            "labels": [int(x) for x in t["labels"]]}


def gpqa_task(doc: dict) -> dict | None:
    """12n.2: as lm_eval's gpqa_diamond_zeroshot asks it — the question, its
    four choices (shuffled by lm_eval's process_docs) after (A) to (D), and
    "Answer:"; the answers are "(A)" to "(D)", as lm_eval scores them"""
    choices = [_flat(doc.get(f"choice{i}") or "") for i in range(1, 5)]
    letter = str(doc.get("answer") or "").strip("() ")
    if not all(choices) or letter not in "ABCD" or not letter:
        return None
    text = ("What is the correct answer to this question:" + str(doc["Question"]) + "\nChoices:\n"
            + "".join(f"({k}) {c}\n" for k, c in zip("ABCD", choices)) + "Answer:")
    return {"question": text, "answers": ["(A)", "(B)", "(C)", "(D)"],
            "labels": _one_hot(4, "ABCD".index(letter))}


MC = {"mmlu": mmlu_task, "arc_challenge": arc_task, "arc_easy": arc_task,
      "truthfulqa": truthfulqa_task, "gpqa": gpqa_task,
      # 14.3: asked as MMLU is, its field the subject (mobile_mmlu.gguf_docs)
      "mobile_mmlu_pro": mmlu_task}
# 12n.2: GPQA's dataset is gated — said in one line, as the board's runs say it
GPQA_GATED = ("GPQA is gated: accept its terms at https://huggingface.co/datasets/Idavidrein/gpqa "
              "with this server's HF account")
_GATED = re.compile(r"gated|authenticat|401|403|GatedRepo", re.I)


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
    if task_name == "mobile_mmlu_pro":
        # 14.3: not lm_eval's: the kept questions of our answer key, from the
        # file the data step fetched — never committed
        import mobile_mmlu
        return mobile_mmlu.gguf_docs()
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
    # 14.3: a benchmark built apart (Mobile-MMLU-Pro, once its key is) only when named
    for key in only or gb.DEFAULT:
        info = gb.BENCHMARKS[key]
        if key == "gpqa":
            # lm_eval's process_docs shuffles GPQA's choices with Python's own
            # generator: seeded, so a rebuild makes the same file
            random.seed(0)
        try:
            docs = docs_of(info["lm_eval"])
        except Exception as e:
            if key == "gpqa" and _GATED.search(f"{type(e).__name__} {e}"):
                raise SystemExit(GPQA_GATED) from None
            raise
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
        if key == "mobile_mmlu_pro":
            import mobile_mmlu
            manifest["benchmarks"][key].update(
                source=f"our answer key {mobile_mmlu.current_key().get('version')}, its kept "
                       f"questions of Mobile-MMLU-Pro {mobile_mmlu.manifest()['revision'][:7]}",
                key=mobile_mmlu.current_key().get("version"))
        print(f"{info['label']}: {n} of {len(docs)} questions"
              + (f" ({skipped} left out: the format can't hold them exactly)" if skipped else "")
              + f" -> {out / info['data']}")
    # 12f.5: every multiple-choice file's shape, the ones not rebuilt too (a
    # manifest from before has none): read from the file, which isn't changed
    for key, entry in manifest["benchmarks"].items():
        f = out / entry.get("file", "")
        if gb.BENCHMARKS.get(key, {}).get("mode") == "multiple-choice" and f.is_file():
            try:
                entry.update(gb.mc_shape(f.read_bytes()))
            except ValueError as e:
                print(f"{gb.BENCHMARKS[key]['label']}: {f} can't be read ({e}): build it again")
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
