#!/usr/bin/env python3
"""Grade free-response answers with a local, pinned judge; write judge.json.

    python scripts/judge.py results/full -m local/my-ckpt            # JUDGE_MODEL from env
    python scripts/judge.py results/full --judge meta-llama/Llama-3.1-8B-Instruct
    python scripts/judge.py results/full --stub                      # tests, no GPU

Reads the harness's --log_samples output for the fr_* tasks (generate_until;
see eval_tasks/fr/AUTHORING.md) and writes results/full/<model>/judge.json
in the same shape-and-place pattern as diagnose.json, so the dashboard's
_beside() loader and the service's freshness key pick it up unchanged.

What makes this a benchmark rather than decoration:

  pinned     the judge is a local model loaded from HF_HOME; its safetensors
             and config are sha256'd into every judge.json. A judge behind an
             API changes when the vendor does and cannot be re-run.
  greedy     temperature 0, one pass; judging the same answers twice writes a
             byte-identical file (a test does it)
  rubric     single answers graded 0–4 against a written, versioned rubric
             with anchors (eval_tasks/fr/rubrics/); never pairwise — that is
             where position bias lives. Length is in the rubric explicitly and
             score-vs-length is reported per category so length bias is
             visible if it appears
  family     a judge never grades a model of its own family (self-preference
             is documented and large). Those cells say so instead
  control    fr_control_mmlu re-asks DIAGNOSE-half MMLU items open-ended; the
             join with the model's own MMLU log answers "knew it, couldn't
             pick it" vs "didn't know it either way" per category

Nothing here writes to the GPU queue itself: the service runs this as the
last step of a suite=judged job, inside the same lock as the evaluation.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import diagnose as dx  # noqa: E402
from fr_build import ALL_TASKS, CONTROL_TASK  # noqa: E402

RUBRIC_DIR = REPO / "eval_tasks" / "fr" / "rubrics"
PROMPT_VERSION = 1
MAX_SCORE = 4
CORRECT_AT = 3           # an open-ended answer scoring >= this counts as "knew it"
LENGTH_BUCKETS = [(0, 20, "≤20 words"), (21, 50, "21–50"), (51, 120, "51–120"), (121, 10**9, ">120")]

PROMPT = """You are grading ONE answer to ONE question against a rubric. Read the rubric, the question, the reference answer, and the candidate answer. Reply with a single digit from 0 to 4 and nothing else.

RUBRIC
{rubric}

QUESTION
{question}

REFERENCE (what a full-marks answer contains)
{reference}

CANDIDATE ANSWER
{answer}

Score (0-4):"""


def family(model_id: str) -> str:
    """Same rule as the dashboard's `family` field: the first alphanumeric run
    of the repo name — 'llama' from 'Llama-3.1-8B-Instruct', 'pythia' from
    'pythia-160m', 'smollm2' from 'SmolLM2-360M'."""
    return re.split(r"[^a-z0-9]", model_id.split("/")[-1].lower())[0]


def rubric_for(task: str) -> tuple[str, str, str]:
    """(text, sha256, version) — the control set is graded with the factual
    rubric: it asks for a fact, and the gold option is the reference."""
    cat = "factual_accuracy" if task == CONTROL_TASK else task[len("fr_"):]
    p = RUBRIC_DIR / f"{cat}.md"
    text = p.read_text(encoding="utf-8")
    m = re.search(r"\(version (\d+)\)", text)
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest(), (m.group(1) if m else "?")


def build_prompt(rubric: str, question: str, reference: str, answer: str) -> str:
    return PROMPT.format(rubric=rubric.strip(), question=question.strip(),
                         reference=reference.strip(), answer=(answer or "").strip() or "(empty)")


def prompt_sha() -> str:
    return hashlib.sha256(f"v{PROMPT_VERSION}\n{PROMPT}".encode("utf-8")).hexdigest()


def words(s: str) -> int:
    return len((s or "").split())


# ---------------------------------------------------------------------------
# graders
# ---------------------------------------------------------------------------

class Grader:
    id = "?"
    weights_sha256 = ""
    config_sha256 = ""

    def grade(self, prompt: str) -> int:
        raise NotImplementedError


_STOP = set("the a an of to in and or is are was were be it its this that for on with as by at from "
            "which who what when where how not no yes".split())


def _content(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _STOP}


class StubGrader(Grader):
    """A deterministic stand-in for tests and dry runs: content-word overlap
    between the candidate and the reference, with the rubric's length clause
    applied. It is NOT a judge — it exists so the plumbing (shapes, hashes,
    determinism, the control join, the calibration round trip) can be tested
    without a GPU. Marked as such in every judge.json it writes."""
    id = "stub/overlap-v1"
    weights_sha256 = hashlib.sha256(b"stub/overlap-v1").hexdigest()
    config_sha256 = weights_sha256

    def grade(self, prompt: str) -> int:
        ref = prompt.split("REFERENCE (what a full-marks answer contains)\n", 1)[1]
        ref, ans = ref.split("\n\nCANDIDATE ANSWER\n", 1)
        ans = ans.rsplit("\n\nScore (0-4):", 1)[0]
        a, r = _content(ans), _content(ref)
        if not a or not r or ans.strip() == "(empty)":
            return 0
        # recall of the reference's content words, so padding cannot dilute a
        # right answer into a wrong one — that is the length clause's job below
        recall = len(a & r) / len(r)
        score = 4 if recall >= 0.8 else 3 if recall >= 0.6 else 2 if recall >= 0.35 \
            else 1 if recall > 0 else 0
        if score >= 3 and words(ans) > max(60, 3 * words(ref)):
            score -= 1                                        # the length clause
        return score


class HFGrader(Grader):
    """The real judge: a local instruct model from HF_HOME, greedy, one pass.
    Records the sha256 of its safetensors and config so the run is pinned."""

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.weights_sha256, self.config_sha256 = self._pin(model_id)
        self.torch = torch

    @staticmethod
    def _pin(model_id: str) -> tuple[str, str]:
        from huggingface_hub import snapshot_download
        d = Path(snapshot_download(model_id, local_files_only=True))
        h = hashlib.sha256()
        for f in sorted(d.glob("*.safetensors")):
            h.update(f.name.encode())
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
        cfg = (d / "config.json").read_bytes() if (d / "config.json").exists() else b""
        return h.hexdigest(), hashlib.sha256(cfg).hexdigest()

    def grade(self, prompt: str) -> int:
        msgs = [{"role": "user", "content": prompt}]
        try:
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                               return_tensors="pt")
        except Exception:                                    # noqa: BLE001 — base model
            ids = self.tok(prompt, return_tensors="pt")["input_ids"]
        ids = ids.to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=4, do_sample=False, temperature=None,
                                      top_p=None, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"[0-4]", text)
        return int(m.group(0)) if m else 0


# ---------------------------------------------------------------------------
# reading the logs
# ---------------------------------------------------------------------------

def _records(model_dir: Path, task: str) -> list[dict]:
    dirs = [d for d in model_dir.glob(f"{task}_*shot") if re.fullmatch(rf"{task}_\d+shot", d.name)]
    files = dx.newest_per_subtask(sorted(f for d in dirs for f in d.rglob("samples_*.jsonl")))
    out = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _answer(rec: dict) -> str:
    r = rec.get("filtered_resps") or rec.get("resps") or []
    while isinstance(r, list) and r:
        r = r[0]
    return str(r) if isinstance(r, str) else ""


def mmlu_outcomes(model_dir: Path) -> dict[str, bool]:
    """doc_hash -> right/wrong on the model's own MMLU run (the MC side of
    the control comparison)."""
    out: dict[str, bool] = {}
    for rec in _records(model_dir, "mmlu"):
        pm = dx.primary_metric(rec)
        dh = rec.get("doc_hash")
        if pm and dh:
            out[dh] = pm[1] >= 0.5
    return out


def length_bucket(n: int) -> str:
    for lo, hi, label in LENGTH_BUCKETS:
        if lo <= n <= hi:
            return label
    return LENGTH_BUCKETS[-1][2]


# ---------------------------------------------------------------------------
# judging one model
# ---------------------------------------------------------------------------

def judge_model(model_dir: Path, grader: Grader, judge_family: str) -> dict | None:
    model_id = model_dir.name.replace("__", "/", 1)
    tasks_present = [t for t in ALL_TASKS if any(model_dir.glob(f"{t}_*shot"))]
    if not tasks_present:
        return None
    head = {"judge": {"id": grader.id, "family": judge_family,
                      "weights_sha256": grader.weights_sha256,
                      "config_sha256": grader.config_sha256,
                      "prompt_sha256": prompt_sha(), "prompt_version": PROMPT_VERSION,
                      "greedy": True, "stub": isinstance(grader, StubGrader),
                      "rubrics": {t: {"sha256": rubric_for(t)[1], "version": rubric_for(t)[2]}
                                  for t in tasks_present}},
            "model": model_id, "split_salt": dx.SPLIT_SALT, "correct_at": CORRECT_AT}
    if family(model_id) == judge_family:
        return {**head, "skipped": f"not judged — same family as judge ({judge_family})",
                "tasks": {}}
    mc = mmlu_outcomes(model_dir) if CONTROL_TASK in tasks_present else {}
    tasks: dict[str, dict] = {}
    for task in tasks_present:
        rubric, _, _ = rubric_for(task)
        items = []
        for rec in sorted(_records(model_dir, task), key=lambda r: str(r.get("doc_hash"))):
            doc = rec.get("doc") or {}
            ans = _answer(rec)
            score = grader.grade(build_prompt(rubric, doc.get("prompt", ""),
                                              doc.get("reference", ""), ans))
            item = {"doc_hash": rec.get("doc_hash"), "id": doc.get("id"),
                    "category": doc.get("category"), "score": int(score),
                    "answer_words": words(ans)}
            if task == CONTROL_TASK:
                item["mmlu_doc_hash"] = doc.get("mmlu_doc_hash")
                item["mc_right"] = mc.get(doc.get("mmlu_doc_hash"))
            items.append(item)
        if not items:
            continue
        dist = collections.Counter(str(it["score"]) for it in items)
        by_len: dict[str, list[int]] = collections.defaultdict(list)
        for it in items:
            by_len[length_bucket(it["answer_words"])].append(it["score"])
        t = {"n": len(items), "mean": round(sum(it["score"] for it in items) / len(items), 4),
             "max": MAX_SCORE,
             "dist": {str(k): dist.get(str(k), 0) for k in range(MAX_SCORE + 1)},
             "score_vs_length": [{"bucket": label, "n": len(by_len[label]),
                                  "mean": round(sum(by_len[label]) / len(by_len[label]), 4)}
                                 for _, _, label in LENGTH_BUCKETS if by_len.get(label)],
             "items": items}
        if task == CONTROL_TASK:
            ctl: dict[str, dict] = {}
            for it in items:
                c = ctl.setdefault(it["category"] or "—", {"n": 0, "mc_wrong": 0, "knew": 0,
                                                            "didnt": 0, "unjoined": 0})
                c["n"] += 1
                if it["mc_right"] is None:
                    c["unjoined"] += 1
                elif not it["mc_right"]:
                    c["mc_wrong"] += 1
                    if it["score"] >= CORRECT_AT:
                        c["knew"] += 1
                    else:
                        c["didnt"] += 1
            t["control"] = ctl
        tasks[task] = t
    return {**head, "tasks": tasks}


def write_judge(model_dir: Path, out: dict, dest: Path | None = None) -> Path:
    d = dest or model_dir
    d.mkdir(parents=True, exist_ok=True)
    p = d / "judge.json"
    # sort_keys + no timestamps: judging twice writes the same bytes
    p.write_text(json.dumps(out, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("-m", "--model", action="append", default=[])
    ap.add_argument("--judge", default=os.environ.get("JUDGE_MODEL", ""),
                    help="judge model id in HF_HOME (default: $JUDGE_MODEL)")
    ap.add_argument("--stub", action="store_true",
                    help="deterministic overlap stand-in; tests and dry runs only")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="write judge.json under this directory instead of beside the results")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    if not a.results.is_dir():
        print(f"no such directory: {a.results}", file=sys.stderr)
        return 2
    if a.stub:
        grader: Grader = StubGrader()
        jf = "stub"
    elif a.judge:
        grader = HFGrader(a.judge)
        jf = family(a.judge)
    else:
        print("no judge: set JUDGE_MODEL or pass --judge (or --stub for a dry run)",
              file=sys.stderr)
        return 2
    want = {m.replace("/", "__") for m in a.model}
    n = 0
    for d in sorted(p for p in a.results.iterdir() if p.is_dir()):
        if want and d.name not in want:
            continue
        out = judge_model(d, grader, jf)
        if out is None:
            continue
        write_judge(d, out, (a.out / d.name) if a.out else None)
        n += 1
        if not a.quiet:
            if out.get("skipped"):
                print(f"{d.name:46} {out['skipped']}")
            else:
                bits = [f"{t}={v['mean']:.2f}/4 (n={v['n']})" for t, v in sorted(out["tasks"].items())]
                print(f"{d.name:46} {' '.join(bits)}")
    if not a.quiet:
        print(f"\nwrote judge.json for {n} model(s) · judge {grader.id}"
              + (" · STUB — not a judgement" if a.stub else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
