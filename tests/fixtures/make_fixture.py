#!/usr/bin/env python3
"""A synthetic results/full tree in the harness's real 0.4.12 shape.

    python tests/fixtures/make_fixture.py OUT_DIR [--report OUT.html] [--no-diagnose]

Why synthetic: the real tree is thirty models by fourteen thousand MMLU items,
root-owned, on the server. Tests need something small, deterministic and shaped
like the truth — every key the report and the diagnosis read, in the form the
harness writes it — with each failure mode diagnose.py detects planted in a
model of its own, so a test can say "this model, this finding" and mean it.

What is in it, one model per behaviour:

  fx/chance-160m         guesses uniformly: at chance everywhere, no finding
  fx/below-135m-it       confidently backs a wrong option — the chat-template
                         pathology; scores below chance, "confidently wrong"
  fx/one-option-70m      answers A almost every time: "one option only"
  fx/skewed-360m         92% of its picks on two of four slots: "answer positions"
  fx/short-pick-410m     picks the shortest option: "option length"
  fx/good-750m           clears chance with an ordinary subject gap
                         (econometrics) — the case a generator is for
  fx/miscount-1b         its results file declares more arc_easy items than
                         its per-item log holds
  local/nodiag-step400   an uploaded checkpoint, quick suite only, and no
                         diagnose.json is written for it

Two children of fx/good-750m close the loop. Both trained on a dataset
derived from MMLU's diagnosis half (tainted on mmlu, parent recorded):
fx/good-750m-tuned-test improved only on the diagnosis half — "the training
taught the test", the warning — and fx/good-750m-tuned-skill improved on both
halves — "the training taught the skill". The frozen report carries their
taint the way the service would compute it from the run/dataset join.

Three models (good, skewed, chance) also sit the exam: a bank drafted by the
fake exam writer across every topic in categories.yaml and accepted by the
fixture (approver "fixture"), the four skill suites' 40 items migrated in
under `other`, and the MMLU control set exam_build builds from the diagnose
half of this very tree. Every question is split by qid. Their
answers are graded by scripts/judge.py's STUB grader, and a synthetic
calibration CSV (human = judge with every seventh row off by one) is
imported so the board has a kappa over the line. The skewed model answers the
control items correctly ("knew it, couldn't pick it"); the chance model does
not ("didn't know it either way").

Three models also carry the permutation control (mmlu_perm, a control task
that never enters an average): fx/skewed-360m clears chance on it while its
mmlu sits at chance ("the format was hiding measurable knowledge"),
fx/chance-160m sits at chance on both ("the knowledge is not there to hide"),
and fx/good-750m clears both. The control's documents go through the real
transform in eval_tasks/mmlu_perm/utils.py.

Every model answers the SAME documents — a benchmark is the same questions
for everyone — so doc_hash, and therefore the report/diagnose split, is
identical across models. Only the responses differ. Seeded; byte-identical
across runs.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
import diagnose as dx  # noqa: E402

SEED = 1234
TS = "2026-08-19T10-02-11.000000"          # the harness's filename timestamp
STALE_TS = "2026-08-18T09-00-00.000000"    # an older re-run left in place
DATE = 1755590531.0
GIT_HASH = "f1x7ure0"
TRANSFORMERS = "4.56.0"

# eight of MMLU's 57 subjects, with the category the harness rolls them into
MMLU_SUBJECTS = {
    "abstract_algebra": "stem", "anatomy": "stem",
    "econometrics": "social_sciences", "high_school_macroeconomics": "social_sciences",
    "us_foreign_policy": "social_sciences", "world_religions": "humanities",
    "professional_law": "humanities", "nutrition": "other",
}
MMLU_PER_SUBJECT = 40      # two-subject categories land above the 30-item noise floor,
                           # one-subject ones below it, so the page shows both states
PERM_SUBJECTS = ["anatomy", "econometrics", "us_foreign_policy", "world_religions"]
PERM_PER_SUBJECT = 40

TASKS = {
    "mmlu":           {"shots": 5, "metrics": ["acc"], "n_options": 4, "norm": False},
    "mmlu_perm":      {"shots": 5, "metrics": ["acc"], "n_options": 4, "norm": False},
    "hellaswag":      {"shots": 5, "metrics": ["acc", "acc_norm"], "n": 80, "n_options": 4,
                       "norm": True},
    "arc_challenge":  {"shots": 5, "metrics": ["acc", "acc_norm"], "n": 60, "n_options": 4,
                       "norm": True},
    "arc_easy":       {"shots": 5, "metrics": ["acc", "acc_norm"], "n": 60, "n_options": 4,
                       "norm": True},
    "winogrande":     {"shots": 5, "metrics": ["acc"], "n": 60, "n_options": 2, "norm": False},
    "piqa":           {"shots": 0, "metrics": ["acc", "acc_norm"], "n": 60, "n_options": 2,
                       "norm": True},
    "truthfulqa_mc2": {"shots": 0, "metrics": ["acc"], "n": 40, "n_options": None,
                       "norm": False},
}
CONTROL = ["mmlu_perm"]                    # a control: in no suite's average
FULL = [t for t in TASKS if t not in CONTROL]
QUICK = ["hellaswag", "arc_easy"]

NODIAG = "local/nodiag-step400"
# judged free response: which models answered, and how well
JUDGED = {"fx/good-750m": 0.75, "fx/skewed-360m": 0.8, "fx/chance-160m": 0.0}
FILLER = ("To put this in context, there are several considerations that could be raised "
          "here, and each of them would take some time to lay out fully; nevertheless the "
          "essential point stands as stated above, and the remaining detail does not alter it "
          "in any material way, which is why the summary given first is the one to keep.")
MISCOUNT = {"model": "fx/miscount-1b", "task": "arc_easy", "declared": 80}
STALE = {"model": "fx/good-750m", "task": "hellaswag", "lines": 10}

# id, policy, kind, params, tasks, per-task policy overrides
MODELS = [
    ("fx/chance-160m",     "chance",     "base",     162_000_000, FULL + CONTROL,
     {"mmlu_perm": "chance"}),
    ("fx/below-135m-it",   "below",      "instruct", 135_000_000, FULL, {}),
    ("fx/one-option-70m",  "one_option", "base",      70_000_000, FULL, {}),
    ("fx/skewed-360m",     "skewed",     "base",     360_000_000, FULL + CONTROL,
     {"mmlu_perm": "good"}),          # the skew was the format: rotated, it knows things
    ("fx/short-pick-410m", "short_pick", "base",     410_000_000, FULL, {}),
    ("fx/good-750m",       "good",       "base",     750_000_000, FULL + CONTROL, {}),
    (MISCOUNT["model"],    "chance",     "base",   1_000_000_000, FULL, {}),
    (NODIAG,               "chance",     "base",      70_000_000, QUICK, {}),
    ("fx/good-750m-tuned-test",  "good_test",  "base", 750_000_000, FULL, {}),
    ("fx/good-750m-tuned-skill", "good_skill", "base", 750_000_000, FULL, {}),
]
# what the service's run/dataset join would say about the two children
TAINT = {"fx/good-750m-tuned-test": ["mmlu"], "fx/good-750m-tuned-skill": ["mmlu"]}
PARENTS = {"fx/good-750m-tuned-test": "fx/good-750m", "fx/good-750m-tuned-skill": "fx/good-750m"}

WORDS = ("ledger", "tariff", "enzyme", "monsoon", "quorum", "isotope", "vector", "treaty",
         "kernel", "plateau", "synapse", "dividend", "glacier", "rhetoric", "lattice",
         "census", "orbit", "solvent", "verdict", "harvest", "pigment", "cipher", "canopy",
         "mortgage", "neuron", "basalt", "referendum", "torque", "parable", "estuary",
         "a", "of", "the", "under", "against", "beyond", "into", "toward", "without", "per")
TOPICS = ("marginal cost", "ring homomorphisms", "the vagus nerve", "fixed exchange rates",
          "containment policy", "the Vedas", "strict liability", "vitamin K", "heteroskedasticity",
          "aggregate demand", "Noether's theorem", "the brachial plexus", "deterrence",
          "monastic orders", "consideration in contract", "iron absorption", "instrumental "
          "variables", "the liquidity trap", "group actions", "the Krebs cycle")

_SANITIZE = re.compile(r"[\"<>:/|\\?*\[\]=]+")


def _rng(seed: int, *parts) -> random.Random:
    return random.Random(f"{seed}:" + ":".join(str(p) for p in parts))


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _phrase(rng: random.Random, k: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(k))


def _options(rng: random.Random, n: int) -> list[str]:
    """n option strings with pairwise-distinct byte lengths, so 'the shortest
    option' names exactly one slot and the length-bias planting is unambiguous."""
    while True:
        ks = [1, 2, 3, 5, 4, 6, 7][:n]
        rng.shuffle(ks)
        opts = [_phrase(rng, k) for k in ks]
        if len({len(o.encode()) for o in opts}) == n:
            return opts


def _blen(s: str) -> int:
    return max(1, len(s.encode("utf-8")))


def perm_rotate():
    """The control's own transform, imported from where the harness loads it,
    so the fixture cannot drift from what a real mmlu_perm run does."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mmlu_perm_utils", REPO / "eval_tasks" / "mmlu_perm" / "utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.rotate


# ---------------------------------------------------------------------------
# documents — one benchmark, shared by every model
# ---------------------------------------------------------------------------

def make_docs(task: str, seed: int = SEED) -> list[dict]:
    """The items of one task. Each entry carries the harness `doc` plus what the
    responder needs (choices, the correct index) and what the tests need (the
    question text as the diagnosis will print it, the group, the subtask)."""
    rng = _rng(seed, "docs", task)
    spec = TASKS[task]
    out: list[dict] = []
    seen_q: set[str] = set()

    def unique(q: str) -> str:
        # the rule test maps a printed question back to its doc_hash, so two
        # items must never share a question
        while q in seen_q:
            q = q[:-1] + " " + rng.choice(WORDS) + "?"
        seen_q.add(q)
        return q

    if task in ("mmlu", "mmlu_perm"):
        rotate = perm_rotate() if task == "mmlu_perm" else None
        subjects = PERM_SUBJECTS if rotate else list(MMLU_SUBJECTS)
        per = PERM_PER_SUBJECT if rotate else MMLU_PER_SUBJECT
        for subject in subjects:
            for i in range(per):
                # gold uniform over the slots either way: round-robin for mmlu;
                # for the control, a key that changes every four items and is
                # then rotated by i mod 4, so every (key, shift) pair occurs
                correct = (i // 4) % 4 if rotate else i % 4
                opts = _options(rng, 4)
                # long enough to carry a 13-gram: the contamination gate must
                # have something to catch, and real MMLU questions are longer still
                q = unique(f"Which of the following best describes {rng.choice(TOPICS)} "
                           f"in {subject.replace('_', ' ')}, as it is treated in a standard "
                           f"introductory course on the subject?")
                doc = {"question": q, "subject": subject, "choices": opts, "answer": correct}
                if rotate:
                    doc = rotate(doc, i)
                    opts, correct = doc["choices"], doc["answer"]
                # the harness logs MMLU's target as the digit string ('0'), not an
                # int — checked against a real samples file on the server
                out.append({"doc": doc, "target": str(correct), "choices": opts,
                            "correct": correct, "group": subject,
                            "subtask": f"{task}_{subject}", "q": q,
                            "ctx": f"{q}\nA. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\n"
                                   f"D. {opts[3]}\nAnswer:"})
        return _hashed(out)

    n = spec["n"]
    for i in range(n):
        if task == "hellaswag":
            correct = i % 4
            opts = _options(rng, 4)
            act = rng.choice(TOPICS).title()
            ctx = unique(f"A person is {_phrase(rng, 4)}. Then they")
            doc = {"ind": i, "activity_label": act, "ctx_a": ctx, "ctx_b": "", "ctx": ctx,
                   "endings": opts, "source_id": f"activitynet~v_{i:05d}", "split": "val",
                   "split_type": "indomain", "label": str(correct),
                   "query": f"{act}: {ctx}", "choices": opts, "gold": correct}
            out.append({"doc": doc, "target": str(correct), "choices": opts, "correct": correct,
                        "group": "—", "subtask": task, "q": doc["query"], "ctx": doc["query"]})
        elif task in ("arc_challenge", "arc_easy"):
            correct = i % 4
            opts = _options(rng, 4)
            q = unique(f"What happens to {rng.choice(TOPICS)} when {_phrase(rng, 3)}?")
            doc = {"id": f"Mercury_{7000000 + i}", "question": q,
                   "choices": {"text": opts, "label": ["A", "B", "C", "D"]},
                   "answerKey": "ABCD"[correct]}
            out.append({"doc": doc, "target": str(correct), "choices": opts, "correct": correct,
                        "group": "—", "subtask": task, "q": q, "ctx": f"Question: {q}\nAnswer:"})
        elif task == "winogrande":
            correct = i % 2
            opts = _options(rng, 2)
            tail = _phrase(rng, 3)
            sentence = unique(f"The {opts[0]} could not fit the {opts[1]} because _ {tail}.")
            # the harness's target for winogrande is the partial sentence after
            # the blank, not an index — which is why diagnose.py cannot find a
            # single answer index for it and says so
            doc = {"sentence": sentence, "option1": opts[0], "option2": opts[1],
                   "answer": str(correct + 1)}
            out.append({"doc": doc, "target": f" {tail}.", "choices": opts, "correct": correct,
                        "group": "—", "subtask": task, "q": "", "ctx": sentence})
        elif task == "piqa":
            correct = i % 2
            opts = _options(rng, 2)
            goal = unique(f"How do you {_phrase(rng, 3)}?")
            doc = {"goal": goal, "sol1": opts[0], "sol2": opts[1], "label": correct}
            out.append({"doc": doc, "target": correct, "choices": opts, "correct": correct,
                        "group": "—", "subtask": task, "q": goal, "ctx": f"Question: {goal}\n"
                        "Answer:"})
        elif task == "truthfulqa_mc2":
            k = rng.randint(4, 7)
            n_true = rng.randint(2, 3)
            opts = _options(rng, k)
            labels = [1] * n_true + [0] * (k - n_true)
            q = unique(f"What is true about {rng.choice(TOPICS)}?")
            doc = {"question": q,
                   "mc1_targets": {"choices": [opts[0]] + opts[n_true:],
                                   "labels": [1] + [0] * (k - n_true)},
                   "mc2_targets": {"choices": opts, "labels": labels}}
            out.append({"doc": doc, "target": 0, "choices": opts, "correct": None,
                        "labels": labels, "group": "—", "subtask": task, "q": q,
                        "ctx": f"Q: {q}\nA:"})
        else:
            raise ValueError(task)
    return _hashed(out)


def _hashed(items: list[dict]) -> list[dict]:
    # the harness hashes the doc it saw; the split is a function of this alone
    for item in items:
        item["doc_hash"] = _sha(json.dumps(item["doc"], sort_keys=True, ensure_ascii=False))
    return items


# ---------------------------------------------------------------------------
# responses — where each model's behaviour is planted
# ---------------------------------------------------------------------------

def policy_probs(rng: random.Random, policy: str, n: int, c: int | None,
                 shortest: int, group: str, half: str = "report") -> list[float]:
    """Per-option probabilities (after any length normalisation) for one item.
    Thresholds are diagnose.py's: lift = p * n, chance is 1.0, 'confident'
    starts at conf_lift(n), at-chance ends at 1.30."""
    u = 1.0 / n

    def peaked(i: int, top: float) -> list[float]:
        rest = (1.0 - top) / (n - 1)
        return [top if j == i else rest for j in range(n)]

    def wobble(p: list[float]) -> list[float]:
        # organic-looking noise, small enough never to reorder the options
        q = [max(1e-6, v * (1 + rng.uniform(-0.04, 0.04))) for v in p]
        s = sum(q)
        return [v / s for v in q]

    if c is None or (n < 3 and policy in ("skewed", "short_pick")):
        policy = "chance"
    if policy == "chance":                       # lift 1.08: no information either way
        return wobble(peaked(rng.randrange(n), u * 1.08))
    if policy == "below":                        # confidently wrong on 85% of items
        wrong = [j for j in range(n) if j != c]
        i = rng.choice(wrong) if rng.random() < 0.85 else c
        return wobble(peaked(i, 0.6 if n > 2 else 0.8))
    if policy == "one_option":
        i = 0 if rng.random() < 0.95 else rng.randrange(n)
        return wobble(peaked(i, 0.5 if n > 2 else 0.8))
    if policy == "skewed":                       # the first two slots hold 92%
        i = rng.randrange(2) if rng.random() < 0.92 else rng.randrange(n)
        return wobble(peaked(i, 0.45))
    if policy == "short_pick":
        i = shortest if rng.random() < 0.7 else rng.randrange(n)
        return wobble(peaked(i, 0.5))
    if policy in ("good", "good_test", "good_skill"):
        p_right = 0.30 if group == "econometrics" else 0.72
        # the children: trained on data derived from the DIAGNOSIS half. One
        # learned the test (only that half improves), one learned the skill
        # (both halves improve — the half the training never saw moved too)
        if policy == "good_skill" or (policy == "good_test" and half == "diagnose"):
            p_right = 0.93
        r = rng.random()
        if r < p_right:
            return wobble(peaked(c, 0.6 if n > 2 else 0.8))
        wrong = rng.choice([j for j in range(n) if j != c])
        if n > 2 and r < p_right + 0.12:         # near miss: correct second, gap 0.05
            p = [(1 - 0.40 - 0.35) / (n - 2)] * n
            p[wrong], p[c] = 0.40, 0.35
            return p
        if n > 2:                                # plain wrong: correct second but gap 0.12
            p = [(1 - 0.40 - 0.28) / (n - 2)] * n
            p[wrong], p[c] = 0.40, 0.28
            return p
        return peaked(wrong, 0.8)
    raise ValueError(policy)


def respond(rng: random.Random, policy: str, item: dict, task: str) -> dict:
    """One harness sample line: the 0.4.12 keys, resps as [[logprob, is_greedy]]
    strings, and the metric values the harness would have scored."""
    spec = TASKS[task]
    choices = item["choices"]
    n = len(choices)
    L = [_blen(s) for s in choices]
    shortest = min(range(n), key=lambda i: L[i])
    probs = policy_probs(rng, policy, n, item["correct"], shortest, item["group"],
                         dx.split_of(item["doc_hash"]))
    z = [math.log(max(p, 1e-9)) for p in probs]
    # acc_norm tasks: the harness scores loglikelihood / byte length, so the
    # planted behaviour lives in normalised space and the raw logprob is that
    # times the length — which is also exactly what diagnose.py undoes
    lps = [zi * li for zi, li in zip(z, L)] if spec["norm"] else z
    pick_raw = max(range(n), key=lambda i: lps[i])
    pick_norm = max(range(n), key=lambda i: z[i])
    rec = {
        "doc_id": item["doc_id"],
        "doc": item["doc"],
        "target": item["target"],
        "arguments": [[item["ctx"], " " + ch] for ch in choices],
        "resps": [[[f"{lp:.4f}", str(i == pick_raw)]] for i, lp in enumerate(lps)],
        "filtered_resps": [[f"{lp:.4f}", str(i == pick_raw)] for i, lp in enumerate(lps)],
        "doc_hash": item["doc_hash"],
        "prompt_hash": _sha(item["ctx"]),
        "target_hash": _sha(str(item["target"])),
        "filter": "none",
        "metrics": list(spec["metrics"]),
    }
    if task == "truthfulqa_mc2":              # mass on every true option
        rec["acc"] = round(sum(p for p, lab in zip(probs, item["labels"]) if lab), 6)
    else:
        rec["acc"] = float(pick_raw == item["correct"])
        if "acc_norm" in spec["metrics"]:
            rec["acc_norm"] = float(pick_norm == item["correct"])
    return rec


# ---------------------------------------------------------------------------
# files — the tree the harness would have left behind
# ---------------------------------------------------------------------------

def safe_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def model_args(model_id: str) -> str:
    # uploaded artifacts are evaluated by absolute path; the report folds that
    # back into local/<name>
    pretrained = (f"/bench/artifacts/{model_id[6:]}" if model_id.startswith("local/")
                  else model_id)
    return f"pretrained={pretrained},dtype=bfloat16"


def _mean_se(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    p = sum(vals) / n if n else 0.0
    return round(p, 6), round(math.sqrt(max(p * (1 - p), 0.0) / n), 6) if n else 0.0


_TEMPLATE = ("{% for message in messages %}<|{{ message['role'] }}|>\n"
             "{{ message['content'] }}<|end|>\n{% endfor %}")


def write_task(out_dir: Path, model_id: str, kind: str, params: int, policy: str,
               task: str, docs: list[dict], seed: int) -> Path:
    spec = TASKS[task]
    shots = spec["shots"]
    task_dir = (out_dir / safe_name(model_id) / f"{task}_{shots}shot"
                / _SANITIZE.sub("__", model_args(model_id)))
    task_dir.mkdir(parents=True, exist_ok=True)
    rng = _rng(seed, "resp", model_id, task)

    lines: dict[str, list[dict]] = collections.defaultdict(list)
    doc_id: dict[str, int] = collections.Counter()
    for item in docs:
        item = dict(item, doc_id=doc_id[item["subtask"]])
        doc_id[item["subtask"]] += 1
        lines[item["subtask"]].append(respond(rng, policy, item, task))
    for st, recs in lines.items():
        with open(task_dir / f"samples_{st}_{TS}.jsonl", "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if model_id == STALE["model"] and task == STALE["task"]:
        # a re-run leaves the older timestamped file beside the new one; a
        # reader that counts both doubles every item in the subtask
        with open(task_dir / f"samples_{task}_{STALE_TS}.jsonl", "w", encoding="utf-8") as fh:
            for r in lines[task][:STALE["lines"]]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    results: dict[str, dict] = {}
    n_samples: dict[str, dict] = {}
    n_shot: dict[str, int] = {}
    hib: dict[str, dict] = {}

    def entry(alias: str, recs: list[dict]) -> dict:
        e = {"alias": alias}
        for m in spec["metrics"]:
            v, se = _mean_se([r[m] for r in recs])
            e[f"{m},none"] = v
            e[f"{m}_stderr,none"] = se
        return e

    grouped = task in ("mmlu", "mmlu_perm")
    for st, recs in lines.items():
        results[st] = entry(("  - " + st[len(task) + 1:]) if grouped else task, recs)
        n = len(recs)
        if model_id == MISCOUNT["model"] and task == MISCOUNT["task"]:
            n = MISCOUNT["declared"]
        n_samples[st] = {"original": n, "effective": n}
        n_shot[st] = shots
        hib[st] = {m: True for m in spec["metrics"]}
    if task == "mmlu":
        cats: dict[str, list[str]] = collections.defaultdict(list)
        for subject, cat in MMLU_SUBJECTS.items():
            cats[f"mmlu_{cat}"].append(f"mmlu_{subject}")
        for cat, kids in cats.items():
            results[cat] = entry(" - " + cat[5:], [r for k in kids for r in lines[k]])
            hib[cat] = {"acc": True}
        results["mmlu"] = entry("mmlu", [r for recs in lines.values() for r in recs])
        hib["mmlu"] = {"acc": True}
        group_subtasks = {"mmlu": sorted(cats), **{c: kids for c, kids in cats.items()}}
    elif grouped:                     # the control: one flat group of subjects
        results[task] = entry(task, [r for recs in lines.values() for r in recs])
        hib[task] = {"acc": True}
        group_subtasks = {task: sorted(lines)}
    else:
        group_subtasks = {task: []}

    blob = {
        "results": results,
        "group_subtasks": group_subtasks,
        "configs": {st: {"task": st, "num_fewshot": shots, "metric_list":
                         [{"metric": m, "aggregation": "mean", "higher_is_better": True}
                          for m in spec["metrics"]], "output_type": "multiple_choice"}
                    for st in lines},
        "versions": {st: 1.0 for st in lines},
        "n-shot": n_shot,
        "higher_is_better": hib,
        "n-samples": n_samples,
        "config": {
            "model": "hf", "model_args": model_args(model_id), "batch_size": "8",
            "batch_sizes": [], "device": "cuda:0", "use_cache": None, "limit": None,
            "bootstrap_iters": 100000, "gen_kwargs": None, "random_seed": 1234,
            "numpy_seed": 1234, "torch_seed": 1234, "fewshot_seed": 1234,
            "model_num_parameters": params, "model_dtype": "torch.bfloat16",
        },
        "git_hash": GIT_HASH,
        "date": DATE + list(TASKS).index(task) * 600,
        "pretty_env_info": "fixture",
        "transformers_version": TRANSFORMERS,
        "upper_git_hash": None,
        "tokenizer_pad_token": ["<|endoftext|>", "0"],
        "tokenizer_eos_token": ["<|endoftext|>", "0"],
        "tokenizer_bos_token": [None, "None"],
        "eot_token_id": 0,
        "max_length": 2048,
        "task_hashes": {st: _sha(st) for st in lines},
        "model_source": "hf",
        "model_name": model_args(model_id).split(",")[0][11:],
        "model_name_sanitized": _SANITIZE.sub("__", model_args(model_id).split(",")[0][11:]),
        "system_instruction": None,
        "system_instruction_sha": None,
        "fewshot_as_multiturn": False,
        "chat_template": _TEMPLATE if kind == "instruct" else None,
        "chat_template_sha": _sha(_TEMPLATE) if kind == "instruct" else None,
        "start_time": 1000.0,
        "end_time": 1000.0 + 90 * (1 + len(docs) / 100),
        "total_evaluation_time_seconds": str(90 * (1 + len(docs) / 100)),
    }
    (task_dir / f"results_{TS}.json").write_text(
        json.dumps(blob, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return task_dir


def write_model_meta(out_dir: Path, model_id: str, kind: str, params: int) -> None:
    layers = max(4, round(math.log(params, 2)) - 18)
    meta = {"model": model_id, "kind": kind, "params": params,
            "kind_reason": ("chat template applied: the repo ships one and the name says -it"
                            if kind == "instruct" else
                            "no chat template in the repo; evaluated as a base model"),
            "arch": "FixtureForCausalLM", "hidden": 64 * layers, "layers": layers,
            "heads": max(2, layers // 2), "ctx": 2048, "vocab": 50304,
            "tmpl_sha": _sha(_TEMPLATE)[:12] if kind == "instruct" else None,
            "tmpl_src": "tokenizer_config.json" if kind == "instruct" else None}
    d = out_dir / safe_name(model_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def write_diagnoses(root: Path, skip: tuple[str, ...] = (NODIAG,),
                    only: list[str] | None = None) -> list[str]:
    """Run scripts/diagnose.py's analysis over the tree and write diagnose.json
    beside each model's results, the way the tool does. Returns the model ids
    written. `skip` keeps the 'no diagnosis on file' path alive."""
    import diagnose as dx
    out_dir = root / "results" / "full"
    written = []
    for model_id, *_ in MODELS:
        if model_id in skip or (only is not None and model_id not in only):
            continue
        d = out_dir / safe_name(model_id)
        if not d.is_dir():
            continue
        out = dx.diagnose_model(d)
        (d / "diagnose.json").write_text(json.dumps(out), encoding="utf-8")
        written.append(model_id)
    return written


def _fr_answer(rng: random.Random, item: dict, p_right: float) -> str:
    """A generated answer the stub grader can score: mostly the reference in
    other words (right), sometimes padded (right but long), else off-topic."""
    r = rng.random()
    if r < p_right:
        ans = f"{item['reference']} In short, that is the answer."
        if rng.random() < 0.25:
            ans += " " + FILLER                        # long: the rubric's length clause
        return ans
    if r < p_right + 0.1:
        return " ".join(item["reference"].split()[:3]) + ", perhaps, though other readings exist."
    return "I am not certain; it may depend on the context and on who is asking."


EXAM_PER_TOPIC = 6          # drafted candidates per topic in the fixture bank
# one topic carries enough questions for its report half to clear the 30-item
# floor, so the gate that guards a proposal has a case that passes
BIG_TOPIC = "economics"
EXAM_BIG_EXTRA = 70


def write_exam(root: Path, out_dir: Path) -> dict:
    """Draft with the fake exam writer, accept everything as the fixture, add
    the migrated skill items, build the harness tasks. Returns the manifest."""
    import exam_build as eb
    from service import llm
    exam_root = root / "exam"
    eb.migrate_seeds(exam_root)
    fake = llm.FakeBatches("fake-exam", root)
    eb.draft(exam_root, fake, eb.TOPICS, per_topic=EXAM_PER_TOPIC, wait=True, poll_s=0)
    eb.draft(exam_root, fake, [BIG_TOPIC], per_topic=EXAM_BIG_EXTRA, wait=True, poll_s=0)
    for c in eb.load_candidates(exam_root, status="candidate"):
        eb.accept(exam_root, c["cid"], approver="fixture")
    return eb.build(out_dir, exam_root)


def write_judged(root: Path, out_dir: Path, seed: int = SEED) -> dict:
    """Build the exam from this tree, write generate_until samples for the
    judged models, grade them with the stub, calibrate synthetically."""
    import csv

    import exam_build as eb
    import judge as jd
    import judge_calibrate as jc
    manifest = write_exam(root, out_dir)
    tasks_dir = eb.tasks_dir(root / "exam")
    items_by_task = {t: [json.loads(ln) for ln in (tasks_dir / f"{t}.jsonl").read_text(
        encoding="utf-8").splitlines() if ln.strip()] for t in manifest["tasks"]}
    for model_id, p_right in JUDGED.items():
        rng = _rng(seed, "fr", model_id)
        for task, items in items_by_task.items():
            task_dir = (out_dir / safe_name(model_id) / f"{task}_0shot"
                        / _SANITIZE.sub("__", model_args(model_id)))
            task_dir.mkdir(parents=True, exist_ok=True)
            recs = []
            for i, it in enumerate(items):
                ans = _fr_answer(rng, it, p_right)
                ctx = it["prompt"] + "\n\nAnswer:"
                recs.append({"doc_id": i, "doc": it, "target": it["reference"],
                             "arguments": [[ctx, {"until": ["\n\n\n"], "max_gen_toks": 256,
                                                  "do_sample": False, "temperature": 0.0}]],
                             "resps": [[ans]], "filtered_resps": [ans],
                             "doc_hash": _sha(json.dumps(it, sort_keys=True, ensure_ascii=False)),
                             "prompt_hash": _sha(ctx), "target_hash": _sha(it["reference"]),
                             "filter": "none", "metrics": ["bypass"], "bypass": 999})
            with open(task_dir / f"samples_{task}_{TS}.jsonl", "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            blob = {"results": {task: {"alias": task, "bypass,none": 999}},
                    "group_subtasks": {task: []},
                    "configs": {task: {"task": task, "output_type": "generate_until",
                                       "metric_list": [{"metric": "bypass"}]}},
                    "versions": {task: 1.0}, "n-shot": {task: 0},
                    "higher_is_better": {task: {"bypass": True}},
                    "n-samples": {task: {"original": len(recs), "effective": len(recs)}},
                    "config": {"model": "hf", "model_args": model_args(model_id), "batch_size": "8",
                               "device": "cuda:0", "limit": None, "random_seed": 1234,
                               "model_num_parameters": dict((m[0], m[3]) for m in MODELS)[model_id]},
                    "git_hash": GIT_HASH, "date": DATE + 9000, "transformers_version": TRANSFORMERS,
                    "chat_template": None, "total_evaluation_time_seconds": "40.0"}
            (task_dir / f"results_{TS}.json").write_text(json.dumps(blob, indent=2),
                                                         encoding="utf-8")
        out = jd.run_stub(out_dir / safe_name(model_id), out_dir, record=True)
        jd.write_judge(out_dir / safe_name(model_id), out)
    # calibration: a person who agrees with the stub on six rows in seven
    cal_csv = root / "calibration.csv"
    jc.export(out_dir, cal_csv, [], 60, seed)
    rows = list(csv.DictReader(open(cal_csv, newline="", encoding="utf-8")))
    judged = {r["id"]: r for r in jc._judged_rows(out_dir, set())}
    for k, r in enumerate(rows):
        js = judged[r["id"]]["judge_score"]
        r["human_score"] = str(min(4, js + 1) if k % 7 == 6 else js)
    with open(cal_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=jc.FIELDS)
        w.writeheader()
        w.writerows(rows)
    cal = jc.import_csv(out_dir, cal_csv)
    return {"exam_root": root / "exam", "tasks_dir": tasks_dir, "manifest": manifest,
            "models": dict(JUDGED), "calibration_csv": cal_csv, "calibration": cal}


def frozen_report(root: Path, path: Path, title: str = "Fixture board") -> Path:
    import report_lm_eval as report
    out_dir = root / "results" / "full"
    runs = report.load_results(out_dir)
    cal_path = out_dir / "judge_calibration.json"
    cal = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else None
    return report.build_report(runs, path, title, calibration=cal,
                               taint=TAINT, parents=PARENTS,
                               judge_identity={"provider": "stub", "model": "overlap-v1",
                                               "id": "stub/overlap-v1", "family": "stub"})


def build(root: Path, seed: int = SEED, diagnose: bool = True,
          report: Path | None = None, judged: bool = True) -> dict:
    """Write the whole tree under root/results/full and return a manifest the
    tests read: models and their planted policies, the shared documents per
    task (with doc_hash and the question text as diagnose.py prints it), and
    which model is the odd one out for each planted inconsistency."""
    root = Path(root)
    out_dir = root / "results" / "full"
    docs = {task: make_docs(task, seed) for task in TASKS}
    models: dict[str, dict] = {}
    for model_id, policy, kind, params, tasks, overrides in MODELS:
        write_model_meta(out_dir, model_id, kind, params)
        for task in tasks:
            write_task(out_dir, model_id, kind, params, overrides.get(task, policy), task,
                       docs[task], seed)
        models[model_id] = {"policy": policy, "kind": kind, "params": params,
                            "tasks": list(tasks), "overrides": dict(overrides),
                            "safe": safe_name(model_id), "dir": out_dir / safe_name(model_id)}
    manifest = {
        "root": root, "out_dir": out_dir, "models": models,
        "docs": {t: [{k: it[k] for k in ("doc_hash", "q", "group", "correct", "subtask")}
                     for it in items] for t, items in docs.items()},
        "nodiag": NODIAG,
        "miscount": {**MISCOUNT, "logged": TASKS[MISCOUNT["task"]]["n"]},
        "stale": {**STALE, "n": TASKS[STALE["task"]]["n"]},
        "diagnosed": write_diagnoses(root) if diagnose else [],
        "judged": write_judged(root, out_dir, seed) if judged else None,
        "taint": dict(TAINT), "parents": dict(PARENTS),
        "report": None,
    }
    if report is not None:
        manifest["report"] = frozen_report(root, Path(report))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="directory that gains results/full/…")
    ap.add_argument("--report", type=Path, default=None,
                    help="also freeze the single-file dashboard here")
    ap.add_argument("--no-diagnose", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    m = build(a.out, seed=a.seed, diagnose=not a.no_diagnose, report=a.report)
    n_files = sum(1 for _ in m["out_dir"].rglob("*.json*"))
    print(f"wrote {len(m['models'])} models, {n_files} files under {m['out_dir']}")
    if m["diagnosed"]:
        print("diagnosed:", ", ".join(m["diagnosed"]))
    if m["report"]:
        print("report:", m["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
