#!/usr/bin/env python3
"""Grade the exam answers with a pinned API judge, in batch; write judge.json.

    python scripts/judge.py results/full -m local/my-ckpt --wait     # JUDGE_* from .env
    python scripts/judge.py results/full --stub                       # tests and dry runs

Reads the harness's --log_samples output for the exam tasks (exam_<topic>,
fr_control_mmlu; scripts/exam_build.py) and writes results/full/<model>/
judge.json in the same shape-and-place pattern as diagnose.json, so the
dashboard's _beside() loader and the service's freshness key pick it up
unchanged. The service runs this as the last step of a suite=judged job:
the grading batch is SUBMITTED inside the GPU lock (seconds, no GPU) and
FINISHED by service/llm_poller.py when the provider completes it — a judge
that waits on an API must never hold the card.

The judging contract, unchanged in substance from the local judge:

  rubric     single answers graded 0–4 against a written, versioned rubric
             with anchors (eval_tasks/fr/rubrics/), never pairwise — that is
             where position bias lives. Length is in the rubric explicitly
             and score-vs-length is reported per topic
  pinned     JUDGE_MODEL must be a DATED model id, not a floating alias.
             Provider, model, prompt sha, rubric sha and batch id ride in
             every judge.json and in every dataset's provenance
  local      JUDGE_PROVIDER=local (a vLLM server on the box) cannot be pinned
             at all — its id is whatever was typed at launch. It is not
             refused; it runs PROVISIONAL: judge.json says so with the base
             URL, the served id and the weights, the reason joins the
             preliminary reasons, and the page greys it, never ranks it and
             leaves it out of every average. No flag turns that off
  canary     a fixed set of thirty answer scripts with known human marks
             (eval_tasks/fr/canary.jsonl) is re-graded at the start of every
             run. judge.json records the canary's mean absolute deviation
             from those marks and from the previous run's; movement past
             JUDGE_CANARY_MAX_DRIFT marks the run preliminary with the
             reason stated. This replaces the byte-identical determinism a
             local greedy judge gave us: a vendor updating the model behind
             the id would otherwise silently re-base every score
  family     a judge never grades a model of its own family, and its
             PROVIDER must differ from the exam writer's and the generator's
             — a loop whose questions, grades and training data all come
             from one family grades itself. ALLOW_SINGLE_PROVIDER_LOOP=1
             overrides for a trial and stamps every judged score with it
  halves     each item records its qid's half; the published topic score is
             the report half; justifications (what went wrong, in words) are
             recorded per item for the step that picks a topic — which may
             read the diagnose half only
  --stub     a deterministic overlap stand-in for tests; the fake backend is
             what CI uses for the batch path
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import diagnose as dx  # noqa: E402
from exam_build import ALL_TASKS, CONTROL_TASK  # noqa: E402

RUBRIC_DIR = REPO / "eval_tasks" / "fr" / "rubrics"
CANARY_PATH = REPO / "eval_tasks" / "fr" / "canary.jsonl"
CANARY_HISTORY = "judge_canary_history.jsonl"      # at the root of results/full
PROMPT_VERSION = 2
MAX_SCORE = 4
CORRECT_AT = 3           # an open-ended answer scoring >= this counts as "knew it"
LENGTH_BUCKETS = [(0, 20, "≤20 words"), (21, 50, "21–50"), (51, 120, "51–120"), (121, 10**9, ">120")]
DATED = re.compile(r"(\d{8}|\d{4}-\d{2}-\d{2})")   # a pinned model id carries its date

PROMPT = """You are grading ONE answer to ONE question against a rubric. Read the rubric, the question, the reference answer, and the candidate answer. Reply with one JSON object and nothing else: {{"score": <integer 0-4>, "justification": <one or two sentences on what the answer got right or wrong against the rubric — about the ANSWER, never quoting the question>}}.

RUBRIC
{rubric}

QUESTION
{question}

REFERENCE (what a full-marks answer contains)
{reference}

CANDIDATE ANSWER
{answer}"""


def family(model_id: str) -> str:
    """Same rule as the dashboard's `family` field: the first alphanumeric run
    of the last path segment — 'llama' from 'Llama-3.1-8B-Instruct', 'claude'
    from 'claude-sonnet-4-5-20250929', 'gpt' from 'gpt-4.1-2025-04-14'."""
    return re.split(r"[^a-z0-9]", model_id.split("/")[-1].lower())[0]


def rubric_for(task: str) -> tuple[str, str, str]:
    """(text, sha256, version). Every exam topic is graded with the exam
    rubric; the control set with the factual one — it asks for a fact, and the
    gold option is the reference."""
    cat = "factual_accuracy" if task == CONTROL_TASK else "exam"
    text = (RUBRIC_DIR / f"{cat}.md").read_text(encoding="utf-8")
    m = re.search(r"\(version (\d+)\)", text)
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest(), (m.group(1) if m else "?")


def build_prompt(rubric: str, question: str, reference: str, answer: str) -> str:
    return PROMPT.format(rubric=rubric.strip(), question=question.strip(),
                         reference=reference.strip(), answer=(answer or "").strip() or "(empty)")


def prompt_sha() -> str:
    return hashlib.sha256(f"v{PROMPT_VERSION}\n{PROMPT}".encode("utf-8")).hexdigest()


def words(s: str) -> int:
    return len((s or "").split())


def parse_grade(text: str) -> tuple[int | None, str]:
    """(score, justification) from the judge's reply; a bare digit still counts."""
    from service import llm
    obj = llm.extract_json(text or "")
    if isinstance(obj, dict) and obj.get("score") is not None:
        try:
            s = int(obj["score"])
            if 0 <= s <= MAX_SCORE:
                return s, str(obj.get("justification") or "").strip()[:600]
        except (TypeError, ValueError):
            pass
    m = re.match(r"\s*(?:score\s*[:=]?\s*)?([0-4])\b", text or "", re.I)
    return (int(m.group(1)) if m else None), ""


# ---------------------------------------------------------------------------
# identity and refusals
# ---------------------------------------------------------------------------

def identity() -> dict:
    """{provider, model, id, family} of the judge this server is configured with."""
    from service import config
    model = config.JUDGE_MODEL
    if model == "stub":
        return {"provider": "stub", "model": "overlap-v1", "id": "stub/overlap-v1", "family": "stub"}
    provider = config.JUDGE_PROVIDER
    return {"provider": provider, "model": model,
            "id": f"{provider}/{model}" if provider and model else (model or ""),
            "family": family(model) if model else ""}


def judge_family(ident: dict, stamp: dict | None = None) -> str:
    """The family the same-family rule compares against. A local judge's is
    its weights' when the server names them: the served id ('chat') is
    whatever someone typed at launch and says nothing about the family."""
    weights = (stamp or {}).get("weights")
    return family(weights) if weights else ident["family"]


def provider_clash() -> str:
    """The exam writer's or generator's provider, when it equals the judge's."""
    from service import config
    p = config.JUDGE_PROVIDER
    if not p or p == "fake":
        return ""
    if p == config.EXAM_PROVIDER:
        return "exam writer"
    if p == config.LLM_PROVIDER:
        return "generator"
    return ""


def single_provider_loop() -> bool:
    from service import config
    return bool(provider_clash()) and config.ALLOW_SINGLE_PROVIDER_LOOP


def blocked() -> str:
    """'' when a judged run can be graded, else the reason — shown on the page
    rather than crashing the container (the exam and the board still work)."""
    from service import config, llm
    if not config.JUDGE_MODEL:
        return ("no judge is configured on this server (JUDGE_MODEL is unset) — the judged "
                "suite is off")
    if config.JUDGE_MODEL == "stub":
        return ""
    p = config.JUDGE_PROVIDER
    if not p:
        return "JUDGE_PROVIDER is unset — the judge is an API call and needs one"
    if p not in llm.PROVIDERS:
        return f"JUDGE_PROVIDER={p!r} is not one of {', '.join(llm.PROVIDERS)}"
    # a local server's id cannot be pinned, so for `local` this refusal
    # becomes a stamp instead (assemble): it runs, and nothing it writes counts
    if p not in ("fake", "local") and not DATED.search(config.JUDGE_MODEL):
        return (f"JUDGE_MODEL={config.JUDGE_MODEL!r} is a floating alias, not a dated model id — "
                f"pin it (e.g. claude-sonnet-4-5-20250929, gpt-4.1-2025-04-14) or a vendor "
                f"update silently re-bases every score")
    if llm.needs_key(p) and not config.JUDGE_API_KEY:
        return "JUDGE_API_KEY is unset — put it in .env, never in docker-compose.yml"
    clash = provider_clash()
    if clash and not config.ALLOW_SINGLE_PROVIDER_LOOP:
        return (f"the judge's provider ({p}) is the same as the {clash}'s — a loop whose "
                f"questions, grades and training data come from one family grades itself. "
                f"Use a different provider, or set ALLOW_SINGLE_PROVIDER_LOOP=1 for a trial "
                f"(every judged score is then stamped with it)")
    return ""


# ---------------------------------------------------------------------------
# the stub — a deterministic stand-in for tests and dry runs
# ---------------------------------------------------------------------------

_STOP = set("the a an of to in and or is are was were be it its this that for on with as by at from "
            "which who what when where how not no yes".split())


def _content(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _STOP}


class StubGrader:
    """Content-word recall of the reference plus the length clause. NOT a
    judge: it exists so the plumbing (shapes, hashes, halves, the canary, the
    control join, the calibration round trip) can be tested without a
    provider. Every file it writes says so."""
    id = "stub/overlap-v1"

    @staticmethod
    def grade(prompt: str) -> tuple[int, str]:
        ref = prompt.split("REFERENCE (what a full-marks answer contains)\n", 1)[1]
        ref, ans = ref.split("\n\nCANDIDATE ANSWER\n", 1)
        a, r = _content(ans), _content(ref)
        if not a or not r or ans.strip() == "(empty)":
            return 0, "no answer, or nothing from the reference in it"
        recall = len(a & r) / len(r)
        score = 4 if recall >= 0.8 else 3 if recall >= 0.6 else 2 if recall >= 0.35 \
            else 1 if recall > 0 else 0
        why = f"the answer covers {recall:.0%} of the reference's substance"
        if score >= 3 and words(ans) > max(60, 3 * words(ref)):
            score -= 1
            why += "; it is far longer than the reference, so the length clause costs a point"
        return score, why


def stub_results(requests) -> dict:
    from service import llm
    out = {}
    for r in requests:
        s, j = StubGrader.grade(r.user)
        out[r.custom_id] = llm.Result(text=json.dumps({"score": s, "justification": j}))
    return out


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


def load_canary() -> list[dict]:
    return [json.loads(x) for x in CANARY_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# a run: plan the requests, then assemble the file from the results
# ---------------------------------------------------------------------------

def plan_requests(model_dir: Path, judge_family: str) -> tuple[list, dict]:
    """Every grading request for one model — the canary first, then every
    answer — plus the plan the results are assembled against later (the
    poller may be in another process by then)."""
    from service import llm
    model_id = model_dir.name.replace("__", "/", 1)
    tasks_present = [t for t in ALL_TASKS if any(model_dir.glob(f"{t}_*shot"))]
    plan = {"model": model_id, "model_dir": str(model_dir), "tasks": {}, "canary": [],
            "skipped": None}
    if not tasks_present:
        return [], plan
    if family(model_id) == judge_family:
        plan["skipped"] = f"not judged — same family as judge ({judge_family})"
        return [], plan
    reqs = []
    rub_exam = rubric_for("exam_x")[0]
    for c in load_canary():
        cid = f"canary:{c['id']}"
        reqs.append(llm.Request(custom_id=cid, system="", max_tokens=200, json=True,
                                user=build_prompt(rub_exam, c["prompt"], c["reference"], c["answer"]),
                                meta={"kind": "canary", "id": c["id"], "human_score": c["human_score"]}))
        plan["canary"].append({"cid": cid, "id": c["id"], "human_score": c["human_score"]})
    mc = mmlu_outcomes(model_dir) if CONTROL_TASK in tasks_present else {}
    safe = model_dir.name
    plan["answer_stats"] = {}
    for task in tasks_present:
        rubric, _, _ = rubric_for(task)
        items = []
        # what the model actually wrote, in aggregate. A model that wrote
        # nothing on a topic has not revealed a gap in that topic, and the
        # step that picks what to train must be able to say so.
        stats = {"n": 0, "empty": 0, "short": 0, "words": 0}
        seen: set[str] = set()
        for i, rec in enumerate(sorted(_records(model_dir, task), key=lambda r: str(r.get("doc_hash")))):
            doc = rec.get("doc") or {}
            ans = _answer(rec)
            qid = doc.get("qid")
            cid = f"judge:{safe}:{task}:{i}"
            reqs.append(llm.Request(custom_id=cid, system="", max_tokens=300, json=True,
                                    user=build_prompt(rubric, doc.get("prompt", ""),
                                                      doc.get("reference", ""), ans),
                                    meta={"kind": "judge", "task": task, "qid": qid}))
            stats["n"] += 1
            stats["words"] += words(ans)
            norm = re.sub(r"\s+", " ", (ans or "").strip().lower())
            if not norm:
                stats["empty"] += 1
            elif words(ans) < 3:
                stats["short"] += 1
            seen.add(norm)
            item = {"cid": cid, "doc_hash": rec.get("doc_hash"), "id": doc.get("id"), "qid": qid,
                    "half": ("diagnose" if task == CONTROL_TASK
                             else dx.split_of(qid) if qid else None),
                    "category": doc.get("category"), "answer_words": words(ans)}
            if task == CONTROL_TASK:
                item["mmlu_doc_hash"] = doc.get("mmlu_doc_hash")
                item["mc_right"] = mc.get(doc.get("mmlu_doc_hash"))
            items.append(item)
        stats["distinct"] = len(seen)
        stats["mean_words"] = round(stats["words"] / stats["n"], 2) if stats["n"] else 0
        plan["answer_stats"][task] = stats
        plan["tasks"][task] = items
    return reqs, plan


def canary_stats(scores: dict[str, int | None], canary: list[dict],
                 previous: dict[str, int] | None, threshold: float) -> dict:
    """Mean absolute deviation of the canary grades from the human marks and
    from the previous run's grades. Drift is movement between runs."""
    pairs = [(scores.get(c["id"]), c["human_score"]) for c in canary]
    graded = [(s, h) for s, h in pairs if s is not None]
    mad_h = round(sum(abs(s - h) for s, h in graded) / len(graded), 4) if graded else None
    mad_p = None
    if previous:
        both = [(scores[k], previous[k]) for k in previous if scores.get(k) is not None]
        mad_p = round(sum(abs(a - b) for a, b in both) / len(both), 4) if both else None
    drifted = mad_p is not None and mad_p > threshold
    return {"n": len(canary), "graded": len(graded), "mad_vs_human": mad_h,
            "mad_vs_previous": mad_p, "threshold": threshold, "drifted": drifted,
            "scores": {c["id"]: scores.get(c["id"]) for c in canary},
            "human": {c["id"]: c["human_score"] for c in canary}}


def previous_canary(results_root: Path, judge_id: str) -> dict[str, int] | None:
    p = Path(results_root) / CANARY_HISTORY
    if not p.exists():
        return None
    last = None
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("judge_id") == judge_id:
            last = row
    return {k: v for k, v in (last or {}).get("scores", {}).items() if v is not None} or None


def record_canary(results_root: Path, judge_id: str, model: str, stats: dict) -> None:
    p = Path(results_root) / CANARY_HISTORY
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"judge_id": judge_id, "model": model, "at": time.time(),
                             "scores": stats["scores"], "mad_vs_human": stats["mad_vs_human"],
                             "mad_vs_previous": stats["mad_vs_previous"]}) + "\n")


def assemble(plan: dict, results: dict, ident: dict, batch_id: str, results_root: Path,
             threshold: float, caveat: bool, record: bool = True) -> dict:
    """judge.json from a plan and the provider's results. Deterministic for a
    deterministic grader (sorted keys, no timestamps in the body)."""
    from service import llm
    tasks_present = list(plan["tasks"])
    head = {"judge": {"id": ident["id"], "provider": ident["provider"], "model": ident["model"],
                      "family": ident["family"], "batch_id": batch_id,
                      "prompt_sha256": prompt_sha(), "prompt_version": PROMPT_VERSION,
                      "stub": ident["provider"] == "stub", "single_provider_loop": caveat,
                      "rubrics": {t: {"sha256": rubric_for(t)[1], "version": rubric_for(t)[2]}
                                  for t in tasks_present}},
            "model": plan["model"], "split_salt": dx.SPLIT_SALT, "correct_at": CORRECT_AT}
    # a local judge: the stamp recorded at submit time (with what the server
    # said it serves), or at the least the mark itself — never nothing
    stamp = plan.get("provisional") or llm.local_mark(ident["provider"], ident["model"], "graded")
    if stamp:
        head["judge"].update(stamp)
        head["judge"]["family"] = judge_family(ident, stamp)
    if plan.get("skipped"):
        return {**head, "skipped": plan["skipped"], "tasks": {}}
    # the canary first: has the judge moved since last time?
    cscores = {}
    for c in plan["canary"]:
        res = results.get(c["cid"])
        cscores[c["id"]] = parse_grade(res.text)[0] if res and not res.error else None
    prev = previous_canary(results_root, ident["id"]) if plan["canary"] else None
    canary = canary_stats(cscores, load_canary(), prev, threshold) if plan["canary"] else None
    if canary and record:
        record_canary(results_root, ident["id"], plan["model"], canary)
    # provisional is a preliminary reason like any other: the page greys it,
    # never ranks it and keeps it out of every average by the same mechanism
    reasons = [stamp["provisional_reason"]] if stamp else []
    if canary and canary["drifted"]:
        reasons.append(f"the judge moved: canary grades differ from the previous run by "
                       f"{canary['mad_vs_previous']} points on average (limit {threshold})")
    if canary and canary["graded"] < canary["n"]:
        reasons.append(f"{canary['n'] - canary['graded']} canary scripts were not graded")
    tasks: dict[str, dict] = {}
    for task, items_meta in plan["tasks"].items():
        items = []
        for m in items_meta:
            res = results.get(m["cid"])
            score, just = parse_grade(res.text) if res and not res.error else (None, "")
            it = {k: v for k, v in m.items() if k != "cid"}
            it["score"] = int(score) if score is not None else 0
            it["graded"] = score is not None
            it["justification"] = just
            items.append(it)
        if not items:
            continue
        dist = collections.Counter(str(it["score"]) for it in items)
        by_len: dict[str, list[int]] = collections.defaultdict(list)
        for it in items:
            by_len[length_bucket(it["answer_words"])].append(it["score"])
        rep = [it["score"] for it in items if it["half"] == "report"]
        dia = [it["score"] for it in items if it["half"] == "diagnose"]

        def _dist(xs):
            c = collections.Counter(xs)
            return {str(k): c.get(k, 0) for k in range(MAX_SCORE + 1)}
        t = {"n": len(items), "mean": round(sum(it["score"] for it in items) / len(items), 4),
             "max": MAX_SCORE, "ungraded": sum(1 for it in items if not it["graded"]),
             "n_report": len(rep), "score_report": round(sum(rep) / len(rep), 4) if rep else None,
             "n_diagnose": len(dia),
             "score_diagnose": round(sum(dia) / len(dia), 4) if dia else None,
             "dist": {str(k): dist.get(str(k), 0) for k in range(MAX_SCORE + 1)},
             # per half, so a before/after comparison can carry a standard error
             "dist_report": _dist(rep), "dist_diagnose": _dist(dia),
             "score_vs_length": [{"bucket": label, "n": len(by_len[label]),
                                  "mean": round(sum(by_len[label]) / len(by_len[label]), 4)}
                                 for _, _, label in LENGTH_BUCKETS if by_len.get(label)],
             "items": items}
        stats = (plan.get("answer_stats") or {}).get(task)
        if stats:
            t["answers"] = stats
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
                    c["knew" if it["score"] >= CORRECT_AT else "didnt"] += 1
            t["control"] = ctl
        tasks[task] = t
    return {**head, "canary": canary, "preliminary_reasons": reasons, "tasks": tasks}


def write_judge(model_dir: Path, out: dict, dest: Path | None = None) -> Path:
    d = dest or model_dir
    d.mkdir(parents=True, exist_ok=True)
    p = d / "judge.json"
    p.write_text(json.dumps(out, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return p


def run_stub(model_dir: Path, results_root: Path | None = None, record: bool = False,
             threshold: float = 0.5) -> dict | None:
    """Grade with the stub, in process. The fixture and the tests."""
    ident = {"provider": "stub", "model": "overlap-v1", "id": "stub/overlap-v1", "family": "stub"}
    reqs, plan = plan_requests(model_dir, "stub")
    if not plan["tasks"] and not plan.get("skipped"):
        return None
    return assemble(plan, stub_results(reqs), ident, "stub", results_root or model_dir.parent,
                    threshold, caveat=False, record=record)


def start_run(model_dir: Path, results_root: Path) -> dict:
    """The service's entry point, called inside the GPU lock: plan, submit
    the batch (seconds), return. The poller finishes it. With JUDGE_MODEL=stub
    the file is written here and now."""
    from service import config, db, llm
    ident = identity()
    if config.JUDGE_MODEL == "stub":
        out = run_stub(model_dir, results_root, record=True, threshold=config.JUDGE_CANARY_MAX_DRIFT)
        if out is None:
            return {"mode": "stub", "written": False}
        write_judge(model_dir, out)
        return {"mode": "stub", "written": True, "skipped": out.get("skipped")}
    why = blocked()
    if why:
        raise RuntimeError(why)
    backend = llm.client("judge")
    stamp = llm.provisional(backend, "graded")
    reqs, plan = plan_requests(model_dir, judge_family(ident, stamp))
    if stamp:
        plan["provisional"] = stamp
    if plan.get("skipped"):
        write_judge(model_dir, assemble(plan, {}, ident, "", results_root,
                                        config.JUDGE_CANARY_MAX_DRIFT, single_provider_loop()))
        return {"mode": "batch", "written": True, "skipped": plan["skipped"]}
    if not reqs:
        return {"mode": "batch", "written": False}
    bid = backend.submit(reqs)
    rid = db.judge_run_create(plan["model"], bid, len(reqs), ident["id"], json.dumps(plan))
    db.batch_add(bid, "judge", rid, len(reqs), backend.name, backend.model)
    return {"mode": "batch", "written": False, "batch_id": bid, "run_id": rid, "n": len(reqs)}


def finish_run(run: dict, results: dict, results_root: Path) -> Path:
    """The poller's half: results in, judge.json out."""
    from service import config
    plan = json.loads(run["plan"])
    ident = identity()
    out = assemble(plan, results, ident, run["batch_id"], results_root,
                   config.JUDGE_CANARY_MAX_DRIFT, single_provider_loop())
    return write_judge(Path(plan["model_dir"]), out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("-m", "--model", action="append", default=[])
    ap.add_argument("--stub", action="store_true",
                    help="deterministic overlap stand-in; tests and dry runs only")
    ap.add_argument("--wait", action="store_true",
                    help="API judge: submit the batch and poll until it completes")
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="write judge.json under this directory instead of beside the results")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    if not a.results.is_dir():
        print(f"no such directory: {a.results}", file=sys.stderr)
        return 2
    from service import config, llm
    if not a.stub:
        why = blocked()
        if why or config.JUDGE_MODEL == "stub":
            print(why or "JUDGE_MODEL=stub: pass --stub", file=sys.stderr)
            return 2
        if not a.wait:
            print("an API judge is asynchronous: pass --wait here, or submit through the "
                  "service (suite=judged) and let its poller finish the batch", file=sys.stderr)
            return 2
    want = {m.replace("/", "__") for m in a.model}
    n = 0
    ident = {"provider": "stub", "model": "overlap-v1", "id": "stub/overlap-v1",
             "family": "stub"} if a.stub else identity()
    backend, stamp = None, {}
    if not a.stub:
        try:
            backend = llm.client("judge")
        except llm.LLMError as e:          # a local server that is down, or serves another id
            print(e, file=sys.stderr)
            return 2
        stamp = llm.provisional(backend, "graded")
    for d in sorted(p for p in a.results.iterdir() if p.is_dir()):
        if want and d.name not in want:
            continue
        if a.stub:
            out = run_stub(d, a.results, record=True)
        else:
            reqs, plan = plan_requests(d, judge_family(ident, stamp))
            if stamp:
                plan["provisional"] = stamp
            if not plan["tasks"] and not plan.get("skipped"):
                continue
            results, bid = {}, ""
            if reqs:
                bid = backend.submit(reqs)
                while True:
                    state, detail = backend.status(bid)
                    if state == "done":
                        break
                    if state == "failed":
                        print(f"{d.name}: batch {bid} failed: {detail}", file=sys.stderr)
                        break
                    time.sleep(a.poll)
                results = backend.fetch(bid) if state == "done" else {}
            out = assemble(plan, results, ident, bid, a.results, config.JUDGE_CANARY_MAX_DRIFT,
                           single_provider_loop())
        if out is None:
            continue
        write_judge(d, out, (a.out / d.name) if a.out else None)
        n += 1
        if not a.quiet:
            if out.get("skipped"):
                print(f"{d.name:46} {out['skipped']}")
            else:
                bits = [f"{t}={v['score_report'] if v['score_report'] is not None else v['mean']:.2f}/4"
                        f"(n={v['n']})" for t, v in sorted(out["tasks"].items())]
                c = out.get("canary") or {}
                print(f"{d.name:46} canary MAD {c.get('mad_vs_human')} vs human, "
                      f"{c.get('mad_vs_previous')} vs previous · {' '.join(bits)}")
    if not a.quiet:
        print(f"\nwrote judge.json for {n} model(s) · judge {ident['id']}"
              + (" · STUB — not a judgement" if a.stub else "")
              + (f" · PROVISIONAL — {stamp['provisional_reason']}" if stamp else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
