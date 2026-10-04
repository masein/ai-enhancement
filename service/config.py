"""Configuration — everything is an environment variable with a sane default.

The service is meant to be launched from the benchmarks directory (the one that
contains results/, eval_tasks/, logs/), like the CLI script. BENCH_ROOT overrides
that if you launch it from elsewhere.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

BENCH_ROOT = Path(os.environ.get("BENCH_ROOT", os.getcwd())).resolve()

RESULTS_ROOT = Path(os.environ.get("OUT_ROOT", BENCH_ROOT / "results"))
OUT_DIR = RESULTS_ROOT / "full"          # service runs are always full-mode (no --limit)
EVAL_TASKS_DIR = Path(os.environ.get("EVAL_TASKS_DIR", BENCH_ROOT / "eval_tasks"))
LOGS_DIR = Path(os.environ.get("LOGS", BENCH_ROOT / "logs"))
DB_PATH = Path(os.environ.get("SERVICE_DB", BENCH_ROOT / "service.sqlite3"))

TITLE = os.environ.get("TITLE", "Team model benchmark")

# GPU etiquette — same numbers and reasoning as scripts/run_benchmarks.sh.
SEED = int(os.environ.get("SEED", "1234"))
MAX_JOB_GB = float(os.environ.get("MAX_JOB_GB", "10"))     # logits+weights+overhead budget
FREE_MARGIN_MIB = int(os.environ.get("FREE_MARGIN_MIB", "512"))
# 12f.0: the server's free disk (the results folder's filesystem, and /):
# amber on the status dot under DISK_AMBER_GB, red under DISK_RED_GB — and
# below red, a run does not start
DISK_AMBER_GB = float(os.environ.get("DISK_AMBER_GB", "10"))
DISK_RED_GB = float(os.environ.get("DISK_RED_GB", "3"))
GPU_POLL_S = int(os.environ.get("GPU_POLL_S", "60"))
GPU_WAIT_MAX_S = int(os.environ.get("GPU_WAIT_MAX_S", str(6 * 3600)))
TASK_TIMEOUT_S = int(os.environ.get("TASK_TIMEOUT_S", str(3 * 3600)))
# 15.3: a task that writes its answers (DeviceMark's, the generative three) is
# given a limit sized for it — its answers still to write, each at the cap, at
# the model's measured pace (tokens a second, pace.json), with LIMIT_HEADROOM
# — never under TASK_TIMEOUT_S. Before any run has measured the model,
# PACE_GUESS_TOK_S. TASK_TIMEOUT_S=0 is no limit at all (a rented GPU's run)
PACE_GUESS_TOK_S = float(os.environ.get("PACE_GUESS_TOK_S", "20"))
LIMIT_HEADROOM = float(os.environ.get("LIMIT_HEADROOM", "1.5"))

# Submission guardrails.
MAX_PARAMS_B = float(os.environ.get("MAX_PARAMS_B", "4"))  # reject >4B params (bf16 ≈ 8 GB weights)
SUBMIT_TOKEN = os.environ.get("SUBMIT_TOKEN", "")          # empty = no token required

# Artifact storage: checkpoints uploaded directly to this box instead of the HF
# Hub. They are ordinary model directories under ARTIFACTS_DIR and get evaluated
# as `local/<name>`. Quotas exist because friends iterate and disks do not.
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", BENCH_ROOT / "artifacts"))
ARTIFACT_MAX_GB = float(os.environ.get("ARTIFACT_MAX_GB", "30"))    # per upload (16b: was 8)
ARTIFACT_QUOTA_GB = float(os.environ.get("ARTIFACT_QUOTA_GB", "150"))  # total dir
# 16b.1: a model added from a browser — a .gguf, or a folder as a .zip — kept
# under BENCH_ROOT, so the host's GGUF worker sees the same path. One file at
# most UPLOAD_MAX_GB; uploads and model folders together at most
# ARTIFACT_QUOTA_GB (16b decision 1: 30 and 150); and none may leave less than
# UPLOAD_FREE_GB free on the disk (masein, 4 Oct: /home also holds HF_HOME)
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", BENCH_ROOT / "uploads"))
UPLOAD_MAX_GB = float(os.environ.get("UPLOAD_MAX_GB", "30"))
UPLOAD_FREE_GB = float(os.environ.get("UPLOAD_FREE_GB", "50"))

# ---------------------------------------------------------------------------
# Custom model code (transformers' trust_remote_code).
#
# A checkpoint whose config.json carries an `auto_map` cannot be loaded without
# executing Python that came with the upload. That is a real capability for a
# team training custom architectures and a real risk on a box that also runs
# other people's week-long jobs, so it is off unless three things line up:
#
#   ALLOW_REMOTE_CODE=1   the operator turned it on for this server
#   EVAL_USER set         and there is an unprivileged user to run it as
#
# EVAL_USER is not optional on purpose. Without it the uploaded code would run
# as root beside everyone's results and the server's Hugging Face token;
# refusing to start that way is the difference between a gate and a nod.
#
# Deliberately NOT gated on a submit token. The tailnet is already the auth
# boundary — the service binds to the Tailscale IP, so anyone who can upload an
# artifact was invited onto the network by hand. A shared secret on top would
# distinguish 'on the tailnet' from 'on the tailnet and knows a string', which
# is a distinction shared secrets do not keep for long (they end up in shell
# history and chat), while charging every friend a --token on every call. The
# protections that do work are below: uploads only, dropped privileges, the HF
# token withheld, and every .py hashed into the run's provenance.
# Uploaded artifacts only — a Hub repo with auto_map stays refused outright,
# because "someone on the tailnet uploaded it" is the only trust signal we have.
# REMOTE_CODE_SHAS, when set, is an allowlist: only those exact .py files run.
# ---------------------------------------------------------------------------
ALLOW_REMOTE_CODE = os.environ.get("ALLOW_REMOTE_CODE", "0") == "1"
EVAL_USER = os.environ.get("EVAL_USER", "").strip()        # e.g. "benchjob"
REMOTE_CODE_SHAS = {s.strip() for s in
                    os.environ.get("REMOTE_CODE_SHAS", "").split(",") if s.strip()}


def remote_code_blocked() -> str:
    """'' when remote code may run on this server, else the reason it may not."""
    if not ALLOW_REMOTE_CODE:
        return ("this server has ALLOW_REMOTE_CODE off, so uploaded model code is "
                "never executed")
    if not EVAL_USER:
        return ("remote code needs EVAL_USER set to an unprivileged account — "
                "without it the uploaded code would run as root next to the "
                "results tree and the server's Hugging Face token")
    try:            # a name that does not resolve is the same as no user at all
        import pwd
        pwd.getpwnam(EVAL_USER)
    except (ImportError, KeyError):
        return (f"EVAL_USER={EVAL_USER!r} does not exist on this machine, so the "
                f"job could not drop privileges and would run as root. Create the "
                f"account (the image ships 'benchjob') or point EVAL_USER at one "
                f"that exists")
    return ""

# 11l: a reasoning model (its chat template opens a <think> block) writes its
# reasoning before the answer, and the exam's 256-token answer budget ran out
# inside it on every one of run #60's 3,730 questions. Its judged runs get
# this many tokens per answer instead; every other model keeps 256, so
# nothing it scored moves. What was used is recorded with the grades.
REASONING_MAX_GEN_TOKS = int(os.environ.get("REASONING_MAX_GEN_TOKS", "2048"))
# 12a.4: Everyday tasks need more. In #70, 8 of Qwen3-1.7B's 45 quick-maths
# answers were empty — its thinking went past 2,048 tokens — and Qwen3-0.6B's
# 5. A reasoning model's everyday answers get this many; the exam's stay above
EVERYDAY_REASONING_MAX_GEN_TOKS = int(os.environ.get("EVERYDAY_REASONING_MAX_GEN_TOKS", "4096"))

# 12d.1: the Playground's chat engine. "hf" loads through lm_eval's own HF
# loader (the one runs use); "fake" streams canned text (tests, no model).
CHAT_BACKEND = os.environ.get("CHAT_BACKEND", "hf")
# a loaded model is unloaded after this long with no message
CHAT_IDLE_UNLOAD_S = int(os.environ.get("CHAT_IDLE_UNLOAD_S", "600"))
# below this many parameters a model may answer on the CPU while a run holds
# the GPU. To be measured on the server after deploy: if a 0.6B model is
# under about 3 words a second there, lower it (brief 12d §2)
CHAT_CPU_MAX_PARAMS_B = float(os.environ.get("CHAT_CPU_MAX_PARAMS_B", "1.0"))
# free GPU memory a model needs beyond its weights (bf16) before it loads —
# activations, the KV cache of one reply, CUDA's context. To be measured on
# the server after deploy (brief 12d §2)
CHAT_GPU_MARGIN_GB = float(os.environ.get("CHAT_GPU_MARGIN_GB", "2.0"))
# how long a run waits for chat to let go of the GPU before taking its lock
CHAT_YIELD_WAIT_S = int(os.environ.get("CHAT_YIELD_WAIT_S", "60"))
# the chats kept listed per person
CHAT_LIST_N = 50

# 12f.1: a model served elsewhere. Questions go this many at a time —
# llama-server starts with one slot, so one unless it was started with more
SERVED_CONCURRENCY = int(os.environ.get("SERVED_CONCURRENCY", "1"))
# a server that stops answering is retried this long, then the run stops
SERVED_RETRY_S = int(os.environ.get("SERVED_RETRY_S", "120"))
# one answer may take this long: a thinking answer on a CPU-offloaded model
SERVED_TIMEOUT_S = int(os.environ.get("SERVED_TIMEOUT_S", "900"))

# The benchmark suite — one place, mirrored from run_benchmarks.sh. quick is for
# iteration (minutes); full is the comparable number. Both write into the same
# tree, so a quick run later "upgrades" to full by running only the missing tasks.
NFEWSHOT = {
    "mmlu": 5, "hellaswag": 5, "arc_challenge": 5, "arc_easy": 5,
    "winogrande": 5, "piqa": 0, "truthfulqa_mc2": 0, "gsm8k": 5,
    "mmlu_perm": 5,          # the control poses MMLU exactly as mmlu does, shots included
    # 12h.1: MMLU-Pro is 5-shot chain of thought, as the harness and its
    # published numbers pose it; IFEval and MATH-500 are asked once, cold
    "ifeval": 0, "mmlu_pro": 5, "hendrycks_math500": 0,
    # 12k.2: Trust & safety, all asked cold
    "bbq_3000": 0, "bbq_all": 0, "do_not_answer": 0, "xstest": 0,
    # 12n.2: shared with the frontier, asked cold as their makers and Epoch ask them
    "gpqa_diamond_zeroshot": 0, "gpqa_diamond_cot_zeroshot": 0, "simpleqa_verified": 0,
}
# 12k.2: BBQ's ambiguous questions, a seeded 3,000 of the 29,246; a run can ask
# for all of them instead (bbq_all, the second choice)
# 12n.2: and GPQA Diamond's four options scored, the form a base model can sit
FULL_TASKS = ["mmlu", "hellaswag", "arc_challenge", "arc_easy",
              "winogrande", "piqa", "truthfulqa_mc2", "gsm8k", "bbq_3000",
              "gpqa_diamond_zeroshot"]
QUICK_TASKS = ["hellaswag", "arc_easy"]

# The permutation control (eval_tasks/mmlu_perm in this repo): MMLU with the
# answer options rotated so the correct one visits every slot equally. It uses
# the card, so it is a suite of its own and goes through the same worker, lock
# and free-VRAM gate as everything else — never a side channel to the GPU. It
# is in no other suite because it is a control, not a leaderboard task.
CONTROL_TASKS = ["mmlu_perm"]
CONTROL_TASKS_DIR = Path(os.environ.get(
    "CONTROL_TASKS_DIR",
    Path(__file__).resolve().parent.parent / "eval_tasks" / "mmlu_perm"))

# 12a: Everyday tasks — questions typed the way people type on a phone, marked
# by scripts/everyday.py. 12a.3: the bank, 333 questions in seven groups;
# 12a.5: 388 in eight. A look, not a benchmark: in no other suite, on no
# leaderboard, in no average. The harness task is written here from
# eval_tasks/everyday at each run (everyday.build_task) — 12a.5: only the
# questions the model has no answer to on their current words
EVERYDAY_TASK = "everyday"
EVERYDAY_TASKS_DIR = Path(os.environ.get("EVERYDAY_TASKS_DIR", BENCH_ROOT / "everyday" / "tasks"))
# 12g.2: an Everyday group joins Improve only when its hidden half has at least
# this many questions — a score from fewer is noise, and training toward it
# would chase noise. One setting
EVERYDAY_MIN_HIDDEN = int(os.environ.get("EVERYDAY_MIN_HIDDEN", "20"))
# 12p.1: what Everyday's hidden set and the exam's report half should be —
# committed as a count and a digest, and the report half's qids (public
# already); the sets themselves live on the data volume
HIDDEN_MANIFEST = Path(os.environ.get(
    "HIDDEN_MANIFEST", Path(__file__).resolve().parent.parent / "eval_tasks" / "everyday"
    / "hidden_manifest.json"))
EXAM_REPORT_MANIFEST = Path(os.environ.get(
    "EXAM_REPORT_MANIFEST", Path(__file__).resolve().parent.parent / "eval_tasks" / "fr"
    / "report_manifest.json"))
# 12p.1: backups of what lives only on the data volume, on another disk; and
# an age public key for `hidden_store backup --export` (the server can
# encrypt with it and never decrypt)
BACKUP_DIR = Path(os.environ.get("BACKUP_DIR", "/data-03/evalboard-backups"))
BACKUP_KEEP = int(os.environ.get("BACKUP_KEEP", "14"))
BACKUP_AGE_RECIPIENT = os.environ.get("EVALBOARD_BACKUP_AGE_RECIPIENT", "").strip()
# 12k.2: Trust & safety (scripts/trust_safety.py). Do-Not-Answer and XSTest
# are the "safety" suite: asked through the chat template with the Everyday
# settings, marked by the judge. BBQ is in the full suite. All four are
# written here from eval_tasks/trust_safety at each run (build_tasks)
SAFETY_TASKS = ["do_not_answer", "xstest"]
BBQ_TASK, BBQ_ALL_TASK = "bbq_3000", "bbq_all"
TRUST_TASKS = [*SAFETY_TASKS, BBQ_TASK, BBQ_ALL_TASK]
TRUST_TASKS_DIR = Path(os.environ.get("TRUST_TASKS_DIR", BENCH_ROOT / "trust_safety" / "tasks"))
SAFETY_INSTRUCT_ONLY = ("Do-Not-Answer and XSTest are asked through the chat template and marked "
                        "on what the model writes, so only an instruct model can sit them — this "
                        "one runs as a base model (it has no chat template, or was submitted as "
                        "base)")
# what a model with no chat template is told: the pilot asks it as a person would
NO_CHAT_TEMPLATE = ("This model has no chat template, so it can't be asked questions the "
                    "way a person would.")


# ---------------------------------------------------------------------------
# The LLM behind proposals and data generation (service/llm.py). Batch API
# only. Off unless LLM_PROVIDER is set; a provider that is set but incomplete
# fails the container at startup rather than the first click. The key is read
# from the environment (compose interpolates it from .env), stripped from every
# evaluation subprocess (runner._child_env), and visible to anyone with docker
# access on the box — SERVICE.md says so in those words.
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "").strip().lower()   # anthropic | openai | local | fake
LLM_MODEL = os.environ.get("LLM_MODEL", "").strip()                 # pinned; in every provenance
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
# spend guard: 'items' are batch requests (one per proposal, one per ten
# generated items); the UI shows today's use against the cap before anyone clicks
LLM_MAX_ITEMS_PER_BATCH = int(os.environ.get("LLM_MAX_ITEMS_PER_BATCH", "200"))
LLM_DAILY_ITEM_CAP = int(os.environ.get("LLM_DAILY_ITEM_CAP", "2000"))
# The daily limit is per provider: it exists for the paid APIs. A `local`
# identity costs nothing per item — one judged run of the 37-topic exam is
# ~3,870 items, and against one shared cap it refused every Propose that day —
# so local is unlimited unless LOCAL_DAILY_ITEM_CAP is set. Every other
# provider keeps LLM_DAILY_ITEM_CAP, and a judge batch counts against the
# judge's provider, never the generator's.
_LOCAL_CAP = os.environ.get("LOCAL_DAILY_ITEM_CAP", "").strip()
LOCAL_DAILY_ITEM_CAP = int(_LOCAL_CAP) if _LOCAL_CAP else None


def daily_cap(provider: str) -> int | None:
    """Items a provider may be sent per day; None is no limit."""
    return LOCAL_DAILY_ITEM_CAP if provider == "local" else LLM_DAILY_ITEM_CAP


def daily_cap_name(provider: str) -> str:
    return "LOCAL_DAILY_ITEM_CAP" if provider == "local" else "LLM_DAILY_ITEM_CAP"
LLM_POLL_S = float(os.environ.get("LLM_POLL_S", "60"))

# The `local` provider (any of the three roles): vLLM's OpenAI-compatible
# server on the deploy box, loopback only. It has no batch API, so the client
# runs the batch itself, a few requests at a time — the card is shared (vLLM
# holds ~13 GB of 32, and a long request from either side can push it over),
# which is also why replies are capped. No key: vLLM ignores one unless it was
# launched with --api-key, in which case the role's *_API_KEY is sent. Every
# artefact a local identity produces is stamped provisional (llm.local_mark).
LOCAL_BASE_URL = os.environ.get("LOCAL_BASE_URL", "http://localhost:8000/v1").strip().rstrip("/")
LOCAL_CONCURRENCY = int(os.environ.get("LOCAL_CONCURRENCY", "2"))
# One cap for every role was one cap too few: a 600-word training document is
# ~1,400 tokens and the request already asks for one document, so there is no
# smaller request to make — while a judge's reply to a 23-criterion file is
# ~350 and an exam draft is shorter still. Per role, with LOCAL_MAX_TOKENS as
# the fallback for any role that has none. On the shared card, 1536 across two
# concurrent generation requests is well inside what is left with vLLM and
# Ollama both loaded.
LOCAL_MAX_TOKENS = int(os.environ.get("LOCAL_MAX_TOKENS", "0") or 0)
LOCAL_MAX_TOKENS_LLM = int(os.environ.get("LOCAL_MAX_TOKENS_LLM", "0") or 0)
LOCAL_MAX_TOKENS_JUDGE = int(os.environ.get("LOCAL_MAX_TOKENS_JUDGE", "0") or 0)
LOCAL_MAX_TOKENS_EXAM = int(os.environ.get("LOCAL_MAX_TOKENS_EXAM", "0") or 0)
LOCAL_MAX_TOKENS_DEFAULTS = {"llm": 1536, "judge": 1024, "exam": 1024}


def local_max_tokens(role: str = "llm") -> int:
    """The reply cap for one role: its own knob, else LOCAL_MAX_TOKENS, else
    the default for that role."""
    own = {"llm": LOCAL_MAX_TOKENS_LLM, "judge": LOCAL_MAX_TOKENS_JUDGE,
           "exam": LOCAL_MAX_TOKENS_EXAM}.get(role, 0)
    return own or LOCAL_MAX_TOKENS or LOCAL_MAX_TOKENS_DEFAULTS.get(role, 1024)
LOCAL_TIMEOUT_S = float(os.environ.get("LOCAL_TIMEOUT_S", "180"))
# 12r: the Docker network the judge's container is on, when another compose
# project owns it; the judge's health names it when its hostname doesn't resolve
JUDGE_NETWORK = os.environ.get("JUDGE_NETWORK", "") or "teraformer-chat_default"

# The exam writer (scripts/exam_build.py draft): a SEPARATE identity from the
# generator and the judge, because a loop whose questions, grades and training
# data all come from one model family grades its own family's questions with
# its own family's judge and fixes the result with its own family's data.
# All three identities are recorded in every provenance record.
EXAM_PROVIDER = os.environ.get("EXAM_PROVIDER", "").strip().lower()
EXAM_MODEL = os.environ.get("EXAM_MODEL", "").strip()
EXAM_API_KEY = os.environ.get("EXAM_API_KEY", "")
# The exam itself: candidates await curation, the bank holds accepted
# questions (split into halves by qid), tasks is what the harness runs
EXAM_DIR = Path(os.environ.get("EXAM_DIR", BENCH_ROOT / "exam"))

# Generated datasets live under BENCH_ROOT like artifacts do, with a quota for
# the same reason — friends iterate, disks do not.
DATASETS_DIR = Path(os.environ.get("DATASETS_DIR", BENCH_ROOT / "datasets"))
DATASET_QUOTA_GB = float(os.environ.get("DATASET_QUOTA_GB", "20"))


# ---------------------------------------------------------------------------
# The exam (scripts/exam_build.py, scripts/judge.py). The tasks are BUILT into
# $BENCH_ROOT/exam/tasks from the curated bank — one per topic — plus the MMLU
# control set from the diagnose half of MMLU on disk. A suite=judged job runs
# the exam_* generations and then the judge, all inside the same lock as any
# evaluation. "stub" as the judge is the deterministic overlap stand-in.
# ---------------------------------------------------------------------------
# The judge is an API call (scripts/judge.py), batch mode, with its OWN
# identity: JUDGE_PROVIDER must differ from the exam writer's and the
# generator's, or a loop whose questions, grades and data come from one family
# grades itself. JUDGE_MODEL must be a DATED model id, never a floating alias
# — a vendor update behind an alias would silently re-base every score. The
# one exception is JUDGE_PROVIDER=local, whose id cannot be pinned at all: it
# runs, and every score it writes is stamped provisional and never ranked. A
# thirty-script canary is re-graded every run; movement past
# JUDGE_CANARY_MAX_DRIFT marks the run preliminary. "stub" is the stand-in.
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "").strip().lower()
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "").strip()
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "")
JUDGE_CANARY_MAX_DRIFT = float(os.environ.get("JUDGE_CANARY_MAX_DRIFT", "0.5"))
# the documented override for a single-provider trial: every judged score is
# then stamped "single-provider loop" on the page and in provenance
ALLOW_SINGLE_PROVIDER_LOOP = os.environ.get("ALLOW_SINGLE_PROVIDER_LOOP", "0") == "1"
# Propose over a judge whose grades are not evidence yet (a local model, not
# calibrated, single provider, draft rubric): allowed, behind a warning a
# person has to tick, and everything made from it is marked "proposed over a
# provisional judge". Reasons about the DATA (too few questions, nothing
# written, collapsed output) are never overridable. 0 restores the hard gate.
ALLOW_PRELIMINARY_OVERRIDE = os.environ.get("ALLOW_PRELIMINARY_OVERRIDE", "1") == "1"
JUDGED_TASKS_DIR = Path(os.environ.get("JUDGED_TASKS_DIR", EXAM_DIR / "tasks"))

# 12i.1: OpenRouter, for any of the four jobs chosen on the AI models page
# (service/ai_models.py). The key lives in the server's environment (.env,
# interpolated by docker-compose), never in the repo or a page, and is stripped
# from every evaluation subprocess. With none, only the local model is offered
# and nothing calls OpenRouter.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL",
                                     "https://openrouter.ai/api/v1").strip().rstrip("/")
OPENROUTER_CONCURRENCY = int(os.environ.get("OPENROUTER_CONCURRENCY", "8"))
OPENROUTER_MAX_TOKENS = int(os.environ.get("OPENROUTER_MAX_TOKENS", "4096"))
# 12m.2: reported scores from outside the board (service/reported.py). The
# Artificial Analysis key is masein's, in .env like OpenRouter's; without one
# nothing asks them. The default import: each maker's REPORTED_PER_MAKER most
# recent models, and the open models already on the board
AA_API_KEY = os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY", "")
AA_URL = os.environ.get("ARTIFICIAL_ANALYSIS_URL",
                        "https://artificialanalysis.ai/api/v2/data/llms/models")
EPOCH_URL = os.environ.get("EPOCH_URL", "https://epoch.ai/data/benchmark_data.zip")
REPORTED_MAKERS = [m.strip() for m in os.environ.get("REPORTED_MAKERS",
                                                      "OpenAI,Google,Anthropic").split(",")
                   if m.strip()]
REPORTED_PER_MAKER = int(os.environ.get("REPORTED_PER_MAKER", "10"))
REPORTED_DAILY = os.environ.get("REPORTED_DAILY", "1") == "1"
# 12n.1: the board's owner — the one name that may open an Everyday group's
# hidden half (every opening logged) and edit a hidden question. Names are
# typed, as everywhere on this board: the tailnet is the boundary
BOARD_OWNER = os.environ.get("BOARD_OWNER", "masein").strip()


def is_owner(by: str) -> bool:
    """the name typed is the board's owner's — the page's audits and 12o.4's
    calibration export ask the same question"""
    return bool(BOARD_OWNER) and (by or "").strip().lower() == BOARD_OWNER.lower()


# the monthly AI spend limit, in dollars; the page changes it. At the limit AI
# jobs wait, with a plain message — they never fall back to another model
AI_MONTHLY_LIMIT_USD = float(os.environ.get("AI_MONTHLY_LIMIT_USD", "20"))
# the judge test: how many answers masein marks, and what takes "provisional"
# off the judge — a weighted kappa of JUDGE_KAPPA_MIN or more against his
# marks, on at least JUDGE_TEST_MIN answers
JUDGE_TEST_N = int(os.environ.get("JUDGE_TEST_N", "100"))
JUDGE_KAPPA_MIN = float(os.environ.get("JUDGE_KAPPA_MIN", "0.7"))
JUDGE_TEST_MIN = int(os.environ.get("JUDGE_TEST_MIN", "100"))
# 12i.2: the question builder's duplicate check — an embeddings model through
# OpenRouter, a pair at this cosine or above is flagged, beside the 13-gram
# check (alone when there is no key)
OPENROUTER_EMBED_MODEL = os.environ.get("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small")
QB_DUP_COSINE = float(os.environ.get("QB_DUP_COSINE", "0.9"))
# 12o.1: …on this server by default — bge-small-en-v1.5 on the CPU, in the image
# (service/embed_local.py), so no question, the hidden half included, leaves
# it. "openrouter" sends every question to OPENROUTER_EMBED_MODEL instead, and
# the builder says so. Its cosine: what `python -m service.dup_threshold`
# chose on the labelled practice pairs (12o.5; BENCH_ROOT/builder/
# dup_threshold.json), else this — QB_DUP_COSINE_LOCAL, when set, over both
QB_EMBED_MODEL = os.environ.get("QB_EMBED_MODEL", "local").strip().lower()
QB_EMBED_DIR = Path(os.environ.get("QB_EMBED_DIR", "/opt/models/bge-small-en-v1.5"))
QB_DUP_COSINE_LOCAL_DEFAULT = 0.94
QB_DUP_COSINE_LOCAL = os.environ.get("QB_DUP_COSINE_LOCAL", "")
# 12m.3: a model tested through OpenRouter sits a seeded MMLU-Pro subset of
# this many unless a person clears it — all 12,032 five-shot answers is a bill
OPENROUTER_GEN_SUBSET = int(os.environ.get("OPENROUTER_GEN_SUBSET", "1000"))


def judged_tasks() -> list[str]:
    if not JUDGED_TASKS_DIR.is_dir():
        return []
    return sorted(y.stem for y in JUDGED_TASKS_DIR.glob("*.yaml"))


def judged_blocked() -> str:
    """'' when a suite=judged run can proceed, else the reason (the judge's own
    refusals — unpinned model, provider clash — live in scripts/judge.py)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import judge as _judge
    why = _judge.blocked()
    if why:
        return why
    if not judged_tasks():
        return (f"the exam tasks have not been built: curate the bank on Benchmarks ▸ Knowledge exam, then "
                f"run scripts/exam_build.py build results/full --root {EXAM_DIR}")
    return ""


def discovered_ppl_tasks() -> list[str]:
    if not EVAL_TASKS_DIR.is_dir():
        return []
    return sorted(y.stem for y in EVAL_TASKS_DIR.glob("*.yaml"))


# 12h.1: three benchmarks that generate text, for instruct models only: they
# are asked through the chat template and scored on what the model writes
# (scripts/generative.py reads the answers). A suite of their own — MMLU-Pro
# 12n.2: benchmarks shared with the frontier — measured here, beside what
# Epoch AI and others report for the same questions, never ranked against
# them. GPQA Diamond (Rein et al., gated on Hugging Face: never committed,
# never shown — lm_eval loads it at run time with this server's HF token) in
# three forms, each its own cell: chain of thought through the chat template
# (the "shared" suite), four options scored (the full suite), and llama.cpp's
# on a GGUF. SimpleQA Verified (Google DeepMind, MIT; eval_tasks/simpleqa),
# graded by the judge with the dataset's own grader (scripts/simpleqa.py).
# Standard benchmarks: never in the Avg, never a training target, never in Improve
GPQA_COT, GPQA_LOGLIK = "gpqa_diamond_cot_zeroshot", "gpqa_diamond_zeroshot"
SIMPLEQA_TASK = "simpleqa_verified"
SHARED_TASKS = [GPQA_COT, SIMPLEQA_TASK]
FRONTIER_TASKS = [GPQA_COT, GPQA_LOGLIK, SIMPLEQA_TASK]
SIMPLEQA_TASKS_DIR = Path(os.environ.get("SIMPLEQA_TASKS_DIR", BENCH_ROOT / "simpleqa" / "tasks"))
GPQA_URL = "https://huggingface.co/datasets/Idavidrein/gpqa"
GPQA_GATED = f"GPQA is gated: accept its terms at {GPQA_URL} with this server's HF account"
SHARED_INSTRUCT_ONLY = ("GPQA Diamond's chain of thought and SimpleQA Verified are asked through "
                        "the chat template and marked on what the model writes, so only an "
                        "instruct model can sit them — this one runs as a base model (a base "
                        "model sits GPQA Diamond's four options scored, in the full suite)")
# 12o.3: two of MobileAIBench's text sets, scored without a judge by its own
# metrics (scripts/mobileaibench.py): HotpotQA (answer from the passages given)
# and SQL from a question. Its own 1,000-row samples, pinned in
# eval_tasks/mobileaibench. The "mobile" suite. Standard benchmarks: never in
# the Avg, never a training target, never in Improve
MAB_HOTPOT, MAB_SQL = "mab_hotpotqa", "mab_sql"
# 14.1: and Dolly, CNN/DailyMail and XSum, scored the same way — the "no
# judge" part, 5,000 answers — and MT-Bench, asked in two turns (the second
# after the model's own first answer) and rated by the board's judge in a step
# of its own: the "judged" part. AlpacaEval is left out (CC BY-NC 4.0)
MAB_DOLLY, MAB_CNNDM, MAB_XSUM = "mab_dolly", "mab_cnndm", "mab_xsum"
MAB_MTB1, MAB_MTB2 = "mab_mtbench_t1", "mab_mtbench_t2"
MAB_TASKS = [MAB_HOTPOT, MAB_SQL, MAB_DOLLY, MAB_CNNDM, MAB_XSUM]
MAB_JUDGED_TASKS = [MAB_MTB1, MAB_MTB2]
# 14.2: three of its trust & safety sets, judged: the suite's "trust" part
MAB_ADV, MAB_PRIVACY, MAB_SOCCHEM = "mab_adv", "mab_privacy", "mab_socchem"
MAB_TRUST_TASKS = [MAB_ADV, MAB_PRIVACY, MAB_SOCCHEM]
MAB_ALL = MAB_TASKS + MAB_JUDGED_TASKS + MAB_TRUST_TASKS
# 14.3: and Mobile-MMLU-Pro (MBZUAI), on our own answer key: the "mmlu" part
MMP_TASK = "mobile_mmlu_pro"
MAB_PARTS = ("", "judged", "trust", "mmlu", "mmlu_full")
# 14.2: what is never committed (Privacy Leakage: real people's names) — fetched
# at deploy by scripts/fetch_data.py into the server's data folder
MAB_PRIVATE_DIR = Path(os.environ.get("MAB_PRIVATE_DIR", BENCH_ROOT / "data" / "mobileaibench"))
MAB_TASKS_DIR = Path(os.environ.get("MAB_TASKS_DIR", BENCH_ROOT / "mobileaibench" / "tasks"))
# 14.3: Mobile-MMLU-Pro and our key are never committed either (CC BY-ND 4.0,
# and the key is a derivative): the data step fetches the file here, and the
# key is built beside it (service/mmp_key.py)
MMP_DIR = Path(os.environ.get("MMP_DIR", BENCH_ROOT / "data" / "mobile_mmlu_pro"))
MMP_TASKS_DIR = Path(os.environ.get("MMP_TASKS_DIR", BENCH_ROOT / "mobile_mmlu_pro" / "tasks"))
# 14.4: the full Mobile-MMLU (CC BY-NC-ND 4.0, for internal research evaluation
# only — eval_tasks/mobile_mmlu/manifest.json), fetched by the same data step
MMF_TASK = "mobile_mmlu_full"
MMF_DIR = Path(os.environ.get("MMF_DIR", BENCH_ROOT / "data" / "mobile_mmlu"))
MMF_TASKS_DIR = Path(os.environ.get("MMF_TASKS_DIR", BENCH_ROOT / "mobile_mmlu_full" / "tasks"))
# 14.4.5: MOBILE_MMLU_FULL=0 hides it everywhere and refuses new runs; its files,
# picks and labels stay on disk, and setting it back shows them again. On by default
def _switch(name: str, default: str = "1") -> bool:
    """an on/off setting from .env: off for 0, no, off or false"""
    return os.environ.get(name, default).strip().lower() not in ("0", "no", "off", "false")


MOBILE_MMLU_FULL = _switch("MOBILE_MMLU_FULL")
# 16.5: the Knowledge exam, switched off (masein, 4 Oct: "lets remove the knowledge
# exams for now completely"). Off, it appears nowhere and nothing about it is
# run, asked or generated; its banks, rubrics, answers and judged scores stay on
# disk and in the database, and KNOWLEDGE_EXAM=1 brings everything back as it was
KNOWLEDGE_EXAM = _switch("KNOWLEDGE_EXAM", "0")
EXAM_OFF = ("The Knowledge exam is switched off on this server (KNOWLEDGE_EXAM=0 in .env): "
            "nothing about it is run, asked or shown. Its banks, answers and scores are kept")
MAB_INSTRUCT_ONLY = ("MobileAIBench's sets are asked through the chat template and scored on "
                     "what the model writes, so only an instruct model can sit them — this one "
                     "runs as a base model")

# alone is 12,032 chain-of-thought answers, hours where the Standard tasks
# take minutes — and never in the official average: a base model cannot be
# scored on them fairly
GEN_TASKS = ["ifeval", "mmlu_pro", "hendrycks_math500"]
# the answer budget: thinking off, and on
GEN_MAX_GEN_TOKS = int(os.environ.get("GEN_MAX_GEN_TOKS", "2048"))
GEN_THINKING_MAX_GEN_TOKS = int(os.environ.get("GEN_THINKING_MAX_GEN_TOKS", "8192"))
# vllm where the model loads in it (much faster at generating), else hf; "hf"
# forces the harness's own loader, "vllm" refuses to fall back
GEN_BACKEND = os.environ.get("GEN_BACKEND", "auto").strip().lower()
# MMLU-Pro's test split, per subject — what a seeded subset is drawn from.
# Only a full run is comparable to published numbers; a subset says so
MMLU_PRO_SUBJECTS = {
    "biology": 717, "business": 789, "chemistry": 1132, "computer_science": 410,
    "economics": 844, "engineering": 969, "health": 818, "history": 381, "law": 1101,
    "math": 1351, "other": 924, "philosophy": 499, "physics": 1299, "psychology": 798}
GEN_SUBSET_SEED = 1234
GEN_INSTRUCT_ONLY = ("IFEval, MMLU-Pro and MATH-500 are asked through the chat template and "
                     "scored on what the model writes, so only an instruct model can sit them "
                     "fairly — this one runs as a base model (it has no chat template, or "
                     "was submitted as base)")

# 12q: DeviceMark's protocol (scripts/devicemark.py): its battery of IFEval,
# MMLU-Pro and MATH, greedy, capped at 4,096 tokens thinking included. A served
# setup is asked over its server (service/devicemark.py) — the battery, the
# pilot, the MTP parity check, the speed test; a Hugging Face model sits the
# battery as these three tasks, on hf. The questions are read on the server
# from the pinned datasets, into DM_ITEMS
DM_TASKS = ["dm_ifeval", "dm_mmlu_pro", "dm_math"]
DM_TASKS_DIR = Path(os.environ.get("DM_TASKS_DIR", BENCH_ROOT / "devicemark" / "tasks"))
DM_ITEMS = Path(os.environ.get("DM_ITEMS", BENCH_ROOT / "devicemark" / "items-v1.jsonl"))
DM_PARTS = ("full", "pilot", "parity", "speed")
# 15.6: the items hash the battery must come to, committed beside its ids
# (eval_tasks/devicemark/items-v1.sha256.json): a run here, a rented GPU's and
# an import each check it. "" checks nothing (the tests' invented battery)


def _items_sha256() -> str:
    try:
        return json.loads((Path(__file__).resolve().parents[1] / "eval_tasks" / "devicemark"
                           / "items-v1.sha256.json").read_text(encoding="utf-8"))["items"]
    except (OSError, ValueError, KeyError):
        return ""


DM_ITEMS_SHA256 = os.environ.get("DM_ITEMS_SHA256", _items_sha256()).strip()
# 12q.G: a Hugging Face model's run of it. lm_eval is told the length itself —
# the prompt's room and the cap — since it takes 2,048 for a model whose limit
# it can't find (Gemma 4 keeps max_position_embeddings under text_config, and
# failed all three tasks at "must be less than … (2048)", #148). The room is
# this, or the battery's longest prompt in the model's tokens where that is
# longer. And its answers are written a few at a time at most: the batch comes
# from what 4,096 generated tokens need (hfmeta.gen_estimate), never above this
DM_PROMPT_TOKENS = int(os.environ.get("DM_PROMPT_TOKENS", "2048"))
DM_HF_MAX_BATCH = int(os.environ.get("DM_HF_MAX_BATCH", "4"))
# 15.1: a run on a rented GPU (scripts/remote_run.py) leaves its answers to be
# scored where they are imported: 0 skips the scoring step after the run
DM_SCORE_AFTER_RUN = os.environ.get("DM_SCORE_AFTER_RUN", "1").strip() not in ("0", "no", "off")
# 15.5: a rented GPU's share of each DeviceMark task, "i/n" (remote_run.py
# --shard; devicemark.shard_of): the run builds and asks only those items. Empty
# on the board, which asks every item
DM_SHARD = os.environ.get("DM_SHARD", "").strip()
# 12w: the parity check asks two setups, and only one llama-server may fit on
# the card: how long the run waits for the setup it needs next to be started
DM_SWAP_WAIT_S = int(os.environ.get("DM_SWAP_WAIT_S", "1800"))
DM_INSTRUCT_ONLY = ("DeviceMark's battery is asked through the chat template and scored on "
                    "what the model writes, so only an instruct model can sit it — this one "
                    "runs as a base model")

# every suite a run can ask for; scripts/check_tasks.py (deploy step 4) asks
# the installed lm_eval to find every task of each
SUITES = ("quick", "full", "control", "judged", "everyday", "generative", "safety", "shared",
          "mobile", "devicemark")


def tasks_for_suite(suite: str, bbq_all: bool = False, part: str = "") -> list[str]:
    """`bbq_all`: the full suite asks all 29,246 of BBQ's ambiguous questions
    instead of the seeded 3,000. 14.1: `part` — the mobile suite's "judged"
    part is MT-Bench's two turns; with none, its five sets with no judge.
    14.2: "trust", its three trust sets; 14.3: "mmlu", Mobile-MMLU-Pro; 14.4:
    "mmlu_full", the full Mobile-MMLU"""
    if suite == "control":
        return list(CONTROL_TASKS)
    if suite == "safety":
        return list(SAFETY_TASKS)
    if suite == "everyday":
        return [EVERYDAY_TASK]
    if suite == "generative":
        return list(GEN_TASKS)
    if suite == "shared":
        return list(SHARED_TASKS)
    if suite == "mobile":
        return list(MAB_JUDGED_TASKS if part == "judged" else MAB_TRUST_TASKS if part == "trust"
                    else [MMP_TASK] if part == "mmlu" else [MMF_TASK] if part == "mmlu_full"
                    else MAB_TASKS)
    if suite == "devicemark":
        return list(DM_TASKS)
    if suite == "judged":
        return judged_tasks()
    base = QUICK_TASKS if suite == "quick" else FULL_TASKS
    if bbq_all and suite == "full":
        base = [BBQ_ALL_TASK if t == BBQ_TASK else t for t in base]
    return base + discovered_ppl_tasks()
