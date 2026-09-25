"""Configuration — everything is an environment variable with a sane default.

The service is meant to be launched from the benchmarks directory (the one that
contains results/, eval_tasks/, logs/), like the CLI script. BENCH_ROOT overrides
that if you launch it from elsewhere.
"""

from __future__ import annotations

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
GPU_POLL_S = int(os.environ.get("GPU_POLL_S", "60"))
GPU_WAIT_MAX_S = int(os.environ.get("GPU_WAIT_MAX_S", str(6 * 3600)))
TASK_TIMEOUT_S = int(os.environ.get("TASK_TIMEOUT_S", str(3 * 3600)))

# Submission guardrails.
MAX_PARAMS_B = float(os.environ.get("MAX_PARAMS_B", "4"))  # reject >4B params (bf16 ≈ 8 GB weights)
SUBMIT_TOKEN = os.environ.get("SUBMIT_TOKEN", "")          # empty = no token required

# Artifact storage: checkpoints uploaded directly to this box instead of the HF
# Hub. They are ordinary model directories under ARTIFACTS_DIR and get evaluated
# as `local/<name>`. Quotas exist because friends iterate and disks do not.
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", BENCH_ROOT / "artifacts"))
ARTIFACT_MAX_GB = float(os.environ.get("ARTIFACT_MAX_GB", "8"))     # per upload
ARTIFACT_QUOTA_GB = float(os.environ.get("ARTIFACT_QUOTA_GB", "150"))  # total dir

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

# The benchmark suite — one place, mirrored from run_benchmarks.sh. quick is for
# iteration (minutes); full is the comparable number. Both write into the same
# tree, so a quick run later "upgrades" to full by running only the missing tasks.
NFEWSHOT = {
    "mmlu": 5, "hellaswag": 5, "arc_challenge": 5, "arc_easy": 5,
    "winogrande": 5, "piqa": 0, "truthfulqa_mc2": 0, "gsm8k": 5,
    "mmlu_perm": 5,          # the control poses MMLU exactly as mmlu does, shots included
}
FULL_TASKS = ["mmlu", "hellaswag", "arc_challenge", "arc_easy",
              "winogrande", "piqa", "truthfulqa_mc2", "gsm8k"]
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
# by scripts/everyday.py. 12a.3: the bank, 333 questions in seven groups. A
# look, not a benchmark: in no other suite, on no leaderboard, in no average.
# The harness task is written here from eval_tasks/everyday at the start of
# each run (everyday.build_task)
EVERYDAY_TASK = "everyday"
EVERYDAY_TASKS_DIR = Path(os.environ.get("EVERYDAY_TASKS_DIR", BENCH_ROOT / "everyday" / "tasks"))
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


# every suite a run can ask for; scripts/check_tasks.py (deploy step 4) asks
# the installed lm_eval to find every task of each
SUITES = ("quick", "full", "control", "judged", "everyday")


def tasks_for_suite(suite: str) -> list[str]:
    if suite == "control":
        return list(CONTROL_TASKS)
    if suite == "everyday":
        return [EVERYDAY_TASK]
    if suite == "judged":
        return judged_tasks()
    base = QUICK_TASKS if suite == "quick" else FULL_TASKS
    return base + discovered_ppl_tasks()
