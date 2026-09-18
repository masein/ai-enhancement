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


# ---------------------------------------------------------------------------
# The LLM behind proposals and data generation (service/llm.py). Batch API
# only. Off unless LLM_PROVIDER is set; a provider that is set but incomplete
# fails the container at startup rather than the first click. The key is read
# from the environment (compose interpolates it from .env), stripped from every
# evaluation subprocess (runner._child_env), and visible to anyone with docker
# access on the box — SERVICE.md says so in those words.
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "").strip().lower()   # anthropic | openai | fake
LLM_MODEL = os.environ.get("LLM_MODEL", "").strip()                 # pinned; in every provenance
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
# spend guard: 'items' are batch requests (one per proposal, one per ten
# generated items); the UI shows today's use against the cap before anyone clicks
LLM_MAX_ITEMS_PER_BATCH = int(os.environ.get("LLM_MAX_ITEMS_PER_BATCH", "200"))
LLM_DAILY_ITEM_CAP = int(os.environ.get("LLM_DAILY_ITEM_CAP", "2000"))
LLM_POLL_S = float(os.environ.get("LLM_POLL_S", "60"))

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
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "").strip()
JUDGED_TASKS_DIR = Path(os.environ.get("JUDGED_TASKS_DIR", EXAM_DIR / "tasks"))


def judged_tasks() -> list[str]:
    if not JUDGED_TASKS_DIR.is_dir():
        return []
    return sorted(y.stem for y in JUDGED_TASKS_DIR.glob("*.yaml"))


def judged_blocked() -> str:
    """'' when a suite=judged run can proceed, else the reason."""
    if not JUDGE_MODEL:
        return ("no judge is configured on this server (JUDGE_MODEL is unset) — the judged "
                "suite is off")
    if not judged_tasks():
        return (f"the exam tasks have not been built: curate the bank on the Exam tab, then "
                f"run scripts/exam_build.py build results/full --root {EXAM_DIR}")
    return ""


def discovered_ppl_tasks() -> list[str]:
    if not EVAL_TASKS_DIR.is_dir():
        return []
    return sorted(y.stem for y in EVAL_TASKS_DIR.glob("*.yaml"))


def tasks_for_suite(suite: str) -> list[str]:
    if suite == "control":
        return list(CONTROL_TASKS)
    if suite == "judged":
        return judged_tasks()
    base = QUICK_TASKS if suite == "quick" else FULL_TASKS
    return base + discovered_ppl_tasks()
