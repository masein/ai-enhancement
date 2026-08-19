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

# The benchmark suite — one place, mirrored from run_benchmarks.sh. quick is for
# iteration (minutes); full is the comparable number. Both write into the same
# tree, so a quick run later "upgrades" to full by running only the missing tasks.
NFEWSHOT = {
    "mmlu": 5, "hellaswag": 5, "arc_challenge": 5, "arc_easy": 5,
    "winogrande": 5, "piqa": 0, "truthfulqa_mc2": 0, "gsm8k": 5,
}
FULL_TASKS = ["mmlu", "hellaswag", "arc_challenge", "arc_easy",
              "winogrande", "piqa", "truthfulqa_mc2", "gsm8k"]
QUICK_TASKS = ["hellaswag", "arc_easy"]


def discovered_ppl_tasks() -> list[str]:
    if not EVAL_TASKS_DIR.is_dir():
        return []
    return sorted(y.stem for y in EVAL_TASKS_DIR.glob("*.yaml"))


def tasks_for_suite(suite: str) -> list[str]:
    base = QUICK_TASKS if suite == "quick" else FULL_TASKS
    return base + discovered_ppl_tasks()
