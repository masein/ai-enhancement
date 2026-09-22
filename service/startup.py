"""What the service reads from the repo, and whether it is there.

A file the service needs at run time but does not carry is invisible until
someone clicks the button that needs it: the image that shipped on
2026-09-20 had no `eval_tasks/fr/_fr_template_yaml`, and the first person to
press "Rebuild the harness tasks" got a 500 with a traceback in the log.
Both demo runs had passed, because the demo runs from the checkout.

So the list lives here, the app refuses to start without it (HANDOFF §13
says a missing file fails at `up`, where the operator is looking), the image
build runs the same check, and a test asserts the image would carry every
path on it.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The 37 topics of the exam, by slug: 36 delivered on 2026-09-21, and Arts,
# empty then, on 2026-09-22. Each has three files in the image:
#   eval_tasks/fr/banks/<slug>_v1.json        exam_build.py import-dir reads it (the deploy step)
#   eval_tasks/fr/rubrics/<slug>.criteria.json   judge.py grades the topic against it
#   eval_tasks/fr/rubrics/<slug>.md           …and puts its prose and anchors in the prompt
DELIVERED_TOPIC_SLUGS = [
    "agriculture", "ai_machine_learning", "anthropology_human_geography",
    "architecture_built_environment", "arts", "biology_life_sciences", "business_management",
    "chemistry_materials_science", "computer_science", "data_information_science", "design",
    "earth_environmental_sciences", "economics", "education", "engineering", "ethics_religion",
    "finance_accounting", "food_veterinary_sciences", "general_multidisciplinary",
    "government_public_policy", "history_archaeology", "it", "language_literature", "law",
    "manufacturing_applied_sciences", "mathematics_statistics", "media_communication",
    "medicine_clinical_health", "philosophy", "physics_astronomy",
    "political_science_international_relations", "psychology_cognitive_sciences",
    "public_health_wellness", "sociology", "software_engineering_programming",
    "systems_cybersecurity", "technology",
]
DELIVERED_TOPIC_FILES = [
    f
    for s in DELIVERED_TOPIC_SLUGS
    for f in (f"eval_tasks/fr/banks/{s}_v1.json", f"eval_tasks/fr/rubrics/{s}.criteria.json",
              f"eval_tasks/fr/rubrics/{s}.md")
]

# Every repo file the RUNNING SERVICE reads. Not the tests', not the demo's —
# the ones a request can reach. Each says which code path needs it.
REQUIRED_REPO_FILES = [
    "scripts/categories.yaml",              # the topic spine: every exam task name
    "eval_tasks/fr/_fr_template_yaml",      # exam_build.build() writes each task from it
    "eval_tasks/fr/canary.jsonl",           # judge.load_canary(): every judged run starts here
    "eval_tasks/fr/rubrics/exam.md",        # the shared rubric, for a topic without its own
    "eval_tasks/fr/rubrics/factual_accuracy.md",   # the control set's rubric
    "eval_tasks/mmlu_perm/_mmlu_perm.yaml",  # suite=control passes this dir to --include_path
    "eval_tasks/mmlu_perm/utils.py",        # …and the tasks in it import this
    "FRIENDS.md",                           # served at /guide
    "clients/bench_client.py",              # served at /client
    *DELIVERED_TOPIC_FILES,
]

# Directories whose contents are read by name at run time: a topic's own
# rubric and criteria file are looked up from the bank, so the set cannot be
# listed, but the directory must exist and not be empty.
REQUIRED_REPO_DIRS = [
    "eval_tasks/fr/rubrics",
    "eval_tasks/fr/banks",
    "eval_tasks/mmlu_perm",
]


def missing_repo_files(root: Path | None = None) -> list[str]:
    """The paths that are not there, in the order listed. Empty means the
    tree this process runs from can serve every request the page makes."""
    base = Path(root or REPO)
    out = [p for p in REQUIRED_REPO_FILES if not (base / p).is_file()]
    out += [f"{p}/ (empty or missing)" for p in REQUIRED_REPO_DIRS
            if not (base / p).is_dir() or not any((base / p).iterdir())]
    return out


def check_repo_files(root: Path | None = None) -> None:
    """Raise with every missing path named. Called from the app's lifespan,
    so the container stops at `up` with the reason on stdout instead of
    serving 500s to whoever gets there first."""
    missing = missing_repo_files(root)
    if missing:
        raise SystemExit(
            "this build cannot serve requests: "
            + ", ".join(missing)
            + f" — missing from {Path(root or REPO)}. In Docker this means the Dockerfile's "
              "COPY list or .dockerignore dropped them; see SERVICE.md § Docker.")
