"""15.5: the runner image, built by the mirror's Actions and pushed to
ghcr.io/masein/evalboard-runner, which is public — so it is checked, every file
in it, before it is pushed (scripts/check_runner_image.py): no exam or Everyday
bank, no trust set, no judge's rubric, no .env file. Here: the check on the
files the runner's and the board's COPY lines take from this checkout, on each
thing it must catch, and the workflow — its triggers, its two tags, the check
before the push, and no secret but the run's own token. No image is built."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import check_runner_image as cri

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "runner-image.yml"
BASE = ["usr/bin/python3", "opt/conda/lib/python3.12/site-packages/torch/__init__.py",
        "usr/share/nltk_data/tokenizers/punkt_tab/english/collocations.tab", "etc/hostname"]


def copies(stage: str) -> list[str]:
    """the sources a Dockerfile stage COPYs into /app"""
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    body = re.search(rf"^FROM \S+ AS {stage}\n(.*?)(?=^FROM |\Z)", text, re.S | re.M)[1]
    return [m[1] for m in re.finditer(r"^COPY (\S+) \S+\s*$", body, re.M)]


def image_of(stage: str) -> list[str]:
    """the files a stage's COPY lines put in /app from this checkout, over a base"""
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout.splitlines()
    out = list(BASE)
    for src in copies(stage):
        out += ["/app/" + f for f in tracked if (f.startswith(src) if src.endswith("/")
                                                  else f == src)]
    return out


def test_the_runner_image_from_this_checkout_holds_nothing_that_stays():
    files = image_of("runner")
    assert copies("runner") == ["scripts/", "service/", "eval_tasks/devicemark/"]
    assert "/app/scripts/remote_run.py" in files and "/app/eval_tasks/devicemark/battery-v1.json" in files
    assert cri.problems(files) == []


def test_the_boards_image_would_never_be_pushed():
    found = cri.problems(image_of("board"))
    for words in ("the Knowledge exam's banks and the judge's rubrics (eval_tasks/fr)",
                  "the Everyday bank (eval_tasks/everyday)", "the trust sets (eval_tasks/trust_safety)"):
        assert any(x.endswith(words) for x in found), words
    assert any(x.startswith("app/eval_tasks/fr/rubrics/") for x in found)
    assert any(x.startswith("app/clients/") and "not the runner's" in x for x in found)


def test_each_thing_that_stays_on_the_server_is_named():
    files = ["/app/scripts/remote_run.py", "/app/eval_tasks/fr/banks/x_v1.json",
             "/app/eval_tasks/everyday/bank.jsonl", "/workspace/eval_tasks/trust_safety/bbq.jsonl",
             "/app/eval_tasks/fr/rubrics/x.md", "/app/.env", "/root/.env", "/app/scripts/.env.local",
             "/app/README.md", "/app/scripts/hidden_half.jsonl", "/app/service/topic_rubric.yaml",
             "/app/service/hidden_store.py", "/app/eval_tasks/simpleqa/data.csv"]
    found = cri.problems(files, env=["PATH=/usr/bin", "HF_TOKEN=x"],
                         cmd=["python", "-m", "uvicorn", "service.app:app"], ports=["8899/tcp"])
    assert found == [
        "app/eval_tasks/fr/banks/x_v1.json: the Knowledge exam's banks and the judge's rubrics "
        "(eval_tasks/fr)",
        "app/eval_tasks/everyday/bank.jsonl: the Everyday bank (eval_tasks/everyday)",
        "workspace/eval_tasks/trust_safety/bbq.jsonl: the trust sets (eval_tasks/trust_safety)",
        "app/eval_tasks/fr/rubrics/x.md: the Knowledge exam's banks and the judge's rubrics "
        "(eval_tasks/fr)",
        "app/.env: an .env file", "root/.env: an .env file", "app/scripts/.env.local: an .env file",
        "app/README.md: not the runner's — app/ holds scripts/, service/, eval_tasks/devicemark/ only",
        "app/scripts/hidden_half.jsonl: a data file named like a bank, a rubric or a hidden half",
        "app/service/topic_rubric.yaml: a data file named like a bank, a rubric or a hidden half",
        "app/eval_tasks/simpleqa/data.csv: a benchmark’s data other than DeviceMark’s "
        "(eval_tasks/simpleqa)",
        "the image's environment sets HF_TOKEN",
        "its start command starts the board: python -m uvicorn service.app:app",
        "it exposes 8899, the board's port"]
    # an empty listing, or another image's, isn't a pass
    assert cri.problems([]) == ["no app/scripts/remote_run.py: this isn't the runner image, or "
                                "the listing is empty"]


def test_the_command_exits_1_and_names_them(tmp_path, capsys):
    listing = tmp_path / "files.txt"
    listing.write_text("\n".join(image_of("runner")) + "\n")
    assert cri.main(["--files", str(listing)]) == 0
    assert "no bank, rubric, trust set or .env file" in capsys.readouterr().out
    listing.write_text("\n".join(image_of("runner") + ["/app/eval_tasks/everyday/bank.jsonl"]))
    assert cri.main(["--files", str(listing)]) == 1
    out = capsys.readouterr().out
    assert "1 thing that must stay on the server:" in out and "the Everyday bank" in out


def test_the_workflow_builds_on_main_and_by_hand_and_checks_before_it_pushes():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "\non:\n  push:\n    branches: [main]\n  workflow_dispatch:\n" in text
    assert "\npermissions:\n  contents: read\n  packages: write\n" in text
    assert "\n  IMAGE: ghcr.io/masein/evalboard-runner\n" in text
    assert "    if: github.repository == 'masein/ai-enhancement'\n" in text
    # the tag is the deploy's EVALBOARD_BUILD: the commit's sha, seven characters
    assert 'echo "short=$(git rev-parse --short=7 HEAD)" >> "$GITHUB_OUTPUT"' in text
    build = text.index("docker build --target runner")
    check = text.index("run: python3 scripts/check_runner_image.py \"$IMAGE:")
    login, push = text.index("docker login ghcr.io"), text.index("docker push")
    assert build < check < login < push
    assert "--build-arg EVALBOARD_BUILD=${{ steps.tag.outputs.short }}" in text[build:check]
    assert '-t "$IMAGE:${{ steps.tag.outputs.short }}" -t "$IMAGE:latest" .' in text[build:check]
    assert ('docker push "$IMAGE:${{ steps.tag.outputs.short }}"\n          docker push '
            '"$IMAGE:latest"') in text[push:]
    # no stored secret: this run's own token, read from stdin
    assert set(re.findall(r"secrets\.(\w+)", text)) == {"GITHUB_TOKEN"}
    assert "--password-stdin" in text[login:push]


def test_ci_checks_the_runner_image_it_builds():
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = ci[ci.index("\n  docker-image:\n"):]
    assert job.index("docker build --target runner -t aienh-runner:ci .") < job.index(
        "run: python3 scripts/check_runner_image.py aienh-runner:ci")
