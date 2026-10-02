"""15.6: constraints.txt — every Python package in the board's image at its
version, made from the built image (scripts/image_packages.py) — and both
Dockerfile stages installing with it, so the board's image built at deploy and
the runner image built by CI hold the same versions, tokenizers included.
ci.yml's image job fails when the two images' lists differ (but for what only
the board stage installs) or either isn't constraints.txt; runner-image.yml
holds the runner image to it before the push. Here: the comparison on given
lists, the committed file, and where the Dockerfile and workflows use it. No
image is built."""

from __future__ import annotations

import re
from pathlib import Path

import image_packages as ip
import remote_bundle as rb

REPO = Path(__file__).resolve().parents[1]
WANT = {"torch": "2.11.0+cu128", "tokenizers": "0.22.2", "datasets": "5.0.1", "numpy": "2.3.5"}
BOARD = {**WANT, "pytest": "8.4.2", "pluggy": "1.6.0", "iniconfig": "2.3.0"}


def test_the_same_packages_pass():
    want = {**BOARD}
    assert ip.compare(dict(BOARD), dict(WANT), want) == []


def test_each_difference_is_named():
    want = {**BOARD}
    board = {**BOARD, "tokenizers": "0.22.3", "rich": "14.0.0"}
    runner = {**WANT, "starlette": "1.7.0"}
    assert ip.compare(board, runner, want) == [
        "tokenizers: the board's 0.22.3, the runner's 0.22.2",
        "starlette 1.7.0: in the runner image only",
        "rich 14.0.0: in the board image only, and not from its own stage",
        "the board image: tokenizers 0.22.3, constraints.txt 0.22.2",
        "the board image: rich 14.0.0, not in constraints.txt",
        "the runner image: starlette 1.7.0, not in constraints.txt"]
    # two images alike, but not as constraints.txt says: still a failure
    stale = {**want, "datasets": "4.6.1", "pyarrow": "25.0.1"}
    assert ip.compare(dict(BOARD), dict(WANT), stale) == [
        "the board image: datasets 5.0.1, constraints.txt 4.6.1",
        "the board image: pyarrow 25.0.1 in constraints.txt, not in the image",
        "the runner image: datasets 5.0.1, constraints.txt 4.6.1",
        "the runner image: pyarrow 25.0.1 in constraints.txt, not in the image"]


def test_the_commands_exit_1_and_say_what_differs(monkeypatch, tmp_path, capsys):
    lists = {"board": dict(BOARD), "runner": dict(WANT)}
    monkeypatch.setattr(ip, "image_list", lambda image: dict(lists[image]))
    c = tmp_path / "constraints.txt"
    c.write_text(ip.HEADER + "".join(f"{p}=={v}\n" for p, v in sorted(BOARD.items())))
    assert ip.main(["compare", "board", "runner", "--constraints", str(c)]) == 0
    assert capsys.readouterr().out.startswith("the board and runner images: 4 packages the same")
    assert ip.main(["check", "runner", "--constraints", str(c)]) == 0
    lists["runner"]["tokenizers"] = "0.23.0"
    assert ip.main(["compare", "board", "runner", "--constraints", str(c)]) == 1
    out = capsys.readouterr().out
    assert "tokenizers: the board's 0.22.2, the runner's 0.23.0" in out
    assert "Make constraints.txt again" in out
    assert ip.main(["check", "runner", "--constraints", str(c)]) == 1
    assert "runner: tokenizers 0.23.0, constraints.txt 0.22.2" in capsys.readouterr().out
    # and `constraints` writes the file from an image, header first
    assert ip.main(["constraints", "board"]) == 0
    made = capsys.readouterr().out
    assert made.startswith(ip.HEADER) and ip.parse(made.splitlines()) == BOARD


def test_constraints_txt_is_the_boards_image_and_agrees_with_requirements():
    text = (REPO / "constraints.txt").read_text()
    pins = ip.parse(text.splitlines())
    body = [x for x in text.splitlines() if x and not x.startswith("#")]
    assert body and all(re.fullmatch(r"[A-Za-z0-9_.\-]+==\S+", x) for x in body)
    assert len(pins) == len(body)                            # one line a package
    for name in ("torch", "tokenizers", "starlette", "pydantic", "safetensors", "numpy",
                 *ip.BOARD_ONLY):
        assert name in pins, name
    # every pin in requirements.txt is the version constraints.txt holds
    for line in (REPO / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==")
            assert pins[ip.norm(name.split("[")[0])] == version, line


def test_both_stages_install_with_it_and_ci_compares_the_images():
    docker = (REPO / "Dockerfile").read_text()
    deps = docker.split("AS deps")[1].split("AS runner")[0]
    board = docker.split("AS board")[1]
    runner = docker.split("AS runner")[1].split("AS board")[0]
    assert "COPY requirements.txt constraints.txt /tmp/" in deps
    assert '-r /tmp/requirements.txt \\\n        $(if [ "$USE_CONSTRAINTS" = "1" ]; then echo ' \
           '"-c /tmp/constraints.txt"; fi)' in deps
    assert "mv /tmp/constraints.txt /opt/evalboard/constraints.txt" in deps
    assert '"pytest>=8" httpx \\\n        $(if [ "$USE_CONSTRAINTS" = "1" ]; then echo ' \
           '"-c /opt/evalboard/constraints.txt"; fi)' in board
    assert "pip install" not in runner                         # the deps stage's, as it is
    assert "constraints.txt" not in (REPO / ".dockerignore").read_text()
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    job = ci[ci.index("\n  docker-image:\n"):]
    assert job.index("--target runner -t aienh-runner:ci") < job.index(
        "python3 scripts/image_packages.py compare aienh-bench:ci aienh-runner:ci")
    wf = (REPO / ".github" / "workflows" / "runner-image.yml").read_text()
    assert wf.index('scripts/image_packages.py check "$IMAGE:') < wf.index("docker login ghcr.io")


def test_tokenizers_is_in_the_setup_record():
    assert "tokenizers" in rb.library_versions()
