"""18c part 1: before step A — the docs' own lines run Harbor from the agent
venv, under sudo or not; the build's files are fetched once from a lock 14
days old and checked, and every SWE-bench image is built with no network; a
clean-up that didn't run keeps the task from being verified; step A3 moves
an existing venv aside.

The docs' lines run for real, in bash, against a venv laid out as uv makes
it (its python a link, its packages' metadata as the lock has them) with
stand-ins for what reaches outside: Harbor, pip's download, Docker, git and
sudo. Invented tasks; no model, nothing fetched."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_bench as ab
import agent_host_mini as hm
import agent_run as ar
from agent18 import INVENTED_PARSER, build_world, tasks_folder, trial
from test_18b1_containment import Env, host_agent, submitted

ROOT = Path(__file__).resolve().parent.parent
DOC = (ROOT / "docs" / "AGENT-RUNS.md").read_text()

PARSER = """#!/bin/bash
cd /testbed
{fetch}
: '>>>>> Start Test Output'
invented-test-runner
: '>>>>> End Test Output'

cd ..
cat > parser.py <<EOF
# /// script
# requires-python = ">=3.11"
# dependencies = ["invented-parser==1.0", "invented-data==2.0"]
# ///
import sys
EOF
uv run parser.py
"""


def doc_lines(needle: str) -> list[str]:
    return [x.strip() for x in DOC.splitlines() if needle in x and x.strip().startswith("cd ")]


# ---------------------------------------------------------------------------
# 1. the docs' lines, run in bash against a venv layout
# ---------------------------------------------------------------------------

HARBOR = r'''#!{python}
# Harbor, stood in for: what it was run as, and a trial as Harbor 0.24.0 writes it
import json, os, sys
from pathlib import Path
with open(os.environ["HARBOR_LOG"], "a") as fh:
    fh.write(json.dumps({{"argv": sys.argv, "path": os.environ.get("PATH", "")}}) + "\n")
a = sys.argv[1:]
if a[0] == "download":
    top = Path(a[a.index("-o") + 1])
    for i, lang in enumerate(["c", "go", "java", "rust"]):
        t = top / f"inv__repo{{i}}-{{100 + i}}"
        (t / "environment").mkdir(parents=True, exist_ok=True)
        (t / "tests").mkdir(exist_ok=True)
        (t / "task.toml").write_text(f'[metadata]\ntags = ["swe-bench", "{{lang}}"]\n')
        (t / "environment" / "Dockerfile").write_text(
            "FROM invented/image:1\nRUN curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh || true\n")
        (t / "tests" / "test.sh").write_text(os.environ["PARSER_TEXT"])
elif a[0] == "run":
    task = Path(a[a.index("-p") + 1]).name
    t = Path(a[a.index("-o") + 1]) / a[a.index("--job-name") + 1] / f"{{task[:32]}}__abc1234"
    t.mkdir(parents=True)
    now = "2026-10-20T10:00:00+00:00"
    (t / "result.json").write_text(json.dumps({{
        "task_name": task, "started_at": now, "finished_at": now, "agent_info": {{"name": "oracle"}},
        "agent_execution": {{"started_at": now}}, "verifier_result": {{"rewards": {{"reward": 1.0}}}}}}))
'''

PIP = r'''# pip's download, stood in for: the wheels of an invented index
import os, shutil, sys
a = sys.argv[1:]
open(os.environ["HARBOR_LOG"] + ".pip", "w").write(" ".join(a))
d = a[a.index("-d") + 1]
os.makedirs(d, exist_ok=True)
for f in os.listdir(os.environ["FAKE_INDEX"]):
    shutil.copy(os.path.join(os.environ["FAKE_INDEX"], f), d)
'''

# the server's disk, stood in for: 500 GB free where Docker keeps its images
SITE = '''import shutil
from collections import namedtuple
shutil.disk_usage = lambda path: namedtuple("usage", "total used free")(10**12, 5 * 10**11, 5 * 10**11)
'''

DOCKER = '''#!/bin/sh
echo "docker $*" >> "$HARBOR_LOG.docker"
case "$*" in
  "info -f {{.DockerRootDir}}") echo /var/lib/docker ;;
  "network inspect"*) echo 172.17.0.1 ;;
  *import_agent.py*) echo "Runs #1: ok" ;;
  "image inspect"*) echo 0 ;;
esac
exit 0
'''

GIT = '''#!/bin/sh
# git, stood in for: DeepSWE's repository at its commit, four invented tasks
case "$*" in
  *checkout*) for i in 1 2 3 4; do t="$2/tasks/inv-deep-$i"; mkdir -p "$t/environment" "$t/tests"
      printf '[metadata]\\nlanguage = "go"\\n[verifier]\\nenvironment_mode = "separate"\\n' > "$t/task.toml"
      echo "FROM invented/deep:1" > "$t/environment/Dockerfile"
      echo "FROM invented/verifier:1" > "$t/tests/Dockerfile"; done ;;
esac
exit 0
'''

SUDO = '''#!/bin/sh
# sudo, stood in for: its own PATH (secure_path), never the caller's
PATH="$FAKESYS:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SUDO_USER=masein
export PATH SUDO_USER
exec "$@"
'''


def venv_world(tmp_path: Path) -> dict:
    """masein's home on the server: the checkout, the agent venv as step A
    makes it, the system's commands; the build's files on an invented index"""
    home = tmp_path / "home"
    venv = home / "agent-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    site = Path(subprocess.run([str(venv / "bin" / "python"), "-c",
                                "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                               capture_output=True, text=True, check=True).stdout.strip())
    for pkg, ver in ar.lock().items():                  # the lock, installed
        d = site / f"{pkg.replace('-', '_')}-{ver}.dist-info"
        d.mkdir(parents=True)
        (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {pkg}\nVersion: {ver}\n")
    (site / "pip").mkdir(exist_ok=True)
    (site / "pip" / "__init__.py").write_text("")
    (site / "pip" / "__main__.py").write_text(PIP)
    (site / "sitecustomize.py").write_text(SITE)
    harbor = venv / "bin" / "harbor"
    harbor.write_text(HARBOR.format(python=venv / "bin" / "python"))
    sysbin = tmp_path / "sys"
    sysbin.mkdir()
    for name, text in (("docker", DOCKER), ("git", GIT), ("sudo", SUDO)):
        (sysbin / name).write_text(text)
    for p in (harbor, *sysbin.iterdir()):
        p.chmod(0o755)
    # the checkout: the runner and its docs, the build's files pinned to the
    # invented index
    repo = home / "benchmarks" / "aienh"
    (repo / "scripts").mkdir(parents=True)
    for name in ("agent_run.py", "agent_bench.py", "agent_host_mini.py", "agent_relay.py"):
        shutil.copy(ROOT / "scripts" / name, repo / "scripts" / name)
    (repo / "docs").mkdir()
    shutil.copy(ROOT / "docs" / "agent-requirements.txt", repo / "docs")
    index = tmp_path / "index"
    index.mkdir()
    lock = ""
    for name in ("invented_parser-1.0-py3-none-any.whl", "invented_data-2.0-py3-none-any.whl"):
        (index / name).write_bytes(b"invented wheel " + name.encode())
        pkg, ver = name.split("-")[:2]
        lock += (f"{pkg.replace('_', '-')}=={ver} \\\n    --hash=sha256:"
                 f"{hashlib.sha256((index / name).read_bytes()).hexdigest()}\n")
    (repo / "docs" / "agent-build-requirements.txt").write_text(lock)
    spec = json.loads((ROOT / "docs" / "agent-build.json").read_text())
    for name in ("python", "uv"):
        f = tmp_path / f"{name}-download.tar.gz"
        f.write_bytes(f"invented {name}".encode())
        spec[name].update(url=f.as_uri(), sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    spec["parser"] = INVENTED_PARSER
    (repo / "docs" / "agent-build.json").write_text(json.dumps(spec))
    bench = tmp_path / "bench"
    (repo / ".env").write_text(f"BENCH_ROOT={bench}\n")
    log = tmp_path / "harbor.log"
    env = {"HOME": str(home), "PATH": f"{sysbin}:/usr/local/bin:/usr/bin:/bin",
           "FAKESYS": str(sysbin), "HARBOR_LOG": str(log), "FAKE_INDEX": str(index),
           "PARSER_TEXT": PARSER.replace("{fetch}", ""), "LANG": "C.UTF-8"}
    return {"home": home, "venv": venv, "bench": bench, "log": log, "env": env, "repo": repo}


def run_line(w: dict, line: str) -> tuple[int, str]:
    p = subprocess.run(["bash", "-c", line], env=w["env"], capture_output=True, text=True,
                       timeout=300)
    return p.returncode, p.stdout + p.stderr


def test_the_docs_lines_run_harbor_from_the_agent_venv_under_sudo(tmp_path):
    w = venv_world(tmp_path)
    a8 = next(x for x in doc_lines("--oracle --tasks 10"))
    a10 = next(x for x in doc_lines("deepswe --oracle --tasks 3"))
    for line in (a8, a10):
        assert line.startswith('cd ~/benchmarks/aienh && sudo env PATH="$HOME/agent-venv/bin:$PATH" '
                               "~/agent-venv/bin/python scripts/agent_run.py "), line
    code, out = run_line(w, a8)
    assert code == 0, out                                   # d94f7f6: harbor: not found
    assert "SWE-bench Multilingual: 4 of 4 · 4 resolved" in out, out
    code, out = run_line(w, a10)
    assert code == 0, out                                   # d94f7f6: FileNotFoundError
    assert "Traceback" not in out and "DeepSWE 1.1: 3 of 3 · 3 resolved" in out, out
    # the docker-group form: the same line without sudo
    code, out = run_line(w, a10.replace("sudo env", "env") + " --run 2")
    assert code == 0 and "3 of 3 · 3 resolved" in out, out
    # the 18b line, sudo with no PATH: the runner still runs the venv's Harbor
    code, out = run_line(w, "cd ~/benchmarks/aienh && sudo ~/agent-venv/bin/python "
                            "scripts/agent_run.py deepswe --oracle --tasks 3 --run 3")
    assert code == 0 and "3 of 3 · 3 resolved" in out, out
    calls = [json.loads(x) for x in w["log"].read_text().splitlines()]
    assert {c["argv"][1] for c in calls} == {"download", "run"}
    harbor = str(w["venv"] / "bin" / "harbor")
    for c in calls:
        assert c["argv"][0] == harbor, c["argv"]
        assert c["path"].split(":")[0] == str(w["venv"] / "bin"), c["path"]   # first on PATH
    assert sum(c["argv"][1] == "run" for c in calls) == 4 + 3 * 3
    # the build's files: fetched once, through the venv's pip, wheels only, hashed
    pip = (tmp_path / "harbor.log.pip").read_text()
    for flag in ("download", "--require-hashes", "--no-deps", "--only-binary=:all:",
                 "--isolated", "--python-version 3.11"):
        assert flag in pip, flag
    run = json.loads(next((w["bench"] / "agent-runs").glob("swebench*/run.json")).read_text())
    assert run["build_lock"]["exclude_newer"] == "2026-09-24" and run["by"] == "masein"


def test_a_python_without_harbor_beside_it_is_refused_in_one_line(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "executable", str(tmp_path / "bin" / "python"))
    assert ar.harbor_path() == tmp_path / "bin" / "harbor"
    why = ar.harbor_line()
    assert why.startswith("Harbor's command isn't beside this python (") and "agent venv" in why
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "harbor").write_text("#!/bin/sh\n")
    (tmp_path / "bin" / "harbor").chmod(0o755)
    assert ar.harbor_line() == ""
    env = ar.job_env()
    assert env["PATH"].split(os.pathsep)[0] == str(tmp_path / "bin")


# ---------------------------------------------------------------------------
# 2. the build's files: once, from a lock 14 days old, each checked
# ---------------------------------------------------------------------------

def test_the_build_lock_is_fourteen_days_old_and_hashed():
    spec = ar.build_spec()
    when = datetime.date.fromisoformat(spec["exclude_newer"])
    assert (datetime.date(2026, 10, 8) - when).days >= ar.FRESH_DAYS
    lock = ar.lock_hashes((ROOT / "docs" / spec["packages"]).read_text())
    assert len(lock) == 78 and lock["swebench"][0] == "4.1.0" and lock["datasets"][0] == "2.16.1"
    assert all(hashes and all(len(h) == 64 for h in hashes) for _, hashes in lock.values())
    for name in ("python", "uv"):
        assert spec[name]["url"].startswith("https://github.com/astral-sh/")
        assert re.fullmatch(r"[0-9a-f]{64}", spec[name]["sha256"])
    assert spec["uv"]["version"] == "0.7.13"                  # the tasks' own uv
    assert spec["parser"] == {"requires-python": ">=3.11",
                              "dependencies": ["swebench==4.1.0", "datasets==2.16.1"]}


def invented_build(tmp_path, monkeypatch) -> dict:
    """docs/agent-build.json and its lock, pinned to invented files"""
    files = {}
    for name in ("python", "uv"):
        f = tmp_path / "dl" / f"{name}.tar.gz"
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(f"invented {name}".encode())
        files[name] = f
    index = tmp_path / "index"
    index.mkdir()
    lock = ""
    for name in ("invented_parser-1.0-py3-none-any.whl", "invented_data-2.0-py3-none-any.whl"):
        (index / name).write_bytes(name.encode())
        pkg, ver = name.split("-")[:2]
        lock += (f"{pkg.replace('_', '-')}=={ver} \\\n    --hash=sha256:"
                 f"{hashlib.sha256(name.encode()).hexdigest()}\n")
    spec = {**ar.build_spec(), "exclude_newer": "2026-09-24", "parser": INVENTED_PARSER,
            "packages": "lock.txt"}
    for name, f in files.items():
        spec[name] = {**spec[name], "url": f.as_uri(),
                      "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "lock.txt").write_text(lock)
    (docs / "agent-build.json").write_text(json.dumps(spec))
    monkeypatch.setattr(ar, "BUILD", docs / "agent-build.json")
    calls = []

    def run(cmd, cwd=None, timeout=None, env=None):
        calls.append(cmd)
        d = Path(cmd[cmd.index("-d") + 1])
        for f in index.iterdir():
            shutil.copy(f, d / f.name)
        return 0, ""
    monkeypatch.setattr(ar, "run", run)
    return {"calls": calls, "files": files, "index": index, "spec": spec, "docs": docs}


def test_the_build_files_are_fetched_once_and_checked_every_run(tmp_path, monkeypatch):
    w = invented_build(tmp_path, monkeypatch)
    today = datetime.date(2026, 10, 20)
    d, info, why = ar.build_files(tmp_path / "bench", today)
    assert why == "" and d.parent.name == "agent-build" and len(w["calls"]) == 1
    pip = w["calls"][0]
    assert pip[:4] == [sys.executable, "-m", "pip", "download"]
    for flag in ("--require-hashes", "--no-deps", "--only-binary=:all:", "--isolated"):
        assert flag in pip
    assert pip[pip.index("--python-version") + 1] == "3.11" and "manylinux_2_17_x86_64" in pip
    assert info["exclude_newer"] == "2026-09-24" and info["packages"] == 2
    assert sorted(p.name for p in d.iterdir()) == ["fetched.json", "python.tar.gz", "uv.tar.gz",
                                                   "wheels"]
    # again: nothing fetched, everything checked
    assert ar.build_files(tmp_path / "bench", today) == (d, info, "") and len(w["calls"]) == 1
    # a file changed after it was fetched: refused, said
    (d / "wheels" / "invented_data-2.0-py3-none-any.whl").write_bytes(b"something else")
    _, _, why = ar.build_files(tmp_path / "bench", today)
    assert "aren't as pinned (invented_data-2.0-py3-none-any.whl isn't the file the lock pins" \
           in why and "remove that folder" in why
    shutil.rmtree(d)
    # a download that isn't the file pinned
    w["files"]["uv"].write_bytes(b"a newer uv")
    _, _, why = ar.build_files(tmp_path / "bench", today)
    assert "isn't the file pinned (its sha256" in why
    # a lock younger than 14 days: refused
    _, _, why = ar.build_files(tmp_path / "bench", datetime.date(2026, 10, 1))
    assert why.startswith("docs/agent-build.json's packages are only 7 days old")


def multilingual_tasks(d: Path) -> list[str]:
    names = tasks_folder(d, 5)
    fetch = ["", "npm install", "composer update",
             "cargo update x@1.0 --precise 0.9 2>/dev/null || true", ""]
    for n, f in zip(names, fetch):
        (d / n / "environment" / "Dockerfile").write_text(
            "FROM invented/image:1\nWORKDIR /testbed\n"
            "RUN curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh || true\n"
            'ENV PATH="/root/.local/bin:${PATH}"\n')
        (d / n / "tests").mkdir()
        (d / n / "tests" / "test.sh").write_text(PARSER.replace("{fetch}", f))
    return names


def test_every_multilingual_image_is_built_with_no_network(tmp_path, monkeypatch):
    src = tmp_path / "src"
    names = multilingual_tasks(src)
    build, info = build_world(tmp_path, monkeypatch)
    b = ab.BENCHES["swebench-multilingual"]
    out, got = ar.offline_tasks(tmp_path, b, src, names, build, info)
    plain = out / names[0] / "environment"
    df = (plain / "Dockerfile").read_text()
    # the task's own uv installer left out; the same uv, a Python and the
    # parser's packages from the build's files, installed with no network
    assert "\nRUN curl -LsSf https://astral.sh/uv" not in df                  # d94f7f6: run
    assert "# left out by evalboard's agent runner (the same uv comes from the build's " \
           "files): curl -LsSf https://astral.sh/uv/0.7.13/install.sh" in df
    assert "COPY evalboard-offline /opt/evalboard-offline" in df
    assert ("ENV UV_PYTHON=/opt/evalboard-python/bin/python3 UV_PYTHON_DOWNLOADS=never "
            "UV_OFFLINE=1 UV_NO_INDEX=1 UV_FIND_LINKS=/opt/evalboard-offline/wheels") in df
    assert df.index("ENV UV_PYTHON") < df.index("uv run parser.py")             # offline first
    assert (plain / "docker-compose.yaml").read_text() == ar.BUILD_NONE         # d94f7f6: none
    assert "network: none" in ar.BUILD_NONE
    # the build's files linked into its build folder, not copied
    for f in ("python.tar.gz", "uv.tar.gz", "wheels/invented_data-2.0-py3-none-any.whl"):
        assert os.path.samefile(plain / "evalboard-offline" / f, build / f)
    # tasks whose own tests fetch: Docker's own network for that, no package's
    # scripts; npm's held to the lock's date; composer and cargo can't be
    npm = (out / names[1] / "environment" / "Dockerfile").read_text()
    assert "npm_config_ignore_scripts=true" in npm and "npm_config_before=2026-09-24" in npm
    composer = (out / names[2] / "environment" / "Dockerfile").read_text()
    assert "( composer update --no-scripts --no-plugins )" in composer
    for n in names[1:4]:
        assert not (out / n / "environment" / "docker-compose.yaml").exists()
    assert got["build_network"] == names[1:4] and got["fetching"] == names[1:4]
    assert got["unpinned"] == names[2:4]                                        # npm is
    assert got["refused"] == {}
    # a task the copy can't be made for: refused, said — never overwritten
    (src / names[4] / "environment" / "docker-compose.yaml").write_text("services: {}\n")
    (src / names[0] / "tests" / "test.sh").write_text(PARSER.replace('"invented-data==2.0"',
                                                                     '"invented-data==3.0"'))
    _, got = ar.offline_tasks(tmp_path, b, src, names, build, info)
    assert got["refused"] == {
        names[0]: "its verifier's parser asks other packages than the build's lock",
        names[4]: "it has its own environment/docker-compose.yaml"}


def test_a_run_keeps_its_build_lock_and_refuses_another():
    assert "build_lock" in ar.SETTINGS
    before = {"benchmark": "x", "build_lock": {"sha256": "ab" * 32, "exclude_newer": "2026-09-24"}}
    now = {**before, "build_lock": {"sha256": "cd" * 32, "exclude_newer": "2026-10-01"}}
    diff = ar.changed_settings(before, now)
    assert len(diff) == 1 and diff[0].startswith("build_lock ")                 # d94f7f6: []


def test_the_build_check_runs_in_a_build_with_no_network(tmp_path, monkeypatch, capsys):
    """--check-reach builds an image as a task's copy is built, its check
    run in the build: nothing may answer; and says what a build with Docker's
    own network reaches"""
    seen = []
    ok = "#7 0.101 PROBE tcp-ok\n#7 0.2 NET lo\n#7 0.3 DONE\n"
    open_ = "#7 0.101 PROBE tcp-ok\n#7 1.0 REACHED 172.17.0.1:8899\n#7 1.1 REACHED 1.1.1.1:443\n" \
            "#7 1.2 DONE\n"

    def run(cmd, cwd=None, timeout=None, env=None):
        seen.append((cmd, env))
        if cmd[:2] == ["docker", "compose"] and "build" in cmd:
            compose = Path(cmd[cmd.index("-f") + 1]).read_text()
            return 0, ok if "network: none" in compose else open_
        if cmd[:3] == ["docker", "network", "inspect"]:
            return 0, "172.17.0.1\n"
        if cmd[:len(ar.BOARD)] == ar.BOARD:                  # 18c.2: the board's servers
            return 0, json.dumps([{"id": "served/x", "base_url": "http://172.17.0.1:8091/v1"}])
        return 0, ""
    monkeypatch.setattr(ar, "run", run)
    monkeypatch.setattr(ar, "check_reach", lambda rdir, targets: "")
    monkeypatch.setenv("BENCH_ROOT", str(tmp_path))
    assert ar.main(["swebench-multilingual", "--check-reach"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[1] == ("a task's build with no network (every SWE-bench Multilingual image, but "
                      "those whose own tests fetch packages) reaches none of them")
    assert out[2].startswith("a build with Docker's own network (") and \
        "reaches 172.17.0.1:8899, 1.1.1.1:443" in out[2]
    builds = [(c, e) for c, e in seen if "build" in c]
    assert all(e["BUILDKIT_PROGRESS"] == "plain" and "--no-cache" in c for c, e in builds)
    # a build with no network that reached something: refused
    ok = open_
    assert ar.main(["swebench-multilingual", "--check-reach"]) == 2
    assert "refused — the task's build reached 172.17.0.1:8899" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 3. a clean-up that didn't run: recorded, and the task never verified
# ---------------------------------------------------------------------------

def test_the_clean_ups_words_are_read_closed():
    assert hm.clean_verdict(0, "CLEAN files=0 processes=0\n") == ""
    assert hm.clean_verdict(1, "CLEAN files=2 processes=1\n") == "exit 1, files=2 processes=1"
    assert hm.clean_verdict(0, "") == "exit 0, no word from it"
    assert hm.clean_verdict(None, "CLEAN files=0 processes=0") == \
        "exit None, files=0 processes=0"
    # it says what it left, and what it can't read counts as left
    assert 'find "$d" -mindepth 1 2>&1' in hm.CLEAN
    assert hm.CLEAN.splitlines()[-1] == '[ "$left" = 0 ] && [ "$n" = 0 ]'      # its exit code


class Failing(Env):
    """Harbor's exec, its clean-up failing one way"""

    def __init__(self, how):
        super().__init__()
        self.how = how

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if command != hm.CLEAN:
            return await super().exec(command, cwd, env, timeout_sec, user)
        self.calls.append({"command": command, "user": user})
        if self.how == "no user":                       # an exec with no `user`
            raise TypeError("exec() got an unexpected keyword argument 'user'")
        if self.how == "timeout":
            raise RuntimeError("Command timed out after 180 seconds")
        return SimpleNamespace(stdout="CLEAN files=2 processes=1\n", stderr="", return_code=1)


@pytest.mark.parametrize("how", ["no user", "timeout", "left"])
def test_a_clean_up_that_didnt_run_is_recorded_and_never_verified(tmp_path, monkeypatch, how):
    agent, t = host_agent(tmp_path, monkeypatch, submitted)
    env = Failing(how)
    with pytest.raises(hm.CleanupFailed) as e:              # d94f7f6: swallowed, verified
        asyncio.run(agent.run("x", env, SimpleNamespace()))
    meta = json.loads((t / ab.HOST_DIR / ab.META).read_text())
    assert meta["clean"]["ok"] is False
    words = {"no user": "TypeError: exec() got an unexpected keyword argument 'user'",
             "timeout": "RuntimeError: Command timed out after 180 seconds",
             "left": "exit 1, files=2 processes=1"}[how]
    assert meta["clean"]["why"] == words and words in str(e.value)
    assert env.calls[-1] == {"command": hm.CLEAN, "user": "root"}


def test_a_clean_up_that_fails_at_the_time_limit_is_never_verified(tmp_path, monkeypatch):
    """Harbor verifies after its time limit (the cancel); a failed clean-up
    there is raised instead, which Harbor never verifies after"""
    import threading
    import time

    def behaviour(agent):
        while not agent.env.stop.is_set():
            time.sleep(0.01)
        raise hm.Stopped("the agent's time is up")
    agent, t = host_agent(tmp_path, monkeypatch, behaviour)
    env = Failing("left")

    async def go():
        await asyncio.wait_for(agent.run("x", env, SimpleNamespace()), timeout=0.3)
    with pytest.raises(hm.CleanupFailed):                    # not a TimeoutError: no verifier
        asyncio.run(go())
    assert threading.active_count() >= 1


def test_a_clean_up_that_ran_is_kept_and_the_task_verified(tmp_path, monkeypatch):
    agent, t = host_agent(tmp_path, monkeypatch, submitted)
    asyncio.run(agent.run("x", Env(), SimpleNamespace()))
    meta = json.loads((t / ab.HOST_DIR / ab.META).read_text())
    assert meta["clean"] == {"exit": 0, "ok": True, "why": ""}


def test_a_task_whose_clean_up_failed_is_an_error_of_ours_never_asked_again(tmp_path):
    rd = tmp_path / "r"
    trial(rd, "inv__c-1", result="", exc="CleanupFailed", message="the clean-up after the agent "
          "didn't run to its end (exit 1, files=2 processes=1): not verified")
    r = ab.read_trial(next((rd / "jobs").glob("*/*")))
    assert r["result"] == "error" and r["final"] is True
    assert r["why"].endswith("after the agent ran: counted not resolved, never asked again")
    assert ar.state_of(rd, "inv__c-1", 1) == ("done", 1)                       # never again
    assert ab.score([r], 300)["errors"] == 1


# ---------------------------------------------------------------------------
# 4. step A3 moves an existing venv aside first
# ---------------------------------------------------------------------------

def test_step_a3_moves_an_existing_venv_aside_first(tmp_path):
    a = DOC[DOC.index("## A."):DOC.index("## B.")]
    step3 = a[a.index("\n3. "):a.index("\n4. ")]
    move = "[ -e ~/agent-venv ] && mv ~/agent-venv ~/agent-venv.old.$(date +%s)"
    assert move in step3 and step3.index(move) < step3.index("uv venv --python 3.12")
    # it works: an existing venv moves aside, a missing one is no error
    home = tmp_path / "home"
    (home / "agent-venv" / "bin").mkdir(parents=True)
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    subprocess.run(["bash", "-c", move], env=env, check=True)
    assert not (home / "agent-venv").exists() and len(list(home.glob("agent-venv.old.*"))) == 1
    assert subprocess.run(["bash", "-c", move], env=env).returncode == 1   # nothing to move
    assert textwrap.dedent(step3).count("```bash") == 2
