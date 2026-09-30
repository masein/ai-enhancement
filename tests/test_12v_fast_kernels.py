"""12v: the hybrid models' fast paths, in the image.

Granite-4.0-H-1B ran out of GPU memory at a batch of one (#132, #144, #147):
its log says "The fast path is not available … Falling back to the naive
implementation", and Qwen3.5's says the same of flash-linear-attention. The
image now carries the three packages transformers looks for:

- mamba_ssm and causal_conv1d, fetched already built for the image's torch and
  CUDA from Hugging Face's kernels-community, every file pinned by commit,
  size and hash (scripts/fast_kernels.json); other bytes are refused, and a
  dropped connection is taken up where it left off;
- fla-core, pinned in requirements-kernels.txt;
- all in /opt/fast-kernels, on every Python's path through one line in
  site-packages — unless FAST_KERNELS=0, the way back with no rebuild;
- and the build checks that Triton's C extension builds, which every fast
  path needs at its first call.

Nothing is fetched here: the server is a stand-in. tests/test_image_deps.py
imports the three in the image."""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import fast_kernels as fk
from service import hfmeta

ROOT = Path(__file__).resolve().parents[1]
DOCKER = (ROOT / "Dockerfile").read_text(encoding="utf-8")


def blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# ---------------------------------------------------------------------------
# what is pinned
# ---------------------------------------------------------------------------

def test_the_two_kernels_are_pinned_file_by_file():
    man = fk.manifest()
    assert set(man["kernels"]) == {"mamba_ssm", "causal_conv1d"}
    for pkg, k in man["kernels"].items():
        assert k["repo"].startswith("kernels-community/") and re.fullmatch(r"[0-9a-f]{40}", k["commit"])
        files = k["files"]
        # the package itself, and its one compiled library, by sha256
        assert "__init__.py" in files and "_ops.py" in files
        [so] = [f for f in files if f.endswith(".so")]
        assert so == f"{k['id']}.abi3.so" and re.fullmatch(r"[0-9a-f]{64}", files[so]["sha256"])
        for rel, want in files.items():
            assert want["size"] >= 0 and ("sha256" in want) != ("blob" in want), rel
            assert re.fullmatch(r"[0-9a-f]{40}", want.get("blob", "0" * 40)), rel
            assert not rel.startswith(("/", "..")) and ".." not in Path(rel).parts, rel
            # the loader's shim for an older `kernels` isn't a part of the package
            assert not rel.startswith(pkg + "/"), rel
        # built for the 5090 (Blackwell, sm_120)
        assert any(a.startswith("12.0") for a in k["archs"])
    assert man["kernels"]["mamba_ssm"]["needs"] == ["einops"]
    # what transformers takes from them is in the files fetched
    mamba = man["kernels"]["mamba_ssm"]["files"]
    assert {"ops/triton/selective_state_update.py", "ops/triton/ssd_combined.py"} <= set(mamba)
    assert "causal_conv1d_interface.py" in man["kernels"]["causal_conv1d"]["files"]
    # the sizes the PR asks the build to fetch
    assert sum(f["size"] for k in man["kernels"].values() for f in k["files"].values()) == 718_285_448


def test_the_builds_are_for_the_images_torch_and_cuda():
    # a new base image means new builds: the variant is named by both
    m = re.search(r"ARG BASE_IMAGE=pytorch/pytorch:(\d+)\.(\d+)\.\d+-cuda(\d+)\.(\d+)-", DOCKER)
    assert m, "the Dockerfile's base image"
    torch_major, torch_minor, cu_major, cu_minor = m.groups()
    assert fk.manifest()["variant"] == (
        f"torch{torch_major}{torch_minor}-cxx11-cu{cu_major}{cu_minor}-x86_64-linux")
    k = fk.manifest()["kernels"]["causal_conv1d"]
    assert fk.url_of(k, "torch211-cxx11-cu128-x86_64-linux", "_ops.py") == (
        f"https://huggingface.co/kernels-community/causal-conv1d/resolve/{k['commit']}"
        "/build/torch211-cxx11-cu128-x86_64-linux/_ops.py")


def test_fla_and_einops_are_pinned():
    kernels = (ROOT / "requirements-kernels.txt").read_text(encoding="utf-8")
    assert re.findall(r"^[A-Za-z].*$", kernels, re.M) == ["fla-core==0.5.2"]
    assert re.search(r"^einops==0\.8\.2$", (ROOT / "requirements.txt").read_text(), re.M)
    # the core only: not the distribution that registers fla's own models with transformers
    assert not re.search(r"^flash-linear-attention", kernels, re.M)


# ---------------------------------------------------------------------------
# the fetch
# ---------------------------------------------------------------------------

class Hub:
    """a stand-in for the Hub: what each URL holds, every request as asked,
    and a connection that drops after `drop_after` bytes, once"""

    def __init__(self, files: dict[str, bytes], drop_after: int | None = None,
                 ranges: bool = True):
        self.files, self.drop_after, self.ranges, self.asked = files, drop_after, ranges, []

    def __call__(self, req, timeout=None):
        rng = req.get_header("Range")
        self.asked.append((req.full_url, rng))
        data = self.files[req.full_url]
        start = int(rng.split("=")[1].rstrip("-")) if rng and self.ranges else 0
        hub, body = self, io.BytesIO(data[start:])

        class Reply:
            status = 206 if rng and hub.ranges else 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n):
                if hub.drop_after is not None and body.tell() >= hub.drop_after:
                    hub.drop_after = None
                    raise ConnectionResetError("the connection dropped")
                return body.read(min(n, 7))                 # in small pieces
        return Reply()


def _small(tmp_path, so: bytes = b"\x7fELF" + bytes(range(200))):
    init, ops = b"from ._ops import ops\n", b"import torch\n"
    man = {"variant": "torch211-cxx11-cu128-x86_64-linux", "kernels": {"causal_conv1d": {
        "repo": "kernels-community/causal-conv1d", "commit": "c" * 40, "id": "_cc",
        "files": {"__init__.py": {"size": len(init), "blob": blob(init)},
                  "sub/_ops.py": {"size": len(ops), "blob": blob(ops)},
                  "_cc.abi3.so": {"size": len(so), "sha256": hashlib.sha256(so).hexdigest()}}}}}
    k = man["kernels"]["causal_conv1d"]
    files = {fk.url_of(k, man["variant"], rel): data
             for rel, data in (("__init__.py", init), ("sub/_ops.py", ops), ("_cc.abi3.so", so))}
    return man, files, tmp_path / "fast-kernels"


def test_the_pinned_files_are_fetched_into_the_package(tmp_path):
    man, files, dest = _small(tmp_path)
    hub = Hub(files)
    said = fk.fetch(dest, man, hub)
    pkg = dest / "causal_conv1d"
    assert (pkg / "__init__.py").read_bytes() == b"from ._ops import ops\n"
    assert (pkg / "sub" / "_ops.py").exists() and (pkg / "_cc.abi3.so").stat().st_size == 204
    assert not list(dest.rglob("*.part"))
    assert said == ["causal_conv1d: 3 files, 239 bytes, kernels-community/causal-conv1d at "
                    "cccccccccccc (torch211-cxx11-cu128-x86_64-linux) · 3 fetched"]
    assert fk.verify(dest, man) == []
    # asked for at the pinned commit, never a branch
    assert all(f"/resolve/{'c' * 40}/build/" in u for u, _ in hub.asked)
    # what is there as pinned isn't fetched again
    hub.asked.clear()
    assert fk.fetch(dest, man, hub)[0].endswith("· 0 fetched") and hub.asked == []


def test_other_bytes_are_refused_and_nothing_is_left(tmp_path):
    man, files, dest = _small(tmp_path)
    so = next(u for u in files if u.endswith(".so"))
    files[so] = files[so][:-1] + b"X"                       # the right size, other bytes
    with pytest.raises(SystemExit, match=r"_cc\.abi3\.so is not the pinned one: 204 bytes"):
        fk.fetch(dest, man, Hub(files))
    assert not (dest / "causal_conv1d" / "_cc.abi3.so").exists()
    assert not list(dest.rglob("*.part"))
    assert "causal_conv1d/_cc.abi3.so" in fk.verify(dest, man)
    # too long, and too short, as well
    for data in (files[so] + b"more", files[so][:50]):
        files[so] = data
        with pytest.raises(SystemExit, match="is not the pinned one"):
            fk.fetch(dest, man, Hub(files))
    # a file changed on disk afterwards is found
    good = _small(tmp_path)
    fk.fetch(dest, good[0], Hub(good[1]))
    (dest / "causal_conv1d" / "__init__.py").write_bytes(b"from ._ops import spo\n")
    assert fk.verify(dest, man) == ["causal_conv1d/__init__.py"]


def test_a_dropped_connection_is_taken_up_where_it_left_off(tmp_path):
    man, files, dest = _small(tmp_path)
    hub = Hub(files, drop_after=70)
    fk.fetch(dest, man, hub)
    assert fk.verify(dest, man) == []
    so = [(u, r) for u, r in hub.asked if u.endswith(".so")]
    assert [r for _, r in so] == [None, "bytes=70-"]        # the rest, not the whole again
    # a server that sends the whole file again is read from its start
    dest2 = tmp_path / "again"
    hub = Hub(files, drop_after=70, ranges=False)
    fk.fetch(dest2, man, hub)
    assert fk.verify(dest2, man) == []
    # one that never holds gives up, and says why
    class Down(Hub):
        def __call__(self, req, timeout=None):
            self.asked.append((req.full_url, None))
            raise ConnectionRefusedError("no route")
    down = Down(files)
    with pytest.raises(SystemExit, match=r"is not the pinned one: 0 bytes.*\(no route\)"):
        fk.fetch(tmp_path / "down", man, down)
    assert len(down.asked) == fk.TRIES


# ---------------------------------------------------------------------------
# on the path, unless switched off
# ---------------------------------------------------------------------------

def _path_has(site_dir: Path, dest: Path, env: dict) -> bool:
    code = (f"import site, sys; site.addsitedir({str(site_dir)!r}); "
            f"print({str(dest)!r} in sys.path, sys.path[-1] == {str(dest)!r})")
    import os
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**{k: v for k, v in os.environ.items() if k != fk.SWITCH}, **env})
    assert r.returncode == 0, r.stderr
    return r.stdout.split() == ["True", "True"]


def test_the_folder_is_on_every_pythons_path_unless_switched_off(tmp_path):
    dest, site_dir = tmp_path / "fast-kernels", tmp_path / "site"
    dest.mkdir()
    p = fk.install_pth(site_dir, dest)
    assert p.name == fk.PTH and p.read_text().count("\n") == 1 and p.read_text().startswith("import ")
    # on by default (deploy step 3 runs with an empty environment), last on the path
    assert _path_has(site_dir, dest, {})
    assert _path_has(site_dir, dest, {"FAST_KERNELS": "1"})
    for off in ("0", "off", "OFF", "false", "no", " 0 "):
        assert not _path_has(site_dir, dest, {"FAST_KERNELS": off}), off


def test_the_memory_estimate_follows_what_is_on_the_path(tmp_path, monkeypatch):
    # 12q.G sizes Granite's batch by whether the Mamba kernels are there: it looks where
    # Python looks, so FAST_KERNELS=0 (the folder off the path) is the slow path's estimate
    import importlib.util
    here = all(importlib.util.find_spec(m) for m in ("mamba_ssm", "causal_conv1d"))
    assert hfmeta.mamba_kernels() is here
    for m in ("mamba_ssm", "causal_conv1d"):
        (tmp_path / m).mkdir()
        (tmp_path / m / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert hfmeta.mamba_kernels() is True


# ---------------------------------------------------------------------------
# the image, the compose file and CI
# ---------------------------------------------------------------------------

def test_the_image_fetches_installs_and_checks_them_at_build():
    assert "ARG WITH_FAST_KERNELS=1" in DOCKER and "WITH_MAMBA" not in DOCKER
    assert ("COPY scripts/fast_kernels.py scripts/fast_kernels.json requirements-kernels.txt "
            "/tmp/fast-kernels/") in DOCKER
    # fla beside the kernels, with no dependency pulled in; and nothing installed with apt
    # (the base image has Triton's compiler: the check builds with it)
    assert "apt-get" not in DOCKER
    assert re.search(r"pip install --no-cache-dir --break-system-packages --no-deps \\\n\s+"
                     r"--target /opt/fast-kernels -r /tmp/fast-kernels/requirements-kernels\.txt",
                     DOCKER)
    # fetched, then checked in a fresh Python: a build where they don't import fails
    assert "python /tmp/fast-kernels/fast_kernels.py --dest /opt/fast-kernels \\\n" in DOCKER
    assert "python /tmp/fast-kernels/fast_kernels.py --dest /opt/fast-kernels --check;" in DOCKER
    assert str(fk.DEST) == "/opt/fast-kernels"
    # no source build of the two is left
    assert "mamba-ssm==" not in DOCKER and "causal-conv1d==" not in DOCKER


def test_compose_has_the_build_arg_and_the_switch():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "WITH_FAST_KERNELS: ${WITH_FAST_KERNELS:-1}" in compose and "WITH_MAMBA" not in compose
    assert "FAST_KERNELS: ${FAST_KERNELS:-1}" in compose
    assert fk.SWITCH == "FAST_KERNELS" and "0" in fk.OFF


def test_ci_installs_fla_and_the_images_triton():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'pip install "triton==3.6.0" -r requirements-kernels.txt' in ci


def test_check_says_when_the_switch_is_off(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("FAST_KERNELS", "0")
    assert fk.main(["--check", "--dest", str(tmp_path)]) == 0
    assert capsys.readouterr().out.startswith("FAST_KERNELS=0: the fast kernels are off")
    # on, and nothing there: every pinned file is named as missing
    monkeypatch.delenv("FAST_KERNELS")
    with pytest.raises(SystemExit, match=r"not as pinned under .*: mamba_ssm/__init__\.py"):
        fk.main(["--check", "--dest", str(tmp_path)])


def test_the_names_checked_are_the_ones_transformers_takes():
    assert set(fk.NAMES) == {"mamba_ssm.ops.triton.selective_state_update",
                             "mamba_ssm.ops.triton.ssd_combined", "causal_conv1d", "fla.modules",
                             "fla.ops.gated_delta_rule"}
    assert {m.split(".")[0] for m in fk.NAMES} == set(fk.MODULES)
    assert json.loads(fk.MANIFEST.read_text())["variant"] == fk.manifest()["variant"]
