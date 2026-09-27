"""12f.4: the image's Python has what its runs import. A served model sits the
generative suite through lm_eval's local-chat-completions, which needs lm_eval's
`api` extra; the image lacked it, and the first two served runs failed at once
("missing package"). The fake-server tests never import lm_eval, so they missed
it. These do — in CI's image-deps job (the image's requirements over a CPU
torch, EVALBOARD_IMAGE_DEPS=1, where a missing lm_eval fails) and in deploy
step 3, inside the image. Elsewhere, with no lm_eval at all, they skip."""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^lm_eval\[([^\]]+)\]==([\d.]+)", re.M)
# a distribution whose module is named otherwise
MODULE = {"beautifulsoup4": "bs4", "pyyaml": "yaml", "scikit-learn": "sklearn"}


def lm_eval():
    try:
        return importlib.import_module("lm_eval")
    except ImportError:
        if os.environ.get("EVALBOARD_IMAGE_DEPS") == "1":
            raise
        pytest.skip("lm_eval isn't installed here: CI's image-deps job and deploy step 3 run this")


def test_the_pin_carries_the_extras_the_runs_use():
    m = PIN.search((ROOT / "requirements.txt").read_text(encoding="utf-8"))
    assert m, "requirements.txt pins lm_eval with its extras"
    assert {"ifeval", "math", "api"} <= {x.strip() for x in m.group(1).split(",")}


def test_what_is_installed_is_the_pin():
    lm_eval()
    assert md.version("lm_eval") == PIN.search(
        (ROOT / "requirements.txt").read_text(encoding="utf-8")).group(2)


def test_local_chat_completions_and_every_package_of_the_api_extra_import():
    lm_eval()
    from lm_eval.api.registry import get_model
    cls = get_model("local-chat-completions")
    assert cls.__module__ == "lm_eval.models.openai_completions"
    importlib.import_module("lm_eval.models.openai_completions")
    # what lm_eval itself says the api extra needs
    extra = [re.split(r"[\s;<>=!~\[]", r, maxsplit=1)[0] for r in md.requires("lm_eval") or []
             if re.search(r"extra\s*==\s*['\"]api['\"]", r)]
    assert extra, "lm_eval declares an api extra"
    for dist in extra:
        name = MODULE.get(dist.lower(), dist.replace("-", "_").lower())
        importlib.import_module(name)
