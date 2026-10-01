"""12q.D (the brief's): what the first hf DeviceMark runs on the server found.
Five failed at "scoring: the IFEval checker failed: Resource 'punkt_tab' not
found" — the image now fetches it at build, pinned (scripts/nltk_data.py;
the check that finds it with no network is tests/test_image_deps.py, in CI's
image-deps job and deploy step 3). And Nemotron-3-Nano-4B can't run here:
transformers 5.5.3's built-in Nemotron-H has no plain MLP layers, and the
repo's own code keeps a cache 5.x's generation replaces — so it is refused
before it is queued, saying why, and their row on the chart says so. The
calibration rests on the three whose raw files match DeviceMark's board.
Nothing is fetched: the script is tried on a zip made here."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

import devicemark as dm
import nltk_data
from conftest import make_service
from service import catalog, config, hfmeta

SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12qd"
NEMO = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
THEIR_NEMO = "nemotron-4b__int8hu__aimodel"


# ---------------------------------------------------------------------------
# punkt_tab, pinned
# ---------------------------------------------------------------------------

def test_the_pin_is_a_commit_a_size_and_a_git_blob():
    assert len(nltk_data.COMMIT) == 40 and len(nltk_data.BLOB) == 40
    assert nltk_data.URL == ("https://raw.githubusercontent.com/nltk/nltk_data/"
                             f"{nltk_data.COMMIT}/packages/tokenizers/punkt_tab.zip")
    assert nltk_data.SIZE == 4_319_076 and str(nltk_data.DEST) == "/usr/share/nltk_data"
    # git's own hash of a file's bytes: `printf 'hello\n' | git hash-object --stdin`
    assert nltk_data.blob_hash(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def _zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("punkt_tab/english/abbrev_types.txt", "e.g\ni.e\n")
        z.writestr("punkt_tab/README", "made here, for the test")
    return buf.getvalue()


def test_the_script_unpacks_the_pinned_bytes_and_refuses_any_others(tmp_path, monkeypatch):
    data = _zip()
    seen = []

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(nltk_data.urllib.request, "urlopen",
                        lambda url, timeout=120: (seen.append(url), Reply(data))[1])
    # pinned to these bytes, it unpacks them where NLTK looks
    monkeypatch.setattr(nltk_data, "SIZE", len(data))
    monkeypatch.setattr(nltk_data, "BLOB", nltk_data.blob_hash(data))
    got = nltk_data.fetch(tmp_path / "nltk_data")
    assert got == tmp_path / "nltk_data" / "tokenizers" / "punkt_tab"
    assert (got / "english" / "abbrev_types.txt").read_text() == "e.g\ni.e\n"
    assert seen == [nltk_data.URL]
    # any other bytes: nothing unpacked
    monkeypatch.setattr(nltk_data, "BLOB", "0" * 40)
    with pytest.raises(SystemExit, match="not the pinned one"):
        nltk_data.fetch(tmp_path / "other")
    assert not (tmp_path / "other").exists()


# ---------------------------------------------------------------------------
# Nemotron-3-Nano-4B can't run here
# ---------------------------------------------------------------------------

def test_the_list_says_why_and_the_approved_entry_agrees():
    why = catalog.cant_run_here(NEMO)
    assert "no plain MLP layers" in why and "5.6.0" in why and "4.53" in why
    assert catalog.cant_run_here("Qwen/Qwen3.5-4B") == ""
    entry = catalog.approved()[NEMO]
    assert entry["commit"].startswith("dfaf35de3e30") and "12q.D" in entry["why"]


def test_preflight_refuses_it_before_anything_is_fetched(monkeypatch):
    monkeypatch.setenv("STUB_PREFLIGHT", "1")          # the stand-in comes after the refusal
    with pytest.raises(hfmeta.PreflightError, match="can't run on this server: transformers"):
        hfmeta.preflight(NEMO, "instruct")
    assert hfmeta.preflight("Qwen/Qwen3.5-4B", "instruct")["params"]


@pytest.mark.parametrize("suite", ["devicemark", "full", "generative"])
def test_it_is_never_queued(tmp_path, monkeypatch, suite):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    try:
        r = client.post("/api/submissions", json={"hf_id": NEMO, "suite": suite,
                                                   "kind": "instruct", "submitter": "masein"},
                        headers={"X-Token": config.SUBMIT_TOKEN or ""})
        assert r.status_code == 422
        assert r.json()["detail"].startswith(f"{NEMO} can't run on this server: ")
        assert r.json()["detail"].endswith("Nothing was queued.")
    finally:
        client.__exit__(None, None, None)


def test_their_row_says_ours_cant_run_here_and_calibration_rests_on_three(tmp_path):
    assert dm.CALIBRATION == ("Qwen/Qwen3.5-4B", "Nanbeige/Nanbeige4.1-3B", "tencent/Youtu-LLM-2B")
    assert set(dm.CALIBRATION) <= set(dm.THEIR_OPEN.values())
    theirs = {r["id"]: r for r in dm.board(tmp_path)["external"]["rows"]}
    assert theirs[THEIR_NEMO]["cant_run"] == catalog.cant_run_here(NEMO)
    assert not any(r.get("cant_run") for i, r in theirs.items() if i != THEIR_NEMO)
    # their numbers, never changed by it
    assert theirs[THEIR_NEMO]["composite"] == {"value": 0.6136, "ci": [0.5782, 0.6511]}


@pytest.mark.dashboard
def test_on_the_chart_their_row_and_their_point_say_so(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&chip=ondevice")
    page.wait_for_selector("[data-dm-chart]")
    cell = page.locator(f"[data-dm-composite='{THEIR_NEMO}']")
    assert cell.locator(f"[data-dm-theirs='{THEIR_NEMO}']").inner_text() == \
        "theirs (int8, scored on a Mac; speed on iPhone 17 Pro): 61.4 ±3.6"
    cant = cell.locator(f"[data-dm-cant='{THEIR_NEMO}']")
    assert cant.inner_text() == "ours: can’t run here"
    assert json.loads(cant.get_attribute("data-tip")) == [catalog.cant_run_here(NEMO)]
    tip = json.loads(page.locator(f"[data-dm-point='{THEIR_NEMO}']").get_attribute("data-tip"))
    assert f"ours: can't run here — {catalog.cant_run_here(NEMO)}" in tip
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator(f"[data-dm-row='{THEIR_NEMO}']").screenshot(path=SCREENS / "nemotron-cant-run.png")
    assert page.errors == []
