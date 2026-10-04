"""14.4.1: the full Mobile-MMLU's data — 16,186 questions in 80 files, one a
field, CC BY-NC-ND 4.0, added on 4 Oct 2026 for internal research evaluation
only (masein's decision, recorded in its manifest). Pinned by revision and each
file's sha256, fetched by the data step into the server's data folder and
never committed; read with each file's field; Pro sits in it, every question
by id and all but one by wording. The files are never in the repo: every test
here uses invented rows (tests/fixtures/mmf_invented/), pinned as the real
files are. Nothing calls Hugging Face."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path

import pytest

import fetch_data
import mobile_mmlu as mmp
from service import config, contamination
from test_14_3_mobile_mmlu import _Resp, put_invented

REPO = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mmf_invented"


def pin_full(monkeypatch) -> list[dict]:
    """the manifest's pins, moved to the invented files"""
    real = mmp.full_manifest
    files = []
    for p in sorted((FIXTURE / "test").glob("*.csv")):
        raw = p.read_bytes()
        files.append({"file": f"test/{p.name}", "url": f"https://example.invalid/{p.name}",
                      "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                      "n": raw.count(b"\n") - 1, "committed": False})

    def pinned():
        return {**real(), "files": files, "n": sum(f["n"] for f in files)}
    monkeypatch.setattr(mmp, "full_manifest", pinned)
    mmp._full_rows.clear()
    return files


def put_full(monkeypatch, d: Path) -> Path:
    (d / "test").mkdir(parents=True, exist_ok=True)
    for p in (FIXTURE / "test").glob("*.csv"):
        (d / "test" / p.name).write_bytes(p.read_bytes())
    monkeypatch.setenv("MMF_DIR", str(d))
    monkeypatch.setattr(config, "MMF_DIR", d)
    pin_full(monkeypatch)
    return d


@pytest.fixture
def full(tmp_path, monkeypatch):
    """the full Mobile-MMLU on this "server": the invented rows"""
    return put_full(monkeypatch, tmp_path / "data" / "mobile_mmlu")


def test_the_manifest_pins_every_file_and_records_the_decision():
    m = mmp.full_manifest()
    assert m["source"] == "https://huggingface.co/datasets/MBZUAI-LLM/Mobile-MMLU"
    assert len(m["revision"]) == 40 and len(m["files"]) == 80 and m["n"] == 16_186
    assert sum(f["n"] for f in m["files"]) == 16_186 == sum(m["fields"].values())
    for f in m["files"]:
        assert f["committed"] is False and len(f["sha256"]) == 64 and f["bytes"] > 0
        assert f["url"] == (f"https://huggingface.co/datasets/MBZUAI-LLM/Mobile-MMLU/resolve/"
                            f"{m['revision']}/{f['file']}")
    # every field is one of the paper's 9 categories', as Pro's are
    assert set(m["fields"]) == set(mmp.FIELD_CATEGORY)
    assert (m["licence"], m["restriction"]) == ("CC BY-NC-ND 4.0", "non-commercial")
    d = m["decision"]
    assert (d["by"], d["on"]) == ("masein", "2026-10-04")
    assert "internal research evaluation only" in d["what"]
    assert m["gated"].startswith("no, checked 4 Oct 2026")
    # how the two sets relate, as checked: Pro in the full set, one question worded apart
    o = m["overlap"]
    assert (o["pro"], o["full"], o["pro_in_full"], o["same_wording"], o["full_only"]) == (
        9497, 16186, 9497, 9496, 6689)
    assert list(o["worded_apart"]) == ["9933ec55"]
    # the paper's full-set column, beside Pro's from the same table
    pc = m["paper_checks"]
    assert pc["models"] == {"Qwen/Qwen2.5-3B-Instruct": 68.1,
                            "meta-llama/Llama-3.2-3B-Instruct": 50.2, "google/gemma-2-2b-it": 38.9}
    assert "Table 2" in pc["table"] and pc["within"] == 3.0


def test_pros_wording_now_says_what_was_decided():
    m = mmp.manifest()
    assert "not_used" not in m and "can't use" not in json.dumps(m)
    assert "internal research evaluation only" in m["full_set"]
    assert "Pro stays the default" in m["full_set"]
    assert "a company can't use" not in mmp.__doc__ and "masein decided" in mmp.__doc__


def test_none_of_the_full_sets_files_is_in_the_repo():
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True,
                             check=True).stdout.decode().split("\0")
    pinned = {f["sha256"] for f in mmp.full_manifest()["files"]}
    sizes = {f["bytes"] for f in mmp.full_manifest()["files"]}
    # the files' header, built here so that this test's own text never holds it
    head = (",".join(["question_id", "Question", "A", "B", "C", "D"]) + "\n").encode()
    holders = []
    for name in filter(None, tracked):
        p = REPO / name
        if not p.is_file() or p.stat().st_size > 30_000_000:
            continue
        raw = p.read_bytes()
        if p.stat().st_size in sizes:
            assert hashlib.sha256(raw).hexdigest() not in pinned, name
        if raw.startswith(head):
            holders.append(name)
    assert holders and all(h.startswith("tests/fixtures/mmf_invented/test/") for h in holders)


def test_the_data_step_fetches_every_file_from_its_revision_and_refuses_another(full, tmp_path):
    items = [i for i in fetch_data.wanted() if i["name"].startswith("Mobile-MMLU (full) ")]
    assert len(items) == 12 and items[0]["dest"] == full / "test" / "cooking_and_recipes.csv"
    real = mmp.full_manifest()["files"]
    item = {**items[0], "dest": tmp_path / "out" / "test" / "cooking_and_recipes.csv"}
    raw = (FIXTURE / "test" / "cooking_and_recipes.csv").read_bytes()
    asked = []
    line = fetch_data.fetch(item, opener=lambda req, timeout: asked.append(req.full_url)
                            or _Resp(raw))
    assert line.startswith("Mobile-MMLU (full) cooking_and_recipes: fetched to ")
    assert asked == [real[0]["url"]]
    # another file — a byte changed — is refused, and nothing is kept
    bad = {**item, "dest": tmp_path / "bad" / "test" / "cooking_and_recipes.csv"}
    with pytest.raises(RuntimeError, match="not fetched — got"):
        fetch_data.fetch(bad, opener=lambda req, timeout: _Resp(raw.replace(b"fridge", b"Fridge")))
    assert not bad["dest"].exists() and not list(bad["dest"].parent.glob("*.part"))


def test_the_real_manifest_asks_the_data_step_for_all_80(monkeypatch):
    items = [i for i in fetch_data.wanted() if i["name"].startswith("Mobile-MMLU (full) ")]
    assert len(items) == 80
    assert {i["sha256"] for i in items} == {f["sha256"] for f in mmp.full_manifest()["files"]}
    assert all(i["dest"].parent.name == "test" for i in items)


def test_each_question_is_read_with_its_files_field(full):
    rows = mmp.load_full()
    assert len(rows) == 18 and mmp.full_available() == ""
    q = mmp.full_by_id()["inv00013"]
    assert (q["field"], q["category"]) == ("first_aid", "Health & Safety")
    assert set(q) == {"id", "question", "A", "B", "C", "D", "field", "category"}


def test_without_the_files_the_set_says_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv("MMF_DIR", str(tmp_path / "empty"))
    mmp._full_rows.clear()
    assert mmp.full_available() == ("Mobile-MMLU (full) isn't on this server: fetch it with "
                                    "the data step (scripts/fetch_data.py)")
    assert mmp.load_full() == []


def test_a_changed_file_is_refused(full):
    p = full / "test" / "pet_care.csv"
    p.write_bytes(p.read_bytes().replace(b"vet", b"Vet"))
    with pytest.raises(ValueError, match="pet_care.csv isn't the pinned file"):
        mmp.load_full()


def test_pro_sits_in_the_full_set_by_id_and_by_wording(full, tmp_path, monkeypatch):
    put_invented(monkeypatch, tmp_path / "data" / "mobile_mmlu_pro")
    o = mmp.overlap()
    assert (o["pro"], o["full"], o["pro_in_full"], o["pro_not_in_full"]) == (12, 18, 12, [])
    assert (o["same_wording"], o["worded_apart"], o["same_options"], o["full_only"]) == (
        11, ["inv00004"], 12, 6)
    out = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(out):
        assert mmp.main(["--overlap"]) == 0
    assert "worded apart inv00004" in out.getvalue() and "only in the full set 6" in out.getvalue()


def test_the_full_set_is_in_the_contamination_index_once_fetched(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MMF_DIR", tmp_path / "nothing")
    before = contamination.BenchmarkIndex(tmp_path / "ix1").refresh().n_pinned
    put_full(monkeypatch, tmp_path / "mmf")
    ix = contamination.BenchmarkIndex(tmp_path / "ix2").refresh()
    assert ix.n_pinned == before + 18
