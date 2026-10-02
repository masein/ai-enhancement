"""15.4: each DeviceMark row's raw per-item run, ready to publish — as
DeviceMark links every row of its board to its raw file. The folder's files,
in their field names where ours mean the same; a recompute script that agrees
with scores.json; the scrub (no key, home path, host name or private address
in any file); DeviceMark rows only; public or private by the model; and the
row's "raw" link, set by the command or the form. Rows made by the board's
runner with 15.1's stand-in lm_eval; nothing runs."""

from __future__ import annotations

import json
import re
import subprocess
import sys

import pytest

import devicemark as dm
import export_devicemark_raw as ex
from service import config, db
from test_12q_devicemark_runs import ME, svc  # noqa: F401 — svc is board's fixture
from test_15_2_import import board, local, row_of  # noqa: F401 — board is the fixture

QWEN = "Qwen/Qwen3.5-4B"
HOST = "teraformer-box"
LEAKS = re.compile(r"(?i)/home/|\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|teraformer-box|"
                   r"\bhf_[A-Za-z0-9]{20,}|\bsk-or-v1-|\bsk-[A-Za-z0-9_\-]{16,}|"
                   r"bearer\s+[A-Za-z0-9._\-]{12,}|\.ts\.net\b")


@pytest.fixture
def row(board, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ex.socket, "gethostname", lambda: HOST)
    monkeypatch.setenv("HF_TOKEN", "hf_FIXTUREsecret0123456789abcdef")
    sub = local(QWEN)
    assert sub["status"] == "done", sub["error"]
    # what a server's log and setup can hold: its paths, addresses, host and keys
    log = config.LOGS_DIR / f"service_{sub['id']}_{QWEN.replace('/', '__')}.log"
    log.write_text(log.read_text() + "\n[service] cache at /home/masein/benchmarks/results/full\n"
                   "judge http://100.74.89.105:8899/v1 and http://gemma-vllm:8000/v1 on "
                   f"{HOST} ({HOST}.tail1234.ts.net)\nAuthorization: Bearer abcdef0123456789xyz\n"
                   "HF_TOKEN=hf_FIXTUREsecret0123456789abcdef api_key=sk-or-v1-0123456789abcdef01\n")
    f = row_of(QWEN) / dm.OUT_NAME
    data = json.loads(f.read_text())
    data["setup"].update(server_flags="--model /home/masein/models/q.gguf --host 100.74.89.105")
    f.write_text(json.dumps(data))
    return sub


def test_a_row_exports_every_item_its_numbers_and_its_log(row, tmp_path):
    dest = ex.export_row(row_of(QWEN), tmp_path / "out", say=lambda *_: None)
    assert dest == tmp_path / "out" / "public" / row_of(QWEN).name
    assert sorted(p.name for p in dest.iterdir()) == ["README.md", "items.jsonl", "log.txt",
                                                       "recompute.py", "scores.json",
                                                       "setup.json"]
    items = [json.loads(x) for x in (dest / "items.jsonl").read_text().splitlines()]
    assert len(items) == 596 and [i["test"] for i in items[:1]] == ["ifeval"]
    # DeviceMark's raw names, where ours mean the same
    for k in ("key", "answer", "capped", "generated_tokens", "prompt_chars", "content_chars",
              "thinking_chars"):
        assert k in items[0], k
    ife, mmlu = items[0], next(i for i in items if i["test"] == "mmlu_pro")
    src = dm.load_items(config.DM_ITEMS)
    assert ife["prompt"] == src[("ifeval", ife["key"])]["text"]          # the prompt as sent
    assert ife["rules"] and set(ife["rules"][0]) == {"id", "strict", "loose"}
    assert ife["ifeval"]["strict"]["prompt"] in (True, False)
    assert {"parsed", "parsed_how", "gold", "correct"} <= set(mmlu)
    scores = json.loads((dest / "scores.json").read_text())
    want = json.loads((row_of(QWEN) / dm.OUT_NAME).read_text())
    assert scores["composite"] == want["composite"]
    assert scores["tests"]["math"]["capped_count"] == 0 and scores["tests"]["math"]["max_tokens"] == 4096
    setup = json.loads((dest / "setup.json").read_text())
    assert setup["battery"] == dm.VERSION and setup["runs"] == [row["id"]]
    assert setup["battery_hashes"] == dm.battery_hashes(src)
    assert f"run #{row['id']}" in (dest / "log.txt").read_text()
    readme = (dest / "README.md").read_text()
    assert "Apache-2.0" in readme and readme.count("MIT") >= 2 and "python recompute.py" in readme


def test_the_recompute_script_agrees_with_scores_json(row, tmp_path):
    dest = ex.export_row(row_of(QWEN), tmp_path / "out", say=lambda *_: None)
    r = subprocess.run([sys.executable, "recompute.py"], cwd=dest, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip().endswith("agrees with scores.json"), r.stdout
    got = json.loads(r.stdout[:r.stdout.rindex("}") + 1])
    want = json.loads((dest / "scores.json").read_text())
    assert got["composite"] == want["composite"]
    for t in dm.BENCHES:
        assert got["tests"][t] == {k: want["tests"][t][k] for k in ("n", "acc", "ci")}
    # and a changed verdict is caught
    lines = (dest / "items.jsonl").read_text().splitlines()
    i = next(n for n, x in enumerate(lines) if json.loads(x)["test"] == "math")
    it = json.loads(lines[i])
    it["correct"] = not it["correct"]
    lines[i] = json.dumps(it)
    (dest / "items.jsonl").write_text("\n".join(lines) + "\n")
    r = subprocess.run([sys.executable, "recompute.py"], cwd=dest, capture_output=True, text=True)
    assert r.returncode == 1 and "differs from scores.json: composite, math" in r.stdout


def test_no_key_home_path_host_or_private_address_is_exported(row, tmp_path):
    dest = ex.export_row(row_of(QWEN), tmp_path / "out", say=lambda *_: None)
    for f in dest.iterdir():
        text = f.read_text()
        m = LEAKS.search(text)
        assert not m, f"{f.name}: {text[max(0, m.start() - 40):m.end() + 40]!r}"
    log = (dest / "log.txt").read_text()
    assert "cache at ~/benchmarks/results/full" in log
    assert "http://[address]:8899/v1" in log and "http://[host]:8000/v1" in log
    assert "Authorization: Bearer [key withheld]" in log
    assert "--host [address]" in (dest / "setup.json").read_text()


def test_only_devicemark_rows_leave_the_server(board, tmp_path):  # noqa: F811
    for suite in ("everyday", "judged", "mobile", "safety"):
        sid = db.add(QWEN, "instruct", suite, ME, "")
        with pytest.raises(SystemExit, match="only DeviceMark rows are exported: Everyday, the "
                                             "Knowledge exam, the reasoning lab, Privacy Leakage "
                                             "and Mobile-MMLU-Pro never leave the server"):
            ex.main(["--run", str(sid), "--out", str(tmp_path / "export")])
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("model,setup,public", [
    ("Qwen/Qwen3.5-4B", {"runtime": "hf transformers (lm_eval)"}, True),
    ("google/gemma-4-E2B-it", {"runtime": "hf transformers (lm_eval)"}, True),
    ("teraformer/Qwen3.6-35B-A3B-k4-LDA", {"runtime": "hf transformers (lm_eval)"}, False),
    ("served/qwen36-phone-mtp", {"runtime": "llama-server", "based_on": "Qwen/Qwen3.6-35B-A3B"},
     False)])
def test_public_for_public_models_and_private_for_in_house_builds(tmp_path, model, setup, public):
    row = tmp_path / model.replace("/", "__")
    row.mkdir()
    (row / "model_meta.json").write_text(json.dumps({"model": model}))
    assert ex.public_by_default(row, setup) is public


def test_the_raw_link_by_the_command_and_on_the_row(row, tmp_path):
    url = "https://huggingface.co/datasets/masein/evalboard-devicemark-raw/tree/main/x"
    assert ex.main(["--link", str(row["id"]), url, "--by", "masein"]) == 0
    assert json.loads((row_of(QWEN) / dm.RAW_NAME).read_text())["url"] == url
    rows = {r["id"]: r for r in dm.rows(config.OUT_DIR)}
    assert rows[f"{QWEN} · thinking"]["raw_url"] == url
    assert dm.model_runs(config.OUT_DIR)[QWEN]["on"]["row"]["raw_url"] == url
    assert ex.main(["--link", str(row["id"]), "--unlink", "--by", "masein"]) == 0
    assert not (row_of(QWEN) / dm.RAW_NAME).exists()
    with pytest.raises(ValueError, match="an https URL"):
        ex.set_link(row_of(QWEN), "file:///etc/passwd", "masein")


def test_the_form_sets_the_raw_link(board, row, tmp_path):  # noqa: F811
    client = board
    url = "https://huggingface.co/datasets/masein/evalboard-devicemark-raw/tree/main/y"
    r = client.put("/api/devicemark/raw", json={"model": QWEN, "thinking": True, "url": url,
                                                  "by": "masein"})
    assert r.status_code == 200 and r.json()["raw"]["url"] == url
    rows = {x["id"]: x for x in client.get("/api/devicemark").json()["rows"]}
    assert rows[f"{QWEN} · thinking"]["raw_url"] == url
    assert client.put("/api/devicemark/raw", json={"model": QWEN, "thinking": True,
                                                   "url": "http://x"}).status_code == 422
    assert client.put("/api/devicemark/raw", json={"model": QWEN, "thinking": False,
                                                   "url": url, "by": "m"}).status_code == 404
    r = client.put("/api/devicemark/raw", json={"model": QWEN, "thinking": True, "url": ""})
    assert r.status_code == 200 and r.json()["raw"] is None
