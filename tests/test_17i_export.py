"""17i, part 3: the export, before anyone runs it (docs/prompts/phase-17i-
seventh-review-hle-reader-export-fetch.md, points 16 to 24). A test a point,
each failing on 511854e. Runs invented on the board; nothing is uploaded."""

from __future__ import annotations

import builtins
import json
import re
import urllib.parse

import export_frontier_raw as efr
import export_safe as es
from service import config, db
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import SERVED, box  # noqa: F401
from test_17g_review import LONG
from test_17h_review import _export_world

ACCT = "teamacct"


def log_of(sid: int):
    return config.LOGS_DIR / f"service_{sid}_{SERVED.replace('/', '__')}.log"


def exported_log(sid: int, tmp_path, *lines: str) -> str:
    log = log_of(sid)
    log.write_text(log.read_text() + "\n" + "\n".join(lines) + "\n")
    return (efr.export_run(sid, tmp_path / "raw") / "log.txt").read_text()


# ---------------------------------------------------------------------------
# 16: a line goes out only in a listed shape; none carries a repository
# ---------------------------------------------------------------------------

def test_16_a_runner_line_goes_out_only_in_a_listed_shape(box, tmp_path, monkeypatch):  # noqa: F811
    sid = _export_world(box, monkeypatch)
    fetching = (f"2026-10-07 09:00:01 fetching llama-server-cuda12.8.tar.gz from "
                f"{ACCT}/evalboard-private at main")
    text = exported_log(sid, tmp_path, fetching,
                        "2026-10-07 09:00:02 anything at all after a time",
                        "2026-10-07 09:00:03 llama-server is up (42 s)",
                        "2026-10-07 09:00:04 · GPQA Diamond 120 of 792 · 2.1 s an answer · "
                        "3.4 h left",
                        "2026-10-07 09:00:05 parity 412 of 1,000")
    assert "evalboard-private" not in text and ACCT not in text, text   # 511854e: through
    assert "anything at all" not in text
    for kept in ("llama-server is up (42 s)", "GPQA Diamond 120 of 792", "parity 412 of 1,000"):
        assert kept in text, kept
    # and no shape on the list holds a "/" but its slots
    for name, shape in es.LOG_SHAPES.items():
        rest = shape.replace(re.escape(es.BOX_PATH) + "/", "").replace(r"\d+/(?:\d+|\?)", "")
        assert "/" not in rest, name


# ---------------------------------------------------------------------------
# 17: the README through the same last step
# ---------------------------------------------------------------------------

def test_17_the_readme_is_scrubbed_as_everything_else(box, tmp_path, monkeypatch):  # noqa: F811
    sid = _export_world(box, monkeypatch)
    db.update(sid, where_ran="rented GPU · RTX 5090 · box root@203.0.113.7 ssh4.vast.ai")
    readme = (efr.export_run(sid, tmp_path / "raw") / "README.md").read_text()
    where = next(x for x in readme.splitlines() if x.startswith("Where it ran"))
    assert "203.0.113.7" not in where and "vast.ai" not in where, where  # 511854e: through


# ---------------------------------------------------------------------------
# 18: every log against every gated and private set
# ---------------------------------------------------------------------------

def test_18_a_log_is_checked_against_every_gated_set_whatever_the_run_asked(
        box, tmp_path, monkeypatch):  # noqa: F811
    sid = _export_world(box, monkeypatch)
    # a SimpleQA run's log, on a box that ran Humanity's Last Exam too
    db.update(sid, tasks=json.dumps(["simpleqa_epoch"]))
    q = LONG[2]["question"]
    text = exported_log(sid, tmp_path, f"[frontier] hle_text_cais: the server failed on {q}")
    assert "boiling point" not in text, text                    # 511854e: through


# ---------------------------------------------------------------------------
# 19: a run written to private/ removes its public/ twin
# ---------------------------------------------------------------------------

def test_19_a_run_made_private_removes_its_public_folder(box, tmp_path, monkeypatch, capsys):  # noqa: F811
    sid = _export_world(box, monkeypatch)
    monkeypatch.setenv("SCRUB_HOSTS", "board-host")
    monkeypatch.setenv("SCRUB_ACCOUNTS", ACCT)
    db.public_set(SERVED, True, "masein")
    monkeypatch.setattr(builtins, "input", lambda prompt="": "yes")
    out = tmp_path / "raw"
    assert efr.main(["--run", str(sid), "--out", str(out)]) == 0
    public = next((out / "public").iterdir())
    db.public_set(SERVED, False, "masein")
    capsys.readouterr()
    assert efr.main(["--run", str(sid), "--out", str(out)]) == 0
    said = capsys.readouterr().out
    assert not public.exists(), said                            # 511854e: left in public/
    assert f"{public} removed: this run is private now" in said
    assert (out / "private" / public.name).exists()


# ---------------------------------------------------------------------------
# 20: the mark decides, never the name; the list shows each file
# ---------------------------------------------------------------------------

def test_20_a_qwen36_file_is_markable_and_the_list_shows_its_file(box, tmp_path, monkeypatch):  # noqa: F811
    import export_devicemark_raw as dmx
    db.served_put({"id": "served/Qwen3.6-35B-A3B-BF16", "name": "BF16", "base_url": "",
                   "key": "", "how": "x", "based_on": "Qwen/Qwen3.6-35B-A3B",
                   "thinking": "auto", "pin": {}, "by": "masein", "at": 0, "file_sha256": {
        "sha256": "ab" * 32, "name": "Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf",
        "source": "hf://unsloth/Qwen3.6-35B-A3B-MTP-GGUF@main/BF16",
        "parts": [{"name": "a", "sha256": "1" * 64}, {"name": "b", "sha256": "2" * 64}]}})
    db.public_set("served/Qwen3.6-35B-A3B-BF16", True, "masein")
    assert efr.known_public({"hf_id": "served/Qwen3.6-35B-A3B-BF16"})   # 511854e: never
    row = tmp_path / "served__Qwen3.6-35B-A3B-BF16"
    row.mkdir()
    (row / "model_meta.json").write_text(json.dumps({"model": "served/Qwen3.6-35B-A3B-BF16"}))
    assert dmx.public_by_default(row, {"based_on": "Qwen/Qwen3.6-35B-A3B"})  # 511854e: never
    said = []
    assert es.confirm(["served/Qwen3.6-35B-A3B-BF16"], ask=lambda p: "no", say=said.append) \
        is False
    line = next(x for x in said if "Qwen3.6-35B-A3B-BF16" in x)
    assert "(2 parts)" in line and "sha256 " + "ab" * 32 in line, line
    assert "from hf://unsloth/Qwen3.6-35B-A3B-MTP-GGUF@main/BF16" in line


# ---------------------------------------------------------------------------
# 21: a value only for a variable on the list; public/ refused without the scrub's names
# ---------------------------------------------------------------------------

def test_21_variables_go_out_by_name_unless_their_value_is_listed():
    got, left = es.env("LLAMA_ARG_HOST=box7.internal LLAMA_ARG_PORT=8080 "
                       "LLAMA_ARG_ALIAS=teamacct-build LLAMA_ARG_HF_FILE=x.gguf "
                       "LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_ARG_CTX_SIZE=81920 "
                       "LLAMA_ARG_API_KEY=k")
    assert got["LLAMA_MOE_ROUTE_MODE"] == "lookahead" and got["LLAMA_ARG_CTX_SIZE"] == "81920"
    for name in ("LLAMA_ARG_HOST", "LLAMA_ARG_PORT", "LLAMA_ARG_ALIAS", "LLAMA_ARG_HF_FILE"):
        assert got[name] == es.WITHHELD, name                  # 511854e: their values
    assert "LLAMA_ARG_API_KEY" not in got and left == 1


def test_21_public_is_refused_while_the_scrub_lacks_its_names(box, tmp_path, monkeypatch,  # noqa: F811
                                                              capsys):
    sid = _export_world(box, monkeypatch)
    monkeypatch.setenv("SCRUB_HOSTS", "board-host")
    monkeypatch.delenv("SCRUB_ACCOUNTS", raising=False)
    db.public_set(SERVED, True, "masein")
    monkeypatch.setattr(builtins, "input", lambda prompt="": "yes")
    out = tmp_path / "raw"
    assert efr.main(["--run", str(sid), "--out", str(out)]) == 1      # 511854e: 0, public
    assert "public/ refused: SCRUB_ACCOUNTS isn't given" in capsys.readouterr().out
    assert not (out / "public").exists() and (out / "private").exists()


# ---------------------------------------------------------------------------
# 22: quotes that slipped the six-word rule
# ---------------------------------------------------------------------------

def test_22_escaped_underscored_and_url_encoded_quotes_are_caught():
    q = es.Questions(["Quelle est la température d'ébullition de l'eau à cette altitude précise?"])
    for line in ("[frontier] " + json.dumps("température d'ébullition de l'eau à cette "
                                            "altitude")[1:-1],
                 "[frontier] la_température_d_ébullition_de_l_eau_à_cette",
                 "[frontier] " + urllib.parse.quote("température d'ébullition de l'eau à cette "
                                                    "altitude")):
        assert q.quotes(line), line                             # 511854e: through


# ---------------------------------------------------------------------------
# 23: a switch never takes the next token
# ---------------------------------------------------------------------------

def test_23_a_flag_that_takes_no_value_carries_nothing_out():
    got, left = es.flags("--jinja hunter2 -c 8192 --cpu-moe --no-mmap tokenish --flash-attn on")
    assert got == ["--jinja", "-c", "8192", "--cpu-moe", "--no-mmap", "--flash-attn", "on"], got
    assert left == 2                                            # 511854e: --jinja hunter2


# ---------------------------------------------------------------------------
# 24: small ones
# ---------------------------------------------------------------------------

def test_24_a_numbered_header_goes_out_and_a_missing_set_is_said(box, tmp_path, monkeypatch,  # noqa: F811
                                                                 capsys):
    import frontier as fb
    sid = _export_world(box, monkeypatch)
    text = (efr.export_run(sid, tmp_path / "raw") / "log.txt").read_text()
    assert f"===== [{sid}] imported" in text, text              # 511854e: dropped as an address
    real = fb.load
    monkeypatch.setattr(fb, "load", lambda task, root=None: (_ for _ in ()).throw(
        RuntimeError("gated")) if task == "hle_text_cais" else real(task, root))
    capsys.readouterr()
    # 17j: and exits 1 — it exited 0, every run's log empty
    assert efr.main(["--run", str(sid), "--out", str(tmp_path / "raw2"), "--private"]) == 1
    assert "no log: Humanity's Last Exam couldn't be loaded" in capsys.readouterr().out


def test_24_public_exits_non_zero_and_closed_stdin_is_a_no(capsys):
    assert efr.main(["--all", "--public"]) == 2                 # 511854e: 0
    assert "--public does nothing" in capsys.readouterr().out

    def closed(prompt=""):
        raise ValueError("I/O operation on closed file.")
    assert es.confirm(["served/x"], ask=closed, say=lambda x: None) is False  # 511854e: raise
