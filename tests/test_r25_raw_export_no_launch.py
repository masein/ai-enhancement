"""r25 (masein and Roohi, 14 Nov): runtime details stay internal. The raw-run exports write no
server launch: not its flags, not its environment, not the speculative-decoding flags' names
(a Frontier task keeps only whether its slots drafted tokens), and a log's line naming the
runtime's variables is left out. #311's scan, run on what they write, finds no runtime switch,
no server flag and no VAR=value. The board's own views are not changed here.

The board's runner with the stand-in lm_eval (15.1); nothing runs."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import devicemark as dm
import export_devicemark_raw as ex
import export_frontier_raw as efr
import export_safe as es
from service import config
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the board's fixture
from test_15_2_import import board, row_of  # noqa: F401 — board is the fixture
from test_15_4_raw_export import QWEN, no_gated_sets, row  # noqa: F401 — fixtures

FLAGS = "-ngl 99 --n-cpu-moe 21 --no-mmap -c 16384 --parallel 1 --jinja"
ENV = "LOOKAHEAD_MOE_ROUTER_NORM=1 GGML_CPU_DISABLE_FUSION=1"
SHELL_LINE = ("2026-10-29 09:00:01 not given to llama-server: GGML_CPU_DISABLE_FUSION, "
              "LLAMA_MOE_ROUTE_MODE, set in this shell — its variables come from --env only, so "
              "each one it gets is recorded")


def scan_terms():
    here = Path(__file__).resolve().parent.parent / "scripts" / "report_data" / "scan_terms.py"
    spec = importlib.util.spec_from_file_location("scan_terms", here)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hits(folder: Path) -> dict:
    """#311's three terms for a launch, by file — the server argv's bare word "llama-server"
    aside: the runner's own "llama-server is up" lines name the program, not a flag"""
    terms = {k: re.compile(scan_terms().TERMS[k])
             for k in ("runtime switches", "server argv", "launch env")}
    out = {}
    for f in sorted(p for p in folder.rglob("*") if p.is_file()):
        text = f.read_text(encoding="utf-8", errors="replace")
        for k, pat in terms.items():
            got = [m.group(0) for m in pat.finditer(text) if m.group(0) != "llama-server"]
            if got:
                out[(f.name, k)] = got
    return out


def test_a_devicemark_setup_goes_out_with_no_launch():
    got = ex.setup_out({"model": "a/b", "cap": 4096, "build": "b9954-a4a76a751",
                        "launch": {"flags": FLAGS, "env": ENV}, "how": "the lda build",
                        "flags": FLAGS, "env": ENV})
    assert got == {"model": "a/b", "cap": 4096, "build": "b9954-a4a76a751"}


def test_a_frontier_task_keeps_only_whether_its_slots_drafted():
    got = efr.task_setup({"task": "gpqa_diamond_epoch", "budget": 32768,
                          "launch": {"flags": FLAGS, "env": ENV},
                          "launch_setup": {"env": {"LOOKAHEAD_MOE_ROUTER_NORM": "1"},
                                           "answers": ["-ctk", "q8_0"],
                                           "spec": ["--spec-type", "--draft-max"], "drafts": False}})
    assert got == {"task": "gpqa_diamond_epoch", "budget": 32768,
                   "launch_setup": {"drafts": False}}


def test_a_log_line_naming_the_runtimes_variables_is_left_out():
    lines, left = es.log_lines(SHELL_LINE + "\n2026-10-29 09:00:30 llama-server is up (29 s)\n",
                               es.Questions([]))
    assert lines == ["2026-10-29 09:00:30 llama-server is up (29 s)"]
    assert left == {"not in a shape on the list": 1}


def test_a_whole_devicemark_export_carries_no_launch_and_the_scan_finds_none(row, tmp_path):  # noqa: F811
    # the row's setup and log as a served run's can be: its launch, and the line naming
    # the shell's runtime variables
    f = row_of(QWEN) / dm.OUT_NAME
    data = json.loads(f.read_text())
    data["setup"].update(launch={"flags": FLAGS, "env": ENV, "build": "b9954-a4a76a751"})
    f.write_text(json.dumps(data))
    log = config.LOGS_DIR / f"service_{row['id']}_{QWEN.replace('/', '__')}.log"
    log.write_text(log.read_text() + "\n" + SHELL_LINE + "\n")
    # first, the scan sees the launch where it is: the row's own record
    assert hits(row_of(QWEN))
    dest = ex.export_row(row_of(QWEN), tmp_path / "out", say=lambda *_: None)
    assert "launch" not in json.loads((dest / "setup.json").read_text())
    assert hits(dest) == {}
    assert "the server's launch left out" in (dest / "README.md").read_text()
