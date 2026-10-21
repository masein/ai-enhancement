"""24b: each build's quant and size, as its files say — the box reads the
GGUF's own general.file_type into its bundle, the import keeps it with the
file's bytes on the served row, and the reports read it (or a local GGUF's
header); what isn't known is left out. Invented files; no model runs."""

from __future__ import annotations

import hashlib

import import_remote as ir
import remote_bundle as rb
from service import db, reports
from test_16_1_sizes import gguf_bytes
from test_17_gguf_box import SERVED, box, bundle_of, register, run_box  # noqa: F401
from test_12q_devicemark_runs import svc  # noqa: F401 — box's fixture


def put(mid: str, **more) -> None:
    db.served_put({"id": mid, "name": mid.split("/")[1], "base_url": "", "key": "", "how": "x",
                   "based_on": "", "thinking": "on", "pin": {}, "by": "masein", "at": 0, **more})


def test_what_the_files_say_and_nothing_more(svc, tmp_path):  # noqa: F811
    put("served/both", gguf_file={"size": 15_800_000_000, "file_type": "Q4_K_M"})
    put("served/size", pin={"size": 15_800_000_000})
    put("served/none")
    put("served/parts", file_sha256={"sha256": "x", "parts": [{"name": "a", "size": 40_000_000_000},
                                                             {"name": "b", "size": 30_000_000_000}]},
        pin={"size": 1})
    local = tmp_path / "here.gguf"
    local.write_bytes(gguf_bytes())                              # general.file_type 15: Q4_K_M
    put("served/local", gguf_path=str(local), pin={"size": 22_000_000_000})
    assert reports.build_file("served/both · thinking") == {"quant": "Q4_K_M",
                                                           "size": 15_800_000_000}
    assert reports.build_file("served/size") == {"size": 15_800_000_000}
    assert reports.build_file("served/none") == {}
    assert reports.build_file("served/parts")["size"] == 70_000_000_000     # the parts summed
    assert reports.build_file("served/local") == {"quant": "Q4_K_M", "size": 22_000_000_000}
    assert reports.build_file("served/nobody") == {}


def test_the_box_reads_the_quant_and_the_import_keeps_it(box, monkeypatch):  # noqa: F811
    data = gguf_bytes() + bytes(4096)                             # a header, and no weights
    box["gguf"].write_bytes(data)
    box["sha"] = hashlib.sha256(data).hexdigest()
    assert run_box(box, "run") == 0
    gg = rb.read(bundle_of(box, "run"))["setup"]["gguf"]
    assert gg["file_type"] == "Q4_K_M" and gg["size"] == len(data)
    register(box["sha"])
    assert ir.import_bundle(bundle_of(box, "run"), "masein", lambda s: None) == 0
    rec = db.served_get(SERVED)
    assert rec["gguf_file"] == {"size": len(data), "file_type": "Q4_K_M"}
    assert reports.build_file(SERVED + " · thinking") == {"quant": "Q4_K_M", "size": len(data)}
