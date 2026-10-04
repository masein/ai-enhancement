"""16.1: two sizes for a model — total, and active where they differ ("35B · 3B
active") — and where each comes from, in order: what a person entered (a Size
field when a served model or a GGUF file is registered, and an edit on its
page), the harness's count, the GGUF file's own header (read by the host
worker), the board model it is based on, and — for a Hub model alone — its
name. A served model's or a GGUF file's name is never read: it only suggests,
in the form. A 35B size on a served or GGUF row never makes the board think
it needs 70 GB of our GPU. No model runs; a fake server, a fake binary."""

from __future__ import annotations

import json
import struct

import pytest

import gguf_header as gh
import report_lm_eval as rep
from conftest import make_service
from fake_openai import FakeServer
from service import chat, config, db, playground, sizes

ME = "masein"
QWEN = "Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL"


# ---------------------------------------------------------------------------
# a name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,total,active", [
    (QWEN, 35e9, 3e9),                        # A3B is its active size, never its total
    ("EleutherAI/pythia-410m", 410e6, None), ("Qwen/Qwen3-0.6B", 6e8, None),
    ("Qwen3-30B-A3B", 30e9, 3e9), ("google/gemma-2-2b-it", 2e9, None),
    ("unsloth/x-Q4_K_XL", None, None), ("model-k4", None, None), ("qwen-32k", None, None)])
def test_a_name_gives_its_total_and_its_active_size(name, total, active):
    assert rep.sizes_from_name(name) == (total, active)
    assert rep.params_from_name(name) == total


# ---------------------------------------------------------------------------
# the order
# ---------------------------------------------------------------------------

def test_the_order_entered_config_file_base_and_a_hub_name_last():
    served = {"served/lda": {"based_on": "org/base-35b", "gguf_path": "/m/lda.gguf"},
              "served/plain": {"based_on": "org/base-35b"},
              "served/nobase": {"based_on": "Qwen/Qwen3.6-35B-A3B"}}
    by_model = {"org/base-35b": {"num_params": 35e9, "archinfo": {"active_params": 3e9}}}
    files = {"/m/lda.gguf": {"header": {"params": 34.7e9, "active_params": 3.3e9}}}
    entered = {"served/lda": {"total": 35e9, "active": 3e9, "by": ME}}
    s = lambda mid, r=None, e=entered: rep.size_of(mid, r or {}, served, e, files, by_model)  # noqa: E731
    assert s("served/lda") == {"total": 35e9, "active": 3e9, "src": "entered", "by": ME}
    assert s("served/lda", e={})["src"] == "file"                       # its file's header
    got = s("served/plain")
    assert (got["total"], got["active"], got["src"], got["base"]) == (35e9, 3e9, "base",
                                                                      "org/base-35b")
    # its base isn't on the board: no size — its name is never read
    assert s("served/nobase")["total"] is None
    assert s("gguf/Qwen3.6-35B-A3B-k4", e={})["total"] is None
    # a Hub model: the harness's count, else its name
    assert s("org/m-1b", {"num_params": 1.2e9})["src"] == "config"
    assert s("EleutherAI/pythia-410m")["src"] == "name"


# ---------------------------------------------------------------------------
# entered: in the forms, on the page, and by the deploy step's command
# ---------------------------------------------------------------------------

@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    yield client, appmod
    client.__exit__(None, None, None)


@pytest.fixture
def fake():
    s = FakeServer()
    yield s
    s.close()


def row(client, appmod, mid: str) -> dict:
    appmod._cache.update(key=None, payload=None, at=0.0)
    return next(m for m in client.get("/api/results").json()["models"] if m["id"] == mid)


def test_a_served_model_takes_the_size_its_form_confirms(svc, fake):
    client, appmod = svc
    r = client.post("/api/served", json={"name": "LDA phone build", "base_url": fake.base,
                                         "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "k4",
                                         "thinking": "off", "size": "35B", "active": "3B",
                                         "by": ME})
    assert r.status_code == 200, r.text
    mid = r.json()["model"]["id"]
    m = row(client, appmod, mid)
    assert (m["params"], m["activeParams"], m["paramsSrc"], m["paramsBy"]) == (35e9, 3e9,
                                                                               "entered", ME)
    # without one: none — its name and its base's name only suggest, in the form
    r = client.post("/api/served", json={"name": "Qwen3.6-35B-A3B other", "base_url": fake.base,
                                         "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "k4",
                                         "thinking": "off", "by": ME})
    assert row(client, appmod, r.json()["model"]["id"])["params"] is None


@pytest.mark.parametrize("total,active,why", [
    ("abc", "", "isn't a size"), ("3B", "35B", "can't be more than its total"),
    ("", "3B", "A size")])
def test_a_size_that_isnt_one_is_refused_and_nothing_is_kept(svc, fake, total, active, why):
    client, _ = svc
    r = client.post("/api/served", json={"name": "Bad size", "base_url": fake.base,
                                         "how": "k4", "thinking": "off", "size": total,
                                         "active": active, "by": ME})
    if total:
        assert r.status_code == 422 and why in r.json()["detail"]
        assert not db.served_all()
    r = client.post("/api/models/size", json={"model": "fx/good-750m", "total": total,
                                              "active": active, "by": ME})
    assert r.status_code == 422 and why in r.json()["detail"]


def test_a_gguf_file_takes_the_size_its_form_confirms(svc, tmp_path):
    client, appmod = svc
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF" + b"\1" * 400)
    r = client.post("/api/gguf/models", json={"name": QWEN, "path": str(f), "how": "k4",
                                              "size": "35B", "active": "3B", "by": ME})
    assert r.status_code == 200, r.text
    m = row(client, appmod, r.json()["model"]["id"])
    # 16.1's bug: a GGUF of a 35B model was 3.0B, its name's last size-looking token
    assert (m["params"], m["activeParams"], m["paramsSrc"]) == (35e9, 3e9, "entered")


def test_a_size_is_edited_on_the_models_page(svc):
    client, appmod = svc
    r = client.post("/api/models/size", json={"model": "fx/good-750m", "total": "0.8B",
                                              "by": ME})
    assert r.status_code == 200, r.text
    m = row(client, appmod, "fx/good-750m")
    assert (m["params"], m["paramsSrc"], m["paramsBy"]) == (8e8, "entered", ME)
    r = client.post("/api/models/size", json={"model": "nobody/here", "total": "1B", "by": ME})
    assert r.status_code == 404


def test_the_deploy_steps_command_sets_every_model_based_on_one(svc, fake, capsys):
    client, appmod = svc
    ids = [client.post("/api/served", json={
        "name": n, "base_url": fake.base, "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "k4",
        "thinking": "off", "by": ME}).json()["model"]["id"] for n in ("one", "two")]
    assert sizes.main(["set", "--based-on", "Qwen/Qwen3.6-35B-A3B", "--total", "35B",
                       "--active", "3B", "--by", ME, "--dry-run"]) == 0
    assert "2 model(s) would be set" in capsys.readouterr().out and not db.sizes_all()
    assert sizes.main(["set", "--based-on", "Qwen/Qwen3.6-35B-A3B", "--total", "35B",
                       "--active", "3B", "--by", ME]) == 0
    for mid in ids:
        m = row(client, appmod, mid)
        assert (m["params"], m["activeParams"], m["paramsBy"]) == (35e9, 3e9, ME)
    assert sizes.main(["list"]) == 0
    assert "35B · 3B active (entered by masein)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the GGUF file's header
# ---------------------------------------------------------------------------

def gguf_bytes(experts: int = 8, used: int = 2) -> bytes:
    """a GGUF header with a mixture of experts' tensors: no weights"""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    def kv(k, t, v):
        return s(k) + struct.pack("<I", t) + v
    kvs = [kv("general.architecture", 8, s("qwen3moe")), kv("general.name", 8, s("Made up")),
           kv("qwen3moe.expert_count", 4, struct.pack("<I", experts)),
           kv("qwen3moe.expert_used_count", 4, struct.pack("<I", used)),
           kv("qwen3moe.context_length", 4, struct.pack("<I", 32768)),
           kv("general.file_type", 4, struct.pack("<I", 15)),
           kv("tokenizer.ggml.tokens", 9, struct.pack("<I", 8) + struct.pack("<Q", 2) + s("a")
              + s("b")),
           kv("tokenizer.chat_template", 8, s("{{ messages }}"))]

    def tensor(name, dims):
        return (s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
                + struct.pack("<I", 12) + struct.pack("<Q", 0))
    ts = [tensor("token_embd.weight", [1000, 100]), tensor("blk.0.attn_q.weight", [100, 100]),
          tensor("blk.0.ffn_gate_exps.weight", [100, 200, experts]),
          tensor("blk.0.ffn_gate_shexp.weight", [100, 50])]
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(ts)) + struct.pack("<Q", len(kvs))
            + b"".join(kvs) + b"".join(ts))


def test_a_gguf_header_says_its_total_and_active_size(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(gguf_bytes())
    h = gh.read(f)
    # 100,000 + 10,000 + 160,000 experts + 5,000 shared; 2 of 8 experts a token
    assert (h["params"], h["active_params"]) == (275_000, 115_000 + 160_000 * 2 // 8)
    assert (h["arch"], h["file_type"], h["context_length"], h["chat_template"]) == (
        "qwen3moe", "Q4_K_M", 32768, True)
    for bad in (b"GGUF" + b"\1" * 400, b"NOTGGUF", gguf_bytes()[:90], b""):
        f.write_bytes(bad)
        assert gh.read(f) is None


def test_the_worker_reads_the_header_and_the_board_takes_its_size(svc, tmp_path, monkeypatch):
    import gguf_data as gd
    import gguf_worker as gw
    from test_gguf_12f3 import docs_of, fake_binary
    client, appmod = svc
    gd.build(config.RESULTS_ROOT / "gguf_data", docs_of=docs_of())
    f = tmp_path / "moe.gguf"
    f.write_bytes(gguf_bytes())
    gid = client.post("/api/gguf/models", json={"name": "a moe build", "path": str(f),
                                                "how": "k4", "by": ME}).json()["model"]["id"]
    assert row(client, appmod, gid)["params"] is None          # not read yet: no size
    monkeypatch.setattr(gw, "_stop", {"why": ""})
    client.post("/api/gguf/runs", json={"model": gid, "benchmarks": ["mmlu"], "by": ME})
    assert gw.Worker(config.RESULTS_ROOT, fake_binary(tmp_path), poll=0.05).once() is True
    files = json.loads((config.RESULTS_ROOT / "gguf_files.json").read_text())
    assert files[str(f)]["header"]["params"] == 275_000
    m = row(client, appmod, gid)
    assert (m["params"], m["activeParams"], m["paramsSrc"]) == (275_000, 155_000, "file")


# ---------------------------------------------------------------------------
# never 70 GB of our GPU
# ---------------------------------------------------------------------------

def test_a_35b_served_or_gguf_row_asks_nothing_of_our_gpu(svc, fake, tmp_path, monkeypatch):
    client, _ = svc
    sid = client.post("/api/served", json={"name": "big served", "base_url": fake.base,
                                           "how": "k4", "thinking": "off", "size": "35B",
                                           "active": "3B", "by": ME}).json()["model"]["id"]
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF" + b"\1" * 400)
    gid = client.post("/api/gguf/models", json={"name": "big gguf", "path": str(f), "how": "k4",
                                                "size": "35B", "by": ME}).json()["model"]["id"]
    monkeypatch.setattr(chat, "gpu_free_bytes", lambda: 4 * 10 ** 9)
    srow = chat.model_row(sid)
    assert chat.Engine().place(srow) == ("served", "")          # its server's memory, not ours
    g = chat.model_row(gid)
    assert g["chat"] is False and g["why_not"] == "gguf" and g["params"] is None
    page = playground.models()
    assert gid not in {m["id"] for m in page["models"]}
    assert playground.GGUF_LINE in page["left_out"]
