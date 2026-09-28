"""12o.1: the question builder's duplicate check embeds on this server — a
small model on the CPU, in the image and pinned — so the hidden half never
leaves it; OpenRouter's embeddings stay a setting, which says it sends every
question out. The cosine for the local model is checked on this bank, on the
server, against the pairs OpenRouter's model flagged, and the builder uses it.

The local model never runs here: a word-count stand-in takes its place, as
the fake OpenRouter's does for OpenRouter's."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from fake_openrouter import embedding
from service import builder, config, db, dup_threshold, embed_local, llm
from test_builder_12i2 import _near_copy_writer, create, rest, review_all, svc  # noqa: F401

REPO = Path(__file__).resolve().parents[1]


class Local:
    """the local model's stand-in: what it was asked, and word-count vectors"""
    def __init__(self, monkeypatch):
        self.asked: list[str] = []
        monkeypatch.setattr(embed_local, "available", lambda: True)
        monkeypatch.setattr(embed_local, "embed", self)

    def __call__(self, texts):
        self.asked += texts
        return [embedding(t) for t in texts]


def no_openrouter(monkeypatch):
    """every call out of the machine, recorded; an embeddings one fails the test"""
    sent = []
    real = llm._http

    def http(method, url, headers, body=None, timeout=60):
        sent.append(url)
        assert "/embeddings" not in url, "a question was sent out to be embedded"
        return real(method, url, headers, body, timeout=timeout)
    monkeypatch.setattr(llm, "_http", http)
    return sent


def reworded():
    import everyday as ev
    bank = next(q for q in ev.load_bank() if 10 < len(q["prompt"].split()) < 30)
    w = bank["prompt"].split()
    return " ".join(w[:6] + ["uh"] + w[6:12] + ["pls"] + w[12:])


def test_by_default_the_check_embeds_here_and_nothing_goes_out(svc, monkeypatch):  # noqa: F811
    client, _, _ = svc
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-or-test")   # a key: still not sent
    local, sent = Local(monkeypatch), no_openrouter(monkeypatch)
    _near_copy_writer(monkeypatch, reworded())
    d = create(client, kind="everyday", group="quick_maths", count=20)
    d = rest(client, review_all(client, d))
    dup = next(f for it in d["items"] if it["n"] == 11 for f in it["flags"] if f["kind"] == "dup")
    assert dup["how"].startswith("cosine ")
    assert d["dedup_how"] == "13-gram and embeddings on this server"
    # the hidden half was embedded — here, and only here
    import everyday as ev
    hidden = {q["prompt"] for q in ev.load_bank() if ev.half(q) == ev.HIDDEN}
    assert hidden and hidden <= set(local.asked)
    assert not [u for u in sent if "/embeddings" in u]
    # cached under the local model's own name: the bank is embedded once
    cache = json.loads((config.BENCH_ROOT / "builder" / "embeddings.json").read_text())
    assert builder._sha(embed_local.IDENT + "\0" + next(iter(hidden))) in cache
    n = len(local.asked)
    create(client, kind="everyday", group="quick_maths", count=20)
    assert len(local.asked) - n <= 20
    page = client.get("/api/builder").json()
    assert page["dedup_how"] == "13-gram and embeddings on this server"
    assert page["dedup_warning"] == ""


def test_without_the_model_in_the_image_it_says_so_and_checks_13_grams(svc, monkeypatch):  # noqa: F811
    client, _, _ = svc
    monkeypatch.setattr(config, "QB_EMBED_DIR", config.BENCH_ROOT / "no-model-here")
    assert not embed_local.available() and embed_local.embed(["x"]) is None
    assert client.get("/api/builder").json()["dedup_how"] == (
        "13-gram — the embedding model isn't in this image: build it again")


def test_openrouter_is_a_setting_that_says_it_sends_the_hidden_half_out(svc, monkeypatch):  # noqa: F811
    client, _, _ = svc
    monkeypatch.setattr(config, "QB_EMBED_MODEL", "openrouter")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-or-test")
    page = client.get("/api/builder").json()
    assert page["dedup_how"] == "13-gram and embeddings through OpenRouter"
    assert page["dedup_warning"] == builder.REMOTE_WARNING
    assert "the hidden half included" in page["dedup_warning"]
    assert builder.dup_cosine() == config.QB_DUP_COSINE


def test_the_local_cosine_is_the_one_checked_on_this_bank(svc, monkeypatch):  # noqa: F811
    assert builder.dup_cosine() == config.QB_DUP_COSINE_LOCAL_DEFAULT
    f = config.BENCH_ROOT / "builder" / "dup_threshold.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"model": embed_local.IDENT, "cosine": 0.87}))
    assert builder.dup_cosine() == 0.87
    # checked with another model's weights: not this one's
    f.write_text(json.dumps({"model": "BAAI/bge-small-en-v1.5@000000000000", "cosine": 0.5}))
    assert builder.dup_cosine() == config.QB_DUP_COSINE_LOCAL_DEFAULT
    monkeypatch.setattr(config, "QB_DUP_COSINE_LOCAL", "0.91")
    assert builder.dup_cosine() == 0.91


@pytest.mark.parametrize("pairs, want", [
    # the remote model's two flags sit at 0.93 and 0.95 locally; the others below 0.9
    ([(0.95, 0.95), (0.91, 0.93), (0.6, 0.89), (0.5, 0.7)],
     {"cosine": 0.93, "flagged": 2, "caught": 2, "extra": 0}),
    # one flagged pair the local model scores low: missing it beats flagging four more
    ([(0.95, 0.96), (0.92, 0.80)] + [(0.7, 0.85)] * 4,
     {"cosine": 0.96, "flagged": 2, "caught": 1, "extra": 0}),
    # nothing flagged: nothing to check against
    ([(0.5, 0.9), (0.6, 0.95)], {"cosine": None, "flagged": 0, "caught": 0, "extra": 0}),
])
def test_the_cosine_chosen_agrees_best_with_the_pairs_it_flagged(pairs, want):
    assert dup_threshold.choose(pairs, 0.9) == want


def test_the_check_on_the_server_reads_the_cache_and_prints_no_question(
        svc, monkeypatch, capsys):  # noqa: F811
    client, _, _ = svc
    # a batch embedded through OpenRouter before 12o.1: its vectors are cached
    monkeypatch.setattr(config, "QB_EMBED_MODEL", "openrouter")
    from fake_openrouter import FakeOpenRouter
    FakeOpenRouter.install(monkeypatch)
    _near_copy_writer(monkeypatch, reworded())
    rest(client, review_all(client, create(client, kind="everyday", group="quick_maths",
                                           count=20)))
    monkeypatch.setattr(config, "QB_EMBED_MODEL", "local")
    Local(monkeypatch)
    dup_threshold.main()
    out = capsys.readouterr().out
    assert "catches" in out, out
    got = json.loads((config.BENCH_ROOT / "builder" / "dup_threshold.json").read_text())
    assert got["model"] == embed_local.IDENT and got["flagged"] >= 1
    assert got["caught"] == got["flagged"] and builder.dup_cosine() == got["cosine"]
    assert f"catches {got['caught']} of them" in out
    import everyday as ev
    assert not [q for q in ev.load_bank() if q["prompt"][:40] in out]
    assert db.qb_list()


def test_the_image_holds_the_model_pinned_as_the_service_expects():
    docker = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert f"ARG BGE_REVISION={embed_local.REVISION}" in docker
    assert f"ARG BGE_SHA256={embed_local.WEIGHTS_SHA256}" in docker
    assert "sha256sum -c" in docker
    assert f"local_dir='{config.QB_EMBED_DIR}'" in docker
    patterns = re.search(r"allow_patterns=\[([^\]]+)\]", docker.replace("\\\n", "")).group(1)
    assert sorted(re.findall(r"'([^']+)'", patterns)) == sorted(embed_local.FILES)
