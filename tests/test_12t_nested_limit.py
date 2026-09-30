"""12t: a model whose config keeps its limit under text_config is told to
lm_eval on every hf run.

lm_eval looks for a model's limit at the top of its config
(n_positions, max_position_embeddings, n_ctx) and takes 2,048 when it finds
none. Gemma 4 keeps max_position_embeddings under text_config. 12q.G told a
DeviceMark run its length; every other run of such a model on hf still got
2,048: the generative three would fail at "requested max tokens to generate
… must be less than model's maximum sequence length (2048)", and a
multiple-choice prompt longer than that was cut from the left, silently.

- Such a model's runs on hf are passed `max_length=<its real limit>`.
- A model whose limit lm_eval finds is told nothing: its commands are what
  they were, to the byte.
- vLLM and served runs are as they were, and a DeviceMark run keeps its own
  length (the prompt's room and the cap).
- scripts/asked_length.py lists the results on disk that were asked at 2,048
  by a model that reads more, from what lm_eval wrote into each results file.

No model runs: lm_eval is a stand-in."""

from __future__ import annotations

import json
from pathlib import Path

import asked_length
import devicemark as dm
import simpleqa as sq
from service import config, db, hfmeta, runner
from test_12n2 import sit_gpqa, sit_simpleqa, svc, twenty  # noqa: F401 — svc is the fixture
from test_12q_devicemark_runs import svc as dm_svc  # noqa: F401 — a fixture
from test_12q_g_hf_length_batch import GEMMA, NESTED
from test_12q_g_hf_length_batch import _run as devicemark_run

PLAIN = {"model_type": "qwen3_5", "hidden_size": 1024, "num_hidden_layers": 24,
         "vocab_size": 248320, "max_position_embeddings": 262144}
MODEL = "org/chat-1b"
LINE = ("[service] lm_eval is told max_length=131072: the model's limit is under text_config "
        "in its config, where lm_eval doesn't look (it would take 2,048)")


def _gpu(monkeypatch, cfg: dict, backend: str = "hf") -> list[list[str]]:
    """a local model with this config, lm_eval a stand-in: every command asked"""
    monkeypatch.setattr(runner, "preflight", lambda hf_id, k, **kw: {
        "kind": "instruct", "params": 70_000_000, "vocab": 50304, "batch": 8, "need_gb": 2.0,
        "remote_code": False, "has_template": True,
        "archinfo": hfmeta._arch_from_config(cfg)})
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    monkeypatch.setattr(runner, "gen_backend", lambda: (backend, "" if backend == "vllm"
                                                        else "no vLLM here"))
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    seen = []

    def run(sid, cmd, *a, **k):
        seen.append(cmd)
        task = cmd[cmd.index("--tasks") + 1]
        model_dir = Path(cmd[cmd.index("--output_path") + 1]).parent
        if task == sq.TASK:
            sit_simpleqa(model_dir, twenty())
        elif task.startswith("gpqa"):
            sit_gpqa(model_dir, task)
        return 0
    monkeypatch.setattr(runner, "_run_task", run)
    return seen


def _submit(client, suite: str) -> int:
    sid = client.post("/api/submissions", json={"hf_id": MODEL, "suite": suite,
                                                "kind": "instruct"}).json()["id"]
    runner.run_submission(db.get(sid))
    return sid


def _args(cmd: list[str]) -> str:
    return cmd[cmd.index("--model_args") + 1]


def _log(sid: int) -> str:
    [f] = config.LOGS_DIR.glob(f"service_{sid}_*.log")
    return f.read_text()


# ---------------------------------------------------------------------------
# which models
# ---------------------------------------------------------------------------

def test_a_limit_under_text_config_is_marked_nested():
    a = hfmeta._arch_from_config(NESTED)
    assert a["ctx"] == 131072 and a["ctx_nested"] is True
    # any of lm_eval's three names at the top, and lm_eval finds it itself
    for k in ("max_position_embeddings", "n_positions", "n_ctx"):
        assert hfmeta.CTX_KEYS.count(k) == 1
        top = hfmeta._arch_from_config({**NESTED, k: 8192})
        assert top["ctx"] == 8192 and "ctx_nested" not in top
    assert "ctx_nested" not in hfmeta._arch_from_config(PLAIN)
    # no limit anywhere: nothing to tell
    none = hfmeta._arch_from_config({"text_config": {"hidden_size": 8}})
    assert none["ctx"] is None and "ctx_nested" not in none


def test_only_a_nested_limit_is_told_to_lm_eval():
    nested = {"archinfo": hfmeta._arch_from_config(NESTED)}
    plain = {"archinfo": hfmeta._arch_from_config(PLAIN)}
    assert runner.nested_limit(nested) == 131072
    assert runner.nested_limit(plain) is None and runner.nested_limit({}) is None
    # a served model's context is its server's, never lm_eval's to find
    assert runner.nested_limit({"archinfo": {"ctx": 32768}}) is None
    assert runner.nested_limit({"archinfo": {"ctx": "?", "ctx_nested": True}}) is None
    assert runner.model_args(runner.load_spec("a/b", nested)) == (
        "pretrained=a/b,dtype=bfloat16,max_length=131072")
    # every other model's model_args are what they were, to the byte
    assert runner.model_args(runner.load_spec("a/b", plain)) == "pretrained=a/b,dtype=bfloat16"
    assert runner.model_args(runner.load_spec("a/b", {**plain, "remote_code": True,
                                                       "revision": "abc123"})) == (
        "pretrained=a/b,dtype=bfloat16,trust_remote_code=True,revision=abc123")
    # the Playground loads through the same spec, and reads only what it read
    spec = runner.load_spec("a/b", plain)
    assert (spec["pretrained"], spec["dtype"], spec["revision"]) == ("a/b", "bfloat16", None)


# ---------------------------------------------------------------------------
# the runs
# ---------------------------------------------------------------------------

def test_a_nested_models_runs_on_hf_are_told_its_limit(svc, monkeypatch):  # noqa: F811
    client, _, _ = svc
    seen = _gpu(monkeypatch, NESTED)
    sid = _submit(client, "shared")
    assert db.get(sid)["status"] == "done", db.get(sid)
    cot, sqa = seen
    # the chain of thought goes as the generative three do; SimpleQA as any other task
    assert _args(cot) == f"pretrained={MODEL},dtype=bfloat16,max_length=131072"
    assert _args(sqa) == f"pretrained={MODEL},dtype=bfloat16,max_length=131072"
    assert _log(sid).count(LINE) == 2                       # said on each task


def test_a_multiple_choice_suite_is_told_it_too(svc, monkeypatch):  # noqa: F811
    client, _, _ = svc
    seen = _gpu(monkeypatch, NESTED)
    sid = _submit(client, "quick")
    assert [c[c.index("--tasks") + 1] for c in seen] == config.tasks_for_suite("quick")
    assert all(_args(c) == f"pretrained={MODEL},dtype=bfloat16,max_length=131072" for c in seen)
    assert _log(sid).count(LINE) == len(seen)


def test_a_model_whose_limit_lm_eval_finds_is_asked_as_before(svc, monkeypatch):  # noqa: F811
    client, _, _ = svc
    seen = _gpu(monkeypatch, PLAIN)
    shared, quick = _submit(client, "shared"), _submit(client, "quick")
    assert len(seen) == 2 + len(config.tasks_for_suite("quick"))
    assert {_args(c) for c in seen} == {f"pretrained={MODEL},dtype=bfloat16"}
    assert "max_length" not in _log(shared) + _log(quick)


def test_vllm_is_given_its_own_length_as_before(svc, monkeypatch):  # noqa: F811
    client, _, _ = svc
    seen = _gpu(monkeypatch, NESTED, backend="vllm")
    monkeypatch.setattr(runner, "gpu_total_mib", lambda: 32000)
    sid = _submit(client, "shared")
    assert db.get(sid)["status"] == "done", db.get(sid)
    cot, sqa = seen
    assert cot[cot.index("--model") + 1] == "vllm"
    assert "max_length=" not in _args(cot) and "max_model_len=" in _args(cot)
    # SimpleQA is a task of the harness's own loader, on hf: told
    assert sqa[sqa.index("--model") + 1] == "hf" and _args(sqa).endswith(",max_length=131072")
    assert _log(sid).count(LINE) == 1


def test_a_devicemark_run_keeps_its_own_length(dm_svc, monkeypatch):  # noqa: F811
    sid, cmds, _ = devicemark_run(monkeypatch, GEMMA, NESTED, params=5.1e9)
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    for c in cmds:
        assert f"max_length={config.DM_PROMPT_TOKENS + dm.CAP}" in _args(c).split(",")
        assert "max_length=131072" not in _args(c)
    log = _log(sid)
    assert "[devicemark] lm_eval is told max_length=6144" in log and LINE not in log


# ---------------------------------------------------------------------------
# what is on disk already
# ---------------------------------------------------------------------------

def _result(out: Path, model: str, task: str, stamp: str, ctx, max_length=...) -> Path:
    d = out / model.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_meta.json").write_text(json.dumps({"model": model, "ctx": ctx}))
    r = d / task / "pretrained__x"
    r.mkdir(parents=True, exist_ok=True)
    blob = {"results": {}, **({} if max_length is ... else {"max_length": max_length})}
    f = r / f"results_{stamp}.json"
    f.write_text(json.dumps(blob))
    return f


def test_the_audit_lists_results_asked_at_2048_by_a_model_that_reads_more(tmp_path, capsys):
    out = tmp_path / "full"
    cut = _result(out, GEMMA, "mmlu_5shot", "2026-09-29T10-00-00", 131072, 2048)
    # asked again since, whole: its newest result is what counts
    _result(out, GEMMA, "arc_easy_25shot", "2026-09-29T10-00-00", 131072, 2048)
    _result(out, GEMMA, "arc_easy_25shot", "2026-09-30T10-00-00", 131072, 131072)
    # a DeviceMark run told its own length isn't the fallback
    _result(out, GEMMA, "dm_math_0shot", "2026-09-30T10-00-00", 131072, 6144)
    # a model that reads 2,048 was asked at its limit; one lm_eval found is at its own
    _result(out, "org/small-2k", "mmlu_5shot", "2026-09-29T10-00-00", 2048, 2048)
    _result(out, "org/chat-1b", "mmlu_5shot", "2026-09-29T10-00-00", 262144, 262144)
    # a served model's results carry no length, and one with no limit on file isn't judged
    _result(out, "served/x", "everyday_0shot", "2026-09-29T10-00-00", 32768)
    _result(out, "org/unknown", "mmlu_5shot", "2026-09-29T10-00-00", None, 2048)
    assert asked_length.under_fallback(out) == [
        {"model": GEMMA, "task": "mmlu_5shot", "asked": 2048, "reads": 131072, "file": str(cut)}]
    assert asked_length.main([str(out)]) == 0
    said = capsys.readouterr().out.splitlines()
    assert said[0] == f"{GEMMA} · mmlu_5shot · asked at 2,048, the model reads 131,072"
    assert said[1].startswith("1 task(s) of 1 model(s): any prompt of theirs longer than 2,048")
    # nothing to list says so
    cut.unlink()
    assert asked_length.main([str(out)]) == 0
    assert capsys.readouterr().out.startswith("none: no result under ")
    assert asked_length.FALLBACK == 2048
