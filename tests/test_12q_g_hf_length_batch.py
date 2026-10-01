"""12q.G: two DeviceMark calibration runs on hf that failed every time.

- Gemma 4 E2B (#148) failed all three tasks at "requested max tokens to
  generate (4096) must be less than model's maximum sequence length (2048)".
  lm_eval looks for the model's limit at the top of its config, and takes 2,048
  when it finds none; Gemma 4 keeps max_position_embeddings under text_config.
  A DeviceMark run on hf now tells lm_eval the length itself: the prompt's room
  and the 4,096 cap. The model's own limit is read from text_config where the
  top has none, and a model that reads fewer tokens is refused at preflight.
- Granite-4.0-H-1B (#132, #144, #147) ran out of GPU memory on MMLU-Pro. Its
  answers are written at a batch sized for them (4 at most), from an estimate
  that counts 4,096 generated tokens and, for a Mamba2 hybrid in an image
  without the Mamba kernels, the slow path's 1.5 GiB for every 256 prompt
  tokens. An out-of-memory error says who held what, and which grew.

No model runs: lm_eval is a stand-in, as in test_12q_e_hf_pairs.py."""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import pytest

import devicemark as dm
from service import config, db, hfmeta, runner
from service import devicemark as sdm
from test_12q_devicemark_runs import ME, row_dir, svc  # noqa: F401 — svc is the fixture

GEMMA, GRANITE, QWEN = "google/gemma-4-E2B-it", "ibm-granite/granite-4.0-h-1b", "Qwen/Qwen3.5-4B"
# the shape of a config that nests its text model: nothing about length at the top
NESTED = {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
          "text_config": {"max_position_embeddings": 131072, "hidden_size": 1536,
                          "num_hidden_layers": 35, "vocab_size": 262144}}
# Granite-4.0-H-1B's, from its config.json: 36 Mamba2 layers and 4 of attention
HYBRID = {"model_type": "granitemoehybrid", "architectures": ["GraniteMoeHybridForCausalLM"],
          "hidden_size": 1536, "num_hidden_layers": 40, "vocab_size": 100352,
          "max_position_embeddings": 131072, "mamba_n_heads": 48, "mamba_d_state": 128,
          "mamba_chunk_size": 256,
          "layer_types": (["mamba"] * 5 + ["attention"] + ["mamba"] * 9 + ["attention"]
                          + ["mamba"] * 9 + ["attention"] + ["mamba"] * 9 + ["attention"]
                          + ["mamba"] * 4)}
PLAIN = {"model_type": "qwen3_5", "hidden_size": 2560, "num_hidden_layers": 36,
         "vocab_size": 248320, "max_position_embeddings": 262144}
# PyTorch's message, with the numbers of #147's log
OOM = ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.00 GiB. GPU 0 has a "
       "total capacity of 31.36 GiB of which 2.37 GiB is free. Process 4121 has 12.90 GiB "
       "memory in use. Including non-PyTorch memory, this process has 15.90 GiB memory in use. "
       "Of the allocated memory 13.41 GiB is allocated by PyTorch, and 2.04 GiB is reserved by "
       "PyTorch but unallocated. If reserved but unallocated memory is large try setting "
       "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.\n")


def _run(monkeypatch, hf_id: str, cfg: dict, *, params: float, kernels: bool = True,
         longest: tuple[int, bool] = (900, True), behave=None, batch: int = 16):
    """a run of the battery on hf, lm_eval a stand-in. `behave(task, batch,
    lf)` may write to the log and return an exit code; a task that returns
    none answers. Returns the submission's id and each lm_eval command"""
    cmds, envs = [], []
    vocab = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]
    monkeypatch.setattr(config, "MAX_JOB_GB", 10.0)
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: {
        "kind": "instruct", "params": params, "vocab": vocab, "batch": batch, "need_gb": 6.0,
        "remote_code": False, "has_template": True, "kind_reason": "chat template",
        "archinfo": {**hfmeta._arch_from_config(cfg), "thinking": "never",
                     "think_end": "</think>"}})
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    monkeypatch.setattr(runner, "gen_backend", lambda: ("hf", "vLLM is not installed here"))
    monkeypatch.setattr(hfmeta, "mamba_kernels", lambda: kernels)
    monkeypatch.setattr(sdm, "prompt_tokens", lambda *a, **k: longest)
    items = dm.load_items(config.DM_ITEMS)

    def run_task(sid, cmd, lf, env, run_as, cwd, on_poll=None):
        task = cmd[cmd.index("--tasks") + 1]
        cmds.append(cmd)
        envs.append(env)
        status = behave(task, int(cmd[cmd.index("--batch_size") + 1]), lf) if behave else None
        lf.flush()
        if status:
            return status
        out = Path(cmd[cmd.index("--output_path") + 1]) / hf_id.replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        b = dm.BENCH_OF[task]
        with open(out / f"samples_{task}_2026-09-30T10-00-00.jsonl", "w") as fh:
            for bb, k in dm.keys_for("full"):
                if bb == b:
                    fh.write(json.dumps({"doc": {"bench": b, "key": k,
                                                 "text": items[(b, k)]["text"]},
                                         "resps": [["\\boxed{A}"]]}) + "\n")
        return 0
    monkeypatch.setattr(runner, "_run_task", run_task)
    # each run here starts with no answers on disk: a task answered already
    # is not asked again (#167's resume), and these tests look at the asking
    for d in row_dir(hf_id).glob("dm_*_0shot"):
        shutil.rmtree(d)
    sid = db.add(hf_id, "instruct", "devicemark", ME, "", part="full")
    runner.run_submission(db.get(sid))
    return sid, cmds, envs


def _arg(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def _log(sid: int) -> str:
    [f] = config.LOGS_DIR.glob(f"service_{sid}_*.log")
    return f.read_text()


# ---------------------------------------------------------------------------
# 1. the length lm_eval is told
# ---------------------------------------------------------------------------

def test_the_models_limit_is_read_from_a_nested_text_config():
    assert "max_position_embeddings" not in NESTED          # where lm_eval looks, and finds none
    assert hfmeta._arch_from_config(NESTED)["ctx"] == 131072
    # the top of the config wins where it has one
    assert hfmeta._arch_from_config({**NESTED, "max_position_embeddings": 8192})["ctx"] == 8192


def test_a_run_tells_lm_eval_the_length_the_prompt_and_the_cap(svc, monkeypatch):  # noqa: F811
    sid, cmds, _ = _run(monkeypatch, GEMMA, NESTED, params=5.1e9)
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    # all three tasks, never lm_eval's 2,048: 2,048 for the prompt and the 4,096 cap
    assert [_arg(c, "--tasks") for c in cmds] == config.DM_TASKS
    for c in cmds:
        assert f"max_length={config.DM_PROMPT_TOKENS + dm.CAP}" in _arg(c, "--model_args").split(",")
    assert config.DM_PROMPT_TOKENS + dm.CAP == 6144
    log = _log(sid)
    assert ("[devicemark] lm_eval is told max_length=6144: 2,048 tokens for the prompt (the "
            "battery's longest is 900 in this model's tokens) and the cap of 4,096 · the model "
            "reads up to 131,072") in log
    assert json.loads((row_dir(GEMMA) / dm.OUT_NAME).read_text())["n"] == 596


def test_the_room_grows_with_the_batterys_longest_prompt(svc, monkeypatch):  # noqa: F811
    # a question of 3,000 tokens is never cut: its room, the template's and the cap
    sid, cmds, _ = _run(monkeypatch, QWEN, PLAIN, params=4.2e9, longest=(3000, True))
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    assert f"max_length={3000 + sdm.PROMPT_SLACK + dm.CAP}" in _arg(cmds[0], "--model_args")


def test_preflight_refuses_a_model_that_reads_too_few_tokens(svc, monkeypatch):  # noqa: F811
    short = {**NESTED, "text_config": {**NESTED["text_config"], "max_position_embeddings": 4096}}
    sid, cmds, _ = _run(monkeypatch, GEMMA, short, params=5.1e9)
    row = db.get(sid)
    assert row["status"] == "failed" and cmds == []         # nothing was loaded
    assert row["error"] == (
        f"{GEMMA} reads 4,096 tokens at most (its config.json), and DeviceMark's protocol "
        "needs 6,144: 2,048 for the prompt (the battery's longest is 900 in its tokens) and the "
        "cap of 4,096 for the answer. It can't sit the battery as the other models do.")
    assert not (row_dir(GEMMA) / dm.OUT_NAME).exists()
    # exactly enough is enough
    ok = {**NESTED, "text_config": {**NESTED["text_config"], "max_position_embeddings": 6144}}
    sid, cmds, _ = _run(monkeypatch, GEMMA, ok, params=5.1e9)
    assert db.get(sid)["status"] == "done" and len(cmds) == 3


def test_a_model_with_no_limit_in_its_config_is_told_the_length_too(svc, monkeypatch):  # noqa: F811
    cfg = {k: v for k, v in PLAIN.items() if k != "max_position_embeddings"}
    sid, cmds, _ = _run(monkeypatch, QWEN, cfg, params=4.2e9)
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    assert "max_length=6144" in _arg(cmds[0], "--model_args")
    assert "the model's config gives no limit of its own" in _log(sid)


def test_only_a_devicemark_run_on_hf_is_told_a_length():
    th = {"mode": "never", "on": False, "think_end": "</think>", "budget": 4096}
    assert "max_length" not in runner.gen_model_args("a/b", th, backend="hf")
    assert runner.gen_model_args("a/b", th, backend="hf", max_length=6144).endswith(
        ",max_length=6144")
    # vLLM is given its own (max_model_len), as before
    v = runner.gen_model_args("a/b", th, backend="vllm", max_length=6144)
    assert "max_length=" not in v and "max_model_len=" in v


# ---------------------------------------------------------------------------
# the battery's longest prompt, in the model's tokens
# ---------------------------------------------------------------------------

def test_the_longest_prompt_is_counted_as_lm_eval_sends_it(monkeypatch):
    asked = {}

    class Tok:
        def apply_chat_template(self, messages, **kw):
            asked.update(kw, messages=messages)
            return "<user>" + messages[0]["content"] + "<assistant>"

        def __call__(self, text, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": text.replace("<", " <").replace(">", "> ").split()}

    class Auto:
        @staticmethod
        def from_pretrained(name, revision=None, trust_remote_code=None):
            asked.update(name=name, revision=revision, code=trust_remote_code)
            return Tok()
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=Auto))
    n, counted = sdm.prompt_tokens("a/b", "abc123", ["one two", "one two three four", "x"],
                                   {"mode": "switch", "on": False})
    assert (n, counted) == (6, True)                        # four words and the template's two
    assert asked["name"] == "a/b" and asked["revision"] == "abc123"
    assert asked["code"] is False                           # the service runs no model's code
    assert asked["enable_thinking"] is False and asked["add_generation_prompt"] is True
    assert asked["tokenize"] is False
    # a model with no switch is sent none
    asked.clear()
    sdm.prompt_tokens("a/b", None, ["one"], {"mode": "never", "on": False})
    assert "enable_thinking" not in asked


def test_a_tokenizer_that_doesnt_load_is_estimated_by_characters(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)  # importing it fails
    assert sdm.prompt_tokens("a/b", None, ["x" * 300, "y" * 3000], {}) == (1000, False)
    assert sdm.prompt_tokens("a/b", None, ["x" * 301], {}) == (101, False)


def test_an_uncounted_longest_says_so(svc, monkeypatch):  # noqa: F811
    sid, _, _ = _run(monkeypatch, QWEN, PLAIN, params=4.2e9, longest=(1200, False))
    assert ("the battery's longest is about 1,200 in this model's tokens, by its characters"
            in _log(sid))


# ---------------------------------------------------------------------------
# 2. the batch its answers are written at, and the memory that takes
# ---------------------------------------------------------------------------

def test_the_slow_paths_tensor_is_the_six_gib_of_the_log():
    # one state-space layer's tensor is chunk x chunk x heads x state in fp32 for every
    # chunk and every answer at a time: 1.5 GiB. A batch of 2 on prompts of two chunks
    # (257 to 512 tokens) asks for the 6.00 GiB of #147's log, exactly
    m = hfmeta._arch_from_config(HYBRID)["mamba"]
    assert m == {"layers": 36, "attention": 4, "heads": 48, "state": 128, "chunk": 256}
    one = m["chunk"] ** 2 * m["heads"] * m["state"] * 4
    assert one == 1.5 * 2 ** 30 and 2 * 2 * one == 6 * 2 ** 30


def test_the_estimate_counts_the_answer_the_prompt_and_the_slow_path(monkeypatch):
    monkeypatch.setattr(config, "MAX_JOB_GB", 10.0)
    arch = hfmeta._arch_from_config(HYBRID)
    # with the kernels, a 1.5B hybrid's answers cost little each: 4 at a time
    fast = hfmeta.gen_estimate(arch, 1.5e9, 100352, prompt=500, new=4096, max_batch=4,
                               kernels=True)
    assert fast["batch"] == 4 and "slow path" not in fast["why"]
    # only its 4 attention layers keep a cache: 4 x 2 x 1,536 x 4,596 tokens x 2 bytes
    each = 4 * 2 * 1536 * 4596 * 2 / 1e9 + 500 * 100352 * 4 / 1e9
    assert fast["need_gb"] == round(3.0 + 1.5 + 4 * each, 2)
    # without them, two chunks of prompt are 2 x 1.5 GiB and half again: one at a time
    slow = hfmeta.gen_estimate(arch, 1.5e9, 100352, prompt=500, new=4096, max_batch=4,
                               kernels=False)
    assert slow["batch"] == 1
    assert slow["need_gb"] == round(3.0 + 1.5 + each + 2 * 1.5 * 2 ** 30 * 1.5 / 1e9, 2)
    assert ("4.8 GB for its Mamba layers on the slow path (2 x 256 prompt tokens; this image "
            "has no Mamba kernels)") in slow["why"]
    # and one that doesn't fit at 1 still runs at 1, with what it needs said
    long = hfmeta.gen_estimate(arch, 1.5e9, 100352, prompt=1500, new=4096, max_batch=4,
                               kernels=False)
    assert long["batch"] == 1 and long["need_gb"] > 10

    # a plain transformer: every layer keeps its cache for the prompt and 4,096 more
    plain = hfmeta._arch_from_config(PLAIN)
    a = hfmeta.gen_estimate(plain, 0.9e9, 248320, prompt=900, new=4096, max_batch=4)
    cache = 36 * 2 * 2560 * (900 + 4096) * 2 / 1e9
    assert a["batch"] == 2                  # 1.8 + 1.5 + 2 x (1.84 + 0.89) = 8.8; 4 would be 14.2
    assert a["need_gb"] == round(1.8 + 1.5 + 2 * (cache + 900 * 248320 * 4 / 1e9), 2)
    assert "1.8 GB of cache for 900 + 4,096 tokens" in a["why"]
    # the cap is the most it ever starts at
    small = {"hidden": 512, "layers": 8}
    assert hfmeta.gen_estimate(small, 0.2e9, 32000, prompt=500, new=4096)["batch"] == 8
    assert hfmeta.gen_estimate(small, 0.2e9, 32000, prompt=500, new=4096,
                               max_batch=4)["batch"] == 4


def test_a_devicemark_run_passes_lm_eval_the_capped_batch(svc, monkeypatch):  # noqa: F811
    # preflight's batch is sized for scoring (16 here): the run never starts above 4
    small = {"model_type": "qwen3_5", "hidden_size": 1024, "num_hidden_layers": 24,
             "vocab_size": 248320, "max_position_embeddings": 262144}
    sid, cmds, _ = _run(monkeypatch, "Qwen/Qwen3.5-0.8B", small, params=0.85e9, batch=16)
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    assert [(_arg(c, "--tasks"), _arg(c, "--batch_size")) for c in cmds] == [
        (t, str(config.DM_HF_MAX_BATCH)) for t in config.DM_TASKS]
    assert config.DM_HF_MAX_BATCH == 4
    assert db.get(sid)["batch"] == 4
    assert "[devicemark] 4 answers written at a time · about " in _log(sid)


def test_granite_without_the_kernels_starts_at_one(svc, monkeypatch):  # noqa: F811
    sid, cmds, _ = _run(monkeypatch, GRANITE, HYBRID, params=1.5e9, kernels=False,
                        longest=(500, True))
    row = db.get(sid)
    assert row["status"] == "done", row["error"]
    assert [_arg(c, "--batch_size") for c in cmds] == ["1", "1", "1"]
    # the wait for the card, and the row, go by what writing takes
    assert row["batch"] == 1 and row["need_gb"] == pytest.approx(9.64, abs=0.02)
    log = _log(sid)
    assert "[devicemark] 1 answer written at a time · about " in log
    assert "for its Mamba layers on the slow path (2 x 256 prompt tokens" in log
    # with them it is a small model like any other
    sid, cmds, _ = _run(monkeypatch, GRANITE, HYBRID, params=1.5e9, kernels=True,
                        longest=(500, True))
    assert [_arg(c, "--batch_size") for c in cmds] == ["4", "4", "4"]


def test_the_run_asks_pytorch_to_hand_back_what_it_frees(svc, monkeypatch):  # noqa: F811
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    _, _, envs = _run(monkeypatch, GRANITE, HYBRID, params=1.5e9, kernels=False)
    assert {e["PYTORCH_CUDA_ALLOC_CONF"] for e in envs} == {"expandable_segments:True"}
    # the operator's own setting is left as it is
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    _, _, envs = _run(monkeypatch, GRANITE, HYBRID, params=1.5e9, kernels=False)
    assert {e["PYTORCH_CUDA_ALLOC_CONF"] for e in envs} == {"max_split_size_mb:512"}


# ---------------------------------------------------------------------------
# an out-of-memory error says who held what, and which grew
# ---------------------------------------------------------------------------

def test_pytorchs_message_is_read_for_who_held_what():
    o = runner.oom_said("Traceback…\n" + OOM)
    assert o == {"asked": 6.0, "own": 15.9, "others": 12.9, "total": 31.36, "spare": 2.04}
    # no process listed (a container that can't see them): what is left of the card
    bare = OOM.replace("Process 4121 has 12.90 GiB memory in use. ", "")
    assert runner.oom_said(bare)["others"] == pytest.approx(31.36 - 2.37 - 15.9)
    # nor its own: what PyTorch had allocated and reserved
    none = bare.replace("Including non-PyTorch memory, this process has 15.90 GiB memory in "
                        "use. ", "")
    assert runner.oom_said(none)["own"] == pytest.approx(13.41 + 2.04)
    # sizes in MiB, and the last message when a retry left two
    two = OOM + OOM.replace("6.00 GiB", "512.00 MiB")
    assert runner.oom_said(two)["asked"] == 0.5
    assert runner.oom_said("RuntimeError: out of memory") is None


def test_the_line_says_which_grew():
    o = runner.oom_said(OOM)
    own = ("this run's own process held 15.9 GiB (2 GiB of it reserved and unused) and asked "
           "for 6 GiB more; the card's other processes held 12.9 GiB")
    # the neighbour held the same when the task started: the run grew
    assert runner.oom_line(o, int(12.9 * 1024)) == (
        own + ", as when the task started: the run grew, not the card")
    assert runner.oom_line(o, int(12.4 * 1024)) == (
        own + " (12.4 GiB when the task started): the run grew, not the card")
    # a neighbour that took 4 GiB more meanwhile: the card got busier
    assert runner.oom_line(o, int(8.9 * 1024)) == (
        own + ", up from 8.9 GiB when the task started: the card got busier")
    # what they held then isn't known: no verdict
    assert runner.oom_line(o) == own
    assert runner.oom_line({**o, "others": None, "spare": 0.2}) == (
        "this run's own process held 15.9 GiB and asked for 6 GiB more")


def test_a_run_out_of_memory_at_one_says_the_run_grew(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(runner, "others_mib", lambda: int(12.9 * 1024))

    def behave(task, batch, lf):
        if task == "dm_mmlu_pro":
            lf.write(OOM)
            return 1
    sid, cmds, _ = _run(monkeypatch, GRANITE, HYBRID, params=1.5e9, kernels=False,
                        longest=(500, True), behave=behave)
    row = db.get(sid)
    assert [_arg(c, "--tasks") for c in cmds] == ["dm_ifeval", "dm_mmlu_pro"]
    assert row["status"] == "failed"
    assert row["error"].startswith(
        "dm_mmlu_pro: ran out of GPU memory at batch 1: this run's own process held 15.9 GiB "
        "(2 GiB of it reserved and unused) and asked for 6 GiB more; the card's other processes "
        "held 12.9 GiB, as when the task started: the run grew, not the card. There is no "
        "smaller batch to try: the model needs more than the card has free.")
    assert "busier" not in row["error"]


def test_a_retry_at_half_the_batch_logs_who_held_what(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(runner, "others_mib", lambda: int(8.0 * 1024))

    def behave(task, batch, lf):
        if task == "dm_mmlu_pro" and batch > 1:
            lf.write(OOM)
            return 1
    small = {"model_type": "qwen3_5", "hidden_size": 1024, "num_hidden_layers": 24,
             "vocab_size": 248320, "max_position_embeddings": 262144}
    sid, cmds, _ = _run(monkeypatch, "Qwen/Qwen3.5-0.8B", small, params=0.85e9, behave=behave)
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    assert [(_arg(c, "--tasks"), _arg(c, "--batch_size")) for c in cmds] == [
        ("dm_ifeval", "4"), ("dm_mmlu_pro", "4"), ("dm_mmlu_pro", "2"), ("dm_mmlu_pro", "1"),
        ("dm_math", "1")]
    log = _log(sid)
    assert ("[service] dm_mmlu_pro ran out of GPU memory: again at batch 2 (this run's own "
            "process held 15.9 GiB") in log
    assert "up from 8 GiB when the task started: the card got busier)" in log
    # the task after it says the batch it runs at, and why
    assert "[devicemark] 1 answer written at a time: 4 ran out of GPU memory" in log


@pytest.mark.parametrize("others", [None, 0, 13209])
def test_the_run_lock_is_released_whatever_the_card_holds(svc, monkeypatch, others):  # noqa: F811
    # what the others hold is read before each task; the run's lock is its own matter, and
    # is given back on an empty card (0 MiB) and where nvidia-smi says nothing (None) too
    released = []
    monkeypatch.setattr(runner, "release_lock", lambda: released.append(1))
    monkeypatch.setattr(runner, "others_mib", lambda: others)
    sid, _, _ = _run(monkeypatch, QWEN, PLAIN, params=4.2e9)
    assert db.get(sid)["status"] == "done" and released == [1]


# ---------------------------------------------------------------------------
# a prompt lm_eval cut isn't the protocol's question
# ---------------------------------------------------------------------------

def test_a_cut_prompt_fails_the_task_and_keeps_none_of_it(svc, monkeypatch):  # noqa: F811
    def behave(task, batch, lf):
        if task == "dm_mmlu_pro":
            lf.write("WARNING Left truncation applied. Original sequence length was 2301, "
                     "truncating to last 2048 tokens. Some content will be lost.\n")
    sid, cmds, _ = _run(monkeypatch, QWEN, PLAIN, params=4.2e9, longest=(700, False),
                        behave=behave)
    row = db.get(sid)
    assert row["status"] == "failed"
    assert [_arg(c, "--tasks") for c in cmds] == ["dm_ifeval", "dm_mmlu_pro"]
    assert row["error"] == (
        "dm_mmlu_pro: a prompt of 2,301 tokens was cut to the 2,048 this run left for it, so "
        "its answers aren't the protocol's and none were kept. Set DM_PROMPT_TOKENS to 2,301 "
        "or more and resubmit.")
    assert not (row_dir(QWEN) / "dm_mmlu_pro_0shot").exists()
    assert (row_dir(QWEN) / "dm_ifeval_0shot").exists()     # the task before it is kept
    assert not (row_dir(QWEN) / dm.OUT_NAME).exists()
