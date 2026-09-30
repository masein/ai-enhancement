"""12f.4: the image's Python has what its runs import. A served model sits the
generative suite through lm_eval's local-chat-completions, which needs lm_eval's
`api` extra; the image lacked it, and the first two served runs failed at once
("missing package"). The fake-server tests never import lm_eval, so they missed
it. These do — in CI's image-deps job (the image's requirements over a CPU
torch, EVALBOARD_IMAGE_DEPS=1, where a missing lm_eval fails) and in deploy
step 3, inside the image. Elsewhere, with no lm_eval at all, they skip."""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import re
import sys
from pathlib import Path

import pytest

from test_12q_devicemark_runs import svc  # noqa: F401 — a fixture
from test_step3_collects import IN_IMAGE

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^lm_eval\[([^\]]+)\]==([\d.]+)", re.M)
# a distribution whose module is named otherwise
MODULE = {"beautifulsoup4": "bs4", "pyyaml": "yaml", "scikit-learn": "sklearn"}


def lm_eval():
    try:
        return importlib.import_module("lm_eval")
    except ImportError:
        if os.environ.get("EVALBOARD_IMAGE_DEPS") == "1":
            raise
        pytest.skip("lm_eval isn't installed here: CI's image-deps job and deploy step 3 run this")


def test_the_pin_carries_the_extras_the_runs_use():
    m = PIN.search((ROOT / "requirements.txt").read_text(encoding="utf-8"))
    assert m, "requirements.txt pins lm_eval with its extras"
    assert {"ifeval", "math", "api"} <= {x.strip() for x in m.group(1).split(",")}


def test_what_is_installed_is_the_pin():
    lm_eval()
    assert md.version("lm_eval") == PIN.search(
        (ROOT / "requirements.txt").read_text(encoding="utf-8")).group(2)


def test_local_chat_completions_and_every_package_of_the_api_extra_import():
    lm_eval()
    from lm_eval.api.registry import get_model
    cls = get_model("local-chat-completions")
    assert cls.__module__ == "lm_eval.models.openai_completions"
    importlib.import_module("lm_eval.models.openai_completions")
    # what lm_eval itself says the api extra needs
    extra = [re.split(r"[\s;<>=!~\[]", r, maxsplit=1)[0] for r in md.requires("lm_eval") or []
             if re.search(r"extra\s*==\s*['\"]api['\"]", r)]
    assert extra, "lm_eval declares an api extra"
    for dist in extra:
        name = MODULE.get(dist.lower(), dist.replace("-", "_").lower())
        importlib.import_module(name)


def test_trust_and_safety_tasks_load_in_the_installed_harness(tmp_path):
    """12k.2: BBQ, Do-Not-Answer and XSTest as a run builds them, found by the
    installed lm_eval under our own names (it ships a `bbq` and a `bbq_ambig`
    of its own, on an unpinned copy), and posing what we mean them to — no
    model is loaded"""
    lm_eval()
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import trust_safety as ts
    from lm_eval.tasks import TaskManager, get_task_dict
    d = ts.build_tasks(tmp_path / "tasks")
    tm = TaskManager(include_path=str(d))
    assert {"bbq", "bbq_ambig"} <= set(tm.all_tasks)          # lm_eval's own, left alone
    tasks = get_task_dict(["bbq_3000", "do_not_answer", "xstest"], task_manager=tm)
    bbq = tasks["bbq_3000"]
    docs = list(bbq.test_docs())
    assert len(docs) == 3000 and bbq.OUTPUT_TYPE == "multiple_choice"
    q = docs[0]
    assert bbq.doc_to_text(q) == f"{q['context']}\n\nQ: {q['question']}\nA:"
    assert bbq.doc_to_choice(q) == q["choices"] and bbq.doc_to_target(q) == q["label"]
    # its scoring: the option with the highest log-likelihood, and the bias score beside acc
    got = bbq.process_results(q, [(-1.0 if i == q["label"] else -9.0, False) for i in range(3)])
    assert got["acc"] == 1 and tuple(got["bias_score"]) == (1, 0, 0)
    assert bbq.aggregation()["bias_score"]([(0, 1, 1), (0, 0, 1)]) == 0.0
    for name, n in (("do_not_answer", 939), ("xstest", 450)):
        t = tasks[name]
        docs = list(t.test_docs())
        assert len(docs) == n and t.OUTPUT_TYPE == "generate_until"
        assert t.doc_to_text(docs[0]) == docs[0]["prompt"]
        assert t.config.generation_kwargs["max_gen_toks"] == 512
    assert len(list(get_task_dict(["bbq_all"], task_manager=tm)["bbq_all"].test_docs())) == 29_246


def test_the_encrypted_export_imports():
    """12p.1b: `hidden_store backup --export` encrypts in-process with pyrage"""
    try:
        import pyrage
    except ImportError:
        if os.environ.get("EVALBOARD_IMAGE_DEPS") == "1":
            raise
        pytest.skip("pyrage isn't installed here: CI's image-deps job and deploy step 3 run this")
    assert re.search(r"^pyrage==1\.4\.0", (ROOT / "requirements.txt").read_text(encoding="utf-8"),
                     re.M)
    assert md.version("pyrage") == "1.4.0" and pyrage.x25519.Identity.generate()


VENDORED = ("instructions.py", "instructions_registry.py", "instructions_util.py", "utils.py")


def test_the_vendored_ifeval_checker_is_lm_evals():
    """12q: scripts/ifeval_official is lm_eval's IFEval checker, but for its two
    marked changes — its imports are the folder's own, and punkt is fetched
    when a sentence is first counted. Both undone, it is the installed one"""
    lm_eval()
    import importlib.util
    theirs = Path(next(iter(importlib.util.find_spec("lm_eval").submodule_search_locations))) \
        / "tasks" / "ifeval"
    for name in VENDORED:
        ours = (ROOT / "scripts" / "ifeval_official" / name).read_text(encoding="utf-8")
        ours = "".join(x for x in ours.splitlines(keepends=True) if not x.startswith("# 12q:"))
        ours = ours.replace("from . import ", "from lm_eval.tasks.ifeval import ")
        ours = ours.replace("\n# download_nltk_resources()  # 12q: on the first sentence counted\n",
                            "\ndownload_nltk_resources()\n")
        ours = ours.replace("    download_nltk_resources()  # 12q: here, not on import\n", "")
        assert ours == (theirs / name).read_text(encoding="utf-8"), name


def test_ifeval_counts_sentences_with_no_network(monkeypatch):
    """12q.D: IFEval's checkers split sentences with NLTK's punkt_tab, and the
    image has it from its build (scripts/nltk_data.py): five DeviceMark runs on
    hf failed at "Resource 'punkt_tab' not found" when the checker went to
    download it. With the network cut off and a download made to fail, the
    checker still counts sentences — in CI's image-deps job, which fetches it
    as the image does, and in deploy step 3, inside the image"""
    lm_eval()
    import socket

    import nltk

    def no_network(*a, **k):
        raise OSError("12q.D: no network in this test")
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(nltk, "download", lambda *a, **k: pytest.fail(
        "the IFEval checker tried to download punkt_tab"))
    nltk.data.find("tokenizers/punkt_tab/english/")
    sys.path.insert(0, str(ROOT / "scripts"))
    from ifeval_official import instructions_registry, instructions_util
    instructions_util._get_sentence_tokenizer.cache_clear()
    text = "It rained all day. We stayed in! Did it ever stop?"
    assert instructions_util.count_sentences(text) == 3
    check = instructions_registry.INSTRUCTION_DICT["length_constraints:number_sentences"]("x")
    check.build_description(num_sentences=3, relation="at least")
    assert check.check_following(text) is True
    check.build_description(num_sentences=4, relation="at least")
    assert check.check_following(text) is False


def test_the_image_fetches_the_pinned_punkt_tab():
    """12q.D: the Dockerfile fetches it with the pinned script, and the build
    fails when it isn't found; CI's image-deps job fetches it the same way"""
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "RUN python /tmp/nltk_data.py --dest /usr/share/nltk_data" in docker
    assert "nltk.data.find('tokenizers/punkt_tab/english/')" in docker
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'python scripts/nltk_data.py --dest "$RUNNER_TEMP/nltk_data"' in ci


def test_lm_eval_takes_the_length_a_devicemark_run_tells_it():
    """12q.G: a DeviceMark run on hf passes lm_eval `max_length` (the prompt's
    room and the cap), because lm_eval takes 2,048 for a model whose limit
    isn't at the top of its config (Gemma 4's is under text_config: "requested
    max tokens to generate (4096) must be less than model's maximum sequence
    length (2048)", #148). And the runner reads lm_eval's own warning for a
    prompt it cut. Both are the installed harness's"""
    lm_eval()
    import inspect

    from lm_eval.models.huggingface import HFLM

    from service import runner
    assert "max_length" in inspect.signature(HFLM.__init__).parameters
    assert HFLM._DEFAULT_MAX_LENGTH == 2048
    src = inspect.getsource(sys.modules[HFLM.__module__])
    assert '("n_positions", "max_position_embeddings", "n_ctx")' in src
    assert "must be less than model's maximum sequence length" in src
    warning = re.search(r'f"(Left truncation applied\. Original sequence length was )'
                        r'\{\w+\}, "\s+f"(truncating to last )\{\w+\}( tokens)', src)
    assert warning, "lm_eval's warning for a cut prompt has other words now"
    said = warning.group(1) + "2301, " + warning.group(2) + "2048" + warning.group(3)
    assert runner._CUT.findall(said) == [("2301", "2048")]


def test_the_longest_prompt_is_counted_on_the_installed_transformers(tmp_path):
    """12q.G: the battery's longest prompt is counted with the model's own
    tokenizer, as lm_eval sends it: the chat template, the thinking switch. A
    tokenizer made here, so nothing is fetched"""
    lm_eval()
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from service import devicemark as sdm
    words = ["[UNK]", "<user>", "<assistant>", "<think>", "one", "two", "three", "four"]
    inner = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    inner.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=inner, unk_token="[UNK]")
    tok.chat_template = ("{% for m in messages %}<user> {{ m['content'] }}{% endfor %}"
                         "{% if add_generation_prompt %} <assistant>{% endif %}"
                         "{% if enable_thinking %} <think>{% endif %}")
    tok.save_pretrained(tmp_path / "tok")
    texts = ["one two", "one two three four", "one"]
    # the longest question's four words, and the template's two
    assert sdm.prompt_tokens(str(tmp_path / "tok"), None, texts,
                             {"mode": "switch", "on": False}) == (6, True)
    # the switch reaches the template
    assert sdm.prompt_tokens(str(tmp_path / "tok"), None, texts,
                             {"mode": "switch", "on": True}) == (7, True)
    # a tokenizer that isn't there is estimated, never an error
    assert sdm.prompt_tokens(str(tmp_path / "none"), None, ["x" * 30], {}) == (10, False)


def test_lm_eval_takes_2048_for_a_limit_under_text_config_and_writes_what_it_used():
    """12t: lm_eval looks for a model's limit at the top of its config only. A
    config that nests its text model has none there, so lm_eval takes 2,048 —
    unless it is told (runner.nested_limit passes max_length). And it writes
    the length it used into its results file, which scripts/asked_length.py
    reads. Both are the installed harness's"""
    lm_eval()
    import inspect
    import types

    import lm_eval.loggers.utils as logged
    from lm_eval.models import huggingface
    from lm_eval.models.huggingface import HFLM

    import asked_length
    nested = types.SimpleNamespace(text_config=types.SimpleNamespace(max_position_embeddings=131072))
    lm = types.SimpleNamespace(
        _max_length=None, _DEFAULT_MAX_LENGTH=HFLM._DEFAULT_MAX_LENGTH,
        model=types.SimpleNamespace(config=nested),
        tokenizer=types.SimpleNamespace(model_max_length=huggingface.TOKENIZER_INFINITY))
    assert HFLM.max_length.fget(lm) == asked_length.FALLBACK == 2048
    lm._max_length = 131072
    assert HFLM.max_length.fget(lm) == 131072
    # a limit at the top is found without being told
    lm._max_length = None
    lm.model.config = types.SimpleNamespace(max_position_embeddings=32768)
    assert HFLM.max_length.fget(lm) == 32768
    assert '"max_length": getattr(lm, "max_length", None)' in inspect.getsource(
        logged.add_tokenizer_info)


def test_the_image_has_what_the_tests_import_at_the_top():
    """Step 3 collects every test module inside the image, and a module that
    fails to import there stops the whole run. tests/test_step3_collects.py
    sorts every distribution the dev and CI requirement files name into "the
    image has it" or not, and collects the suite without the second kind.
    This is the other half: the first kind imports, in the image"""
    lm_eval()
    for name in sorted(IN_IMAGE):
        importlib.import_module(name)


def test_an_hf_devicemark_run_in_a_test_fetches_no_tokenizer(svc, monkeypatch):  # noqa: F811
    """A DeviceMark run on hf counts each answer with the model's tokenizer
    (devicemark.mark_hf). Where transformers is installed that went to the Hub
    for every model id a test names — in step 3, from the server. The tests'
    shared fixture stands a refusal in for the loader; here, beside the real
    transformers, a whole run asks for no host but this one"""
    lm_eval()
    import socket

    import transformers

    from service import db
    from test_12q_g_hf_length_batch import PLAIN, QWEN, _run
    asked = []
    here = ("127.0.0.1", "localhost", "::1", None)
    connect, lookup = socket.socket.connect, socket.getaddrinfo

    def no_connect(self, address, *a, **k):
        if not isinstance(address, tuple) or address[0] in here:
            return connect(self, address, *a, **k)
        asked.append(address[0])
        raise OSError("no network in this test")

    def no_lookup(host, *a, **k):
        if host in here:
            return lookup(host, *a, **k)
        asked.append(host)
        raise OSError("no network in this test")
    monkeypatch.setattr(socket.socket, "connect", no_connect)
    monkeypatch.setattr(socket, "getaddrinfo", no_lookup)
    sid, cmds, _ = _run(monkeypatch, QWEN, PLAIN, params=4.2e9)
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    assert len(cmds) == 3 and asked == []
    with pytest.raises(OSError, match="no tokenizer is fetched in a test"):
        transformers.AutoTokenizer.from_pretrained(QWEN)
