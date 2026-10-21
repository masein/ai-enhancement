"""25: IFBench, checked by its own code (scripts/ifbench_official, AI2's,
vendored at 1c40f0c) on the reply alone. The vendored files are the
commit's but for their marked lines; they give the numbers IFBench's own
run_eval gives on its sample_output.jsonl (tests/fixtures/ifbench, credited
there); five of its new kinds of constraint followed and broken; the
thinking never counts; an answer that ran out of room follows nothing; and
importing the checkers downloads and makes nothing. No model runs."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import frontier as fb
import instruction_following as inf

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "scripts" / "ifbench_official"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ifbench"
TASK = "ifbench_test"
# allenai/IFBench at 1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d: each file's sha256
UPSTREAM = {
    "__init__.py": "2a806107f115facaa3e7e6046ca97247237b4b113b514bcdd763b2530141f694",
    "classic_instructions.py": "d26e642f2e71e4f57218e5e05f098a22e36311485779baac401161d3228c9973",
    "evaluation_lib.py": "e681a4b03a6a9dfb540fbd9ef6b963de503a5ffac557ed611c326ee385850cf6",
    "instructions_registry.py": "5119f2919261c77566f698f2f367883aff56cf89443bac5a56b6e30ddbd5aef9",
    "instructions_util.py": "6996a6efaf6e3c92a2e926881eadc8c0262fb8d06c8cb56394ef7322f69e1930",
    "instructions.py": "a76a3b300b531b95200e53427c6b1fbb72b671cf6d358796797c0fa4b6c7c44f",
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}
# IFBench's run_eval on IFBench_test.jsonl and sample_output.jsonl (run in a
# container at 1c40f0c, 21 Oct): 300 prompts, 344 instructions
RUN_EVAL = {"prompt_strict": 84 / 300, "inst_strict": 102 / 344,
            "prompt_loose": 94 / 300, "inst_loose": 117 / 344}


def undone(text: str) -> str:
    """a vendored file as the commit holds it: the header gone, each marked
    line as it was"""
    out = []
    for line in text.split("\n"):
        if line.startswith("# 25: off: "):
            out.append(line[len("# 25: off: "):])
        elif line.startswith("# 25: "):
            continue
        elif "  # 25: was: " in line:
            out.append(line.split("  # 25: was: ", 1)[1])
        else:
            out.append(line)
    return "\n".join(out)


def test_the_vendored_files_are_the_commits_but_for_their_marked_lines():
    assert sorted(p.name for p in VENDORED.iterdir() if p.is_file()) == sorted(UPSTREAM)
    for name, want in UPSTREAM.items():
        text = (VENDORED / name).read_text(encoding="utf-8")
        assert hashlib.sha256(undone(text).encode()).hexdigest() == want, name
    # nothing but imports and what made or fetched things on import is changed
    marked = [line for p in VENDORED.glob("*.py") for line in p.read_text().split("\n")
              if "# 25: was: " in line or line.startswith("# 25: off: ")]
    assert all("import" in m or "nltk_data" in m or "NLTK_DATA" in m
               or m.endswith("download_nltk_resources()") for m in marked), marked
    note = (ROOT / "scripts" / "VENDORED.txt").read_text()
    assert "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d" in note and "Apache-2.0" in note


def rows() -> list[dict]:
    with gzip.open(FIXTURES / "IFBench_test.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def test_the_checkers_agree_with_ifbench_s_own_run_eval_on_its_sample_output():
    with gzip.open(FIXTURES / "sample_output.jsonl.gz", "rt", encoding="utf-8") as f:
        said = [json.loads(x) for x in f]
    reply = {r["prompt"]: r["response"] for r in said}
    reply.update({r["prompt"].strip(): r["response"] for r in said})
    follows = []
    for r in rows():
        kw = [{k: v for k, v in (d or {}).items() if v is not None} for d in r["kwargs"]]
        got = reply.get(r["prompt"], reply.get(r["prompt"].strip()))
        follows.append(inf.check("ifbench", r["prompt"], r["instruction_id_list"], kw, got or ""))
    n = inf.numbers(follows)
    assert (n["prompts"], n["instructions"]) == (300, 344)
    for k, want in RUN_EVAL.items():
        assert n[k]["v"] == pytest.approx(want, abs=1e-12), k
    # asked again: the same, from what was kept
    assert inf.check("ifbench", rows()[0]["prompt"], rows()[0]["instruction_id_list"],
                     [{k: v for k, v in d.items() if v is not None} for d in rows()[0]["kwargs"]],
                     reply[rows()[0]["prompt"]]) == follows[0]


# five of IFBench's new kinds of constraint, each followed by one reply and
# broken by another — invented prompts, the instructions and arguments real
CASES = [
    ("format:no_whitespace", {}, "Hello,world!", "Hello world"),
    ("format:options", {"options": "yes/no/maybe"}, "Maybe.", "Yes, and here is why."),
    ("count:word_count_range", {"min_words": 5.0, "max_words": 7.0},
     "One two three four five six.", "Too short."),
    ("format:title_case", {}, "The Quick Brown Fox Jumps", "The quick brown fox jumps"),
    ("words:alphabet", {}, "Apples bake carefully daily.", "Apples bake slowly."),
]


def item(k: int, iid: str, kw: dict) -> dict:
    return {"id": f"inv{k}", "question": f"Invented prompt {k}.", "instructions": [iid],
            "kwargs": [kw]}


def official(it: dict, reply: str) -> dict:
    """IFBench's own evaluation_lib, as its run_eval calls it"""
    from ifbench_official import evaluation_lib as E
    inp = E.InputExample(key=0, instruction_id_list=it["instructions"], prompt=it["question"],
                         kwargs=[dict(x) for x in it["kwargs"]])
    got = {it["question"]: reply}
    return {"strict": E.test_instruction_following_strict(inp, got).follow_instruction_list,
            "loose": E.test_instruction_following_loose(inp, got).follow_instruction_list}


def test_five_new_kinds_of_constraint_followed_and_broken():
    follows = []
    for k, (iid, kw, good, bad) in enumerate(CASES):
        it = item(k, iid, kw)
        need = fb.prompt(TASK, it)[1]
        assert fb.prompt(TASK, it)[0] == it["question"]         # the prompt as it is
        for reply, want in ((good, True), (bad, False)):
            sc = fb.score(TASK, reply, "stop", need)
            assert sc["ok"] is want, (iid, reply)
            assert sc["follow"]["loose"] == official(it, reply)["loose"]
            assert sc["follow"]["strict"] == official(it, reply)["strict"]
            follows.append(sc["follow"])
    # a line before the reply: strict breaks no_whitespace, loose sets the line aside
    it = item(9, "format:no_whitespace", {})
    sc = fb.score(TASK, "Sure:\nHello,world!", "stop", fb.prompt(TASK, it)[1])
    assert sc["follow"]["strict"] == [False] and sc["follow"]["loose"] == [True]
    assert sc["ok"] is True                                   # IFBench's measure is loose
    follows.append(sc["follow"])
    n = inf.numbers(follows)
    assert n["prompt_loose"]["v"] == pytest.approx(6 / 11)
    assert n["prompt_strict"]["v"] == pytest.approx(5 / 11)
    assert n["inst_loose"]["v"] == pytest.approx(6 / 11)
    assert n["prompt_loose"]["se"] == pytest.approx(
        ((6 / 11) * (5 / 11) * 11 / 10 / 11) ** 0.5)


def test_the_thinking_never_counts_and_an_answer_cut_off_follows_nothing():
    it = item(0, "format:title_case", {})
    need = fb.prompt(TASK, it)[1]
    plain = fb.score(TASK, "The Quick Brown Fox Jumps", "stop", need)
    thought = fb.score(TASK, "<think>\nlower case words, all of them\n</think>\n\n"
                             "The Quick Brown Fox Jumps", "stop", need)
    assert thought["follow"] == plain["follow"] and thought["ok"] is True
    # ran out of room: cut at the limit, or its thinking never closed
    for ans, fin in (("The Quick Brown Fox Jumps", "length"), ("<think>\nTitle Case", "stop")):
        sc = fb.score(TASK, ans, fin, need)
        assert sc["ran_out"] and sc["ok"] is False
        assert sc["follow"]["strict"] == sc["follow"]["loose"] == [False]
    # one run, said as DeviceMark's IFEval says it
    run = {"ok": True, "read": None, "follow": plain["follow"]}
    assert fb.verdict(TASK, run) == "strict: 1 of 1 instructions followed · loose: 1 of 1"


def test_importing_the_checkers_downloads_and_makes_nothing(tmp_path):
    """the image has NLTK's data; a check never reaches the network for it,
    and the package folder gets no .nltk_data of its own"""
    code = ("import nltk, sys\n"
            "def boom(*a, **k): raise SystemExit('a download was asked for')\n"
            "nltk.download = boom\n"
            "import ifbench_official\n"
            "from ifbench_official import instructions_util as U\n"
            "print(U.count_sentences('One. Two.'), len(U.nltk.word_tokenize('a b')))\n"
            "import instruction_following as inf\n"
            "print(inf.check('ifbench', 'p', ['ratio:stop_words'], [{'percentage': 50}], "
            "'the cat sat'))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                       env={**os.environ, "PYTHONPATH": os.pathsep.join(
                           [str(ROOT / "scripts"), os.environ.get("PYTHONPATH", "")])})
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split("\n")[0] == "2 2"
    assert not (VENDORED / ".nltk_data").exists()


def test_the_image_fetches_ifbench_s_nltk_data_pinned(tmp_path, monkeypatch):
    """scripts/nltk_data.py: English stopwords and the part-of-speech tagger
    beside punkt_tab, each at the nltk_data commit that last changed it, its
    size and git blob hash checked — tried on zips made here, nothing fetched"""
    import io
    import zipfile

    import nltk_data
    assert set(nltk_data.MORE) == {"stopwords", "averaged_perceptron_tagger_eng"}
    for name, p in nltk_data.MORE.items():
        assert len(p["commit"]) == len(p["blob"]) == 40 and p["size"] > 0
        assert nltk_data.url_of(name) == (
            f"https://raw.githubusercontent.com/nltk/nltk_data/{p['commit']}/packages/"
            f"{p['folder']}/{name}.zip")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("stopwords/english", "the\na\n")
    data = buf.getvalue()

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    seen = []
    monkeypatch.setattr(nltk_data.urllib.request, "urlopen",
                        lambda url, timeout=120: (seen.append(url), Reply(data))[1])
    monkeypatch.setitem(nltk_data.MORE, "stopwords", {
        **nltk_data.MORE["stopwords"], "size": len(data), "blob": nltk_data.blob_hash(data)})
    got = nltk_data.fetch_more("stopwords", tmp_path / "nd")
    assert got == tmp_path / "nd" / "corpora" / "stopwords"
    assert (got / "english").read_text() == "the\na\n" and seen == [nltk_data.url_of("stopwords")]
    monkeypatch.setitem(nltk_data.MORE, "stopwords", {**nltk_data.MORE["stopwords"],
                                                      "blob": "0" * 40})
    with pytest.raises(SystemExit, match="stopwords.zip is not the pinned one"):
        nltk_data.fetch_more("stopwords", tmp_path / "other")
    assert not (tmp_path / "other").exists()
