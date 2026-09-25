"""12h.1: IFEval, MMLU-Pro and MATH-500, and nine small instruct models.

No model runs here, or anywhere but the server: every answer below is a saved
chat answer (tests/fixtures/generative_answers.jsonl), with what the board
must read out of it and the verdict it must get. The rest is what can be
tested without weights: the approved list for models that ship their own
code, the thinking switch, the command a run builds, a thinking run as a row
of its own, and a base model kept out of these columns and every average."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import generative as gen
import report_lm_eval as report
from generative_fixture import write_run
from service import catalog, config, hfmeta, runner

REPO = Path(__file__).resolve().parents[1]
FIX = [json.loads(line) for line in
       (REPO / "tests" / "fixtures" / "generative_answers.jsonl").read_text(encoding="utf-8")
       .splitlines() if line.strip()]
NEMOTRON = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
PINNED = "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"


# ---------------------------------------------------------------------------
# reading chat answers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("i", range(len(FIX)), ids=[f"{f['task']}#{n}" for n, f in enumerate(FIX)])
def test_every_chat_answer_yields_its_answer_and_verdict(i):
    f = FIX[i]
    got = gen.read(f["task"], f["rec"])
    assert (got["extracted"], got["correct"], got["ran_out"]) == \
        (f["extracted"], f["correct"], f["ran_out"]), f["note"]


def test_the_fixtures_cover_every_benchmark_and_the_awkward_cases():
    by = {t: [f for f in FIX if f["task"] == t] for t in gen.TASKS}
    assert all(len(v) >= 4 for v in by.values())
    answers = " ".join(f["rec"]["resps"][0][0] for f in FIX)
    for case in ("The answer is (C).", "**C**", "I think it's C because", "\\boxed{\\frac{1}{2}}",
                 "0.5", "<think>"):
        assert case in answers, case
    assert all(any(f["ran_out"] for f in v) for v in by.values())


def test_maths_answers_are_compared_as_maths():
    pytest.importorskip("math_verify")        # in the image and the local check
    assert gen.math_equal("0.5", "\\frac{1}{2}")
    assert gen.math_equal("\\dfrac{3}{4}", "\\frac{3}{4}")
    assert gen.math_equal("(3, -2)", "(3,-2)")
    assert not gen.math_equal("3", "5")
    assert not gen.math_equal(None, "5") and not gen.math_equal("5", None)
    # the harness's own comparison is a string one: these differ as strings
    assert "0.5" != "\\frac{1}{2}"


def test_without_math_verify_the_fallback_still_reads_numbers():
    assert gen._plain_equal("0.5", "\\frac{1}{2}")
    assert gen._plain_equal("\\sqrt{2}", "\\sqrt{2}")
    assert not gen._plain_equal("2", "3")


def test_the_letter_rules():
    opts = [f"option {i}" for i in range(10)]
    assert gen.extract_letter("A bold claim, but the answer is (B).", opts) == "B"
    assert gen.extract_letter("Answer: **J**", opts) == "J"
    assert gen.extract_letter("Answer: J", opts[:4]) is None      # only four options here
    assert gen.extract_letter("the answer is (c)", opts) is None   # a letter is a capital
    assert gen.extract_letter("I'd go with (D) here.", opts) == "D"
    assert gen.last_boxed("so \\boxed{\\frac{1}{\\sqrt{2}}} is it") == "\\frac{1}{\\sqrt{2}}"
    assert gen.extract_math("x = 3, y = 4, so the answer is 7 because 3 + 4 = 7") == "7"


# ---------------------------------------------------------------------------
# a run, scored, and the board's rows
# ---------------------------------------------------------------------------

def payload(out_dir: Path) -> dict:
    return report.build_payload(report.merge_runs(report.load_results(out_dir)), "t",
                                source=str(out_dir))


def test_our_reading_is_the_score_and_the_harness_one_is_not(tmp_path):
    write_run(tmp_path, "org/chat-2b", thinking=False)
    p = payload(tmp_path)
    m = next(x for x in p["models"] if x["id"] == "org/chat-2b")
    ours = {t: gen.score(t, [f["rec"] for f in FIX if f["task"] == t]) for t in gen.TASKS}
    # MMLU-Pro and MATH-500 are ours (the harness said 0.0 for both)
    for t in ("mmlu_pro", "hendrycks_math500"):
        assert p["cells"][t]["org/chat-2b"]["v"] == pytest.approx(ours[t]["acc"])
        assert ours[t]["acc"] > 0.5
    # IFEval is the harness's own checker: its prompt-level strict score
    assert p["cells"]["ifeval"]["org/chat-2b"]["v"] == 0.5
    assert set(gen.TASKS) <= set(p["accTasks"]) and p["genTasks"] == list(gen.TASKS)
    # the answers that ran out of room, counted per task
    assert all(m["gen"]["tasks"][t]["ran_out"] == 1 for t in gen.TASKS)
    assert m["gen"]["backend"] == "vllm"
    # never in any average: not required, not in the diagnostic one either
    assert not set(gen.TASKS) & set(p["required"])
    assert m["partialAvg"] is None and m["avg"] is None


def test_thinking_on_is_a_row_of_its_own_never_averaged_with_off(tmp_path):
    write_run(tmp_path, "org/chat-2b", thinking=False)
    write_run(tmp_path, "org/chat-2b", thinking=True)
    p = payload(tmp_path)
    ids = sorted(m["id"] for m in p["models"])
    assert ids == ["org/chat-2b", "org/chat-2b · thinking"]
    think = next(m for m in p["models"] if m["id"] == "org/chat-2b · thinking")
    assert think["thinkingRow"] is True and think["name"] == "chat-2b · thinking"
    assert think["gen"]["thinking"]["on"] is True
    # each row has its own three cells; nothing merged, nothing averaged
    for t in gen.TASKS:
        assert set(p["cells"][t]) == {"org/chat-2b", "org/chat-2b · thinking"}


def test_a_score_far_below_the_published_one_is_flagged():
    assert report.far_below("Qwen/Qwen3.5-2B", "ifeval", 0.40) == \
        {"published": 61.2, "note": "no thinking", "gap": 21.2}
    assert report.far_below("Qwen/Qwen3.5-2B", "ifeval", 0.50) is None      # 11.2 under
    assert report.far_below("Qwen/Qwen3.5-2B", "hendrycks_math500", 0.0) is None   # nothing published
    assert report.far_below("org/unknown", "ifeval", 0.0) is None


def test_the_published_table_is_the_briefs():
    pub = report._PUBLISHED
    assert pub["Qwen/Qwen3.5-2B"] == {"ifeval": (61.2, "no thinking"),
                                      "mmlu_pro": (55.3, "no thinking")}
    assert pub["tencent/Youtu-LLM-2B"]["hendrycks_math500"] == (93.7, "")
    assert pub["google/gemma-4-E2B-it"] == {"mmlu_pro": (60.0, "")}
    assert set(pub) <= {m["id"] for m in catalog.MODELS}


# ---------------------------------------------------------------------------
# thinking, the command, the subset
# ---------------------------------------------------------------------------

def meta_for(mode, default_on=False, think_end="</think>"):
    return {"archinfo": {"thinking": mode, "thinking_default_on": default_on,
                         "think_end": think_end}}


def test_thinking_is_off_by_default_and_said_out_loud():
    off = runner.gen_thinking({}, meta_for("switch", default_on=True))
    assert off == {"mode": "switch", "on": False, "separate": False,
                   "budget": 2048, "think_end": "</think>"}
    args = runner.gen_model_args("Qwen/Qwen3-1.7B", off, backend="hf")
    assert args == "pretrained=Qwen/Qwen3-1.7B,dtype=bfloat16,enable_thinking=False"
    on = runner.gen_thinking({"thinking": True}, meta_for("switch"))
    assert on["on"] and on["separate"] and on["budget"] == 8192
    assert "enable_thinking=True,think_end_token=</think>" in \
        runner.gen_model_args("x/y", on, backend="hf")
    # a model that cannot stop thinks, on its own row; one that never thinks is asked plainly
    always = runner.gen_thinking({}, meta_for("always"))
    assert always["on"] and not always["separate"]
    assert runner.gen_model_args("x/y", always, backend="hf") == \
        "pretrained=x/y,dtype=bfloat16,think_end_token=</think>"
    never = runner.gen_thinking({"thinking": True}, meta_for("never"))
    assert not never["on"] and runner.gen_model_args("x/y", never, backend="hf") == \
        "pretrained=x/y,dtype=bfloat16"


def test_the_nine_and_their_thinking_switches():
    ids = [m["id"] for m in catalog.MODELS]
    assert ids == ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B",
                   "LiquidAI/LFM2.5-1.2B-Instruct", "ibm-granite/granite-4.0-h-1b",
                   "google/gemma-4-E2B-it", NEMOTRON, "tencent/Youtu-LLM-2B",
                   "Nanbeige/Nanbeige4.1-3B"]
    modes = {m["id"]: catalog.thinking_of(m["id"], None) for m in catalog.MODELS}
    assert modes["Qwen/Qwen3.5-2B"] == {"mode": "switch", "default_on": False,
                                        "think_end": "</think>"}
    assert modes[NEMOTRON]["default_on"] is True               # turn it off
    assert modes["Nanbeige/Nanbeige4.1-3B"]["mode"] == "always"
    assert modes["LiquidAI/LFM2.5-1.2B-Instruct"]["mode"] == "never"
    assert modes["google/gemma-4-E2B-it"]["think_end"] == "<channel|>"
    # any other model: its template decides
    qwen3 = "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>"
    assert catalog.thinking_of("Qwen/Qwen3-1.7B", qwen3) == \
        {"mode": "switch", "default_on": True, "think_end": "</think>"}
    qwen35 = "{%- if enable_thinking is defined and enable_thinking is true %}<think>\n"
    assert catalog.thinking_of("org/other", qwen35)["default_on"] is False
    assert catalog.thinking_of("org/plain", "{{ messages }}")["mode"] == "never"


def test_the_command_on_vllm_and_on_hf(tmp_path):
    th = runner.gen_thinking({}, meta_for("switch"))
    args = runner.gen_model_args("Qwen/Qwen3.5-2B", th, backend="vllm", gpu_util=0.42)
    assert args.endswith("gpu_memory_utilization=0.42,max_model_len=8192")
    v = runner.lm_eval_cmd(args, "mmlu_pro", 5, 8, tmp_path, chat=True, max_gen_toks=2048,
                           backend="vllm", samples=tmp_path / "subset.json")
    assert v[v.index("--model") + 1] == "vllm" and "--device" not in v
    assert v[v.index("--batch_size") + 1] == "auto"
    assert v[v.index("--samples") + 1] == str(tmp_path / "subset.json")
    assert v[v.index("--gen_kwargs") + 1] == "max_gen_toks=2048" and "--apply_chat_template" in v
    h = runner.lm_eval_cmd("pretrained=x/y", "ifeval", 0, 8, tmp_path, chat=True, backend="hf")
    assert h[h.index("--device") + 1] == "cuda:0" and "--samples" not in h
    assert runner.gpu_util_for(20000, 32000) == 0.53 and runner.gpu_util_for(40000, 32000) == 0.9


def test_a_subset_is_seeded_fixed_and_in_proportion():
    a, b = runner.mmlu_pro_subset(1200), runner.mmlu_pro_subset(1200)
    assert a == b and sum(len(v) for v in a.values()) == 1200
    assert abs(len(a["mmlu_pro_math"]) - 1200 * 1351 / 12032) < 1
    assert all(len(set(v)) == len(v) and max(v) < config.MMLU_PRO_SUBJECTS[k[9:]]
               for k, v in a.items() if v)
    assert runner.mmlu_pro_subset(1200, seed=7) != a
    assert sum(config.MMLU_PRO_SUBJECTS.values()) == 12032


def test_the_three_are_a_suite_and_in_deploy_step_4():
    assert config.tasks_for_suite("generative") == ["ifeval", "mmlu_pro", "hendrycks_math500"]
    assert "generative" in config.SUITES            # check_tasks.py walks config.SUITES
    assert config.NFEWSHOT["mmlu_pro"] == 5 and config.NFEWSHOT["ifeval"] == 0


# ---------------------------------------------------------------------------
# models that ship their own code
# ---------------------------------------------------------------------------

def test_the_approved_list_refuses_another_commit_and_an_unlisted_repo():
    assert catalog.approved_code(NEMOTRON) == (True, "", PINNED)
    assert catalog.approved_code(NEMOTRON, PINNED[:12])[0] is True
    ok, why, commit = catalog.approved_code(NEMOTRON, "0123456789abcdef")
    assert not ok and commit is None and "only at commit dfaf35de3e30" in why
    ok, why, _ = catalog.approved_code("someone/custom-model")
    assert not ok and "Code from the Hub is never executed here" in why
    # the list is in the repo, reviewed: one entry today
    assert list(catalog.approved()) == [NEMOTRON]


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """huggingface_hub, as far as preflight asks it: a repo whose config
    ships an auto_map, and a chat template"""
    files = {}

    class Err(Exception):
        pass
    errors = types.SimpleNamespace(EntryNotFoundError=Err, GatedRepoError=Err,
                                   RepositoryNotFoundError=Err)

    def download(repo, name, revision=None):
        key = (repo, name, revision) if (repo, name, revision) in files else (repo, name, None)
        if key not in files:
            raise Err(name)
        p = tmp_path / f"{abs(hash(key))}-{name}"
        p.write_text(files[key], encoding="utf-8")
        return str(p)

    class Api:
        def model_info(self, repo):
            return types.SimpleNamespace(safetensors=types.SimpleNamespace(total=4e9),
                                         siblings=[types.SimpleNamespace(rfilename=n)
                                                   for (r, n, _) in files if r == repo])
    mod = types.ModuleType("huggingface_hub")
    mod.HfApi, mod.hf_hub_download = Api, download
    mod.errors = errors
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    monkeypatch.delenv("STUB_PREFLIGHT", raising=False)

    def repo(hf_id, model_type, auto_map=True, revision=None):
        cfg = {"model_type": model_type, "vocab_size": 131072,
               **({"auto_map": {"AutoModelForCausalLM": "modeling.X"}} if auto_map else {})}
        files[(hf_id, "config.json", revision)] = json.dumps(cfg)
        files[(hf_id, "tokenizer_config.json", None)] = json.dumps(
            {"chat_template": "{{ enable_thinking if enable_thinking is defined else True }}"})
    return repo


def test_own_code_runs_only_from_the_approved_list(hub, monkeypatch):
    # built into transformers: its own class loads it, the repo's code never runs
    hub(NEMOTRON, "nemotron_h")
    monkeypatch.setattr(hfmeta, "native_architecture", lambda t: t == "nemotron_h")
    meta = hfmeta.preflight(NEMOTRON, "instruct")
    assert meta["remote_code"] is False and meta["archinfo"]["own_code"] == "built in"
    assert meta["archinfo"]["thinking"] == "switch"
    # not built in, not on the list: refused, as always
    hub("someone/custom", "custom_arch")
    monkeypatch.setattr(hfmeta, "native_architecture", lambda t: False)
    with pytest.raises(hfmeta.PreflightError, match="Code from the Hub is never executed here"):
        hfmeta.preflight("someone/custom", "instruct")
    # on the list, but the server runs no model code: refused, saying so
    monkeypatch.setattr(config, "ALLOW_REMOTE_CODE", False)
    with pytest.raises(hfmeta.PreflightError, match="on the approved list at commit dfaf35de3e30"):
        hfmeta.preflight(NEMOTRON, "instruct")
    # on the list and allowed: its own code, at the pinned commit only
    monkeypatch.setattr(config, "remote_code_blocked", lambda: "")
    hub(NEMOTRON, "nemotron_h", revision=PINNED)
    meta = hfmeta.preflight(NEMOTRON, "instruct")
    assert meta["remote_code"] is True and meta["revision"] == PINNED
    assert meta["archinfo"]["own_code"] == "approved"


# ---------------------------------------------------------------------------
# the submit form's words, and the queue
# ---------------------------------------------------------------------------

@pytest.fixture
def svc(tmp_path, monkeypatch):
    from conftest import make_service
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client
    client.__exit__(None, None, None)


def test_the_queue_takes_thinking_and_a_subset_for_the_three_only(svc):
    from service import db
    r = svc.post("/api/submissions", json={"hf_id": "Qwen/Qwen3.5-2B", "suite": "generative"})
    assert r.status_code == 200
    on = svc.post("/api/submissions", json={"hf_id": "Qwen/Qwen3.5-2B", "suite": "generative",
                                            "thinking": True})
    assert on.status_code == 200 and on.json()["id"] != r.json()["id"]   # another row
    row = db.get(on.json()["id"])
    assert row["thinking"] == 1 and row["subset"] == 0
    sub = svc.post("/api/submissions", json={"hf_id": "Qwen/Qwen3.5-4B", "suite": "generative",
                                             "subset": 1200})
    assert db.get(sub.json()["id"])["subset"] == 1200
    for body, want in (({"suite": "full", "thinking": True}, "suite generative"),
                       ({"suite": "generative", "subset": 20000}, "from 1 to 12031"),
                       ({"suite": "generative", "kind": "base"}, "only an instruct model")):
        bad = svc.post("/api/submissions", json={"hf_id": "Qwen/Qwen3.5-2B", **body})
        assert bad.status_code == 422 and want in bad.json()["detail"], body


def test_the_nine_are_suggested_with_their_thinking(svc):
    items = svc.get("/api/models/suggest", params={"q": "qwen3.5"}).json()["items"]
    hit = next(i for i in items if i["id"] == "Qwen/Qwen3.5-2B")
    assert hit["kind"] == "instruct" and hit["thinking"] == "switch"
    nan = svc.get("/api/models/suggest", params={"q": "nanbeige"}).json()["items"]
    assert nan[0]["id"] == "Nanbeige/Nanbeige4.1-3B" and nan[0]["thinking"] == "always"


# ---------------------------------------------------------------------------
# scripts/trial_generative.py, on a fake model
# ---------------------------------------------------------------------------

def test_the_trial_has_help():
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "trial_generative.py"), "--help"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "--model" in out.stdout and "--limit" in out.stdout


def test_the_trial_prints_each_item_and_the_estimate(monkeypatch, capsys):
    import trial_generative as trial

    def fake_harness(cmd, cwd, log):
        """a model that answers with the saved chat answers, whatever it is asked"""
        task = cmd[cmd.index("--tasks") + 1]
        d = Path(cwd) / "fake"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"samples_{task}_2026-09-25T00-00-00.jsonl", "w", encoding="utf-8") as fh:
            for i, f in enumerate(x for x in FIX if x["task"] == task):
                fh.write(json.dumps({"doc_id": i, **f["rec"]}) + "\n")
        return 0
    monkeypatch.setattr(trial, "run_harness", fake_harness)
    monkeypatch.setattr("service.hfmeta.preflight", lambda *a, **k: {
        "kind": "instruct", "batch": 8, "remote_code": False, **meta_for("switch", True)})
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)
    monkeypatch.setattr(runner, "gen_backend", lambda: ("hf", "vLLM is not installed in this image"))
    t = iter(range(0, 10_000, 26))
    monkeypatch.setattr(trial.time, "time", lambda: next(t))
    assert trial.main(["--model", "Qwen/Qwen3-1.7B", "--limit", "4"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == ("Qwen/Qwen3-1.7B · thinking off (switch) · 2048 tokens per "
                                   "answer · on hf (vLLM is not installed in this image)")
    assert "MMLU-Pro — 13 items in 26 s, 2.00 s per item" in out
    assert "    1  ✓  read C  ·  answer C" in out
    assert "wrote: The answer is (C)." in out
    assert "(ran out of room)" in out and "(nothing to read)" in out
    assert "MMLU-Pro: 10 of 13 right · 1 unreadable · 1 ran out of room" in out
    assert "IFEval: 2 of 4 right · 0 unreadable · 1 ran out of room" in out
    assert "A full MMLU-Pro run at this rate: about 6 h 41 min for 12,032" in out
    assert "Over 3 hours: masein decides" in out
    assert out.rstrip().endswith("trial OK: every benchmark answered and read")


def test_the_estimate_under_three_hours_leaves_nothing_to_decide(capsys):
    import trial_generative as trial
    assert trial.duration(9600) == "about 2 h 40 min"
    trial.estimate(0.5)
    out = capsys.readouterr().out
    assert "about 1 h 40 min for 12,032" in out and "masein decides" not in out


# ---------------------------------------------------------------------------
# scripts/trial_standard.py: the Standard tasks again, beside the board
# ---------------------------------------------------------------------------

SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def standard_results(d: Path, task: str, v: float, se: float) -> None:
    """a results file as lm_eval writes one, for one Standard task"""
    d.mkdir(parents=True, exist_ok=True)
    (d / "results_2026-09-25T00-00-00.json").write_text(json.dumps({
        "results": {task: {"alias": task, "acc,none": v - 0.02, "acc_stderr,none": se,
                           "acc_norm,none": v, "acc_norm_stderr,none": se}},
        "group_subtasks": {task: []}, "n-shot": {task: config.NFEWSHOT.get(task, 0)},
        "n-samples": {task: {"original": 100, "effective": 100}},
        "higher_is_better": {task: {"acc": True, "acc_norm": True}},
        "config": {"model": "hf", "model_args": f"pretrained={SMOL},dtype=bfloat16",
                   "batch_size": "8", "limit": None},
        "date": 1790000000.0}), encoding="utf-8")


def test_the_standard_trial_has_help():
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "trial_standard.py"), "--help"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "--model" in out.stdout and "--tasks" in out.stdout


def test_the_standard_trial_puts_each_task_beside_the_board(tmp_path, monkeypatch, capsys):
    import trial_standard as trial
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "full")
    board = tmp_path / "full" / SMOL.replace("/", "__")
    standard_results(board / "hellaswag_10shot" / "run", "hellaswag", 0.4100, 0.0050)
    standard_results(board / "arc_easy_0shot" / "run", "arc_easy", 0.6000, 0.0100)
    now = {"hellaswag": (0.4200, 0.0050), "arc_easy": (0.5500, 0.0100)}
    seen = []

    def fake_harness(cmd, cwd, log):
        """the service's own command, answered by numbers set here"""
        seen.append(cmd)
        task = cmd[cmd.index("--tasks") + 1]
        v, se = now[task]
        standard_results(Path(cmd[cmd.index("--output_path") + 1]) / "run", task, v, se)
        return 0
    monkeypatch.setattr(trial, "run_harness", fake_harness)
    monkeypatch.setattr("service.hfmeta.preflight", lambda *a, **k: {
        "kind": "instruct", "batch": 8, "remote_code": False})
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)
    assert trial.main(["--model", SMOL]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith(f"{SMOL} · instruct · transformers ")
    # the quick suite, each task as a run asks it: its shots, the chat template, whole
    assert [c[c.index("--tasks") + 1] for c in seen] == ["hellaswag", "arc_easy"]
    assert all("--apply_chat_template" in c and "--limit" not in c for c in seen)
    assert seen[0][seen[0].index("--num_fewshot") + 1] == str(config.NFEWSHOT["hellaswag"])
    assert out[1] == ("hellaswag · acc_norm 0.4200 ± 0.0050 · board 0.4100 ± 0.0050 · "
                      "+0.0100, inside the noise")
    assert out[2] == ("arc_easy · acc_norm 0.5500 ± 0.0100 · board 0.6000 ± 0.0100 · "
                      "-0.0500, 3.5 standard errors — a real difference")
    assert out[-1] == "trial OK: every task ran"


def test_the_standard_trial_says_when_a_task_fails_or_is_not_on_the_board(
        tmp_path, monkeypatch, capsys):
    import trial_standard as trial
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "full")

    def fake_harness(cmd, cwd, log):
        task = cmd[cmd.index("--tasks") + 1]
        if task == "arc_easy":
            Path(log).write_text("Traceback …\nValueError: the model would not load\n")
            return 1
        standard_results(Path(cmd[cmd.index("--output_path") + 1]) / "run", task, 0.42, 0.005)
        return 0
    monkeypatch.setattr(trial, "run_harness", fake_harness)
    monkeypatch.setattr("service.hfmeta.preflight", lambda *a, **k: {
        "kind": "base", "batch": 8, "remote_code": False})
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)
    assert trial.main(["--model", SMOL]) == 1
    out = capsys.readouterr().out
    assert "hellaswag · acc_norm 0.4200 ± 0.0050 · not on the board for this model" in out
    assert "arc_easy: lm_eval exited 1. The end of its output:" in out
    assert "    | ValueError: the model would not load" in out
    assert out.rstrip().endswith("trial FAILED: arc_easy")
