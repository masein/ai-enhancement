"""The models the board suggests, and what it knows about each before any run.

12h.1 adds nine small instruct models that can sit IFEval, MMLU-Pro and
MATH-500. Their notes are from each model card and each repo's files, read on
2026-09-25; the real load happens on the server (scripts/trial_generative.py),
and what it shows wins over anything written here.

Thinking. Several of these can think before answering, and it moves the score
a lot (Qwen3.5-2B publishes IFEval 61.2 without and 78.6 with it). Each
model's chat template says how:

  switch   the template reads `enable_thinking`; the board passes it False by
           default (a phone assistant answers straight away) and True for a
           "Think before answering" run, which is a row of its own
  always   the model thinks whatever it is told; its row says "thinking"
  never    the template has no thinking at all

Own code. A repo that ships its own model code runs it only from the approved
list (service/approved_code.json): repo plus pinned commit. transformers 5.5.3
has all nine architectures built in, so none of them needs its own code today.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
APPROVED_PATH = HERE / "approved_code.json"
# 12q.D: the models that can't run here, and why — refused before a run starts
CANT_RUN_PATH = HERE / "cant_run_here.json"

# name, size, how it thinks, where its thinking ends, and what to watch when
# it loads. Hugging Face names checked on 2026-09-25. 15.7: "opens" — its chat
# template opens the thinking itself when it's on ('<think>\n' after the
# assistant's turn: Qwen3.5 and Nemotron, read on 2 Oct), so a reply carries
# no opening marker; Youtu and Nanbeige write <think> themselves, Gemma 4 its
# <|channel>
MODELS = [
    {"id": "Qwen/Qwen3.5-0.8B", "params": 0.8e9, "thinking": "switch", "default_on": False,
     "opens": True,
     "watch": "the card loads it as multimodal (Qwen3_5ForConditionalGeneration); needs "
              "transformers 5.5+"},
    {"id": "Qwen/Qwen3.5-2B", "params": 2e9, "thinking": "switch", "default_on": False,
     "opens": True,
     "watch": "multimodal on the card; needs transformers 5.5+"},
    {"id": "Qwen/Qwen3.5-4B", "params": 4e9, "thinking": "switch", "default_on": False,
     "opens": True,
     "watch": "multimodal on the card; needs transformers 5.5+"},
    {"id": "LiquidAI/LFM2.5-1.2B-Instruct", "params": 1.2e9, "thinking": "never",
     "watch": "convolution + attention hybrid (lfm2); LFM 1.0 licence"},
    {"id": "ibm-granite/granite-4.0-h-1b", "params": 1.5e9, "thinking": "never",
     "watch": "Mamba2 hybrid; slow without mamba_ssm and causal-conv1d in the image"},
    {"id": "google/gemma-4-E2B-it", "params": 2e9, "thinking": "switch", "default_on": False,
     "think_end": "<channel|>",
     "watch": "multimodal (Gemma4ForConditionalGeneration); needs transformers 5.5+; thinks "
              "inside <|channel>…<channel|>"},
    {"id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "params": 4e9, "thinking": "switch",
     "default_on": True, "opens": True,
     "watch": "Mamba2 hybrid; the repo ships its own code, and transformers 5.5.3 has "
              "nemotron_h built in, so it loads without it; thinks unless told not to"},
    {"id": "tencent/Youtu-LLM-2B", "params": 2e9, "thinking": "switch", "default_on": True,
     "watch": "the model card asks for transformers 4.56–4.57.1, but the repo's config was "
              "saved with 5.0.0.dev0, ships no code, and transformers 5.5.3 has youtu built "
              "in; thinks unless told not to"},
    {"id": "Nanbeige/Nanbeige4.1-3B", "params": 3e9, "thinking": "always",
     "watch": "a Llama checkpoint: the standard loader; a reasoning model with no off switch; "
              "slow tokenizer"},
]
_BY_ID = {m["id"]: m for m in MODELS}


def known(hf_id: str) -> dict | None:
    return _BY_ID.get(hf_id)


def thinking_of(hf_id: str, template: str | None) -> dict:
    """{mode, default_on, think_end}: how this model thinks. The catalogue
    first; for any other model, its chat template — a template that reads
    `enable_thinking` has a switch (and says in it which way it defaults),
    anything else is taken not to think. A model that thinks with no switch
    and is not in the catalogue is found by its first run, as #60 found
    Qwen3."""
    k = known(hf_id)
    if k:
        mode = k["thinking"]
        return {"mode": mode, "default_on": mode == "always" or bool(k.get("default_on")),
                "think_end": k.get("think_end", "</think>"), "opens": bool(k.get("opens"))}
    t = template or ""
    if "enable_thinking" in t:
        # Qwen3 and Youtu: `enable_thinking is defined and enable_thinking is
        # false` turns it off, so it is on unless told; Qwen3.5 and Gemma 4
        # read `is true` or `| default(false)`
        on = ("enable_thinking is false" in t.replace("  ", " ")
              or "else True" in t)
        return {"mode": "switch", "default_on": on,
                "think_end": "<channel|>" if "<|channel>" in t else "</think>",
                "opens": opens_thinking(t)}
    return {"mode": "never", "default_on": False, "think_end": "</think>", "opens": False}


def opens_thinking(template: str | None) -> bool:
    """15.7: does this chat template open the thinking itself when it's on —
    the prompt it makes ending in <think> — so the reply carries no opening
    marker? Rendered as transformers renders it (jinja2, sandboxed); False
    where it can't be"""
    if not template:
        return False
    try:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def fail(msg):
            raise ValueError(msg)
        env.globals["raise_exception"] = fail
        prompt = env.from_string(template).render(
            messages=[{"role": "user", "content": "x"}], add_generation_prompt=True,
            enable_thinking=True, bos_token="", eos_token="", tools=None)
    except Exception:                               # noqa: BLE001 — unknown: read as not
        return False
    return prompt.rstrip().endswith("<think>")


def approved() -> dict[str, dict]:
    """repo -> {commit, why, added}: the repos whose own model code may run,
    each at one commit only. Read from service/approved_code.json, which is in
    the repo and changes only through a reviewed PR."""
    try:
        data = json.loads(APPROVED_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict) and v.get("commit")}


def cant_run_here(hf_id: str) -> str:
    """12q.D: why a model can't run on this server ('' when it can), from
    service/cant_run_here.json — a reviewed list, as the approved one is"""
    try:
        entry = json.loads(CANT_RUN_PATH.read_text(encoding="utf-8")).get(hf_id)
    except (OSError, ValueError):
        return ""
    return str((entry or {}).get("why") or "")


def approved_code(hf_id: str, revision: str | None = None) -> tuple[bool, str, str | None]:
    """(allowed, why not, commit to run). A repo not on the list is refused;
    a repo on it runs its own code at the listed commit, and any other commit
    is refused."""
    entry = approved().get(hf_id)
    if not entry:
        return False, (f"{hf_id} requires trust_remote_code=True (custom modeling code in the "
                       f"repo). Code from the Hub is never executed here, whatever the server "
                       f"settings. If this is your model, upload it as an artifact "
                       f"(POST /api/artifacts/<name>) and submit it with "
                       f"allow_remote_code=true."), None
    commit = entry["commit"]
    if revision and revision != commit and not (len(revision) >= 7
                                                and commit.startswith(revision)):
        return False, (f"{hf_id} may run its own code only at commit {commit[:12]}, the one "
                       f"on the approved list (service/approved_code.json); {revision[:12]} "
                       f"is not it"), None
    return True, "", commit
