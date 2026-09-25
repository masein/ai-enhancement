"""12h.1: a model's IFEval, MMLU-Pro and MATH-500 run, as the harness writes it
and the board reads it — built from the saved chat answers in
generative_answers.jsonl, with no model. The harness's own MMLU-Pro and
MATH-500 numbers are written wrong on purpose (0.0), so a test sees the
board's reading win."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIX = [json.loads(line) for line in (HERE / "generative_answers.jsonl")
       .read_text(encoding="utf-8").splitlines() if line.strip()]
SHOTS = {"ifeval": 0, "mmlu_pro": 5, "hendrycks_math500": 0}


METRIC = {"ifeval": "prompt_level_strict_acc", "mmlu_pro": "exact_match",
          "hendrycks_math500": "exact_match"}


def write_task(d: Path, task: str, *, margs: str, backend: str = "vllm", ifeval: float = 0.5,
               subset: int = 0, stamp: str = "2026-09-25T00-00-00") -> None:
    """one of the three as the harness writes it, into folder `d`: its samples
    (the saved chat answers) and its results. `subset`: its MMLU-Pro was a
    seeded subset of this many (12a.5b) — the harness says so per subject,
    answering fewer than each holds"""
    metric = METRIC[task]
    d.mkdir(parents=True, exist_ok=True)
    recs = [f["rec"] for f in FIX if f["task"] == task]
    with open(d / f"samples_{task}_{stamp}.jsonl", "w", encoding="utf-8") as fh:
        for i, r in enumerate(recs):
            fh.write(json.dumps({"doc_id": i, **r}) + "\n")
    filt = "custom-extract" if task == "mmlu_pro" else "none"
    v = ifeval if task == "ifeval" else 0.0
    results = {task: {"alias": task, f"{metric},{filt}": v, f"{metric}_stderr,{filt}": 0.1}}
    if task == "ifeval":
        results[task]["inst_level_strict_acc,none"] = 0.7
    n_samples = {task: {"original": len(recs), "effective": len(recs)}}
    if task == "mmlu_pro" and subset:
        from service import config, runner
        took = runner.mmlu_pro_subset(subset)
        n_samples = {f"mmlu_pro_{s}": {"original": n, "effective": len(took[f"mmlu_pro_{s}"])}
                     for s, n in config.MMLU_PRO_SUBJECTS.items()}
    (d / f"results_{stamp}.json").write_text(json.dumps({
        "results": results, "group_subtasks": {task: []}, "n-shot": {task: SHOTS[task]},
        "n-samples": n_samples,
        "higher_is_better": {task: {metric: True}},
        "config": {"model": backend, "model_args": margs, "batch_size": "auto",
                   "apply_chat_template": True, "limit": None},
        "chat_template": "applied", "date": 1790000000.0,
        "configs": {task: {"generation_kwargs": {"max_gen_toks": 2048}}}}), encoding="utf-8")


def write_run(out_dir: Path, model: str, *, thinking: bool | None = None,
              backend: str = "vllm", ifeval: float = 0.5,
              tasks: tuple[str, ...] = tuple(SHOTS), subset: int = 0) -> Path:
    """`tasks`: which of the three it sat (12h.2: a model missing one);
    `subset`: its MMLU-Pro a seeded subset of this many (12a.5b)"""
    import generative as gen
    safe = model.replace("/", "__") + ("__thinking" if thinking else "")
    mdir = out_dir / safe
    margs = f"pretrained={model},dtype=bfloat16" + (
        f",enable_thinking={thinking}" if thinking is not None else "")
    for task in METRIC:
        if task in tasks:
            write_task(mdir / f"{task}_{SHOTS[task]}shot" / "run", task, margs=margs,
                       backend=backend, ifeval=ifeval, subset=subset)
    if not (mdir / "model_meta.json").exists():
        (mdir / "model_meta.json").write_text(json.dumps(
            {"model": model + (" · thinking" if thinking else ""), "kind": "instruct"}),
            encoding="utf-8")
    gen.write(mdir, gen.mark(mdir, extra={
        "thinking": {"mode": "switch", "on": bool(thinking), "budget": 8192 if thinking else 2048},
        "backend": backend}))
    return mdir
