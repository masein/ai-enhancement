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


def write_run(out_dir: Path, model: str, *, thinking: bool | None = None,
              backend: str = "vllm", ifeval: float = 0.5) -> Path:
    import generative as gen
    safe = model.replace("/", "__") + ("__thinking" if thinking else "")
    mdir = out_dir / safe
    margs = f"pretrained={model},dtype=bfloat16" + (
        f",enable_thinking={thinking}" if thinking is not None else "")
    for task, metric in (("ifeval", "prompt_level_strict_acc"), ("mmlu_pro", "exact_match"),
                         ("hendrycks_math500", "exact_match")):
        d = mdir / f"{task}_{SHOTS[task]}shot" / "run"
        d.mkdir(parents=True, exist_ok=True)
        recs = [f["rec"] for f in FIX if f["task"] == task]
        with open(d / f"samples_{task}_2026-09-25T00-00-00.jsonl", "w", encoding="utf-8") as fh:
            for i, r in enumerate(recs):
                fh.write(json.dumps({"doc_id": i, **r}) + "\n")
        filt = "custom-extract" if task == "mmlu_pro" else "none"
        v = ifeval if task == "ifeval" else 0.0
        results = {task: {"alias": task, f"{metric},{filt}": v, f"{metric}_stderr,{filt}": 0.1}}
        if task == "ifeval":
            results[task]["inst_level_strict_acc,none"] = 0.7
        (d / "results_2026-09-25T00-00-00.json").write_text(json.dumps({
            "results": results, "group_subtasks": {task: []}, "n-shot": {task: SHOTS[task]},
            "n-samples": {task: {"original": len(recs), "effective": len(recs)}},
            "higher_is_better": {task: {metric: True}},
            "config": {"model": backend, "model_args": margs, "batch_size": "auto",
                       "apply_chat_template": True, "limit": None},
            "chat_template": "applied", "date": 1790000000.0,
            "configs": {task: {"generation_kwargs": {"max_gen_toks": 2048}}}}), encoding="utf-8")
    if not (mdir / "model_meta.json").exists():
        (mdir / "model_meta.json").write_text(json.dumps(
            {"model": model + (" · thinking" if thinking else ""), "kind": "instruct"}),
            encoding="utf-8")
    gen.write(mdir, gen.mark(mdir, extra={
        "thinking": {"mode": "switch", "on": bool(thinking), "budget": 8192 if thinking else 2048},
        "backend": backend}))
    return mdir
