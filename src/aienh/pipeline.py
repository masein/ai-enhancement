"""
The pipeline — every stage wired together, one command.

This is the shape of the thing you said you'd be building: a repeatable path from
raw data to a scored, versioned, comparable model, with a dashboard at the end.

    data -> pretrain (dense)  ->|
    data -> pretrain (MoE)    ->| eval -> registry -> dashboard
                    |
                    +-> SFT -> eval
                          |
                          +-> GRPO -> eval
                          +-> distill (teacher = SFT model) -> eval

Four properties make this a pipeline rather than a pile of scripts, and they are
the ones worth defending in review:

  1. Every stage's output is a checkpoint with its config and tokenizer INSIDE it.
     No stage depends on ambient state or on remembering which flags you passed.
  2. Every stage is evaluated by the SAME suite, at the same seed, with the
     template the model was trained for — and the suite is hashed into the result.
  3. Every result is appended to an append-only registry with its parent, so
     lineage is reconstructable months later.
  4. Every stage is skippable and resumable, because you will re-run one stage
     forty times and the others twice.

Scale presets: `smoke` (a couple of minutes on a laptop CPU, for CI), `small`
(the default, ~15-30 min on a MacBook), `full` (a real if tiny run). The point of
`smoke` is that CI runs the WHOLE pipeline on every commit — a pipeline that is
only ever exercised at full scale is a pipeline that is broken half the time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .dashboard import build_dashboard
from .distill import DistillConfig, distill, inspect_soft_labels
from .evaluate import CHAT_TEMPLATE, RAW_TEMPLATE, DEFAULT_SUITE, run_suite
from .grpo import GRPOConfig, grpo
from .model import ModelConfig
from .registry import from_train_result, leaderboard, record, render_table
from .sft import SFTConfig, sft
from .train import TrainConfig, train
from .utils import git_sha, pick_device

SCALES = {
    "smoke": dict(
        model=dict(n_layer=2, n_head=4, n_embd=96, block_size=64),
        train=dict(steps=120, micro_batch_size=32, n_docs=6000, block_size=64),
        sft=dict(n_examples=1200, epochs=4.0),
        grpo=dict(iterations=8, prompts_per_iter=6, group_size=6, eval_every=4, eval_n=50),
        distill=dict(steps=80),
        eval=dict(n=100),
    ),
    "small": dict(
        model=dict(n_layer=4, n_head=6, n_embd=192, block_size=64),
        train=dict(steps=800, micro_batch_size=48, n_docs=20000, block_size=64),
        # 8 epochs, not 12, on purpose: it lands the SFT model at ~36% greedy
        # exact-match with pass@8 ~82%, i.e. a large pass@1->pass@k gap. That is
        # the regime where RL has something to reinforce (doc 07). Raise to 12
        # and the model saturates at ~81% with no headroom, and the GRPO stage
        # can only add noise — which is itself worth running once to see.
        sft=dict(n_examples=4000, epochs=8.0),
        grpo=dict(iterations=150, prompts_per_iter=8, group_size=8, eval_every=25),
        distill=dict(steps=300),
        eval=dict(n=200),
    ),
    "full": dict(
        model=dict(n_layer=6, n_head=8, n_embd=320, block_size=96),
        train=dict(steps=3000, micro_batch_size=64, n_docs=40000, block_size=96),
        sft=dict(n_examples=8000, epochs=12.0),
        grpo=dict(iterations=300, prompts_per_iter=12, group_size=8, eval_every=50),
        distill=dict(steps=1200),
        eval=dict(n=400),
    ),
}

ALL_STAGES = ["pretrain_dense", "pretrain_moe", "sft", "grpo", "distill", "dashboard"]


def _eval_and_record(result: dict, kind: str, device, template: str, suite: list[str],
                     eval_n: int, parent: str | None, out_dir: str, notes: str = "") -> dict:
    """Score a model, write the registry row, dump the raw eval JSON next to the run."""
    print(f"  -- eval ({kind}, template={template!r}) --")
    task_kwargs = {t: {"n": eval_n} for t in suite if not t.startswith("ppl_")}
    ev = run_suite(result["model"], result["tokenizer"], device, tasks=suite,
                   task_kwargs=task_kwargs, template=template)
    rec = from_train_result(result, kind=kind, eval_out=ev, parent=parent, notes=notes)
    rec.config = {**(result.get("config") or {}), "eval_template": template}
    record(rec)
    run_dir = Path(out_dir) / result["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval.json").write_text(json.dumps(ev, indent=2, default=str))
    return ev


def run_pipeline(scale: str = "small", stages: list[str] | None = None,
                 out_dir: str = "runs", tracker: str = "auto", seed: int = 1337,
                 corpus: str | dict | None = None) -> dict:
    stages = stages or ALL_STAGES
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; have {sorted(SCALES)}")
    S = SCALES[scale]
    device = pick_device()
    suite = DEFAULT_SUITE
    eval_n = S["eval"]["n"]
    corpus = corpus or {"arithmetic": 0.7, "stories": 0.2, "code": 0.1}
    t_start = time.time()

    print("=" * 78)
    print(f"PIPELINE  scale={scale}  device={device.type}  git={git_sha() or 'n/a'}")
    print(f"  stages: {', '.join(stages)}")
    print(f"  corpus: {corpus}")
    print(f"  suite:  {suite}  (eval n={eval_n})")
    print("=" * 78)

    artifacts: dict[str, dict] = {}
    base = None

    # ---- 1. pretrain, dense --------------------------------------------
    if "pretrain_dense" in stages:
        mc = ModelConfig(**S["model"])
        tc = TrainConfig(corpus=corpus, tokenizer="char", seed=seed, out_dir=out_dir,
                         tracker=tracker, name_prefix="pre-dense", tags=["dense"],
                         **S["train"])
        base = train(mc, tc)
        artifacts["pretrain_dense"] = base
        _eval_and_record(base, "pretrain", device, RAW_TEMPLATE, suite, eval_n,
                         None, out_dir, notes="dense baseline")

    # ---- 2. pretrain, MoE, SAME token budget ---------------------------
    if "pretrain_moe" in stages:
        # Same depth/width/steps as the dense run, so the only difference is the FFN.
        # More total parameters, similar FLOPs per token: that is the comparison MoE
        # is actually claiming to win, and it is the one people forget to run.
        # n_shared_experts=0 keeps the comparison strictly FLOP-matched: each expert
        # is 1/top_k the dense width, so the 2 experts a token runs sum to exactly one
        # dense MLP. A shared expert is extra always-on capacity and extra FLOPs —
        # worth trying (--set n_shared_experts=1), but not in the matched baseline.
        mc = ModelConfig(**S["model"], moe=True, n_experts=8, top_k=2,
                         n_shared_experts=0, aux_loss_coef=0.01)
        tc = TrainConfig(corpus=corpus, tokenizer="char", seed=seed, out_dir=out_dir,
                         tracker=tracker, name_prefix="pre-moe", tags=["moe"],
                         **S["train"])
        moe = train(mc, tc)
        artifacts["pretrain_moe"] = moe
        _eval_and_record(moe, "pretrain", device, RAW_TEMPLATE, suite, eval_n,
                         None, out_dir, notes="MoE, 8 experts top-2 + 1 shared")
        if base:
            d, m = base["summary"], moe["summary"]
            print(f"\n  [dense vs MoE] params {d['params_total']:,} -> {m['params_total']:,} "
                  f"({m['params_total'] / d['params_total']:.2f}x total, "
                  f"{m['params_active'] / d['params_active']:.2f}x active/token)")
            print(f"                 val ppl {d['eval/final_val_ppl']:.3f} -> "
                  f"{m['eval/final_val_ppl']:.3f}   "
                  f"wall {d['perf/wall_clock_s']:.0f}s -> {m['perf/wall_clock_s']:.0f}s")

    # ---- 3. SFT --------------------------------------------------------
    sft_out = None
    if "sft" in stages:
        if base is None:
            raise RuntimeError("sft needs pretrain_dense in the same run "
                               "(or load a checkpoint yourself and call sft())")
        cfg = SFTConfig(seed=seed, out_dir=out_dir, tracker=tracker,
                        template=CHAT_TEMPLATE, tags=["post"], **S["sft"])
        sft_out = sft(base["model"], base["tokenizer"], cfg, device=device, parent=base["name"])
        artifacts["sft"] = sft_out
        # NOTE the template switch: this model was trained on "Q: ...\nA:" so it is
        # evaluated that way. Scoring it with the raw template measures format
        # mismatch, not capability — see scripts/demo_template_mismatch.py.
        _eval_and_record(sft_out, "sft", device, CHAT_TEMPLATE, suite, eval_n,
                         base["name"], out_dir, notes="response-only loss")

    # ---- 4. GRPO -------------------------------------------------------
    if "grpo" in stages:
        if sft_out is None:
            raise RuntimeError("grpo needs sft in the same run")
        import copy
        policy = copy.deepcopy(sft_out["model"])
        cfg = GRPOConfig(seed=seed, out_dir=out_dir, tracker=tracker,
                         template=CHAT_TEMPLATE, tags=["post", "rlvr"], **S["grpo"])
        rl = grpo(policy, sft_out["tokenizer"], cfg, device=device, parent=sft_out["name"])
        artifacts["grpo"] = rl
        _eval_and_record(rl, "grpo", device, CHAT_TEMPLATE, suite, eval_n,
                         sft_out["name"], out_dir, notes="RLVR on exact-match reward")

    # ---- 5. distillation ----------------------------------------------
    if "distill" in stages:
        teacher_run = sft_out or base
        if teacher_run is None:
            raise RuntimeError("distill needs a teacher (run sft or pretrain_dense)")
        teacher = teacher_run["model"]
        print("\n  -- what a soft label actually looks like --")
        print(inspect_soft_labels(teacher, teacher_run["tokenizer"],
                                 CHAT_TEMPLATE.format(q="17 + 45 ="), device))

        # A deliberately smaller student: half the width, half the depth.
        sm = dict(S["model"])
        sm["n_embd"] = max(48, sm["n_embd"] // 2)
        sm["n_layer"] = max(1, sm["n_layer"] // 2)
        sm["n_head"] = max(2, sm["n_head"] // 2)
        sm["vocab_size"] = teacher.cfg.vocab_size
        for mode in ("offline", "online"):
            student = ModelConfig(**sm)
            from .model import build_model
            cfg = DistillConfig(mode=mode, seed=seed, out_dir=out_dir, tracker=tracker,
                                cache_path=f"{out_dir}/teacher_cache.npz",
                                tags=["distill"], **S["distill"])
            kd = distill(build_model(student), teacher, teacher_run["tokenizer"], cfg,
                         device=device, parent=teacher_run["name"],
                         teacher_name=teacher_run["name"])
            artifacts[f"distill_{mode}"] = kd
            _eval_and_record(kd, "distill", device, CHAT_TEMPLATE, suite, eval_n,
                             teacher_run["name"], out_dir, notes=f"KD {mode}")

    # ---- 6. dashboard --------------------------------------------------
    if "dashboard" in stages:
        p = build_dashboard(runs_dir=out_dir, out_path="artifacts/dashboard.html")
        print(f"\n  dashboard -> {p} ({p.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 78)
    print(render_table(leaderboard()))
    print(f"\nPIPELINE done in {(time.time() - t_start) / 60:.1f} min")
    print("=" * 78)
    return artifacts
