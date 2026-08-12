"""
One entry point:  python -m aienh <command> [flags]

    data        run the preprocessing pipeline and print the stage report
    train       pretrain a model  (--moe for a mixture-of-experts FFN)
    sft         fine-tune a checkpoint on instruction data
    grpo        RL post-training on a verifiable reward
    distill     train a small student from a big teacher
    eval        score a checkpoint and write a registry row
    pipeline    all of the above, wired together end to end
    dashboard   regenerate the HTML leaderboard from the registry
    leaderboard print the registry as a table
    sample      generate text from a checkpoint

The config-building commands (train / sft / grpo / distill) accept
--config path.yaml (or .json) whose keys override the defaults, plus --set key=value
for one-offs. Both flat and nested keys work, and a key is routed to whichever config
declares it:

    --set lr=1e-3 --set n_layer=6 --set moe=true
    --set model.n_embd=256 --set train.steps=2000
    --set corpus='{"stories":0.8,"code":0.2}'

An unknown key is an error, not a silent no-op. Configs are the unit of
reproducibility: a run you cannot re-launch from a file is a run you cannot defend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _coerce(cur, v):
    """Coerce a string from the command line to the field's existing type."""
    if not isinstance(v, str):
        return v
    if isinstance(cur, bool):
        return v.lower() in ("1", "true", "yes", "on")
    if isinstance(cur, dict) or v.strip()[:1] in ("{", "["):
        # `corpus` is str | dict, so the target type alone cannot tell us what to do;
        # the shape of the value can. Without this, --set corpus='{"stories":0.8}'
        # stores the literal string and dies much later in build_dataset.
        return json.loads(v)
    if isinstance(cur, bool):
        return v
    if isinstance(cur, int) and not isinstance(cur, bool):
        return int(v)
    if isinstance(cur, float):
        return float(v)
    return v


def _apply(cfg_obj, overrides: dict):
    """Set dataclass fields from a dict, coercing to the field's existing type."""
    for k, v in overrides.items():
        if not hasattr(cfg_obj, k):
            raise SystemExit(
                f"unknown config key {k!r} for {type(cfg_obj).__name__}. "
                f"Known keys: {', '.join(sorted(vars(cfg_obj)))}")
        setattr(cfg_obj, k, _coerce(getattr(cfg_obj, k), v))
    return cfg_obj


def _route(overrides: dict, *configs):
    """
    Send each override to whichever config actually declares that field.

    Accepts both shapes, because both are natural:
        --set n_layer=6            flat  -> routed by field ownership
        --set model.n_layer=6      explicit prefix
        {model: {...}, train: {...}}   nested, as in configs/pretrain_dense.yaml

    An unknown key is an error, not a silent no-op: a typo'd hyperparameter that is
    quietly ignored is the worst possible failure for reproducibility.
    """
    named = {name: obj for name, obj in configs}
    for key, val in overrides.items():
        if key in named and isinstance(val, dict):          # nested section
            _apply(named[key], val)
            continue
        if "." in key:                                      # explicit prefix
            section, field = key.split(".", 1)
            if section not in named:
                raise SystemExit(f"unknown config section {section!r}; "
                                 f"have {', '.join(named)}")
            _apply(named[section], {field: val})
            continue
        owners = [obj for obj in named.values() if hasattr(obj, key)]
        if not owners:
            raise SystemExit(
                f"unknown config key {key!r}. It is not a field of "
                f"{' or '.join(type(o).__name__ for o in named.values())}.")
        if len(owners) > 1:
            raise SystemExit(f"ambiguous config key {key!r} — it exists on "
                             f"{', '.join(type(o).__name__ for o in owners)}. "
                             f"Qualify it, e.g. model.{key}=…")
        _apply(owners[0], {key: val})
    return [obj for _, obj in configs]


def _overrides(args) -> dict:
    from .utils import load_config_file
    out: dict = {}
    if args.config:
        out.update(load_config_file(args.config))
    for item in args.set or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aienh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        """Commands that build a config object: they honour --config and --set."""
        p.add_argument("--config", help="YAML/JSON file of config overrides")
        p.add_argument("--set", action="append", metavar="KEY=VAL",
                       help="inline override, repeatable (e.g. --set lr=1e-3)")
        p.add_argument("--tracker", default="auto", choices=["auto", "wandb", "local"])
        p.add_argument("--out-dir", default="runs")
        return p

    def plain(p):
        """Commands with no config object. They deliberately do NOT take --config:
        a flag that is accepted and ignored is worse than one that is absent."""
        p.add_argument("--out-dir", default="runs")
        return p

    p = plain(sub.add_parser("data", help="run the preprocessing pipeline"))
    p.add_argument("--corpus", default="dirty")
    p.add_argument("--n-docs", type=int, default=4000)
    p.add_argument("--tokenizer", default="char", choices=["char", "bpe"])
    p.add_argument("--block-size", type=int, default=64)

    p = common(sub.add_parser("train", help="pretrain a model"))
    p.add_argument("--moe", action="store_true")
    p.add_argument("--corpus", default="stories",
                   help="corpus name, or JSON mixture like '{\"stories\":0.8,\"code\":0.2}'")

    p = common(sub.add_parser("sft", help="supervised fine-tune a checkpoint"))
    p.add_argument("checkpoint")

    p = common(sub.add_parser("grpo", help="RL post-training (verifiable reward)"))
    p.add_argument("checkpoint")

    p = common(sub.add_parser("distill", help="distil a small student from a teacher"))
    p.add_argument("checkpoint", help="teacher checkpoint")
    p.add_argument("--mode", default=None, choices=["offline", "online", "on_policy"],
                   help="overrides `mode` in --config (default: online)")

    p = plain(sub.add_parser("eval", help="score a checkpoint"))
    p.add_argument("checkpoint")
    p.add_argument("--tasks", default=None, help="comma-separated task names")
    p.add_argument("--template", default="raw", choices=["raw", "chat"])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--kind", default="eval")
    p.add_argument("--no-record", action="store_true")

    p = plain(sub.add_parser("pipeline", help="run every stage end to end"))
    p.add_argument("--tracker", default="auto", choices=["auto", "wandb", "local"])
    p.add_argument("--scale", default="small", choices=["smoke", "small", "full"])
    p.add_argument("--stages", default=None, help="comma-separated subset")
    p.add_argument("--seed", type=int, default=1337)

    sub.add_parser("dashboard", help="regenerate the HTML leaderboard")
    sub.add_parser("leaderboard", help="print the registry")

    p = sub.add_parser("sample", help="generate from a checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--greedy", action="store_true")

    args = ap.parse_args(argv)

    # ---- dispatch -------------------------------------------------------
    if args.cmd == "data":
        from .data import build_dataset
        build_dataset(corpus=args.corpus, tokenizer_kind=args.tokenizer,
                      block_size=args.block_size, n_docs=args.n_docs)
        return 0

    if args.cmd == "train":
        from .model import ModelConfig
        from .train import TrainConfig, train
        corpus = json.loads(args.corpus) if args.corpus.strip().startswith("{") else args.corpus
        mc = ModelConfig(moe=args.moe)
        tc = TrainConfig(corpus=corpus, tracker=args.tracker, out_dir=args.out_dir,
                         name_prefix="pre-moe" if args.moe else "pre-dense")
        _route(_overrides(args), ("model", mc), ("train", tc))
        if mc.moe and not args.moe:
            tc.name_prefix = "pre-moe"      # --set moe=true should still be labelled
        train(mc, tc)
        return 0

    if args.cmd == "sft":
        from .sft import SFTConfig, sft
        from .train import load_checkpoint
        model, tok, blob = load_checkpoint(args.checkpoint)
        cfg = _apply(SFTConfig(tracker=args.tracker, out_dir=args.out_dir), _overrides(args))
        sft(model, tok, cfg, parent=blob.get("name"))
        return 0

    if args.cmd == "grpo":
        from .grpo import GRPOConfig, grpo
        from .train import load_checkpoint
        model, tok, blob = load_checkpoint(args.checkpoint)
        cfg = _apply(GRPOConfig(tracker=args.tracker, out_dir=args.out_dir), _overrides(args))
        grpo(model, tok, cfg, parent=blob.get("name"))
        return 0

    if args.cmd == "distill":
        from .distill import DistillConfig, distill
        from .model import ModelConfig, build_model
        from .train import load_checkpoint
        teacher, tok, blob = load_checkpoint(args.checkpoint)
        tc = teacher.cfg
        student_cfg = ModelConfig(**{**tc.to_dict(),
                                     "n_embd": max(48, tc.n_embd // 2),
                                     "n_layer": max(1, tc.n_layer // 2),
                                     "n_head": max(2, tc.n_head // 2)})
        cfg = _apply(DistillConfig(tracker=args.tracker, out_dir=args.out_dir),
                     _overrides(args))
        # An explicit flag beats the config file, so apply it last. Otherwise
        # `--mode on_policy --config configs/distill_offline.yaml` runs offline.
        if args.mode:
            cfg.mode = args.mode
        distill(build_model(student_cfg), teacher, tok, cfg,
                parent=blob.get("name"), teacher_name=blob.get("name"))
        return 0

    if args.cmd == "eval":
        from .evaluate import CHAT_TEMPLATE, DEFAULT_SUITE, RAW_TEMPLATE, run_suite
        from .registry import RunRecord, now, record
        from .train import load_checkpoint
        from .utils import git_sha, pick_device
        model, tok, blob = load_checkpoint(args.checkpoint)
        device = pick_device()
        # config_hash comes from the checkpoint the stage wrote; without it two rows
        # can share a name and differ in config, which is the mistake the registry
        # exists to prevent.
        tasks = args.tasks.split(",") if args.tasks else DEFAULT_SUITE
        template = RAW_TEMPLATE if args.template == "raw" else CHAT_TEMPLATE
        kwargs = {t: {"n": args.n} for t in tasks if not t.startswith("ppl_")}
        ev = run_suite(model, tok, device, tasks=tasks, task_kwargs=kwargs, template=template)
        name = blob.get("name") or Path(args.checkpoint).parent.name
        print(json.dumps({"points": ev["points"], "suite_hash": ev["suite_hash"]}, indent=2))
        if not args.no_record:
            metrics = {}
            for task, r in ev["results"].items():
                metrics[f"{task}/{r['metric']}"] = r["value"]
                if r.get("stderr"):
                    metrics[f"{task}/stderr"] = r["stderr"]
            record(RunRecord(
                name=name, kind=args.kind, created_at=now(),
                config_hash=blob.get("config_hash", ""),
                points=ev["points"], suite_hash=ev["suite_hash"],
                parent=blob.get("parent"), checkpoint=str(args.checkpoint),
                params_total=model.n_params(), params_active=model.n_active_params(),
                git_sha=git_sha(), metrics=metrics, breakdown=ev["breakdown"],
                config={"eval_template": template}))
            print(f"recorded {name} in runs/registry.jsonl")
        return 0

    if args.cmd == "pipeline":
        from .pipeline import ALL_STAGES, run_pipeline
        run_pipeline(scale=args.scale,
                     stages=args.stages.split(",") if args.stages else ALL_STAGES,
                     out_dir=args.out_dir, tracker=args.tracker, seed=args.seed)
        return 0

    if args.cmd == "dashboard":
        from .dashboard import build_dashboard
        p = build_dashboard()
        print(f"wrote {p} ({p.stat().st_size / 1024:.1f} KB)")
        return 0

    if args.cmd == "leaderboard":
        from .registry import leaderboard, render_table
        print(render_table(leaderboard()))
        return 0

    if args.cmd == "sample":
        from .train import load_checkpoint, sample_text
        model, tok, _ = load_checkpoint(args.checkpoint)
        from .utils import pick_device
        print(sample_text(model, tok, args.prompt, pick_device(), args.max_new_tokens,
                          temperature=args.temperature, greedy=args.greedy))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
