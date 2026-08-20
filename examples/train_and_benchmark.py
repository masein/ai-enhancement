#!/usr/bin/env python3
"""Train a tiny model for real, benchmark every checkpoint — the end-to-end
pipeline test, and the working sample behind API.md's integration pattern.

Full mode:
  1. fine-tunes a small causal LM (default EleutherAI/pythia-14m) on a streamed
     public corpus (default roneneldan/TinyStories) for --steps optimizer steps
  2. every --checkpoint-every steps (and at the end) pushes the checkpoint to
     the Hub as <push-to>-step<N> — weights AND tokenizer; the harness needs both
  3. submits each checkpoint to the benchmark service the moment it is pushed —
     non-blocking, and a submit failure never kills training
  4. after training, waits for the evaluations and prints a task × checkpoint
     table: capability-vs-steps, next to the loss you watched during training

Connectivity check first (no training, no HF account, ~a minute of queue time):

    python examples/train_and_benchmark.py --bench http://teraformer-5090-3:8899 --dry-run

The real thing (needs `hf auth login` with a WRITE token):

    python examples/train_and_benchmark.py \
        --bench http://teraformer-5090-3:8899 \
        --push-to <your-hf-username>/bench-demo \
        --steps 200 --checkpoint-every 100 --suite quick

Offline rehearsal (trains and saves locally, never touches the Hub or the
service — useful for checking your environment before spending anything):

    python examples/train_and_benchmark.py --local-only --steps 20

Dependencies:  pip install torch transformers datasets huggingface_hub

Two sharp edges, named:
  * Checkpoints are pushed as SEPARATE repos (…-step100, …-step200). The service
    identifies a run by repo id, so each checkpoint needs its own id — pushing
    commits to one repo would make every submission "the same model".
  * Push public (the default). A private repo is only benchmarkable if the
    server's HF account has read access to it.
"""

from __future__ import annotations

import argparse
import getpass
import math
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "clients"))
from bench_client import Bench, BenchError  # noqa: E402


# ---------------------------------------------------------------------------
# data: stream a text corpus, pack it into fixed-length blocks
# ---------------------------------------------------------------------------

def iter_texts(dataset_id, field):
    """Documents from a Hub dataset (streamed) or a local .txt/.jsonl file —
    local files loop forever so a short corpus can still feed many steps."""
    import json
    p = Path(dataset_id)
    if p.exists():
        while True:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if p.suffix in (".jsonl", ".ndjson"):
                        try:
                            yield json.loads(line).get(field) or ""
                        except json.JSONDecodeError:
                            continue
                    else:
                        yield line
        return
    from datasets import load_dataset
    for row in load_dataset(dataset_id, split="train", streaming=True):
        yield row.get(field) or ""


def packed_batches(dataset_id, field, tokenizer, seq_len, batch_size):
    """Tokenize, concatenate, and cut into (batch, seq_len) blocks — the
    standard causal-LM packing, minus every optimization."""
    import torch

    buf, batch = [], []
    for text in iter_texts(dataset_id, field):
        if not text:
            continue
        buf.extend(tokenizer(text)["input_ids"])
        buf.append(tokenizer.eos_token_id)
        while len(buf) >= seq_len:
            batch.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(batch) == batch_size:
                yield torch.tensor(batch, dtype=torch.long)
                batch = []


# ---------------------------------------------------------------------------
# the training loop — deliberately the shortest honest version
# ---------------------------------------------------------------------------

def train(args, bench):
    import tempfile

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name or f"demo-{int(time.time()) % 100000}"
    print(f"device: {device} · base: {args.base_model} · run: {run_name} · "
          f"{args.steps} steps of {args.batch_size}×{args.seq_len} tokens")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # .float() is load-bearing: newer transformers loads checkpoints in their
    # STORED dtype (pythia ships fp16), and full fine-tuning in fp16 without a
    # loss scaler diverges to NaN within a few hundred steps. fp32 is cheap at
    # this size and always stable.
    model = AutoModelForCausalLM.from_pretrained(args.base_model).float().to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup = min(50, max(10, args.steps // 10))   # LR warmup — cheap divergence insurance

    # run tracking: curves appear live on the dashboard's Training tab. If no
    # service is reachable, run=None and everything still works locally.
    run = None
    if bench:
        try:
            run = bench.init(run_name, project=args.project, submitter=args.submitter,
                             config={"base_model": args.base_model, "dataset": args.dataset,
                                     "lr": args.lr, "batch_size": args.batch_size,
                                     "seq_len": args.seq_len, "steps": args.steps,
                                     "device": device},
                             hf_prefix=args.push_to or f"local/{run_name}")
            print(f"  tracking as training run #{run.id} — watch the Training tab")
        except BenchError as e:
            print(f"  run tracking unavailable (non-fatal): {e}")

    submitted = []          # (step, model_id, submission_id | None)

    def checkpoint(step, running_loss):
        # 1) materialize the checkpoint
        if args.local_only:
            out = Path(f"checkpoint-step{step}")
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            print(f"  [ckpt] step {step}: saved locally to {out}/ (local-only mode)")
            return
        if args.push_to:                       # Hub mode — needs a write token
            model_id = f"{args.push_to}-step{step}"
            print(f"  [ckpt] step {step}: pushing {model_id} to the Hub ...")
            model.push_to_hub(model_id)
            tokenizer.push_to_hub(model_id)    # without this, evaluation cannot load it
        else:                                  # artifact mode — no HF account at all
            with tempfile.TemporaryDirectory() as td:
                model.save_pretrained(td)
                tokenizer.save_pretrained(td)
                try:
                    model_id = bench.upload_artifact(f"{run_name}-step{step}", td)
                except BenchError as e:
                    print(f"  [ckpt] step {step}: upload failed (non-fatal): {e}")
                    return
            print(f"  [ckpt] step {step}: uploaded as {model_id}")
        # 2) mark it on the run + queue the benchmark, without waiting
        if run:
            sid = run.log_checkpoint(step, model_id, suite=args.suite,
                                     note=f"{run_name} step {step}, loss {running_loss:.3f}")
        else:
            try:
                sid = bench.submit(model_id, suite=args.suite, submitter=args.submitter,
                                   note=f"step {step}, loss {running_loss:.3f}")
            except BenchError as e:
                print(f"  [ckpt] step {step}: submit failed (non-fatal): {e}")
                sid = None
        if sid:
            print(f"  [ckpt] step {step}: benchmark queued as #{sid}")
        submitted.append((step, model_id, sid))

    batches = packed_batches(args.dataset, args.field, tokenizer,
                             args.seq_len, args.batch_size)
    t0, running, nan_warned = time.time(), None, False
    for step in range(1, args.steps + 1):
        lr_now = args.lr * min(1.0, step / warmup)
        for g in opt.param_groups:
            g["lr"] = lr_now
        ids = next(batches).to(device)
        loss = model(input_ids=ids, labels=ids).loss
        loss_val = loss.item()
        if not math.isfinite(loss_val):
            # a NaN step must not update weights — one bad step poisons the model
            opt.zero_grad(set_to_none=True)
            if not nan_warned:
                print(f"  step {step}: loss is not finite — skipping the optimizer "
                      f"step. If this persists, lower --lr (currently {args.lr:g}).")
                nan_warned = True
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = loss_val if running is None else 0.95 * running + 0.05 * loss_val
        tok_s = step * args.batch_size * args.seq_len / (time.time() - t0)
        if run:
            m = {"loss": loss_val, "lr": lr_now, "grad_norm": float(gnorm),
                 "tokens_per_s": tok_s}
            if device == "cuda":
                m["gpu_mem_gb"] = torch.cuda.memory_allocated() / 1e9
            run.log(m, step=step)
        if step % 10 == 0 or step == 1:
            print(f"  step {step:>5}  loss {loss.item():.3f}  (avg {running:.3f}, "
                  f"CE nats/token — compare the nats/byte column on the dashboard)  "
                  f"{tok_s:,.0f} tok/s")
        if step % args.checkpoint_every == 0 or step == args.steps:
            checkpoint(step, running)

    if run:
        run.finish()
    return submitted


def collect(bench, submitted):
    """Wait for every submitted checkpoint, then print task × step scores."""
    table: dict[str, dict[int, float]] = {}
    for step, repo, sid in submitted:
        if sid is None:
            continue
        try:
            bench.wait(sid, poll_s=20, echo=True)
        except BenchError as e:
            print(f"  #{sid} ({repo}): {e}")
            continue
        for task, s in bench.scores(repo).items():
            table.setdefault(task, {})[step] = s["value"]
    if not table:
        print("no scores came back — check the queue on the dashboard")
        return
    steps = sorted({s for by in table.values() for s in by})
    print("\ncapability vs training step")
    print(f"{'task':<22}" + "".join(f"step {s:<9}" for s in steps))
    for task in sorted(table):
        row = "".join(f"{table[task].get(s, float('nan')):<14.4f}" for s in steps)
        print(f"{task:<22}{row}")
    print("\n(the same numbers are on the dashboard's leaderboard — these repos are "
          "ordinary models there, filterable by your name in provenance)")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--bench", help="benchmark service base URL")
    ap.add_argument("--token", default="", help="X-Token if the server requires one")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip training; submit a known public model to prove the service works")
    ap.add_argument("--local-only", action="store_true",
                    help="train + save locally; never touch the Hub or the service")
    ap.add_argument("--push-to", help="OPTIONAL Hub repo prefix (needs a write token). "
                    "Without it, checkpoints upload to the service's artifact storage — "
                    "no HF account needed.")
    ap.add_argument("--run-name", help="training-run name on the dashboard (default: demo-<time>)")
    ap.add_argument("--project", default="default")
    ap.add_argument("--base-model", default="EleutherAI/pythia-14m")
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--field", default="text")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--suite", default="quick", choices=["quick", "full"])
    ap.add_argument("--submitter", default=getpass.getuser())
    ap.add_argument("--no-wait", action="store_true",
                    help="don't wait for evaluations at the end (watch the dashboard instead)")
    args = ap.parse_args()

    bench = Bench(args.bench, args.token) if args.bench else None

    if args.dry_run:
        if not bench:
            ap.error("--dry-run needs --bench")
        probe = "EleutherAI/pythia-31m"
        print(f"dry run: submitting {probe} (quick suite) to {args.bench}")
        sid = bench.submit(probe, suite="quick", submitter=args.submitter,
                           note="pipeline dry run")
        bench.wait(sid, poll_s=10, echo=True)
        print("scores:")
        for task, s in bench.scores(probe).items():
            print(f"  {task:<22} {s['value']:.4f}"
                  + (f" ± {s['stderr']:.4f}" if s.get("stderr") else "")
                  + f"  ({s['metric']})")
        print("service OK — the whole pipeline works.")
        return 0

    if not args.local_only and not args.bench:
        ap.error("full mode needs --bench (or use --local-only / --dry-run). "
                 "Checkpoints go to the service's artifact storage by default; "
                 "add --push-to user/prefix to use the HF Hub instead.")
    if args.push_to and "/" not in args.push_to:
        ap.error("--push-to must look like username/repo-prefix")

    submitted = train(args, bench)
    if args.local_only or args.no_wait or not submitted:
        return 0
    print("\ntraining done — collecting benchmark results "
          "(safe to Ctrl-C; the dashboard keeps everything)")
    collect(bench, submitted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
