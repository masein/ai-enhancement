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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} · base: {args.base_model} · {args.steps} steps "
          f"of {args.batch_size}×{args.seq_len} tokens")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    submitted = []          # (step, repo_id, submission_id | None)

    def checkpoint(step, running_loss):
        if args.local_only:
            out = Path(f"checkpoint-step{step}")
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            print(f"  [ckpt] step {step}: saved locally to {out}/ (local-only mode)")
            return
        repo = f"{args.push_to}-step{step}"
        print(f"  [ckpt] step {step}: pushing {repo} ...")
        model.push_to_hub(repo)
        tokenizer.push_to_hub(repo)      # without this, evaluation cannot load it
        try:
            sid = bench.submit(repo, suite=args.suite, kind="base",
                               submitter=args.submitter,
                               note=f"step {step}, loss {running_loss:.3f}")
            print(f"  [ckpt] step {step}: submitted as #{sid} (not waiting — training continues)")
            submitted.append((step, repo, sid))
        except BenchError as e:
            # the pattern API.md preaches: benchmarking must never kill training
            print(f"  [ckpt] step {step}: submit failed (non-fatal): {e}")
            submitted.append((step, repo, None))

    batches = packed_batches(args.dataset, args.field, tokenizer,
                             args.seq_len, args.batch_size)
    t0, running = time.time(), None
    for step in range(1, args.steps + 1):
        ids = next(batches).to(device)
        loss = model(input_ids=ids, labels=ids).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = loss.item() if running is None else 0.95 * running + 0.05 * loss.item()
        if step % 10 == 0 or step == 1:
            tok_s = step * args.batch_size * args.seq_len / (time.time() - t0)
            print(f"  step {step:>5}  loss {loss.item():.3f}  (avg {running:.3f}, "
                  f"CE nats/token — compare the nats/byte column on the dashboard)  "
                  f"{tok_s:,.0f} tok/s")
        if step % args.checkpoint_every == 0 or step == args.steps:
            checkpoint(step, running)

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
    ap.add_argument("--push-to", help="Hub repo prefix, e.g. youruser/bench-demo")
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

    if not args.local_only:
        if not args.bench or not args.push_to:
            ap.error("full mode needs --bench and --push-to (or use --local-only / --dry-run)")
        if "/" not in (args.push_to or ""):
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
