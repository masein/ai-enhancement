"""
Pretraining: the training loop, in full, with nothing hidden behind a Trainer class.

The four words in your list — batch size, forward pass, backward pass, optimizer —
are literally the four lines in the inner loop below. Everything else in this file
is bookkeeping around them.

    forward   run the batch through the model, get logits, compare to targets
              -> a single scalar `loss`
    backward  loss.backward(): autograd walks the graph in reverse and leaves
              d(loss)/d(param) in `param.grad` for every parameter
    optimizer optimizer.step(): use those gradients to update the weights
              (AdamW: per-parameter adaptive step, momentum + variance estimates)
    batch     how many sequences you average the gradient over before stepping.
              Bigger batch = less noisy gradient = you can afford a bigger lr,
              up to a limit ("critical batch size") past which you are just
              burning FLOPs for no extra signal.

Two details that trip people up:

  gradient accumulation
      An "effective batch" of 512 sequences will not fit in memory. So you run
      micro-batches of, say, 16, call backward on each (gradients ACCUMULATE into
      .grad by default — this is the one place PyTorch's mutable default is
      convenient), and only call optimizer.step() after 32 of them. Mathematically
      identical to a batch of 512, ~32x less memory, same wall-clock FLOPs.
      Crucially you must divide each micro-batch's loss by grad_accum, or your
      effective learning rate silently scales with grad_accum.

  zero_grad
      Because gradients accumulate, you must clear them before each new step.
      Forget it and you are training on a running sum of every batch you have
      ever seen. set_to_none=True frees the memory instead of writing zeros.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

from .data import build_dataset
from .model import GPT, ModelConfig, build_model
from .tokenizer import tokenizer_to_dict
from .tracking import Tracker
from .utils import (config_hash, cosine_lr, env_info, human, pick_device,
                    pick_dtype, run_name, set_seed)


@dataclass
class TrainConfig:
    # --- data ---
    corpus: str | dict = "stories"     # name, or {"stories": 0.8, "code": 0.2}
    n_docs: int = 6000
    tokenizer: str = "char"            # "char" | "bpe"
    vocab_size: int = 512              # only used by bpe
    block_size: int = 128
    clean: bool = True                 # run the dedup/filter pipeline

    # --- optimisation ---
    steps: int = 600
    micro_batch_size: int = 16
    grad_accum: int = 1                # effective batch = micro_batch * grad_accum
    lr: float = 3e-3
    min_lr_ratio: float = 0.1          # cosine decays to lr * this
    warmup_frac: float = 0.05
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- eval / logging ---
    eval_every: int = 50
    eval_batches: int = 20
    log_every: int = 10
    sample_every: int = 0              # 0 = off; else generate a sample every N steps
    sample_prompt: str = ""

    # --- infra ---
    seed: int = 1337
    device: str = "auto"
    dtype: str = "auto"
    out_dir: str = "runs"
    project: str = "aienh"
    name_prefix: str = "pre"
    run_name: str | None = None
    tracker: str = "auto"
    save: bool = True
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.grad_accum * self.block_size


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------

def get_batch(blocks: np.ndarray, batch_size: int, device: torch.device,
              generator: np.random.Generator):
    """
    Sample `batch_size` packed blocks at random and split into inputs/targets.

    `blocks` is [n_blocks, block_size + 1]. x = block[:-1], y = block[1:], i.e.
    y is x shifted left by one. Position t of x predicts position t of y, for
    every t at once — that is why one forward pass gives you block_size training
    signals per sequence instead of one.

    Sampling with replacement (rather than shuffling epochs) is standard for
    pretraining: at real scale you do less than one epoch anyway, so "epoch" stops
    being a useful unit and you count TOKENS instead.
    """
    idx = generator.integers(0, len(blocks), size=batch_size)
    batch = torch.from_numpy(blocks[idx].astype(np.int64))
    x = batch[:, :-1].to(device, non_blocking=True)
    y = batch[:, 1:].to(device, non_blocking=True)
    return x, y


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay — the default for essentially every LLM run.
    See utils.cosine_lr for why."""
    return cosine_lr(step, cfg.steps, cfg.lr, cfg.warmup_frac, cfg.min_lr_ratio)


@torch.no_grad()
def estimate_loss(model: GPT, blocks: np.ndarray, cfg: TrainConfig,
                  device: torch.device, generator: np.random.Generator) -> float:
    """Average loss over a few batches.

    model.eval() / no_grad matter: eval() disables dropout (so the number is
    deterministic), no_grad() stops building the autograd graph (so it is ~2x
    faster and uses far less memory). Forgetting eval() is a classic source of
    "val loss is worse than train loss and I don't know why"."""
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch(blocks, cfg.micro_batch_size, device, generator)
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def train(model_cfg: ModelConfig, cfg: TrainConfig, dataset: dict | None = None) -> dict:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    dtype = pick_dtype(device, cfg.dtype)

    # ---- data ----------------------------------------------------------
    if dataset is None:
        dataset = build_dataset(
            corpus=cfg.corpus, tokenizer_kind=cfg.tokenizer, vocab_size=cfg.vocab_size,
            block_size=cfg.block_size, n_docs=cfg.n_docs, seed=cfg.seed, clean=cfg.clean,
        )
    tok = dataset["tokenizer"]
    model_cfg.vocab_size = tok.vocab_size
    model_cfg.block_size = cfg.block_size

    # ---- model / optimizer --------------------------------------------
    model = build_model(model_cfg).to(device)
    opt = model.configure_optimizer(cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2))

    chash = config_hash(model_cfg, cfg, {"data": dataset["label"]})
    name = cfg.run_name or run_name(cfg.name_prefix, chash, cfg.seed)

    total_tokens = cfg.steps * cfg.tokens_per_step
    header = {
        "params_total": model.n_params(),
        "params_active": model.n_active_params(),
        "params_non_embedding": model.n_params(non_embedding=True),
        "tokens_per_step": cfg.tokens_per_step,
        "total_train_tokens": total_tokens,
        # Chinchilla-optimal is ~20 tokens per parameter. Far below that and the
        # model is undertrained (add data/steps); far above and you are paying for
        # marginal gains — which is nevertheless what every deployed model does,
        # because inference cost dominates training cost over a model's lifetime.
        "tokens_per_param": round(total_tokens / max(1, model.n_params()), 2),
        "dataset_tokens": dataset["n_train_tokens"],
        "epochs_over_data": round(total_tokens / max(1, dataset["n_train_tokens"]), 2),
    }

    print(f"\n=== {name} ===")
    print(f"  device={device.type} dtype={str(dtype).split('.')[-1]} "
          f"vocab={tok.vocab_size} block={cfg.block_size}")
    print(f"  params={human(header['params_total'])} "
          f"(active/token={human(header['params_active'])}) "
          f"moe={model_cfg.moe}{f' E={model_cfg.n_experts} k={model_cfg.top_k}' if model_cfg.moe else ''}")
    print(f"  batch: {cfg.micro_batch_size} x {cfg.grad_accum} accum x {cfg.block_size} ctx "
          f"= {human(cfg.tokens_per_step)} tokens/step")
    print(f"  budget: {cfg.steps} steps = {human(total_tokens)} tokens "
          f"({header['tokens_per_param']} tokens/param, "
          f"{header['epochs_over_data']} epochs over the data)")

    tracker = Tracker(
        project=cfg.project, name=name, mode=cfg.tracker, out_dir=cfg.out_dir,
        tags=[*cfg.tags, "pretrain"],
        config={
            "model": model_cfg.to_dict(), "train": cfg.to_dict(),
            "data": {"label": dataset["label"], **{k: dataset[k] for k in
                     ("n_train_tokens", "n_val_tokens", "n_train_docs", "n_val_docs")},
                     "pipeline": dataset["report"].to_dict(),
                     "contamination": dataset.get("contamination")},
            "derived": header, "env": env_info(), "config_hash": chash,
        },
    )

    rng = np.random.default_rng(cfg.seed)
    # Evaluation uses freshly-seeded generators every time (below), so the same
    # batches are drawn at every eval point and across runs — otherwise a moving
    # sample adds noise to exactly the comparison you care about.
    train_blocks, val_blocks = dataset["train"], dataset["val"]

    autocast = (torch.autocast(device_type=device.type, dtype=dtype)
                if dtype != torch.float32 and device.type in ("cuda", "cpu")
                else torch.autocast(device_type="cpu", enabled=False))

    model.train()
    best_val = float("inf")
    t0 = time.time()
    history: list[dict] = []

    for step in range(cfg.steps):
        # 1. learning rate for this step (schedules are per-step, not per-epoch)
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        # 2. clear last step's gradients
        opt.zero_grad(set_to_none=True)

        loss_acc, aux_acc = 0.0, 0.0
        for _ in range(cfg.grad_accum):
            x, y = get_batch(train_blocks, cfg.micro_batch_size, device, rng)

            # 3. FORWARD: inputs -> logits -> scalar loss
            with autocast:
                _, loss, aux = model(x, targets=y)
                total = loss if aux is None else loss + aux

            # 4. BACKWARD: fill .grad for every parameter.
            #    Divide by grad_accum so the accumulated gradient is the MEAN over
            #    the effective batch, not the sum.
            (total / cfg.grad_accum).backward()

            loss_acc += loss.item() / cfg.grad_accum
            aux_acc += (0.0 if aux is None else float(aux)) / cfg.grad_accum

        # 5. clip: rescale the whole gradient vector if its norm exceeds a
        #    threshold. Cheap insurance against one bad batch blowing up the run.
        #    Watch this number: a spiking grad_norm precedes a loss spike.
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        gnorm = float(gnorm.detach())

        # 6. OPTIMIZER STEP: weights <- weights - lr * adaptive_step(grad)
        opt.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            elapsed = time.time() - t0
            tok_s = cfg.tokens_per_step * (step + 1) / max(elapsed, 1e-6)
            row = {
                "train/loss": loss_acc, "train/ppl": math.exp(min(20, loss_acc)),
                "train/aux_loss": aux_acc, "train/lr": lr,
                "train/grad_norm": gnorm, "perf/tokens_per_s": tok_s,
                "progress/tokens": (step + 1) * cfg.tokens_per_step,
            }
            row.update(model.moe_stats())
            tracker.log(row, step=step)
            print(f"  step {step:5d}/{cfg.steps}  loss {loss_acc:.4f}  "
                  f"ppl {row['train/ppl']:8.2f}  lr {lr:.2e}  "
                  f"gnorm {gnorm:5.2f}  {human(tok_s)} tok/s")

        if cfg.eval_every and (step % cfg.eval_every == 0 or step == cfg.steps - 1):
            vl = estimate_loss(model, val_blocks, cfg, device, np.random.default_rng(cfg.seed + 1))
            tl = estimate_loss(model, train_blocks, cfg, device, np.random.default_rng(cfg.seed + 2))
            best_val = min(best_val, vl)
            tracker.log({"eval/val_loss": vl, "eval/val_ppl": math.exp(min(20, vl)),
                         "eval/train_loss": tl,
                         # val - train, so it grows POSITIVE as the model overfits.
                         # On a small corpus with many epochs it will open up; that is
                         # the memorisation you are trying to see. Both sides are
                         # measured with a fixed generator seed so the gap is not
                         # polluted by which batches happened to be drawn.
                         "eval/gap": vl - tl}, step=step)
            history.append({"step": step, "val_loss": vl, "train_loss": tl})
            print(f"    eval  train {tl:.4f} | val {vl:.4f} | ppl {math.exp(min(20, vl)):.2f}")

        if cfg.sample_every and step and step % cfg.sample_every == 0:
            print("    sample:", repr(sample_text(model, tok, cfg.sample_prompt, device, 80)))

    # ---- finish --------------------------------------------------------
    wall = time.time() - t0
    final_val = estimate_loss(model, val_blocks, cfg, device, np.random.default_rng(cfg.seed + 1))
    summary = {
        "eval/final_val_loss": final_val,
        "eval/final_val_ppl": math.exp(min(20, final_val)),
        "eval/best_val_loss": best_val,
        "perf/wall_clock_s": round(wall, 1),
        "perf/tokens_per_s": cfg.tokens_per_step * cfg.steps / max(wall, 1e-6),
        **header,
    }
    tracker.set_summary(summary)

    ckpt_path = None
    if cfg.save:
        ckpt_path = Path(cfg.out_dir) / name / "model.pt"
        torch.save({
            "model": model.state_dict(),
            "model_cfg": model_cfg.to_dict(),
            "tokenizer": tokenizer_to_dict(tok),
            "train_cfg": cfg.to_dict(),
            "summary": summary,
            "name": name,
            "config_hash": chash,
        }, ckpt_path)
        tracker.log_artifact(ckpt_path, name=f"{name}-ckpt", kind="model")
        print(f"  saved {ckpt_path}")

    tracker.finish()
    print(f"=== {name} done in {wall:.1f}s  val_ppl={summary['eval/final_val_ppl']:.2f} ===")

    return {"name": name, "model": model, "tokenizer": tok, "config_hash": chash,
            "summary": summary, "history": history, "config": cfg.to_dict(),
            "checkpoint": str(ckpt_path) if ckpt_path else None,
            "dataset": dataset, "device": device}


def sample_text(model: GPT, tok, prompt: str, device, max_new_tokens: int = 100,
                temperature: float = 0.8, greedy: bool = False) -> str:
    ids = tok.encode(prompt) or [tok.eos_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens, temperature=temperature, greedy=greedy)
    return tok.decode(out[0].tolist())


def load_checkpoint(path: str | Path, device=None):
    """Rebuild (model, tokenizer) from a checkpoint. Everything needed to
    reproduce inference is inside the file — no 'you also need config X' step."""
    from .tokenizer import tokenizer_from_dict
    device = device or pick_device()
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["model_cfg"])
    model = build_model(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, tokenizer_from_dict(blob["tokenizer"]), blob
