"""
Supervised fine-tuning — the first and highest-leverage step of post-training.

Post-training, in order, is: SFT -> preference optimisation (RLHF/DPO) -> RL with
verifiable rewards (GRPO). SFT is where a base model that merely CONTINUES text
learns to RESPOND in a format. Nearly all of the "it feels like a chatbot now"
delta comes from this step; the later steps mostly adjust *which* response.

Mechanically, SFT is pretraining with one change:

    pretraining  loss on every token
    SFT          loss on the RESPONSE tokens only; the prompt is masked out

That is it. Same forward, same backward, same AdamW. The mask matters because you
do not want to teach the model to generate prompts — you want P(response | prompt).
Training on prompt tokens too is not catastrophic (it acts like extra language
modelling) but it dilutes the signal and, on templated data, teaches the model to
hallucinate new questions. `mask_prompt=False` is exposed so you can measure the
difference yourself instead of taking my word for it.

The other differences from pretraining are all about scale, not mechanism:

    data        thousands to ~1M curated examples, not billions of tokens.
                Quality >> quantity here; this is the step where 10k
                hand-checked examples beat 1M scraped ones.
    lr          ~10x lower than pretraining. You are nudging a trained model, not
                building one. Too high and you get catastrophic forgetting — the
                model's general ability degrades while it learns your format.
    epochs      1-3. SFT sets are small enough to overfit fast; past ~3 epochs
                you memorise the training answers and the eval numbers fall.
    packing     usually off (or masked), because you want clean per-example
                boundaries rather than sequences that run across examples.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

from .data import ARITH_MAX_OPERAND, arith_split
from .evaluate import CHAT_TEMPLATE
from .tokenizer import tokenizer_to_dict
from .tracking import Tracker
from .utils import config_hash, cosine_lr, env_info, pick_device, run_name, set_seed


@dataclass
class SFTConfig:
    n_examples: int = 3000
    epochs: float = 2.0
    batch_size: int = 32
    lr: float = 8e-4              # below the pretraining lr (3e-3), tuned on this task
    weight_decay: float = 0.0     # usually off or tiny for short fine-tunes
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    mask_prompt: bool = True      # flip to False to see why masking matters
    max_len: int = 64
    seed: int = 1337
    eval_every: int = 25
    log_every: int = 10
    template: str = CHAT_TEMPLATE
    out_dir: str = "runs"
    project: str = "aienh"
    name_prefix: str = "sft"
    run_name: str | None = None
    tracker: str = "auto"
    save: bool = True
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def build_sft_data(n: int = 3000, seed: int = 0, template: str = CHAT_TEMPLATE,
                   max_operand: int = ARITH_MAX_OPERAND, val_frac: float = 0.1,
                   split: str = "train") -> tuple[list[dict], list[dict]]:
    """
    Instruction data: {prompt, response} pairs in a fixed template.

    The template is a contract. Whatever you train here is what you must use at
    inference and in evaluation — a model SFT'd on "Q: ...\\nA:" scores badly when
    prompted raw, and that looks exactly like a capability regression on a
    dashboard. This is the single most common false alarm in benchmarking work.
    """
    rng = random.Random(seed)
    # Every distinct problem on the training side of the split, and no more. Asking
    # for more examples than the task contains would silently duplicate them, which
    # inflates the apparent dataset size and quietly turns 3 epochs into 6.
    pool = [(a, b) for a in range(max_operand + 1) for b in range(max_operand + 1)
            if not split or arith_split(a, b) == split]
    rng.shuffle(pool)
    rows = [{"prompt": template.format(q=f"{a} + {b} ="), "response": f" {a + b}"}
            for a, b in pool[:n]]
    n_val = max(1, int(len(rows) * val_frac))
    return rows[n_val:], rows[:n_val]


def encode_example(ex: dict, tok, max_len: int, mask_prompt: bool = True):
    """
    Returns (x, y, mask) for ONE example, as python lists.

        ids  = prompt_ids + response_ids + [EOS]
        x    = ids[:-1]        what the model sees
        y    = ids[1:]         what it should predict
        mask = 1 where y is a response token (or EOS), else 0

    Note EOS is inside the mask: the model must learn to STOP. Forget this and
    your model rambles past the answer forever, which shows up as a format /
    parse failure in eval rather than as a training bug.
    """
    p_ids = tok.encode(ex["prompt"])
    r_ids = tok.encode(ex["response"]) + [tok.eos_id]
    ids = (p_ids + r_ids)[:max_len + 1]
    x, y = ids[:-1], ids[1:]
    if mask_prompt:
        # y[i] corresponds to ids[i+1]; response starts at index len(p_ids)
        mask = [1.0 if (i + 1) >= len(p_ids) else 0.0 for i in range(len(y))]
    else:
        mask = [1.0] * len(y)
    return x, y, mask


def collate(batch: list[dict], tok, max_len: int, device, mask_prompt: bool = True):
    """Right-pad to the longest example in the batch; padded positions get mask 0
    so they contribute nothing to the loss. (Causal attention means right padding
    cannot leak into earlier positions.)"""
    encoded = [encode_example(ex, tok, max_len, mask_prompt) for ex in batch]
    L = max(len(x) for x, _, _ in encoded)
    pad = tok.pad_id
    X = torch.full((len(batch), L), pad, dtype=torch.long)
    Y = torch.full((len(batch), L), pad, dtype=torch.long)
    M = torch.zeros((len(batch), L), dtype=torch.float)
    for i, (x, y, m) in enumerate(encoded):
        X[i, :len(x)] = torch.tensor(x)
        Y[i, :len(y)] = torch.tensor(y)
        M[i, :len(m)] = torch.tensor(m)
    return X.to(device), Y.to(device), M.to(device)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

@torch.no_grad()
def sft_eval(model, tok, rows: list[dict], cfg: SFTConfig, device) -> float:
    model.eval()
    losses = []
    for s in range(0, len(rows), cfg.batch_size):
        X, Y, M = collate(rows[s:s + cfg.batch_size], tok, cfg.max_len, device, cfg.mask_prompt)
        _, loss, _ = model(X, targets=Y, loss_mask=M)
        losses.append(float(loss))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def sft(model, tok, cfg: SFTConfig, device=None, parent: str | None = None) -> dict:
    device = device or pick_device()
    set_seed(cfg.seed)
    model = model.to(device)

    train_rows, val_rows = build_sft_data(cfg.n_examples, cfg.seed, cfg.template)
    steps = max(1, int(cfg.epochs * len(train_rows) / cfg.batch_size))

    opt = model.configure_optimizer(cfg.lr, cfg.weight_decay)
    chash = config_hash(model.cfg, cfg, {"parent": parent})
    name = cfg.run_name or run_name(cfg.name_prefix, chash, cfg.seed)

    print(f"\n=== {name} (SFT{'' if cfg.mask_prompt else ', prompt NOT masked'}) ===")
    print(f"  parent={parent}  examples={len(train_rows)}  epochs={cfg.epochs} "
          f"-> {steps} steps @ batch {cfg.batch_size}  lr={cfg.lr:g}")
    print(f"  template={cfg.template!r}")

    tracker = Tracker(project=cfg.project, name=name, mode=cfg.tracker, out_dir=cfg.out_dir,
                      tags=[*cfg.tags, "sft"],
                      config={"sft": cfg.to_dict(), "model": model.cfg.to_dict(),
                              "parent": parent, "env": env_info(), "config_hash": chash,
                              "n_train": len(train_rows), "n_val": len(val_rows)})

    rng = random.Random(cfg.seed)
    model.train()
    t0 = time.time()

    for step in range(steps):
        lr = cosine_lr(step, steps, cfg.lr, cfg.warmup_frac, min_ratio=0.1)
        for g in opt.param_groups:
            g["lr"] = lr

        batch = [rng.choice(train_rows) for _ in range(cfg.batch_size)]
        X, Y, M = collate(batch, tok, cfg.max_len, device, cfg.mask_prompt)

        opt.zero_grad(set_to_none=True)
        _, loss, aux = model(X, targets=Y, loss_mask=M)
        total = loss if aux is None else loss + aux
        total.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        gnorm = float(gnorm.detach())
        opt.step()

        if step % cfg.log_every == 0 or step == steps - 1:
            # tokens_in_loss / tokens_total tells you how much of each batch is
            # actually training the model. With a long prompt and a short answer
            # this can be under 20% — which is why SFT throughput looks terrible
            # compared to pretraining on the same hardware.
            frac = float(M.mean())
            tracker.log({"sft/loss": float(loss.detach()), "sft/lr": lr,
                         "sft/grad_norm": gnorm, "sft/loss_token_frac": frac},
                        step=step)
            print(f"  step {step:4d}/{steps}  loss {float(loss.detach()):.4f}  "
                  f"lr {lr:.2e}  gnorm {gnorm:5.2f}  loss_tokens {frac:.0%}")

        if cfg.eval_every and (step % cfg.eval_every == 0 or step == steps - 1):
            vl = sft_eval(model, tok, val_rows, cfg, device)
            tracker.log({"sft/val_loss": vl}, step=step)
            print(f"    val loss {vl:.4f}")

    wall = time.time() - t0
    summary = {"sft/final_val_loss": sft_eval(model, tok, val_rows, cfg, device),
               "perf/wall_clock_s": round(wall, 1), "steps": steps,
               "params_total": model.n_params(), "params_active": model.n_active_params()}
    tracker.set_summary(summary)

    ckpt = None
    if cfg.save:
        ckpt = Path(cfg.out_dir) / name / "model.pt"
        torch.save({"model": model.state_dict(), "model_cfg": model.cfg.to_dict(),
                    "tokenizer": tokenizer_to_dict(tok), "sft_cfg": cfg.to_dict(), "config_hash": chash,
                    "parent": parent, "name": name, "summary": summary}, ckpt)
        tracker.log_artifact(ckpt, name=f"{name}-ckpt")
    tracker.finish()
    print(f"=== {name} done in {wall:.1f}s  val_loss={summary['sft/final_val_loss']:.4f} ===")

    return {"name": name, "model": model, "tokenizer": tok, "config_hash": chash,
            "summary": summary, "checkpoint": str(ckpt) if ckpt else None,
            "config": cfg.to_dict(), "device": device, "parent": parent,
            "template": cfg.template}
