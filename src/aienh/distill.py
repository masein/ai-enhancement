"""
Knowledge distillation: soft labels, hard labels, offline / online / on-policy.

HARD vs SOFT LABELS
    hard label   the one correct token. A one-hot vector. Information content:
                 log2(V) bits at most, and in practice ~0 bits of nuance — it says
                 "62" and nothing about what else was plausible.
    soft label   the teacher's full distribution over the vocabulary. It says
                 "62 with p=0.7, 61 with p=0.15, 72 with p=0.05, ...". That extra
                 structure is the "dark knowledge": it encodes which mistakes are
                 near-misses and which are absurd, and it is a far richer training
                 signal per token than one-hot.

    Practical consequence: a student trained on soft labels needs fewer tokens to
    reach the same quality, because each token carries more information. That is
    the entire reason distillation works, and why every small production model is
    distilled from a big one rather than pretrained from scratch.

    Temperature T flattens the teacher distribution (T>1) to expose the low-
    probability structure. The classic loss is
        L = alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE(hard)
    The T^2 is not cosmetic: dividing logits by T shrinks the gradients by 1/T^2,
    so you multiply back to keep the soft term's magnitude comparable to the hard
    term as you tune T.

THREE FLAVOURS — the distinction you were asked about
    OFFLINE   Run the teacher ONCE over the corpus, store its outputs, then train
              the student against the stored targets. Teacher GPU time is paid
              once; student training is then as cheap as ordinary pretraining, and
              you can iterate on the student twenty times without touching the
              teacher.
              Cost: storage, and staleness — the targets only cover text the
              teacher was run on, which is text the *teacher* would produce or saw,
              not text the *student* produces.
              Storage math, real numbers: full logits at V=128k, fp16, 1B tokens
              = 2 * 128e3 * 1e9 = 256 TB. Nobody does that. You store top-k
              (k=8..64): k=16 costs (2 bytes value + 4 bytes index) * 16 * 1e9
              = 96 GB, which is a normal disk. This module does exactly that, and
              renormalises over the stored k at train time.

    ONLINE    Teacher and student both in memory; teacher runs forward (no grad)
              on the same batch as the student, every step. No storage, always
              consistent with whatever data you feed, and you can use the FULL
              distribution rather than a top-k approximation.
              Cost: teacher forward on every step forever — typically 2-4x the
              student's own cost, and the teacher must fit alongside the student.

    ON-POLICY (a.k.a. sequence-level KD / GKD)
              The student GENERATES, and the teacher scores the student's own
              samples. This fixes the deepest problem with both of the above:
              teacher-forced training only ever shows the student states the
              teacher would visit, so at inference the student walks into states
              it never saw and compounds errors ("exposure bias"). Training on the
              student's own trajectories removes the mismatch, and empirically
              this is what closes the last chunk of the teacher-student gap.
              Cost: generation in the training loop — the slowest option, and it
              shares the sampling machinery (and the wall-clock profile) of RL.

WHICH TO USE: offline while you are iterating on student architecture and data
(cheap, repeatable, and the teacher bill is sunk); online when the teacher is
small enough to co-reside or you need full distributions; on-policy as the last
stage, once the student is otherwise trained, to fix generation-time behaviour.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .sft import build_sft_data, collate
from .tokenizer import tokenizer_to_dict
from .tracking import Tracker
from .utils import config_hash, cosine_lr, env_info, pick_device, run_name, set_seed


@dataclass
class DistillConfig:
    mode: str = "online"          # "offline" | "online" | "on_policy"
    steps: int = 300
    batch_size: int = 32
    lr: float = 1e-3
    temperature: float = 2.0
    alpha: float = 0.9            # weight on the soft (KL) term
    top_k: int = 16               # offline only: how many teacher logits to store
    max_new_tokens: int = 6       # on_policy only: rollout length
    temperature_sample: float = 1.0   # on_policy only: rollout temperature (NOT the KD T)
    grad_clip: float = 1.0
    warmup_frac: float = 0.05
    max_len: int = 64
    n_examples: int = 3000
    seed: int = 1337
    log_every: int = 20
    eval_every: int = 50
    cache_path: str = "runs/teacher_cache.npz"
    out_dir: str = "runs"
    project: str = "aienh"
    name_prefix: str = "kd"
    run_name: str | None = None
    tracker: str = "auto"
    save: bool = True
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# the pedagogical bit: look at an actual soft label
# ---------------------------------------------------------------------------

@torch.no_grad()
def inspect_soft_labels(teacher, tok, prompt: str, device, temperatures=(1.0, 2.0, 5.0),
                        top: int = 6) -> str:
    """
    Print the teacher's distribution for the next token at several temperatures.

    Do this once, on a real checkpoint, and the soft/hard distinction stops being
    abstract: you can see the entropy, you can see which wrong answers are
    near-misses, and you can see the hard label throwing all of it away.
    """
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    logits, _, _ = teacher(ids)
    logits = logits[0, -1].float()
    lines = [f"prompt={prompt!r}"]
    for T in temperatures:
        p = F.softmax(logits / T, dim=-1)
        H = float(-(p * p.clamp_min(1e-12).log()).sum())
        vals, idx = p.topk(top)
        shown = "  ".join(f"{tok.decode([int(i)])!r}:{float(v):.3f}" for v, i in zip(vals, idx))
        lines.append(f"  T={T:<4g} entropy={H:5.3f} nats   {shown}")
    lines.append("  hard label would be: " +
                 repr(tok.decode([int(logits.argmax())])) + "  (everything else discarded)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def kd_loss_full(student_logits, teacher_logits, targets, mask, T: float, alpha: float):
    """
    Soft + hard loss with the teacher's FULL distribution (online mode).

    KL(teacher || student) is the "forward" direction, which is mode-covering: the
    student is penalised for putting low probability anywhere the teacher puts
    mass, so it tries to cover all of the teacher's behaviour. (Reverse KL is
    mode-seeking, and produces sharper, less diverse students. This choice is a
    real design decision with visible consequences, not a formality.)
    """
    p_t = F.softmax(teacher_logits.float() / T, dim=-1)
    logp_s = F.log_softmax(student_logits.float() / T, dim=-1)
    logp_t = torch.log(p_t.clamp_min(1e-12))
    kl_tok = (p_t * (logp_t - logp_s)).sum(-1)                  # [B, T]
    m = mask
    soft = (kl_tok * m).sum() / m.sum().clamp_min(1.0) * (T ** 2)

    ce_tok = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)).float(),
                             targets.reshape(-1), reduction="none").view_as(m)
    hard = (ce_tok * m).sum() / m.sum().clamp_min(1.0)
    return alpha * soft + (1 - alpha) * hard, float(soft.detach()), float(hard.detach())


def kd_loss_topk(student_logits, t_vals, t_idx, targets, mask, T: float, alpha: float):
    """
    Same idea against a stored top-k teacher (offline mode).

    The teacher's probabilities are RENORMALISED over the stored k, so you are
    matching a truncated distribution. That is an approximation, and the error
    grows as k shrinks: with k=1 you are back to hard labels (plus a temperature),
    which defeats the point. k in 8..64 is the usual range; measure it rather than
    guessing, because the right k depends on how peaked your teacher is.
    """
    t_probs = F.softmax(t_vals.float() / T, dim=-1)              # [B, T, k]
    logp_s_full = F.log_softmax(student_logits.float() / T, dim=-1)
    logp_s = torch.gather(logp_s_full, -1, t_idx)                # [B, T, k]
    # BOTH sides must live on the same support. The teacher was renormalised over
    # the stored k above, so the student must be too — otherwise the "KL" also
    # penalises all the student's mass outside the top-k, which is a different
    # objective with a large k-dependent floor. (Measured before this line existed:
    # a student with logits IDENTICAL to the teacher's scored 6.91 at k=16 instead
    # of 0.0, and the offline and online losses were not comparable at all.)
    logp_s = logp_s - logp_s.logsumexp(-1, keepdim=True)
    kl_tok = (t_probs * (torch.log(t_probs.clamp_min(1e-12)) - logp_s)).sum(-1)
    m = mask
    soft = (kl_tok * m).sum() / m.sum().clamp_min(1.0) * (T ** 2)

    ce_tok = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)).float(),
                             targets.reshape(-1), reduction="none").view_as(m)
    hard = (ce_tok * m).sum() / m.sum().clamp_min(1.0)
    return alpha * soft + (1 - alpha) * hard, float(soft.detach()), float(hard.detach())


# ---------------------------------------------------------------------------
# offline: build the teacher cache
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_teacher_cache(teacher, tok, rows: list[dict], cfg: DistillConfig, device) -> dict:
    """
    Run the teacher once over the dataset and store top-k logits per position.

    Also prints the storage arithmetic for full logits vs top-k at production
    scale, because that ratio is the whole reason top-k caching exists.
    """
    teacher.eval()
    all_x, all_y, all_m, all_v, all_i = [], [], [], [], []
    for s in range(0, len(rows), cfg.batch_size):
        X, Y, M = collate(rows[s:s + cfg.batch_size], tok, cfg.max_len, device)
        logits, _, _ = teacher(X)
        vals, idx = logits.float().topk(cfg.top_k, dim=-1)
        # Pad every batch to the same width so this stacks into one array.
        L = cfg.max_len
        def padr(t, width, value=0):
            out = torch.full((t.size(0), width, *t.shape[2:]), value, dtype=t.dtype, device=t.device)
            out[:, :t.size(1)] = t[:, :width]
            return out
        all_x.append(padr(X.unsqueeze(-1), L).squeeze(-1).cpu().numpy())
        all_y.append(padr(Y.unsqueeze(-1), L).squeeze(-1).cpu().numpy())
        all_m.append(padr(M.unsqueeze(-1), L).squeeze(-1).cpu().numpy())
        all_v.append(padr(vals, L).cpu().numpy().astype(np.float16))
        all_i.append(padr(idx, L).cpu().numpy().astype(np.int32))

    cache = {
        "x": np.concatenate(all_x), "y": np.concatenate(all_y),
        "m": np.concatenate(all_m), "vals": np.concatenate(all_v),
        "idx": np.concatenate(all_i),
    }
    Path(cfg.cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.cache_path, **cache)

    n_tok = int(cache["m"].sum())
    V = teacher.cfg.vocab_size
    on_disk = Path(cfg.cache_path).stat().st_size
    print(f"  [offline] cached {len(cache['x'])} sequences, {n_tok:,} supervised tokens "
          f"-> {on_disk / 1e6:.2f} MB on disk (top_k={cfg.top_k})")
    print(f"  [offline] full logits for the same tokens would be "
          f"{2 * V * n_tok / 1e6:.2f} MB (fp16, V={V}) "
          f"= {2 * V / (6 * cfg.top_k):.0f}x more")
    print(f"  [offline] at production scale (V=128k, 1B tokens): "
          f"full = 256 TB, top_k={cfg.top_k} = {6 * cfg.top_k * 1e9 / 1e12:.2f} TB")
    return cache


# ---------------------------------------------------------------------------
# the training loop
# ---------------------------------------------------------------------------

def distill(student, teacher, tok, cfg: DistillConfig, device=None,
            parent: str | None = None, teacher_name: str | None = None) -> dict:
    device = device or pick_device()
    set_seed(cfg.seed)
    student, teacher = student.to(device), teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():          # the teacher is frozen, always
        p.requires_grad_(False)

    train_rows, val_rows = build_sft_data(cfg.n_examples, cfg.seed)
    opt = student.configure_optimizer(cfg.lr, weight_decay=0.0)
    chash = config_hash(student.cfg, cfg, {"parent": parent, "teacher": teacher_name})
    name = cfg.run_name or run_name(f"{cfg.name_prefix}-{cfg.mode}", chash, cfg.seed)

    print(f"\n=== {name} (distill, mode={cfg.mode}) ===")
    print(f"  teacher={teacher_name} ({teacher.n_params():,} params) -> "
          f"student ({student.n_params():,} params, "
          f"{student.n_params() / max(1, teacher.n_params()):.1%} of teacher)")
    print(f"  T={cfg.temperature} alpha={cfg.alpha} (alpha=1 -> pure soft, 0 -> pure hard)")

    cache = None
    if cfg.mode == "offline":
        cached = Path(cfg.cache_path)
        if cached.exists():
            # This branch is the entire economic argument for offline KD: the teacher
            # bill is sunk, so iterating on the student is as cheap as pretraining.
            cache = dict(np.load(cached))
            print(f"  [offline] reusing teacher cache {cached} "
                  f"({cached.stat().st_size / 1e6:.2f} MB) — teacher not run")
        else:
            cache = build_teacher_cache(teacher, tok, train_rows, cfg, device)

    tracker = Tracker(project=cfg.project, name=name, mode=cfg.tracker, out_dir=cfg.out_dir,
                      tags=[*cfg.tags, "distill", cfg.mode],
                      config={"distill": cfg.to_dict(), "student": student.cfg.to_dict(),
                              "teacher": teacher.cfg.to_dict(), "teacher_name": teacher_name,
                              "parent": parent, "env": env_info(), "config_hash": chash})

    rng = random.Random(cfg.seed)
    student.train()
    t0 = time.time()

    for step in range(cfg.steps):
        lr = cosine_lr(step, cfg.steps, cfg.lr, cfg.warmup_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)

        if cfg.mode == "offline":
            sel = rng.sample(range(len(cache["x"])), min(cfg.batch_size, len(cache["x"])))
            X = torch.from_numpy(cache["x"][sel].astype(np.int64)).to(device)
            Y = torch.from_numpy(cache["y"][sel].astype(np.int64)).to(device)
            M = torch.from_numpy(cache["m"][sel].astype(np.float32)).to(device)
            tv = torch.from_numpy(cache["vals"][sel].astype(np.float32)).to(device)
            ti = torch.from_numpy(cache["idx"][sel].astype(np.int64)).to(device)
            s_logits, _, aux = student(X)
            loss, soft, hard = kd_loss_topk(s_logits, tv, ti, Y, M, cfg.temperature, cfg.alpha)

        elif cfg.mode == "online":
            batch = [rng.choice(train_rows) for _ in range(cfg.batch_size)]
            X, Y, M = collate(batch, tok, cfg.max_len, device)
            with torch.no_grad():
                t_logits, _, _ = teacher(X)
            s_logits, _, aux = student(X)
            loss, soft, hard = kd_loss_full(s_logits, t_logits, Y, M, cfg.temperature, cfg.alpha)

        elif cfg.mode == "on_policy":
            # The student generates; the teacher grades its OWN trajectories.
            #
            # Prompts are sampled so they all share one encoded length. Truncating to
            # the shortest prompt in the batch instead (the obvious shortcut) silently
            # cuts the answer marker off the longer ones — measured: 6 of 8 prompts in
            # a typical batch lost their trailing "A:" — so the student would be
            # distilled on inputs that never occur at inference, which defeats the
            # entire point of going on-policy.
            enc_all = [(r, tok.encode(r["prompt"])) for r in train_rows]
            by_len: dict[int, list] = {}
            for r, e in enc_all:
                by_len.setdefault(len(e), []).append(e)
            L = max(by_len, key=lambda k: len(by_len[k]))       # the most common length
            pool = by_len[L]
            ctx = torch.tensor([rng.choice(pool) for _ in range(cfg.batch_size)],
                               dtype=torch.long, device=device)
            with torch.no_grad():
                gen = student.generate(ctx, max_new_tokens=cfg.max_new_tokens,
                                       temperature=cfg.temperature_sample)
                t_logits, _, _ = teacher(gen[:, :-1])
            student.train()          # generate() flipped us to eval; restore BEFORE
                                     # the training forward, or dropout stays off
            s_logits, _, aux = student(gen[:, :-1])
            Y = gen[:, 1:]
            M = torch.zeros_like(Y, dtype=torch.float)
            M[:, L - 1:] = 1.0                      # supervise only the generated part
            loss, soft, hard = kd_loss_full(s_logits, t_logits, Y, M, cfg.temperature, cfg.alpha)
        else:
            raise ValueError(f"unknown mode {cfg.mode!r}")

        total = loss if aux is None else loss + aux
        total.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        gnorm = float(gnorm.detach())
        opt.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            tracker.log({"kd/loss": float(loss.detach()), "kd/soft_kl": soft, "kd/hard_ce": hard,
                         "kd/lr": lr, "kd/grad_norm": gnorm}, step=step)
            print(f"  step {step:4d}/{cfg.steps}  loss {float(loss.detach()):.4f}  "
                  f"soft_kl {soft:.4f}  hard_ce {hard:.4f}  lr {lr:.2e}")

    wall = time.time() - t0
    summary = {"perf/wall_clock_s": round(wall, 1), "mode": cfg.mode,
               "params_total": student.n_params(), "params_active": student.n_active_params(),
               "teacher_params": teacher.n_params()}
    tracker.set_summary(summary)

    ckpt = None
    if cfg.save:
        ckpt = Path(cfg.out_dir) / name / "model.pt"
        torch.save({"model": student.state_dict(), "model_cfg": student.cfg.to_dict(),
                    "tokenizer": tokenizer_to_dict(tok), "distill_cfg": cfg.to_dict(), "config_hash": chash,
                    "parent": parent, "name": name, "summary": summary}, ckpt)
        tracker.log_artifact(ckpt, name=f"{name}-ckpt")
    tracker.finish()
    print(f"=== {name} done in {wall:.1f}s ===")

    return {"name": name, "model": student, "tokenizer": tok, "config_hash": chash,
            "summary": summary, "checkpoint": str(ckpt) if ckpt else None,
            "config": cfg.to_dict(), "device": device, "parent": parent}
