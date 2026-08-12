"""
GRPO — Group Relative Policy Optimization. RL post-training, minus the critic.

WHERE IT FITS
    SFT teaches format from demonstrations. RLHF/DPO aligns to human preference.
    GRPO optimises against a REWARD you can compute — a unit test passing, an
    answer matching, a format check. This family is called RLVR (RL with
    Verifiable Rewards) and it is what produced the recent jump in reasoning
    models. If a reward can be computed by code, you do not need a reward model
    at all, and most of RLHF's failure modes (reward hacking, reward-model drift)
    go away with it.

THE IDEA, in one paragraph
    PPO needs a *critic* — a second network the size of the policy, trained to
    predict the expected return of a partial generation, used to turn a
    single end-of-sequence reward into a per-token advantage. That is expensive
    and finicky. GRPO throws it away and gets the baseline from a GROUP instead:
    sample G completions for the SAME prompt, and score each one relative to its
    peers. If a completion beat its group's average, push its tokens up; if it
    lost, push them down.

        A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)

    That is the whole trick, and why it is called *group relative*. Only relative
    ranking within the group matters, so a reward function that is monotone but
    badly scaled still works — a real practical advantage.

THE OBJECTIVE (what the code below computes)
    For each sampled token, with A_i the advantage of the completion it belongs to:

        ratio      = pi_new(token) / pi_old(token)
        surrogate  = min(ratio * A, clip(ratio, 1-eps_low, 1+eps_high) * A)
        loss       = -mean_over_tokens(surrogate) + beta * KL(pi_new || pi_ref)

    ratio + clip is inherited from PPO: it is a trust region, stopping a single
    update from moving the policy so far that the samples you collected are no
    longer representative of it. With one gradient step per batch of samples
    (mu=1) the ratio is exactly 1 and clipping never fires — clipping only earns
    its keep when you reuse a batch for several updates, which is what mu>1 does.

    The KL term to a frozen REFERENCE model (usually the SFT checkpoint) is a
    leash: it stops the policy drifting into degenerate text that games the
    reward. Note that current TRL defaults beta=0.0 — i.e. no KL penalty at all —
    because on verifiable-reward tasks practitioners found the leash costs more
    capability than the drift it prevents. That is a real, contested tradeoff, not
    a settled fact: keep beta=0 for short runs on verifiable rewards, raise it if
    you see the model's general ability collapse while reward climbs.

KNOWN VARIANTS, and why they exist (all switchable below)
    Dr. GRPO       divide by a constant instead of the group's std. Dividing by
                   std up-weights easy/hard prompts whose rewards happen to have
                   low variance, which biases the gradient. `scale_rewards=False`.
    DAPO           four fixes: (1) clip-higher — a larger upper clip bound so
                   low-probability good tokens can still be raised; (2) dynamic
                   sampling — drop groups where every sample got the same reward,
                   since their advantages are all zero and they contribute pure
                   nothing while still costing generation; (3) token-level loss
                   normalisation — divide by total tokens across the group rather
                   than averaging per-sequence first, so long completions are not
                   gradient-diluted; (4) overlong reward shaping.
                   All four are implemented here. TRL's default loss_type is now
                   "dapo", i.e. (3).
    GSPO           use one SEQUENCE-level importance ratio instead of per-token
                   ratios. Motivation is specifically MoE: which experts fire can
                   change between the sampling policy and the updating policy, so
                   per-token ratios pick up huge variance that has nothing to do
                   with the policy improving. Relevant to you if your team's model
                   is MoE — `ratio_mode="sequence"`.

WHAT TO WATCH (these are the metrics that tell you an RL run is failing)
    reward/mean            should climb. If it is flat, check the reward function
                           actually fires — log its components separately.
    reward/std_within_group  if this goes to ~0, every sample in every group gets
                           the same score, all advantages are 0, and you are
                           burning GPUs to compute nothing. Either the task is
                           solved or the reward is too coarse.
    policy/clip_frac       a few percent is healthy; >20% means your steps are
                           fighting the trust region.
    policy/entropy         collapsing entropy = the policy is becoming
                           deterministic. Reward often looks great right up to
                           the point the model can only say one thing.
    gen/completion_len     watch for length hacking: a reward that accidentally
                           pays for verbosity will produce it.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import ARITH_MAX_OPERAND, arith_split
from .evaluate import CHAT_TEMPLATE, parse_int
from .tokenizer import tokenizer_to_dict
from .tracking import Tracker
from .utils import config_hash, cosine_lr, env_info, pick_device, run_name, set_seed


@dataclass
class GRPOConfig:
    # --- the RL knobs ---
    iterations: int = 60
    prompts_per_iter: int = 8         # distinct prompts per iteration
    group_size: int = 8               # G: completions sampled per prompt
    mu: int = 1                       # gradient steps per sampled batch (ratio==1 if 1)
    lr: float = 2e-4
    temperature: float = 1.0          # sampling temperature for rollouts
    max_new_tokens: int = 6
    epsilon_low: float = 0.2
    epsilon_high: float = 0.28        # DAPO "clip-higher"; set == epsilon_low for vanilla
    beta: float = 0.0                 # KL-to-reference coefficient (TRL default is 0.0)
    scale_rewards: bool = True        # False = Dr. GRPO (no /std)
    dynamic_sampling: bool = True     # DAPO: drop zero-variance groups
    loss_norm: str = "token"          # "token" (DAPO) | "sequence" (original GRPO)
    ratio_mode: str = "token"         # "token" (GRPO/DAPO) | "sequence" (GSPO)
    length_penalty_after: int = 0     # DAPO overlong shaping; 0 = off
    grad_clip: float = 1.0
    warmup_frac: float = 0.05         # RL wants a warmup too
    format_reward: float = 0.2        # weight of the format component

    # --- bookkeeping ---
    template: str = CHAT_TEMPLATE
    max_operand: int = ARITH_MAX_OPERAND
    seed: int = 1337
    log_every: int = 5
    eval_every: int = 20            # held-out eval every N iters (0 = off)
    eval_n: int = 100               # items per in-run eval; costs generation
    out_dir: str = "runs"
    project: str = "aienh"
    name_prefix: str = "grpo"
    run_name: str | None = None
    tracker: str = "auto"
    save: bool = True
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------

_NUM_ONLY = re.compile(r"^ ?\d+$")


def reward_arith(completion: str, gold: int, format_weight: float = 0.2,
                 length_penalty_after: int = 0, n_tokens: int = 0) -> dict:
    """
    A verifiable reward, decomposed. ALWAYS return the components, never just the
    total: a rising total with a falling correctness component is reward hacking,
    and a single scalar hides it perfectly.

    Here: 1.0 for the right answer, plus a small bonus for clean formatting, minus
    an optional soft penalty for overlong generations (DAPO's overlong shaping —
    it discourages rambling without hard-truncating the reward signal).
    """
    body = completion.split("<|endoftext|>")[0]
    pred = parse_int(body)
    correct = float(pred == gold)
    fmt = float(bool(_NUM_ONLY.match(body.rstrip())))
    total = correct + format_weight * fmt
    if length_penalty_after and n_tokens > length_penalty_after:
        total -= 0.05 * (n_tokens - length_penalty_after)
    return {"total": total, "correct": correct, "format": fmt, "pred": pred}


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_rollouts(policy, tok, items: list[dict], G: int, cfg: GRPOConfig, device):
    """
    Sample G completions per prompt. Grouped by prompt token length so the batch
    needs no left padding (see evaluate.generate_batch for why that matters).

    Each completion is truncated at its first EOS: tokens after EOS were never
    really "chosen" by the policy in any meaningful sense and must not be trained on.
    """
    policy.eval()
    rollouts = []
    by_len: dict[int, list[dict]] = {}
    for it in items:
        ids = tok.encode(it["prompt"])
        by_len.setdefault(len(ids), []).append({**it, "prompt_ids": ids})

    for L, group in by_len.items():
        rows = [g for g in group for _ in range(G)]
        ctx = torch.tensor([g["prompt_ids"] for g in rows], dtype=torch.long, device=device)
        out = policy.generate(ctx, cfg.max_new_tokens, temperature=cfg.temperature,
                              eos_id=tok.eos_id)
        for g, row in zip(rows, out):
            comp = row[L:].tolist()
            if tok.eos_id in comp:                       # keep EOS, drop everything after
                comp = comp[:comp.index(tok.eos_id) + 1]
            rollouts.append({**g, "completion_ids": comp,
                             "completion": tok.decode(comp)})
    return rollouts


def make_scoring_batch(rollouts: list[dict], tok, device):
    """Assemble rollouts into (x, y, completion_mask). Right-padded; the mask marks
    exactly the sampled completion tokens, which are the only ones RL touches."""
    seqs = [r["prompt_ids"] + r["completion_ids"] for r in rollouts]
    L = max(len(s) for s in seqs)
    pad = tok.pad_id
    X = torch.full((len(seqs), L - 1), pad, dtype=torch.long)
    Y = torch.full((len(seqs), L - 1), pad, dtype=torch.long)
    M = torch.zeros((len(seqs), L - 1), dtype=torch.float)
    for i, (r, s) in enumerate(zip(rollouts, seqs)):
        x, y = s[:-1], s[1:]
        X[i, :len(x)] = torch.tensor(x)
        Y[i, :len(y)] = torch.tensor(y)
        start = len(r["prompt_ids"]) - 1                 # y index of the first completion token
        M[i, start:start + len(r["completion_ids"])] = 1.0
    return X.to(device), Y.to(device), M.to(device)


def token_logprobs(model, X, Y):
    """log pi(y_t | x_<=t) for every position. Shape [B, T]."""
    logits, _, aux = model(X)
    logp = F.log_softmax(logits.float(), dim=-1)
    return torch.gather(logp, -1, Y.unsqueeze(-1)).squeeze(-1), logits, aux


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def grpo(policy, tok, cfg: GRPOConfig, device=None, parent: str | None = None,
         reference=None) -> dict:
    """
    `reference` is a frozen copy of the starting policy, used only for the KL
    penalty. It is skipped entirely when beta == 0.0 (the default), which also
    saves you a full forward pass per step.
    """
    device = device or pick_device()
    set_seed(cfg.seed)
    policy = policy.to(device)

    if cfg.beta > 0:
        if reference is None:
            import copy
            reference = copy.deepcopy(policy).to(device)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)

    opt = policy.configure_optimizer(cfg.lr, weight_decay=0.0)
    chash = config_hash(policy.cfg, cfg, {"parent": parent})
    name = cfg.run_name or run_name(cfg.name_prefix, chash, cfg.seed)
    rng = random.Random(cfg.seed)

    print(f"\n=== {name} (GRPO) ===")
    print(f"  parent={parent}  {cfg.iterations} iters x {cfg.prompts_per_iter} prompts "
          f"x G={cfg.group_size} = {cfg.iterations * cfg.prompts_per_iter * cfg.group_size} rollouts")
    print(f"  eps=({cfg.epsilon_low}, {cfg.epsilon_high}) beta={cfg.beta} mu={cfg.mu} "
          f"scale_rewards={cfg.scale_rewards} loss_norm={cfg.loss_norm} ratio={cfg.ratio_mode}")

    tracker = Tracker(project=cfg.project, name=name, mode=cfg.tracker, out_dir=cfg.out_dir,
                      tags=[*cfg.tags, "grpo", "rlvr"],
                      config={"grpo": cfg.to_dict(), "model": policy.cfg.to_dict(),
                              "parent": parent, "env": env_info(), "config_hash": chash})

    t0 = time.time()
    sample_rows = []
    best_held = 0.0

    for it in range(cfg.iterations):
        # A short warmup then gentle decay, same reasoning as supervised training:
        # the first updates come from the noisiest advantage estimates you will ever
        # have, and an RL run that destabilises early rarely recovers.
        lr_now = cosine_lr(it, cfg.iterations, cfg.lr, cfg.warmup_frac, min_ratio=0.3)
        for g in opt.param_groups:
            g["lr"] = lr_now

        # ---- 1. prompts ------------------------------------------------
        items = []
        while len(items) < cfg.prompts_per_iter:
            a, b = rng.randint(0, cfg.max_operand), rng.randint(0, cfg.max_operand)
            if arith_split(a, b) != "train":     # never RL on eval problems
                continue
            items.append({"prompt": cfg.template.format(q=f"{a} + {b} ="), "gold": a + b})

        # ---- 2. rollouts ----------------------------------------------
        rollouts = sample_rollouts(policy, tok, items, cfg.group_size, cfg, device)

        # ---- 3. reward -------------------------------------------------
        for r in rollouts:
            rw = reward_arith(r["completion"], r["gold"], cfg.format_reward,
                              cfg.length_penalty_after, len(r["completion_ids"]))
            r.update(reward=rw["total"], correct=rw["correct"], fmt=rw["format"])

        # ---- 4. group-relative advantages ------------------------------
        groups: dict[str, list[dict]] = {}
        for r in rollouts:
            groups.setdefault(r["prompt"], []).append(r)

        kept, skipped, group_stds = [], 0, []
        for _, grp in groups.items():
            rs = torch.tensor([g["reward"] for g in grp], dtype=torch.float)
            std = float(rs.std(unbiased=False))
            group_stds.append(std)
            if cfg.dynamic_sampling and std < 1e-6:
                # DAPO dynamic sampling: every sample scored the same, so every
                # advantage is exactly 0 and this group's gradient is identically
                # zero. Drop it — and log that you did, because if this number is
                # most of your batch, your effective batch size is a fiction.
                skipped += 1
                continue
            adv = rs - rs.mean()
            if cfg.scale_rewards:
                adv = adv / (rs.std(unbiased=False) + 1e-4)
            for g, a in zip(grp, adv.tolist()):
                g["adv"] = a
                kept.append(g)

        if not kept:
            print(f"  iter {it:4d}: all {len(groups)} groups had zero reward variance — "
                  f"nothing to learn from, skipping")
            tracker.log({"grpo/groups_skipped": skipped,
                         "reward/mean": float(np.mean([r['reward'] for r in rollouts]))}, step=it)
            continue

        # ---- 5. old logprobs (the policy that produced the samples) ----
        X, Y, M = make_scoring_batch(kept, tok, device)
        A = torch.tensor([r["adv"] for r in kept], dtype=torch.float, device=device).unsqueeze(-1)
        with torch.no_grad():
            old_logp, _, _ = token_logprobs(policy, X, Y)
            ref_logp = None
            if cfg.beta > 0:
                ref_logp, _, _ = token_logprobs(reference, X, Y)

        # ---- 6. update -------------------------------------------------
        policy.train()
        for _ in range(cfg.mu):
            new_logp, logits, aux = token_logprobs(policy, X, Y)

            if cfg.ratio_mode == "sequence":
                # GSPO: one length-normalised ratio for the whole sequence, shared
                # by all its tokens. Kills the per-token variance that MoE routing
                # changes inject.
                seq_delta = ((new_logp - old_logp) * M).sum(-1) / M.sum(-1).clamp_min(1.0)
                ratio = seq_delta.exp().unsqueeze(-1).expand_as(new_logp)
            else:
                ratio = (new_logp - old_logp).exp()

            unclipped = ratio * A
            clipped = ratio.clamp(1 - cfg.epsilon_low, 1 + cfg.epsilon_high) * A
            surrogate = torch.min(unclipped, clipped)

            if cfg.loss_norm == "token":
                # DAPO: one denominator for the whole batch of tokens.
                pg_loss = -(surrogate * M).sum() / M.sum().clamp_min(1.0)
            else:
                # Original GRPO: average within each sequence first, then across
                # sequences — which makes each token in a long completion count
                # for less.
                per_seq = (surrogate * M).sum(-1) / M.sum(-1).clamp_min(1.0)
                pg_loss = -per_seq.mean()

            kl_val = 0.0
            loss = pg_loss
            if cfg.beta > 0:
                # k3 estimator (Schulman): exp(d) - d - 1 with d = ref - new.
                # Always >= 0, unbiased, far lower variance than (new - ref)^2.
                d = ref_logp - new_logp
                kl_tok = d.exp() - d - 1
                kl = (kl_tok * M).sum() / M.sum().clamp_min(1.0)
                kl_val = float(kl)
                loss = loss + cfg.beta * kl
            if aux is not None:
                loss = loss + aux

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                policy.parameters(), cfg.grad_clip).detach())
            opt.step()

        # ---- 7. metrics ------------------------------------------------
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            ent_tok = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
            entropy = float((ent_tok * M).sum() / M.sum().clamp_min(1.0))
            # A token is *clipped* only when clipping actually changed the
            # surrogate, which depends on the sign of the advantage as well as the
            # ratio. Counting every token outside the band (the obvious version)
            # roughly doubles the number, so the ">20% is trouble" rule of thumb
            # would fire at a true ~10%. This is what TRL measures.
            clip_frac = float((((surrogate < unclipped).float()) * M).sum()
                              / M.sum().clamp_min(1.0))
            row = {
                "reward/mean": float(np.mean([r["reward"] for r in rollouts])),
                "reward/correct_frac": float(np.mean([r["correct"] for r in rollouts])),
                "reward/format_frac": float(np.mean([r["fmt"] for r in rollouts])),
                "reward/std_within_group": float(np.mean(group_stds)),
                "adv/abs_mean": float(A.abs().mean()),
                "policy/loss": float(pg_loss.detach()), "policy/kl_to_ref": kl_val,
                "policy/ratio_mean": float((ratio * M).sum() / M.sum().clamp_min(1.0)),
                "policy/clip_frac": clip_frac, "policy/entropy": entropy,
                "policy/grad_norm": gnorm, "policy/lr": lr_now,
                "gen/completion_len": float(np.mean([len(r["completion_ids"]) for r in kept])),
                "grpo/groups_skipped": skipped,
                "grpo/rollouts_used": len(kept),
            }
        tracker.log(row, step=it)

        # ---- 7b. held-out eval DURING the run ---------------------------
        # Reward is a claim about the reward function, not about the model. An RL run
        # that only watches reward cannot see itself degrading a held-out metric —
        # which is the most common way RL runs fail. So measure it here, on the test
        # side of the problem space, while there is still time to stop.
        if cfg.eval_every and (it % cfg.eval_every == 0 or it == cfg.iterations - 1):
            from .evaluate import task_arith_exact
            held = task_arith_exact(policy, tok, device, n=cfg.eval_n, seed=1234,
                                    template=cfg.template, split="test")
            policy.train()
            tracker.log({"eval/arith_exact": held.value,
                         "eval/arith_exact_stderr": held.stderr}, step=it)
            flag = ""
            if held.value < best_held - 3 * max(held.stderr, 1e-6):
                flag = (f"  !! {best_held - held.value:.3f} below this run's best "
                        f"({best_held:.3f}), >3 stderr: the run is damaging the model")
            best_held = max(best_held, held.value)
            print(f"    iter {it:4d} held-out exact_match {held.value:.3f} "
                  f"+/- {held.stderr:.3f} (n={held.n}){flag}")

        if it % cfg.log_every == 0 or it == cfg.iterations - 1:
            print(f"  iter {it:4d}/{cfg.iterations}  reward {row['reward/mean']:.3f}  "
                  f"acc {row['reward/correct_frac']:.3f}  ent {entropy:.3f}  "
                  f"clip {clip_frac:.2%}  skipped {skipped}/{len(groups)}  "
                  f"len {row['gen/completion_len']:.1f}")
            if len(sample_rows) < 40:
                for r in kept[:2]:
                    sample_rows.append([it, r["prompt"], r["completion"], r["gold"],
                                        r["reward"], round(r["adv"], 3)])

    wall = time.time() - t0
    tracker.log_table("rollout_samples",
                      ["iter", "prompt", "completion", "gold", "reward", "advantage"],
                      sample_rows)
    summary = {"perf/wall_clock_s": round(wall, 1),
               "params_total": policy.n_params(), "params_active": policy.n_active_params()}
    tracker.set_summary(summary)

    ckpt = None
    if cfg.save:
        ckpt = Path(cfg.out_dir) / name / "model.pt"
        torch.save({"model": policy.state_dict(), "model_cfg": policy.cfg.to_dict(),
                    "tokenizer": tokenizer_to_dict(tok), "grpo_cfg": cfg.to_dict(), "config_hash": chash,
                    "parent": parent, "name": name, "summary": summary}, ckpt)
        tracker.log_artifact(ckpt, name=f"{name}-ckpt")
    tracker.finish()
    print(f"=== {name} done in {wall:.1f}s ===")

    return {"name": name, "model": policy, "tokenizer": tok, "config_hash": chash,
            "summary": summary, "checkpoint": str(ckpt) if ckpt else None,
            "config": cfg.to_dict(), "device": device, "parent": parent}
