"""
The model: a decoder-only transformer, with a switch for dense or MoE feed-forward.

Read this file top to bottom once. It is ~300 lines and it is the entire "how an
LLM works" answer in executable form. Everything else in the repo is plumbing
around it.

Shapes are annotated throughout with these letters:
    B  batch size
    T  sequence length (time / tokens)
    C  n_embd, the model/residual width
    H  n_head
    V  vocab size
    E  n_experts

The one-sentence version: embed tokens, then repeatedly (a) let positions look at
earlier positions and mix information — attention, (b) transform each position
independently through a wider layer — the MLP, then project the final vector to a
score for every token in the vocabulary and train it to put the highest score on
whatever token actually came next.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 512
    block_size: int = 128          # max context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128              # must be divisible by n_head
    dropout: float = 0.0
    bias: bool = False             # modern LLMs mostly drop biases; costs nothing to keep off
    norm: str = "rms"              # "rms" (Llama/Qwen) or "layer" (GPT-2)
    pos: str = "rope"              # "rope" (modern) or "learned" (GPT-2)
    tie_weights: bool = True       # share input embedding and output projection

    # --- MoE ---------------------------------------------------------------
    moe: bool = False
    n_experts: int = 4
    top_k: int = 2                 # experts activated per token
    n_shared_experts: int = 0      # DeepSeek-style always-on expert(s)
    aux_loss_coef: float = 0.01    # load-balancing loss weight
    router_z_loss_coef: float = 0.001
    moe_every: int = 1             # put MoE in every Nth block (1 = all of them)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """x / rms(x) * gain. No mean subtraction, no bias: fewer ops than LayerNorm
    and empirically just as good, which is why every recent model uses it."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def make_norm(cfg: ModelConfig) -> nn.Module:
    return RMSNorm(cfg.n_embd) if cfg.norm == "rms" else nn.LayerNorm(cfg.n_embd, bias=cfg.bias)


def build_rope_cache(head_dim: int, max_T: int, base: float = 10000.0, device=None):
    """Rotary position embeddings.

    Instead of adding a position vector to the token vector, RoPE *rotates* each
    2-dimensional slice of q and k by an angle proportional to the position. The
    dot product q·k then depends only on (position_q - position_k), so attention
    becomes naturally relative — and the same weights extrapolate to longer
    contexts than they were trained on, which absolute learned positions cannot do.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_T, device=device).float()
    freqs = torch.outer(t, inv_freq)            # [T, head_dim/2]
    return torch.cos(freqs), torch.sin(freqs)   # each [T, head_dim/2]


def apply_rope(x, cos, sin):
    """x: [B, H, T, head_dim] -> rotated. cos/sin: [T, head_dim/2]."""
    T = x.size(-2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)     # [1, 1, T, hd/2]
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]         # even / odd dims form the 2-D pairs
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class CausalSelfAttention(nn.Module):
    """
    Every position builds a query ("what am I looking for?"), a key ("what do I
    offer?") and a value ("what will I hand over?"). Scores = q·k / sqrt(d),
    masked so position t can only see positions <= t, softmaxed into weights,
    then used to average the values. That mask is the only reason a decoder LM
    can be trained on all T positions at once instead of one at a time.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.use_rope = cfg.pos == "rope"

    def forward(self, x, cos=None, sin=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # [B, T, C] -> [B, H, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # Fused kernel: does the mask, softmax and matmuls without materialising
        # the [B, H, T, T] score matrix. This is "flash attention" in one call.
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward: widen 4x, non-linearity, project back.
    This is where most of a dense model's parameters live (~2/3 of them)."""

    def __init__(self, cfg: ModelConfig, hidden_mult: int = 4):
        super().__init__()
        hidden = hidden_mult * cfg.n_embd
        self.fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class MoEFeedForward(nn.Module):
    """
    Mixture of Experts, replacing the single MLP with E of them plus a router.

    Per token: the router scores all E experts, the top-k are selected, each
    runs the token through its own MLP, and the outputs are combined weighted by
    the (renormalised) router probabilities. Each expert here is 1/k the width of a
    dense MLP, so with E=8, k=2 you get 4x the FFN parameters at the same FFN FLOPs
    per token as a dense model — that is the entire economic argument for MoE.

    The catch, and the thing you will actually debug: routing is a discrete
    choice, so nothing makes experts get equal traffic. Left alone the router
    collapses onto a few experts and the rest are dead weight. Hence two
    auxiliary losses:

      load-balancing loss   E * sum_i (fraction of tokens to expert i)
                                   * (mean router probability of expert i)
                            minimised when both are uniform (= 1/E), so it pushes
                            traffic flat.
      router z-loss         penalises large router logits, which keeps the
                            softmax numerically sane in low precision.

    `self.stats` holds last-forward diagnostics — log them, they are how you see
    a collapse happening instead of guessing.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.top_k = min(cfg.top_k, cfg.n_experts)
        self.router = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        # Each expert is 1/top_k the width of a dense MLP, so the k experts a token
        # actually runs sum to exactly one dense MLP's FLOPs. That makes the
        # dense-vs-MoE comparison in this repo genuinely FLOP-matched at the same
        # depth and width — the only comparison from which "MoE wins" means anything.
        #
        # This is the "fine-grained expert" design (DeepSeek-V3 and successors).
        # Mixtral-style MoE instead gives every expert the FULL dense width, so top-2
        # routing costs 2x the FFN FLOPs of a same-width dense model; there, the
        # matched comparison is against a dense model of the MoE's ACTIVE size, not
        # its width. Both are legitimate; they answer different questions, and
        # conflating them is how MoE results get oversold.
        mult = max(1, 4 // max(1, self.top_k))
        self.experts = nn.ModuleList([MLP(cfg, hidden_mult=mult)
                                      for _ in range(cfg.n_experts)])
        # A shared expert is ADDITIONAL always-on capacity: it adds FLOPs on top of
        # the routed budget. Set n_shared_experts=0 for a strictly matched comparison.
        self.shared = nn.ModuleList(
            [MLP(cfg, hidden_mult=mult) for _ in range(cfg.n_shared_experts)]
        )
        self.stats: dict[str, torch.Tensor | float] = {}

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(B * T, C)                       # flatten tokens: [N, C]
        N = xf.size(0)

        logits = self.router(xf)                       # [N, E]
        probs = F.softmax(logits, dim=-1)
        topv, topi = probs.topk(self.top_k, dim=-1)    # [N, k]
        topv = topv / topv.sum(-1, keepdim=True)       # renormalise the kept mass

        out = torch.zeros_like(xf)
        # Loop over experts (not tokens): gather the tokens routed here, run one
        # batched matmul, scatter back. This is the readable version of what a
        # production kernel does with a permutation + grouped GEMM.
        counts = torch.zeros(self.n_experts, device=x.device)
        for e, expert in enumerate(self.experts):
            hit = (topi == e)                          # [N, k] bool
            if not hit.any():
                continue
            tok_idx, slot_idx = hit.nonzero(as_tuple=True)
            counts[e] = tok_idx.numel()
            weight = topv[tok_idx, slot_idx].unsqueeze(-1)   # [n_e, 1]
            out.index_add_(0, tok_idx, expert(xf[tok_idx]) * weight)

        for expert in self.shared:                     # always-on capacity
            out = out + expert(xf)

        # --- auxiliary losses ------------------------------------------------
        frac = counts / max(1, N * self.top_k)         # f_i: share of routed slots
        mean_prob = probs.mean(0)                      # P_i: mean router prob
        aux = self.n_experts * (frac * mean_prob).sum() * self.cfg.aux_loss_coef
        z_loss = logits.logsumexp(-1).pow(2).mean() * self.cfg.router_z_loss_coef

        self.stats = {
            "expert_frac": frac.detach(),
            # 1.0 = perfectly balanced, 0.0 = fully collapsed onto one expert
            "balance": (1.0 - (frac - 1.0 / self.n_experts).abs().sum() / (2 * (1 - 1.0 / self.n_experts))).detach(),
            "router_entropy": (-(probs * probs.clamp_min(1e-9).log()).sum(-1)).mean().detach(),
            "aux_loss": aux.detach(),
        }
        return out.view(B, T, C), aux + z_loss


class Block(nn.Module):
    """Pre-norm residual block: x = x + attn(norm(x)); x = x + ffn(norm(x)).

    Pre-norm (normalise *before* the sublayer) is what makes deep stacks trainable
    without warmup tricks — the residual path stays an identity, so gradients
    reach layer 0 unattenuated."""

    def __init__(self, cfg: ModelConfig, use_moe: bool):
        super().__init__()
        self.n1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.n2 = make_norm(cfg)
        self.use_moe = use_moe
        self.ffn = MoEFeedForward(cfg) if use_moe else MLP(cfg)

    def forward(self, x, cos=None, sin=None):
        x = x + self.attn(self.n1(x), cos, sin)
        if self.use_moe:
            h, aux = self.ffn(self.n2(x))
            return x + h, aux
        return x + self.ffn(self.n2(x)), None


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = (
            nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos == "learned" else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            Block(cfg, use_moe=cfg.moe and (i % cfg.moe_every == 0))
            for i in range(cfg.n_layer)
        ])
        self.norm_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            # Same matrix for "token -> vector" and "vector -> token score".
            # Saves V*C parameters and usually helps small models.
            self.lm_head.weight = self.tok_emb.weight

        if cfg.pos == "rope":
            cos, sin = build_rope_cache(cfg.n_embd // cfg.n_head, cfg.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale down the projections that write into the residual stream, so the
        # stream's variance doesn't grow with depth. (GPT-2 paper trick.)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # -- parameter accounting -------------------------------------------
    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n

    def n_active_params(self) -> int:
        """Parameters actually touched per token. For a dense model this equals
        n_params. For MoE it is much smaller, and it is the number that predicts
        inference cost — quote both, never just one. (Shared experts always run, so
        they count as active.)"""
        total = self.n_params()
        if not self.cfg.moe:
            return total
        inactive = 0
        for blk in self.blocks:
            if not blk.use_moe:
                continue
            per_expert = sum(p.numel() for p in blk.ffn.experts[0].parameters())
            skipped = self.cfg.n_experts - min(self.cfg.top_k, self.cfg.n_experts)
            inactive += per_expert * skipped
        return total - inactive

    # -- forward ---------------------------------------------------------
    def forward(self, idx, targets=None, loss_mask=None, reduction: str = "mean"):
        """
        idx      [B, T] int64 token ids
        targets  [B, T] int64 next-token ids, or None for inference
        loss_mask[B, T] 1.0 where the token should contribute to the loss.
                        This single argument is the difference between
                        pretraining (mask everything in) and SFT (mask the
                        prompt out).

        Returns (logits [B, T, V], loss or None, aux_loss or None).
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)

        cos = sin = None
        if self.cfg.pos == "rope":
            cos, sin = self.rope_cos, self.rope_sin

        aux_total = None
        for blk in self.blocks:
            x, aux = blk(x, cos, sin)
            if aux is not None:
                aux_total = aux if aux_total is None else aux_total + aux

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Cross-entropy over the vocabulary = -log(probability the model gave
            # the correct next token). Averaged over tokens, this is the training
            # loss; exponentiated, it is perplexity.
            flat_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="none",
            )
            if loss_mask is not None:
                m = loss_mask.reshape(-1)
                loss = (flat_loss * m).sum() / m.sum().clamp_min(1.0)
            elif reduction == "mean":
                loss = flat_loss.mean()
            else:
                loss = flat_loss.view(B, T)
        return logits, loss, aux_total

    def moe_stats(self) -> dict[str, float]:
        """Averaged router diagnostics from the last forward pass."""
        keys = ("balance", "router_entropy")
        acc: dict[str, list[float]] = {k: [] for k in keys}
        fracs = []
        for blk in self.blocks:
            if blk.use_moe and blk.ffn.stats:
                for k in keys:
                    acc[k].append(float(blk.ffn.stats[k]))
                fracs.append(blk.ffn.stats["expert_frac"])
        if not fracs:
            return {}
        out = {f"moe/{k}": sum(v) / len(v) for k, v in acc.items() if v}
        mean_frac = torch.stack(fracs).mean(0)
        for i, f in enumerate(mean_frac.tolist()):
            out[f"moe/expert_{i}_frac"] = f
        return out

    # -- generation ------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, greedy: bool = False,
                 eos_id: int | None = None):
        """
        Autoregressive sampling: predict, append, repeat. Naive (no KV cache), so
        it is O(T^2) — fine here, and the place you would optimise first for real
        inference.

        temperature < 1 sharpens the distribution (more deterministic),
        > 1 flattens it (more diverse). greedy=True always takes the argmax,
        which is what you want when *measuring* a model, since it removes
        sampling noise from your benchmark numbers.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]                  # only the last position matters
            if greedy:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None:
                    kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx

    # -- optimizer -------------------------------------------------------
    def configure_optimizer(self, lr: float, weight_decay: float = 0.1,
                            betas: tuple[float, float] = (0.9, 0.95)):
        """
        AdamW with two parameter groups.

        Weight decay pulls weights toward zero (a regulariser). You apply it to
        matrices that do the actual mixing, and NOT to norm gains, biases or
        embeddings — decaying those just fights the model's ability to represent
        scale, and every reference implementation excludes them.
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)


def build_model(cfg: ModelConfig) -> GPT:
    return GPT(cfg)


if __name__ == "__main__":  # python -m aienh.model
    for moe in (False, True):
        cfg = ModelConfig(vocab_size=256, n_layer=2, n_head=4, n_embd=128,
                          moe=moe, n_experts=8, top_k=2)
        m = build_model(cfg)
        x = torch.randint(0, 256, (2, 32))
        logits, loss, aux = m(x, targets=x)
        print(f"moe={moe!s:5s} params={m.n_params():,} active={m.n_active_params():,} "
              f"loss={loss.item():.3f} "
              f"aux={None if aux is None else round(float(aux.detach()), 5)}")
        if moe:
            print("        ", {k: round(v, 3) for k, v in m.moe_stats().items()})
