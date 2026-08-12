"""
Tests that check the things that silently produce wrong numbers.

Run:  pytest -q tests/    (or:  python tests/test_smoke.py  with no pytest installed)

The bias here is deliberate: these are not "does it run" tests, they are "is the
maths right" tests. Every one of them corresponds to a mistake that produces
plausible-looking output — a shifted target, a mask that leaks the prompt, a
loglikelihood that scores the wrong position. Those are the bugs that cost weeks,
because nothing crashes and the loss still goes down.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from aienh.data import (build_dataset, contamination_report, exact_dedup, mix_corpora,
                        near_dedup, normalise, pack, quality_filter, split_docs)
from aienh.evaluate import binom_stderr, loglikelihood, parse_int, points_from_results, suite_hash
from aienh.grpo import make_scoring_batch, reward_arith
from aienh.model import ModelConfig, build_model
from aienh.sft import collate, encode_example
from aienh.tokenizer import build_tokenizer, tokenizer_from_dict, tokenizer_to_dict
from aienh.utils import config_hash, cosine_lr


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

def test_char_roundtrip():
    tok = build_tokenizer("char", ["hello world 123"])
    for s in ("hello world 123", "Q: 7 + 8 =\nA: 15", "!@#$%"):
        assert tok.decode(tok.encode(s)) == s, s


def test_bpe_roundtrip_and_compression():
    text = "the cat sat on the mat. " * 200
    tok = build_tokenizer("bpe", [text], vocab_size=600)
    assert tok.decode(tok.encode("the cat sat on the mat.")) == "the cat sat on the mat."
    # BPE must be strictly shorter than char-level on text it was trained on,
    # otherwise the merges learned nothing.
    char = build_tokenizer("char", [text])
    assert len(tok.encode(text[:400])) < len(char.encode(text[:400]))


def test_bpe_handles_unseen_bytes():
    """Byte-level means nothing is unrepresentable — the reason to prefer it."""
    tok = build_tokenizer("bpe", ["only ascii here"], vocab_size=300)
    assert tok.decode(tok.encode("emoji \U0001f600 and é")) == "emoji \U0001f600 and é"


def test_tokenizer_serialisation():
    for kind in ("char", "bpe"):
        tok = build_tokenizer(kind, ["some text 42"], vocab_size=300)
        clone = tokenizer_from_dict(tokenizer_to_dict(tok))
        assert clone.encode("some text 42") == tok.encode("some text 42")
        assert clone.vocab_size == tok.vocab_size


# ---------------------------------------------------------------------------
# data pipeline
# ---------------------------------------------------------------------------

def test_pack_is_a_shifted_pair():
    """x[t] must predict y[t] where y is x shifted by one. Get this wrong by one and
    the model trains fine, converges, and is useless."""
    tok = build_tokenizer("char", ["abcdefghij"])
    blocks = pack(["abcdefghij"] * 40, tok, block_size=8)
    assert blocks.shape[1] == 9                       # block_size + 1
    x, y = blocks[:, :-1], blocks[:, 1:]
    assert (x[0, 1:] == y[0, :-1]).all()


def test_pack_rejects_too_small_corpus():
    tok = build_tokenizer("char", ["abc"])
    try:
        pack(["abc"], tok, block_size=1024)
    except ValueError as e:
        assert "too small" in str(e)
    else:
        raise AssertionError("expected ValueError for an under-filled block")


def test_filters_do_what_they_say():
    assert quality_filter(["ok"], min_chars=12) == []
    assert quality_filter(["buy now " * 20]) == []                  # repetition
    assert quality_filter(["!!!!!!!!!!!!!!!!!!"]) == []             # alpha ratio
    assert quality_filter(["the dog ran to the park quietly ."]) != []


def test_dedup_levels():
    """Exact dedup catches copies; near-dedup catches copies with boilerplate glued
    on. Note the document length: SimHash distance is a *fraction-of-content*
    signal, so appending three words to an 8-word document is a big change
    (measured hamming 13) while appending the same three words to a 34-word
    document is a small one (measured hamming 4). Near-dedup is unreliable on very
    short documents, and that is a property of the method, not a bug to tune away."""
    long_doc = ("the dog ran to the park and found a red ball while the sun was "
                "going down and everyone laughed at the noise it made today")
    docs = [long_doc, long_doc, long_doc + " Read more ."]
    assert len(exact_dedup(docs)) == 2
    assert len(near_dedup(exact_dedup(docs), max_hamming=8)) == 1
    # a genuinely different document survives the same threshold
    other = "a robot carried the wooden chair past the quiet river until the bell rang"
    assert len(near_dedup([long_doc, other], max_hamming=8)) == 2


def test_normalise_is_idempotent():
    s = normalise("  a  b\r\n\n\n\nc  ")
    assert normalise(s) == s


def test_split_is_disjoint():
    docs = [f"document number {i} with some words" for i in range(100)]
    tr, va = split_docs(docs, val_frac=0.2, seed=0)
    assert len(tr) + len(va) == 100
    assert not (set(tr) & set(va))


def test_mixture_respects_weights():
    src = {"a": [f"a{i}" for i in range(1000)], "b": [f"b{i}" for i in range(1000)]}
    out = mix_corpora(src, {"a": 0.75, "b": 0.25}, total_docs=400, seed=0)
    frac_a = sum(d.startswith("a") for d in out) / len(out)
    assert abs(frac_a - 0.75) < 0.02, frac_a


def test_contamination_is_detected():
    train = ["the quick brown fox jumps over the lazy dog again and again today"]
    rep = contamination_report(train, train, n=5)
    assert rep["exact_overlap_pct"] == 100.0 and rep["ngram_overlap_pct"] == 100.0
    clean = contamination_report(train, ["completely different words appear here now ok"], n=5)
    assert clean["ngram_overlap_pct"] == 0.0


def test_build_dataset_end_to_end():
    ds = build_dataset("arithmetic", n_docs=2000, block_size=32, verbose=False)
    assert ds["train"].shape[1] == 33 and ds["n_train_tokens"] > 0
    assert ds["report"].to_dict()["exact_dedup"]["in"] > 0


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def _tiny(**kw):
    cfg = ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, **kw)
    return build_model(cfg)


def test_forward_shapes_and_loss():
    m = _tiny()
    x = torch.randint(0, 64, (3, 16))
    logits, loss, aux = m(x, targets=x)
    assert logits.shape == (3, 16, 64) and aux is None
    # An untrained model is uniform over the vocab, so loss ~ ln(V).
    assert abs(float(loss) - math.log(64)) < 0.6


def test_causality():
    """Changing a LATER token must not change an EARLIER position's logits. If this
    fails, the model can see the future and every metric you produce is fiction."""
    m = _tiny().eval()
    x = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        a, _, _ = m(x)
        x2 = x.clone()
        x2[0, -1] = (x[0, -1] + 1) % 64
        b, _, _ = m(x2)
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5)
    assert not torch.allclose(a[0, -1], b[0, -1], atol=1e-5)


def test_loss_mask_ignores_masked_positions():
    m = _tiny()
    x = torch.randint(0, 64, (2, 16))
    mask = torch.zeros(2, 16)
    mask[:, -1] = 1.0
    _, masked, _ = m(x, targets=x, loss_mask=mask)
    logits, _, _ = m(x)
    manual = F.cross_entropy(logits[:, -1].float(), x[:, -1])
    assert abs(float(masked) - float(manual)) < 1e-4


def test_moe_active_params_and_aux_loss():
    m = _tiny(moe=True, n_experts=8, top_k=2)
    x = torch.randint(0, 64, (2, 16))
    _, _, aux = m(x, targets=x)
    assert aux is not None and float(aux) > 0
    assert m.n_active_params() < m.n_params()
    stats = m.moe_stats()
    assert abs(sum(v for k, v in stats.items() if "_frac" in k) - 1.0) < 1e-3
    assert 0.0 <= stats["moe/balance"] <= 1.0


def test_moe_router_is_balanced_at_init():
    """At init the router is random, so traffic should be roughly uniform. A
    collapsed router at step 0 means the init or the top-k is wrong."""
    m = _tiny(moe=True, n_experts=4, top_k=1)
    m(torch.randint(0, 64, (8, 16)))
    fr = [v for k, v in m.moe_stats().items() if "_frac" in k]
    assert max(fr) < 0.6, fr


def test_weight_tying_and_optimizer_groups():
    m = _tiny(tie_weights=True)
    assert m.lm_head.weight is m.tok_emb.weight
    opt = m.configure_optimizer(1e-3, weight_decay=0.1)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])


def test_rope_and_learned_positions_both_run():
    for pos in ("rope", "learned"):
        m = _tiny(pos=pos)
        out, loss, _ = m(torch.randint(0, 64, (2, 16)), targets=torch.randint(0, 64, (2, 16)))
        assert torch.isfinite(loss)


def test_generate_is_deterministic_when_greedy():
    m = _tiny().eval()
    x = torch.randint(0, 64, (1, 4))
    a = m.generate(x, 6, greedy=True)
    b = m.generate(x, 6, greedy=True)
    assert torch.equal(a, b) and a.shape == (1, 10)


# ---------------------------------------------------------------------------
# sft / grpo mechanics
# ---------------------------------------------------------------------------

def test_sft_mask_covers_response_and_eos_only():
    tok = build_tokenizer("char", ["Q: 1 + 2 =\nA: 3"])
    ex = {"prompt": "Q: 1 + 2 =\nA:", "response": " 3"}
    x, y, mask = encode_example(ex, tok, max_len=64, mask_prompt=True)
    n_resp = len(tok.encode(" 3")) + 1                   # +1 for EOS
    assert sum(mask) == n_resp
    # the masked-in positions must be the LAST ones
    assert all(m == 1.0 for m in mask[-n_resp:])
    x2, y2, m2 = encode_example(ex, tok, max_len=64, mask_prompt=False)
    assert sum(m2) == len(y2)


def test_collate_pads_without_polluting_the_loss():
    tok = build_tokenizer("char", ["Q: 1 + 2 =\nA: 3"])
    rows = [{"prompt": "Q: 1 + 2 =\nA:", "response": " 3"},
            {"prompt": "Q: 11 + 22 =\nA:", "response": " 33"}]
    X, Y, M = collate(rows, tok, 64, torch.device("cpu"))
    assert X.shape == Y.shape == M.shape
    assert float(M.sum()) == sum(sum(encode_example(r, tok, 64)[2]) for r in rows)


def test_grpo_scoring_mask_marks_exactly_the_completion():
    tok = build_tokenizer("char", ["Q: 1 + 2 =\nA: 3"])
    roll = [{"prompt_ids": tok.encode("Q: 1 + 2 =\nA:"), "completion_ids": tok.encode(" 3") + [tok.eos_id]},
            {"prompt_ids": tok.encode("Q: 11 + 2 =\nA:"), "completion_ids": tok.encode(" 13")}]
    X, Y, M = make_scoring_batch(roll, tok, torch.device("cpu"))
    for i, r in enumerate(roll):
        assert float(M[i].sum()) == len(r["completion_ids"])
        # the marked targets must literally be the completion tokens
        marked = Y[i][M[i] > 0].tolist()
        assert marked == r["completion_ids"]


def test_group_advantages_are_zero_mean():
    r = torch.tensor([1.2, 0.0, 1.0, 0.2])
    adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
    assert abs(float(adv.mean())) < 1e-5
    assert abs(float(adv.std(unbiased=False)) - 1.0) < 1e-3


def test_reward_components_are_separable():
    good = reward_arith(" 62<|endoftext|>", 62)
    bad_fmt = reward_arith(" the answer is 62", 62)
    wrong = reward_arith(" 61", 62)
    assert good["correct"] == 1.0 and good["format"] == 1.0
    assert bad_fmt["correct"] == 1.0 and bad_fmt["format"] == 0.0   # right, badly formatted
    assert wrong["correct"] == 0.0 and wrong["format"] == 1.0       # wrong, well formatted
    assert good["total"] > bad_fmt["total"] > wrong["total"]


def test_kd_offline_and_online_agree_without_truncation():
    """
    With k = vocab size there is no truncation, so the top-k loss must equal the
    full-distribution loss — and a student identical to the teacher must score 0.

    This is the test that catches the support mismatch: if the student's log-probs are
    gathered from a full-vocabulary softmax while the teacher is renormalised over k,
    the "KL" has a large k-dependent floor. Measured before the fix: a student with
    logits identical to the teacher's scored 6.91 at k=16 instead of 0.0, and the
    offline and online losses were not on the same scale at all.
    """
    from aienh.distill import kd_loss_full, kd_loss_topk
    torch.manual_seed(0)
    B, T, V = 2, 5, 11
    teacher = torch.randn(B, T, V)
    mask = torch.ones(B, T)
    targets = torch.randint(0, V, (B, T))

    # identical student => zero soft loss, in both implementations
    _, soft_full, _ = kd_loss_full(teacher.clone(), teacher, targets, mask, T=2.0, alpha=1.0)
    vals, idx = teacher.topk(V, dim=-1)
    _, soft_topk, _ = kd_loss_topk(teacher.clone(), vals, idx, targets, mask, T=2.0, alpha=1.0)
    assert abs(soft_full) < 1e-4, soft_full
    assert abs(soft_topk) < 1e-4, soft_topk

    # and they agree on a student that differs
    student = torch.randn(B, T, V)
    _, a, _ = kd_loss_full(student, teacher, targets, mask, T=2.0, alpha=1.0)
    _, b, _ = kd_loss_topk(student, vals, idx, targets, mask, T=2.0, alpha=1.0)
    assert abs(a - b) < 1e-4, (a, b)


def test_moe_is_flop_matched_to_dense():
    """Each expert is 1/top_k the dense width, so the k experts a token actually runs
    sum to one dense MLP. Without this the dense-vs-MoE comparison in the pipeline is
    not FLOP-matched and 'MoE wins' means nothing."""
    dense = _tiny()
    moe = _tiny(moe=True, n_experts=8, top_k=2)
    dense_ffn = sum(p.numel() for p in dense.blocks[0].ffn.parameters())
    expert = sum(p.numel() for p in moe.blocks[0].ffn.experts[0].parameters())
    assert abs(expert * 2 - dense_ffn) <= 2 * 32, (expert, dense_ffn)   # bias-free: exact
    assert moe.n_params() > dense.n_params()             # more total capacity
    # active is dense + routers only, i.e. within a few percent
    assert moe.n_active_params() < dense.n_params() * 1.05


def test_grpo_update_moves_probabilities_the_right_way():
    """
    The test that says the RL implementation is actually correct: after a few GRPO
    steps, a completion with POSITIVE advantage must become more likely and one with
    NEGATIVE advantage less likely.

    Worth having because a sign error here does not crash, does not NaN, and produces
    a training curve that looks plausible while the model gets worse. If you ever
    suspect your RL code, run this shape of test before you tune anything.
    """
    from aienh.grpo import make_scoring_batch, token_logprobs
    torch.manual_seed(0)
    tok = build_tokenizer("char", ["Q: 1 + 2 =\nA: 3"])
    model = build_model(ModelConfig(vocab_size=tok.vocab_size, block_size=32,
                                    n_layer=2, n_head=2, n_embd=64))
    p = tok.encode("Q: 1 + 2 =\nA:")
    roll = [{"prompt_ids": p, "completion_ids": tok.encode(" 3"), "adv": +1.0},
            {"prompt_ids": p, "completion_ids": tok.encode(" 9"), "adv": -1.0}]
    dev = torch.device("cpu")
    X, Y, M = make_scoring_batch(roll, tok, dev)
    A = torch.tensor([r["adv"] for r in roll]).unsqueeze(-1)

    def seq_logprob():
        with torch.no_grad():
            lp, _, _ = token_logprobs(model, X, Y)
            return ((lp * M).sum(-1)).tolist()

    before = seq_logprob()
    opt = model.configure_optimizer(1e-2, 0.0)
    for _ in range(5):
        new_logp, _, _ = token_logprobs(model, X, Y)
        old = new_logp.detach().clone()
        ratio = (new_logp - old).exp()
        surr = torch.min(ratio * A, ratio.clamp(0.8, 1.28) * A)
        loss = -(surr * M).sum() / M.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    after = seq_logprob()
    assert after[0] > before[0], f"positive advantage should raise logP: {before[0]} -> {after[0]}"
    assert after[1] < before[1], f"negative advantage should lower logP: {before[1]} -> {after[1]}"


def test_arith_split_is_stable_and_disjoint():
    """Decontamination by construction: the split must be deterministic across
    processes (md5, not hash()) and no problem may appear on both sides."""
    from aienh.data import ARITH_MAX_OPERAND, arith_split
    from aienh.evaluate import arith_items
    from aienh.sft import build_sft_data
    n = ARITH_MAX_OPERAND + 1
    assignments = {(a, b): arith_split(a, b) for a in range(n) for b in range(n)}
    assert all(arith_split(a, b) == v for (a, b), v in assignments.items())   # stable
    n_test = sum(v == "test" for v in assignments.values())
    assert 0.1 < n_test / len(assignments) < 0.3, n_test / len(assignments)

    # the real guarantee: no eval item is in the SFT data
    train_rows, val_rows = build_sft_data(4000)
    seen = {(int(r["prompt"].split()[1]), int(r["prompt"].split()[3]))
            for r in train_rows + val_rows}
    assert not any((it["a"], it["b"]) in seen for it in arith_items(300))
    # and the deliberately-contaminated variant really is contaminated
    assert all((it["a"], it["b"]) in seen for it in arith_items(300, split="train"))


def test_ppo_clip_bounds_the_update():
    A = torch.tensor([1.0, -1.0])
    for ratio in (torch.tensor([5.0, 5.0]), torch.tensor([0.01, 0.01])):
        surr = torch.min(ratio * A, ratio.clamp(0.8, 1.2) * A)
        # clipping only ever REDUCES the surrogate, never increases it
        assert (surr <= ratio * A + 1e-6).all()


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def test_loglikelihood_matches_manual_computation():
    """The most bug-prone function in any harness: an off-by-one here silently
    scores the wrong token and every MC benchmark quietly becomes noise."""
    torch.manual_seed(0)
    m = _tiny().eval()
    tok = build_tokenizer("char", ["abcdefgh"])
    m.cfg.vocab_size = tok.vocab_size
    m = build_model(ModelConfig(vocab_size=tok.vocab_size, block_size=16,
                                n_layer=2, n_head=2, n_embd=32)).eval()
    ctx, cont = "abc", "de"
    (total, n, greedy), = loglikelihood(m, tok, ctx, [cont], torch.device("cpu"))
    ids = tok.encode(ctx) + tok.encode(cont)
    with torch.no_grad():
        logits, _, _ = m(torch.tensor([ids]))
        lp = F.log_softmax(logits.float(), -1)[0]
    manual = sum(float(lp[len(tok.encode(ctx)) + j - 1, ids[len(tok.encode(ctx)) + j]])
                 for j in range(len(tok.encode(cont))))
    assert n == len(tok.encode(cont))
    assert abs(total - manual) < 1e-4


def test_loglikelihood_batching_is_padding_safe():
    """Scoring several continuations together must give the same numbers as one at
    a time — otherwise your eval depends on batch size."""
    torch.manual_seed(0)
    tok = build_tokenizer("char", ["abcdefgh"])
    m = build_model(ModelConfig(vocab_size=tok.vocab_size, block_size=16,
                                n_layer=2, n_head=2, n_embd=32)).eval()
    conts = ["d", "de", "def"]
    together = loglikelihood(m, tok, "abc", conts, torch.device("cpu"))
    for cont, (t, _, _) in zip(conts, together):
        (alone, _, _), = loglikelihood(m, tok, "abc", [cont], torch.device("cpu"))
        assert abs(t - alone) < 1e-4, cont


def test_parser_behaviour_is_explicit():
    assert parse_int(" 62<|endoftext|>abc") == 62
    assert parse_int("the answer is 7") == 7
    assert parse_int(" -3") == -3
    assert parse_int("no digits here") is None


def test_stderr_matches_the_formula():
    assert abs(binom_stderr(0.5, 200) - math.sqrt(0.25 / 200)) < 1e-12
    assert binom_stderr(0.5, 1) == 0.0
    # the number quoted in the docs: ~3.5 points at p=0.5, n=200
    assert abs(binom_stderr(0.5, 200) - 0.0354) < 0.001


def test_points_are_monotone_and_bounded():
    from aienh.evaluate import TaskResult
    lo = {"arith_exact": TaskResult("arith_exact", "exact_match", 0.0, 100)}
    hi = {"arith_exact": TaskResult("arith_exact", "exact_match", 1.0, 100)}
    assert points_from_results(lo)[0] == 0.0
    assert points_from_results(hi)[0] == 100.0
    mid = {"arith_exact": TaskResult("arith_exact", "exact_match", 0.5, 100)}
    assert 0 < points_from_results(mid)[0] < 100


def test_suite_hash_changes_with_template_and_seed():
    a = suite_hash(["arith_exact"], 1234, "{q}")
    assert a == suite_hash(["arith_exact"], 1234, "{q}")          # stable
    assert a != suite_hash(["arith_exact"], 1234, "Q: {q}\nA:")   # template matters
    assert a != suite_hash(["arith_exact"], 9999, "{q}")          # seed matters
    assert a != suite_hash(["arith_exact", "format_ok"], 1234, "{q}")


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------

def test_config_hash_is_stable_and_sensitive():
    a = ModelConfig(n_embd=128)
    b = ModelConfig(n_embd=128)
    c = ModelConfig(n_embd=256)
    assert config_hash(a) == config_hash(b) != config_hash(c)


def test_lr_schedule_shape():
    total = 100
    lrs = [cosine_lr(s, total, 1e-3, warmup_frac=0.1, min_ratio=0.1) for s in range(total)]
    assert lrs[0] < lrs[9]                      # warming up
    assert abs(max(lrs) - 1e-3) < 1e-9          # peaks at base lr
    assert lrs[-1] < lrs[50] < max(lrs)         # then decays
    assert lrs[-1] >= 1e-4 - 1e-9               # never below the floor


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
