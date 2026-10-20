"""23: two builds compared question by question — the sums, as plain
functions (docs/prompts/phase-23-two-builds-question-by-question.md). No
dependency: the binomial sums are exact and short.

- One run a question (HLE, MATH Level 5, MMLU-Pro, SimpleQA): b questions A
  right and B wrong, c the other way, n both answered. d = (b − c)/n, its
  SE = √((b + c − n·d²)/(n − 1))/√n, the 95% range d ± 1.96·SE, and the
  verdict the exact McNemar test, two-sided at 5%.
- Several runs a question (GPQA ×4, OTIS ×8), and ARC-AGI-2 by task: each
  question's mean over its runs (a task's score), d the mean of the
  differences, SE their standard deviation (n − 1) over √n (Miller 2024), the
  verdict whether the 95% range leaves out 0.
- How often a build disagrees with itself between two of its runs, and with
  the other build."""

from __future__ import annotations

from math import sqrt

Z95 = 1.96
A_HIGHER, B_HIGHER, CANT_TELL = "A higher", "B higher", "can't tell"


def mean_se(d: list[float]) -> tuple[float, float]:
    """the mean of paired differences and its standard error: their standard
    deviation (with n − 1) over √n — 0 for fewer than two"""
    n = len(d)
    if not n:
        return 0.0, 0.0
    mu = sum(d) / n
    if n < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in d) / (n - 1)
    return mu, sqrt(var / n)


def mcnemar_p(b: int, c: int) -> float:
    """the exact McNemar test, two-sided: min(1, 2·P(X ≤ min(b, c))) with
    X ~ Binomial(b + c, ½) — 1 when no question differs"""
    m = b + c
    if m == 0:
        return 1.0
    k, term, tail = min(b, c), 1, 0
    for i in range(k + 1):
        tail += term
        term = term * (m - i) // (i + 1)
    return min(1.0, 2 * tail / 2 ** m)


def one_run(b: int, c: int, n: int) -> dict:
    """{d, se, lo, hi, p, verdict} for one run a question"""
    if n <= 0:
        return {"d": None, "se": None, "lo": None, "hi": None, "p": None, "verdict": CANT_TELL}
    d = (b - c) / n
    se = sqrt(max(0.0, (b + c - n * d * d) / (n - 1)) / n) if n > 1 else 0.0
    p = mcnemar_p(b, c)
    verdict = CANT_TELL if p >= 0.05 or b == c else (A_HIGHER if b > c else B_HIGHER)
    return {"d": d, "se": se, "lo": d - Z95 * se, "hi": d + Z95 * se, "p": p,
            "verdict": verdict}


def several_runs(sa: list[float], sb: list[float]) -> dict:
    """{d, se, lo, hi, verdict} from each question's mean over its runs (or
    each ARC task's score), A's and B's, in the same order"""
    if not sa:
        return {"d": None, "se": None, "lo": None, "hi": None, "verdict": CANT_TELL}
    d, se = mean_se([x - y for x, y in zip(sa, sb)])
    lo, hi = d - Z95 * se, d + Z95 * se
    verdict = A_HIGHER if lo > 0 else B_HIGHER if hi < 0 else CANT_TELL
    return {"d": d, "se": se, "lo": lo, "hi": hi, "verdict": verdict}


def arc_tasks(grids: dict[str, list[bool]], group: dict[str, str]) -> dict[str, float]:
    """ARC-AGI-2 as summary() scores it: a test grid is solved when either
    attempt is, a task scores the share of its grids solved"""
    by: dict[str, list[float]] = {}
    for g, runs in grids.items():
        if runs:
            by.setdefault(group.get(g, g), []).append(1.0 if any(runs) else 0.0)
    return {t: sum(v) / len(v) for t, v in by.items()}


def self_disagree(k: int, r: int) -> float | None:
    """the chance two different runs of one build differ on a question it was
    asked k times and got right r times: 2·r·(k − r)/(k·(k − 1))"""
    if k < 2:
        return None
    return 2 * r * (k - r) / (k * (k - 1))


def cross_disagree(pa: float, pb: float) -> float:
    """the chance a run of A and a run of B differ: pA·(1 − pB) + pB·(1 − pA)"""
    return pa * (1 - pb) + pb * (1 - pa)
