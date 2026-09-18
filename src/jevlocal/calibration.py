"""Calibration: softmax, NLL, ECE, per-head temperature scaling.

Temperature scaling divides logits by T before softmax. T is fit on a
calibration set (fit split) and reported on a held-out eval split.
Pilot n=24 results are preliminary only; the decision set needs 150-300
held-out items with confidence intervals (see plans/faz2.md).
"""

import math


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not logits:
        raise ValueError("logits must not be empty")
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    return [e / s for e in exps]


def nll(probs_of_gold: list[float]) -> float:
    return sum(-math.log(max(p, 1e-12)) for p in probs_of_gold) / max(1, len(probs_of_gold))


def ece(confidences: list[float], hits: list[bool], bins: int = 5) -> float:
    total = 0.0
    n = len(confidences)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c == 0)]
        if not idx:
            continue
        acc = sum(hits[i] for i in idx) / len(idx)
        avg = sum(confidences[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(acc - avg)
    return total


def fit_temperature(logits_list: list[list[float]], gold_idx: list[int],
                    grid: list[float] | None = None) -> tuple[float, float]:
    """Grid-search T minimizing NLL. Returns (best_T, best_NLL)."""
    if not logits_list:
        raise ValueError("logits_list must not be empty")
    if grid is None:
        grid = [round(0.25 + 0.25 * i, 2) for i in range(12)]  # 0.25..3.0
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        pg = [softmax(logits, t)[g] for logits, g in zip(logits_list, gold_idx)]
        v = nll(pg)
        if v < best_nll:
            best_t, best_nll = t, v
    return best_t, best_nll
