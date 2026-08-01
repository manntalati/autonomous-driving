"""
P11-4 — Calibration and failure-prediction metrics.

Three questions, three metrics. Each answers something the others cannot.

    1. Does the confidence mean what it says?   -> ECE, reliability diagram
    2. Does it RANK errors correctly?           -> AUROC / AUPRC
    3. What does acting on it buy?              -> risk-coverage curve

A model can be perfectly ranked but badly calibrated (AUROC 0.95, ECE 0.30): it
sorts detections correctly but its "0.9" means 60%. Temperature scaling fixes
calibration without touching ranking, which is why they must be reported apart.

THE PHASE 9 CONNECTION
----------------------
The headline experiment is ECE on raw detector scores, day vs night. The expected
result — the crux of the project — is roughly calibrated in daylight and badly
OVER-confident at night. That is the quantitative statement of "it fails
silently". Then show the introspection head restores calibration at night.

No sklearn dependency: everything here is a few lines of numpy, and the project
does not otherwise pull scikit-learn in.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
    equal_mass: bool = False,
) -> Tuple[float, Dict]:
    """
    Expected Calibration Error.

    Args:
        confidences: (N,) predicted probabilities in [0, 1].
        correct: (N,) binary ground truth (1 = prediction was right).
        n_bins: number of confidence bins.
        equal_mass: use quantile bin edges instead of equal-width. Equal-width
            bins leave high-confidence bins nearly empty when scores cluster low,
            which is exactly this detector's regime — report both views.

    Returns: (ece, bin_stats) with per-bin count, mean confidence, mean accuracy,
    ready to plot as a reliability diagram.

        ECE = sum_b (n_b / N) * |acc(b) - conf(b)|

    A perfectly calibrated model has ECE 0: of everything it calls 70% likely,
    exactly 70% is correct. ECE is sensitive to n_bins — always state the value.
    """
    conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
    corr = np.asarray(correct, dtype=np.float64).reshape(-1)
    if len(conf) == 0:
        return float("nan"), {"edges": [], "count": [], "confidence": [], "accuracy": []}

    if equal_mass:
        edges = np.unique(np.quantile(conf, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    counts, confs, accs = [], [], []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # include the right edge on the last bin so conf == 1.0 is not dropped
        m = (conf >= lo) & (conf < hi) if hi < edges[-1] else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        counts.append(n)
        if n == 0:
            confs.append(float("nan"))
            accs.append(float("nan"))
            continue
        c, a = float(conf[m].mean()), float(corr[m].mean())
        confs.append(c)
        accs.append(a)
        ece += (n / len(conf)) * abs(a - c)

    return float(ece), {
        "edges": edges.tolist(), "count": counts,
        "confidence": confs, "accuracy": accs, "n_bins": n_bins,
    }


def bootstrap_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for ECE.

    With ~121 night frames the night-set ECE is noisy, and the day/night ECE
    comparison is the headline claim — it needs error bars, not a bare point
    estimate.

    Returns: {"ece", "lo", "hi"} at the 95% percentile interval.
    """
    conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
    corr = np.asarray(correct, dtype=np.float64).reshape(-1)
    point, _ = expected_calibration_error(conf, corr, n_bins)
    if len(conf) < 2:
        return {"ece": point, "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(conf), len(conf))
        e, _ = expected_calibration_error(conf[idx], corr[idx], n_bins)
        boots.append(e)
    return {"ece": point,
            "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5))}


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    AUROC via the rank (Mann-Whitney U) identity, with tie correction.
    Equivalent to the trapezoidal ROC area and cheaper.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within tied score groups
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision — area under the precision-recall curve."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    cum_tp = np.cumsum(y)
    precision = cum_tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def failure_prediction_auroc(reliability: np.ndarray, correct: np.ndarray) -> Dict[str, float]:
    """
    How well does the reliability score separate correct from incorrect detections?

    Args:
        reliability: (N,) predicted P(correct) — the introspection head, or the
            raw detector score for the baseline.
        correct: (N,) binary ground truth.

    Returns: {"auroc", "auprc", "auprc_baseline", "n", "positive_rate"}

    AUROC is threshold-free: the probability a randomly chosen correct detection
    outranks a randomly chosen incorrect one. 0.5 is chance.

    ALWAYS REPORT THE BASELINE. The comparison that matters is
    AUROC(introspection head) vs AUROC(raw detector score). If the head does not
    beat the raw score it has learned nothing useful — a publishable negative
    result, as Phase 4 did with the ViT.

    AUPRC is included with the positive base rate as its chance line: under
    imbalance AUROC can look respectable while minority-class precision is poor.

    Compute per condition (day/night) and per class. The interesting question is
    whether AUROC holds up at night even as mAP collapses — i.e. whether the
    stack still knows it cannot see.
    """
    rel = np.asarray(reliability, dtype=np.float64).reshape(-1)
    corr = np.asarray(correct, dtype=np.int64).reshape(-1)
    base = float(corr.mean()) if len(corr) else float("nan")
    return {
        "auroc": _auroc(rel, corr),
        "auprc": _auprc(rel, corr),
        "auprc_baseline": base,
        "n": int(len(corr)),
        "positive_rate": base,
    }


def risk_coverage_curve(
    reliability: np.ndarray,
    correct: np.ndarray,
    n_points: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Selective-prediction curve: error rate as a function of how much you keep.

    Args:
        reliability: (N,) predicted P(correct).
        correct: (N,) binary ground truth.
        n_points: number of coverage levels.

    Returns: {"coverage", "risk", "aurc"} — aurc is the area under the curve,
    lower is better.

    THIS IS THE ONE FOR THE README. ECE and AUROC are abstract; this answers the
    practitioner's actual question: "if I discard the least reliable 30% of
    detections, how much does my error rate drop?" It turns uncertainty into an
    engineering decision.

    Target sentence, computed on the night set:
        "Abstaining on the least-reliable X% of night detections cuts the false
         positive rate from A to B, while retaining Y% of true positives."

    Plot the head and the raw-score baseline on the same axes — the gap between
    the curves is exactly what Phase 11 contributes.
    """
    rel = np.asarray(reliability, dtype=np.float64).reshape(-1)
    corr = np.asarray(correct, dtype=np.float64).reshape(-1)
    n = len(rel)
    if n == 0:
        return {"coverage": np.zeros(0), "risk": np.zeros(0), "aurc": float("nan")}

    order = np.argsort(-rel, kind="mergesort")   # most reliable first
    corr_sorted = corr[order]
    cum_correct = np.cumsum(corr_sorted)
    kept = np.arange(1, n + 1)
    risk_all = 1.0 - cum_correct / kept
    coverage_all = kept / n

    idx = np.unique(np.linspace(0, n - 1, min(n_points, n)).astype(int))
    coverage, risk = coverage_all[idx], risk_all[idx]
    aurc = float(np.trapezoid(risk_all, coverage_all)) if n > 1 else float("nan")
    return {"coverage": coverage, "risk": risk, "aurc": aurc}


def selective_summary(reliability: np.ndarray, correct: np.ndarray,
                      coverage: float = 0.7) -> Dict[str, float]:
    """
    The README sentence, computed: what abstaining actually buys at one operating point.

    Returns error rate over all detections vs over the retained fraction, plus the
    share of true positives retained.
    """
    rel = np.asarray(reliability, dtype=np.float64).reshape(-1)
    corr = np.asarray(correct, dtype=np.float64).reshape(-1)
    n = len(rel)
    if n == 0:
        return {}
    k = max(1, int(round(coverage * n)))
    keep = np.argsort(-rel, kind="mergesort")[:k]
    total_tp = corr.sum()
    return {
        "coverage": k / n,
        "error_all": float(1.0 - corr.mean()),
        "error_kept": float(1.0 - corr[keep].mean()),
        "tp_retained": float(corr[keep].sum() / total_tp) if total_tp else float("nan"),
        "n_all": n,
        "n_kept": k,
    }


def temperature_scale(logits: np.ndarray, correct: np.ndarray,
                      lo: float = 0.05, hi: float = 10.0, steps: int = 200) -> float:
    """
    Fit a single temperature T minimising NLL; apply as sigmoid(logits / T).

    Returns: the fitted scalar T.

    The cheapest calibration fix (Guo et al., 2017). Dividing by a positive
    constant preserves ordering, so it cannot change AUROC at all — which makes
    it a clean control: any AUROC gain from the introspection head comes from the
    extra signals, not from recalibration.

    Fitted by a grid + local refine over log T rather than gradient descent; the
    objective is 1-D, smooth and convex enough that this is exact to ~1e-3 and
    avoids an optimiser dependency.

    WORTH RUNNING: fit T on DAY data, then measure ECE on NIGHT data. Expect it to
    help far less than a night-fitted T would — the miscalibration is caused by
    distribution shift, not a fixed scaling error, and one global constant cannot
    track a shift it never observed. That negative result is the argument for why
    Phase 11 needs out-of-model signals rather than another transform of the
    detector's own logits.
    """
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(correct, dtype=np.float64).reshape(-1)
    if len(z) == 0:
        return 1.0

    def nll(t: float) -> float:
        s = z / t
        # Numerically stable binary cross-entropy from logits:
        #   max(s,0) - s*y + log(1 + exp(-|s|))
        return float(np.mean(np.maximum(s, 0) - s * y + np.log1p(np.exp(-np.abs(s)))))

    grid = np.exp(np.linspace(np.log(lo), np.log(hi), steps))
    best = min(grid, key=nll)
    # local refine around the grid winner
    fine = np.linspace(max(lo, best * 0.8), min(hi, best * 1.25), steps)
    return float(min(fine, key=nll))


def platt_scale(logits: np.ndarray, correct: np.ndarray,
                iters: int = 2000, lr: float = 0.1) -> Tuple[float, float]:
    """
    Fit a 2-parameter affine calibration: sigmoid(a * z + b).

    Returns: (a, b).

    WHY TWO PARAMETERS AND NOT ONE
    ------------------------------
    Temperature scaling (a = 1/T, b = 0) can only rescale logits. Training with
    `pos_weight` to counter class imbalance shifts the model's implicit prior by
    roughly log(pos_weight) — a BIAS, not a scale — so a single temperature cannot
    remove it. Measured here: temperature-only cut the head's ECE from 0.157 to
    0.121 but left it well above the raw score's 0.060; the residual is the prior
    shift, which `b` absorbs.

    Like temperature scaling, this is monotonic in the logit whenever a > 0, so it
    cannot change AUROC — the ranking is untouched and only calibration moves.

    Fitted by gradient descent on NLL (2 parameters, convex in practice). Fit on a
    split disjoint from the evaluation sets.

    `iters` defaults to 2000, not a few hundred: on a known +2.0 bias injection,
    200 iterations recovered only b = -1.39 while 2000 reached -1.97. An
    under-converged fit fails silently — the numbers still look plausible — so the
    default is set past the point where the estimate stops moving.
    """
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(correct, dtype=np.float64).reshape(-1)
    if len(z) == 0:
        return 1.0, 0.0

    a, b = 1.0, 0.0
    n = len(z)
    for _ in range(iters):
        s = a * z + b
        p = 1.0 / (1.0 + np.exp(-np.clip(s, -50, 50)))
        diff = p - y
        ga, gb = float((diff * z).mean()), float(diff.mean())
        a -= lr * ga
        b -= lr * gb
        if abs(ga) < 1e-9 and abs(gb) < 1e-9:
            break
    # a must stay positive or the calibration would invert the ranking
    return (float(a), float(b)) if a > 0 else (1.0, 0.0)
