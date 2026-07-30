"""
P11-4 — Calibration and failure-prediction metrics.

Three questions, three different metrics. Report all three; each answers
something the others cannot.

    1. Does the confidence mean what it says?        -> ECE, reliability diagram
    2. Does it RANK errors correctly?                -> AUROC / AUPR
    3. What does acting on it actually buy?          -> risk-coverage curve

A model can be perfectly ranked but badly calibrated (AUROC 0.95, ECE 0.30) —
it sorts detections correctly but its "0.9" means 60%. Temperature scaling fixes
calibration without touching ranking, which is why they must be reported apart.

THE PHASE 9 CONNECTION
----------------------
The headline Phase 11 experiment is to compute ECE for the raw detector scores on
day frames and on night frames. The expected result — and the crux of the whole
project — is that the detector is roughly calibrated in daylight and badly
OVER-confident at night. That is the quantitative statement of "it fails
silently", and it is the single most important number this phase produces.
Then show the introspection head restores calibration at night.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, Dict]:
    """
    Expected Calibration Error.

    Args:
        confidences: (N,) predicted probabilities in [0, 1].
        correct: (N,) binary ground truth (1 = the prediction was right).
        n_bins: number of equal-width confidence bins.

    Returns:
        (ece, bin_stats) where bin_stats holds per-bin count, mean confidence and
        mean accuracy — everything needed to plot a reliability diagram.

    ECE partitions predictions into bins by confidence and takes the
    weighted mean gap between confidence and accuracy:

        ECE = sum_b (n_b / N) * |acc(b) - conf(b)|

    A perfectly calibrated model has ECE 0: of everything it calls 70% likely,
    exactly 70% is correct.

    CAVEATS TO RESPECT
        - Equal-width bins leave the high-confidence bins nearly empty for a
          detector whose scores cluster low. Report bin counts alongside ECE, and
          consider equal-MASS bins (quantile edges) as a second view.
        - ECE is sensitive to n_bins; state the value used.
        - With ~121 night frames the night-set ECE will be noisy. Bootstrap a
          confidence interval rather than reporting a bare point estimate — the
          day/night ECE comparison is the headline claim and it needs error bars.
    """
    raise NotImplementedError("P11-4")


def failure_prediction_auroc(
    reliability: np.ndarray,
    correct: np.ndarray,
) -> Dict[str, float]:
    """
    How well does the reliability score separate correct from incorrect detections?

    Args:
        reliability: (N,) predicted P(correct) — from the introspection head, or
            the raw detector score for the baseline.
        correct: (N,) binary ground truth.

    Returns: {"auroc": ..., "auprc": ..., "auprc_baseline": ...}

    AUROC is threshold-free: the probability that a randomly chosen correct
    detection is ranked above a randomly chosen incorrect one. 0.5 is chance.

    ALWAYS REPORT THE BASELINE. The comparison that matters is
        AUROC(introspection head) vs AUROC(raw detector score).
    If the head does not beat the raw score, it has learned nothing useful, and
    that is a publishable negative result — say so, as Phase 4 did with the ViT.

    Include AUPRC as well, with the positive base rate as its chance line. When
    classes are imbalanced (and they will be), AUROC can look respectable while
    precision on the minority class is poor; AUPRC exposes that.

    Compute these per condition (day / night) and per class. The interesting
    result is whether introspection generalises to night better than detection
    does — i.e. whether AUROC holds up at night even as mAP collapses. If it
    does, that is the strongest possible support for the project's thesis: the
    stack cannot see well at night, but it still knows that it cannot.
    """
    raise NotImplementedError("P11-4")


def risk_coverage_curve(
    reliability: np.ndarray,
    correct: np.ndarray,
    n_points: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Selective-prediction curve: accuracy as a function of how much you keep.

    Args:
        reliability: (N,) predicted P(correct).
        correct: (N,) binary ground truth.
        n_points: number of coverage levels to evaluate.

    Returns: {"coverage": (n,), "risk": (n,), "aurc": float}
        coverage — fraction of detections retained (sorted by reliability desc)
        risk     — error rate among the retained detections
        aurc     — area under the risk-coverage curve; lower is better

    THIS IS THE ONE TO PUT IN THE README. ECE and AUROC are abstract; this curve
    answers the question a practitioner actually has: "if I discard the least
    reliable 30% of detections, how much does my error rate drop?" It converts
    uncertainty into an engineering decision.

    The headline sentence to aim for, computed on the night set:

        "Abstaining on the least-reliable X% of night detections cuts the false
         positive rate from A to B, while retaining Y% of true positives."

    Report the curve for both the introspection head and the raw-score baseline
    on the same axes. The gap between the two curves is exactly what Phase 11
    contributes, and it is legible at a glance in a way no scalar is.
    """
    raise NotImplementedError("P11-4")


def temperature_scale(
    logits: np.ndarray,
    correct: np.ndarray,
) -> float:
    """
    Fit a single temperature T that minimises NLL on a held-out split; apply as
    `sigmoid(logits / T)`.

    Returns: the fitted scalar T.

    The cheapest possible calibration fix (Guo et al., 2017) — one parameter,
    fit by 1-D optimisation, and it cannot change AUROC at all because dividing
    by a positive constant preserves ordering. That property makes it a clean
    control: any AUROC improvement your introspection head shows is genuinely
    from the extra signals, not from recalibration.

    Worth a specific experiment here: fit T on DAY data, then measure ECE on
    NIGHT data. Expect it to help much less than a temperature fit on night data
    would — because the miscalibration is caused by distribution shift, not by a
    fixed scaling error, and a single global constant cannot track a shift it has
    never observed. That negative result is a clean argument for why the
    Phase 11 head needs out-of-model signals (radar, temporal) rather than yet
    another transform of the detector's own logits.
    """
    raise NotImplementedError("P11-4")
