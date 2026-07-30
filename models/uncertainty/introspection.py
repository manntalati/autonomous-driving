"""
P11-3 / P11-5 — Introspection head and the per-frame trust score.

    IntrospectionHead : per-detection  -> P(this detection is correct)
    TrustScorer       : per-frame      -> is the stack inside its ODD right now?

REFRAMING THE PROBLEM
---------------------
Detection asks "what is in this scene?". Introspection asks a different and much
easier question: "given this detection and its context, is the detector right?"
That is binary classification over a handful of scalar features, learnable from a
few thousand examples — unlike the detection problem itself, which Phase 9 showed
is data-starved. This asymmetry is why the approach works at this project's scale
and is worth stating plainly in the write-up.

The literature calls this *introspection*, *failure prediction*, or *learned
confidence*. The metric that matters is not accuracy — it is whether the
predicted probability ranks correct detections above incorrect ones (AUROC) and
whether the probability means what it says (calibration).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class IntrospectionHead(nn.Module):
    """
    Small MLP: detection feature vector -> P(correct).

    Args:
        in_features: width of the vector from `assemble_features`.
        hidden: hidden width. Keep it small (32-64) and use 2 layers. There are
            only a few thousand training detections and roughly a dozen features;
            a large head will memorise them.
        dropout: regularisation.

    Shapes: (M, in_features) -> (M,) logits. Apply sigmoid outside, or return
    logits and let BCEWithLogitsLoss handle it (numerically preferable).

    HOW TO BUILD THE TRAINING LABELS
    --------------------------------
    Run the frozen detector over a labelled split. For each surviving detection,
    match against GT (same matcher as the mAP code) and label:
        1 = true positive (IoU/centre-distance match with an unmatched GT)
        0 = false positive
    Missed GT objects produce no row — this head scores detections that exist, it
    does not predict misses. Say so explicitly; a reader will ask.

    THE SPLIT DISCIPLINE — the most important thing on this page
    -----------------------------------------------------------
    You need THREE disjoint splits:
        1. detector training     — trains the detector
        2. introspection training— trains this head, using the FROZEN detector
        3. final evaluation      — reports AUROC/ECE, touched once

    If you train the head on split 1, every detection there is one the detector
    has already fit, so false positives are rare and unrepresentative, and the
    head learns a failure distribution that does not exist at test time. Your
    AUROC will look excellent and mean nothing.

    Concretely, given the Phase 9 inventory: use trainval train (72 scenes) for
    the detector, trainval val (13 scenes) for this head, and the mini night
    scenes plus mini day scenes as the held-out evaluation. Note the consequence
    and be upfront about it — the head is trained on daytime failures only and
    tested on night failures, so it is itself subject to domain shift. Whether
    introspection generalises across the ODD boundary better than detection does
    is a genuinely interesting question and one of the better results this phase
    can produce. Do not hide it; measure it.

    CLASS IMBALANCE
    ---------------
    At a low score threshold most detections are false positives; at a high one
    most are true. The positive rate therefore depends entirely on the threshold
    you harvested at. Fix the threshold, report the resulting positive rate, and
    use `pos_weight` in BCEWithLogitsLoss rather than resampling — resampling
    distorts the base rate and breaks calibration, which is the one thing this
    head must get right.
    """

    def __init__(self, in_features: int, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        raise NotImplementedError("P11-3")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Args: (M, in_features). Returns: (M,) logits."""
        raise NotImplementedError("P11-3")


class TrustScorer:
    """
    Aggregate per-detection reliabilities into one per-frame trust score, and
    decide whether the stack is inside its operational design domain.

    Args:
        head: a trained IntrospectionHead.
        odd_threshold: trust below this means "outside ODD — do not rely on me".
        scaler: the persisted feature normaliser from training.

    This is the object Phase 12's agent consumes. Its output contract:

        {
          "trust": float,            # 0-1, per-frame
          "in_odd": bool,            # trust >= odd_threshold
          "reason": str,             # human-readable driver of a low score
          "per_detection": [...]     # reliability per detection
        }

    The `reason` field is what makes abstention useful rather than opaque.
    "Low trust" tells a driver nothing; "low trust: 6 of 8 detections lack radar
    corroboration, mean epistemic variance 3.1x the daytime baseline" tells them
    what the system is struggling with. Derive it from whichever feature group
    deviates most from its training-set mean — that is cheap and honest, and it
    avoids inventing an explanation the model did not actually use.

    AGGREGATION — not just the mean
    -------------------------------
    Averaging per-detection reliability hides the case that matters. One highly
    unreliable detection of a pedestrian in the ego lane is far more dangerous
    than five unreliable parked cars at 45 m. Weight by risk: proximity, whether
    the object is in the ego path, and class vulnerability. State the weighting
    explicitly — a reviewer will and should ask why a scalar summarises a scene.

    CALIBRATE THE THRESHOLD, DO NOT GUESS IT
    ----------------------------------------
    Pick `odd_threshold` from the risk-coverage curve on the introspection
    validation split, at an explicitly chosen operating point — e.g. the largest
    coverage whose selective error rate stays under some target. Then report
    what that threshold does on the night set. A threshold hand-tuned until the
    night demo looked good is not a result.
    """

    def __init__(self, head: IntrospectionHead, odd_threshold: float = 0.5, scaler=None) -> None:
        raise NotImplementedError("P11-5")

    def score_frame(self, detections, scores, labels, **signals) -> Dict:
        """Returns the dict described above. Must run without GT — this is inference."""
        raise NotImplementedError("P11-5")
