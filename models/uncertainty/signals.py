"""
P11-2 — Failure signals that do not depend on the detector's own confidence.

The core problem with using a detector's score to decide whether to trust it is
circularity: the score is produced by the same weights that produced the error.
When the model is wrong because it is out of distribution, it is confidently
wrong, and the score offers no independent evidence.

So the introspection head is fed signals that come from OUTSIDE the detector's
forward pass. Three independent sources, each failing in a different way:

    1. Cross-modal disagreement — a second sensor with different physics
       (radar: radio time-of-flight + Doppler; camera: photons + learned priors).
       Darkness breaks one and not the other, so disagreement is informative
       exactly in the ODD-boundary regime Phase 9 identified.
    2. Temporal instability — an object that appears, vanishes, and reappears
       across consecutive frames is one the model cannot commit to. This needs no
       second sensor and no labels. Phase 6 already measured the aggregate version
       (flicker rate 0.135); here it becomes a per-detection feature.
    3. Epistemic variance — from mc_dropout.py.

Independence is the point. Any one signal can be fooled; a detection that all
three flag is very likely wrong. Report the per-signal AUROC alongside the
combined one so it is clear each is pulling weight rather than one dominating.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def radar_camera_disagreement(
    detections: torch.Tensor,
    radar_points: np.ndarray,
    search_radius: float = 3.0,
    velocity_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Per-detection radar agreement score.

    Args:
        detections: (M, 5) BEV boxes [x, y, length, width, yaw] in the ego frame.
        radar_points: (N, 6) ego-frame radar returns from `load_radar_points`.
        search_radius: metres around a detection centre to search for returns.
        velocity_threshold: m/s above which a return counts as "moving".

    Returns:
        (M, F) float tensor of agreement features per detection:
            n_returns      — radar returns within search_radius
            min_distance   — distance to the nearest return (search_radius if none)
            mean_rcs       — mean RCS of nearby returns (0 if none)
            max_speed      — max |v| among nearby returns
            is_corroborated— 1.0 if n_returns > 0 else 0.0

    INTERPRETATION
    --------------
    A camera detection with zero nearby radar returns is a false-positive
    candidate: the camera claims solid matter, and the radar sees nothing there.
    A detection corroborated by several high-RCS returns is very likely real.

    BE CAREFUL — the converse is much weaker
    ----------------------------------------
    "No radar return" does NOT reliably mean "no object". Radar misses pedestrians
    routinely (low RCS, soft body, absorbed rather than reflected), and misses
    anything in a sensor blind spot or occluded by a closer strong reflector.
    So this feature is strong evidence for cars and weak-to-misleading for
    pedestrians. Compute per-class AUROC for this signal specifically; if it turns
    out uninformative for pedestrians, that is a real finding to report, not a bug
    to hide. It also argues for letting the introspection head see the class label
    so it can learn to weight this feature differently per class.

    Also: `search_radius` is a real hyperparameter with a physical meaning
    (radar angular error grows with range). Consider making it range-dependent
    rather than constant, and say which you chose.
    """
    raise NotImplementedError("P11-2")


def temporal_instability(
    track_history: Dict[int, List[Optional[dict]]],
    window: int = 5,
) -> Dict[int, float]:
    """
    Per-track instability over a sliding window of recent frames.

    Args:
        track_history: track_id -> list of per-frame detection dicts (None where
            the track was not detected in that frame), most recent last.
        window: number of recent frames to consider.

    Returns:
        track_id -> instability in [0, 1]; higher means less stable.

    SUGGESTED COMPONENTS (combine, then justify the weighting)
        - miss rate: fraction of the window where the track was absent
        - score variance across the window
        - centre jitter: variance of the BEV centre after subtracting a constant-
          velocity fit, so genuinely moving objects are not penalised for moving

    This reuses the tracking idea from `evaluation/eval_flicker.py`, but note the
    key difference: eval_flicker tracks by ground-truth `instance_token`, which is
    unavailable at inference. Here you must track *predictions*, so you need a
    real associator — greedy nearest-centre matching with a distance gate is
    sufficient and matches the P10-4 matcher. Do not reach for the GT tokens; the
    whole point is a signal computable at runtime with no labels.
    """
    raise NotImplementedError("P11-2")


def assemble_features(
    detections: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    epistemic: Optional[Dict[str, torch.Tensor]] = None,
    radar: Optional[torch.Tensor] = None,
    temporal: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Concatenate all available signals into the introspection head's input.

    Returns: (M, D) float tensor, one row per detection.

    FEATURE LIST (start here; ablate later)
        geometry   — range, bearing, box area, aspect ratio, distance to image edge
        class      — one-hot class label (3 dims)
        confidence — the detector's own score (include it; it IS informative, just
                     not sufficient)
        epistemic  — score_var, box_var
        radar      — the 5 columns from radar_camera_disagreement
        temporal   — instability scalar

    NORMALISE. Range is 0-51, area is 0-10000, variance may be ~1e-4. An MLP on
    raw values will be dominated by whichever feature happens to be largest.
    Fit a StandardScaler on the introspection TRAINING split only and persist it
    with the checkpoint — refitting at eval time leaks statistics from the test
    set and will flatter your AUROC.

    Handle missing signals explicitly. Radar features are unavailable when radar
    failed to load; temporal features are undefined for the first frames of a
    scene. Use a zero fill PLUS a binary "was this available" indicator column
    rather than a bare zero, so the head can distinguish "no radar returns nearby"
    (evidence of a false positive) from "radar not available" (no evidence at all).
    Conflating those two is a subtle and damaging bug.
    """
    raise NotImplementedError("P11-2")
