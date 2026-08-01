"""
P11-2 — Failure signals that do not depend on the detector's own confidence.

The problem with using a detector's score to decide whether to trust it is
circularity: the score comes from the same weights that produced the error. When
the model is wrong because it is out of distribution, it is confidently wrong.

So the introspection head is fed signals from OUTSIDE the detector's forward
pass. Three sources, each failing differently:

    1. Cross-modal disagreement — a second sensor with different physics
       (radar: radio time-of-flight + Doppler; camera: photons + learned priors).
       Darkness breaks one and not the other.
    2. Temporal instability — an object that appears, vanishes and reappears is
       one the model cannot commit to. Needs no second sensor and no labels.
       Phase 6 measured the aggregate version (flicker 0.135); this is the
       per-detection form.
    3. Epistemic variance — from mc_dropout.py.

Independence is the point: any one can be fooled, but a detection all three flag
is very likely wrong. Report per-signal AUROC alongside the combined figure so it
is clear each pulls weight rather than one dominating.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Column layout of radar_camera_disagreement's output.
RADAR_FEATURES = ["n_returns", "min_distance", "mean_rcs", "max_speed", "is_corroborated"]


def radar_camera_disagreement(
    detections: np.ndarray,
    radar_points: np.ndarray,
    search_radius: float = 3.0,
    range_scaled: bool = True,
) -> np.ndarray:
    """
    Per-detection radar agreement features.

    Args:
        detections: (M, 5) BEV boxes [x, y, length, width, yaw] in the ego frame.
        radar_points: (N, 6) ego-frame radar returns from `load_radar_points`.
        search_radius: base search radius in metres.
        range_scaled: grow the radius with range. Radar angular error is angular,
            so its cross-range position error grows linearly with distance; a
            constant radius is too tight far away and too loose up close.
            Effective radius = search_radius * (1 + range / 50).

    Returns: (M, 5) float32, columns RADAR_FEATURES.

    INTERPRETATION — asymmetric, and the asymmetry matters
    ------------------------------------------------------
    A camera detection with no nearby radar return is a false-positive candidate:
    the camera claims solid matter, the radar sees nothing.

    The converse is much weaker. "No radar return" does NOT reliably mean "no
    object" — radar routinely misses pedestrians (low RCS, soft body, absorbed
    rather than reflected) and anything occluded by a closer strong reflector.
    This feature is therefore strong evidence for cars and weak-to-misleading for
    pedestrians. Compute per-class AUROC for it specifically; if it is
    uninformative for pedestrians, that is a finding to report. It also argues for
    letting the introspection head see the class label so it can weight this
    feature differently per class.
    """
    det = np.asarray(detections, dtype=np.float64).reshape(-1, 5)
    pts = np.asarray(radar_points, dtype=np.float64).reshape(-1, 6)
    out = np.zeros((len(det), len(RADAR_FEATURES)), dtype=np.float32)
    if len(det) == 0:
        return out

    rng = np.sqrt((det[:, :2] ** 2).sum(axis=1))
    radii = search_radius * (1.0 + rng / 50.0) if range_scaled else np.full(len(det), search_radius)

    if len(pts) == 0:
        out[:, 1] = radii            # min_distance saturates when nothing is near
        return out

    d = np.sqrt(((det[:, None, :2] - pts[None, :, :2]) ** 2).sum(axis=-1))   # (M, N)
    speed = np.sqrt(pts[:, 4] ** 2 + pts[:, 5] ** 2)

    for i in range(len(det)):
        near = d[i] <= radii[i]
        n = int(near.sum())
        out[i, 0] = n
        out[i, 1] = float(d[i].min()) if len(pts) else radii[i]
        out[i, 2] = float(pts[near, 3].mean()) if n else 0.0
        out[i, 3] = float(speed[near].max()) if n else 0.0
        out[i, 4] = 1.0 if n else 0.0
    # Cap min_distance so an isolated detection does not produce an outlier that
    # dominates the standardiser.
    out[:, 1] = np.minimum(out[:, 1], radii)
    return out


def temporal_instability(
    track_history: Dict[int, List[Optional[dict]]],
    window: int = 5,
) -> Dict[int, float]:
    """
    Per-track instability over a sliding window of recent frames.

    Args:
        track_history: track_id -> list of per-frame detection dicts (None where
            the track was not detected), most recent last. Each dict needs
            "score" and "xy".
        window: number of recent frames to consider.

    Returns: track_id -> instability in [0, 1]; higher is less stable.

    Components:
        miss_rate     — fraction of the window where the track was absent
        score_std     — clipped std of the detection score
        jitter        — centre deviation from a constant-velocity fit, normalised.
                        Subtracting the linear fit means genuinely moving objects
                        are not penalised for moving; only erratic motion counts.

    Components that could not be MEASURED are excluded from the average rather
    than counted as zero. A track seen in only one frame of five has no score
    variance and no motion fit; averaging those in as 0.0 would report it as
    fairly stable (0.27) when it is the least stable thing in the scene. With
    fewer than two observations, the miss rate alone is the estimate.

    This reuses the tracking idea from evaluation/eval_flicker.py but differs in
    one essential way: eval_flicker tracks by ground-truth `instance_token`, which
    does not exist at inference. Here the caller must supply predicted tracks from
    a real associator (SceneTracker), because the entire point is a signal
    computable at runtime with no labels.
    """
    out: Dict[int, float] = {}
    for tid, hist in track_history.items():
        h = hist[-window:]
        if not h:
            out[tid] = 1.0
            continue
        present = [x for x in h if x is not None]
        miss_rate = 1.0 - len(present) / len(h)

        if len(present) >= 2:
            scores = np.array([p["score"] for p in present], dtype=np.float64)
            score_std = float(np.clip(scores.std() * 2.0, 0.0, 1.0))
            xy = np.array([p["xy"] for p in present], dtype=np.float64)
            t = np.arange(len(xy), dtype=np.float64)
            # constant-velocity fit per axis; residual is erratic motion only
            resid = np.zeros_like(xy)
            for ax in range(2):
                coef = np.polyfit(t, xy[:, ax], 1)
                resid[:, ax] = xy[:, ax] - np.polyval(coef, t)
            jitter = float(np.clip(np.linalg.norm(resid, axis=1).mean() / 2.0, 0.0, 1.0))
            score = (miss_rate + score_std + jitter) / 3.0
        else:
            score = miss_rate      # the only component that is measurable here

        out[tid] = float(np.clip(score, 0.0, 1.0))
    return out


# Feature block layout of assemble_features. Kept explicit so the ablation can
# drop a block by name and the introspection head can report per-block importance.
GEOMETRY_FEATURES = ["range", "bearing", "area", "aspect", "abs_y"]
FEATURE_BLOCKS = {
    "geometry": GEOMETRY_FEATURES,
    "class": ["cls_0", "cls_1", "cls_2"],
    "confidence": ["score"],
    "epistemic": ["score_var", "box_var", "epistemic_avail"],
    "radar": RADAR_FEATURES + ["radar_avail"],
    "temporal": ["instability", "temporal_avail"],
}
FEATURE_NAMES = [n for block in FEATURE_BLOCKS.values() for n in block]


def assemble_features(
    detections: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 3,
    epistemic: Optional[Dict[str, np.ndarray]] = None,
    radar: Optional[np.ndarray] = None,
    temporal: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Concatenate all available signals into the introspection head's input.

    Args:
        detections: (M, 5) BEV boxes.
        scores: (M,) detector confidence.
        labels: (M,) class ids.
        epistemic: {"score_var": (M,), "box_var": (M,)} or None.
        radar: (M, 5) from radar_camera_disagreement, or None.
        temporal: (M,) instability per detection, or None.

    Returns: (M, D) float32, columns FEATURE_NAMES.

    MISSING SIGNALS ARE EXPLICIT, NOT ZERO
    --------------------------------------
    Every optional block carries a binary "was this available" indicator. A bare
    zero would conflate "no radar returns nearby" (evidence of a false positive)
    with "radar not available" (no evidence at all) — a subtle and damaging bug,
    since the first should lower reliability and the second should not move it.

    NORMALISATION is deliberately NOT done here. Range is 0-51, area is 0-10000,
    and variance may be ~1e-4; an MLP on raw values is dominated by whichever
    feature is largest. Fit the standardiser on the introspection TRAINING split
    only (see FeatureScaler below) and persist it with the checkpoint — refitting
    at eval time leaks test statistics and flatters AUROC.
    """
    det = np.asarray(detections, dtype=np.float64).reshape(-1, 5)
    m = len(det)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    x, y, length, width = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
    geom = np.stack([
        np.sqrt(x ** 2 + y ** 2),                        # range
        np.degrees(np.arctan2(y, np.maximum(np.abs(x), 1e-6))),  # bearing
        length * width,                                  # area
        length / np.maximum(width, 1e-6),                # aspect
        np.abs(y),                                       # lateral offset from ego path
    ], axis=1)

    onehot = np.zeros((m, num_classes), dtype=np.float64)
    if m:
        onehot[np.arange(m), np.clip(labels, 0, num_classes - 1)] = 1.0

    if epistemic is not None:
        epi = np.stack([
            np.asarray(epistemic["score_var"], dtype=np.float64).reshape(-1),
            np.asarray(epistemic["box_var"], dtype=np.float64).reshape(-1),
            np.ones(m),
        ], axis=1)
    else:
        epi = np.zeros((m, 3))

    if radar is not None:
        rad = np.concatenate([np.asarray(radar, dtype=np.float64).reshape(m, -1),
                              np.ones((m, 1))], axis=1)
    else:
        rad = np.zeros((m, len(RADAR_FEATURES) + 1))

    if temporal is not None:
        tmp = np.stack([np.asarray(temporal, dtype=np.float64).reshape(-1), np.ones(m)], axis=1)
    else:
        tmp = np.zeros((m, 2))

    feats = np.concatenate([geom, onehot, scores.reshape(-1, 1), epi, rad, tmp], axis=1)
    assert feats.shape[1] == len(FEATURE_NAMES), \
        f"feature width {feats.shape[1]} != {len(FEATURE_NAMES)} names"
    return feats.astype(np.float32)


class FeatureScaler:
    """
    Standardiser fitted on the introspection training split only.

    Persisted alongside the head so inference uses training-time statistics.
    Refitting at eval time would leak test-set statistics into the features.
    """

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "FeatureScaler":
        x = np.asarray(x, dtype=np.float64)
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), 1e-6)   # guard constant columns
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("FeatureScaler used before fit()")
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict) -> "FeatureScaler":
        self.mean, self.std = state["mean"], state["std"]
        return self
