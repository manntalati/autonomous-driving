"""
P11-3 / P11-5 — Introspection head and the per-frame trust score.

    IntrospectionHead : per-detection  -> P(this detection is correct)
    TrustScorer       : per-frame      -> is the stack inside its ODD right now?

REFRAMING THE PROBLEM
---------------------
Detection asks "what is in this scene?". Introspection asks a different and much
easier question: "given this detection and its context, is the detector right?"
That is binary classification over a dozen scalar features, learnable from a few
thousand examples — unlike detection itself, which Phase 9 showed is data-starved.
That asymmetry is why this works at this project's scale.

The metric that matters is not accuracy but whether the predicted probability
ranks correct detections above incorrect ones (AUROC), and whether the probability
means what it says (calibration).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from models.uncertainty.signals import FEATURE_BLOCKS, FEATURE_NAMES, FeatureScaler

CLASS_NAMES = ["car", "pedestrian", "cyclist"]


class IntrospectionHead(nn.Module):
    """
    Small MLP: detection feature vector -> P(correct) logit.

    Args:
        in_features: width from `assemble_features` (len(FEATURE_NAMES)).
        hidden: hidden width. Small (32-64), 2 layers — there are only a few
            thousand training detections over ~15 features; a large head memorises.
        dropout: regularisation.

    Shapes: (M, in_features) -> (M,) logits. Returns logits, not probabilities, so
    BCEWithLogitsLoss can be used (numerically preferable to sigmoid + BCE).

    TRAINING LABELS
    ---------------
    Run the frozen detector over a labelled split; match each surviving detection
    against GT with the same matcher the mAP code uses:
        1 = true positive, 0 = false positive.
    Missed GT objects produce no row — this head scores detections that exist, it
    does not predict misses. State that explicitly; a reader will ask.

    THE SPLIT DISCIPLINE — the most important thing here
    ---------------------------------------------------
    Three disjoint splits are required:
        1. detector training      — trains the detector
        2. introspection training — trains this head, on the FROZEN detector
        3. final evaluation       — reports AUROC/ECE, touched once

    Training the head on split 1 means every detection there is one the detector
    already fit: false positives are rare and unrepresentative, the head learns a
    failure distribution that does not exist at test time, and AUROC looks
    excellent while meaning nothing.

    Given the Phase 9 inventory: detector on trainval train (72 scenes), head on
    trainval val (13 scenes), evaluation on the mini day and night scenes. Note
    the consequence honestly — the head is trained on daytime failures and tested
    on night failures, so it is itself subject to domain shift. Whether
    introspection generalises across the ODD boundary better than detection does
    is one of the more interesting results available here. Measure it; do not
    hide it.

    CLASS IMBALANCE AND CALIBRATION (measured, not assumed)
    -------------------------------------------------------
    The positive rate depends entirely on the score threshold detections were
    harvested at — at threshold 0.05 it is 0.115 (2,340 TP / 17,945 FP). Fix the
    threshold and report the rate.

    `pos_weight` is used to handle the imbalance rather than resampling, but it
    is NOT calibration-neutral: up-weighting the minority class deliberately
    pushes predicted probabilities away from the true base rate. Measured on the
    first run, the head reached ECE 0.157 against the raw score's 0.060 — better
    ranked but worse calibrated. `train_introspection.train_head` therefore fits a
    2-parameter affine calibration afterwards. One parameter is not enough:
    pos_weight shifts the implicit prior by ~log(pos_weight), a bias rather than a
    scale, and a temperature-only fit only reached ECE 0.121. The affine form is
    monotonic for a > 0, so calibration improves while AUROC is untouched.
    """

    def __init__(self, in_features: int = len(FEATURE_NAMES), hidden: int = 64,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.in_features = in_features
        # Affine calibration sigmoid(a*z + b), fitted after training (see
        # train_introspection). (1.0, 0.0) is a no-op, so an unfitted head behaves
        # exactly as before. Two parameters, not one: pos_weight shifts the prior
        # (a bias) as well as sharpening, and a temperature alone cannot undo that.
        self.calib_a: float = 1.0
        self.calib_b: float = 0.0
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Args: (M, in_features). Returns: (M,) logits."""
        if features.dim() != 2:
            raise ValueError(f"expected (M, F), got {tuple(features.shape)}")
        return self.net(features).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args: (M, F). Returns: (M,) calibrated probabilities in [0, 1].

        Applies the fitted affine calibration. Monotonic for a > 0, so this
        changes ECE but never AUROC.
        """
        self.eval()
        return torch.sigmoid(self(features) * self.calib_a + self.calib_b)


def block_of(feature_name: str) -> str:
    """Which FEATURE_BLOCKS group a feature belongs to (for the reason string)."""
    for block, names in FEATURE_BLOCKS.items():
        if feature_name in names:
            return block
    return "other"


class TrustScorer:
    """
    Aggregate per-detection reliabilities into one per-frame trust score, and
    decide whether the stack is inside its operational design domain.

    Args:
        head: trained IntrospectionHead.
        scaler: the FeatureScaler persisted from training.
        odd_threshold: trust below this means "outside ODD — do not rely on me".
        vulnerable_classes: class ids weighted up in the risk aggregation.

    Output contract — this is what Phase 12's agent consumes:

        {
          "trust": float,          # 0-1, per-frame
          "in_odd": bool,          # trust >= odd_threshold
          "reason": str,           # human-readable driver of a low score
          "per_detection": [...]   # reliability per detection
        }

    RISK-WEIGHTED AGGREGATION, NOT A PLAIN MEAN
    -------------------------------------------
    Averaging hides the case that matters. One unreliable pedestrian detection in
    the ego lane is far more dangerous than five unreliable parked cars at 45 m.
    Weight by proximity, ego-path membership, and class vulnerability. The
    weighting is stated explicitly because a reviewer should ask why a scalar
    summarises a scene.

    CALIBRATE THE THRESHOLD, DO NOT GUESS IT
    ----------------------------------------
    Pick odd_threshold from the risk-coverage curve on the introspection
    validation split at an explicitly chosen operating point, then report what it
    does on the night set. A threshold hand-tuned until the night demo looked good
    is not a result.
    """

    def __init__(self, head: IntrospectionHead, scaler: Optional[FeatureScaler] = None,
                 odd_threshold: float = 0.5, vulnerable_classes: Sequence[int] = (1, 2),
                 geometry: str = "image") -> None:
        """
        Args:
            geometry — the coordinate convention of the boxes this scorer is fed:
                "image" (pixels, [cx, cy, w, h]) or "bev" (metres, [x, y, l, w, yaw]).

        THIS MUST MATCH WHAT THE HEAD WAS TRAINED ON. `train_introspection`
        harvests 2-D detections and builds pixel-space geometry features, so a
        head trained by it expects "image". Feeding it BEV metres puts every
        feature many sigma from the training mean, reliability collapses to ~0,
        and the frame is reported as outside the ODD on every frame — which looks
        like a dramatic finding and is actually a unit mismatch.

        The risk WEIGHTING (proximity, ego-corridor, class vulnerability) is only
        meaningful in metres, so it is applied for "bev" and falls back to uniform
        weighting for "image". Risk-weighted trust over image detections would
        need a BEV-trained head; that is a real gap, not something to fake by
        reinterpreting pixels as metres.
        """
        if geometry not in ("image", "bev"):
            raise ValueError(f"geometry must be 'image' or 'bev', got {geometry!r}")
        self.head = head
        self.scaler = scaler
        self.odd_threshold = odd_threshold
        self.vulnerable_classes = set(vulnerable_classes)
        self.geometry = geometry

    def _weights(self, detections: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Risk weight per detection: closer, in-path, and vulnerable count more.

        Uniform for image-space boxes — `hypot(cx, cy)` on pixel coordinates is
        not a range, and pretending otherwise would silently weight detections by
        their distance from the top-left corner of the frame.
        """
        det = np.asarray(detections, dtype=np.float64).reshape(-1, 5)
        if len(det) == 0:
            return np.zeros(0)
        vulnerable = np.array([2.0 if int(l) in self.vulnerable_classes else 1.0
                               for l in np.asarray(labels).reshape(-1)])
        if self.geometry != "bev":
            return vulnerable                      # class risk only; no usable geometry
        rng = np.sqrt((det[:, :2] ** 2).sum(axis=1))
        proximity = 1.0 / (1.0 + rng / 20.0)              # ~1.0 at 0 m, ~0.3 at 50 m
        in_path = np.where(np.abs(det[:, 1]) <= 2.0, 2.0, 1.0)   # ego corridor
        return proximity * in_path * vulnerable

    def _reason(self, features: np.ndarray) -> str:
        """
        Name the feature block deviating most from its training-set mean.

        Derived from the scaler's statistics rather than invented: this reports
        which inputs are unusual, not a story about how the head reasoned.
        """
        if self.scaler is None or self.scaler.mean is None or len(features) == 0:
            return "low predicted reliability"
        z = np.abs(self.scaler.transform(features)).mean(axis=0)
        block_dev: Dict[str, List[float]] = {}
        for name, dev in zip(FEATURE_NAMES, z):
            block_dev.setdefault(block_of(name), []).append(float(dev))
        worst = max(block_dev.items(), key=lambda kv: np.mean(kv[1]))
        label = {
            "epistemic": "model uncertainty far above its daytime baseline",
            "radar": "detections lack radar corroboration",
            "temporal": "detections are unstable across frames",
            "geometry": "object geometry unlike anything in training",
            "confidence": "detector confidence unusually distributed",
            "class": "unusual class mix",
        }.get(worst[0], "inputs unlike training data")
        return f"{label} ({np.mean(worst[1]):.1f}s from training mean)"

    def score_frame(self, detections, scores, labels, features: np.ndarray) -> Dict:
        """
        Score one frame. Runs without GT — this is inference.

        Args:
            detections: (M, 5) BEV boxes.
            scores: (M,) detector confidences.
            labels: (M,) class ids.
            features: (M, F) from assemble_features (unscaled).
        """
        det = np.asarray(detections, dtype=np.float64).reshape(-1, 5)
        labels = np.asarray(labels).reshape(-1)

        if len(det) == 0:
            # No detections is not the same as an untrustworthy frame; it is a
            # frame with nothing to vouch for. Treat as neutral-in-ODD so the
            # agent does not announce degraded perception on an empty road.
            return {"trust": 1.0, "in_odd": True, "reason": "no detections",
                    "per_detection": []}

        feats = np.asarray(features, dtype=np.float32).reshape(len(det), -1)
        x = self.scaler.transform(feats) if self.scaler is not None else feats
        rel = self.head.predict_proba(torch.from_numpy(np.asarray(x, dtype=np.float32))).numpy()

        w = self._weights(det, labels)
        trust = float((rel * w).sum() / max(w.sum(), 1e-9))
        in_odd = trust >= self.odd_threshold
        return {
            "trust": trust,
            "in_odd": bool(in_odd),
            "reason": "within operational design domain" if in_odd else self._reason(feats),
            "per_detection": [
                {"index": i, "reliability": float(rel[i]), "score": float(scores[i]),
                 "label": int(labels[i]),
                 "class": CLASS_NAMES[int(labels[i])] if int(labels[i]) < len(CLASS_NAMES) else "?",
                 "range_m": float(np.hypot(det[i, 0], det[i, 1])),
                 "risk_weight": float(w[i])}
                for i in range(len(det))
            ],
        }

    def save(self, path: str) -> None:
        torch.save({
            "head": self.head.state_dict(),
            "in_features": self.head.in_features,
            "calib_a": getattr(self.head, "calib_a", 1.0),
            "calib_b": getattr(self.head, "calib_b", 0.0),
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "odd_threshold": self.odd_threshold,
            "geometry": self.geometry,
        }, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "TrustScorer":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        head = IntrospectionHead(in_features=blob["in_features"])
        head.load_state_dict(blob["head"])
        head.calib_a = float(blob.get("calib_a", 1.0))
        head.calib_b = float(blob.get("calib_b", 0.0))
        head.eval()
        scaler = None
        if blob.get("scaler") is not None:
            scaler = FeatureScaler().load_state_dict(blob["scaler"])
        # Older checkpoints predate the field; they were all trained by
        # train_introspection on 2-D detections, so "image" is the right default.
        return cls(head, scaler, blob.get("odd_threshold", 0.5),
                   geometry=blob.get("geometry", "image"))
