"""
Export the per-bin and per-coverage data the Phase 11 figures need.

`train_introspection` writes scalar summaries (ECE, AUROC, AURC). A reliability
diagram needs the per-bin confidence/accuracy pairs and a risk-coverage plot needs
the whole curve, so this re-harvests just the night cell — 121 frames, seconds —
and writes the curve points out.

    python -m tools.export_curves          # -> logs/curve_data.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from nuscenes.nuscenes import NuScenes

from data.dataset import NIGHT_SCENES, version_from_data_root
from evaluation.calibration import expected_calibration_error, risk_coverage_curve
from models.detection.train_detector import build_detector
from models.uncertainty.introspection import TrustScorer
from models.uncertainty.train_introspection import build_loader, harvest_detections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detector_dropout.yaml")
    ap.add_argument("--ckpt", default="checkpoints/detector_dropout_best.pt")
    ap.add_argument("--introspection", default="checkpoints/introspection_nomc.pt")
    ap.add_argument("--mini-root", default="data/raw/v1.0-mini")
    ap.add_argument("--out", default="logs/curve_data.json")
    ap.add_argument("--n-bins", type=int, default=12)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cpu")

    model = build_detector(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    scorer = TrustScorer.load(args.introspection)
    root = Path(args.mini_root)
    nusc = NuScenes(version=version_from_data_root(root), dataroot=str(root), verbose=False)

    out = {}
    for cell, scenes in (("unseen_night", set(NIGHT_SCENES)),):
        print(f"[harvest] {cell}: {sorted(scenes)}")
        h = harvest_detections(model, build_loader(nusc, root, scenes), device, cfg["num_classes"])
        x = torch.from_numpy(scorer.scaler.transform(h["features"]))
        rel = scorer.head.predict_proba(x).numpy()
        raw = h["score"]
        corr = h["correct"]

        entry = {"n": int(len(corr)), "positive_rate": float(corr.mean())}
        for key, s in (("head", rel), ("raw", raw)):
            ece, bins = expected_calibration_error(s, corr, n_bins=args.n_bins)
            # Equal-MASS bins for the reliability diagram: with equal-width bins
            # the high-confidence bins hold almost no detections at night, so the
            # curve's right tail swings wildly on a handful of samples. Quantile
            # edges give every plotted point comparable support.
            _, bins_eq = expected_calibration_error(s, corr, n_bins=args.n_bins,
                                                    equal_mass=True)
            rc = risk_coverage_curve(s, corr)
            entry[key] = {
                "ece": float(ece),
                "bins": {k: (v if isinstance(v, (int, float, list)) else list(v))
                         for k, v in bins.items()},
                "bins_equal_mass": {k: (v if isinstance(v, (int, float, list)) else list(v))
                                    for k, v in bins_eq.items()},
                "coverage": rc["coverage"].tolist(),
                "risk": rc["risk"].tolist(),
                "aurc": float(rc["aurc"]),
            }
            print(f"  {key:5s} ECE {ece:.4f}  AURC {rc['aurc']:.4f}")
        out[cell] = entry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
