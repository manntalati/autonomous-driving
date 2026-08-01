"""
P11 — Harvest detections, label them, train the introspection head, evaluate.

PIPELINE
    1. harvest_detections  — run the FROZEN detector over a split, keep every
       surviving detection with its features and a TP/FP label.
    2. train_head          — fit the IntrospectionHead on the harvest.
    3. evaluate            — ECE / AUROC / risk-coverage, split by condition,
       always against the raw-score baseline.

THE SPLIT DISCIPLINE (enforced here, not just documented)
---------------------------------------------------------
    detector training      trainval train (72 scenes)  — the detector saw this
    introspection training trainval val   (13 scenes)  — head trains here
    evaluation             mini day + mini night       — touched once

`main` refuses to train the head on scenes the detector was fitted on, because
that is the one mistake that produces a great-looking AUROC that means nothing.

Usage:
    python -m models.uncertainty.train_introspection configs/detector.yaml \
        --ckpt checkpoints/detector_best.pt --out checkpoints/introspection.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader

from data.dataset import (
    NIGHT_SCENES,
    NuScenesDetectionDataset,
    collate_fn,
    get_scene_split,
    version_from_data_root,
)
from evaluation.calibration import (
    bootstrap_ece,
    expected_calibration_error,
    failure_prediction_auroc,
    risk_coverage_curve,
    selective_summary,
    platt_scale,
)
from models.detection.box_utils import compute_iou
from models.detection.train_detector import _pick_device, build_detector
from models.uncertainty.introspection import IntrospectionHead, TrustScorer
from models.uncertainty.mc_dropout import (
    MCDropoutPredictor,
    active_dropout_p,
    gather_anchor_uncertainty,
)
from models.uncertainty.signals import FEATURE_NAMES, FeatureScaler, assemble_features

# A detection counts as correct if it overlaps an unmatched GT of the same class
# at or above this IoU — the same convention models/detection/map.py uses.
TP_IOU = 0.5


def _label_detections(boxes, labels, gt_boxes, gt_labels, iou_threshold=TP_IOU) -> np.ndarray:
    """
    TP/FP label per detection, greedy by input order (already score-sorted).
    Each GT may be claimed once, so duplicate detections are false positives.
    """
    n = len(boxes)
    out = np.zeros(n, dtype=np.int64)
    if n == 0 or len(gt_boxes) == 0:
        return out
    ious = compute_iou(boxes, gt_boxes).cpu().numpy()
    used = np.zeros(len(gt_boxes), dtype=bool)
    lab = labels.cpu().numpy()
    glab = gt_labels.cpu().numpy()
    for i in range(n):
        cand = ious[i].copy()
        cand[used] = -1.0
        cand[glab != lab[i]] = -1.0          # class must match
        j = int(np.argmax(cand))
        if cand[j] >= iou_threshold:
            out[i] = 1
            used[j] = True
    return out


@torch.no_grad()
def harvest_detections(
    model,
    loader,
    device,
    num_classes: int = 3,
    mc: Optional[MCDropoutPredictor] = None,
) -> Dict[str, np.ndarray]:
    """
    Run the frozen detector over a loader and collect per-detection rows.

    Returns dict of arrays: features (M, F), correct (M,), score (M,), label (M,),
    frame (M,) — frame index, so results can be grouped per frame later.

    Detections here are 2-D image boxes, so the geometry block of
    `assemble_features` is fed [cx, cy, w, h, 0] rather than a BEV box: the
    feature extractor only reads centre/extent, and keeping one feature layout
    across the 2-D and BEV paths avoids a second scaler and a second head.
    """
    model.eval()
    feats_all, correct_all, score_all, label_all, frame_all = [], [], [], [], []
    frame_idx = 0

    for images, targets in loader:
        images = images.to(device)
        cls_logits, bbox_deltas, anchors = model(images, return_raw=True)
        image_size = (images.shape[-2], images.shape[-1])
        boxes_l, scores_l, labels_l, idx_l = model.postprocess(
            cls_logits, bbox_deltas, anchors, image_size, return_indices=True
        )

        mc_stats = None
        if mc is not None:
            mc_stats = mc(images)

        for b in range(images.shape[0]):
            boxes, scores, labels, aidx = boxes_l[b], scores_l[b], labels_l[b], idx_l[b]
            if len(boxes) == 0:
                frame_idx += 1
                continue
            gt_boxes = targets[b]["boxes"].to(device)
            gt_labels = targets[b]["labels"].to(device)
            correct = _label_detections(boxes, labels, gt_boxes, gt_labels)

            # geometry expects [x, y, length, width, yaw]; supply the 2-D analogue
            bx = boxes.cpu().numpy()
            cx = (bx[:, 0] + bx[:, 2]) / 2.0
            cy = (bx[:, 1] + bx[:, 3]) / 2.0
            w = np.maximum(bx[:, 2] - bx[:, 0], 1e-6)
            h = np.maximum(bx[:, 3] - bx[:, 1], 1e-6)
            geom_boxes = np.stack([cx, cy, w, h, np.zeros_like(cx)], axis=1)

            epi = None
            if mc_stats is not None:
                one = {k: (v[b:b + 1] if torch.is_tensor(v) and v.dim() > 2 else v)
                       for k, v in mc_stats.items()}
                u = gather_anchor_uncertainty(one, aidx, labels)
                epi = {"score_var": u["score_var"].cpu().numpy(),
                       "box_var": u["box_var"].cpu().numpy()}

            f = assemble_features(geom_boxes, scores.cpu().numpy(), labels.cpu().numpy(),
                                  num_classes=num_classes, epistemic=epi)
            feats_all.append(f)
            correct_all.append(correct)
            score_all.append(scores.cpu().numpy())
            label_all.append(labels.cpu().numpy())
            frame_all.append(np.full(len(f), frame_idx))
            frame_idx += 1

    if not feats_all:
        return {k: np.zeros(0) for k in ("features", "correct", "score", "label", "frame")}
    return {
        "features": np.concatenate(feats_all).astype(np.float32),
        "correct": np.concatenate(correct_all).astype(np.int64),
        "score": np.concatenate(score_all).astype(np.float32),
        "label": np.concatenate(label_all).astype(np.int64),
        "frame": np.concatenate(frame_all).astype(np.int64),
    }


def train_head(harvest: Dict[str, np.ndarray], epochs: int = 200, lr: float = 1e-3,
               hidden: int = 64, seed: int = 0, verbose: bool = True
               ) -> Tuple[IntrospectionHead, FeatureScaler]:
    """
    Fit the introspection head on a harvest.

    `pos_weight` handles imbalance rather than resampling: resampling distorts the
    base rate and breaks calibration, which is the one thing this head must get
    right.
    """
    torch.manual_seed(seed)
    x_raw, y = harvest["features"], harvest["correct"]
    if len(x_raw) == 0:
        raise RuntimeError("empty harvest — the detector produced no detections")

    scaler = FeatureScaler().fit(x_raw)
    x = torch.from_numpy(scaler.transform(x_raw))
    t = torch.from_numpy(y).float()

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    if verbose:
        print(f"  harvest: {len(y)} detections, {n_pos} TP / {n_neg} FP "
              f"(positive rate {y.mean():.3f}), pos_weight {pos_weight.item():.2f}")

    head = IntrospectionHead(in_features=x.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    head.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = lossf(head(x), t)
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 50 == 0:
            print(f"    epoch {ep + 1:3d}  loss {loss.item():.4f}")
    head.eval()

    # pos_weight up-weights the minority class to help ranking, but it also
    # deliberately pushes predicted probabilities away from the true base rate —
    # so the head trains well-ranked and badly CALIBRATED (measured: ECE 0.157 vs
    # 0.060 for the raw score). Temperature scaling repairs that: dividing logits
    # by a positive constant cannot change the ordering, so AUROC is untouched
    # while ECE drops. Fitted on the introspection training harvest, which is
    # disjoint from every evaluation split.
    with torch.no_grad():
        logits = head(x).numpy()
    head.calib_a, head.calib_b = platt_scale(logits, y)
    if verbose:
        print(f"  fitted calibration: a={head.calib_a:.3f} b={head.calib_b:.3f}")
    return head, scaler


def evaluate(head: IntrospectionHead, scaler: FeatureScaler,
             harvest: Dict[str, np.ndarray], name: str) -> Dict:
    """
    ECE / AUROC / risk-coverage for the head, always beside the raw-score baseline.
    """
    x = torch.from_numpy(scaler.transform(harvest["features"]))
    rel = head.predict_proba(x).numpy()
    raw = harvest["score"]
    corr = harvest["correct"]

    out = {
        "cell": name,
        "n_detections": int(len(corr)),
        "positive_rate": float(corr.mean()) if len(corr) else float("nan"),
        "head": {
            **failure_prediction_auroc(rel, corr),
            "ece": bootstrap_ece(rel, corr, n_boot=300),
            "aurc": risk_coverage_curve(rel, corr)["aurc"],
            "selective_70": selective_summary(rel, corr, 0.7),
        },
        "baseline_raw_score": {
            **failure_prediction_auroc(raw, corr),
            "ece": bootstrap_ece(raw, corr, n_boot=300),
            "aurc": risk_coverage_curve(raw, corr)["aurc"],
            "selective_70": selective_summary(raw, corr, 0.7),
        },
    }
    return out


def build_loader(nusc, data_root, scenes, batch_size=4):
    ds = NuScenesDetectionDataset(nusc, data_root, split="val", scenes=scenes)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2,
                      collate_fn=collate_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ckpt", default="checkpoints/detector_best.pt")
    ap.add_argument("--out", default="checkpoints/introspection.pt")
    ap.add_argument("--trainval-root", default="data/raw/v1.0-trainval")
    ap.add_argument("--mini-root", default="data/raw/v1.0-mini")
    ap.add_argument("--mc-samples", type=int, default=0,
                    help="MC-dropout passes; 0 disables (requires head_dropout > 0)")
    ap.add_argument("--report", default="logs/introspection_eval.json")
    ap.add_argument("--device", default=None,
                    help="force a device (e.g. 'cpu'). Default: auto. Use 'cpu' to "
                         "run this alongside a GPU training job without contending "
                         "for the single MPS device.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(args.device) if args.device else _pick_device()
    print(f"device: {device}")
    model = build_detector(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    mc = None
    if args.mc_samples > 0:
        ps = [p for p in active_dropout_p(model) if p > 0]
        if not ps:
            raise SystemExit(
                "--mc-samples requires dropout: set head_dropout > 0 in the config "
                "and fine-tune, then re-run. Ensembles are the alternative."
            )
        mc = MCDropoutPredictor(model, num_samples=args.mc_samples)

    tv_root, mini_root = Path(args.trainval_root), Path(args.mini_root)
    nusc_tv = NuScenes(version=version_from_data_root(tv_root), dataroot=str(tv_root), verbose=False)
    nusc_mini = NuScenes(version=version_from_data_root(mini_root), dataroot=str(mini_root), verbose=False)
    tv_train, tv_val = get_scene_split(nusc_tv, tv_root)

    mini_all = set(get_scene_split(nusc_mini, mini_root)[0]) | set(get_scene_split(nusc_mini, mini_root)[1])
    mini_night = set(NIGHT_SCENES) & mini_all
    mini_day = (mini_all - mini_night) - set(tv_train)

    # Enforce the split discipline rather than trusting it.
    leak = set(tv_val) & set(tv_train)
    if leak:
        raise SystemExit(f"introspection split overlaps detector training: {sorted(leak)}")

    print(f"[harvest] introspection-train = trainval val ({len(tv_val)} scenes)")
    train_h = harvest_detections(model, build_loader(nusc_tv, tv_root, set(tv_val)),
                                 device, cfg["num_classes"], mc)
    head, scaler = train_head(train_h)

    scorer = TrustScorer(head, scaler)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scorer.save(args.out)
    print(f"[saved] {args.out}")

    report = {"train_cell": evaluate(head, scaler, train_h, "introspection_train")}
    for name, nusc, root, scenes in [
        ("unseen_day", nusc_mini, mini_root, mini_day),
        ("unseen_night", nusc_mini, mini_root, mini_night),
    ]:
        if not scenes:
            continue
        print(f"\n[eval] {name}: {sorted(scenes)}")
        h = harvest_detections(model, build_loader(nusc, root, scenes), device,
                               cfg["num_classes"], mc)
        report[name] = evaluate(head, scaler, h, name)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.report, "w"), indent=2, default=float)

    print("\n" + "=" * 78)
    print(f"{'cell':<20}{'n':>7}{'AUROC head':>12}{'AUROC raw':>11}{'ECE head':>10}{'ECE raw':>9}")
    print("-" * 78)
    for k, r in report.items():
        if "head" not in r:
            continue
        print(f"{k:<20}{r['n_detections']:>7}{r['head']['auroc']:>12.3f}"
              f"{r['baseline_raw_score']['auroc']:>11.3f}"
              f"{r['head']['ece']['ece']:>10.3f}{r['baseline_raw_score']['ece']['ece']:>9.3f}")
    print("=" * 78)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
