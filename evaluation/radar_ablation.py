"""
P10-4 — Radar ablation: does radar close the *night* gap specifically?

THE EXPERIMENT
--------------
A 2x2. Two models (camera-only, camera+radar) crossed with two conditions
(day, night), all on unseen scenes:

                          unseen day        unseen night
    camera only            A                 B
    camera + radar         C                 D

    Day benefit   = C - A
    Night benefit = D - B

The Phase 10 claim is `Night benefit >> Day benefit`. If radar merely adds
capacity, both improve about equally and the ODD argument fails.

Range stratification splits every cell into near/mid/far, because radar measures
range directly where camera depth degrades. Averaging over range would blur
exactly the effect worth showing.

THE n=1 CAVEAT — state this next to the headline number
-------------------------------------------------------
Each arm is a SINGLE training run, so this ablation can only support a LARGE
radar effect. Run-to-run variance is substantial: two runs of the identical
camera-only config produced epoch-1 BEV mIoU of 0.124 and 0.185, a ~50% swing
from data order and initialisation alone. Both arms are now seeded (seed: 0) and
trained for a fixed 12 epochs, which removes data-order variance and matches
training maturity — but it does not make one run a distribution.

Two residual asymmetries seeding cannot remove:
  * Building the radar branch consumes extra RNG draws, so the shared modules
    constructed after it (BEVEncoder, detection and seg heads) start from
    different weights in the radar arm than in the camera arm.
  * The radar arm has more parameters, so any gain is partly capacity. That is
    precisely what the day-vs-night contrast is designed to separate: added
    capacity should help both conditions, whereas illumination invariance should
    help night specifically.

So the defensible claim is the DIFFERENCE OF DIFFERENCES (night benefit vs day
benefit), plus the direction of the learned fusion gate — not a small absolute
mAP delta. If night and day benefit land within ~0.02 of each other, report "no
detectable effect at this sample size" rather than reading a sign off noise.
A real effect size needs 3+ seeds per arm (~19 h at the measured 16 min/epoch),
which is the honest next step if the single-run result looks marginal.

Usage:
    python -m evaluation.radar_ablation \
        --camera-config configs/bev_surround_p10.yaml \
        --camera-ckpt   checkpoints/bev_surround_p10_last.pt \
        --radar-config  configs/bev_radar.yaml \
        --radar-ckpt    checkpoints/bev_radar_last.pt
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

from data.bev_dataset import NuScenesBEVDataset, bev_collate_fn
from data.dataset import NIGHT_SCENES, get_scene_split, version_from_data_root
from evaluation.bev_matching import box_ranges, compute_bev_map
from models.bev.bev_detector import decode_bev_detections
from models.bev.train_bev import (
    _pick_device,
    _stack_calibration,
    _stack_radar,
    build_bev_detector,
)

RANGE_BUCKETS: List[Tuple[str, float, float]] = [
    ("near", 0.0, 20.0),
    ("mid", 20.0, 35.0),
    ("far", 35.0, 51.2),
]


def bucket_by_range(boxes, labels, scores=None) -> Dict[str, tuple]:
    """
    Split BEV boxes into RANGE_BUCKETS by distance from the ego origin.

    Args:
        boxes: (M, 5) [x, y, length, width, yaw] ego-frame BEV boxes.
        labels: (M,) class ids.
        scores: (M,) optional confidence scores.
    Returns: bucket name -> (boxes, labels, scores) subset.

    Range is sqrt(x^2 + y^2), matching the `range_m` convention the MCP tools
    already report, so these numbers stay comparable across the project.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 5)
    labels = np.asarray(labels).reshape(-1)
    scores = None if scores is None else np.asarray(scores, dtype=np.float64).reshape(-1)
    rng = box_ranges(boxes)
    out = {}
    for name, lo, hi in RANGE_BUCKETS:
        m = (rng >= lo) & (rng < hi)
        out[name] = (boxes[m], labels[m], None if scores is None else scores[m])
    return out


@torch.no_grad()
def evaluate_cell(model, loader, device, num_classes: int, xbound, ybound,
                  score_threshold: float = 0.1, stratify: bool = True) -> Dict:
    """
    Evaluate one model on one condition cell, optionally stratified by range.

    Returns overall mAP/AP plus, when stratify=True, per-bucket mAP, per-bucket AP
    and per-bucket GT object counts. Matching is centre-distance (see
    evaluation/bev_matching.py) — `compute_map` assumes axis-aligned image boxes
    and cannot be reused for rotated BEV boxes.
    """
    model.eval()
    preds: List[dict] = []
    gts: List[dict] = []
    gates: List[float] = []

    for images, targets in loader:
        images = images.to(device)
        intrinsics, cam_to_ego = _stack_calibration(targets, device)
        radar_bev = _stack_radar(targets, device)
        outputs = model(images, intrinsics, cam_to_ego, radar_bev=radar_bev)
        if "gate" in outputs:
            gates.append(outputs["gate"])

        heat = torch.sigmoid(outputs["heatmap"])
        for b in range(images.shape[0]):
            boxes, scores, labels = decode_bev_detections(
                heat[b], outputs["regression"][b], xbound, ybound,
                score_threshold=score_threshold,
            )
            preds.append({
                "boxes": np.asarray(boxes).reshape(-1, 5).tolist(),
                "scores": np.asarray(scores).reshape(-1).tolist(),
                "labels": np.asarray(labels).reshape(-1).tolist(),
            })
            gts.append({
                "boxes": targets[b]["boxes"].cpu().numpy().reshape(-1, 5).tolist(),
                "labels": targets[b]["labels"].cpu().numpy().reshape(-1).tolist(),
            })

    mean_ap, per_class = compute_bev_map(preds, gts, num_classes)
    result: Dict = {
        "mAP": mean_ap,
        "AP": per_class,
        "num_frames": len(gts),
        "num_gt": int(sum(len(g["labels"]) for g in gts)),
    }
    if gates:
        result["mean_gate"] = float(np.mean(gates))

    if stratify:
        buckets: Dict[str, Dict] = {}
        for name, lo, hi in RANGE_BUCKETS:
            p_b, g_b = [], []
            for p, g in zip(preds, gts):
                pb = bucket_by_range(p["boxes"], p["labels"], p["scores"])[name]
                gb = bucket_by_range(g["boxes"], g["labels"])[name]
                p_b.append({"boxes": pb[0].tolist(), "labels": pb[1].tolist(), "scores": pb[2].tolist()})
                g_b.append({"boxes": gb[0].tolist(), "labels": gb[1].tolist()})
            m, ap = compute_bev_map(p_b, g_b, num_classes)
            n_gt = int(sum(len(x["labels"]) for x in g_b))
            # Report the GT count next to every AP: an AP over a handful of boxes
            # is not a measurement, and the far/night bucket will be thin.
            buckets[name] = {"mAP": m, "AP": ap, "num_gt": n_gt, "range": [lo, hi]}
        result["buckets"] = buckets
    return result


def build_loader(cfg: dict, nusc, data_root, scenes, use_radar: bool) -> DataLoader:
    ds = NuScenesBEVDataset(
        nusc, data_root, split="val",
        image_size=tuple(cfg.get("image_size", [448, 800])),
        xbound=tuple(cfg["xbound"]), ybound=tuple(cfg["ybound"]), dbound=tuple(cfg["dbound"]),
        cameras=cfg.get("cameras"),
        use_radar=use_radar,
        radar_channels=cfg.get("radar_channels"),
        radar_dilate=cfg.get("radar_dilate", 1),
        scenes=scenes,
    )
    return DataLoader(ds, batch_size=cfg.get("batch_size", 1), shuffle=False,
                      num_workers=2, collate_fn=bev_collate_fn)


def load_model(cfg, ckpt, device):
    model = build_bev_detector(cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-config", default="configs/bev_surround.yaml")
    ap.add_argument("--camera-ckpt", default="checkpoints/bev_surround_best.pt")
    ap.add_argument("--radar-config", default="configs/bev_radar.yaml")
    ap.add_argument("--radar-ckpt", default="checkpoints/bev_radar_best.pt")
    ap.add_argument("--trainval-root", default="data/raw/v1.0-trainval")
    ap.add_argument("--mini-root", default="data/raw/v1.0-mini")
    ap.add_argument("--out", default="logs/radar_ablation.json")
    args = ap.parse_args()

    device = _pick_device()
    cam_cfg = yaml.safe_load(open(args.camera_config))
    rad_cfg = yaml.safe_load(open(args.radar_config))
    num_classes = cam_cfg["num_classes"]

    tv_root, mini_root = Path(args.trainval_root), Path(args.mini_root)
    nusc_tv = NuScenes(version=version_from_data_root(tv_root), dataroot=str(tv_root), verbose=False)
    nusc_mini = NuScenes(version=version_from_data_root(mini_root), dataroot=str(mini_root), verbose=False)
    _, tv_val = get_scene_split(nusc_tv, tv_root)

    conditions = [
        ("unseen_day", nusc_tv, tv_root, set(tv_val)),
        ("unseen_night", nusc_mini, mini_root, set(NIGHT_SCENES)),
    ]
    models = [
        ("camera", cam_cfg, args.camera_ckpt, False),
        ("camera_radar", rad_cfg, args.radar_ckpt, True),
    ]

    results: Dict[str, Dict] = {}
    for mname, cfg, ckpt, use_radar in models:
        if not Path(ckpt).exists():
            print(f"[skip] {mname}: {ckpt} not found — train it first")
            continue
        model = load_model(cfg, ckpt, device)
        for cname, nusc, root, scenes in conditions:
            loader = build_loader(cfg, nusc, root, scenes, use_radar)
            print(f"\n[{mname} / {cname}] {len(scenes)} scenes, {len(loader.dataset)} frames")
            r = evaluate_cell(model, loader, device, num_classes,
                              tuple(cfg["xbound"]), tuple(cfg["ybound"]))
            results[f"{mname}/{cname}"] = r
            gate = f" | gate {r['mean_gate']:.3f}" if "mean_gate" in r else ""
            print(f"  mAP {r['mAP']:.4f} | GT {r['num_gt']}{gate}")
            for bname, b in r.get("buckets", {}).items():
                print(f"    {bname:5s} mAP {b['mAP']:.4f}  (GT {b['num_gt']})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)

    def get(m, c, key="mAP"):
        return results.get(f"{m}/{c}", {}).get(key)

    print("\n" + "=" * 62)
    print(f"{'':<16}{'unseen day':>14}{'unseen night':>16}")
    print("-" * 62)
    for m in ("camera", "camera_radar"):
        d, n = get(m, "unseen_day"), get(m, "unseen_night")
        print(f"{m:<16}{d if d is None else f'{d:>14.4f}'}{n if n is None else f'{n:>16.4f}'}")
    print("=" * 62)
    a, b = get("camera", "unseen_day"), get("camera", "unseen_night")
    c, d = get("camera_radar", "unseen_day"), get("camera_radar", "unseen_night")
    if None not in (a, b, c, d):
        print(f"day benefit  : {c - a:+.4f}")
        print(f"night benefit: {d - b:+.4f}   <-- the Phase 10 claim")
        gd, gn = get("camera_radar", "unseen_day", "mean_gate"), get("camera_radar", "unseen_night", "mean_gate")
        if gd is not None and gn is not None:
            print(f"mean gate    : day {gd:.3f} -> night {gn:.3f} "
                  f"({'shifts toward radar' if gn < gd else 'NO shift toward radar'})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
