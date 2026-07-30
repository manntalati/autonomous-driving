"""
P9-1 — Day/night audit.

Decomposes reported detection accuracy into "the model has not generalised yet"
and "the model has left its operational design domain (ODD)".

The two data roots on disk form a natural experiment:
  * v1.0-trainval blob — 85 scenes (scene-0001..0102), 3376 CAM_FRONT keyframes,
    100% daytime, clear weather. This is the training domain.
  * v1.0-mini — 10 scenes, 3 of which are night (1077, 1094 "after rain",
    1100 "difficult lighting"). Only scene-0061 overlaps the blob, and it is a
    day scene, so the night scenes are genuinely unseen.

We therefore evaluate one detector checkpoint on three cells:

    (seen,   day)   trainval train scenes    — fit ceiling
    (unseen, day)   trainval val scenes      — pure generalisation
    (unseen, night) mini night scenes        — generalisation + domain shift

(unseen, day) -> (unseen, night) is the day->night transfer gap: both are
unseen, so generalisation is held constant and the delta is domain shift alone.

Usage:
    python -m evaluation.day_night_audit configs/detector.yaml \
        --ckpt checkpoints/detector_best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader

from data.dataset import (
    NuScenesDetectionDataset,
    collate_fn,
    get_scene_split,
    version_from_data_root,
)
from models.detection.losses import DetectionLoss
from models.detection.train_detector import build_detector, val_one_epoch, _pick_device

# nuScenes scene descriptions are hand-written and reliably say "Night" for night
# scenes; there is no structured lighting field in the schema.
NIGHT_KEYWORD = "night"
RAIN_KEYWORD = "rain"


def classify_scenes(nusc: NuScenes, scene_names: set[str]) -> dict[str, dict]:
    """Tag each scene with its lighting/weather condition from its description."""
    out = {}
    for scene in nusc.scene:
        if scene["name"] not in scene_names:
            continue
        desc = scene["description"].lower()
        out[scene["name"]] = {
            "night": NIGHT_KEYWORD in desc,
            "rain": RAIN_KEYWORD in desc,
            "description": scene["description"],
        }
    return out


def scenes_on_disk(nusc: NuScenes, data_root: Path) -> set[str]:
    """Scene names whose CAM_FRONT keyframes are actually present (partial blobs)."""
    train_scenes, val_scenes = get_scene_split(nusc, data_root)
    return set(train_scenes) | set(val_scenes)


def build_eval_loader(nusc, data_root, scenes, batch_size, cameras=None) -> DataLoader:
    ds = NuScenesDetectionDataset(
        nusc, data_root, split="val", cameras=cameras, scenes=scenes
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ckpt", default="checkpoints/detector_best.pt")
    ap.add_argument("--trainval-root", default="data/raw/v1.0-trainval")
    ap.add_argument("--mini-root", default="data/raw/v1.0-mini")
    ap.add_argument("--out", default="logs/day_night_audit.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = _pick_device()
    batch_size = cfg.get("batch_size", 4)

    model = build_detector(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    loss_fn = DetectionLoss(cfg["num_classes"])

    tv_root, mini_root = Path(args.trainval_root), Path(args.mini_root)
    nusc_tv = NuScenes(version=version_from_data_root(tv_root), dataroot=str(tv_root), verbose=False)
    nusc_mini = NuScenes(version=version_from_data_root(mini_root), dataroot=str(mini_root), verbose=False)

    tv_train, tv_val = get_scene_split(nusc_tv, tv_root)
    mini_all = scenes_on_disk(nusc_mini, mini_root)
    mini_cond = classify_scenes(nusc_mini, mini_all)
    mini_night = {n for n, c in mini_cond.items() if c["night"]}
    mini_day = {n for n, c in mini_cond.items() if not c["night"]}

    # Guard the core assumption: the night scenes must not be in the training set.
    contaminated = mini_night & set(tv_train)
    if contaminated:
        raise SystemExit(f"night scenes present in training data: {sorted(contaminated)}")

    cells = [
        ("seen_day",     nusc_tv,   tv_root,   set(tv_train)),
        ("unseen_day",   nusc_tv,   tv_root,   set(tv_val)),
        ("unseen_night", nusc_mini, mini_root, mini_night),
        ("unseen_miniday", nusc_mini, mini_root, mini_day - set(tv_train)),
    ]

    results = {}
    for name, nusc, root, scenes in cells:
        if not scenes:
            print(f"[skip] {name}: no scenes")
            continue
        loader = build_eval_loader(nusc, root, scenes, batch_size)
        n = len(loader.dataset)
        print(f"\n[{name}] {len(scenes)} scenes, {n} frames — {sorted(scenes)}")
        metrics = val_one_epoch(model, loader, loss_fn, device, cfg["num_classes"])
        metrics["num_frames"] = n
        metrics["num_scenes"] = len(scenes)
        metrics["scenes"] = sorted(scenes)
        results[name] = metrics
        print(f"  loss {metrics['loss']:.3f} | mAP {metrics['mAP']:.4f} | AP {[round(a,4) for a in metrics['AP']]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"ckpt": args.ckpt, "cells": results, "mini_conditions": mini_cond},
              open(args.out, "w"), indent=2)

    print("\n" + "=" * 68)
    print(f"{'cell':<16}{'frames':>8}{'mAP':>9}{'car':>9}{'ped':>9}{'cyc':>9}")
    print("-" * 68)
    for name, m in results.items():
        ap_ = m["AP"] + [float("nan")] * (3 - len(m["AP"]))
        print(f"{name:<16}{m['num_frames']:>8}{m['mAP']:>9.4f}"
              f"{ap_[0]:>9.4f}{ap_[1]:>9.4f}{ap_[2]:>9.4f}")
    print("=" * 68)
    if "unseen_day" in results and "unseen_night" in results:
        d, n = results["unseen_day"]["mAP"], results["unseen_night"]["mAP"]
        drop = (d - n) / d * 100 if d > 0 else float("nan")
        print(f"day->night transfer gap: {d:.4f} -> {n:.4f}  ({drop:+.1f}% relative)")
    if "seen_day" in results and "unseen_day" in results:
        s, u = results["seen_day"]["mAP"], results["unseen_day"]["mAP"]
        print(f"generalisation gap (day):{s:.4f} -> {u:.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
