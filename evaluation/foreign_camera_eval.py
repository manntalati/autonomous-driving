"""
P13 — score a detector on simulated foreign cameras.

    python -m evaluation.foreign_camera_eval configs/detector.yaml \
        --ckpt checkpoints/detector_best.pt

Evaluates the same held-out frames as the Phase 9 audit, but pushed through the
`data.foreign_camera` degradation chain, so cross-camera robustness gets a number.
The `native` row is the undegraded control — without it a low score could mean
either "the camera simulation is hard" or "this split is hard".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader

from data.dataset import (NuScenesDetectionDataset, collate_fn, get_scene_split,
                          version_from_data_root)
from data.foreign_camera import PRESETS, ForeignCamera, simulate, transform_boxes
from models.detection.losses import DetectionLoss
from models.detection.train_detector import _pick_device, build_detector, val_one_epoch


class ForeignCameraDataset(NuScenesDetectionDataset):
    """
    A detection dataset whose IMAGES are degraded to look like another camera.

    A subclass, not a wrapper that monkeypatches `_load_image`: replacing a bound
    method on an instance and stashing the original alongside it does not survive
    the pickling that DataLoader workers do, and the stashed reference comes back
    pointing at the patched version — infinite recursion on the first worker read.

    GT boxes are moved with the pixels. The FOV simulation shrinks image content
    toward the centre; leaving the boxes at their original coordinates would score
    misalignment rather than robustness, and even a perfect detector would read
    near zero. Both paths derive the transform from `foreign_camera.fov_transform`
    so they cannot drift apart.
    """

    def __init__(self, *a, camera: ForeignCamera, **kw) -> None:
        super().__init__(*a, **kw)
        self.camera = camera

    def _load_image(self, sd_token: str):
        from PIL import Image
        img, w, h = super()._load_image(sd_token)
        # Seed per frame so the degradation is deterministic across runs — an
        # eval whose noise changes between invocations is not a benchmark.
        rng = np.random.default_rng(abs(hash(sd_token)) % (2 ** 32))
        return Image.fromarray(simulate(np.asarray(img), self.camera, rng)), w, h

    def _get_2d_boxes(self, sd_token: str, img_w: int, img_h: int):
        boxes, labels = super()._get_2d_boxes(sd_token, img_w, img_h)
        if not boxes:
            return boxes, labels
        moved = transform_boxes(boxes, img_w, img_h, self.camera.hfov_deg)
        out_b, out_l = [], []
        for b, l in zip(np.asarray(moved).reshape(-1, 4).tolist(), labels):
            # The shrink can push a box below the 2px floor the base class uses;
            # drop those rather than scoring the detector on sub-pixel targets.
            if (b[2] - b[0]) >= 2 and (b[3] - b[1]) >= 2:
                out_b.append(b)
                out_l.append(l)
        return out_b, out_l


def fov_normalized_val_transform(hfov_deg: float):
    """
    The val pipeline with FOV normalisation prepended — the DEPLOYED BYO path.

    `demo.byo_video.fov_normalize` centre-crops to the training FOV before
    inference. The plain benchmark skips that step and therefore measures the
    worst case; this reproduces what a user actually gets.

    Built from albumentations so the GROUND-TRUTH BOXES FOLLOW THE CROP. Doing the
    crop with raw numpy would leave labels in the old coordinate frame — the same
    misalignment bug that made the first version of this benchmark meaningless.

    The crop fraction equals the shrink ratio the simulation applied, so the crop
    lands exactly on the content region. What cannot be recovered is resolution:
    content downscaled to 44% and back up has genuinely lost detail, which is
    physically right — a wide lens really does trade angular resolution for
    coverage.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    from data.transforms import INPUT_H, INPUT_W, MEAN, NATIVE_H, NATIVE_W, STD
    from demo.byo_video import crop_fraction

    f = crop_fraction(hfov_deg)
    cw = max(int(round(NATIVE_W * f)), 32)
    ch = min(max(int(round(cw * INPUT_H / INPUT_W)), 32), NATIVE_H)
    steps = []
    if f < 1.0:
        steps.append(A.CenterCrop(height=ch, width=cw))
    steps += [A.Resize(INPUT_H, INPUT_W), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
    return A.Compose(steps, bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["labels"], min_visibility=0.3, clip=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ckpt", default="checkpoints/detector_best.pt")
    ap.add_argument("--trainval-root", default="data/raw/v1.0-trainval")
    ap.add_argument("--out", default="logs/foreign_camera_eval.json")
    ap.add_argument("--cameras", default="phone,dashcam,action_cam")
    ap.add_argument("--fov-normalize", action="store_true",
                    help="apply the deployed BYO FOV normalisation before inference")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = _pick_device()
    model = build_detector(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    loss_fn = DetectionLoss(cfg["num_classes"])

    root = Path(args.trainval_root)
    nusc = NuScenes(version=version_from_data_root(root), dataroot=str(root), verbose=False)
    _, val_scenes = get_scene_split(nusc, root)

    def loader(camera=None):
        kw = dict(split="val", scenes=set(val_scenes))
        if camera is not None and args.fov_normalize:
            kw["transform"] = fov_normalized_val_transform(camera.hfov_deg)
        ds = (NuScenesDetectionDataset(nusc, root, **kw) if camera is None
              else ForeignCameraDataset(nusc, root, camera=camera, **kw))
        return DataLoader(ds, batch_size=cfg.get("batch_size", 4), shuffle=False,
                          num_workers=2, collate_fn=collate_fn)

    results = {}
    cells = [("native", None)] + [(n, PRESETS[n]) for n in args.cameras.split(",")]
    for name, cam in cells:
        dl = loader(cam)
        desc = "" if cam is None else f" ({cam.hfov_deg:.0f}° FOV, q{cam.jpeg_quality})"
        print(f"\n[{name}]{desc} {len(dl.dataset)} frames")
        m = val_one_epoch(model, dl, loss_fn, device, cfg["num_classes"])
        results[name] = {"mAP": m["mAP"], "AP": m["AP"], "loss": m["loss"],
                         "frames": len(dl.dataset)}
        print(f"  mAP {m['mAP']:.4f} | AP {[round(a, 3) for a in m['AP']]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"ckpt": args.ckpt, "cells": results}, open(args.out, "w"), indent=2)

    base = results["native"]["mAP"]
    print("\n" + "=" * 60)
    print(f"{'camera':<14}{'mAP':>9}{'vs native':>12}{'retained':>11}")
    print("-" * 60)
    for k, r in results.items():
        rel = "" if k == "native" else f"{r['mAP'] - base:+.4f}"
        pct = "" if k == "native" else f"{r['mAP'] / base * 100:.0f}%"
        print(f"{k:<14}{r['mAP']:>9.4f}{rel:>12}{pct:>11}")
    print("=" * 60)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
