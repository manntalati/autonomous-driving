"""
P6-4 — flicker (temporal-consistency) evaluation for the Phase 6 temporal detector.

Runs the trained temporal detector frame-by-frame over each validation scene,
pairs detections with GT objects tracked by nuScenes `instance_token`, and
reports the flicker rate (fraction of objects detected at t-1 and t+1 but
missed at t — see evaluation/temporal_metrics.compute_flicker_rate).

Predictions and GT are compared in NATIVE image pixels: GT is projected at
native resolution, and the detector's resized-space boxes are scaled back —
this keeps instance tracking independent of the input transform.

    python -m evaluation.eval_flicker configs/temporal.yaml checkpoints/temporal_best.pt
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility

from data.dataset import get_scene_split, version_from_data_root, LABEL_MAP
from data.transforms import MEAN, STD, INPUT_H, INPUT_W
from models.temporal.train_temporal import build_temporal_detector, _pick_device
from evaluation.temporal_metrics import compute_flicker_rate

CAM = "CAM_FRONT"
_MEAN = np.array(MEAN, dtype=np.float32)
_STD = np.array(STD, dtype=np.float32)


def _load_frame(nusc: NuScenes, data_root: str, cam_sd_token: str) -> torch.Tensor:
    """Load + resize + ImageNet-normalise a CAM_FRONT image → (3, H, W) tensor."""
    sd = nusc.get("sample_data", cam_sd_token)
    img = Image.open(Path(data_root) / sd["filename"]).convert("RGB")
    img = img.resize((INPUT_W, INPUT_H), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()


def _gt_boxes_native(nusc: NuScenes, cam_sd_token: str):
    """2D GT boxes in native pixels + their nuScenes instance tokens."""
    sd = nusc.get("sample_data", cam_sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = np.array(cs["camera_intrinsic"])
    w, h = sd["width"], sd["height"]
    _, boxes, _ = nusc.get_sample_data(cam_sd_token, box_vis_level=BoxVisibility.ANY)
    out_boxes, out_inst = [], []
    for box in boxes:
        if box.name not in LABEL_MAP:
            continue
        pts = view_points(box.corners(), K, normalize=True)
        x1, y1 = float(np.clip(pts[0].min(), 0, w)), float(np.clip(pts[1].min(), 0, h))
        x2, y2 = float(np.clip(pts[0].max(), 0, w)), float(np.clip(pts[1].max(), 0, h))
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            continue
        out_boxes.append([x1, y1, x2, y2])
        out_inst.append(nusc.get("sample_annotation", box.token)["instance_token"])
    return np.array(out_boxes, dtype=np.float32).reshape(-1, 4), out_inst


def evaluate_flicker(cfg_path: str, ckpt_path: str) -> float:
    """Load the trained temporal detector and report its mean flicker rate."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = _pick_device()
    model = build_temporal_detector(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    nusc = NuScenes(version=version_from_data_root(cfg["data_root"]), dataroot=cfg["data_root"], verbose=False)
    seq_len = cfg.get("seq_len", 3)
    _, val_scenes = get_scene_split(nusc, cfg["data_root"])

    scene_rates = []
    for scene in nusc.scene:
        if scene["name"] not in val_scenes:
            continue
        # ordered CAM_FRONT sample_data tokens for the scene
        cam_tokens, t = [], scene["first_sample_token"]
        while t != "":
            sample = nusc.get("sample", t)
            cam_tokens.append(sample["data"][CAM])
            t = sample["next"]
        frames = [_load_frame(nusc, cfg["data_root"], tk) for tk in cam_tokens]

        seq_pred, seq_gt, seq_inst = [], [], []
        for i, cam_sd in enumerate(cam_tokens):
            # 3-frame window ending at frame i, left-padded at the scene start
            window = [frames[max(0, i - k)] for k in reversed(range(seq_len))]
            batch = torch.stack(window).unsqueeze(0).to(device)  # (1, T, 3, H, W)
            with torch.no_grad():
                boxes_list, _, _ = model(batch)
            pred = boxes_list[0].cpu().numpy().reshape(-1, 4)
            # resized (448×800) → native pixels
            sd = nusc.get("sample_data", cam_sd)
            scale = np.array([sd["width"] / INPUT_W, sd["height"] / INPUT_H] * 2, dtype=np.float32)
            if pred.shape[0] > 0:
                pred = pred * scale
            gt, inst = _gt_boxes_native(nusc, cam_sd)
            seq_pred.append(pred)
            seq_gt.append(gt)
            seq_inst.append(inst)

        scene_rates.append(compute_flicker_rate(seq_pred, seq_gt, seq_inst))

    flicker = float(np.mean(scene_rates)) if scene_rates else 0.0
    print(f"[flicker] mean flicker rate over {len(scene_rates)} val scenes: {flicker:.4f}")
    return flicker


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/temporal.yaml"
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/temporal_best.pt"
    evaluate_flicker(cfg_path, ckpt_path)
