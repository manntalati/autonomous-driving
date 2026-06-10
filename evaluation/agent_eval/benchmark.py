from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import random
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from data.dataset import LABEL_MAP, CLASS_NAMES, MINI_VAL_SCENES

_BEV_X_MAX = 51.2
_BEV_X_MIN = 0.0
_LANE_HALF = 2.0
_LANE_OUTER = 6.0

@dataclass
class BenchQuestion:
    qid: str
    question: str
    scene: str
    frame_idx: int
    qtype: str
    gt_answer: Any
    expects_tools: list[str]

def _ego_boxes(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    cam_sd_token = sample["data"]["CAM_FRONT"]
    sd = nusc.get("sample_data", cam_sd_token)
    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
    ego_t = np.array(ego_pose["translation"])
    ego_rot_inv = Quaternion(ego_pose["rotation"]).inverse
    out = []
    for box in nusc.get_boxes(cam_sd_token):
        if box.name not in LABEL_MAP:
            continue
        center = ego_rot_inv.rotate(np.array(box.center) - ego_t)
        cls = CLASS_NAMES[LABEL_MAP[box.name]]
        out.append((cls, float(center[0]), float(center[1])))
    return out

def gen_counting(nusc, scene_name: str, sample_token: str, frame_idx: int) -> list[BenchQuestion]:
    sample = nusc.get("sample", sample_token)
    cam_sd_token = sample["data"]["CAM_FRONT"]
    _, boxes, _ = nusc.get_sample_data(cam_sd_token, box_vis_level=BoxVisibility.ANY)
    counts = {c: 0 for c in CLASS_NAMES}
    for box in boxes:
        if box.name in LABEL_MAP:
            counts[CLASS_NAMES[LABEL_MAP[box.name]]] += 1
    out = []
    for cls, n in counts.items():
        out.append(BenchQuestion(
            qid=f"{scene_name}:{frame_idx}:count_{cls}",
            question=f"How many {cls}s are in {scene_name} frame {frame_idx}?",
            scene=scene_name, frame_idx=frame_idx, qtype="count",
            gt_answer=n,
            expects_tools=["load_frame", "detect_objects"],
        ))
    return out

def gen_presence(nusc, scene_name, sample_token, frame_idx) -> list[BenchQuestion]:
    sample = nusc.get("sample", sample_token)
    cam_sd_token = sample["data"]["CAM_FRONT"]
    _, boxes, _ = nusc.get_sample_data(cam_sd_token, box_vis_level=BoxVisibility.ANY)
    present = {c: False for c in CLASS_NAMES}
    for box in boxes:
        if box.name in LABEL_MAP:
            present[CLASS_NAMES[LABEL_MAP[box.name]]] = True
    out = []
    for cls, p in present.items():
        out.append(BenchQuestion(
            qid=f"{scene_name}:{frame_idx}:presence_{cls}",
            question=f"Are there any {cls}s in {scene_name} frame {frame_idx}?",
            scene=scene_name, frame_idx=frame_idx, qtype="presence",
            gt_answer=bool(p),
            expects_tools=["load_frame", "detect_objects"],
        ))
    return out


def gen_nearest(nusc, scene_name, sample_token, frame_idx) -> list[BenchQuestion]:
    boxes = _ego_boxes(nusc, sample_token)
    out = []
    for cls in CLASS_NAMES:
        candidates = [(x, y) for c, x, y in boxes
                      if c == cls and _BEV_X_MIN <= x <= _BEV_X_MAX]
        if not candidates:
            continue
        nearest = min((x*x + y*y) ** 0.5 for x, y in candidates)
        out.append(BenchQuestion(
            qid=f"{scene_name}:{frame_idx}:nearest_{cls}",
            question=f"How far is the nearest {cls} ahead in {scene_name} frame {frame_idx}? Answer in meters.",
            scene=scene_name, frame_idx=frame_idx, qtype="nearest",
            gt_answer=round(nearest, 1),
            expects_tools=["load_frame", "bev_map"],
        ))
    return out

def gen_spatial(nusc, scene_name, sample_token, frame_idx) -> list[BenchQuestion]:
    boxes = _ego_boxes(nusc, sample_token)
    boxes = [(c, x, y) for c, x, y in boxes if _BEV_X_MIN <= x <= _BEV_X_MAX]

    def in_lane(y, lane):
        if lane == "current": return abs(y) <= _LANE_HALF
        if lane == "left": return _LANE_HALF < y <= _LANE_OUTER
        if lane == "right": return -_LANE_OUTER <= y < -_LANE_HALF

    out = []
    for cls in CLASS_NAMES:
        for lane in ["left", "current", "right"]:
            present = any(c == cls and in_lane(y, lane) for c, x, y in boxes)
            out.append(BenchQuestion(
                qid=f"{scene_name}:{frame_idx}:spatial_{cls}_{lane}",
                question=f"Is there a {cls} in the {lane} lane in {scene_name} frame {frame_idx}?",
                scene=scene_name, frame_idx=frame_idx, qtype="spatial",
                gt_answer=bool(present),
                expects_tools=["load_frame", "bev_map"],
            ))
    return out

def build_benchmark(nusc: NuScenes, val_scenes: tuple[str, ...] = tuple(MINI_VAL_SCENES), seed: int = 0, n_per_frame: int = 4, out_path: str | Path = "logs/agent_eval_benchmark.json") -> list[BenchQuestion]:
    rng = random.Random(seed)
    all_questions: list[BenchQuestion] = []

    for scene in nusc.scene:
        if scene["name"] not in val_scenes:
            continue
        tok, idx = scene["first_sample_token"], 0
        while tok != "":
            qs = (
                gen_counting(nusc, scene["name"], tok, idx)
                + gen_presence(nusc, scene["name"], tok, idx)
                + gen_nearest(nusc, scene["name"], tok, idx)
                + gen_spatial(nusc, scene["name"], tok, idx)
            )
            if len(qs) > n_per_frame:
                qs = rng.sample(qs, n_per_frame)
            all_questions.extend(qs)
            tok = nusc.get("sample", tok)["next"]
            idx += 1

    all_questions.sort(key=lambda q: q.qid)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([q.__dict__ for q in all_questions], f, indent=2)

    return all_questions
