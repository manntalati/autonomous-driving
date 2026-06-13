from __future__ import annotations
import os
from PIL import Image
from nuscenes.nuscenes import NuScenes

CAMERAS = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
]

def _scene_sample_tokens(nusc: NuScenes, scene_name: str) -> list[str]:
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    tokens, tok = [], scene["first_sample_token"]
    while tok:
        tokens.append(tok)
        tok = nusc.get("sample", tok)["next"]
    return tokens

def scene_length(nusc: NuScenes, scene_name: str) -> int:
    return len(_scene_sample_tokens(nusc, scene_name))

def load_surround_images(nusc: NuScenes, scene_name: str, frame_idx: int) -> dict[str, Image.Image]:
    """Return {camera_name: PIL.Image} for the 6 nuScenes cameras at this frame.
    Order: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT.
    """
    tokens = _scene_sample_tokens(nusc, scene_name)
    sample = nusc.get("sample", tokens[frame_idx])
    out: dict[str, Image.Image] = {}
    for cam in CAMERAS:
        sd = nusc.get("sample_data", sample["data"][cam])
        out[cam] = Image.open(os.path.join(nusc.dataroot, sd["filename"])).copy()
    return out
