"""
Sequence dataset for temporal fusion (Phase 6).

Serves length-`seq_len` windows of consecutive CAM_FRONT keyframes from a
single scene, paired with the GT boxes of the CURRENT (last) frame.

Per-frame image loading + 3D→2D box projection are reused from
NuScenesDetectionDataset. A deterministic (val-style) transform is used for
every frame so a window stays temporally coherent — random per-frame
augmentation would shift each frame differently and break the correspondence
the temporal module relies on.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import torch
from nuscenes.nuscenes import NuScenes
from torch.utils.data import Dataset

from data.dataset import NuScenesDetectionDataset, get_scene_split
from data.transforms import get_val_transforms


class NuScenesSequenceDataset(Dataset):
    """
    Windows of consecutive keyframes. window[-1] is the current frame; the
    earlier entries are its past context.
    """

    def __init__(self, nusc: NuScenes, data_root: str | Path, split: str = "train", seq_len: int = 3, cameras: Optional[List[str]] = None) -> None:
        """
        Args:
          nusc — NuScenes instance.
          data_root — path to v1.0-mini.
          split — "train" or "val".
          seq_len — frames per window (current + seq_len-1 past).
          cameras — camera channels (default ["CAM_FRONT"]).
        """
        self.nusc = nusc
        self.split = split
        self.seq_len = seq_len
        # inner per-frame dataset — deterministic transform keeps windows coherent
        self.frame_ds = NuScenesDetectionDataset(
            nusc, data_root, split=split, cameras=cameras, transform=get_val_transforms()
        )
        # sample_token → index into frame_ds (one CAM_FRONT entry per token)
        self.token_to_idx = {tok: i for i, (tok, _) in enumerate(self.frame_ds.index)}
        self.index = self._build_index()

    def _build_index(self) -> List[List[str]]:
        """
        Walk each scene's sample linked list; emit every length-seq_len window
        of consecutive sample tokens. window[-1] is the current frame.
        Returns: list of windows, each a list of seq_len sample tokens.
        """
        windows: List[List[str]] = []
        train_scenes, val_scenes = get_scene_split(self.nusc, self.frame_ds.data_root)
        scenes = train_scenes if self.split == "train" else val_scenes
        for scene in self.nusc.scene:
            if scene["name"] not in scenes:
                continue
            tokens: List[str] = []
            token = scene["first_sample_token"]
            while token != "":
                tokens.append(token)
                token = self.nusc.get("sample", token)["next"]
            for i in range(self.seq_len - 1, len(tokens)):
                windows.append(tokens[i - self.seq_len + 1: i + 1])
        return windows

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        """
        Returns: (frames, target) where
          frames — (seq_len, 3, H, W) tensor; frames[-1] is the current frame.
          target — the CURRENT frame's detection target dict ('boxes','labels','meta').
        Pipeline:
          1. for each sample token in the window: self.frame_ds[self.token_to_idx[token]]
             → (image, target).
          2. stack the seq_len images into (seq_len, 3, H, W).
          3. return that stack + the LAST frame's target.
        """
        window = self.index[idx]
        images = []
        target = None
        for token in window:
            image, target = self.frame_ds[self.token_to_idx[token]]
            images.append(image)
        frames = torch.stack(images, dim=0)
        return frames, target


def sequence_collate_fn(batch):
    """
    Stack frame sequences into (B, seq_len, 3, H, W); keep targets as a list
    (variable box counts per frame).
    Returns: (frames, targets).
    """
    frames = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return frames, targets
