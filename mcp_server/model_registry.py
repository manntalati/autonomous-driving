from pathlib import Path
import yaml
import torch
from nuscenes.nuscenes import NuScenes
from demo.pipeline import PerceptionPipeline
from data.dataset import version_from_data_root

class ModelRegistry:
    def __init__(self, cfg_path: str | Path):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.pipeline = PerceptionPipeline(self.cfg, self.device)
        self.nusc = NuScenes(version=version_from_data_root(self.cfg["data_root"]), dataroot=self.cfg['data_root'], verbose=False)