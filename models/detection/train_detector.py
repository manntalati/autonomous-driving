from models.detection.detector import FPNDetector

def build_detector(cfg: dict) -> FPNDetector:
    """
    Load Phase 1 backbone (optionally pretrained), attach FPN + head, return full detector.
    """
    pass

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler) -> dict:
    pass

def val_one_epoch(model, loader, loss_fn, device) -> dict:
    pass

def main(cfg_path: str) -> None:
    """
    Load config → build model → get_loaders() → train loop with mAP eval each epoch.
    """
    pass
