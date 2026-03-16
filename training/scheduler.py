from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional, Union
import torch
import torch.nn as nn

def build_optimizer(model: nn.Module, optimizer_type: Literal["sgd", "adamw"] = "adamw", lr: float = 1e-3, weight_decay: float = 1e-4, momentum: float = 0.9) -> torch.optim.Optimizer:
    decay_parameters = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay_parameters = [p for n, p in model.named_parameters() if p.ndim < 2]
    parameter_groups = [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0}
    ]
    if optimizer_type == 'sgd':
        return torch.optim.SGD(parameter_groups, lr=lr, momentum=momentum)
    else:
        return torch.optim.AdamW(parameter_groups, lr=lr)

def build_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: Literal["cosine", "plateau"] = "cosine", epochs: int = 30, min_lr: float = 1e-6, patience: int = 5) -> Union[torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    else:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, min_lr=min_lr)

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, ckpt_path: str = "best_model.pt") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.ckpt_path = Path(ckpt_path)
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.ckpt_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def load_best(self, model: nn.Module) -> nn.Module:
        model.load_state_dict(torch.load(self.ckpt_path, map_location="cpu", weights_only=True))
        return model