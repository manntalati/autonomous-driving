"""
P1-5 — LR Scheduling & Early Stopping
========================================
CS 444 concept: learning rate schedules, regularization, preventing overfitting.

Provides:
  - build_optimizer()  — SGD with momentum or AdamW + weight decay
  - build_scheduler()  — CosineAnnealingLR or ReduceLROnPlateau
  - EarlyStopping      — stops training when val loss stagnates + saves best ckpt
"""

from __future__ import annotations
import copy
from pathlib import Path
from typing import Literal, Optional
import torch
import torch.nn as nn


# ── Optimizer factory ─────────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    optimizer_type: Literal["sgd", "adamw"] = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Create an optimizer with weight decay applied only to non-bias parameters.

    Weight decay on bias and BN parameters hurts — filter them out.

    Args:
        model:          The model whose parameters to optimize.
        optimizer_type: 'sgd' (momentum) or 'adamw'.
        lr:             Initial learning rate.
        weight_decay:   L2 regularization strength.
        momentum:       Momentum for SGD (ignored for AdamW).

    Returns:
        Configured optimizer.
    """
    # TODO P1-5a: Split model parameters into two groups:
    #   decay_params:    weight tensors (ndim >= 2)
    #   no_decay_params: biases + BN weight/bias (ndim < 2)
    #
    # Then:
    #   param_groups = [
    #       {"params": decay_params,    "weight_decay": weight_decay},
    #       {"params": no_decay_params, "weight_decay": 0.0},
    #   ]
    #
    # Return torch.optim.AdamW(param_groups, lr=lr)
    #   or  torch.optim.SGD(param_groups, lr=lr, momentum=momentum)
    raise NotImplementedError


# ── Scheduler factory ─────────────────────────────────────────────────────────

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: Literal["cosine", "plateau"] = "cosine",
    epochs: int = 30,
    min_lr: float = 1e-6,
    patience: int = 5,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create a learning rate scheduler.

    Args:
        optimizer:      The optimizer to schedule.
        scheduler_type: 'cosine' (CosineAnnealingLR) or 'plateau' (ReduceLROnPlateau).
        epochs:         Total training epochs (used by cosine).
        min_lr:         Minimum LR floor.
        patience:       Epochs without improvement before plateau reduces LR.

    Returns:
        Configured LR scheduler.

    CS 444 note:
        Cosine annealing smoothly decays LR following a cosine curve — allows
        the model to first explore broadly then settle into a minimum.
        ReduceLROnPlateau is adaptive — reduce only when val loss stops improving.
    """
    # TODO P1-5b: Return the appropriate scheduler.
    #   'cosine' → torch.optim.lr_scheduler.CosineAnnealingLR(
    #                  optimizer, T_max=epochs, eta_min=min_lr)
    #   'plateau' → torch.optim.lr_scheduler.ReduceLROnPlateau(
    #                  optimizer, mode='min', patience=patience, min_lr=min_lr)
    raise NotImplementedError


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss hasn't improved for `patience` epochs.

    Also saves the best model checkpoint to disk.

    Args:
        patience:  Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        ckpt_path: Where to save the best model state dict (.pt file).
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        ckpt_path: str = "best_model.pt",
    ) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.ckpt_path  = Path(ckpt_path)
        self.best_loss: float = float("inf")
        self.counter:   int   = 0
        self.should_stop: bool = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """Check improvement and update counter / save checkpoint.

        Args:
            val_loss: Current epoch validation loss.
            model:    Model to checkpoint if improved.

        Side effects:
            - Saves model state dict if val_loss improved.
            - Sets self.should_stop = True if patience exhausted.
        """
        # TODO P1-5c: Implement early stopping logic.
        #   If val_loss < self.best_loss - self.min_delta:
        #       Update self.best_loss, reset self.counter.
        #       Save model.state_dict() to self.ckpt_path.
        #   Else:
        #       Increment self.counter.
        #       If self.counter >= self.patience: self.should_stop = True
        raise NotImplementedError

    def load_best(self, model: nn.Module) -> nn.Module:
        """Restore the best checkpoint into model and return it."""
        model.load_state_dict(torch.load(self.ckpt_path, map_location="cpu"))
        return model
