"""
P1-4 — Training Loop
======================
CS 444 concept: mini-batch SGD, forward/backward/update cycle, metric tracking.

Trainer wraps a model, optimizer, loss, and dataloaders into clean
train_epoch / val_epoch methods. Called from a top-level train script.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class AverageMeter:
    """Running mean for a scalar metric (loss, accuracy, etc.)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    """Manages training and validation for a classification backbone.

    Args:
        model:      The nn.Module to train.
        optimizer:  Optimizer (SGD with momentum or AdamW).
        criterion:  Loss function (nn.CrossEntropyLoss).
        train_loader: DataLoader for training set.
        val_loader:   DataLoader for validation set.
        device:     'cuda', 'mps', or 'cpu'.
        scaler:     Optional GradScaler for AMP (mixed precision, P1-5).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scaler = scaler

    # ------------------------------------------------------------------ #
    # Train one epoch                                                      #
    # ------------------------------------------------------------------ #

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one full pass over the training set.

        Steps per batch:
          1. Move data to device.
          2. Zero gradients.
          3. Forward pass (with AMP context if scaler is set).
          4. Compute loss.
          5. Backward pass (scaler.scale if AMP).
          6. Optimizer step (scaler.step + update if AMP).
          7. Accumulate loss and top-1 accuracy.

        Returns:
            {'loss': avg_loss, 'acc': avg_top1_accuracy}
        """
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter  = AverageMeter()

        for batch_idx, batch in enumerate(self.train_loader):
            # TODO P1-4a: Unpack batch.
            # The dataset returns dicts; for the backbone subtask we pass
            # images + labels. Adjust key names to match your dataset's collate_fn.
            # images: (B, C, H, W), labels: (B,) integer class indices
            raise NotImplementedError

            # TODO P1-4b: Zero gradients, forward, loss, backward, step.
            # If self.scaler is not None, use AMP:
            #   with torch.autocast(device_type=self.device):
            #       logits = self.model(images)
            #       loss = self.criterion(logits, labels)
            #   self.scaler.scale(loss).backward()
            #   self.scaler.step(self.optimizer)
            #   self.scaler.update()
            # Else:
            #   logits = self.model(images)
            #   loss   = self.criterion(logits, labels)
            #   loss.backward()
            #   self.optimizer.step()
            raise NotImplementedError

            # TODO P1-4c: Compute top-1 accuracy and update meters.
            # top1 = (logits.argmax(dim=1) == labels).float().mean().item()
            # loss_meter.update(loss.item(), images.size(0))
            # acc_meter.update(top1, images.size(0))
            raise NotImplementedError

        return {"loss": loss_meter.avg, "acc": acc_meter.avg}

    # ------------------------------------------------------------------ #
    # Validation epoch                                                     #
    # ------------------------------------------------------------------ #

    def val_epoch(self) -> Dict[str, float]:
        """Run one full pass over the validation set (no gradients).

        Returns:
            {'loss': avg_loss, 'acc': avg_top1_accuracy}
        """
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter  = AverageMeter()

        with torch.no_grad():
            for batch in self.val_loader:
                # TODO P1-4d: Same as train_epoch but no backward.
                # No AMP needed for inference (though it doesn't hurt).
                raise NotImplementedError

        return {"loss": loss_meter.avg, "acc": acc_meter.avg}

    # ------------------------------------------------------------------ #
    # Full training run                                                    #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        epochs: int,
        scheduler=None,
        early_stopping=None,
    ) -> None:
        """Run train_epoch + val_epoch for `epochs` iterations.

        Args:
            epochs:        Number of epochs.
            scheduler:     Optional LR scheduler (from training/scheduler.py).
            early_stopping: Optional EarlyStopping instance (from training/scheduler.py).
        """
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics   = self.val_epoch()

            # TODO P1-4e: Step the scheduler (after validation, not after each batch).
            # For ReduceLROnPlateau: scheduler.step(val_metrics['loss'])
            # For others:            scheduler.step()

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} | "
                f"{elapsed:.1f}s"
            )

            # TODO P1-4f: Call early_stopping(val_metrics['loss'], self.model).
            # If early_stopping.should_stop: break
