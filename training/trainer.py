from __future__ import annotations
from typing import Dict, Optional
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class AverageMeter:
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
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str = "cpu", scaler: Optional[torch.cuda.amp.GradScaler] = None) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scaler = scaler

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        for images, targets in self.train_loader:
            images = images.to(self.device)
            labels = torch.cat([t['labels'] for t in targets]).to(self.device)
            self.optimizer.zero_grad()
            if self.scaler:
                with torch.autocast(device_type=self.device):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
            loss_meter.update(loss.item(), images.size(0))
            accuracy_meter.update(accuracy, images.size(0))
            
        return {'loss': loss_meter.avg, 'acc': accuracy_meter.avg}

    def val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                labels = torch.cat([t['labels'] for t in targets]).to(self.device)
                if self.scaler:
                    with torch.autocast(device_type=self.device):
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                
                accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
                loss_meter.update(loss.item(), images.size(0))
                accuracy_meter.update(accuracy, images.size(0))

        return {'loss': loss_meter.avg, 'acc': accuracy_meter.avg}

    def fit(self, epochs: int, scheduler=None, early_stopping=None) -> None:
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.val_epoch()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} | "
                f"{elapsed:.1f}s"
            )

            if early_stopping is not None:
                early_stopping(val_metrics['loss'], self.model)
                if early_stopping.should_stop:
                    break
