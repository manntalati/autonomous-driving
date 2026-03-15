"""
P1-4/5 — Backbone Training Script
=====================================
Run: python -m training.train_backbone

Wires together:
  - NuScenesDetectionDataset (from data/)
  - ResNetBackbone (models/backbone/resnet.py)
  - Trainer (training/trainer.py)
  - build_optimizer + build_scheduler + EarlyStopping (training/scheduler.py)
"""

from __future__ import annotations
import argparse
import torch
from data.dataloader import get_loaders
from models.backbone.resnet import ResNetBackbone
from training.trainer import Trainer
from training.scheduler import build_optimizer, build_scheduler, EarlyStopping


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ResNet backbone (Phase 1)")
    p.add_argument("--data-root",  default="data/raw/v1.0-mini")
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--wd",         type=float, default=1e-4)
    p.add_argument("--optimizer",  default="adamw", choices=["sgd", "adamw"])
    p.add_argument("--scheduler",  default="cosine", choices=["cosine", "plateau"])
    p.add_argument("--patience",   type=int, default=7)
    p.add_argument("--amp",        action="store_true", help="Enable mixed precision")
    p.add_argument("--ckpt",       default="best_backbone.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────── #
    train_loader, val_loader = get_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
    )

    # ── Model ─────────────────────────────────────────────────────────── #
    model = ResNetBackbone(num_classes=3)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer + scheduler + early stopping ────────────────────────── #
    optimizer = build_optimizer(
        model,
        optimizer_type=args.optimizer,
        lr=args.lr,
        weight_decay=args.wd,
    )
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        epochs=args.epochs,
    )
    early_stopping = EarlyStopping(patience=args.patience, ckpt_path=args.ckpt)

    # ── Mixed precision scaler (optional) ─────────────────────────────── #
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device == "cuda") else None

    # ── Trainer ───────────────────────────────────────────────────────── #
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        scaler=scaler,
    )

    trainer.fit(
        epochs=args.epochs,
        scheduler=scheduler,
        early_stopping=early_stopping,
    )

    print(f"\nBest checkpoint saved to {args.ckpt}")


if __name__ == "__main__":
    main()
