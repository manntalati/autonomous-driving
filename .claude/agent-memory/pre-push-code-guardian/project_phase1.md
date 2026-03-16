---
name: project_phase1
description: Phase 1 CNN backbone status — which sub-tasks are complete vs stubbed with NotImplementedError
type: project
---

Phase 1 CNN Backbone status as of 2026-03-16:

- P1-1: LinearClassifier — COMPLETE (models/backbone/linear_classifier.py)
- P1-2: MLP — COMPLETE (models/backbone/mlp.py)
- P1-3: ResNetBackbone (ConvBlock, ResidualBlock, _make_stage, multi-scale C3/C4/C5) — COMPLETE (models/backbone/resnet.py)
- P1-4: Trainer (training/trainer.py) — COMPLETE. train_epoch, val_epoch, fit all implemented.
- P1-5: build_optimizer, build_scheduler, EarlyStopping (training/scheduler.py) — COMPLETE.

## Known Bugs in Phase 1 (reported, not fixed)

- CRITICAL: train_backbone.py calls get_loaders(data_root=..., batch_size=...) but dataloader.get_loaders signature is get_loaders(nusc: NuScenes, cameras=None). Will crash on startup.
- CRITICAL: Trainer.train_epoch concatenates all annotation labels from all images (torch.cat([t['labels'] for t in targets])) and uses them as per-image classification labels. If any image has != 1 annotation, label count != batch size and CrossEntropyLoss crashes.
- WARNING: EarlyStopping.load_best uses torch.load without weights_only=True — FutureWarning in PyTorch 2.4, will default to True in future versions.
- WARNING: build_scheduler return type annotated as _LRScheduler but ReduceLROnPlateau does not inherit from _LRScheduler.
- INFO: fit() does not call early_stopping.load_best() at end of training — best weights not restored in memory automatically.

## Tests

- 101 tests across tests/test_backbone.py and tests/test_scheduler_trainer.py — all pass as of 2026-03-16.
- tests/test_scheduler_trainer.py is new (created 2026-03-16) covering build_optimizer, build_scheduler, EarlyStopping, AverageMeter, Trainer.
- Trainer tests use a synthetic single-label-per-image DataLoader to avoid the label-count/batch-size mismatch bug.

**Why:** P1-4 and P1-5 were filled in by the student.

**How to apply:** All critical bugs above are reported but unresolved — do not silently fix them, highlight them if the student asks about training crashes.
