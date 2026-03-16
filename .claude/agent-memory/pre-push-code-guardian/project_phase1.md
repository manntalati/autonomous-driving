---
name: project_phase1
description: Phase 1 CNN backbone status — which sub-tasks are complete vs stubbed with NotImplementedError
type: project
---

Phase 1 CNN Backbone status as of 2026-03-15:

- P1-1: LinearClassifier — COMPLETE (models/backbone/linear_classifier.py)
- P1-2: MLP — COMPLETE (models/backbone/mlp.py)
- P1-3: ResNetBackbone (ConvBlock, ResidualBlock, _make_stage, multi-scale C3/C4/C5) — COMPLETE (models/backbone/resnet.py)
- P1-4: Trainer (training/trainer.py) — SKELETON ONLY. train_epoch, val_epoch, fit all have raise NotImplementedError in key spots.
- P1-5: build_optimizer, build_scheduler, EarlyStopping (training/scheduler.py) — SKELETON ONLY. All three implementations raise NotImplementedError.

**Why:** P1-4 and P1-5 are intentional stubs — the student fills these in as the learning exercise.

**How to apply:** Do not flag the NotImplementedError stubs as bugs. They are intentional pedagogical placeholders. The train_backbone.py script will crash if run end-to-end until P1-4/P1-5 are filled in — this is expected.
