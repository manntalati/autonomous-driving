---
name: project_conventions
description: Codebase conventions discovered during Phase 1 review — tensor shapes, data formats, known coupling points
type: project
---

## Tensor / Data Conventions

- Bounding boxes: [N, 4] in pascal_voc (xyxy) format, float32
- Labels: [N,] integer class indices (long), 3 classes: car=0, pedestrian=1, cyclist=2
- Images: [B, C, H, W] float32, normalized with ImageNet mean/std (MEAN=(0.485,0.456,0.406), STD=(0.229,0.224,0.225))
- Input resolution: 448x800 (H x W)
- ResNet multi-scale outputs: C3=[B,128,H,W], C4=[B,256,H/2,W/2], C5=[B,512,H/4,W/4]

## DataLoader / Dataset

- collate_fn returns (images_tensor, list_of_target_dicts) — NOT a batched tensor for targets
- train_backbone.py calls get_loaders() from data/dataloader.py — BUT dataloader.py requires a NuScenes object as first argument while train_backbone.py calls get_loaders(data_root=..., batch_size=...) with keyword args only. This is a KNOWN API MISMATCH bug.
- pin_memory=True in both train and val DataLoaders

## Known API Mismatch (Critical Bug)

data/dataloader.py::get_loaders() signature: get_loaders(nusc: NuScenes, cameras=None)
training/train_backbone.py calls: get_loaders(data_root=args.data_root, batch_size=args.batch_size)

These are incompatible. The script will crash on startup. Student needs to reconcile these APIs.

## CI

- .github/workflows/ci.yml installs CPU-only torch separately, then greps requirements.txt to exclude torch lines
- The grep filter `grep -v "^torch"` only excludes lines starting with "torch" — torchvision line also starts with "torch" so it is excluded too (correct behavior). But nuscenes-devkit and other deps with optional GPU extensions may cause issues in CI.
- No `pytest.ini` or `conftest.py` — tests rely on running from project root for relative imports to work.

## .gitignore gaps

- Only ignores specific __pycache__ dirs explicitly — does not use a blanket `**/__pycache__/` pattern
- Missing: *.pyc, *.pyo, .DS_Store, checkpoints/*.pt, notebooks/.ipynb_checkpoints
