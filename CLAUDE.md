# CLAUDE.md — Autonomous Driving Perception Stack

## Project Goal
Build a full autonomous driving perception pipeline from scratch in PyTorch, covering the entire modern CV stack as a deep learning capstone.

## CS 444 Skills to Showcase (in order)
1. Linear classifiers → Phase 1
2. Multi-layer networks + manual backprop → Phase 1
3. CNNs (ResNet-style) → Phase 1
4. Dense prediction (U-Net) → Phase 3
5. Object detection (SSD/YOLO) → Phase 2
6. Vision Transformers → Phase 4
7. Attention / Transformers → Phases 4, 5, 6
8. Temporal/Recurrent modules → Phase 6

## Phases
- **Phase 0** — Project setup & data pipeline (KITTI or nuScenes mini)
- **Phase 1** — CNN backbone from scratch (linear classifier → MLP → ResNet)
- **Phase 2** — 2D object detection (SSD/YOLO-style, mAP eval)
- **Phase 3** — Semantic segmentation (U-Net decoder, mIoU eval)
- **Phase 4** — Vision Transformer integration (hybrid CNN-ViT)
- **Phase 5** — Bird's Eye View transform (Lift-Splat-Shoot style)
- **Phase 6** — Temporal fusion across frames (cross-attention tracking)
- **Phase 7** — End-to-end integration & demo

## Stack
- PyTorch (custom training loops, mixed precision)
- nuScenes mini dataset (10 scenes, ~404 samples, 6 cameras)
- OpenCV, Matplotlib for visualization
- Metrics: mAP (detection), mIoU (segmentation)

## Phase Progress
- [✅] Phase 0 — Setup & Data Pipeline
- [✅] Phase 1 — CNN Backbone
- [🔄] Phase 2 — 2D Detection (in progress)
- [ ] Phase 3 — Segmentation
- [ ] Phase 4 — ViT Integration
- [ ] Phase 5 — BEV Transform
- [ ] Phase 6 — Temporal Fusion
- [ ] Phase 7 — Integration & Demo

## Completed Work
### Phase 2 (in progress)
- `models/detection/box_utils.py` — core detection utilities ✅
  - `compute_iou` — pairwise (N, M) IoU matrix between two sets of boxes
  - `match_anchors_to_gt` — assign each anchor its best GT match + fg/bg label
  - `encode_boxes` — convert GT boxes to (dx, dy, dw, dh) deltas for training targets
  - `decode_boxes` — inverse: apply predicted deltas to anchors → final [x1,y1,x2,y2] boxes
  - `nms` — non-maximum suppression returning kept box indices
- `models/detection/anchors.py` — `AnchorGenerator` ✅
  - `generate_for_level` — tiles anchors across one feature map level; returns `(H*W*num_ratios, 4)` in `[x1,y1,x2,y2]`
  - `generate_all` — loops over all FPN levels, concatenates, clamps to image bounds; returns `(total_anchors, 4)`
- `models/detection/fpn.py` — `FPN` ✅
  - `__init__` — builds `lateral_convs` and `output_convs` as `nn.ModuleList` (one per FPN level)
  - `forward` — takes `(C3, C4, C5)`, applies lateral convs, top-down upsample+add pathway, output convs; returns `(P5, P4, P3)`
- `models/detection/head.py` — `DetectionHead` ✅
  - `__init__` — shared 4-conv tower (`nn.Sequential`), `cls_head` (`num_anchors * num_classes` out), `reg_head` (`num_anchors * 4` out)
  - `forward` — loops over FPN levels, runs tower then heads, reshapes to `(B, H*W*A, C)` and `(B, H*W*A, 4)`; returns `(cls_logits_list, bbox_deltas_list)`
- `models/detection/losses.py` — `focal_loss`, `smooth_l1_loss`, `DetectionLoss` skeletons (not yet implemented)
- `models/detection/detector.py` — `FPNDetector` skeleton (not yet implemented)
- `models/detection/map.py` — `compute_ap`, `compute_map` skeletons (not yet implemented)
- `models/detection/train_detector.py` — training script skeleton (not yet implemented)

### Phase 1 ✅
- `models/backbone/linear_classifier.py` — `LinearClassifier`: flatten + single `nn.Linear`, forward pass ✅
- `models/backbone/mlp.py` — `MLP`: 3-layer `nn.Sequential` with ReLU + Dropout, forward pass ✅
- `models/backbone/resnet.py` — `ConvBlock`, `ResidualBlock` (with skip connection), `_make_stage`, `ResNetBackbone` (stem + 4 stages + classifier head + multi-scale C3/C4/C5 output) ✅
- `training/trainer.py` — `Trainer`: `train_epoch` (AMP + standard paths), `val_epoch`, `fit` (scheduler + early stopping) ✅
- `training/scheduler.py` — `build_optimizer` (param group weight decay filtering), `build_scheduler` (cosine/plateau), `EarlyStopping` ✅
- `training/train_backbone.py` — end-to-end training script (wires data → model → trainer)

### Phase 0 ✅
- `requirements.txt` — curated project deps (torch, torchvision, albumentations, nuscenes-devkit, etc.)
- Full folder structure: models/{backbone,detection,segmentation,bev,temporal}, data/, training/, evaluation/, utils/, notebooks/, demo/, configs/
- `notebooks/p0_2_data_exploration.ipynb` — schema walkthrough, 6-camera viz, bbox projection, intrinsic/extrinsic params, LiDAR projection, instance trajectory
- `data/transforms.py` — train (flip, crop, jitter, blur, normalize) + val pipelines via albumentations; bbox-aware
- `data/dataset.py` — NuScenesDetectionDataset: 3-class detection (car/ped/cyclist), 3D→2D box projection, scene-level train/val split, collate_fn
- `data/dataloader.py` — get_loaders() factory returning train/val DataLoaders; pin_memory, configurable cameras
- `utils/visualize.py` — draw_boxes (cv2, per-class color), visualize_batch (un-normalize + grid), visualize_sample

## Key Decisions
- Dataset: nuScenes mini only (not full trainval) — sufficient for showcasing DL skills
- 3 detection classes: car (0), pedestrian (1), cyclist (2) — 12 of 23 nuScenes categories mapped
- Train/val split: first 8 scenes train, last 2 val — split by scene to prevent data leakage
- Input resolution: 448×800 (downsampled from native 900×1600)

## Collaboration Notes
- Act as project manager: present tickets phase by phase
- Break down tasks into method-level skeletons; user fills in implementations
- Review user code when asked — point out bugs, never silently fix them
- Update CLAUDE.md as phases are completed
- Emphasize DL concepts from CS 444 throughout implementation
