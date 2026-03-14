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
- KITTI / nuScenes mini dataset
- OpenCV, Matplotlib for visualization
- Metrics: mAP (detection), mIoU (segmentation)

## Phase Progress
- [ ] Phase 0 — Setup & Data Pipeline
- [ ] Phase 1 — CNN Backbone
- [ ] Phase 2 — 2D Detection
- [ ] Phase 3 — Segmentation
- [ ] Phase 4 — ViT Integration
- [ ] Phase 5 — BEV Transform
- [ ] Phase 6 — Temporal Fusion
- [ ] Phase 7 — Integration & Demo

## Collaboration Notes
- Act as project manager: present tickets phase by phase
- Update CLAUDE.md as phases are completed
- Emphasize DL concepts from CS 444 throughout implementation
