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
- [✅] Phase 2 — 2D Detection (pretrained backbone trained; mAP 0.129, car AP 0.291)
- [✅] Phase 3 — Segmentation (U-Net trained; mIoU 0.327, drivable IoU 0.587)
- [✅] Phase 4 — ViT Integration (hybrid CNN-ViT trained; mIoU 0.320, ≈ ResNet 0.327)
- [ ] Phase 5 — BEV Transform
- [ ] Phase 6 — Temporal Fusion
- [ ] Phase 7 — Integration & Demo

## Completed Work
### Phase 2 (in progress)
- `models/detection/box_utils.py` — core detection utilities (vectorized) ✅
  - `compute_iou` — pairwise (N, M) IoU via broadcasting; device-agnostic
  - `match_anchors_to_gt` — vectorized `ious.max(dim=1)` matching; returns `(LongTensor, LongTensor)` on-device
  - `encode_boxes` — vectorized GT → delta encoding with `clamp_min` guard for zero-area anchors
  - `decode_boxes` — vectorized inverse: apply predicted deltas to anchors
  - `nms` — delegates to `torchvision.ops.nms` for fused kernel (CUDA/CPU/MPS)
- `models/detection/anchors.py` — `AnchorGenerator` ✅
  - `generate_for_level` — vectorized via `meshgrid` + broadcast; order: `row → col → ratio` to match head reshape
  - `generate_all` — generates + caches anchors keyed on `(sizes, image_size, device)` for O(1) repeat calls
- `models/detection/fpn.py` — `FPN` ✅
  - `__init__` — builds `lateral_convs` and `output_convs` as `nn.ModuleList` (one per FPN level)
  - `forward` — takes `(C3, C4, C5)`, applies lateral convs, top-down upsample+add pathway, output convs; returns `(P5, P4, P3)`
- `models/detection/head.py` — `DetectionHead` ✅
  - `__init__` — shared 4-conv tower (`nn.Sequential`), `cls_head` (`num_anchors * num_classes` out), `reg_head` (`num_anchors * 4` out)
  - `forward` — loops over FPN levels, runs tower then heads, reshapes to `(B, H*W*A, C)` and `(B, H*W*A, 4)`; returns `(cls_logits_list, bbox_deltas_list)`
- `models/detection/losses.py` — `focal_loss`, `smooth_l1_loss`, `DetectionLoss` ✅
  - `focal_loss` — thin wrapper around `torchvision.ops.sigmoid_focal_loss` with `reduction="none"`; returns per-element loss so the caller controls normalization
  - `smooth_l1_loss` — manual Huber: `0.5 * x^2 / beta` for `|x| < beta`, `|x| - 0.5 * beta` otherwise; returns per-element tensor
  - `DetectionLoss.__init__` — stores `num_classes`, `cls_weight`, `reg_weight`
  - `DetectionLoss.forward` — concatenates all FPN levels into `(B, N, C)` / `(B, N, 4)`; per image: calls `match_anchors_to_gt` → builds sigmoid one-hot `cls_target` → `focal_loss` on all anchors; `encode_boxes` + `smooth_l1_loss` on positives only; normalizes totals by `max(num_pos, 1)` (RetinaNet convention); returns `(total_loss, log_dict)` including `num_pos`
- `models/detection/detector.py` — `FPNDetector` ✅
  - `__init__` — stores `backbone`, `fpn`, `head`, `anchor_generator`, `num_classes`, and NMS/top-k hyperparams (`score_threshold`, `nms_threshold`, `max_detections`)
  - `forward` — runs backbone → FPN → head, generates anchors via cached `generate_all`; training branch (or `return_raw=True`) returns `(cls_logits, bbox_deltas, anchors)`, eval branch calls `postprocess`
  - `postprocess` — concatenates FPN levels; per image: `sigmoid` scores → `decode_boxes` → clamp to image bounds → per-class score threshold → per-class `nms` (guarded against empty input) → global top-k by score; returns `(boxes_list, scores_list, labels_list)`
- `models/detection/map.py` — `compute_ap`, `compute_map` ✅
  - `compute_ap` — 11-point interpolation: samples max precision at 11 recall thresholds `[0.0..1.0]`, averages; handles empty recall buckets
  - `compute_map` — per class: filters preds/GT by label, matches via IoU (greedy, each GT matched once), accumulates TP/FP across dataset, sorts by score descending, builds cumulative PR curve, calls `compute_ap`; returns `(mAP, [AP per class])`
- `models/detection/train_detector.py` — full training script ✅
  - `build_detector` — instantiates `ResNetBackbone` (with optional ImageNet pretrained weights) → `FPN` → `DetectionHead` → `AnchorGenerator` → `FPNDetector` from cfg dict
  - `train_one_epoch` — AMP loop (gated on CUDA): unpack batch via `_unpack_batch`, forward → `DetectionLoss` → `scaler` backward/step; tracks `pos_per_img`; returns avg loss dict
  - `val_one_epoch` — `model.eval()` + `return_raw=True` for loss without BN drift; also runs `postprocess` on same logits to collect predictions for `compute_map`; returns loss + mAP + per-class AP
  - `main` — loads yaml cfg, builds model/optimizer/scheduler/loss/scaler, creates `checkpoints/` dir, runs epoch loop with per-epoch mAP eval and early stopping

### Phase 2 — Bugs Found and Fixed
A series of critical bugs were discovered and resolved during the first training attempts:

1. **Batch unpacking mismatch** — `train_one_epoch` expected `(images, gt_boxes, gt_labels)` but `collate_fn` returns `(images, targets_dict_list)`. Fixed with `_unpack_batch` helper.
2. **FPN channel mismatch** — `FPN(in_channels=[512, 1024, 2048])` hardcoded ResNet-50 channels; our ResNet-18 backbone outputs `[128, 256, 512]`. Fixed in `build_detector`.
3. **Backbone constructor misuse** — `ResNetBackbone(cfg["num_classes"])` passed `3` as `in_channels`, accidentally setting `num_classes=None` (classification head) to be omitted only by coincidence. Fixed to `ResNetBackbone()`.
4. **CUDA-hardcoded autocast** — `torch.autocast(device_type="cuda")` failed silently on MPS. Fixed with device-type gating.
5. **Missing checkpoints directory** — `torch.save` would crash on first checkpoint. Fixed with `Path.mkdir(parents=True, exist_ok=True)`.
6. **BN stat drift during validation** — `val_one_epoch` called `model.train()` to get raw logits, polluting BatchNorm running stats. Fixed by adding `return_raw` flag to `FPNDetector.forward` and using `model.eval()` in val.
7. **GradScaler deprecation** — Migrated from `torch.cuda.amp.GradScaler()` to `torch.amp.GradScaler("cuda", enabled=...)`.
8. **Stride/scale ↔ FPN level misalignment** — Config had `strides: [8, 16, 32]` (fine→coarse) but FPN returns `(P5, P4, P3)` (coarse→fine). P5 anchors (stride 32) were generated with stride 8, covering only the top-left ~196×108 pixels. Most GT boxes had no matching anchors. Fixed by flipping config to `strides: [32, 16, 8]`, `scales: [128, 64, 32]`.
9. **Anchor ↔ prediction index ordering mismatch** — `DetectionHead` reshapes via `permute(0,2,3,1).reshape(B, H*W*A, C)` giving `row→col→ratio` order, but `AnchorGenerator` tiled in `ratio→row→col` order. Regression targets were assigned to wrong spatial locations. Fixed by changing anchor ordering to `row→col→ratio` via `(H, W, A, 4)` layout.
10. **Pure-Python loops in box_utils and anchors** — `compute_iou`, `match_anchors_to_gt`, `encode_boxes`, `decode_boxes`, `nms`, and `generate_for_level` all used Python `for` loops over 22k+ anchors. Each training batch took multiple seconds on CPU. Vectorized all ops (broadcasting, `meshgrid`, `torchvision.ops.nms`); detection math per batch dropped from seconds to ~3–5 ms.

### Phase 2 — Training Observations and Pretrained Backbone Decision
Training on nuScenes mini (~320 train images, 80 val images) revealed that from-scratch backbone training is insufficient for this dataset size:

**Run 1 (from-scratch, broken stride/anchor ordering):**
- Loss plateaued at train ~0.66, val ~0.89 from epoch 6 onward.
- No mAP evaluation was wired yet.
- Root cause: stride/scale misalignment + anchor ordering bug meant anchors didn't match GT boxes or predictions correctly.

**Run 2 (from-scratch, bugs fixed, scales=[128,64,32]):**
- pos/img stabilized at ~43 (healthy anchor matching).
- Car AP reached 0.126 by epoch 12, then early stopping triggered.
- Pedestrian/cyclist AP stuck at 0.000.
- Train/val gap small (~0.22) — underfitting, not overfitting.

**Run 3 (from-scratch, scales=[96,48,16] to try covering smaller pedestrians):**
- pos/img dropped to ~19 (shrunk P5/P4 too much, lost car matches).
- Car AP peaked at 0.109 — worse than run 2.
- Ped AP reached 0.002, cyclist 0.002 — marginal improvement.
- Conclusion: anchor geometry helps at the margin, but the core problem is feature quality.

**Class distribution analysis (train split):**
- car: 2306, ped: 869, cyclist: 178 (13:1 car:cyclist ratio)
- Median box widths: car 59px, ped 19px, cyclist 38px
- Imbalance is moderate — focal loss should handle it. The real bottleneck is that a backbone trained from scratch on 320 images can't learn discriminative features for small/rare classes.

**Decision: ImageNet-pretrained ResNet-18 backbone.**
- Architecture unchanged — same `ResNetBackbone` with `ConvBlock`, `ResidualBlock`, `_make_stage`
- Added `load_pretrained()` method that maps torchvision ResNet-18 state dict keys to our naming convention (120 params mapped)
- Controlled by `pretrained: true` in `configs/detector.yaml`
- Rationale: the backbone needs to have learned low/mid-level visual features (edges, textures, shapes) from a large dataset. 320 nuScenes images is ~370× less than COCO (which RetinaNet was trained on). Pretrained weights give the detector head useful features to work with from epoch 1.

### Phase 2 — Pretrained-Backbone Training Results (Run 4)
Training run with `pretrained: true`, scales `[128, 64, 32]`, strides `[32, 16, 8]`, 40-epoch cap.

**Outcome:** early stopped at epoch 18, best checkpoint at epoch 8 (`checkpoints/detector_best.pt`).

| Metric | From-scratch (Run 2) | Pretrained (Run 4) | Δ |
|---|---|---|---|
| Car AP | 0.126 | **0.291** | +130% |
| Pedestrian AP | 0.000 | 0.001 | flat |
| Cyclist AP | 0.000 | **0.097** | new signal |
| **mAP** | **0.042** | **0.129** | **+207%** |
| Best epoch | 12 | 8 | converges faster |
| Train loss at best | 0.51 | 0.37 | — |
| Val loss at best | 0.84 | 0.69 | — |

**Observations:**
- Pretrained ImageNet features tripled overall mAP and unlocked cyclist detection (was zero before).
- Convergence is dramatically faster — 8 epochs vs 12, and val loss is lower (0.69 vs 0.84).
- After epoch 8, val loss oscillates (0.69–0.99) while train loss keeps dropping — classic mild overfitting on 320 images.
- **Pedestrian AP remains stuck at ~0.** Even with strong features, the detection geometry can't match pedestrian sizes:
  - Median pedestrian width in train split: 19px.
  - Smallest anchor: scale 32 on stride-8 feature map → smallest box ~22px on a feature pixel covering 8 input pixels. Recall is brittle.
  - Real fix is architectural: add a P2 FPN level (stride 4) or shrink scales to `[64, 32, 16]`. Deferred to a future polish pass — not blocking Phase 3.

**Logs:** `logs/detector_pretrained_run.log` (per-epoch loss + mAP).

### Phase 3 — Complete ✅ (U-Net trained, mIoU 0.327)
**Data pipeline (done ✅):**
- `data/seg_labels.py` — offline BEV-polygon → camera mask projection ✅
  - `load_map_for_log` — resolves scene→log→location, builds + module-caches `NuScenesMap`.
  - `get_camera_extrinsics` — composes `camera→ego` (calibrated_sensor) ∘ `ego→world` (ego_pose) via quaternions, inverts to `world→camera`; returns `(R_w2c, t_w2c, K)`.
  - `sample_map_polygons` — `get_records_in_radius` for all CLASS_MAP layers; resolves polygons via `extract_polygon` (handles `drivable_area`'s plural `polygon_tokens`).
  - `project_polygon_to_image` — lifts polygon to z=0, applies `P_cam = R·P + t`, drops `z < z_min` vertices *before* the K-projection divide. Wrapped in `np.errstate(...)` to suppress numpy's spurious Apple-Silicon SIMD-matmul FP warnings (inputs/outputs verified finite).
  - `rasterize_mask` — `cv2.fillPoly` in ascending class-ID order so higher IDs paint over lower.
  - `build_seg_label_for_sample` / `generate_all_masks` — end-to-end; iterates `TRAIN_SCENES | VAL_SCENES`, skips existing PNGs.
  - 5-class scheme: background / drivable / lane / ped_crossing / walkway.
- `data/seg_dataset.py` — `NuScenesSegmentationDataset` ✅ — `_build_index` mirrors detection split, skips pairs with no cached mask; `__getitem__` loads image+mask, applies joint albumentations transform, casts mask to long.
- `data/transforms.py` — `get_seg_train_transforms` / `get_seg_val_transforms` ✅ — same image augs as detection, NEAREST mask resampling so class IDs aren't blended, no bbox_params.
- `data/dataloader.py` — `get_seg_loaders` (concrete) ✅
- `configs/segmenter.yaml` — config ✅
- **Masks generated:** 404 CAM_FRONT masks cached → `data/raw/v1.0-mini/seg_masks/`. Verified: train 324 / val 80 samples, batches `images (B,3,448,800) float32` + `masks (B,448,800) int64`. Sample mask class mix ~54% bg / 22% drivable / 23% lane / <1% walkway / ~0% ped_crossing (ped_crossing is rare).
- Base map PNGs (`maps/{hash}.png`, semantic_prior) restored from `data/v1.0-mini.tgz` — required by the `NuScenes()` constructor.

**Model + training (implemented ✅, trained ✅):**
- `models/segmentation/unet.py` — `UpBlock` + `UNet` ✅ — `UpBlock` upsamples (bilinear) → concat skip → 2 ConvBlocks (or pure 2× upsample when `skip=None`); `UNet` decodes C5→C4→C3→stride-4 then 1×1 classifier + final 4× upsample. Verified `(B,3,H,W)→(B,5,H,W)`, backward OK.
- `models/segmentation/losses.py` — `dice_loss` + `SegmentationLoss` ✅ — soft multi-class Dice (softmax probs, one-hot GT, `sum(dim=(0,2,3))` per class then mean); `SegmentationLoss` returns `(total, log_dict)` with `loss`/`ce_loss`/`dice_loss`.
- `models/segmentation/train_seg.py` — `train_one_epoch` / `val_one_epoch` / `main` ✅ — AMP loop (CUDA-gated), val uses streaming `ConfusionMatrixMeter`; smoke-tested end-to-end on real seg loaders.
- `evaluation/seg_metrics.py` — `ConfusionMatrixMeter` + `compute_miou` ✅ — streaming confusion matrix via `np.bincount`; per-class IoU with NaN for absent classes; `miou` = `nanmean`.
- Tests: `tests/test_unet.py`, `test_seg_losses.py`, `test_seg_metrics.py`, `test_seg_transforms.py` — full suite **300 passing**.

### Phase 3 — Training Results
Trained `python -m models.segmentation.train_seg configs/segmenter.yaml` — pretrained ResNet-18 backbone, 40-epoch cap, cosine LR, early-stop patience 10 on mIoU.

**Outcome:** early stopped at epoch 20, best checkpoint at epoch 10 (`checkpoints/segmenter_best.pt`). **mIoU 0.327.**

| Class | IoU @ best (epoch 10) |
|---|---|
| background | 0.861 |
| drivable | 0.587 |
| lane | 0.177 |
| ped_crossing | 0.000 |
| walkway | 0.011 |
| **mIoU** | **0.327** |

**Observations:**
- background + drivable dominate — large, frequent classes the model learns well.
- `lane` modest (~0.18): thin structures, softened by the decoder's single 4× final upsample from stride-4.
- `ped_crossing` (~0% of pixels) and `walkway` (<1%) stuck near 0 — severe class imbalance; even with the Dice term, classes that barely appear can't be learned from 324 images.
- Overfitting from ~epoch 5: train loss fell steadily 1.39→0.52 while val loss rose 1.23→1.46 and mIoU oscillated 0.24–0.33 — same small-dataset pattern as Phase 2 detection.

**Logs:** `logs/segmenter_run.log` (per-epoch loss + mIoU).

Known limitation: `project_polygon_to_image` filters `z < z_min` per-vertex rather than clipping polygon edges at the camera plane — polygons straddling the camera plane are slightly distorted. Fine for road/drivable at `radius=50m`; revisit if bottom-edge artifacts appear.

### Phase 4 — Complete ✅ (hybrid CNN-ViT trained, mIoU 0.320)
Built a Vision Transformer from scratch and a hybrid CNN-ViT backbone, then compared segmentation mIoU against the Phase 3 ResNet U-Net.

**Design:** `HybridCNNViT` returns `(C3@128/s8, C4@256/s16, C5@512/s32)` — identical to `ResNetBackbone` — so it is a drop-in backbone for the existing `UNet`. The comparison reuses the Phase 3 decoder, loss, and `train_seg.py` unchanged; only `cfg["backbone"]` differs.

- `models/backbone/vit.py` — `PatchEmbedding`, `PositionalEncoding`, `MultiHeadSelfAttention`, `TransformerEncoderBlock`, `ViT` ✅ — from-scratch ViT; classifier-head or feature-map output.
- `models/backbone/hybrid.py` — `HybridCNNViT` ✅ — CNN front reuses `ResNetBackbone` (stem + stages 1-3, ImageNet-pretrained); ViT encoder over patch-embedded C4 produces C5 (trained from scratch).
- `configs/segmenter_hybrid.yaml` — config ✅ (`backbone: hybrid`, ViT hyperparams; `embed_dim` must be 512).
- `models/segmentation/train_seg.py` — `build_segmenter` branches on `cfg["backbone"]` ("resnet" | "hybrid") ✅.
- Tests: `tests/test_vit.py`, `test_hybrid.py` — full suite **325 passing**.

### Phase 4 — Training Results
Trained `python -m models.segmentation.train_seg configs/segmenter_hybrid.yaml` — identical setup to Phase 3 (U-Net decoder, Dice+CE loss, 40-epoch cap, cosine LR, patience 10); only the backbone differs. Early stopped at epoch 20, best at epoch 10.

| Class | ResNet U-Net (Phase 3) | Hybrid CNN-ViT (Phase 4) |
|---|---|---|
| background | 0.861 | 0.862 |
| drivable | 0.587 | 0.567 |
| lane | 0.177 | 0.168 |
| ped_crossing | 0.000 | 0.005 |
| walkway | 0.011 | 0.000 |
| **mIoU** | **0.327** | **0.320** |

**Conclusion:** the hybrid landed essentially tied with the pure ResNet (0.320 vs 0.327 — within run-to-run noise), not ahead of it. This is the textbook ViT-vs-CNN result at small data scale: the ViT stage trains from scratch on only 324 images, with none of the CNN's locality / translation-equivariance inductive bias, so attention's global reasoning doesn't pay off — the fully-pretrained ResNet stage4 is simply a better C5 extractor in this regime. ViT advantages appear at large data scale (original ViT needed JFT-300M). The CNN front (pretrained C3/C4) carries most of the load in both models. **Logs:** `logs/segmenter_hybrid_run.log`.

Note: checkpoint paths are now config-driven via `cfg["ckpt_path"]` — `segmenter.yaml` → `checkpoints/segmenter_resnet_best.pt`, `segmenter_hybrid.yaml` → `checkpoints/segmenter_hybrid_best.pt`, so the two runs no longer collide. (The old `checkpoints/segmenter_best.pt` on disk is the last hybrid run under the pre-fix name — safe to delete; `.pt` files are gitignored.)

Recommended order: bottom-up — `PatchEmbedding` and `MultiHeadSelfAttention` first (independently testable), then `TransformerEncoderBlock`, then `ViT`, then `HybridCNNViT`.

### Phase 1 ✅
- `models/backbone/linear_classifier.py` — `LinearClassifier`: flatten + single `nn.Linear`, forward pass ✅
- `models/backbone/mlp.py` — `MLP`: 3-layer `nn.Sequential` with ReLU + Dropout, forward pass ✅
- `models/backbone/resnet.py` — `ConvBlock`, `ResidualBlock` (with skip connection), `_make_stage`, `ResNetBackbone` (stem + 4 stages + classifier head + multi-scale C3/C4/C5 output), `load_pretrained` (ImageNet ResNet-18 weight mapping) ✅
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
- ImageNet-pretrained backbone: necessary for nuScenes mini — 320 train images is too few to learn visual features from scratch (car AP saturated at 0.12, ped/cyclist AP stuck at 0). Architecture is unchanged; only initial weight values differ.
- Anchor config aligned to FPN output order: `strides: [32, 16, 8]`, `scales: [128, 64, 32]` matching `(P5, P4, P3)` coarse-to-fine
- Anchor tiling order `row→col→ratio` to match `DetectionHead` reshape `permute(0,2,3,1)`

## Collaboration Notes
- Act as project manager: present tickets phase by phase
- Break down tasks into method-level skeletons; user fills in implementations
- Review user code when asked — point out bugs, never silently fix them
- Update CLAUDE.md as phases are completed
- Emphasize DL concepts from CS 444 throughout implementation
