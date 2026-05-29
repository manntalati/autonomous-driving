

https://github.com/user-attachments/assets/9be2ee6f-5d44-409b-a746-28f3d2ffa66b

# Autonomous Driving Perception Stack

A full autonomous driving perception pipeline built **from scratch** in PyTorch. This project implements the core vision system that self-driving cars use to understand the world: detecting vehicles, pedestrians, and cyclists, segmenting roads and lanes, projecting everything into a bird's eye view, and tracking objects across time.

Built as a deep learning capstone covering the entire modern CV stack: linear classifiers, CNNs, object detection, dense prediction, Vision Transformers, BEV transforms, and temporal attention.

---
## Demo

[Demo](https://github.com/user-attachments/assets/af175355-e19e-4206-8d8a-1035cfab3e47)

Camera with detections + segmentation on the left, BEV panel on the right!

---

## Architecture Overview

```
                         Input: Dashcam Video Frames
                                    |
                    +---------------+----------------+
                    |                                |
            [CNN Backbone]                  [ViT Encoder]
            ResNet-style                  Patch Embed + MSA
            (local features)             (global reasoning)
                    |                                |
                    +---------------+----------------+
                                    |
                          [Hybrid Feature Map]
                                    |
                +-------------------+-------------------+
                |                   |                   |
        [Detection Head]   [Segmentation Head]   [BEV Transform]
         SSD/YOLO-style      U-Net Decoder       Lift-Splat-Shoot
         Cars, Peds, Cyc    Road, Lane, Sidewalk  Top-down projection
                |                   |                   |
                +-------------------+-------------------+
                                    |
                        [Temporal Fusion Module]
                        Cross-attention across frames
                        Object tracking & consistency
                                    |
                            [Final Output]
                  Annotated video + BEV map + tracks
```

In depth architecture: [Architecture Plan](architecture_plan.md)

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | PyTorch |
| Dataset | nuScenes mini (10 scenes, ~404 samples, 6 cameras) |
| Computer Vision | OpenCV |
| Visualization | Matplotlib, OpenCV |
| Training | Custom training loops, LR scheduling, mixed precision |
| Evaluation | mAP (detection), mIoU (segmentation), FPS benchmarks |

---

## Project Roadmap

### Phase 0: Project Setup & Data Pipeline

> **Goal:** Establish the repo, download data, and build a robust data loading pipeline.

| Ticket | Task | Status |
|---|---|---|
| `P0-1` | Set up repo structure, virtual env, and dependencies (PyTorch, OpenCV, etc.) | [✅] |
| `P0-2` | Download & explore dataset (KITTI or nuScenes mini split) — understand annotations, camera params | [✅] |
| `P0-3` | Build a custom `Dataset` class with data loading, augmentations, and train/val splits | [✅] |
| `P0-4` | Build a `DataLoader` pipeline with visualization utilities (draw bboxes, overlay masks) | [✅] |

---

### Phase 1: Image Classification Backbone (from scratch)

> **Goal:** Build the CNN feature extractor that serves as the foundation for everything else.

| Ticket | Task | Status |
|---|---|---|
| `P1-1` | Implement a simple linear classifier on image patches as a baseline | [✅] |
| `P1-2` | Build a multi-layer network (MLP) baseline | [✅] |
| `P1-3` | Build a CNN backbone from scratch (ResNet-style) — conv blocks, batch norm, skip connections | [✅] |
| `P1-4` | Train the backbone on a traffic sign/object classification subtask | [✅] |
| `P1-5` | Implement training best practices: LR scheduling, weight decay, augmentation, early stopping | [✅] |

---

### Phase 2: 2D Object Detection

> **Goal:** Detect cars, pedestrians, and cyclists with bounding boxes in driving scenes.

| Ticket | Task | Status |
|---|---|---|
| `P2-1` | Implement anchor box generation and IoU computation from scratch | [✅] |
| `P2-2` | Build a single-stage detector head (SSD/YOLO-style) on top of the CNN backbone | [✅] |
| `P2-3` | Implement the detection loss (classification + bbox regression + NMS) | [✅] |
| `P2-4` | Train on nuScenes 2D detection (cars, pedestrians, cyclists) | [✅] |
| `P2-5` | Evaluate with mAP, visualize predictions vs ground truth | [✅] |

---

### Phase 3: Dense Prediction / Semantic Segmentation

> **Goal:** Classify every pixel — road, lane, sidewalk, vehicle, sky — for full scene understanding.

| Ticket | Task | Status |
|---|---|---|
| `P3-1` | Build a decoder (upsampling path) — bilinear upsample + skip connections (U-Net style) | [✅] |
| `P3-2` | Implement pixel-wise cross-entropy loss and Dice loss | [✅] |
| `P3-3` | Train for drivable/lane/ped-crossing/walkway segmentation | [✅] |
| `P3-4` | Evaluate with mIoU (per-class IoU) | [✅] |

---

### Phase 4: Vision Transformer Integration

> **Goal:** Add global reasoning capability via a Vision Transformer, then fuse with CNN features.

| Ticket | Task | Status |
|---|---|---|
| `P4-1` | Implement patch embedding + positional encoding from scratch | [✅] |
| `P4-2` | Build multi-head self-attention and a full ViT encoder block | [✅] |
| `P4-3` | Augment CNN backbone with ViT — compare segmentation mIoU | [✅] |
| `P4-4` | Implement a hybrid CNN-ViT architecture (CNN early features -> ViT for global reasoning) | [✅] |

---

### Phase 5: Bird's Eye View (BEV) Transform

> **Goal:** Project front-view perception into a top-down map — the representation self-driving cars actually plan on.

| Ticket | Task | Status |
|---|---|---|
| `P5-1` | Implement camera intrinsic/extrinsic projection math | [✅] |
| `P5-2` | Build a learned BEV transform module (Lift-Splat-Shoot style) | [✅] |
| `P5-3` | Project front-view features into a top-down BEV grid + BEV detection head | [✅] |
| `P5-4` | Visualize BEV outputs (detected objects from above) | [✅] |

---

### Phase 6: Temporal Fusion Across Frames

> **Goal:** Leverage video sequences for smoother, more consistent predictions over time.

| Ticket | Task | Status |
|---|---|---|
| `P6-1` | Build a sequence data loader that serves consecutive frames | [✅] |
| `P6-2` | Implement temporal attention (cross-attention between current and past frame features) | [✅] |
| `P6-3` | Temporal-fusion module enriching the detector with past-frame features | [✅] |
| `P6-4` | Temporal-consistency (flicker) metric + frame-by-frame evaluation | [✅] |

---

### Phase 7: Integration & Demo

> **Goal:** Unify everything into one end-to-end system and produce a polished demo.

| Ticket | Task | Status |
|---|---|---|
| `P7-1` | Unify all modules into one end-to-end pipeline (image -> detections + segmentation + BEV) | [✅] |
| `P7-2` | Run inference over a scene's frames, render annotated detection + segmentation outputs | [✅] |
| `P7-3` | Interactive Streamlit demo — scene picker, frame scrubber, all outputs | [✅] |
| `P7-4` | Write up results: architecture diagram, benchmarks, ablation studies | [✅] |

---

### Phase 2 Training Log

Training the FPN-based RetinaNet detector on nuScenes mini (320 train images, 80 val images) required debugging several critical issues before the model could learn:

**Bugs fixed before first successful run:**
- Batch unpacking mismatch (`collate_fn` returns dict targets, not separate tensors)
- FPN channel mismatch (ResNet-50 channels hardcoded; our ResNet-18 uses 128/256/512)
- Stride/scale ↔ FPN level misalignment (anchors assigned to wrong feature pyramid levels)
- Anchor ↔ prediction index ordering (anchors tiled in different order than head's reshape)
- Pure-Python loops in box utilities vectorized for ~100–1000× speedup

**From-scratch backbone results (scales=[128,64,32]):**
| Metric | Best Value | Epoch |
|---|---|---|
| Car AP | 0.126 | 12 |
| Pedestrian AP | 0.000 | — |
| Cyclist AP | 0.000 | — |
| mAP | 0.042 | 12 |

**Analysis:** Training loss dropped steadily (0.68 → 0.51) but val loss plateaued at ~0.84 with a small train-val gap — classic underfitting. The backbone couldn't learn discriminative visual features from only 320 images. Car detection worked marginally because cars are large and frequent. Pedestrians (median 19px wide, 869 instances) and cyclists (178 instances) produced zero AP despite anchor scales that geometrically covered them.

**Solution:** ImageNet-pretrained ResNet-18 backbone via `load_pretrained()`. The Phase 1 architecture (`ConvBlock`, `ResidualBlock`, `_make_stage`, `ResNetBackbone`) is unchanged — only the initial weight values differ. This gives the detector head pre-learned low/mid-level features (edges, textures, object parts) from 1.2M ImageNet images, allowing the detection head to focus on *where* objects are rather than *what visual features matter*.

**Pretrained backbone results (same anchor config, `pretrained: true`):**
| Metric | From-scratch | Pretrained | Δ |
|---|---|---|---|
| Car AP | 0.126 | **0.291** | +130% |
| Pedestrian AP | 0.000 | 0.001 | flat |
| Cyclist AP | 0.000 | **0.097** | new signal |
| mAP | 0.042 | **0.129** | +207% |
| Best epoch | 12 | 8 | faster |

Training early-stopped at epoch 18 (best at 8). Pretrained features tripled overall mAP and unlocked cyclist detection. Pedestrians remain stuck at ~0 AP — the detection geometry (smallest scale 32 on stride-8 feature map) can't recall objects with median width 19px. A future polish pass will add a P2 FPN level (stride 4) or shrink scales to `[64, 32, 16]`.

---

### Phase 3 Training Log

Segmentation labels are generated offline by projecting nuScenes map-expansion polygons (drivable area, lane, ped_crossing, walkway) through each camera's intrinsics + extrinsics into image space — 404 cached `CAM_FRONT` masks, 5 classes. A U-Net decoder sits on the pretrained ResNet-18 backbone: bilinear-upsample blocks merge encoder skips (C5→C4→C3), then a 1×1 classifier and final 4× upsample to input resolution. Loss is cross-entropy + soft multi-class Dice.

**U-Net segmentation results (`pretrained: true`, 40-epoch cap):**
| Class | IoU (best, epoch 10) |
|---|---|
| background | 0.861 |
| drivable | 0.587 |
| lane | 0.177 |
| ped_crossing | 0.000 |
| walkway | 0.011 |
| **mIoU** | **0.327** |

**Analysis:** Training early-stopped at epoch 20 (best at 10). Background and drivable surface — large, frequent classes — segment well. `lane` is modest (~0.18): lane polygons are thin and the decoder's single 4× final upsample softens fine boundaries. `ped_crossing` (~0% of pixels) and `walkway` (<1%) stay near zero — severe class imbalance means the rare classes never get enough signal, even with the Dice term. Train loss fell steadily (1.39 → 0.52) while validation loss rose (1.23 → 1.46) from ~epoch 5 — the same overfitting pattern seen in Phase 2, driven by the 324-image training set. A future polish pass could add a P2-level skip (stride 4) for sharper lanes or oversample rare-class crops.

---

### Phase 4 Training Log

A Vision Transformer is built from scratch — patch embedding, learned positional encoding, multi-head self-attention, pre-norm encoder blocks — and combined into a hybrid backbone: a ResNet CNN front extracts local features C3/C4 (strides 8/16), then a ViT encoder applies global self-attention over the patch-embedded C4 to produce C5 (stride 32). Because the hybrid returns `(C3, C4, C5)` with the same channels/strides as `ResNetBackbone`, it drops straight into the Phase 3 U-Net — the mIoU comparison holds the decoder, loss, and training loop fixed and varies only the backbone.

**Hybrid CNN-ViT vs ResNet backbone (U-Net segmentation, same setup):**
| Class | ResNet U-Net | Hybrid CNN-ViT |
|---|---|---|
| background | 0.861 | 0.862 |
| drivable | 0.587 | 0.567 |
| lane | 0.177 | 0.168 |
| ped_crossing | 0.000 | 0.005 |
| walkway | 0.011 | 0.000 |
| **mIoU** | **0.327** | **0.320** |

**Analysis:** The hybrid finished essentially tied with the pure ResNet (0.320 vs 0.327 — within run-to-run noise), not ahead of it. This is the expected ViT-vs-CNN outcome at small data scale. The ViT stage trains from scratch on only 324 images and has none of the convolutional inductive biases (locality, translation equivariance) that let CNNs generalize from little data; meanwhile the ResNet's deep stage is ImageNet-pretrained. Self-attention's global reasoning is a real advantage — but it shows up at large data scale (the original ViT needed JFT-300M to beat CNNs). Here the pretrained CNN front does most of the work in both models. The honest takeaway: a hybrid is only worth its extra cost when there is enough data, or a pretrained ViT, to train the attention stack properly.

---

### Phase 5 Training Log

The bird's-eye-view stage implements Lift-Splat-Shoot from scratch. Every CAM_FRONT feature-map pixel predicts a categorical distribution over discrete depths; its context feature is "lifted" into a 3D frustum (one weighted copy per depth bin), each frustum point is placed in the ego frame using the camera intrinsics + extrinsics, and all points are "splatted" — sum-pooled — into a 64×64 top-down BEV grid. A centre-based detection head then predicts a per-class object-centre heatmap and box regression (offset, size, heading) in BEV space, supervised against nuScenes 3D boxes projected to the ego frame.

**BEV detector results (CAM_FRONT, 40-epoch cap):**
| | epoch 1 | best (epoch 12) | epoch 22 |
|---|---|---|---|
| train loss | 27.77 | 4.20 | 3.44 |
| val loss | 7.88 | **7.74** | 8.52 |

**Analysis:** Training early-stopped at epoch 22 (best at 12). The pipeline trains end-to-end — train loss fell ~8× as the CenterNet-style heatmap focal loss collapsed and the depth/splat learned to place features in the right BEV cells. But validation loss barely moved (7.88 → 7.74) while train kept dropping: strong overfitting, the recurring consequence of a 324-image training set. The model learns the geometry and the train distribution but doesn't generalise — the same data ceiling seen across detection, segmentation, and the hybrid backbone. A top-down visualization of decoded BEV detections (ticket P5-4) is the qualitative check and remains to be built.

---

### Phase 6 Training Log

The temporal stage fuses information across consecutive video frames. A sequence dataloader serves 3-frame windows of consecutive keyframes; a shared backbone runs on every frame; a from-scratch **cross-attention** module lets the current frame's deepest features (the query) attend to a memory built from the two past frames' features (keys/values), each tagged with a learned temporal embedding. The fused features then feed the unchanged Phase 2 FPN detector — so the comparison isolates exactly what temporal fusion adds.

**3-frame temporal detector vs Phase 2 single-frame baseline:**
| Metric | Single-frame | Temporal (3-frame) |
|---|---|---|
| Car AP | 0.291 | 0.257 |
| Pedestrian AP | 0.001 | 0.003 |
| Cyclist AP | 0.097 | 0.143 |
| **mAP** | **0.129** | **0.135** |

**Analysis:** Training early-stopped at epoch 34 (best at 24). Temporal fusion finished marginally above the single-frame detector (mAP 0.135 vs 0.129, ~5% relative) — a real but modest gain, carried by cyclist AP while car AP slipped slightly. The temporal model carries extra parameters (the cross-attention stack) and overfit hard — validation loss climbed from 1.4 to 6.0 while training loss fell toward 0.01 — on only 308 three-frame windows. The honest takeaway matches the rest of the project: the mechanism is sound and implemented from scratch, but a 300-sample regime can't show temporal attention's full value; it would pay off more clearly with longer sequences and far more data.

**Temporal consistency (P6-4):** beyond mAP, `evaluation/eval_flicker.py` runs the detector frame-by-frame over the validation scenes and measures *flicker* — objects detected at t-1 and t+1 but dropped in the middle frame, tracked by nuScenes instance id. The temporal detector's mean flicker rate is **0.135**: about 1 in 7 consistently-visible objects still blinks out for a single frame — a concrete handle on the stability that mAP alone can't see.

---

### Phase 7 Integration & Demo

The final phase unifies the trained phases into one perception system. `PerceptionPipeline` loads three models — the Phase 6 temporal detector, the Phase 4 hybrid CNN-ViT U-Net segmenter, and the Phase 5 Lift-Splat-Shoot BEV detector — and runs all of them on each frame: 2D detections, a semantic segmentation mask, and top-down BEV detections (the deferred P5-4 visualization, decoded from the BEV heatmap). An interactive Streamlit app lets you pick a scene, scrub through its frames, and view the camera image with boxes + segmentation overlaid alongside a bird's-eye-view panel.

**End-to-end benchmark** (MPS, 448×800 input, 3-frame temporal window):
| Model | Parameters |
|---|---|
| Temporal detector | 16.6M |
| Hybrid U-Net segmenter | 27.6M |
| BEV detector | 12.0M |
| **Total** | **56.2M** |

The full pipeline runs all three models in **57 ms/frame (17.5 FPS)** — interactive speed for the demo.

**Ablation summary** (experiments run across phases):
| Comparison | Result |
|---|---|
| Detection backbone: from-scratch → ImageNet-pretrained | mAP 0.042 → 0.129 |
| Segmentation backbone: ResNet vs hybrid CNN-ViT | mIoU 0.327 vs 0.320 (≈ tie) |
| Detection: single-frame vs 3-frame temporal | mAP 0.129 → 0.135; flicker 0.135 |

The consistent thread across every phase: each module works and is built from scratch, but nuScenes mini (~400 samples) is small enough that pretrained features help enormously while extra capacity (ViT, temporal attention) mostly overfits. The architecture is sound; the data is the ceiling.

Run the demo with `streamlit run demo/app.py` (after `pip install streamlit`).

---

## Getting Started

```bash
# Clone the repo
git clone <repo-url>
cd autonomous-driving

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Place nuScenes mini at data/raw/v1.0-mini/
# Download from https://www.nuscenes.org/nuscenes#download (mini split, ~4GB)

# Verify data pipeline end-to-end
python -m data.dataloader
```

---

## Project Structure

```
autonomous-driving/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── configs/                  # Training configs and hyperparameters
├── data/
│   ├── dataset.py            # NuScenesDetectionDataset — 3-class detection, 3D→2D projection
│   ├── dataloader.py         # get_loaders() — train/val DataLoader factory
│   ├── transforms.py         # albumentations pipelines (train + val, bbox-aware)
│   └── raw/v1.0-mini/        # nuScenes mini dataset
├── models/                   # All model architectures
│   ├── backbone/             # CNN and ViT backbones
│   ├── detection/            # Object detection head
│   ├── segmentation/         # Semantic segmentation decoder
│   ├── bev/                  # Bird's eye view transform
│   └── temporal/             # Temporal fusion module
├── training/                 # Training loops, losses, schedulers
├── evaluation/               # Metrics (mAP, mIoU) and eval scripts
├── utils/
│   └── visualize.py          # draw_boxes, visualize_batch, visualize_sample
├── notebooks/
│   └── p0_2_data_exploration.ipynb  # nuScenes schema, cameras, intrinsics, LiDAR projection
└── demo/                     # End-to-end inference and demo scripts
```
