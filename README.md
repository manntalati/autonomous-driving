# Autonomous Driving Perception Stack

A full autonomous driving perception pipeline built **from scratch** in PyTorch — detection, segmentation, bird's-eye-view, and temporal fusion — and then an investigation into the thing that actually breaks these systems in the field.

**The question this project asks:** what happens to a perception stack when it leaves the conditions it was trained on, and can it tell that it has?

The training data here is 3,376 keyframes of clear daylight. Evaluated on held-out daytime frames the detector reaches **mAP 0.285**. Evaluated on night footage it has never seen, it collapses to **mAP 0.095** — a 67% drop, with cyclists falling 87%. Nothing warns you this is happening. The detector reports the same confidence scores it always did.

That silent failure — not raw accuracy — is the open problem in autonomous driving deployment. Phases 0–8 build the perception stack. Phases 9–12 measure where it breaks, close the gap with radar, and wrap it in a monitor that knows when its own output should not be trusted.

Built as a deep learning capstone covering the modern CV stack: linear classifiers, CNNs, object detection, dense prediction, Vision Transformers, BEV transforms, temporal attention — plus uncertainty quantification, sensor fusion, and an agentic runtime monitor.

---
## Demo

[Demo](https://github.com/user-attachments/assets/af175355-e19e-4206-8d8a-1035cfab3e47)

Camera with detections + segmentation on the left, BEV panel on the right!

Demo w/ Agent<img width="1512" height="831" alt="Screenshot 2026-06-13 at 10 19 07 AM" src="https://github.com/user-attachments/assets/e80e776f-a8d3-45c9-9e33-ffc53f234b7e" />


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
| Dataset | nuScenes — trainval blob (85 scenes, 3,376 keyframes, all daytime) + mini (10 scenes, 3 of them night) |
| Computer Vision | OpenCV |
| Visualization | Matplotlib, OpenCV |
| Training | Custom training loops, LR scheduling, mixed precision |
| Sensors | 6 cameras, LiDAR, 5 radars |
| Evaluation | mAP (detection), mIoU (segmentation), flicker rate, ECE / AUROC / risk-coverage (Phase 11), FPS benchmarks |
| Agent / MCP | Anthropic API, MCP Python SDK (stdio transport) |

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

### Phase 9: Day→Night ODD Audit

> **Goal:** Quantify how far the perception stack degrades outside its operational design domain, and separate that from ordinary overfitting.

| Ticket | Task | Status |
|---|---|---|
| `P9-1` | Condition-labelled evaluation harness (`evaluation/day_night_audit.py`) + `scenes=` dataset override | [✅] |
| `P9-2` | Report both splits; write up the day→night transfer gap | [✅] |
| `P9-3` | Correct the split documentation; expose the official devkit split | [✅] |

---

### Phase 10: Radar Fusion

> **Goal:** Close the night gap with the one sensor that does not care about illumination. Radar measures range and radial velocity directly; a camera infers both from appearance.

| Ticket | Task | Status |
|---|---|---|
| `P10-1` | Radar point-cloud loader (5 sensors → ego frame, motion-compensated) | [ ] |
| `P10-2` | Radar BEV rasterizer (occupancy / radial velocity / RCS channels) | [ ] |
| `P10-3` | Camera+radar BEV fusion module | [ ] |
| `P10-4` | Day/night ablation, reported separately at near and long range | [ ] |

---

### Phase 11: Trust Layer

> **Goal:** Make the stack's uncertainty explicit and calibrated, so a downstream consumer can decide when to stop trusting it.

| Ticket | Task | Status |
|---|---|---|
| `P11-1` | Epistemic uncertainty via MC-dropout / ensemble over the detector | [ ] |
| `P11-2` | Cross-modal disagreement signal (radar ⟂ camera) | [ ] |
| `P11-3` | Introspection head: signals → P(detection is correct) | [ ] |
| `P11-4` | Calibration + failure-prediction metrics (ECE, AUROC, risk-coverage) | [ ] |
| `P11-5` | Per-frame trust score + ODD boundary threshold | [ ] |

---

### Phase 12: Streaming Autonomy Monitor

> **Goal:** A continuously running agent that watches the drive, speaks only when something matters, and abstains when the stack is outside its ODD.

| Ticket | Task | Status |
|---|---|---|
| `P12-1` | Frame stream player (12 Hz mini sweeps for demo, 2 Hz keyframes for eval) | [ ] |
| `P12-2` | Fast tier: per-frame state tracking + deterministic event detector (no LLM) | [ ] |
| `P12-3` | Slow tier: event-triggered LLM advisories with memory of what was already said | [ ] |
| `P12-4` | Abstention behaviour driven by the Phase 11 trust score | [ ] |
| `P12-5` | Streaming eval: warning lead time, false-alarm rate, abstention correctness | [ ] |

---

### Phase 8 (Agentic): MCP Perception Tool API

> **Goal:** Expose the trained perception pipeline as an MCP tool API that an LLM agent can call autonomously to understand driving scenes.

| Ticket | Task | Status |
|---|---|---|
| `A0-1` | MCP server scaffolding (FastMCP, stdio transport, `ping` tool) | [✅] |
| `A0-2` | Hand-built Anthropic tool-use loop (`mcp_tools_to_anthropic`, `run_agent`) | [✅] |
| `A1-1` | `SceneStore` — nuScenes frame loader + UUID per-frame cache | [✅] |
| `A1-2` | `ModelRegistry` — pipeline singleton + `run_perception(frame_id)` | [✅] |
| `A1-3` | Core tools: `list_scenes`, `load_frame`, `detect_objects`, `segment_scene`, `bev_map` | [✅] |
| `A1-4` | Driving-decision tools: `check_lane_switch_safety`, `check_turn_clearance`, `check_obstacle_stop`, `check_pedestrian_crossing`, `estimate_following_distance`, `scene_summary` | [✅] |
| `A2-1` | Orchestrator agent (spatial-reasoning system prompt + chained tool calls) | [ ] |
| `A3-1` | Eval harness (GT-derived question bank, accuracy / tool-call count / latency / cost) | [ ] |
| `A4-1` | Interactive Streamlit agent demo with live tool-trace panel | [ ] |

Detailed agentic roadmap: [docs/agentic_perception_roadmap.md](docs/agentic_perception_roadmap.md)

**Quickstart:**
```bash
# Set your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-..." > .env

# Ask a scene question — the agent will call load_frame + detect_objects automatically
python -m agent.run "how many cars are in scene-0103 frame 5?"

# Safety decision example — triggers load_frame + bev_map + check_lane_switch_safety
python -m agent.run "is it safe to change lanes left in scene-0103 frame 10?"
```

**MCP tool inventory:**

| Tool | Purpose |
|---|---|
| `list_scenes` | List all available nuScenes scenes |
| `load_frame(scene, idx)` | Load a frame, get a `frame_id` for subsequent calls |
| `detect_objects(frame_id)` | 2D bounding boxes with class + confidence |
| `segment_scene(frame_id)` | Per-class pixel coverage + ahead-region flags |
| `bev_map(frame_id)` | Top-down object positions in real-world metres |
| `check_lane_switch_safety(frame_id, dir)` | Safe to change left/right? |
| `check_turn_clearance(frame_id, dir)` | Turn clearance at intersection |
| `check_obstacle_stop(frame_id)` | Should the vehicle stop? Nearest obstacle distance |
| `check_pedestrian_crossing(frame_id)` | Crossing ahead + pedestrian occupancy |
| `estimate_following_distance(frame_id)` | Metres to nearest car ahead |
| `scene_summary(frame_id)` | Full structured scene JSON for holistic reasoning |

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

### Phase 9 Day→Night ODD Audit

Every earlier phase blamed its modest numbers on dataset size. That explanation was incomplete, and the data on disk turned out to contain a clean natural experiment.

**The setup.** Two nuScenes roots are present. The downloaded trainval blob is 85 scenes (scene-0001…0102), 3,376 CAM_FRONT keyframes — and **100% daytime, clear weather**; a keyword scan over all 85 hand-written scene descriptions finds no mention of night, rain, or dusk. The mini split contains three night scenes: 1077, 1094 ("Night, after rain"), and 1100 ("Night… difficult lighting"). Only scene-0061 overlaps the blob, and it is a day scene. **The night scenes are therefore provably unseen by any trainval-trained model** — a zero-shot day→night transfer benchmark with no contamination.

`evaluation/day_night_audit.py` evaluates one checkpoint across four condition cells (and refuses to report anything if a night scene is found in the training set):

| Cell | Frames | mAP | Car | Ped | Cyclist |
|---|---|---|---|---|---|
| seen, day (trainval train) | 2,863 | 0.615 | 0.761 | 0.496 | 0.587 |
| unseen, day (trainval val) | 513 | 0.285 | 0.575 | 0.157 | 0.124 |
| **unseen, night (mini)** | 121 | **0.095** | 0.188 | 0.081 | 0.016 |
| unseen, day (mini) — *control* | 244 | 0.304 | 0.486 | 0.320 | 0.105 |

**Day→night transfer gap: 0.285 → 0.095, a 67% relative collapse.** Both rows are unseen data, so generalisation is held constant and the difference is domain shift alone.

**The control rules out the obvious confound.** One could object that the night scenes come from mini, so the drop might reflect a dataset difference rather than darkness. It does not: mini's *daytime* scenes score **0.304**, statistically indistinguishable from trainval's held-out day scenes at 0.285 — same source as the night frames, none of the collapse. The degradation tracks lighting, not provenance.

**Vulnerable road users degrade worst.** Against the mini-day control, cyclist AP falls 85% (0.105 → 0.016) and pedestrian AP 75% (0.320 → 0.081), versus 61% for cars. The classes that disappear at night are exactly the ones an autonomous vehicle cannot afford to miss — small, unlit, and unpredictable. This is the safety case for Phases 10–12, and it is measured rather than asserted.

**A correction to the Phase 2 record.** Earlier phases reported pedestrian AP ≈ 0.001 and attributed it to anchor geometry — the smallest anchor being too large for a 19-pixel pedestrian — with a proposed fix of adding a P2 FPN level. That diagnosis was wrong. Pedestrian AP is **0.496** on seen daytime data with the same anchor configuration. The anchors were never the problem; training-set size and night conditions were. The P2 remedy has been dropped.

**Why this reframes the project.** The recurring "the data is the ceiling" conclusion holds, but it is not the whole story. There are two distinct failure modes stacked on top of each other: a generalisation gap (0.615 → 0.285) from limited data, and a domain-shift gap (0.285 → 0.095) from operating outside the training distribution. Only the first is fixed by collecting more of the same data. The second is what makes deployed perception dangerous, because the model's confidence scores do not drop when it happens — it fails silently. Phases 10–12 address that directly.

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
├── configs/                  # Training + agent configs and hyperparameters
│   └── agent.yaml            # Checkpoint paths + thresholds for the MCP server
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
├── demo/                     # End-to-end inference and demo scripts
├── mcp_server/               # MCP tool API (agentic perception platform)
│   ├── server.py             # FastMCP instance — ping + register_all_tools()
│   ├── scene_store.py        # SceneStore — frame loading, UUID cache, calibration
│   ├── model_registry.py     # ModelRegistry singleton — pipeline + run_perception()
│   └── perception_tools.py   # All 11 perception tools (5 core + 6 driving-decision)
├── agent/                    # LLM agent that calls the MCP tools
│   ├── run.py                # CLI entry point
│   ├── loop.py               # Hand-built Anthropic tool-use loop
│   ├── mcp_client.py         # stdio MCP client wrapper
│   └── config.py             # .env loader (ANTHROPIC_API_KEY)
└── docs/
    └── agentic_perception_roadmap.md  # Phase-by-phase agentic roadmap
```
