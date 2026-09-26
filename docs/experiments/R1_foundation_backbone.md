# R1: Frozen foundation backbone vs the night gap

> **DRAFT.** The thresholds below are proposals. Finalize them, and resolve the
> open decisions at the end, before the first R1 training run. Then set the status
> to `registered` and freeze everything above **Result**.

| | |
|---|---|
| **Status** | draft |
| **Question** | Q2: is the night gap about data, representation, or geometry? |
| **Registered** | *not yet* |
| **Time-box ends** | registration + 2 weeks |
| **Depends on** | P14-3 (seeded detector training), so that the baseline has 3 seeds |

## Why this experiment

Phase 9 measured a 67% day→night collapse (mAP 0.285 → 0.095) in a detector whose
backbone is an ImageNet-pretrained ResNet-18, fine-tuned on 85 daytime scenes.
There are two readings of that result:

- **Data:** the detector has never seen a night pixel, and no representation fixes
  that.
- **Representation:** a backbone pretrained on far broader imagery already encodes
  features that survive low light. The 85-scene fine-tune merely fails to build
  them.

A frozen self-supervised foundation backbone separates the two cheaply. It adds
no night *labels*, so any gap it closes comes from the representation.

## Baseline

| | |
|---|---|
| Scorecard row | `results/scorecard/baseline_resnet18.json` *(to be scored once P14-3 exists)* |
| Config | `configs/detector.yaml` |
| Checkpoints | `checkpoints/detector_resnet18_s{seed}.pt`, seeds `0,1,2` |
| Today's single-seed reference | `checkpoints/detector_best.pt`: `det.unseen_day.mAP` 0.2853, `det.unseen_night.mAP` 0.0951, `det.night_gap_rel` 0.667 |

The single-seed numbers are context, not the comparison. R1 is compared against
the 3-seed baseline row.

## The one change

Replace the detector's image backbone with a **frozen DINOv2 ViT-S/14**, plus
a small trainable adapter. The adapter maps the ViT's stride-14 patch tokens to
the C3/C4/C5 feature maps the existing FPN expects: 128/256/512 channels at
strides 8/16/32. A ViTDet-style simple pyramid does this with one upsample for
C3, a projection for C4 and a downsample for C5.

The backbone and its adapter count as one coherent change; the adapter only
exists to fit the new backbone to the FPN. Everything else stays identical to
`configs/detector.yaml`: the FPN, the head, the anchors, the losses, the LR
schedule, the 85 trainval scenes and the input pipeline. The config diff should be
a `backbone:` key and the adapter's settings, and nothing else.

## Hypothesis

With no night training data, a frozen DINOv2 backbone shrinks the relative
day→night gap by at least a third, and does so by raising night accuracy, not by
lowering day accuracy.

## Metric and threshold

| Role | Cell | Baseline | Threshold |
|---|---|---|---|
| Primary | `det.night_gap_rel` | 3-seed mean (≈ 0.667 today) | R1 mean ≤ ⅔ × baseline mean (≈ 0.445), **and** R1 mean below the baseline's lower 95% bound |
| Guard | `det.unseen_night.mAP` | 3-seed mean (≈ 0.095 today) | R1 mean above the baseline's upper 95% bound |
| Reported, not decided on | `det.unseen_day.mAP`, `det.seen_day.mAP`, `det.unseen_miniday.mAP`, `det.foreign.*` | | |
| Reported, not decided on | per-class night AP (pedestrian, cyclist) from the row's `details` | | |

The threshold is **⅔ × the baseline's 3-seed mean gap**. It is fixed once the
baseline row is scored and before any R1 training. It will only equal 0.445 if the
3-seed mean matches today's single run.

**Why a relative gap plus a guard.** A stronger backbone will probably lift day mAP
too. An absolute gap (day − night) can then *grow* even when night improves a
lot, so it penalizes the outcome the hypothesis predicts. The relative gap
normalizes for overall accuracy, as the Phase 13 "retained vs native" column
does. On its own, though, the relative gap can be gamed: a model that gets worse
in daylight and no better at night also shrinks it. The guard closes that hole.
Night accuracy itself must rise beyond the baseline's spread.

## Failure condition

- **Refuted:** the R1 mean `det.night_gap_rel` lies inside the ResNet baseline's
  95% interval. The backbone makes no detectable difference to the gap.
- **Not confirmed (guard):** the primary passes, but the R1 mean
  `det.unseen_night.mAP` does not clear the baseline's upper 95% bound. The gap
  shrank without night getting better, which means day got worse.
- **Not confirmed (partial):** the R1 mean gap is outside the baseline's interval
  but above ⅔ × the baseline mean. The effect is real but smaller than claimed.
  Report the size, and do not round it up to a confirmation.
- **Confirmed:** the primary and the guard both pass.

## Protocol

- **Seeds:** 0, 1 and 2 for both arms, trained under the same protocol.
- **Reproduce first (rule 4):** before any night claim, check that the frozen
  features are being used correctly in-domain:
  1. Load the official weights. Confirm that the preprocessing (ImageNet
     mean/std, which `data/transforms.py` already uses) and the output token grid
     match the reference implementation on one image.
  2. Run a linear probe on the Phase 1 classification subtask with frozen DINOv2
     features, against the same probe on the ImageNet ResNet-18. DINOv2's
     published linear-probe results say it should not lose. If it does, suspect
     the implementation (normalization, input size, token handling) before the
     idea.
- **Input size:** 448×800 is not a multiple of 14. Pad the width to 812 and
  leave the height at 448 (32 patches), rather than resizing, because the
  anchors are defined in pixels.
- **Training:** `python -m models.detection.train_detector configs/detector_dinov2.yaml`,
  with the baseline's epoch budget and checkpoint rule (see the open decisions).
- **Scoring:**
  ```bash
  python -m evaluation.scorecard --name r1_dinov2_vits14 \
      --det-config configs/detector_dinov2.yaml \
      --det-ckpt 'checkpoints/detector_dinov2_s{seed}.pt' --seeds 0,1,2
  ```
  BEV cells use the unchanged baseline BEV model, so they match the baseline row.
- **Compute budget:** only the adapter, FPN and head train. Measure images/sec in
  the first epoch, and record it here before committing to all three seeds.

## Known confounders

1. **Pretraining data vs architecture.** DINOv2 differs from the baseline in both
   architecture (ViT vs CNN) and pretraining data (LVD-142M vs ImageNet-1k,
   self-supervised vs supervised). R1 tests "this backbone", not which of the two
   matters. The follow-up arm, which is *not* part of R1, is an ImageNet-supervised
   ViT-S. It shares DINOv2's architecture but not its data.
2. **Frozen vs fine-tuned.** The baseline fine-tunes its backbone at 0.1× LR, and
   R1 keeps its backbone frozen. That keeps the test clean: frozen features cannot
   adapt toward daytime-only statistics. But it means R1 does not show what a
   fine-tuned DINOv2 would do.
3. **Model selection on `unseen_day`.** The detector trainer early-stops on
   val mAP, and the val scenes are the scorecard's `unseen_day` cell. So day mAP
   is optimistically biased in both arms, while night mAP is untouched by
   selection. This affects both arms alike, but it inflates the gap's
   denominator slightly.
4. **Small night set.** The night cell has 121 frames across 3 scenes, and
   cyclists are rare in it. Per-class night AP is reported with GT counts, and
   no claim rests on a single class.

## Open decisions (resolve before registering)

- [ ] ViT-S/14 first (cheap), or go straight to ViT-B/14?
- [ ] DINOv2 or DINOv3. The gameplan allows either. Confirm DINOv3's current
      availability, license and patch size before choosing it; that information
      was not looked up when this draft was written.
- [ ] Stopping rule: keep the baseline's early stopping on val mAP (matches the
      existing checkpoint, but see confounder 3), or train both arms for a fixed
      number of epochs (the Phase 10/13 lesson: a symmetric criterion does not
      give symmetric maturity)?
- [ ] Adapter design (simple pyramid vs FPN-style lateral convs), frozen once
      chosen.

---

## Result

*(filled in after the run)*

## Verdict

## Amendments
