"""
P11-1 — Epistemic uncertainty via MC-dropout.

THE DISTINCTION THAT MATTERS
----------------------------
    Aleatoric uncertainty — noise inherent in the data. A pedestrian 60 m away at
        night occupies 8 blurry pixels; no model, however good, recovers the
        detail. More data does not help.
    Epistemic uncertainty — uncertainty in the *model's parameters*, from having
        seen too little relevant data. More data DOES help.

Phase 9's night collapse is overwhelmingly epistemic: the model has never seen a
night frame. That is why epistemic uncertainty is the right signal for an ODD
monitor — it is high exactly where the model is operating outside its training
distribution, which is exactly the condition we need to detect.

A softmax/sigmoid score does NOT measure this. A network trained only on daylight
will happily emit score 0.9 on a night frame; the score reflects how well the
input matches learned decision boundaries, not whether those boundaries were ever
calibrated for this input. This is the entire reason detections fail *silently*,
and it is the sentence to lead the Phase 11 write-up with.

MC-DROPOUT AS APPROXIMATE BAYESIAN INFERENCE
--------------------------------------------
Gal & Ghahramani (2016) showed that a network trained with dropout, then run at
test time with dropout STILL ACTIVE, samples from an approximate posterior over
weights. Run T stochastic forward passes, and the spread across passes estimates
epistemic uncertainty:

    mean over T  -> the prediction
    variance over T -> the epistemic uncertainty

Cheap, needs no retraining, and requires only that dropout layers exist and are
left in train mode. Its weakness: it underestimates uncertainty relative to a
true posterior, and quality depends heavily on where dropout sits.

WARNING SPECIFIC TO THIS CODEBASE
---------------------------------
`ResNetBackbone` and `DetectionHead` use BatchNorm and, as far as Phase 2 built
them, no dropout in the detection tower. Two consequences:

  1. You must ADD dropout (e.g. Dropout2d after each conv block in the head
     tower) and fine-tune, or MC-dropout has nothing to sample and every pass
     returns an identical result. Verify variance > 0 before trusting any of it.
  2. Do NOT call `model.train()` to enable dropout — that also puts BatchNorm in
     train mode, which updates running statistics from your eval data and
     corrupts the checkpoint. This is the exact bug Phase 2 already hit and fixed
     with the `return_raw` flag. Enable dropout modules selectively; see
     `enable_dropout` below.

If fine-tuning with dropout is too expensive, use a small deep ensemble instead
(`EnsemblePredictor`): train 3-5 detectors from different seeds. Ensembles give
strictly better-calibrated epistemic estimates than MC-dropout, at the cost of
N training runs. With ~3,300 frames and the convergence times in the logs
(best epoch ~8-12), 3 seeds is genuinely affordable, and it is the stronger
result to report.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def enable_dropout(model: nn.Module) -> int:
    """
    Put ONLY the dropout layers into train mode, leaving BatchNorm in eval mode.

    Args: model — a detector already set to .eval().
    Returns: number of dropout modules switched on (assert this is > 0).

    Implementation: iterate `model.modules()` and call `.train()` on any module
    that is an instance of `nn.Dropout`, `nn.Dropout2d`, or `nn.Dropout3d`.
    Everything else keeps whatever mode it already had.
    """
    raise NotImplementedError("P11-1")


class MCDropoutPredictor(nn.Module):
    """
    Wrap a trained detector and run T stochastic passes to estimate uncertainty.

    Args:
        model: a trained FPNDetector (or BEVDetector) containing dropout layers.
        num_samples: T, the number of stochastic passes. 10-30 is the usual range;
            variance estimates stabilise slowly, so measure where yours plateaus
            rather than picking a number. Cost is linear in T — at Phase 7's
            57 ms/frame, T=20 means ~1.1 s/frame, which is why Phase 12 keeps the
            LLM tier event-triggered and does not run this on every frame.

    Returns per detection:
        score_mean — mean sigmoid score across passes (use this, not the single
            deterministic score; averaging is already a mild calibration win)
        score_var  — variance of the score across passes (epistemic signal)
        box_var    — variance of the decoded box coordinates across passes; a box
            whose corners jitter between passes is one the model cannot localise

    THE HARD PART — matching detections across passes
    -------------------------------------------------
    Each pass produces its own post-NMS detection set, and they will not align:
    different counts, different order, objects appearing in some passes only.
    You cannot simply stack and take a variance.

    Recommended approach: run the passes at the RAW anchor level, before NMS.
    Anchors are a fixed, ordered grid identical across passes, so per-anchor
    variance is well defined and needs no matching at all. Compute mean/variance
    over the (B, N_anchors, C) logit tensors, then run postprocess ONCE on the
    mean logits and carry each surviving detection's anchor index so you can look
    up its variance. `FPNDetector.forward(..., return_raw=True)` already gives
    exactly this tensor — Phase 2 built the hook you need here.

    The alternative (greedy IoU matching between passes' final detections) is
    more intuitive but introduces a second matching threshold that silently
    shapes your uncertainty estimates. Avoid it.
    """

    def __init__(self, model: nn.Module, num_samples: int = 20) -> None:
        super().__init__()
        raise NotImplementedError("P11-1")

    @torch.no_grad()
    def forward(self, images: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Args: images — (B, 3, H, W), plus whatever kwargs the wrapped model needs.
        Returns dict with keys: 'score_mean', 'score_var', 'box_mean', 'box_var',
        each (B, N_anchors, ...).
        """
        raise NotImplementedError("P11-1")


class EnsemblePredictor(nn.Module):
    """
    Deep ensemble — the stronger alternative to MC-dropout.

    Args:
        models: list of independently-trained detectors (different seeds; the
            data order and init differ, which is enough for useful diversity).

    Same output contract as MCDropoutPredictor, so the Phase 11 introspection head
    can consume either without changes. Keep that interface identical — it lets
    you report an MC-dropout vs ensemble ablation for free, which is a genuinely
    interesting comparison and costs no extra design work.

    Memory note: 3-5 detectors at 16.6M parameters each is fine on MPS, but run
    them sequentially and accumulate statistics rather than holding all outputs.
    """

    def __init__(self, models: List[nn.Module]) -> None:
        super().__init__()
        raise NotImplementedError("P11-1")

    @torch.no_grad()
    def forward(self, images: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("P11-1")
