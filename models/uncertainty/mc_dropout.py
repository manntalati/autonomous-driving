"""
P11-1 — Epistemic uncertainty via MC-dropout and deep ensembles.

THE DISTINCTION THAT MATTERS
----------------------------
    Aleatoric uncertainty — noise inherent in the data. A pedestrian 60 m away at
        night occupies 8 blurry pixels; no model recovers the detail. More data
        does not help.
    Epistemic uncertainty — uncertainty in the model's *parameters*, from having
        seen too little relevant data. More data DOES help.

Phase 9's night collapse is overwhelmingly epistemic: the model has never seen a
night frame. That is why epistemic uncertainty is the right signal for an ODD
monitor — it is high exactly where the model operates outside its training
distribution.

A sigmoid score does NOT measure this. A network trained only on daylight will
happily emit 0.9 on a night frame; the score reflects how well the input matches
learned decision boundaries, not whether those boundaries were ever calibrated
for this input. That is the whole reason detections fail *silently*.

MC-DROPOUT AS APPROXIMATE BAYESIAN INFERENCE
--------------------------------------------
Gal & Ghahramani (2016): a network trained with dropout, then run at test time
with dropout STILL ACTIVE, samples from an approximate posterior over weights.
T stochastic passes give mean (prediction) and variance (epistemic uncertainty).

TWO TRAPS, BOTH HANDLED HERE
----------------------------
1. `model.train()` would also put BatchNorm in train mode, updating running
   statistics from eval data and corrupting the checkpoint — the exact bug
   Phase 2 hit and fixed with `return_raw`. `enable_dropout` flips ONLY dropout
   modules.
2. Post-NMS detection sets differ between passes (different counts, different
   order), so they cannot be stacked. We sample at the RAW ANCHOR level instead,
   where the anchor grid is fixed and identical across passes, making per-anchor
   variance well-defined with no matching step at all.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn

_DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)


def enable_dropout(model: nn.Module) -> int:
    """
    Put ONLY the dropout layers into train mode, leaving BatchNorm in eval mode.

    Args: model — a detector already set to .eval().
    Returns: number of dropout modules switched on.

    Callers should assert the return is > 0 and that at least one has p > 0;
    otherwise every "stochastic" pass is identical and the variance is zero.

    Prefer `stochastic_dropout` below, which restores the previous mode. Leaving
    dropout enabled leaks randomness into whatever the caller does next.
    """
    count = 0
    for m in model.modules():
        if isinstance(m, _DROPOUT_TYPES):
            m.train()
            count += 1
    return count


@contextmanager
def stochastic_dropout(model: nn.Module):
    """
    Enable dropout for the duration of the block, then restore the previous modes.

    WHY THIS MUST RESTORE (a bug this caused)
    -----------------------------------------
    `harvest_detections` runs the deterministic forward FIRST and then samples:

        cls, box, anchors = model(images, return_raw=True)   # meant to be deterministic
        mc_stats = mc(images)                                # samples

    With a bare `enable_dropout`, dropout stayed on after the first batch, so from
    batch 2 onward the "deterministic" pass was itself sampled — randomising the
    detection set, and making the MC arm's detections differ from the non-MC arm's.
    That silently destroys the controlled comparison the whole phase rests on: the
    two arms are supposed to differ only in whether epistemic FEATURES are present,
    not in which detections they score.

    Yields the number of dropout modules enabled.
    """
    previous = [(m, m.training) for m in model.modules() if isinstance(m, _DROPOUT_TYPES)]
    try:
        for m, _ in previous:
            m.train()
        yield len(previous)
    finally:
        for m, was_training in previous:
            m.train(was_training)


def active_dropout_p(model: nn.Module) -> List[float]:
    """Dropout probabilities present in the model — for verifying MC-dropout is live."""
    return [float(m.p) for m in model.modules() if isinstance(m, _DROPOUT_TYPES)]


def _stack_levels(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate the per-FPN-level lists the detector returns into (B, N, C)."""
    return torch.cat(tensors, dim=1)


class MCDropoutPredictor(nn.Module):
    """
    Wrap a trained detector and run T stochastic passes to estimate uncertainty.

    Args:
        model: trained FPNDetector containing dropout layers with p > 0.
        num_samples: T. 10-30 typical; measure where your variance estimate
            plateaus rather than guessing. Cost is linear in T — at Phase 7's
            57 ms/frame, T=20 is ~1.1 s/frame, which is why Phase 12 keeps the
            LLM tier event-triggered and never runs this every frame.

    Returns per anchor (not per detection — see the module docstring):
        score_mean (B, N, C), score_var (B, N, C),
        box_mean   (B, N, 4), box_var   (B, N, 4),
        anchors    (N, 4)
    """

    def __init__(self, model: nn.Module, num_samples: int = 20) -> None:
        super().__init__()
        self.model = model
        self.num_samples = num_samples

    @torch.no_grad()
    def forward(self, images: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        if self.num_samples < 2:
            raise ValueError("num_samples must be >= 2 to estimate a variance")
        self.model.eval()
        ps = [p for p in active_dropout_p(self.model) if p > 0]
        if not ps:
            raise RuntimeError(
                "MC-dropout requires dropout layers with p > 0. Build the detector "
                "with head_dropout > 0 in the config and fine-tune, or use "
                "EnsemblePredictor instead."
            )

        scores, boxes = [], []
        anchors = None
        # Scoped: dropout is restored to eval on exit, so the caller's next
        # deterministic forward really is deterministic.
        with stochastic_dropout(self.model):
            for _ in range(self.num_samples):
                cls_logits, bbox_deltas, anc = self.model(images, return_raw=True, **kwargs)
                scores.append(torch.sigmoid(_stack_levels(cls_logits)))
                boxes.append(_stack_levels(bbox_deltas))
                anchors = anc

        s = torch.stack(scores)        # (T, B, N, C)
        b = torch.stack(boxes)         # (T, B, N, 4)
        return {
            "score_mean": s.mean(0),
            "score_var": s.var(0, unbiased=False),
            "box_mean": b.mean(0),
            "box_var": b.var(0, unbiased=False),
            "anchors": anchors,
        }


class EnsemblePredictor(nn.Module):
    """
    Deep ensemble — the stronger alternative to MC-dropout.

    Args:
        models: independently-trained detectors (different seeds). Data order and
            init differ, which is enough for useful diversity.

    Identical output contract to MCDropoutPredictor, so the introspection head
    consumes either without changes — which makes an MC-dropout vs ensemble
    ablation free.

    Ensembles give better-calibrated epistemic estimates than MC-dropout at the
    cost of N training runs. With ~3,300 frames and best epochs at 8-12, three
    seeds is genuinely affordable and is the stronger result to report.

    Models are run sequentially and statistics accumulated, so peak memory is one
    model's activations rather than N.
    """

    def __init__(self, models: List[nn.Module]) -> None:
        super().__init__()
        if len(models) < 2:
            raise ValueError("an ensemble needs at least 2 models")
        self.models = nn.ModuleList(models)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        scores, boxes = [], []
        anchors = None
        for m in self.models:
            m.eval()
            cls_logits, bbox_deltas, anc = m(images, return_raw=True, **kwargs)
            scores.append(torch.sigmoid(_stack_levels(cls_logits)))
            boxes.append(_stack_levels(bbox_deltas))
            anchors = anc
        s = torch.stack(scores)
        b = torch.stack(boxes)
        return {
            "score_mean": s.mean(0),
            "score_var": s.var(0, unbiased=False),
            "box_mean": b.mean(0),
            "box_var": b.var(0, unbiased=False),
            "anchors": anchors,
        }


def gather_anchor_uncertainty(
    stats: Dict[str, torch.Tensor],
    anchor_indices: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Look up per-detection uncertainty from per-anchor statistics.

    Args:
        stats: output of MCDropoutPredictor/EnsemblePredictor for ONE image.
        anchor_indices: (M,) index into the anchor axis for each surviving detection.
        labels: (M,) predicted class of each detection.

    Returns: {"score_var": (M,), "box_var": (M,)} — the score variance is read at
    the detection's own predicted class, and the box variance is summed over the
    four coordinates.

    This is the bridge between anchor-level sampling and detection-level features.
    It requires postprocess to carry the anchor index of each surviving detection;
    see `postprocess_with_indices` in models/uncertainty/signals.py.
    """
    sv = stats["score_var"][0]            # (N, C)
    bv = stats["box_var"][0]              # (N, 4)
    idx = anchor_indices.long()
    return {
        "score_var": sv[idx, labels.long()],
        "box_var": bv[idx].sum(dim=-1),
    }
