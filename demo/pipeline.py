"""
Unified perception pipeline (Phase 7).

Loads the three trained models — the Phase 6 temporal detector, the Phase 4
hybrid U-Net segmenter, and the Phase 5 BEV detector — and runs all of them on
one frame, returning detections + a segmentation mask + BEV detections.
"""
from __future__ import annotations
from pathlib import Path
import yaml
import torch

from models.temporal.train_temporal import build_temporal_detector
from models.segmentation.train_seg import build_segmenter
from models.bev.train_bev import build_bev_detector
from models.bev.bev_detector import decode_bev_detections


def _load(model, checkpoint: str, device: torch.device):
    """Load a checkpoint into a model, move to device, set eval."""
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


class PerceptionPipeline:
    """Detection + segmentation + BEV inference behind one interface."""

    def __init__(self, cfg: dict, device: torch.device) -> None:
        """
        Args: cfg — parsed configs/demo.yaml; device — torch device.
        Builds each model from its training config and loads its checkpoint.
        """
        self.cfg = cfg
        self.device = device

        det_cfg = yaml.safe_load(open(cfg["detector"]["config"]))
        self.detector = _load(build_temporal_detector(det_cfg),
                              cfg["detector"]["checkpoint"], device)

        seg_cfg = yaml.safe_load(open(cfg["segmenter"]["config"]))
        self.segmenter = _load(build_segmenter(seg_cfg),
                               cfg["segmenter"]["checkpoint"], device)

        self.bev_cfg = yaml.safe_load(open(cfg["bev"]["config"]))
        self.bev = _load(build_bev_detector(self.bev_cfg),
                         cfg["bev"]["checkpoint"], device)

        # Phase 10 A/B arm. Optional: the demo still runs with camera-only BEV if
        # the radar model has not been trained. Both arms come from the ablation
        # (12 fixed epochs, seed 0, final checkpoints), so a side-by-side panel is
        # showing the same comparison the numbers came from.
        # BOTH arms are loaded from the ablation, never mixed with the Phase-8
        # `bev` model above. That model was trained under a different protocol
        # (early stopping on mIoU, different epoch count), so pairing it against
        # the radar arm would reintroduce precisely the training-maturity confound
        # the ablation was restructured to remove.
        self.bev_radar = None
        self.bev_radar_cfg = None
        self.bev_ab_camera = None
        self.bev_ab_camera_cfg = None
        rad, cam = cfg.get("bev_radar"), cfg.get("bev_ab_camera")
        if (rad and cam and Path(rad["checkpoint"]).exists()
                and Path(cam["checkpoint"]).exists()):
            self.bev_radar_cfg = yaml.safe_load(open(rad["config"]))
            self.bev_radar = _load(build_bev_detector(self.bev_radar_cfg),
                                   rad["checkpoint"], device)
            self.bev_ab_camera_cfg = yaml.safe_load(open(cam["config"]))
            self.bev_ab_camera = _load(build_bev_detector(self.bev_ab_camera_cfg),
                                       cam["checkpoint"], device)

    @torch.no_grad()
    def process_frame(self, frame_window: torch.Tensor, intrinsic: torch.Tensor,
                      cam_to_ego: torch.Tensor, bev_surround=None) -> dict:
        """
        Run all three models on one timestep.
        Args:
          frame_window — (T, 3, H, W) the last T CAM_FRONT frames; LAST = current.
          intrinsic — (3, 3) CAM_FRONT K for the current frame.
          cam_to_ego — (4, 4) CAM_FRONT→ego transform for the current frame.
          bev_surround — optional (images (N,3,H,W), intrinsics (N,3,3),
                         cam_to_ego (N,4,4)) for the BEV path — all N surround
                         cameras of the current frame. If None, BEV runs on
                         CAM_FRONT only (N=1, front-half BEV map).
        Returns: dict with
          'boxes' (N,4), 'scores' (N,), 'labels' (N,)   — 2D detections (current frame),
          'seg_mask' (H, W) int                         — segmentation of the current frame,
          'bev_boxes' (M,5), 'bev_scores' (M,), 'bev_labels' (M,) — BEV detections,
          'bev_seg' (X, Y) int                          — BEV semantic-map argmax.
        """
        device = self.device
        frame_window = frame_window.to(device)
        current = frame_window[-1].unsqueeze(0)

        # ── detection ──
        boxes_list, scores_list, labels_list = self.detector(frame_window.unsqueeze(0))
        boxes, scores, labels = boxes_list[0].cpu(), scores_list[0].cpu(), labels_list[0].cpu()
        keep = scores >= self.cfg.get("det_score_threshold", 0.3)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # ── segmentation ──
        seg_mask = self.segmenter(current).argmax(dim=1)[0].cpu()

        # ── BEV detection + segmentation ──
        if bev_surround is not None:
            imgs_n, k_n, c2e_n = bev_surround
            bev_out = self.bev(
                imgs_n.unsqueeze(0).to(device),         # (1, N, 3, H, W)
                k_n.unsqueeze(0).to(device),            # (1, N, 3, 3)
                c2e_n.unsqueeze(0).to(device),          # (1, N, 4, 4)
            )
        else:
            bev_out = self.bev(
                current.unsqueeze(1),                   # (1, 1, 3, H, W) — CAM_FRONT only
                intrinsic.to(device).view(1, 1, 3, 3),
                cam_to_ego.to(device).view(1, 1, 4, 4),
            )
        bev_boxes, bev_scores, bev_labels = decode_bev_detections(
            bev_out["heatmap"][0].cpu(), bev_out["regression"][0].cpu(),
            tuple(self.bev_cfg["xbound"]), tuple(self.bev_cfg["ybound"]),
            score_threshold=self.cfg.get("bev_score_threshold", 0.3),
            max_detections=self.cfg.get("bev_max_detections", 50),
        )
        bev_seg = bev_out["seg"][0].argmax(dim=0).cpu()   # (X, Y) BEV semantic map

        return {
            "boxes": boxes, "scores": scores, "labels": labels,
            "seg_mask": seg_mask,
            "bev_boxes": bev_boxes, "bev_scores": bev_scores, "bev_labels": bev_labels,
            "bev_seg": bev_seg,
        }

    @property
    def has_radar_arm(self) -> bool:
        return self.bev_radar is not None

    @torch.no_grad()
    def process_bev_pair(self, images: torch.Tensor, intrinsic: torch.Tensor,
                         cam_to_ego: torch.Tensor, radar_bev: torch.Tensor) -> dict:
        """
        Run BOTH BEV arms on one frame for the Phase 10 side-by-side panel.

        Args:
          images — (1, N, 3, H, W) camera batch.
          intrinsic — (1, N, 3, 3); cam_to_ego — (1, N, 4, 4).
          radar_bev — (1, 5, X, Y) rasterized radar grid.

        Returns: {"camera": {...}, "camera_radar": {...}, "gate": float|None},
        each inner dict holding boxes / scores / labels / seg.

        Both arms decode with the SAME score threshold. Using different thresholds
        would let the panel show a difference that came from the decode step
        rather than from radar.
        """
        if self.bev_radar is None or self.bev_ab_camera is None:
            raise RuntimeError("A/B panel needs both ablation checkpoints")

        # Validate the contract instead of letting a wrong rank surface as a
        # confusing channel-count error deep inside the first convolution. A
        # caller that forgets the batch dimension gets (6,3,H,W) flattened to 18
        # "channels"; this says what actually went wrong.
        if images.dim() != 5:
            raise ValueError(
                f"images must be (B, N, 3, H, W); got {tuple(images.shape)}. "
                "Add a batch dimension — e.g. images.unsqueeze(0)."
            )
        want = next(self.bev_radar.parameters()).device
        for name, t in (("images", images), ("intrinsic", intrinsic),
                        ("cam_to_ego", cam_to_ego), ("radar_bev", radar_bev)):
            if t is not None and t.device != want:
                raise ValueError(
                    f"{name} is on {t.device} but the model is on {want}; "
                    "move every input with .to(pipeline.device)."
                )

        thr = self.cfg.get("bev_score_threshold", 0.3)
        topk = self.cfg.get("bev_max_detections", 50)
        out = {}
        for name, model, mcfg, radar in (
            ("camera", self.bev_ab_camera, self.bev_ab_camera_cfg, None),
            ("camera_radar", self.bev_radar, self.bev_radar_cfg, radar_bev),
        ):
            o = model(images, intrinsic, cam_to_ego, radar_bev=radar)
            b, s, l = decode_bev_detections(
                o["heatmap"][0].cpu(), o["regression"][0].cpu(),
                tuple(mcfg["xbound"]), tuple(mcfg["ybound"]),
                score_threshold=thr, max_detections=topk,
            )
            out[name] = {"boxes": b, "scores": s, "labels": l,
                         "seg": o["seg"][0].argmax(dim=0).cpu()}
            if "gate" in o:
                out["gate"] = o["gate"]
        out.setdefault("gate", None)
        return out
