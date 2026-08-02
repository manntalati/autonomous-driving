import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

CLASS_COLORS = [
    (0,   0,   255),   # car        — red
    (0,   255,  0 ),   # pedestrian — green
    (255,  0,   0 ),   # cyclist    — blue
]

CLASS_NAME = {
    0: 'car',
    1: 'pedestrian',
    2: 'cyclist'
}

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Segmentation class colours (BGR) — background / drivable / lane / ped_crossing / walkway.
SEG_COLORS = np.array([
    (0,   0,   0  ),   # background  — black (not drawn)
    (128, 64,  128),   # drivable    — purple
    (0,   255, 255),   # lane        — yellow
    (60,  20,  220),   # ped_crossing— crimson
    (232, 35,  244),   # walkway     — magenta
], dtype=np.uint8)


def overlay_segmentation(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5,
                          skip_classes=None) -> np.ndarray:
    """
    Blend a per-pixel segmentation mask over an image.
    Args: image — (H, W, 3) uint8; mask — (H, W) int class IDs; alpha — blend weight;
          skip_classes — iterable of class IDs to leave un-painted (useful for the
          drivable class, whose road-surface polygons paint over vehicles on the road
          and obscure the detection boxes).
    Returns: (H, W, 3) uint8 — image with non-background, non-skipped classes blended.
    """
    mask = np.asarray(mask)
    out = image.astype(np.float32).copy()
    colours = SEG_COLORS[mask].astype(np.float32)   # (H, W, 3)
    fg = mask > 0                                    # leave background untouched
    if skip_classes:
        for c in skip_classes:
            fg &= mask != c
    out[fg] = (1.0 - alpha) * out[fg] + alpha * colours[fg]
    return out.astype(np.uint8)


def draw_bev(bev_boxes, scores, labels, xbound, ybound, seg=None, canvas_px: int = 400) -> np.ndarray:
    """
    Render the BEV scene as a top-down image (P5-4 / P8-5).
    Args:
      bev_boxes — (N, 5) [x, y, length, width, yaw] ego-frame BEV boxes.
      scores / labels — (N,) detection scores and class ids.
      xbound/ybound — BEV grid extent the canvas spans.
      seg — optional (X, Y) BEV semantic-map class grid; drawn as a coloured
            (dimmed) background — drivable / lane / walkway road layout.
      canvas_px — output square size in pixels.
    Returns: (canvas_px, canvas_px, 3) uint8 top-down view — ego forward = up,
      with the road map underneath, range rings, ego marker, and detection boxes.
    """
    x_lo, x_hi, _ = xbound
    y_lo, y_hi, _ = ybound

    if seg is not None:
        seg = np.asarray(seg)
        bg = SEG_COLORS[seg][::-1, :, :]                       # flip x so forward = up
        canvas = cv2.resize(bg, (canvas_px, canvas_px), interpolation=cv2.INTER_NEAREST)
        canvas = (canvas.astype(np.float32) * 0.55).astype(np.uint8)   # dim so boxes pop
    else:
        canvas = np.zeros((canvas_px, canvas_px, 3), dtype=np.uint8)

    def to_px(x, y):
        col = int((y - y_lo) / (y_hi - y_lo) * canvas_px)
        row = int(canvas_px - (x - x_lo) / (x_hi - x_lo) * canvas_px)  # forward → up
        return col, row

    # range rings every 10 m + ego-vehicle marker
    ego = to_px(0.0, 0.0)
    metres_per_px = (x_hi - x_lo) / canvas_px
    for r in range(10, int(max(x_hi, y_hi)) + 1, 10):
        cv2.circle(canvas, ego, int(r / metres_per_px), (110, 110, 110), 1)
    cv2.circle(canvas, ego, 5, (255, 255, 255), -1)

    for i in range(len(bev_boxes)):
        x, y, length, width, yaw = [float(v) for v in bev_boxes[i]]
        local = np.array([[ length / 2,  width / 2], [ length / 2, -width / 2],
                          [-length / 2, -width / 2], [-length / 2,  width / 2]])
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        world = local @ rot.T + np.array([x, y])
        pts = np.array([to_px(wx, wy) for wx, wy in world], dtype=np.int32)
        color = tuple(int(v) for v in CLASS_COLORS[int(labels[i])])
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
    return canvas


# ── Phase 10 visualisation ──────────────────────────────────────────────────
# The P10-4 range buckets. Drawing them on the BEV canvas is what makes the
# headline finding legible: camera-only mAP is 0.239 near and 0.012 far (in
# daylight), while camera+radar holds 0.102 far. Without the band boundaries a
# viewer cannot see *where* on the canvas the two models diverge.
RANGE_BANDS = [("near", 0.0, 20.0), ("mid", 20.0, 35.0), ("far", 35.0, 51.2)]

# Radar returns are drawn as CROSSES, not dots: shape carries identity so the
# overlay never depends on colour alone (the three class colours are already
# pure R/G/B, and a fourth hue would be one confusion away from a cyclist box).
RADAR_COLOR = (255, 255, 0)      # BGR — cyan


def draw_radar_points(canvas: np.ndarray, radar_points, xbound, ybound,
                      color=RADAR_COLOR, min_size: int = 2, max_size: int = 5) -> np.ndarray:
    """
    Overlay ego-frame radar returns on an existing BEV canvas.

    Args:
      canvas — (P, P, 3) uint8 BEV image from draw_bev (modified in place and returned).
      radar_points — (N, 6) [x, y, z, rcs, vx, vy] ego-frame returns.
      xbound/ybound — the extent the canvas spans, matching draw_bev.
      min_size/max_size — cross half-length in px, scaled by RCS so strong
        reflectors read as larger marks.

    Returns: the canvas.
    """
    pts = np.asarray(radar_points, dtype=np.float64).reshape(-1, 6)
    if len(pts) == 0:
        return canvas
    p = canvas.shape[0]
    x_lo, x_hi, _ = xbound
    y_lo, y_hi, _ = ybound

    rcs = pts[:, 3]
    lo, hi = float(rcs.min()), float(rcs.max())
    span = max(hi - lo, 1e-6)

    for (x, y, _z, r, _vx, _vy) in pts:
        col = int((y - y_lo) / (y_hi - y_lo) * p)
        row = int(p - (x - x_lo) / (x_hi - x_lo) * p)
        if not (0 <= col < p and 0 <= row < p):
            continue
        s = int(min_size + (r - lo) / span * (max_size - min_size))
        cv2.line(canvas, (col - s, row), (col + s, row), color, 1, cv2.LINE_AA)
        cv2.line(canvas, (col, row - s), (col, row + s), color, 1, cv2.LINE_AA)
    return canvas


def draw_range_bands(canvas: np.ndarray, xbound, ybound, label: bool = True) -> np.ndarray:
    """
    Draw the near/mid/far boundaries used by the P10-4 ablation.

    Rings at 20 m and 35 m, brighter than draw_bev's decorative 10 m rings so the
    analysis boundaries are the ones that read. Returns the canvas.
    """
    p = canvas.shape[0]
    x_lo, x_hi, _ = xbound
    metres_per_px = (x_hi - x_lo) / p
    ego = (int((0.0 - ybound[0]) / (ybound[1] - ybound[0]) * p),
           int(p - (0.0 - x_lo) / (x_hi - x_lo) * p))
    for name, _lo, hi in RANGE_BANDS[:-1]:
        cv2.circle(canvas, ego, int(hi / metres_per_px), (200, 200, 200), 1, cv2.LINE_AA)
        if label:
            ty = ego[1] - int(hi / metres_per_px)
            if 10 < ty < p - 4:
                cv2.putText(canvas, f"{hi:.0f}m", (ego[0] + 4, ty - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def _banner(canvas: np.ndarray, text: str, sub: str = "", height: int = 34) -> np.ndarray:
    """Title strip above a panel. Text uses neutral ink, never a data colour."""
    p = canvas.shape[1]
    bar = np.full((height, p, 3), 24, dtype=np.uint8)
    cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
    if sub:
        (tw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(bar, sub, (p - tw - 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (170, 170, 170), 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def draw_bev_comparison(left_bev: np.ndarray, right_bev: np.ndarray,
                        left_title: str = "camera only", right_title: str = "camera + radar",
                        left_sub: str = "", right_sub: str = "", gap_px: int = 2) -> np.ndarray:
    """
    Side-by-side BEV panels for the Phase 10 A/B comparison.

    Args: two equal-size BEV canvases (from draw_bev, optionally with radar and
      range bands already overlaid), plus per-panel titles and right-aligned
      sub-labels (e.g. detection counts).

    The panels are separated by a 2px surface gap rather than a border — the same
    spacer rule the charts use. Identity comes from the titles and from position,
    so the two sides need no colour coding of their own and the class colours
    inside each panel keep their usual meaning.
    """
    if left_bev.shape != right_bev.shape:
        raise ValueError(f"panels must match: {left_bev.shape} vs {right_bev.shape}")
    l = _banner(left_bev, left_title, left_sub)
    r = _banner(right_bev, right_title, right_sub)
    gap = np.full((l.shape[0], gap_px, 3), 24, dtype=np.uint8)
    return np.hstack([l, gap, r])


def _ascii(text: str) -> str:
    """
    OpenCV's Hershey fonts are ASCII-only — anything outside renders as '???'.
    Reason strings come from TrustScorer and titles are author-written, so both
    can pick up en/em dashes and typographic quotes. Transliterate the common
    offenders and drop the rest rather than shipping '???' into a demo.
    """
    for a, b in (("—", "-"), ("–", "-"), ("’", "'"), ("‘", "'"),
                 ("“", '"'), ("”", '"'), ("×", "x"), ("≥", ">="), ("≤", "<=")):
        text = text.replace(a, b)
    return text.encode("ascii", "ignore").decode()


def draw_trust_banner(width: int, trust: float, in_odd: bool, reason: str = "",
                      height: int = 46) -> np.ndarray:
    """
    Phase 11/12 trust strip: per-frame trust score and ODD state.

    Uses the reserved STATUS palette (good / critical), never a categorical hue,
    and always ships text alongside the colour — state is never colour-alone.
    """
    good, critical = (122, 175, 27), (72, 73, 227)      # BGR of #1baf7a / #e34948
    color = good if in_odd else critical
    bar = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.rectangle(bar, (0, 0), (6, height), color, -1)          # status keyline
    state = "IN ODD" if in_odd else "OUTSIDE ODD - DO NOT RELY"
    cv2.putText(bar, _ascii(f"trust {trust:.2f}   {state}"), (14, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    if reason:
        cv2.putText(bar, _ascii(reason)[:88], (14, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (170, 170, 170), 1, cv2.LINE_AA)
    # trust meter, right-aligned
    mx0, mx1 = width - 130, width - 14
    cv2.rectangle(bar, (mx0, 12), (mx1, 22), (70, 70, 70), -1)
    cv2.rectangle(bar, (mx0, 12), (mx0 + int((mx1 - mx0) * max(0.0, min(1.0, trust))), 22),
                  color, -1)
    return bar


def draw_boxes(image, boxes, labels, scores=None) -> np.ndarray:
    """
    Draw colored bounding boxes and class labels onto an image.
    Args: image — (H, W, 3) uint8 numpy array; boxes — list of [x1,y1,x2,y2];
          labels — list of int class indices; scores — optional list of floats shown in label text.
    Returns: annotated (H, W, 3) uint8 numpy array (copy of input).
    """
    image_copy = image.copy()
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = CLASS_COLORS[label]
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness=2)
        if scores is not None:
            text_string = f"{CLASS_NAME[label]} {scores[i]:.2f}"
        else:
            text_string = CLASS_NAME[label]
        cv2.putText(image_copy, text_string, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image_copy


def visualize_batch(images, targets, class_names) -> None:
    """
    Display a batch of images with their GT boxes in a matplotlib grid.
    Args: images — (B, 3, H, W) float tensor (ImageNet-normalized);
          targets — list of B target dicts with 'boxes', 'labels', 'meta' keys;
          class_names — list of class name strings (unused directly, labels drive colors).
    """
    B = images.shape[0]
    ncols = min(B, 4)
    nrows = (B + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i in range(B):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img * _STD + _MEAN, 0, 1)
        img = (img * 255).astype(np.uint8)

        boxes  = targets[i]['boxes'].tolist()
        labels = targets[i]['labels'].tolist()
        annotated = draw_boxes(img, boxes, labels)

        title = targets[i].get('meta', {}).get('camera', f'sample {i}')
        axes[i].imshow(annotated)
        axes[i].set_title(title, fontsize=9)
        axes[i].axis('off')

    for j in range(B, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_sample(dataset, idx) -> None:
    """
    Visualize a single dataset sample by index.
    Args: dataset — NuScenesDetectionDataset instance; idx — sample index.
    """
    image_tensor, targets = dataset[idx]
    images = image_tensor.unsqueeze(0)
    class_names = list(CLASS_NAME.values())
    visualize_batch(images, [targets], class_names)
