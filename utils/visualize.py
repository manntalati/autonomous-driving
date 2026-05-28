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
