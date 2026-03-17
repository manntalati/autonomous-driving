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
