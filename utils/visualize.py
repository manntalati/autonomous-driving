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
    image_tensor, targets = dataset[idx]
    images = image_tensor.unsqueeze(0)
    class_names = list(CLASS_NAME.values())
    visualize_batch(images, [targets], class_names)