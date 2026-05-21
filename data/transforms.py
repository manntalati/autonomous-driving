import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
NATIVE_H, NATIVE_W = 900, 1600
INPUT_H, INPUT_W = 448, 800


def get_train_transforms(input_h=INPUT_H, input_w=INPUT_W):
    """
    Augmented transform pipeline for training: resize, flip, shift/scale/rotate, crop, color jitter, blur, noise, normalize, to tensor.
    Bbox-aware: boxes are clipped and filtered by min_visibility=0.3.
    Args: input_h/input_w — target image resolution (default 448×800).
    Returns: albumentations Compose pipeline.
    """
    return A.Compose(
        [
            A.Resize(input_h, input_w),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=5,
                border_mode=0,
                p=0.3,
            ),
            A.RandomCrop(
                height=int(input_h * 0.9),
                width=int(input_w * 0.9),
                p=0.3,
            ),
            A.Resize(input_h, input_w),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.3,
            clip=True,
        ),
    )


def get_val_transforms(input_h=INPUT_H, input_w=INPUT_W):
    """
    Minimal transform pipeline for validation: resize, normalize, to tensor. No augmentation.
    Bbox-aware: boxes are clipped and filtered by min_visibility=0.3.
    Args: input_h/input_w — target image resolution (default 448×800).
    Returns: albumentations Compose pipeline.
    """
    return A.Compose(
        [
            A.Resize(input_h, input_w),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.3,
            clip=True,
        ),
    )


def get_seg_train_transforms(input_h=INPUT_H, input_w=INPUT_W):
    """
    Segmentation training pipeline. Same image augs as detection, but with a mask target
    instead of bboxes. The mask is resampled with NEAREST so class IDs aren't blended.
    Args: input_h/input_w — target image resolution (default 448×800).
    Returns: albumentations Compose pipeline accepting kwargs `image=`, `mask=`.
    Notes:
      - Use A.Resize(..., interpolation=cv2.INTER_LINEAR for image, mask_interpolation=cv2.INTER_NEAREST).
      - Skip GaussianBlur/GaussNoise on mask automatically (only image-level ops).
      - HorizontalFlip flips mask too — fine for road/lane symmetry.
      - No bbox_params.
    """
    return A.Compose(
        [
            A.Resize(
                input_h, input_w,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                mask_interpolation=cv2.INTER_NEAREST,
                fill_mask=0,
                p=0.3,
            ),
            A.RandomCrop(
                height=int(input_h * 0.9),
                width=int(input_w * 0.9),
                p=0.3,
            ),
            A.Resize(
                input_h, input_w,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


def get_seg_val_transforms(input_h=INPUT_H, input_w=INPUT_W):
    """
    Segmentation validation pipeline: resize + normalize only.
    Args: input_h/input_w — target image resolution.
    Returns: albumentations Compose with NEAREST resampling for the mask.
    """
    return A.Compose(
        [
            A.Resize(
                input_h, input_w,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )
