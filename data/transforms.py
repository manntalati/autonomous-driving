import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
NATIVE_H, NATIVE_W = 900, 1600
INPUT_H, INPUT_W = 448, 800


def get_train_transforms(input_h=INPUT_H, input_w=INPUT_W):
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
