import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

SCALE = 1.1
__all__ = ["get_train_transforms", "get_test_transforms"]

def get_train_transforms(img_size: int):
    train_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=int(img_size * SCALE)),
            A.PadIfNeeded(
                min_height=int(img_size * SCALE),
                min_width=int(img_size * SCALE),
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.RandomCrop(width=img_size, height=img_size),
            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
            A.OneOf(
                [
                    A.ShiftScaleRotate(
                        rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                    ),
                    # A.IAAAffine(shear=15, p=0.5, mode="constant"),
                    A.GaussNoise(),
                ],
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Blur(p=0.1),
            A.CLAHE(p=0.1),
            A.Posterize(p=0.1),
            A.ToGray(p=0.1),
            A.ChannelShuffle(p=0.05),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[], ),
    )

    return train_transforms


def get_test_transforms(img_size: int):
    test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),  # keep the ratio & clip hw length
            A.PadIfNeeded(  # pad image if hw < min_height, min_width
                min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
    )

    return test_transforms
