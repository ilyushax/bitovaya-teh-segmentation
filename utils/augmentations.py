import albumentations as A
from typing import List


def get_transforms() -> List:
    train_transforms = A.Compose(
        [
            A.Resize(512, 512),
            A.RandomCrop(384, 384),

            # Geometric invariant transforms
            # A.Rotate(limit=(-15,15),p=0.2),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.ElasticTransform(p=0.1),
            # A.Perspective(p=0.5),

            A.CLAHE(clip_limit=30.0, p=0.2),
            A.Sharpen(alpha=(0.3, 0.9), p=0.5),
            A.GaussNoise(var_limit=(5.0, 50.0), p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.7, contrast_limit=0.4, p=0.2
            ),
            A.HueSaturationValue(
                hue_shift_limit=70, sat_shift_limit=90,
                val_shift_limit=50, p=0.2
            ),
            A.Blur(blur_limit=(3, 39), p=0.5),
            A.Normalize(),  # Normalize the image
        ]
    )

    test_transforms = A.Compose(
        [
            A.Resize(384, 384),
            A.Normalize(),
        ]
    )

    return train_transforms, test_transforms
