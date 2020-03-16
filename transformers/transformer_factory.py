from albumentations import (
    Resize,
    Normalize,
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    RandomBrightness,
    OneOf,
    Compose)
from albumentations.pytorch import ToTensor


class Transformer:
    def __init__(self, height=2048, width=1365):
        self.augmentation_pipeline = Compose(
            [
                Resize(height, width, p=1.0),
                OneOf(
                    [
                        HorizontalFlip(),
                        VerticalFlip()
                    ],
                    p=0.5
                ),
                RandomRotate90(p=0.5),
                ToTensor()
            ]
        )

    def get_augmented(self, original_image):
        augmented = self.augmentation_pipeline()(
            {
                "image": original_image
            }
        )
        image = augmented["image"]
        return image
