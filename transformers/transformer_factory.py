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


class TransformerFactory:
    def __init__(self, height=2048, width=1365, pipe_type="default"):
        self.height = height
        self.width = width
        self.pipe_type = pipe_type

    def get_augmented(self, original_image, ):
        self.augmentation_pipeline = None

        if(self.pipe_type == "image"):
            self.augmentation_pipeline = Compose(
                [
                    Resize(self.height, self.width, p=1.0),
                    Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
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
        else:
            self.augmentation_pipeline = Compose(
                [
                    Resize(self.height, self.width, p=1.0),
                    Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
                    ToTensor()
                ]
            )

        augmented = self.augmentation_pipeline(
            image=original_image
        )
        image = augmented["image"]
        return image
