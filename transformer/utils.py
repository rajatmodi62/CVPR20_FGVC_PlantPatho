from albumentations import (
    Resize,
    Normalize,
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    Blur,
    MotionBlur,
    InvertImg,
    IAASharpen,
    GridDistortion,
    IAAPiecewiseAffine,
    Rotate,
    RandomScale,
    ShiftScaleRotate,
    ElasticTransform,
    IAAEmboss,
    OneOf,
    Compose)
from albumentations.pytorch import ToTensor
import random
import json
from os import path
from itertools import chain

op_obj = {
    'RandomContrast': RandomContrast,
    'RandomBrightness': RandomBrightness,
    'RandomGamma': RandomGamma,
    'Blur': Blur,
    'MotionBlur': MotionBlur,
    'InvertImg': InvertImg,
    'Rotate': Rotate,
    'ShiftScaleRotate': ShiftScaleRotate,
    'RandomScale': RandomScale,
    'GridDistortion': GridDistortion,
    'ElasticTransform': ElasticTransform
}


class DefaultTransformer:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, original_image):
        self.augmentation_pipeline = Compose(
            [
                Resize(self.height, self.width, always_apply=True),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    always_apply=True
                ),
                ToTensor()
            ]
        )

        augmented = self.augmentation_pipeline(
            image=original_image
        )
        image = augmented["image"]
        return image

    def __str__(self):
        string = str(self.height) + "x" + str(self.width) + " | " + "default"
        return string


class ImageTransformer:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, original_image):
        self.augmentation_pipeline = Compose(
            [
                Resize(650, 650, always_apply=True),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(rotate_limit=25.0, p=0.7),
                OneOf([IAAEmboss(p=1),
                       IAASharpen(p=1),
                       Blur(p=1)], p=0.5),
                IAAPiecewiseAffine(p=0.5),
                Resize(self.height, self.width, always_apply=True),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    always_apply=True
                ),
                ToTensor()
            ]
        )

        augmented = self.augmentation_pipeline(
            image=original_image
        )
        image = augmented["image"]
        return image

    def __str__(self):
        string = str(self.height) + "x" + str(self.width) + \
            " | " + "image"
        return string


class PolicyTransformer:
    def __init__(self, height, width, auto_aug_policy=None):
        self.height = height
        self.width = width
        self.auto_aug_policy = auto_aug_policy

        self.base_augmentation_pipeline = Compose([
            Resize(self.height, self.width, always_apply=True),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
                      always_apply=True),
            ToTensor()
        ])

        self.aug_list = []
        if isinstance(self.auto_aug_policy, list):
            for sub_policy in self.auto_aug_policy:
                op_1, params_1 = sub_policy[0]
                op_2, params_2 = sub_policy[1]
                aug = Compose([
                    Resize(512, 512, always_apply=True),
                    op_obj[op_1](**params_1),
                    op_obj[op_2](**params_2),
                ])
                self.aug_list.append(aug)
        else:
            if path.exists('transformer/best_policy.json'):
                with open('transformer/best_policy.json') as f:
                    policy_array = json.load(f)
                    sub_policy_pool = chain.from_iterable(policy_array)

                    for sub_policy in sub_policy_pool:
                        op_1, params_1 = sub_policy[0]
                        op_2, params_2 = sub_policy[1]
                        aug = Compose([
                            Resize(700, 700, always_apply=True),
                            op_obj[op_1](**params_1),
                            op_obj[op_2](**params_2),
                        ])
                        self.aug_list.append(aug)
            else:
                print("[ Policy JSON not found ]")
                exit()

    def __call__(self, original_image):
        modified_image = original_image

        if len(self.aug_list) > 0:
            aug = random.choice(self.aug_list)
            modified_image = aug(image=original_image)['image']

        modified_image = self.base_augmentation_pipeline(image=modified_image)[
            'image']

        return modified_image

    def __str__(self):
        string = None
        if isinstance(self.auto_aug_policy, list):
            if len(self.aug_list) > 0:
                string = str(self.height) + "x" + str(self.width) + \
                    " | " + "policy (Search Mode)"
            else:
                string = str(self.height) + "x" + \
                    str(self.width) + " | " + "default"
        else:
            string = str(self.height) + "x" + \
                str(self.width) + " | " + "policy"
        return string
