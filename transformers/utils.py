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
    GridDistortion,
    Rotate,
    RandomScale,
    ShiftScaleRotate,
    ElasticTransform,
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


class ImageTransformer:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, original_image):
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

        augmented = self.augmentation_pipeline(
            image=original_image
        )
        image = augmented["image"]
        return image

    def __str__(self):
        string = str(self.height) + "x" + str(self.width) + \
            " | " + "RGB-Norm" + " | " + "Rand-Rotate90"
        return string


class DefaultTransformer:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, original_image):
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

    def __str__(self):
        string = str(self.height) + "x" + str(self.width) + " | " + "RGB-Norm"
        return string


class PolicyTransformer:
    def __init__(self, height, width, policy=None):
        self.height = height
        self.width = width
        self.policy = policy

        self.base_augmentation_pipeline = Compose([
            Resize(self.height, self.width, p=1.0),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
            ToTensor()
        ])

        self.aug_list = []
        if isinstance(self.policy, list):
            for sub_policy in policy:
                op_1, params_1 = sub_policy[0]
                op_2, params_2 = sub_policy[1]
                aug = Compose([
                    Resize(512, 512, p=1.0),
                    op_obj[op_1](**params_1),
                    op_obj[op_2](**params_2),
                ])
                self.aug_list.append(aug)
        else:
            if path.exists('transformers/best_policy.json'):
                with open('transformers/best_policy.json') as f:
                    policy_array = json.load(f)
                    sub_policy_pool = chain.from_iterable(policy_array)

                    for sub_policy in sub_policy_pool:
                        op_1, params_1 = sub_policy[0]
                        op_2, params_2 = sub_policy[1]
                        aug = Compose([
                            Resize(512, 512, p=1.0),
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
        if isinstance(self.policy, list):
            if len(self.aug_list) > 0: 
                string = str(self.height) + "x" + str(self.width) + \
                " | " + "Policy (Search Mode)"
            else:
                string = str(self.height) + "x" + str(self.width) + " | " + "RGB-Norm" 
        else:
            string = str(self.height) + "x" + \
                str(self.width) + " | " + "Policy"
        return string
