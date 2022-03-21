from typing import List
from pathlib import Path

from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread as mask_imread
from catalyst import utils
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import cv2


class SegmentationDataset(Dataset):
    def __init__(
            self,
            images: List[Path],
            masks: List[Path] = None,
            transforms=None
    ) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image = utils.imread(image_path)
        # print(image.shape)

        result = {"image": image}

        if self.masks is not None:
            mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
            result["mask"] = (mask/255)[...,None]

        if self.transforms is not None:
            result = self.transforms(**result)

        result["filename"] = image_path.name

        return result


def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]


def hard_transforms():
    result = [
        albu.RandomRotate90(),
        albu.Cutout(),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        albu.GridDistortion(p=0.3),
        albu.HueSaturationValue(p=0.3)
    ]

    return result


def resize_transforms(image_size=224):
    BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose([
        albu.SmallestMaxSize(pre_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

    ])

    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    # random_crop_big = albu.Compose([
    #     albu.LongestMaxSize(pre_size, p=1),
    #     albu.RandomCrop(
    #         image_size, image_size, p=1
    #     )

    # ])

    # Converts the image to a square of size image_size x image_size
    result = [
        albu.OneOf([
            random_crop,
            rescale,
        ], p=1)
    ]

    return result

def mask_trans(x, **kwargs):
    return x.transpose(2, 0, 1)

def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Lambda(mask=mask_trans), albu.Normalize(), ToTensorV2()]


def compose(transforms_to_compose):
    # combine all augmentations into single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result