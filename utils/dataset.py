#!/usr/bin/env python3

import os
from random import random
from typing import Callable, Dict, Union

import torch
from torch import Tensor
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .utils import class2one_hot, sset


def make_dataset(root, subset):
    assert subset in ['train', 'val', 'test']
    items = []

    img_path = os.path.join(root, subset, 'img')
    mask_path = os.path.join(root, subset, 'gt')

    images = os.listdir(img_path)
    labels = os.listdir(mask_path)

    images.sort()
    labels.sort()

    for it_im, it_gt in zip(images, labels):
        item = (os.path.join(img_path, it_im), os.path.join(mask_path, it_gt))
        items.append(item)

    return items


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, transform=None,
                 mask_transform=None, augment=False, equalize=False):
        self.root_dir: str = root_dir
        self.transform: Callable = transform
        self.mask_transform: Callable = mask_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.imgs = make_dataset(root_dir, subset)

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random() > 0.5:
            angle = random() * 90 - 45
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask

    def __getitem__(self, index) -> Dict[str, Union[Tensor, int, str]]:
        img_path, mask_path = self.imgs[index]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augmentation:
            img, mask = self.augment(img, mask)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)

        _, W, H = img.shape
        assert mask.shape == (2, W, H)

        return {"img": img,
                "full_mask": mask,
                "weak_mask": mask,
                "path": img_path,
                "true_size": torch.tensor([0, 7845], dtype=torch.float32),
                "bounds": torch.tensor([[-1, -1], [7845, 7845]], dtype=torch.float32)}
