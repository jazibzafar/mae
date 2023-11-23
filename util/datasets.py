# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from mae_utils import MAETransform, img_loader
from torch.utils.data import random_split


def build_dataset(args):
    # Replace below with my own MAE transform.
    # transform = build_transform(is_train, args)
    transform = MAETransform(args.input_size)
    # TODO: This below is a problem. There's no train/val divide in my
    # TODO: data. Rewrite this to read the data properly. [X] Done.
    # root = os.path.join(args.data_path, 'train' if is_train else 'val')
    # dataset = datasets.ImageFolder(root, transform=transform)
    root = args.data_path
    dataset = datasets.ImageFolder(root, transform=transform, loader=img_loader)
    train_dataset, val_dataset = random_split(dataset, [args.train_ratio, 1-args.train_ratio])
    print(dataset)

    return train_dataset, val_dataset

# TODO: Change the transforms for my data. [X] Use MAE Transform instead of this.
# I'm not using the function below at all.
def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
