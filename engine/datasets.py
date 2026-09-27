# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import datasets, transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform


def build_dataset(is_train, args, folder_name=None):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR-100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'CIFAR-10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10

    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000

    elif args.data_set == 'EVO_IMNET':
        root = os.path.join(args.data_path, folder_name)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000

    elif args.data_set == 'EVO_CIFAR-100':
        root = os.path.join(args.data_path, folder_name)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 100

    elif args.data_set == 'EVO_CIFAR-10':
        root = os.path.join(args.data_path, folder_name)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 10
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32

    mean = IMAGENET_INCEPTION_MEAN
    std = IMAGENET_INCEPTION_STD

    if args.imagenet_default_mean_and_std:
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

    if args.cifar_default_mean_and_std:
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,   # 颜色抖动
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,   # 随机擦除
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=3), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            crop_pct = 224 / 256
            size = int(args.input_size / crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=3),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
