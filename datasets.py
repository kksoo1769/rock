# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform


def build_dataset(is_train, is_test, args):
    transform = build_transform(is_train, is_test, args)

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

    if is_test:
        assert args.data_set == 'image_folder'
        dataset = TestDataset(args.test_data_path, transform=transform)
        nb_classes = args.nb_classes
    else:
        if args.data_set == 'CIFAR':
            dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
            nb_classes = 100
        elif args.data_set == 'IMNET':
            print("reading from datapath", args.data_path)
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
            nb_classes = 1000
        elif args.data_set == "image_folder":
            root = args.data_path if is_train else args.val_data_path
            dataset = datasets.ImageFolder(root, transform=transform)
            nb_classes = args.nb_classes
            assert len(dataset.class_to_idx) == nb_classes
        else:
            raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, is_test, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        # transform = create_transform(
        #     input_size=args.input_size,
        #     scale=args.scale,
        #     hflip=.5,
        #     vflip=.5,
        #     is_training=True,
        #     color_jitter=args.color_jitter,
        #     auto_augment=args.aa,
        #     interpolation=args.train_interpolation,
        #     re_prob=args.reprob,
        #     re_mode=args.remode,
        #     re_count=args.recount,
        #     mean=mean,
        #     std=std,
        # )
        # if not resize_im:
        #     transform.transforms[0] = transforms.RandomCrop(
        #         args.input_size, padding=4)
        # return transform
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=args.scale, interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=args.color_jitter, contrast=args.color_jitter),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if not is_test:
                if args.crop_pct is None:
                    args.crop_pct = 224 / 256
                size = int(args.input_size / args.crop_pct)
                t.append(
                    # to maintain same ratio w.r.t. 224 images
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
                )
                t.append(transforms.CenterCrop(args.input_size))
            else:
                t.append(transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.samples = []
        self.transform = transform

        for path, _, filenames in sorted(os.walk(root)):
            for filename in sorted(filenames):
                if filename.lower().endswith('jpg'):
                    self.samples.append(os.path.join(path, filename))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        image = datasets.folder.default_loader(path)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, path