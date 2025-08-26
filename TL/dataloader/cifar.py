# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: cifar.py
@time: 2022/4/19 11:19
"""

import os
import random
from PIL import Image
import torch
from typing import Any, Callable, Optional, Tuple

from torchvision import datasets, transforms
from torch.utils.data import Dataset

# --- CORRECTED Imports for Integrated Data Augmentation ---
from Augment.transforms_factory import create_transform
from Augment.autoaugment import SNNAugmentWide
from dataloader.dataloader_utils import DataLoaderX


class ExpandDVSChannel(object):
    """
    Expands a 2-channel DVS tensor to 3 channels.
    """
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.size(0) == 2:
            third_channel = tensor[0, :, :] + tensor[1, :, :]
            return torch.cat([tensor, third_channel.unsqueeze(0)], dim=0)
        return tensor


# your own data dir
USER_NAME = 'tr'
DIR = {'CIFAR10': f'/home/user/{USER_NAME}/Dataset/Event_Camera_Datasets/CIFAR10/cifar10',
       'CIFAR10DVS': f'/home/user/{USER_NAME}/Dataset/Event_Camera_Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.9_mat',
       }


class DVSAlignedTLCIFAR10(Dataset):
    """
    A robustly designed CIFAR10 Dataset for Transfer Learning.
    """
    def __init__(self, root: str, dvs_root: str, train: bool = True,
                 transform: Optional[Callable] = None, dvs_transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False, T: int = 4):
        super().__init__()
        self.train = train
        self.transform = transform
        self.dvs_transform = dvs_transform
        self.target_transform = target_transform
        self.imgx = transforms.ToPILImage()
        self.T = T

        dvs_data_path = os.path.join(dvs_root, 'train' if self.train else 'test')
        dvs_files_by_class = [[] for _ in range(10)]
        dvs_class_list = sorted(os.listdir(dvs_data_path))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(dvs_class_list)}
        for dvs_class in dvs_class_list:
            class_idx = class_to_idx[dvs_class]
            class_path = os.path.join(dvs_data_path, dvs_class)
            for file_name in sorted(os.listdir(class_path)):
                dvs_files_by_class[class_idx].append(os.path.join(class_path, file_name))

        rgb_dataset = datasets.CIFAR10(root=root, train=self.train, download=download)
        rgb_data_by_class = [[] for _ in range(10)]
        for img, label in zip(rgb_dataset.data, rgb_dataset.targets):
            rgb_data_by_class[label].append(img)

        self.paired_samples = []
        for class_idx in range(10):
            dvs_samples = dvs_files_by_class[class_idx]
            rgb_samples = rgb_data_by_class[class_idx]
            num_dvs_samples = len(dvs_samples)
            for i in range(num_dvs_samples):
                dvs_file = dvs_samples[i]
                rgb_sample = rgb_samples[i % len(rgb_samples)]
                self.paired_samples.append(((rgb_sample, dvs_file), class_idx))
        
        if self.train:
            random.shuffle(self.paired_samples)

    def __len__(self) -> int:
        return len(self.paired_samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        (rgb_data, dvs_file), target = self.paired_samples[index]

        rgb_img = Image.fromarray(rgb_data)
        if self.transform:
            rgb_img = self.transform(rgb_img)

        dvs_img = torch.load(dvs_file)
        native_t = dvs_img.size(0)
        if native_t > self.T:
            indices = torch.randperm(native_t)[:self.T]
            indices = torch.sort(indices).values
            dvs_img = dvs_img[indices]
        elif native_t < self.T:
            repeats = (self.T + native_t - 1) // native_t
            dvs_img = dvs_img.repeat(repeats, 1, 1, 1)[:self.T]

        if self.dvs_transform:
            dvs_img = self.dvs_trans(dvs_img)

        if self.target_transform:
            target = self.target_transform(target)

        return (rgb_img, dvs_img), target

    def dvs_trans(self, dvs_img):
        transformed_dvs_img = []
        for t in range(dvs_img.size(0)):
            data = self.imgx(dvs_img[t, ...])
            transformed_dvs_img.append(self.dvs_transform(data))
        return torch.stack(transformed_dvs_img, dim=0)


def get_dvs_aligned_tl_cifar10(batch_size, train_set_ratio=1.0, dvs_train_set_ratio=1.0, T=4):
    """
    Gets train/test loaders using the DVSAlignedTLCIFAR10 dataset with integrated, advanced data augmentation.
    """
    # --- 1. Define Advanced RGB Augmentation --- 
    rgb_trans_train = create_transform(
        input_size=128,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
    )

    # --- 2. Define DVS Augmentation --- 
    dvs_trans_train = transforms.Compose([
        transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        ExpandDVSChannel(),
        SNNAugmentWide()
    ])

    # --- 3. Define Validation/Test Transforms (simpler) ---
    rgb_trans_val = create_transform(input_size=128, is_training=False)
    dvs_trans_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        ExpandDVSChannel(),
    ])

    # --- 4. Create Datasets ---
    train_dataset = DVSAlignedTLCIFAR10(DIR['CIFAR10'], DIR['CIFAR10DVS'], train=True,
                                        transform=rgb_trans_train, dvs_transform=dvs_trans_train, 
                                        download=True, T=T)

    val_dataset = DVSAlignedTLCIFAR10(DIR['CIFAR10'], DIR['CIFAR10DVS'], train=False,
                                      transform=rgb_trans_val, dvs_transform=dvs_trans_val, 
                                      download=True, T=T)

    # --- 5. Create DataLoaders ---
    train_loader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    val_loader = DataLoaderX(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

    return train_loader, val_loader