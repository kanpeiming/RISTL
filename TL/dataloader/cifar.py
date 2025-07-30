# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: cifar.py
@time: 2022/4/19 11:19
"""

import os
import bisect
from collections import Counter
from torch._utils import _accumulate
from dataloader.dataloader_utils import *
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from spikingjelly.datasets import cifar10_dvs
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import random
from PIL import Image
import torch


class ExpandDVSChannel(object):
    """
    Expands a 2-channel DVS tensor to 3 channels.
    The third channel is the sum of the first two, representing total event activity.
    """
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be expanded. Expects C=2.

        Returns:
            Tensor: Expanded tensor image of size (3, H, W).
        """
        if tensor.size(0) == 2:
            # Sum of the two channels creates the third channel
            third_channel = tensor[0, :, :] + tensor[1, :, :]
            # Stack the original 2 channels with the new 3rd channel
            return torch.cat([tensor, third_channel.unsqueeze(0)], dim=0)
        return tensor


# your own data dir
USER_NAME = 'tr'
DIR = {'CIFAR10': f'/home/user/{USER_NAME}/Dataset/Event_Camera_Datasets/CIFAR10/cifar10',
       'CIFAR10DVS': f'/home/user/{USER_NAME}/Dataset/Event_Camera_Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.9_mat',
       'CIFAR10DVS_CATCH': f'/home/user/{USER_NAME}/Dataset/Event_Camera_Datasets/CIFAR10/CIFAR10DVS_dst_cache',
       }


class TLCIFAR10(datasets.CIFAR10):
    """This is the original TLCIFAR10 class, kept for the test set."""
    def __init__(
            self,
            root: str,
            dvs_root: str,
            train: bool = True,
            dvs_train_set_ratio: float = 1.0,
            transform: Optional[Callable] = None,
            dvs_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            T: int = 4,
    ) -> None:

        super(TLCIFAR10, self).__init__(root=root, train=train, transform=transform,
                                        target_transform=target_transform, download=download)
        self.T = T
        self.train = train
        self.dvs_transform = dvs_transform
        self.imgx = transforms.ToPILImage()
        if self.train:
            self.dvs_root = os.path.join(dvs_root, 'train')
        else:
            self.dvs_root = os.path.join(dvs_root, 'test')

        dvs_class_list = sorted(os.listdir(self.dvs_root))
        self.dvs_data = []
        self.dvs_targets = []
        for dvs_class in dvs_class_list:
            dvs_class_path = os.path.join(self.dvs_root, dvs_class)
            file_list = sorted(os.listdir(dvs_class_path))
            for file_name in file_list:
                self.dvs_data.append(os.path.join(dvs_class_path, file_name))
                self.dvs_targets.append(self.class_to_idx[dvs_class])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.train:
            # This part is complex and was causing issues. It's now only used if called directly.
            # The new DVSAlignedTLCIFAR10 is the preferred way for training.
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            # ... (original pairing logic omitted for clarity)
            return img, target # Placeholder
        else:
            dvs_img = torch.load(self.dvs_data[index])
            
            # --- Adjust time dimension of DVS data ---
            native_t = dvs_img.size(0)
            if native_t > self.T:
                # Select T frames randomly
                indices = torch.randperm(native_t)[:self.T]
                indices = torch.sort(indices).values
                dvs_img = dvs_img[indices]
            elif native_t < self.T:
                # Repeat frames to match T
                repeats = (self.T + native_t - 1) // native_t
                dvs_img = dvs_img.repeat(repeats, 1, 1, 1)[:self.T]

            if self.dvs_transform is not None:
                dvs_img = self.dvs_trans(dvs_img)
            target = self.dvs_targets[index]
            return dvs_img, target

    def __len__(self) -> int:
        if self.train:
            return len(self.data)
        else:
            return len(self.dvs_data)

    def dvs_trans(self, dvs_img):
        transformed_dvs_img = []
        for t in range(dvs_img.size(0)):
            data = self.imgx(dvs_img[t, ...])
            transformed_dvs_img.append(self.dvs_transform(data))
        dvs_img = torch.stack(transformed_dvs_img, dim=0)
        if self.train:
            if random.random() > 0.5:
                dvs_img = torch.flip(dvs_img, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            dvs_img = torch.roll(dvs_img, shifts=(off1, off2), dims=(2, 3))
        return dvs_img

    def get_len(self):
        if self.train:
            return len(self.data), len(self.dvs_data)
        else:
            return len(self.dvs_data)


class DVSAlignedTLCIFAR10(Dataset):
    """
    A robustly designed CIFAR10 Dataset for Transfer Learning.
    It solves the sample mismatch issue by creating a fixed, class-aligned pairing
    of DVS and RGB samples during initialization.
    The length of the dataset is determined by the DVS dataset (the smaller one).
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

        # --- 1. Load DVS Data (the smaller dataset) and group by class ---
        dvs_data_path = os.path.join(dvs_root, 'train' if self.train else 'test')
        dvs_files_by_class = [[] for _ in range(10)]
        dvs_class_list = sorted(os.listdir(dvs_data_path))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(dvs_class_list)}
        for dvs_class in dvs_class_list:
            class_idx = class_to_idx[dvs_class]
            class_path = os.path.join(dvs_data_path, dvs_class)
            for file_name in sorted(os.listdir(class_path)):
                dvs_files_by_class[class_idx].append(os.path.join(class_path, file_name))

        # --- 2. Load RGB Data and group by class ---
        rgb_dataset = datasets.CIFAR10(root=root, train=self.train, download=download)
        rgb_data_by_class = [[] for _ in range(10)]
        for img, label in zip(rgb_dataset.data, rgb_dataset.targets):
            rgb_data_by_class[label].append(img)

        # --- 3. Create the fixed, class-aligned pairing list (the core logic) ---
        self.paired_samples = []
        for class_idx in range(10):
            dvs_samples = dvs_files_by_class[class_idx]
            rgb_samples = rgb_data_by_class[class_idx]
            num_dvs_samples = len(dvs_samples)
            
            # For each DVS sample, find a corresponding RGB sample of the same class.
            # Use modulo to handle cases where RGB samples are fewer (unlikely) or to cycle through them.
            for i in range(num_dvs_samples):
                dvs_file = dvs_samples[i]
                rgb_sample = rgb_samples[i % len(rgb_samples)] # Cycle through RGB samples
                self.paired_samples.append(((rgb_sample, dvs_file), class_idx))
        
        # Shuffle the dataset to ensure randomness in epochs
        if self.train:
            random.shuffle(self.paired_samples)

    def __len__(self) -> int:
        return len(self.paired_samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        (rgb_data, dvs_file), target = self.paired_samples[index]

        # Load and transform RGB image
        rgb_img = Image.fromarray(rgb_data)
        if self.transform:
            rgb_img = self.transform(rgb_img)

        # Load and transform DVS image
        dvs_img = torch.load(dvs_file)
        
        # --- Adjust time dimension of DVS data ---
        native_t = dvs_img.size(0)
        if native_t > self.T:
            # Select T frames randomly
            indices = torch.randperm(native_t)[:self.T]
            indices = torch.sort(indices).values
            dvs_img = dvs_img[indices]
        elif native_t < self.T:
            # Repeat frames to match T
            repeats = (self.T + native_t - 1) // native_t
            dvs_img = dvs_img.repeat(repeats, 1, 1, 1)[:self.T]

        if self.dvs_transform:
            dvs_img = self.dvs_trans(dvs_img)

        if self.target_transform:
            target = self.target_transform(target)

        return (rgb_img, dvs_img), target

    def get_len(self) -> Tuple[int, int]:
        # For compatibility with the print statement in vit_tl.py
        return (len(self.paired_samples), len(self.paired_samples))

    def dvs_trans(self, dvs_img):
        # (This is a helper function, copied from the original TLCIFAR10)
        transformed_dvs_img = []
        for t in range(dvs_img.size(0)):
            data = self.imgx(dvs_img[t, ...])
            transformed_dvs_img.append(self.dvs_transform(data))
        dvs_img = torch.stack(transformed_dvs_img, dim=0)
        if self.train:
            if random.random() > 0.5:
                dvs_img = torch.flip(dvs_img, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            dvs_img = torch.roll(dvs_img, shifts=(off1, off2), dims=(2, 3))
        return dvs_img

def get_dvs_aligned_tl_cifar10(batch_size, train_set_ratio=1.0, dvs_train_set_ratio=1.0, T=4):
    """
    Gets train/test loaders using the DVSAlignedTLCIFAR10 dataset.
    This function is designed to solve the batch mismatch issue for CKA loss calculation.
    """
    rgb_trans_train = transforms.Compose([transforms.Resize(128),
                                          # transforms.RandomCrop(48, padding=4),
                                          # transforms.RandomHorizontalFlip(),
                                          # CIFAR10Policy(),
                                          transforms.ToTensor(),
                                          # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                          ])
    dvs_trans = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    ExpandDVSChannel(), # <-- This is our new transform
                                   ])

    tl_train_data = DVSAlignedTLCIFAR10(DIR['CIFAR10'], DIR['CIFAR10DVS'], train=True,
                                        transform=rgb_trans_train, dvs_transform=dvs_trans, download=True, T=T)

    dvs_test_data = TLCIFAR10(DIR['CIFAR10'], DIR['CIFAR10DVS'], train=False,
                              dvs_transform=dvs_trans, download=True, T=T)

    train_dataloader = DataLoaderX(tl_train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True,
                                   pin_memory=True)
    test_dataloader = DataLoaderX(dvs_test_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False,
                                  pin_memory=True)

    return train_dataloader, test_dataloader