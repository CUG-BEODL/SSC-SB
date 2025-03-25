from torch.utils import data
from torchvision import transforms
import numpy as np
# from torchvision.utils import draw_segmentation_masks
import pandas as pd
import random
import glob
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from cugdt.Tiff import *
from glob import glob
from utils import *


def random_crop(dataset, target_size):
    """
    Randomly crops a subregion from the dataset.
    Ensures at least 50% of the cropped region contains valid data.
    """
    x_start = np.random.randint(0, int(target_size * 0.5))
    y_start = np.random.randint(0, int(target_size * 0.5))
    crop_size = np.random.randint(int(target_size * 0.5), int(target_size * 1))
    cropped_data = dataset[:, x_start:x_start + crop_size, y_start:y_start + crop_size]

    mask = np.mean(cropped_data, axis=0) != 0
    try:
        if (np.sum(mask) / (cropped_data.shape[1] * cropped_data.shape[2])) > 0.5:
            return resize_image(cropped_data, target_size)
        else:
            return dataset
    except Exception:
        return dataset


def add_gaussian_noise(dataset, std_dev=0.05):
    """
    Adds Gaussian noise to the dataset while preserving masked regions.
    """
    mask = np.mean(dataset, axis=0) == 0
    noise = np.random.normal(0, std_dev, dataset.shape)
    noisy_dataset = dataset + noise
    noisy_dataset[:, mask] = 0
    return noisy_dataset


class DataAugmentation:
    """
    Performs data augmentation including cropping, noise addition,
    flipping, and rotation.
    """

    def __init__(self, target_size):
        self.target_size = target_size

    def apply_transforms(self, dataset):
        """
        Applies random transformations to augment the dataset.
        """
        if random.random() < 0.5:
            dataset = random_crop(dataset, self.target_size)
        if random.random() < 0.5:
            dataset = add_gaussian_noise(dataset)
        if random.random() < 0.5:
            dataset = np.flip(dataset, axis=1)  # Horizontal flip
        if random.random() < 0.5:
            dataset = np.flip(dataset, axis=0)  # Vertical flip
        if random.random() < 0.5:
            dataset = np.rot90(dataset, k=1, axes=(1, 2))  # Rotate 90 degrees
        if random.random() < 0.5:
            dataset = np.rot90(dataset, k=2, axes=(1, 2))  # Rotate 180 degrees
        if random.random() < 0.5:
            dataset = np.rot90(dataset, k=3, axes=(1, 2))  # Rotate 270 degrees

        return dataset


class MaskDataset(data.Dataset):
    """
       Custom dataset class for handling developed and undeveloped street blocks.
       Includes support for time-series sampling and augmentation.
       """

    def __init__(self, Dset, UDset, len_ts, channels, batch_size, block_size, sample_size, type, normalize=True):
        super(MaskDataset, self).__init__()
        self.type = type
        self.Dset = Dset
        self.UDset = UDset
        self.len_ts = len_ts
        self.channels = channels
        self.normalize = normalize
        self.batch_size = batch_size
        self.block_size = block_size
        self.sample_size = sample_size

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset and applies data augmentation.
        """
        if index % 2 == 0:
            dataset = self.Dset[np.random.choice(range(0, self.Dset.shape[0]), 1)[0]]
            y = 0
        else:
            dataset = self.UDset[np.random.choice(range(0, self.UDset.shape[0]), 1)[0]]
            y = 1
        start = np.random.choice(range(0, self.len_ts - self.sample_size), 1)[0]
        dataset = dataset[start * self.channels:(start + self.sample_size) * self.channels]
        augmenter = DataAugmentation(self.block_size)
        dataset = augmenter.apply_transforms(dataset)
        return dataset.copy(), y

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.batch_size * 100


def load_data(args):
    """
       Loads the developed and undeveloped datasets and prepares DataLoaders.
       """
    developed_dataset = np.load(args.D_set)
    undeveloped_dataset = np.load(args.UD_set)

    # 维度校验
    assert developed_dataset.shape[2] == args.block_size
    assert developed_dataset.shape[3] == args.block_size
    assert developed_dataset.shape[1] == args.len_ts * args.channels
    assert undeveloped_dataset.shape[1] == args.len_ts * args.channels

    train_ds = MaskDataset(developed_dataset,
                           undeveloped_dataset,
                           args.len_ts,
                           args.channels,
                           args.batch_size,
                           args.block_size,
                           args.sample_size,
                           type='train')

    train_dl = data.DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return train_dl


if __name__ == '__main__':
    developed_dataset = np.load('dataset/Changsha_undeveloped_sampleSize_64.npy')
    print(developed_dataset.shape)
