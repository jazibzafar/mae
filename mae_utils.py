from __future__ import print_function, division
from tifffile import imread
import albumentations as A
import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import rasterio


def read_random_block(filename, nbands, blocksize, dtype=np.uint8):
    """Reads a random block from a tiled Tiff file.

    Args:
        filename: The file name
        nbands: Number of image bands
        blocksize: Block size in pixels

    Returns:
        Numpy array containing the data
    """
    arr = np.zeros((nbands, blocksize, blocksize), dtype=dtype)
    with rasterio.open(filename) as img:
        blocks = list(img.block_windows(0))
        # window = np.random.choice(blocks)[1] - throws an error
        len_blocks = len(blocks)
        window_idx = np.random.choice(range(0, len_blocks))
        window = blocks[window_idx][1]
        for band in range(1, nbands+1):
            arr[band-1] = img.read(band, window=window)
    return arr.transpose((1, 2, 0))


class MAETransform:
    def __init__(self, input_size: int, random_crop: bool = False):
        self.input_size = input_size
        self.random_crop = random_crop

        if self.random_crop:
            self.transforms = A.RandomCrop(height= self.input_size, width=self.input_size, always_apply=True)

        self.transforms = A.Compose([
            # A.RandomCrop(height=self.input_size, # commented out because using SSLDataset not GEOTIFF4
            #              width=self.input_size,
            #              always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=(0.2, 0.3),
                                       contrast_limit=(0.2, 0.3),
                                       p=0.2),
            A.RandomGamma(gamma_limit=(100, 140), p=0.2),
            A.RandomToneCurve(scale=0.1, p=0.2)
        ])

    def __call__(self, image):
        # Make sure image is a np array
        if type(image) == 'torch.Tensor':
            image = image.numpy()

        crop = ToTensor()(self.transforms(image=image)['image'])
        return crop


class MAETransformDumb:
    def __init__(self, input_size: int):
        self.input_size = input_size

        self.transforms = A.CenterCrop(height=self.input_size,
                                       width=self.input_size,
                                       always_apply=True)

    def __call__(self, image):
        # Make sure image is a np array
        if type(image) == 'torch.Tensor':
            image = image.numpy()

        crop = ToTensor()(self.transforms(image=image)['image'])
        return crop


class GEOTIFF4(Dataset):
    def __init__(self, file_list, root_dir, transform=None):
        with open(file_list) as f:
            self.file_list = f.read().splitlines()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_name = os.path.join(self.root_dir, self.file_list[item])
        image = imread(img_name)
        if self.transform:
            image = self.transform(image=image)
        # image = ToTensor()(image['image'])
        return image


class SSLDataset:
    def __init__(self, root_dir, file_list, nbands, blocksize, augmentations):
        """Dataset for self-supervised learning on large sets of tiled Tiff files.

        Args:
            files: List of input files
            nbands: Number of image bands
            blocksize: Block size of the Tiff tiles
            augmentations: TODO
        """
        super().__init__()
        with open(file_list) as f:
            self.file_list = f.read().splitlines()
        self.root_dir = root_dir
        self.augmentations = augmentations
        self.blocksize = blocksize
        self.nbands = nbands

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file = os.path.join(self.root_dir, self.file_list[item])
        block = read_random_block(file, self.nbands, self.blocksize)
        if self.augmentations:
            block = self.augmentations(image=block)
        return block
