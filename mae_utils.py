from __future__ import print_function, division
from tifffile import imread
import albumentations as A
import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import rasterio


def img_loader(path):
    return imread(path)


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
    def __init__(self, input_size: int):
        self.input_size = input_size

        self.transforms = A.Compose([
            A.RandomCrop(height=self.input_size,
                         width=self.input_size,
                         always_apply=True),
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
    def __init__(self, root_dir, file_list, nbands, augmentations, blocksize=224):
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


import torch
from torch.utils.data import IterableDataset
from osgeo import gdal
import numpy as np
import webdataset as wds
from typing import Any, Callable, List, Optional, Set, Tuple


typemapping_gdal_to_numpy = {
  1: "uint8",
  2: "uint16",
  3: "int16",
  4: "uint32",
  5: "int32",
  6: "float32",
  7: "float64",
  10: "complex64",
  11: "complex128",
}


class GeoWebDS(IterableDataset):
    def __init__(
            self,
            *,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        # self.batchsize = 32
        self.cropsize = 320
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform
        num_shards = 32
        imgs_per_shard = 256
        num_nodes = 1
        num_workers = 4
        # self.num_patches = num_shards * imgs_per_shard * (2240 // self.cropsize)**2
        self.num_patches = 1000000000000  # set it to sth really high for now, so that the generator doesnt get exhausted during trainng
        self.dataset = wds.DataPipeline(
                                        # wds.SimpleShardList(root),
                                        wds.ResampledShards(root),
                                        wds.shuffle(8),
                                        wds.split_by_node,
                                        wds.split_by_worker,
                                        wds.tarfile_to_samples(),
                                        wds.to_tuple("tif"),
                                        wds.map(GeoWebDS.preprocess),
                                        self.slicer,
                                        wds.shuffle(256),
                                        wds.map(self.transform),
                                        wds.map(GeoWebDS.fake_target)
                                    ).with_length(self.num_patches)

    @staticmethod
    def read_geotif_from_bytestream(data: bytes) -> np.ndarray:
        gdal.FileFromMemBuffer("/vsimem/tmp", data)
        ds = gdal.Open("/vsimem/tmp")
        bands = ds.RasterCount
        ys = ds.RasterYSize
        xs = ds.RasterXSize
        # dtype = typemapping_gdal_to_numpy[ds.GetRasterBand(1).DataType]
        arr = np.empty((bands, ys, xs), dtype="float32")  # CHW
        for b in range(1, bands + 1):
            band = ds.GetRasterBand(b)
            arr[b - 1, :, :] = band.ReadAsArray()
        return torch.from_numpy(arr) / 255

    @staticmethod
    def preprocess(sample):
        return GeoWebDS.read_geotif_from_bytestream(sample[0])

    @staticmethod
    def slice_image(samples, tilesize: int):
        for img in samples:
            for y in range(0, img.shape[1], tilesize):
                for x in range(0, img.shape[2], tilesize):
                    yield img[:, y:y + tilesize, x:x + tilesize]  # CHW

    def slicer(self, img):
        return GeoWebDS.slice_image(img, self.cropsize)

    @staticmethod
    def fake_target(x):
        return x, 0

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.num_patches