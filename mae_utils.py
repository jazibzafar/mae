from __future__ import print_function, division
from tifffile import imread
import albumentations as A
import torch
import os
from torch.utils.data import Dataset, IterableDataset
from torchvision.transforms import ToTensor
import numpy as np
import rasterio
import webdataset as wds
from osgeo import gdal
import matplotlib.pyplot as plt


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


class GeoWebDataset(IterableDataset):
    def __init__(self,
                 *,
                 root,
                 n_bands,
                 augmentations,
                 num_workers,
                 num_nodes=1,
                 num_shards=100,
                 imgs_per_shard=250):
        self.root = root
        self.n_bands = n_bands
        self.augmentations = augmentations
        self.num_workers = num_workers
        self.num_nodes = num_nodes
        self.num_shards = num_shards
        self.imgs_per_shard = imgs_per_shard
        self.cropsize = 224
        #
        self.num_patches = 1000000000000  # set it to sth really high for now, so that the generator doesnt get exhausted during trainng
        self.dataset = wds.DataPipeline(wds.ResampledShards(self.root),
                                        wds.shuffle(8),
                                        wds.split_by_node,
                                        wds.tarfile_to_samples(),
                                        wds.to_tuple("tif"),
                                        wds.map(GeoWebDataset.preprocess),
                                        self.slicer,
                                        wds.shuffle(100),  # buffer of size 100
                                        wds.map(self.augmentations),
                                        wds.split_by_worker,
                                        ).with_length(self.num_patches)

    @staticmethod
    def read_geotif_from_bytestream(data: bytes) -> np.ndarray:
        gdal.FileFromMemBuffer("/vsimem/tmp", data)
        ds = gdal.Open("/vsimem/tmp")
        bands = ds.RasterCount
        ys = ds.RasterYSize
        xs = ds.RasterXSize
        # arr = np.empty((bands, ys, xs), dtype="float32")  # CHW
        # for b in range(1, bands + 1):
        #     band = ds.GetRasterBand(b)
        #     arr[b - 1, :, :] = band.ReadAsArray()
        # return torch.from_numpy(arr) / 255
        arr = np.empty((ys, xs, bands), dtype="uint8")  # HWC
        for b in range(1, bands + 1):
            band = ds.GetRasterBand(b)
            arr[:, :, b - 1] = band.ReadAsArray()
        return arr

    @staticmethod
    def preprocess(sample):
        return GeoWebDataset.read_geotif_from_bytestream(sample[0])

    @staticmethod
    def slice_image(samples, tilesize: int):
        for img in samples:
            for y in range(0, img.shape[1], tilesize):
                for x in range(0, img.shape[2], tilesize):
                    yield img[:, y:y + tilesize, x:x + tilesize]  # CHW

    def slicer(self, img):
        return GeoWebDataset.slice_image(img, self.cropsize)

    def __iter__(self):
        return iter(self.dataset)

    # def __len__(self):
    #     return self.imgs_per_shard * self.num_shards


class FakeDataset(Dataset):
    def __init__(self, shape, length):
        self.data = np.zeros(shape, dtype=np.float32)
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.data


# Some random functions useful for visualization
def to_rgb(img_in, mode='CHW'):
    if mode == 'CHW':
        return img_in[:3, :, :]
    elif mode == 'HWC':
        return img_in[:, :, :3]
    else:
        print("mode = CHW or HWC")


def change_mode(img_in, mode_in):  # hacky; implemented only for torch tensors.
    if mode_in == 'CHW':
        return torch.permute(img_in, (1, 2, 0))  # HWC
    elif mode_in == 'HWC':
        return torch.permute(img_in, (2, 0, 1))  # CHW


def show_image(image, title=''):
    plt.imshow(to_rgb(image, mode='HWC'))
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

