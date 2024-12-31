from torchvision import transforms
from enum import Enum

import pandas as pd

import PIL
import torch
import os

_CLASSNAMES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class VisADataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for VisA.
    """

    def __init__(
            self,
            source,
            classname='leather',
            resize=288,
            imagesize=288,
            split=DatasetSplit.TRAIN,
            **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.resize = resize
        self.imgsize = imagesize
        self.imagesize = (3, self.imgsize, self.imgsize)
        self.classname = classname

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split != DatasetSplit.TRAIN and mask_path is not None:
            mask_gt = PIL.Image.open(mask_path).convert('F')
            mask_gt = self.transform_mask(mask_gt)
            mask_gt = torch.where(mask_gt > 0, 1, 0).to(torch.float32)
        else:
            mask_gt = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask_gt": mask_gt,
            "is_anomaly": int(anomaly != "normal"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        csv_path = os.path.join(self.source, "split_csv/1cls.csv")
        df = pd.read_csv(csv_path)
        flag = 'train' if self.split == DatasetSplit.TRAIN else 'test'

        anomaly_types = ['normal'] if flag == 'train' else ['normal', 'anomaly']
        imgpaths_per_class[self.classname] = {}
        maskpaths_per_class[self.classname] = {}

        for anomaly in anomaly_types:
            relative_img_path = df.loc[
                (df['object'] == self.classname) & (df['split'] == flag) & (df['label'] == anomaly), 'image'].values.tolist()
            absolute_img_path = [os.path.join(self.source, x) for x in relative_img_path]
            imgpaths_per_class[self.classname][anomaly] = absolute_img_path

            if flag == 'test' and anomaly != 'normal':
                relative_msk_path = df.loc[(df['object'] == self.classname) & (df['label'] == anomaly), 'mask'].values.tolist()
                absolute_msk_path = [os.path.join(self.source, x) for x in relative_msk_path]
                maskpaths_per_class[self.classname][anomaly] = absolute_msk_path
            else:
                maskpaths_per_class[self.classname][anomaly] = None

        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "normal":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
