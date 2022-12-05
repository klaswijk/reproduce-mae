import os
import json
import torch
import numpy as np
from collections import namedtuple
from pandas import read_csv
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image


DatasetInfo = namedtuple(
    "DatasetInfo", ["image_size", "n_classes", "multilabel"])
info = {
    "cifar10": DatasetInfo(image_size=32, n_classes=10, multilabel=False),
    "imagenette": DatasetInfo(image_size=160, n_classes=10, multilabel=False),
    "coco": DatasetInfo(image_size=96, n_classes=80, multilabel=True)
}


class ImageNetteDataset(Dataset):
    def __init__(self, path, transform=None, test=False):
        # Should have 10000 train and 3395 val ("test") images
        self.img_labels = read_csv(path + "noisy_imagenette.csv")
        if test:
            self.img_labels = self.img_labels[10000:]
        else:
            self.img_labels = self.img_labels[:10000]
        self.img_dir = path
        self.transform = transform
        targets = {
            "n01440764": 0,
            "n02979186": 1,
            "n03028079": 2,
            "n03417042": 3,
            "n03445777": 4,
            "n02102040": 5,
            "n03000684": 6,
            "n03394916": 7,
            "n03425413": 8,
            "n03888257": 9,
        }
        self.target_transform = transforms.Lambda(targets.get)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = image.convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CocoMultilabel(Dataset):

    def __init__(self, path, transform=None, test=False, version="2017"):
        datatype = "val" if test else "train"
        with open(f"{path}/coco/annotations/instances_{datatype}{version}.json", "r") as f:
            instances = json.load(f)

        self.transform = transform
        self.datapath = f"{path}/coco/{datatype}{version}"
        self.images = [image["file_name"] for image in instances["images"]]
        self.image_ids = [image["id"] for image in instances["images"]]
        self.label_names = [cat["name"] for cat in instances["categories"]]
        self.labels = {id: np.zeros(80, dtype=np.float32)
                       for id in self.image_ids}

        # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        ignore = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]

        for annotation in instances["annotations"]:
            cat_id = annotation["category_id"]
            cat_id -= np.searchsorted(ignore, cat_id)
            image_id = annotation["image_id"]
            self.labels[image_id][cat_id - 1] = 1

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = Image.open(f"{self.datapath}/{self.images[idx]}")
        image = image.convert('RGB')
        label = self.labels[self.image_ids[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transform(dataset_name, transform_type):
    transform_list = [
        transforms.RandomCrop(info[dataset_name].image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    if transform_type == "finetune":
        transform_list = [transforms.RandAugment()] + transform_list
    if dataset_name == "coco":
        size = int(info[dataset_name].image_size * 1.1)
        transform_list = [transforms.Resize(
            size, antialias=True)] + transform_list
    return transforms.Compose(transform_list)


def get_dataloader(dataset_name, train, device, checkpoint, transform_type=None):
    # Get transform
    if not train:
        transform_type = None
    transform = get_transform(dataset_name, transform_type)

    # Get dataset
    data_path = checkpoint["data_path"]
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_path,
            train=train,
            download=True,
            transform=transform
        )
    elif dataset_name == "imagenette":
        dataset = ImageNetteDataset(
            data_path + "/imagenette2-160/",
            transform=transform,
            test=not train
        )
    elif dataset_name == "coco":
        dataset = CocoMultilabel(
            data_path,
            transform=transform,
            test=not train
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    # Use a subset of the data for debugging
    limit = checkpoint["config"]["data"]["limit"]
    if limit and limit > -1:
        dataset = Subset(dataset, range(limit))

    # Return dataloader
    num_workers = 4
    pin_memory = str(device) != "cpu"
    pin_memory_device = str(device) if str(device) != "cpu" else ""
    batch_size = checkpoint["config"]["batch_size"]

    if train:
        # Train/val split
        val_ratio = checkpoint["config"]["data"]["val_ratio"]
        valsize = int(val_ratio * len(dataset))
        trainsize = len(dataset) - valsize
        trainset, valset = random_split(
            dataset,
            lengths=[trainsize, valsize],
            generator=torch.Generator().manual_seed(42)  # "deterministic"
        )
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            pin_memory_device=pin_memory_device
        )
        valloader = DataLoader(
            valset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            pin_memory_device=pin_memory_device
        )
        return trainloader, valloader
    else:
        testloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            pin_memory_device=pin_memory_device
        )
        return testloader
